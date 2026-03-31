"""File system watcher that monitors for new Claude Code session files and auto-ingests them.

Uses polling (no external dependencies like watchdog) to detect new/modified
JSONL files in ~/.claude/projects/ and processes them through the existing
parse -> extract -> store pipeline.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

from claude_memory.config import MemoryConfig
from claude_memory.db import MemoryDB
from claude_memory.extractor import MemoryExtractor
from claude_memory.generator import ClaudemdGenerator
from claude_memory.parser import parse_session_file
from claude_memory.search import MemorySearch

logger = logging.getLogger(__name__)


class SessionWatcher:
    """Polls for new/modified JSONL session files and auto-ingests them.

    Monitors ``~/.claude/projects/`` (or a specific project directory)
    recursively for ``.jsonl`` files.  Tracks known files and their mtimes
    in a dict so only genuinely new or modified files are processed.
    """

    def __init__(
        self,
        db: MemoryDB,
        config: MemoryConfig | None = None,
        interval: float = 5.0,
        auto_generate: bool = True,
        project_filter: str | None = None,
    ) -> None:
        self.db = db
        self.config = config or MemoryConfig()
        self.interval = interval
        self.auto_generate = auto_generate
        self.project_filter = project_filter

        # Internal state
        self._known_files: dict[Path, float] = {}  # path -> mtime
        self._running = False
        self._files_processed = 0
        self._memories_extracted = 0
        self._errors = 0

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def watch_dir(self) -> Path:
        """The root directory being monitored."""
        return self.config.projects_dir

    @property
    def files_processed(self) -> int:
        return self._files_processed

    @property
    def memories_extracted(self) -> int:
        return self._memories_extracted

    @property
    def errors(self) -> int:
        return self._errors

    def start(self) -> None:
        """Start the polling loop.  Blocks until :meth:`stop` is called or
        a SIGINT/SIGTERM is received.
        """
        self._running = True

        # Install signal handlers for graceful shutdown (only works in main thread)
        prev_sigint = None
        prev_sigterm = None
        _in_main_thread = threading.current_thread() is threading.main_thread()

        if _in_main_thread:
            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)

            def _handle_signal(signum: int, frame: object) -> None:
                logger.info("Received signal %d, stopping watcher...", signum)
                self.stop()

            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)

        try:
            # Initial scan to populate known files (don't process existing ones)
            self._initial_scan()

            while self._running:
                new_files = self._scan()
                for filepath in new_files:
                    if not self._running:
                        break
                    # Wait for file to stabilize (session may still be writing)
                    time.sleep(2)
                    self._process_file(filepath)
                # Sleep in small increments so we can respond to stop() quickly
                self._interruptible_sleep(self.interval)
        finally:
            # Restore original signal handlers
            if _in_main_thread:
                signal.signal(signal.SIGINT, prev_sigint)
                signal.signal(signal.SIGTERM, prev_sigterm)

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self._running = False

    def summary(self) -> dict[str, int]:
        """Return a summary of processing activity."""
        return {
            "files_processed": self._files_processed,
            "memories_extracted": self._memories_extracted,
            "errors": self._errors,
        }

    # ── Scanning ─────────────────────────────────────────────────────────

    def _initial_scan(self) -> None:
        """Populate ``_known_files`` with existing files so they aren't
        treated as "new" on the first polling cycle.
        """
        for filepath in self._iter_jsonl_files():
            try:
                self._known_files[filepath] = filepath.stat().st_mtime
            except OSError:
                pass

    def _scan(self) -> list[Path]:
        """Scan for new or modified session files.

        Returns a list of paths that are either previously unseen or whose
        mtime has changed since the last scan.
        """
        changed: list[Path] = []
        current_files: set[Path] = set()

        for filepath in self._iter_jsonl_files():
            current_files.add(filepath)
            try:
                mtime = filepath.stat().st_mtime
            except OSError:
                continue

            prev_mtime = self._known_files.get(filepath)
            if prev_mtime is None or mtime != prev_mtime:
                changed.append(filepath)
                self._known_files[filepath] = mtime

        # Clean up entries for deleted files
        deleted = set(self._known_files) - current_files
        for p in deleted:
            del self._known_files[p]

        return changed

    def _iter_jsonl_files(self):
        """Yield all .jsonl files under the watched directory tree."""
        root = self.watch_dir
        if not root.exists():
            return

        try:
            if self.project_filter is not None:
                # Only scan the specific project directory
                normalized = str(Path(self.project_filter).resolve())
                encoded = normalized.replace("/", "-")
                project_dir = root / encoded
                if project_dir.exists():
                    yield from project_dir.glob("*.jsonl")
            else:
                yield from root.rglob("*.jsonl")
        except PermissionError:
            logger.warning("Permission denied scanning %s", root)
        except OSError as exc:
            logger.warning("Error scanning %s: %s", root, exc)

    # ── Processing ───────────────────────────────────────────────────────

    def _process_file(self, filepath: Path) -> None:
        """Ingest a single session file through the parse -> extract -> store pipeline."""
        session_id = filepath.stem

        # Skip already-processed sessions
        if self.db.is_session_processed(session_id):
            logger.debug("Session %s already processed, skipping", session_id[:8])
            return

        logger.info("Processing session %s from %s", session_id[:8], filepath)

        try:
            # 1. Parse
            messages = parse_session_file(filepath)
            if not messages:
                logger.info("No messages in %s, skipping", filepath.name)
                return

            # 2. Determine project path from the file location
            project_path = self._project_path_from_file(filepath)

            # 3. Extract memories
            extractor = MemoryExtractor()
            memories = extractor.extract_all(messages, session_id, project_path)

            # 4. Generate session summary
            summary = extractor.generate_summary(messages, session_id, project_path)

            # 5. Store
            self.db.insert_session(summary)
            for mem in memories:
                self.db.insert_memory(mem)

            self._files_processed += 1
            self._memories_extracted += len(memories)

            logger.info(
                "Ingested session %s: %d messages, %d memories extracted",
                session_id[:8], len(messages), len(memories),
            )

            # 6. Optionally regenerate CLAUDE.md
            if self.auto_generate:
                self._regenerate_context(project_path)

        except Exception as exc:
            self._errors += 1
            logger.error("Error processing %s: %s", filepath, exc)

    def _project_path_from_file(self, filepath: Path) -> str:
        """Derive the original project path from a session file's location.

        Session files live under ``~/.claude/projects/<encoded-path>/``.
        The encoded name is the project path with ``/`` replaced by ``-``.
        """
        # The parent directory name is the encoded project path
        encoded_name = filepath.parent.name
        if encoded_name.startswith("-"):
            # Decode: -Users-foo-Desktop-myproject -> /Users/foo/Desktop/myproject
            decoded = "/" + encoded_name[1:].replace("-", "/")
            return decoded
        # Fallback: use the parent directory path
        return str(filepath.parent)

    def _regenerate_context(self, project_path: str) -> None:
        """Regenerate the CLAUDE.md context file for a project."""
        try:
            searcher = MemorySearch(self.db)
            gen = ClaudemdGenerator(self.db, searcher)
            dest = gen.write_to_memory_dir(project_path, self.config)
            logger.info("Regenerated context at %s", dest)
        except Exception as exc:
            logger.warning("Failed to regenerate context for %s: %s", project_path, exc)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in 0.25s increments so stop() takes effect quickly."""
        elapsed = 0.0
        while elapsed < seconds and self._running:
            time.sleep(min(0.25, seconds - elapsed))
            elapsed += 0.25
