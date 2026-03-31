"""Tests for the SessionWatcher file-system watcher."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from claude_memory.config import MemoryConfig
from claude_memory.db import MemoryDB
from claude_memory.watcher import SessionWatcher

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def watcher_env(tmp_path):
    """Set up an isolated watcher environment with a fake projects dir and DB."""
    # Create a fake ~/.claude/projects/ structure
    claude_home = tmp_path / ".claude"
    projects_dir = claude_home / "projects"
    projects_dir.mkdir(parents=True)

    db_path = tmp_path / "test_memory.db"
    db = MemoryDB(db_path=db_path)

    config = MemoryConfig(
        claude_home=claude_home,
        db_path=db_path,
    )

    yield {
        "tmp_path": tmp_path,
        "claude_home": claude_home,
        "projects_dir": projects_dir,
        "db": db,
        "config": config,
    }

    db.close()


def _create_session_jsonl(directory: Path, session_id: str = "test-session-0001") -> Path:
    """Create a minimal valid session JSONL file in the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{session_id}.jsonl"

    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Let's use FastAPI for this project"},
            "timestamp": "2026-03-28T10:00:00Z",
            "cwd": "/tmp/test-project",
            "gitBranch": "main",
            "sessionId": session_id,
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I'll set up a FastAPI project. Let's use FastAPI "
                            "because it provides automatic OpenAPI documentation."
                        ),
                    },
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": "/tmp/test-project/main.py",
                            "content": "from fastapi import FastAPI\napp = FastAPI()\n",
                        },
                    },
                ],
            },
            "timestamp": "2026-03-28T10:01:00Z",
            "sessionId": session_id,
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "I prefer pytest over unittest"},
            "timestamp": "2026-03-28T10:02:00Z",
            "cwd": "/tmp/test-project",
            "sessionId": session_id,
            "isMeta": False,
        },
    ]

    with filepath.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    return filepath


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestScanDetectsNewFile:
    """Test that _scan detects newly created JSONL files."""

    def test_scan_detects_new_file(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Initial scan to set baseline
        watcher._initial_scan()
        assert len(watcher._known_files) == 0

        # Create a new session file
        project_dir = env["projects_dir"] / "-tmp-test-project"
        _create_session_jsonl(project_dir, "new-session-001")

        # Scan should detect the new file
        changed = watcher._scan()
        assert len(changed) == 1
        assert changed[0].stem == "new-session-001"


class TestScanIgnoresKnownFile:
    """Test that _scan does not re-detect unchanged files."""

    def test_scan_ignores_known_file(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Create a file before initial scan
        project_dir = env["projects_dir"] / "-tmp-test-project"
        _create_session_jsonl(project_dir, "existing-session")

        # Initial scan picks it up
        watcher._initial_scan()
        assert len(watcher._known_files) == 1

        # Subsequent scan should find nothing new
        changed = watcher._scan()
        assert len(changed) == 0


class TestScanDetectsModifiedFile:
    """Test that _scan re-detects files whose mtime has changed."""

    def test_scan_detects_modified_file(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Create a file and do initial scan
        project_dir = env["projects_dir"] / "-tmp-test-project"
        filepath = _create_session_jsonl(project_dir, "mod-session")
        watcher._initial_scan()
        assert len(watcher._known_files) == 1

        # First scan - no changes
        changed = watcher._scan()
        assert len(changed) == 0

        # Modify the file (append a new line and update mtime)
        time.sleep(0.05)  # Ensure mtime difference on filesystems with coarse granularity
        with filepath.open("a") as f:
            msg = {
                "type": "user",
                "message": {"role": "user", "content": "Another message"},
                "timestamp": "2026-03-28T10:05:00Z",
                "cwd": "/tmp/test-project",
                "sessionId": "mod-session",
                "isMeta": False,
            }
            f.write(json.dumps(msg) + "\n")

        # Force a different mtime (some filesystems have 1s granularity)
        import os
        new_mtime = filepath.stat().st_mtime + 1.0
        os.utime(filepath, (new_mtime, new_mtime))

        # Scan should detect the modification
        changed = watcher._scan()
        assert len(changed) == 1
        assert changed[0].stem == "mod-session"


class TestProcessFile:
    """Test end-to-end file processing through the watcher."""

    def test_process_file(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Create a session file
        project_dir = env["projects_dir"] / "-tmp-test-project"
        filepath = _create_session_jsonl(project_dir, "process-session")

        # Process the file
        watcher._process_file(filepath)

        # Verify session was stored
        assert env["db"].is_session_processed("process-session")

        # Verify memories were extracted
        assert watcher.files_processed == 1
        assert watcher.memories_extracted > 0
        assert watcher.errors == 0

    def test_process_file_skips_already_processed(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Create and process a session file
        project_dir = env["projects_dir"] / "-tmp-test-project"
        filepath = _create_session_jsonl(project_dir, "dup-session")
        watcher._process_file(filepath)
        first_count = watcher.files_processed

        # Process the same file again - should skip
        watcher._process_file(filepath)
        assert watcher.files_processed == first_count


class TestWatcherStop:
    """Test that the watcher shuts down cleanly."""

    def test_watcher_stop(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=0.5,
            auto_generate=False,
        )

        # Start watcher in a background thread
        started = threading.Event()
        stopped = threading.Event()

        def _run():
            started.set()
            watcher.start()
            stopped.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Wait for watcher to start
        assert started.wait(timeout=5.0), "Watcher did not start in time"

        # Signal stop
        watcher.stop()

        # Wait for clean shutdown
        assert stopped.wait(timeout=5.0), "Watcher did not stop in time"

        # Verify it's no longer running
        assert not watcher._running

    def test_watcher_stop_returns_summary(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=0.5,
            auto_generate=False,
        )

        summary = watcher.summary()
        assert "files_processed" in summary
        assert "memories_extracted" in summary
        assert "errors" in summary
        assert summary["files_processed"] == 0


class TestWatcherWithProjectFilter:
    """Test that project_filter restricts which files are watched."""

    def test_watcher_with_project_filter(self, watcher_env):
        env = watcher_env
        tmp = env["tmp_path"]

        # Use paths rooted in tmp_path so resolve() doesn't change them
        # (avoids macOS /tmp -> /private/tmp symlink issues)
        project_a_path = str((tmp / "project-a").resolve())
        project_b_path = str((tmp / "project-b").resolve())

        # Encode paths the same way Claude Code does (replace / with -)
        encoded_a = project_a_path.replace("/", "-")
        encoded_b = project_b_path.replace("/", "-")

        project_a_dir = env["projects_dir"] / encoded_a
        project_b_dir = env["projects_dir"] / encoded_b
        _create_session_jsonl(project_a_dir, "session-a")
        _create_session_jsonl(project_b_dir, "session-b")

        # Create watcher that only watches project A
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
            project_filter=project_a_path,
        )

        # Initial scan should only find project A's file
        watcher._initial_scan()
        assert len(watcher._known_files) == 1

        known_stems = {p.stem for p in watcher._known_files}
        assert "session-a" in known_stems
        assert "session-b" not in known_stems

    def test_watcher_without_filter_sees_all(self, watcher_env):
        env = watcher_env

        # Create files in two different project directories
        project_a_dir = env["projects_dir"] / "-tmp-project-a"
        project_b_dir = env["projects_dir"] / "-tmp-project-b"
        _create_session_jsonl(project_a_dir, "session-a")
        _create_session_jsonl(project_b_dir, "session-b")

        # Create watcher without filter
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Initial scan should find both files
        watcher._initial_scan()
        assert len(watcher._known_files) == 2


class TestProjectPathDecoding:
    """Test that project paths are correctly derived from file locations."""

    def test_project_path_from_file(self, watcher_env):
        env = watcher_env
        watcher = SessionWatcher(
            db=env["db"],
            config=env["config"],
            interval=1.0,
            auto_generate=False,
        )

        # Simulate a file path like ~/.claude/projects/-Users-foo-Desktop-myproject/session.jsonl
        project_dir = env["projects_dir"] / "-Users-foo-Desktop-myproject"
        project_dir.mkdir(parents=True)
        filepath = project_dir / "some-session.jsonl"

        result = watcher._project_path_from_file(filepath)
        assert result == "/Users/foo/Desktop/myproject"
