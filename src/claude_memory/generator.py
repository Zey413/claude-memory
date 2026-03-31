"""Generate CLAUDE.md context files from stored memories."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from claude_memory.config import MemoryConfig, project_path_to_claude_dir
from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType, ProjectContext
from claude_memory.search import MemorySearch
from claude_memory.utils import format_duration

logger = logging.getLogger(__name__)


# ── Internal helpers ────────────────────────────────────────────────────────

_MAX_ITEMS_PER_SECTION = 10

# Common generic directory names that should be skipped for project naming
_GENERIC_DIR_NAMES = frozenset({
    "src", "app", "lib", "pkg", "packages", "code", "project",
    "workspace", "repo", "main", "build", "dist", "out",
})

# Stop-words to exclude from topic clustering keywords
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "not",
    "so", "yet", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "about", "up", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "all", "any", "if", "when", "what", "which", "who",
    "how", "use", "using", "used", "need", "add", "new",
})

# Section priority order for token budget (lower index = higher priority)
_SECTION_PRIORITY = [
    "Key Decisions",
    "Active TODOs",
    "Code Patterns & Conventions",
    "Recent Sessions",
    "User Preferences",
    "Known Issues & Solutions",
]


def _bullet(text: str) -> str:
    """Format a single bullet point, collapsing to one line."""
    return f"- {text.strip()}"


def _section(title: str, items: list[str]) -> str:
    """Render a markdown section. Returns empty string when *items* is empty."""
    if not items:
        return ""
    capped = items[:_MAX_ITEMS_PER_SECTION]
    body = "\n".join(_bullet(item) for item in capped)
    return f"## {title}\n\n{body}\n"


def _memory_one_liner(memory: Memory, score_indicator: str = "") -> str:
    """Condense a memory into a single descriptive line.

    If *score_indicator* is provided it is appended in parentheses.
    """
    line = memory.title
    if memory.content and memory.content != memory.title:
        # Append a short content excerpt when it adds useful detail.
        extra = memory.content.replace("\n", " ").strip()
        if len(extra) > 120:
            extra = extra[:117] + "..."
        line = f"**{memory.title}** — {extra}"
    if score_indicator:
        line = f"{line} ({score_indicator})"
    return line


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: word-count * 1.3."""
    return int(len(text.split()) * 1.3)


def _extract_keywords(memory: Memory) -> set[str]:
    """Extract meaningful keywords from a memory's title and tags."""
    words: set[str] = set()
    # Words from title (lowered, alpha-only, len >= 3)
    for word in re.split(r"[^a-zA-Z0-9]+", memory.title.lower()):
        if len(word) >= 3 and word not in _STOP_WORDS:
            words.add(word)
    # Tags are already good keywords
    for tag in memory.tags:
        tag_lower = tag.lower().strip()
        if tag_lower and tag_lower not in _STOP_WORDS:
            words.add(tag_lower)
    return words


def _smart_project_name(project_path: str) -> str:
    """Resolve a human-readable project name, skipping generic dir names.

    Falls back to ``project_name_from_path`` for normal paths.
    """
    p = Path(project_path)
    name = p.name
    if name.lower() in _GENERIC_DIR_NAMES and p.parent != p:
        name = p.parent.name
    return name


# ── Generator ───────────────────────────────────────────────────────────────


class ClaudemdGenerator:
    """Build and write CLAUDE.md context files from the memory store."""

    def __init__(self, db: MemoryDB, search: MemorySearch):
        self.db = db
        self.search = search

    # ── Internal file helper ────────────────────────────────────────────

    @staticmethod
    def _atomic_write(dest: Path, content: str) -> None:
        """Write content to *dest* atomically via a temp file + os.replace().

        Creates parent directories with exist_ok=True.  Raises readable
        errors on PermissionError and OSError.
        """
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(
                f"Cannot create directory {dest.parent}: permission denied"
            )
        except OSError as exc:
            raise OSError(f"Cannot create directory {dest.parent}: {exc}") from exc

        # Write to a temp file in the same directory, then atomically replace
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", dir=str(dest.parent), prefix=".claude_mem_"
            )
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            fd = None  # Mark as closed
            os.replace(tmp_path, str(dest))
            tmp_path = None  # Mark as consumed
        except PermissionError:
            raise PermissionError(
                f"Cannot write to {dest}: permission denied"
            )
        except OSError as exc:
            raise OSError(f"Failed to write {dest}: {exc}") from exc
        finally:
            # Clean up fd if still open
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            # Clean up temp file if replace didn't happen
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ── Sorting & Scoring helpers ───────────────────────────────────────

    def _sort_by_importance(self, memories: list[Memory]) -> list[Memory]:
        """Sort memories by importance_score descending.

        If all scores are 0 (never consolidated), fall back to recency
        (newest first).
        """
        scores = {m.id: self.db.get_importance_score(m.id) for m in memories}
        all_zero = all(s == 0.0 for s in scores.values())
        if all_zero:
            return sorted(memories, key=lambda m: m.created_at, reverse=True)
        return sorted(memories, key=lambda m: scores.get(m.id, 0.0), reverse=True)

    def _score_indicator(self, memory: Memory) -> str:
        """Return a star indicator based on importance score.

        ★ for score > 0.7, ☆ for score > 0.4, empty string otherwise.
        """
        score = self.db.get_importance_score(memory.id)
        if score > 0.7:
            return "\u2605"
        if score > 0.4:
            return "\u2606"
        return ""

    # ── Topic Clustering ────────────────────────────────────────────────

    def _cluster_by_topic(self, memories: list[Memory]) -> dict[str, list[Memory]]:
        """Group memories into topic clusters based on shared keywords.

        A memory joins a cluster when it shares 2+ keywords with at least
        one other memory in that cluster.  The cluster is named after the
        most common keyword among its members.  Memories that don't fit
        any cluster go into ``"Other"``.
        """
        if not memories:
            return {}

        # Pre-compute keywords for each memory
        kw_map: dict[str, set[str]] = {}
        for mem in memories:
            kw_map[mem.id] = _extract_keywords(mem)

        # Build clusters via pairwise keyword overlap
        # Use a simple union-find approach
        parent: dict[str, str] = {m.id: m.id for m in memories}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Merge memories sharing 2+ keywords
        mem_list = list(memories)
        for i in range(len(mem_list)):
            for j in range(i + 1, len(mem_list)):
                shared = kw_map[mem_list[i].id] & kw_map[mem_list[j].id]
                if len(shared) >= 2:
                    union(mem_list[i].id, mem_list[j].id)

        # Group by root
        groups: dict[str, list[Memory]] = {}
        for mem in memories:
            root = find(mem.id)
            groups.setdefault(root, []).append(mem)

        # Name each cluster
        result: dict[str, list[Memory]] = {}
        other: list[Memory] = []
        for _root, group in groups.items():
            if len(group) < 2:
                other.extend(group)
                continue
            # Find most common keyword in the group
            keyword_counter: Counter[str] = Counter()
            for mem in group:
                keyword_counter.update(kw_map[mem.id])
            if keyword_counter:
                cluster_name = keyword_counter.most_common(1)[0][0].title()
            else:
                cluster_name = "Other"
            # Handle duplicate cluster names by merging
            if cluster_name in result:
                result[cluster_name].extend(group)
            else:
                result[cluster_name] = list(group)

        if other:
            result["Other"] = other

        return result

    # ── Clustered section rendering ─────────────────────────────────────

    def _clustered_section(
        self,
        title: str,
        memories: list[Memory],
        *,
        checkbox: bool = False,
    ) -> str:
        """Render a section with topic sub-headings.

        When *checkbox* is True, each bullet is prefixed with ``[ ]``.
        """
        if not memories:
            return ""

        sorted_memories = self._sort_by_importance(memories)
        clusters = self._cluster_by_topic(sorted_memories)

        lines: list[str] = [f"## {title}\n"]

        for cluster_name, mems in clusters.items():
            mems = self._sort_by_importance(mems)
            lines.append(f"### {cluster_name}\n")
            for mem in mems[:_MAX_ITEMS_PER_SECTION]:
                indicator = self._score_indicator(mem)
                text = _memory_one_liner(mem, score_indicator=indicator)
                if checkbox:
                    lines.append(f"- [ ] {text.strip()}")
                else:
                    lines.append(f"- {text.strip()}")
            lines.append("")  # blank line after cluster

        return "\n".join(lines)

    # ── Recent Activity section ─────────────────────────────────────────

    def _recent_activity_section(self, project_path: str) -> str:
        """Build the 'Recent Activity' section from recent sessions.

        Shows last 1-3 sessions with summary, key files, and open threads.
        """
        recent = self.db.get_recent_sessions(project_path=project_path, limit=3)
        if not recent:
            return ""

        lines: list[str] = ["## Recent Activity\n"]

        for sess in recent:
            date_str = sess.started_at.strftime("%Y-%m-%d")
            dur = format_duration(sess.duration_minutes) if sess.duration_minutes else "?"
            summary = (
                sess.summary_text.replace("\n", " ").strip()
                if sess.summary_text else "No summary"
            )
            if len(summary) > 100:
                summary = summary[:97] + "..."
            files_count = len(sess.files_modified) if sess.files_modified else 0
            files_info = f" — modified {files_count} files" if files_count else ""
            lines.append(f"- **Session** ({date_str}, {dur}): {summary}{files_info}")

        # Key files touched (from all recent sessions)
        all_files: list[str] = []
        for sess in recent:
            if sess.files_modified:
                all_files.extend(sess.files_modified)
        if all_files:
            # Deduplicate, keep order, limit to 5
            seen: set[str] = set()
            unique_files: list[str] = []
            for f in all_files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            display_files = unique_files[:5]
            lines.append(f"- **Key files touched**: {', '.join(display_files)}")

        # Open threads: from recent TODOs and key topics
        threads: list[str] = []
        for sess in recent:
            if sess.key_topics:
                threads.extend(sess.key_topics)
        # Also grab recent TODO titles
        todos = self.search.by_type(project_path, MemoryType.TODO, limit=3)
        for t in todos:
            threads.append(t.title)
        if threads:
            # Deduplicate
            seen_threads: set[str] = set()
            unique_threads: list[str] = []
            for t in threads:
                if t not in seen_threads:
                    seen_threads.add(t)
                    unique_threads.append(t)
            lines.append(f"- **Open threads**: {', '.join(unique_threads[:5])}")

        lines.append("")
        return "\n".join(lines)

    # ── Public API ───────────────────────────────────────────────────────

    def generate_project_context(self, project_path: str) -> str:
        """Generate the full CLAUDE.md markdown content for *project_path*."""
        project_name = _smart_project_name(project_path)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        sections: list[str] = []

        # Header
        sections.append(
            f"# {project_name}\n\n"
            f"> Auto-generated by claude-memory on {now_str}. "
            "Do not edit manually — changes will be overwritten.\n"
        )

        # Recent Activity (new — right after header)
        activity = self._recent_activity_section(project_path)
        if activity:
            sections.append(activity)

        # 1. Key Decisions (importance-sorted)
        limit = _MAX_ITEMS_PER_SECTION
        decisions = self.search.by_type(
            project_path, MemoryType.DECISION, limit=limit,
        )
        decisions = self._sort_by_importance(decisions)
        sections.append(_section(
            "Key Decisions",
            [_memory_one_liner(m, self._score_indicator(m)) for m in decisions],
        ))

        # 2. Active TODOs (clustered by topic)
        todos = self.search.by_type(
            project_path, MemoryType.TODO, limit=limit,
        )
        sections.append(self._clustered_section(
            "Active TODOs", todos, checkbox=True,
        ))

        # 3. Code Patterns & Conventions (clustered by topic)
        patterns = self.search.by_type(
            project_path, MemoryType.PATTERN, limit=limit,
        )
        sections.append(self._clustered_section(
            "Code Patterns & Conventions", patterns,
        ))

        # 4. Recent Sessions (last 5 summaries)
        recent_sessions = self.db.get_recent_sessions(project_path=project_path, limit=5)
        session_lines: list[str] = []
        for sess in recent_sessions:
            date_str = sess.started_at.strftime("%Y-%m-%d")
            duration_str = format_duration(sess.duration_minutes) if sess.duration_minutes else "?"
            summary = (
                sess.summary_text.replace("\n", " ").strip()
                if sess.summary_text else "No summary"
            )
            if len(summary) > 120:
                summary = summary[:117] + "..."
            branch_info = f" (`{sess.git_branch}`)" if sess.git_branch else ""
            session_lines.append(f"{date_str}{branch_info} [{duration_str}]: {summary}")
        sections.append(_section("Recent Sessions", session_lines))

        # 5. User Preferences (importance-sorted)
        preferences = self.search.by_type(
            project_path, MemoryType.PREFERENCE, limit=limit,
        )
        preferences = self._sort_by_importance(preferences)
        sections.append(_section(
            "User Preferences",
            [_memory_one_liner(m, self._score_indicator(m)) for m in preferences],
        ))

        # 6. Known Issues & Solutions (importance-sorted)
        issues = self.search.by_type(
            project_path, MemoryType.ISSUE, limit=limit,
        )
        solutions = self.search.by_type(
            project_path, MemoryType.SOLUTION, limit=limit,
        )
        issues = self._sort_by_importance(issues)
        solutions = self._sort_by_importance(solutions)
        issue_lines: list[str] = []
        for m in issues:
            issue_lines.append(f"\U0001f41b {_memory_one_liner(m, self._score_indicator(m))}")
        for m in solutions:
            issue_lines.append(f"\u2705 {_memory_one_liner(m, self._score_indicator(m))}")
        sections.append(_section("Known Issues & Solutions", issue_lines))

        # Assemble – drop empty sections
        body = "\n".join(s for s in sections if s)
        return body

    def generate_with_budget(
        self,
        project_path: str,
        token_budget: int = 4000,
    ) -> str:
        """Generate CLAUDE.md content constrained to a token budget.

        Fills sections in priority order (Decisions > TODOs > Patterns >
        Sessions > Preferences > Issues).  Sections that would exceed the
        remaining budget are truncated or dropped, lowest-priority first.

        Appends a token-estimate footer comment.
        """
        project_name = _smart_project_name(project_path)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        header = (
            f"# {project_name}\n\n"
            f"> Auto-generated by claude-memory on {now_str}. "
            "Do not edit manually — changes will be overwritten.\n"
        )

        # Recent Activity section (high value, included before priority sections)
        activity = self._recent_activity_section(project_path)

        # Build each priority section independently
        limit = _MAX_ITEMS_PER_SECTION

        section_builders: dict[str, str] = {}

        # Key Decisions
        decisions = self._sort_by_importance(
            self.search.by_type(project_path, MemoryType.DECISION, limit=limit)
        )
        section_builders["Key Decisions"] = _section(
            "Key Decisions",
            [_memory_one_liner(m, self._score_indicator(m)) for m in decisions],
        )

        # Active TODOs
        todos = self.search.by_type(project_path, MemoryType.TODO, limit=limit)
        section_builders["Active TODOs"] = self._clustered_section(
            "Active TODOs", todos, checkbox=True,
        )

        # Code Patterns
        patterns = self.search.by_type(project_path, MemoryType.PATTERN, limit=limit)
        section_builders["Code Patterns & Conventions"] = self._clustered_section(
            "Code Patterns & Conventions", patterns,
        )

        # Recent Sessions
        recent_sessions = self.db.get_recent_sessions(project_path=project_path, limit=5)
        session_lines: list[str] = []
        for sess in recent_sessions:
            date_str = sess.started_at.strftime("%Y-%m-%d")
            dur = format_duration(sess.duration_minutes) if sess.duration_minutes else "?"
            summary = (
                sess.summary_text.replace("\n", " ").strip()
                if sess.summary_text else "No summary"
            )
            if len(summary) > 120:
                summary = summary[:117] + "..."
            branch_info = f" (`{sess.git_branch}`)" if sess.git_branch else ""
            session_lines.append(f"{date_str}{branch_info} [{dur}]: {summary}")
        section_builders["Recent Sessions"] = _section("Recent Sessions", session_lines)

        # User Preferences
        prefs = self._sort_by_importance(
            self.search.by_type(project_path, MemoryType.PREFERENCE, limit=limit)
        )
        section_builders["User Preferences"] = _section(
            "User Preferences",
            [_memory_one_liner(m, self._score_indicator(m)) for m in prefs],
        )

        # Known Issues & Solutions
        issues = self._sort_by_importance(
            self.search.by_type(project_path, MemoryType.ISSUE, limit=limit)
        )
        solutions = self._sort_by_importance(
            self.search.by_type(project_path, MemoryType.SOLUTION, limit=limit)
        )
        issue_lines_list: list[str] = []
        for m in issues:
            issue_lines_list.append(
                f"\U0001f41b {_memory_one_liner(m, self._score_indicator(m))}"
            )
        for m in solutions:
            issue_lines_list.append(
                f"\u2705 {_memory_one_liner(m, self._score_indicator(m))}"
            )
        section_builders["Known Issues & Solutions"] = _section(
            "Known Issues & Solutions", issue_lines_list,
        )

        # --- Budget allocation ---
        parts: list[str] = [header]
        if activity:
            parts.append(activity)

        # Add sections in priority order, checking budget
        for section_name in _SECTION_PRIORITY:
            section_text = section_builders.get(section_name, "")
            if not section_text:
                continue
            candidate = "\n".join(parts) + "\n" + section_text
            if _estimate_tokens(candidate) <= token_budget:
                parts.append(section_text)
            # If adding this section blows the budget, skip it

        body = "\n".join(s for s in parts if s)
        token_est = _estimate_tokens(body)
        body += f"\n<!-- Token estimate: ~{token_est} tokens -->\n"
        return body

    def build_project_context(self, project_path: str) -> ProjectContext:
        """Build a structured ``ProjectContext`` model from stored data."""
        project_name = _smart_project_name(project_path)
        total_sessions = self.db.count_sessions(project_path=project_path)
        total_memories = self.db.count_memories(project_path=project_path)

        limit = _MAX_ITEMS_PER_SECTION
        recent_sessions = self.db.get_recent_sessions(
            project_path=project_path, limit=1,
        )
        last_session_at = (
            recent_sessions[0].started_at if recent_sessions else None
        )

        decisions = self.search.by_type(
            project_path, MemoryType.DECISION, limit=limit,
        )
        todos = self.search.by_type(
            project_path, MemoryType.TODO, limit=limit,
        )
        patterns = self.search.by_type(
            project_path, MemoryType.PATTERN, limit=limit,
        )

        return ProjectContext(
            project_path=project_path,
            project_name=project_name,
            total_sessions=total_sessions,
            total_memories=total_memories,
            last_session_at=last_session_at,
            key_decisions=[m.title for m in decisions],
            active_todos=[m.title for m in todos],
            common_patterns=[m.title for m in patterns],
        )

    def write_to_memory_dir(self, project_path: str, config: MemoryConfig | None = None) -> Path:
        """Write context to ``~/.claude/projects/<PATH>/memory/context.md``."""
        claude_dir = project_path_to_claude_dir(project_path, config)
        memory_dir = claude_dir / "memory"
        dest = memory_dir / "context.md"
        content = self.generate_project_context(project_path)
        self._atomic_write(dest, content)
        return dest

    def write_to_project_root(self, project_path: str) -> Path:
        """Write context to ``<project>/CLAUDE.md``."""
        dest = Path(project_path) / "CLAUDE.md"
        content = self.generate_project_context(project_path)
        self._atomic_write(dest, content)
        return dest

    def render_to_string(self, project_path: str) -> str:
        """Return the rendered CLAUDE.md content without writing to disk."""
        return self.generate_project_context(project_path)
