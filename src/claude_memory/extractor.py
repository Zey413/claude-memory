"""Rule-based memory extractor for Claude Code sessions.

Extracts structured memories (decisions, patterns, TODOs, error/fix pairs,
preferences) from parsed session messages using lightweight heuristics.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import PurePosixPath

from claude_memory.models import Memory, MemoryType, SessionSummary
from claude_memory.parser import ParsedMessage, ToolUse
from claude_memory.utils import content_hash, project_name_from_path, truncate

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Stopwords for topic extraction (common English words to filter out)
# --------------------------------------------------------------------------- #

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "was", "are",
    "be", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "not", "no", "so", "if",
    "then", "than", "too", "very", "just", "about", "up", "out", "how",
    "what", "which", "who", "when", "where", "why", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "because", "until", "while",
    "here", "there", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "mine", "yours", "ours", "theirs", "been", "being",
    "am", "were", "also", "like", "get", "got", "make", "made", "need",
    "let", "know", "see", "look", "want", "use", "used", "using", "file",
    "please", "ok", "okay", "yes", "yeah", "sure", "right", "well", "now",
    "one", "two", "new", "way", "going", "think", "try", "thing", "go",
    "don", "doesn", "didn", "won", "isn", "aren", "wasn", "weren",
    "haven", "hasn", "hadn", "ll", "ve", "re", "de",
})

# Minimum word length to consider for topics
_MIN_WORD_LEN = 3

# --------------------------------------------------------------------------- #
#  Regex patterns for extraction
# --------------------------------------------------------------------------- #

# Decision patterns (case-insensitive)
_DECISION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:let'?s|let us)\s+(use|go with|implement|create|switch to|try)\s+(.+)", re.I),
    re.compile(r"(?:decided? to|choosing|going to)\s+(.+)", re.I),
    re.compile(r"instead of\s+(.+?),\s+(?:we|i)(?:'ll)?\s+(.+)", re.I),
    re.compile(r"the\s+(?:best|right|better|correct)\s+(?:approach|choice|option|way|solution)\s+(?:is|would be)\s+(.+)", re.I),
    re.compile(r"(?:we|i)\s+should\s+(?:use|go with|implement|create|adopt)\s+(.+)", re.I),
]

# Preference patterns
_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"i\s+prefer\s+(.+)", re.I),
    re.compile(r"always\s+use\s+(.+)", re.I),
    re.compile(r"(?:don'?t|do not|never)\s+use\s+(.+)", re.I),
    re.compile(r"please\s+(?:always\s+)?use\s+(.+)", re.I),
    re.compile(r"switch\s+to\s+(.+)", re.I),
    re.compile(r"i\s+(?:like|want)\s+(?:to\s+use\s+)?(.+?)(?:\s+instead| better| more)", re.I),
]

# TODO / future-work patterns
_TODO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bTODO\b[:\s]*(.+)", re.I),
    re.compile(r"\bFIXME\b[:\s]*(.+)", re.I),
    re.compile(r"\bneed to\b\s+(.+)", re.I),
    re.compile(r"\bshould later\b\s+(.+)", re.I),
    re.compile(r"\bremember to\b\s+(.+)", re.I),
    re.compile(r"\bcome back (?:to|and)\b\s+(.+)", re.I),
]

# Error indicators in tool output
_ERROR_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"\berror\b", re.I),
    re.compile(r"\bfailed\b", re.I),
    re.compile(r"\bexception\b", re.I),
    re.compile(r"\btraceback\b", re.I),
    re.compile(r"\bcommand not found\b", re.I),
    re.compile(r"\bpermission denied\b", re.I),
    re.compile(r"\bno such file\b", re.I),
    re.compile(r"\bsyntax error\b", re.I),
    re.compile(r"\bexit code [1-9]\d*\b", re.I),
    re.compile(r"\bnon-zero exit\b", re.I),
]

# Word tokenisation
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #

def _sentence_around_match(text: str, match: re.Match[str], context_chars: int = 200) -> str:
    """Extract a reasonable sentence/context window around a regex match."""
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)
    snippet = text[start:end].strip()
    # Try to trim to sentence boundaries
    if start > 0:
        dot = snippet.find(". ")
        if dot != -1 and dot < context_chars // 2:
            snippet = snippet[dot + 2:]
    if end < len(text):
        dot = snippet.rfind(". ")
        if dot != -1 and dot > len(snippet) - context_chars // 2:
            snippet = snippet[: dot + 1]
    return snippet.strip()


def _has_error(text: str) -> bool:
    """Check if text contains error indicators."""
    return any(pat.search(text) for pat in _ERROR_INDICATORS)


def _file_path_from_tool(tool: ToolUse) -> str | None:
    """Extract the target file path from a tool invocation, if any."""
    inp = tool.input_data
    if tool.name in ("Write", "Edit", "Read"):
        return inp.get("file_path") or inp.get("path")
    if tool.name == "Bash":
        # Crude: look for obvious path arguments in the command
        cmd = inp.get("command", "")
        # Match quoted or unquoted paths starting with / or ./
        m = re.search(r"""(?:["'])(/[^"']+)(?:["'])""", cmd)
        if m:
            return m.group(1)
    return None


def _directory_of(filepath: str) -> str:
    """Return the parent directory string for a filepath."""
    return str(PurePosixPath(filepath).parent)


# --------------------------------------------------------------------------- #
#  MemoryExtractor
# --------------------------------------------------------------------------- #

class MemoryExtractor:
    """Extracts structured memories from parsed Claude Code session messages."""

    # Number of messages to look ahead when searching for a fix after an error
    ERROR_FIX_LOOKAHEAD = 5

    # ----- public interface ------------------------------------------------ #

    def extract_all(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> list[Memory]:
        """Run all extraction rules and return deduplicated memories."""
        memories: list[Memory] = []
        memories.extend(self._extract_decisions(messages, session_id, project_path))
        memories.extend(self._extract_file_patterns(messages, session_id, project_path))
        memories.extend(self._extract_todos(messages, session_id, project_path))
        memories.extend(self._extract_errors_and_fixes(messages, session_id, project_path))
        memories.extend(self._extract_preferences(messages, session_id, project_path))
        return self._deduplicate(memories)

    def generate_summary(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> SessionSummary:
        """Generate a high-level session summary from parsed messages."""
        user_msgs = [m for m in messages if m.msg_type == "user" and not m.is_meta]
        assistant_msgs = [m for m in messages if m.msg_type == "assistant"]

        # Timestamps
        timestamps = [m.timestamp for m in messages if m.timestamp is not None]
        started_at = min(timestamps) if timestamps else None
        ended_at = max(timestamps) if timestamps else None
        duration_minutes: float | None = None
        if started_at and ended_at:
            duration_minutes = (ended_at - started_at).total_seconds() / 60.0

        # Tool usage counts
        tool_counts: Counter[str] = Counter()
        for msg in assistant_msgs:
            for tu in msg.tool_uses:
                tool_counts[tu.name] += 1

        # Files modified and read
        files_modified: list[str] = []
        files_read: list[str] = []
        seen_mod: set[str] = set()
        seen_read: set[str] = set()

        for msg in messages:
            for tu in msg.tool_uses:
                fpath = _file_path_from_tool(tu)
                if fpath is None:
                    continue
                if tu.name in ("Write", "Edit"):
                    if fpath not in seen_mod:
                        files_modified.append(fpath)
                        seen_mod.add(fpath)
                elif tu.name == "Read":
                    if fpath not in seen_read:
                        files_read.append(fpath)
                        seen_read.add(fpath)

        # Git branch (use last non-None value)
        git_branch: str | None = None
        for msg in reversed(messages):
            if msg.git_branch:
                git_branch = msg.git_branch
                break

        # Key topics via word frequency on user messages
        key_topics = self._extract_topics(user_msgs)

        # Short summary text
        summary_parts: list[str] = []
        proj_name = project_name_from_path(project_path)
        summary_parts.append(f"Session on {proj_name}")
        if git_branch:
            summary_parts.append(f"(branch: {git_branch})")
        if duration_minutes is not None:
            summary_parts.append(f"lasting {duration_minutes:.0f} min")
        summary_parts.append(f"with {len(user_msgs)} user messages")
        if tool_counts:
            top_tools = tool_counts.most_common(3)
            tool_str = ", ".join(f"{n}×{name}" for name, n in top_tools)
            summary_parts.append(f"using {tool_str}")
        if key_topics:
            summary_parts.append(f"— topics: {', '.join(key_topics[:5])}")
        summary_text = " ".join(summary_parts)

        return SessionSummary(
            session_id=session_id,
            project_path=project_path,
            git_branch=git_branch,
            started_at=started_at or ended_at or __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            ended_at=ended_at,
            duration_minutes=duration_minutes,
            message_count=len(messages),
            user_message_count=len(user_msgs),
            assistant_message_count=len(assistant_msgs),
            tool_uses=dict(tool_counts),
            files_modified=files_modified,
            files_read=files_read,
            summary_text=summary_text,
            key_topics=key_topics,
        )

    # ----- private extraction methods -------------------------------------- #

    def _extract_decisions(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> list[Memory]:
        """Extract architecture/design decisions from messages."""
        memories: list[Memory] = []
        for msg in messages:
            if not msg.text_content:
                continue
            for pattern in _DECISION_PATTERNS:
                for match in pattern.finditer(msg.text_content):
                    context = _sentence_around_match(msg.text_content, match)
                    title = truncate(match.group(0))
                    memories.append(Memory(
                        session_id=session_id,
                        project_path=project_path,
                        memory_type=MemoryType.DECISION,
                        title=title,
                        content=context,
                        tags=self._auto_tags(context),
                        source_line_start=msg.index,
                        source_line_end=msg.index,
                        confidence=0.7,
                    ))
        return memories

    def _extract_file_patterns(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> list[Memory]:
        """Track Write/Edit tool uses and group by directory to find patterns."""
        dir_files: dict[str, list[str]] = {}
        for msg in messages:
            for tu in msg.tool_uses:
                if tu.name not in ("Write", "Edit"):
                    continue
                fpath = _file_path_from_tool(tu)
                if fpath is None:
                    continue
                parent = _directory_of(fpath)
                dir_files.setdefault(parent, []).append(fpath)

        memories: list[Memory] = []
        for directory, files in dir_files.items():
            if len(files) < 2:
                continue
            unique_files = sorted(set(files))
            title = truncate(f"Modified {len(unique_files)} files in {directory}")
            content_lines = [f"Directory: {directory}", f"Files ({len(unique_files)}):"]
            for f in unique_files:
                content_lines.append(f"  - {f}")
            memories.append(Memory(
                session_id=session_id,
                project_path=project_path,
                memory_type=MemoryType.PATTERN,
                title=title,
                content="\n".join(content_lines),
                tags=["file-pattern", project_name_from_path(project_path)],
                confidence=0.9,
            ))
        return memories

    def _extract_todos(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> list[Memory]:
        """Extract TODO / future-work items from messages and tool uses."""
        memories: list[Memory] = []

        for msg in messages:
            # Check for TaskCreate tool uses
            for tu in msg.tool_uses:
                if tu.name == "TaskCreate":
                    desc = tu.input_data.get("description", "") or tu.input_data.get("task", "")
                    if desc:
                        memories.append(Memory(
                            session_id=session_id,
                            project_path=project_path,
                            memory_type=MemoryType.TODO,
                            title=truncate(desc),
                            content=desc,
                            tags=["task"],
                            source_line_start=msg.index,
                            source_line_end=msg.index,
                            confidence=0.95,
                        ))

            # Scan text for TODO-like patterns
            if not msg.text_content:
                continue
            for pattern in _TODO_PATTERNS:
                for match in pattern.finditer(msg.text_content):
                    captured = match.group(1).strip()
                    # Filter out very short or clearly non-actionable hits
                    if len(captured) < 10:
                        continue
                    context = _sentence_around_match(msg.text_content, match, context_chars=150)
                    memories.append(Memory(
                        session_id=session_id,
                        project_path=project_path,
                        memory_type=MemoryType.TODO,
                        title=truncate(captured),
                        content=context,
                        tags=self._auto_tags(context),
                        source_line_start=msg.index,
                        source_line_end=msg.index,
                        confidence=0.6,
                    ))
        return memories

    def _extract_errors_and_fixes(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> list[Memory]:
        """Find Bash errors followed by resolution within the next few messages."""
        memories: list[Memory] = []

        for i, msg in enumerate(messages):
            if msg.msg_type != "assistant":
                continue

            # Look for Bash tool uses with error output
            error_tools = [
                tu for tu in msg.tool_uses
                if tu.name == "Bash" and _has_error(tu.output)
            ]
            if not error_tools:
                # Also check the text content for error mentions when tools
                # don't have output attached (common when results aren't
                # linked back).
                has_bash = any(tu.name == "Bash" for tu in msg.tool_uses)
                if has_bash and _has_error(msg.text_content):
                    error_tools = [tu for tu in msg.tool_uses if tu.name == "Bash"]
                if not error_tools:
                    continue

            error_text = error_tools[0].output or msg.text_content

            # Look ahead for the fix
            fix_text: str | None = None
            fix_index: int | None = None
            for j in range(i + 1, min(i + 1 + self.ERROR_FIX_LOOKAHEAD, len(messages))):
                candidate = messages[j]
                if candidate.msg_type != "assistant":
                    continue
                # A fix is indicated by a successful Bash run, or an Edit/Write
                has_fix_tool = any(
                    tu.name in ("Bash", "Edit", "Write") for tu in candidate.tool_uses
                )
                if has_fix_tool and not _has_error(candidate.text_content):
                    fix_text = candidate.text_content
                    fix_index = candidate.index
                    break

            # Record the error (and fix if found)
            error_title = truncate(error_text.split("\n")[0] if error_text else "Error encountered")

            memories.append(Memory(
                session_id=session_id,
                project_path=project_path,
                memory_type=MemoryType.ISSUE,
                title=error_title,
                content=truncate(error_text, max_length=500),
                tags=["error", "bash"],
                source_line_start=msg.index,
                source_line_end=msg.index,
                confidence=0.75,
            ))

            if fix_text and fix_index is not None:
                fix_title = truncate(f"Fix: {error_title}")
                memories.append(Memory(
                    session_id=session_id,
                    project_path=project_path,
                    memory_type=MemoryType.SOLUTION,
                    title=fix_title,
                    content=truncate(fix_text, max_length=500),
                    tags=["fix", "bash"],
                    source_line_start=msg.index,
                    source_line_end=fix_index,
                    confidence=0.65,
                ))

        return memories

    def _extract_preferences(
        self,
        messages: list[ParsedMessage],
        session_id: str,
        project_path: str,
    ) -> list[Memory]:
        """Extract user style/tool preferences."""
        memories: list[Memory] = []
        for msg in messages:
            # Only look at user messages for preferences
            if msg.role != "user" or not msg.text_content:
                continue
            for pattern in _PREFERENCE_PATTERNS:
                for match in pattern.finditer(msg.text_content):
                    context = _sentence_around_match(msg.text_content, match)
                    title = truncate(match.group(0))
                    memories.append(Memory(
                        session_id=session_id,
                        project_path=project_path,
                        memory_type=MemoryType.PREFERENCE,
                        title=title,
                        content=context,
                        tags=["preference"],
                        source_line_start=msg.index,
                        source_line_end=msg.index,
                        confidence=0.8,
                    ))
        return memories

    # ----- deduplication --------------------------------------------------- #

    def _deduplicate(self, memories: list[Memory]) -> list[Memory]:
        """Remove near-duplicate memories based on content hashing."""
        seen: dict[str, Memory] = {}
        unique: list[Memory] = []
        for mem in memories:
            h = content_hash(mem.content)
            if h in seen:
                existing = seen[h]
                # Keep the one with higher confidence
                if mem.confidence > existing.confidence:
                    unique = [m for m in unique if m is not existing]
                    unique.append(mem)
                    seen[h] = mem
            else:
                seen[h] = mem
                unique.append(mem)
        return unique

    # ----- auto tagging ---------------------------------------------------- #

    @staticmethod
    def _auto_tags(text: str, max_tags: int = 5) -> list[str]:
        """Generate simple tags from text via word frequency."""
        words = _WORD_RE.findall(text.lower())
        filtered = [w for w in words if w not in _STOPWORDS and len(w) >= _MIN_WORD_LEN]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(max_tags)]

    # ----- topic extraction ------------------------------------------------ #

    @staticmethod
    def _extract_topics(user_messages: list[ParsedMessage], max_topics: int = 10) -> list[str]:
        """Extract key topics from user messages via word frequency analysis."""
        word_counter: Counter[str] = Counter()
        for msg in user_messages:
            if not msg.text_content:
                continue
            words = _WORD_RE.findall(msg.text_content.lower())
            filtered = [w for w in words if w not in _STOPWORDS and len(w) >= _MIN_WORD_LEN]
            word_counter.update(filtered)

        # Return most common words as topic labels
        return [word for word, _ in word_counter.most_common(max_topics)]
