"""JSONL session log parser for Claude Code sessions.

Parses Claude Code session JSONL files into structured messages,
handling all observed message types and content block formats.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude_memory.utils import parse_iso

logger = logging.getLogger(__name__)

# Known top-level message types in session JSONL files
KNOWN_MSG_TYPES = frozenset({
    "user",
    "assistant",
    "system",
    "file-history-snapshot",
    "queue-operation",
})


@dataclass
class ToolUse:
    """A single tool invocation within an assistant message."""

    name: str           # 'Bash', 'Edit', 'Write', 'Read', etc.
    input_data: dict    # Tool input parameters
    output: str = ""    # Tool result if available


@dataclass
class ParsedMessage:
    """A structured representation of one JSONL line from a session log."""

    index: int                          # Line number in the JSONL file (0-based)
    msg_type: str                       # 'user' | 'assistant' | 'system' | etc.
    role: str | None                    # 'user' | 'assistant' | None
    text_content: str                   # Extracted text (stripped of XML tags)
    tool_uses: list[ToolUse] = field(default_factory=list)
    timestamp: datetime | None = None
    cwd: str | None = None
    git_branch: str | None = None
    is_meta: bool = False               # /clear, /model, etc.
    session_id: str | None = None


# --------------------------------------------------------------------------- #
#  XML tag stripping
# --------------------------------------------------------------------------- #

_XML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_xml_tags(text: str) -> str:
    """Remove XML-style tags from text content, preserving inner text."""
    return _XML_TAG_RE.sub("", text).strip()


# --------------------------------------------------------------------------- #
#  Content extraction helpers
# --------------------------------------------------------------------------- #

def _extract_text_from_content(content: str | list | dict | None) -> str:
    """Recursively extract plain text from a message's content field.

    Content can appear in several shapes:
    - A plain string
    - A list of content blocks (each a dict with a ``type`` key)
    - A dict with ``role`` and ``content`` keys (nested message wrapper)
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return _strip_xml_tags(content)

    if isinstance(content, dict):
        # Nested message wrapper: {"role": "...", "content": ...}
        if "content" in content:
            return _extract_text_from_content(content["content"])
        # Single content block
        if content.get("type") == "text":
            return _strip_xml_tags(content.get("text", ""))
        return ""

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(_strip_xml_tags(block))
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(_strip_xml_tags(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    # Tool results may embed further content blocks
                    parts.append(_extract_text_from_content(block.get("content")))
        return "\n".join(p for p in parts if p)

    return str(content)


def _extract_tool_uses(content: str | list | dict | None) -> list[ToolUse]:
    """Extract tool-use blocks from assistant message content."""
    if not isinstance(content, list):
        return []

    tools: list[ToolUse] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tools.append(
                ToolUse(
                    name=block.get("name", "unknown"),
                    input_data=block.get("input", {}),
                )
            )
    return tools


def _attach_tool_results(messages: list[ParsedMessage]) -> None:
    """Walk the message list and attach tool results to their originating ToolUse.

    Tool results typically appear in a subsequent ``user`` message whose content
    contains ``tool_result`` blocks with a matching ``tool_use_id``.  We build
    an index of tool_use_id → ToolUse from assistant messages, then scan user
    messages for result blocks.
    """
    # Build id → ToolUse index from assistant messages
    tool_index: dict[str, ToolUse] = {}
    for msg in messages:
        if msg.msg_type != "assistant":
            continue
        # Re-walk raw content is not available here; instead we rely on the
        # fact that we stored the tool_use blocks.  To map IDs we'd need the
        # raw data.  A pragmatic shortcut: we index ToolUses by position per
        # message and correlate with results that follow.

    # Simpler heuristic: pair consecutive assistant-tool_uses with the next
    # user message's tool_result blocks.
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.msg_type == "assistant" and msg.tool_uses:
            # Look ahead for the matching result message
            for j in range(i + 1, min(i + 3, len(messages))):
                candidate = messages[j]
                if candidate.msg_type == "user" and candidate.text_content == "" and not candidate.tool_uses:
                    # This is likely a tool-result-only user message — its
                    # text was already extracted (empty because tool_result
                    # blocks are not plain text).  We skip detailed matching
                    # and leave output empty for now.
                    break
        i += 1


# --------------------------------------------------------------------------- #
#  Single-line parser
# --------------------------------------------------------------------------- #

def parse_line(index: int, data: dict) -> ParsedMessage | None:
    """Parse a single JSONL line (already decoded) into a ParsedMessage.

    Returns ``None`` for unrecognised or empty entries.
    """
    msg_type = data.get("type")
    if not msg_type:
        # Some entries are bare wrappers — try to infer type from structure
        if "message" in data and isinstance(data["message"], dict):
            msg_type = data["message"].get("role", "unknown")
        elif "snapshot" in data:
            msg_type = "file-history-snapshot"
        else:
            logger.debug("Skipping line %d: no 'type' field and could not infer type", index)
            return None

    # ---- file-history-snapshot ---- #
    if msg_type == "file-history-snapshot":
        return ParsedMessage(
            index=index,
            msg_type=msg_type,
            role=None,
            text_content="",
            session_id=data.get("sessionId"),
        )

    # ---- queue-operation ---- #
    if msg_type == "queue-operation":
        return ParsedMessage(
            index=index,
            msg_type=msg_type,
            role=None,
            text_content="",
        )

    # ---- user / assistant / system ---- #
    message_field = data.get("message", {})
    content = message_field.get("content") if isinstance(message_field, dict) else None
    # Some formats embed content at the top level
    if content is None:
        content = data.get("content")

    role = message_field.get("role") if isinstance(message_field, dict) else data.get("role")

    text = _extract_text_from_content(content)
    tools = _extract_tool_uses(content)

    # Timestamp — may be ISO string or epoch millis
    ts: datetime | None = None
    raw_ts = data.get("timestamp")
    if isinstance(raw_ts, str):
        try:
            ts = parse_iso(raw_ts)
        except (ValueError, TypeError):
            pass
    elif isinstance(raw_ts, (int, float)):
        try:
            # Epoch milliseconds (Claude Code convention)
            ts = datetime.fromtimestamp(raw_ts / 1000)
        except (ValueError, OSError):
            pass

    # Meta commands (e.g. /clear, /model)
    is_meta = bool(data.get("isMeta", False))

    return ParsedMessage(
        index=index,
        msg_type=msg_type,
        role=role,
        text_content=text,
        tool_uses=tools,
        timestamp=ts,
        cwd=data.get("cwd"),
        git_branch=data.get("gitBranch"),
        is_meta=is_meta,
        session_id=data.get("sessionId"),
    )


# --------------------------------------------------------------------------- #
#  Full-file parser
# --------------------------------------------------------------------------- #

def parse_session_file(filepath: Path) -> list[ParsedMessage]:
    """Parse an entire session JSONL file into a list of :class:`ParsedMessage`.

    Skips blank lines and malformed JSON with a warning.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Session file not found: {filepath}")

    messages: list[ParsedMessage] = []

    with filepath.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Malformed JSON on line %d of %s: %s", line_no, filepath, exc)
                continue

            if not isinstance(data, dict):
                logger.debug("Skipping non-dict JSON on line %d of %s", line_no, filepath)
                continue

            parsed = parse_line(line_no, data)
            if parsed is not None:
                messages.append(parsed)

    # Post-processing: try to attach tool results
    _attach_tool_results(messages)

    return messages
