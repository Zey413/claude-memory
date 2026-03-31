"""Tests for the JSONL session parser."""

import json
from pathlib import Path

from claude_memory.parser import parse_line, parse_session_file


def test_parse_session_file(sample_jsonl):
    """Test parsing a complete session JSONL file."""
    filepath, session_id = sample_jsonl
    messages = parse_session_file(filepath)

    assert len(messages) >= 4  # At least 4 meaningful messages
    # Check user messages
    user_msgs = [m for m in messages if m.role == "user"]
    assert len(user_msgs) >= 2

    # Check assistant messages with tool uses
    asst_msgs = [m for m in messages if m.role == "assistant"]
    assert len(asst_msgs) >= 2

    # Check that Write tools were extracted
    all_tools = []
    for msg in asst_msgs:
        all_tools.extend(msg.tool_uses)
    write_tools = [t for t in all_tools if t.name == "Write"]
    assert len(write_tools) >= 2


def test_parse_line_user_message():
    """Test parsing a single user message."""
    data = {
        "type": "user",
        "message": {"role": "user", "content": "Hello world"},
        "timestamp": "2026-03-28T10:00:00Z",
        "cwd": "/tmp/project",
        "isMeta": False,
    }
    msg = parse_line(0, data)
    assert msg is not None
    assert msg.role == "user"
    assert "Hello world" in msg.text_content
    assert msg.cwd == "/tmp/project"
    assert not msg.is_meta


def test_parse_line_assistant_with_tools():
    """Test parsing an assistant message with tool_use blocks."""
    data = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me create a file."},
                {
                    "type": "tool_use",
                    "name": "Write",
                    "input": {"file_path": "/tmp/test.py", "content": "print('hello')"},
                },
            ],
        },
        "timestamp": "2026-03-28T10:01:00Z",
    }
    msg = parse_line(0, data)
    assert msg is not None
    assert msg.role == "assistant"
    assert "create a file" in msg.text_content
    assert len(msg.tool_uses) == 1
    assert msg.tool_uses[0].name == "Write"
    assert msg.tool_uses[0].input_data["file_path"] == "/tmp/test.py"


def test_parse_line_file_history_snapshot():
    """Test parsing a file-history-snapshot entry."""
    data = {"type": "file-history-snapshot", "snapshot": {"files": []}}
    msg = parse_line(0, data)
    assert msg is not None
    assert msg.msg_type == "file-history-snapshot"


def test_parse_line_malformed():
    """Test that malformed data returns None."""
    msg = parse_line(0, {})
    assert msg is None


def test_parse_session_file_not_found(tmp_path):
    """Test that FileNotFoundError is raised for missing files."""
    import pytest
    with pytest.raises(FileNotFoundError):
        parse_session_file(tmp_path / "nonexistent.jsonl")


# ── Expanded tests ────────────────────────────────────────────────────────────


def test_empty_jsonl_file(tmp_path):
    """Parsing an empty JSONL file returns an empty list."""
    filepath = tmp_path / "empty.jsonl"
    filepath.write_text("")
    messages = parse_session_file(filepath)
    assert messages == []


def test_blank_lines_skipped(tmp_path):
    """Blank lines and whitespace-only lines are skipped."""
    filepath = tmp_path / "blanks.jsonl"
    content = '\n\n{"type":"user","message":{"role":"user","content":"Hello"}}\n\n\n'
    filepath.write_text(content)
    messages = parse_session_file(filepath)
    assert len(messages) == 1
    assert messages[0].text_content == "Hello"


def test_meta_messages_skipped_from_content(tmp_path):
    """Messages with isMeta flag are parsed but flagged accordingly."""
    filepath = tmp_path / "meta.jsonl"
    lines = [
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "/clear"},
            "isMeta": True,
            "timestamp": "2026-03-28T10:00:00Z",
        }),
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "/model claude-3"},
            "isMeta": True,
            "timestamp": "2026-03-28T10:00:05Z",
        }),
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "Hello, real message"},
            "isMeta": False,
            "timestamp": "2026-03-28T10:00:10Z",
        }),
    ]
    filepath.write_text("\n".join(lines))
    messages = parse_session_file(filepath)

    meta_msgs = [m for m in messages if m.is_meta]
    non_meta = [m for m in messages if not m.is_meta]
    assert len(meta_msgs) == 2
    assert len(non_meta) == 1
    assert non_meta[0].text_content == "Hello, real message"


def test_various_timestamp_formats(tmp_path):
    """Parse both ISO string and epoch millisecond timestamps."""
    filepath = tmp_path / "timestamps.jsonl"
    lines = [
        # ISO format with Z
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "ISO timestamp"},
            "timestamp": "2026-03-28T10:00:00Z",
        }),
        # Epoch milliseconds
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "Epoch timestamp"},
            "timestamp": 1774965600000,  # approx 2026-03-28T10:00:00 UTC
        }),
        # No timestamp
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "No timestamp"},
        }),
    ]
    filepath.write_text("\n".join(lines))
    messages = parse_session_file(filepath)

    assert len(messages) == 3
    assert messages[0].timestamp is not None  # ISO parsed
    assert messages[1].timestamp is not None  # Epoch parsed
    assert messages[2].timestamp is None       # Missing


def test_parse_queue_operation():
    """Parse a queue-operation entry."""
    data = {"type": "queue-operation", "operation": "enqueue"}
    msg = parse_line(0, data)
    assert msg is not None
    assert msg.msg_type == "queue-operation"
    assert msg.text_content == ""


def test_parse_line_with_git_branch():
    """Git branch info is captured from the data."""
    data = {
        "type": "user",
        "message": {"role": "user", "content": "Fix bug"},
        "timestamp": "2026-03-28T10:00:00Z",
        "gitBranch": "feature/login",
    }
    msg = parse_line(0, data)
    assert msg is not None
    assert msg.git_branch == "feature/login"


def test_parse_error_fix_session_fixture():
    """Parse the error_fix_session.jsonl fixture file."""
    fixture = Path(__file__).parent / "fixtures" / "error_fix_session.jsonl"
    messages = parse_session_file(fixture)
    assert len(messages) >= 8

    # Should have both user and assistant messages
    user_msgs = [m for m in messages if m.role == "user"]
    asst_msgs = [m for m in messages if m.role == "assistant"]
    assert len(user_msgs) >= 3
    assert len(asst_msgs) >= 4

    # Should have tool uses
    all_tools = []
    for msg in messages:
        all_tools.extend(msg.tool_uses)
    tool_names = [t.name for t in all_tools]
    assert "Bash" in tool_names
    assert "Edit" in tool_names
