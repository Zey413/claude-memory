"""Tests for the JSONL session parser."""

from claude_memory.parser import parse_session_file, parse_line


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
