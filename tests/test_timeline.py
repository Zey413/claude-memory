"""Tests for the session timeline and replay functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_memory.display import display_timeline, render_to_string
from claude_memory.timeline import TimelineBuilder

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_session_jsonl(tmp_path: Path, messages: list[dict], name: str = "test-session") -> Path:
    """Write a list of message dicts as JSONL and return the path."""
    filepath = tmp_path / f"{name}.jsonl"
    with filepath.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return filepath


# ── Tests ────────────────────────────────────────────────────────────────────


def test_build_from_jsonl(sample_jsonl):
    """Build a timeline from the sample_session fixture."""
    filepath, session_id = sample_jsonl
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert timeline.session_id == session_id
    assert len(timeline.events) > 0
    # Should have user messages and tool events
    event_types = {e.event_type for e in timeline.events}
    assert "user_message" in event_types


def test_timeline_events(sample_jsonl):
    """Verify events are extracted correctly from the sample fixture."""
    filepath, _ = sample_jsonl
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    # sample_jsonl has: 2 user messages (non-meta), 2 Write tools, 1 Bash tool, 1 TaskCreate
    user_events = [e for e in timeline.events if e.event_type == "user_message"]
    assert len(user_events) == 2

    file_write_events = [e for e in timeline.events if e.event_type == "file_write"]
    assert len(file_write_events) == 2

    bash_events = [e for e in timeline.events if e.event_type == "bash_command"]
    assert len(bash_events) == 1


def test_user_message_event(tmp_path):
    """User messages become user_message events with truncated summary."""
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello world, this is a test"},
            "timestamp": "2026-03-28T10:00:00Z",
            "isMeta": False,
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.event_type == "user_message"
    assert "Hello world" in event.summary
    assert event.timestamp is not None


def test_tool_use_events(tmp_path):
    """Write/Edit/Read/Bash all become events with correct types."""
    messages = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Write",
                         "input": {"file_path": "/tmp/a.py", "content": "x"}},
                ],
            },
            "timestamp": "2026-03-28T10:01:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Edit",
                         "input": {"file_path": "/tmp/b.py",
                                   "old_string": "x", "new_string": "y"}},
                ],
            },
            "timestamp": "2026-03-28T10:02:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/c.py"}},
                ],
            },
            "timestamp": "2026-03-28T10:03:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
                ],
            },
            "timestamp": "2026-03-28T10:04:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    types = [e.event_type for e in timeline.events]
    assert "file_write" in types
    assert "file_edit" in types
    assert "file_read" in types
    assert "bash_command" in types


def test_file_write_event(tmp_path):
    """Write tool creates a file_write event with file info."""
    messages = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "/tmp/project/main.py", "content": "print('hello')"},
                    },
                ],
            },
            "timestamp": "2026-03-28T10:00:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.event_type == "file_write"
    assert "main.py" in event.summary
    assert "/tmp/project/main.py" in event.files


def test_bash_command_event(tmp_path):
    """Bash tool creates a bash_command event with command summary."""
    messages = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "python -m pytest tests/ -v --tb=short"},
                    },
                ],
            },
            "timestamp": "2026-03-28T10:00:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.event_type == "bash_command"
    assert event.summary.startswith("$ ")
    assert "pytest" in event.summary


def test_timeline_properties(tmp_path):
    """Test duration, counts, and files_modified properties."""
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Start the project"},
            "timestamp": "2026-03-28T10:00:00Z",
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Write",
                         "input": {"file_path": "/tmp/a.py", "content": "x"}},
                ],
            },
            "timestamp": "2026-03-28T10:05:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Edit",
                         "input": {"file_path": "/tmp/b.py",
                                   "old_string": "x", "new_string": "y"}},
                ],
            },
            "timestamp": "2026-03-28T10:10:00Z",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Looks good"},
            "timestamp": "2026-03-28T10:10:00Z",
            "isMeta": False,
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert timeline.duration_minutes == pytest.approx(10.0)
    assert timeline.user_message_count == 2
    assert timeline.tool_use_count == 2
    assert "/tmp/a.py" in timeline.files_modified
    assert "/tmp/b.py" in timeline.files_modified


def test_activity_summary(tmp_path):
    """Activity summary returns correct stats."""
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2026-03-28T10:00:00Z",
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Write",
                         "input": {"file_path": "/tmp/f.py", "content": "x"}},
                ],
            },
            "timestamp": "2026-03-28T10:05:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "echo hello"}},
                ],
            },
            "timestamp": "2026-03-28T10:10:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)
    summary = builder.get_activity_summary(timeline)

    assert summary["session_id"] == filepath.stem
    assert summary["total_events"] == 3
    assert summary["user_messages"] == 1
    assert summary["tool_uses"] == 2
    assert summary["duration_minutes"] == pytest.approx(10.0)
    assert "/tmp/f.py" in summary["files_modified"]
    assert "user_message" in summary["event_counts"]
    assert "file_write" in summary["event_counts"]
    assert "bash_command" in summary["event_counts"]


def test_empty_session(tmp_path):
    """An empty JSONL file produces an empty timeline."""
    filepath = tmp_path / "empty-session.jsonl"
    filepath.write_text("")
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert timeline.session_id == "empty-session"
    assert len(timeline.events) == 0
    assert timeline.started_at is None
    assert timeline.ended_at is None
    assert timeline.duration_minutes is None
    assert timeline.user_message_count == 0
    assert timeline.tool_use_count == 0
    assert timeline.files_modified == []


def test_event_type_filter(tmp_path):
    """Filtering events by type works in display_timeline."""
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Do stuff"},
            "timestamp": "2026-03-28T10:00:00Z",
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            "timestamp": "2026-03-28T10:01:00Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Write",
                         "input": {"file_path": "/tmp/x.py", "content": "x"}},
                ],
            },
            "timestamp": "2026-03-28T10:02:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    # Filter to only bash_command
    output = render_to_string(display_timeline, timeline, limit=50, event_type="bash_command")
    assert "bash_command" in output
    # file_write should NOT appear in the table rows (but may appear in summary)
    # user_message should NOT appear in the table rows
    # We check the event table doesn't show them
    assert "user_message" not in output
    assert "file_write" not in output


def test_timeline_timestamps(tmp_path):
    """Start and end timestamps are set correctly from events."""
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "First"},
            "timestamp": "2026-03-28T09:00:00Z",
            "isMeta": False,
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Middle"},
            "timestamp": "2026-03-28T10:30:00Z",
            "isMeta": False,
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Last"},
            "timestamp": "2026-03-28T12:00:00Z",
            "isMeta": False,
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert timeline.started_at is not None
    assert timeline.ended_at is not None
    # started_at should be the earliest timestamp
    assert timeline.started_at.hour == 9
    assert timeline.started_at.minute == 0
    # ended_at should be the latest timestamp
    assert timeline.ended_at.hour == 12
    assert timeline.ended_at.minute == 0
    # Duration should be 3 hours = 180 minutes
    assert timeline.duration_minutes == pytest.approx(180.0)


def test_display_timeline_renders(tmp_path):
    """display_timeline renders without error and includes key elements."""
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Build a feature"},
            "timestamp": "2026-03-28T10:00:00Z",
            "isMeta": False,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Write",
                         "input": {"file_path": "/tmp/app.py", "content": "x"}},
                ],
            },
            "timestamp": "2026-03-28T10:05:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)
    timeline.project_path = "/tmp/test-project"

    output = render_to_string(display_timeline, timeline)
    assert "Session Timeline" in output
    assert "Summary" in output
    assert "user_message" in output or "user messages" in output
    assert "app.py" in output


def test_unknown_tool_becomes_tool_use(tmp_path):
    """Unknown tool types produce a generic tool_use event."""
    messages = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "CustomTool", "input": {"key": "value"}},
                ],
            },
            "timestamp": "2026-03-28T10:00:00Z",
        },
    ]
    filepath = _make_session_jsonl(tmp_path, messages)
    builder = TimelineBuilder()
    timeline = builder.build_from_jsonl(filepath)

    assert len(timeline.events) == 1
    event = timeline.events[0]
    assert event.event_type == "tool_use"
    assert "CustomTool" in event.summary
