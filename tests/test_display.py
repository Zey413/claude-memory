"""Tests for the rich display module."""

from datetime import datetime, timezone

from claude_memory.display import (
    TYPE_STYLES,
    display_memory_table,
    display_search_results,
    display_sessions,
    display_stats,
    render_to_string,
)
from claude_memory.models import Memory, MemoryType, SearchResult, SessionSummary


def _make_memory(**overrides) -> Memory:
    """Create a sample memory for tests."""
    defaults = {
        "session_id": "test-session-001",
        "project_path": "/tmp/test-project",
        "memory_type": MemoryType.DECISION,
        "title": "Use pytest for testing",
        "content": "Decided to use pytest as the main testing framework.",
        "tags": ["testing", "tooling"],
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return Memory(**defaults)


def test_display_memory_table():
    """Verify memory table renders without error."""
    memories = [
        _make_memory(),
        _make_memory(
            memory_type=MemoryType.TODO,
            title="Add rich output",
            tags=["cli"],
        ),
        _make_memory(
            memory_type=MemoryType.ISSUE,
            title="Broken import",
            tags=[],
        ),
    ]
    output = render_to_string(display_memory_table, memories, title="Test Memories")
    assert "Test Memories" in output
    assert "Use pytest" in output
    assert "Add rich output" in output
    assert "Broken import" in output
    assert "decision" in output
    assert "todo" in output
    assert "issue" in output


def test_display_stats():
    """Verify stats panel renders."""
    stats = {
        "total_memories": 42,
        "total_sessions": 7,
        "total_tags": 15,
        "db_size_bytes": 2048,
        "memories_by_type": {
            "decision": 10,
            "todo": 5,
        },
        "memories_by_project": {
            "/home/user/project-a": 20,
            "/home/user/project-b": 22,
        },
    }
    output = render_to_string(display_stats, stats)
    assert "Claude Memory Statistics" in output
    assert "42" in output
    assert "7" in output
    assert "15" in output
    assert "2.0 KB" in output
    assert "decision" in output
    assert "todo" in output
    assert "project-a" in output
    assert "project-b" in output


def test_display_sessions():
    """Verify session table renders."""
    sessions = [
        SessionSummary(
            session_id="abc12345-def6-7890-abcd-ef1234567890",
            project_path="/tmp/test-project",
            git_branch="main",
            started_at=datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc),
            ended_at=datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc),
            duration_minutes=30.0,
            message_count=50,
            summary_text="Worked on CLI improvements",
            key_topics=["cli", "testing"],
        ),
        SessionSummary(
            session_id="zzz99999-aaa0-1111-bbbb-cc2222222222",
            project_path="/tmp/test-project",
            started_at=datetime(2025, 1, 16, 14, 0, tzinfo=timezone.utc),
            message_count=20,
            summary_text="Fixed database bugs",
            key_topics=["database"],
        ),
    ]
    output = render_to_string(display_sessions, sessions)
    assert "Sessions" in output
    assert "abc12345" in output
    assert "zzz99999" in output
    assert "main" in output
    assert "CLI improvements" in output
    assert "database" in output


def test_display_search_results():
    """Verify search results render."""
    results = [
        SearchResult(
            memory=_make_memory(title="Auth decision"),
            score=0.95,
        ),
        SearchResult(
            memory=_make_memory(
                memory_type=MemoryType.PATTERN,
                title="Auth pattern",
            ),
            score=0.72,
        ),
    ]
    output = render_to_string(display_search_results, results, "auth")
    assert "auth" in output
    assert "Auth decision" in output
    assert "Auth pattern" in output
    assert "0.95" in output
    assert "0.72" in output


def test_type_styles_complete():
    """All MemoryType values have styles defined."""
    for mt in MemoryType:
        assert mt.value in TYPE_STYLES, f"Missing style for MemoryType.{mt.name} ({mt.value})"
