"""Tests for the database layer."""

from datetime import datetime, timezone

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType, SessionSummary


def test_insert_and_get_memory(tmp_db):
    """Test inserting and retrieving a memory."""
    mem = Memory(
        session_id="session-1",
        project_path="/tmp/project",
        memory_type=MemoryType.DECISION,
        title="Use FastAPI",
        content="We decided to use FastAPI for the REST API.",
        tags=["api", "python"],
    )
    mem_id = tmp_db.insert_memory(mem)
    assert mem_id == mem.id

    retrieved = tmp_db.get_memory(mem.id)
    assert retrieved is not None
    assert retrieved.title == "Use FastAPI"
    assert retrieved.memory_type == MemoryType.DECISION
    assert "api" in retrieved.tags


def test_search_fts(tmp_db):
    """Test full-text search."""
    mem = Memory(
        session_id="session-1",
        project_path="/tmp/project",
        memory_type=MemoryType.DECISION,
        title="Use FastAPI framework",
        content="We chose FastAPI because it's fast and modern.",
        tags=["api"],
    )
    tmp_db.insert_memory(mem)

    results = tmp_db.search_fts("FastAPI")
    assert len(results) >= 1
    assert results[0].memory.title == "Use FastAPI framework"


def test_insert_session(tmp_db):
    """Test inserting a session summary."""
    summary = SessionSummary(
        session_id="session-1",
        project_path="/tmp/project",
        started_at=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 28, 11, 0, tzinfo=timezone.utc),
        duration_minutes=60.0,
        message_count=50,
        user_message_count=20,
        assistant_message_count=30,
        summary_text="Working on REST API.",
    )
    sid = tmp_db.insert_session(summary)
    assert sid == "session-1"

    retrieved = tmp_db.get_session("session-1")
    assert retrieved is not None
    assert retrieved.duration_minutes == 60.0


def test_count_memories(tmp_db):
    """Test memory counting."""
    assert tmp_db.count_memories() == 0

    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO, title="Test", content="Content",
    )
    tmp_db.insert_memory(mem)
    assert tmp_db.count_memories() == 1
    assert tmp_db.count_memories(project_path="/tmp/p1") == 1
    assert tmp_db.count_memories(project_path="/tmp/p2") == 0


def test_delete_memory(tmp_db):
    """Test memory deletion."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO, title="Delete me", content="Content",
    )
    tmp_db.insert_memory(mem)
    assert tmp_db.count_memories() == 1

    deleted = tmp_db.delete_memory(mem.id)
    assert deleted
    assert tmp_db.count_memories() == 0


def test_get_stats(tmp_db):
    """Test stats collection."""
    stats = tmp_db.get_stats()
    assert stats["total_memories"] == 0
    assert stats["total_sessions"] == 0


def test_reset(tmp_db):
    """Test database reset."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO, title="Test", content="Content",
    )
    tmp_db.insert_memory(mem)
    assert tmp_db.count_memories() == 1

    tmp_db.reset()
    assert tmp_db.count_memories() == 0


def test_session_processed_check(tmp_db):
    """Test session processing detection."""
    assert not tmp_db.is_session_processed("session-1")

    summary = SessionSummary(
        session_id="session-1",
        project_path="/tmp/project",
        started_at=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
        summary_text="Test",
    )
    tmp_db.insert_session(summary)
    assert tmp_db.is_session_processed("session-1")
