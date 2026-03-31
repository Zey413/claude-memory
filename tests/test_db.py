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


# ── Expanded tests ────────────────────────────────────────────────────────────


def test_fts_special_characters(tmp_db):
    """FTS search handles special characters (quotes, asterisks) without crashing."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title='Handle "quoted" strings',
        content="We need to handle strings like 'hello' and wildcards like *.",
        tags=["strings"],
    )
    tmp_db.insert_memory(mem)

    # Should not raise — queries with special chars should be handled
    results = tmp_db.search_fts('"quoted"')
    assert isinstance(results, list)

    results2 = tmp_db.search_fts("wildcards")
    assert len(results2) >= 1


def test_large_content_memory(tmp_db):
    """Test storing and retrieving a memory with ~10KB of content."""
    large_content = "This is a line of large content.\n" * 300  # ~10KB
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.CONTEXT,
        title="Large context block",
        content=large_content,
    )
    mem_id = tmp_db.insert_memory(mem)
    retrieved = tmp_db.get_memory(mem_id)
    assert retrieved is not None
    assert len(retrieved.content) == len(large_content)
    assert retrieved.content == large_content


def test_get_memories_by_type(tmp_db):
    """Filter memories by type returns only matching type."""
    decision = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="A decision", content="Decided something",
    )
    todo = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO,
        title="A todo", content="Need to do something",
    )
    tmp_db.insert_memory(decision)
    tmp_db.insert_memory(todo)

    decisions = tmp_db.get_memories_by_type("/tmp/p1", MemoryType.DECISION)
    assert len(decisions) == 1
    assert decisions[0].memory_type == MemoryType.DECISION

    todos = tmp_db.get_memories_by_type("/tmp/p1", MemoryType.TODO)
    assert len(todos) == 1
    assert todos[0].memory_type == MemoryType.TODO


def test_get_memories_by_project(tmp_db):
    """Filter memories by project path."""
    m1 = Memory(
        session_id="s1", project_path="/tmp/projA",
        memory_type=MemoryType.TODO, title="A task", content="A",
    )
    m2 = Memory(
        session_id="s1", project_path="/tmp/projB",
        memory_type=MemoryType.TODO, title="B task", content="B",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    a_mems = tmp_db.get_memories_by_project("/tmp/projA")
    assert len(a_mems) == 1
    assert a_mems[0].title == "A task"

    b_mems = tmp_db.get_memories_by_project("/tmp/projB")
    assert len(b_mems) == 1
    assert b_mems[0].title == "B task"


def test_stats_with_data(tmp_db):
    """get_stats returns accurate data when database is populated."""
    # Insert some memories
    for i in range(3):
        tmp_db.insert_memory(Memory(
            session_id="s1", project_path="/tmp/p1",
            memory_type=MemoryType.DECISION,
            title=f"Decision {i}", content=f"Content {i}",
            tags=["tag-a"],
        ))
    tmp_db.insert_memory(Memory(
        session_id="s2", project_path="/tmp/p2",
        memory_type=MemoryType.TODO,
        title="A todo", content="Todo content",
        tags=["tag-b"],
    ))

    # Insert a session
    summary = SessionSummary(
        session_id="s1", project_path="/tmp/p1",
        started_at=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
        summary_text="Test session",
    )
    tmp_db.insert_session(summary)

    stats = tmp_db.get_stats()
    assert stats["total_memories"] == 4
    assert stats["total_sessions"] == 1
    assert stats["total_tags"] >= 2  # tag-a and tag-b
    assert stats["memories_by_type"]["decision"] == 3
    assert stats["memories_by_type"]["todo"] == 1
    assert stats["memories_by_project"]["/tmp/p1"] == 3
    assert stats["memories_by_project"]["/tmp/p2"] == 1
    assert stats["db_size_bytes"] > 0


def test_tag_operations(tmp_db):
    """Test add/get tag operations."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Tagged memory", content="Content",
        tags=["initial-tag"],
    )
    tmp_db.insert_memory(mem)

    # Check initial tag exists
    tags = tmp_db.get_all_tags()
    tag_names = [t.name for t in tags]
    assert "initial-tag" in tag_names

    # Add a new tag
    tmp_db.add_tag_to_memory(mem.id, "new-tag")
    tags = tmp_db.get_all_tags()
    tag_names = [t.name for t in tags]
    assert "new-tag" in tag_names

    # Remove a tag
    tmp_db.remove_tag_from_memory(mem.id, "initial-tag")
    # The tag still exists in the tags table, but is no longer linked
    retrieved = tmp_db.get_memory(mem.id)
    assert "initial-tag" not in retrieved.tags
    assert "new-tag" in retrieved.tags


def test_get_memory_not_found(tmp_db):
    """get_memory returns None for non-existent ID."""
    assert tmp_db.get_memory("nonexistent-id") is None


def test_delete_memory_not_found(tmp_db):
    """Deleting a non-existent memory returns False."""
    assert tmp_db.delete_memory("nonexistent-id") is False


def test_context_manager(tmp_path):
    """MemoryDB works as a context manager."""
    db_path = tmp_path / "ctx_mgr.db"
    with MemoryDB(db_path=db_path) as db:
        mem = Memory(
            session_id="s1", project_path="/tmp",
            memory_type=MemoryType.TODO, title="Test", content="C",
        )
        db.insert_memory(mem)
        assert db.count_memories() == 1
    # After context exits, db is closed — no crash
