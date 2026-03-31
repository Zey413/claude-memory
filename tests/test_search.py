"""Tests for the search engine."""

from datetime import datetime, timezone

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType
from claude_memory.search import MemorySearch


def test_search_basic(tmp_db):
    """Test basic full-text search."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Choose PostgreSQL for database",
        content="We decided PostgreSQL is best for our needs.",
        tags=["database", "postgres"],
    )
    tmp_db.insert_memory(mem)

    search = MemorySearch(tmp_db)
    results = search.search("PostgreSQL")
    assert len(results) >= 1
    assert results[0].memory.id == mem.id


def test_search_with_tag_filter(tmp_db):
    """Test search with tag filtering."""
    mem1 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Use Redis for caching",
        content="Redis is a good caching solution.",
        tags=["caching", "redis"],
    )
    mem2 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Use Redis for sessions",
        content="Redis can store session data too.",
        tags=["session", "redis"],
    )
    tmp_db.insert_memory(mem1)
    tmp_db.insert_memory(mem2)

    search = MemorySearch(tmp_db)
    # Search with tag filter
    results = search.search("Redis", tags=["caching"])
    assert len(results) >= 1
    assert any("caching" in r.memory.tags for r in results)


def test_recent_memories(tmp_db):
    """Test recent memory retrieval."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO,
        title="Fix bug", content="Need to fix the login bug.",
    )
    tmp_db.insert_memory(mem)

    search = MemorySearch(tmp_db)
    recent = search.recent(days=7)
    assert len(recent) >= 1


def test_by_project(tmp_db):
    """Test filtering by project."""
    mem1 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO, title="P1 task", content="Task for P1",
    )
    mem2 = Memory(
        session_id="s1", project_path="/tmp/p2",
        memory_type=MemoryType.TODO, title="P2 task", content="Task for P2",
    )
    tmp_db.insert_memory(mem1)
    tmp_db.insert_memory(mem2)

    search = MemorySearch(tmp_db)
    p1_memories = search.by_project("/tmp/p1")
    assert len(p1_memories) == 1
    assert p1_memories[0].title == "P1 task"
