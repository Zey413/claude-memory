"""Tests for the search engine."""


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


# ── Expanded tests ────────────────────────────────────────────────────────────


def test_empty_query(tmp_db):
    """Empty string search should not crash and returns a list."""
    mem = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Some memory", content="Some content here.",
    )
    tmp_db.insert_memory(mem)

    search = MemorySearch(tmp_db)
    results = search.search("")
    assert isinstance(results, list)


def test_search_by_type(tmp_db):
    """Search with memory_type filter only returns matching type."""
    d = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Decision about architecture",
        content="We decided on microservices architecture.",
        tags=["architecture"],
    )
    t = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO,
        title="Architecture refactoring needed",
        content="Need to refactor the architecture layer.",
        tags=["architecture"],
    )
    tmp_db.insert_memory(d)
    tmp_db.insert_memory(t)

    search = MemorySearch(tmp_db)
    results = search.search("architecture", memory_type=MemoryType.DECISION)
    assert len(results) >= 1
    assert all(r.memory.memory_type == MemoryType.DECISION for r in results)


def test_related_memories(tmp_db):
    """related() returns memories linked by session, tags, or project+type."""
    # Two memories in the same session
    m1 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Use FastAPI", content="Decided on FastAPI.",
        tags=["api", "python"],
    )
    m2 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Use PostgreSQL", content="Decided on PostgreSQL.",
        tags=["database"],
    )
    # Different session, overlapping tag
    m3 = Memory(
        session_id="s2", project_path="/tmp/p1",
        memory_type=MemoryType.PATTERN,
        title="API structure", content="RESTful patterns.",
        tags=["api"],
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    search = MemorySearch(tmp_db)
    related = search.related(m1.id)
    related_ids = {r.id for r in related}
    # m2 shares session_id, m3 shares "api" tag
    assert m2.id in related_ids
    assert m3.id in related_ids
    # m1 itself should not be in the results
    assert m1.id not in related_ids


def test_related_memories_not_found(tmp_db):
    """related() returns empty list for non-existent memory."""
    search = MemorySearch(tmp_db)
    related = search.related("nonexistent")
    assert related == []


def test_by_tag(tmp_db):
    """by_tag returns memories that carry a specific tag."""
    m1 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Tagged A", content="Content A",
        tags=["important"],
    )
    m2 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.TODO,
        title="Tagged B", content="Content B",
        tags=["trivial"],
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    search = MemorySearch(tmp_db)
    results = search.by_tag("important")
    assert len(results) == 1
    assert results[0].title == "Tagged A"


def test_by_type(tmp_db):
    """by_type returns memories of the specified type for a project."""
    m = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.PATTERN,
        title="Code pattern", content="We use repository pattern.",
    )
    tmp_db.insert_memory(m)

    search = MemorySearch(tmp_db)
    patterns = search.by_type("/tmp/p1", MemoryType.PATTERN)
    assert len(patterns) == 1
    assert patterns[0].memory_type == MemoryType.PATTERN

    # Different type should be empty
    todos = search.by_type("/tmp/p1", MemoryType.TODO)
    assert len(todos) == 0


def test_search_with_project_filter(tmp_db):
    """Search scoped to a specific project."""
    m1 = Memory(
        session_id="s1", project_path="/tmp/p1",
        memory_type=MemoryType.DECISION,
        title="Python Flask app", content="Using Flask framework.",
    )
    m2 = Memory(
        session_id="s1", project_path="/tmp/p2",
        memory_type=MemoryType.DECISION,
        title="Python Django app", content="Using Django framework.",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    search = MemorySearch(tmp_db)
    results = search.search("Python", project_path="/tmp/p1")
    assert len(results) >= 1
    assert all(r.memory.project_path == "/tmp/p1" for r in results)
