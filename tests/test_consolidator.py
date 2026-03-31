"""Tests for the memory consolidation engine."""

from datetime import datetime, timedelta, timezone

from claude_memory.consolidator import (
    ConsolidationReport,
    MemoryConsolidator,
    _jaccard_similarity,
    _word_set,
)
from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_memory(
    session_id: str = "s1",
    project_path: str = "/tmp/project",
    memory_type: MemoryType = MemoryType.DECISION,
    title: str = "Test memory",
    content: str = "Some content",
    tags: list[str] | None = None,
    confidence: float = 1.0,
    created_at: datetime | None = None,
) -> Memory:
    """Convenience factory for test memories."""
    mem = Memory(
        session_id=session_id,
        project_path=project_path,
        memory_type=memory_type,
        title=title,
        content=content,
        tags=tags or [],
        confidence=confidence,
    )
    if created_at is not None:
        mem.created_at = created_at
        mem.updated_at = created_at
    return mem


# ── Scoring tests ────────────────────────────────────────────────────────────


def test_score_memories(tmp_db: MemoryDB):
    """Insert memories, score them, verify scores are reasonable."""
    m1 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI",
        content="We decided to use FastAPI.",
        tags=["api", "python"],
        confidence=1.0,
    )
    m2 = _make_memory(
        memory_type=MemoryType.CONTEXT,
        title="Project overview",
        content="Overview of the project.",
        confidence=0.5,
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    consolidator = MemoryConsolidator(tmp_db)
    scored = consolidator.score_memories()
    assert scored == 2

    score1 = tmp_db.get_importance_score(m1.id)
    score2 = tmp_db.get_importance_score(m2.id)

    # Decision with tags and full confidence should score higher
    assert score1 > score2
    # Both should be positive
    assert score1 > 0
    assert score2 > 0


def test_score_recency_decay(tmp_db: MemoryDB):
    """Older memories should have lower scores than recent ones, all else equal."""
    now = datetime.now(timezone.utc)
    recent = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Recent decision",
        content="Very recent.",
        created_at=now,
    )
    old = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Old decision",
        content="Very old.",
        created_at=now - timedelta(days=90),
    )
    tmp_db.insert_memory(recent)
    tmp_db.insert_memory(old)

    consolidator = MemoryConsolidator(tmp_db)
    consolidator.score_memories()

    score_recent = tmp_db.get_importance_score(recent.id)
    score_old = tmp_db.get_importance_score(old.id)
    assert score_recent > score_old


# ── Duplicate detection ──────────────────────────────────────────────────────


def test_find_duplicates(tmp_db: MemoryDB):
    """Insert near-duplicates, verify they're found."""
    m1 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI for REST API",
        content="We decided to use FastAPI for the REST API because it is modern.",
    )
    m2 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI for REST API",
        content="We decided to use FastAPI for the REST API because it is modern and fast.",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    consolidator = MemoryConsolidator(tmp_db)
    # Score first so duplicates have scores
    consolidator.score_memories()
    pairs = consolidator.find_duplicates()
    assert len(pairs) == 1
    keep, remove = pairs[0]
    # Both should be actual Memory objects
    assert keep.id in (m1.id, m2.id)
    assert remove.id in (m1.id, m2.id)
    assert keep.id != remove.id


def test_find_no_false_positives(tmp_db: MemoryDB):
    """Different memories are NOT flagged as duplicates."""
    m1 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI",
        content="FastAPI for the web framework.",
    )
    m2 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use PostgreSQL database",
        content="PostgreSQL for persistent storage with migrations.",
    )
    m3 = _make_memory(
        memory_type=MemoryType.TODO,
        title="Write unit tests",
        content="Need to write comprehensive unit tests for all modules.",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    consolidator = MemoryConsolidator(tmp_db)
    consolidator.score_memories()
    pairs = consolidator.find_duplicates()
    assert len(pairs) == 0


def test_duplicates_require_same_type(tmp_db: MemoryDB):
    """Near-duplicates of different types should NOT be flagged."""
    m1 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI for REST API",
        content="We decided to use FastAPI for the REST API.",
    )
    m2 = _make_memory(
        memory_type=MemoryType.SOLUTION,
        title="Use FastAPI for REST API",
        content="We decided to use FastAPI for the REST API.",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    consolidator = MemoryConsolidator(tmp_db)
    consolidator.score_memories()
    pairs = consolidator.find_duplicates()
    assert len(pairs) == 0


# ── Archive stale TODOs ──────────────────────────────────────────────────────


def test_archive_stale_todos(tmp_db: MemoryDB):
    """Old TODOs get archived (type changed to context)."""
    now = datetime.now(timezone.utc)
    old_todo = _make_memory(
        memory_type=MemoryType.TODO,
        title="Old TODO",
        content="This is an old TODO.",
        created_at=now - timedelta(days=60),
    )
    tmp_db.insert_memory(old_todo)

    consolidator = MemoryConsolidator(tmp_db)
    archived = consolidator.archive_stale_todos(days=30)
    assert len(archived) == 1
    assert archived[0].id == old_todo.id

    # Verify in DB the type was changed
    refreshed = tmp_db.get_memory(old_todo.id)
    assert refreshed is not None
    assert refreshed.memory_type == MemoryType.CONTEXT


def test_archive_recent_todos(tmp_db: MemoryDB):
    """Recent TODOs are kept (not archived)."""
    now = datetime.now(timezone.utc)
    recent_todo = _make_memory(
        memory_type=MemoryType.TODO,
        title="Recent TODO",
        content="This is a recent TODO.",
        created_at=now - timedelta(days=5),
    )
    tmp_db.insert_memory(recent_todo)

    consolidator = MemoryConsolidator(tmp_db)
    archived = consolidator.archive_stale_todos(days=30)
    assert len(archived) == 0

    # Verify type unchanged
    refreshed = tmp_db.get_memory(recent_todo.id)
    assert refreshed is not None
    assert refreshed.memory_type == MemoryType.TODO


# ── Merge duplicates ─────────────────────────────────────────────────────────


def test_merge_duplicates(tmp_db: MemoryDB):
    """Verify lower-scored duplicate is deleted."""
    m1 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI for REST API",
        content="We decided to use FastAPI for the REST API because it is modern.",
    )
    m2 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Use FastAPI for REST API",
        content="We decided to use FastAPI for the REST API because it is modern and fast.",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    consolidator = MemoryConsolidator(tmp_db)
    consolidator.score_memories()
    pairs = consolidator.find_duplicates()
    assert len(pairs) == 1

    merged_count = consolidator.merge_duplicates(pairs)
    assert merged_count == 1

    # Only one memory should remain
    assert tmp_db.count_memories() == 1


# ── Full pipeline ────────────────────────────────────────────────────────────


def test_consolidation_report(tmp_db: MemoryDB):
    """Full pipeline produces valid report."""
    now = datetime.now(timezone.utc)

    # Add some normal memories
    m1 = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Architecture choice",
        content="Decided on microservices architecture.",
        tags=["architecture"],
    )
    # Add a duplicate pair
    m2 = _make_memory(
        memory_type=MemoryType.PATTERN,
        title="Logging pattern used everywhere",
        content="We use structured logging with JSON format in all services.",
    )
    m3 = _make_memory(
        memory_type=MemoryType.PATTERN,
        title="Logging pattern used everywhere",
        content="We use structured logging with JSON format in all services and modules.",
    )
    # Add a stale TODO
    m4 = _make_memory(
        memory_type=MemoryType.TODO,
        title="Fix old bug",
        content="Need to fix that old bug.",
        created_at=now - timedelta(days=60),
    )

    for m in [m1, m2, m3, m4]:
        tmp_db.insert_memory(m)

    consolidator = MemoryConsolidator(tmp_db)
    report = consolidator.consolidate()

    assert isinstance(report, ConsolidationReport)
    assert report.memories_scored > 0
    assert report.duplicates_found >= 1
    assert report.duplicates_merged >= 1
    assert report.todos_archived >= 1
    assert isinstance(report.top_memories, list)


# ── DB-level importance score tests ──────────────────────────────────────────


def test_importance_score_db(tmp_db: MemoryDB):
    """Verify score persists in DB."""
    mem = _make_memory(
        memory_type=MemoryType.DECISION,
        title="Test score",
        content="Testing importance score persistence.",
    )
    tmp_db.insert_memory(mem)

    # Default score should be 0.0
    assert tmp_db.get_importance_score(mem.id) == 0.0

    # Update and verify
    tmp_db.update_importance_score(mem.id, 0.85)
    assert tmp_db.get_importance_score(mem.id) == 0.85

    # Verify it persists after re-fetching
    score = tmp_db.get_importance_score(mem.id)
    assert abs(score - 0.85) < 1e-6


def test_top_memories(tmp_db: MemoryDB):
    """Verify top command returns sorted results."""
    m1 = _make_memory(title="Low score", content="Low importance.")
    m2 = _make_memory(title="High score", content="High importance.")
    m3 = _make_memory(title="Mid score", content="Medium importance.")

    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    tmp_db.update_importance_score(m1.id, 0.1)
    tmp_db.update_importance_score(m2.id, 0.9)
    tmp_db.update_importance_score(m3.id, 0.5)

    top = tmp_db.get_top_memories(limit=3)
    assert len(top) == 3
    # Verify descending order by checking scores
    scores = [tmp_db.get_importance_score(m.id) for m in top]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 0.9
    assert scores[1] == 0.5
    assert scores[2] == 0.1


def test_top_memories_with_project_filter(tmp_db: MemoryDB):
    """Top memories can be filtered by project."""
    m1 = _make_memory(project_path="/tmp/projA", title="A mem", content="A")
    m2 = _make_memory(project_path="/tmp/projB", title="B mem", content="B")

    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    tmp_db.update_importance_score(m1.id, 0.9)
    tmp_db.update_importance_score(m2.id, 0.8)

    top_a = tmp_db.get_top_memories(project_path="/tmp/projA", limit=10)
    assert len(top_a) == 1
    assert top_a[0].id == m1.id


# ── Jaccard similarity unit tests ────────────────────────────────────────────


def test_jaccard_identical():
    """Identical sets have similarity 1.0."""
    assert _jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint():
    """Disjoint sets have similarity 0.0."""
    assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    """Partial overlap gives expected value."""
    sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
    # intersection = {b, c} = 2, union = {a, b, c, d} = 4 → 0.5
    assert abs(sim - 0.5) < 1e-6


def test_jaccard_empty():
    """Both empty sets give similarity 1.0."""
    assert _jaccard_similarity(set(), set()) == 1.0


def test_word_set():
    """word_set correctly tokenises and lowercases."""
    words = _word_set("Hello World hello")
    assert words == {"hello", "world"}
