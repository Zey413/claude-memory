"""Tests for the CLAUDE.md generator."""

from datetime import datetime, timezone

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType, SessionSummary
from claude_memory.generator import ClaudemdGenerator
from claude_memory.search import MemorySearch


def _populate_test_data(db: MemoryDB):
    """Insert some test data into the database."""
    # Add memories
    memories = [
        Memory(
            session_id="s1", project_path="/tmp/test-proj",
            memory_type=MemoryType.DECISION,
            title="Use FastAPI", content="Decided to use FastAPI for the REST API.",
            tags=["api"],
        ),
        Memory(
            session_id="s1", project_path="/tmp/test-proj",
            memory_type=MemoryType.TODO,
            title="Add authentication", content="Need to add JWT auth.",
            tags=["auth"],
        ),
        Memory(
            session_id="s1", project_path="/tmp/test-proj",
            memory_type=MemoryType.PATTERN,
            title="Repository pattern", content="Using repository pattern for DB access.",
            tags=["architecture"],
        ),
    ]
    for m in memories:
        db.insert_memory(m)

    # Add session
    summary = SessionSummary(
        session_id="s1",
        project_path="/tmp/test-proj",
        started_at=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 28, 11, 0, tzinfo=timezone.utc),
        duration_minutes=60.0,
        message_count=50,
        user_message_count=20,
        assistant_message_count=30,
        summary_text="Built REST API with FastAPI.",
    )
    db.insert_session(summary)


def test_generate_project_context(tmp_db):
    """Test generating CLAUDE.md content."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    content = gen.generate_project_context("/tmp/test-proj")
    assert "test-proj" in content
    assert "Key Decisions" in content
    assert "FastAPI" in content
    assert "Active TODOs" in content
    assert "authentication" in content


def test_build_project_context(tmp_db):
    """Test building a ProjectContext model."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    ctx = gen.build_project_context("/tmp/test-proj")
    assert ctx.project_name == "test-proj"
    assert ctx.total_memories == 3
    assert ctx.total_sessions == 1
    assert len(ctx.key_decisions) >= 1


def test_render_to_string(tmp_db):
    """Test render_to_string returns non-empty markdown."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    output = gen.render_to_string("/tmp/test-proj")
    assert len(output) > 0
    assert "test-proj" in output


def test_write_to_memory_dir(tmp_db, tmp_path):
    """Test writing to memory directory."""
    _populate_test_data(tmp_db)

    from claude_memory.config import MemoryConfig
    config = MemoryConfig(claude_home=tmp_path / ".claude")

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    path = gen.write_to_memory_dir("/tmp/test-proj", config=config)
    assert path.exists()
    content = path.read_text()
    assert "test-proj" in content
