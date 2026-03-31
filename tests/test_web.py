"""Tests for the FastAPI web dashboard backend."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from claude_memory.models import Memory, MemoryType, SessionSummary
from claude_memory.web import app as app_module
from claude_memory.web.app import app, init_app

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_app(tmp_path):
    """Initialise the app with a temporary database for each test."""
    db_path = tmp_path / "test_web.db"
    init_app(db_path)
    yield
    # Cleanup: close the DB
    if app_module._db is not None:
        app_module._db.close()
        app_module._db = None
        app_module._search = None


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def populated_db():
    """Insert sample data into the test database."""
    db = app_module._db
    assert db is not None

    # Insert a session
    session = SessionSummary(
        session_id="sess-001",
        project_path="/tmp/test-project",
        git_branch="main",
        started_at=datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 28, 11, 0, 0, tzinfo=timezone.utc),
        duration_minutes=60.0,
        message_count=20,
        user_message_count=8,
        assistant_message_count=12,
        summary_text="Built a REST API with FastAPI",
        key_topics=["fastapi", "rest", "api"],
        files_modified=["main.py", "test_main.py"],
    )
    db.insert_session(session)

    # Insert memories
    memories = [
        Memory(
            id="mem-001",
            session_id="sess-001",
            project_path="/tmp/test-project",
            memory_type=MemoryType.DECISION,
            title="Use FastAPI for REST API",
            content="Decided to use FastAPI for the REST API because it is modern and fast.",
            tags=["fastapi", "architecture"],
            confidence=0.9,
            created_at=datetime(2026, 3, 28, 10, 5, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 28, 10, 5, 0, tzinfo=timezone.utc),
        ),
        Memory(
            id="mem-002",
            session_id="sess-001",
            project_path="/tmp/test-project",
            memory_type=MemoryType.PATTERN,
            title="Pytest for testing",
            content="Use pytest with fixtures for all unit tests.",
            tags=["testing", "pytest"],
            confidence=1.0,
            created_at=datetime(2026, 3, 28, 10, 10, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 28, 10, 10, 0, tzinfo=timezone.utc),
        ),
        Memory(
            id="mem-003",
            session_id="sess-001",
            project_path="/tmp/other-project",
            memory_type=MemoryType.TODO,
            title="Add integration tests",
            content="Need to add integration tests for the API endpoints.",
            tags=["testing"],
            confidence=0.8,
            created_at=datetime(2026, 3, 28, 10, 15, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 28, 10, 15, 0, tzinfo=timezone.utc),
        ),
    ]
    for mem in memories:
        db.insert_memory(mem)

    return db


# ── Tests ─────────────────────────────────────────────────────────────────


def test_search_endpoint(client, populated_db):
    """Search returns results."""
    resp = client.get("/api/search", params={"q": "FastAPI"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    # Check result structure
    hit = data[0]
    assert "id" in hit
    assert "title" in hit
    assert "score" in hit
    assert "highlight" in hit


def test_search_with_type_filter(client, populated_db):
    """Search with type filter narrows results."""
    resp = client.get("/api/search", params={"q": "test", "type": "pattern"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    for item in data:
        assert item["memory_type"] == "pattern"


def test_search_missing_query(client, populated_db):
    """Search without query parameter returns 422."""
    resp = client.get("/api/search")
    assert resp.status_code == 422


def test_list_memories_endpoint(client, populated_db):
    """List returns array."""
    resp = client.get("/api/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3


def test_list_memories_with_project_filter(client, populated_db):
    """List with project filter returns only matching memories."""
    resp = client.get("/api/memories", params={"project": "/tmp/test-project"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert item["project_path"] == "/tmp/test-project"


def test_list_memories_with_offset(client, populated_db):
    """List with offset skips items."""
    resp = client.get("/api/memories", params={"limit": 2, "offset": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_stats_endpoint(client, populated_db):
    """Stats returns dict with expected keys."""
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "total_memories" in data
    assert "total_sessions" in data
    assert "total_tags" in data
    assert "memories_by_type" in data
    assert "memories_by_project" in data
    assert "db_size_bytes" in data
    assert data["total_memories"] == 3
    assert data["total_sessions"] == 1


def test_stats_with_project_filter(client, populated_db):
    """Stats with project filter returns project-specific counts."""
    resp = client.get("/api/stats", params={"project": "/tmp/test-project"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_memories"] == 2
    assert data["project_filter"] == "/tmp/test-project"


def test_sessions_endpoint(client, populated_db):
    """Sessions returns array."""
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    session = data[0]
    assert session["session_id"] == "sess-001"
    assert session["summary_text"] == "Built a REST API with FastAPI"


def test_projects_endpoint(client, populated_db):
    """Projects returns array."""
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    project = data[0]
    assert "project_path" in project
    assert "memory_count" in project
    assert "session_count" in project


def test_graph_endpoint(client, populated_db):
    """Graph returns nodes and edges."""
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    assert len(data["nodes"]) == 3  # 3 memories


def test_get_memory_endpoint(client, populated_db):
    """Single memory by ID."""
    resp = client.get("/api/memories/mem-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "mem-001"
    assert data["title"] == "Use FastAPI for REST API"
    assert data["memory_type"] == "decision"


def test_get_memory_not_found(client, populated_db):
    """Missing memory returns 404."""
    resp = client.get("/api/memories/nonexistent")
    assert resp.status_code == 404


def test_delete_memory_endpoint(client, populated_db):
    """Delete returns success."""
    resp = client.delete("/api/memories/mem-003")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert data["id"] == "mem-003"

    # Verify it's gone
    resp2 = client.get("/api/memories/mem-003")
    assert resp2.status_code == 404


def test_delete_memory_not_found(client, populated_db):
    """Deleting missing memory returns 404."""
    resp = client.delete("/api/memories/nonexistent")
    assert resp.status_code == 404


def test_top_memories_endpoint(client, populated_db):
    """Top returns sorted list."""
    resp = client.get("/api/top")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3
    for item in data:
        assert "importance_score" in item
        assert "title" in item
        assert "id" in item


def test_top_memories_with_limit(client, populated_db):
    """Top with limit returns correct count."""
    resp = client.get("/api/top", params={"limit": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1


def test_timeline_endpoint(client, populated_db):
    """Timeline returns session info and events."""
    resp = client.get("/api/timeline/sess-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-001"
    assert "session" in data
    assert "events" in data
    assert isinstance(data["events"], list)


def test_timeline_not_found(client, populated_db):
    """Timeline for missing session returns 404."""
    resp = client.get("/api/timeline/nonexistent")
    assert resp.status_code == 404


def test_dashboard_html(client):
    """GET / returns HTML."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Claude Memory Dashboard" in resp.text


def test_empty_database(client):
    """Endpoints work with an empty database."""
    resp = client.get("/api/memories")
    assert resp.status_code == 200
    assert resp.json() == []

    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []

    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_memories"] == 0


def test_graph_empty(client):
    """Graph with empty DB returns empty nodes and edges."""
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []
