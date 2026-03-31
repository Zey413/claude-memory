"""Tests for the MCP server tools."""

from __future__ import annotations

import pytest

# Import the module-level state helpers and tools
from claude_memory import mcp_server
from claude_memory.models import Memory, MemoryType

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _init_mcp_db(tmp_path):
    """Initialise the MCP module-level DB for every test."""
    db_path = tmp_path / "mcp_test.db"
    mcp_server.init_db(db_path)
    yield
    # Cleanup
    if mcp_server._db is not None:
        mcp_server._db.close()
        mcp_server._db = None
        mcp_server._search = None


@pytest.fixture
def populated_db():
    """Insert a handful of memories into the shared MCP db."""
    db = mcp_server._get_db()

    memories = [
        Memory(
            session_id="sess-1",
            project_path="/tmp/project-alpha",
            memory_type=MemoryType.DECISION,
            title="Use FastAPI for REST layer",
            content="Decided to use FastAPI because of async support and automatic docs.",
            tags=["api", "fastapi"],
        ),
        Memory(
            session_id="sess-1",
            project_path="/tmp/project-alpha",
            memory_type=MemoryType.PATTERN,
            title="Repository pattern for DB access",
            content="All database access goes through repository classes.",
            tags=["database", "architecture"],
        ),
        Memory(
            session_id="sess-2",
            project_path="/tmp/project-alpha",
            memory_type=MemoryType.TODO,
            title="Add integration tests",
            content="Need to add integration tests for the API endpoints.",
            tags=["testing"],
        ),
        Memory(
            session_id="sess-3",
            project_path="/tmp/project-beta",
            memory_type=MemoryType.ISSUE,
            title="Memory leak in worker",
            content="Workers leak memory when processing large batches.",
            tags=["bug", "performance"],
        ),
    ]
    for m in memories:
        db.insert_memory(m)

    return memories


# ── Tests ────────────────────────────────────────────────────────────────────

class TestMemorySearch:
    """Tests for the memory_search MCP tool."""

    def test_basic_search(self, populated_db):
        results = mcp_server.memory_search(query="FastAPI")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["title"] == "Use FastAPI for REST layer"

    def test_search_with_type_filter(self, populated_db):
        results = mcp_server.memory_search(query="tests", type="todo")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(r["type"] == "todo" for r in results)

    def test_search_with_project_filter(self, populated_db):
        results = mcp_server.memory_search(
            query="memory", project="/tmp/project-beta",
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(r["project"] == "/tmp/project-beta" for r in results)

    def test_search_with_limit(self, populated_db):
        results = mcp_server.memory_search(query="API", limit=1)
        assert len(results) <= 1

    def test_search_empty_query(self, populated_db):
        results = mcp_server.memory_search(query="")
        assert isinstance(results, list)

    def test_search_no_results(self, populated_db):
        results = mcp_server.memory_search(query="zzz_nonexistent_zzz")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_result_fields(self, populated_db):
        results = mcp_server.memory_search(query="FastAPI")
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "title" in r
        assert "content" in r
        assert "type" in r
        assert "confidence" in r
        assert "tags" in r
        assert "score" in r
        assert "project" in r
        assert "created_at" in r


class TestMemoryList:
    """Tests for the memory_list MCP tool."""

    def test_list_all(self, populated_db):
        results = mcp_server.memory_list()
        assert isinstance(results, list)
        assert len(results) == 4

    def test_list_by_project(self, populated_db):
        results = mcp_server.memory_list(project="/tmp/project-alpha")
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(r["project"] == "/tmp/project-alpha" for r in results)

    def test_list_by_type(self, populated_db):
        results = mcp_server.memory_list(
            project="/tmp/project-alpha", type="decision",
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["type"] == "decision"

    def test_list_with_limit(self, populated_db):
        results = mcp_server.memory_list(limit=2)
        assert len(results) <= 2

    def test_list_empty_db(self):
        """Empty DB should return an empty list, not error."""
        results = mcp_server.memory_list()
        assert results == []

    def test_list_result_fields(self, populated_db):
        results = mcp_server.memory_list(limit=1)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "title" in r
        assert "content" in r
        assert "type" in r
        assert "confidence" in r
        assert "tags" in r
        assert "project" in r
        assert "created_at" in r


class TestMemoryStats:
    """Tests for the memory_stats MCP tool."""

    def test_global_stats(self, populated_db):
        stats = mcp_server.memory_stats()
        assert isinstance(stats, dict)
        assert stats["total_memories"] == 4
        assert "memories_by_type" in stats
        assert "memories_by_project" in stats

    def test_project_scoped_stats(self, populated_db):
        stats = mcp_server.memory_stats(project="/tmp/project-alpha")
        assert stats["project_memories"] == 3
        assert stats["project_sessions"] == 0  # No sessions inserted

    def test_stats_empty_db(self):
        stats = mcp_server.memory_stats()
        assert stats["total_memories"] == 0

    def test_stats_type_breakdown(self, populated_db):
        stats = mcp_server.memory_stats()
        by_type = stats["memories_by_type"]
        assert by_type.get("decision") == 1
        assert by_type.get("pattern") == 1
        assert by_type.get("todo") == 1
        assert by_type.get("issue") == 1


class TestMemoryContext:
    """Tests for the memory_context MCP tool."""

    def test_context_generation(self, populated_db):
        ctx = mcp_server.memory_context(project_path="/tmp/project-alpha")
        assert isinstance(ctx, str)
        # Should contain project name derived from path
        assert "project-alpha" in ctx

    def test_context_contains_decisions(self, populated_db):
        ctx = mcp_server.memory_context(project_path="/tmp/project-alpha")
        assert "FastAPI" in ctx

    def test_context_contains_todos(self, populated_db):
        ctx = mcp_server.memory_context(project_path="/tmp/project-alpha")
        assert "integration tests" in ctx

    def test_context_empty_project(self):
        ctx = mcp_server.memory_context(project_path="/tmp/no-such-project")
        assert isinstance(ctx, str)
        # Should still return a valid string (header at minimum)
        assert len(ctx) > 0


class TestListProjectsResource:
    """Tests for the memory://projects resource."""

    def test_list_projects(self, populated_db):
        result = mcp_server.list_projects()
        assert isinstance(result, str)
        assert "/tmp/project-alpha" in result
        assert "/tmp/project-beta" in result

    def test_list_projects_empty(self):
        result = mcp_server.list_projects()
        assert "No projects found" in result


class TestErrorHandling:
    """Ensure tools return error dicts/strings instead of crashing."""

    def test_search_invalid_type(self, populated_db):
        """Invalid type value should return an error entry."""
        results = mcp_server.memory_search(query="test", type="invalid_type")
        assert isinstance(results, list)
        assert len(results) == 1
        assert "error" in results[0]

    def test_list_invalid_type(self, populated_db):
        results = mcp_server.memory_list(
            project="/tmp/project-alpha", type="invalid_type",
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert "error" in results[0]
