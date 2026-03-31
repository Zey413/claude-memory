"""MCP (Model Context Protocol) server for real-time memory access.

Exposes claude-memory tools and resources so Claude Code can query
memories directly during a session via the stdio transport.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from claude_memory.db import MemoryDB
from claude_memory.models import MemoryType
from claude_memory.search import MemorySearch

logger = logging.getLogger(__name__)

# ── Singleton state ──────────────────────────────────────────────────────────

_db: MemoryDB | None = None
_search: MemorySearch | None = None


def _get_db() -> MemoryDB:
    """Return the shared MemoryDB instance."""
    if _db is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _db


def _get_search() -> MemorySearch:
    """Return the shared MemorySearch instance."""
    if _search is None:
        raise RuntimeError("Search not initialised — call init_db() first")
    return _search


def init_db(db_path: Path | None = None) -> None:
    """Initialise (or re-initialise) the module-level DB connection."""
    global _db, _search  # noqa: PLW0603
    if _db is not None:
        _db.close()
    _db = MemoryDB(db_path=db_path)
    _search = MemorySearch(_db)


# ── MCP server ───────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="claude-memory",
    instructions="Query and search the claude-memory knowledge base.",
)


# ── Tool 1: memory_search ───────────────────────────────────────────────────

@mcp.tool()
def memory_search(
    query: str,
    type: str | None = None,
    project: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search memories by keyword.

    Args:
        query: Search query string
        type: Optional memory type filter (decision, pattern, issue, solution,
              preference, context, todo, learning)
        project: Optional project path filter
        limit: Maximum number of results (default 10)

    Returns:
        List of matching memories with title, content, type, confidence, tags.
    """
    try:
        search = _get_search()
        memory_type = MemoryType(type) if type else None
        results = search.search(
            query=query,
            project_path=project,
            memory_type=memory_type,
            limit=limit,
        )
        return [
            {
                "id": r.memory.id,
                "title": r.memory.title,
                "content": r.memory.content,
                "type": r.memory.memory_type.value,
                "confidence": r.memory.confidence,
                "tags": r.memory.tags,
                "score": r.score,
                "project": r.memory.project_path,
                "created_at": r.memory.created_at.isoformat(),
            }
            for r in results
        ]
    except Exception as exc:
        logger.exception("memory_search failed")
        return [{"error": str(exc)}]


# ── Tool 2: memory_list ─────────────────────────────────────────────────────

@mcp.tool()
def memory_list(
    project: str | None = None,
    type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List recent memories.

    Args:
        project: Optional project path filter
        type: Optional memory type filter
        limit: Maximum number of results (default 20)

    Returns:
        List of memories.
    """
    try:
        db = _get_db()
        if type and project:
            memories = db.get_memories_by_type(
                project, MemoryType(type), limit=limit,
            )
        elif project:
            memories = db.get_memories_by_project(project, limit=limit)
        else:
            memories = db.get_recent_memories(days=365, limit=limit)
        return [
            {
                "id": m.id,
                "title": m.title,
                "content": m.content,
                "type": m.memory_type.value,
                "confidence": m.confidence,
                "tags": m.tags,
                "project": m.project_path,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
    except Exception as exc:
        logger.exception("memory_list failed")
        return [{"error": str(exc)}]


# ── Tool 3: memory_stats ────────────────────────────────────────────────────

@mcp.tool()
def memory_stats(project: str | None = None) -> dict:
    """Get memory system statistics.

    Args:
        project: Optional project path to scope stats

    Returns:
        Stats dict with counts, types, projects.
    """
    try:
        db = _get_db()
        stats = db.get_stats()
        if project:
            project_count = db.count_memories(project_path=project)
            session_count = db.count_sessions(project_path=project)
            stats["project_memories"] = project_count
            stats["project_sessions"] = session_count
        return stats
    except Exception as exc:
        logger.exception("memory_stats failed")
        return {"error": str(exc)}


# ── Tool 4: memory_context ──────────────────────────────────────────────────

@mcp.tool()
def memory_context(project_path: str) -> str:
    """Get full project context (CLAUDE.md-style).

    Args:
        project_path: Absolute path to the project

    Returns:
        Generated context string in CLAUDE.md format.
    """
    try:
        from claude_memory.generator import ClaudemdGenerator

        db = _get_db()
        search = _get_search()
        gen = ClaudemdGenerator(db, search)
        return gen.render_to_string(project_path)
    except Exception as exc:
        logger.exception("memory_context failed")
        return f"Error generating context: {exc}"


# ── Resource: memory://projects ──────────────────────────────────────────────

@mcp.resource("memory://projects")
def list_projects() -> str:
    """List all projects with memories.

    Returns a newline-separated list of project paths that have stored memories.
    """
    try:
        db = _get_db()
        stats = db.get_stats()
        projects = stats.get("memories_by_project", {})
        if not projects:
            return "No projects found."
        lines = [
            f"{path} ({count} memories)"
            for path, count in sorted(projects.items())
        ]
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("list_projects failed")
        return f"Error listing projects: {exc}"


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Claude Memory MCP server")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: ~/.claude-memory/memory.db)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio"],
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    init_db(db_path)

    try:
        mcp.run(transport=args.transport)
    finally:
        if _db is not None:
            _db.close()


if __name__ == "__main__":
    main()
