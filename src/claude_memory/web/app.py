"""FastAPI web dashboard backend for claude-memory."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from claude_memory.db import MemoryDB
from claude_memory.graph import GraphBuilder
from claude_memory.models import MemoryType
from claude_memory.search import MemorySearch

logger = logging.getLogger(__name__)

# Valid memory type values for input validation
_VALID_MEMORY_TYPES = frozenset(t.value for t in MemoryType)

# ── Singleton state (mirrors mcp_server.py pattern) ────────────────────────

_db: MemoryDB | None = None
_search: MemorySearch | None = None


def _get_db() -> MemoryDB:
    """Return the shared MemoryDB instance."""
    if _db is None:
        raise RuntimeError("Database not initialised — call init_app() first")
    return _db


def _get_search() -> MemorySearch:
    """Return the shared MemorySearch instance."""
    if _search is None:
        raise RuntimeError("Search not initialised — call init_app() first")
    return _search


def init_app(db_path: Path | None = None) -> None:
    """Initialise (or re-initialise) the module-level DB connection.

    The SQLite connection is opened with ``check_same_thread=False`` so it
    can be shared across the ASGI event-loop thread and any background
    threads used by the test client.
    """
    import sqlite3

    global _db, _search  # noqa: PLW0603
    if _db is not None:
        _db.close()
    _db = MemoryDB(db_path=db_path)
    # Re-open with check_same_thread=False for ASGI compatibility
    _db.conn.close()
    _db.conn = sqlite3.connect(str(_db.db_path), check_same_thread=False)
    _db.conn.row_factory = sqlite3.Row
    _db.conn.execute("PRAGMA journal_mode=WAL")
    _db.conn.execute("PRAGMA foreign_keys=ON")
    _search = MemorySearch(_db)


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(title="Claude Memory Dashboard", version="0.5.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _validate_limit(limit: int) -> int:
    """Clamp limit to 1-500 range."""
    return max(1, min(limit, 500))


def _validate_offset(offset: int) -> int:
    """Ensure offset is >= 0."""
    return max(0, offset)


def _validate_query(q: str) -> str:
    """Strip whitespace and enforce max 500 chars."""
    q = q.strip()
    if len(q) > 500:
        q = q[:500]
    return q


def _validate_memory_id(memory_id: str) -> str:
    """Validate that memory_id is a non-empty string."""
    memory_id = memory_id.strip()
    if not memory_id:
        raise HTTPException(status_code=400, detail="memory_id must be a non-empty string")
    return memory_id


def _validate_memory_type(type_str: str | None) -> MemoryType | None:
    """Validate type filter against MemoryType values. Returns None if input is None."""
    if type_str is None:
        return None
    if type_str not in _VALID_MEMORY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid memory type '{type_str}'."
                f" Must be one of: {sorted(_VALID_MEMORY_TYPES)}"
            ),
        )
    return MemoryType(type_str)


def _memory_to_dict(mem) -> dict:
    """Convert a Memory to a JSON-serializable dict."""
    return {
        "id": mem.id,
        "session_id": mem.session_id,
        "project_path": mem.project_path,
        "memory_type": mem.memory_type.value,
        "title": mem.title,
        "content": mem.content,
        "tags": mem.tags,
        "confidence": mem.confidence,
        "created_at": mem.created_at.isoformat(),
        "updated_at": mem.updated_at.isoformat(),
    }


def _session_to_dict(s) -> dict:
    """Convert a SessionSummary to a JSON-serializable dict."""
    return {
        "session_id": s.session_id,
        "project_path": s.project_path,
        "git_branch": s.git_branch,
        "started_at": s.started_at.isoformat(),
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "duration_minutes": s.duration_minutes,
        "message_count": s.message_count,
        "user_message_count": s.user_message_count,
        "assistant_message_count": s.assistant_message_count,
        "summary_text": s.summary_text,
        "key_topics": s.key_topics,
        "files_modified": s.files_modified,
    }


# ── API Routes ────────────────────────────────────────────────────────────


@app.get("/api/search")
async def search_memories(
    q: str = Query(..., description="Search query"),
    type: str | None = None,
    project: str | None = None,
    limit: int = 20,
):
    """Full-text search over memories."""
    try:
        _get_db()
        search = _get_search()
        q = _validate_query(q)
        limit = _validate_limit(limit)
        mt = _validate_memory_type(type)
        results = search.search(
            query=q,
            project_path=project,
            memory_type=mt,
            limit=limit,
        )
        return [
            {
                **_memory_to_dict(r.memory),
                "score": r.score,
                "highlight": r.highlight,
            }
            for r in results
        ]
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memories")
async def list_memories(
    project: str | None = None,
    type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List memories with optional filters."""
    try:
        db = _get_db()
        limit = _validate_limit(limit)
        offset = _validate_offset(offset)
        mt = _validate_memory_type(type)
        if mt and project:
            memories = db.get_memories_by_type(
                project, mt, limit=limit + offset,
            )
        elif project:
            memories = db.get_memories_by_project(project, limit=limit + offset)
        else:
            memories = db.get_all_memories(limit=limit + offset)
        # Apply offset
        memories = memories[offset:][:limit]
        return [_memory_to_dict(m) for m in memories]
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("List memories error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: str):
    """Get a single memory by ID."""
    try:
        db = _get_db()
        memory_id = _validate_memory_id(memory_id)
        mem = db.get_memory(memory_id)
        if mem is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return _memory_to_dict(mem)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Get memory error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    try:
        db = _get_db()
        memory_id = _validate_memory_id(memory_id)
        deleted = db.delete_memory(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted", "id": memory_id}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Delete memory error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/stats")
async def get_stats(project: str | None = None):
    """Get system statistics."""
    try:
        db = _get_db()
        stats = db.get_stats()
        if project:
            stats["project_filter"] = project
            stats["total_memories"] = db.count_memories(project)
            stats["total_sessions"] = db.count_sessions(project)
        return stats
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions")
async def list_sessions(project: str | None = None, limit: int = 20):
    """List session summaries."""
    try:
        db = _get_db()
        limit = _validate_limit(limit)
        sessions = db.get_recent_sessions(project_path=project, limit=limit)
        return [_session_to_dict(s) for s in sessions]
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Sessions error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/projects")
async def list_projects():
    """List all projects with memory counts."""
    try:
        db = _get_db()
        stats = db.get_stats()
        projects = []
        for project_path, count in stats.get("memories_by_project", {}).items():
            session_count = db.count_sessions(project_path)
            projects.append({
                "project_path": project_path,
                "memory_count": count,
                "session_count": session_count,
            })
        return projects
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Projects error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/graph")
async def get_graph(project: str | None = None):
    """Get knowledge graph as JSON (nodes + edges)."""
    try:
        db = _get_db()
        builder = GraphBuilder(db)
        graph = builder.build(project_path=project)
        data = json.loads(builder.export_json(graph))
        return data
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Graph error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/timeline/{session_id}")
async def get_timeline(session_id: str):
    """Get session timeline events."""
    try:
        db = _get_db()
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id must be a non-empty string")
        session = db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # Return session info with events placeholder (timeline requires JSONL file)
        return {
            "session_id": session_id,
            "session": _session_to_dict(session),
            "events": [],
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Timeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/top")
async def get_top_memories(project: str | None = None, limit: int = 10):
    """Get top memories by importance score."""
    try:
        db = _get_db()
        limit = _validate_limit(limit)
        memories = db.get_top_memories(project_path=project, limit=limit)
        result = []
        for m in memories:
            d = _memory_to_dict(m)
            d["importance_score"] = db.get_importance_score(m.id)
            result.append(d)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Top memories error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Static Files + SPA ─────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return (
        "<h1>Claude Memory Dashboard</h1>"
        "<p>Static files not found.</p>"
    )
