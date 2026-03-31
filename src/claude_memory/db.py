"""SQLite database with FTS5 full-text search for memory storage."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from claude_memory.config import MemoryConfig
from claude_memory.models import Memory, MemoryType, SearchResult, SessionSummary, Tag
from claude_memory.utils import iso_now, parse_iso

logger = logging.getLogger(__name__)

# Retry configuration for "database is locked" errors
_LOCKED_RETRIES = 3
_LOCKED_SLEEP = 0.1


# ── Schema Migrations ────────────────────────────────────────────────────────

MIGRATIONS: dict[int, list[str]] = {
    1: [
        # Core memories table
        """CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            project_path TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source_line_start INTEGER,
            source_line_end INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        # Sessions table
        """CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            project_path TEXT NOT NULL,
            git_branch TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_minutes REAL,
            message_count INTEGER DEFAULT 0,
            user_message_count INTEGER DEFAULT 0,
            assistant_message_count INTEGER DEFAULT 0,
            tool_uses_json TEXT DEFAULT '{}',
            files_modified_json TEXT DEFAULT '[]',
            files_read_json TEXT DEFAULT '[]',
            summary_text TEXT DEFAULT '',
            key_topics_json TEXT DEFAULT '[]',
            processed_at TEXT
        )""",
        # Tags table
        """CREATE TABLE IF NOT EXISTS tags (
            name TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0,
            last_used TEXT
        )""",
        # Memory-tag junction table
        """CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
            tag_name TEXT REFERENCES tags(name) ON DELETE CASCADE,
            PRIMARY KEY (memory_id, tag_name)
        )""",
        # FTS5 virtual table
        """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            title,
            content,
            tags,
            content=memories,
            content_rowid=rowid,
            tokenize='porter unicode61'
        )""",
        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_path)",
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at)",
        # Schema version table
        """CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )""",
    ],
}


class MemoryDB:
    """SQLite database with FTS5 for memory storage and search."""

    def __init__(self, db_path: Path | None = None, config: MemoryConfig | None = None):
        cfg = config or MemoryConfig()
        self.db_path = db_path or cfg.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._execute("PRAGMA journal_mode=WAL")
        self._execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Execute helper with retry ────────────────────────────────────────

    def _execute(
        self, sql: str, params: tuple | list = (),
        *, commit: bool = False,
    ) -> sqlite3.Cursor:
        """Execute a SQL statement with retry logic for 'database is locked'.

        Wraps cursor.execute with automatic retry on sqlite3.OperationalError
        when the database is locked. Other OperationalErrors are re-raised.
        """
        last_err: Exception | None = None
        for attempt in range(_LOCKED_RETRIES):
            try:
                cursor = self.conn.execute(sql, params)
                if commit:
                    self.conn.commit()
                return cursor
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower():
                    last_err = exc
                    logger.warning(
                        "Database locked (attempt %d/%d), retrying in %.1fs...",
                        attempt + 1, _LOCKED_RETRIES, _LOCKED_SLEEP,
                    )
                    time.sleep(_LOCKED_SLEEP)
                else:
                    raise
        # All retries exhausted
        raise last_err  # type: ignore[misc]

    # ── Migration ─────────────────────────────────────────────────────────

    def _get_version(self) -> int:
        """Get current schema version."""
        try:
            row = self._execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] if row and row[0] is not None else 0
        except sqlite3.OperationalError:
            return 0

    def _migrate(self) -> None:
        """Apply pending migrations with transaction rollback on failure."""
        current = self._get_version()
        for ver in sorted(MIGRATIONS):
            if ver > current:
                try:
                    for sql in MIGRATIONS[ver]:
                        self._execute(sql)
                    self._execute(
                        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                        (ver, iso_now()),
                    )
                    self.conn.commit()
                except (sqlite3.OperationalError, sqlite3.IntegrityError) as exc:
                    logger.error("Migration to version %d failed: %s — rolling back", ver, exc)
                    self.conn.rollback()
                    raise

    # ── Memory CRUD ───────────────────────────────────────────────────────

    def insert_memory(self, memory: Memory) -> str:
        """Insert a memory and update FTS index. Returns the memory ID.

        Handles IntegrityError (duplicate ID) and OperationalError gracefully.
        """
        try:
            self._execute(
                """INSERT INTO memories
                   (id, session_id, project_path, memory_type, title, content,
                    confidence, source_line_start, source_line_end, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.session_id,
                    memory.project_path,
                    memory.memory_type.value,
                    memory.title,
                    memory.content,
                    memory.confidence,
                    memory.source_line_start,
                    memory.source_line_end,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                ),
            )
            # Update FTS
            rowid = self._execute(
                "SELECT rowid FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()[0]
            tags_str = " ".join(memory.tags)
            self._execute(
                "INSERT INTO memories_fts (rowid, title, content, tags) VALUES (?, ?, ?, ?)",
                (rowid, memory.title, memory.content, tags_str),
            )
            # Insert tags
            for tag in memory.tags:
                self._ensure_tag(tag)
                self._execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag_name) VALUES (?, ?)",
                    (memory.id, tag),
                )
            self.conn.commit()
            return memory.id
        except sqlite3.IntegrityError as exc:
            self.conn.rollback()
            logger.warning("Duplicate memory %s, skipping: %s", memory.id, exc)
            return memory.id
        except sqlite3.OperationalError as exc:
            self.conn.rollback()
            logger.error("Failed to insert memory %s: %s", memory.id, exc)
            raise

    def get_memory(self, memory_id: str) -> Memory | None:
        """Get a memory by ID."""
        row = self._execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def get_memories_by_type(
        self,
        project_path: str,
        memory_type: MemoryType,
        limit: int = 50,
    ) -> list[Memory]:
        """Get memories filtered by type and project."""
        rows = self._execute(
            """SELECT * FROM memories
               WHERE project_path = ? AND memory_type = ?
               ORDER BY created_at DESC LIMIT ?""",
            (project_path, memory_type.value, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memories_by_project(
        self,
        project_path: str,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories for a project."""
        rows = self._execute(
            """SELECT * FROM memories
               WHERE project_path = ?
               ORDER BY created_at DESC LIMIT ?""",
            (project_path, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_recent_memories(
        self,
        days: int = 7,
        project_path: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """Get recent memories, optionally filtered by project."""
        sql = """SELECT * FROM memories
                 WHERE datetime(created_at) >= datetime('now', ?)"""
        params: list = [f"-{days} days"]
        if project_path:
            sql += " AND project_path = ?"
            params.append(project_path)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        # Remove FTS entry first
        row = self._execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row:
            self._execute(
                "INSERT INTO memories_fts (memories_fts, rowid, title, content, tags) "
                "VALUES ('delete', ?, '', '', '')",
                (row[0],),
            )
        cursor = self._execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count_memories(self, project_path: str | None = None) -> int:
        """Count total memories, optionally by project."""
        if project_path:
            row = self._execute(
                "SELECT COUNT(*) FROM memories WHERE project_path = ?",
                (project_path,),
            ).fetchone()
        else:
            row = self._execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    # ── Session CRUD ──────────────────────────────────────────────────────

    def insert_session(self, summary: SessionSummary) -> str:
        """Insert or update a session summary."""
        self._execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, project_path, git_branch, started_at, ended_at,
                duration_minutes, message_count, user_message_count,
                assistant_message_count, tool_uses_json, files_modified_json,
                files_read_json, summary_text, key_topics_json, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.session_id,
                summary.project_path,
                summary.git_branch,
                summary.started_at.isoformat(),
                summary.ended_at.isoformat() if summary.ended_at else None,
                summary.duration_minutes,
                summary.message_count,
                summary.user_message_count,
                summary.assistant_message_count,
                json.dumps(summary.tool_uses),
                json.dumps(summary.files_modified),
                json.dumps(summary.files_read),
                summary.summary_text,
                json.dumps(summary.key_topics),
                iso_now(),
            ),
        )
        self.conn.commit()
        return summary.session_id

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Get a session summary by ID."""
        row = self._execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    def get_recent_sessions(
        self,
        project_path: str | None = None,
        limit: int = 10,
    ) -> list[SessionSummary]:
        """Get recent sessions, optionally filtered by project."""
        if project_path:
            rows = self._execute(
                """SELECT * FROM sessions
                   WHERE project_path = ?
                   ORDER BY started_at DESC LIMIT ?""",
                (project_path, limit),
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def is_session_processed(self, session_id: str) -> bool:
        """Check if a session has already been processed."""
        row = self._execute(
            "SELECT processed_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None and row["processed_at"] is not None

    def count_sessions(self, project_path: str | None = None) -> int:
        """Count total sessions."""
        if project_path:
            row = self._execute(
                "SELECT COUNT(*) FROM sessions WHERE project_path = ?",
                (project_path,),
            ).fetchone()
        else:
            row = self._execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    # ── Tag Operations ────────────────────────────────────────────────────

    def _ensure_tag(self, name: str) -> None:
        """Create tag if it doesn't exist, update if it does."""
        self._execute(
            """INSERT INTO tags (name, count, last_used)
               VALUES (?, 1, ?)
               ON CONFLICT(name) DO UPDATE SET
                   count = count + 1,
                   last_used = ?""",
            (name, iso_now(), iso_now()),
        )

    def get_all_tags(self) -> list[Tag]:
        """Get all tags sorted by count."""
        rows = self._execute(
            "SELECT * FROM tags ORDER BY count DESC"
        ).fetchall()
        return [
            Tag(name=r["name"], count=r["count"], last_used=parse_iso(r["last_used"]))
            for r in rows
        ]

    def add_tag_to_memory(self, memory_id: str, tag_name: str) -> None:
        """Add a tag to a memory."""
        self._ensure_tag(tag_name)
        self._execute(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_name) VALUES (?, ?)",
            (memory_id, tag_name),
        )
        self.conn.commit()

    def remove_tag_from_memory(self, memory_id: str, tag_name: str) -> None:
        """Remove a tag from a memory."""
        self._execute(
            "DELETE FROM memory_tags WHERE memory_id = ? AND tag_name = ?",
            (memory_id, tag_name),
        )
        self.conn.commit()

    # ── FTS5 Search ───────────────────────────────────────────────────────

    def search_fts(
        self,
        query: str,
        project_path: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Full-text search over memories with BM25 ranking.

        Returns an empty list if the FTS query is malformed or empty.
        """
        fts_query = self._prepare_fts_query(query)

        sql = """
            SELECT m.*, bm25(memories_fts, 2.0, 1.0, 1.5) AS rank
            FROM memories m
            JOIN memories_fts ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ?
        """
        params: list = [fts_query]

        if project_path:
            sql += " AND m.project_path = ?"
            params.append(project_path)
        if memory_type:
            sql += " AND m.memory_type = ?"
            params.append(memory_type.value)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = self._execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            # Malformed FTS query (unbalanced quotes, bad syntax, etc.)
            logger.warning("FTS query failed for %r: %s — returning empty results", query, exc)
            return []

        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            score = abs(row["rank"])  # BM25 returns negative scores
            snippet = self._make_snippet(memory.content, query)
            results.append(SearchResult(memory=memory, score=score, highlight=snippet))
        return results

    def _prepare_fts_query(self, query: str) -> str:
        """Convert natural language query to FTS5 syntax."""
        words = query.strip().split()
        if not words:
            return '""'
        if len(words) == 1:
            # Prefix match for single words
            return f'"{words[0]}"*'
        # OR individual words for multi-word queries
        return " OR ".join(f'"{w}"' for w in words)

    def _make_snippet(self, content: str, query: str, max_len: int = 150) -> str:
        """Create a text snippet around the query match."""
        query_lower = query.lower()
        content_lower = content.lower()
        pos = content_lower.find(query_lower)
        if pos == -1:
            # Try individual words
            for word in query.split():
                pos = content_lower.find(word.lower())
                if pos != -1:
                    break
        if pos == -1:
            return content[:max_len] + ("..." if len(content) > max_len else "")

        start = max(0, pos - max_len // 3)
        end = min(len(content), pos + max_len * 2 // 3)
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get overall system statistics."""
        total_memories = self._execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        total_sessions = self._execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_tags = self._execute("SELECT COUNT(*) FROM tags").fetchone()[0]

        type_counts = {}
        for row in self._execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories GROUP BY memory_type"
        ).fetchall():
            type_counts[row["memory_type"]] = row["cnt"]

        project_counts = {}
        for row in self._execute(
            "SELECT project_path, COUNT(*) as cnt FROM memories GROUP BY project_path"
        ).fetchall():
            project_counts[row["project_path"]] = row["cnt"]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_memories": total_memories,
            "total_sessions": total_sessions,
            "total_tags": total_tags,
            "memories_by_type": type_counts,
            "memories_by_project": project_counts,
            "db_size_bytes": db_size,
        }

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop all data and recreate schema."""
        self._execute("DROP TABLE IF EXISTS memory_tags")
        self._execute("DROP TABLE IF EXISTS tags")
        self._execute("DROP TABLE IF EXISTS memories_fts")
        self._execute("DROP TABLE IF EXISTS memories")
        self._execute("DROP TABLE IF EXISTS sessions")
        self._execute("DROP TABLE IF EXISTS schema_version")
        self.conn.commit()
        self._migrate()

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        # Get tags for this memory
        tag_rows = self._execute(
            "SELECT tag_name FROM memory_tags WHERE memory_id = ?",
            (row["id"],),
        ).fetchall()
        tags = [tr["tag_name"] for tr in tag_rows]

        return Memory(
            id=row["id"],
            session_id=row["session_id"],
            project_path=row["project_path"],
            memory_type=MemoryType(row["memory_type"]),
            title=row["title"],
            content=row["content"],
            confidence=row["confidence"],
            source_line_start=row["source_line_start"],
            source_line_end=row["source_line_end"],
            tags=tags,
            created_at=parse_iso(row["created_at"]),
            updated_at=parse_iso(row["updated_at"]),
        )

    def _row_to_session(self, row: sqlite3.Row) -> SessionSummary:
        """Convert a database row to a SessionSummary object."""
        return SessionSummary(
            session_id=row["session_id"],
            project_path=row["project_path"],
            git_branch=row["git_branch"],
            started_at=parse_iso(row["started_at"]),
            ended_at=parse_iso(row["ended_at"]) if row["ended_at"] else None,
            duration_minutes=row["duration_minutes"],
            message_count=row["message_count"],
            user_message_count=row["user_message_count"],
            assistant_message_count=row["assistant_message_count"],
            tool_uses=json.loads(row["tool_uses_json"] or "{}"),
            files_modified=json.loads(row["files_modified_json"] or "[]"),
            files_read=json.loads(row["files_read_json"] or "[]"),
            summary_text=row["summary_text"] or "",
            key_topics=json.loads(row["key_topics_json"] or "[]"),
        )
