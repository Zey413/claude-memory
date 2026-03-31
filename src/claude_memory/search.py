"""Higher-level memory search engine wrapping FTS5 and filtered queries."""

from __future__ import annotations

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType, SearchResult


class MemorySearch:
    """Facade over MemoryDB that provides richer search capabilities."""

    def __init__(self, db: MemoryDB):
        self.db = db

    def search(
        self,
        query: str,
        project_path: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Full-text search with optional project, type, and tag filters.

        Uses ``db.search_fts()`` for the core FTS5 query, then applies
        post-filtering on tags if provided.
        """
        # Fetch more results than requested when tag filtering is needed,
        # because post-filtering may discard some matches.
        fetch_limit = limit * 3 if tags else limit

        results = self.db.search_fts(
            query=query,
            project_path=project_path,
            memory_type=memory_type,
            limit=fetch_limit,
        )

        if tags:
            required = set(tags)
            results = [
                r for r in results
                if required.intersection(r.memory.tags)
            ]

        return results[:limit]

    def recent(
        self,
        project_path: str | None = None,
        days: int = 7,
        limit: int = 50,
    ) -> list[Memory]:
        """Return memories created within the last *days* days."""
        return self.db.get_recent_memories(
            days=days,
            project_path=project_path,
            limit=limit,
        )

    def by_tag(self, tag: str, limit: int = 50) -> list[Memory]:
        """Return memories that carry a specific tag.

        Performs a direct SQL query through the memory_tags junction table.
        """
        rows = self.db.conn.execute(
            """SELECT m.*
               FROM memories m
               JOIN memory_tags mt ON m.id = mt.memory_id
               WHERE mt.tag_name = ?
               ORDER BY m.created_at DESC
               LIMIT ?""",
            (tag, limit),
        ).fetchall()
        return [self.db._row_to_memory(r) for r in rows]

    def related(self, memory_id: str, limit: int = 10) -> list[Memory]:
        """Find memories related to a given memory.

        Relatedness is determined by three signals (unioned, then
        deduplicated):
          1. Same session_id.
          2. Overlapping tags.
          3. Same project + same memory_type.

        The source memory itself is always excluded from results.
        """
        memory = self.db.get_memory(memory_id)
        if memory is None:
            return []

        seen_ids: set[str] = {memory.id}
        related: list[Memory] = []

        def _collect(candidates: list[Memory]) -> None:
            for m in candidates:
                if m.id not in seen_ids:
                    seen_ids.add(m.id)
                    related.append(m)

        # 1. Same session
        session_rows = self.db.conn.execute(
            """SELECT * FROM memories
               WHERE session_id = ? AND id != ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (memory.session_id, memory.id, limit),
        ).fetchall()
        _collect([self.db._row_to_memory(r) for r in session_rows])

        # 2. Overlapping tags
        if memory.tags:
            placeholders = ", ".join("?" for _ in memory.tags)
            tag_rows = self.db.conn.execute(
                f"""SELECT DISTINCT m.*
                    FROM memories m
                    JOIN memory_tags mt ON m.id = mt.memory_id
                    WHERE mt.tag_name IN ({placeholders})
                      AND m.id != ?
                    ORDER BY m.created_at DESC
                    LIMIT ?""",
                (*memory.tags, memory.id, limit),
            ).fetchall()
            _collect([self.db._row_to_memory(r) for r in tag_rows])

        # 3. Same project + same type
        type_rows = self.db.conn.execute(
            """SELECT * FROM memories
               WHERE project_path = ? AND memory_type = ? AND id != ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (memory.project_path, memory.memory_type.value, memory.id, limit),
        ).fetchall()
        _collect([self.db._row_to_memory(r) for r in type_rows])

        return related[:limit]

    def by_project(self, project_path: str, limit: int = 100) -> list[Memory]:
        """Return all memories belonging to a project."""
        return self.db.get_memories_by_project(project_path, limit=limit)

    def by_type(
        self,
        project_path: str,
        memory_type: MemoryType,
        limit: int = 50,
    ) -> list[Memory]:
        """Return memories of a given type within a project."""
        return self.db.get_memories_by_type(project_path, memory_type, limit=limit)
