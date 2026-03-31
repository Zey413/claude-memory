"""Higher-level memory search engine wrapping FTS5 and filtered queries."""

from __future__ import annotations

import logging

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType, SearchResult

__all__ = [
    "MemorySearch",
]

logger = logging.getLogger(__name__)


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

    # ── Semantic search ──────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        project_path: str | None = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """Semantic similarity search using embeddings.

        Falls back to FTS if embedding dependencies are unavailable or
        no memories have been embedded yet.
        """
        try:
            from claude_memory.embedding import EmbeddingEngine, is_available
        except ImportError:
            logger.info("Embedding module unavailable, falling back to FTS")
            return self.search(query, project_path=project_path, limit=limit)

        if not is_available():
            logger.info("sentence-transformers not installed, falling back to FTS")
            return self.search(query, project_path=project_path, limit=limit)

        pairs = self.db.get_memories_with_embeddings(project_path)
        if not pairs:
            logger.info("No embedded memories found, falling back to FTS")
            return self.search(query, project_path=project_path, limit=limit)

        engine = EmbeddingEngine.get_instance()
        query_vec = engine.encode(query)

        scored: list[tuple[Memory, float]] = []
        for mem, emb_bytes in pairs:
            mem_vec = engine.deserialize(emb_bytes)
            score = engine.cosine_similarity(query_vec, mem_vec)
            if score >= min_score:
                scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for mem, score in scored[:limit]:
            snippet = mem.content[:150] + ("..." if len(mem.content) > 150 else "")
            results.append(SearchResult(memory=mem, score=score, highlight=snippet))
        return results

    def hybrid_search(
        self,
        query: str,
        project_path: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Combine FTS and semantic results using reciprocal rank fusion.

        Falls back to FTS-only when embeddings are not available.
        """
        fts_results = self.search(query, project_path=project_path, limit=limit * 2)

        try:
            from claude_memory.embedding import is_available
        except ImportError:
            return fts_results[:limit]

        if not is_available():
            return fts_results[:limit]

        pairs = self.db.get_memories_with_embeddings(project_path)
        if not pairs:
            return fts_results[:limit]

        sem_results = self.semantic_search(
            query, project_path=project_path, limit=limit * 2, min_score=0.0,
        )

        # Reciprocal rank fusion (k = 60 is a standard constant)
        k = 60
        rrf_scores: dict[str, float] = {}
        rrf_memories: dict[str, SearchResult] = {}

        for rank, r in enumerate(fts_results):
            mid = r.memory.id
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            rrf_memories[mid] = r

        for rank, r in enumerate(sem_results):
            mid = r.memory.id
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            if mid not in rrf_memories:
                rrf_memories[mid] = r

        sorted_ids = sorted(rrf_scores, key=lambda mid: rrf_scores[mid], reverse=True)

        results: list[SearchResult] = []
        for mid in sorted_ids[:limit]:
            sr = rrf_memories[mid]
            results.append(
                SearchResult(
                    memory=sr.memory,
                    score=rrf_scores[mid],
                    highlight=sr.highlight,
                )
            )
        return results

    # ── Existing helpers ─────────────────────────────────────────────────

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
