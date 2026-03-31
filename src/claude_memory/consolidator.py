"""Memory consolidation — dedup, merge, score, archive."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType

# ── Report Model ──────────────────────────────────────────────────────────────

class ConsolidationReport(BaseModel):
    """Report produced by a consolidation run."""

    duplicates_found: int = 0
    duplicates_merged: int = 0
    todos_archived: int = 0
    memories_scored: int = 0
    top_memories: list[dict] = Field(default_factory=list)  # Top 5 by score


# ── Scoring weights ──────────────────────────────────────────────────────────

TYPE_WEIGHTS: dict[str, float] = {
    "decision": 1.0,
    "solution": 0.9,
    "pattern": 0.8,
    "issue": 0.7,
    "preference": 0.7,
    "todo": 0.6,
    "learning": 0.5,
    "context": 0.4,
}

_HALF_LIFE_DAYS = 30.0


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets of strings."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def _word_set(text: str) -> set[str]:
    """Convert text to a set of lowercase words (simple tokenisation)."""
    return set(text.lower().split())


# ── Consolidator ──────────────────────────────────────────────────────────────

class MemoryConsolidator:
    """Engine for deduplication, scoring, archival, and merging of memories."""

    def __init__(self, db: MemoryDB):
        self.db = db

    # ── Full pipeline ─────────────────────────────────────────────────────

    def consolidate(self, project_path: str | None = None) -> ConsolidationReport:
        """Run full consolidation pipeline. Returns report."""
        report = ConsolidationReport()

        # 1. Score all memories
        scored = self.score_memories(project_path)
        report.memories_scored = scored

        # 2. Find and merge duplicates
        pairs = self.find_duplicates(project_path)
        report.duplicates_found = len(pairs)
        merged = self.merge_duplicates(pairs)
        report.duplicates_merged = merged

        # 3. Archive stale TODOs
        archived = self.archive_stale_todos()
        report.todos_archived = len(archived)

        # 4. Get top memories for the report
        top = self.db.get_top_memories(project_path=project_path, limit=5)
        report.top_memories = [
            {
                "id": m.id,
                "title": m.title,
                "type": m.memory_type.value,
                "score": self.db.get_importance_score(m.id),
            }
            for m in top
        ]

        return report

    # ── Scoring ───────────────────────────────────────────────────────────

    def score_memories(self, project_path: str | None = None) -> int:
        """Calculate importance scores for all memories.

        Scoring factors:
        - Type weight: decision=1.0, solution=0.9, pattern=0.8,
          issue=0.7, preference=0.7, todo=0.6, learning=0.5, context=0.4
        - Recency: exponential decay (halve every 30 days)
        - Confidence: direct multiplier
        - Tag count: more tags = more connected = more important
        - Final score = type_weight * recency * confidence * (1 + 0.1 * tag_count)

        Store score in the importance_score column.
        Returns the number of memories scored.
        """
        memories = self.db.get_all_memories(project_path=project_path)
        now = datetime.now(timezone.utc)
        count = 0

        for mem in memories:
            type_weight = TYPE_WEIGHTS.get(mem.memory_type.value, 0.4)

            # Recency: exponential decay with half-life of 30 days
            age_days = (now - mem.created_at).total_seconds() / 86400.0
            recency = math.pow(0.5, age_days / _HALF_LIFE_DAYS)

            confidence = mem.confidence if mem.confidence is not None else 1.0

            tag_count = len(mem.tags)

            score = type_weight * recency * confidence * (1.0 + 0.1 * tag_count)

            self.db.update_importance_score(mem.id, score)
            count += 1

        return count

    # ── Duplicate detection ───────────────────────────────────────────────

    def find_duplicates(
        self, project_path: str | None = None
    ) -> list[tuple[Memory, Memory]]:
        """Find near-duplicate memories using Jaccard similarity on word sets.

        Two memories are near-duplicates if:
        - Same type AND
        - Jaccard similarity of title+content words > 0.6

        Returns list of (keep, remove) pairs.
        The memory with the higher importance score is kept.
        """
        memories = self.db.get_all_memories(project_path=project_path)

        # Group by type to reduce comparisons
        by_type: dict[str, list[Memory]] = {}
        for mem in memories:
            key = mem.memory_type.value
            by_type.setdefault(key, []).append(mem)

        seen_ids: set[str] = set()
        pairs: list[tuple[Memory, Memory]] = []

        for _type_key, group in by_type.items():
            for i in range(len(group)):
                if group[i].id in seen_ids:
                    continue
                words_i = _word_set(group[i].title + " " + group[i].content)
                for j in range(i + 1, len(group)):
                    if group[j].id in seen_ids:
                        continue
                    words_j = _word_set(group[j].title + " " + group[j].content)
                    sim = _jaccard_similarity(words_i, words_j)
                    if sim > 0.6:
                        # Determine which to keep (higher importance score)
                        score_i = self.db.get_importance_score(group[i].id)
                        score_j = self.db.get_importance_score(group[j].id)
                        if score_i >= score_j:
                            keep, remove = group[i], group[j]
                        else:
                            keep, remove = group[j], group[i]
                        pairs.append((keep, remove))
                        seen_ids.add(remove.id)

        return pairs

    # ── Archiving stale TODOs ─────────────────────────────────────────────

    def archive_stale_todos(self, days: int = 30) -> list[Memory]:
        """Archive TODOs older than N days (change type to 'context').

        Returns list of archived memories.
        """
        now = datetime.now(timezone.utc)
        all_memories = self.db.get_all_memories()
        archived: list[Memory] = []

        for mem in all_memories:
            if mem.memory_type != MemoryType.TODO:
                continue
            age_days = (now - mem.created_at).total_seconds() / 86400.0
            if age_days > days:
                self.db.update_memory_type(mem.id, MemoryType.CONTEXT)
                mem.memory_type = MemoryType.CONTEXT
                archived.append(mem)

        return archived

    # ── Merging duplicates ────────────────────────────────────────────────

    def merge_duplicates(self, pairs: list[tuple[Memory, Memory]]) -> int:
        """Merge near-duplicate pairs: keep higher-scoring, delete other.

        Returns count of removed memories.
        """
        removed = 0
        for _keep, remove in pairs:
            if self.db.delete_memory(remove.id):
                removed += 1
        return removed
