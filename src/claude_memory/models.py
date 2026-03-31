"""Data models for Claude Memory system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

__all__ = [
    "MemoryType",
    "Memory",
    "SessionSummary",
    "Tag",
    "ProjectContext",
    "SearchResult",
]


class MemoryType(str, Enum):
    """Categories of extractable memories."""

    DECISION = "decision"       # Architecture/design decisions
    PATTERN = "pattern"         # Code patterns, conventions
    ISSUE = "issue"             # Bugs found, errors encountered
    SOLUTION = "solution"       # How issues were resolved
    PREFERENCE = "preference"   # User preferences, style choices
    CONTEXT = "context"         # Project context, domain knowledge
    TODO = "todo"               # Unfinished work, future plans
    LEARNING = "learning"       # New concepts, techniques discovered


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _short_id() -> str:
    return uuid4().hex[:12]


class Memory(BaseModel):
    """A single extracted memory unit."""

    id: str = Field(default_factory=_short_id)
    session_id: str
    project_path: str
    memory_type: MemoryType
    title: str  # Short summary (< 120 chars)
    content: str  # Full memory content
    tags: list[str] = Field(default_factory=list)
    source_line_start: int | None = None
    source_line_end: int | None = None
    confidence: float = 1.0  # 0.0-1.0 extraction confidence
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)


class SessionSummary(BaseModel):
    """Summary of a complete Claude Code session."""

    session_id: str
    project_path: str
    git_branch: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_minutes: float | None = None
    message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    tool_uses: dict[str, int] = Field(default_factory=dict)
    files_modified: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    summary_text: str = ""
    key_topics: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)


class Tag(BaseModel):
    """Reusable tag for organizing memories."""

    name: str
    count: int = 0
    last_used: datetime = Field(default_factory=_now_utc)


class ProjectContext(BaseModel):
    """Aggregated context for a project across all sessions."""

    project_path: str
    project_name: str
    total_sessions: int = 0
    total_memories: int = 0
    last_session_at: datetime | None = None
    primary_language: str | None = None
    key_decisions: list[str] = Field(default_factory=list)
    active_todos: list[str] = Field(default_factory=list)
    common_patterns: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now_utc)


class SearchResult(BaseModel):
    """A search result with relevance score."""

    memory: Memory
    score: float = 0.0
    highlight: str = ""  # Text snippet with match highlighted
