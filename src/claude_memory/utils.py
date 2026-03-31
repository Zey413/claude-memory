"""Shared utility functions."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def iso_now() -> str:
    """Get current UTC datetime as ISO 8601 string."""
    return now_utc().isoformat()


def parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 datetime string."""
    # Handle various formats
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def content_hash(content: str) -> str:
    """Generate a short hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def truncate(text: str, max_length: int = 120) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def project_name_from_path(project_path: str) -> str:
    """Extract a human-readable project name from a path.

    Examples:
        /Users/foo/Desktop/quant-trading-system → quant-trading-system
        /home/bar/projects/my-app → my-app
    """
    return Path(project_path).name


def format_duration(minutes: float) -> str:
    """Format duration in minutes to human-readable string."""
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_mins = minutes % 60
        if remaining_mins > 0:
            return f"{hours:.0f}h {remaining_mins:.0f}m"
        return f"{hours:.0f}h"
    days = hours / 24
    return f"{days:.1f}d"
