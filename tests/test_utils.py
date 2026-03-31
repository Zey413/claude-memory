"""Tests for shared utility functions."""

from __future__ import annotations

from datetime import datetime, timezone

from claude_memory.utils import (
    content_hash,
    format_duration,
    iso_now,
    now_utc,
    parse_iso,
    project_name_from_path,
    truncate,
)

# ── now_utc ───────────────────────────────────────────────────────────────────


def test_now_utc_returns_utc():
    """now_utc returns a timezone-aware UTC datetime."""
    dt = now_utc()
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


def test_now_utc_is_recent():
    """now_utc returns a time very close to the current time."""
    before = datetime.now(timezone.utc)
    dt = now_utc()
    after = datetime.now(timezone.utc)
    assert before <= dt <= after


# ── iso_now ───────────────────────────────────────────────────────────────────


def test_iso_now_format():
    """iso_now returns a valid ISO 8601 string."""
    result = iso_now()
    # Should be parseable back
    parsed = parse_iso(result)
    assert isinstance(parsed, datetime)


# ── parse_iso ─────────────────────────────────────────────────────────────────


def test_parse_iso_with_z_suffix():
    """Parse ISO string ending with Z (UTC)."""
    dt = parse_iso("2026-03-28T10:00:00Z")
    assert dt.year == 2026
    assert dt.month == 3
    assert dt.day == 28
    assert dt.hour == 10


def test_parse_iso_with_offset():
    """Parse ISO string with explicit offset."""
    dt = parse_iso("2026-03-28T10:00:00+00:00")
    assert dt.year == 2026


def test_parse_iso_without_tz():
    """Parse ISO string without timezone info."""
    dt = parse_iso("2026-03-28T10:00:00")
    assert dt.year == 2026


# ── content_hash ──────────────────────────────────────────────────────────────


def test_content_hash_deterministic():
    """Same input always produces the same hash."""
    h1 = content_hash("hello world")
    h2 = content_hash("hello world")
    assert h1 == h2


def test_content_hash_different_inputs():
    """Different inputs produce different hashes."""
    h1 = content_hash("hello world")
    h2 = content_hash("goodbye world")
    assert h1 != h2


def test_content_hash_length():
    """Hash is truncated to 16 hex characters."""
    h = content_hash("any string")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_empty_string():
    """Empty string still produces a valid hash."""
    h = content_hash("")
    assert len(h) == 16


# ── truncate ──────────────────────────────────────────────────────────────────


def test_truncate_long_string():
    """Long strings are truncated with ellipsis."""
    long = "x" * 200
    result = truncate(long, max_length=120)
    assert len(result) == 120
    assert result.endswith("...")


def test_truncate_short_string():
    """Short strings are returned unchanged."""
    short = "hello"
    result = truncate(short, max_length=120)
    assert result == "hello"


def test_truncate_exact_length():
    """String at exact max_length is not truncated."""
    exact = "x" * 120
    result = truncate(exact, max_length=120)
    assert result == exact
    assert "..." not in result


def test_truncate_custom_max():
    """Custom max_length is respected."""
    result = truncate("a" * 50, max_length=20)
    assert len(result) == 20
    assert result.endswith("...")


# ── project_name_from_path ────────────────────────────────────────────────────


def test_project_name_from_path_unix():
    """Extract project name from Unix-style path."""
    assert project_name_from_path("/Users/foo/Desktop/my-project") == "my-project"


def test_project_name_from_path_nested():
    """Extract project name from deeply nested path."""
    assert project_name_from_path("/home/user/code/org/repo") == "repo"


def test_project_name_from_path_simple():
    """Single component path."""
    assert project_name_from_path("myproject") == "myproject"


# ── format_duration ───────────────────────────────────────────────────────────


def test_format_duration_seconds():
    """Sub-minute durations show seconds."""
    assert format_duration(0.5) == "30s"


def test_format_duration_minutes():
    """Durations under an hour show minutes."""
    assert format_duration(45) == "45m"


def test_format_duration_hours():
    """Multi-hour durations show hours and minutes."""
    result = format_duration(75)  # 1h 15m
    assert "1h" in result
    assert "15m" in result


def test_format_duration_exact_hours():
    """Exact hour durations don't show minutes."""
    assert format_duration(120) == "2h"


def test_format_duration_days():
    """Multi-day durations show days."""
    result = format_duration(1500)  # 25 hours
    assert "d" in result
