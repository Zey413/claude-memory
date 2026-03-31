"""Tests for configuration and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from claude_memory.config import (
    MemoryConfig,
    discover_projects,
    find_latest_session,
    find_session_files,
    project_path_to_claude_dir,
)


@pytest.fixture
def mock_claude_home(tmp_path):
    """Create a mock ~/.claude directory structure."""
    claude_home = tmp_path / ".claude"
    projects_dir = claude_home / "projects"
    projects_dir.mkdir(parents=True)
    return claude_home


@pytest.fixture
def mock_config(mock_claude_home, tmp_path):
    """Create a MemoryConfig pointing at mock directories."""
    return MemoryConfig(
        claude_home=mock_claude_home,
        db_path=tmp_path / "test.db",
    )


# ── Path encoding ─────────────────────────────────────────────────────────────


def test_project_path_to_claude_dir(mock_config):
    """Verify that project paths are correctly encoded to Claude directory names."""
    result = project_path_to_claude_dir("/Users/foo/Desktop/myproject", config=mock_config)
    assert result.name == "-Users-foo-Desktop-myproject"
    assert result.parent == mock_config.projects_dir


def test_project_path_to_claude_dir_nested(mock_config):
    """Encoding preserves deeper nesting — the resolved path is encoded."""
    from pathlib import Path
    # Use resolved path so we're not affected by macOS /home → /System/Volumes/Data symlinks
    resolved = str(Path("/home/user/code/org/repo").resolve())
    expected_name = resolved.replace("/", "-")
    result = project_path_to_claude_dir("/home/user/code/org/repo", config=mock_config)
    assert result.name == expected_name


# ── discover_projects ─────────────────────────────────────────────────────────


def test_discover_projects(mock_config, mock_claude_home):
    """Discover projects that have Claude session data."""
    proj_dir = mock_claude_home / "projects" / "-Users-foo-Desktop-myproject"
    proj_dir.mkdir()

    projects = discover_projects(config=mock_config)
    assert len(projects) == 1
    decoded_path, claude_dir = projects[0]
    assert decoded_path == "/Users/foo/Desktop/myproject"
    assert claude_dir == proj_dir


def test_discover_projects_empty(mock_config):
    """No projects found when projects dir is empty."""
    projects = discover_projects(config=mock_config)
    assert projects == []


def test_discover_projects_missing_dir(tmp_path):
    """Graceful handling when projects dir doesn't exist."""
    config = MemoryConfig(claude_home=tmp_path / "nonexistent")
    projects = discover_projects(config=config)
    assert projects == []


def test_discover_projects_multiple(mock_config, mock_claude_home):
    """Discover multiple projects."""
    (mock_claude_home / "projects" / "-Users-foo-proj-a").mkdir()
    (mock_claude_home / "projects" / "-Users-foo-proj-b").mkdir()

    projects = discover_projects(config=mock_config)
    assert len(projects) == 2


# ── find_session_files ────────────────────────────────────────────────────────


def test_find_session_files(mock_config, mock_claude_home):
    """Find JSONL session files for a project, newest first."""
    import time
    from pathlib import Path

    # project_path_to_claude_dir resolves the path, so we must match
    project_path = "/tmp/myproject"
    resolved = str(Path(project_path).resolve())
    encoded = resolved.replace("/", "-")
    proj_dir = mock_claude_home / "projects" / encoded
    proj_dir.mkdir(parents=True)

    # Create two JSONL files with different mod times
    f1 = proj_dir / "session-aaa.jsonl"
    f1.write_text("{}\n")
    time.sleep(0.05)  # Ensure different mtime
    f2 = proj_dir / "session-bbb.jsonl"
    f2.write_text("{}\n")

    files = find_session_files(project_path, config=mock_config)
    assert len(files) == 2
    # Newest should be first
    assert files[0].stem == "session-bbb"
    assert files[1].stem == "session-aaa"


def test_find_session_files_no_dir(mock_config):
    """Return empty list when the project's claude dir doesn't exist."""
    files = find_session_files("/tmp/nonexistent", config=mock_config)
    assert files == []


# ── find_latest_session ───────────────────────────────────────────────────────


def test_find_latest_session(mock_config, mock_claude_home):
    """find_latest_session returns the newest session file."""
    import time
    from pathlib import Path

    project_path = "/tmp/myproject"
    resolved = str(Path(project_path).resolve())
    encoded = resolved.replace("/", "-")
    proj_dir = mock_claude_home / "projects" / encoded
    proj_dir.mkdir(parents=True)

    f1 = proj_dir / "session-old.jsonl"
    f1.write_text("{}\n")
    time.sleep(0.05)
    f2 = proj_dir / "session-new.jsonl"
    f2.write_text("{}\n")

    session_id, filepath = find_latest_session(project_path, config=mock_config)
    assert session_id == "session-new"
    assert filepath == f2


def test_find_latest_session_no_files(mock_config, mock_claude_home):
    """Raise FileNotFoundError when no session files exist."""
    from pathlib import Path

    project_path = "/tmp/myproject"
    resolved = str(Path(project_path).resolve())
    encoded = resolved.replace("/", "-")
    proj_dir = mock_claude_home / "projects" / encoded
    proj_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        find_latest_session(project_path, config=mock_config)


# ── MemoryConfig defaults ────────────────────────────────────────────────────


def test_memory_config_defaults():
    """Verify default paths are sensible."""
    config = MemoryConfig()
    assert config.claude_home == Path.home() / ".claude"
    assert config.db_path == Path.home() / ".claude-memory" / "memory.db"
    assert config.projects_dir == Path.home() / ".claude" / "projects"
    assert config.settings_file == Path.home() / ".claude" / "settings.json"
    assert config.history_file == Path.home() / ".claude" / "history.jsonl"


def test_memory_config_custom_paths(tmp_path):
    """Custom paths are respected."""
    config = MemoryConfig(
        claude_home=tmp_path / "custom-claude",
        db_path=tmp_path / "custom.db",
    )
    assert config.claude_home == tmp_path / "custom-claude"
    assert config.db_path == tmp_path / "custom.db"
    assert config.projects_dir == tmp_path / "custom-claude" / "projects"


def test_missing_claude_home(tmp_path):
    """Graceful handling when claude_home doesn't exist."""
    config = MemoryConfig(claude_home=tmp_path / "no-such-dir")
    # These should work without errors
    assert config.projects_dir == tmp_path / "no-such-dir" / "projects"
    # discover_projects should return empty, not crash
    projects = discover_projects(config=config)
    assert projects == []


def test_ensure_dirs(tmp_path):
    """ensure_dirs creates the database parent directory."""
    db_path = tmp_path / "sub" / "deep" / "memory.db"
    config = MemoryConfig(db_path=db_path)
    config.ensure_dirs()
    assert db_path.parent.exists()
