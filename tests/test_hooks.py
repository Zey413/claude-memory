"""Tests for the hook management module."""

from __future__ import annotations

import json

import pytest

from claude_memory.config import MemoryConfig
from claude_memory.hooks import HookManager


@pytest.fixture
def hook_config(tmp_path):
    """Create a MemoryConfig pointing at a temp directory."""
    claude_home = tmp_path / ".claude"
    claude_home.mkdir()
    return MemoryConfig(claude_home=claude_home)


@pytest.fixture
def hook_manager(hook_config):
    """Create a HookManager with a temp settings path."""
    return HookManager(config=hook_config)


# ── Install tests ─────────────────────────────────────────────────────────────


def test_install_hook_empty_settings(hook_manager):
    """Install hook into fresh (non-existent) settings file."""
    assert not hook_manager.settings_path.exists()
    result = hook_manager.install_session_end_hook()
    assert result is True
    assert hook_manager.settings_path.exists()

    settings = json.loads(hook_manager.settings_path.read_text())
    hooks = settings["hooks"]["SessionEnd"]
    assert len(hooks) == 1
    assert "claude-memory" in hooks[0]["hooks"][0]["command"]


def test_install_hook_existing_hooks(hook_manager):
    """Install hook when other hooks already exist in settings."""
    # Pre-populate settings with an existing hook
    existing = {
        "hooks": {
            "SessionEnd": [
                {"hooks": [{"type": "command", "command": "echo existing-hook"}]}
            ]
        }
    }
    hook_manager.settings_path.write_text(json.dumps(existing))

    result = hook_manager.install_session_end_hook()
    assert result is True

    settings = json.loads(hook_manager.settings_path.read_text())
    session_end = settings["hooks"]["SessionEnd"]
    # Should now have two entries: the existing one + ours
    assert len(session_end) == 2
    assert "existing-hook" in session_end[0]["hooks"][0]["command"]
    assert "claude-memory" in session_end[1]["hooks"][0]["command"]


def test_uninstall_hook(hook_manager):
    """Install then uninstall; settings file should no longer contain our hook."""
    hook_manager.install_session_end_hook()
    assert hook_manager.is_installed()

    removed = hook_manager.uninstall_hook()
    assert removed is True
    assert not hook_manager.is_installed()

    settings = json.loads(hook_manager.settings_path.read_text())
    assert settings["hooks"]["SessionEnd"] == []


def test_uninstall_hook_not_found(hook_manager):
    """Uninstalling when hook was never installed returns False."""
    assert hook_manager.uninstall_hook() is False


def test_hook_script_generation(hook_manager):
    """Verify the generated hook script content."""
    script = hook_manager.generate_hook_script()
    assert "#!/bin/bash" in script
    assert "claude-memory" in script
    assert "nohup" in script
    assert "$PROJECT_DIR" in script


def test_install_hook_idempotent(hook_manager):
    """Installing twice should not duplicate the hook."""
    first = hook_manager.install_session_end_hook()
    assert first is True

    second = hook_manager.install_session_end_hook()
    assert second is False  # Already installed

    settings = json.loads(hook_manager.settings_path.read_text())
    session_end = settings["hooks"]["SessionEnd"]
    assert len(session_end) == 1


def test_write_hook_script(hook_manager, tmp_path):
    """Test writing hook script to disk."""
    output_dir = tmp_path / "scripts"
    path = hook_manager.write_hook_script(output_dir)
    assert path.exists()
    assert path.name == "hook_session_end.sh"
    content = path.read_text()
    assert "claude-memory" in content
    # Script should be executable
    import stat
    assert path.stat().st_mode & stat.S_IXUSR


def test_hook_command_content(hook_manager):
    """Verify the hook command contains the expected flags."""
    cmd = hook_manager._get_hook_command()
    assert "--project" in cmd
    assert "--latest" in cmd
    assert "$CWD" in cmd
    assert "|| true" in cmd  # Fail-safe


def test_is_installed_no_settings_file(hook_manager):
    """is_installed returns False when settings file doesn't exist."""
    assert not hook_manager.is_installed()
