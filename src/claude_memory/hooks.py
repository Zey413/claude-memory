"""Hook management for Claude Code integration."""

from __future__ import annotations

import json
from pathlib import Path

from claude_memory.config import MemoryConfig


class HookManager:
    """Manage Claude Code hooks for auto-triggering memory extraction."""

    HOOK_MARKER = "claude-memory"

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self.settings_path = self.config.settings_file

    def is_installed(self) -> bool:
        """Check if the SessionEnd hook is already installed."""
        if not self.settings_path.exists():
            return False
        settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        hooks = settings.get("hooks", {}).get("SessionEnd", [])
        for entry in hooks:
            for hook in entry.get("hooks", []):
                if self.HOOK_MARKER in hook.get("command", ""):
                    return True
        return False

    def install_session_end_hook(self) -> bool:
        """Install the SessionEnd hook into Claude Code settings.json.

        Returns True if installed, False if already exists.
        """
        if self.is_installed():
            return False

        # Load or create settings
        if self.settings_path.exists():
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        else:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings = {}

        # Build the hook entry
        hook_command = self._get_hook_command()
        hook_entry = {
            "hooks": [
                {
                    "type": "command",
                    "command": hook_command,
                }
            ]
        }

        # Add to SessionEnd hooks (non-destructive — preserves existing hooks)
        hooks = settings.setdefault("hooks", {})
        session_end = hooks.setdefault("SessionEnd", [])
        session_end.append(hook_entry)

        # Write back
        self.settings_path.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return True

    def uninstall_hook(self) -> bool:
        """Remove the SessionEnd hook from Claude Code settings.json.

        Returns True if removed, False if not found.
        """
        if not self.settings_path.exists():
            return False

        settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        session_end = settings.get("hooks", {}).get("SessionEnd", [])

        new_entries = []
        removed = False
        for entry in session_end:
            new_hooks = []
            for hook in entry.get("hooks", []):
                if self.HOOK_MARKER not in hook.get("command", ""):
                    new_hooks.append(hook)
                else:
                    removed = True
            if new_hooks:
                entry["hooks"] = new_hooks
                new_entries.append(entry)

        if removed:
            settings["hooks"]["SessionEnd"] = new_entries
            self.settings_path.write_text(
                json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        return removed

    def _get_hook_command(self) -> str:
        """Generate the hook command string."""
        return (
            'claude-memory ingest --project "$CWD" --latest '
            "2>>/tmp/claude-memory-hook.log || true"
        )

    def generate_hook_script(self) -> str:
        """Generate a standalone shell script for the hook."""
        return """#!/bin/bash
# Claude Code SessionEnd hook for claude-memory
# Runs extraction in background to avoid blocking Claude Code exit

PROJECT_DIR="${CWD:-$(pwd)}"

# Run in background
nohup claude-memory ingest \\
    --project "$PROJECT_DIR" \\
    --latest \\
    >> /tmp/claude-memory-hook.log 2>&1 &
"""

    def write_hook_script(self, output_dir: Path | None = None) -> Path:
        """Write the hook script to a file."""
        if output_dir is None:
            output_dir = Path.home() / ".claude-memory"
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / "hook_session_end.sh"
        script_path.write_text(self.generate_hook_script(), encoding="utf-8")
        script_path.chmod(0o755)
        return script_path
