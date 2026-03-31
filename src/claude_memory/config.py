"""Configuration and path resolution for Claude Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Default paths
CLAUDE_HOME = Path.home() / ".claude"
MEMORY_DB_DIR = Path.home() / ".claude-memory"
DEFAULT_DB_NAME = "memory.db"


@dataclass
class MemoryConfig:
    """Central configuration for the memory system."""

    claude_home: Path = field(default_factory=lambda: CLAUDE_HOME)
    db_path: Path = field(default_factory=lambda: MEMORY_DB_DIR / DEFAULT_DB_NAME)

    @property
    def projects_dir(self) -> Path:
        """Directory containing per-project Claude data."""
        return self.claude_home / "projects"

    @property
    def history_file(self) -> Path:
        """Global history.jsonl file."""
        return self.claude_home / "history.jsonl"

    @property
    def settings_file(self) -> Path:
        """Claude Code settings.json."""
        return self.claude_home / "settings.json"

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def project_path_to_claude_dir(project_path: str, config: MemoryConfig | None = None) -> Path:
    """Convert an absolute project path to its Claude project directory.

    Claude Code encodes project paths by replacing '/' with '-':
        /Users/foo/Desktop/myproject → -Users-foo-Desktop-myproject
        → ~/.claude/projects/-Users-foo-Desktop-myproject/

    Args:
        project_path: Absolute path to the project directory.
        config: Optional config (uses defaults if not provided).

    Returns:
        Path to the Claude project directory.
    """
    cfg = config or MemoryConfig()
    # Normalize and encode the path
    normalized = str(Path(project_path).resolve())
    encoded = normalized.replace("/", "-")
    return cfg.projects_dir / encoded


def discover_projects(config: MemoryConfig | None = None) -> list[tuple[str, Path]]:
    """Discover all projects that have Claude session data.

    Returns:
        List of (decoded_project_path, claude_project_dir) tuples.
    """
    cfg = config or MemoryConfig()
    projects = []

    if not cfg.projects_dir.exists():
        return projects

    for entry in cfg.projects_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("-"):
            # Decode: -Users-foo-Desktop-myproject → /Users/foo/Desktop/myproject
            decoded = entry.name.replace("-", "/", 1)  # Fix leading dash
            # Actually the encoding replaces ALL / with -, so we need to reverse
            decoded = "/" + entry.name[1:].replace("-", "/")
            projects.append((decoded, entry))

    return projects


def find_session_files(project_path: str, config: MemoryConfig | None = None) -> list[Path]:
    """Find all session JSONL files for a project, sorted by modification time (newest first).

    Args:
        project_path: Absolute path to the project directory.
        config: Optional config.

    Returns:
        List of Path objects to .jsonl session files, newest first.
    """
    claude_dir = project_path_to_claude_dir(project_path, config)
    if not claude_dir.exists():
        return []

    jsonl_files = sorted(
        claude_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return jsonl_files


def find_latest_session(project_path: str, config: MemoryConfig | None = None) -> tuple[str, Path]:
    """Find the most recently modified session JSONL for a project.

    Args:
        project_path: Absolute path to the project directory.
        config: Optional config.

    Returns:
        Tuple of (session_id, path_to_jsonl).

    Raises:
        FileNotFoundError: If no session files exist.
    """
    files = find_session_files(project_path, config)
    if not files:
        claude_dir = project_path_to_claude_dir(project_path, config)
        raise FileNotFoundError(f"No session files found in {claude_dir}")

    latest = files[0]
    session_id = latest.stem  # UUID is the filename without .jsonl
    return session_id, latest
