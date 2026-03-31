"""Tests for the CLI."""

import json

from click.testing import CliRunner

from claude_memory.cli import cli
from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType


def _insert_sample_memory(db: MemoryDB, **overrides) -> Memory:
    """Insert a sample memory into the DB and return it."""
    defaults = {
        "session_id": "test-session-001",
        "project_path": "/tmp/test-project",
        "memory_type": MemoryType.DECISION,
        "title": "Use pytest for testing",
        "content": "Decided to use pytest as the main testing framework.",
        "tags": ["testing", "tooling"],
        "confidence": 0.9,
    }
    defaults.update(overrides)
    mem = Memory(**defaults)
    db.insert_memory(mem)
    return mem


def test_cli_help():
    """Test that CLI shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Claude Code cross-session memory system" in result.output


def test_cli_version():
    """Test version output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.5.0" in result.output


def test_cli_stats_empty(tmp_path):
    """Test stats command with empty database."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "stats"])
    assert result.exit_code == 0
    assert "Total memories:" in result.output
    assert "0" in result.output


def test_cli_list_empty(tmp_path):
    """Test list command with no memories."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "list"])
    assert result.exit_code == 0
    assert "No memories found" in result.output


# ── Export tests ─────────────────────────────────────────────────────────────


def test_export_json(tmp_path):
    """Test that export produces valid JSON with correct fields."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "export"])
    assert result.exit_code == 0

    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    entry = data[0]
    assert entry["session_id"] == "test-session-001"
    assert entry["project_path"] == "/tmp/test-project"
    assert entry["memory_type"] == "decision"
    assert entry["title"] == "Use pytest for testing"
    assert entry["content"] == "Decided to use pytest as the main testing framework."
    assert entry["tags"] == ["testing", "tooling"]
    assert entry["confidence"] == 0.9
    assert "created_at" in entry


def test_export_to_file(tmp_path):
    """Test exporting to a file."""
    db_path = tmp_path / "test.db"
    out_path = tmp_path / "export.json"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "export", "--output", str(out_path),
    ])
    assert result.exit_code == 0
    assert "Exported 1 memories" in result.output

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(data) == 1


# ── Import tests ─────────────────────────────────────────────────────────────


def test_import_roundtrip(tmp_path):
    """Export memories, then import into a fresh DB; verify same data."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    mem = _insert_sample_memory(db)
    db.close()

    # Export
    export_file = tmp_path / "export.json"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "export", "--output", str(export_file),
    ])
    assert result.exit_code == 0

    # Import into a fresh DB
    db_path2 = tmp_path / "test2.db"
    result = runner.invoke(cli, [
        "--db", str(db_path2), "import-data", "--input", str(export_file),
    ])
    assert result.exit_code == 0
    assert "1 imported" in result.output
    assert "0 skipped" in result.output
    assert "0 errors" in result.output

    # Verify data
    db2 = MemoryDB(db_path=db_path2)
    imported = db2.get_memory(mem.id)
    assert imported is not None
    assert imported.title == mem.title
    assert imported.content == mem.content
    assert imported.memory_type == mem.memory_type
    assert imported.session_id == mem.session_id
    assert imported.tags == mem.tags
    db2.close()


def test_import_skip_duplicates(tmp_path):
    """Import with skip-duplicates skips existing memories."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    # Export
    export_file = tmp_path / "export.json"
    runner = CliRunner()
    runner.invoke(cli, [
        "--db", str(db_path), "export", "--output", str(export_file),
    ])

    # Import into same DB (duplicates)
    result = runner.invoke(cli, [
        "--db", str(db_path), "import-data", "--input", str(export_file),
    ])
    assert result.exit_code == 0
    assert "0 imported" in result.output
    assert "1 skipped" in result.output


def test_import_validation_errors(tmp_path):
    """Import entries with missing fields counts as errors."""
    bad_data = [
        {"title": "missing required fields"},
    ]
    import_file = tmp_path / "bad.json"
    import_file.write_text(json.dumps(bad_data), encoding="utf-8")

    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "import-data", "--input", str(import_file),
    ])
    assert result.exit_code == 0
    assert "0 imported" in result.output
    assert "1 errors" in result.output


# ── JSON output mode tests ───────────────────────────────────────────────────


def test_json_output_mode_stats(tmp_path):
    """--json-output flag on stats produces valid JSON."""
    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "--json-output", "stats",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "total_memories" in data
    assert "total_sessions" in data
    assert "total_tags" in data


def test_json_output_mode_list(tmp_path):
    """--json-output flag on list produces valid JSON array."""
    db_path = tmp_path / "test.db"
    project_path = str(tmp_path / "myproject")
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=project_path)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "--json-output", "list",
        "--project", project_path,
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["title"] == "Use pytest for testing"


def test_json_output_mode_sessions(tmp_path):
    """--json-output flag on sessions produces valid JSON array."""
    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "--json-output", "sessions",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 0


def test_json_output_mode_projects(tmp_path):
    """--json-output flag on projects produces valid JSON array."""
    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "--json-output", "projects",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


# ── Reset confirmation tests ─────────────────────────────────────────────────


def test_reset_confirmation(tmp_path):
    """Reset without --force asks for confirmation; 'n' aborts."""
    db_path = tmp_path / "test.db"
    runner = CliRunner()
    # Ensure DB exists first
    db = MemoryDB(db_path=db_path)
    db.close()

    result = runner.invoke(cli, ["--db", str(db_path), "reset"], input="n\n")
    assert result.exit_code != 0  # Aborted
    assert "Are you sure?" in result.output


def test_reset_force(tmp_path):
    """Reset with --force skips confirmation and proceeds."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "reset", "--force"])
    assert result.exit_code == 0
    assert "Database reset complete" in result.output

    # Verify DB is empty
    db2 = MemoryDB(db_path=db_path)
    assert db2.count_memories() == 0
    db2.close()


def test_reset_confirmation_yes(tmp_path):
    """Reset without --force with 'y' input proceeds normally."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "reset"], input="y\n")
    assert result.exit_code == 0
    assert "Database reset complete" in result.output
