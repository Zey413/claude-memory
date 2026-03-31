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
    assert "0.6.0" in result.output


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


# ── Search command tests ────────────────────────────────────────────────────


def test_search_command(tmp_path):
    """Test basic search via CLI finds matching memories."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, title="FastAPI architecture", content="Using FastAPI for REST.")
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "search", "FastAPI"])
    assert result.exit_code == 0
    assert "FastAPI" in result.output


def test_search_no_results(tmp_path):
    """Test search for nonexistent query returns no results message."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "search", "xyznonexistent"])
    assert result.exit_code == 0
    assert "No memories found" in result.output


# ── List with filter tests ──────────────────────────────────────────────────


def test_list_with_type_filter(tmp_path):
    """Test list --type decision filters by memory type."""
    db_path = tmp_path / "test.db"
    project_path = str(tmp_path / "myproject")
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=project_path, memory_type=MemoryType.DECISION,
                          title="A decision")
    _insert_sample_memory(db, project_path=project_path, memory_type=MemoryType.TODO,
                          title="A todo")
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "list",
        "--type", "decision", "--project", project_path,
    ])
    assert result.exit_code == 0
    assert "A decision" in result.output


def test_list_with_project_filter(tmp_path):
    """Test list --project filters memories to specific project."""
    db_path = tmp_path / "test.db"
    project_a = str(tmp_path / "project-a")
    project_b = str(tmp_path / "project-b")
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=project_a, title="Memory A")
    _insert_sample_memory(db, project_path=project_b, title="Memory B")
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "--json-output", "list",
        "--project", project_a,
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["title"] == "Memory A"


# ── Generate command tests ──────────────────────────────────────────────────


def test_generate_stdout(tmp_path):
    """Test generate --target stdout prints markdown to console."""
    db_path = tmp_path / "test.db"
    project_path = tmp_path / "myproject"
    project_path.mkdir()
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=str(project_path))
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "generate",
        "--project", str(project_path), "--target", "stdout",
    ])
    assert result.exit_code == 0
    assert "myproject" in result.output
    assert "#" in result.output


def test_generate_to_dir(tmp_path):
    """Test generate --target project_root writes CLAUDE.md file."""
    db_path = tmp_path / "test.db"
    project_path = tmp_path / "myproject"
    project_path.mkdir()
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=str(project_path))
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "generate",
        "--project", str(project_path), "--target", "project_root",
    ])
    assert result.exit_code == 0
    assert "Generated CLAUDE.md" in result.output
    assert (project_path / "CLAUDE.md").exists()


# ── Projects and sessions commands ──────────────────────────────────────────


def test_projects_command(tmp_path):
    """Test projects command runs without error."""
    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "projects"])
    assert result.exit_code == 0
    # The projects command reads ~/.claude/projects so it may find real data
    # or no data — either way it shouldn't crash
    assert result.output is not None


def test_sessions_command(tmp_path):
    """Test sessions command with no sessions."""
    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "sessions"])
    assert result.exit_code == 0
    assert "No sessions found" in result.output


# ── Export with filters ─────────────────────────────────────────────────────


def test_export_with_type_filter(tmp_path):
    """Test export --type decision only exports decisions."""
    db_path = tmp_path / "test.db"
    project_path = str(tmp_path / "proj")
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=project_path,
                          memory_type=MemoryType.DECISION, title="A decision")
    _insert_sample_memory(db, project_path=project_path,
                          memory_type=MemoryType.TODO, title="A todo")
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "export",
        "--project", project_path, "--type", "decision",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["memory_type"] == "decision"


def test_export_with_project_filter(tmp_path):
    """Test export --project filters to one project."""
    db_path = tmp_path / "test.db"
    proj_a = str(tmp_path / "proj-a")
    proj_b = str(tmp_path / "proj-b")
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db, project_path=proj_a, title="Memory A")
    _insert_sample_memory(db, project_path=proj_b, title="Memory B")
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "export", "--project", proj_a,
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["title"] == "Memory A"


# ── Tag command tests ───────────────────────────────────────────────────────


def test_tag_add(tmp_path):
    """Test tag --add adds a tag to a memory."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    mem = _insert_sample_memory(db, tags=[])
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "tag",
        "--memory-id", mem.id, "--add", "important",
    ])
    assert result.exit_code == 0
    assert "Added tag: important" in result.output


def test_tag_remove(tmp_path):
    """Test tag --remove removes a tag from a memory."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    mem = _insert_sample_memory(db, tags=["testing", "tooling"])
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--db", str(db_path), "tag",
        "--memory-id", mem.id, "--remove", "tooling",
    ])
    assert result.exit_code == 0
    assert "Removed tag: tooling" in result.output


# ── Consolidate command tests ───────────────────────────────────────────────


def test_consolidate_command(tmp_path):
    """Test consolidate runs without errors."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "consolidate"])
    assert result.exit_code == 0
    assert "Consolidation complete" in result.output
    assert "Memories scored:" in result.output


def test_consolidate_dry_run(tmp_path):
    """Test consolidate --dry-run shows report without modifying DB."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "consolidate", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "Memories to score:" in result.output


# ── Top command test ────────────────────────────────────────────────────────


def test_top_command(tmp_path):
    """Test top shows memories (or message to consolidate first)."""
    db_path = tmp_path / "test.db"
    db = MemoryDB(db_path=db_path)
    _insert_sample_memory(db)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", str(db_path), "top"])
    assert result.exit_code == 0
    # With un-scored memories, it returns them with 0 scores
    assert "Top" in result.output or "No memories found" in result.output
