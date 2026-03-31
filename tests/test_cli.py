"""Tests for the CLI."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_memory.cli import cli


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
    assert "0.1.0" in result.output


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
