"""Click-based CLI for claude-memory."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from claude_memory import __version__
from claude_memory.config import (
    discover_projects,
    find_latest_session,
    find_session_files,
)
from claude_memory.db import MemoryDB
from claude_memory.extractor import MemoryExtractor
from claude_memory.generator import ClaudemdGenerator
from claude_memory.hooks import HookManager
from claude_memory.models import Memory, MemoryType
from claude_memory.parser import parse_session_file
from claude_memory.search import MemorySearch

logger = logging.getLogger(__name__)


# ── Color helpers ────────────────────────────────────────────────────────────

TYPE_COLORS: dict[str, str] = {
    "decision": "cyan",
    "pattern": "green",
    "issue": "red",
    "solution": "blue",
    "preference": "magenta",
    "context": "white",
    "todo": "yellow",
    "learning": "bright_green",
}


def _get_db(ctx: click.Context) -> MemoryDB:
    """Get or create the DB from Click context."""
    db_path = ctx.obj.get("db_path") if ctx.obj else None
    return MemoryDB(db_path=db_path)


def _resolve_project(project: str | None) -> str:
    """Resolve project path to an absolute path."""
    if project is None:
        return str(Path.cwd().resolve())
    return str(Path(project).resolve())


def _is_json_mode(ctx: click.Context) -> bool:
    """Check if --json-output was set on the CLI group."""
    return bool(ctx.obj.get("json_mode"))


# ── Main Group ────────────────────────────────────────────────────────────────

@click.group()
@click.option("--db", type=click.Path(), default=None,
              help="Database path (default: ~/.claude-memory/memory.db)")
@click.option("--json-output", "json_mode", is_flag=True,
              help="Output in machine-readable JSON format")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose/debug logging")
@click.version_option(version=__version__, prog_name="claude-memory")
@click.pass_context
def cli(ctx: click.Context, db: str | None, json_mode: bool, verbose: bool) -> None:
    """Claude Code cross-session memory system.

    Extract, index, and recall context from past Claude Code sessions.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db) if db else None
    ctx.obj["json_mode"] = json_mode
    ctx.obj["verbose"] = verbose

    # Configure logging level based on --verbose
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(name)s: %(message)s",
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--session-id", "-s", help="Specific session UUID to ingest")
@click.option("--project", "-p", type=click.Path(exists=True),
              help="Project path (default: current directory)")
@click.option("--latest", is_flag=True, help="Ingest the most recent session only")
@click.option("--all", "ingest_all", is_flag=True,
              help="Ingest all unprocessed sessions for a project")
@click.pass_context
def ingest(ctx: click.Context, session_id: str | None, project: str | None,
           latest: bool, ingest_all: bool) -> None:
    """Extract memories from Claude Code session logs.

    \b
    Examples:
        claude-memory ingest --latest
        claude-memory ingest --session-id abc123-def456
        claude-memory ingest --all --project /path/to/project
    """
    project_path = _resolve_project(project)
    db = _get_db(ctx)
    extractor = MemoryExtractor()

    try:
        if session_id:
            _ingest_session(db, extractor, session_id, project_path)
        elif latest:
            try:
                sid, filepath = find_latest_session(project_path)
                _ingest_session(db, extractor, sid, project_path, filepath)
            except FileNotFoundError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        elif ingest_all:
            _ingest_all_sessions(db, extractor, project_path)
        else:
            click.echo("Please specify --session-id, --latest, or --all", err=True)
            sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nIngestion interrupted by user.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Error during ingestion: {exc}", err=True)
        if ctx.obj.get("verbose"):
            logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        db.close()


def _ingest_session(
    db: MemoryDB,
    extractor: MemoryExtractor,
    session_id: str,
    project_path: str,
    filepath: Path | None = None,
) -> None:
    """Ingest a single session."""
    if db.is_session_processed(session_id):
        click.echo(f"Session {session_id[:8]}... already processed, skipping.")
        return

    if filepath is None:
        files = find_session_files(project_path)
        matching = [f for f in files if f.stem == session_id]
        if not matching:
            click.echo(f"Error: Session file not found for {session_id}", err=True)
            return
        filepath = matching[0]

    click.echo(f"Parsing session {session_id[:8]}...")
    messages = parse_session_file(filepath)
    if not messages:
        click.echo("  No messages found, skipping.")
        return

    click.echo(f"  {len(messages)} messages parsed")

    # Extract memories
    memories = extractor.extract_all(messages, session_id, project_path)
    click.echo(f"  {len(memories)} memories extracted")

    # Generate summary
    summary = extractor.generate_summary(messages, session_id, project_path)

    # Store
    db.insert_session(summary)
    for mem in memories:
        db.insert_memory(mem)

    click.echo(f"  Session {session_id[:8]}... ingested successfully")


def _ingest_all_sessions(
    db: MemoryDB,
    extractor: MemoryExtractor,
    project_path: str,
) -> None:
    """Ingest all unprocessed sessions for a project."""
    files = find_session_files(project_path)
    if not files:
        click.echo("No session files found.")
        return

    total = 0
    skipped = 0
    for filepath in files:
        session_id = filepath.stem
        if db.is_session_processed(session_id):
            skipped += 1
            continue
        _ingest_session(db, extractor, session_id, project_path, filepath)
        total += 1

    click.echo(f"\nDone: {total} sessions ingested, {skipped} already processed.")


# ── Search ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--project", "-p", help="Filter by project path")
@click.option("--type", "-t", "memory_type",
              type=click.Choice([t.value for t in MemoryType], case_sensitive=False),
              help="Filter by memory type")
@click.option("--tag", multiple=True, help="Filter by tags")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(ctx: click.Context, query: str, project: str | None,
           memory_type: str | None, tag: tuple, limit: int, as_json: bool) -> None:
    """Search memories by keyword or topic.

    \b
    Examples:
        claude-memory search "authentication"
        claude-memory search "database" --type decision
        claude-memory search "pytest" --tag testing
    """
    db = _get_db(ctx)
    json_mode = as_json or _is_json_mode(ctx)
    try:
        searcher = MemorySearch(db)
        mt = MemoryType(memory_type) if memory_type else None
        tags = list(tag) if tag else None
        project_path = _resolve_project(project) if project else None

        results = searcher.search(
            query=query,
            project_path=project_path,
            memory_type=mt,
            tags=tags,
            limit=limit,
        )

        if json_mode:
            output = [
                {
                    "id": r.memory.id,
                    "type": r.memory.memory_type.value,
                    "title": r.memory.title,
                    "content": r.memory.content,
                    "score": r.score,
                    "tags": r.memory.tags,
                    "session_id": r.memory.session_id,
                    "created_at": r.memory.created_at.isoformat(),
                }
                for r in results
            ]
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            if not results:
                click.echo("No memories found.")
                return
            click.echo(f"Found {len(results)} memories:\n")
            for r in results:
                _print_memory(r.memory, score=r.score)
    except KeyboardInterrupt:
        click.echo("\nSearch interrupted.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Error during search: {exc}", err=True)
        if ctx.obj.get("verbose"):
            logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        db.close()


# ── List ──────────────────────────────────────────────────────────────────────

@cli.command("list")
@click.option("--project", "-p", help="Filter by project")
@click.option("--type", "-t", "memory_type",
              type=click.Choice([t.value for t in MemoryType], case_sensitive=False),
              help="Filter by memory type")
@click.option("--recent", "-r", default=30, help="Days to look back (default: 30)")
@click.option("--limit", "-n", default=50, help="Max results")
@click.pass_context
def list_memories(ctx: click.Context, project: str | None,
                  memory_type: str | None, recent: int, limit: int) -> None:
    """List memories with optional filters.

    \b
    Examples:
        claude-memory list --recent 7
        claude-memory list --type todo --project .
    """
    db = _get_db(ctx)
    json_mode = _is_json_mode(ctx)
    try:
        project_path = _resolve_project(project) if project else None
        if memory_type and project_path:
            memories = db.get_memories_by_type(
                project_path, MemoryType(memory_type), limit=limit
            )
        elif project_path:
            memories = db.get_memories_by_project(project_path, limit=limit)
        else:
            memories = db.get_recent_memories(days=recent, limit=limit)

        if json_mode:
            output = [_memory_to_dict(m) for m in memories]
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            if not memories:
                click.echo("No memories found.")
                return

            click.echo(f"Found {len(memories)} memories:\n")
            for mem in memories:
                _print_memory(mem)
    finally:
        db.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--project", "-p", help="Filter by project")
@click.option("--limit", "-n", default=10, help="Max results")
@click.pass_context
def sessions(ctx: click.Context, project: str | None, limit: int) -> None:
    """List processed session summaries.

    \b
    Examples:
        claude-memory sessions
        claude-memory sessions --project /path/to/project
    """
    db = _get_db(ctx)
    json_mode = _is_json_mode(ctx)
    try:
        project_path = _resolve_project(project) if project else None
        session_list = db.get_recent_sessions(project_path=project_path, limit=limit)

        if json_mode:
            output = [
                {
                    "session_id": s.session_id,
                    "project_path": s.project_path,
                    "git_branch": s.git_branch,
                    "started_at": s.started_at.isoformat(),
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "duration_minutes": s.duration_minutes,
                    "message_count": s.message_count,
                    "summary_text": s.summary_text,
                    "key_topics": s.key_topics,
                }
                for s in session_list
            ]
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            if not session_list:
                click.echo("No sessions found.")
                return

            click.echo(f"Found {len(session_list)} sessions:\n")
            for s in session_list:
                date = s.started_at.strftime("%Y-%m-%d %H:%M")
                date_str = click.style(date, dim=True)
                duration = ""
                if s.duration_minutes:
                    from claude_memory.utils import format_duration
                    duration = f" ({format_duration(s.duration_minutes)})"
                branch = f" [{s.git_branch}]" if s.git_branch else ""
                click.echo(f"  {s.session_id[:8]}... | {date_str}{duration}{branch}")
                click.echo(f"    {s.summary_text}")
                if s.key_topics:
                    click.echo(f"    Topics: {', '.join(s.key_topics[:5])}")
                click.echo()
    finally:
        db.close()


# ── Generate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--project", "-p", type=click.Path(exists=True),
              default=".", help="Project path")
@click.option("--target",
              type=click.Choice(["memory_dir", "project_root", "stdout"]),
              default="memory_dir",
              help="Where to write the generated context")
@click.pass_context
def generate(ctx: click.Context, project: str, target: str) -> None:
    """Generate CLAUDE.md project context from memories.

    \b
    Examples:
        claude-memory generate
        claude-memory generate --target project_root
        claude-memory generate --target stdout
    """
    project_path = _resolve_project(project)
    db = _get_db(ctx)
    try:
        searcher = MemorySearch(db)
        gen = ClaudemdGenerator(db, searcher)

        if target == "stdout":
            content = gen.render_to_string(project_path)
            click.echo(content)
        elif target == "project_root":
            path = gen.write_to_project_root(project_path)
            click.echo(f"Generated CLAUDE.md at {path}")
        else:  # memory_dir
            path = gen.write_to_memory_dir(project_path)
            click.echo(f"Generated context at {path}")
    except KeyboardInterrupt:
        click.echo("\nGeneration interrupted.", err=True)
        sys.exit(130)
    except PermissionError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except OSError as exc:
        click.echo(f"Error writing file: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error during generation: {exc}", err=True)
        if ctx.obj.get("verbose"):
            logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        db.close()


# ── Stats ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show memory system statistics."""
    db = _get_db(ctx)
    json_mode = _is_json_mode(ctx)
    try:
        s = db.get_stats()

        if json_mode:
            click.echo(json.dumps(s, indent=2, ensure_ascii=False))
        else:
            click.echo("Claude Memory Statistics")
            click.echo("=" * 40)
            click.echo(f"Total memories:  {click.style(str(s['total_memories']), bold=True)}")
            click.echo(f"Total sessions:  {click.style(str(s['total_sessions']), bold=True)}")
            click.echo(f"Total tags:      {click.style(str(s['total_tags']), bold=True)}")

            db_size = s["db_size_bytes"]
            if db_size > 1_048_576:
                size_str = f"{db_size / 1_048_576:.1f} MB"
            elif db_size > 1024:
                size_str = f"{db_size / 1024:.1f} KB"
            else:
                size_str = f"{db_size} B"
            click.echo(f"Database size:   {click.style(size_str, bold=True)}")

            if s["memories_by_type"]:
                click.echo("\nBy type:")
                for mtype, count in sorted(s["memories_by_type"].items()):
                    color = TYPE_COLORS.get(mtype, "white")
                    styled_type = click.style(mtype, fg=color)
                    styled_count = click.style(str(count), bold=True)
                    click.echo(f"  {styled_type:<30} {styled_count}")

            if s["memories_by_project"]:
                click.echo("\nBy project:")
                for proj, count in sorted(s["memories_by_project"].items()):
                    name = Path(proj).name if proj else "unknown"
                    click.echo(f"  {name:<30} {click.style(str(count), bold=True)}")
    finally:
        db.close()


# ── Install Hook ──────────────────────────────────────────────────────────────

@cli.command("install-hook")
@click.pass_context
def install_hook(ctx: click.Context) -> None:
    """Install SessionEnd hook into Claude Code settings.

    Adds a hook to ~/.claude/settings.json that auto-triggers
    memory extraction when a Claude Code session ends.
    """
    manager = HookManager()
    if manager.install_session_end_hook():
        click.echo("SessionEnd hook installed successfully!")
        click.echo("Memory extraction will run automatically when sessions end.")
    else:
        click.echo("Hook is already installed.")


@cli.command("uninstall-hook")
@click.pass_context
def uninstall_hook(ctx: click.Context) -> None:
    """Remove the SessionEnd hook from Claude Code settings."""
    manager = HookManager()
    if manager.uninstall_hook():
        click.echo("SessionEnd hook removed.")
    else:
        click.echo("Hook not found.")


# ── Tag Management ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--memory-id", "-m", required=True, help="Memory ID to tag")
@click.option("--add", "-a", multiple=True, help="Tags to add")
@click.option("--remove", "-r", multiple=True, help="Tags to remove")
@click.pass_context
def tag(ctx: click.Context, memory_id: str, add: tuple, remove: tuple) -> None:
    """Manage memory tags.

    \b
    Examples:
        claude-memory tag -m abc123 -a important -a architecture
        claude-memory tag -m abc123 -r obsolete
    """
    if not add and not remove:
        click.echo("Specify --add or --remove", err=True)
        sys.exit(1)

    db = _get_db(ctx)
    try:
        mem = db.get_memory(memory_id)
        if not mem:
            click.echo(f"Memory {memory_id} not found.", err=True)
            sys.exit(1)

        for t in add:
            db.add_tag_to_memory(memory_id, t)
            click.echo(f"  Added tag: {t}")
        for t in remove:
            db.remove_tag_from_memory(memory_id, t)
            click.echo(f"  Removed tag: {t}")
    finally:
        db.close()


# ── Reset ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def reset(ctx: click.Context, force: bool) -> None:
    """Reset the memory database (destructive)."""
    if not force:
        click.confirm("Are you sure? This will delete all memories.", abort=True)
    db = _get_db(ctx)
    try:
        db.reset()
        click.echo("Database reset complete. All memories deleted.")
    finally:
        db.close()


# ── Projects ──────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def projects(ctx: click.Context) -> None:
    """List all discovered projects with Claude session data."""
    json_mode = _is_json_mode(ctx)
    found = discover_projects()

    if json_mode:
        output = [
            {
                "path": str(decoded_path),
                "sessions": len(list(claude_dir.glob("*.jsonl"))),
            }
            for decoded_path, claude_dir in found
        ]
        click.echo(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if not found:
            click.echo("No projects found with Claude session data.")
            return

        click.echo(f"Found {len(found)} projects:\n")
        for decoded_path, claude_dir in found:
            jsonl_count = len(list(claude_dir.glob("*.jsonl")))
            click.echo(f"  {decoded_path}")
            click.echo(f"    Sessions: {jsonl_count}")
            click.echo()


# ── Export ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file path (default: stdout)")
@click.option("--project", "-p", help="Filter by project path")
@click.option("--type", "-t", "memory_type",
              type=click.Choice([t.value for t in MemoryType], case_sensitive=False),
              help="Filter by memory type")
@click.pass_context
def export(ctx: click.Context, output: str | None, project: str | None,
           memory_type: str | None) -> None:
    """Export memories to JSON.

    \b
    Examples:
        claude-memory export
        claude-memory export --output memories.json
        claude-memory export --project /path/to/project --type decision
    """
    db = _get_db(ctx)
    try:
        project_path = _resolve_project(project) if project else None
        if memory_type and project_path:
            memories = db.get_memories_by_type(
                project_path, MemoryType(memory_type), limit=10000
            )
        elif project_path:
            memories = db.get_memories_by_project(project_path, limit=10000)
        else:
            memories = db.get_recent_memories(days=36500, limit=10000)

        data = [_memory_to_dict(m) for m in memories]
        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if output:
            Path(output).write_text(json_str, encoding="utf-8")
            click.echo(f"Exported {len(data)} memories to {output}")
        else:
            click.echo(json_str)
    finally:
        db.close()


# ── Import ────────────────────────────────────────────────────────────────────

@cli.command("import-data")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True), help="Input JSON file path")
@click.option("--skip-duplicates/--overwrite", default=True,
              help="Skip duplicate IDs (default) or overwrite them")
@click.pass_context
def import_data(ctx: click.Context, input_path: str, skip_duplicates: bool) -> None:
    """Import memories from a JSON file.

    \b
    Examples:
        claude-memory import-data --input memories.json
        claude-memory import-data --input backup.json --overwrite
    """
    db = _get_db(ctx)
    try:
        raw = Path(input_path).read_text(encoding="utf-8")
        entries = json.loads(raw)
        if not isinstance(entries, list):
            click.echo("Error: JSON file must contain an array.", err=True)
            sys.exit(1)

        imported = 0
        skipped = 0
        errors = 0

        for entry in entries:
            try:
                # Validate required fields
                required = ["session_id", "project_path", "memory_type", "title", "content"]
                missing = [f for f in required if f not in entry]
                if missing:
                    errors += 1
                    continue

                # Check for existing memory
                mem_id = entry.get("id")
                if mem_id and db.get_memory(mem_id):
                    if skip_duplicates:
                        skipped += 1
                        continue
                    else:
                        # Overwrite: delete then re-insert
                        db.delete_memory(mem_id)

                mem = Memory(
                    id=mem_id if mem_id else None,
                    session_id=entry["session_id"],
                    project_path=entry["project_path"],
                    memory_type=MemoryType(entry["memory_type"]),
                    title=entry["title"],
                    content=entry["content"],
                    tags=entry.get("tags", []),
                    confidence=entry.get("confidence", 1.0),
                )
                # Preserve created_at if present
                if "created_at" in entry:
                    from claude_memory.utils import parse_iso
                    mem.created_at = parse_iso(entry["created_at"])
                    mem.updated_at = mem.created_at

                db.insert_memory(mem)
                imported += 1
            except Exception:
                errors += 1

        click.echo(f"{imported} imported, {skipped} skipped, {errors} errors")
    finally:
        db.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

TYPE_ICONS = {
    "decision": "🧠",
    "pattern": "📐",
    "issue": "🐛",
    "solution": "✅",
    "preference": "⚙️",
    "context": "📋",
    "todo": "📝",
    "learning": "💡",
}


def _memory_to_dict(mem: Memory) -> dict:
    """Convert a Memory to a JSON-serializable dict."""
    return {
        "id": mem.id,
        "session_id": mem.session_id,
        "project_path": mem.project_path,
        "memory_type": mem.memory_type.value,
        "title": mem.title,
        "content": mem.content,
        "tags": mem.tags,
        "confidence": mem.confidence,
        "created_at": mem.created_at.isoformat(),
    }


def _print_memory(mem, score: float | None = None) -> None:
    """Print a formatted memory entry with color."""
    icon = TYPE_ICONS.get(mem.memory_type.value, "📌")
    color = TYPE_COLORS.get(mem.memory_type.value, "white")
    type_label = click.style(f"[{mem.memory_type.value}]", fg=color)
    score_str = f" (score: {score:.2f})" if score is not None else ""
    click.echo(f"  {icon} {type_label} {mem.title}{score_str}")
    date_str = click.style(
        f"ID: {mem.id} | Session: {mem.session_id[:8]}...", dim=True
    )
    click.echo(f"     {date_str}")
    if mem.tags:
        click.echo(f"     Tags: {', '.join(mem.tags)}")
    # Show first 2 lines of content
    lines = mem.content.split("\n")
    preview = lines[0][:100]
    if len(lines) > 1:
        preview += f" (+{len(lines)-1} lines)"
    click.echo(f"     {preview}")
    click.echo()
