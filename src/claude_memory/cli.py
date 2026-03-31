"""Click-based CLI for claude-memory."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from claude_memory import __version__
from claude_memory.config import (
    MemoryConfig,
    discover_projects,
    find_latest_session,
    find_session_files,
)
from claude_memory.db import MemoryDB
from claude_memory.display import (
    display_memory_table,
    display_projects,
    display_search_results,
    display_sessions,
    display_stats,
    display_timeline,
)
from claude_memory.extractor import MemoryExtractor
from claude_memory.generator import ClaudemdGenerator
from claude_memory.graph import GraphBuilder
from claude_memory.hooks import HookManager
from claude_memory.models import Memory, MemoryType
from claude_memory.parser import parse_session_file
from claude_memory.search import MemorySearch
from claude_memory.watcher import SessionWatcher

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
@click.option("--semantic", is_flag=True,
              help="Use semantic similarity search")
@click.option("--hybrid", is_flag=True,
              help="Combine FTS + semantic search")
@click.pass_context
def search(ctx: click.Context, query: str, project: str | None,
           memory_type: str | None, tag: tuple, limit: int, as_json: bool,
           semantic: bool, hybrid: bool) -> None:
    """Search memories by keyword or topic.

    \b
    Examples:
        claude-memory search "authentication"
        claude-memory search "database" --type decision
        claude-memory search "pytest" --tag testing
        claude-memory search "API design" --semantic
        claude-memory search "database patterns" --hybrid
    """
    db = _get_db(ctx)
    json_mode = as_json or _is_json_mode(ctx)
    try:
        searcher = MemorySearch(db)
        mt = MemoryType(memory_type) if memory_type else None
        tags = list(tag) if tag else None
        project_path = _resolve_project(project) if project else None

        if semantic:
            results = searcher.semantic_search(
                query=query,
                project_path=project_path,
                limit=limit,
            )
        elif hybrid:
            results = searcher.hybrid_search(
                query=query,
                project_path=project_path,
                limit=limit,
            )
        else:
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
            display_search_results(results, query)
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


# ── Embed ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--project", "-p", help="Filter by project path")
@click.option("--force", is_flag=True, help="Re-embed all memories (even already embedded)")
@click.pass_context
def embed(ctx: click.Context, project: str | None, force: bool) -> None:
    """Generate embeddings for all memories (requires embeddings extra).

    \b
    Examples:
        claude-memory embed
        claude-memory embed --project /path/to/project
        claude-memory embed --force
    """
    try:
        from claude_memory.embedding import _MODEL_ID, EmbeddingEngine, is_available
    except ImportError:
        click.echo(
            "Error: Embedding dependencies not installed.\n"
            "Install with: pip install 'claude-memory[embeddings]'",
            err=True,
        )
        sys.exit(1)

    if not is_available():
        click.echo(
            "Error: sentence-transformers and/or numpy not installed.\n"
            "Install with: pip install 'claude-memory[embeddings]'",
            err=True,
        )
        sys.exit(1)

    db = _get_db(ctx)
    try:
        project_path = _resolve_project(project) if project else None
        all_memories = db.get_all_memories(project_path)

        if not all_memories:
            click.echo("No memories found to embed.")
            return

        # Determine which memories need embedding
        if force:
            to_embed = all_memories
        else:
            to_embed = [m for m in all_memories if db.get_embedding(m.id) is None]

        if not to_embed:
            total = db.count_embedded(project_path)
            click.echo(f"All {total} memories already embedded. Use --force to re-embed.")
            return

        click.echo(f"Embedding {len(to_embed)} memories...")
        engine = EmbeddingEngine.get_instance()

        # Batch encode for efficiency
        texts = [f"{m.title} {m.content}" for m in to_embed]
        vectors = engine.encode_batch(texts)

        for mem, vec in zip(to_embed, vectors):
            blob = engine.serialize(vec)
            db.store_embedding(mem.id, blob, _MODEL_ID)

        click.echo(f"Done: {len(to_embed)} memories embedded with model '{_MODEL_ID}'.")
    except KeyboardInterrupt:
        click.echo("\nEmbedding interrupted.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Error during embedding: {exc}", err=True)
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

            display_memory_table(memories, title=f"Found {len(memories)} memories")
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

            display_sessions(session_list)
    finally:
        db.close()


# ── Replay ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('session_id', required=False)
@click.option('--project', '-p', type=click.Path(exists=True),
              help='Project path (default: current directory)')
@click.option('--limit', default=50, help='Max events to show')
@click.option('--type', 'event_type', default=None,
              help='Filter by event type (user_message, tool_use, file_write, bash_command)')
@click.pass_context
def replay(ctx, session_id, project, limit, event_type):
    """Replay a session timeline showing key events.

    If no session_id is given, replays the most recent session.

    \b
    Examples:
        claude-memory replay
        claude-memory replay abc123-def456
        claude-memory replay --project /path/to/project
        claude-memory replay --type bash_command
    """
    from claude_memory.timeline import TimelineBuilder

    project_path = _resolve_project(project)

    try:
        if session_id:
            # Find the session file by ID (or prefix match)
            files = find_session_files(project_path)
            matching = [f for f in files if f.stem == session_id or f.stem.startswith(session_id)]
            if not matching:
                click.echo(f"Error: Session file not found for {session_id}", err=True)
                sys.exit(1)
            filepath = matching[0]
        else:
            # Use the latest session
            try:
                _sid, filepath = find_latest_session(project_path)
            except FileNotFoundError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

        builder = TimelineBuilder()
        timeline = builder.build_from_jsonl(filepath)
        timeline.project_path = project_path

        display_timeline(timeline, limit=limit, event_type=event_type)

    except KeyboardInterrupt:
        click.echo("\nReplay interrupted.", err=True)
        sys.exit(130)
    except Exception as exc:
        click.echo(f"Error during replay: {exc}", err=True)
        if ctx.obj.get("verbose"):
            logger.exception("Full traceback:")
        sys.exit(1)


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
            display_stats(s)
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


# ── Serve (MCP) ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--transport", default="stdio", type=click.Choice(["stdio"]),
              help="Transport type for MCP server (default: stdio)")
@click.pass_context
def serve(ctx: click.Context, transport: str) -> None:
    """Start the MCP server for real-time memory access.

    Launches a Model Context Protocol server that exposes memory
    search, listing, stats, and context generation as MCP tools.

    \b
    Examples:
        claude-memory serve
        claude-memory serve --transport stdio
    """
    try:
        from claude_memory.mcp_server import init_db
        from claude_memory.mcp_server import mcp as mcp_server
    except ImportError:
        click.echo(
            "Error: MCP dependencies not installed.\n"
            "Install with: pip install 'claude-memory[mcp]'",
            err=True,
        )
        sys.exit(1)

    db_path = ctx.obj.get("db_path") if ctx.obj else None
    init_db(db_path)
    mcp_server.run(transport=transport)


# ── Consolidate ──────────────────────────────────────────────────────────────

@cli.command()
@click.option('--project', default=None, help='Filter by project path')
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
@click.pass_context
def consolidate(ctx, project, dry_run):
    """Consolidate memories: dedup, score, archive stale TODOs."""
    from claude_memory.consolidator import MemoryConsolidator

    db = _get_db(ctx)
    json_mode = _is_json_mode(ctx)
    try:
        consolidator = MemoryConsolidator(db)
        project_path = _resolve_project(project) if project else None

        if dry_run:
            # Score first so find_duplicates has scores to work with
            scored = consolidator.score_memories(project_path)
            pairs = consolidator.find_duplicates(project_path)
            # Count stale TODOs without archiving
            from datetime import datetime, timezone
            all_mems = db.get_all_memories(project_path)
            now = datetime.now(timezone.utc)
            stale_count = sum(
                1 for m in all_mems
                if m.memory_type == MemoryType.TODO
                and (now - m.created_at).total_seconds() / 86400.0 > 30
            )
            click.echo("Dry run — no changes made:")
            click.echo(f"  Memories to score:     {scored}")
            click.echo(f"  Duplicates found:      {len(pairs)}")
            click.echo(f"  Stale TODOs to archive: {stale_count}")
            if pairs:
                click.echo("\n  Duplicate pairs:")
                for keep, remove in pairs:
                    click.echo(f"    KEEP: [{keep.memory_type.value}] {keep.title}")
                    click.echo(f"    DROP: [{remove.memory_type.value}] {remove.title}")
                    click.echo()
        else:
            report = consolidator.consolidate(project_path)
            if json_mode:
                click.echo(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
            else:
                click.echo("Consolidation complete:")
                click.echo(f"  Memories scored:    {report.memories_scored}")
                click.echo(f"  Duplicates found:   {report.duplicates_found}")
                click.echo(f"  Duplicates merged:  {report.duplicates_merged}")
                click.echo(f"  TODOs archived:     {report.todos_archived}")
                if report.top_memories:
                    click.echo("\n  Top memories:")
                    for tm in report.top_memories:
                        click.echo(f"    [{tm['type']}] {tm['title']} (score: {tm['score']:.3f})")
    finally:
        db.close()


# ── Top ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option('--project', default=None, help='Filter by project path')
@click.option('--limit', default=10, help='Number of top memories to show')
@click.pass_context
def top(ctx, project, limit):
    """Show top memories by importance score."""
    db = _get_db(ctx)
    json_mode = _is_json_mode(ctx)
    try:
        project_path = _resolve_project(project) if project else None
        memories = db.get_top_memories(project_path=project_path, limit=limit)

        if not memories:
            click.echo("No memories found. Run 'consolidate' first to score memories.")
            return

        if json_mode:
            output = [
                {
                    **_memory_to_dict(m),
                    "importance_score": db.get_importance_score(m.id),
                }
                for m in memories
            ]
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            click.echo(f"Top {len(memories)} memories by importance:\n")
            for i, mem in enumerate(memories, 1):
                score = db.get_importance_score(mem.id)
                icon = TYPE_ICONS.get(mem.memory_type.value, "📌")
                color = TYPE_COLORS.get(mem.memory_type.value, "white")
                type_label = click.style(f"[{mem.memory_type.value}]", fg=color)
                click.echo(f"  {i}. {icon} {type_label} {mem.title} (score: {score:.3f})")
                if mem.tags:
                    click.echo(f"     Tags: {', '.join(mem.tags)}")
                click.echo()
    finally:
        db.close()


# ── Diff ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument('session_id_1')
@click.argument('session_id_2', required=False, default=None)
@click.pass_context
def diff(ctx, session_id_1, session_id_2):
    """Show differences between two sessions (or latest vs session)."""
    db = _get_db(ctx)
    try:
        session1 = db.get_session(session_id_1)
        if session1 is None:
            # Try prefix match
            all_sessions = db.get_recent_sessions(limit=100)
            matches = [s for s in all_sessions if s.session_id.startswith(session_id_1)]
            if matches:
                session1 = matches[0]
            else:
                click.echo(f"Session {session_id_1} not found.", err=True)
                sys.exit(1)

        if session_id_2:
            session2 = db.get_session(session_id_2)
            if session2 is None:
                all_sessions = db.get_recent_sessions(limit=100)
                matches = [s for s in all_sessions if s.session_id.startswith(session_id_2)]
                if matches:
                    session2 = matches[0]
                else:
                    click.echo(f"Session {session_id_2} not found.", err=True)
                    sys.exit(1)
        else:
            # Use latest session as session2
            recent = db.get_recent_sessions(limit=2)
            if len(recent) < 2:
                click.echo("Need at least two sessions to diff.", err=True)
                sys.exit(1)
            session2 = recent[0] if recent[0].session_id != session1.session_id else recent[1]

        # Get memories for each session
        all_mems = db.get_all_memories()
        mems1 = [m for m in all_mems if m.session_id == session1.session_id]
        mems2 = [m for m in all_mems if m.session_id == session2.session_id]

        date1 = session1.started_at.strftime('%Y-%m-%d')
        sid1 = session1.session_id[:8]
        click.echo(f"Session A: {sid1}... ({date1})")
        click.echo(f"  Summary: {session1.summary_text or 'N/A'}")
        click.echo(f"  Memories: {len(mems1)}")
        click.echo()

        date2 = session2.started_at.strftime('%Y-%m-%d')
        sid2 = session2.session_id[:8]
        click.echo(f"Session B: {sid2}... ({date2})")
        click.echo(f"  Summary: {session2.summary_text or 'N/A'}")
        click.echo(f"  Memories: {len(mems2)}")
        click.echo()

        # Show types breakdown
        types1 = {}
        for m in mems1:
            types1[m.memory_type.value] = types1.get(m.memory_type.value, 0) + 1
        types2 = {}
        for m in mems2:
            types2[m.memory_type.value] = types2.get(m.memory_type.value, 0) + 1

        all_types = sorted(set(list(types1.keys()) + list(types2.keys())))
        if all_types:
            click.echo("Memory types comparison:")
            click.echo(f"  {'Type':<15} {'Session A':>10} {'Session B':>10}")
            click.echo(f"  {'-'*15} {'-'*10} {'-'*10}")
            for t in all_types:
                a = types1.get(t, 0)
                b = types2.get(t, 0)
                click.echo(f"  {t:<15} {a:>10} {b:>10}")
            click.echo()

        # Show unique topics
        topics1 = set(session1.key_topics) if session1.key_topics else set()
        topics2 = set(session2.key_topics) if session2.key_topics else set()
        only_a = topics1 - topics2
        only_b = topics2 - topics1
        common = topics1 & topics2

        if common:
            click.echo(f"Common topics: {', '.join(common)}")
        if only_a:
            click.echo(f"Only in A: {', '.join(only_a)}")
        if only_b:
            click.echo(f"Only in B: {', '.join(only_b)}")
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

        display_projects(found)


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


# ── Watch ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--interval", default=5.0, help="Poll interval in seconds")
@click.option("--project", default=None, help="Watch specific project only")
@click.option("--no-generate", is_flag=True, help="Skip auto-generating CLAUDE.md")
@click.pass_context
def watch(ctx: click.Context, interval: float, project: str | None,
          no_generate: bool) -> None:
    """Watch for new sessions and auto-ingest memories.

    Monitors ~/.claude/projects/ for new or modified JSONL session files
    and automatically ingests them through the memory extraction pipeline.

    \b
    Examples:
        claude-memory watch
        claude-memory watch --interval 10
        claude-memory watch --project /path/to/project
        claude-memory watch --no-generate
    """
    db = _get_db(ctx)
    config = MemoryConfig()
    if ctx.obj.get("db_path"):
        config.db_path = ctx.obj["db_path"]

    project_filter = str(Path(project).resolve()) if project else None

    watcher = SessionWatcher(
        db=db,
        config=config,
        interval=interval,
        auto_generate=not no_generate,
        project_filter=project_filter,
    )

    watch_target = project_filter or str(watcher.watch_dir)
    click.echo(f"Watching for new sessions in: {watch_target}")
    click.echo(f"Poll interval: {interval}s | Auto-generate: {not no_generate}")
    click.echo("Press Ctrl+C to stop.\n")

    try:
        watcher.start()
    except KeyboardInterrupt:
        pass
    finally:
        summary = watcher.summary()
        click.echo("\nWatch summary:")
        click.echo(f"  Files processed:    {summary['files_processed']}")
        click.echo(f"  Memories extracted: {summary['memories_extracted']}")
        click.echo(f"  Errors:             {summary['errors']}")
        db.close()


# ── Graph ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option('--project', '-p', default=None, help='Filter by project path')
@click.option('--format', 'fmt', type=click.Choice(['summary', 'dot', 'json']),
              default='summary', help='Output format')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output file path (default: stdout)')
@click.pass_context
def graph(ctx, project, fmt, output):
    """Analyze cross-project memory relationships."""
    db = _get_db(ctx)
    try:
        project_path = _resolve_project(project) if project else None
        builder = GraphBuilder(db)
        g = builder.build(project_path)

        if fmt == 'dot':
            content = builder.export_dot(g)
        elif fmt == 'json':
            content = builder.export_json(g)
        else:
            summary = builder.get_summary(g)
            lines = [
                "Knowledge Graph Summary",
                "=" * 40,
                f"  Nodes:                {summary['node_count']}",
                f"  Edges:                {summary['edge_count']}",
                f"  Clusters:             {summary['cluster_count']}",
                f"  Projects:             {summary['project_count']}",
                f"  Cross-project edges:  {summary['cross_project_edges']}",
            ]
            if summary['hub_memories']:
                lines.append("\n  Hub memories:")
                for h in summary['hub_memories']:
                    lines.append(f"    - {h['title']} (degree: {h['degree']})")
            content = "\n".join(lines)

        if output:
            Path(output).write_text(content, encoding='utf-8')
            click.echo(f"Graph output written to {output}")
        else:
            click.echo(content)
    finally:
        db.close()


@cli.command("shared-patterns")
@click.option('--limit', default=10, help='Max patterns to show')
@click.pass_context
def shared_patterns(ctx, limit):
    """Find patterns and decisions shared across projects."""
    db = _get_db(ctx)
    json_mode = _is_json_mode(ctx)
    try:
        builder = GraphBuilder(db)
        patterns = builder.find_shared_patterns()[:limit]

        if json_mode:
            click.echo(json.dumps(patterns, indent=2, ensure_ascii=False))
        else:
            if not patterns:
                click.echo("No shared patterns found across projects.")
                return
            click.echo(f"Found {len(patterns)} shared patterns:\n")
            for i, p in enumerate(patterns, 1):
                click.echo(f"  {i}. {p['pattern']}")
                click.echo(f"     Projects ({p['count']}): {', '.join(p['projects'])}")
                click.echo()
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
