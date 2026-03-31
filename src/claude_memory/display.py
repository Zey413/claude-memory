"""Rich terminal display helpers for claude-memory CLI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from claude_memory.models import Memory, SearchResult, SessionSummary
    from claude_memory.timeline import SessionTimeline

console = Console()

# Memory type styling
TYPE_STYLES: dict[str, str] = {
    "decision": "bold cyan",
    "todo": "bold yellow",
    "pattern": "bold green",
    "issue": "bold red",
    "solution": "bold blue",
    "preference": "bold magenta",
    "context": "dim white",
    "learning": "bold white",
}

TYPE_ICONS: dict[str, str] = {
    "decision": "\U0001f9e0",
    "pattern": "\U0001f4d0",
    "issue": "\U0001f41b",
    "solution": "\u2705",
    "preference": "\u2699\ufe0f",
    "context": "\U0001f4cb",
    "todo": "\U0001f4dd",
    "learning": "\U0001f4a1",
}


def display_memory(memory: Memory, verbose: bool = False, score: float | None = None) -> None:
    """Display a single memory as a rich Panel."""
    mtype = memory.memory_type.value
    style = TYPE_STYLES.get(mtype, "white")
    icon = TYPE_ICONS.get(mtype, "\U0001f4cc")

    # Build title
    title_text = f"{icon} [{mtype.upper()}] {memory.title}"
    if score is not None:
        title_text += f"  (score: {score:.2f})"

    # Build body lines
    lines: list[str] = []
    lines.append(f"[dim]ID: {memory.id} | Session: {memory.session_id[:8]}...[/dim]")
    if memory.tags:
        tag_str = ", ".join(memory.tags)
        lines.append(f"[italic]Tags: {tag_str}[/italic]")

    if verbose:
        lines.append("")
        lines.append(memory.content)
    else:
        # Show preview: first 2 lines
        content_lines = memory.content.split("\n")
        preview = content_lines[0][:100]
        if len(content_lines) > 1:
            preview += f" (+{len(content_lines) - 1} lines)"
        lines.append(preview)

    body = "\n".join(lines)
    panel = Panel(body, title=title_text, title_align="left", border_style=style, padding=(0, 1))
    console.print(panel)


def display_memory_table(memories: list[Memory], title: str = "Memories") -> None:
    """Display memories in a rich Table."""
    table = Table(title=title, show_lines=False, pad_edge=True)
    table.add_column("Type", style="bold", width=12)
    table.add_column("Title", min_width=30)
    table.add_column("Tags", style="italic")
    table.add_column("Session", style="dim", width=10)
    table.add_column("Date", style="dim", width=12)

    for mem in memories:
        mtype = mem.memory_type.value
        style = TYPE_STYLES.get(mtype, "white")
        icon = TYPE_ICONS.get(mtype, "\U0001f4cc")
        type_text = Text(f"{icon} {mtype}", style=style)
        tags_str = ", ".join(mem.tags) if mem.tags else ""
        session_str = mem.session_id[:8] + "..."
        date_str = mem.created_at.strftime("%Y-%m-%d")
        table.add_row(type_text, mem.title, tags_str, session_str, date_str)

    console.print(table)


def display_stats(stats: dict) -> None:
    """Display stats as a rich Panel with two columns."""
    # Build a table inside a panel
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="bold")
    grid.add_column()

    grid.add_row("Total memories:", str(stats["total_memories"]))
    grid.add_row("Total sessions:", str(stats["total_sessions"]))
    grid.add_row("Total tags:", str(stats["total_tags"]))

    db_size = stats["db_size_bytes"]
    if db_size > 1_048_576:
        size_str = f"{db_size / 1_048_576:.1f} MB"
    elif db_size > 1024:
        size_str = f"{db_size / 1024:.1f} KB"
    else:
        size_str = f"{db_size} B"
    grid.add_row("Database size:", size_str)

    panel = Panel(
        grid, title="Claude Memory Statistics",
        border_style="bright_blue", padding=(1, 2),
    )
    console.print(panel)

    # By type
    if stats.get("memories_by_type"):
        type_table = Table(title="By Type", show_lines=False)
        type_table.add_column("Type", style="bold")
        type_table.add_column("Count", justify="right")
        for mtype, count in sorted(stats["memories_by_type"].items()):
            style = TYPE_STYLES.get(mtype, "white")
            type_text = Text(mtype, style=style)
            type_table.add_row(type_text, str(count))
        console.print(type_table)

    # By project
    if stats.get("memories_by_project"):
        proj_table = Table(title="By Project", show_lines=False)
        proj_table.add_column("Project", style="bold")
        proj_table.add_column("Count", justify="right")
        for proj, count in sorted(stats["memories_by_project"].items()):
            name = Path(proj).name if proj else "unknown"
            proj_table.add_row(name, str(count))
        console.print(proj_table)


def display_sessions(sessions: list[SessionSummary]) -> None:
    """Display session list as a rich Table."""
    table = Table(title="Sessions", show_lines=True)
    table.add_column("ID", style="bold", width=10)
    table.add_column("Date", style="dim", width=18)
    table.add_column("Duration", width=10)
    table.add_column("Branch", style="cyan", width=16)
    table.add_column("Summary", min_width=30)
    table.add_column("Topics", style="italic")

    for s in sessions:
        session_str = s.session_id[:8] + "..."
        date_str = s.started_at.strftime("%Y-%m-%d %H:%M")

        duration = ""
        if s.duration_minutes:
            from claude_memory.utils import format_duration
            duration = format_duration(s.duration_minutes)

        branch = s.git_branch or ""
        summary = s.summary_text[:80] if s.summary_text else ""
        topics = ", ".join(s.key_topics[:5]) if s.key_topics else ""
        table.add_row(session_str, date_str, duration, branch, summary, topics)

    console.print(table)


def display_projects(projects: list[tuple]) -> None:
    """Display projects as a rich Tree."""
    tree = Tree("[bold]Claude Projects[/bold]")

    for decoded_path, claude_dir in projects:
        jsonl_count = len(list(claude_dir.glob("*.jsonl")))
        branch = tree.add(f"[bold]{decoded_path}[/bold]")
        branch.add(f"[dim]Sessions: {jsonl_count}[/dim]")

    console.print(tree)


def display_search_results(results: list[SearchResult], query: str) -> None:
    """Display search results with highlighted matches."""
    count = len(results)
    console.print(
        f"\nFound [bold]{count}[/bold] memories"
        f" matching [yellow]\"{query}\"[/yellow]:\n"
    )

    for r in results:
        display_memory(r.memory, score=r.score)


# Timeline event type styling
TIMELINE_EVENT_STYLES: dict[str, str] = {
    "user_message": "bold yellow",
    "tool_use": "bold white",
    "file_write": "bold green",
    "file_edit": "bold cyan",
    "file_read": "dim white",
    "bash_command": "bold magenta",
    "decision": "bold blue",
    "error": "bold red",
    "fix": "bold green",
}

TIMELINE_EVENT_ICONS: dict[str, str] = {
    "user_message": "\U0001f4ac",   # speech balloon
    "tool_use": "\U0001f527",       # wrench
    "file_write": "\U0001f4dd",     # memo
    "file_edit": "\u270f\ufe0f",    # pencil
    "file_read": "\U0001f4c4",      # document
    "bash_command": "\U0001f4bb",   # laptop
    "decision": "\U0001f9e0",       # brain
    "error": "\u274c",              # cross mark
    "fix": "\u2705",                # check mark
}


def display_timeline(
    timeline: SessionTimeline,
    limit: int = 50,
    event_type: str | None = None,
) -> None:
    """Display a session timeline using rich.

    Shows a header panel with session info, a table of events,
    and a summary footer with statistics.
    """
    from claude_memory.utils import format_duration

    # --- Header ---
    header_lines: list[str] = []
    header_lines.append(f"[bold]Session:[/bold] {timeline.session_id}")
    if timeline.project_path:
        header_lines.append(f"[bold]Project:[/bold] {timeline.project_path}")
    if timeline.started_at:
        header_lines.append(
            f"[bold]Started:[/bold] {timeline.started_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    if timeline.duration_minutes is not None:
        header_lines.append(
            f"[bold]Duration:[/bold] {format_duration(timeline.duration_minutes)}"
        )
    header = "\n".join(header_lines)
    console.print(Panel(header, title="Session Timeline", border_style="bright_blue"))

    # --- Filter events ---
    events = timeline.events
    if event_type:
        events = [e for e in events if e.event_type == event_type]
    events = events[:limit]

    if not events:
        console.print("[dim]No events to display.[/dim]")
        return

    # --- Events table ---
    table = Table(show_lines=False, pad_edge=True, expand=True)
    table.add_column("Time", style="dim", width=12)
    table.add_column("Type", width=16)
    table.add_column("Description", min_width=30)
    table.add_column("Files", style="cyan", max_width=30)

    for event in events:
        time_str = event.timestamp.strftime("%H:%M:%S") if event.timestamp else ""
        style = TIMELINE_EVENT_STYLES.get(event.event_type, "white")
        icon = TIMELINE_EVENT_ICONS.get(event.event_type, "\U0001f4cc")
        type_text = Text(f"{icon} {event.event_type}", style=style)
        files_str = ", ".join(Path(f).name for f in event.files) if event.files else ""
        table.add_row(time_str, type_text, event.summary, files_str)

    console.print(table)

    # --- Summary footer ---
    total = len(timeline.events)
    shown = len(events)
    summary_parts: list[str] = []
    summary_parts.append(f"[bold]{total}[/bold] total events")
    if shown < total:
        summary_parts.append(f"([bold]{shown}[/bold] shown)")
    summary_parts.append(
        f"[yellow]{timeline.user_message_count}[/yellow] user messages"
    )
    summary_parts.append(
        f"[magenta]{timeline.tool_use_count}[/magenta] tool uses"
    )
    if timeline.files_modified:
        summary_parts.append(
            f"[green]{len(timeline.files_modified)}[/green] files modified"
        )
    console.print(
        Panel(
            " | ".join(summary_parts),
            title="Summary",
            border_style="dim",
        )
    )


def render_to_string(render_func, *args, **kwargs) -> str:
    """Capture rich output to a string (useful for testing)."""
    string_io = StringIO()
    temp_console = Console(file=string_io, force_terminal=True, width=200)
    # Temporarily swap console
    import claude_memory.display as mod
    original = mod.console
    mod.console = temp_console
    try:
        render_func(*args, **kwargs)
    finally:
        mod.console = original
    return string_io.getvalue()
