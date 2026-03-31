"""Session timeline and replay functionality."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude_memory.parser import ParsedMessage, parse_session_file


@dataclass
class TimelineEvent:
    """A single event in a session timeline."""

    timestamp: datetime | None
    event_type: str  # "user_message", "tool_use", "file_write", "file_read",
    #                  "bash_command", "decision", "error", "fix"
    summary: str  # One-line description
    details: str = ""  # Optional extended info
    files: list[str] = field(default_factory=list)  # Files involved


@dataclass
class SessionTimeline:
    """Complete timeline for a session."""

    session_id: str
    project_path: str
    events: list[TimelineEvent] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def duration_minutes(self) -> float | None:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() / 60
        return None

    @property
    def user_message_count(self) -> int:
        return sum(1 for e in self.events if e.event_type == "user_message")

    @property
    def tool_use_count(self) -> int:
        return sum(
            1
            for e in self.events
            if e.event_type
            in ("tool_use", "file_write", "file_edit", "file_read", "bash_command")
        )

    @property
    def files_modified(self) -> list[str]:
        files: set[str] = set()
        for e in self.events:
            if e.event_type in ("file_write", "file_edit"):
                files.update(e.files)
        return sorted(files)


class TimelineBuilder:
    """Build session timelines from JSONL logs or DB data."""

    def build_from_jsonl(self, filepath: Path) -> SessionTimeline:
        """Parse a JSONL file and build a timeline."""
        messages = parse_session_file(filepath)
        timeline = SessionTimeline(
            session_id=filepath.stem,
            project_path="",
        )

        for msg in messages:
            events = self._message_to_events(msg)
            timeline.events.extend(events)

        # Set start/end times
        if timeline.events:
            timestamps = [e.timestamp for e in timeline.events if e.timestamp]
            if timestamps:
                timeline.started_at = min(timestamps)
                timeline.ended_at = max(timestamps)

        return timeline

    def _message_to_events(self, msg: ParsedMessage) -> list[TimelineEvent]:
        """Convert a ParsedMessage to timeline events."""
        events: list[TimelineEvent] = []

        if msg.role == "user" and msg.text_content and not msg.is_meta:
            # User message -- truncate to 80 chars
            summary = msg.text_content[:80].replace("\n", " ")
            if len(msg.text_content) > 80:
                summary += "..."
            events.append(
                TimelineEvent(
                    timestamp=msg.timestamp,
                    event_type="user_message",
                    summary=summary,
                )
            )

        if msg.role == "assistant":
            for tool in msg.tool_uses:
                event = self._tool_to_event(tool, msg.timestamp)
                if event:
                    events.append(event)

        return events

    def _tool_to_event(self, tool, timestamp) -> TimelineEvent | None:
        """Convert a tool use to a timeline event."""
        name = tool.name
        inp = tool.input_data or {}

        if name == "Write":
            path = inp.get("file_path", "unknown")
            return TimelineEvent(
                timestamp=timestamp,
                event_type="file_write",
                summary=f"Created {Path(path).name}",
                files=[path],
            )
        elif name == "Edit":
            path = inp.get("file_path", "unknown")
            return TimelineEvent(
                timestamp=timestamp,
                event_type="file_edit",
                summary=f"Edited {Path(path).name}",
                files=[path],
            )
        elif name == "Read":
            path = inp.get("file_path", "unknown")
            return TimelineEvent(
                timestamp=timestamp,
                event_type="file_read",
                summary=f"Read {Path(path).name}",
                files=[path],
            )
        elif name == "Bash":
            cmd = inp.get("command", "")
            short_cmd = cmd[:60].replace("\n", " ")
            if len(cmd) > 60:
                short_cmd += "..."
            return TimelineEvent(
                timestamp=timestamp,
                event_type="bash_command",
                summary=f"$ {short_cmd}",
            )
        # Unknown tool types are returned as generic tool_use events
        return TimelineEvent(
            timestamp=timestamp,
            event_type="tool_use",
            summary=f"Tool: {name}",
        )

    def get_activity_summary(self, timeline: SessionTimeline) -> dict:
        """Generate activity summary stats."""
        type_counts: dict[str, int] = {}
        for e in timeline.events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        return {
            "session_id": timeline.session_id,
            "duration_minutes": timeline.duration_minutes,
            "total_events": len(timeline.events),
            "event_counts": type_counts,
            "files_modified": timeline.files_modified,
            "user_messages": timeline.user_message_count,
            "tool_uses": timeline.tool_use_count,
        }
