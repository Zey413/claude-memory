# claude-memory

Cross-session memory system for Claude Code. Extract, index, and recall context across sessions.

## Features

- **Automatic Memory Extraction**: Parses Claude Code session logs (JSONL) to extract decisions, patterns, TODOs, issues, solutions, and preferences
- **Full-Text Search**: SQLite FTS5-powered search with BM25 ranking
- **CLAUDE.md Generation**: Auto-generates project context files that Claude Code reads on session start
- **Hook Integration**: SessionEnd hook auto-triggers extraction when sessions end
- **Lightweight**: Only requires `click` and `pydantic` — no vector DB, no LLM API calls

## Installation

```bash
pip install claude-memory
```

Or install from source:

```bash
git clone https://github.com/Zey413/claude-memory.git
cd claude-memory
pip install -e ".[dev]"
```

## Quick Start

### 1. Ingest sessions

```bash
# Ingest the most recent session
claude-memory ingest --latest

# Ingest all sessions for current project
claude-memory ingest --all

# Ingest sessions for a specific project
claude-memory ingest --all --project /path/to/project
```

### 2. Search memories

```bash
claude-memory search "authentication"
claude-memory search "database" --type decision
claude-memory search "pytest" --tag testing
```

### 3. Generate CLAUDE.md

```bash
# Write to ~/.claude/projects/<PATH>/memory/context.md
claude-memory generate

# Write to project root
claude-memory generate --target project_root

# Print to stdout
claude-memory generate --target stdout
```

### 4. Install auto-extraction hook

```bash
claude-memory install-hook
```

This adds a `SessionEnd` hook to `~/.claude/settings.json` that automatically extracts memories when a Claude Code session ends.

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Extract memories from session logs |
| `search` | Full-text search over memories |
| `list` | List memories with filters |
| `sessions` | List processed session summaries |
| `generate` | Generate CLAUDE.md context |
| `stats` | Show system statistics |
| `projects` | List discovered projects |
| `install-hook` | Install SessionEnd hook |
| `uninstall-hook` | Remove SessionEnd hook |
| `tag` | Manage memory tags |
| `reset` | Reset database (destructive) |

## Memory Types

| Type | Description |
|------|-------------|
| `decision` | Architecture/design decisions |
| `pattern` | Code patterns, conventions |
| `issue` | Bugs found, errors encountered |
| `solution` | How issues were resolved |
| `preference` | User preferences, style choices |
| `context` | Project context, domain knowledge |
| `todo` | Unfinished work, future plans |
| `learning` | New concepts, techniques discovered |

## Architecture

```
JSONL session logs -> Parser -> Rule-based Extractor -> SQLite + FTS5
                                                          |
                                        CLI queries + CLAUDE.md generation
                                                          |
                                        SessionEnd hook (auto-trigger)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/ tests/
```

## License

MIT
