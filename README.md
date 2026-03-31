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
| `export` | Export memories to JSON |
| `import-data` | Import memories from JSON |
| `consolidate` | Dedup, score, archive stale memories |
| `top` | Show top memories by importance score |
| `diff` | Compare memories between sessions |
| `watch` | Auto-ingest new sessions (daemon mode) |
| `serve` | Start MCP server for real-time access |
| `reset` | Reset database (destructive) |

### Global Options

| Option | Description |
|--------|-------------|
| `--json-output` | Machine-readable JSON output |
| `-v, --verbose` | Enable debug logging |
| `--db PATH` | Custom database path |

## v0.3.0 Features

### MCP Server Integration

Claude Code can query memories in real-time during a session:

```bash
# Start the MCP server
claude-memory serve

# Or add to your Claude Code MCP config:
# claude-memory-mcp (entry point)
```

Exposes 4 tools: `memory_search`, `memory_list`, `memory_stats`, `memory_context`.

### Watch Mode (Auto-Ingestion)

```bash
# Watch for new sessions and auto-ingest
claude-memory watch

# Custom poll interval and project filter
claude-memory watch --interval 10 --project /path/to/project
```

### Memory Consolidation

```bash
# Score, dedup, and archive stale memories
claude-memory consolidate

# Preview changes without applying
claude-memory consolidate --dry-run

# Show top memories by importance
claude-memory top --limit 5
```

### Rich Terminal UI

Beautiful output with tables, panels, and color-coded memory types powered by the `rich` library.

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
                         ┌──────────────────────────────────────────────────┐
                         │           Claude Code Runtime                    │
                         │                                                  │
                         │  Every session is recorded as a .jsonl file:     │
                         │  ~/.claude/projects/<PATH>/<SESSION_ID>.jsonl    │
                         │                                                  │
                         │  Each line = one JSON message:                   │
                         │    - user messages (prompts, preferences)        │
                         │    - assistant messages (text + tool_use blocks) │
                         │    - file-history-snapshot, queue-operation      │
                         └──────────────────────┬───────────────────────────┘
                                                │
                                                │ SessionEnd hook triggers
                                                │ `claude-memory ingest --latest`
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          claude-memory Pipeline                               │
│                                                                               │
│  ┌─────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │             │    │                      │    │                          │  │
│  │  Parser     │───▶│  Rule-based          │───▶│  SQLite + FTS5          │  │
│  │  (parser.py)│    │  Extractor           │    │  (db.py)                │  │
│  │             │    │  (extractor.py)       │    │                          │  │
│  │ JSONL ──▶   │    │                      │    │  memories table          │  │
│  │ ParsedMsg   │    │  5 extraction rules: │    │  sessions table          │  │
│  │             │    │  - Decisions          │    │  tags table              │  │
│  └─────────────┘    │  - File patterns     │    │  memories_fts (FTS5)     │  │
│                     │  - TODOs             │    │                          │  │
│                     │  - Error/fix pairs   │    │  BM25 ranking            │  │
│                     │  - Preferences       │    │  Porter stemming         │  │
│                     └──────────────────────┘    └────────────┬─────────────┘  │
│                                                              │                │
│                     ┌────────────────────────────────────────┼────────────┐   │
│                     │                                        │            │   │
│                     ▼                                        ▼            │   │
│          ┌──────────────────┐                     ┌─────────────────┐     │   │
│          │  CLAUDE.md       │                     │  CLI            │     │   │
│          │  Generator       │                     │  (cli.py)       │     │   │
│          │  (generator.py)  │                     │                 │     │   │
│          │                  │                     │  search, list,  │     │   │
│          │  Writes to:      │                     │  ingest, stats, │     │   │
│          │  - memory/       │                     │  generate, ...  │     │   │
│          │    context.md    │                     │                 │     │   │
│          │  - CLAUDE.md     │                     │  11 commands    │     │   │
│          └──────────────────┘                     └─────────────────┘     │   │
│                     │                                                     │   │
│                     ▼                                                     │   │
│          ┌──────────────────────────────────────────────────────────┐     │   │
│          │  Next Claude Code session auto-reads CLAUDE.md          │     │   │
│          │  → Context from past sessions is available immediately  │     │   │
│          └──────────────────────────────────────────────────────────┘     │   │
└───────────────────────────────────────────────────────────────────────────────┘
```

## How It Works — Extraction Principles

### Data Source: Claude Code Session Logs

Claude Code automatically saves every conversation to `~/.claude/projects/<PATH>/<SESSION_ID>.jsonl`. Each line is a JSON object representing one message in the conversation:

```jsonl
{"type":"user","message":{"role":"user","content":"Let's build a REST API"},"timestamp":"...","cwd":"/my/project"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I'll use FastAPI..."},{"type":"tool_use","name":"Write","input":{"file_path":"main.py","content":"..."}}]}}
```

These files contain **everything** — every prompt you typed, every response Claude gave, every file read/written, every command executed. For the quant project, 7 sessions produced **3,635 JSONL lines (7.1 MB)** of raw conversation data.

### Step 1: Parse (parser.py)

The parser reads each JSONL line and converts it into a structured `ParsedMessage`:

```
Raw JSONL line  ──▶  ParsedMessage
                       ├── role: "user" | "assistant"
                       ├── text_content: (XML tags stripped)
                       ├── tool_uses: [ToolUse(name, input_data)]
                       ├── timestamp, cwd, git_branch
                       └── is_meta: (skip /clear, /model commands)
```

From the assistant messages, it extracts **tool_use blocks** — these tell us exactly which files were created (`Write`), edited (`Edit`), read (`Read`), and what shell commands were run (`Bash`).

### Step 2: Extract (extractor.py)

Five rule-based extractors scan the parsed messages using **regex patterns and structural analysis** (no LLM calls needed):

| Extractor | What it finds | How | Example |
|-----------|--------------|-----|---------|
| **Decisions** | Architecture/design choices | Regex: `"let's use X"`, `"decided to X"`, `"going to implement X"` | "Let's use FastAPI for the REST API" |
| **File Patterns** | Which files were modified together | Track `Write`/`Edit` tool_use blocks, group by directory | "Modified 8 files in /tests" |
| **TODOs** | Unfinished work | `TaskCreate` tool uses + regex for `TODO`, `FIXME`, `need to` | "Add API documentation" |
| **Error/Fix Pairs** | Bugs and their solutions | Find `Bash` failures (error keywords in output) → look ahead 5 messages for resolution | "Fixed import error by..." |
| **Preferences** | User style choices | Regex on user messages: `"I prefer"`, `"always use"`, `"don't use"` | "I prefer pytest over unittest" |

### Step 3: Store & Index (db.py)

Extracted memories go into SQLite with **FTS5 full-text search**:

```
memories table (id, session_id, project, type, title, content, confidence)
    │
    ├── memories_fts (FTS5 virtual table with Porter stemming + BM25 ranking)
    ├── memory_tags (many-to-many junction table)
    └── sessions table (summary stats per session)
```

### Step 4: Generate (generator.py)

The generator aggregates memories into a `CLAUDE.md` file that Claude Code **automatically reads on session start**:

```markdown
# Project Memory — quant
> Auto-generated by claude-memory

## Key Decisions
- Use Alembic for database migrations
- ...

## Active TODOs
- [ ] Add rollback support
- ...

## Recent Sessions
- 2026-03-30 (13h): Built memory system, 77 interactions
- ...
```

### Real-World Results (quant project)

| Metric | Value |
|--------|-------|
| Sessions processed | 7 |
| Total JSONL lines | 3,635 |
| Raw data size | 7.1 MB |
| User interactions | 1,120 |
| Tool invocations | 836 |
| **Memories extracted** | **117** |
| Breakdown: TODOs | 80 (68%) |
| Breakdown: Patterns | 31 (27%) |
| Breakdown: Decisions | 4 (3%) |
| Breakdown: Preferences | 2 (2%) |
| Database size | 248 KB |

The 117 memories are distilled from 7.1 MB of raw conversation into 248 KB of structured, searchable knowledge — a **29x compression ratio** while retaining the most important context.

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
