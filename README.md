# claude-memory

[![PyPI](https://img.shields.io/pypi/v/claude-memory)](https://pypi.org/project/claude-memory/)
[![Tests](https://img.shields.io/badge/tests-259%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Cross-session memory system for Claude Code. Extract, index, and recall context across sessions.

## Features

### Core
- **Automatic Memory Extraction** — rule-based parsing of Claude Code session logs (JSONL) into structured memories
- **Full-Text Search** — SQLite FTS5 with BM25 ranking and Porter stemming
- **CLAUDE.md Generation** — auto-generates project context files that Claude Code reads on session start

### AI-Powered
- **Semantic Search** — sentence-transformer embeddings for meaning-based recall
- **Hybrid Search** — combines FTS5 keyword matching with semantic similarity for best results

### Analysis
- **Knowledge Graph** — visualize cross-project memory relationships and shared patterns
- **Memory Consolidation** — dedup, importance scoring, and stale TODO archival
- **Session Diff & Replay** — compare sessions or replay a session timeline event by event

### Integration
- **MCP Server** — Model Context Protocol server lets Claude query memories in real-time during sessions
- **SessionEnd Hook** — auto-triggers extraction when Claude Code sessions end
- **Watch Mode** — daemon that monitors for new sessions and auto-ingests

### User Interface
- **Web Dashboard** — full-featured browser UI with search, browse, graph, stats, timeline, and session views
- **Rich Terminal** — color-coded tables, panels, and formatted output via the `rich` library
- **23-Command CLI** — comprehensive command-line interface for every operation

## Quick Start

```bash
# 1. Install
pip install claude-memory

# 2. Ingest your Claude Code sessions
claude-memory ingest --all

# 3. Search your memories
claude-memory search "authentication"
```

That's it. Memories are extracted from your existing Claude Code session logs and indexed for instant recall.

## Installation

```bash
pip install claude-memory                    # Core (FTS5 search, CLI, Rich output)
pip install 'claude-memory[web]'             # + Web Dashboard (FastAPI)
pip install 'claude-memory[embeddings]'      # + Semantic search (sentence-transformers)
pip install 'claude-memory[mcp]'             # + MCP server
pip install 'claude-memory[web,embeddings]'  # Full install
```

Or install from source:

```bash
git clone https://github.com/Zey413/claude-memory.git
cd claude-memory
pip install -e ".[dev,web,embeddings,mcp]"
```

## Web Dashboard

Launch with `claude-memory ui` (requires `pip install 'claude-memory[web]'`):

```bash
claude-memory ui --open          # Opens browser to http://localhost:8765
```

The dashboard provides six views:

| View | Description |
|------|-------------|
| **Search** | Real-time keyword and semantic search across all memories |
| **Browse** | Filter and explore memories by type, project, and tags |
| **Graph** | Interactive knowledge graph of memory relationships |
| **Stats** | System-wide statistics, memory type distribution, and project breakdown |
| **Timeline** | Chronological event timeline for session replay |
| **Sessions** | Session history with metadata and extraction summaries |

## CLI Commands

All 23 commands available via `claude-memory <command>`:

| Command | Description |
|---------|-------------|
| **Ingestion** | |
| `ingest` | Extract memories from Claude Code session logs |
| `watch` | Auto-ingest new sessions in daemon mode |
| `install-hook` | Install SessionEnd hook for auto-extraction |
| `uninstall-hook` | Remove the SessionEnd hook |
| **Search & Browse** | |
| `search` | Full-text, semantic (`--semantic`), or hybrid (`--hybrid`) search |
| `list` | List memories with type/project/tag filters |
| `top` | Show top memories ranked by importance score |
| `sessions` | List processed session summaries |
| `projects` | List all discovered projects with session data |
| **Analysis** | |
| `graph` | Analyze cross-project memory relationships (summary/dot/json) |
| `shared-patterns` | Find patterns and decisions shared across projects |
| `consolidate` | Dedup, score, and archive stale memories |
| `diff` | Compare memories between two sessions |
| `replay` | Replay a session timeline showing key events |
| `stats` | Show system-wide memory statistics |
| **Generation & Export** | |
| `generate` | Generate CLAUDE.md project context from memories |
| `export` | Export memories to JSON |
| `import-data` | Import memories from a JSON file |
| **AI & Embeddings** | |
| `embed` | Generate sentence-transformer embeddings for all memories |
| **Servers** | |
| `serve` | Start MCP server for real-time memory access |
| `ui` | Launch the web dashboard |
| **Management** | |
| `tag` | Add/remove/list memory tags |
| `reset` | Reset the memory database (destructive) |

### Global Options

| Option | Description |
|--------|-------------|
| `--db PATH` | Custom database path |
| `--json-output` | Machine-readable JSON output |
| `-v, --verbose` | Enable debug logging |

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `decision` | Architecture and design decisions | "Use FastAPI for the REST API" |
| `pattern` | Code patterns and conventions | "Modified 8 files in /tests" |
| `issue` | Bugs found, errors encountered | "Import error in module X" |
| `solution` | How issues were resolved | "Fixed by adding \_\_init\_\_.py" |
| `preference` | User preferences and style choices | "Prefer pytest over unittest" |
| `context` | Project context, domain knowledge | "Service uses PostgreSQL 15" |
| `todo` | Unfinished work, future plans | "Add API rate limiting" |
| `learning` | New concepts, techniques discovered | "FTS5 supports Porter stemming" |

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
                                                │ SessionEnd hook / watch mode
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
│                     │  - TODOs             │    │  embeddings (optional)   │  │
│                     │  - Error/fix pairs   │    │                          │  │
│                     │  - Preferences       │    │  BM25 ranking            │  │
│                     └──────────────────────┘    │  Porter stemming         │  │
│                                                 └────────────┬─────────────┘  │
│                                                              │                │
│         ┌────────────────────────────────────────────────────┼──────────┐     │
│         │                    │                    │           │          │     │
│         ▼                    ▼                    ▼           ▼          │     │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐ ┌─────────┐   │     │
│  │ CLAUDE.md    │  │ Web Dashboard  │  │ MCP Server   │ │  CLI    │   │     │
│  │ Generator    │  │ (FastAPI)      │  │ (mcp_server) │ │ (click) │   │     │
│  │              │  │                │  │              │ │         │   │     │
│  │ Writes to:   │  │ 6 views:       │  │ 4 tools:     │ │ 23 cmds │   │     │
│  │ memory/      │  │ search, browse │  │ search, list │ │         │   │     │
│  │ context.md   │  │ graph, stats   │  │ stats,       │ │         │   │     │
│  │              │  │ timeline,      │  │ context      │ │         │   │     │
│  │              │  │ sessions       │  │              │ │         │   │     │
│  └──────┬───────┘  └────────────────┘  └──────────────┘ └─────────┘   │     │
│         │                                                              │     │
│         ▼                                                              │     │
│  ┌──────────────────────────────────────────────────────────────────┐  │     │
│  │  Next Claude Code session auto-reads CLAUDE.md                  │  │     │
│  │  → Context from past sessions is available immediately          │  │     │
│  └──────────────────────────────────────────────────────────────────┘  │     │
└───────────────────────────────────────────────────────────────────────────────┘
```

## How It Works — Extraction Principles

### Data Source: Claude Code Session Logs

Claude Code automatically saves every conversation to `~/.claude/projects/<PATH>/<SESSION_ID>.jsonl`. Each line is a JSON object representing one message in the conversation:

```jsonl
{"type":"user","message":{"role":"user","content":"Let's build a REST API"},"timestamp":"...","cwd":"/my/project"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I'll use FastAPI..."},{"type":"tool_use","name":"Write","input":{"file_path":"main.py","content":"..."}}]}}
```

These files contain **everything** — every prompt you typed, every response Claude gave, every file read/written, every command executed.

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

From assistant messages, it extracts **tool_use blocks** — these tell us exactly which files were created (`Write`), edited (`Edit`), read (`Read`), and what shell commands were run (`Bash`).

### Step 2: Extract (extractor.py)

Five rule-based extractors scan the parsed messages using **regex patterns and structural analysis** (no LLM calls needed):

| Extractor | What it finds | How |
|-----------|--------------|-----|
| **Decisions** | Architecture/design choices | Regex: `"let's use X"`, `"decided to X"`, `"going to implement X"` |
| **File Patterns** | Which files were modified together | Track `Write`/`Edit` tool_use blocks, group by directory |
| **TODOs** | Unfinished work | `TaskCreate` tool uses + regex for `TODO`, `FIXME`, `need to` |
| **Error/Fix Pairs** | Bugs and their solutions | Find `Bash` failures → look ahead 5 messages for resolution |
| **Preferences** | User style choices | Regex on user messages: `"I prefer"`, `"always use"`, `"don't use"` |

### Step 3: Store & Index (db.py)

Extracted memories go into SQLite with **FTS5 full-text search**:

```
memories table (id, session_id, project, type, title, content, confidence)
    │
    ├── memories_fts (FTS5 virtual table with Porter stemming + BM25 ranking)
    ├── memory_tags (many-to-many junction table)
    ├── sessions table (summary stats per session)
    └── embeddings table (optional, for semantic search)
```

### Step 4: Generate (generator.py)

The generator aggregates memories into a `CLAUDE.md` file that Claude Code **automatically reads on session start**:

```markdown
# Project Memory — my-project
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

## MCP Server

Claude Code can query memories in real-time during a session:

```bash
claude-memory serve
```

Exposes 4 tools via the Model Context Protocol: `memory_search`, `memory_list`, `memory_stats`, `memory_context`.

Add to your Claude Code MCP config using the `claude-memory-mcp` entry point.

## Development

```bash
# Install all dev dependencies
pip install -e ".[dev,web,embeddings,mcp]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=claude_memory

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## License

MIT
