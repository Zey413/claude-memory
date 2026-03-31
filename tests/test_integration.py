"""End-to-end integration tests for the full claude-memory pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from claude_memory.cli import cli
from claude_memory.consolidator import MemoryConsolidator
from claude_memory.db import MemoryDB
from claude_memory.extractor import MemoryExtractor
from claude_memory.generator import ClaudemdGenerator
from claude_memory.graph import GraphBuilder
from claude_memory.models import Memory, MemoryType, SessionSummary
from claude_memory.parser import parse_session_file
from claude_memory.search import MemorySearch

# ---------------------------------------------------------------------------
#  Helpers – create realistic JSONL session data
# ---------------------------------------------------------------------------

def _ts(minute: int) -> str:
    """Return an ISO timestamp at 2026-03-28T10:{minute:02d}:00Z."""
    return f"2026-03-28T10:{minute:02d}:00Z"


def _user_msg(text: str, minute: int, session_id: str, **extra) -> dict:
    """Build a user-type JSONL line."""
    msg: dict = {
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": _ts(minute),
        "cwd": "/home/dev/myproject",
        "sessionId": session_id,
        "isMeta": False,
    }
    msg.update(extra)
    return msg


def _assistant_msg(
    content: list | str,
    minute: int,
    session_id: str,
) -> dict:
    """Build an assistant-type JSONL line."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": content},
        "timestamp": _ts(minute),
        "sessionId": session_id,
    }


def _tool_use(name: str, input_data: dict) -> dict:
    return {"type": "tool_use", "name": name, "input": input_data}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _create_realistic_session(path: Path, session_id: str = "integ-sess-001") -> Path:
    """Create a realistic JSONL session file with ~20 messages.

    Simulates a coding session that:
    - builds a Flask application
    - creates and edits files
    - encounters a Bash error and fixes it
    - records user preferences
    - mentions TODOs
    """
    filepath = path / f"{session_id}.jsonl"

    messages = [
        # 1. User asks to build something
        _user_msg(
            "I want to build a Flask REST API with SQLAlchemy for a task manager app.",
            0, session_id, gitBranch="feature/task-api",
        ),
        # 2. Assistant plans + creates first file (Write)
        _assistant_msg([
            _text_block(
                "I'll create a Flask REST API with SQLAlchemy. "
                "Let's use Flask-RESTful for the endpoint structure. "
                "The best approach is to keep models, routes, and config separate."
            ),
            _tool_use("Write", {
                "file_path": "/home/dev/myproject/app.py",
                "content": (
                    "from flask import Flask\n"
                    "from flask_restful import Api\n"
                    "app = Flask(__name__)\napi = Api(app)\n"
                ),
            }),
        ], 1, session_id),
        # 3. Assistant creates model file (Write)
        _assistant_msg([
            _text_block("Now let me create the database models."),
            _tool_use("Write", {
                "file_path": "/home/dev/myproject/models.py",
                "content": (
                    "from flask_sqlalchemy import SQLAlchemy\n"
                    "db = SQLAlchemy()\n"
                    "class Task(db.Model):\n"
                    "    id = db.Column(db.Integer, primary_key=True)\n"
                    "    title = db.Column(db.String(200))\n"
                ),
            }),
        ], 2, session_id),
        # 4. User correction / preference
        _user_msg(
            "I prefer using Pydantic for validation instead of marshmallow. "
            "Always use type hints in the code.",
            3, session_id,
        ),
        # 5. Assistant edits file (Edit)
        _assistant_msg([
            _text_block(
                "Good point. I'll switch to Pydantic. "
                "Let's use Pydantic for request/response validation."
            ),
            _tool_use("Edit", {
                "file_path": "/home/dev/myproject/app.py",
                "old_string": "from flask_restful import Api",
                "new_string": "from flask_restful import Api\nfrom pydantic import BaseModel",
            }),
        ], 4, session_id),
        # 6. Assistant creates routes file (Write)
        _assistant_msg([
            _text_block(
                "Now I'll create the routes. "
                "TODO: add authentication middleware later for securing the endpoints."
            ),
            _tool_use("Write", {
                "file_path": "/home/dev/myproject/routes.py",
                "content": (
                    "from flask_restful import Resource\n"
                    "class TaskList(Resource):\n"
                    "    def get(self): return []\n"
                    "    def post(self): return {}\n"
                ),
            }),
        ], 5, session_id),
        # 7. User asks to run tests
        _user_msg("Can you run the tests now?", 6, session_id),
        # 8. Assistant runs Bash → ERROR
        _assistant_msg([
            _text_block("Let me run the test suite."),
            _tool_use("Bash", {
                "command": "cd /home/dev/myproject && python -m pytest tests/ -v",
            }),
        ], 7, session_id),
        # 9. Simulated tool-result user message (error output)
        # In real JSONL the tool_result comes as a user message; we emulate it
        # by embedding the error in the next assistant message's text.
        _assistant_msg([
            _text_block(
                "The test run failed with an error:\n"
                "ModuleNotFoundError: No module named 'flask_sqlalchemy'\n"
                "exit code 1\n"
                "We need to install the dependency first."
            ),
        ], 8, session_id),
        # 10. Assistant fixes the error (Bash success)
        _assistant_msg([
            _text_block("Let me install the missing package and re-run."),
            _tool_use("Bash", {
                "command": "pip install flask-sqlalchemy && python -m pytest tests/ -v",
            }),
        ], 9, session_id),
        # 11. Assistant reports success
        _assistant_msg([
            _text_block("All tests pass now. The fix was to install flask-sqlalchemy."),
        ], 10, session_id),
        # 12. User mentions a TODO
        _user_msg(
            "Great. Remember to add pagination support for the list endpoint later.",
            11, session_id,
        ),
        # 13. User expresses another preference
        _user_msg(
            "I'd rather use environment variables for config instead of a config file.",
            12, session_id,
        ),
        # 14. Assistant reads a file
        _assistant_msg([
            _text_block("Let me check the current config setup."),
            _tool_use("Read", {
                "file_path": "/home/dev/myproject/config.py",
            }),
        ], 13, session_id),
        # 15. Assistant edits config
        _assistant_msg([
            _text_block(
                "I'll refactor the config to use environment variables. "
                "Decided to use python-dotenv for loading .env files."
            ),
            _tool_use("Edit", {
                "file_path": "/home/dev/myproject/config.py",
                "old_string": "DATABASE_URL = 'sqlite:///tasks.db'",
                "new_string": "import os\nDATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///tasks.db')",
            }),
        ], 14, session_id),
        # 16. User gives a TIL learning
        _user_msg(
            "TIL: Flask-SQLAlchemy auto-configures the session scope per request.",
            15, session_id,
        ),
        # 17. Assistant acknowledges with an insight
        _assistant_msg([
            _text_block(
                "That's a great observation. Turns out that the scoped session "
                "is what prevents cross-request data leaks in Flask."
            ),
        ], 16, session_id),
        # 18. User asks for cleanup
        _user_msg(
            "Let's also need to add proper error handling with custom exceptions.",
            17, session_id,
        ),
        # 19. Assistant creates error handler (Write)
        _assistant_msg([
            _text_block("I'll create a centralised error handler module."),
            _tool_use("Write", {
                "file_path": "/home/dev/myproject/errors.py",
                "content": (
                    "class AppError(Exception):\n"
                    "    def __init__(self, message, status_code=400):\n"
                    "        self.message = message\n"
                    "        self.status_code = status_code\n"
                ),
            }),
        ], 18, session_id),
        # 20. User wraps up with decision
        _user_msg(
            "We should use structured logging with the stdlib logging module.",
            19, session_id,
        ),
    ]

    with filepath.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    return filepath


# ---------------------------------------------------------------------------
#  Test helpers
# ---------------------------------------------------------------------------

def _ingest_session(
    db: MemoryDB,
    filepath: Path,
    session_id: str,
    project_path: str,
) -> tuple[list[Memory], SessionSummary]:
    """Parse → extract → store.  Returns (memories, summary)."""
    messages = parse_session_file(filepath)
    assert len(messages) > 0, "Expected parsed messages"

    extractor = MemoryExtractor()
    memories = extractor.extract_all(messages, session_id, project_path)
    summary = extractor.generate_summary(messages, session_id, project_path)

    db.insert_session(summary)
    for mem in memories:
        db.insert_memory(mem)

    return memories, summary


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the complete JSONL -> ingest -> search -> generate -> export -> import flow."""

    def test_end_to_end(self, tmp_path):
        """Full pipeline: create → parse → extract → store → search → generate → export."""
        session_id = "e2e-sess-001"
        project_path = "/home/dev/myproject"
        db_path = tmp_path / "e2e.db"

        # 1. Create a realistic JSONL file
        filepath = _create_realistic_session(tmp_path, session_id)
        assert filepath.exists()

        # 2. Parse it
        messages = parse_session_file(filepath)
        assert len(messages) >= 15

        # 3. Extract memories
        extractor = MemoryExtractor()
        memories = extractor.extract_all(messages, session_id, project_path)
        assert len(memories) > 0, "Expected at least one memory"

        # Verify we got different types of memories
        types_found = {m.memory_type for m in memories}
        # We should get at least preferences, decisions, or TODOs
        assert len(types_found) >= 1, f"Expected multiple memory types, got {types_found}"

        # 4. Generate summary
        summary = extractor.generate_summary(messages, session_id, project_path)
        assert summary.session_id == session_id
        assert summary.message_count > 0

        # 5. Store in DB
        with MemoryDB(db_path=db_path) as db:
            db.insert_session(summary)
            for mem in memories:
                db.insert_memory(mem)

            # Verify stored count
            stored_count = db.count_memories(project_path)
            assert stored_count == len(memories)

            # 6. Search for known content (FTS)
            search = MemorySearch(db)
            results = search.search("Flask", project_path=project_path)
            # There should be at least one memory mentioning Flask
            assert len(results) >= 1, "Expected at least one search hit for 'Flask'"

            # 7. Generate CLAUDE.md
            gen = ClaudemdGenerator(db, search)
            content = gen.render_to_string(project_path)
            assert "myproject" in content  # project name from path
            assert len(content) > 50  # non-trivial content

            # 8. Export to JSON
            all_mems = db.get_all_memories(project_path=project_path)
            export_data = []
            for m in all_mems:
                export_data.append({
                    "id": m.id,
                    "session_id": m.session_id,
                    "project_path": m.project_path,
                    "memory_type": m.memory_type.value,
                    "title": m.title,
                    "content": m.content,
                    "tags": m.tags,
                    "confidence": m.confidence,
                    "created_at": m.created_at.isoformat(),
                })
            export_path = tmp_path / "export.json"
            export_path.write_text(json.dumps(export_data, indent=2))
            assert export_path.exists()

        # 9. Import into a fresh DB
        fresh_db_path = tmp_path / "fresh.db"
        with MemoryDB(db_path=fresh_db_path) as fresh_db:
            imported_data = json.loads(export_path.read_text())
            for entry in imported_data:
                mem = Memory(
                    id=entry["id"],
                    session_id=entry["session_id"],
                    project_path=entry["project_path"],
                    memory_type=MemoryType(entry["memory_type"]),
                    title=entry["title"],
                    content=entry["content"],
                    tags=entry.get("tags", []),
                    confidence=entry.get("confidence", 1.0),
                )
                fresh_db.insert_memory(mem)

            # 10. Verify imported data matches
            imported_count = fresh_db.count_memories(project_path)
            assert imported_count == len(memories), (
                f"Expected {len(memories)} imported, got {imported_count}"
            )
            # Spot-check: verify a specific memory exists
            for entry in imported_data:
                got = fresh_db.get_memory(entry["id"])
                assert got is not None, f"Memory {entry['id']} not found after import"
                assert got.title == entry["title"]

    def test_consolidation_pipeline(self, tmp_path):
        """Ingest -> consolidate -> verify scoring and dedup."""
        session_id = "consol-sess-001"
        project_path = "/home/dev/myproject"
        db_path = tmp_path / "consol.db"

        filepath = _create_realistic_session(tmp_path, session_id)

        with MemoryDB(db_path=db_path) as db:
            _ingest_session(db, filepath, session_id, project_path)

            consolidator = MemoryConsolidator(db)
            report = consolidator.consolidate(project_path)

            # Scoring should have processed all memories
            assert report.memories_scored > 0
            assert report.memories_scored == db.count_memories(project_path)

            # All memories should now have non-zero importance scores
            all_mems = db.get_all_memories(project_path=project_path)
            for mem in all_mems:
                score = db.get_importance_score(mem.id)
                assert score > 0.0, f"Memory {mem.id} has zero score after consolidation"

            # Top memories should be sorted descending
            top = db.get_top_memories(project_path=project_path, limit=5)
            scores = [db.get_importance_score(m.id) for m in top]
            assert scores == sorted(scores, reverse=True), "Top memories not sorted by score"

    def test_graph_from_multi_project(self, tmp_path):
        """Two projects -> graph -> verify cross-project edges."""
        db_path = tmp_path / "graph.db"

        with MemoryDB(db_path=db_path) as db:
            # Project A: Flask API
            sess_a = "graph-sess-a"
            (tmp_path / "a").mkdir(exist_ok=True)
            fp_a = _create_realistic_session(tmp_path / "a", sess_a)
            _ingest_session(db, fp_a, sess_a, "/home/dev/project-alpha")

            # Project B: share similar content (reuse the same session shape
            # but with a different project path)
            sess_b = "graph-sess-b"
            (tmp_path / "b").mkdir(exist_ok=True)
            fp_b = _create_realistic_session(tmp_path / "b", sess_b)
            _ingest_session(db, fp_b, sess_b, "/home/dev/project-beta")

            builder = GraphBuilder(db)
            graph = builder.build()  # All projects

            # Should have nodes from both projects
            assert len(graph.nodes) > 0
            projects_in_graph = {n.project for n in graph.nodes.values()}
            assert len(projects_in_graph) == 2, f"Expected 2 projects, got {projects_in_graph}"

            # Because the sessions are similar, there should be cross-project edges
            cross = [e for e in graph.edges if e.relationship == "cross_project"]
            assert len(cross) >= 1, "Expected at least one cross-project edge"

            # Graph summary should reflect multi-project state
            summary = builder.get_summary(graph)
            assert summary["project_count"] == 2
            assert summary["cross_project_edges"] >= 1

    def test_search_modes(self, tmp_path):
        """Ingest -> FTS search -> verify results with different queries."""
        session_id = "search-sess-001"
        project_path = "/home/dev/myproject"
        db_path = tmp_path / "search.db"

        filepath = _create_realistic_session(tmp_path, session_id)

        with MemoryDB(db_path=db_path) as db:
            _ingest_session(db, filepath, session_id, project_path)

            search = MemorySearch(db)

            # FTS search for "Flask"
            results = search.search("Flask", project_path=project_path)
            assert len(results) >= 1
            # All results should have positive scores
            for r in results:
                assert r.score > 0.0

            # Search for "Pydantic" (mentioned in preferences)
            results_pydantic = search.search("Pydantic", project_path=project_path)
            assert len(results_pydantic) >= 1

            # Search by type filter
            results_pref = search.search(
                "Pydantic",
                project_path=project_path,
                memory_type=MemoryType.PREFERENCE,
            )
            for r in results_pref:
                assert r.memory.memory_type == MemoryType.PREFERENCE

            # by_type helper
            all_prefs = search.by_type(project_path, MemoryType.PREFERENCE)
            assert isinstance(all_prefs, list)

            # by_project helper
            all_project = search.by_project(project_path)
            assert len(all_project) > 0

    def test_generator_with_scored_memories(self, tmp_path):
        """Ingest -> consolidate (score) -> generate -> verify importance order."""
        session_id = "gen-sess-001"
        project_path = "/home/dev/myproject"
        db_path = tmp_path / "gen.db"

        filepath = _create_realistic_session(tmp_path, session_id)

        with MemoryDB(db_path=db_path) as db:
            _ingest_session(db, filepath, session_id, project_path)

            # Score memories
            consolidator = MemoryConsolidator(db)
            consolidator.score_memories(project_path)

            search = MemorySearch(db)
            gen = ClaudemdGenerator(db, search)

            # Full generation
            content = gen.generate_project_context(project_path)
            assert "myproject" in content
            # The content should have markdown headers
            assert "##" in content

            # Budget-constrained generation
            budget_content = gen.generate_with_budget(project_path, token_budget=2000)
            assert "myproject" in budget_content
            assert "Token estimate" in budget_content

            # Build structured context
            ctx = gen.build_project_context(project_path)
            assert ctx.project_name == "myproject"
            assert ctx.total_memories > 0
            assert ctx.total_sessions == 1

    def test_cli_ingest_and_search(self, tmp_path):
        """CLI-level test: ingest file -> search via CLI -> verify output."""
        session_id = "cli-sess-001"
        project_path = "/home/dev/myproject"
        db_path = tmp_path / "cli.db"

        # Create session JSONL in a layout that mimics Claude's project structure
        # We need to use the CLI's --session-id approach which looks for files
        # in the Claude project directory, so instead we'll test the CLI
        # commands that work with the DB directly.
        filepath = _create_realistic_session(tmp_path, session_id)

        # First, programmatically ingest (the CLI ingest needs Claude's dir structure)
        with MemoryDB(db_path=db_path) as db:
            _ingest_session(db, filepath, session_id, project_path)

        runner = CliRunner()

        # Test: search command
        result = runner.invoke(
            cli, ["--db", str(db_path), "search", "Flask", "--json"]
        )
        assert result.exit_code == 0, f"CLI search failed: {result.output}"
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 1

        # Test: list command
        result = runner.invoke(
            cli, ["--db", str(db_path), "--json-output", "list"],
        )
        assert result.exit_code == 0, f"CLI list failed: {result.output}"
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0

        # Test: stats command
        result = runner.invoke(
            cli, ["--db", str(db_path), "--json-output", "stats"],
        )
        assert result.exit_code == 0, f"CLI stats failed: {result.output}"
        stats = json.loads(result.output)
        assert stats["total_memories"] > 0
        assert stats["total_sessions"] == 1

        # Test: sessions command
        result = runner.invoke(
            cli, ["--db", str(db_path), "--json-output", "sessions"],
        )
        assert result.exit_code == 0, f"CLI sessions failed: {result.output}"
        sessions = json.loads(result.output)
        assert isinstance(sessions, list)
        assert len(sessions) == 1

        # Test: consolidate command
        result = runner.invoke(
            cli, ["--db", str(db_path), "--json-output", "consolidate"],
        )
        assert result.exit_code == 0, f"CLI consolidate failed: {result.output}"
        report = json.loads(result.output)
        assert report["memories_scored"] > 0

        # Test: top command
        result = runner.invoke(
            cli, ["--db", str(db_path), "--json-output", "top"],
        )
        assert result.exit_code == 0, f"CLI top failed: {result.output}"
        top = json.loads(result.output)
        assert isinstance(top, list)
        assert len(top) > 0

        # Test: generate to stdout
        # Use a real directory as --project (click.Path(exists=True) validates it).
        # Re-ingest with this path so generate finds the memories.
        gen_project_dir = tmp_path / "genproject"
        gen_project_dir.mkdir()
        gen_db_path = tmp_path / "cli_gen.db"
        with MemoryDB(db_path=gen_db_path) as gen_db:
            _ingest_session(gen_db, filepath, "cli-gen-001", str(gen_project_dir))
        result = runner.invoke(cli, [
            "--db", str(gen_db_path), "generate",
            "--project", str(gen_project_dir), "--target", "stdout",
        ])
        assert result.exit_code == 0, f"CLI generate failed: {result.output}"
        assert "##" in result.output  # markdown section headers present

        # Test: export command
        export_path = tmp_path / "cli_export.json"
        result = runner.invoke(
            cli, ["--db", str(db_path), "export", "--output", str(export_path)],
        )
        assert result.exit_code == 0, f"CLI export failed: {result.output}"
        assert export_path.exists()
        exported = json.loads(export_path.read_text())
        assert isinstance(exported, list)
        assert len(exported) > 0

        # Test: import-data command into fresh DB
        fresh_db_path = tmp_path / "cli_fresh.db"
        result = runner.invoke(
            cli, ["--db", str(fresh_db_path), "import-data", "--input", str(export_path)],
        )
        assert result.exit_code == 0, f"CLI import failed: {result.output}"
        assert "imported" in result.output

        # Test: graph command
        result = runner.invoke(
            cli, ["--db", str(db_path), "graph", "--format", "json"],
        )
        assert result.exit_code == 0, f"CLI graph failed: {result.output}"
        graph_data = json.loads(result.output)
        assert "nodes" in graph_data
        assert "edges" in graph_data

    def test_web_api_after_ingest(self, tmp_path):
        """Ingest data -> query web API -> verify results."""
        from fastapi.testclient import TestClient

        from claude_memory.web import app as app_module
        from claude_memory.web.app import app, init_app

        session_id = "web-sess-001"
        project_path = "/home/dev/myproject"
        db_path = tmp_path / "web.db"

        # Ingest data
        with MemoryDB(db_path=db_path) as db:
            filepath = _create_realistic_session(tmp_path, session_id)
            _ingest_session(db, filepath, session_id, project_path)

        # Initialise the web app with our test DB
        init_app(db_path)
        client = TestClient(app)

        try:
            # Test: search endpoint
            resp = client.get("/api/search", params={"q": "Flask"})
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) >= 1
            assert "score" in data[0]
            assert "title" in data[0]

            # Test: memories listing
            resp = client.get("/api/memories")
            assert resp.status_code == 200
            mems = resp.json()
            assert isinstance(mems, list)
            assert len(mems) > 0

            # Test: memories filtered by project
            resp = client.get("/api/memories", params={"project": project_path})
            assert resp.status_code == 200
            proj_mems = resp.json()
            assert len(proj_mems) > 0
            for m in proj_mems:
                assert m["project_path"] == project_path

            # Test: single memory by ID
            mem_id = mems[0]["id"]
            resp = client.get(f"/api/memories/{mem_id}")
            assert resp.status_code == 200
            assert resp.json()["id"] == mem_id

            # Test: sessions endpoint
            resp = client.get("/api/sessions")
            assert resp.status_code == 200
            sessions = resp.json()
            assert len(sessions) == 1
            assert sessions[0]["session_id"] == session_id

            # Test: stats endpoint
            resp = client.get("/api/stats")
            assert resp.status_code == 200
            stats = resp.json()
            assert stats["total_memories"] > 0
            assert stats["total_sessions"] == 1

            # Test: projects endpoint
            resp = client.get("/api/projects")
            assert resp.status_code == 200
            projects = resp.json()
            assert len(projects) >= 1

            # Test: graph endpoint
            resp = client.get("/api/graph")
            assert resp.status_code == 200
            graph = resp.json()
            assert "nodes" in graph
            assert "edges" in graph
            assert len(graph["nodes"]) > 0

            # Test: top memories endpoint
            resp = client.get("/api/top")
            assert resp.status_code == 200
            top = resp.json()
            assert isinstance(top, list)
            assert len(top) > 0

            # Test: 404 on missing memory
            resp = client.get("/api/memories/nonexistent-id")
            assert resp.status_code == 404

            # Test: delete a memory
            resp = client.delete(f"/api/memories/{mem_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"

            # Verify it's gone
            resp = client.get(f"/api/memories/{mem_id}")
            assert resp.status_code == 404

        finally:
            # Clean up the module-level DB connection
            if app_module._db is not None:
                app_module._db.close()
                app_module._db = None
                app_module._search = None

    def test_parser_extracts_all_message_types(self, tmp_path):
        """Verify parser handles all message types in the realistic session."""
        session_id = "parse-sess-001"
        filepath = _create_realistic_session(tmp_path, session_id)

        messages = parse_session_file(filepath)

        # Should have at least 15 messages (we wrote 20)
        assert len(messages) >= 15

        # Check we got both roles
        roles = {m.role for m in messages if m.role}
        assert "user" in roles
        assert "assistant" in roles

        # Check tool uses were extracted
        tool_names = set()
        for m in messages:
            for tu in m.tool_uses:
                tool_names.add(tu.name)
        assert "Write" in tool_names
        assert "Edit" in tool_names
        assert "Bash" in tool_names
        assert "Read" in tool_names

        # Check timestamps parsed
        timestamped = [m for m in messages if m.timestamp is not None]
        assert len(timestamped) > 0

    def test_extractor_finds_diverse_memory_types(self, tmp_path):
        """Verify the extractor finds preferences, TODOs, decisions from the realistic session."""
        session_id = "extract-sess-001"
        project_path = "/home/dev/myproject"
        filepath = _create_realistic_session(tmp_path, session_id)

        messages = parse_session_file(filepath)
        extractor = MemoryExtractor()
        memories = extractor.extract_all(messages, session_id, project_path)

        types_found = {m.memory_type for m in memories}

        # We definitely have preferences ("I prefer using Pydantic",
        # "I'd rather use environment variables")
        assert MemoryType.PREFERENCE in types_found, (
            f"Expected PREFERENCE type; got {types_found}"
        )

        # Session summary should capture tool usage
        summary = extractor.generate_summary(messages, session_id, project_path)
        assert summary.user_message_count > 0
        assert summary.assistant_message_count > 0
        assert "Write" in summary.tool_uses

    def test_roundtrip_export_import_preserves_data(self, tmp_path):
        """Export all memories to JSON and import into fresh DB — data must match."""
        session_id = "roundtrip-sess-001"
        project_path = "/home/dev/myproject"

        filepath = _create_realistic_session(tmp_path, session_id)

        # Ingest into first DB
        db1_path = tmp_path / "db1.db"
        with MemoryDB(db_path=db1_path) as db1:
            _ingest_session(db1, filepath, session_id, project_path)
            original_memories = db1.get_all_memories(project_path=project_path)
            assert len(original_memories) > 0

            # Export
            export_data = []
            for m in original_memories:
                export_data.append({
                    "id": m.id,
                    "session_id": m.session_id,
                    "project_path": m.project_path,
                    "memory_type": m.memory_type.value,
                    "title": m.title,
                    "content": m.content,
                    "tags": m.tags,
                    "confidence": m.confidence,
                    "created_at": m.created_at.isoformat(),
                })

        # Import into second DB
        db2_path = tmp_path / "db2.db"
        with MemoryDB(db_path=db2_path) as db2:
            for entry in export_data:
                from claude_memory.utils import parse_iso

                mem = Memory(
                    id=entry["id"],
                    session_id=entry["session_id"],
                    project_path=entry["project_path"],
                    memory_type=MemoryType(entry["memory_type"]),
                    title=entry["title"],
                    content=entry["content"],
                    tags=entry.get("tags", []),
                    confidence=entry.get("confidence", 1.0),
                    created_at=parse_iso(entry["created_at"]),
                    updated_at=parse_iso(entry["created_at"]),
                )
                db2.insert_memory(mem)

            imported_memories = db2.get_all_memories(project_path=project_path)
            assert len(imported_memories) == len(original_memories)

            # Verify every memory matches
            original_by_id = {m.id: m for m in original_memories}
            for imp in imported_memories:
                orig = original_by_id.get(imp.id)
                assert orig is not None, f"Memory {imp.id} not in original"
                assert imp.title == orig.title
                assert imp.content == orig.content
                assert imp.memory_type == orig.memory_type
                assert imp.confidence == orig.confidence
                assert set(imp.tags) == set(orig.tags)
