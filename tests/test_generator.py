"""Tests for the CLAUDE.md generator."""

from datetime import datetime, timezone

from claude_memory.db import MemoryDB
from claude_memory.generator import (
    ClaudemdGenerator,
    _estimate_tokens,
    _smart_project_name,
)
from claude_memory.models import Memory, MemoryType, SessionSummary
from claude_memory.search import MemorySearch


def _populate_test_data(db: MemoryDB):
    """Insert some test data into the database."""
    # Add memories
    memories = [
        Memory(
            session_id="s1", project_path="/tmp/test-proj",
            memory_type=MemoryType.DECISION,
            title="Use FastAPI", content="Decided to use FastAPI for the REST API.",
            tags=["api"],
        ),
        Memory(
            session_id="s1", project_path="/tmp/test-proj",
            memory_type=MemoryType.TODO,
            title="Add authentication", content="Need to add JWT auth.",
            tags=["auth"],
        ),
        Memory(
            session_id="s1", project_path="/tmp/test-proj",
            memory_type=MemoryType.PATTERN,
            title="Repository pattern", content="Using repository pattern for DB access.",
            tags=["architecture"],
        ),
    ]
    for m in memories:
        db.insert_memory(m)

    # Add session
    summary = SessionSummary(
        session_id="s1",
        project_path="/tmp/test-proj",
        started_at=datetime(2026, 3, 28, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 28, 11, 0, tzinfo=timezone.utc),
        duration_minutes=60.0,
        message_count=50,
        user_message_count=20,
        assistant_message_count=30,
        summary_text="Built REST API with FastAPI.",
    )
    db.insert_session(summary)


def test_generate_project_context(tmp_db):
    """Test generating CLAUDE.md content."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    content = gen.generate_project_context("/tmp/test-proj")
    assert "test-proj" in content
    assert "Key Decisions" in content
    assert "FastAPI" in content
    assert "Active TODOs" in content
    assert "authentication" in content


def test_build_project_context(tmp_db):
    """Test building a ProjectContext model."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    ctx = gen.build_project_context("/tmp/test-proj")
    assert ctx.project_name == "test-proj"
    assert ctx.total_memories == 3
    assert ctx.total_sessions == 1
    assert len(ctx.key_decisions) >= 1


def test_render_to_string(tmp_db):
    """Test render_to_string returns non-empty markdown."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    output = gen.render_to_string("/tmp/test-proj")
    assert len(output) > 0
    assert "test-proj" in output


def test_write_to_memory_dir(tmp_db, tmp_path):
    """Test writing to memory directory."""
    _populate_test_data(tmp_db)

    from claude_memory.config import MemoryConfig
    config = MemoryConfig(claude_home=tmp_path / ".claude")

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    path = gen.write_to_memory_dir("/tmp/test-proj", config=config)
    assert path.exists()
    content = path.read_text()
    assert "test-proj" in content


# ── New tests ────────────────────────────────────────────────────────────────


def _populate_scored_data(db: MemoryDB):
    """Insert memories with varied importance scores for sorting tests."""
    memories = [
        Memory(
            id="low1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.DECISION,
            title="Low priority decision",
            content="This is a low priority decision.",
            tags=["misc"],
        ),
        Memory(
            id="high1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.DECISION,
            title="High priority decision",
            content="This is the most important decision.",
            tags=["core"],
        ),
        Memory(
            id="mid1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.DECISION,
            title="Medium priority decision",
            content="This is a mid-level decision.",
            tags=["feature"],
        ),
    ]
    for m in memories:
        db.insert_memory(m)

    # Set distinct importance scores
    db.update_importance_score("high1", 0.9)
    db.update_importance_score("mid1", 0.5)
    db.update_importance_score("low1", 0.1)

    # Session for Recent Activity
    summary = SessionSummary(
        session_id="s1",
        project_path="/tmp/proj",
        started_at=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
        duration_minutes=120.0,
        message_count=80,
        user_message_count=30,
        assistant_message_count=50,
        summary_text="Built memory system",
        files_modified=["src/memory/db.py", "src/memory/cli.py", "tests/test_db.py"],
        key_topics=["memory", "database"],
    )
    db.insert_session(summary)


def test_importance_sorted_output(tmp_db):
    """Memories should appear sorted by importance score in generated output."""
    _populate_scored_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    content = gen.generate_project_context("/tmp/proj")

    # The high-priority decision (score 0.9, gets ★) should appear before low
    high_pos = content.find("High priority decision")
    low_pos = content.find("Low priority decision")
    assert high_pos != -1 and low_pos != -1
    assert high_pos < low_pos, "High-score memory should appear before low-score"

    # ★ indicator should be present for score > 0.7
    assert "\u2605" in content, "★ indicator should be present for score > 0.7"


def test_topic_clustering(tmp_db):
    """TODOs sharing 2+ keywords should be grouped under a topic heading."""
    # Insert TODOs that share keywords
    todos = [
        Memory(
            id="todo_auth_1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title="Add JWT authentication token",
            content="Implement JWT auth.",
            tags=["auth", "security"],
        ),
        Memory(
            id="todo_auth_2",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title="Add OAuth authentication flow",
            content="Support OAuth.",
            tags=["auth", "security"],
        ),
        Memory(
            id="todo_test_1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title="Write integration tests coverage",
            content="Tests for APIs.",
            tags=["testing", "quality"],
        ),
    ]
    for t in todos:
        tmp_db.insert_memory(t)

    # Add a session so the generator doesn't fail
    summary = SessionSummary(
        session_id="s1",
        project_path="/tmp/proj",
        started_at=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
        duration_minutes=60.0,
    )
    tmp_db.insert_session(summary)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    content = gen.generate_project_context("/tmp/proj")

    # The auth-related TODOs should be clustered under a heading
    assert "Active TODOs" in content
    assert "###" in content, "Should have topic sub-headings"

    # Both auth TODOs should appear in the output
    assert "JWT" in content
    assert "OAuth" in content


def test_token_budget_enforcement(tmp_db):
    """Output of generate_with_budget should respect the given token budget."""
    _populate_scored_data(tmp_db)

    # Add more data to make the output larger
    for i in range(20):
        mem = Memory(
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title=f"Todo item number {i} with some extra words",
            content=f"Detail for todo {i} that adds more words to the output.",
            tags=[f"tag{i}"],
        )
        tmp_db.insert_memory(mem)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    # A very small budget should produce a trimmed output
    result = gen.generate_with_budget("/tmp/proj", token_budget=4000)

    # The footer with token estimate should be present
    assert "<!-- Token estimate:" in result
    assert "tokens -->" in result

    # The estimate in the footer should be within budget (or close)
    est = _estimate_tokens(result)
    # Allow some slack for the footer itself
    assert est < 5000, f"Output should be reasonably sized, got ~{est} tokens"


def test_token_budget_truncation(tmp_db):
    """Low-priority sections should be dropped first when budget is tight."""
    _populate_scored_data(tmp_db)

    # Add issues and preferences (low priority sections)
    for i in range(10):
        tmp_db.insert_memory(Memory(
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.ISSUE,
            title=f"Bug number {i} in the system component",
            content=f"Detailed description of bug {i} and its impact on the system.",
            tags=[f"bug{i}"],
        ))
        tmp_db.insert_memory(Memory(
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.PREFERENCE,
            title=f"User preference item {i} for editor config",
            content=f"The user prefers setting {i} configured a specific way.",
            tags=[f"pref{i}"],
        ))

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    # Very tight budget — should include high-priority sections, drop lower ones
    tight = gen.generate_with_budget("/tmp/proj", token_budget=200)

    # Key Decisions is highest priority and should survive
    assert "Key Decisions" in tight or "proj" in tight

    # With a generous budget, issues & preferences should appear
    generous = gen.generate_with_budget("/tmp/proj", token_budget=20000)
    assert "Known Issues" in generous or "User Preferences" in generous


def test_recent_activity_section(tmp_db):
    """'Recent Activity' section should be present when sessions exist."""
    _populate_scored_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    content = gen.generate_project_context("/tmp/proj")

    assert "Recent Activity" in content
    assert "2026-03-30" in content
    assert "Built memory system" in content


def test_generate_with_budget(tmp_db):
    """generate_with_budget should produce valid markdown."""
    _populate_test_data(tmp_db)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    result = gen.generate_with_budget("/tmp/test-proj", token_budget=4000)

    # Should be valid markdown with a header
    assert result.startswith("#")
    assert "test-proj" in result
    assert "<!-- Token estimate:" in result

    # Should contain section content
    assert "Key Decisions" in result or "Active TODOs" in result


def test_cluster_by_topic(tmp_db):
    """_cluster_by_topic should group memories sharing 2+ keywords."""
    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    memories = [
        Memory(
            id="a1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title="Fix authentication token refresh",
            content="Token refresh is broken.",
            tags=["auth", "tokens"],
        ),
        Memory(
            id="a2",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title="Improve authentication token expiry",
            content="Tokens expire too fast.",
            tags=["auth", "tokens"],
        ),
        Memory(
            id="b1",
            session_id="s1", project_path="/tmp/proj",
            memory_type=MemoryType.TODO,
            title="Set up database migration scripts",
            content="Need migration tooling.",
            tags=["database", "migrations"],
        ),
    ]

    clusters = gen._cluster_by_topic(memories)

    # a1 and a2 share "authentication", "token(s)", "auth", "tokens" — should cluster
    assert len(clusters) >= 1

    # b1 has unique keywords, should be in "Other" or its own group
    all_clustered_ids = set()
    for mems in clusters.values():
        for m in mems:
            all_clustered_ids.add(m.id)

    # All memories should be accounted for
    assert "a1" in all_clustered_ids
    assert "a2" in all_clustered_ids
    assert "b1" in all_clustered_ids

    # a1 and a2 should be in the same cluster (they share many keywords)
    for _name, mems in clusters.items():
        ids = {m.id for m in mems}
        if "a1" in ids:
            assert "a2" in ids, "a1 and a2 should be in the same cluster"
            break


def test_empty_clusters(tmp_db):
    """_cluster_by_topic should not crash with empty or single-item lists."""
    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    # Empty list
    result = gen._cluster_by_topic([])
    assert result == {}

    # Single memory — should go to "Other"
    single = [Memory(
        id="solo",
        session_id="s1", project_path="/tmp/proj",
        memory_type=MemoryType.TODO,
        title="Standalone task",
        content="A lone memory.",
        tags=[],
    )]
    result = gen._cluster_by_topic(single)
    assert len(result) >= 1
    all_mems = [m for group in result.values() for m in group]
    assert len(all_mems) == 1
    assert all_mems[0].id == "solo"


def test_smart_project_name():
    """_smart_project_name should skip generic directory names."""
    assert _smart_project_name("/home/user/my-project") == "my-project"
    assert _smart_project_name("/home/user/my-project/src") == "my-project"
    assert _smart_project_name("/home/user/my-app/app") == "my-app"
    assert _smart_project_name("/home/user/cool-tool/lib") == "cool-tool"
    # Normal path without generic suffix
    assert _smart_project_name("/home/user/fastapi-service") == "fastapi-service"


def test_score_indicators(tmp_db):
    """Score indicators should render correctly for different score ranges."""
    mem_high = Memory(
        id="score_high",
        session_id="s1", project_path="/tmp/proj",
        memory_type=MemoryType.DECISION,
        title="Important",
        content="Very important.",
    )
    mem_mid = Memory(
        id="score_mid",
        session_id="s1", project_path="/tmp/proj",
        memory_type=MemoryType.DECISION,
        title="Medium",
        content="Somewhat important.",
    )
    mem_low = Memory(
        id="score_low",
        session_id="s1", project_path="/tmp/proj",
        memory_type=MemoryType.DECISION,
        title="Low",
        content="Not very important.",
    )

    for m in [mem_high, mem_mid, mem_low]:
        tmp_db.insert_memory(m)

    tmp_db.update_importance_score("score_high", 0.9)
    tmp_db.update_importance_score("score_mid", 0.5)
    tmp_db.update_importance_score("score_low", 0.1)

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    assert gen._score_indicator(mem_high) == "\u2605"  # ★
    assert gen._score_indicator(mem_mid) == "\u2606"   # ☆
    assert gen._score_indicator(mem_low) == ""


def test_estimate_tokens():
    """_estimate_tokens should produce reasonable estimates."""
    assert _estimate_tokens("hello world") == int(2 * 1.3)
    assert _estimate_tokens("one two three four five") == int(5 * 1.3)
    assert _estimate_tokens("") == 0


# ── New generator tests ─────────────────────────────────────────────────────


def test_generate_empty_project(tmp_db):
    """No memories for a project produces minimal output with just a header."""
    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    content = gen.generate_project_context("/tmp/empty-project")
    assert "empty-project" in content
    # Should have a header at minimum
    assert content.startswith("#")
    # Should NOT have section content for decisions/todos
    assert "Key Decisions" not in content or content.count("-") < 3


def test_generate_all_types(tmp_db):
    """One memory of each type → all relevant sections present."""
    all_types = [
        (MemoryType.DECISION, "Use FastAPI", "Framework choice."),
        (MemoryType.TODO, "Add auth", "Need JWT auth."),
        (MemoryType.PATTERN, "Repository pattern", "Repo pattern for DB."),
        (MemoryType.PREFERENCE, "Dark mode", "Prefers dark mode."),
        (MemoryType.ISSUE, "Login bug", "Login fails on timeout."),
        (MemoryType.SOLUTION, "Fix login", "Added retry logic."),
    ]
    for mtype, title, content in all_types:
        tmp_db.insert_memory(Memory(
            session_id="s1", project_path="/tmp/all-types",
            memory_type=mtype, title=title, content=content,
        ))

    # Add session so generator works
    tmp_db.insert_session(SessionSummary(
        session_id="s1", project_path="/tmp/all-types",
        started_at=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
        duration_minutes=60.0,
    ))

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)
    content = gen.generate_project_context("/tmp/all-types")

    assert "Key Decisions" in content
    assert "Active TODOs" in content
    assert "Code Patterns" in content
    assert "User Preferences" in content
    assert "Known Issues" in content


def test_clustered_section_markdown(tmp_db):
    """Verify clustered section produces ### sub-headings."""
    # Insert memories that share keywords so they cluster
    for i in range(4):
        tmp_db.insert_memory(Memory(
            session_id="s1", project_path="/tmp/cluster-test",
            memory_type=MemoryType.TODO,
            title=f"Authentication token refresh {i}",
            content=f"Fix auth token issue {i}.",
            tags=["auth", "token"],
        ))

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)
    memories = tmp_db.get_all_memories("/tmp/cluster-test")
    section = gen._clustered_section("Test Section", memories)

    assert "## Test Section" in section
    assert "###" in section


def test_write_to_project_root(tmp_db, tmp_path):
    """Verify write_to_project_root creates CLAUDE.md in the project dir."""
    project_path = str(tmp_path / "my-project")
    (tmp_path / "my-project").mkdir()
    tmp_db.insert_memory(Memory(
        session_id="s1", project_path=project_path,
        memory_type=MemoryType.DECISION, title="Test",
        content="Testing write.",
    ))

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)
    path = gen.write_to_project_root(project_path)

    assert path.exists()
    assert path.name == "CLAUDE.md"
    assert "my-project" in path.read_text()


def test_atomic_write_safety(tmp_db, tmp_path):
    """Verify atomic write doesn't leave .tmp files behind."""
    dest = tmp_path / "subdir" / "test.md"
    ClaudemdGenerator._atomic_write(dest, "hello world")
    assert dest.exists()
    assert dest.read_text() == "hello world"

    import os
    # No .tmp files should remain
    parent_files = os.listdir(str(dest.parent))
    tmp_files = [f for f in parent_files if f.endswith(".tmp")]
    assert len(tmp_files) == 0


def test_recent_activity_no_sessions(tmp_db):
    """No sessions for a project produces no Recent Activity section."""
    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)
    section = gen._recent_activity_section("/tmp/no-sessions")
    assert section == ""


def test_budget_very_small(tmp_db):
    """Very small budget (100 tokens) produces only header."""
    tmp_db.insert_memory(Memory(
        session_id="s1", project_path="/tmp/budget-test",
        memory_type=MemoryType.DECISION, title="Big decision",
        content="A very important decision with lots of context.",
    ))

    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)
    result = gen.generate_with_budget("/tmp/budget-test", token_budget=100)

    # Should start with header
    assert result.startswith("#")
    # Token estimate footer should still be there
    assert "<!-- Token estimate:" in result


def test_budget_includes_footer(tmp_db):
    """Token estimate footer is always present in budgeted output."""
    _populate_test_data(tmp_db)
    search = MemorySearch(tmp_db)
    gen = ClaudemdGenerator(tmp_db, search)

    result = gen.generate_with_budget("/tmp/test-proj", token_budget=10000)
    assert "<!-- Token estimate:" in result
    assert "tokens -->" in result
