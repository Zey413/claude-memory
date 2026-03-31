"""Tests for the memory extractor."""

from claude_memory.extractor import MemoryExtractor
from claude_memory.models import Memory, MemoryType
from claude_memory.parser import ParsedMessage, ToolUse, parse_session_file


def test_extract_all(sample_jsonl):
    """Test the full extraction pipeline."""
    filepath, session_id = sample_jsonl
    messages = parse_session_file(filepath)
    extractor = MemoryExtractor()

    memories = extractor.extract_all(messages, session_id, "/tmp/test-project")
    assert len(memories) > 0

    # Should extract at least one decision (FastAPI)
    types = {m.memory_type for m in memories}
    # We should get some combination of decisions, patterns, todos
    assert len(types) >= 1


def test_extract_decisions(sample_jsonl):
    """Test decision extraction from session."""
    filepath, session_id = sample_jsonl
    messages = parse_session_file(filepath)
    extractor = MemoryExtractor()

    decisions = extractor._extract_decisions(messages, session_id, "/tmp/test-project")
    # "Let's use FastAPI" should be detected
    assert len(decisions) >= 1
    assert all(m.memory_type == MemoryType.DECISION for m in decisions)


def test_extract_file_patterns(sample_jsonl):
    """Test file pattern extraction."""
    filepath, session_id = sample_jsonl
    messages = parse_session_file(filepath)
    extractor = MemoryExtractor()

    patterns = extractor._extract_file_patterns(messages, session_id, "/tmp/test-project")
    assert len(patterns) >= 1
    assert all(m.memory_type == MemoryType.PATTERN for m in patterns)


def test_extract_todos(sample_jsonl):
    """Test TODO extraction."""
    filepath, session_id = sample_jsonl
    messages = parse_session_file(filepath)
    extractor = MemoryExtractor()

    todos = extractor._extract_todos(messages, session_id, "/tmp/test-project")
    # Should find "Add API documentation" from TaskCreate
    # and/or "add integration tests later" from text
    assert len(todos) >= 1
    assert all(m.memory_type == MemoryType.TODO for m in todos)


def test_generate_summary(sample_jsonl):
    """Test session summary generation."""
    filepath, session_id = sample_jsonl
    messages = parse_session_file(filepath)
    extractor = MemoryExtractor()

    summary = extractor.generate_summary(messages, session_id, "/tmp/test-project")
    assert summary.session_id == session_id
    assert summary.message_count > 0
    assert summary.user_message_count >= 2
    assert summary.assistant_message_count >= 2
    assert len(summary.files_modified) >= 2  # main.py and test_main.py
    assert summary.summary_text  # Non-empty summary


def test_deduplication():
    """Test that duplicate memories are removed."""
    extractor = MemoryExtractor()
    mem1 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.DECISION,
        title="Use FastAPI", content="Let's use FastAPI",
    )
    mem2 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.DECISION,
        title="Use FastAPI copy", content="Let's use FastAPI",
    )
    mem3 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.TODO,
        title="Add tests", content="Need to add tests",
    )

    result = extractor._deduplicate([mem1, mem2, mem3])
    assert len(result) == 2  # mem1 and mem3 (mem2 is duplicate content)


# --------------------------------------------------------------------------- #
#  New tests for improved extraction quality
# --------------------------------------------------------------------------- #


def _make_msg(
    index: int,
    msg_type: str,
    text: str,
    role: str | None = None,
    tool_uses: list[ToolUse] | None = None,
) -> ParsedMessage:
    """Helper to build a ParsedMessage for unit tests."""
    return ParsedMessage(
        index=index,
        msg_type=msg_type,
        role=role or msg_type,
        text_content=text,
        tool_uses=tool_uses or [],
    )


def test_extract_errors_and_fixes():
    """Error with Bash tool followed by a successful fix should produce ISSUE + SOLUTION."""
    extractor = MemoryExtractor()

    error_tool = ToolUse(
        name="Bash",
        input_data={"command": "npm install"},
        output="Permission denied: /usr/local/lib",
    )
    fix_tool = ToolUse(
        name="Bash",
        input_data={"command": "sudo npm install"},
        output="added 42 packages in 3s",
    )

    messages = [
        _make_msg(0, "assistant", "Running install…", tool_uses=[error_tool]),
        _make_msg(1, "assistant", "Retrying with sudo", tool_uses=[fix_tool]),
    ]

    memories = extractor._extract_errors_and_fixes(messages, "s1", "/tmp/proj")

    issues = [m for m in memories if m.memory_type == MemoryType.ISSUE]
    solutions = [m for m in memories if m.memory_type == MemoryType.SOLUTION]
    assert len(issues) >= 1
    assert len(solutions) >= 1
    assert "Permission denied" in issues[0].content


def test_extract_errors_new_indicators():
    """New error indicators (Connection refused, OOM, segfault, etc.) should be detected."""
    extractor = MemoryExtractor()

    for error_text in [
        "Connection refused on port 5432",
        "segfault at 0x0",
        "process killed",
        "OOM: cannot allocate memory",
        "DEPRECATION warning: this API is removed in v2",
        "WARN: config file missing",
        "Timeout waiting for response",
    ]:
        tool = ToolUse(name="Bash", input_data={"command": "run"}, output=error_text)
        messages = [_make_msg(0, "assistant", "Running…", tool_uses=[tool])]
        memories = extractor._extract_errors_and_fixes(messages, "s1", "/tmp/p")
        assert len(memories) >= 1, f"Should detect error in: {error_text}"
        assert memories[0].memory_type == MemoryType.ISSUE


def test_extract_preferences_extended():
    """New preference patterns should be detected."""
    extractor = MemoryExtractor()

    cases = [
        ("my preference is tabs over spaces", "my preference is"),
        ("I'd rather use yarn than npm", "I'd rather"),
        ("can we use TypeScript for this project", "can we use"),
        ("instead of using REST, let's try GraphQL", "instead of using"),
    ]

    for text, expected_frag in cases:
        messages = [_make_msg(0, "user", text, role="user")]
        prefs = extractor._extract_preferences(messages, "s1", "/tmp/p")
        assert len(prefs) >= 1, f"Should detect preference in: {text!r}"
        assert prefs[0].memory_type == MemoryType.PREFERENCE
        assert expected_frag.lower() in prefs[0].title.lower()


def test_extract_learnings():
    """Learning patterns on both assistant and user messages should be extracted."""
    extractor = MemoryExtractor()

    # Assistant learning patterns
    assistant_texts = [
        "turns out the config needs to be in YAML format",
        "the trick is to set NODE_ENV before running the build",
        "key insight: the cache expires after 5 minutes",
        "important to note that the API rate-limits at 100 req/s",
        "the reason is that Python GIL prevents true parallelism",
    ]
    for text in assistant_texts:
        messages = [_make_msg(0, "assistant", text)]
        learnings = extractor._extract_learnings(messages, "s1", "/tmp/p")
        assert len(learnings) >= 1, f"Should detect learning in assistant msg: {text!r}"
        assert learnings[0].memory_type == MemoryType.LEARNING
        assert learnings[0].confidence == 0.65

    # User learning patterns
    user_texts = [
        "TIL: you can use --force to override the lock",
        "I learned that you need to restart after config changes",
        "now I understand why the tests were flaky",
        "good to know that the API supports pagination",
    ]
    for text in user_texts:
        messages = [_make_msg(0, "user", text, role="user")]
        learnings = extractor._extract_learnings(messages, "s1", "/tmp/p")
        assert len(learnings) >= 1, f"Should detect learning in user msg: {text!r}"
        assert learnings[0].memory_type == MemoryType.LEARNING


def test_extract_learnings_in_pipeline():
    """Learnings should appear in the full extract_all pipeline."""
    extractor = MemoryExtractor()
    messages = [
        _make_msg(0, "assistant", "turns out the config needs a trailing slash"),
    ]
    memories = extractor.extract_all(messages, "s1", "/tmp/p")
    learning_mems = [m for m in memories if m.memory_type == MemoryType.LEARNING]
    assert len(learning_mems) >= 1


def test_decision_filtering():
    """Questions should be filtered out of decision extraction."""
    extractor = MemoryExtractor()

    # These look like decisions but are actually questions — should be filtered
    question_texts = [
        "should we use Redis for caching?",
        "shall we go with PostgreSQL?",
        "can we implement this with async?",
    ]
    for text in question_texts:
        messages = [_make_msg(0, "user", text, role="user")]
        decisions = extractor._extract_decisions(messages, "s1", "/tmp/p")
        assert len(decisions) == 0, f"Question should be filtered: {text!r}"

    # These are real decisions — should NOT be filtered
    decision_texts = [
        "the plan is to use Redis for caching",
        "we'll go with PostgreSQL for the database",
        "I'm going to implement a retry mechanism",
        "the approach is to use event sourcing",
    ]
    for text in decision_texts:
        messages = [_make_msg(0, "user", text, role="user")]
        decisions = extractor._extract_decisions(messages, "s1", "/tmp/p")
        assert len(decisions) >= 1, f"Decision should be detected: {text!r}"
        assert decisions[0].memory_type == MemoryType.DECISION


def test_dedup_normalization():
    """Whitespace-only differences should be deduped, case-insensitive titles should dedup."""
    extractor = MemoryExtractor()

    # Whitespace normalization: content differs only by whitespace
    mem1 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.DECISION,
        title="Use Redis", content="Let's   use   Redis   for caching",
        confidence=0.7,
    )
    mem2 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.DECISION,
        title="Use Redis v2", content="Let's use Redis for caching",
        confidence=0.8,
    )
    result = extractor._deduplicate([mem1, mem2])
    assert len(result) == 1
    # Should keep the higher-confidence one
    assert result[0].confidence == 0.8

    # Case-insensitive title dedup
    mem3 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.DECISION,
        title="Use PostgreSQL", content="Content A",
        confidence=0.6,
    )
    mem4 = Memory(
        session_id="s1", project_path="/tmp", memory_type=MemoryType.DECISION,
        title="use postgresql", content="Content B (different)",
        confidence=0.9,
    )
    result2 = extractor._deduplicate([mem3, mem4])
    assert len(result2) == 1
    # Should keep higher-confidence one
    assert result2[0].confidence == 0.9


def test_empty_session_extraction():
    """Empty message list should produce no memories and not raise."""
    extractor = MemoryExtractor()

    memories = extractor.extract_all([], "s1", "/tmp/p")
    assert memories == []
