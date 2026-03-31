"""Tests for the memory extractor."""

from claude_memory.extractor import MemoryExtractor
from claude_memory.models import MemoryType
from claude_memory.parser import parse_session_file


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
    from claude_memory.models import Memory

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
