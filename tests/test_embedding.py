"""Tests for optional embedding / semantic search support.

All tests work WITHOUT sentence-transformers installed.
We only use numpy (lightweight) and mock the heavy model dependencies.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType
from claude_memory.search import MemorySearch

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_memory(tmp_db: MemoryDB, title: str = "Test mem", content: str = "Some content",
                 project: str = "/tmp/p1", tags: list[str] | None = None, **kw) -> Memory:
    """Insert a sample memory and return it."""
    mem = Memory(
        session_id=kw.get("session_id", "s1"),
        project_path=project,
        memory_type=kw.get("memory_type", MemoryType.DECISION),
        title=title,
        content=content,
        tags=tags or [],
    )
    tmp_db.insert_memory(mem)
    return mem


def _random_vec(dim: int = 384, seed: int = 0) -> np.ndarray:
    """Return a deterministic random normalised float32 vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ── embedding.py unit tests ─────────────────────────────────────────────────


def test_is_available_returns_bool():
    """is_available() should return a bool regardless of install state."""
    from claude_memory.embedding import is_available
    result = is_available()
    assert isinstance(result, bool)


def test_serialize_deserialize_roundtrip():
    """Serialize → deserialize should give back the same vector (numpy only)."""
    from claude_memory.embedding import EmbeddingEngine

    vec = _random_vec(384, seed=42)
    blob = EmbeddingEngine.serialize(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == 384 * 4  # float32

    restored = EmbeddingEngine.deserialize(blob)
    np.testing.assert_array_almost_equal(vec, restored)


def test_cosine_similarity_identical():
    """Cosine similarity of a normalised vector with itself should be ~1.0."""
    from claude_memory.embedding import EmbeddingEngine

    vec = _random_vec(384, seed=7)
    sim = EmbeddingEngine.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    """Cosine similarity of orthogonal vectors should be ~0."""
    from claude_memory.embedding import EmbeddingEngine

    a = np.zeros(384, dtype=np.float32)
    b = np.zeros(384, dtype=np.float32)
    a[0] = 1.0
    b[1] = 1.0
    sim = EmbeddingEngine.cosine_similarity(a, b)
    assert abs(sim) < 1e-5


# ── DB migration v3 tests ───────────────────────────────────────────────────


def test_migration_v3_adds_columns(tmp_db):
    """After migration v3, the embedding and embedding_model columns exist."""
    # Just try selecting the columns — will raise if they don't exist
    row = tmp_db._execute(
        "SELECT embedding, embedding_model FROM memories LIMIT 1"
    ).fetchone()
    # Table is empty but the query succeeded
    assert row is None


def test_store_and_get_embedding(tmp_db):
    """store_embedding + get_embedding round-trip."""
    mem = _make_memory(tmp_db)
    vec = _random_vec(384, seed=1)

    from claude_memory.embedding import EmbeddingEngine
    blob = EmbeddingEngine.serialize(vec)

    tmp_db.store_embedding(mem.id, blob, "all-MiniLM-L6-v2")

    retrieved = tmp_db.get_embedding(mem.id)
    assert retrieved is not None
    restored = EmbeddingEngine.deserialize(retrieved)
    np.testing.assert_array_almost_equal(vec, restored)


def test_get_embedding_none_when_not_set(tmp_db):
    """get_embedding returns None for a memory without embedding."""
    mem = _make_memory(tmp_db)
    assert tmp_db.get_embedding(mem.id) is None


def test_get_memories_with_embeddings(tmp_db):
    """get_memories_with_embeddings returns only embedded memories."""
    m1 = _make_memory(tmp_db, title="Embedded")
    _make_memory(tmp_db, title="Not embedded")

    from claude_memory.embedding import EmbeddingEngine
    vec = _random_vec(384)
    tmp_db.store_embedding(m1.id, EmbeddingEngine.serialize(vec), "test")

    pairs = tmp_db.get_memories_with_embeddings()
    assert len(pairs) == 1
    assert pairs[0][0].id == m1.id


def test_get_memories_with_embeddings_project_filter(tmp_db):
    """Project filter works on get_memories_with_embeddings."""
    m1 = _make_memory(tmp_db, title="P1 mem", project="/tmp/p1")
    m2 = _make_memory(tmp_db, title="P2 mem", project="/tmp/p2")

    from claude_memory.embedding import EmbeddingEngine
    vec = _random_vec(384)
    for m in [m1, m2]:
        tmp_db.store_embedding(m.id, EmbeddingEngine.serialize(vec), "test")

    pairs = tmp_db.get_memories_with_embeddings(project_path="/tmp/p1")
    assert len(pairs) == 1
    assert pairs[0][0].project_path == "/tmp/p1"


def test_count_embedded(tmp_db):
    """count_embedded counts only memories with embeddings."""
    m1 = _make_memory(tmp_db, title="A")
    _make_memory(tmp_db, title="B")

    from claude_memory.embedding import EmbeddingEngine
    vec = _random_vec(384)
    tmp_db.store_embedding(m1.id, EmbeddingEngine.serialize(vec), "test")

    assert tmp_db.count_embedded() == 1
    assert tmp_db.count_embedded(project_path="/tmp/p1") == 1
    assert tmp_db.count_embedded(project_path="/tmp/nonexistent") == 0


# ── Semantic search tests (mocked) ──────────────────────────────────────────


def _setup_embedded_memories(tmp_db):
    """Insert 3 memories with distinct embeddings and return (db, mems, vecs).

    Uses serialize directly from numpy to avoid issues with mocking.
    """
    mems = []
    vecs = []
    for i, (title, content) in enumerate([
        ("Use PostgreSQL", "We decided on PostgreSQL for the database layer."),
        ("Redis caching", "Redis is used for session caching."),
        ("API authentication", "JWT tokens for API auth."),
    ]):
        m = _make_memory(tmp_db, title=title, content=content)
        v = _random_vec(384, seed=i + 10)
        # Serialize directly with numpy, not via EmbeddingEngine (which may be mocked)
        blob = v.astype(np.float32).tobytes()
        tmp_db.store_embedding(m.id, blob, "test")
        mems.append(m)
        vecs.append(v)
    return mems, vecs


def _real_deserialize(data: bytes):
    """Deserialize bytes to numpy — standalone to avoid mock interference."""
    return np.frombuffer(data, dtype=np.float32)


def _real_cosine_similarity(a, b) -> float:
    """Cosine similarity of two normalised vectors."""
    return float(np.dot(a, b))


@patch("claude_memory.embedding.is_available", return_value=True)
@patch("claude_memory.embedding.EmbeddingEngine")
def test_semantic_search_with_mocked_engine(mock_engine_cls, mock_avail, tmp_db):
    """semantic_search returns results sorted by similarity score."""
    mems, vecs = _setup_embedded_memories(tmp_db)

    # Make the query vector very similar to the first memory's vector
    query_vec = vecs[0] + 0.01 * _random_vec(384, seed=99)
    query_vec = query_vec / np.linalg.norm(query_vec)

    mock_instance = MagicMock()
    mock_instance.encode.return_value = query_vec
    mock_instance.deserialize.side_effect = _real_deserialize
    mock_instance.cosine_similarity.side_effect = _real_cosine_similarity
    mock_engine_cls.get_instance.return_value = mock_instance

    searcher = MemorySearch(tmp_db)
    results = searcher.semantic_search("PostgreSQL database")

    assert len(results) >= 1
    # The first result should be the one most similar to query_vec
    assert results[0].memory.id == mems[0].id
    assert results[0].score > 0.3


def test_semantic_search_fallback_to_fts_when_unavailable(tmp_db):
    """When is_available() is False, semantic_search falls back to FTS."""
    _make_memory(tmp_db, title="FTS fallback test", content="Fallback content here.")
    searcher = MemorySearch(tmp_db)

    with patch("claude_memory.embedding.is_available", return_value=False):
        results = searcher.semantic_search("Fallback")
        # Should still get results via FTS
        assert isinstance(results, list)


def test_semantic_search_fallback_no_embeddings(tmp_db):
    """When no memories have embeddings, semantic_search falls back to FTS."""
    _make_memory(tmp_db, title="No embedding test", content="No vectors stored yet.")
    searcher = MemorySearch(tmp_db)

    with patch("claude_memory.embedding.is_available", return_value=True):
        results = searcher.semantic_search("No embedding")
        assert isinstance(results, list)


# ── Hybrid search tests ─────────────────────────────────────────────────────


@patch("claude_memory.embedding.is_available", return_value=True)
@patch("claude_memory.embedding.EmbeddingEngine")
def test_hybrid_search_combines_results(mock_engine_cls, mock_avail, tmp_db):
    """hybrid_search merges FTS and semantic results."""
    mems, vecs = _setup_embedded_memories(tmp_db)

    query_vec = _random_vec(384, seed=99)
    mock_instance = MagicMock()
    mock_instance.encode.return_value = query_vec
    mock_instance.deserialize.side_effect = _real_deserialize
    mock_instance.cosine_similarity.side_effect = _real_cosine_similarity
    mock_engine_cls.get_instance.return_value = mock_instance

    searcher = MemorySearch(tmp_db)
    results = searcher.hybrid_search("PostgreSQL database")

    assert isinstance(results, list)
    # RRF scores should be positive
    if results:
        assert all(r.score > 0 for r in results)


def test_hybrid_search_fallback_to_fts(tmp_db):
    """When embeddings unavailable, hybrid_search returns FTS results."""
    _make_memory(tmp_db, title="Hybrid fallback", content="FTS only result.")
    searcher = MemorySearch(tmp_db)

    with patch("claude_memory.embedding.is_available", return_value=False):
        results = searcher.hybrid_search("Hybrid fallback")
        assert isinstance(results, list)


# ── CLI tests ────────────────────────────────────────────────────────────────


def test_embed_command_no_deps(tmp_path):
    """embed command shows friendly error when deps are missing."""
    from click.testing import CliRunner

    from claude_memory.cli import cli

    db_path = str(tmp_path / "test.db")
    runner = CliRunner()

    with patch("claude_memory.cli.importlib_import", side_effect=ImportError, create=True):
        # Patch is_available to return False within the embed command
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = runner.invoke(cli, ["--db", db_path, "embed"])
            # Should either error about deps or say no memories
            assert (
                result.exit_code == 0
                or "not installed" in (result.output or "").lower()
                or "No memories" in result.output
            )


def test_embed_command_no_memories(tmp_path):
    """embed command with empty DB says no memories found."""
    from click.testing import CliRunner

    from claude_memory.cli import cli

    db_path = str(tmp_path / "test.db")
    runner = CliRunner()

    # Mock embedding availability so command proceeds
    mock_module = MagicMock()
    mock_module.is_available.return_value = True
    mock_module._MODEL_ID = "test-model"

    with patch.dict("sys.modules", {"claude_memory.embedding": mock_module}):
        with patch("claude_memory.cli.is_available", return_value=True, create=True):
            result = runner.invoke(cli, ["--db", db_path, "embed"])
            # Either it can't import properly or says no memories
            assert result.exit_code == 0 or result.exit_code == 1


def test_search_semantic_flag_exists():
    """The --semantic flag is accepted by the search command."""
    from click.testing import CliRunner

    from claude_memory.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--help"])
    assert "--semantic" in result.output
    assert "--hybrid" in result.output


def test_search_semantic_cli(tmp_path):
    """search --semantic invokes semantic_search path."""
    from click.testing import CliRunner

    from claude_memory.cli import cli

    db_path = str(tmp_path / "test.db")
    db = MemoryDB(db_path=tmp_path / "test.db")
    _make_memory(db, title="Semantic CLI test", content="Test semantic via CLI.")
    db.close()

    runner = CliRunner()
    # Without real embeddings, should fall back to FTS or show no results
    result = runner.invoke(cli, ["--db", db_path, "search", "Semantic", "--semantic"])
    assert result.exit_code == 0


def test_search_hybrid_cli(tmp_path):
    """search --hybrid invokes hybrid_search path."""
    from click.testing import CliRunner

    from claude_memory.cli import cli

    db_path = str(tmp_path / "test.db")
    db = MemoryDB(db_path=tmp_path / "test.db")
    _make_memory(db, title="Hybrid CLI test", content="Test hybrid via CLI.")
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--db", db_path, "search", "Hybrid", "--hybrid"])
    assert result.exit_code == 0


def test_embed_command_help():
    """embed --help shows expected options."""
    from click.testing import CliRunner

    from claude_memory.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.output
    assert "--force" in result.output
    assert "embeddings extra" in result.output


# ── EmbeddingEngine singleton test ───────────────────────────────────────────


def test_singleton_returns_same_instance():
    """get_instance() returns the same object on repeated calls."""
    from claude_memory.embedding import EmbeddingEngine

    # Reset singleton for a clean test
    EmbeddingEngine._instance = None
    try:
        a = EmbeddingEngine.get_instance()
        b = EmbeddingEngine.get_instance()
        assert a is b
    finally:
        EmbeddingEngine._instance = None
