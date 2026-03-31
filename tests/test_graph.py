"""Tests for the cross-project knowledge graph module."""

from __future__ import annotations

import json
from datetime import datetime

from claude_memory.db import MemoryDB
from claude_memory.graph import (
    GraphBuilder,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
)
from claude_memory.models import Memory, MemoryType

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_memory(
    session_id: str = "s1",
    project_path: str = "/tmp/project",
    memory_type: MemoryType = MemoryType.DECISION,
    title: str = "Test memory",
    content: str = "Some content",
    tags: list[str] | None = None,
    confidence: float = 1.0,
    created_at: datetime | None = None,
) -> Memory:
    """Convenience factory for test memories."""
    mem = Memory(
        session_id=session_id,
        project_path=project_path,
        memory_type=memory_type,
        title=title,
        content=content,
        tags=tags or [],
        confidence=confidence,
    )
    if created_at is not None:
        mem.created_at = created_at
        mem.updated_at = created_at
    return mem


# ── Empty graph ──────────────────────────────────────────────────────────────


def test_build_empty_graph(tmp_db: MemoryDB):
    """No memories → empty graph."""
    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    assert graph.get_clusters() == []


# ── Build with memories ──────────────────────────────────────────────────────


def test_build_with_memories(tmp_db: MemoryDB):
    """Add memories, verify nodes created."""
    m1 = _make_memory(title="Memory one", content="First memory")
    m2 = _make_memory(title="Memory two", content="Second memory")
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    assert len(graph.nodes) == 2
    assert m1.id in graph.nodes
    assert m2.id in graph.nodes
    assert graph.nodes[m1.id].title == "Memory one"
    assert graph.nodes[m2.id].title == "Memory two"


# ── Tag edges ────────────────────────────────────────────────────────────────


def test_tag_edges(tmp_db: MemoryDB):
    """Memories sharing tags get connected."""
    m1 = _make_memory(
        session_id="s1", title="Use pytest for testing",
        content="Testing framework choice", tags=["testing", "python"],
    )
    m2 = _make_memory(
        session_id="s2", title="Use mypy for type checking",
        content="Type checking tool", tags=["testing", "python"],
    )
    m3 = _make_memory(
        session_id="s3", title="Deploy to AWS",
        content="Cloud deployment", tags=["aws"],
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    # m1 and m2 share "testing" and "python" tags → should be connected
    tag_edges = [e for e in graph.edges if e.relationship == "shared_tag"]
    connected_pairs = {(e.source, e.target) for e in tag_edges}
    assert (
        (m1.id, m2.id) in connected_pairs or (m2.id, m1.id) in connected_pairs
    ), "m1 and m2 should be connected via shared tags"

    # m3 should NOT have shared_tag edges to m1 or m2
    for e in tag_edges:
        pair = {e.source, e.target}
        assert not ({m3.id, m1.id} == pair), "m3 should not share tags with m1"
        assert not ({m3.id, m2.id} == pair), "m3 should not share tags with m2"


# ── Session edges ────────────────────────────────────────────────────────────


def test_session_edges(tmp_db: MemoryDB):
    """Same-session memories connected."""
    m1 = _make_memory(session_id="session-A", title="First in session", content="A1")
    m2 = _make_memory(session_id="session-A", title="Second in session", content="A2")
    m3 = _make_memory(session_id="session-B", title="Different session", content="B1")
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    session_edges = [e for e in graph.edges if e.relationship == "same_session"]
    connected_pairs = {frozenset({e.source, e.target}) for e in session_edges}

    assert frozenset({m1.id, m2.id}) in connected_pairs
    assert frozenset({m1.id, m3.id}) not in connected_pairs
    assert frozenset({m2.id, m3.id}) not in connected_pairs


# ── Title similarity edges ───────────────────────────────────────────────────


def test_title_similarity_edges(tmp_db: MemoryDB):
    """Similar titles connected."""
    m1 = _make_memory(
        session_id="s1", title="Use FastAPI for REST API endpoints",
        content="Framework choice",
    )
    m2 = _make_memory(
        session_id="s2", title="Use FastAPI for REST API routes",
        content="Routing decision",
    )
    m3 = _make_memory(
        session_id="s3", title="Deploy PostgreSQL database cluster",
        content="Database deployment",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    title_edges = [e for e in graph.edges if e.relationship == "similar_title"]
    connected_pairs = {frozenset({e.source, e.target}) for e in title_edges}

    # m1 and m2 have very similar titles
    assert frozenset({m1.id, m2.id}) in connected_pairs

    # m3 should NOT be similar to m1 or m2
    assert frozenset({m1.id, m3.id}) not in connected_pairs
    assert frozenset({m2.id, m3.id}) not in connected_pairs


# ── Cross-project edges ──────────────────────────────────────────────────────


def test_cross_project_edges(tmp_db: MemoryDB):
    """Cross-project similar memories connected."""
    m1 = _make_memory(
        project_path="/tmp/projectA", session_id="s1",
        memory_type=MemoryType.PATTERN,
        title="Use structured logging with JSON format",
        content="We use structured logging with JSON format in all services",
    )
    m2 = _make_memory(
        project_path="/tmp/projectB", session_id="s2",
        memory_type=MemoryType.PATTERN,
        title="Use structured logging with JSON format",
        content="We use structured logging with JSON format in all modules",
    )
    m3 = _make_memory(
        project_path="/tmp/projectA", session_id="s1",
        memory_type=MemoryType.PATTERN,
        title="Use structured logging with JSON format in services",
        content="We use structured logging with JSON format in all services here",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    cross_edges = [e for e in graph.edges if e.relationship == "cross_project"]
    cross_pairs = {frozenset({e.source, e.target}) for e in cross_edges}

    # m1 (projectA) and m2 (projectB) are same type, different project, similar content
    assert frozenset({m1.id, m2.id}) in cross_pairs

    # m1 and m3 are same project → no cross-project edge
    assert frozenset({m1.id, m3.id}) not in cross_pairs


# ── Get neighbors ────────────────────────────────────────────────────────────


def test_get_neighbors(tmp_db: MemoryDB):
    """Returns correct neighbor list."""
    graph = KnowledgeGraph()
    n1 = GraphNode(id="a", title="A", memory_type="decision", project="/p", tags=[])
    n2 = GraphNode(id="b", title="B", memory_type="decision", project="/p", tags=[])
    n3 = GraphNode(id="c", title="C", memory_type="decision", project="/p", tags=[])
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    graph.add_edge(GraphEdge(source="a", target="b", relationship="shared_tag"))
    graph.add_edge(GraphEdge(source="a", target="c", relationship="same_session"))

    neighbors = graph.get_neighbors("a")
    assert len(neighbors) == 2
    neighbor_ids = {n.id for n, _e in neighbors}
    assert neighbor_ids == {"b", "c"}

    # b's neighbors should include a
    neighbors_b = graph.get_neighbors("b")
    assert len(neighbors_b) == 1
    assert neighbors_b[0][0].id == "a"


# ── Get clusters ─────────────────────────────────────────────────────────────


def test_get_clusters(tmp_db: MemoryDB):
    """Finds connected components."""
    graph = KnowledgeGraph()
    # Cluster 1: a - b
    graph.add_node(GraphNode(id="a", title="A", memory_type="decision", project="/p", tags=[]))
    graph.add_node(GraphNode(id="b", title="B", memory_type="decision", project="/p", tags=[]))
    graph.add_edge(GraphEdge(source="a", target="b", relationship="shared_tag"))

    # Cluster 2: c - d - e
    graph.add_node(GraphNode(id="c", title="C", memory_type="decision", project="/p", tags=[]))
    graph.add_node(GraphNode(id="d", title="D", memory_type="decision", project="/p", tags=[]))
    graph.add_node(GraphNode(id="e", title="E", memory_type="decision", project="/p", tags=[]))
    graph.add_edge(GraphEdge(source="c", target="d", relationship="same_session"))
    graph.add_edge(GraphEdge(source="d", target="e", relationship="same_session"))

    # Cluster 3: f (isolated)
    graph.add_node(GraphNode(id="f", title="F", memory_type="decision", project="/p", tags=[]))

    clusters = graph.get_clusters()
    assert len(clusters) == 3

    # Sort clusters by size for predictable assertions
    clusters.sort(key=len)
    assert clusters[0] == ["f"]
    assert clusters[1] == ["a", "b"]
    assert clusters[2] == ["c", "d", "e"]


# ── Hub memories ─────────────────────────────────────────────────────────────


def test_hub_memories(tmp_db: MemoryDB):
    """Most connected nodes returned."""
    m1 = _make_memory(
        session_id="s1", title="Hub memory central",
        content="Central", tags=["a", "b", "c"],
    )
    m2 = _make_memory(session_id="s1", title="Leaf memory one", content="Leaf1", tags=["a"])
    m3 = _make_memory(session_id="s1", title="Leaf memory two", content="Leaf2", tags=["b"])
    m4 = _make_memory(session_id="s1", title="Leaf memory three", content="Leaf3", tags=["c"])
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)
    tmp_db.insert_memory(m4)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    hubs = builder.find_hub_memories(graph, top_n=1)
    assert len(hubs) >= 1
    # The hub should be the most connected node — m1 shares tags with all others
    # and is in the same session, so it should have the highest degree
    top_hub = hubs[0]
    assert graph.degree(top_hub.id) >= graph.degree(m2.id)
    assert graph.degree(top_hub.id) >= graph.degree(m3.id)
    assert graph.degree(top_hub.id) >= graph.degree(m4.id)


# ── Shared patterns ──────────────────────────────────────────────────────────


def test_find_shared_patterns(tmp_db: MemoryDB):
    """Cross-project patterns found."""
    m1 = _make_memory(
        project_path="/tmp/projectA", session_id="s1",
        memory_type=MemoryType.PATTERN,
        title="Use structured logging",
        content="Structured logging pattern",
    )
    m2 = _make_memory(
        project_path="/tmp/projectB", session_id="s2",
        memory_type=MemoryType.PATTERN,
        title="Use structured logging",
        content="Structured logging in project B",
    )
    m3 = _make_memory(
        project_path="/tmp/projectC", session_id="s3",
        memory_type=MemoryType.DECISION,
        title="Use PostgreSQL database",
        content="Database decision",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    patterns = builder.find_shared_patterns()

    assert len(patterns) >= 1
    top = patterns[0]
    assert "pattern" in top
    assert "projects" in top
    assert "count" in top
    assert top["count"] >= 2
    assert "/tmp/projectA" in top["projects"]
    assert "/tmp/projectB" in top["projects"]


# ── Export DOT ───────────────────────────────────────────────────────────────


def test_export_dot(tmp_db: MemoryDB):
    """Valid DOT output."""
    m1 = _make_memory(session_id="s1", title="Memory A", content="Content A", tags=["x"])
    m2 = _make_memory(session_id="s1", title="Memory B", content="Content B", tags=["x"])
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()
    dot = builder.export_dot(graph)

    assert dot.startswith("graph KnowledgeGraph {")
    assert dot.strip().endswith("}")
    assert m1.id in dot
    assert m2.id in dot
    assert "Memory A" in dot
    assert "Memory B" in dot


# ── Export JSON ──────────────────────────────────────────────────────────────


def test_export_json(tmp_db: MemoryDB):
    """Valid JSON output."""
    m1 = _make_memory(session_id="s1", title="Memory X", content="Content X", tags=["t"])
    m2 = _make_memory(session_id="s1", title="Memory Y", content="Content Y", tags=["t"])
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()
    json_str = builder.export_json(graph)

    data = json.loads(json_str)
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    node_ids = {n["id"] for n in data["nodes"]}
    assert m1.id in node_ids
    assert m2.id in node_ids

    # Each node should have required fields
    for node in data["nodes"]:
        assert "id" in node
        assert "title" in node
        assert "memory_type" in node
        assert "project" in node
        assert "tags" in node

    # Edges should have required fields
    for edge in data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "relationship" in edge
        assert "weight" in edge


# ── Graph summary ────────────────────────────────────────────────────────────


def test_graph_summary(tmp_db: MemoryDB):
    """Stats are correct."""
    m1 = _make_memory(
        project_path="/tmp/projA", session_id="s1",
        title="Decision A", content="Content A", tags=["shared"],
    )
    m2 = _make_memory(
        project_path="/tmp/projA", session_id="s1",
        title="Decision B", content="Content B", tags=["shared"],
    )
    m3 = _make_memory(
        project_path="/tmp/projB", session_id="s2",
        title="Isolated memory", content="Unique content here",
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()
    summary = builder.get_summary(graph)

    assert summary["node_count"] == 3
    assert summary["edge_count"] >= 1  # at least the shared_tag edge
    assert summary["cluster_count"] >= 1
    assert summary["project_count"] == 2
    assert isinstance(summary["hub_memories"], list)
    assert isinstance(summary["cross_project_edges"], int)


# ── No self-loops ────────────────────────────────────────────────────────────


def test_no_self_loops(tmp_db: MemoryDB):
    """Nodes don't connect to themselves."""
    m1 = _make_memory(
        session_id="s1", title="Self memory",
        content="Should not self-loop", tags=["a"],
    )
    tmp_db.insert_memory(m1)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    for edge in graph.edges:
        assert edge.source != edge.target, (
            f"Self-loop found: {edge.source} -> {edge.target}"
        )

    # Also verify add_edge rejects self-loops explicitly
    graph2 = KnowledgeGraph()
    n = GraphNode(id="x", title="X", memory_type="decision", project="/p", tags=[])
    graph2.add_node(n)
    graph2.add_edge(GraphEdge(source="x", target="x", relationship="shared_tag"))
    assert len(graph2.edges) == 0, "Self-loop should have been rejected"
    assert graph2.degree("x") == 0


# ── New graph tests ─────────────────────────────────────────────────────────


def test_graph_large_dataset(tmp_db: MemoryDB):
    """50 memories → graph builds without crash."""
    for i in range(50):
        mem = _make_memory(
            session_id=f"s{i % 5}",
            project_path=f"/tmp/project-{i % 3}",
            memory_type=MemoryType.DECISION if i % 2 == 0 else MemoryType.PATTERN,
            title=f"Memory number {i} about topic {i % 7}",
            content=f"Content for memory {i} with details about area {i % 7}.",
            tags=[f"tag{i % 4}", f"tag{i % 6}"],
        )
        tmp_db.insert_memory(mem)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()

    assert len(graph.nodes) == 50
    # Should have some edges from shared tags, sessions, etc.
    assert len(graph.edges) > 0


def test_dot_export_valid_syntax(tmp_db: MemoryDB):
    """Verify DOT export starts/ends correctly and has node declarations."""
    m1 = _make_memory(title="Node A", content="Content A", tags=["x"])
    m2 = _make_memory(title="Node B", content="Content B", tags=["x"])
    m3 = _make_memory(title="Node C", content="Content C", tags=["y"])
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)
    tmp_db.insert_memory(m3)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()
    dot = builder.export_dot(graph)

    # Valid DOT format checks
    assert dot.startswith("graph KnowledgeGraph {")
    assert dot.strip().endswith("}")
    # Should have node declarations with labels
    assert 'label="Node A"' in dot
    assert 'label="Node B"' in dot
    assert 'label="Node C"' in dot
    # Should contain type attributes
    assert 'type="decision"' in dot


def test_json_export_parseable(tmp_db: MemoryDB):
    """Verify JSON export is valid JSON with correct structure."""
    m1 = _make_memory(title="JSON A", content="Content A", tags=["t1"])
    m2 = _make_memory(title="JSON B", content="Content B", tags=["t1"])
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()
    json_str = builder.export_json(graph)

    # Must be valid JSON
    data = json.loads(json_str)
    assert isinstance(data, dict)
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2

    # Verify node fields
    for node in data["nodes"]:
        assert "id" in node
        assert "title" in node
        assert "memory_type" in node
        assert "project" in node
        assert "tags" in node
        assert "importance" in node

    # Verify edge fields
    for edge in data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "relationship" in edge
        assert "weight" in edge


def test_shared_patterns_empty(tmp_db: MemoryDB):
    """No cross-project data → empty shared patterns."""
    # All memories in the same project
    m1 = _make_memory(
        project_path="/tmp/same-project",
        memory_type=MemoryType.PATTERN,
        title="Only one project pattern",
        content="Some pattern.",
    )
    tmp_db.insert_memory(m1)

    builder = GraphBuilder(tmp_db)
    patterns = builder.find_shared_patterns()
    assert patterns == []


def test_graph_summary_keys(tmp_db: MemoryDB):
    """Verify get_summary returns all expected keys."""
    m1 = _make_memory(
        project_path="/tmp/projA", title="A1", content="C1", tags=["x"],
    )
    m2 = _make_memory(
        project_path="/tmp/projB", title="B1", content="C2", tags=["x"],
    )
    tmp_db.insert_memory(m1)
    tmp_db.insert_memory(m2)

    builder = GraphBuilder(tmp_db)
    graph = builder.build()
    summary = builder.get_summary(graph)

    expected_keys = {
        "node_count", "edge_count", "cluster_count",
        "project_count", "cross_project_edges", "hub_memories",
    }
    assert expected_keys == set(summary.keys())
    assert summary["node_count"] == 2
    assert summary["project_count"] == 2
    assert isinstance(summary["hub_memories"], list)
    assert isinstance(summary["cross_project_edges"], int)
