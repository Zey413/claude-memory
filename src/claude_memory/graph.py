"""Cross-project knowledge graph for memory relationships."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations

from claude_memory.db import MemoryDB
from claude_memory.models import Memory, MemoryType


def _word_set(text: str) -> set[str]:
    """Convert text to a set of lowercase words (simple tokenisation)."""
    return set(text.lower().split())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets of strings."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


@dataclass
class GraphNode:
    """A node in the knowledge graph (represents a memory)."""

    id: str
    title: str
    memory_type: str
    project: str
    tags: list[str]
    importance: float = 0.0


@dataclass
class GraphEdge:
    """An edge connecting two memories."""

    source: str  # memory ID
    target: str  # memory ID
    relationship: str  # "shared_tag", "same_session", "similar_title", "cross_project"
    weight: float = 1.0


@dataclass
class KnowledgeGraph:
    """In-memory graph of memory relationships."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    # Internal adjacency index: node_id -> list of (neighbor_id, edge_index)
    _adjacency: dict[str, list[tuple[str, int]]] = field(
        default_factory=lambda: defaultdict(list), repr=False,
    )

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph. Skips self-loops."""
        if edge.source == edge.target:
            return
        idx = len(self.edges)
        self.edges.append(edge)
        self._adjacency[edge.source].append((edge.target, idx))
        self._adjacency[edge.target].append((edge.source, idx))

    def get_neighbors(self, node_id: str) -> list[tuple[GraphNode, GraphEdge]]:
        """Return a list of (neighbor_node, connecting_edge) for the given node."""
        result: list[tuple[GraphNode, GraphEdge]] = []
        for neighbor_id, edge_idx in self._adjacency.get(node_id, []):
            neighbor_node = self.nodes.get(neighbor_id)
            if neighbor_node is not None:
                result.append((neighbor_node, self.edges[edge_idx]))
        return result

    def get_clusters(self) -> list[list[str]]:
        """Find connected components using BFS.

        Returns a list of clusters, where each cluster is a sorted list
        of node IDs.
        """
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for node_id in self.nodes:
            if node_id in visited:
                continue
            # BFS from this node
            cluster: list[str] = []
            queue: deque[str] = deque([node_id])
            visited.add(node_id)
            while queue:
                current = queue.popleft()
                cluster.append(current)
                for neighbor_id, _edge_idx in self._adjacency.get(current, []):
                    if neighbor_id not in visited and neighbor_id in self.nodes:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            clusters.append(sorted(cluster))

        return clusters

    def degree(self, node_id: str) -> int:
        """Return the number of edges incident to a node."""
        return len(self._adjacency.get(node_id, []))


class GraphBuilder:
    """Build knowledge graph from memory database."""

    def __init__(self, db: MemoryDB):
        self.db = db

    def build(self, project_path: str | None = None) -> KnowledgeGraph:
        """Build full graph from all memories."""
        graph = KnowledgeGraph()
        memories = self.db.get_all_memories(project_path)

        # 1. Add all memories as nodes
        for mem in memories:
            graph.add_node(GraphNode(
                id=mem.id,
                title=mem.title,
                memory_type=mem.memory_type.value,
                project=mem.project_path,
                tags=mem.tags,
                importance=0.0,
            ))

        # 2. Build edges
        self._add_tag_edges(graph, memories)
        self._add_session_edges(graph, memories)
        self._add_title_similarity_edges(graph, memories)
        self._add_cross_project_edges(graph, memories)

        return graph

    def _add_tag_edges(self, graph: KnowledgeGraph, memories: list[Memory]) -> None:
        """Connect memories that share tags.

        Build inverted index: tag -> [memory_ids].
        For each tag with 2+ memories, add edges between all pairs.
        Limit to max 5 edges per tag to avoid explosion.
        """
        tag_index: dict[str, list[str]] = defaultdict(list)
        for mem in memories:
            for tag in mem.tags:
                tag_index[tag].append(mem.id)

        for _tag, mem_ids in tag_index.items():
            if len(mem_ids) < 2:
                continue
            # Limit to 5 edges per tag (take first 6 IDs → max C(6,2)=15,
            # but we cap the pairs themselves)
            pairs = list(combinations(mem_ids, 2))
            for src, tgt in pairs[:5]:
                graph.add_edge(GraphEdge(
                    source=src, target=tgt,
                    relationship="shared_tag", weight=1.0,
                ))

    def _add_session_edges(self, graph: KnowledgeGraph, memories: list[Memory]) -> None:
        """Connect memories from the same session."""
        session_index: dict[str, list[str]] = defaultdict(list)
        for mem in memories:
            session_index[mem.session_id].append(mem.id)

        for _sid, mem_ids in session_index.items():
            if len(mem_ids) < 2:
                continue
            pairs = list(combinations(mem_ids, 2))
            for src, tgt in pairs[:10]:
                graph.add_edge(GraphEdge(
                    source=src, target=tgt,
                    relationship="same_session", weight=0.8,
                ))

    def _add_title_similarity_edges(
        self, graph: KnowledgeGraph, memories: list[Memory],
    ) -> None:
        """Connect memories with similar titles using word overlap.

        Jaccard similarity > 0.3 on title words.
        """
        # Pre-compute word sets for titles
        title_words: list[tuple[str, set[str]]] = []
        for mem in memories:
            ws = _word_set(mem.title)
            if ws:
                title_words.append((mem.id, ws))

        for i in range(len(title_words)):
            mid_i, words_i = title_words[i]
            for j in range(i + 1, len(title_words)):
                mid_j, words_j = title_words[j]
                if mid_i == mid_j:
                    continue
                sim = _jaccard_similarity(words_i, words_j)
                if sim > 0.3:
                    graph.add_edge(GraphEdge(
                        source=mid_i, target=mid_j,
                        relationship="similar_title", weight=sim,
                    ))

    def _add_cross_project_edges(
        self, graph: KnowledgeGraph, memories: list[Memory],
    ) -> None:
        """Find same-type memories across different projects with similar content.

        Group by type, then find cross-project pairs with word overlap > 0.4.
        """
        type_index: dict[str, list[Memory]] = defaultdict(list)
        for mem in memories:
            type_index[mem.memory_type.value].append(mem)

        for _mtype, group in type_index.items():
            # Pre-compute content word sets
            content_data: list[tuple[str, str, set[str]]] = []
            for mem in group:
                ws = _word_set(mem.title + " " + mem.content)
                content_data.append((mem.id, mem.project_path, ws))

            for i in range(len(content_data)):
                mid_i, proj_i, words_i = content_data[i]
                for j in range(i + 1, len(content_data)):
                    mid_j, proj_j, words_j = content_data[j]
                    # Must be different projects
                    if proj_i == proj_j:
                        continue
                    sim = _jaccard_similarity(words_i, words_j)
                    if sim > 0.4:
                        graph.add_edge(GraphEdge(
                            source=mid_i, target=mid_j,
                            relationship="cross_project", weight=sim,
                        ))

    def find_shared_patterns(self) -> list[dict]:
        """Find patterns/decisions that appear across multiple projects.

        Returns: [{pattern, projects: [...], count}]
        """
        # Look at pattern and decision memories
        all_memories = self.db.get_all_memories()
        relevant = [
            m for m in all_memories
            if m.memory_type in (MemoryType.PATTERN, MemoryType.DECISION)
        ]

        # Group by content similarity across projects
        # Use title as the "pattern key" (simplified)
        pattern_map: dict[str, dict[str, set[str]]] = {}  # title_key -> {title, projects}

        for mem in relevant:
            title_lower = mem.title.lower().strip()
            # Find existing similar pattern
            matched = False
            for key, info in pattern_map.items():
                sim = _jaccard_similarity(_word_set(key), _word_set(title_lower))
                if sim > 0.5:
                    info["projects"].add(mem.project_path)
                    matched = True
                    break
            if not matched:
                pattern_map[title_lower] = {
                    "title": mem.title,
                    "projects": {mem.project_path},
                }

        # Filter to patterns appearing in 2+ projects
        results: list[dict] = []
        for _key, info in pattern_map.items():
            if len(info["projects"]) >= 2:
                results.append({
                    "pattern": info["title"],
                    "projects": sorted(info["projects"]),
                    "count": len(info["projects"]),
                })

        results.sort(key=lambda x: x["count"], reverse=True)
        return results

    def find_hub_memories(
        self, graph: KnowledgeGraph, top_n: int = 10,
    ) -> list[GraphNode]:
        """Find most connected memories (hubs).

        Sort by degree centrality.
        """
        node_degrees: list[tuple[int, str]] = []
        for node_id in graph.nodes:
            deg = graph.degree(node_id)
            node_degrees.append((deg, node_id))

        # Sort by degree descending, then by node_id for stability
        node_degrees.sort(key=lambda x: (-x[0], x[1]))

        result: list[GraphNode] = []
        for deg, node_id in node_degrees[:top_n]:
            if deg > 0:
                result.append(graph.nodes[node_id])

        return result

    def export_dot(self, graph: KnowledgeGraph) -> str:
        """Export graph in DOT format for Graphviz visualization."""
        lines: list[str] = ["graph KnowledgeGraph {"]

        # Nodes
        for node_id, node in graph.nodes.items():
            label = node.title.replace('"', '\\"')
            lines.append(
                f'  "{node_id}" [label="{label}" type="{node.memory_type}"];'
            )

        # Edges
        for edge in graph.edges:
            lines.append(
                f'  "{edge.source}" -- "{edge.target}" '
                f'[label="{edge.relationship}" weight={edge.weight:.2f}];'
            )

        lines.append("}")
        return "\n".join(lines)

    def export_json(self, graph: KnowledgeGraph) -> str:
        """Export graph as JSON for web visualization."""
        data = {
            "nodes": [
                {
                    "id": n.id,
                    "title": n.title,
                    "memory_type": n.memory_type,
                    "project": n.project,
                    "tags": n.tags,
                    "importance": n.importance,
                }
                for n in graph.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relationship": e.relationship,
                    "weight": e.weight,
                }
                for e in graph.edges
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def get_summary(self, graph: KnowledgeGraph) -> dict:
        """Return graph statistics.

        Includes: nodes, edges, clusters, hub memories, cross-project connections.
        """
        clusters = graph.get_clusters()
        hubs = self.find_hub_memories(graph, top_n=5)
        cross_project_count = sum(
            1 for e in graph.edges if e.relationship == "cross_project"
        )

        # Collect unique projects
        projects: set[str] = set()
        for node in graph.nodes.values():
            projects.add(node.project)

        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "cluster_count": len(clusters),
            "project_count": len(projects),
            "cross_project_edges": cross_project_count,
            "hub_memories": [
                {"id": h.id, "title": h.title, "degree": graph.degree(h.id)}
                for h in hubs
            ],
        }
