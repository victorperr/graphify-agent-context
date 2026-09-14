"""Deterministic, dependency-light access to Graphify's node-link output."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping


class GraphifyStoreError(ValueError):
    """Raised when a Graphify document cannot be queried safely."""


class GraphifyStore:
    """Query a Graphify graph without making the LLM responsible for graph logic.

    The store deliberately keeps results small and explainable. Every returned
    node and edge retains its original metadata, while context packets also
    include a human-readable evidence trail for agent prompts and traces.
    """

    def __init__(self, graph: Mapping[str, Any]):
        self._graph = dict(graph)
        self._nodes = self._index_nodes(self._graph.get("nodes", []))
        self._edges = self._index_edges(self._graph.get(
            "links", self._graph.get("edges", [])))
        self._outgoing: dict[str, list[dict[str, Any]]] = {}
        self._incoming: dict[str, list[dict[str, Any]]] = {}
        for edge in self._edges:
            self._outgoing.setdefault(edge["source"], []).append(edge)
            self._incoming.setdefault(edge["target"], []).append(edge)
        self._hyperedges = list(self._graph.get("hyperedges", []))

    @classmethod
    def from_json(cls, path: str | Path) -> "GraphifyStore":
        """Load a graphify `graph.json` file."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls(json.load(stream))

    @staticmethod
    def _index_nodes(nodes: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if "id" not in node:
                raise GraphifyStoreError("Every graph node must have an 'id'.")
            node_id = str(node["id"])
            indexed[node_id] = {**node, "id": node_id}
        return indexed

    @staticmethod
    def _index_edges(edges: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        indexed = []
        for edge in edges:
            source = edge.get("source", edge.get("from"))
            target = edge.get("target", edge.get("to"))
            if source is None or target is None:
                raise GraphifyStoreError(
                    "Every graph edge must have source and target.")
            indexed.append(
                {**edge, "source": str(source), "target": str(target)})
        return indexed

    @staticmethod
    def _confidence_rank(node_or_edge: Mapping[str, Any]) -> int:
        value = str(node_or_edge.get("confidence", "EXTRACTED")).upper()
        return {"EXTRACTED": 3, "INFERRED": 2, "AMBIGUOUS": 1}.get(value, 0)

    @staticmethod
    def _validate_limit(limit: int, name: str) -> None:
        if limit < 1:
            raise GraphifyStoreError(f"{name} must be greater than zero.")

    def search(self, query: str, *, limit: int = 8, min_confidence: str = "AMBIGUOUS") -> list[dict[str, Any]]:
        """Find relevant nodes using predictable token matching.

        This is intentionally not semantic search: it works offline, exposes
        why a node matched, and can be replaced by a vector retriever later.
        """
        self._validate_limit(limit, "limit")
        required_rank = self._confidence_rank({"confidence": min_confidence})
        tokens = {token for token in query.lower().split() if token}
        ranked = []
        for node in self._nodes.values():
            if self._confidence_rank(node) < required_rank:
                continue
            haystack = " ".join(str(node.get(key, "")) for key in (
                "id", "label", "name", "source_file", "file_type")).lower()
            matched = sorted(token for token in tokens if token in haystack)
            if matched:
                ranked.append(
                    (len(matched), self._confidence_rank(node), node))
        ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2]["id"])))
        return [{**node, "matched_terms": sorted(token for token in tokens if token in " ".join(map(str, node.values())).lower())} for _, _, node in ranked[:limit]]

    def neighbors(self, node_id: str, *, direction: str = "both", limit: int = 20) -> dict[str, Any]:
        """Return a bounded neighborhood and the edges that explain it."""
        self._validate_limit(limit, "limit")
        if node_id not in self._nodes:
            return {"node_id": node_id, "nodes": [], "edges": [], "error": "node_not_found"}
        if direction not in {"in", "out", "both"}:
            raise GraphifyStoreError(
                "direction must be 'in', 'out', or 'both'.")
        selected = []
        node_ids = {node_id}
        if direction in {"out", "both"}:
            selected.extend(self._outgoing.get(node_id, []))
        if direction in {"in", "both"}:
            selected.extend(self._incoming.get(node_id, []))
        for edge in selected:
            node_ids.update((edge["source"], edge["target"]))
        selected = selected[:limit]
        return {"node_id": node_id, "nodes": [self._nodes[key] for key in node_ids if key in self._nodes], "edges": selected}

    def shortest_path(self, source_id: str, target_id: str, *, max_hops: int = 6) -> dict[str, Any]:
        """Find a short directed path, returning an explicit no-path result."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return {"path": [], "length": None, "error": "node_not_found"}
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        while queue:
            current, path = queue.popleft()
            if current == target_id:
                return {"path": path, "length": len(path) - 1}
            if len(path) - 1 >= max_hops:
                continue
            for edge in self._outgoing.get(current, []):
                if edge["target"] not in visited:
                    visited.add(edge["target"])
                    queue.append((edge["target"], [*path, edge["target"]]))
        return {"path": [], "length": None, "error": "no_path"}

    def context(self, query: str, *, limit: int = 5, hops: int = 1, min_confidence: str = "INFERRED") -> dict[str, Any]:
        """Build a compact, traceable context packet for an agent or graph node."""
        self._validate_limit(limit, "limit")
        if hops < 0:
            raise GraphifyStoreError("hops must not be negative.")
        matches = self.search(query, limit=limit,
                              min_confidence=min_confidence)
        evidence = []
        for match in matches:
            discovered = {match["id"]}
            frontier = {match["id"]}
            for _ in range(max(0, hops)):
                next_frontier = {
                    edge["target"]
                    for node_id in frontier
                    for edge in self._outgoing.get(node_id, [])
                    if edge["target"] not in discovered
                }
                discovered.update(next_frontier)
                frontier = next_frontier
            neighborhood = {
                "node_id": match["id"],
                "nodes": [self._nodes[node_id] for node_id in discovered if node_id in self._nodes],
                "edges": [
                    edge for edge in self._edges
                    if edge["source"] in discovered and edge["target"] in discovered
                ][:30],
            }
            evidence.append({"anchor": match, "neighborhood": neighborhood})
        return {"query": query, "matches": matches, "evidence": evidence, "hyperedges": self._hyperedges[:limit]}
