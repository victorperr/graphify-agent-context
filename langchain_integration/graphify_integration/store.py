"""Deterministic, dependency-light access to Graphify's node-link output."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

CONFIDENCE_RANKS = {"EXTRACTED": 3, "INFERRED": 2, "AMBIGUOUS": 1}
MAX_RESULTS = 25
MAX_FIELD_CHARS = 500
# snake_case identifiers stay whole, so "get_current_user?" is one token.
WORD_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can do does did for from has have how in is it its of on or "
    "that the their them there these this those to was were what when where which who whom "
    "whose why with".split())


def _query_tokens(query: str) -> set[str]:
    """Identifier-like words from a natural-language query, without stop words or 1-2 letter noise."""
    return {token for token in WORD_RE.findall(query.lower()) if len(token) >= 3 and token not in STOPWORDS}


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    return token[:-1] if token.endswith("s") and not token.endswith("ss") and len(token) > 4 else token


def _match_score(token: str, haystack: str, words: set[str], parts: set[str]) -> int:
    """3 = whole identifier, 2 = one snake_case part (or its plural), 1 = substring of 4+ characters."""
    singular = _singular(token)
    if token in words or singular in words:
        return 3
    if token in parts or singular in parts:
        return 2
    return 1 if len(token) >= 4 and token in haystack else 0


class GraphifyStoreError(ValueError):
    """Raised when a Graphify document cannot be queried safely."""


class GraphifyStore:
    """Query a Graphify graph without making the LLM responsible for graph logic.

    Every result is bounded: `limit` values are capped at `MAX_RESULTS`, string
    fields longer than `MAX_FIELD_CHARS` are truncated, and context packets list
    each node once. Records without a `confidence` label are treated as
    EXTRACTED, because Graphify labels edges but usually not nodes.
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
        return CONFIDENCE_RANKS.get(value, 0)

    @staticmethod
    def _required_rank(min_confidence: str) -> int:
        rank = CONFIDENCE_RANKS.get(str(min_confidence).upper())
        if rank is None:
            raise GraphifyStoreError(
                f"min_confidence must be one of {', '.join(CONFIDENCE_RANKS)}.")
        return rank

    @staticmethod
    def _bounded_limit(limit: int, name: str) -> int:
        if limit < 1:
            raise GraphifyStoreError(f"{name} must be greater than zero.")
        return min(limit, MAX_RESULTS)

    @staticmethod
    def _compact(record: Mapping[str, Any]) -> dict[str, Any]:
        """Truncate long string fields such as source code so one node cannot flood a prompt."""
        return {key: value[:MAX_FIELD_CHARS] + "...[truncated]" if isinstance(value, str) and len(value) > MAX_FIELD_CHARS
                else value for key, value in record.items()}

    def search(self, query: str, *, limit: int = 8, min_confidence: str = "AMBIGUOUS") -> list[dict[str, Any]]:
        """Find relevant nodes using predictable token matching.

        This is intentionally not semantic search: it works offline, exposes
        why a node matched, and can be replaced by a vector retriever later.
        """
        limit = self._bounded_limit(limit, "limit")
        required_rank = self._required_rank(min_confidence)
        tokens = _query_tokens(query)
        ranked = []
        for node in self._nodes.values():
            if self._confidence_rank(node) < required_rank:
                continue
            haystack = " ".join(str(node.get(key, "")) for key in (
                "id", "label", "name", "source_file", "file_type")).lower()
            words = set(WORD_RE.findall(haystack))
            parts = {part for word in words for part in word.split("_")}
            scores = {token: _match_score(token, haystack, words, parts) for token in tokens}
            matched = sorted(token for token, score in scores.items() if score)
            if matched:
                ranked.append((sum(scores.values()), matched, self._confidence_rank(node), node))
        ranked.sort(key=lambda item: (-item[0], -len(item[1]), -item[2], item[3]["id"]))
        return [{**self._compact(node), "matched_terms": matched} for _, matched, _, node in ranked[:limit]]

    def neighbors(self, node_id: str, *, direction: str = "both", limit: int = 20) -> dict[str, Any]:
        """Return a bounded neighborhood and the edges that explain it."""
        limit = self._bounded_limit(limit, "limit")
        if direction not in {"in", "out", "both"}:
            raise GraphifyStoreError(
                "direction must be 'in', 'out', or 'both'.")
        if node_id not in self._nodes:
            return {"node_id": node_id, "nodes": [], "edges": [], "error": "node_not_found"}
        selected = []
        if direction in {"out", "both"}:
            selected.extend(self._outgoing.get(node_id, []))
        if direction in {"in", "both"}:
            selected.extend(edge for edge in self._incoming.get(node_id, [])
                            if not (direction == "both" and edge["source"] == node_id))
        selected = selected[:limit]
        node_ids = [node_id, *dict.fromkeys(
            other for edge in selected for other in (edge["source"], edge["target"]) if other != node_id)]
        return {"node_id": node_id,
                "nodes": [self._compact(self._nodes[key]) for key in node_ids if key in self._nodes],
                "edges": [self._compact(edge) for edge in selected]}

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
        """Build a compact, traceable context packet for an agent or graph node.

        Traversal follows only edges and nodes that meet `min_confidence`.
        Matched nodes appear once, in `matches`; each evidence entry refers to
        its anchor by `anchor_id`. Hyperedges are included only when they
        touch a node in the packet.
        """
        limit = self._bounded_limit(limit, "limit")
        required_rank = self._required_rank(min_confidence)
        if hops < 0:
            raise GraphifyStoreError("hops must not be negative.")
        matches = self.search(query, limit=limit,
                              min_confidence=min_confidence)
        evidence = []
        packet_node_ids: set[str] = set()
        for match in matches:
            discovered = {match["id"]}
            frontier = {match["id"]}
            edges: list[dict[str, Any]] = []
            for _ in range(hops):
                next_frontier = set()
                for node_id in sorted(frontier):
                    for edge in self._outgoing.get(node_id, []):
                        target = self._nodes.get(edge["target"])
                        if (target is None or self._confidence_rank(edge) < required_rank
                                or self._confidence_rank(target) < required_rank):
                            continue
                        edges.append(edge)
                        if edge["target"] not in discovered:
                            next_frontier.add(edge["target"])
                discovered.update(next_frontier)
                frontier = next_frontier
            packet_node_ids.update(discovered)
            evidence.append({
                "anchor_id": match["id"],
                "nodes": [self._compact(self._nodes[node_id]) for node_id in sorted(discovered - {match["id"]})],
                "edges": [self._compact(edge) for edge in edges[:30]],
            })
        hyperedges = [
            self._compact(hyperedge) for hyperedge in self._hyperedges
            if self._confidence_rank(hyperedge) >= required_rank
            and packet_node_ids.intersection(map(str, hyperedge.get("nodes", hyperedge.get("members", []))))
        ]
        return {"query": query, "matches": matches, "evidence": evidence, "hyperedges": hyperedges[:limit]}
