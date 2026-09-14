import pytest

from graphify_integration import GraphifyStore


GRAPH = {
    "nodes": [
        {"id": "auth", "label": "authentication service", "confidence": "EXTRACTED"},
        {"id": "users", "label": "users route", "confidence": "INFERRED"},
        {"id": "db", "label": "database", "confidence": "EXTRACTED"},
    ],
    "links": [
        {"source": "users", "target": "auth", "relation": "calls"},
        {"source": "auth", "target": "db", "relation": "uses"},
    ],
}


def test_search_is_confidence_aware_and_deterministic():
    store = GraphifyStore(GRAPH)
    assert [node["id"] for node in store.search("authentication")] == ["auth"]
    assert store.search("users", min_confidence="EXTRACTED") == []


def test_neighbors_and_shortest_path_preserve_evidence():
    store = GraphifyStore(GRAPH)
    assert store.neighbors("auth", direction="out")[
        "edges"][0]["relation"] == "uses"
    assert store.shortest_path("users", "db") == {
        "path": ["users", "auth", "db"], "length": 2}
    assert {node["id"] for node in store.context("users", hops=2)[
        "evidence"][0]["neighborhood"]["nodes"]} == {"users", "auth", "db"}


def test_invalid_limit_is_reported():
    with pytest.raises(ValueError):
        GraphifyStore(GRAPH).search("auth", limit=0)


def test_context_rejects_negative_hops():
    with pytest.raises(ValueError, match="hops"):
        GraphifyStore(GRAPH).context("auth", hops=-1)
