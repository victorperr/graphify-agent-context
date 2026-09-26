import pytest

from graphify_integration import GraphifyStore
from graphify_integration.store import MAX_RESULTS


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
    evidence = store.context("users", hops=2)["evidence"][0]
    assert evidence["anchor_id"] == "users"
    assert {node["id"] for node in evidence["nodes"]} == {"auth", "db"}


def test_invalid_limit_is_reported():
    with pytest.raises(ValueError):
        GraphifyStore(GRAPH).search("auth", limit=0)


def test_context_rejects_negative_hops():
    with pytest.raises(ValueError, match="hops"):
        GraphifyStore(GRAPH).context("auth", hops=-1)


def star_graph(size: int) -> dict:
    return {"nodes": [{"id": "hub"}] + [{"id": f"leaf{i}"} for i in range(size)],
            "links": [{"source": "hub", "target": f"leaf{i}"} for i in range(size)]}


def test_neighbors_returns_only_nodes_of_returned_edges():
    result = GraphifyStore(star_graph(50)).neighbors("hub", limit=2)
    assert len(result["edges"]) == 2
    assert [node["id"] for node in result["nodes"]] == ["hub", "leaf0", "leaf1"]


def test_neighbors_lists_a_self_loop_once():
    store = GraphifyStore({"nodes": [{"id": "a"}], "links": [{"source": "a", "target": "a"}]})
    assert len(store.neighbors("a")["edges"]) == 1


def test_limits_are_capped_and_long_fields_truncated():
    store = GraphifyStore({"nodes": [{"id": f"node{i}", "code": "x" * 10_000} for i in range(100)], "links": []})
    results = store.search("node", limit=1_000)
    assert len(results) == MAX_RESULTS
    assert len(results[0]["code"]) < 600


def test_unknown_min_confidence_is_rejected():
    with pytest.raises(ValueError, match="min_confidence"):
        GraphifyStore(GRAPH).search("auth", min_confidence="EXTRACTD")


def test_context_lists_each_match_once():
    packet = GraphifyStore(GRAPH).context("auth")
    assert [match["id"] for match in packet["matches"]] == ["auth"]
    assert "anchor" not in packet["evidence"][0]


def test_context_traversal_respects_min_confidence():
    graph = {"nodes": [{"id": "api"}, {"id": "db"}, {"id": "cache"}],
             "links": [{"source": "api", "target": "db", "confidence": "EXTRACTED"},
                       {"source": "api", "target": "cache", "confidence": "AMBIGUOUS"}]}
    evidence = GraphifyStore(graph).context("api", min_confidence="INFERRED")["evidence"][0]
    assert [node["id"] for node in evidence["nodes"]] == ["db"]
    assert [edge["target"] for edge in evidence["edges"]] == ["db"]


def test_context_only_includes_hyperedges_touching_the_packet():
    graph = {**GRAPH, "hyperedges": [{"id": "billing", "nodes": ["invoice", "payment"]},
                                     {"id": "login_flow", "nodes": ["users", "auth"]}]}
    packet = GraphifyStore(graph).context("auth")
    assert [hyperedge["id"] for hyperedge in packet["hyperedges"]] == ["login_flow"]


API_GRAPH = {
    "nodes": [
        {"id": "dependency:get_current_user", "label": "get_current_user"},
        {"id": "dependency:get_current_active_superuser", "label": "get_current_active_superuser"},
        {"id": "route:app/api/routes/users.py:read_user_me", "label": "GET /users/me"},
        {"id": "theme_settings", "label": "theme settings"},
        {"id": "whatever_isolated", "label": "unrelated"},
    ],
    "links": [],
}


def test_search_handles_question_punctuation():
    results = GraphifyStore(API_GRAPH).search("Which function decodes get_current_user?")
    assert results[0]["id"] == "dependency:get_current_user"
    assert results[0]["matched_terms"] == ["get_current_user"]


def test_search_ignores_stop_words():
    assert [node["id"] for node in GraphifyStore(API_GRAPH).search("What is the theme?")] == ["theme_settings"]


def test_search_ranks_whole_identifiers_above_parts_and_matches_plurals():
    ids = [node["id"] for node in GraphifyStore(API_GRAPH).search("routes for users")]
    assert ids == ["route:app/api/routes/users.py:read_user_me", "dependency:get_current_user"]


def test_search_ranks_exact_identifier_above_substring():
    # "admin_authentication" sorts first alphabetically, so only scoring can put "auth" on top.
    store = GraphifyStore({"nodes": [{"id": "admin_authentication"}, {"id": "auth"}], "links": []})
    assert [node["id"] for node in store.search("auth")] == ["auth", "admin_authentication"]


def test_matched_terms_explain_the_ranking():
    store = GraphifyStore({"nodes": [{"id": "n1", "label": "auth", "description": "database"}], "links": []})
    assert store.search("auth database")[0]["matched_terms"] == ["auth"]
