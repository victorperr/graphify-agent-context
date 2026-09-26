"""Optional LangChain and LangGraph adapters."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .store import GraphifyStore


def create_graphify_tools(store: GraphifyStore) -> list[Any]:
    """Create LangChain tools; LangChain remains an optional dependency."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "Install the optional 'langchain' dependency to create tools.") from exc

    def search_graph(query: str, limit: int = 8) -> str:
        """Search Graphify nodes by concept, file, symbol, or technology."""
        return json.dumps(store.search(query, limit=limit), default=str)

    def inspect_graph_node(node_id: str, direction: str = "both") -> str:
        """Inspect the calls, imports, and dependencies around a Graphify node."""
        return json.dumps(store.neighbors(node_id, direction=direction), default=str)

    def trace_graph_path(source_id: str, target_id: str, max_hops: int = 6) -> str:
        """Trace a directed relationship between two Graphify node IDs."""
        return json.dumps(store.shortest_path(source_id, target_id, max_hops=max_hops), default=str)

    def graph_context(query: str, limit: int = 5) -> str:
        """Return bounded, confidence-aware evidence for answering a question."""
        return json.dumps(store.context(query, limit=limit), default=str)

    return [
        StructuredTool.from_function(search_graph),
        StructuredTool.from_function(inspect_graph_node),
        StructuredTool.from_function(trace_graph_path),
        StructuredTool.from_function(graph_context),
    ]


def _message_text(message: Any) -> str:
    """Read text from a LangChain message object, a dict, or a (role, content) tuple."""
    if isinstance(message, Mapping):
        content = message.get("content", "")
    elif isinstance(message, tuple) and len(message) == 2:
        content = message[1]
    else:
        content = getattr(message, "content", message)
    if isinstance(content, list):  # multimodal content blocks
        return " ".join(block if isinstance(block, str) else str(block.get("text", ""))
                        for block in content if isinstance(block, (str, Mapping)))
    return str(content or "")


def graphify_context_node(store: GraphifyStore):
    """Return a LangGraph-compatible node: state in, evidence in state out."""
    def node(state: Mapping[str, Any]) -> dict[str, Any]:
        messages = state.get("messages") or []
        query = state.get("question") or state.get("input") or (
            _message_text(messages[-1]) if messages else "")
        if not str(query).strip():
            return {"graphify_context": {"query": "", "matches": [], "evidence": [], "hyperedges": [], "error": "empty_query"}}
        return {"graphify_context": store.context(str(query))}

    return node
