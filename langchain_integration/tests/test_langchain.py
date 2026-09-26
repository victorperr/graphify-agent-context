import json
from types import SimpleNamespace

import pytest

from graphify_integration import GraphifyStore
from graphify_integration.langchain import create_graphify_tools, graphify_context_node

STORE = GraphifyStore({"nodes": [{"id": "auth", "label": "authentication"}], "links": []})


@pytest.mark.parametrize("message", [
    SimpleNamespace(content="authentication"),  # LangChain message objects expose .content
    {"role": "user", "content": "authentication"},
    ("user", "authentication"),
    SimpleNamespace(content=[{"type": "text", "text": "authentication"}]),
])
def test_context_node_reads_every_message_shape(message):
    context = graphify_context_node(STORE)({"messages": [message]})["graphify_context"]
    assert [match["id"] for match in context["matches"]] == ["auth"]


def test_context_node_handles_missing_messages():
    context = graphify_context_node(STORE)({"messages": []})["graphify_context"]
    assert context["error"] == "empty_query"


def test_tools_return_bounded_json():
    pytest.importorskip("langchain_core")
    tools = {tool.name: tool for tool in create_graphify_tools(STORE)}
    assert set(tools) == {"search_graph", "inspect_graph_node", "trace_graph_path", "graph_context"}
    assert json.loads(tools["search_graph"].invoke({"query": "auth", "limit": 10_000}))[0]["id"] == "auth"
