# Graphify integration for LangChain and LangGraph

Give an agent a bounded, explainable view of a Graphify `graph.json` without dumping the entire code graph into its context.

## Install

```bash
pip install -e .
pip install -e ".[langchain]"  # optional adapters
```

## LangChain

```python
from graphify_integration import GraphifyStore
from graphify_integration.langchain import create_graphify_tools

store = GraphifyStore.from_json("graphify-out/graph.json")
tools = create_graphify_tools(store)
# Bind `tools` to any LangChain chat model or agent.
```

The tools are intentionally small: `search_graph`, `inspect_graph_node`, `trace_graph_path`, and `graph_context`. `graph_context` is the useful default for agents: it returns matching anchors, their evidence neighborhoods, confidence metadata, and hyperedges in one bounded packet.

## LangGraph

```python
from graphify_integration.langchain import graphify_context_node

graph.add_node("graphify_context", graphify_context_node(store))
```

Connect that node before the model node. It reads `question`, `input`, or the latest message and writes `graphify_context` to state.

The core store has no LangChain or NetworkX dependency, so it is straightforward to test, deploy in a service, or replace the lexical search with a vector index later without changing the agent contract.
