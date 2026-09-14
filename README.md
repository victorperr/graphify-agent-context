# Graphify Agent Context

Graphify Agent Context is a dependency-light adapter that turns a Graphify code graph into bounded, explainable evidence for AI agents.

Instead of putting an entire repository into an LLM context window, it exposes deterministic tools for:

- lexical node search with confidence filtering
- bounded neighborhood inspection
- directed shortest-path tracing
- compact multi-hop context packets with evidence metadata
- optional FastAPI route, dependency, Pydantic model, and middleware extraction

The graph remains the source of truth. The LLM chooses when to ask a question; this package performs the graph traversal and keeps the result small enough to inspect and cite.

## Why this is useful

Code agents often have retrieval, but not reliable structural retrieval. Graphify provides relationships; this project adds a stable tool contract around those relationships. The result is useful for architecture Q&A, onboarding, impact analysis, API documentation, and agent traces where a response should distinguish extracted facts from inferred ones.

This is an integration layer, not a replacement for Graphify, LangChain, LangGraph, or a vector database. Lexical search is deliberately deterministic and can later be replaced behind the same store contract.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[langchain]"
```

The core store does not require LangChain, NetworkX, FastAPI, or an LLM.

## LangChain and LangGraph

```python
from graphify_integration import GraphifyStore
from graphify_integration.langchain import create_graphify_tools

store = GraphifyStore.from_json("graphify-out/graph.json")
tools = create_graphify_tools(store)
```

The adapters expose `search_graph`, `inspect_graph_node`, `trace_graph_path`, and `graph_context`. `graph_context` is the recommended default for an agent because it returns bounded evidence rather than an unstructured graph dump.

For LangGraph, add `graphify_context_node(store)` before the model node. It reads `question`, `input`, or the latest message and writes `graphify_context` to state.

## FastAPI analysis

```bash
fastapi-graphify ./my-api --output FASTAPI_REPORT.md
fastapi-graphify ./my-api --graph graphify-out/graph.json --format json --output fastapi-report.json
```

The AST pass is best effort and skips malformed or unreadable Python files. It has no runtime dependency on FastAPI. Graph metadata is used when source analysis is incomplete.

## Development

```bash
pytest -q
```

The tests cover deterministic ranking, confidence filters, bounded traversal, explicit error results, FastAPI extraction, and domain graph relationships.

## Project status

This is an alpha portfolio project. The most valuable next steps are benchmark fixtures from real Graphify outputs, richer Python symbol resolution, and an MCP server exposing the same bounded query contract.

## License

MIT. See [LICENSE](LICENSE).
