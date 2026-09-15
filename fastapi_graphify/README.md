# FastAPI Graphify

FastAPI Graphify is a small post-processing layer for an existing Graphify `graph.json`. It combines Graphify's cross-file structure with Python AST facts that are especially valuable to API agents:

- route method, path, handler, path parameters, response models, and status codes
- Pydantic models and source locations
- `Depends(...)` declarations and nested dependency relationships
- middleware declarations
- typed FastAPI domain nodes and edges for downstream LangChain or LangGraph tools

It has no runtime dependency on FastAPI, NetworkX, or an LLM.

## CLI

```bash
fastapi-graphify ./my-api --graph graphify-out/graph.json --output FASTAPI_REPORT.md
fastapi-graphify ./my-api --graph graphify-out/graph.json --format json --output fastapi-report.json
```

## Python

```python
from fastapi_graphify import FastAPIAnalyzer

analyzer = FastAPIAnalyzer(graph_path="graphify-out/graph.json")
report = analyzer.analyze(".")

print(report.to_json())
print(report.domain_nodes)
print(report.domain_edges)
```

The AST pass is best-effort: malformed or unreadable Python files are skipped, while Graphify metadata remains available. This makes it suitable for CI reporting and for feeding bounded architecture evidence to an agent.