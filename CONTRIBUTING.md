# Contributing

## Local setup

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Keep graph query results bounded and deterministic. New agent-facing behavior should preserve source metadata and include focused tests for missing nodes, invalid limits, and confidence filtering where relevant.

Please open an issue before a large feature so the public API and Graphify compatibility can be discussed first.