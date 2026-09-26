"""Compare a file-search agent with a Graphify-context agent on known-answer questions.

Both agents run the same model, system prompt, turn limit, and tool-output cap;
only the tools differ. Grading is deterministic: every expected term must appear
in the final answer (case-insensitive).

    python evals/run_eval.py --repo ../full-stack-fastapi-template/backend \
        --graph ../full-stack-fastapi-template/backend/graphify-out/graph.json --update-readme
"""

from __future__ import annotations

import argparse
import json
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from fastapi_graphify import FastAPIAnalyzer
from graphify_integration import GraphifyStore

ROOT = Path(__file__).resolve().parents[1]

MAX_TURNS = 12
MAX_TOOL_CHARS = 12_000
SKIPPED_PARTS = {"node_modules", "__pycache__", "graphify-out"}
SYSTEM = (
    "You answer questions about a FastAPI codebase using only the provided tools. "
    "Investigate until you are confident, then reply in one or two sentences naming the "
    "exact identifiers (functions, models, paths). Do not guess."
)
Tool = tuple[dict[str, Any], Callable[..., Any]]


def tool(name: str, description: str, properties: dict[str, Any], required: list[str], fn: Callable[..., Any]) -> Tool:
    parameters = {"type": "object", "properties": properties, "required": required}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}, fn


def file_tools(repo: Path) -> list[Tool]:
    """Baseline: the list/grep/read toolset a typical coding agent gets."""
    files = sorted(path for path in repo.rglob("*") if path.is_file() and not any(
        part.startswith(".") or part in SKIPPED_PARTS for part in path.relative_to(repo).parts))
    rel = {path: path.relative_to(repo).as_posix() for path in files}

    def list_files(glob: str = "*") -> list[str]:
        return [rel[path] for path in files if fnmatch(rel[path], glob)][:300]

    def grep(pattern: str, glob: str = "*") -> list[str]:
        regex, hits = re.compile(pattern, re.IGNORECASE), []
        for path in (path for path in files if fnmatch(rel[path], glob)):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            hits += [f"{rel[path]}:{number}: {line.strip()}" for number,
                     line in enumerate(lines, 1) if regex.search(line)]
        return hits[:60]

    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = (repo / path).resolve()
        if not target.is_relative_to(repo):
            raise ValueError("path escapes the repository")
        lines = target.read_text(encoding="utf-8").splitlines()[start_line - 1:end_line]
        return "\n".join(f"{start_line + offset}: {line}" for offset, line in enumerate(lines))

    text = {"type": "string"}
    return [
        tool("list_files", "List repository files matching a glob such as '*.py'.",
             {"glob": text}, [], list_files),
        tool("grep", "Search file lines with a case-insensitive regex; returns path:line: text (max 60).",
             {"pattern": text, "glob": text}, ["pattern"], grep),
        tool("read_file", "Read a line range of a repository file.",
             {"path": text, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"], read_file),
    ]


def graph_tools(repo: Path, graph_path: Path | None) -> list[Tool]:
    """Treatment: the Graphify graph enriched with FastAPI domain nodes, queried through GraphifyStore."""
    analyzer = FastAPIAnalyzer(graph_path=graph_path) if graph_path else FastAPIAnalyzer()
    report = analyzer.analyze(repo)
    store = GraphifyStore({"nodes": report.graph_nodes + report.domain_nodes,
                           "links": report.graph_edges + report.domain_edges})
    text, limit = {"type": "string"}, {"type": "integer"}
    return [
        tool("graph_context", "Return bounded, confidence-aware evidence (matches plus their neighborhoods) for a question.",
             {"query": text, "limit": limit}, ["query"], lambda query, limit=5: store.context(query, limit=limit)),
        tool("search_graph", "Search graph nodes by concept, file, symbol, route path, or model name.",
             {"query": text, "limit": limit}, ["query"], lambda query, limit=8: store.search(query, limit=limit)),
        tool("inspect_graph_node", "List the incoming and outgoing edges of one graph node id.",
             {"node_id": text}, ["node_id"], lambda node_id: store.neighbors(node_id)),
    ]


def run_agent(client: OpenAI, model: str, tools: list[Tool], question: str) -> dict[str, Any]:
    schemas = [schema for schema, _ in tools]
    handlers = {schema["function"]["name"]: fn for schema, fn in tools}
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    tokens = tool_calls = 0
    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(model=model, messages=messages, tools=schemas)
        tokens += response.usage.total_tokens  # prompt tokens already include cached tokens
        choice = response.choices[0]
        message = choice.message
        if not message.tool_calls:
            return {"answer": message.content or "", "stop_reason": "refusal" if message.refusal else choice.finish_reason,
                    "tokens": tokens, "tool_calls": tool_calls}
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            tool_calls += 1
            try:
                output = handlers[call.function.name](**json.loads(call.function.arguments or "{}"))
                output = output if isinstance(output, str) else json.dumps(output, default=str)
            except Exception as exc:  # tool failures go back to the model, as they would in production
                output = f"ERROR {type(exc).__name__}: {exc}"
            if len(output) > MAX_TOOL_CHARS:
                output = output[:MAX_TOOL_CHARS] + "\n[truncated]"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
    return {"answer": "", "stop_reason": "max_turns", "tokens": tokens, "tool_calls": tool_calls}


def grade(answer: str, expected: list[str]) -> bool:
    return all(term.lower() in answer.lower() for term in expected)


def render_table(summary: dict[str, dict[str, float]], model: str, count: int) -> str:
    lines = [f"Evaluated on {count} known-answer questions about "
             f"[full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template) with `{model}`.", "",
             "| Setup | Accuracy | Avg tokens / question | Avg tool calls |", "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {row['accuracy']:.0%} | {row['tokens']:,.0f} | {row['tool_calls']:.1f} |"
              for name, row in summary.items()]
    if {"file_search", "graph_context"} <= summary.keys():
        base, graph = summary["file_search"], summary["graph_context"]
        ratio = base["tokens"] / graph["tokens"] if graph["tokens"] else float("inf")
        cost = f"{ratio:.1f}x fewer tokens" if ratio >= 1 else f"{1 / ratio:.1f}x more tokens"
        delta = (graph["accuracy"] - base["accuracy"]) * 100
        lines = [f"**graph_context vs file search: {delta:+.0f} pts accuracy with {cost}.**", "", *lines]
    return "\n".join(lines)


def update_readme(table: str) -> None:
    readme = ROOT / "README.md"
    content, replaced = re.subn(r"(<!-- eval:start -->\n).*?(\n<!-- eval:end -->)",
                                lambda match: match.group(1) + table + match.group(2),
                                readme.read_text(encoding="utf-8"), flags=re.DOTALL)
    if not replaced:
        raise SystemExit("README.md has no <!-- eval:start --> / <!-- eval:end --> markers.")
    readme.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True, help="Backend directory of the evaluated FastAPI project")
    parser.add_argument("--graph", type=Path, help="Graphify graph.json for the same project")
    parser.add_argument("--questions", type=Path, default=ROOT / "evals" / "questions.jsonl")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--setups", nargs="+", choices=("file_search", "graph_context"),
                        default=["file_search", "graph_context"])
    parser.add_argument("--limit", type=int, help="Run only the first N questions")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    lines = args.questions.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()][:args.limit]
    builders = {"file_search": lambda: file_tools(repo), "graph_context": lambda: graph_tools(repo, args.graph)}
    client = OpenAI()
    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {}
    for setup in args.setups:
        tools, rows = builders[setup](), []
        with (results_dir / f"{setup}.jsonl").open("w", encoding="utf-8") as out:
            for question in questions:
                started = time.perf_counter()
                row = {"id": question["id"], **run_agent(client, args.model, tools, question["question"])}
                row.update(correct=grade(row["answer"], question["expected"]),
                           seconds=round(time.perf_counter() - started, 1))
                out.write(json.dumps(row) + "\n")
                rows.append(row)
                print(f"[{setup}] {row['id']} {'PASS' if row['correct'] else 'FAIL'} "
                      f"tokens={row['tokens']} tool_calls={row['tool_calls']}")
        summary[setup] = {key: sum(float(row[key]) for row in rows) / len(rows)
                          for key in ("tokens", "tool_calls")} | {"accuracy": sum(row["correct"] for row in rows) / len(rows)}

    table = render_table(summary, args.model, len(questions))
    print("\n" + table)
    if args.update_readme:
        update_readme(table)


if __name__ == "__main__":
    main()
