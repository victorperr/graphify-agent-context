"""FastAPI-specific post-processing for Graphify node-link graphs.

The analyzer is deliberately independent from FastAPI, NetworkX, and an LLM.
Graphify supplies cross-file structure; Python AST inspection supplies details
that are syntax-level facts, such as route decorators and function parameters.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


HTTP_METHODS = {"get", "post", "put", "patch",
                "delete", "options", "head", "trace", "websocket"}
STATUS_CODE_RE = re.compile(r"\b([1-5][0-9]{2})\b")
PATH_PARAM_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)[^}]*\}")


@dataclass(frozen=True)
class Route:
    path: str
    method: str
    handler: str
    file: str
    line: int
    parameters: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    response_models: list[str] = field(default_factory=list)
    status_codes: list[int] = field(default_factory=list)
    router: str = ""


@dataclass(frozen=True)
class Model:
    name: str
    file: str
    line: int
    bases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Dependency:
    name: str
    file: str
    line: int
    dependencies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Middleware:
    name: str
    file: str
    line: int
    kind: str


@dataclass
class FastAPIReport:
    """Serializable FastAPI architecture facts and graph-derived insights."""

    routes: list[Route] = field(default_factory=list)
    models: list[Model] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    middleware: list[Middleware] = field(default_factory=list)
    graph_nodes: list[dict[str, Any]] = field(default_factory=list)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)
    domain_nodes: list[dict[str, Any]] = field(default_factory=list)
    domain_edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# FastAPI Architecture Report", ""]
        lines += [f"## Routes ({len(self.routes)})", "",
                  "| Method | Path | Handler | Dependencies | Source |", "| --- | --- | --- | --- | --- |"]
        for route in self.routes:
            deps = ", ".join(route.dependencies) or "-"
            lines.append(
                f"| {route.method} | `{route.path}` | `{route.handler}` | {deps} | `{route.file}:{route.line}` |")
        lines += ["", f"## Pydantic Models ({len(self.models)})", ""]
        for model in self.models:
            lines.append(f"- `{model.name}` in `{model.file}:{model.line}`")
        lines += ["", f"## Dependencies ({len(self.dependencies)})", ""]
        for dependency in self.dependencies:
            nested = ", ".join(dependency.dependencies) or "-"
            lines.append(f"- `{dependency.name}` calls: {nested}")
        lines += ["", f"## Middleware ({len(self.middleware)})", ""]
        for item in self.middleware:
            lines.append(
                f"- `{item.kind}` `{item.name}` in `{item.file}:{item.line}`")
        lines += ["", "## Graphify Coverage", "", f"- Graph nodes: {len(self.graph_nodes)}", f"- Graph edges: {len(self.graph_edges)}",
                  f"- FastAPI domain nodes: {len(self.domain_nodes)}", f"- FastAPI domain edges: {len(self.domain_edges)}", ""]
        return "\n".join(lines)


class FastAPIAnalyzer:
    """Enrich an existing Graphify graph with FastAPI-specific facts."""

    def __init__(self, graph: Mapping[str, Any] | None = None, *, graph_path: str | Path | None = None):
        if graph is not None and graph_path is not None:
            raise ValueError("Pass graph or graph_path, not both.")
        if graph_path is not None:
            with Path(graph_path).open(encoding="utf-8") as stream:
                graph = json.load(stream)
        self.graph = dict(graph or {})
        self._nodes = {str(node["id"]): dict(node)
                       for node in self.graph.get("nodes", []) if "id" in node}
        self._edges = list(self.graph.get(
            "links", self.graph.get("edges", [])))

    def analyze(self, source_root: str | Path | None = None) -> FastAPIReport:
        """Analyze Python files and attach Graphify data to one report."""
        report = FastAPIReport(graph_nodes=list(
            self._nodes.values()), graph_edges=self._edges.copy())
        if source_root is None:
            self._extract_from_graph(report)
            self._build_domain_graph(report)
            return report
        root = Path(source_root)
        for path in sorted(root.rglob("*.py")):
            if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            self._extract_file(path, root, report)
        self._extract_from_graph(report)
        self._build_domain_graph(report)
        return report

    def write_report(self, report: FastAPIReport, output_path: str | Path, *, format: str = "markdown") -> None:
        """Write a report as Markdown or JSON."""
        if format not in {"markdown", "json"}:
            raise ValueError("format must be 'markdown' or 'json'.")
        output = Path(output_path)
        output.write_text(report.to_markdown() if format ==
                          "markdown" else report.to_json(), encoding="utf-8")

    def _extract_file(self, path: Path, root: Path, report: FastAPIReport) -> None:
        try:
            tree = ast.parse(path.read_text(
                encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            return
        relative = path.relative_to(root).as_posix()
        visitor = _FastAPIVisitor(relative)
        visitor.visit(tree)
        report.routes.extend(visitor.routes)
        report.models.extend(visitor.models)
        report.dependencies.extend(visitor.dependencies)
        report.middleware.extend(visitor.middleware)

    def _extract_from_graph(self, report: FastAPIReport) -> None:
        """Use Graphify metadata when source AST input is unavailable or incomplete."""
        known_routes = {(route.file, route.handler) for route in report.routes}
        for node_id, node in self._nodes.items():
            text = _node_text(node)
            route_match = re.search(
                r"@(app|router)\.(\w+)\(\s*['\"]([^'\"]+)", text)
            if not route_match:
                continue
            app, method, path = route_match.groups()
            handler = str(node.get("name", node.get("label", node_id)))
            source = str(node.get("source_file", node.get("file", "graph")))
            if (source, handler) in known_routes:
                continue
            report.routes.append(Route(path, method.upper(), handler, source, int(
                node.get("line", 0)), sorted(set(PATH_PARAM_RE.findall(path))), router=app))

    @staticmethod
    def _build_domain_graph(report: FastAPIReport) -> None:
        """Materialize typed FastAPI nodes and relationships for downstream tools."""
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        def add_node(node_id: str, node_type: str, label: str, **metadata: Any) -> None:
            nodes.setdefault(
                node_id, {"id": node_id, "type": node_type, "label": label, **metadata})

        for route in report.routes:
            route_id = f"route:{route.file}:{route.handler}"
            add_node(route_id, "fastapi_route", f"{route.method} {route.path}", method=route.method,
                     path=route.path, handler=route.handler, source_file=route.file, line=route.line)
            for model in route.response_models:
                model_name = model.split(".")[-1]
                model_id = f"model:{model_name}"
                add_node(model_id, "pydantic_model", model_name)
                edges.append(
                    {"source": route_id, "target": model_id, "relation": "returns_model"})
            for dependency in route.dependencies:
                dependency_name = _dependency_name(dependency)
                dependency_id = f"dependency:{dependency_name}"
                add_node(dependency_id, "fastapi_dependency", dependency_name)
                edges.append(
                    {"source": route_id, "target": dependency_id, "relation": "depends_on"})
        for model in report.models:
            add_node(f"model:{model.name}", "pydantic_model",
                     model.name, source_file=model.file, line=model.line)
        for dependency in report.dependencies:
            dependency_id = f"dependency:{dependency.name}"
            add_node(dependency_id, "fastapi_dependency", dependency.name,
                     source_file=dependency.file, line=dependency.line)
            for nested in dependency.dependencies:
                nested_name = _dependency_name(nested)
                nested_id = f"dependency:{nested_name}"
                add_node(nested_id, "fastapi_dependency", nested_name)
                edges.append({"source": dependency_id,
                             "target": nested_id, "relation": "calls"})
        for item in report.middleware:
            add_node(f"middleware:{item.name}", "fastapi_middleware",
                     item.name, kind=item.kind, source_file=item.file, line=item.line)
        report.domain_nodes = sorted(
            nodes.values(), key=lambda node: node["id"])
        report.domain_edges = sorted(edges, key=lambda edge: (
            edge["source"], edge["target"], edge["relation"]))


class _FastAPIVisitor(ast.NodeVisitor):
    def __init__(self, file: str):
        self.file = file
        self.routes: list[Route] = []
        self.models: list[Model] = []
        self.dependencies: list[Dependency] = []
        self.middleware: list[Middleware] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [_expression(base) for base in node.bases]
        if any(base.rsplit(".", 1)[-1] == "BaseModel" for base in bases):
            self.models.append(Model(node.name, self.file, node.lineno, bases))
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_route = False
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            target = call.func if call else decorator
            if not isinstance(target, ast.Attribute) or target.attr.lower() not in HTTP_METHODS:
                if isinstance(target, ast.Attribute) and target.attr == "middleware":
                    kind = _call_argument(call, 0) or _keyword_argument(
                        call, "type") or "http"
                    self.middleware.append(Middleware(
                        node.name, self.file, node.lineno, kind))
                continue
            is_route = True
            method = target.attr.upper()
            path = _call_argument(call, 0) or "/"
            dependencies = [_expression(default) for default in node.args.defaults +
                            node.args.kw_defaults if default and "Depends" in _expression(default)]
            response_models = [_expression(keyword.value) for keyword in (
                call.keywords if call else []) if keyword.arg == "response_model"]
            decorator_text = _expression(decorator)
            status_codes = sorted(
                {int(code) for code in STATUS_CODE_RE.findall(decorator_text)})
            self.routes.append(Route(path, method, node.name, self.file, node.lineno, sorted(set(
                PATH_PARAM_RE.findall(path))), dependencies, response_models, status_codes, _expression(target.value)))
        if not is_route and any("Depends(" in _expression(default) for default in node.args.defaults + node.args.kw_defaults if default):
            dependencies = [_expression(default) for default in node.args.defaults +
                            node.args.kw_defaults if default and "Depends" in _expression(default)]
            self.dependencies.append(Dependency(
                node.name, self.file, node.lineno, dependencies))


def _expression(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    return _legacy_expression(node)


def _legacy_expression(node: ast.AST) -> str:
    """Render the AST forms needed by the extractor on Python 3.8."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_legacy_expression(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Call):
        arguments = [_legacy_expression(argument) for argument in node.args]
        arguments.extend(
            f"{keyword.arg}={_legacy_expression(keyword.value)}"
            for keyword in node.keywords
        )
        return f"{_legacy_expression(node.func)}({', '.join(arguments)})"
    if isinstance(node, ast.Dict):
        pairs = zip(node.keys, node.values)
        return "{" + ", ".join(
            f"{_legacy_expression(key)}: {_legacy_expression(value)}"
            for key, value in pairs if key is not None
        ) + "}"
    return ast.dump(node)


def _call_argument(call: ast.Call | None, index: int) -> str:
    if call and len(call.args) > index and isinstance(call.args[index], ast.Constant) and isinstance(call.args[index].value, str):
        return call.args[index].value
    return ""


def _keyword_argument(call: ast.Call | None, name: str) -> str:
    if call:
        for keyword in call.keywords:
            if keyword.arg == name:
                return _expression(keyword.value).strip("'\"")
    return ""


def _dependency_name(value: str) -> str:
    prefix = "Depends("
    return value[len(prefix):].rstrip(")") if value.startswith(prefix) else value


def _node_text(node: Mapping[str, Any]) -> str:
    return " ".join(str(node.get(key, "")) for key in ("id", "label", "name", "source", "source_file", "code"))
