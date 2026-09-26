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
DEPENDENCY_CALLS = {"Depends", "Security"}
MODEL_BASES = {"BaseModel", "SQLModel"}
# Return annotations FastAPI accepts but that carry no model worth a graph node.
NON_MODEL_RETURNS = {"Any", "None", "bool", "int",
                     "float", "str", "bytes", "dict", "list", "Response"}
HTTP_STATUS_NAME_RE = re.compile(r"HTTP_([1-5][0-9]{2})_")
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
    table: bool = False


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
        modules = []
        for path in sorted(root.rglob("*.py")):
            if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            tree = self._parse(path)
            if tree is not None:
                modules.append((path.relative_to(root).as_posix(), tree))
        # Aliases such as `CurrentUser = Annotated[User, Depends(get_current_user)]` are
        # usually defined in one module and used in others, so we need to collect them from all modules first.
        aliases = _dependency_aliases(tree for _, tree in modules)
        classes: list[Model] = []
        for relative, tree in modules:
            visitor = _FastAPIVisitor(relative, aliases)
            visitor.visit(tree)
            report.routes.extend(visitor.routes)
            report.dependencies.extend(visitor.dependencies)
            report.middleware.extend(visitor.middleware)
            classes.extend(visitor.classes)
        report.models.extend(_model_classes(classes))
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

    @staticmethod
    def _parse(path: Path) -> ast.Module | None:
        try:
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return None

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
            # Merge, so a model first seen as a route's response model still gets its source location.
            nodes.setdefault(
                node_id, {"id": node_id, "type": node_type, "label": label}).update(metadata)

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
                     model.name, source_file=model.file, line=model.line, table=model.table)
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
    def __init__(self, file: str, aliases: Mapping[str, list[str]] | None = None):
        self.file = file
        self.aliases = aliases or {}
        self.routers: dict[str, tuple[str, list[str]]] = {}
        self.routes: list[Route] = []
        self.classes: list[Model] = []
        self.dependencies: list[Dependency] = []
        self.middleware: list[Middleware] = []

    def visit_Module(self, node: ast.Module) -> None:
        # Record `router = APIRouter(prefix=..., dependencies=...)` before visiting the routes that use it.
        for statement in node.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) \
                    and _name(statement.value.func) in {"APIRouter", "FastAPI"}:
                prefix = _keyword_value(statement.value, "prefix")
                router = (prefix.value if isinstance(prefix, ast.Constant) and isinstance(prefix.value, str) else "",
                          _dependency_calls(_keyword_value(statement.value, "dependencies")))
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        self.routers[target.id] = router
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        table = any(keyword.arg == "table" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                    for keyword in node.keywords)
        self.classes.append(Model(node.name, self.file, node.lineno,
                                  [_expression(base) for base in node.bases], table))
        self.generic_visit(node)

    def _parameter_dependencies(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Dependencies from `x=Depends(f)` defaults, inline `Annotated[..., Depends(f)]`, and aliases of it."""
        arguments = node.args
        dependencies = [dependency for default in arguments.defaults + arguments.kw_defaults
                        for dependency in _dependency_calls(default)]
        for argument in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
            dependencies += _dependency_calls(argument.annotation)
            if isinstance(argument.annotation, (ast.Name, ast.Attribute)):
                dependencies += self.aliases.get(
                    _name(argument.annotation), [])
        return list(dict.fromkeys(dependencies))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_route = False
        parameter_dependencies = self._parameter_dependencies(node)
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
            router = _expression(target.value)
            prefix, router_dependencies = self.routers.get(router, ("", []))
            path = (prefix + _call_argument(call, 0)) or "/"
            dependencies = list(dict.fromkeys(
                router_dependencies + _dependency_calls(_keyword_value(call, "dependencies")) + parameter_dependencies))
            response_models = [_expression(keyword.value) for keyword in (
                call.keywords if call else []) if keyword.arg == "response_model"]
            if not response_models and node.returns is not None and _name(node.returns) not in NON_MODEL_RETURNS:
                # FastAPI uses the return annotation as the response model when response_model is absent.
                response_models = [_expression(node.returns)]
            self.routes.append(Route(path, method, node.name, self.file, node.lineno, sorted(set(
                PATH_PARAM_RE.findall(path))), dependencies, response_models, _status_codes(call), router))
        if not is_route and parameter_dependencies:
            self.dependencies.append(Dependency(
                node.name, self.file, node.lineno, parameter_dependencies))


def _name(node: ast.AST) -> str:
    """Last dotted component of an expression: `status.HTTP_200_OK` -> `HTTP_200_OK`, `list[X]` -> `list`."""
    if isinstance(node, ast.Subscript):
        node = node.value
    return _expression(node).rsplit(".", 1)[-1]


def _keyword_value(call: ast.Call | None, name: str) -> ast.AST | None:
    return next((keyword.value for keyword in (call.keywords if call else []) if keyword.arg == name), None)


def _annotated_type(node: ast.AST | None) -> str:
    """The type in `Annotated[Type, ...]`, which `Depends()` without arguments resolves to."""
    if isinstance(node, ast.Subscript) and _name(node.value) == "Annotated" \
            and isinstance(node.slice, ast.Tuple) and node.slice.elts:
        return _expression(node.slice.elts[0])
    return ""


def _dependency_calls(node: ast.AST | None) -> list[str]:
    """`Depends(...)` and `Security(...)` calls inside an expression, normalized to `Depends(target)`."""
    if node is None:
        return []
    implied = _annotated_type(node)
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _name(child.func) in DEPENDENCY_CALLS:
            target = child.args[0] if child.args else _keyword_value(
                child, "dependency")
            name = _expression(target) if target is not None else implied
            if name:
                found.append(f"Depends({name})")
    return found


def _dependency_aliases(trees: Iterable[ast.Module]) -> dict[str, list[str]]:
    """Module-level `Alias = Annotated[T, Depends(f)]` (also `Alias: TypeAlias = ...` and `type Alias = ...`)."""
    aliases: dict[str, list[str]] = {}
    for tree in trees:
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name, value = statement.targets[0].id, statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                name, value = statement.target.id, statement.value
            elif type(statement).__name__ == "TypeAlias":  # Python 3.12 `type Alias = ...`
                name, value = statement.name.id, statement.value
            else:
                continue
            if _annotated_type(value) and (dependencies := _dependency_calls(value)):
                aliases[name] = dependencies
    return aliases


def _model_classes(classes: list[Model]) -> list[Model]:
    """Classes deriving from BaseModel or SQLModel, directly or through other project classes."""
    known = set(MODEL_BASES)
    changed = True
    while changed:
        changed = False
        for model in classes:
            if model.name not in known and any(base.rsplit(".", 1)[-1] in known for base in model.bases):
                known.add(model.name)
                changed = True
    return [model for model in classes if model.name in known]


def _status_code(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and str(node.value).isdigit():
        return int(node.value)
    match = HTTP_STATUS_NAME_RE.search(_expression(node))
    return int(match.group(1)) if match else None


def _status_codes(call: ast.Call | None) -> list[int]:
    """Codes from `status_code=` and the keys of `responses=`, read from the AST rather than the decorator text."""
    codes = []
    status_code = _keyword_value(call, "status_code")
    if status_code is not None:
        codes.append(_status_code(status_code))
    responses = _keyword_value(call, "responses")
    if isinstance(responses, ast.Dict):
        codes += [_status_code(key)
                  for key in responses.keys if key is not None]
    return sorted({code for code in codes if code is not None})


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
