from pathlib import Path

from fastapi_graphify import FastAPIAnalyzer


def test_analyzer_extracts_routes_models_dependencies_and_middleware(tmp_path: Path):
    source = tmp_path / "app.py"
    source.write_text(
        """from fastapi import Depends, FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass User(BaseModel):\n    name: str\ndef auth(): ...\n@app.middleware(\"http\")\nasync def audit(request, call_next): ...\n@app.get(\"/users/{user_id}\", response_model=User, responses={404: {}})\nasync def read_user(user_id: str, current=Depends(auth)): ...\n""",
        encoding="utf-8",
    )
    report = FastAPIAnalyzer(
        {"nodes": [{"id": "app"}], "links": []}).analyze(tmp_path)

    assert report.routes[0].path == "/users/{user_id}"
    assert report.routes[0].parameters == ["user_id"]
    assert report.routes[0].dependencies == ["Depends(auth)"]
    assert report.routes[0].response_models == ["User"]
    assert report.routes[0].status_codes == [404]
    assert report.models[0].name == "User"
    assert report.middleware[0].kind == "http"
    assert {node["type"] for node in report.domain_nodes} == {
        "fastapi_route", "pydantic_model", "fastapi_dependency", "fastapi_middleware"}
    assert {edge["relation"] for edge in report.domain_edges} == {
        "depends_on", "returns_model"}


def test_graph_metadata_can_supply_routes_without_source():
    graph = {
        "nodes": [{"id": "read", "label": "@router.post('/items') read"}], "links": []}
    report = FastAPIAnalyzer(graph).analyze()
    assert report.routes[0].method == "POST"
    assert report.routes[0].path == "/items"
