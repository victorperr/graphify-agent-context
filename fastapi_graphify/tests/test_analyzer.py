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


TEMPLATE_LIKE_PROJECT = {
    "app/api/deps.py": """
from typing import Annotated
from fastapi import Depends
from sqlmodel import Session
def get_db(): yield None
SessionDep = Annotated[Session, Depends(get_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]
def get_current_user(session: SessionDep, token: TokenDep) -> User: ...
CurrentUser = Annotated[User, Depends(get_current_user)]
def get_current_active_superuser(current_user: CurrentUser) -> User: ...
""",
    "app/models.py": """
from sqlmodel import SQLModel
class ItemBase(SQLModel):
    title: str
class Item(ItemBase, table=True):
    owner_id: int
class ItemPublic(ItemBase):
    id: int
class NotAModel:
    pass
""",
    "app/api/routes/users.py": """
from typing import Any
from fastapi import APIRouter, Depends, status
from app.api.deps import CurrentUser, SessionDep, get_current_active_superuser
router = APIRouter(prefix="/users", tags=["users"])
@router.get("/", dependencies=[Depends(get_current_active_superuser)], response_model=UsersPublic)
def read_users(session: SessionDep, skip: int = 0, limit: int = 100): ...
@router.post("/signup", status_code=status.HTTP_201_CREATED)
def register_user(session: SessionDep) -> UserPublic: ...
@router.get("/me")
def read_user_me(current_user: CurrentUser) -> Any: ...
""",
    "app/api/routes/items.py": """
from typing import Annotated
from fastapi import APIRouter, Depends, Security
router = APIRouter(prefix="/items", dependencies=[Depends(get_db)])
@router.post("/", response_model=ItemPublic)
def create_item(user: Annotated[User, Security(get_current_user)]): ...
""",
}


def analyze_template_like_project(tmp_path: Path):
    for relative, source in TEMPLATE_LIKE_PROJECT.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(source, encoding="utf-8")
    report = FastAPIAnalyzer().analyze(tmp_path)
    return report, {route.handler: route for route in report.routes}


def test_routes_resolve_annotated_aliases_decorator_and_router_dependencies(tmp_path: Path):
    _, routes = analyze_template_like_project(tmp_path)
    assert routes["read_users"].dependencies == [
        "Depends(get_current_active_superuser)", "Depends(get_db)"]
    assert routes["read_user_me"].dependencies == ["Depends(get_current_user)"]
    assert routes["create_item"].dependencies == ["Depends(get_db)", "Depends(get_current_user)"]


def test_nested_dependencies_follow_annotated_aliases(tmp_path: Path):
    report, _ = analyze_template_like_project(tmp_path)
    dependencies = {dependency.name: dependency.dependencies for dependency in report.dependencies}
    assert dependencies["get_current_user"] == ["Depends(get_db)", "Depends(reusable_oauth2)"]
    assert dependencies["get_current_active_superuser"] == ["Depends(get_current_user)"]
    assert {"source": "dependency:get_current_active_superuser", "target": "dependency:get_current_user",
            "relation": "calls"} in report.domain_edges


def test_router_prefix_status_codes_and_return_annotations(tmp_path: Path):
    _, routes = analyze_template_like_project(tmp_path)
    assert (routes["read_users"].method, routes["read_users"].path) == ("GET", "/users/")
    assert routes["register_user"].path == "/users/signup"
    assert routes["register_user"].status_codes == [201]
    assert routes["read_users"].status_codes == []  # `limit: int = 100` is not a status code
    assert routes["register_user"].response_models == ["UserPublic"]
    assert routes["read_user_me"].response_models == []  # `-> Any` names no model


def test_sqlmodel_and_inherited_models_are_detected(tmp_path: Path):
    report, _ = analyze_template_like_project(tmp_path)
    models = {model.name: model for model in report.models}
    assert set(models) == {"ItemBase", "Item", "ItemPublic"}
    assert models["Item"].table and not models["ItemPublic"].table
    item_public = next(node for node in report.domain_nodes if node["id"] == "model:ItemPublic")
    assert item_public["source_file"] == "app/models.py"  # kept although a route referenced it first


def test_graph_metadata_can_supply_routes_without_source():
    graph = {
        "nodes": [{"id": "read", "label": "@router.post('/items') read"}], "links": []}
    report = FastAPIAnalyzer(graph).analyze()
    assert report.routes[0].method == "POST"
    assert report.routes[0].path == "/items"
