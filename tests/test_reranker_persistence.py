"""Offline regression for Apply -> cleared input -> Save credentials.

Compile the actual route and file writer without starting server storage/jobs.
No provider calls, live environment reads, or real credentials are involved.
"""
import ast
import asyncio
import copy
import json
import os
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values
from reranker_engine import RerankerEngine


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return copy.deepcopy(self.body)


@pytest.fixture
def route(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "server.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"api_config_update", "_write_dashboard_env_values", "_quote_env_value"}
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            definitions.append(node)
    assert {node.name for node in definitions} == names
    env_path = tmp_path / ".env"
    env_path.write_text("OMBRE_RERANKER_API_KEY=old-key\nKEEP_ME=untouched\n", encoding="utf-8")
    for key in ("OMBRE_RERANKER_API_KEY", "KEEP_ME"):
        monkeypatch.setenv(key, "fixture-original")

    async def no_gateway(_payload):
        return None

    namespace = {
        "__file__": str(source),
        "os": os, "re": re, "json": json, "_json_lib": json,
        "config": {"_runtime_config_path": str(tmp_path / "state" / "config.runtime.yaml"),
                   "reranker": {"api_key": "live-rerank-key", "enabled": True,
                                 "model": "test-reranker", "base_url": "https://rerank.invalid/v1"}},
        "RerankerEngine": RerankerEngine,
        "_require_dashboard_auth": lambda _: None,
        "_dashboard_env_path": lambda: str(env_path),
        "_hot_update_gateway_config": no_gateway,
    }
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace, env_path


def invoke(namespace, body):
    response = asyncio.run(namespace["api_config_update"](Request(body)))
    assert response.status_code == 200
    return json.loads(response.body)


@pytest.mark.parametrize("payload", [{}, {"api_key": ""}])
def test_explicit_save_persists_current_own_key_after_input_cleared(route, payload):
    ns, path = route
    inode = path.stat().st_ino
    result = invoke(ns, {"reranker": payload, "persist_env": True})
    assert dotenv_values(path, interpolate=False)["OMBRE_RERANKER_API_KEY"] == "live-rerank-key"
    assert "KEEP_ME=untouched" in path.read_text(encoding="utf-8")
    assert path.stat().st_ino == inode
    assert "env.OMBRE_RERANKER_API_KEY" in result["updated"]
    assert "reranker.api_key_from_runtime" in result["updated"]
    assert "live-rerank-key" not in json.dumps(result)


def test_new_key_wins_over_prior_runtime_key(route):
    ns, path = route
    result = invoke(ns, {"reranker": {"api_key": "new-key"}, "persist_env": True})
    assert dotenv_values(path, interpolate=False)["OMBRE_RERANKER_API_KEY"] == "new-key"
    assert "reranker.api_key" in result["updated"]
    assert "reranker.api_key_from_runtime" not in result["updated"]


def test_runtime_only_then_persist_without_retyping(route):
    ns, path = route
    before = path.read_bytes()
    invoke(ns, {"reranker": {"api_key": "applied-not-saved"}, "persist_env": False})
    assert path.read_bytes() == before
    result = invoke(ns, {"reranker": {}, "persist_env": True})
    assert dotenv_values(path, interpolate=False)["OMBRE_RERANKER_API_KEY"] == "applied-not-saved"
    assert "env.OMBRE_RERANKER_API_KEY" in result["updated"]


def test_dashboard_save_persists_own_key_to_env_not_yaml(route):
    ns, path = route
    result = invoke(ns, {"reranker": {}, "persist": True, "persist_env": True})
    assert dotenv_values(path, interpolate=False)["OMBRE_RERANKER_API_KEY"] == "live-rerank-key"
    yaml_path = Path(ns["config"]["_runtime_config_path"])
    assert yaml_path.exists()
    assert "live-rerank-key" not in yaml_path.read_text(encoding="utf-8")
    assert "persisted_to_runtime_yaml" in result["updated"]
    assert "persisted_to_env" in result["updated"]


@pytest.mark.parametrize("body", [{"reranker": {}}, {"persist_env": True}])
def test_unrequested_scope_or_persistence_never_writes_reranker(route, body):
    ns, path = route
    before = path.read_bytes()
    result = invoke(ns, body)
    assert path.read_bytes() == before
    assert "env.OMBRE_RERANKER_API_KEY" not in result["updated"]


def test_missing_own_key_does_not_persist_engine_cross_provider_fallback(route):
    ns, path = route
    ns["config"]["reranker"]["api_key"] = ""
    ns["config"]["embedding"] = {"api_key": "embedding-only"}
    before = path.read_bytes()
    result = invoke(ns, {"reranker": {}, "persist_env": True})
    assert path.read_bytes() == before
    assert "env.OMBRE_RERANKER_API_KEY" not in result["updated"]


def test_dashboard_auth_precedes_persistence(route):
    from starlette.responses import JSONResponse
    ns, path = route
    before = path.read_bytes()
    ns["_require_dashboard_auth"] = lambda _: JSONResponse({"error": "unauthorized"}, status_code=401)
    response = asyncio.run(ns["api_config_update"](Request({"reranker": {}, "persist_env": True})))
    assert response.status_code == 401
    assert path.read_bytes() == before
