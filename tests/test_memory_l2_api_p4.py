"""P4 owner and resolver API coverage without importing the server startup."""
from __future__ import annotations

import ast
import asyncio
import copy
import logging
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from test_memory_l2_p4 import owner_payload, setup as setup_l2


def run(task):
    return asyncio.run(task)


def _api_app(service, authority):
    server_path = Path(__file__).resolve().parents[1] / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"), filename=str(server_path))
    handler_node = next(
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "api_memory_l2"
    )
    route_specs = []
    for decorator in handler_node.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        if decorator.func.attr != "custom_route" or not decorator.args:
            continue
        path = ast.literal_eval(decorator.args[0])
        methods = ast.literal_eval(next(
            keyword.value for keyword in decorator.keywords if keyword.arg == "methods"
        ))
        route_specs.append((path, methods))

    # Keep the production handler body and route declarations; only replace
    # the module-level app/startup dependencies with this temporary harness.
    handler_node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[handler_node], type_ignores=[]))
    namespace = {
        "_require_owner_or_internal_memory_read": lambda request: _auth(
            request, {"Bearer owner-test", "Bearer internal-read-test"},
        ),
        "_require_owner_or_internal_index_maintenance": lambda request: _auth(
            request, {"Bearer owner-test"},
        ),
        "reflection_engine": service.engine,
        "memory_authority_store": authority,
        "memory_commit_service": service.revisions.commit,
        "_memory_l2_service": lambda: service,
        "logger": logging.getLogger("test-memory-l2-api"),
    }
    exec(compile(module, str(server_path), "exec"), namespace)
    handler = namespace["api_memory_l2"]
    app = Starlette(routes=[
        Route(path, handler, methods=methods) for path, methods in route_specs
    ])
    return app, route_specs


def _auth(request, accepted):
    if request.headers.get("authorization", "") in accepted:
        return None
    return JSONResponse({"error": {"code": "unauthorized"}}, status_code=401)


async def _send(app, method, path, *, token="owner-test", content=None, json_body=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        return await client.request(
            method, path, headers=headers, content=content, json=json_body,
        )


def test_retired_presentation_writes_and_authenticated_owner_history(tmp_path):
    service, authority, model_calls = setup_l2(tmp_path)
    app, route_specs = _api_app(service, authority)
    expected_routes = {
        ("/api/memory-authority/memories/{memory_id}/presentation", ("GET",)),
        ("/api/memory-authority/memories/{memory_id}/presentation/prepare", ("POST",)),
        ("/api/memory-authority/memories/{memory_id}/presentation/revisions", ("POST",)),
        ("/api/memory-authority/presentations/resolve", ("POST",)),
    }
    assert {(path, tuple(methods)) for path, methods in route_specs} == expected_routes

    run(service.run_once(1))  # Explicit legacy test data, outside the current worker.
    base = "/api/memory-authority/memories/m1/presentation"
    model_text = service.detail("m1")["l2"]["presentation"]["document"]["full_injection_text"]
    edit = owner_payload(service, request="legacy-owner-edit")
    edit["document"]["full_injection_text"] = "Previously saved owner L2 manuscript."
    run(service.save("m1", edit))
    before = copy.deepcopy(service.detail("m1")["l2"])

    for action, payload in (("/prepare", {"expected_l1_revision": 2, "request_id": "retired"}),
                            ("/revisions", {**edit, "request_id": "no-write"})):
        denied = run(_send(app, "POST", base + action, json_body=payload, token="internal-read-test"))
        gone = run(_send(app, "POST", base + action, json_body=payload))
        assert denied.status_code == 401
        assert gone.status_code == 410 and gone.json()["error"]["code"] == "l2_retired"
        assert gone.headers["Cache-Control"] == "no-store"
    assert service.detail("m1")["l2"] == before and len(model_calls) == 1

    first = run(_send(app, "GET", base + "?presentation_revision=1"))
    current = run(_send(app, "GET", base))
    assert first.status_code == current.status_code == 200
    assert first.json()["retired"] is True
    assert first.json()["l2"]["presentation"]["document"]["full_injection_text"] == model_text
    assert current.json()["l2"]["presentation"]["document"]["full_injection_text"] == edit["document"]["full_injection_text"]


def test_live_resolver_is_retired_even_for_available_owner_versions(tmp_path):
    service, authority, model_calls = setup_l2(tmp_path)
    run(service.run_once(1))
    app, _routes = _api_app(service, authority)
    assert len(model_calls) == 1

    # The old resolver could return this owner version as a ready product.
    owner_version = owner_payload(service, request="api-resolve-ready-version")
    approved = run(service.save("m1", owner_version))
    assert approved["presentation_revision"] == 2

    denied = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation", token="",
    ))
    assert denied.status_code == 401

    invalid_query = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation?unexpected=1",
    ))
    assert invalid_query.status_code == 400
    assert invalid_query.json()["error"]["code"] == "unknown_query_field"

    before = copy.deepcopy(service.detail("m1")["l2"])
    resolve_path = "/api/memory-authority/presentations/resolve"
    denied = run(_send(app, "POST", resolve_path, token="", json_body={"items": []}))
    assert denied.status_code == 401
    statement_id = before["presentation"]["document"]["coherent_sections"][0]["statement_refs"][0]
    resolved = run(_send(
        app, "POST", resolve_path,
        token="internal-read-test",
        json_body={"items": [{
            "memory_id": "m1", "revision": 2,
            "statement_ids": [statement_id], "allowed_uses": ["historical_evidence"],
            "statement_uses": {statement_id: "historical_evidence"},
            "narrative_allowed": True,
        }]},
    ))
    assert resolved.status_code == 410
    assert resolved.json() == {"error": {"code": "l2_retired"}}
    assert resolved.headers["Cache-Control"] == "no-store"
    assert service.detail("m1")["l2"] == before
    assert len(model_calls) == 1
