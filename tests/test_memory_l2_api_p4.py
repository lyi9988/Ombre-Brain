"""P4 owner and resolver API coverage without importing the server startup."""
from __future__ import annotations

import ast
import asyncio
import copy
import json
import logging
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from test_memory_l1_revisions import request_for
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


def test_real_api_handler_routes_owner_versions_save_and_stale_409_preserves_draft(tmp_path):
    service, authority, model_calls = setup_l2(tmp_path)
    app, route_specs = _api_app(service, authority)
    expected_routes = {
        ("/api/memory-authority/memories/{memory_id}/presentation", ("GET",)),
        ("/api/memory-authority/memories/{memory_id}/presentation/prepare", ("POST",)),
        ("/api/memory-authority/memories/{memory_id}/presentation/revisions", ("POST",)),
        ("/api/memory-authority/presentations/resolve", ("POST",)),
    }
    assert {(path, tuple(methods)) for path, methods in route_specs} == expected_routes

    run(service.run_once(1))
    assert len(model_calls) == 1
    first = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation?presentation_revision=1",
    ))
    assert first.status_code == 200
    assert first.json()["l2"]["presentation"]["presentation_revision"] == 1
    original_text = first.json()["l2"]["presentation"]["document"]["full_injection_text"]

    edit = owner_payload(service, request="api-owner-edit")
    edit["document"]["full_injection_text"] = "API owner L2 edit, version two."
    saved = run(_send(
        app, "POST", "/api/memory-authority/memories/m1/presentation/revisions",
        json_body=edit,
    ))
    assert saved.status_code == 200
    assert saved.json()["presentation_revision"] == 2
    assert saved.json()["l2"]["presentation"]["document"]["full_injection_text"] == edit["document"]["full_injection_text"]

    old_read = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation?presentation_revision=1",
    ))
    current_read = run(_send(app, "GET", "/api/memory-authority/memories/m1/presentation"))
    assert old_read.status_code == current_read.status_code == 200
    assert old_read.json()["l2"]["presentation"]["document"]["full_injection_text"] == original_text
    assert current_read.json()["l2"]["presentation"]["document"]["full_injection_text"] == edit["document"]["full_injection_text"]

    # The client retains its exact unsaved draft when its first-pass CAS is old.
    stale_draft = owner_payload(service, request="api-stale-l1-draft")
    stale_draft["document"]["full_injection_text"] = "Keep this unsaved draft after 409."
    draft_bytes = json.dumps(stale_draft, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    changed_l1 = request_for(service.revisions, request_id="api-l1-change")
    changed_l1["document"]["narrative"]["body"] += " A new first-pass revision."
    run(service.revisions.save("m1", changed_l1))
    conflict = run(_send(
        app, "POST", "/api/memory-authority/memories/m1/presentation/revisions",
        content=draft_bytes,
    ))
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "revision_conflict"
    assert json.loads(draft_bytes)["document"]["full_injection_text"] == "Keep this unsaved draft after 409."
    historical_read = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation?presentation_revision=2",
    ))
    assert historical_read.status_code == 200
    assert historical_read.json()["l2"]["presentation"]["document"]["full_injection_text"] == edit["document"]["full_injection_text"]
    assert len(model_calls) == 1


def test_prepare_api_only_enqueues_without_running_the_model(tmp_path):
    service, authority, model_calls = setup_l2(tmp_path)
    app, _routes = _api_app(service, authority)
    pending_before = service.store.pending(100)

    response = run(_send(
        app, "POST", "/api/memory-authority/memories/m1/presentation/prepare",
        json_body={"expected_l1_revision": 2, "request_id": "api-prepare-only"},
    ))

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["state"] == "pending"
    pending_after = service.store.pending(100)
    assert len(pending_after) >= len(pending_before)
    assert any(job["task_id"] == response.json()["task_id"] for job in pending_after)
    assert model_calls == []


def test_api_auth_validation_size_limit_and_resolve_are_read_only(tmp_path):
    service, authority, model_calls = setup_l2(tmp_path)
    run(service.run_once(1))
    app, _routes = _api_app(service, authority)
    assert len(model_calls) == 1

    # The test model's new manuscript requires owner review. Save an owner
    # version through the real route so resolve exercises a ready product.
    owner_version = owner_payload(service, request="api-resolve-ready-version")
    approved = run(_send(
        app, "POST", "/api/memory-authority/memories/m1/presentation/revisions",
        json_body=owner_version,
    ))
    assert approved.status_code == 200
    assert approved.json()["presentation_revision"] == 2

    denied = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation", token="",
    ))
    assert denied.status_code == 401

    invalid_query = run(_send(
        app, "GET", "/api/memory-authority/memories/m1/presentation?unexpected=1",
    ))
    assert invalid_query.status_code == 400
    assert invalid_query.json()["error"]["code"] == "unknown_query_field"

    oversized = run(_send(
        app, "POST", "/api/memory-authority/memories/m1/presentation/prepare",
        content=b"x" * 524289,
    ))
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "request_too_large"

    before = copy.deepcopy(service.detail("m1")["l2"])
    statement_id = before["presentation"]["document"]["coherent_sections"][0]["statement_refs"][0]
    resolved = run(_send(
        app, "POST", "/api/memory-authority/presentations/resolve",
        token="internal-read-test",
        json_body={"items": [{
            "memory_id": "m1", "revision": 2,
            "statement_ids": [statement_id], "allowed_uses": ["historical_evidence"],
            "statement_uses": {statement_id: "historical_evidence"},
            "narrative_allowed": True,
        }]},
    ))
    assert resolved.status_code == 200
    item = resolved.json()["items"][0]
    assert item["state"] == "ready" and item["text"]
    assert item["presentation_revision"] == 2
    assert service.detail("m1")["l2"] == before
    assert len(model_calls) == 1
