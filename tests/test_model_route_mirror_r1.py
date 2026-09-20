import pytest
import asyncio
import json
import httpx

from gateway import GatewayService
from model_route_mirror import (
    ModelRouteMirrorConflict,
    ModelRouteMirrorStore,
    ModelRouteMirrorValidationError,
    route_slice_sha256,
)


def routes():
    return [
        {
            "route_id": "memory_query_planner",
            "enabled": True,
            "candidates": [
                {
                    "provider_id": "ombre",
                    "model_id": "deepseek-v4-flash",
                    "position": 0,
                    "overrides": {"temperature": 0, "max_tokens": 320},
                }
            ],
        },
        {
            "route_id": "autonomy_reasoner",
            "enabled": False,
            "candidates": [],
        },
    ]


def test_secretless_route_mirror_is_revisioned_resolvable_and_idempotent(tmp_path):
    store = ModelRouteMirrorStore(tmp_path / "model-routes.sqlite3")
    payload = routes()
    digest = route_slice_sha256(payload)

    first = store.put(
        revision=1,
        routes=payload,
        route_sha256=digest,
        source_authority="aizizhu.model_registry",
        request_id="sync-1",
    )
    second = store.put(
        revision=1,
        routes=payload,
        route_sha256=digest,
        source_authority="aizizhu.model_registry",
        request_id="sync-1",
    )

    assert first == second
    assert first["secretless"] is True
    resolved = store.resolve("memory_query_planner")
    assert resolved["revision"] == 1
    assert resolved["candidates"][0]["model_id"] == "deepseek-v4-flash"
    assert store.resolve("autonomy_reasoner") is None
    assert "key" not in str(first).lower()


def test_route_mirror_rejects_secrets_hash_drift_and_stale_revision(tmp_path):
    store = ModelRouteMirrorStore(tmp_path / "model-routes.sqlite3")
    payload = routes()
    store.put(
        revision=2,
        routes=payload,
        route_sha256=route_slice_sha256(payload),
        source_authority="aizizhu.model_registry",
        request_id="sync-2",
    )

    leaked = routes()
    leaked[0]["candidates"][0]["api_key"] = "secret"
    with pytest.raises(ModelRouteMirrorValidationError, match="secrets"):
        route_slice_sha256(leaked)
    with pytest.raises(ModelRouteMirrorValidationError, match="mismatch"):
        store.put(
            revision=3,
            routes=payload,
            route_sha256="0" * 64,
            source_authority="aizizhu.model_registry",
            request_id="sync-3",
        )
    with pytest.raises(ModelRouteMirrorConflict, match="stale"):
        store.put(
            revision=1,
            routes=payload,
            route_sha256=route_slice_sha256(payload),
            source_authority="aizizhu.model_registry",
            request_id="sync-stale",
        )


def test_same_request_id_with_different_route_payload_conflicts(tmp_path):
    store = ModelRouteMirrorStore(tmp_path / "model-routes.sqlite3")
    payload = routes()
    store.put(
        revision=1,
        routes=payload,
        route_sha256=route_slice_sha256(payload),
        source_authority="aizizhu.model_registry",
        request_id="same-request",
    )
    changed = routes()
    changed[0]["candidates"][0]["model_id"] = "gemini-3-flash-preview"
    with pytest.raises(ModelRouteMirrorConflict, match="payload differs"):
        store.put(
            revision=2,
            routes=changed,
            route_sha256=route_slice_sha256(changed),
            source_authority="aizizhu.model_registry",
            request_id="same-request",
        )


def test_gateway_model_route_mirror_endpoint_is_secretless_and_no_store(tmp_path):
    payload = routes()

    class Request:
        headers = {"Authorization": "Bearer synthetic"}
        path_params = {}
        query_params = {}

        def __init__(self, method, body=None):
            self.method = method
            self._body = body

        async def json(self):
            return self._body

    service = GatewayService.__new__(GatewayService)
    service._authorize = lambda _header: None
    service.model_route_mirror = ModelRouteMirrorStore(tmp_path / "model-routes.sqlite3")
    put = asyncio.run(service.handle_model_route_mirror(Request("PUT", {
        "revision": 1,
        "route_sha256": route_slice_sha256(payload),
        "routes": payload,
        "source_authority": "aizizhu.model_registry",
        "request_id": "gateway-sync-1",
    })))
    get = asyncio.run(service.handle_model_route_mirror(Request("GET")))

    assert put.status_code == get.status_code == 200
    assert put.headers["cache-control"] == "no-store"
    body = json.loads(get.body.decode("utf-8"))
    assert body["mirror"]["revision"] == 1
    assert "synthetic" not in str(body)


def test_internal_route_uses_mirrored_model_and_safe_overrides(tmp_path):
    store = ModelRouteMirrorStore(tmp_path / "model-routes.sqlite3")
    payload = [{
        "route_id": "memory_query_planner",
        "enabled": True,
        "candidates": [{
            "provider_id": "ombre",
            "model_id": "deepseek-v4-flash",
            "position": 0,
            "overrides": {"temperature": 0.1, "max_tokens": 123},
        }],
    }]
    store.put(
        revision=7,
        routes=payload,
        route_sha256=route_slice_sha256(payload),
        source_authority="aizizhu.model_registry",
        request_id="sync-7",
    )
    seen = []
    service = GatewayService.__new__(GatewayService)
    service.model_route_mirror = store

    async def forward(request_payload):
        seen.append(request_payload)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"ok":true}'}}]
        })

    service._forward_upstream = forward
    content, error, debug = asyncio.run(service._call_mirrored_internal_route(
        "memory_query_planner",
        {"messages": [], "temperature": 0.9, "max_tokens": 999, "stream": False},
    ))

    assert error is None
    assert content == '{"ok":true}'
    assert seen[0]["model"] == "deepseek-v4-flash"
    assert seen[0]["temperature"] == 0.1
    assert seen[0]["max_tokens"] == 123
    assert debug["revision"] == 7
    assert len(debug["route_sha256"]) == 64
