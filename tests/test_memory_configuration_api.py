import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from gateway import GatewayService, create_gateway_app
from memory_system_config import REQUEST_CONFIGURATION, configuration, read_overlay


def service_at(path):
    service = GatewayService.__new__(GatewayService)
    service.config = {"gateway": {"inject_max_cards": 2}, "unrelated": {"kept": True}}
    service.gateway_cfg = service.config["gateway"]
    service.reranker_engine = SimpleNamespace(candidate_limit=20)
    service._runtime_overlay_path = str(path)
    service._authorize = lambda auth: None if auth == "Bearer owner" else JSONResponse({"error": "unauthorized"}, 401)
    return service


def test_real_routes_save_readback_preview_conflict_and_credential_boundary(tmp_path):
    service = service_at(tmp_path / "runtime.yaml")
    app = create_gateway_app(config=service.config, service=service)
    app.state.gateway_service = service
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            path = "/api/memory-authority/configuration"
            assert (await client.get(path)).status_code == 401
            headers = {"Authorization": "Bearer owner"}
            first = (await client.get(path, headers=headers)).json()
            assert first["schema_version"] == "memory-config-v1"
            draft = copy.deepcopy(first["values"])
            draft["limits"].update(inject_max_cards=7, diffusion_inject_max_items=5,
                                   memory_detail_recall_max_ids=12, dynamic_top_k=20, semantic_candidate_top_k=5)
            draft["catalog"]["topics"].append("摄影")
            body = {"expected_revision": first["revision"], "request_id": "local-owner-save", "values": draft}
            saved = await client.put(path, headers=headers, json=body)
            assert saved.status_code == 200
            saved = saved.json()
            assert saved["status"] == "ok" and saved["applied"] is True
            assert saved["result"]["values"] == draft
            assert service.inject_max_cards == 7 and service.memory_detail_recall_max_ids == 12
            assert service.dynamic_top_k == 20 and service.semantic_candidate_top_k == 5
            readback = (await client.get(path, headers=headers)).json()
            assert readback["revision"] == saved["saved_revision"]
            assert readback["application"]["formulas"]["state"] == "awaiting_consumer"
            again = (await client.put(path, headers=headers, json=body)).json()
            assert again["replayed"] is True and again["applied"] is True
            conflict = await client.put(path, headers=headers, json={**body, "request_id": "stale"})
            assert conflict.status_code == 409
            preview = await client.post(path + "/preview", headers=headers,
                json={"values": draft, "inputs": {"relevance": .8}})
            result = preview.json()["result"]
            assert result["formulas"]["final"]["value"] == .8
            assert result["provider_requests"] == 0 and result["saved"] is False
            before = (tmp_path / "runtime.yaml").read_bytes()
            bad = copy.deepcopy(draft)
            bad["limits"]["inject_max_cards"] = 201
            rejected = await client.put(path, headers=headers, json={**body,
                "expected_revision": readback["revision"], "request_id": "invalid", "values": bad})
            assert rejected.status_code == 400
            assert (tmp_path / "runtime.yaml").read_bytes() == before
            assert "api_key" not in json.dumps(readback)
            assert "unrelated" not in readback
    asyncio.run(run())


def test_frozen_limits_survive_concurrent_save_and_stream_iteration(tmp_path):
    service = service_at(tmp_path / "runtime.yaml")
    service._maybe_reload_runtime_overlay = lambda: None
    seen = []
    async def handler(request):
        seen.append(service.inject_max_cards)
        # Simulate a different request applying new settings while this one
        # yields. The runtime setter must not change this request's view.
        service.inject_max_cards = 11
        assert service.inject_max_cards == 2
        async def stream():
            seen.append(service.inject_max_cards)
            yield str(service.inject_max_cards).encode()
        return StreamingResponse(stream())
    service._handle_chat_unfrozen = handler
    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                       "headers": [(b"authorization", b"Bearer owner")]})
    async def run():
        response = await service.handle_chat(request)
        assert REQUEST_CONFIGURATION.get() is None
        assert service.inject_max_cards == 11
        assert b"".join([part async for part in response.body_iterator]) == b"2"
        assert REQUEST_CONFIGURATION.get() is None
    asyncio.run(run())
    assert seen == [2, 2]


def test_old_replay_reports_not_currently_applied_instead_of_overwriting_new_state(tmp_path):
    service = service_at(tmp_path / "runtime.yaml")
    app = create_gateway_app(config=service.config, service=service)
    app.state.gateway_service = service
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                     headers={"Authorization": "Bearer owner"}) as client:
            path = "/api/memory-authority/configuration"
            first = (await client.get(path)).json()
            old = {"expected_revision": first["revision"], "request_id": "first", "values": first["values"]}
            saved = (await client.put(path, json=old)).json()
            newer = copy.deepcopy(first["values"])
            newer["limits"]["inject_max_cards"] = 8
            await client.put(path, json={"expected_revision": saved["saved_revision"],
                                         "request_id": "second", "values": newer})
            replay = (await client.put(path, json=old)).json()
            assert replay["replayed"] is True and replay["applied"] is False
            assert service.inject_max_cards == 8
            assert read_overlay(tmp_path / "runtime.yaml")["gateway"]["inject_max_cards"] == 8
    asyncio.run(run())
