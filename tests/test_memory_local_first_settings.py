"""Exercise the local-first owner setting through Gateway's real config route."""
import asyncio
import copy

import httpx

from gateway import GatewayService, create_gateway_app, memory_use


def _base_config(tmp_path):
    root = tmp_path / "gateway"
    buckets = root / "buckets"
    state = root / "state"
    buckets.mkdir(parents=True)
    state.mkdir(parents=True)
    return {
        "_runtime_config_path": str(state / "config.runtime.yaml"),
        "buckets_dir": str(buckets),
        "state_dir": str(state),
        "gateway": {
            "upstream_base_url": "",
            "upstream_default_model": "",
            "upstream_models": [],
            "domain_sentinel_enabled": False,
            "memory_sentinel_enabled": False,
            "query_planner_enabled": False,
            "model_request_trace_path": str(state / "model-traces.sqlite3"),
            "model_route_mirror_path": str(state / "model-routes.sqlite3"),
            "prompt_plan_mirror_path": str(state / "prompt-plans.sqlite3"),
        },
        "embedding": {
            "enabled": False,
            "api_key": "",
            "base_url": "http://provider-disabled.invalid",
        },
        "reranker": {
            "enabled": False,
            "api_key": "",
            "base_url": "http://provider-disabled.invalid",
        },
        "memory_authority": {"enabled": False},
        "memory_system": {},
    }


def _app(config):
    service = GatewayService(config)
    app = create_gateway_app(config=config, service=service)
    app.state.gateway_service = service
    return service, app


def test_local_first_setting_survives_real_owner_api_save_reload_legacy_client_and_conflict(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "memory-local-first-test-owner")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "")
    base = _base_config(tmp_path)
    service, app = _app(copy.deepcopy(base))
    overlay_path = tmp_path / "gateway" / "state" / "config.runtime.yaml"
    owner_headers = {"Authorization": "Bearer memory-local-first-test-owner"}
    endpoint = "/api/memory-authority/configuration"

    # The test exercises configuration only. It must not use a provider or upstream key.
    assert service.embedding_engine.enabled is False
    assert service.reranker_engine.enabled is False
    assert service.upstream_api_key == ""
    assert memory_use.option(service, "local_first_enabled", False) is False

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers=owner_headers,
        ) as client:
            initial_response = await client.get(endpoint)
            assert initial_response.status_code == 200
            initial = initial_response.json()
            assert initial["values"]["recall"]["local_first_enabled"] is False
            assert memory_use.option(service, "local_first_enabled", False) is False

            enabled = copy.deepcopy(initial["values"])
            enabled["recall"]["local_first_enabled"] = True
            save_response = await client.put(endpoint, json={
                "expected_revision": initial["revision"],
                "request_id": "local-first-enable",
                "values": enabled,
            })
            assert save_response.status_code == 200
            saved = save_response.json()
            assert saved["status"] == "ok" and saved["applied"] is True
            assert saved["result"]["values"]["recall"]["local_first_enabled"] is True
            assert memory_use.option(service, "local_first_enabled", False) is True

            readback_response = await client.get(endpoint)
            assert readback_response.status_code == 200
            readback = readback_response.json()
            assert readback["revision"] == saved["saved_revision"]
            assert readback["values"]["recall"]["local_first_enabled"] is True

            # Rebuild from the unchanged base config; GET must read the temporary overlay
            # and apply it to the new service instance used by the normal runtime option.
            restarted, restarted_app = _app(copy.deepcopy(base))
            assert memory_use.option(restarted, "local_first_enabled", False) is False
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://gateway.test",
                headers=owner_headers,
            ) as restarted_client:
                restarted_readback = await restarted_client.get(endpoint)
            assert restarted_readback.status_code == 200
            assert restarted_readback.json()["values"]["recall"]["local_first_enabled"] is True
            assert memory_use.option(restarted, "local_first_enabled", False) is True

            # A client from before this field existed can save another value without
            # turning the newly persisted local-first setting back off.
            legacy_values = copy.deepcopy(readback["values"])
            legacy_values["recall"].pop("local_first_enabled")
            legacy_values["limits"]["inject_max_cards"] += 1
            legacy_response = await client.put(endpoint, json={
                "expected_revision": readback["revision"],
                "request_id": "legacy-client-omits-local-first",
                "values": legacy_values,
            })
            assert legacy_response.status_code == 200
            legacy_saved = legacy_response.json()
            assert legacy_saved["result"]["values"]["recall"]["local_first_enabled"] is True
            assert memory_use.option(service, "local_first_enabled", False) is True

            current = (await client.get(endpoint)).json()
            before_invalid = overlay_path.read_bytes()
            for invalid_value in ("false", 0):
                invalid_values = copy.deepcopy(current["values"])
                invalid_values["recall"]["local_first_enabled"] = invalid_value
                invalid_response = await client.put(endpoint, json={
                    "expected_revision": current["revision"],
                    "request_id": f"invalid-local-first-{type(invalid_value).__name__}",
                    "values": invalid_values,
                })
                assert invalid_response.status_code == 400
                assert invalid_response.json()["error"] == {
                    "code": "boolean_required",
                    "field": "recall.local_first_enabled",
                }
                assert overlay_path.read_bytes() == before_invalid
                assert memory_use.option(service, "local_first_enabled", False) is True

            stale_values = copy.deepcopy(current["values"])
            stale_values["recall"]["local_first_enabled"] = False
            conflict_response = await client.put(endpoint, json={
                "expected_revision": initial["revision"],
                "request_id": "stale-local-first-disable",
                "values": stale_values,
            })
            assert conflict_response.status_code == 409
            assert conflict_response.json()["error"]["code"] == "configuration_revision_conflict"
            assert conflict_response.json()["error"]["current_revision"] == current["revision"]
            assert overlay_path.read_bytes() == before_invalid
            assert memory_use.option(service, "local_first_enabled", False) is True

            final_readback = (await client.get(endpoint)).json()
            assert final_readback["values"]["recall"]["local_first_enabled"] is True

    asyncio.run(scenario())
