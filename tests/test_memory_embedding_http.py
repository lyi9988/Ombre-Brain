"""Offline HTTP boundary tests for owner-controlled embedding maintenance."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
from types import SimpleNamespace

import pytest
from starlette.responses import JSONResponse

from memory_embedding_jobs import EmbeddingJobError


class FakeBrainRequest:
    def __init__(self, method, *, action=None, headers=None, body=None):
        self.method = method
        self.path_params = {} if action is None else {"action": action}
        self.headers = dict(headers or {})
        self.cookies = {}
        self._body = copy.deepcopy(body)

    async def json(self):
        return copy.deepcopy(self._body)


class RecordingJobs:
    def __init__(self):
        self.calls = []
        self.errors = {}

    def _call(self, action, body=None):
        self.calls.append((action, copy.deepcopy(body)))
        error = self.errors.get(action)
        if error:
            raise error
        return {"action": action}

    def view(self):
        return self._call("view")

    async def preview(self):
        return self._call("preview")

    def start(self, body):
        return self._call("start", body)

    def continue_full(self, body):
        return self._call("continue_full", body)

    def stop(self, body):
        return self._call("stop", body)


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture
def brain_api(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("OMBRE_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.setenv("OMBRE_STATE_DIR", str(state_dir))
    monkeypatch.setenv(
        "OMBRE_RUNTIME_CONFIG_PATH", str(state_dir / "config.runtime.yaml")
    )

    server = importlib.import_module("server")
    config = server.load_config()
    config["state_dir"] = str(state_dir)
    config["_runtime_config_path"] = str(state_dir / "config.runtime.yaml")
    monkeypatch.setattr(server, "config", config)

    jobs = RecordingJobs()
    monkeypatch.setattr(server, "memory_embedding_jobs", jobs)
    monkeypatch.setattr(
        server,
        "_dashboard_authenticated",
        lambda request: request.headers.get("x-test-owner") == "owner-session",
    )
    monkeypatch.setattr(
        server,
        "_authorized_memory_write",
        lambda request: request.headers.get("authorization")
        == "Bearer internal-memory-token",
    )
    monkeypatch.setattr(
        server,
        "_require_dashboard_auth",
        lambda _request: JSONResponse({"error": "unauthorized"}, status_code=401),
    )
    return SimpleNamespace(server=server, jobs=jobs)


def test_brain_embedding_index_routes_require_auth_before_any_job_action(brain_api):
    actions = [
        ("GET", None, None),
        ("POST", "preview", {}),
        ("POST", "start", {"confirm_provider_use": True}),
        ("POST", "continue", {"confirm_provider_use": True}),
        ("POST", "stop", {"job_id": "job-1"}),
    ]

    for method, action, body in actions:
        response = asyncio.run(brain_api.server.api_memory_embedding_index(
            FakeBrainRequest(method, action=action, body=body)
        ))
        assert response.status_code == 401
    assert brain_api.jobs.calls == []


def test_brain_embedding_index_routes_dispatch_continue_and_all_controls(brain_api):
    headers = {"authorization": "Bearer internal-memory-token"}
    continue_body = {
        "job_id": "job-1",
        "expected_space_token": "space-token",
        "confirm_provider_use": True,
    }
    actions = [
        ("GET", None, None, "view"),
        ("POST", "preview", {}, "preview"),
        ("POST", "start", {"job_id": "new-job"}, "start"),
        ("POST", "continue", continue_body, "continue_full"),
        ("POST", "stop", {"job_id": "job-1"}, "stop"),
    ]

    for method, action, body, expected_action in actions:
        response = asyncio.run(brain_api.server.api_memory_embedding_index(
            FakeBrainRequest(method, action=action, headers=headers, body=body)
        ))
        assert response.status_code == 200
        assert response_json(response) == {"action": expected_action}

    assert brain_api.jobs.calls == [
        ("view", None),
        ("preview", None),
        ("start", {"job_id": "new-job"}),
        ("continue_full", continue_body),
        ("stop", {"job_id": "job-1"}),
    ]


def test_brain_embedding_index_maps_job_errors_to_400_and_409(brain_api):
    brain_api.jobs.errors.update({
        "start": EmbeddingJobError("confirm_provider_use_required", 400),
        "continue_full": EmbeddingJobError("embedding_configuration_changed", 409),
    })
    headers = {"x-test-owner": "owner-session"}

    invalid_start = asyncio.run(brain_api.server.api_memory_embedding_index(
        FakeBrainRequest("POST", action="start", headers=headers, body={})
    ))
    stale_continue = asyncio.run(brain_api.server.api_memory_embedding_index(
        FakeBrainRequest("POST", action="continue", headers=headers, body={})
    ))

    assert invalid_start.status_code == 400
    assert response_json(invalid_start) == {"error": "confirm_provider_use_required"}
    assert stale_continue.status_code == 409
    assert response_json(stale_continue) == {
        "error": "embedding_configuration_changed"
    }
    assert brain_api.jobs.calls == [("start", {}), ("continue_full", {})]


class FakeGatewayRequest:
    def __init__(self, method, *, action=None, authorization="Bearer owner-token", body=None):
        self.method = method
        self.path_params = {} if action is None else {"action": action}
        self.headers = {"Authorization": authorization}
        self._body = copy.deepcopy(body)

    async def json(self):
        return copy.deepcopy(self._body)


class FakeGatewayResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return copy.deepcopy(self._payload)


def gateway_service(monkeypatch, gateway_module, *, internal_token="internal-token"):
    service = gateway_module.GatewayService.__new__(gateway_module.GatewayService)
    service.gateway_token = "" if internal_token is None else "gateway-fallback-token"
    monkeypatch.setattr(
        service, "_authorize",
        lambda header: None if header == "Bearer owner-token"
        else JSONResponse({"error": "unauthorized"}, status_code=401),
    )
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.invalid:8000")
    if internal_token is None:
        monkeypatch.delenv("OMBRE_MEMORY_WRITE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", internal_token)
    return service


def test_gateway_embedding_index_transport_whitelist_and_internal_token(
    monkeypatch,
):
    import gateway

    service = gateway_service(monkeypatch, gateway)
    requests = []
    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            clients.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def request(self, method, url, *, headers, json):
            requests.append((method, url, headers, json))
            return FakeGatewayResponse(200, {"status": "ok"})

    monkeypatch.setattr(gateway.httpx, "AsyncClient", FakeClient)
    responses = []
    actions = [
        ("GET", None, None, "/api/memory-authority/embedding-index"),
        ("POST", "preview", {}, "/api/memory-authority/embedding-index/preview"),
        ("POST", "start", {"confirm_provider_use": True}, "/api/memory-authority/embedding-index/start"),
        ("POST", "continue", {"job_id": "job-1"}, "/api/memory-authority/embedding-index/continue"),
        ("POST", "stop", {"job_id": "job-1"}, "/api/memory-authority/embedding-index/stop"),
    ]

    for method, action, body, _path in actions:
        response = asyncio.run(service.handle_memory_embedding_index(
            FakeGatewayRequest(method, action=action, body=body)
        ))
        responses.append(response)
        assert response.status_code == 200
        assert json.loads(response.body.decode("utf-8")) == {"status": "ok"}

    assert len(clients) == len(actions)
    assert [request[1] for request in requests] == [
        "http://brain.invalid:8000" + path for _method, _action, _body, path in actions
    ]
    assert all(request[2] == {"Authorization": "Bearer internal-token"}
               for request in requests)
    assert [request[3] for request in requests] == [body for _m, _a, body, _p in actions]
    assert all("internal-token" not in response.body.decode("utf-8")
               for response in responses)
    assert len(requests) == len(actions)


def test_gateway_embedding_index_transport_fails_closed_without_auth_or_upstream(
    monkeypatch,
):
    import gateway

    service = gateway_service(monkeypatch, gateway)
    requests = []

    class UnexpectedClient:
        def __init__(self, **_kwargs):
            requests.append("client-created")

    monkeypatch.setattr(gateway.httpx, "AsyncClient", UnexpectedClient)
    denied = asyncio.run(service.handle_memory_embedding_index(
        FakeGatewayRequest("GET", authorization="Bearer wrong-token")
    ))
    unknown = asyncio.run(service.handle_memory_embedding_index(
        FakeGatewayRequest("POST", action="delete", body={})
    ))
    assert denied.status_code == 401
    assert unknown.status_code == 404
    assert requests == []

    service = gateway_service(monkeypatch, gateway, internal_token=None)
    unconfigured = asyncio.run(service.handle_memory_embedding_index(
        FakeGatewayRequest("GET")
    ))
    assert unconfigured.status_code == 503
    assert json.loads(unconfigured.body.decode("utf-8")) == {
        "error": "embedding_maintenance_unconfigured"
    }
    assert requests == []


def test_gateway_embedding_index_transport_maps_upstream_failures_to_503(monkeypatch):
    import gateway

    service = gateway_service(monkeypatch, gateway)

    class FailedUpstreamClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def request(self, *_args, **_kwargs):
            return FakeGatewayResponse(500, {"secret": "never-forward"})

    monkeypatch.setattr(gateway.httpx, "AsyncClient", FailedUpstreamClient)
    response = asyncio.run(service.handle_memory_embedding_index(
        FakeGatewayRequest("GET")
    ))
    assert response.status_code == 503
    assert json.loads(response.body.decode("utf-8")) == {
        "error": "embedding_maintenance_unavailable"
    }
    assert "never-forward" not in response.body.decode("utf-8")
