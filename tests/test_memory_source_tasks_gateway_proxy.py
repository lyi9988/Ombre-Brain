"""Real loopback coverage for the Gateway source-task transport.

Both the Gateway route and its Brain-side HTTP peer run over 127.0.0.1.  The
peer is synthetic; these tests do not start Brain, call a model, or touch owner
or production data.
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import uvicorn

from gateway import GatewayService, create_gateway_app


@contextmanager
def source_task_peer():
    state = {"calls": [], "responses": {}}

    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            state["calls"].append({
                "method": self.command,
                "path": self.path,
                "body": raw,
                "authorization": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
            })
            mode = parse_qs(urlsplit(self.path).query).get("mode", ["ok"])[0]
            status, body, extra_headers = state["responses"].get(
                mode, (200, {"items": [], "status": "ok"}, {})
            )
            if isinstance(body, bytes):
                encoded = body
            else:
                encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            for name, value in extra_headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = _serve
        do_POST = _serve

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def source_task_gateway():
    service = GatewayService.__new__(GatewayService)
    service.config = {"gateway": {}}
    service.gateway_cfg = service.config["gateway"]
    service.gateway_token = "owner-token"
    app = create_gateway_app(config=service.config, service=service)
    app.state.gateway_service = service
    return app


@contextmanager
def gateway_loopback(app):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, log_level="critical", access_log=False, lifespan="off"
    ))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "Gateway loopback server did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


def test_source_task_routes_auth_body_and_query_over_loopback(monkeypatch):
    with source_task_peer() as (brain_url, peer):
        monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", brain_url)
        monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "brain-write-token")
        peer["responses"]["accepted"] = (202, {"status": "ok", "items": []}, {})
        with gateway_loopback(source_task_gateway()) as gateway_url:
            with httpx.Client(base_url=gateway_url, trust_env=False, timeout=3) as client:
                denied = client.get(
                    "/api/memory-authority/source-tasks?limit=2",
                )
                assert denied.status_code == 401
                assert peer["calls"] == []

                listed = client.get(
                    "/api/memory-authority/source-tasks?limit=2&cursor=a%2Fb&limit=3",
                    headers={"Authorization": "Bearer owner-token"},
                )
                assert listed.status_code == 200
                assert listed.json() == {"items": [], "status": "ok"}
                assert peer["calls"][-1]["method"] == "GET"
                assert peer["calls"][-1]["path"] == (
                    "/api/memory-authority/source-tasks?limit=2&cursor=a%2Fb&limit=3"
                )
                assert peer["calls"][-1]["authorization"] == "Bearer brain-write-token"

                body = b'{"request_id":"local-1","source_id":"diary:test","metadata":{"v":1}}'
                created = client.post(
                    "/api/memory-authority/action-source?mode=accepted&cursor=x%2Fy",
                    content=body,
                    headers={
                        "Authorization": "Bearer owner-token",
                        "Content-Type": "application/json",
                    },
                )
                assert created.status_code == 202
                assert created.json() == {"status": "ok", "items": []}
                assert peer["calls"][-1]["method"] == "POST"
                assert peer["calls"][-1]["path"] == (
                    "/api/memory-authority/action-source?mode=accepted&cursor=x%2Fy"
                )
                assert peer["calls"][-1]["body"] == body
                assert peer["calls"][-1]["content_type"] == "application/json"

                peer["responses"]["conflict"] = (409, {"status": "existing"}, {})
                retried = client.post(
                    "/api/memory-authority/source-tasks/task%20%CE%B1/retry"
                    "?mode=conflict&reason=owner",
                    content=b'{"request_id":"retry-1"}',
                    headers={
                        "Authorization": "Bearer owner-token",
                        "Content-Type": "application/json",
                    },
                )
                assert retried.status_code == 409
                assert retried.json() == {"status": "existing"}
                assert peer["calls"][-1]["path"] == (
                    "/api/memory-authority/source-tasks/task%20%CE%B1/retry"
                    "?mode=conflict&reason=owner"
                )
                assert peer["calls"][-1]["body"] == b'{"request_id":"retry-1"}'
                assert len(peer["calls"]) == 3


def test_source_task_body_limit_receipt_shape_and_upstream_statuses(monkeypatch):
    with source_task_peer() as (brain_url, peer):
        monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", brain_url)
        monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "brain-write-token")
        for status in (400, 401, 403, 404, 409, 413, 503):
            peer["responses"][f"status-{status}"] = (
                status, {"status": f"upstream-{status}"}, {}
            )
        peer["responses"].update({
            "server-error": (500, {"error": "private-upstream-detail"}, {}),
            "redirect": (307, b"{}", {"Location": "/followed"}),
            "array": (200, b"[]", {}),
        })

        with gateway_loopback(source_task_gateway()) as gateway_url:
            with httpx.Client(base_url=gateway_url, trust_env=False, timeout=3) as client:
                headers = {
                    "Authorization": "Bearer owner-token",
                    "Content-Type": "application/json",
                }
                oversized = client.post(
                    "/api/memory-authority/action-source",
                    content=b"x" * 8193,
                    headers=headers,
                )
                assert oversized.status_code == 413
                assert oversized.json() == {"error": "request_too_large"}
                assert peer["calls"] == []

                for status in (400, 401, 403, 404, 409, 413, 503):
                    before = len(peer["calls"])
                    response = client.post(
                        f"/api/memory-authority/action-source?mode=status-{status}",
                        content=json.dumps({"request_id": f"status-{status}"}),
                        headers=headers,
                    )
                    assert response.status_code == status
                    assert response.json() == {"status": f"upstream-{status}"}
                    assert len(peer["calls"]) == before + 1

                before = len(peer["calls"])
                failed = client.post(
                    "/api/memory-authority/source-tasks/task-1/retry?mode=server-error",
                    content=b'{"request_id":"one-retry-attempt"}',
                    headers=headers,
                )
                assert failed.status_code == 503
                assert failed.json() == {"error": "memory_tasks_unavailable"}
                assert "private-upstream-detail" not in failed.text
                assert len(peer["calls"]) == before + 1

                before = len(peer["calls"])
                redirected = client.post(
                    "/api/memory-authority/action-source?mode=redirect",
                    content=b'{"request_id":"no-follow"}',
                    headers=headers,
                )
                assert redirected.status_code == 503
                assert len(peer["calls"]) == before + 1

                before = len(peer["calls"])
                bad_shape = client.get(
                    "/api/memory-authority/source-tasks?mode=array",
                    headers={"Authorization": "Bearer owner-token"},
                )
                assert bad_shape.status_code == 503
                assert bad_shape.json() == {"error": "memory_tasks_unavailable"}
                assert len(peer["calls"]) == before + 1
