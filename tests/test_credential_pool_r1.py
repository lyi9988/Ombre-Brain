"""Offline Gateway credential-pool tests against AST-extracted production methods.

Only the listed key-selection/retry methods come from gateway.py. Dispatch
resolution, capacity, trace and receipt collaborators are local test doubles;
provider HTTP is always intercepted by httpx.MockTransport.
"""
import ast
import asyncio
import copy
import os
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / "gateway.py"


def _load_gateway_methods():
    tree = ast.parse(GATEWAY.read_text(encoding="utf-8"), filename=str(GATEWAY))
    service_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GatewayService"
    )
    wanted = {
        "_api_key_entries_from_config",
        "_payload_for_upstream_model",
        "_forward_upstream",
        "_available_upstream_api_keys",
        "_should_retry_upstream_status",
    }
    methods = [
        copy.deepcopy(node) for node in service_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in methods} != wanted:
        raise AssertionError(f"GatewayService methods changed: found {sorted(node.name for node in methods)}")
    for method in methods:
        method.decorator_list = []
    retryable = next(
        copy.deepcopy(node) for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RETRYABLE_UPSTREAM_STATUS_CODES"
                for target in node.targets)
    )
    cls = ast.ClassDef(
        name="ExtractedGatewayCredentialMethods", bases=[], keywords=[],
        body=methods or [ast.Pass()], decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[retryable, cls], type_ignores=[]))
    namespace = {
        "Any": Any,
        "copy": copy,
        "deepcopy": copy.deepcopy,
        "httpx": httpx,
        "os": os,
        "time": time,
        "logger": SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        ),
        "_usage_from_httpx_response": lambda response: {},
    }
    exec(compile(module, str(GATEWAY), "exec"), namespace)
    return namespace["ExtractedGatewayCredentialMethods"]


GatewayMethods = _load_gateway_methods()


class _AdoptionStub:
    def begin(self, trace, payload):
        return None

    def response(self, attempt, response, protocol):
        return None


class GatewayCredentialPoolTests(unittest.TestCase):
    def _service(self, handler):
        keys = {
            "PRIMARY_KEY": "synthetic-primary",
            "SECONDARY_KEY": "synthetic-secondary",
        }
        raw = {
            "name": "synthetic-upstream",
            "base_url": "https://mock.invalid/v1",
            "credential_primary_env": "PRIMARY_KEY",
            "api_key_envs": ["PRIMARY_KEY", "SECONDARY_KEY"],
            "api_keys": [{"api_key": "synthetic-inline-backup", "label": "owner-backup"}],
        }
        with patch.dict(os.environ, keys):
            service = GatewayMethods()
            entries = service._api_key_entries_from_config(raw)
        upstream = {**raw, "api_keys": entries}
        route = {"upstream": upstream, "upstream_model": "synthetic-provider-model"}

        service.upstream_key_cooldown_seconds = 0
        service._resolve_upstream_for_model = lambda model: route
        service._capacity_response = lambda *args, **kwargs: None
        service._trace_from_payload = lambda payload: {}
        service.model_request_trace = SimpleNamespace(record_attempt=lambda **kwargs: None)
        service.attachment_adoption = _AdoptionStub()
        service._cool_down_upstream_key = lambda *args, **kwargs: None
        service._clear_upstream_key_cooldown = lambda *args, **kwargs: None
        service.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False)
        return service, entries

    @staticmethod
    def _forward(service, status=200):
        async def run():
            try:
                return await service._forward_upstream({
                    "model": "synthetic-public-model",
                    "messages": [{"role": "user", "content": "synthetic test"}],
                })
            finally:
                await service.http_client.aclose()
        return asyncio.run(run())

    def test_primary_env_pool_precedes_and_preserves_independent_inline_backup(self):
        calls = []
        def provider(request):
            calls.append(request)
            return httpx.Response(200, json={"ok": True})
        service, entries = self._service(provider)
        self.assertEqual([item["label"] for item in entries], [
            "env:PRIMARY_KEY", "env:SECONDARY_KEY", "owner-backup",
        ])
        self.assertEqual([item["value"] for item in entries], [
            "synthetic-primary", "synthetic-secondary", "synthetic-inline-backup",
        ])
        response = self._forward(service)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers["authorization"], "Bearer synthetic-primary")

    def test_auth_failure_on_primary_retries_secondary_once_then_succeeds(self):
        calls = []
        def provider(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(401, json={"error": "synthetic auth failure"})
            return httpx.Response(200, json={"ok": True})
        service, entries = self._service(provider)
        response = self._forward(service)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 2)
        self.assertEqual([r.headers["authorization"] for r in calls], [
            "Bearer synthetic-primary", "Bearer synthetic-secondary",
        ])
        self.assertNotIn("synthetic-inline-backup", [
            r.headers["authorization"] for r in calls])

    def test_success_on_primary_does_not_try_secondary(self):
        calls = []
        service, _ = self._service(lambda request: calls.append(request) or httpx.Response(200))
        response = self._forward(service)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers["authorization"], "Bearer synthetic-primary")

    def test_nonretryable_400_does_not_try_secondary(self):
        calls = []
        service, _ = self._service(lambda request: calls.append(request) or httpx.Response(400))
        self.assertFalse(service._should_retry_upstream_status(400))
        response = self._forward(service)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers["authorization"], "Bearer synthetic-primary")


if __name__ == "__main__":
    unittest.main()
