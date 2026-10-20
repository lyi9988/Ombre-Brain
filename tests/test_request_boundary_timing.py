"""Offline handler/trace boundaries; no provider, key or body observations."""
import asyncio
import json
import sqlite3
import time

from starlette.requests import Request
from starlette.responses import JSONResponse

import gateway
from gateway import GatewayService
from model_request_trace import ModelRequestTraceStore


def test_handler_offsets_and_retry_preparations_use_only_their_own_request(tmp_path, monkeypatch):
    class Clock:
        now = 10.0
        def perf_counter(self):
            return self.now
        def time_ns(self):
            return 123_000_000
    clock = Clock()
    monkeypatch.setattr(gateway, "time", clock)
    service = GatewayService.__new__(GatewayService)
    service.config = {"gateway": {}}
    service.gateway_cfg = {}
    service._authorize = lambda auth: None
    service._maybe_reload_runtime_overlay = lambda: None
    service._memory_configuration_snapshot = lambda apply: {"revision": 1, "values": {}}
    service.recall_policy = {}
    service._session_id_for_request = lambda request: "synthetic-session"
    service._client_label_from_request = lambda request, route: "reality"
    service._trace_context_from_request = lambda request, **kwargs: {
        "trace_id": request.headers["X-Guyan-Trace-Id"], "logical_request_id": "r:model:1",
        "request_ordinal": 1, "retry_reason": request.headers.get("X-Guyan-Model-Retry-Reason", "")}
    service._summarize_messages_for_debug = lambda messages: []
    service._strip_favorite_memory_marker_from_payload = lambda payload: (payload, False)
    service._continuation_phase_enabled = lambda request: False
    service._prompt_plan_for_request = lambda *args, **kwargs: {}
    service._canonical_turn_key = lambda request: "synthetic-canonical"
    service.model_request_trace = ModelRequestTraceStore(tmp_path / "trace.sqlite")
    async def canonical(request, payload, session, query):
        clock.now += 0.25
        return payload, {}
    service._prepare_canonical_turn = canonical
    async def prepare(payload, *args, **kwargs):
        clock.now += 0.6
        return payload, [], {"retrieval_runtime": {"synthetic": True},
                             "memory_detail_recall_debug": {},
                             "prepare_timing_debug": {"total_ms": 600, "steps_ms": {}}}
    service.prepare_payload = prepare
    original_record = service._record_logical_trace
    def record(*args):
        original_record(*args)
        clock.now += 0.05
    service._record_logical_trace = record
    async def upstream(payload, *args, **kwargs):
        service.model_request_trace.record_attempt(trace_id=payload["_ombre_trace_context"]["trace_id"],
            ordinal=1, provider="fixture", upstream="fixture", model="fixture", payload={})
        return JSONResponse({"ok": True})
    service._stream_upstream = upstream
    async def run():
        for retry in (False, True):
            clock.now += 10
            headers = [(b"x-guyan-trace-id", b"same-trace")]
            if retry:
                headers.append((b"x-guyan-model-retry-reason", b"sdk_transport_retry"))
            request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": headers})
            request._json = {"model": "fixture", "stream": True,
                             "messages": [{"role": "user", "content": "PRIVATE_BODY_DO_NOT_OBSERVE"}]}
            response = await service.handle_chat(request)
            assert json.loads(response.body) == {"ok": True}
    asyncio.run(run())
    trace = service.model_request_trace.get("same-trace", view="metadata")
    assert len(trace["preparations"]) == 2
    for preparation in trace["preparations"]:
        timing = preparation["request_boundary_timing"]
        assert timing == {"received_at_unix_ms": 123, "request_parsed_ms": 0,
                          "canonical_enter_ms": 0, "canonical_exit_ms": 250,
                          "prepare_enter_ms": 250, "prepare_exit_ms": 850, "trace_record_done_ms": 900}
        assert len(preparation["attempt_ids"]) == 1
    assert trace["preparations"][0]["attempt_ids"] != trace["preparations"][1]["attempt_ids"]
    assert "PRIVATE_BODY" not in json.dumps(trace)
    assert all(isinstance(value, (int, float)) for value in trace["metadata"]["request_boundary_timing"].values())


def test_boundary_projection_rejects_text_and_does_not_update_previous_prepare(tmp_path):
    store = ModelRequestTraceStore(tmp_path / "trace.sqlite")
    store.begin_logical({"trace_id": "t", "metadata": {"prepare_timing_debug": {"total_ms": 1},
        "request_boundary_timing": {"prepare_enter_ms": 2, "private_body": "NEVER COPY", "prepare_exit_ms": True}}})
    store.update_metadata("t", {"request_boundary_timing": {"trace_record_done_ms": 4, "request_parsed_ms": -1}})
    store.begin_logical({"trace_id": "t", "metadata": {}})
    store.update_metadata("t", {"request_boundary_timing": {"trace_record_done_ms": 999}})
    preparation = store.get("t")["preparations"][0]
    assert preparation["request_boundary_timing"] == {"prepare_enter_ms": 2, "trace_record_done_ms": 4}
    assert "NEVER COPY" not in json.dumps(preparation)


def test_added_exit_observation_does_not_wait_for_busy_trace_store(tmp_path):
    store = ModelRequestTraceStore(tmp_path / "trace.sqlite")
    store.begin_logical({"trace_id": "t", "metadata": {}})
    with sqlite3.connect(store.path) as other_writer:
        other_writer.execute("BEGIN IMMEDIATE")
        started = time.perf_counter()
        store.update_metadata("t", {"request_boundary_timing": {"trace_record_done_ms": 4}}, nonblocking=True)
        assert time.perf_counter() - started < 1
    assert "request_boundary_timing" not in store.get("t")["metadata"]
