"""Real L0/Authority/L2 path, synthetic source IO only; no external calls."""
import asyncio
import copy
import hashlib
import json

import httpx
import pytest

from memory_l1_contract import canonical_json
from test_memory_l2_dependency_scope_r2 import _persist_source_bound_l1, _resolve_item
from test_memory_l2_p4 import run
from test_memory_l2_gateway_contract_p4 import _gateway_runtime, _BrainReadTransport
from test_memory_l2_gateway_contract_p4 import _gateway_service
from memory_l1_reads import MemoryL1Reads
from types import SimpleNamespace


@pytest.mark.parametrize("failure,kind", [
    ("timeout", "timeout"), ("http", "http_status"),
    ("read", "read_error"), ("revision", "identity_mismatch"),
    ("anchor", "anchor_mismatch"), ("missing", "unavailable"),
])
def test_persisted_l2_resolve_distinguishes_failures_without_qualifying_source(tmp_path, failure, kind):
    service, env, calls = _persist_source_bound_l1(tmp_path)
    service.prepare("m1", {"expected_l1_revision": 2, "request_id": "diagnostic"})
    assert run(service.run_once(1))["results"][0]["state"] == "ready"
    before = copy.deepcopy(service.detail("m1")["l2"])
    source = env["sources"]
    adapter = next(iter(source.adapters.values()))
    original = adapter.read
    read_count = []

    async def read(binding):
        read_count.append(binding)
        if failure == "timeout":
            raise asyncio.TimeoutError("PRIVATE_URL")
        if failure == "http":
            request = httpx.Request("GET", "https://private.invalid/?token=PRIVATE_SECRET")
            response = httpx.Response(503, request=request)
            response.raise_for_status()
        if failure == "read":
            raise RuntimeError("PRIVATE_SECRET")
        result = await original(binding)
        if failure == "revision":
            result["source_revision"] = "PRIVATE_CHANGED_REVISION"
        elif failure == "anchor":
            result["text"] = "PRIVATE_CHANGED_BODY"
        elif failure == "missing":
            return {"state": "unavailable", "reason": "source_not_found"}
        return result

    adapter.read = read
    result = run(service.resolve({"items": [_resolve_item("st-a")]}))["items"][0]
    assert result["state"] == "stale" and result["reason"] == "source_dependency_changed"
    assert result["text"] == "" and result["units"] == []
    detail = result["source_diagnostic"]["failures"][0]["detail"]
    assert detail["failure_kind"] == kind
    assert len(read_count) == 1 and len(calls) == 1
    raw = json.dumps(result["source_diagnostic"], ensure_ascii=False)
    for secret in ("PRIVATE_", "source:A", "合成 L0"):
        assert secret not in raw
    if failure == "http":
        assert detail["http_status"] == 503
    if failure == "revision":
        change = detail["field_changes"][0]
        assert change == {
            "field": "source_revision",
            "expected_sha256": hashlib.sha256(canonical_json(read_count[0]["source_revision"]).encode()).hexdigest(),
            "current_sha256": hashlib.sha256(canonical_json("PRIVATE_CHANGED_REVISION").encode()).hexdigest(),
        }
    # A failed read must not mutate/rewrite the persisted owner presentation.
    assert service.detail("m1")["l2"] == before
    adapter.read = original
    recovered = run(service.resolve({"items": [_resolve_item("st-a")]}))["items"][0]
    assert recovered["state"] == "ready" and recovered["text"]
    assert "source_diagnostic" not in recovered and len(calls) == 1


def test_cancel_propagates_without_marking_presentation_stale(tmp_path):
    service, env, calls = _persist_source_bound_l1(tmp_path)
    service.prepare("m1", {"expected_l1_revision": 2, "request_id": "cancel"})
    assert run(service.run_once(1))["results"][0]["state"] == "ready"
    before = copy.deepcopy(service.detail("m1")["l2"])

    async def cancelled(_binding):
        raise asyncio.CancelledError()

    next(iter(env["sources"].adapters.values())).read = cancelled
    with pytest.raises(asyncio.CancelledError):
        run(service.resolve({"items": [_resolve_item("st-a")]}))
    assert service.detail("m1")["l2"] == before and len(calls) == 1


@pytest.mark.parametrize("failure", ["timeout", "revision"])
def test_real_brain_batch_through_gateway_http_keeps_source_diagnostics(tmp_path, monkeypatch, failure):
    runtime = _gateway_runtime(monkeypatch)
    service, env, calls = _persist_source_bound_l1(tmp_path)
    service.prepare("m1", {"expected_l1_revision": 2, "request_id": "http-diagnostic"})
    assert run(service.run_once(1))["results"][0]["state"] == "ready"
    adapter = next(iter(env["sources"].adapters.values()))
    original = adapter.read

    async def read(binding):
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE_URL")
        result = await original(binding)
        result["source_revision"] = "PRIVATE_REVISION"
        return result

    adapter.read = read
    transport = _BrainReadTransport(service, MemoryL1Reads(service.revisions, env["sources"]))
    client = runtime.httpx.AsyncClient
    monkeypatch.setattr(runtime.httpx, "AsyncClient",
                        lambda *args, **kwargs: client(*args, **{**kwargs, "transport": transport}))
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.contract.test")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "synthetic-read-token")
    presentations, traces = run(runtime.resolve_presentations(
        SimpleNamespace(gateway_token="synthetic-read-token"),
        [{"bucket_id": "b1", "memory_id": "m1", "revision": 2,
          "usage": {"items": [{"statement_id": "st-a", "usage": "historical_evidence"}],
                    "narrative_allowed": True,
                    "statement_selection": {"direct_ids": ["st-a"], "support_ids": []}}}],
        request={}))
    assert presentations == {}
    assert len(transport.calls) == 1 and len(calls) == 1
    trace = traces["b1"]
    assert trace["fallback_reason"] == "source_dependency_changed"
    assert trace["source_diagnostic"]["failures"][0]["detail"]["failure_kind"] == (
        "timeout" if failure == "timeout" else "identity_mismatch")
    assert "PRIVATE_" not in json.dumps(traces)


def test_final_authority_rejection_retains_scoped_diagnostic_never_stale_body(tmp_path, monkeypatch):
    runtime = _gateway_runtime(monkeypatch)
    body = "synthetic-current-body"
    metadata = {"memory_id": "m1", "revision": 2,
                "body_sha256": hashlib.sha256(body.encode()).hexdigest()}
    record = {"state": "l1", "memory_id": "m1", "revision": 2}
    reads = []

    def index_metadata(ids):
        reads.append(ids)
        return {"b1": metadata} if len(reads) == 1 else {}

    view = SimpleNamespace(enabled=True, memory_index_metadata_map=index_metadata,
                           memory_l1_recall_map=lambda ids: {"b1": record})
    gateway = _gateway_service(view, "synthetic-read-token")
    request = {"_semantic_records": {"b1": record}, "usage_by_bucket": {
        "b1": {"state": "l1", "items": [{"statement_id": "st-a", "usage": "historical_evidence"}],
               "narrative_allowed": True, "statement_selection": {"direct_ids": ["st-a"], "support_ids": []}}}}

    async def qualified(_service, _request, _query, candidates, **kwargs):
        return candidates

    async def brain_reply(*args, **kwargs):
        return {"status": "ok", "items": [{"memory_id": "m1", "state": "stale", "text": "", "units": [],
            "reason": "source_dependency_changed", "source_diagnostic": {
                "state": "ok", "failures": [{"binding_id_sha256": "a" * 64,
                    "detail": {"failure_kind": "timeout", "body": "PRIVATE_BODY"}}]}}]}

    # Isolate the post-resolver Authority race; real L0 and HTTP paths are
    # covered above. Exercise the complete native project_sections function.
    monkeypatch.setattr(runtime, "verify_sources", qualified)
    monkeypatch.setattr(runtime, "read_batch", brain_reply)
    sections = run(runtime.project_sections(gateway, request, "synthetic question",
        [{"id": "b1", "content": body}],
        {"direct": ("[bucket_id:b1] synthetic-current-body", ["b1"], 4000)}))
    assert sections == {"direct": "", "targeted": "", "diffused": "", "recent": ""}
    trace = request["presentation_trace"][0]
    assert trace["source"] == "none" and trace["presentation_revision"] is None
    assert trace["source_diagnostic_scope"] == "resolver_observation_before_final_eligibility"
    assert trace["source_diagnostic"]["failures"][0]["detail"]["failure_kind"] == "timeout"
    assert "PRIVATE_BODY" not in json.dumps(request["presentation_trace"])
    assert not request["injected_usage_bucket_ids"]
