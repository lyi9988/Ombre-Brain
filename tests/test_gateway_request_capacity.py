"""Offline final-wire checks: real helpers/trace/receipts, mocked provider HTTP."""
import asyncio
import copy
import json

import httpx
import pytest

from attachment_adoption_support import (attachment_payload, gateway_service, openai_response,
                                         r1_tool_lookup_query, r2_response_lookup_query)
from gateway import GatewayService
from gateway_request_capacity import has_full_attachment, route_limits
from model_request_trace import ModelRequestTraceStore
from request_capacity import HEADER, assess_capacity, encode_header
from starlette.requests import Request


PATHS = ["openai", "anthropic", "openai_stream", "anthropic_stream"]


@pytest.mark.parametrize("path", PATHS)
def test_output_only_overflow_blocks_unknown_image_cost_on_every_wire_path(tmp_path, path):
    service, route, sent = _harness(tmp_path, window=5000, output=1000)
    body = {"model": "public-model", "max_tokens": 6000,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "offline image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]}],
        "_ombre_trace_context": {"trace_id": "trace-1"}}
    original = copy.deepcopy(body)
    async def run():
        response = await _send(service, route, body, path)
        assert response.status_code == 413
        report = response.json()["context_capacity"]
        assert report["reason"] == "output_reservation_exceeds_context_window"
        assert report["reserved_output_tokens"] == 6000
        assert not sent and not service.attachment_adoption.rows
        assert service.model_request_trace.get("trace-1")["attempts"] == []
        assert body["messages"] == original["messages"] and body["max_tokens"] == 6000
        await service.http_client.aclose()
    asyncio.run(run())


async def _send(service, route, payload, path):
    if path == "openai":
        return await service._forward_upstream(payload)
    if path == "anthropic":
        return await service._forward_anthropic_upstream(payload, route)
    if path == "openai_stream":
        return await service._open_upstream_stream(route, payload)
    return await service._open_anthropic_upstream_stream(route, payload)


def _harness(tmp_path, *, window=100000, output=4000):
    sent = []
    def provider(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response(finish_reason="stop"))
    service, route = gateway_service(provider)
    service.gateway_cfg = {"upstreams": [{"name": route["upstream"]["name"],
        "models": [{"id": "public-model", "upstream_model": "upstream-model",
                    "context_window": window, "max_output_tokens": output}]}]}
    route["upstream"].update(protocol="anthropic", anthropic_version="2023-06-01")
    service.model_request_trace = ModelRequestTraceStore(tmp_path / "trace.sqlite")
    service.model_request_trace.begin_logical({"trace_id": "trace-1"})
    return service, route, sent


@pytest.mark.parametrize("path", PATHS)
def test_all_final_helpers_reject_full_schema_before_dispatch_or_adoption(tmp_path, path):
    service, route, sent = _harness(tmp_path, window=6000, output=4000)
    payload, _block, private, _ref = attachment_payload()
    payload["max_tokens"] = 4000
    payload["tools"] = [{"type": "function", "function": {"name": "full_schema",
        "description": "schema-explains-every-field" * 400,
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}}]
    original = copy.deepcopy(payload)
    service.attachment_adoption.begin = lambda *_args: pytest.fail("blocked request began adoption")
    async def run():
        response = await _send(service, route, payload, path)
        assert response.status_code == 413
        body = response.json()
        assert body["error"]["code"] == "context_capacity_exceeded"
        assert body["context_capacity"]["components"]["tools_utf8"] > 5000
        assert body["context_capacity"]["reserved_output_tokens"] == 4000
        assert private not in response.text
        proxy = service._proxy_anthropic_error_response(response)
        assert json.loads(proxy.body)["error"]["code"] == "context_capacity_exceeded"
        assert not sent
        assert not service.attachment_adoption.rows
        assert payload["messages"] == original["messages"]
        assert payload["tools"] == original["tools"]
        assert payload["max_tokens"] == original["max_tokens"]
        trace = service.model_request_trace.get("trace-1")
        assert trace["attempts"] == []
        assert trace["metadata"]["context_capacity"]["state"] == "exceeded"
        assert private not in json.dumps(trace)
        await service.http_client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("path", PATHS)
def test_unknown_chat_available_but_attachment_unknown_stops_and_preserves_body(tmp_path, path):
    service, route, sent = _harness(tmp_path, window=None, output=None)
    service.gateway_cfg = {}
    plain = {"model": "public-model", "messages": [{"role": "user", "content": "normal chat"}],
             "_ombre_trace_context": {"trace_id": "trace-1"}}
    payload, block, _private, _ref = attachment_payload()
    # Legacy caller without either new capacity or adoption metadata is still
    # recognized by its actual attachment body wrapper.
    payload["_ombre_trace_context"].pop("attachment_adoption")
    async def run():
        response = await _send(service, route, plain, path)
        assert response.status_code == 200
        assert plain["_ombre_trace_context"]["context_capacity"]["state"] == "capacity_unverified"
        rejected = await _send(service, route, payload, path)
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "context_capacity_unknown"
        assert payload["messages"][0]["content"] == block
        assert len(sent) == 1
        assert "_ombre_trace_context" not in sent[0]
        assert service.attachment_adoption.lookup(r2_response_lookup_query())["state"] == "unavailable"
        await response.aclose()
        await service.http_client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("path", PATHS)
def test_known_header_alias_dispatches_exact_attachment_and_keeps_output_budget(tmp_path, path):
    service, route, sent = _harness(tmp_path)
    service.gateway_cfg = {}
    payload, block, _private, _ref = attachment_payload()
    payload["max_tokens"] = 7777
    payload["_ombre_trace_context"]["context_capacity_contract"] = {
        "version": 1, "model": "public-model", "context_window": 100000,
        "reserved_output_tokens": 8000, "provider_id": "provider-actual",
        "registry_revision": "owner-applied-7", "capacity_source": "model_registry.applied"}
    async def run():
        response = await _send(service, route, payload, path)
        assert response.status_code == 200
        report = payload["_ombre_trace_context"]["context_capacity"]
        assert report["state"] == "within_capacity"
        assert report["model"] == "upstream-model"
        assert report["registry_revision"] == "owner-applied-7"
        assert report["reserved_output_tokens"] == 8000
        assert sent[0]["max_tokens"] == 7777
        content = sent[0]["messages"][0]["content"]
        assert content == block or content == [{"type": "text", "text": block}]
        assert "_ombre_trace_context" not in sent[0]
        await response.aclose()
        await service.http_client.aclose()
    asyncio.run(run())


def test_explicit_actual_route_config_merges_conservatively_and_preserves_budget(tmp_path):
    service, route, sent = _harness(tmp_path, window=50000, output=8000)
    service.gateway_cfg["upstreams"][0].update(context_window=40000, max_output_tokens=10000)
    payload = {"model": "public-model", "max_completion_tokens": 12000,
        "messages": [{"role": "user", "content": "unchanged"}],
        "_ombre_trace_context": {"trace_id": "trace-1", "context_capacity_contract": {
            "version": 1, "model": "public-model", "context_window": 45000,
            "reserved_output_tokens": 11000, "registry_revision": 9}}}
    async def run():
        response = await service._forward_upstream(payload)
        assert response.status_code == 200
        report = payload["_ombre_trace_context"]["context_capacity"]
        assert report["state"] == "within_capacity"
        assert report["context_window"] == 40000
        assert report["reserved_output_tokens"] == 12000
        assert sent[0]["max_completion_tokens"] == 12000
        assert sent[0]["messages"] == payload["messages"]
        assert "context_capacity" not in sent[0]
        await service.http_client.aclose()
    asyncio.run(run())


def test_unknown_resolves_only_actual_route_explicit_fields_and_model_mismatch_stays_unknown(tmp_path):
    service, route, sent = _harness(tmp_path)
    service.gateway_cfg["upstreams"].insert(0, {"name": "unrelated", "context_window": 1,
        "models": [{"id": "public-model", "context_window": 1}]})
    limits = route_limits(service.gateway_cfg, route)
    assert limits["context_window"] == 100000
    payload, _block, _private, _ref = attachment_payload()
    payload["_ombre_trace_context"]["context_capacity_contract"] = {
        "version": 1, "model": "WRONG_MODEL", "context_window": 200000,
        "reserved_output_tokens": 1}
    async def run():
        response = await service._forward_upstream(payload)
        assert response.status_code == 400
        assert response.json()["context_capacity"]["reason"] == "capacity_model_mismatch"
        assert sent == []
        await service.http_client.aclose()
    asyncio.run(run())


def test_header_metadata_and_owner_config_roundtrip_are_body_free(tmp_path):
    service, route, _sent = _harness(tmp_path)
    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
        "headers": [(HEADER.lower().encode(), encode_header({"model": "public-model",
            "context_window": 40000, "reserved_output_tokens": 5000,
            "provider_id": "provider-actual", "registry_revision": 7}).encode())]})
    trace = service._trace_context_from_request(request, session_id="s", request_type="initial", client_id="reality")
    assert trace["context_capacity_contract"]["context_window"] == 40000
    service.upstreams = [dict(route["upstream"], models=["public-model"],
                              model_map={"public-model": "upstream-model"})]
    owner_payload = service._gateway_upstreams_config_payload()
    assert owner_payload[0]["models"][0]["context_window"] == 100000
    assert owner_payload[0]["models"][0]["max_output_tokens"] == 4000
    safe = service._sanitize_gateway_upstreams_config(owner_payload)
    assert safe[0]["models"][0]["context_window"] == 100000
    assert "provider-secret" not in json.dumps(owner_payload)
    asyncio.run(service.http_client.aclose())


def test_legacy_injection_prefix_cannot_hide_full_attachment_but_mentions_stay_chat():
    payload, block, _private, _ref = attachment_payload()
    payload["messages"][0]["content"] = "<ombre_live_context>memory</ombre_live_context>\nCurrent user message:\n" + block
    assert has_full_attachment(payload, {}) is True
    assert has_full_attachment({"messages": [{"role": "user", "content": 'I saw the key "attachment_source" in documentation'}]}, {}) is False


def test_continuation_recalculates_complete_tail_and_response_schema(tmp_path):
    service, route, sent = _harness(tmp_path, window=8000, output=2000)
    payload = {"model": "public-model", "max_tokens": 2000,
        "messages": [{"role": "user", "content": "hello"}],
        "_ombre_trace_context": {"trace_id": "trace-1"}}
    async def run():
        assert (await service._forward_upstream(payload)).status_code == 200
        payload["messages"].extend([
            {"role": "assistant", "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "complete tool result " * 500}])
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "result", "schema": {"description": "full schema " * 100}}}
        original = copy.deepcopy(payload["messages"])
        response = await service._forward_upstream(payload)
        assert response.status_code == 413
        assert response.json()["context_capacity"]["components"]["response_format_utf8"] > 1000
        assert payload["messages"] == original
        assert len(sent) == 1
        events = service.model_request_trace.get("trace-1")["events"]
        assessed = [item["payload"]["state"] for item in events if item["type"] == "capacity.assessed"]
        assert assessed == ["within_capacity", "exceeded"]
        await service.http_client.aclose()
    asyncio.run(run())


def test_memory_detail_capacity_rejection_is_not_replaced_with_prior_success(tmp_path):
    service, route, sent = _harness(tmp_path, window=5000, output=1000)
    service.memory_detail_recall_enabled = True
    service.memory_detail_recall_max_ids = 1
    service._parse_memory_detail_request = lambda _text: (["bucket-1"], "prior answer")
    async def detail(_ids):
        return "DETAIL_BODY " * 1000, []
    service._build_memory_detail_recall_context = detail
    payload = {"model": "public-model", "max_tokens": 1000,
        "messages": [{"role": "user", "content": "complete question"}],
        "_ombre_trace_context": {"trace_id": "trace-1"}}
    prior = httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "request detail"}}]})
    async def run():
        response, debug = await service._maybe_retry_with_memory_detail(forward_payload=payload,
            upstream_response=prior, injection_debug={"injected_bucket_ids": ["bucket-1"]})
        assert response.status_code == 413
        assert debug["skip_reason"] == "context_capacity_exceeded"
        assert not sent
        assert not service.attachment_adoption.rows
        await service.http_client.aclose()
    asyncio.run(run())


def test_key_retry_reassesses_actual_configuration_before_next_physical_send(tmp_path):
    service, route, sent = _harness(tmp_path, window=30000, output=4000)
    route["upstream"]["api_keys"].append({"label": "second", "value": "second-test-key"})
    attempts = []
    def provider(request):
        attempts.append(json.loads(request.content))
        service.gateway_cfg["upstreams"][0]["models"][0]["context_window"] = 1000
        return httpx.Response(503, json={"error": "synthetic unavailable"})
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    payload = {"model": "public-model", "messages": [{"role": "user", "content": "hello"}],
        "_ombre_trace_context": {"trace_id": "trace-1"}}
    async def run():
        response = await service._forward_upstream(payload)
        assert response.status_code == 413
        assert len(attempts) == 1
        reports = [item["payload"] for item in service.model_request_trace.get("trace-1")["events"]
                   if item["type"] == "capacity.assessed"]
        assert [item["state"] for item in reports] == ["within_capacity", "exceeded"]
        await service.http_client.aclose()
    asyncio.run(run())


def test_memory_detail_block_revokes_only_exact_logical_prior_receipts(tmp_path):
    service, _route, _sent = _harness(tmp_path, window=5000, output=1000)
    service.memory_detail_recall_enabled = True
    service.memory_detail_recall_max_ids = 1
    service._parse_memory_detail_request = lambda _text: (["bucket-1"], "prior answer")
    async def detail(_ids):
        return "DETAIL_BODY " * 1000, []
    service._build_memory_detail_recall_context = detail
    sends = []
    def provider(_request):
        call_id = "call-1" if not sends else "call-other"
        sends.append(call_id)
        return httpx.Response(200, json=openai_response(tool_call_id=call_id))
    payload, _block, _text, _ref = attachment_payload()
    payload["max_tokens"] = 1000
    other = copy.deepcopy(payload)
    other["_ombre_trace_context"]["logical_request_id"] = "logical-other"
    async def run():
        await service.http_client.aclose()
        service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        first = await service._forward_upstream(payload)
        await service._forward_upstream(other)
        assert service.attachment_adoption.lookup(r2_response_lookup_query())["state"] == "available"
        assert service.attachment_adoption.lookup(r1_tool_lookup_query())["state"] == "available"
        incomplete = copy.deepcopy(payload["_ombre_trace_context"])
        incomplete["attachment_adoption"].pop("profile_id")
        service.attachment_adoption.block_dispatch(incomplete, "context_capacity_exceeded")
        assert service.attachment_adoption.lookup(r2_response_lookup_query())["state"] == "available"
        response, _debug = await service._maybe_retry_with_memory_detail(forward_payload=payload,
            upstream_response=first, injection_debug={"injected_bucket_ids": ["bucket-1"]})
        assert response.status_code == 413
        for query in (r2_response_lookup_query(), r1_tool_lookup_query()):
            row = service.attachment_adoption.lookup(query)
            assert row["state"] == "unavailable"
            assert row["reason"] == "logical_dispatch_blocked"
            assert row["prior_physical_attempt_id"].startswith("adopt-")
            assert "attachments" not in row
        assert service.attachment_adoption.lookup(r2_response_lookup_query(logical_request_id="logical-other"))["state"] == "available"
        assert service.attachment_adoption.lookup(r1_tool_lookup_query(tool_call_id="call-other"))["state"] == "available"
        assert len(sends) == 2
        await service.http_client.aclose()
    asyncio.run(run())


def test_real_prepare_selection_sources_and_l2_push_previously_fitting_request_over_capacity(tmp_path, monkeypatch):
    from test_gateway_sdk_tool_phase import minimal_service, async_return
    from test_memory_recall_gateway_r2 import (_authority_fixture, _gateway_service,
        _install_source_http, _natural_candidate, _base_config)
    from test_memory_recall_projection_r2 import _make_plan
    from recall_policy import RecallPolicy

    body = "完整合成来源叙述。" * 140
    l2 = "OWNER_READABLE_L2 独立可读记忆正文。" * 120
    _authority, view, _records, buckets = _authority_fixture(tmp_path, [("bucket-l1", body)])
    buckets[0]["metadata"]["type"] = "source"
    source_calls = _install_source_http(monkeypatch, presentations={"memory:bucket-l1": {
        "state": "ready", "text": l2, "presentation_revision": 4, "units": []}})
    candidates = [_natural_candidate(buckets[0], score=0.95, keyword_score=0.9)]
    service, calls = _gateway_service(tmp_path, monkeypatch, view, buckets, candidates, _base_config())
    defaults = minimal_service(tmp_path / "prepare")
    for key, value in defaults.__dict__.items():
        service.__dict__.setdefault(key, value)
    service.identity = {"ai_name": "顾衍"}
    service.inject_total_budget = 12000
    service.retrieval_mode = "bucket"
    service.recalled_budget = 10000
    service.related_memory_budget = 0
    service.memory_detail_recall_budget = 10000
    service.recent_budget = 10000
    service.upstream_default_model = "public-model"
    service.persona_engine.enabled = False
    service._list_gateway_buckets = async_return(buckets)
    # Candidate provider is offline; selection, source eligibility, L2
    # resolution, real formatter, Composer and physical helper remain real.
    for method in ("_select_dynamic_buckets", "_format_recalled_moments",
                   "_source_record_synthetic_moment_for_bucket", "_inject_context_messages",
                   "_build_memory_recall_projection", "_reconcile_memory_recall_injection",
                   "_rejected_bucket_evidence"):
        service.__dict__.pop(method, None)
    from types import MethodType
    service._recall_policy.has_strong_score = MethodType(RecallPolicy.has_strong_score, service._recall_policy)
    service._recall_policy.specific_query_terms = lambda query: [str(query)]
    service._route_memory_sentinel = async_return({"route": "ordinary", "route_reason_codes": []})
    service._route_domain_sentinel = async_return({"route": "ordinary"})
    physical, route, _unused = _harness(tmp_path / "physical", window=5500, output=1000)
    service.gateway_cfg = physical.gateway_cfg
    service.http_client = physical.http_client
    service.model_request_trace = physical.model_request_trace
    service.attachment_adoption = physical.attachment_adoption
    service._resolve_upstream_for_model = lambda _model: route
    service._available_upstream_api_keys = physical._available_upstream_api_keys
    plan = _make_plan(service, memory_budget=10000)
    payload = {"model": "public-model", "max_tokens": 1000,
        "messages": [{"role": "system", "content": "SYSTEM"},
                     {"role": "user", "content": "完整合成来源叙述"}],
        "_ombre_trace_context": {"trace_id": "trace-1", "runtime_scope": "chat"}}
    before = assess_capacity({key: value for key, value in payload.items() if not key.startswith("_")},
                             {"context_window": 5500, "reserved_output_tokens": 1000})
    assert before["state"] == "within_capacity"
    async def run():
        prepared, _ids, debug = await service.prepare_payload(payload, "synthetic-session",
            include_debug=True, prompt_plan=plan)
        text = json.dumps(prepared["messages"], ensure_ascii=False)
        assert l2 in text, {"calls": calls, "source_calls": list(source_calls),
                             "l2_calls": source_calls.presentation_requests, "debug": debug}
        assert calls["candidate"]
        assert source_calls and source_calls.presentation_requests
        response = await service._forward_upstream(prepared)
        assert response.status_code == 413
        assert response.json()["context_capacity"]["state"] == "exceeded"
        assert not _unused
        assert prepared["max_tokens"] == 1000
        assert l2 in json.dumps(prepared["messages"], ensure_ascii=False)
        assert not service.attachment_adoption.rows
        evidence = {"before_gateway_injection": before, "after_gateway_injection": response.json()["context_capacity"],
            "selected_candidate_count": len(calls["selected"]), "source_check_http_count": len(source_calls),
            "presentation_resolve_http_count": len(source_calls.presentation_requests),
            "l2_presentation_revision": 4, "complete_l2_in_final_messages": True,
            "physical_provider_send_count": len(_unused), "attachment_receipt_count": len(service.attachment_adoption.rows),
            "actual_max_tokens_unchanged": prepared["max_tokens"] == payload["max_tokens"]}
        (tmp_path / "capacity-chain-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        await service.http_client.aclose()
    asyncio.run(run())
