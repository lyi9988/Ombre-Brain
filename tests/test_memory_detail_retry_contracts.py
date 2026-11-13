"""T02 real preparation, qualification, Composer and serialized provider contracts."""
import asyncio
from copy import deepcopy
import json
from types import MethodType

import httpx
import pytest

from attachment_adoption_support import gateway_service
from memory_delivery import MemoryDeliveryStore, final_fragments
from recall_policy import RecallPolicy
from test_gateway_sdk_tool_phase import minimal_service, async_return
from test_memory_recall_gateway_r2 import (
    _authority_fixture, _gateway_service,
    _natural_candidate, _base_config)
from test_memory_recall_projection_r2 import _make_plan

HTTP_CLIENT = httpx.AsyncClient
RAW = "现住址 DISALLOWED_WHOLE_BUCKET_BODY"


def _response(content):
    return httpx.Response(200, json={"choices": [{"message": {
        "role": "assistant", "content": content}, "finish_reason": "stop"}]})


def _sources(monkeypatch, *, withdrawn=False, withdrawn_ids=()):
    def handler(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/presentations/resolve"):
            return httpx.Response(200, json={"items": [{"memory_id": item["memory_id"],
                "state": "stale", "text": "", "presentation_revision": None} for item in payload["items"]]})
        return httpx.Response(200, json={"items": [{"memory_id": item["memory_id"],
            "revision": item["revision"], "state": "ok", "sources": {
                binding: {"state": "withdrawn" if withdrawn or item["memory_id"] in withdrawn_ids or binding.startswith("unavailable:") else "available",
                    "eligible_source": not withdrawn and item["memory_id"] not in withdrawn_ids and not binding.startswith("unavailable:")}
                for binding in item["binding_ids"]}} for item in payload["items"]]})
    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return HTTP_CLIENT(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.synthetic.invalid")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "synthetic-source-check-token")


def _harness(tmp_path, monkeypatch, *, count=1, operit=False, overlapping_core=False):
    ids = [f"bucket-{index}" for index in range(count)]
    import test_memory_recall_gateway_r2 as recall_fixture
    original_semantic = recall_fixture._semantic_document
    def semantic(*args, **kwargs):
        document, binding = original_semantic(*args, **kwargs)
        missing = deepcopy(document["evidence_bindings"][0])
        missing["binding_id"] = "unavailable:" + binding
        missing["source_id"] += ":withdrawn"
        document["evidence_bindings"].append(missing)
        document["statements"][0]["evidence_refs"] = [missing["binding_id"]]
        return document, binding
    monkeypatch.setattr(recall_fixture, "_semantic_document", semantic)
    authority, view, _, buckets = _authority_fixture(tmp_path, [(bid, RAW) for bid in ids])
    for bucket in buckets:
        bucket["metadata"]["type"] = "source"
    _sources(monkeypatch)
    candidates = [_natural_candidate(bucket, score=.95, keyword_score=.9) for bucket in buckets]
    service, _ = _gateway_service(tmp_path, monkeypatch, view, buckets, candidates, _base_config())
    defaults = minimal_service(tmp_path / "prepare")
    for key, value in defaults.__dict__.items():
        service.__dict__.setdefault(key, value)
    service.identity = {"ai_name": "synthetic"}
    service.inject_total_budget = 16000
    service.retrieval_mode = "bucket"
    service.recalled_budget = 10000
    service.related_memory_budget = 0
    service.memory_detail_recall_budget = 10000
    service.memory_detail_recall_enabled = True
    service.memory_detail_recall_max_ids = 5
    service.recent_budget = 10000
    service.upstream_default_model = "public-model"
    service.persona_engine.enabled = False
    service.core_memory_interval_rounds = 1
    service._should_inject_interval = lambda *args: True
    service._build_core_memory_block = async_return("ORDER_A_CORE")
    service._list_gateway_buckets = async_return(buckets)
    for method in ("_select_dynamic_buckets", "_format_recalled_moments",
                   "_source_record_synthetic_moment_for_bucket", "_inject_context_messages",
                   "_build_memory_recall_projection", "_reconcile_memory_recall_injection",
                   "_rejected_bucket_evidence", "_extract_bucket_ids_from_context",
                   "_dynamic_recall_search_query"):
        service.__dict__.pop(method, None)
    service._dynamic_recall_search_query = lambda query, *args, **kwargs: query
    service._recall_policy.has_strong_score = MethodType(RecallPolicy.has_strong_score, service._recall_policy)
    service._recall_policy.specific_query_terms = lambda query: [str(query)]
    service._route_memory_sentinel = async_return({"route": "ordinary", "route_reason_codes": []})
    service._route_domain_sentinel = async_return({"route": "ordinary"})
    sent = []
    def provider(request):
        sent.append(json.loads(request.content))
        return _response('[memory_detail ids="' + ",".join(ids) + '"]detail needed' if len(sent) == 1 else "done")
    physical, route = gateway_service(provider)
    # The source HTTP fixture patches AsyncClient globally; use its original
    # class so this transport is genuinely the provider's serialized wire.
    asyncio.run(physical.http_client.aclose())
    service.http_client = HTTP_CLIENT(transport=httpx.MockTransport(provider))
    service.gateway_cfg = physical.gateway_cfg
    service.model_request_trace = physical.model_request_trace
    service.attachment_adoption = physical.attachment_adoption
    service._resolve_upstream_for_model = lambda model: route
    service._available_upstream_api_keys = physical._available_upstream_api_keys
    service._clear_upstream_key_cooldown = physical._clear_upstream_key_cooldown
    service.memory_delivery = MemoryDeliveryStore(tmp_path / "delivery.sqlite")
    plan = _make_plan(service, memory_budget=10000)
    payload = {"model": "public-model", "max_tokens": 1000, "messages": [
        {"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "现住址"}],
        "_ombre_trace_context": {"trace_id": "trace-1", "runtime_scope": "chat",
            "conversation_id": "conversation", "user_source_event_id": "user-event",
            "assistant_source_event_id": "assistant-event"}}
    if overlapping_core:
        async def core(buckets_arg):
            from memory_delivery import record_carrier
            record_carrier(buckets[0], "CORE_OVERLAP_OLD_BODY", "core")
            return "ORDER_A_CORE CORE_OVERLAP_OLD_BODY"
        service._build_core_memory_block = core
    if operit:
        service.operit_context_rewrite_enabled = True
        service._rewrite_operit_context_for_forward = lambda messages: (messages,
            "OPERIT_STABLE_MARKER", "OPERIT_ACTIVITY_MARKER", {})
    prepared, selected, debug = asyncio.run(service.prepare_payload(
        payload, "synthetic-session", include_debug=True, prompt_plan=plan))
    assert prepared.get("_ombre_memory_detail_context"), debug
    assert selected, json.dumps(debug, ensure_ascii=False, default=str)
    debug["injected_bucket_ids"] = list(selected)
    service._prepare_memory_delivery(prepared, debug)
    return service, prepared, debug, sent, authority, ids, buckets


def _retry(service, prepared, debug, *, response=None):
    async def run():
        initial = response or await service._forward_upstream(prepared)
        return await service._maybe_retry_with_memory_detail(
            forward_payload=prepared, upstream_response=initial, injection_debug=debug)
    return asyncio.run(run())


def test_partial_qualification_composer_and_new_receipt_match_each_wire(tmp_path, monkeypatch):
    service, prepared, debug, sent, _, ids, _ = _harness(tmp_path, monkeypatch)
    first_manifest = deepcopy(service._trace_from_payload(prepared)["memory_delivery_manifest"]["fragments"])
    response, detail = _retry(service, prepared, debug)
    second_manifest = service._trace_from_payload(prepared)["memory_delivery_manifest"]["fragments"]
    assert response.status_code == 200 and detail["retried"], detail
    assert len(sent) == 2
    for wire in sent:
        text = json.dumps(wire, ensure_ascii=False)
        assert "现住址在白桦街" in text
        assert "旧住址在青石路" not in text and RAW not in text
        assert text.count("CANONICAL-MEMORY-BEGIN") == 1
        assert text.index("ORDER_A_CORE") < text.index("CANONICAL-MEMORY-BEGIN")
        assert "_ombre_memory_detail_context" not in text
    with service.memory_delivery._connect() as db:
        attempts = db.execute("SELECT * FROM memory_delivery_attempts ORDER BY created_at,attempt_id").fetchall()
    assert len(attempts) == 2 and len({row["attempt_id"] for row in attempts}) == 2
    assert all(row["successful"] == 1 for row in attempts)
    for row, wire, manifest in zip(attempts, sent, [first_manifest, second_manifest]):
        fragments = json.loads(row["fragments_json"])
        assert fragments and final_fragments(manifest, wire) == fragments
    assert any(row.get("coverage", {}).get("section") == "targeted"
               for row in json.loads(attempts[-1]["fragments_json"]))
    asyncio.run(service.http_client.aclose())


@pytest.mark.parametrize("change", ["withdrawn", "revision"])
def test_current_source_or_revision_change_after_first_wire_blocks_retry(tmp_path, monkeypatch, change):
    service, prepared, debug, sent, authority, ids, buckets = _harness(tmp_path, monkeypatch)
    first = asyncio.run(service._forward_upstream(prepared))
    if change == "withdrawn":
        monkeypatch.setattr(httpx, "AsyncClient", HTTP_CLIENT)
        _sources(monkeypatch, withdrawn=True)
    else:
        from memory_commit_service import body_sha256
        from memory_l1_contract import l1_content_sha256
        memory_id = "memory:" + ids[0]
        old = authority.get_memory_revision(memory_id, 1)
        metadata = deepcopy(old["metadata"])
        body = "CHANGED_UNVERIFIED_BODY"
        metadata["memory_l1_content_sha256"] = l1_content_sha256(body, metadata["memory_l1"])
        operation = authority.prepare_memory_commit(memory_id=memory_id, bucket_id=ids[0],
            expected_revision=1, body_sha256=body_sha256(body), snapshot_path="synthetic-revision-2.md",
            metadata=metadata, source_refs=old["source_refs"], decision_source="synthetic-revision",
            idempotency_key="synthetic-change", actor="test")
        authority.record_body_written(operation["operation_id"], observed_body_sha256=body_sha256(body))
        authority.finalize_memory_commit(operation["operation_id"])
        assert authority.get_memory(memory_id)["active_revision"] == 2
        # The first-round bucket projection remains at r1 and must not bypass
        # the newly authoritative body/revision.
    response, detail = _retry(service, prepared, debug, response=first)
    assert len(sent) == 1 and not detail["retried"], detail
    assert "CHANGED_UNVERIFIED_BODY" not in json.dumps(sent)
    asyncio.run(service.http_client.aclose())


@pytest.mark.parametrize("gate", ["off", "budget", "no_marker", "stream", "disabled"])
def test_detail_gates_never_add_a_provider_attempt(tmp_path, monkeypatch, gate):
    service, prepared, debug, sent, _, ids, _ = _harness(tmp_path, monkeypatch)
    first = asyncio.run(service._forward_upstream(prepared))
    if gate in ("off", "budget"):
        blocks = prepared["_ombre_memory_detail_context"]["prompt_plan"]["gateway_slice"]["blocks"]
        block = next(row for row in blocks if row["source_id"] == "ombre.memory_recall")
        if gate == "off":
            block.update(mode="off", enabled=False)
        else:
            block["token_budget"] = 0
    elif gate == "stream":
        prepared["stream"] = True
    elif gate == "disabled":
        service.memory_detail_recall_enabled = False
    else:
        first = _response("No detail request.")
    _, detail = _retry(service, prepared, debug, response=first)
    assert len(sent) == 1 and not detail["retried"], detail
    asyncio.run(service.http_client.aclose())


def test_four_qualified_ids_and_explicit_five_zero_limit(tmp_path, monkeypatch):
    service, prepared, debug, sent, _, ids, _ = _harness(tmp_path, monkeypatch, count=4)
    retry, details = asyncio.run(service._prepare_memory_detail_retry(prepared, ids))
    assert retry is not None, details
    fragments = service._trace_from_payload(retry)["memory_delivery_manifest"]["fragments"]
    assert set(ids) <= {row["bucket_id"] for row in fragments}
    with pytest.raises(ValueError, match="max_ids"):
        asyncio.run(service._prepare_memory_detail_retry(prepared, ids + ["fifth", "sixth"]))
    service.memory_detail_recall_max_ids = 0
    with pytest.raises(ValueError, match="max_ids"):
        asyncio.run(service._prepare_memory_detail_retry(prepared, ids))
    asyncio.run(service.http_client.aclose())


def test_operit_context_and_manifest_only_scope_survive_real_second_wire(tmp_path, monkeypatch):
    service, prepared, debug, sent, _, ids, _ = _harness(tmp_path, monkeypatch, operit=True)
    first = asyncio.run(service._forward_upstream(prepared))
    trace = service._trace_from_payload(prepared)
    prior_scope = deepcopy(trace["memory_delivery_manifest"]["scope"])
    for key in ("conversation_id", "user_source_event_id", "assistant_source_event_id"):
        trace.pop(key, None)
    _, detail = _retry(service, prepared, debug, response=first)
    assert detail["retried"], detail
    assert service._trace_from_payload(prepared)["memory_delivery_manifest"]["scope"] == prior_scope
    for wire in sent:
        texts = json.dumps(wire["messages"], ensure_ascii=False)
        assert texts.count("OPERIT_STABLE_MARKER") == 1
        assert texts.count("OPERIT_ACTIVITY_MARKER") == 1
        stable = next(row for row in wire["messages"] if "OPERIT_STABLE_MARKER" in str(row["content"]))
        assert stable["role"] == "system"
    asyncio.run(service.http_client.aclose())


def test_cache_continuation_retry_keeps_exact_tool_protocol_tail(tmp_path, monkeypatch):
    import time
    service, prepared, debug, sent, _, ids, _ = _harness(tmp_path, monkeypatch)
    base = deepcopy(prepared["_ombre_memory_detail_context"]["base_messages"])
    tail = [{"role": "assistant", "content": "", "tool_calls": [{"id": "call-1",
        "type": "function", "function": {"name": "synthetic_read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "TOOL_TAIL_MUST_SURVIVE"}]
    entry = {"prepared_payload": prepared, "base_forward_messages": base,
        "stable_context": "", "dynamic_context": "", "injected_ids": ids,
        "debug_payload": debug, "snapshot_parent_key": "parent", "source_phase": "initial",
        "created_at": time.monotonic()}
    identity = {"baseline_messages": base, "tool_tail": tail, "snapshot_parent_key": "parent",
                "transport_session_id": "synthetic-session", "key_hash": "synthetic"}
    replay = service._prepare_snapshot_replay(entry, identity, deepcopy(prepared),
        continuation_phase=True, started_at=time.perf_counter(), include_debug=True)
    assert replay is not None
    cached, _, _ = replay
    retry, detail = asyncio.run(service._prepare_memory_detail_retry(cached, ids))
    assert retry is not None, detail
    asyncio.run(service._forward_upstream(retry))
    assert sent[-1]["messages"][-2:] == tail
    assert "_ombre_memory_detail_context" not in json.dumps(sent[-1])
    asyncio.run(service.http_client.aclose())


@pytest.mark.parametrize("change", ["withdrawn", "revision"])
def test_overlapping_core_cannot_leak_or_stand_in_for_requalified_target(tmp_path, monkeypatch, change):
    service, prepared, debug, sent, authority, ids, _ = _harness(
        tmp_path, monkeypatch, count=2, overlapping_core=True)
    first = asyncio.run(service._forward_upstream(prepared))
    assert "CORE_OVERLAP_OLD_BODY" in json.dumps(sent[0])
    recipe = prepared["_ombre_memory_detail_context"]
    assert any(row["bucket_id"] == ids[0] and row["section"] == "core"
               for row in recipe["carrier_spans"])
    first_fragments = service._trace_from_payload(prepared)["memory_delivery_manifest"]["fragments"]
    assert any(row["bucket_id"] == ids[0] and row["coverage"]["section"] == "core"
               for row in first_fragments)
    if change == "withdrawn":
        _sources(monkeypatch, withdrawn_ids={"memory:" + ids[0]})
    else:
        from memory_commit_service import body_sha256
        from memory_l1_contract import l1_content_sha256
        memory_id = "memory:" + ids[0]
        old = authority.get_memory_revision(memory_id, 1)
        metadata = deepcopy(old["metadata"])
        body = "NEW_CURRENT_AUTHORITY_BODY"
        metadata["memory_l1_content_sha256"] = l1_content_sha256(body, metadata["memory_l1"])
        operation = authority.prepare_memory_commit(memory_id=memory_id, bucket_id=ids[0],
            expected_revision=1, body_sha256=body_sha256(body), snapshot_path="new-core-r2.md",
            metadata=metadata, source_refs=old["source_refs"], decision_source="synthetic-revision",
            idempotency_key="change-core-owner", actor="test")
        authority.record_body_written(operation["operation_id"], observed_body_sha256=body_sha256(body))
        authority.finalize_memory_commit(operation["operation_id"])
    # The old core text alone cannot turn a failed targeted request into success.
    denied, denied_debug = asyncio.run(service._prepare_memory_detail_retry(prepared, [ids[0]]))
    assert denied is None and denied_debug["skip_reason"] == "detail_not_eligible_or_not_compiled"
    # A second still-valid requested memory makes this a real second wire:
    # the revoked/changed memory must disappear from both core and targeted.
    _, detail = _retry(service, prepared, debug, response=first)
    assert detail["retried"] and len(sent) == 2, detail
    assert ids[0] in detail["missing_ids"]
    assert "CORE_OVERLAP_OLD_BODY" not in json.dumps(sent[1])
    manifest = service._trace_from_payload(prepared)["memory_delivery_manifest"]["fragments"]
    assert not any(row["bucket_id"] == ids[0] for row in manifest)
    assert any(row["bucket_id"] == ids[1] and row["coverage"]["section"] == "targeted"
               for row in manifest)
    asyncio.run(service.http_client.aclose())


def test_retained_partial_prefix_never_promotes_to_complete_coverage():
    from memory_delivery import compiled_candidates, sha
    prefix = "Already clipped semantic prefix."
    original = {"bucket_id": "retained-memory", "text": prefix, "text_sha256": sha(prefix),
        "coverage": {"section": "core", "wire_partial": True, "statement_ids": [], "unit_ids": []}}
    projection = {"compiled_memory_blocks": [{"source_id": "ombre.core_memory", "content": prefix}]}
    result = compiled_candidates(projection, [original])
    assert len(result) == 1 and result[0]["text"] == prefix
    assert result[0]["coverage"]["wire_partial"] is True
    assert result[0]["coverage"]["statement_ids"] == [] and result[0]["coverage"]["unit_ids"] == []
    receipt = final_fragments(result, {"messages": [{"role": "system", "content": prefix}]})
    assert len(receipt) == 1 and receipt[0]["coverage"]["wire_partial"] is True
    assert original["coverage"]["wire_partial"] is True


def test_native_anthropic_response_shape_does_not_trigger_detail_retry(tmp_path, monkeypatch):
    service, prepared, debug, sent, _, ids, _ = _harness(tmp_path, monkeypatch)
    native = httpx.Response(200, json={"type": "message", "role": "assistant",
        "content": [{"type": "text", "text": '[memory_detail ids="bucket-0"]native text'}],
        "stop_reason": "end_turn"})
    response, detail = _retry(service, prepared, debug, response=native)
    assert response is native and not detail["retried"] and not sent
    assert detail["skip_reason"] == "no_assistant_message"
    asyncio.run(service.http_client.aclose())
