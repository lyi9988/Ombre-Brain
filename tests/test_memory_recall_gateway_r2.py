"""Gateway recall integration contracts over temporary Authority SQL and mocked HTTP."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from gateway import GatewayService, create_gateway_app, memory_use
from memory_authority import MemoryAuthorityStore
from memory_authority_view import MemoryAuthorityRecallView
from memory_commit_service import body_sha256
from memory_l1_contract import l1_content_sha256, make_text_anchor, semantic_sha256
from memory_l1_schema import document_template, split_document, statement_template, unknown
from memory_recall_semantics import preview_pipeline, score_document
from memory_system_config import values_from_config
from prompt_plan_mirror import PromptPlanMirrorStore
from starlette.responses import JSONResponse
from test_recall_natural_r1 import _natural_finish_service


FIXED_NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)


def _typed(value):
    return unknown("known", value)


def _statement(statement_id, content, binding_id, *, kind="fact", topics=(),
               stability="phase", start="2025-01-01", end=None):
    item = statement_template()
    item.update(
        statement_id=statement_id,
        statement_revision_id=f"revision:{statement_id}",
        content=content,
        subject_ref=_typed("entity:user"),
        evidence_refs=[binding_id],
    )
    item["classification"].update(
        kind=_typed(kind), topics=list(topics), stability=_typed(stability),
        modality=_typed("assertion"), basis=_typed("record"),
    )
    item["conditions"] = {
        "state": "not_applicable", "text": None, "predicate": None,
        "exceptions": [], "evidence_refs": [],
    }
    item["time"]["event"].update(
        start=_typed(start), end=unknown("not_stated"),
    )
    applicability = item["time"]["applicability"]
    applicability.update(
        temporal_scope="time_range", start=_typed(start),
        end=(_typed(end) if end else unknown("unbounded", evidence_refs=[binding_id])),
        recurrence_text=None,
    )
    return item


def _semantic_document(bucket_id, body, *, current="现住址在白桦街。", old="旧住址在青石路。",
                       topic="housing", profile_kind="fact", profile_stability="phase"):
    binding_id = f"source:{bucket_id}"
    source_text = f"授权合成来源 {bucket_id}。"
    catalog = {"topics": [topic]}
    document = document_template(
        body, f"合成记录 {bucket_id}", catalog=catalog,
        catalog_revision="gateway-recall-r2",
    )
    document["entities"] = [{
        "entity_id": "entity:user", "entity_type": _typed("user"),
        "name": "主人", "aliases": [],
    }]
    document["retrieval"].update(keywords=["现住址"], tags=["housing"])
    document["evidence_bindings"] = [{
        "binding_id": binding_id,
        "source_namespace": "snapshot.synthetic.diary",
        "source_id": f"source-id:{bucket_id}",
        "source_revision": 1,
        "author_ref": None,
        "speaker_ref": None,
        "anchors": [make_text_anchor(
            source_text, target_namespace="l0", target_id=f"source-id:{bucket_id}",
            target_revision=1, text_view_id="synthetic.exact-v1",
            start=0, end=len(source_text),
        )],
    }]
    document["statements"] = [
        _statement(
            f"statement:{bucket_id}:old", old, binding_id, kind="fact", topics=[topic],
            stability="phase", start="2019-01-01", end="2024-01-01",
        ),
        _statement(
            f"statement:{bucket_id}:current", current, binding_id,
            kind=profile_kind, topics=[topic], stability=profile_stability, start="2025-01-01",
        ),
    ]
    return split_document(document)[1], binding_id


def _commit_l1(authority, bucket_id, body, *, topic="housing", current=None,
               old=None, profile_kind="fact", profile_stability="phase"):
    semantic, binding_id = _semantic_document(
        bucket_id, body, topic=topic, current=current or "现住址在白桦街。",
        old=old or "旧住址在青石路。", profile_kind=profile_kind,
        profile_stability=profile_stability,
    )
    metadata = {
        "name": f"合成记录 {bucket_id}",
        "memory_l1": semantic,
        "memory_l1_semantic_sha256": semantic_sha256(semantic),
        "memory_l1_content_sha256": l1_content_sha256(body, semantic),
    }
    prepared = authority.prepare_memory_commit(
        memory_id=f"memory:{bucket_id}", bucket_id=bucket_id,
        expected_revision=0, body_sha256=body_sha256(body),
        snapshot_path=f"revisions/memory:{bucket_id}/1.md", metadata=metadata,
        source_refs=[f"synthetic:{bucket_id}"], decision_source="gateway-test",
        idempotency_key=f"gateway-recall:{bucket_id}", actor="test",
    )
    authority.record_body_written(
        prepared["operation_id"], observed_body_sha256=body_sha256(body),
    )
    authority.finalize_memory_commit(prepared["operation_id"])
    return {"memory_id": f"memory:{bucket_id}", "revision": 1,
            "bucket_id": bucket_id, "binding_id": binding_id, "document": semantic}


def _base_config(*, mode="fast_local", local_threshold=0.5):
    values = values_from_config({})
    values["features"].append({
        "id": "keyword_score", "label": "词法分数", "value_type": "number",
        "source": "runtime", "missing": "unknown", "input_key": "keyword_score",
        "expression": "", "description": "synthetic runtime input",
        "minimum": 0, "maximum": 1,
    })
    values["formulas"].append({
        "id": "keyword_rank", "label": "词法排序",
        "expression": "keyword_score", "weights": {},
    })
    values["recall"].update(
        mode=mode, formula_id="keyword_rank", local_threshold=local_threshold,
        semantic_threshold=1.0, rerank_threshold=1.0,
    )
    return values


def _scoring_profile_config():
    values = _base_config(local_threshold=0.5)
    values["features"].append({
        "id": "base_relevance", "label": "基础相关性", "value_type": "number",
        "source": "runtime", "missing": "unknown", "input_key": "base_relevance",
        "expression": "", "description": "synthetic base score",
        "minimum": 0, "maximum": 1,
    })
    formulas = [
        ("current_preference", "max(base_relevance, keyword_score)"),
        ("current_fact", "min(base_relevance, keyword_score)"),
        ("history_preference", "min(base_relevance, keyword_score)"),
        ("history_fact", "keyword_score"),
    ]
    values["formulas"].extend({"id": ident, "expression": expression, "weights": {}}
                              for ident, expression in formulas)
    values["recall"]["formula_id"] = None
    values["recall"]["scoring_profiles"] = [
        {"id": "current_preference_profile", "selector": {
            "query_purpose": "current", "kind": "preference",
            "topic": "housing", "stability": "stable",
        }, "formula_id": "current_preference"},
        {"id": "current_fact_profile", "selector": {
            "query_purpose": "current", "kind": "fact",
            "topic": "housing", "stability": "phase",
        }, "formula_id": "current_fact"},
        {"id": "history_preference_profile", "selector": {
            "query_purpose": "history", "kind": "preference",
            "topic": "housing", "stability": "stable",
        }, "formula_id": "history_preference"},
        {"id": "history_fact_profile", "selector": {
            "query_purpose": "history", "kind": "fact",
            "topic": "housing", "stability": "phase",
        }, "formula_id": "history_fact"},
    ]
    return values


def _freshness_config():
    values = values_from_config({})
    values["time_profiles"] = [{
        "id": "fresh_current_fact",
        "selector": {"kind": "fact", "query_purpose": "current"},
        "function": "none", "weight": 1.0,
    }]
    values["recall"].update(
        mode="fast_local", formula_id=None, local_threshold=0.5,
        semantic_threshold=1.0, rerank_threshold=1.0,
    )
    return values


def _install_source_http(monkeypatch, *, state="available", presentations=None):
    class RequestCalls(list):
        pass

    requests = RequestCalls()
    presentation_requests = []
    presentations = presentations or {}
    original_client = httpx.AsyncClient

    def handler(request):
        payload = json.loads(request.content)
        call = {
            "url": str(request.url), "authorization": request.headers.get("Authorization"),
            "payload": payload,
        }
        if request.url.path.endswith("/presentations/resolve"):
            presentation_requests.append(call)
            rows = []
            for item in payload["items"]:
                resolution = presentations.get(item["memory_id"], {
                    "state": "stale", "text": "", "presentation_revision": None,
                    "reason": "presentation_missing",
                })
                rows.append({"memory_id": item["memory_id"], **resolution})
            return httpx.Response(200, json={"status": "ok", "items": rows})
        requests.append(call)
        rows = []
        for item in payload["items"]:
            sources = {
                binding_id: {
                    "state": state,
                    "eligible_source": state == "available",
                    "reason": None if state == "available" else "synthetic_source_withdrawal",
                }
                for binding_id in item["binding_ids"]
            }
            rows.append({
                "memory_id": item["memory_id"], "revision": item["revision"],
                "state": "ok", "sources": sources,
            })
        return httpx.Response(200, json={"items": rows})

    transport = httpx.MockTransport(handler)

    def mock_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(memory_use.httpx, "AsyncClient", mock_client)
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.synthetic.invalid")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "synthetic-source-check-token")
    requests.presentation_requests = presentation_requests
    return requests


def _natural_candidate(bucket, *, score, keyword_score):
    return {
        "bucket": bucket, "score": score, "semantic_score": 0.0,
        "keyword_score": keyword_score, "natural_semantic": {},
        "_test_labels": ["keyword_match"],
    }


def _gateway_service(tmp_path, monkeypatch, view, buckets, candidates, config):
    service, calls, _metadata = _natural_finish_service(
        monkeypatch, buckets, {"*": candidates},
    )
    service.memory_authority_view = view
    service.memory_authority_enabled = True
    service.gateway_tz = timezone.utc
    service.config = {
        "state_dir": str(tmp_path),
        "memory_authority": {"enabled": True},
        "memory_system": copy.deepcopy(config),
    }
    service.gateway_token = "synthetic-source-check-token"
    service.first_card_min_score = 0.55
    service.second_card_min_score = 0.5
    service.second_card_relative_score = 0.85
    service.automatic_recall_enabled = True
    service._query_date_recall_hint = lambda _query: None
    return service, calls


def _recall_input(query="现在住在哪里？"):
    return {
        "q_current": query, "q_context": query,
        "metadata": {"query_views": []},
        "_use_context": {
            "now": FIXED_NOW, "target_time": None, "query": query,
            "purpose": "current", "overview": False,
            "facts": {"hour": {"state": "known", "value": 12},
                      "weekday": {"state": "known", "value": 5}},
        },
    }


def _authority_fixture(tmp_path, rows):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})
    records, buckets = {}, []
    for bucket_id, body in rows:
        record = _commit_l1(authority, bucket_id, body)
        records[bucket_id] = record
        buckets.append({"id": bucket_id, "content": body, "metadata": {"name": bucket_id}})
    view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path), "memory_authority": {"enabled": True},
    })
    return authority, view, records, buckets


def test_fast_local_uses_configured_formula_and_threshold_in_real_selection_and_projection(tmp_path, monkeypatch):
    rows = [
        ("bucket-low-keyword", "低词法命中的合成记忆。"),
        ("bucket-high-keyword", "高词法命中的合成记忆。"),
    ]
    _authority, view, records, buckets = _authority_fixture(tmp_path, rows)
    candidates = [
        _natural_candidate(buckets[0], score=0.95, keyword_score=0.2),
        _natural_candidate(buckets[1], score=0.2, keyword_score=0.8),
    ]
    source_requests = _install_source_http(monkeypatch)
    config = _base_config(local_threshold=0.5)
    service, calls = _gateway_service(tmp_path, monkeypatch, view, buckets, candidates, config)
    request = _recall_input()

    selected, _suppressed, debug = asyncio.run(service._finish_natural_selection(
        request["q_context"], "session:gateway-recall-r2", buckets,
        recall_input=request,
    ))

    assert calls["candidate_options"][-1]["allow_semantic"] is False
    assert calls["rerank"] == []
    assert [item["id"] for item in selected] == ["bucket-high-keyword"]
    assert request["selected_buckets"] == ["bucket-high-keyword"]
    score = debug["configured_computation"]["scores"][0]["computation"]
    assert score["formula_id"] == "keyword_rank"
    assert score["score"] == pytest.approx(0.8)
    assert len(source_requests) == 1
    assert source_requests[0]["authorization"] == "Bearer synthetic-source-check-token"
    assert source_requests[0]["url"].endswith("/api/memory-authority/recall-sources")

    raw = "[bucket_id:bucket-high-keyword] 原始故事仍保留作叙述背景。"
    projected = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-high-keyword"], 4000),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))["direct"]
    assert "current_basis" in projected
    assert "historical_evidence" in projected
    assert "原始故事仍保留作叙述背景" in projected
    assert request["injected_usage_bucket_ids"] == ["bucket-high-keyword"]
    final = service._build_memory_recall_projection(
        recalled_memory=projected, targeted_memory_detail="", related_memory="",
        recalled_moments=[], targeted_memory_detail_debug={},
        memory_sentinel_debug={"route": "natural"}, natural_input=request,
    )
    assert final["recall_status"]["status"] == "selected"
    assert final["selected_bucket_ids"] == ["bucket-high-keyword"]
    assert "historical_evidence" in final["fact_body"]
    assert "current_basis" in final["fact_body"]

    stricter_config = _base_config(local_threshold=0.9)
    stricter, stricter_calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, candidates, stricter_config,
    )
    stricter_input = _recall_input()
    no_selection, _suppressed, strict_debug = asyncio.run(stricter._finish_natural_selection(
        stricter_input["q_context"], "session:gateway-recall-r2-strict", buckets,
        recall_input=stricter_input,
    ))
    assert no_selection == []
    assert strict_debug["final_bucket_ids"] == []
    assert stricter_calls["candidate_options"][-1]["allow_semantic"] is False
    assert stricter_calls["rerank"] == []


def test_freshness_profile_does_not_lift_a_low_base_relevance_above_its_base(tmp_path, monkeypatch):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})
    body = "新近记录，但对当前问题相关性很低。"
    _commit_l1(authority, "bucket-fresh-low", body)
    view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path), "memory_authority": {"enabled": True},
    })
    bucket = {"id": "bucket-fresh-low", "content": body, "metadata": {"name": "fresh-low"}}
    candidate = _natural_candidate(bucket, score=0.2, keyword_score=0.9)
    _install_source_http(monkeypatch)
    service, calls = _gateway_service(
        tmp_path, monkeypatch, view, [bucket], [candidate], _freshness_config(),
    )
    request = _recall_input()

    selected, suppressed, _debug = asyncio.run(service._finish_natural_selection(
        request["q_context"], "session:freshness-contract", [bucket],
        recall_input=request,
    ))

    assert selected == []
    rejected = next(item for item in suppressed if item["bucket"]["id"] == bucket["id"])
    score = rejected["_configured_score"]
    assert score["statements"][0]["features"]["freshness"] == 1.0
    assert score["score"] == pytest.approx(0.2)
    assert score["score"] <= rejected["_raw_base_score"]
    assert calls["candidate_options"][-1]["allow_semantic"] is False
    assert calls["rerank"] == []

    preview = preview_pipeline(_freshness_config(), {"relevance": 0.2}, {
        "now": "2026-10-17T12:00:00Z",
        "event_time": "2026-10-17T12:00:00Z",
        "purpose": "current", "kind": "fact",
    })
    assert preview["provider_requests"] == 0 and preview["saved"] is False
    assert preview["pipeline"]["score"] == pytest.approx(0.2)
    assert preview["pipeline"]["statements"][0]["features"]["time_factor"] == pytest.approx(1.0)


def test_scoring_profiles_change_real_candidate_choice_by_purpose_kind_topic_and_stability(
    tmp_path, monkeypatch,
):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})
    preference_body = "主人偏好与住所相关的安排。"
    fact_body = "当前住所记录。"
    _commit_l1(
        authority, "bucket-preference", preference_body,
        profile_kind="preference", profile_stability="stable",
    )
    _commit_l1(authority, "bucket-fact", fact_body, profile_kind="fact")
    view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path), "memory_authority": {"enabled": True},
    })
    buckets = [
        {"id": "bucket-preference", "content": preference_body, "metadata": {"name": "preference"}},
        {"id": "bucket-fact", "content": fact_body, "metadata": {"name": "fact"}},
    ]
    candidates = [
        _natural_candidate(buckets[0], score=0.95, keyword_score=0.05),
        _natural_candidate(buckets[1], score=0.3, keyword_score=0.85),
    ]
    _install_source_http(monkeypatch)
    config = _scoring_profile_config()

    selected_by_purpose = {}
    for purpose in ("current", "history"):
        service, calls = _gateway_service(tmp_path, monkeypatch, view, buckets, candidates, config)
        request = _recall_input(
            "现在住在哪里？" if purpose == "current" else "回想过去住在哪里？",
        )
        request["_use_context"]["purpose"] = purpose
        selected, _suppressed, debug = asyncio.run(service._finish_natural_selection(
            request["q_context"], f"session:scoring-profile:{purpose}", buckets,
            recall_input=request,
        ))
        selected_by_purpose[purpose] = [item["id"] for item in selected]
        assert calls["candidate_options"][-1]["allow_semantic"] is False
        assert calls["rerank"] == []
        expected_bucket = "bucket-preference" if purpose == "current" else "bucket-fact"
        assert selected_by_purpose[purpose] == [expected_bucket]
        selected_score = next(
            row["computation"] for row in debug["configured_computation"]["scores"]
            if row["bucket_id"] == expected_bucket
        )
        rows = selected_score["statements"]
        if purpose == "current":
            current = next(row for row in rows if row["statement_id"] == "statement:bucket-preference:current")
            assert current["scoring_profile"] == "current_preference_profile"
            assert current["formula_id"] == "current_preference"
        else:
            current = next(row for row in rows if row["statement_id"] == "statement:bucket-fact:current")
            assert current["scoring_profile"] == "history_fact_profile"
            assert current["formula_id"] == "history_fact"

    assert selected_by_purpose == {
        "current": ["bucket-preference"],
        "history": ["bucket-fact"],
    }


def test_withdrawn_source_cannot_reintroduce_the_old_raw_bucket_body(tmp_path, monkeypatch):
    _authority, view, _records, buckets = _authority_fixture(
        tmp_path, [("bucket-revoked", "撤回后不应注入的原始故事正文。")],
    )
    source_requests = _install_source_http(
        monkeypatch, state="withdrawn",
        presentations={"memory:bucket-revoked": {
            "state": "ready", "text": "Must not be read", "presentation_revision": 9,
        }},
    )
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    request = _recall_input()
    request["selected_buckets"] = ["bucket-revoked"]
    raw = "[bucket_id:bucket-revoked] 撤回后不应注入的原始故事正文。"

    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-revoked"], 4000),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))
    assert sections["direct"] == ""
    assert source_requests.presentation_requests == []
    assert request["injected_usage_bucket_ids"] == []
    final = service._build_memory_recall_projection(
        recalled_memory=sections["direct"], targeted_memory_detail="", related_memory="",
        recalled_moments=[], targeted_memory_detail_debug={},
        memory_sentinel_debug={"route": "natural"}, natural_input=request,
    )
    assert final["recall_status"]["injected_count"] == 0
    assert "撤回后不应注入的原始故事正文" not in final["fact_body"]
    assert "撤回后不应注入的原始故事正文" not in final["body"]


def test_usage_card_off_source_does_not_fall_back_to_l1_or_ready_l2(tmp_path, monkeypatch):
    _authority, view, _records, buckets = _authority_fixture(
        tmp_path, [("bucket-prompt-off", "主人关闭卡片后不得回退的原始故事。")],
    )
    source_requests = _install_source_http(monkeypatch, presentations={
        "memory:bucket-prompt-off": {
            "state": "ready", "text": "Ready L2 must remain suppressed",
            "presentation_revision": 4,
        },
    })
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    mirror = PromptPlanMirrorStore(tmp_path / "prompt-plan.sqlite3")
    plan_sha = "a" * 64
    mirror.put_plan("preset:usage-card-off", 1, plan_sha256=plan_sha, gateway_slice={
        "blocks": [{
            "block_id": "block:usage-card-off",
            "source_id": "ombre.memory_usage_card_prompt",
            "scope": "memory.usage_card",
            "stage": "ombre_post_injection",
            "mode": "off",
            "role": "system",
            "lane": "instruction",
            "anchor": "instructions.end",
            "order": 1,
            "enabled": False,
        }],
    })
    mirror.put_binding(
        "memory.usage_card", preset_id="preset:usage-card-off",
        preset_revision=1, plan_sha256=plan_sha, aiz_binding_revision=1,
    )
    service.prompt_plan_mirror = mirror
    request = _recall_input()
    request["selected_buckets"] = ["bucket-prompt-off"]
    raw = "[bucket_id:bucket-prompt-off] 主人关闭卡片后不得回退的原始故事。"

    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-prompt-off"], 4000),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))
    assert sections["direct"] == ""
    assert len(source_requests.presentation_requests) == 1
    assert request["injected_usage_bucket_ids"] == []
    trace = request["presentation_trace"][0]
    assert trace["source"] == "none"
    assert trace["presentation_revision"] is None
    assert trace["fallback_reason"] == "usage_card_prompt_unavailable"
    final = service._build_memory_recall_projection(
        recalled_memory=sections["direct"], targeted_memory_detail="", related_memory="",
        recalled_moments=[], targeted_memory_detail_debug={},
        memory_sentinel_debug={"route": "natural"}, natural_input=request,
    )
    assert "主人关闭卡片后不得回退的原始故事" not in final["fact_body"]
    assert "主人关闭卡片后不得回退的原始故事" not in final["body"]
    assert "Ready L2 must remain suppressed" not in final["body"]


def test_ready_l2_is_batch_resolved_and_used_with_per_statement_use_constraints(tmp_path, monkeypatch):
    _authority, view, _records, buckets = _authority_fixture(
        tmp_path, [("bucket-l2-ready", "L1 raw original sentinel; full source narrative.")],
    )
    source_requests = _install_source_http(monkeypatch, presentations={
        "memory:bucket-l2-ready": {
            "state": "ready", "text": "L2 coherent presentation sentinel.",
            "presentation_revision": 7, "reason": None,
        },
    })
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    request = _recall_input()
    raw = "[bucket_id:bucket-l2-ready] L1 raw original sentinel; full source narrative."

    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-l2-ready"], 4000),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))

    assert len(source_requests.presentation_requests) == 1
    call = source_requests.presentation_requests[0]
    assert call["url"].endswith("/api/memory-authority/presentations/resolve")
    assert call["authorization"] == "Bearer synthetic-source-check-token"
    item = call["payload"]["items"][0]
    expected_statement_uses = {
        use["statement_id"]: use["usage"]
        for use in request["usage_by_bucket"]["bucket-l2-ready"]["items"]
    }
    assert item["statement_uses"] == expected_statement_uses
    assert set(item["allowed_uses"]) == set(expected_statement_uses.values())
    assert item["narrative_allowed"] is True
    assert "current_basis" in sections["direct"]
    assert "historical_evidence" in sections["direct"]
    assert "L2 coherent presentation sentinel." in sections["direct"]
    assert "L1 raw original sentinel" not in sections["direct"]
    trace = request["presentation_trace"][0]
    assert trace["source"] == "l2"
    assert trace["presentation_revision"] == 7
    assert trace["fallback_reason"] is None

    projection = service._build_memory_recall_projection(
        recalled_memory=sections["direct"], targeted_memory_detail="", related_memory="",
        recalled_moments=[], targeted_memory_detail_debug={},
        memory_sentinel_debug={"route": "natural"}, natural_input=request,
    )
    assert projection["presentation_trace"] == request["presentation_trace"]


def test_stale_l2_falls_back_to_l1_and_records_exact_reason(tmp_path, monkeypatch):
    _authority, view, _records, buckets = _authority_fixture(
        tmp_path, [("bucket-l2-stale", "L1 fallback narrative sentinel.")],
    )
    source_requests = _install_source_http(monkeypatch, presentations={
        "memory:bucket-l2-stale": {
            "state": "stale", "text": "Stale L2 must not be used",
            "presentation_revision": None, "reason": "dependency_version_stale",
        },
    })
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    request = _recall_input()
    raw = "[bucket_id:bucket-l2-stale] L1 fallback narrative sentinel."

    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-l2-stale"], 4000),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))

    assert len(source_requests.presentation_requests) == 1
    assert "L1 fallback narrative sentinel." in sections["direct"]
    assert "Stale L2 must not be used" not in sections["direct"]
    trace = request["presentation_trace"][0]
    assert trace["resolver_state"] == "stale"
    assert trace["source"] == "l1"
    assert trace["presentation_revision"] is None
    assert trace["fallback_reason"] == "dependency_version_stale"


def test_l2_is_kept_as_a_whole_unit_when_it_exceeds_the_remaining_budget(tmp_path, monkeypatch):
    _authority, view, _records, buckets = _authority_fixture(
        tmp_path, [("bucket-l2-large", "Short L1 fallback with its condition preserved.")],
    )
    _install_source_http(monkeypatch, presentations={
        "memory:bucket-l2-large": {
            "state": "ready",
            "text": "L2 complete condition unit and its exception. " * 120,
            "presentation_revision": 8, "reason": None,
        },
    })
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    request = _recall_input()
    raw = "[bucket_id:bucket-l2-large] Short L1 fallback with its condition preserved."

    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-l2-large"], 400),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))

    assert "Short L1 fallback with its condition preserved." in sections["direct"]
    assert "L2 complete condition unit" not in sections["direct"]
    trace = request["presentation_trace"][0]
    assert trace["source"] == "l1"
    assert trace["presentation_revision"] is None
    assert trace["fallback_reason"] == "presentation_exceeds_remaining_budget"


def test_large_l2_selects_complete_units_that_fit_the_existing_budget(tmp_path, monkeypatch):
    _authority, view, _records, buckets = _authority_fixture(
        tmp_path, [("bucket-l2-units", "L1 fallback must not replace a fitting L2 unit.")],
    )
    current_unit = (
        "Complete current unit: the arrangement applies only under its stated condition, "
        "with the listed exception preserved."
    )
    oversized_unit = "Complete historical unit with all conditions. " * 120
    _install_source_http(monkeypatch, presentations={
        "memory:bucket-l2-units": {
            "state": "ready",
            "text": "Full preferred L2 narrative. " * 120,
            "presentation_revision": 9,
            "reason": None,
            "units": [
                {"unit_id": "unit:current", "statement_refs": [
                    "statement:bucket-l2-units:current",
                ], "text": current_unit},
                {"unit_id": "unit:historical", "statement_refs": [
                    "statement:bucket-l2-units:old",
                ], "text": oversized_unit},
            ],
        },
    })
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    request = _recall_input()
    raw = "[bucket_id:bucket-l2-units] L1 fallback must not replace a fitting L2 unit."

    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-l2-units"], 400),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))

    assert current_unit in sections["direct"]
    assert oversized_unit not in sections["direct"]
    assert "Full preferred L2 narrative." not in sections["direct"]
    assert "L1 fallback must not replace a fitting L2 unit." not in sections["direct"]
    trace = request["presentation_trace"][0]
    assert trace["source"] == "l2"
    assert trace["presentation_revision"] == 9
    assert trace["selected_unit_ids"] == ["unit:current"]
    assert trace["partial_presentation"] is True
    assert trace["fallback_reason"] is None


def test_real_configuration_preview_route_accepts_scenario_and_legacy_payload_without_providers(tmp_path):
    values = _freshness_config()
    service = GatewayService.__new__(GatewayService)
    service.config = {"memory_system": copy.deepcopy(values)}
    service.gateway_tz = timezone.utc
    service._authorize = lambda authorization: (
        None if authorization == "Bearer owner"
        else JSONResponse({"error": "unauthorized"}, status_code=401)
    )
    provider_calls = []

    async def unexpected_provider(*_args, **_kwargs):
        provider_calls.append("called")
        raise AssertionError("configuration preview must not call a provider")

    service.embedding_engine = SimpleNamespace(
        enabled=True, search_similar_queries=unexpected_provider,
    )
    service.reranker_engine = SimpleNamespace(rerank=unexpected_provider)
    app = create_gateway_app(config={"gateway": {}}, service=service)
    app.state.gateway_service = service

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
            headers={"Authorization": "Bearer owner"},
        ) as client:
            path = "/api/memory-authority/configuration/preview"
            scenario = {
                "now": "2026-10-17T12:00:00Z", "purpose": "current", "kind": "fact",
            }
            preview_response = await client.post(path, json={
                "values": values, "inputs": {"relevance": 0.2}, "scenario": scenario,
            })
            assert preview_response.status_code == 200
            result = preview_response.json()["result"]
            assert result["provider_requests"] == 0 and result["saved"] is False

            document = document_template("Synthetic preview only", catalog=values["catalog"])
            statement = statement_template()
            statement["content"] = "Synthetic preview claim"
            statement["classification"]["kind"] = _typed("fact")
            document["statements"] = [statement]
            direct = score_document(document, values, {
                "now": FIXED_NOW, "target_time": None, "purpose": "current", "facts": {},
            }, {"relevance": 0.2})
            assert result["pipeline"]["score"] == pytest.approx(direct["score"])
            assert result["pipeline"]["statements"][0]["score"] == pytest.approx(
                direct["statements"][0]["score"],
            )

            legacy_response = await client.post(path, json={
                "values": _base_config(), "inputs": {"relevance": 0.7, "keyword_score": 0.3},
            })
            assert legacy_response.status_code == 200
            legacy = legacy_response.json()["result"]
            assert legacy["formulas"]["final"]["value"] == pytest.approx(0.7)
            assert legacy["provider_requests"] == 0 and legacy["saved"] is False

            malformed = await client.post(path, json={
                "values": values, "inputs": {"relevance": 0.2}, "scenario": ["not", "an object"],
            })
            assert malformed.status_code == 400

    asyncio.run(run())
    assert provider_calls == []


def test_score_pool_binds_legacy_importance_confidence_and_local_unknown_inputs(tmp_path):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})
    record = _commit_l1(authority, "bucket-runtime-inputs", "仅用于评分输入契约的合成记录。")
    bucket_id = "bucket-runtime-inputs"
    values = _base_config(mode="fast_local")
    for ident, input_key in (
        ("importance_probe", "importance"),
        ("confidence_probe", "confidence"),
        ("semantic_probe", "semantic_score"),
        ("interest_probe", "interest_affinity"),
    ):
        values["features"].append({
            "id": ident, "label": ident, "value_type": "number", "source": "runtime",
            "missing": "unknown", "input_key": input_key, "expression": "",
            "description": "synthetic bound-input observation",
        })
    values["formulas"].append({
        "id": "observed_runtime_inputs",
        "expression": (
            "importance_probe + confidence_probe + coalesce(semantic_probe, 0) "
            "+ coalesce(interest_probe, 0)"
        ),
        "weights": {},
    })
    values["recall"]["formula_id"] = "observed_runtime_inputs"

    bucket = {
        "id": bucket_id, "content": "仅用于评分输入契约的合成记录。",
        "importance": 1.5, "confidence": 0.15,
        "metadata": {"importance": 8.5, "confidence": 0.76},
    }
    request = {
        "allow_semantic": False,
        "_semantic_records": {bucket_id: {
            "state": "l1", "memory_id": record["memory_id"], "revision": record["revision"],
            "document": record["document"],
        }},
        "_use_context": {
            "now": FIXED_NOW, "target_time": None, "query": "合成输入观察",
            "purpose": "current", "overview": False, "facts": {},
        },
    }
    candidate = {
        "bucket": bucket, "score": 0.42, "keyword_score": 0.2,
        "semantic_score": 0.963, "rerank_score": None,
    }
    service = SimpleNamespace(
        config={"memory_system": values}, gateway_tz=timezone.utc,
    )

    scored = memory_use.score_pool(service, request, "合成输入观察", [candidate])

    calculation = scored[0]["_configured_score"]
    current = next(
        row for row in calculation["statements"]
        if row["statement_id"] == f"statement:{bucket_id}:current"
    )
    assert values["recall"]["mode"] == "fast_local"
    assert request["allow_semantic"] is False
    assert current["features"]["importance"] == pytest.approx(8.5)
    assert current["features"]["confidence"] == pytest.approx(0.76)
    assert current["features"]["semantic_score"] is None
    assert current["features"]["interest_affinity"] is None
    bound = current["calculation"]["features"]
    assert bound["importance_probe"] == pytest.approx(8.5)
    assert bound["confidence_probe"] == pytest.approx(0.76)
    assert bound["semantic_probe"] is None
    assert bound["interest_probe"] is None
    assert current["calculation"]["formulas"]["observed_runtime_inputs"]["value"] == pytest.approx(9.26)
