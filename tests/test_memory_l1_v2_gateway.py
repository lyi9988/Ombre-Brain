"""Local contract checks for the ordinary natural-card Recall path."""

import asyncio
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from gateway import GatewayService, create_gateway_app
from memory_system_config import REQUEST_CONFIGURATION, configuration
from memory_commit_service import BucketMemoryProjection, MemoryCommitService, body_sha256
from memory_l1_contract import l1_content_sha256, make_text_anchor, semantic_sha256
from memory_l1_v2 import SCHEMA_VERSION, TEXT_PROJECTION_VERSION, prepare_card, retrieval_text
from memory_delivery import compiled_candidates, payload_texts, sha
import memory_recall_runtime as memory_use
from attachment_adoption_support import openai_response
from test_memory_recall_gateway_r2 import (
    _base_config, _gateway_service, _install_source_http, _natural_candidate, _recall_input,
)
from test_memory_recall_projection_r2 import _compose, _fixture, _projection
from test_memory_local_first_settings import _base_config as _real_gateway_config


BODY = "2026年11月8日，我和主人约定先等她回复，再做下一步。那次没有执行。"


@pytest.fixture
def final_source_case(tmp_path, monkeypatch):
    with monkeypatch.context() as setup:
        authority, service, _, _, _ = _fixture(tmp_path, setup)
    card = _commit_card(authority, "final-source-chat", body="这一次已经完成。", origin="chat_l1")
    query = "那次安排后来完成了吗？"
    request = _recall_input(query)
    inputs = {section: (f"[bucket_id:{card['id']}]", [card["id"]], 500)
              for section in ("direct", "diffused", "recent")}
    state = SimpleNamespace(calls=[], response="available", delay=0.0, started=None, cancelled=False)
    original_client = httpx.AsyncClient

    async def handle(http_request):
        assert http_request.url.path == "/api/memory-authority/recall-sources"
        payload = json.loads(http_request.content)
        state.calls.append(payload)
        if state.response == "block":
            state.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                state.cancelled = True
        if state.delay:
            await asyncio.sleep(state.delay)
        if state.response == "http_error":
            return httpx.Response(503, json={"error": {"code": "recall_sources_unavailable"}})
        return httpx.Response(200, json={"items": [
            {"memory_id": item["memory_id"], "revision": item["revision"], "state": "ok", "sources": {
                key: {"state": state.response, "eligible_source": state.response == "available"}
                for key in item["binding_ids"]}} for item in payload["items"]]})

    transport = httpx.MockTransport(handle)

    def client(*args, **kwargs):
        return original_client(*args, **{**kwargs, "transport": transport})

    monkeypatch.setattr(memory_use.httpx, "AsyncClient", client)
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.synthetic.invalid")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "synthetic-source-check-token")
    frozen = {"service": id(service), "values": memory_use.values(service)}
    token = REQUEST_CONFIGURATION.set(frozen)
    try:
        yield SimpleNamespace(service=service, request=request, query=query, card=card, inputs=inputs, http=state)
    finally:
        REQUEST_CONFIGURATION.reset(token)


@pytest.mark.parametrize(("main_remaining", "owner_timeout"), [(-5.0, 15.0), (-5.0, 1.0), (0.01, 15.0)])
def test_final_source_budget_observes_a_fresh_chat_source_and_restores_the_deadline(
        final_source_case, monkeypatch, main_remaining, owner_timeout):
    case = final_source_case

    async def run():
        case.service.recall_timeout_seconds = owner_timeout
        memory_use.prepare(case.service, case.request, case.query, [case.card])
        case.request["_deadline"] = time.monotonic() + 5
        assert await memory_use.verify_sources(case.service, case.request, case.query, [{"bucket": case.card}])
        case.http.delay = 0.005
        original = time.monotonic() + main_remaining
        case.request["_deadline"] = original
        resolve_presentations = memory_use.resolve_presentations

        async def resolve(service, candidates, *, request):
            assert request["_deadline"] == original
            return await resolve_presentations(service, candidates, request=request)

        monkeypatch.setattr(memory_use, "resolve_presentations", resolve)
        sections = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        assert sections["direct"].count(case.card["content"]) == 1
        assert case.request["_deadline"] == original
        assert len(case.http.calls) == 2 and case.http.calls[0] == case.http.calls[1]
        stats = case.request["recall_io"]["final_source_recheck"]
        assert stats["passes"] == stats["calls"] == 1
        assert 0 < stats["extra_budget_ms"] <= min(2000, owner_timeout * 1000)
        assert 0 <= stats["extra_duration_ms"] <= stats["duration_ms"]
        if main_remaining < 0:
            assert stats["extra_budget_ms"] == round(min(2.0, owner_timeout) * 1000)
            assert stats["extra_duration_ms"] == stats["duration_ms"]
            assert stats["duration_ms"] >= 4

    asyncio.run(run())


@pytest.mark.parametrize(("response", "reason"), [
    ("withdrawn", "l0_source_not_eligible"), ("http_error", "l0_source_verification_unavailable"),
])
def test_final_source_budget_keeps_withdrawal_and_http_failure_excluded(final_source_case, response, reason):
    case = final_source_case

    async def run():
        memory_use.prepare(case.service, case.request, case.query, [case.card])
        assert await memory_use.verify_sources(case.service, case.request, case.query, [{"bucket": case.card}])
        case.http.response = response
        original = time.monotonic() - 5
        case.request["_deadline"] = original
        sections = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        assert not any(sections.values())
        assert case.request["_deadline"] == original
        assert len(case.http.calls) == 2
        assert case.request["presentation_trace"][0]["fallback_reason"] == reason
        assert case.request["usage_by_bucket"][case.card["id"]]["state"] == "excluded"

    asyncio.run(run())


def test_final_source_budget_leaves_the_normal_single_batch_unchanged(final_source_case):
    case = final_source_case

    async def run():
        original = time.monotonic() + 10
        case.request["_deadline"] = original
        sections = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        assert sections["direct"].count(case.card["content"]) == 1
        assert len(case.http.calls) == 1
        assert case.request["_deadline"] == original
        stats = case.request["recall_io"]["final_source_recheck"]
        assert stats["passes"] == stats["calls"] == 1
        assert stats["extra_budget_ms"] == stats["extra_duration_ms"] == 0
        assert "_final_source_deadline" not in REQUEST_CONFIGURATION.get()["source_validation"]

    asyncio.run(run())


def test_final_source_budget_reuses_one_absolute_deadline_across_passes(final_source_case, monkeypatch):
    case = final_source_case
    observed_deadlines = []
    read_batch = memory_use.read_batch

    async def read(service, request, stage, path, payload):
        observed_deadlines.append(request["_deadline"])
        return await read_batch(service, request, stage, path, payload)

    monkeypatch.setattr(memory_use, "read_batch", read)

    async def run():
        original = time.monotonic() - 5
        case.request["_deadline"] = original
        first = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        second_request = {**case.request}
        second = await memory_use.project_sections(case.service, second_request, case.query, [case.card], case.inputs)
        assert first["direct"].count(case.card["content"]) == second["direct"].count(case.card["content"]) == 1
        assert observed_deadlines[0] == observed_deadlines[1]
        assert case.request["_deadline"] == second_request["_deadline"] == original
        stats = case.request["recall_io"]["final_source_recheck"]
        assert stats["extra_budget_ms"] == 2000
        assert stats["passes"] == stats["calls"] == len(case.http.calls) == 2

    asyncio.run(run())


def test_final_source_budget_cannot_renew_an_expired_grace_period(final_source_case):
    case = final_source_case

    async def run():
        case.service.recall_timeout_seconds = 1.0
        original = time.monotonic() - 5
        case.request["_deadline"] = original
        first = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        assert first["direct"].count(case.card["content"]) == 1
        await asyncio.sleep(1.02)
        second = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        assert not any(second.values())
        assert len(case.http.calls) == 1 and case.request["_deadline"] == original
        stats = case.request["recall_io"]["final_source_recheck"]
        assert stats["extra_budget_ms"] == 1000 and stats["passes"] == 2 and stats["calls"] == 1
        assert "final_source_deadline" in case.request["incomplete_reasons"]

    asyncio.run(run())


def test_final_source_budget_propagates_external_cancel_and_cleans_up_the_http_read(final_source_case):
    case = final_source_case

    async def run():
        original = time.monotonic() - 5
        case.request["_deadline"] = original
        case.http.response, case.http.started = "block", asyncio.Event()
        task = asyncio.create_task(memory_use.project_sections(
            case.service, case.request, case.query, [case.card], case.inputs))
        await asyncio.wait_for(case.http.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert case.http.cancelled
        assert case.request["_deadline"] == original
        assert len(case.http.calls) == 1
        assert case.request["recall_io"]["final_source_recheck"]["calls"] == 1

    asyncio.run(run())


def test_final_source_budget_timeout_keeps_the_source_excluded_and_restores_the_main_deadline(final_source_case):
    case = final_source_case

    async def run():
        case.service.recall_timeout_seconds = 0.025
        original = time.monotonic() - 5
        case.request["_deadline"] = original
        case.http.response, case.http.started = "block", asyncio.Event()
        sections = await memory_use.project_sections(case.service, case.request, case.query, [case.card], case.inputs)
        assert not any(sections.values()) and case.http.cancelled
        assert case.request["_deadline"] == original
        assert "final_source_deadline" in case.request["incomplete_reasons"]
        stats = case.request["recall_io"]["final_source_recheck"]
        assert stats["extra_budget_ms"] == 25 and stats["calls"] == 1
        assert case.request["recall_io"]["sources"]["timeouts"] == 1

    asyncio.run(run())


def _card_document(bucket_id, *, body=BODY, dated=True, event_day="2026-11-08", scope_note=None,
                   title="桥面安排", keyword="蓝桦工程", origin="auto_l1"):
    source_id = "synthetic:" + bucket_id
    binding = {
        "binding_id": "source_1", "source_namespace": "synthetic", "source_id": source_id,
        "source_revision": 1, "author_ref": "owner", "speaker_ref": "owner",
        "anchors": [make_text_anchor(body, target_namespace="l0", target_id=source_id,
                                    target_revision=1, text_view_id="original", start=0, end=len(body))],
    }
    card = {
        "title": title, "body": body, "kind": "commitment", "stability": "phase",
        "keywords": [keyword], "people": ["p1"], "sources": ["source_1"],
        "primary_topic": "projects_creation",
    }
    if dated:
        card["event_time"] = {"start": event_day, "precision": "day", "sources": ["source_1"]}
    if scope_note:
        card["scope_notes"] = [{"focus": "桥面安排", "rule": scope_note, "sources": ["source_1"]}]
    document = prepare_card(card, people=[{"ref": "p1", "person_id": "owner", "name": "林柚",
                                         "kind": "human", "aliases": ["小柚"]}],
                            bindings=[binding], generation={"origin": origin})
    document.pop("body")
    return document, source_id


def _commit_card(authority, bucket_id, *, body=BODY, dated=True, origin="auto_l1"):
    document, source_id = _card_document(bucket_id, body=body, dated=dated, origin=origin)
    metadata = {
        "memory_l1": document,
        "memory_l1_semantic_sha256": semantic_sha256(document),
        "memory_l1_content_sha256": l1_content_sha256(body, document),
    }
    prepared = authority.prepare_memory_commit(
        memory_id="memory:" + bucket_id, bucket_id=bucket_id, expected_revision=0,
        body_sha256=body_sha256(body), snapshot_path=f"revisions/{bucket_id}/1.md",
        metadata=metadata, source_refs=[source_id], decision_source="gateway-test",
        idempotency_key=bucket_id, actor="test",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=body_sha256(body))
    authority.finalize_memory_commit(prepared["operation_id"])
    return {"id": bucket_id, "content": body,
            "metadata": {"name": "桥面安排", "type": "dynamic", "created": "2026-11-13T09:00:00",
                         "memory_l1_schema_version": SCHEMA_VERSION}}


def test_active_chat_preserves_the_draft_when_recalled_by_event_date(tmp_path, monkeypatch):
    with monkeypatch.context() as setup:
        authority, service, view, _, plan = _fixture(tmp_path, setup)
    body = "我和你约好，蓝桦工程先等你回复再推进，当时还没有执行。"
    card = _commit_card(authority, "bucket-active-chat", body=body, origin="chat_l1")
    bucket_id = card["id"]
    query = "2026年11月8日蓝桦工程的安排是什么？"
    request = _recall_input(query)
    inputs = {"direct": (f"[bucket_id:{bucket_id}]", [bucket_id], 500)}
    calls = _install_source_http(monkeypatch)

    sections = asyncio.run(memory_use.project_sections(service, request, query, [card], inputs))

    assert sections["direct"] == body
    assert len(calls) == 1
    record = view.memory_l1_recall_map([bucket_id])[bucket_id]
    assert record["state"] == "l1_v2"
    assert record["document"]["event_time"]["start"] == "2026-11-08"
    projection = _projection(service, request, sections)
    messages, *_ = _compose(service, plan, projection, sections, date="", recent="")
    assert sum(row["content"].count(body) for row in messages) == 1


def test_v2_normal_projection_validates_sources_and_compiles_the_full_body_once(tmp_path, monkeypatch):
    with monkeypatch.context() as setup:
        authority, service, view, _, plan = _fixture(tmp_path, setup)
    card = _commit_card(authority, "bucket-v2")
    bid = card["id"]
    metadata = view.memory_index_metadata_map([bid])[bid]
    assert metadata["text_projection_version"] == TEXT_PROJECTION_VERSION
    assert view.memory_l1_recall_map([bid])[bid]["state"] == "l1_v2"
    query = "蓝桦工程的安排是什么？"
    request = _recall_input(query)
    marker = f"[bucket_id:{bid}]"
    inputs = {section: (marker, [bid], 500) for section in ("direct", "diffused", "recent")}
    calls = _install_source_http(monkeypatch)
    sections = asyncio.run(memory_use.project_sections(service, request, query, [card], inputs))
    assert sections == {"targeted": "", "direct": BODY, "diffused": "", "recent": ""}
    assert len(calls) == 1
    assert calls[0]["payload"]["items"] == [
        {"memory_id": "memory:bucket-v2", "revision": 1, "binding_ids": ["source_1"]}
    ]
    assert calls.presentation_requests == []
    projection = _projection(service, request, sections)
    messages, *_ = _compose(service, plan, projection, sections, date="", recent="")
    assert sum(row["content"].count(BODY) for row in messages) == 1
    compiled = compiled_candidates(projection, projection["delivery_candidates"])
    assert len(compiled) == 1
    fragment = compiled[0]
    assert fragment["l1_revision"] == 1 and fragment["l2_revision"] is None
    assert fragment["text"] == BODY and fragment["text_sha256"] == body_sha256(BODY)
    assert fragment["coverage"]["statement_ids"] == []
    assert fragment["source_refs"][0]["binding_id"] == "source_1"
    guard = memory_use.snapshot_guard(service, request, complete=True)
    with monkeypatch.context() as unavailable:
        unavailable.setattr(view, "memory_l1_recall_map", lambda _ids: {})
        assert not asyncio.run(memory_use.snapshot_sources_current(service, {"memory_guard": guard}, {}))
    monkeypatch.undo()
    with monkeypatch.context() as withdrawn_sources:
        _install_source_http(withdrawn_sources, state="withdrawn")
        assert not asyncio.run(memory_use.snapshot_sources_current(service, {"memory_guard": guard}, {}))
        withdrawn = asyncio.run(memory_use.project_sections(service, request, query, [card], inputs))
        assert not any(withdrawn.values())

    _install_source_http(monkeypatch)
    request["composer_recall_enabled"] = False
    disabled = asyncio.run(memory_use.project_sections(service, request, query, [card], inputs))
    assert not any(disabled.values())


def test_dated_selection_uses_event_day_and_reranks_card_text(tmp_path, monkeypatch):
    with monkeypatch.context() as setup:
        authority, _, view, _, _ = _fixture(tmp_path, setup)
    card = _commit_card(authority, "bucket-v2")
    undated = _commit_card(authority, "bucket-undated", body="我曾和主人聊过另一个蓝桦工程。", dated=False)
    bodies = [card, undated]
    items = [_natural_candidate(bucket, score=.75, keyword_score=.75) for bucket in bodies]
    for item in items:
        item["_rerank_for_test"] = .96
    config = _base_config(mode="fast_hybrid", local_threshold=None)
    config["recall"]["rerank_threshold"] = .8
    service, calls = _gateway_service(tmp_path, monkeypatch, view, bodies, items, config)
    source_checks = _install_source_http(monkeypatch)
    query = "2026年11月8日聊过蓝桦工程的哪些安排？"
    request = _recall_input(query)
    request["date_recall_requested"] = True
    request["_use_context"]["target_time"] = datetime(2026, 11, 8, tzinfo=timezone.utc)

    # The fixture substitutes the upstream candidate builder, so restrict the
    # returned rows to the same eligible IDs the real builder receives.
    base_builder = service._dynamic_bucket_candidate_items

    async def candidates(query, session, buckets, **kwargs):
        items, suppressed = await base_builder(query, session, buckets, **kwargs)
        eligible = {row["id"] for row in buckets}
        return [item for item in items if item["bucket"]["id"] in eligible], suppressed

    service._dynamic_bucket_candidate_items = candidates
    selected, _, debug = asyncio.run(service._finish_natural_selection(
        query, "synthetic-session", bodies, recall_input=request,
    ))
    assert [bucket["id"] for bucket in selected] == ["bucket-v2"]
    assert calls["rerank"][0]["documents"] == {"bucket-v2": "桥面安排\n\n" + BODY}
    calculation = debug["configured_computation"]["scores"][0]["computation"]
    assert calculation["card_features"]["stability"] == "phase"
    assert calculation["inner_state_affinity"]["state"] == "unknown"
    assert source_checks.presentation_requests == []
    assert service._date_recall_buckets_for_date(bodies, "2026-11-13", ["蓝桦工程"]) == []
    document = view.memory_l1_recall_map(["bucket-v2"])["bucket-v2"]["document"]
    assert memory_use.l1_v2_lexical_features("桥面安排、小柚、蓝桦工程与创作", document) == {
        "title_match": 1.0, "keyword_match": 1.0, "entity_match": 1.0, "topic_match": 1.0,
    }


def test_v2_lexical_and_vector_hits_merge_in_the_real_candidate_builder(tmp_path, monkeypatch):
    from memory_authority import MemoryAuthorityStore

    config = _real_gateway_config(tmp_path)
    config["memory_authority"]["enabled"] = True
    authority = MemoryAuthorityStore(config)
    card = _commit_card(authority, "bucket-v2")
    distractor = _commit_card(authority, "bucket-distractor", body="我曾聊过另一件事。", dated=False)
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-v2-test")
    service = GatewayService(config)
    query = "小柚提过什么？"
    request = _recall_input(query)
    vector_calls = []

    async def vectors(_request, ids):
        vector_calls.append(ids)
        _request["semantic_hits"] = {card["id"]: {"index_status": "verified"}}
        return {card["id"]: .91}

    async def rerank(_query, items, **kwargs):
        assert len([item for item in items if item["bucket"]["id"] == card["id"]]) == 1
        assert kwargs["documents_override"][card["id"]] == "桥面安排\n\n" + BODY
        for item in items:
            item["rerank_score"] = .98 if item["bucket"]["id"] == card["id"] else .1
        return items

    monkeypatch.setattr(service, "_natural_semantic_candidates", vectors)
    monkeypatch.setattr(service, "_rerank_scored_bucket_candidates", rerank)
    monkeypatch.setattr(service, "_get_keyword_candidates", lambda *_args: {})
    calls = _install_source_http(monkeypatch)

    async def run():
        try:
            return await service._finish_natural_selection(
                query, "gateway-v2-test", [card, distractor], recall_input=request,
            )
        finally:
            await service.http_client.aclose()

    selected, _, debug = asyncio.run(run())
    assert vector_calls == [{card["id"], distractor["id"]}]
    assert [bucket["id"] for bucket in selected] == [card["id"]]
    chosen = [row for row in debug["configured_computation"]["candidate_scores"]
              if row["bucket_id"] == card["id"]]
    assert len(chosen) == 1
    assert chosen[0]["keyword_score"] > 0
    assert chosen[0]["semantic_score"] == .91
    assert chosen[0]["computation"]["card_features"]["lexical"]["entity_match"] == 1
    assert calls.presentation_requests == []


DATE_QUERY = "回到2026年10月11日的招聘会，我最后投了几份简历？"


@pytest.mark.parametrize(("query", "expected"), [
    (DATE_QUERY, True),
    ("2026年10月11日招聘会结果怎么样？我投了几份简历？", True),
    ("10月11日的面试有结果吗", True),
    ("10月11日聊过什么？", True),
    ("2027年1月10日的招聘会要带简历吗？", False),
    ("10月11日出生的人是什么星座？", False),
    ("10月11日的招聘会已经结束了。", False),
    ("今天的状态怎么样？", False),
    ("刚才的回答里写过10月11日吗？", False),
    ("你现在怎么看招聘会经历？", False),
    ("今天的招聘会最后投了几份简历？", True),
    ("今天的招聘会需要带几份简历？", False),
])
def test_date_route_accepts_plain_event_questions_without_treating_all_dates_as_recall(monkeypatch, query, expected):
    monkeypatch.setattr("gateway.reference_now", lambda _tz: datetime(2026, 11, 15, tzinfo=timezone.utc))
    service = GatewayService.__new__(GatewayService)
    service.gateway_tz = timezone.utc
    assert service._query_requests_date_recall(query) is expected


def test_plain_dated_event_filters_other_days_before_rerank_and_delivery(tmp_path, monkeypatch):
    from memory_authority import MemoryAuthorityStore

    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-v2-date-test")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "fixture")
    config = _real_gateway_config(tmp_path)
    config["memory_authority"]["enabled"] = True
    config["memory_system"]["recall"] = {"mode": "fast_hybrid"}
    config["gateway"].update(upstream_base_url="https://provider.fixture/v1", upstream_default_model="public-model",
                             core_memory_interval_rounds=0, favorite_memory_interval_rounds=0)
    authority = MemoryAuthorityStore(config)
    service = GatewayService(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, service.bucket_mgr))
    previous = "2026年10月10日，我提醒你明天去招聘会前把简历带上。"
    target = "2026年10月11日，你从招聘会回来，一份简历也没投出去。"
    ranks, sent = [], []

    async def sources(_service, _request, _stage, path, payload):
        assert path == "/api/memory-authority/recall-sources"
        return {"items": [{**item, "state": "ok", "sources": {
            bid: {"state": "available", "eligible_source": True} for bid in item["binding_ids"]}}
            for item in payload["items"]]}

    async def rerank(_query, items, **kwargs):
        ranks.append(list(kwargs["documents_override"].values()))
        for item in items:
            item["rerank_score"] = .95
        return items

    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response(finish_reason="stop"))

    monkeypatch.setattr(memory_use, "read_batch", sources)
    monkeypatch.setattr(service, "_rerank_scored_bucket_candidates", rerank)
    monkeypatch.setattr(service, "_prompt_plan_for_request", lambda *_args, **_kwargs: {
        "scope": "talk.initial", "gateway_slice": {"blocks": [{
            "source_id": "ombre.memory_recall", "block_id": "l1", "scope": "talk.initial",
            "anchor": "current_user.before", "mode": "live_source", "role": "system",
            "wrapper_text": "顾衍的记忆\n{{content}}",
        }]},
    })

    async def adopt(bucket_id, text, day):
        title = "招聘会"
        doc, source = _card_document(bucket_id, body=text, event_day=day, title=title, keyword=title)
        await commit.commit_memory(memory_id=bucket_id, bucket_id=bucket_id, expected_revision=0,
            body=text, metadata={"name": title, "type": "dynamic", "memory_l1": doc,
                "memory_l1_schema_version": SCHEMA_VERSION,
                "memory_l1_semantic_sha256": semantic_sha256(doc),
                "memory_l1_content_sha256": l1_content_sha256(text, doc)},
            source_refs=[source], decision_source="owner_review", idempotency_key=bucket_id, actor="owner")

    async def scenario():
        await adopt("fixture-wrong-day", previous, "2026-10-10")
        await adopt("fixture-right-day", target, "2026-10-11")
        await service.http_client.aclose()
        service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        app = create_gateway_app(config=config, service=service)
        app.state.gateway_service = service
        async with service.http_client, httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway.fixture") as client:
            response = await client.post("/v1/chat/completions", headers={
                "Authorization": "Bearer gateway-v2-date-test",
                "X-Guyan-Conversation-Id": "date-fixture",
                "X-Ombre-Canonical-Source-Event-Id": "date-user",
                "X-Ombre-Canonical-Assistant-Source-Event-Id": "date-assistant"},
                json={"model": "public-model", "messages": [{"role": "user", "content": DATE_QUERY}]})
        assert response.status_code == 200
        assert ranks == [["招聘会\n\n" + target]]
        wire = "\n".join(payload_texts(sent[-1]))
        assert wire.count(target) == 1 and previous not in wire

    asyncio.run(scenario())


def test_l1_v2_http_talk_delivery_survives_restart_and_new_revision_blocks_the_old_target(tmp_path, monkeypatch):
    from memory_authority import MemoryAuthorityStore

    config = _real_gateway_config(tmp_path)
    config["memory_authority"]["enabled"] = True
    config["gateway"].update(upstream_base_url="https://provider.fixture/v1", upstream_default_model="public-model",
                             core_memory_interval_rounds=0, favorite_memory_interval_rounds=0)
    config["memory_system"]["recall"] = {"mode": "fast_hybrid"}
    config["embedding"].update(enabled=True, api_key="fixture", base_url="https://embedding.fixture", model="fixture")
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-v2-test")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "fixture")
    authority = MemoryAuthorityStore(config)
    service = GatewayService(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, service.bucket_mgr))
    note = "仅收到明确确认后再继续。"
    revised = "2026年11月8日，我和主人约定先查蓝桦工程的桥面清单，再等她明确确认。那次没有执行。"
    bucket_id = "bucket-v2-wire"
    memory_id = "memory-v2-wire"
    sent, source_requests, embedding_requests = [], [], []

    def upstream(request):
        assert request.url.host == "provider.fixture"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response(finish_reason="stop"))

    async def sources(_service, _request, _stage, path, payload):
        assert path == "/api/memory-authority/recall-sources"
        source_requests.append(payload)
        return {"items": [{**item, "state": "ok", "sources": {
            bid: {"state": "available", "eligible_source": True} for bid in item["binding_ids"]}}
            for item in payload["items"]]}

    async def query_embeddings(texts, **_kwargs):
        embedding_requests.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    async def rerank(_query, items, **kwargs):
        for item in items:
            assert kwargs["documents_override"][item["bucket"]["id"]].startswith("桥面安排\n\n")
            item["rerank_score"] = .95
        return items

    monkeypatch.setattr(memory_use, "read_batch", sources)
    monkeypatch.setattr(service.embedding_engine, "query_embeddings", query_embeddings)
    monkeypatch.setattr(service, "_rerank_scored_bucket_candidates", rerank)

    async def adopt(body, expected):
        document, source_id = _card_document(bucket_id, scope_note=note)
        receipt = await commit.commit_memory(
            memory_id=memory_id, bucket_id=bucket_id, expected_revision=expected, body=body,
            metadata={"name": "桥面安排", "type": "dynamic", "memory_l1": document,
                      "memory_l1_schema_version": SCHEMA_VERSION,
                      "memory_l1_semantic_sha256": semantic_sha256(document),
                      "memory_l1_content_sha256": l1_content_sha256(body, document)},
            source_refs=[source_id], decision_source="owner_review" if not expected else "owner_revision",
            idempotency_key=f"fixture-{expected}", actor="owner")
        return document, receipt

    def index(body, document, revision):
        service.embedding_engine._store_embedding(bucket_id, [1.0, 0.0],
            source_revision=revision, source_sha256=body_sha256(body), parent_memory_id=memory_id,
            input_text=retrieval_text(body, document), text_projection_version=TEXT_PROJECTION_VERSION)

    async def scenario():
        await service.http_client.aclose()
        service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        app = create_gateway_app(config=config, service=service)
        app.state.gateway_service = service
        plan = {"scope": "talk.initial", "gateway_slice": {"blocks": [{
            "source_id": "ombre.memory_recall", "block_id": "l1", "scope": "talk.initial",
            "anchor": "current_user.before", "mode": "live_source", "role": "system",
            "wrapper_text": "顾衍的记忆\n{{content}}",
        }]}}
        monkeypatch.setattr(service, "_prompt_plan_for_request", lambda *_args, **_kwargs: plan)
        auth = {"Authorization": "Bearer gateway-v2-test"}
        turn = {"conversation_id": "fixture-conversation", "user_source_event_id": "user-1",
                "assistant_source_event_id": "assistant-1"}
        async with service.http_client, httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway.fixture") as client:
            async def talk(scope):
                return await client.post("/v1/chat/completions", headers={**auth,
                    "X-Guyan-Conversation-Id": scope["conversation_id"],
                    "X-Ombre-Canonical-Source-Event-Id": scope["user_source_event_id"],
                    "X-Ombre-Canonical-Assistant-Source-Event-Id": scope["assistant_source_event_id"]},
                    json={"model": "public-model", "messages": [{"role": "user", "content": "蓝桦工程的桥面安排是什么？"}]})

            document, first = await adopt(BODY, 0)
            assert first["revision"] == 1
            index(BODY, document, 1)
            assert (await talk(turn)).json()["choices"][0]["message"]["content"] == "完成"
            first_wire = "\n".join(payload_texts(sent[-1]))
            assert first_wire.count(BODY) == first_wire.count(note) == 1
            assert "source_1" not in first_wire and "evidence_bindings" not in first_wire
            assert len(embedding_requests) == 1
            from memory_delivery import MemoryDeliveryStore
            service.memory_delivery = MemoryDeliveryStore(service.memory_delivery.path)
            path = "/internal/memory-delivery/lookup"
            initial = (await client.post(path, json=turn, headers=auth)).json()
            fragment = initial["context"]["fragments"][0]
            assert len(initial["context"]["fragments"]) == 1
            assert initial["fragment_states"] == [{"fragment_id": fragment["fragment_id"], "state": "available"}]
            assert fragment["l1_revision"] == 1 and fragment["l2_revision"] is None
            assert fragment["coverage"]["l1_body_sha256"] == sha(BODY)
            assert fragment["text_sha256"] == sha(fragment["text"])
            assert fragment["source_refs"][0]["anchors"] == document["evidence_bindings"][0]["anchors"]

            _, correction = await adopt(revised, 1)
            assert correction["revision"] == 2
            original = (await client.post(path, json=turn, headers=auth)).json()
            assert original["context"] == initial["context"]
            assert original["fragment_states"][0]["state"] == "changed"
            next_turn = {**turn, "user_source_event_id": "user-2", "assistant_source_event_id": "assistant-2"}
            stale_hits = await service.embedding_engine.search_similar_queries(
                ["桥面安排"], eligible_ids={bucket_id}, index_metadata=service.memory_authority_view.memory_index_metadata_map([bucket_id]))
            assert stale_hits == [] and len(embedding_requests) == 1
            stale = await talk(next_turn)
            assert stale.status_code == 200
            stale_wire = "\n".join(payload_texts(sent[-1]))
            assert BODY not in stale_wire
            assert len(embedding_requests) == 1, "old embedding is rejected before any embedding provider work"
            index(revised, document, 2)
            last_turn = {**turn, "user_source_event_id": "user-3", "assistant_source_event_id": "assistant-3"}
            assert (await talk(last_turn)).status_code == 200
            latest_wire = "\n".join(payload_texts(sent[-1]))
            assert latest_wire.count(revised) == 1 and BODY not in latest_wire
            assert len(embedding_requests) == 2
            assert source_requests

    asyncio.run(scenario())


@pytest.mark.parametrize(("trial", "query", "vector", "rerank", "selected"), [
    (False, "另外还有什么？", .71, None, False),
    (True, "另外还有什么？", .95, None, False),
    (True, "蓝桦工程最近如何？", .62, None, True),
    (True, "蓝桦工程最近如何？", .90, .59, False),
    (True, "另外还有什么？", 0, .60, True),
])
def test_l1_v2_trial_admission_requires_independent_fallback_evidence(
        tmp_path, monkeypatch, trial, query, vector, rerank, selected):
    from memory_authority import MemoryAuthorityStore

    config = _real_gateway_config(tmp_path)
    config["memory_authority"]["enabled"] = True
    authority = MemoryAuthorityStore(config)
    card = _commit_card(authority, "bucket-v2")
    options = _base_config(mode="fast_hybrid", local_threshold=None)
    options["recall"].update(l1_v2_trial=trial, semantic_threshold=.72, rerank_threshold=.65)
    service = GatewayService(config)
    service.config["memory_system"] = options
    item = _natural_candidate(card, score=.86, keyword_score=0)
    item.update(semantic_score=vector, natural_semantic={"index_status": "verified"},
                l1_local_match=memory_use.l1_v2_lexical_features(query,
                    service.memory_authority_view.memory_l1_recall_map([card["id"]])[card["id"]]["document"]))
    if rerank is not None:
        item["rerank_score"] = rerank

    async def candidates(*_args, **_kwargs):
        return [item], []

    async def no_planner(_query):
        pytest.fail("Fast-hybrid must not call the Query Planner")

    async def run():
        request = _recall_input(query)
        request["_use_context"]["now"] = datetime(2026, 11, 14, tzinfo=timezone.utc)
        async with service.http_client:
            return await service._finish_natural_selection(query, "v2-trial", [card],
                                                           recall_input=request)

    monkeypatch.setattr(service, "_dynamic_bucket_candidate_items", candidates)
    monkeypatch.setattr(service, "_bucket_evidence_labels", lambda *_args: [])
    monkeypatch.setattr(service, "query_planner_enabled", True)
    monkeypatch.setattr(service, "_call_query_planner", no_planner)
    _install_source_http(monkeypatch)
    policy = configuration(service.config)
    assert policy["validation"]["valid"] and policy["values"]["recall"]["l1_v2_trial"] == trial
    result, _, debug = asyncio.run(run())
    assert bool(result) is selected
    calculation = debug["configured_computation"]["candidate_scores"][0]["computation"]
    assert (calculation["card_features"]["time_weight"] < 1) is trial
    historical = memory_use.l1_v2_time_weight(
        service.memory_authority_view.memory_l1_recall_map([card["id"]])[card["id"]]["document"],
        {"purpose": "history", "now": datetime(2026, 11, 14, tzinfo=timezone.utc)})
    assert historical == 1
