"""Local contract checks for the ordinary natural-card Recall path."""

import asyncio
from datetime import datetime, timezone

from gateway import GatewayService
from memory_commit_service import body_sha256
from memory_l1_contract import l1_content_sha256, make_text_anchor, semantic_sha256
from memory_l1_v2 import SCHEMA_VERSION, TEXT_PROJECTION_VERSION, prepare_card
from memory_delivery import compiled_candidates
import memory_recall_runtime as memory_use
from test_memory_recall_gateway_r2 import (
    _base_config, _gateway_service, _install_source_http, _natural_candidate, _recall_input,
)
from test_memory_recall_projection_r2 import _compose, _fixture, _projection
from test_memory_local_first_settings import _base_config as _real_gateway_config


BODY = "2026年11月8日，我和主人约定先等她回复，再做下一步。那次没有执行。"


def _commit_card(authority, bucket_id, *, body=BODY, dated=True):
    source_id = "synthetic:" + bucket_id
    binding = {
        "binding_id": "source_1", "source_namespace": "synthetic", "source_id": source_id,
        "source_revision": 1, "author_ref": "owner", "speaker_ref": "owner",
        "anchors": [make_text_anchor(body, target_namespace="l0", target_id=source_id,
                                    target_revision=1, text_view_id="original", start=0, end=len(body))],
    }
    card = {
        "title": "桥面安排", "body": body, "kind": "commitment", "stability": "phase",
        "keywords": ["蓝桦工程"], "people": ["p1"], "sources": ["source_1"],
        "primary_topic": "projects_creation",
    }
    if dated:
        card["event_time"] = {"start": "2026-11-08", "precision": "day", "sources": ["source_1"]}
    document = prepare_card(card, people=[{"ref": "p1", "person_id": "owner", "name": "林柚",
                                         "kind": "human", "aliases": ["小柚"]}],
                            bindings=[binding], generation={"origin": "auto_l1"})
    document.pop("body")
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
