"""Natural select/admit/pick regressions; temporary state and provider doubles only."""
import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from recall_input import build_recall_input
from test_memory_local_first_settings import _app, _base_config


TARGET = "chat_memory_" + "a" * 32
OTHER = "chat_memory_" + "b" * 32
PREVIOUS = "我对Reality聊天搜索的愿望是什么？"
BODY = "希望 Reality 的聊天搜索区分原文和渲染文本，方便寻找过去的消息。"


def case(tmp_path, monkeypatch, *, skip=5):
    config = _base_config(tmp_path)
    config["gateway"].update(skip_recent_rounds=skip, cooldown_hours=6, inject_max_cards=2)
    service, _app_unused = _app(config)
    buckets = [{"id": TARGET, "content": BODY, "metadata": {"type": "dynamic", "name": "Reality聊天搜索愿望"}},
               {"id": OTHER, "content": "海边出门要带伞，周末天气多变。", "metadata": {"type": "dynamic", "name": "海边出行"}}]
    metadata = {bucket["id"]: {"memory_id": "memory:" + bucket["id"], "revision": 1,
        "body_sha256": hashlib.sha256(bucket["content"].encode()).hexdigest()} for bucket in buckets}
    service.memory_authority_view = SimpleNamespace(enabled=False,
        memory_index_metadata_map=lambda _ids: metadata)
    calls = {"candidate": 0, "rerank": 0, "pick": []}

    async def candidates(_query, _session, eligible, **_options):
        calls["candidate"] += 1
        # History supplied the highest-scoring exact/lexical evidence. Even
        # these strong values must not masquerade as a current-turn request.
        return [{"bucket": bucket, "score": .97, "semantic_score": .93,
                 "keyword_score": .9169, "exact_anchor_match": True,
                 "natural_semantic": {"index_status": "verified", "unit_id": bucket["id"]}}
                for bucket in eligible], []

    async def rerank(_query, items, **_options):
        calls["rerank"] += 1
        return [{**item, "score": .97, "rerank_score": .96} for item in items]

    picker = service._pick_dynamic_cards

    def pick(items, **options):
        calls["pick"].append([item["bucket"]["id"] for item in items])
        return picker(items, **options)

    monkeypatch.setattr(service, "_dynamic_bucket_candidate_items", candidates)
    monkeypatch.setattr(service, "_rerank_scored_bucket_candidates", rerank)
    monkeypatch.setattr(service, "_pick_dynamic_cards", pick)
    service.state_store.record_success("same-session", [TARGET])
    return service, buckets, metadata, calls


def select(service, buckets, current, *, previous=PREVIOUS, session="same-session", history=True):
    messages = ([{"role": "user", "content": previous},
                 {"role": "assistant", "content": BODY}] if history else [])
    messages.append({"role": "user", "content": current})
    request = build_recall_input(messages, current_query=current,
        text_extractor=str, cleaner=str, recent_turns=4)
    selected, suppressed, debug = asyncio.run(service._select_dynamic_buckets(
        request["q_context"], session, buckets, natural_input=request,
        allow_semantic=True, allow_query_planner=False, allow_rerank=True))
    stage = next(row for row in debug["candidate_stages"] if row["stage"] == "natural.admit_candidates")
    target = next((row for row in stage["candidate_evidence"]["candidates"]
                   if row["bucket_ref"] == hashlib.sha256(TARGET.encode()).hexdigest()[:16]), None)
    return [bucket["id"] for bucket in selected], suppressed, debug, request, target


def test_recent_context_only_strong_card_is_suppressed_before_real_picker(tmp_path, monkeypatch):
    service, buckets, _metadata, calls = case(tmp_path, monkeypatch)
    ids, suppressed, debug, request, evidence = select(service, buckets, "周末去海边要带哪些东西？")
    assert TARGET not in ids and OTHER in ids, (evidence, debug['recent_repeat'], service._specific_query_terms(request['q_current']))
    assert TARGET not in calls["pick"][-1]
    assert calls["candidate"] == calls["rerank"] == 1
    assert next(item for item in suppressed if item["bucket"]["id"] == TARGET)["admission_reason"] == "recent_context_only"
    assert evidence["recent_repeat"] == {"decision": "suppressed", "reason": "recent_context_only",
                                         "previous_admission_reason": "verified_local_evidence"}
    assert evidence["admitted"] is False and evidence["selected"] is False
    assert debug["recent_repeat"]["suppressed_count"] == request["recent_repeat"]["suppressed_count"] == 1


@pytest.mark.parametrize("current", [
    "Reality聊天搜索应该区分什么？",
    "那条记忆的原文细节呢？",
    f"请看 [bucket_id:{TARGET}]",
    "然后呢？",
    "我还记得那个愿望，你记得吗？",
    "你知道我喜欢什么吗？",
    "周末带伞吗？Reality聊天搜索应该区分什么？",
])
def test_current_requests_followups_preference_overview_and_multiquestions_remain_available(tmp_path, monkeypatch, current):
    service, buckets, _metadata, calls = case(tmp_path, monkeypatch)
    ids, _suppressed, _debug, _request, evidence = select(service, buckets, current)
    assert TARGET in ids and TARGET in calls["pick"][-1], evidence
    assert evidence["recent_repeat"]["decision"] == "bypassed"
    assert evidence["selected"] is True


def test_bare_followup_uses_nearest_user_not_assistant_or_older_topic(tmp_path, monkeypatch):
    service, buckets, _metadata, _calls = case(tmp_path, monkeypatch)
    ids, _suppressed, _debug, _request, evidence = select(
        service, buckets, "然后呢？", previous="周末去海边要带伞吗？")
    assert TARGET not in ids, evidence
    assert evidence["recent_repeat"]["reason"] == "recent_context_only"


@pytest.mark.parametrize("scope", ["disabled", "different_session", "window_elapsed", "no_history"])
def test_policy_respects_owner_round_window_session_and_current_only_semantics(tmp_path, monkeypatch, scope):
    service, buckets, _metadata, _calls = case(tmp_path, monkeypatch, skip=0 if scope == "disabled" else 5)
    if scope == "window_elapsed":
        for _ in range(5):
            service.state_store.record_success("same-session", [OTHER])
    ids, _suppressed, debug, _request, _evidence = select(service, buckets, "周末去海边要带哪些东西？",
        session="other-session" if scope == "different_session" else "same-session",
        history=scope != "no_history")
    assert TARGET in ids
    if scope == "disabled":
        assert debug["recent_repeat"]["status"] == "disabled"


def test_explicit_repeat_request_cannot_bypass_authority_body_validation(tmp_path, monkeypatch):
    service, buckets, metadata, calls = case(tmp_path, monkeypatch)
    metadata[TARGET]["body_sha256"] = "0" * 64
    ids, _suppressed, _debug, request, evidence = select(service, buckets, "那条Reality记忆的原文细节呢？")
    assert TARGET not in ids and TARGET not in calls["pick"][-1]
    assert "bucket_revision_mismatch" in request["incomplete_reasons"]
    assert evidence is None


@pytest.mark.parametrize("source_state", ["available", "revoked"])
def test_verified_correction_keeps_declared_support_but_not_revoked_source(tmp_path, monkeypatch, source_state):
    from memory_authority import MemoryAuthorityStore
    from memory_authority_view import MemoryAuthorityRecallView
    from memory_commit_service import body_sha256
    from memory_l1_contract import l1_content_sha256, semantic_sha256
    from memory_l1_schema import scene_template, unknown
    from test_memory_recall_gateway_r2 import _semantic_document, _install_source_http

    service, buckets, _metadata, calls = case(tmp_path, monkeypatch)
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "typed")})
    document, binding = _semantic_document(TARGET, BODY, topic="travel",
        current="我没有购买机票，出发日期尚未确定。", old="只有改签确认后才安排接送。")
    support, direct = [statement["statement_id"] for statement in document["statements"]]
    # Both pieces remain eligible; the condition is explicit scene support,
    # not a guessed dependency from cosine or shared bucket identity.
    document["statements"][0]["time"]["applicability"]["end"] = unknown("unbounded", evidence_refs=[binding])
    scene = scene_template(catalog={"topics": ["travel"]}, catalog_revision="gateway-recall-r2")
    scene.update(scene_id="scene:departure-condition", scene_revision_id="scene-rev:1",
                 title="出发条件", statement_refs=[direct, support])
    document["scenes"] = [scene]
    metadata = {"memory_l1": document, "memory_l1_semantic_sha256": semantic_sha256(document),
                "memory_l1_content_sha256": l1_content_sha256(BODY, document)}
    prepared = authority.prepare_memory_commit(memory_id="memory:" + TARGET, bucket_id=TARGET,
        expected_revision=0, body_sha256=body_sha256(BODY), snapshot_path="revisions/repeat/1.md",
        metadata=metadata, source_refs=["synthetic:repeat"], decision_source="repeat-test",
        idempotency_key="repeat-support", actor="test")
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=body_sha256(BODY))
    authority.finalize_memory_commit(prepared["operation_id"])
    service.memory_authority_view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path / "typed"), "memory_authority": {"enabled": True}})
    service.memory_authority_enabled = True
    source_calls = _install_source_http(monkeypatch, state=source_state)
    ids, _suppressed, _debug, request, evidence = select(
        service, buckets, "我已经购买了机票，几点出发？")
    assert source_calls, "exercise actual source validation against mocked L0 receipts"
    if source_state == "available":
        assert TARGET in ids and TARGET in calls["pick"][-1], (evidence, request['usage_by_bucket'][TARGET])
        assert evidence["recent_repeat"]["reason"] == "qualified_correction_background"
        selection = request["usage_by_bucket"][TARGET]["statement_selection"]
        assert direct in selection["direct_ids"]
        assert support in selection["required_support_ids"]
        assert support in selection["support_ids"]
        assert selection["missing_support_ids"] == []
    else:
        assert TARGET not in ids and TARGET not in calls["pick"][-1]
        assert request["usage_by_bucket"][TARGET]["state"] == "excluded"
