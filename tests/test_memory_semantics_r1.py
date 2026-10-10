"""Synthetic E10-B evidence/authority tests; no network, models or live data."""
import asyncio
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from chat_memory_tool import ChatMemoryTool
from memory_authority import MemoryProposal, MemoryIngestionPolicy, RevisionConflict
from memory_candidate_edit import save_owner_candidate
from memory_narrative import narrative_body_hash
from memory_semantics import build_annotations, current_annotations, guard_metadata, source_keys
from memory_source_provenance import (canonical_source, bridge_from_coverage, archive_bridge,
                                     daily_sources, raw_event_source, digest)
from reflection_engine import ReflectionEngine
from test_chat_memory_tool import envelope, service
from test_memory_authority_r1 import reflection_config, attach_fake_commit_service
from test_daily_chat_memory_quality_review_v4 import patch_model, model_candidates_json

TEXT = "🙂我喜欢喝茶。晚上不喝咖啡，白天可以。"
BODY = "她告诉我自己喜欢喝茶，也说明晚上不喝咖啡，白天可以。我保留这个条件，不把它改成一概不喝。"


def annotation(**changes):
    return {"semantic_kind": "preference", "subject": "user", "assertion_basis": "owner_statement",
            "evidence": "晚上不喝咖啡，白天可以。", "topic": "咖啡", "value": "不喝", "polarity": "negative",
            "conditions": "晚上", "exceptions": "白天可以", **changes}


def source(**changes):
    return canonical_source({"event_id": "evt-1", "version_id": "v1", "role": "user", "text": TEXT,
                             "created_at": "2026-10-08T23:00:00+08:00", **changes},
                            conversation_id="c", profile_id="haven_xiaoyu")


def semantic(proposed=None, sources=None, body=BODY, proposal_id="p"):
    return build_annotations(body=body, proposed=proposed, sources=sources or [source()], proposal_id=proposal_id)


def chat_envelope(**changes):
    value = envelope(content=BODY, kind="preference", event_time={}, annotations=[annotation()], **changes)
    value["context"]["sources"][0]["text"] = TEXT
    return value


def coverage(text=TEXT):
    return {"conversation_id": "c", "items": [{"event_id": "evt-1", "version_id": "v1", "message_index": 1,
            "turn_member": True, "source_text_sha256": digest(text), "recorded_at_ms": 1791471600000}]}


def archived(text=TEXT, bridge=None):
    bridge = bridge if bridge is not None else bridge_from_coverage(
        coverage(text), [{"role": "system", "content": "rules"}, {"role": "user", "content": text}],
        profile_id="haven_xiaoyu")
    return {"id": 701, "source": "gateway", "role": "user", "text": text, "session_id": "transport",
            "conversation_id": "transport", "created_at": "2026-10-09T01:00:00Z",
            "metadata": {"profile_id": "haven_xiaoyu", "round_id": 301,
                         "canonical_source": archive_bridge(bridge, role="user", text=text)}}


def test_optional_and_legacy_do_not_require_a_questionnaire():
    result = semantic()
    assert result["items"] == [] and result["state"] == "current"
    assert guard_metadata({"source": "old"}, BODY) == {"source": "old"}
    assert not current_annotations({"source": "old"}, BODY)


def test_two_preferences_keep_subject_conditions_negation_and_unicode_span():
    first = annotation(evidence="我喜欢喝茶。", topic="茶", value="喜欢", polarity="positive")
    first.pop("conditions"); first.pop("exceptions")
    result = semantic([first, annotation()])
    assert len(result["items"]) == 2 and not result["issues"]
    tea, coffee = result["items"]
    assert tea["conditions_status"] == "not_stated"
    assert tea["evidence_refs"][0]["span"] == [1, 7]  # emoji is one Unicode codepoint
    assert coffee["qualifiers"]["conditions"] == "晚上" and coffee["polarity"] == "negative"
    assert coffee["subject_ref"]["role"] == "user"
    assert coffee["execution_authority"] is False and coffee["interpretation_status"] == "proposed"
    assert tea["annotation_id"] != coffee["annotation_id"]


@pytest.mark.parametrize("proposed", [[], None, {}, [None], [annotation(semantic_kind=[])],
    [annotation(assertion_basis=[])], [annotation(subject=[])], [annotation(polarity=[])],
    [annotation(assertion_state=[])], [annotation(owner_confirmed=True)], [annotation(evidence="absent")],
    [annotation(conditions="永久")], [annotation(valid_time="明天")], [annotation(evidence="x" * 601)]])
def test_bad_optional_fields_never_gain_evidence_authority(proposed):
    result = semantic(proposed)
    assert not result["items"]


def test_repeated_quote_is_unresolved_not_guessed():
    assert not semantic([annotation()], [source(), source(event_id="evt-2")])["items"]
    assert not semantic([annotation()], [source(text=TEXT + TEXT)])["items"]


def test_assistant_interpretation_cannot_be_owner_confirmation():
    interpretation = source(text="我觉得她可能不想被催。", role="assistant")
    value = {"semantic_kind": "boundary", "subject": "user", "assertion_basis": "owner_statement",
             "evidence": interpretation["text"]}
    assert semantic([value], [interpretation])["issues"][0]["reason"] == "speaker_basis_mismatch"
    value["assertion_basis"] = "assistant_interpretation"
    item = semantic([value], [interpretation])["items"][0]
    assert item["assertion_basis"] == "assistant_interpretation" and item["execution_authority"] is False


@pytest.mark.parametrize("state", ["changed", "cancelled", "fulfilled", "uncertain"])
def test_claim_state_is_not_action_state_or_authorization(state):
    value = semantic([annotation(assertion_state=state)])["items"][0]
    assert value["assertion_state"] == state and value["execution_authority"] is False
    assert "due_at" not in value and "goal_id" not in value


def test_runtime_bridge_preserves_namespace_version_scope_and_actual_message_time():
    raw = archived()
    resolved = raw_event_source(raw, profile_id="haven_xiaoyu")
    assert resolved["source_key"] == source()["source_key"]
    assert resolved["namespace"] == "canonical_event" and resolved["archive_ref"]["namespace"] == "raw_event"
    assert resolved["archive_ref"]["event_id"] == "701" and resolved["event_id"] == "evt-1"
    assert resolved["recorded_at"] != raw["created_at"]
    bridge = raw["metadata"]["canonical_source"]
    assert "text" not in bridge and "rules" not in json.dumps(bridge)
    assert source(version_id="v2")["source_key"] != resolved["source_key"]
    assert source(event_id="evt-later")["source_key"] != resolved["source_key"]


@pytest.mark.parametrize("field,value", [("text", TEXT + "changed"), ("source", "import"), ("role", "assistant")])
def test_changed_or_non_gateway_archive_cannot_claim_bridge(field, value):
    raw = archived(); raw[field] = value
    assert raw_event_source(raw, profile_id="haven_xiaoyu")["bridge_status"] == "unresolved"


def test_missing_ambiguous_foreign_profile_or_bad_position_bridge_is_unresolved():
    assert raw_event_source(archived(bridge=[]), profile_id="haven_xiaoyu")["namespace"] == "raw_event"
    assert raw_event_source(archived(), profile_id="another")["bridge_status"] == "unresolved"
    c = coverage(); c["items"][0]["message_index"] = -1
    assert bridge_from_coverage(c, [{"role": "user", "content": TEXT}], profile_id="p") == []
    c = coverage(); c["items"][0]["source_text_sha256"] = "0" * 64
    assert bridge_from_coverage(c, [{}, {"role": "user", "content": TEXT}], profile_id="p") == []
    b = bridge_from_coverage(coverage(), [{}, {"role": "user", "content": TEXT}], profile_id="p")
    assert archive_bridge(b + b, role="user", text=TEXT) == {}


@pytest.mark.parametrize("mode", ["auto", "review"])
def test_chat_annotations_use_original_policy_commit_and_idempotency(tmp_path, mode):
    tool, projection = service(tmp_path, mode)
    payload = chat_envelope()
    first = asyncio.run(tool.submit(payload))
    second = asyncio.run(tool.submit(payload))
    assert first == second
    assert first["memory_status"] == ("committed" if mode == "auto" else "pending")
    row = tool.store.get_candidate(first["candidate_id"])
    assert row["proposal"]["proposed_body"] == BODY and row["proposal"]["owner_explicit"] is False
    annotations = row["proposal"]["metadata"]["semantic_annotations"]
    assert annotations["items"][0]["semantic_kind"] == "preference"
    keys = source_keys(row["proposal"])
    assert len(tool.store.find_candidates_by_sources(keys)) == 1
    if mode == "auto":
        memory = tool.store.get_memory_revision(first["memory_id"], 1)
        assert memory["metadata"]["semantic_annotations"] == annotations
        assert len(projection.revisions) == 1
    else:
        assert not projection.revisions


def test_owner_title_preserves_annotations_body_edit_invalidates_and_reindexes(tmp_path):
    tool, _ = service(tmp_path, "review")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    before = tool.store.get_candidate(ident)
    keys = source_keys(before["proposal"])
    row = save_owner_candidate(tool.store, ident, edit={"title": "新标题"}, expected_revision=1, request_id="title")
    assert current_annotations(row["proposal"]["metadata"], BODY)
    row = save_owner_candidate(tool.store, ident, edit={"content": "正文已经更改，不再陈述咖啡偏好。"}, expected_revision=2, request_id="body")
    assert row["status"] == "pending"
    assert row["proposal"]["metadata"]["semantic_annotations"]["state"] == "invalidated"
    assert not tool.store.find_candidates_by_sources(keys)
    assert tool.store.get_candidate(ident)["revision"] == 3
    with sqlite3.connect(tool.store.path) as db:
        assert db.execute("select count(*) from candidate_revisions where candidate_id=?", (ident,)).fetchone()[0] == 3


def test_index_failure_rolls_back_candidate_in_same_transaction(tmp_path):
    tool, _ = service(tmp_path, "review")
    with sqlite3.connect(tool.store.path) as db:
        db.execute("CREATE TRIGGER fail_index BEFORE INSERT ON candidate_source_index BEGIN SELECT RAISE(ABORT,'synthetic'); END")
    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(tool.submit(chat_envelope()))
    assert tool.store.list_candidates(status="all") == []


def test_daily_and_chat_link_by_runtime_evidence_without_overwriting_author(tmp_path):
    tool, projection = service(tmp_path, "review")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    original = copy.deepcopy(tool.store.get_candidate(ident))
    turns = tool.engine._raw_event_turn_payloads([archived()], limit=80)
    raw = {"title": "另一次提炼咖啡偏好", "kind": "stable_preference", "confidence": .95,
           "content": "她明确告诉我，晚上不想喝咖啡，但白天可以。我不能忽略这个时间条件。",
           "source_turn_ids": [301], "source_event_ids": [701], "annotations": [annotation()]}
    audit = {}
    normalized = tool.engine._normalize_daily_chat_memory_candidates("2026-10-09", [raw], turns, mode="review", audit=audit)
    assert len(normalized) == 1, audit
    item = normalized[0]
    assert item["semantic_annotations"]["links"][0]["target_candidate_id"] == ident
    assert item["semantic_annotations"]["links"][0]["fact_equivalence"] == "unproven"
    assert "shared_source_not_same_fact" in item["soft_flags"]
    assert tool.store.get_candidate(ident) == original and not projection.revisions
    calls = patch_model(tool.engine, model_candidates_json([]))
    asyncio.run(tool.engine._extract_window_candidates("2026-10-09", turns))
    assert len(calls) == 1
    assert calls[0]["existing_source_memories"][0]["status"] == "pending"
    assert calls[0]["existing_source_memories"][0]["comparison_only"] is True
    assert "_semantic_sources" not in json.dumps(calls[0])


def test_whole_turn_same_kind_and_longer_text_are_not_fact_identity(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode="review"))
    turns = engine._raw_event_turn_payloads([archived()], limit=80)
    base = {"title": "明确的饮食偏好", "kind": "stable_preference", "confidence": .95,
            "source_turn_ids": [301], "source_event_ids": [701]}
    tea = {**base, "content": "她告诉我自己喜欢喝茶，我记下了她的饮食偏好，以后选择饮品时会留意。"}
    coffee = {**base, "content": "她告诉我自己晚上不喝咖啡，白天可以。这是一个带时间条件的饮食偏好，不是永久不喝。"}
    audit = {}
    rows = engine._normalize_daily_chat_memory_candidates("2026-10-09", [tea, coffee, tea], turns,
                                                         max_candidates=8, mode="review", audit=audit)
    assert len(rows) == 2 and rows[0]["source_hash"] == rows[1]["source_hash"], audit
    assert rows[0]["content"] == tea["content"] and rows[1]["content"] == coffee["content"]
    assert audit["merged_duplicates"] == 1
    legacy = {"kind": "preference", "source_hash": rows[0]["source_hash"], "content": "另一个事实"}
    assert not engine._daily_chat_memory_exact_duplicate(rows[0], [legacy])
    later = copy.deepcopy(archived()); later["id"] = 702; later["metadata"]["round_id"] = 302
    later["metadata"]["canonical_source"] = {}
    other = engine._normalize_daily_chat_memory_candidates("2026-10-09", [{**tea, "source_turn_ids": [302], "source_event_ids": [702]}],
                      engine._raw_event_turn_payloads([later], 80), mode="review")
    assert other[0]["id"] != rows[0]["id"]


def test_final_commit_body_edit_invalidates_metadata_and_excludes_old_candidate_lookup(tmp_path):
    tool, _ = service(tmp_path, "auto")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    candidate = tool.store.get_candidate(ident)
    keys = source_keys(candidate["proposal"])
    revision = tool.store.get_memory_revision(ident, 1)
    asyncio.run(tool.engine._memory_authority_service(object()).commit_memory(
        memory_id=ident, bucket_id=ident, expected_revision=1, body="主人后来修订的另一段正文。",
        metadata=revision["metadata"], source_refs=revision["source_refs"],
        decision_source="owner", idempotency_key="owner-body-edit", actor="owner"))
    newer = tool.store.get_memory_revision(ident, 2)
    assert newer["metadata"]["semantic_annotations"]["state"] == "invalidated"
    assert tool.store.find_candidates_by_sources(keys) == []


@pytest.mark.parametrize("text", [
    "<silent>我喜欢喝茶。</silent>真实可见原话。", "<think>我喜欢喝茶。</think>真实可见原话。",
    '<msg reason="我喜欢喝茶。">真实可见原话。</msg>',
    "<tool_result>我喜欢喝茶。</tool_result>真实可见原话。",
    "<attachment>我喜欢喝茶。</attachment>真实可见原话。",
    "【相关记忆】\n我喜欢喝茶。", "近期素材：我喜欢喝茶。", "<silent>我喜欢喝茶。",
])
def test_hidden_or_control_text_cannot_be_verified_evidence(text):
    item = {"semantic_kind": "preference", "subject": "user", "assertion_basis": "owner_statement", "evidence": "我喜欢喝茶。"}
    result = semantic([item], [source(text=text)])
    assert not result["items"] and result["issues"][0]["reason"] == "evidence_not_owner_visible"


def test_visible_quote_retains_actual_span_after_hidden_material():
    text = "<silent>internal</silent>🙂我喜欢喝茶。"
    item = {"semantic_kind": "preference", "subject": "user", "assertion_basis": "owner_statement", "evidence": "我喜欢喝茶。"}
    value = semantic([item], [source(text=text)])["items"][0]["evidence_refs"][0]
    assert text[slice(*value["span"])] == item["evidence"]
    assert value["snapshot_sha256"] == digest(text)


def test_direct_proposal_constructor_cannot_bypass_body_guard(tmp_path):
    tool, _ = service(tmp_path, "review")
    proposal = MemoryProposal(proposal_id="direct", source_type="chat_tool", proposed_body="Changed body",
        metadata={"semantic_annotations": semantic([annotation()])})
    row = tool.store.put_candidate(proposal, MemoryIngestionPolicy().evaluate(proposal))
    assert row["proposal"]["metadata"]["semantic_annotations"]["state"] == "invalidated"
    assert not source_keys(row["proposal"])


def test_revision_identity_cannot_reindex_another_candidate(tmp_path):
    tool, _ = service(tmp_path, "review")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    row = tool.store.get_candidate(ident)
    proposal = MemoryProposal.from_mapping({**row["proposal"], "proposal_id": "other"})
    with pytest.raises(ValueError, match="identity"):
        tool.store.revise_candidate(ident, expected_revision=1, proposal=proposal, request_id="wrong-id", actor="owner")
    assert tool.store.get_candidate(ident) == row
    assert len(tool.store.find_candidates_by_sources(source_keys(row["proposal"]))) == 1


def test_edit_index_rollback_and_concurrent_revision_gate(tmp_path):
    tool, _ = service(tmp_path, "review")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    before = tool.store.get_candidate(ident)
    with sqlite3.connect(tool.store.path) as db:
        db.execute("CREATE TRIGGER fail_reindex BEFORE INSERT ON candidate_source_index BEGIN SELECT RAISE(ABORT,'synthetic'); END")
    with pytest.raises(sqlite3.IntegrityError):
        save_owner_candidate(tool.store, ident, edit={"title": "新标题"}, expected_revision=1, request_id="failed")
    assert tool.store.get_candidate(ident) == before
    assert len(tool.store.find_candidates_by_sources(source_keys(before["proposal"]))) == 1
    with sqlite3.connect(tool.store.path) as db:
        db.execute("DROP TRIGGER fail_reindex")
    from concurrent.futures import ThreadPoolExecutor
    def save(title):
        try:
            return save_owner_candidate(tool.store, ident, edit={"title": title}, expected_revision=1, request_id=title)["revision"]
        except RevisionConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["first", "second"]))
    assert sorted(map(str, results)) == ["2", "conflict"]
    assert len(tool.store.find_candidates_by_sources(source_keys(before["proposal"]))) == 1


def test_scopes_and_versions_do_not_link_same_text(tmp_path):
    tool, _ = service(tmp_path, "review")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    original = tool.store.get_candidate(ident)
    for changes in ({"conversation_id": "other"}, {"profile_id": "other"}):
        bound = canonical_source({"event_id": "evt-1", "version_id": "v1", "role": "user", "text": TEXT},
            **{"conversation_id": "c", "profile_id": "haven_xiaoyu", **changes})
        assert not tool.store.find_candidates_by_sources([bound["source_key"]])
    assert not tool.store.find_candidates_by_sources([source(version_id="new-version")["source_key"]])
    assert tool.store.get_candidate(ident) == original


def test_kind_view_is_not_a_new_authority_or_action(tmp_path):
    tool, _ = service(tmp_path, "review")
    ident = asyncio.run(tool.submit(chat_envelope()))["candidate_id"]
    row = save_owner_candidate(tool.store, ident, edit={"kind": "key_event"}, expected_revision=1, request_id="kind")
    annotation = current_annotations(row["proposal"]["metadata"], BODY)["items"][0]
    assert annotation["semantic_kind"] == "preference" and annotation["execution_authority"] is False
    assert row["proposal"]["memory_type"] == "key_event" and row["status"] == "pending"
