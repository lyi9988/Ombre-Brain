"""Offline end-to-end contracts for chat and diary L1 adoption into L2."""
from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from chat_memory_tool import ChatMemoryTool, candidate_id, chat_evidence_sources
from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from memory_l1_ingestion import bind_sources, model_template
from memory_l1_schema import statement_template
from memory_l0_sources import MemoryL0Sources
from memory_l2_service import MemoryL2Service
from memory_l1_contract import semantic_sha256
from bucket_manager import BucketManager
from gateway_state import GatewayStateStore
from reflection_engine import ReflectionEngine
from test_chat_memory_preparation import setup as setup_chat
from test_memory_authority_r1 import reflection_config
from test_memory_diary_l1_p2b import (
    accept,
    make_document,
    make_scene,
    make_statement,
    prepare as prepare_diary,
    setup_diary,
    capture,
    run,
)


def _configure_l2_engine(engine, sources):
    engine._memory_l1_prompt = lambda _source_id: ("synthetic L2 prompt", {"prompt_revision": "l2-p1"})
    engine._memory_task_route = lambda stage: {
        "stage": stage,
        "client": object(),
        "configured": True,
        "model": "synthetic-l2",
        "connection": "reflection",
        "thinking_mode": "",
        "reasoning_effort": "",
        "max_tokens": 16000,
    }
    engine._memory_l1_sources = lambda: sources
    engine._completion_content = lambda response: response.choices[0].message.content
    engine._completion_finish_reason = lambda response: response.choices[0].finish_reason

    async def complete(_client, **kwargs):
        material = json.loads(kwargs["messages"][1]["content"])
        ids = [item["statement_id"] for item in material["l1_document"]["statements"]]
        ids.extend(item["statement_id"] for item in material.get("shared_statements", []))
        ids = list(dict.fromkeys(ids))
        assert ids, "L2 must be grounded in at least one persisted L1 statement"
        document = {
            "full_injection_text": "合成的第二遍完整回忆。",
            "coherent_sections": [{
                "section_id": "synthetic-section",
                "text": "合成的可检索连贯段落。",
                "statement_refs": ids,
            }],
            "optional_use_variants": [],
            "review": {"required": False, "reasons": []},
        }
        return NS(choices=[NS(message=NS(content=json.dumps(document, ensure_ascii=False)), finish_reason="stop")])

    engine._daily_chat_memory_create_completion = complete
    return engine


def _l2_engine(sources):
    return _configure_l2_engine(NS(config={}, temperature=0.2), sources)


async def _run_l2(engine, revisions):
    service = MemoryL2Service(engine, revisions)
    pending_before = service.store.pending(20)
    assert pending_before
    _prompt, _route, generation = service.generation_context()
    dependency = service.store.current_dependency(pending_before[0]["memory_id"], generation)
    await service._material(pending_before[0]["memory_id"], dependency["first_pass_revision"])
    result = await service.run_once(20)
    assert result["processed"] == len(pending_before)
    assert all(item["state"] == "ready" for item in result["results"]), result
    for pending, item in zip(pending_before, result["results"]):
        state = service.detail(pending["memory_id"])["l2"]
        assert state["state"] == "ready"
        assert state["presentation"]["document"]["full_injection_text"] == "合成的第二遍完整回忆。"
    return service


@pytest.mark.parametrize("mode", ["auto", "review"])
def test_chat_auto_and_review_share_source_bound_l1_commit_then_persisted_l2(tmp_path, mode):
    env, engine, envelope, model_calls = setup_chat(tmp_path, mode=mode)
    canonical = chat_evidence_sources(envelope["context"], profile_id=engine.memory_semantic_profile_id)
    snapshots, _bindings = bind_sources(canonical, config=engine.config)
    by_source_id = {item["source_id"]: item for item in snapshots}
    l0 = MemoryL0Sources(env["authority"])

    async def locate(binding, *, verify_only=False, **_kwargs):
        snapshot = by_source_id.get(binding["source_id"])
        if not snapshot or snapshot["source_revision"] != binding["source_revision"]:
            return {"state": "changed", "eligible_source": False}
        if verify_only:
            return {"state": "available", "eligible_source": True}
        return {"state": "available", "eligible_source": True, **copy.deepcopy(snapshot)}

    readers = NS(locate=locate, retain_copy=l0.retain_copy)
    engine._memory_l1_sources = lambda: readers
    task_store = ChatMemoryPreparation(env["authority"])
    task_store.enqueue(envelope)
    prepared = asyncio.run(prepare_chat_tasks(engine, env["buckets"]))["results"][0]
    memory_id = candidate_id(envelope["operation_id"])
    candidate = env["authority"].get_candidate(memory_id)
    assert candidate["proposal"]["proposed_body"] == envelope["arguments"]["narrative"]["body"]
    assert candidate["proposal"]["metadata"]["memory_l1"]["evidence_bindings"]
    assert len(model_calls) == 1

    if mode == "review":
        assert prepared["memory_status"] == "pending"
        assert candidate["status"] == "pending"
        assert env["authority"].get_memory(memory_id) is None
        assert MemoryL2Service(engine, env["revisions"]).store.pending(10) == []
        accepted = env["authority"].decide_candidate(
            memory_id,
            action="accept",
            expected_revision=candidate["revision"],
            request_id="stage03-review-accept",
            actor="owner",
        )
        assert accepted["status"] == "accepted"
        asyncio.run(ChatMemoryTool(engine, env["buckets"]).commit(memory_id))
        assert env["authority"].get_candidate(memory_id)["status"] == "committed"
    else:
        assert prepared["memory_status"] == "committed"

    stored = env["authority"].get_memory(memory_id)
    assert stored and stored["recall_policy"] == "enabled"
    revision = env["authority"].get_memory_revision(memory_id, 1)
    assert revision["source_refs"]
    assert revision["metadata"]["memory_l1"]["evidence_bindings"]
    assert len(MemoryL2Service(engine, env["revisions"]).store.pending(10)) == 1
    asyncio.run(prepare_chat_tasks(engine, env["buckets"]))
    assert len(MemoryL2Service(engine, env["revisions"]).store.pending(10)) == 1


@pytest.mark.parametrize("mode", ["auto", "review"])
def test_diary_auto_and_review_adopt_full_root_and_scene_into_l2(tmp_path, mode):
    env = setup_diary(tmp_path, "L0第一句。合成日记原文。L0最后一句。")
    source = capture(env)
    statement = make_statement("new:statement", "主人记录了这次合成日记。")
    statement["evidence_refs"] = ["diary-source"]
    scene = make_scene(
        env,
        source,
        "new:scene",
        "日记场景",
        statement_refs=["new:statement"],
    )
    document = make_document(
        env,
        source,
        "L1保留完整日记正文。",
        [scene],
        statements=[statement],
    )
    candidate = prepare_diary(
        env,
        source,
        document,
        {"new:scene": "L1保留的独立场景正文。"},
        request_id="stage03-diary-" + mode,
        mode=mode,
        confidence=0.99,
    )
    root_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    assert candidate["proposal"]["proposed_body"] == "L1保留完整日记正文。"

    if mode == "review":
        assert candidate["status"] == "pending"
        before = awaitless(env["diary"].adopt(candidate["candidate_id"]))
        assert before["status"] == "pending" and before["root"] is None
        assert env["authority"].get_memory(root_id) is None
        assert MemoryL2Service(_l2_engine(env["sources"]), env["revisions"]).store.pending(10) == []
        accept(env, candidate)

    adopted = run(env["diary"].adopt(candidate["candidate_id"]))
    assert adopted["status"] == "complete"
    assert adopted["root"]["revision"] == 1
    assert len(adopted["scenes"]) == 1 and adopted["scenes"][0]["state"] == "ready"
    root = env["authority"].get_memory(root_id)
    child_id = adopted["scenes"][0]["memory_id"]
    child = env["authority"].get_memory(child_id)
    assert root["recall_policy"] == child["recall_policy"] == "enabled"
    root_revision = env["authority"].get_memory_revision(root_id, 1)
    child_revision = env["authority"].get_memory_revision(child_id, 1)
    binding = candidate["proposal"]["metadata"]["diary_l1_bundle"]["source_binding"]
    expected_source_ref = "source_snapshot:" + semantic_sha256(binding)
    assert root_revision["source_refs"] == child_revision["source_refs"] == [expected_source_ref]
    assert root_revision["decision_source"] == "diary_first_pass"
    assert child_revision["decision_source"] == "diary_scene"
    assert len(MemoryL2Service(_l2_engine(env["sources"]), env["revisions"]).store.pending(10)) == 2
    replayed = run(env["diary"].adopt(candidate["candidate_id"]))
    assert replayed["status"] == "complete"
    assert env["authority"].get_memory(root_id)["active_revision"] == 1
    assert env["authority"].get_memory(child_id)["active_revision"] == 1
    assert len(MemoryL2Service(_l2_engine(env["sources"]), env["revisions"]).store.pending(10)) == 2
    awaitless(_run_l2(_l2_engine(env["sources"]), env["revisions"]))


def test_auto_daily_complete_semantic_l1_commits_to_authority_and_persists_l2(tmp_path):
    config = reflection_config(tmp_path, mode="auto")
    config["persona"] = {"profile_id": "stage03-daily-profile", "canonical_session_id": "stage03-daily-profile"}
    engine = ReflectionEngine(config)
    turns_store = GatewayStateStore(str(tmp_path / "buckets" / "gateway_state.db"))
    source_text = "我已经决定十一月初去苏州看家人，预计在十一月三日出发，日期要以公司排班为准。"
    turn_id = turns_store.record_conversation_turn(
        profile_id=engine.memory_semantic_profile_id,
        session_id="stage03-daily-session",
        round_id=1,
        user_text=source_text,
        assistant_text="我记下这项安排，等排班确定后再更新日期。",
        created_at=datetime(2026, 10, 25, 9, tzinfo=timezone.utc),
    )
    turns = turns_store.list_conversation_turns_between(
        profile_id=engine.memory_semantic_profile_id,
        start_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
        end_at=datetime(2026, 10, 26, tzinfo=timezone.utc),
        limit=10,
        chronological=True,
    )
    assert len(turns) == 1 and turns[0]["id"] == turn_id
    engine._memory_task_route = lambda stage: {
        "stage": stage,
        "client": object(),
        "configured": True,
        "model": "synthetic-auto-l1",
        "connection": "reflection",
        "thinking_mode": "",
        "reasoning_effort": "",
        "max_tokens": 16000,
    }
    engine._memory_l1_prompt = lambda _source_id, **_kwargs: (
        "synthetic complete daily L1 prompt", {"prompt_revision": "daily-l1-p1"}
    )
    title = "十一月苏州探亲安排"
    body = "主人已把十一月初赴苏州看望家人列入近期计划，出发日期仍需按公司排班确定。"
    model_calls = []

    async def complete(_client, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        model_calls.append((kwargs["task_stage"], payload))
        assert kwargs["task_stage"] == "auto_l1"
        assert payload["l1_contract"]["source_texts"][0]["text"] == source_text
        document = model_template(catalog=engine._memory_l1_revisions().catalog_provider()[0])
        document["entry"]["form"] = "episode"
        document["narrative"] = {"title": title, "body": body}
        document["retrieval"]["keywords"] = ["苏州探亲", "排班日期"]
        statement = statement_template()
        statement.pop("statement_revision_id", None)
        statement.update(
            statement_id="new:statement",
            content="主人计划十一月赴苏州看望家人，日期受排班安排影响。",
            evidence_refs=["source:1"],
        )
        document["statements"] = [statement]
        response = {
            "candidates": [{
                "kind": "key_event",
                "confidence": 0.99,
                "reason": "主人明确说明一项具体的近期安排。",
                "source_refs": ["source:1"],
                "document": document,
            }]
        }
        return NS(choices=[NS(message=NS(content=json.dumps(response, ensure_ascii=False)), finish_reason="stop")])

    engine._daily_chat_memory_create_completion = complete
    extracted, stats = awaitless(engine._extract_window_candidates("2026-10-25", turns))
    assert not stats["parse_failure"] and len(extracted) == 1
    assert extracted[0]["source_turn_ids"] == [str(turn_id)]
    assert extracted[0]["document"]["narrative"]["body"] == body
    audit = {}
    normalized = engine._normalize_daily_chat_memory_candidates(
        "2026-10-25", extracted, turns, mode="auto", min_confidence=0.9, audit=audit
    )
    assert len(normalized) == 1, audit
    candidate = normalized[0]
    assert candidate["l1_document"]["narrative"]["body"] == body
    assert candidate["memory_l1"]["statements"][0]["evidence_refs"]
    candidate_id_value = candidate["id"]

    buckets = BucketManager(config)
    first = awaitless(engine._write_daily_chat_memory_candidates(candidate_batch := [candidate], buckets))
    assert first["created"] == 1 and first["failed"] == 0, first
    assert len(model_calls) == 1
    row = engine.memory_authority_store.get_candidate(candidate_id_value)
    assert row["status"] == "committed"
    memory = engine.memory_authority_store.get_memory(candidate_id_value)
    assert memory["active_revision"] == 1 and memory["recall_policy"] == "enabled"
    revision = engine.memory_authority_store.get_memory_revision(candidate_id_value, 1)
    assert revision["decision_source"] == "auto"
    assert revision["source_refs"] == [f"conversation_turn:{turn_id}"]
    detail = awaitless(engine._memory_l1_revisions(buckets).detail(candidate_id_value))
    assert detail["l1"]["state"] == "current"
    assert detail["body"] == detail["l1"]["document"]["narrative"]["body"] == body
    assert detail["l1"]["document"]["statements"][0]["evidence_refs"]
    source_binding = detail["l1"]["document"]["evidence_bindings"][0]
    current_source = awaitless(engine._memory_l1_sources().locate(source_binding))
    assert current_source["eligible_source"] and current_source["text"] == source_text
    assert len(MemoryL2Service(_l2_engine(engine._memory_l1_sources()), engine._memory_l1_revisions(buckets)).store.pending(10)) == 1

    replay = awaitless(engine._write_daily_chat_memory_candidates(candidate_batch, buckets))
    assert replay["exists"] == 1 and replay["failed"] == 0
    assert engine.memory_authority_store.get_memory(candidate_id_value)["active_revision"] == 1
    assert len(MemoryL2Service(_l2_engine(engine._memory_l1_sources()), engine._memory_l1_revisions(buckets)).store.pending(10)) == 1
    awaitless(_run_l2(_l2_engine(engine._memory_l1_sources()), engine._memory_l1_revisions(buckets)))


def awaitless(coro):
    return asyncio.run(coro)
