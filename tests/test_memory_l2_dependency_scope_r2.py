"""L2 source checks follow semantic dependencies, not the full L1 inventory."""
from __future__ import annotations

import asyncio
import copy
import json
import pytest
from types import SimpleNamespace as NS

from memory_l1_schema import (
    document_template,
    generation_template,
    split_document,
    statement_template,
)
from memory_l2_service import MemoryL2Service, source_dependency_ids
from test_memory_diary_l1_p2b import CATALOG_REVISION, DAY, setup_diary
from test_memory_l1_revisions import request_for
from test_memory_l2_p4 import run, setup as setup_l2


def _persist_source_bound_l1(tmp_path):
    env = setup_diary(tmp_path / "authority", "合成 L0 来源正文。")
    source_a = run(env["sources"].capture_diary(DAY, binding_id="source:A"))
    source_b = run(env["sources"].capture_diary(DAY, binding_id="source:B"))

    document = document_template(
        "只由 A 支持的完整合成记忆。",
        "L2 来源范围测试",
        catalog=env["catalog"],
        catalog_revision=CATALOG_REVISION,
    )
    document["evidence_bindings"] = [source_a["binding"], source_b["binding"]]
    statement = statement_template()
    statement.update(
        statement_id="st-a",
        statement_revision_id="sr-a",
        content="只引用 A 的语义陈述。",
        evidence_refs=["source:A"],
    )
    document["statements"] = [statement]
    document["generation"] = generation_template(
        env["catalog"], CATALOG_REVISION, "first_pass_model"
    )
    body, semantic = split_document(document)

    run(env["commit"].commit_memory(
        memory_id="m1",
        bucket_id="b1",
        expected_revision=0,
        body="只由 A 支持的完整合成记忆。",
        metadata={"name": "来源范围合成记忆"},
        source_refs=[],
        decision_source="synthetic-test",
        idempotency_key="seed-source-scope",
        actor="test",
    ))
    run(env["commit"].commit_memory(
        memory_id="m1",
        bucket_id="b1",
        expected_revision=1,
        body=body,
        metadata={"name": "来源范围合成记忆", "memory_l1": semantic},
        source_refs=[],
        decision_source="synthetic-test",
        idempotency_key="l1-source-scope",
        actor="test",
        l1_write_intent=True,
    ))

    engine_service, _, calls = setup_l2(tmp_path / "engine")
    service = MemoryL2Service(engine_service.engine, env["revisions"])
    service.engine._memory_l1_sources = lambda: env["sources"]
    return service, env, calls


def _selective_source_io(env):
    original = env["sources"].locate
    state = {"unavailable": set(), "calls": []}

    async def locate(binding, *, start=None, end=None, verify_only=False):
        binding_id = binding["binding_id"]
        state["calls"].append((binding_id, verify_only))
        if binding_id in state["unavailable"]:
            return {
                "state": "unavailable",
                "eligible_source": False,
                "reason": "synthetic_source_unavailable",
            }
        return await original(
            binding, start=start, end=end, verify_only=verify_only
        )

    env["sources"].locate = locate
    return state


def _resolve_item(statement_id, revision=2):
    return {
        "memory_id": "m1",
        "revision": revision,
        "statement_ids": [statement_id],
        "allowed_uses": ["historical_evidence"],
        "statement_uses": {statement_id: "historical_evidence"},
        "narrative_allowed": True,
    }


def test_only_semantic_bindings_block_generation_and_resolve(tmp_path):
    service, env, calls = _persist_source_bound_l1(tmp_path)
    source_io = _selective_source_io(env)
    service.prepare("m1", {"expected_l1_revision": 2, "request_id": "a-required"})

    source_io["unavailable"] = {"source:A"}
    result = run(service.run_once(1))
    assert result["results"][0]["reason"] == "l2_source_not_available"
    assert not calls
    assert {binding_id for binding_id, _ in source_io["calls"]} == {"source:A"}

    source_io["calls"].clear()
    source_io["unavailable"] = {"source:B"}
    service.prepare("m1", {"expected_l1_revision": 2, "request_id": "ignore-inventory-b"})
    result = run(service.run_once(1))
    assert result["results"][0]["state"] == "ready"
    assert len(calls) == 1
    material = json.loads(calls[0]["messages"][1]["content"])
    assert [source["binding_id"] for source in material["sources"]] == ["source:A"]
    assert [binding["binding_id"] for binding in material["l1_document"]["evidence_bindings"]] == ["source:A"]
    assert {binding_id for binding_id, _ in source_io["calls"]} == {"source:A"}

    source_io["calls"].clear()
    resolved = run(service.resolve({"items": [_resolve_item("st-a")]}))["items"][0]
    assert resolved["state"] == "ready"
    assert resolved["text"]
    assert {binding_id for binding_id, _ in source_io["calls"]} == {"source:A"}

    source_io["unavailable"] = {"source:A"}
    source_io["calls"].clear()
    resolved = run(service.resolve({"items": [_resolve_item("st-a")]}))["items"][0]
    assert resolved["state"] == "stale"
    assert not resolved["text"]
    assert resolved["reason"] == "source_dependency_changed"

    source_io["unavailable"].clear()
    env["source_state"]["record"]["revision_id"] = "source-r2"
    resolved = run(service.resolve({"items": [_resolve_item("st-a")]}))["items"][0]
    assert resolved["state"] == "stale"
    assert not resolved["text"]
    assert resolved["reason"] == "source_dependency_changed"
    env["source_state"]["record"].pop("revision_id")

    current = run(env["revisions"].detail("m1"))
    edit = {
        "expected_revision": current["current_revision"],
        "request_id": "advance-l1-after-l2",
        "document": copy.deepcopy(current["l1"]["document"]),
        "edit_note": "synthetic dependency change",
    }
    edit["document"]["narrative"]["title"] = "新的 L1 修订"
    run(env["revisions"].save("m1", edit))
    resolved = run(service.resolve({"items": [_resolve_item("st-a")]}))["items"][0]
    assert resolved["state"] == "stale"
    assert not resolved["text"]


def test_shared_scene_statements_and_active_ring_refs_are_source_dependencies():
    document = document_template("child scene", "共享场景")
    document["entry"]["scene_ref"] = {
        "scene_id": "scene-1",
        "scene_revision_id": "scene-r1",
    }
    document["entry"]["shared_statement_refs"] = ["st-shared"]
    document["evidence_bindings"] = [
        {"binding_id": "source:A"},
        {"binding_id": "source:B"},
        {"binding_id": "source:C"},
    ]
    own = statement_template()
    own.update(statement_id="st-own", evidence_refs=["source:A"])
    document["statements"] = [own]
    shared = statement_template()
    shared.update(statement_id="st-shared", evidence_refs=["source:B"])
    active_ring = {"state": "active", "source_refs": ["source:C"]}

    dependencies = source_dependency_ids(
        document,
        shared_statements=[shared],
        rings=[active_ring],
    )
    assert set(dependencies) == {"source:A", "source:B", "source:C"}

    assert set(source_dependency_ids(document, statement_ids=["st-own"])) == {"source:A"}

    legacy = document_template("legacy", "no explicit source refs")
    legacy["evidence_bindings"] = copy.deepcopy(document["evidence_bindings"])
    assert set(source_dependency_ids(legacy)) == {"source:A", "source:B", "source:C"}


def test_active_ring_source_is_verified_before_l2_generation(tmp_path):
    service, env, calls = _persist_source_bound_l1(tmp_path)
    source_io = _selective_source_io(env)
    source_io["unavailable"] = {"source:B"}
    env["authority"].append_ring(
        memory_id="m1",
        content="Active ring backed by B.",
        kind="note",
        source_refs=["source:B"],
        idempotency_key="active-ring-source-b",
        actor="test",
    )
    service.prepare("m1", {"expected_l1_revision": 2, "request_id": "active-ring-b"})

    result = run(service.run_once(1))
    assert result["results"][0]["reason"] == "l2_source_not_available"
    assert not calls
    assert "source:B" in {binding_id for binding_id, _ in source_io["calls"]}


@pytest.mark.parametrize('use_variant', [False, True])
def test_resolve_returns_only_complete_selected_sections_with_statement_refs(tmp_path, use_variant):
    service, _, _calls = setup_l2(tmp_path)
    run(service.run_once(1))
    request = request_for(service.revisions, "trip-claims")
    statement_specs = [
        ("new:old-plan", "原计划是 11 月 1 日去苏州。"),
        ("new:date-change", "11 月 1 日要加班，行程改到暂定 11 月 3 日。"),
        ("new:ticket-state", "票尚未购买，等排班稳定后再决定。"),
        ("new:coffee-preference", "她仍然喜欢咖啡的味道。"),
    ]
    request["document"]["statements"] = []
    for statement_id, content in statement_specs:
        statement = statement_template()
        statement.update(statement_id=statement_id, content=content)
        request["document"]["statements"].append(statement)
    saved = run(service.revisions.save("m1", request))
    statements = saved["result"]["l1"]["document"]["statements"]
    ids_by_content = {item["content"]: item["statement_id"] for item in statements}
    trip_ids = [ids_by_content[content] for _, content in statement_specs[:3]]
    unrelated_id = ids_by_content[statement_specs[3][1]]

    trip_text = "原计划 11 月 1 日去苏州；因 11 月 1 日要加班，暂改到 11 月 3 日。排班稳定前不买票，票尚未购买。"
    unrelated_text = "她仍然喜欢咖啡的味道。"

    async def complete(_client, **kwargs):
        material = json.loads(kwargs["messages"][1]["content"])
        ids = {item["content"]: item["statement_id"] for item in material["l1_document"]["statements"]}
        output = {
            "full_injection_text": trip_text + "\n\n" + unrelated_text,
            "coherent_sections": [
                {
                    "section_id": "trip-revision",
                    "text": trip_text,
                    "statement_refs": [ids[content] for _, content in statement_specs[:3]],
                },
                {
                    "section_id": "coffee-preference",
                    "text": unrelated_text,
                    "statement_refs": [ids[statement_specs[3][1]]],
                },
            ],
            "optional_use_variants": [],
            "review": {"required": False, "reasons": []},
        }
        if use_variant:
            output['optional_use_variants'] = [{'variant_id': 'date-only', 'text': '暂改到 11 月 3 日。',
                'statement_refs': [ids[statement_specs[1][1]]], 'intended_use': 'historical_evidence'}]
        return NS(text=json.dumps(output, ensure_ascii=False), finish="stop")

    service.engine._daily_chat_memory_create_completion = complete
    service.prepare("m1", {"expected_l1_revision": 3, "request_id": "trip-l2"})
    result = run(service.run_once(1))
    assert result["results"][0]["state"] == "ready"

    item = {
        "memory_id": "m1",
        "revision": 3,
        "statement_ids": trip_ids,
        "allowed_uses": ["historical_evidence"],
        "statement_uses": {statement_id: "historical_evidence" for statement_id in trip_ids},
        # Even when narrative use is permitted, unselected claims stay out.
        "narrative_allowed": True,
    }
    if use_variant:
        item.update(statement_ids=[*trip_ids, unrelated_id],
            statement_uses={sid: 'historical_evidence' for sid in [*trip_ids, unrelated_id]},
            matched_statement_ids=[trip_ids[1]], support_statement_ids=[trip_ids[2]])
    resolved = run(service.resolve({"items": [item]}))["items"][0]
    assert resolved["state"] == "ready"
    assert resolved["text"].endswith(trip_text)
    assert len(resolved["units"]) == (2 if use_variant else 1)
    assert next(unit for unit in resolved['units'] if unit['unit_id'] == 'trip-revision') == {
        "unit_id": "trip-revision",
        "text": trip_text,
        "statement_refs": trip_ids,
    }
    assert "票尚未购买" in resolved["text"]
    assert "暂改到 11 月 3 日" in resolved["text"]
    assert "咖啡" not in resolved["text"]
    assert unrelated_id not in {ref for unit in resolved['units'] for ref in unit['statement_refs']}
    assert set(resolved['coverage']['used_statement_ids']) == set(trip_ids)
