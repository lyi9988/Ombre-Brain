"""P2-B whole-diary and scene-bundle behavior through temporary Authority/Commit."""
from __future__ import annotations

import asyncio
import copy

import pytest

from bucket_manager import BucketManager
from memory_authority import IdempotencyConflict, MemoryAuthorityStore, MemoryIngestionPolicy
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_diary_l1 import DiaryL1Service
from memory_l0_sources import McpDiarySourceAdapter, MemoryL0Sources
from memory_l1_contract import make_text_anchor
from memory_l1_index import scene_memory_id
from memory_l1_reads import MemoryL1Reads
from memory_l1_schema import (
    catalog_with_defaults,
    document_template,
    scene_template,
    statement_template,
)
from memory_l1_service import L1OwnerError, MemoryL1RevisionService


DAY = "2026-10-14"
CATALOG_REVISION = "catalog-diary-p2b-test"


def run(coro):
    return asyncio.run(coro)


def setup_diary(tmp_path, source_text, *, record_fields=None, policy=None):
    config = {"buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state")}
    authority = MemoryAuthorityStore(config)
    buckets = BucketManager(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, buckets))
    catalog = catalog_with_defaults()
    revisions = MemoryL1RevisionService(
        authority, commit, buckets, lambda: (catalog, CATALOG_REVISION)
    )
    state = {"record": {"id": "synthetic-diary", "content": source_text, **(record_fields or {})}}

    async def fake_mcp_read(day):
        assert day == DAY
        return copy.deepcopy(state["record"])

    adapter = McpDiarySourceAdapter("synthetic-diary-mcp", fake_mcp_read)
    sources = MemoryL0Sources(authority, [adapter])
    ingestion_policy = policy or MemoryIngestionPolicy()
    diary = DiaryL1Service(revisions, sources, ingestion_policy)
    reads = MemoryL1Reads(revisions, sources)
    return {
        "authority": authority,
        "buckets": buckets,
        "catalog": catalog,
        "commit": commit,
        "diary": diary,
        "policy": ingestion_policy,
        "reads": reads,
        "revisions": revisions,
        "source_state": state,
        "sources": sources,
    }


def capture(env):
    return run(env["sources"].capture_diary(DAY, binding_id="test-capture"))["source"]


def generation_for(source, **overrides):
    value = {
        "prompt_revision": "diary-prompt-r7",
        "model_route_revision": "diary-route-r3",
        "task_id": "synthetic-first-pass-task",
        "source_revision": source["source_revision"],
    }
    value.update(overrides)
    return value


def make_scene(env, source, scene_id, title, *, statement_refs=(), start=0, end=None):
    if end is None:
        end = len(source["text"])
    anchor = make_text_anchor(
        source["text"],
        target_namespace="l0",
        target_id=source["source_id"],
        target_revision=source["source_revision"],
        text_view_id=source["text_view_id"],
        start=start,
        end=end,
    )
    scene = scene_template(env["catalog"], CATALOG_REVISION)
    scene.update(
        scene_id=scene_id,
        title=title,
        anchors=[anchor],
        statement_refs=list(statement_refs),
    )
    return scene


def make_statement(statement_id, content):
    statement = statement_template()
    statement.update(statement_id=statement_id, content=content)
    return statement


def make_document(env, source, body, scenes, *, title="完整日记", statements=(), relations=()):
    document = document_template(
        body,
        title,
        catalog=env["catalog"],
        catalog_revision=CATALOG_REVISION,
    )
    document["entry"]["form"] = "diary_full"
    document["scenes"] = list(scenes)
    document["statements"] = list(statements)
    document["relations"] = list(relations)
    return document


def full_ranges(source):
    return [{"start": 0, "end": len(source["text"])}]


def prepare(env, source, document, scene_bodies, *, request_id="diary-r1", ranges=None,
            mode="review", confidence=0.4, generation=None):
    return run(env["diary"].prepare(
        day=DAY,
        document=document,
        scene_bodies=scene_bodies,
        generation=generation or generation_for(source),
        input_ranges=full_ranges(source) if ranges is None else ranges,
        request_id=request_id,
        confidence=confidence,
        requested_mode=mode,
    ))


def accept(env, candidate):
    return env["authority"].decide_candidate(
        candidate["candidate_id"],
        action="accept",
        expected_revision=candidate["revision"],
        request_id=candidate["candidate_id"] + ":owner-accept",
        actor="owner",
        reason_codes=["SYNTHETIC_TEST_ACCEPT"],
    )


def owner_save(env, memory_id, revision, mutate, request_id):
    detail = run(env["revisions"].detail(memory_id, revision))
    document = copy.deepcopy(detail["l1"]["document"])
    mutate(document)
    return run(env["revisions"].save(memory_id, {
        "expected_revision": revision,
        "request_id": request_id,
        "document": document,
        "edit_note": "synthetic P2-B owner correction",
    }))


def test_long_source_and_first_pass_are_distinct_full_bodies_children_edit_independently_and_pin_shared_facts(tmp_path):
    source_text = "L0开头。" + ("原文中保留的合成日记段落。" * 650) + "L0最后一句不能丢。"
    l1_body = "L1第一遍的完整连贯整理。" + ("独立整理正文内容。" * 610) + "L1整理末句。"
    env = setup_diary(tmp_path, source_text)
    source = capture(env)
    half = len(source_text) // 2
    statements = [
        make_statement("new:statement-a", "第一场景共享陈述：主人带伞出门。"),
        make_statement("new:statement-b", "第二场景共享陈述：雨停后回家。"),
    ]
    scenes = [
        make_scene(env, source, "new:scene-a", "出门", statement_refs=["new:statement-a"], end=half),
        make_scene(env, source, "new:scene-b", "回家", statement_refs=["new:statement-b"], start=half),
    ]
    scene_bodies = {
        "new:scene-a": "场景一独立正文，写清出门与带伞的连贯过程。",
        "new:scene-b": "场景二独立正文，写清雨停后回家的连贯过程。",
    }
    document = make_document(env, source, l1_body, scenes, statements=statements)

    candidate = prepare(env, source, document, scene_bodies, request_id="long-diary", mode="auto", confidence=0.99)
    assert candidate["status"] == "accepted"
    assert candidate["policy"]["action"] == "auto_accept"
    assert len(candidate["proposal"]["proposed_body"]) > 4_000
    assert candidate["proposal"]["proposed_body"] == l1_body
    assert candidate["proposal"]["proposed_body"] != source_text
    assert candidate["proposal"]["metadata"]["diary_l1_bundle"]["coverage"] == {
        "coordinate_unit": "unicode_codepoint",
        "source_length": len(source_text),
        "input_ranges": [{"start": 0, "end": len(source_text)}],
        "missing_ranges": [],
        "state": "complete",
    }

    result = run(env["diary"].adopt(candidate["candidate_id"]))
    assert result["status"] == "complete"
    root_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    root = run(env["revisions"].detail(root_id, 1))
    assert root["body"] == l1_body
    assert root["l1"]["document"]["narrative"]["body"] == l1_body
    assert root["l1"]["coverage"]["source_length"] == len(source_text)
    binding = root["l1"]["document"]["evidence_bindings"][0]
    assert binding["author_ref"] is None
    assert binding["speaker_ref"] is None
    located = run(env["sources"].locate(binding))
    assert located["eligible_source"] is True
    assert located["text"] == source_text
    assert len(located["text"]) > 4_000

    root_doc = root["l1"]["document"]
    by_title = {scene["title"]: scene for scene in root_doc["scenes"]}
    child_ids = {}
    for title, expected_body, statement_prefix in (
        ("出门", scene_bodies["new:scene-a"], "第一场景"),
        ("回家", scene_bodies["new:scene-b"], "第二场景"),
    ):
        scene = by_title[title]
        child_id = scene_memory_id(root_id, scene["scene_id"])
        child_ids[title] = child_id
        child = run(env["revisions"].detail(child_id, 1))
        assert child["body"] == expected_body
        assert child["l1"]["document"]["narrative"]["body"] == expected_body
        assert child["l1"]["document"]["statements"] == []
        shared_ids = scene["statement_refs"]
        assert child["l1"]["document"]["entry"]["parent_ref"] == {"memory_id": root_id, "revision": 1}
        assert child["l1"]["document"]["entry"]["scene_ref"] == {
            "scene_id": scene["scene_id"], "scene_revision_id": scene["scene_revision_id"]
        }
        assert child["l1"]["document"]["entry"]["shared_statement_refs"] == shared_ids
        assert len(shared_ids) == 1
        root_statement = next(item for item in root_doc["statements"] if item["statement_id"] in shared_ids)
        assert root_statement["content"].startswith(statement_prefix)

    scene_a = by_title["出门"]
    child_a_id = child_ids["出门"]
    owner_child_body = "主人独立改写的场景一正文，不改变整篇日记。"
    owner_save(env, child_a_id, 1, lambda doc: doc["narrative"].update(body=owner_child_body), "edit-child-a")
    assert run(env["revisions"].detail(child_a_id, 2))["body"] == owner_child_body
    assert run(env["revisions"].detail(root_id, 1))["body"] == l1_body

    old_scene_view = run(env["reads"].scene(root_id, 1, scene_a["scene_id"]))
    assert old_scene_view["child"]["state"] == "ready"
    assert old_scene_view["child"]["detail"]["requested_revision"] == 2
    assert old_scene_view["child"]["detail"]["body"] == owner_child_body
    old_statement = old_scene_view["shared_statements"][0]["content"]

    root_statement_id = scene_a["statement_refs"][0]
    owner_save(
        env,
        root_id,
        1,
        lambda doc: next(item for item in doc["statements"] if item["statement_id"] == root_statement_id).update(
            content="主人修订后的共享陈述：出门时带了一把透明伞。"
        ),
        "edit-root-shared-statement",
    )
    current_root = run(env["revisions"].detail(root_id, 2))
    assert current_root["body"] == l1_body
    assert next(item for item in current_root["l1"]["document"]["statements"]
                if item["statement_id"] == root_statement_id)["content"] != old_statement

    pinned_view = run(env["reads"].scene(root_id, 1, scene_a["scene_id"]))
    current_view = run(env["reads"].scene(root_id, 2, scene_a["scene_id"]))
    assert pinned_view["child"]["state"] == "ready"
    assert pinned_view["shared_statements"][0]["content"] == old_statement
    assert current_view["child"]["state"] == "stale"
    assert current_view["child"].get("detail") is None
    assert current_view["shared_statements"][0]["content"] != old_statement


def test_partial_coverage_forces_review_and_review_candidate_requires_accept_before_adopt(tmp_path):
    text = "头部内容。" + ("覆盖边界合成文本。" * 500) + "最后三字XYZ"
    env = setup_diary(tmp_path, text)
    source = capture(env)
    document = make_document(
        env, source, "L1已整理但本轮没有覆盖L0末尾。",
        [make_scene(env, source, "new:scene", "完整场景")],
    )
    candidate = prepare(
        env, source, document, {"new:scene": "该场景第一遍整理正文。"},
        request_id="partial-input", mode="auto", confidence=0.99,
        ranges=[{"start": 0, "end": len(text) - 3}],
    )
    assert candidate["status"] == "pending"
    assert candidate["policy"]["action"] == "queue_review"
    assert candidate["policy"]["decision_policy"] == "review"
    coverage = candidate["proposal"]["metadata"]["diary_l1_bundle"]["coverage"]
    assert coverage["state"] == "partial"
    assert coverage["missing_ranges"] == [{"start": len(text) - 3, "end": len(text)}]

    before = run(env["diary"].adopt(candidate["candidate_id"]))
    assert before["status"] == "pending"
    assert before["root"] is None
    root_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    assert env["authority"].get_memory(root_id) is None

    accepted = accept(env, candidate)
    assert accepted["status"] == "accepted"
    adopted = run(env["diary"].adopt(candidate["candidate_id"]))
    assert adopted["status"] == "partial"
    assert adopted["root"]["revision"] == 1
    assert adopted["scenes"][0]["state"] == "ready"
    assert env["authority"].get_candidate(candidate["candidate_id"])["status"] == "committed"


def test_generation_requires_all_four_fields_and_actual_fake_mcp_source_version(tmp_path):
    env = setup_diary(tmp_path, "真实假MCP捕获的合成日记来源。")
    source = capture(env)
    document = make_document(
        env, source, "第一遍整理。", [make_scene(env, source, "new:scene", "场景")]
    )
    bodies = {"new:scene": "场景正文。"}

    missing = generation_for(source)
    del missing["task_id"]
    with pytest.raises(L1OwnerError) as missing_error:
        prepare(env, source, document, bodies, request_id="missing-generation", generation=missing)
    assert missing_error.value.code == "generation_context_required"

    stale = generation_for(source, source_revision="stale-fake-mcp-revision")
    with pytest.raises(L1OwnerError) as version_error:
        prepare(env, source, document, bodies, request_id="bad-source-version", generation=stale)
    assert version_error.value.status == 409
    assert version_error.value.code == "diary_source_changed_during_generation"
    assert env["authority"].list_candidates(status="pending") == []


def test_changed_or_withdrawn_source_blocks_adoption_without_root_commit(tmp_path):
    for mutation, request_id in (("changed", "source-changed"), ("withdrawn", "source-withdrawn")):
        case = tmp_path / request_id
        env = setup_diary(case, "最初版本的合成日记原文。", record_fields={"revision_id": "mcp-r1"})
        source = capture(env)
        document = make_document(
            env, source, "已完成的第一遍整理。", [make_scene(env, source, "new:scene", "场景")]
        )
        candidate = prepare(
            env, source, document, {"new:scene": "已完成的场景正文。"},
            request_id=request_id, mode="auto", confidence=0.99,
        )
        assert candidate["status"] == "accepted"
        if mutation == "changed":
            env["source_state"]["record"]["revision_id"] = "mcp-r2"
        else:
            env["source_state"]["record"]["revoked"] = True

        with pytest.raises(L1OwnerError) as exc:
            run(env["diary"].adopt(candidate["candidate_id"]))
        assert exc.value.status == 409
        assert exc.value.code == "source_not_currently_readable"
        root_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
        assert env["authority"].get_memory(root_id) is None


def test_request_id_reuse_with_different_first_pass_payload_conflicts(tmp_path):
    env = setup_diary(tmp_path, "幂等冲突测试使用的合成日记。")
    source = capture(env)
    document = make_document(
        env, source, "第一份完整整理。", [make_scene(env, source, "new:scene", "场景")]
    )
    prepare(env, source, document, {"new:scene": "第一份场景正文。"}, request_id="same-request")

    changed = copy.deepcopy(document)
    changed["narrative"]["body"] = "同一请求标识却提交不同的第一遍正文。"
    with pytest.raises(IdempotencyConflict):
        prepare(env, source, changed, {"new:scene": "第一份场景正文。"}, request_id="same-request")


def test_explicit_split_merge_relations_and_stable_scene_revision_boundary(tmp_path):
    source_text = "原始日记场景边界依据。" * 18
    env = setup_diary(tmp_path, source_text)
    source = capture(env)
    split_scene_id, merge_scene_id = "new:scene-left", "new:scene-right"

    def endpoint(object_id):
        return {"memory_id": None, "revision": "this_revision", "object_type": "scene", "object_id": object_id}

    relations = [
        {"relation_id": "new:rel-split", "from": endpoint(split_scene_id), "to": endpoint(merge_scene_id),
         "kind": "split_from", "evidence_refs": [], "origin": "model", "confirmation_state": "proposed"},
        {"relation_id": "new:rel-merge", "from": endpoint(merge_scene_id), "to": endpoint(split_scene_id),
         "kind": "merged_from", "evidence_refs": [], "origin": "owner", "confirmation_state": "confirmed"},
    ]
    document = make_document(
        env,
        source,
        "第一遍保留边界关系的整篇日记。",
        [
            make_scene(env, source, split_scene_id, "拆分后左段", start=0, end=30),
            make_scene(env, source, merge_scene_id, "合并后的右段", start=30, end=len(source_text)),
        ],
        relations=relations,
    )
    bodies = {split_scene_id: "拆分后的左段正文。", merge_scene_id: "合并后的右段正文。"}
    first = prepare(env, source, document, bodies, request_id="scene-identities-r1", mode="auto", confidence=0.99)
    assert first["status"] == "accepted"
    run(env["diary"].adopt(first["candidate_id"]))
    root_id = first["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    first_doc = run(env["revisions"].detail(root_id, 1))["l1"]["document"]
    first_scenes = {scene["title"]: scene for scene in first_doc["scenes"]}
    first_ids = {scene["scene_id"] for scene in first_doc["scenes"]}
    first_relations = first_doc["relations"]
    assert {relation["kind"] for relation in first_relations} == {"split_from", "merged_from"}
    for relation in first_relations:
        assert relation["origin"] == "model", "a first-pass model cannot promote its relation to owner provenance"
        assert relation["confirmation_state"] == "proposed", "model relation suggestions are not owner confirmations"
        assert relation["from"]["memory_id"] == root_id
        assert relation["to"]["memory_id"] == root_id
        assert relation["from"]["revision"] == 1
        assert relation["to"]["revision"] == 1
        assert relation["from"]["object_id"] in first_ids
        assert relation["to"]["object_id"] in first_ids

    stable = first_scenes["拆分后左段"]
    revised_doc = copy.deepcopy(first_doc)
    producer_scene = next(scene for scene in revised_doc["scenes"] if scene["scene_id"] == stable["scene_id"])
    producer_scene["title"] = "边界调整后的左段"
    producer_scene["scene_revision_id"] = "producer-supplied-stale-managed-value"
    producer_scene["anchors"] = [make_text_anchor(
        source_text,
        target_namespace="l0",
        target_id=source["source_id"],
        target_revision=source["source_revision"],
        text_view_id=source["text_view_id"],
        start=1,
        end=29,
    )]
    second = prepare(
        env,
        source,
        revised_doc,
        {stable["scene_id"]: "边界调整后的场景正文。"},
        request_id="scene-identities-r2",
        mode="review",
    )
    second_doc = second["proposal"]["metadata"]["memory_l1"]
    changed_scene = next(scene for scene in second_doc["scenes"] if scene["scene_id"] == stable["scene_id"])
    assert changed_scene["scene_id"] == stable["scene_id"]
    assert changed_scene["scene_revision_id"] != stable["scene_revision_id"]
    assert changed_scene["scene_revision_id"] != "producer-supplied-stale-managed-value"
    assert changed_scene["title"] == "边界调整后的左段"
    assert second["proposal"]["metadata"]["diary_l1_bundle"]["scene_bodies"][stable["scene_id"]] == "边界调整后的场景正文。"


def test_scene_two_failure_keeps_root_and_scene_one_readable_then_retries_without_rewriting_successes(tmp_path, monkeypatch):
    source_text = "场景一的合成原文。场景二的合成原文。"
    env = setup_diary(tmp_path, source_text)
    source = capture(env)
    split = len(source_text) // 2
    document = make_document(
        env,
        source,
        "保留整篇的第一遍整理。",
        [
            make_scene(env, source, "new:scene-one", "场景一", start=0, end=split),
            make_scene(env, source, "new:scene-two", "场景二", start=split, end=len(source_text)),
        ],
    )
    candidate = prepare(
        env,
        source,
        document,
        {"new:scene-one": "第一场景首次写入成功的正文。", "new:scene-two": "第二场景稍后重试的正文。"},
        request_id="scene-failure-retry",
        mode="auto",
        confidence=0.99,
    )
    scene_ids = [item["scene_id"] for item in candidate["proposal"]["metadata"]["memory_l1"]["scenes"]]
    root_idem = candidate["candidate_id"] + ":root"
    second_idem = candidate["candidate_id"] + ":scene:" + scene_ids[1]
    commit_calls = {}
    original_commit = env["commit"].commit_memory

    async def fail_second_scene_once(**kwargs):
        key = kwargs["idempotency_key"]
        commit_calls[key] = commit_calls.get(key, 0) + 1
        if key == second_idem and commit_calls[key] == 1:
            raise RuntimeError("synthetic second-scene projection failure")
        return await original_commit(**kwargs)

    monkeypatch.setattr(env["commit"], "commit_memory", fail_second_scene_once)
    first = run(env["diary"].adopt(candidate["candidate_id"]))
    assert first["status"] == "partial"
    assert first["scenes"][0]["state"] == "ready"
    assert first["scenes"][1]["state"] == "failed"
    root_id = first["root"]["memory_id"]
    assert first["root"]["revision"] == 1
    assert run(env["revisions"].detail(root_id, 1))["body"] == "保留整篇的第一遍整理。"

    first_view = run(env["reads"].scene(root_id, 1, scene_ids[0]))
    failed_view = run(env["reads"].scene(root_id, 1, scene_ids[1]))
    assert first_view["child"]["state"] == "ready"
    assert first_view["child"]["detail"]["body"] == "第一场景首次写入成功的正文。"
    assert failed_view["child"]["state"] == "failed"
    assert failed_view["child"].get("detail") is None
    assert commit_calls[root_idem] == 1
    assert commit_calls[candidate["candidate_id"] + ":scene:" + scene_ids[0]] == 1

    # An owner edit after scene one's successful commit must survive the same-candidate retry.
    first_child_id = scene_memory_id(root_id, scene_ids[0])
    owner_body = "主人在重试前独立修改的场景一正文。"
    owner_save(env, first_child_id, 1, lambda doc: doc["narrative"].update(body=owner_body), "retry-owner-edit-scene-one")
    retried = run(env["diary"].adopt(candidate["candidate_id"]))
    assert retried["status"] == "complete"
    assert {scene["state"] for scene in retried["scenes"]} == {"ready"}
    assert commit_calls[root_idem] == 1
    assert commit_calls[candidate["candidate_id"] + ":scene:" + scene_ids[0]] == 1
    assert commit_calls[second_idem] == 2
    assert env["authority"].get_memory(root_id)["active_revision"] == 1
    assert env["authority"].get_memory(first_child_id)["active_revision"] == 2
    assert run(env["revisions"].detail(first_child_id, 2))["body"] == owner_body
    assert run(env["reads"].scene(root_id, 1, scene_ids[1]))["child"]["state"] == "ready"


def test_replaying_old_candidate_cannot_overwrite_later_owner_root_or_child_edits(tmp_path):
    env = setup_diary(tmp_path, "旧候选重放保护用的合成日记。")
    source = capture(env)
    document = make_document(
        env, source, "候选生成时的整篇L1正文。", [make_scene(env, source, "new:scene", "主场景")]
    )
    candidate = prepare(
        env, source, document, {"new:scene": "候选生成时的场景正文。"},
        request_id="old-candidate", mode="auto", confidence=0.99,
    )
    adopted = run(env["diary"].adopt(candidate["candidate_id"]))
    root_id = adopted["root"]["memory_id"]
    scene_id = adopted["scenes"][0]["scene_id"]
    child_id = scene_memory_id(root_id, scene_id)

    owner_root_body = "主人后来修改的整篇L1正文。"
    owner_child_body = "主人后来单独修改的场景正文。"
    owner_save(env, root_id, 1, lambda doc: doc["narrative"].update(body=owner_root_body), "edit-root-after-candidate")
    owner_save(env, child_id, 1, lambda doc: doc["narrative"].update(body=owner_child_body), "edit-child-after-candidate")

    replay = run(env["diary"].adopt(candidate["candidate_id"]))
    assert replay["status"] == "superseded"
    assert env["authority"].get_memory(root_id)["active_revision"] == 2
    assert env["authority"].get_memory(child_id)["active_revision"] == 2
    assert run(env["revisions"].detail(root_id, 2))["body"] == owner_root_body
    assert run(env["revisions"].detail(child_id, 2))["body"] == owner_child_body


def test_tombstoned_root_rejects_accepted_diary_candidate_before_writing_scenes(tmp_path):
    env = setup_diary(tmp_path, "根已撤回时不得写入子场景的合成日记。")
    source = capture(env)
    document = make_document(
        env, source, "已准备的整篇第一遍整理。", [make_scene(env, source, "new:scene", "主场景")]
    )
    candidate = prepare(
        env, source, document, {"new:scene": "已准备的独立场景正文。"},
        request_id="inactive-root", mode="auto", confidence=0.99,
    )
    root_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    scene_id = candidate["proposal"]["metadata"]["memory_l1"]["scenes"][0]["scene_id"]

    run(env["commit"].commit_memory(
        memory_id=root_id,
        bucket_id=root_id,
        expected_revision=0,
        body="并发形成的先行根版本。",
        metadata={"name": "先行版本"},
        source_refs=["synthetic:concurrent-root"],
        decision_source="synthetic_concurrent_root",
        idempotency_key="concurrent-root-before-adopt",
        actor="test",
    ))
    run(env["commit"].tombstone_memory(
        memory_id=root_id,
        idempotency_key="tombstone-before-adopt",
        actor="test",
    ))

    with pytest.raises(L1OwnerError) as exc:
        run(env["diary"].adopt(candidate["candidate_id"]))
    assert exc.value.status == 409
    assert exc.value.code == "diary_root_not_active"
    assert env["authority"].get_memory(root_id)["state"] == "tombstoned"
    assert env["authority"].get_memory(scene_memory_id(root_id, scene_id)) is None


def test_root_revision_change_after_child_source_reread_blocks_stale_scene_write(tmp_path, monkeypatch):
    env = setup_diary(tmp_path, "来源重查与根revision竞态保护的合成日记。")
    source = capture(env)
    document = make_document(
        env, source, "第一遍整篇整理。", [make_scene(env, source, "new:scene", "主场景")]
    )
    candidate = prepare(
        env, source, document, {"new:scene": "只能绑定根revision一的场景正文。"},
        request_id="root-changes-after-source-reread", mode="auto", confidence=0.99,
    )
    root_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    scene_id = candidate["proposal"]["metadata"]["memory_l1"]["scenes"][0]["scene_id"]
    original_locate = env["sources"].locate
    locate_calls = 0

    async def reread_source_then_advance_root(binding, **kwargs):
        nonlocal locate_calls
        result = await original_locate(binding, **kwargs)
        locate_calls += 1
        # First read gates the root commit. The next read occurs immediately
        # before the child write, where a concurrent root revision must win.
        if locate_calls == 2:
            root_revision = env["authority"].get_memory_revision(root_id, 1)
            await env["commit"].commit_memory(
                memory_id=root_id,
                bucket_id=root_id,
                expected_revision=1,
                body="来源核验后主人并发保存的新版根正文。",
                metadata=copy.deepcopy(root_revision["metadata"]),
                source_refs=root_revision["source_refs"],
                decision_source="owner_l1_edit",
                idempotency_key="concurrent-root-r2-after-source-reread",
                actor="owner",
                l1_write_intent=True,
            )
        return result

    monkeypatch.setattr(env["sources"], "locate", reread_source_then_advance_root)
    adopted = run(env["diary"].adopt(candidate["candidate_id"]))

    assert locate_calls == 2
    assert adopted["status"] == "partial"
    assert adopted["scenes"][0]["state"] == "failed"
    assert adopted["scenes"][0]["issue"]["code"] == "diary_root_dependency_changed"
    assert env["authority"].get_memory(root_id)["active_revision"] == 2
    assert run(env["revisions"].detail(root_id, 2))["body"] == "来源核验后主人并发保存的新版根正文。"
    assert env["authority"].get_memory(scene_memory_id(root_id, scene_id)) is None

    old_root_scene = run(env["reads"].scene(root_id, 1, scene_id))
    assert old_root_scene["child"]["state"] == "failed"
    assert old_root_scene["child"]["issue"]["code"] == "diary_root_dependency_changed"
    assert old_root_scene["child"].get("detail") is None
