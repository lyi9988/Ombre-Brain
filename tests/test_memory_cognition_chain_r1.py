"""Revision-bound cognition-chain and live-source gates over SQLite fixtures."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from memory_authority_view import MemoryAuthorityRecallView
from memory_commit_service import body_sha256
from memory_l1_contract import make_text_anchor, semantic_sha256
from memory_l1_index import scene_memory_id
from memory_recall_runtime import add_local_candidates
from memory_system_config import values_from_config
from test_memory_cognition_l1_r1 import _cognition_diary_payload
from test_memory_recall_contract_r2 import run, setup_diary


def _cognitive_diary(env, request_id="cognition-chain-r1", *, extra_statements=(), day=None):
    payload = _cognition_diary_payload(env, request_id)
    if day is not None:
        payload["day"] = day
        source = run(env["sources"].capture_diary(day, binding_id="cognition-fixture"))["source"]
        payload["document"]["scenes"][0]["anchors"] = [make_text_anchor(
            source["text"], target_namespace="l0", target_id=source["source_id"],
            target_revision=source["source_revision"], text_view_id=source["text_view_id"],
            start=0, end=len(source["text"]),
        )]
        payload["input_ranges"] = [{"start": 0, "end": len(source["text"])}]
        payload["generation"]["source_revision"] = source["source_revision"]
    for index, content in enumerate(extra_statements, start=1):
        statement = copy.deepcopy(payload["document"]["statements"][0])
        statement.update(
            statement_id=f"new:cognition-extra-{request_id}-{index}",
            statement_revision_id=None,
            content=content,
        )
        statement["cognition"] = {
            **statement["cognition"],
            "open_questions": [f"{content} 对应的合成未解问题？"],
        }
        payload["document"]["statements"].append(statement)
        payload["document"]["scenes"][0]["statement_refs"].append(statement["statement_id"])
    candidate = run(env["diary"].prepare(**payload))
    assert candidate["status"] == "accepted"
    adopted = run(env["diary"].adopt(candidate["candidate_id"]))
    assert adopted["status"] == "complete"
    memory_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    document = run(env["revisions"].detail(memory_id, 1))["l1"]["document"]
    return {
        "memory_id": memory_id,
        "child_id": scene_memory_id(memory_id, document["scenes"][0]["scene_id"]),
        "document": document,
        "binding_id": document["evidence_bindings"][0]["binding_id"],
        "statement_id": document["statements"][0]["statement_id"],
        "statement_ids": [item["statement_id"] for item in document["statements"]],
    }


def _revise(env, memory_id, expected_revision, request_id, edit):
    detail = run(env["revisions"].detail(memory_id, expected_revision))
    document = copy.deepcopy(detail["l1"]["document"])
    edit(document)
    return run(env["revisions"].save(memory_id, {
        "expected_revision": expected_revision,
        "request_id": request_id,
        "document": document,
        "edit_note": "synthetic cognition-chain revision",
    }))


def _view(env):
    return MemoryAuthorityRecallView({
        **env["config"], "memory_authority": {"enabled": True},
    })


def _write_counts(authority):
    conn = authority._connect()
    try:
        return tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                     for table in ("memory_revisions", "commit_log", "memory_l0_source_copies"))
    finally:
        conn.close()


def test_committed_revision_history_keeps_current_l0_evidence_readable_without_writes(tmp_path):
    env = setup_diary(tmp_path)
    committed = _cognitive_diary(env)
    first_revision_id = committed["document"]["statements"][0]["statement_revision_id"]

    revised = _revise(env, committed["memory_id"], 1, "cognition-chain-edit-r2", lambda document: (
        document["statements"][0].update(
            content="合成的认知陈述经过修正。",
            cognition={**document["statements"][0]["cognition"],
                       "open_questions": ["修订后仍待确认的问题？"]},
        )
    ))
    assert revised["revision"] == 2
    latest_statement = revised["result"]["l1"]["document"]["statements"][0]
    assert latest_statement["statement_revision_id"] != first_revision_id

    view = _view(env)
    graph = view.cognition_chain([(committed["memory_id"], 2)], depth=2, limit=8)
    assert [(node["memory_id"], node["revision"], node["historical"]) for node in graph["nodes"]] == [
        (committed["memory_id"], 2, False),
        (committed["memory_id"], 1, True),
    ]
    prior_statement = graph["nodes"][1]["document"]["statements"][0]
    assert prior_statement["statement_revision_id"] == first_revision_id
    assert prior_statement["content"] == committed["document"]["statements"][0]["content"]

    source = env["sources"]
    original_locate = source.locate
    locate_calls = []

    async def tracked_locate(binding, **kwargs):
        locate_calls.append((binding["binding_id"], kwargs.get("verify_only")))
        return await original_locate(binding, **kwargs)

    source.locate = tracked_locate
    before = _write_counts(env["authority"])
    historical = run(env["reads"].recall_source_status([{
        "memory_id": committed["memory_id"], "revision": 1,
        "binding_ids": [committed["binding_id"]], "allow_historical": True,
    }]))["items"][0]
    after = _write_counts(env["authority"])
    assert historical["state"] == "ok"
    assert historical["historical"] is True
    assert historical["sources"][committed["binding_id"]]["eligible_source"] is True
    assert locate_calls == [(committed["binding_id"], True)]
    assert after == before

    def unexpected_source_read(*_args, **_kwargs):
        raise AssertionError("old revisions require an explicit historical source check")

    source.locate = unexpected_source_read
    current_only = run(env["reads"].recall_source_status([{
        "memory_id": committed["memory_id"], "revision": 1,
        "binding_ids": [committed["binding_id"]],
    }]))["items"][0]
    assert current_only["state"] == "memory_not_current_or_eligible"
    assert current_only["sources"] == {}


@pytest.mark.parametrize(
    ("state", "recall_policy"),
    [("active", "manual_only"), ("tombstoned", "enabled")],
)
def test_chain_and_historical_source_gate_require_active_enabled_authority(
    tmp_path, state, recall_policy,
):
    env = setup_diary(tmp_path)
    committed = _cognitive_diary(env, f"cognition-chain-ineligible-{state}-{recall_policy}")
    env["authority"].change_memory_state(
        memory_id=committed["memory_id"], state=state, recall_policy=recall_policy,
        idempotency_key=f"chain-ineligible:{state}:{recall_policy}", actor="test",
        expected_revision=1, expected_state="active",
    )

    graph = _view(env).cognition_chain([(committed["memory_id"], 1)], depth=2, limit=8)
    assert graph["nodes"] == []
    assert graph["issues"] == [{
        "memory_id": committed["memory_id"], "revision": 1,
        "reason": "related_revision_unavailable",
    }]

    locate_calls = []
    async def tracked_locate(*args, **kwargs):
        locate_calls.append((args, kwargs))
        return {"state": "available", "eligible_source": True}
    env["sources"].locate = tracked_locate
    source_status = run(env["reads"].recall_source_status([{
        "memory_id": committed["memory_id"], "revision": 1,
        "binding_ids": [committed["binding_id"]], "allow_historical": True,
    }]))["items"][0]
    assert source_status["state"] == "memory_not_current_or_eligible"
    assert source_status["sources"] == {}
    assert locate_calls == []


def test_withdrawn_l0_source_stays_ineligible_for_an_old_revision(tmp_path):
    env = setup_diary(tmp_path)
    committed = _cognitive_diary(env, "cognition-chain-withdrawn-source")
    _revise(env, committed["memory_id"], 1, "cognition-chain-withdrawn-r2", lambda document: (
        document["statements"][0].update(content="新的合成认知版本。")
    ))
    env["source_state"]["record"]["revoked"] = True

    result = run(env["reads"].recall_source_status([{
        "memory_id": committed["memory_id"], "revision": 1,
        "binding_ids": [committed["binding_id"]], "allow_historical": True,
    }]))["items"][0]
    assert result["historical"] is True
    assert result["state"] == "ok"
    assert result["sources"][committed["binding_id"]]["state"] == "withdrawn"
    assert result["sources"][committed["binding_id"]]["eligible_source"] is False


def test_child_shared_statement_comes_from_its_pinned_root_revision(tmp_path):
    env = setup_diary(tmp_path)
    committed = _cognitive_diary(env, "cognition-chain-shared-root")
    original_statement = committed["document"]["statements"][0]
    _revise(env, committed["memory_id"], 1, "cognition-chain-shared-root-r2", lambda document: (
        document["statements"][0].update(content="新 root 版本的合成陈述。")
    ))

    graph = _view(env).cognition_chain([(committed["child_id"], 1)], depth=1, limit=8)
    assert len(graph["nodes"]) == 1
    node = graph["nodes"][0]
    assert node["document"]["entry"]["parent_ref"] == {
        "memory_id": committed["memory_id"], "revision": 1,
    }
    assert [item["statement_id"] for item in node["document"]["statements"]] == [
        original_statement["statement_id"],
    ]
    assert node["document"]["statements"][0]["content"] == original_statement["content"]
    assert node["historical"] is True
    binding_ids = [item["binding_id"] for item in node["document"]["evidence_bindings"]]
    current_source = run(env["reads"].recall_source_status([{
        "memory_id": committed["child_id"], "revision": 1,
        "binding_ids": binding_ids,
    }]))["items"][0]
    assert current_source["state"] == "dependency_changed"
    assert current_source["sources"] == {}
    historical_source = run(env["reads"].recall_source_status([{
        "memory_id": committed["child_id"], "revision": 1,
        "binding_ids": binding_ids, "allow_historical": True,
    }]))["items"][0]
    assert historical_source["state"] == "ok"
    assert all(row["eligible_source"] is True for row in historical_source["sources"].values())


def test_two_statement_endpoints_to_one_revision_are_unioned_in_via_paths(tmp_path):
    env = setup_diary(tmp_path)
    target = _cognitive_diary(env, "cognition-chain-multi-target", day="2026-10-15", extra_statements=(
        "第一个有关联的认知陈述。", "第二个有关联的认知陈述。", "未连接的认知陈述。",
    ))
    source = _cognitive_diary(env, "cognition-chain-multi-source")

    def add_two_statement_edges(document):
        source_endpoint = {"memory_id": None, "revision": "this_revision",
                           "object_type": "statement", "object_id": source["statement_id"]}
        document["relations"] = []
        for index, statement_id in enumerate(target["statement_ids"][1:3], start=1):
            document["relations"].append({
                "relation_id": f"new:cognition-multi-edge-{index}",
                "from": copy.deepcopy(source_endpoint),
                "to": {"memory_id": target["memory_id"], "revision": 1,
                       "object_type": "statement", "object_id": statement_id},
                "kind": "supplements", "evidence_refs": [], "origin": "owner",
                "confirmation_state": "confirmed",
            })

    _revise(env, source["memory_id"], 1, "cognition-chain-multi-source-r2", add_two_statement_edges)
    graph = _view(env).cognition_chain([(source["memory_id"], 2)], depth=2, limit=8)
    target_node = next(node for node in graph["nodes"] if node["memory_id"] == target["memory_id"])
    assert {(path["object_type"], path["object_id"]) for path in target_node["via_paths"]} == {
        ("statement", target["statement_ids"][1]),
        ("statement", target["statement_ids"][2]),
    }
    assert len([node for node in graph["nodes"]
                if (node["memory_id"], node["revision"]) == (target["memory_id"], 1)]) == 1


def test_source_epoch_fencing_rejects_a_new_revision_during_historical_check(tmp_path):
    env = setup_diary(tmp_path)
    committed = _cognitive_diary(env, "cognition-chain-concurrent-source")
    _revise(env, committed["memory_id"], 1, "cognition-chain-concurrent-r2", lambda document: (
        document["statements"][0].update(content="第二版合成认知陈述。")
    ))
    entered = asyncio.Event()
    release = asyncio.Event()
    original_locate = env["sources"].locate

    async def delayed_locate(binding, **kwargs):
        assert kwargs.get("verify_only") is True
        entered.set()
        await release.wait()
        return await original_locate(binding, **kwargs)

    env["sources"].locate = delayed_locate

    async def scenario():
        request = asyncio.create_task(env["reads"].recall_source_status([{
            "memory_id": committed["memory_id"], "revision": 1,
            "binding_ids": [committed["binding_id"]], "allow_historical": True,
        }]))
        await entered.wait()
        detail = await env["revisions"].detail(committed["memory_id"], 2)
        document = copy.deepcopy(detail["l1"]["document"])
        document["statements"][0]["content"] = "第三版在校验期间形成。"
        await env["revisions"].save(committed["memory_id"], {
            "expected_revision": 2, "request_id": "cognition-chain-concurrent-r3",
            "document": document, "edit_note": "concurrent synthetic revision",
        })
        release.set()
        return await request

    result = run(scenario())["items"][0]
    assert result["state"] == "dependency_changed"
    assert result["historical"] is True


def test_explicit_cycles_are_bounded_and_missing_legacy_l1_is_reported(tmp_path):
    env = setup_diary(tmp_path)
    committed = _cognitive_diary(env, "cognition-chain-cycle-and-issue")
    related = _cognitive_diary(env, "cognition-chain-cycle-target", day="2026-10-15")
    assert related["memory_id"] != committed["memory_id"]
    legacy_id = "legacy-cognition-without-l1"
    legacy_body = "Synthetic legacy memory with no L1 revision."
    env["authority"].import_legacy_memory(
        memory_id=legacy_id, bucket_id=legacy_id,
        body_sha256=body_sha256(legacy_body),
        snapshot_path=str(tmp_path / "legacy-snapshot.md"), metadata={}, source_refs=[],
    )

    def add_related_and_legacy_edges(document):
        local = {"memory_id": None, "revision": "this_revision",
                 "object_type": "statement", "object_id": committed["statement_id"]}
        document["relations"] = [
            {"relation_id": "new:cognition-cycle-forward-edge", "from": local,
             "to": {"memory_id": related["memory_id"], "revision": 1,
                    "object_type": "statement", "object_id": related["statement_id"]},
             "kind": "supplements", "evidence_refs": [], "origin": "owner",
             "confirmation_state": "confirmed"},
            {"relation_id": "new:cognition-legacy-edge", "from": local,
             "to": {"memory_id": legacy_id, "revision": 1,
                    "object_type": "memory", "object_id": legacy_id},
             "kind": "supplements", "evidence_refs": [], "origin": "owner",
             "confirmation_state": "confirmed"},
        ]

    _revise(env, committed["memory_id"], 1, "cognition-chain-cycle-r2", add_related_and_legacy_edges)

    # Simulate an imported historical edge that closes a cycle. Normal owner
    # saves reject references to a not-yet-existing revision, so this fixture
    # inserts the back-edge only after both referenced revisions exist.
    conn = env["authority"]._connect()
    try:
        row = conn.execute(
            "SELECT metadata_json FROM memory_revisions WHERE memory_id=? AND revision=1",
            (related["memory_id"],),
        ).fetchone()
        metadata = json.loads(row[0])
        related_document = metadata["memory_l1"]
        related_document["relations"] = [{
            "relation_id": "imported:cognition-cycle-back-edge",
            "from": {"memory_id": related["memory_id"], "revision": 1,
                     "object_type": "statement", "object_id": related["statement_id"]},
            "to": {"memory_id": committed["memory_id"], "revision": 2,
                   "object_type": "statement", "object_id": committed["statement_id"]},
            "kind": "supplements", "evidence_refs": [], "origin": "program",
            "confirmation_state": "confirmed",
        }]
        metadata["memory_l1_semantic_sha256"] = semantic_sha256(related_document)
        conn.execute(
            "UPDATE memory_revisions SET metadata_json=? WHERE memory_id=? AND revision=1",
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
             related["memory_id"]),
        )
        conn.commit()
    finally:
        conn.close()
    view = _view(env)

    complete = view.cognition_chain([(committed["memory_id"], 2)], depth=5, limit=20)
    versions = [(node["memory_id"], node["revision"]) for node in complete["nodes"]]
    assert versions == [
        (committed["memory_id"], 2),
        (committed["memory_id"], 1),
        (related["memory_id"], 1),
    ]
    assert len(versions) == len(set(versions))
    assert not any(node["memory_id"] == legacy_id for node in complete["nodes"])
    assert sum(1 for node in complete["nodes"]
               if (node["memory_id"], node["revision"]) == (committed["memory_id"], 2)) == 1
    assert {issue["memory_id"] for issue in complete["issues"]} == {legacy_id}
    assert complete["issues"][0]["reason"] == "related_revision_unavailable"
    returned_to_seed = complete["nodes"][0]["via_paths"]
    assert len(returned_to_seed) == 1
    assert returned_to_seed[0]["object_type"] == "statement"
    assert returned_to_seed[0]["object_id"] == committed["statement_id"]

    capped = view.cognition_chain([(committed["memory_id"], 2)], depth=5, limit=1)
    assert len(capped["nodes"]) == 1
    assert capped["truncated"] is True

    _revise(env, committed["memory_id"], 2, "cognition-chain-cycle-r3", lambda document: (
        document["statements"][0].update(content="第三版仍带有显式循环关系。")
    ))
    depth_capped = view.cognition_chain([(committed["memory_id"], 3)], depth=1, limit=20)
    assert all(node["depth"] <= 1 for node in depth_capped["nodes"])
    assert depth_capped["depth_limited"] is True
    assert depth_capped["truncated"] is True


def test_local_cognition_role_cue_is_weaker_than_question_and_owner_weights_control_it():
    config = values_from_config({})

    def add_candidate(query, role, open_questions, weights=None):
        bucket_id = "bucket-" + ("role" if query == "问题" else "question")
        document = {
            "retrieval": {"keywords": [], "tags": []},
            "entities": [],
            "statements": [{
                "classification": {"kind": {"state": "known", "value": "fact"},
                                   "topics": []},
                "cognition": {
                    "role": {"state": "known", "value": role},
                    "open_questions": list(open_questions),
                },
            }],
        }
        values = copy.deepcopy(config)
        if weights:
            values["recall"]["local_match_weights"].update(weights)
        service = SimpleNamespace(config={"memory_system": values})
        request = {
            "_semantic_records": {bucket_id: {"state": "l1", "document": document}},
            "_semantic_buckets": {bucket_id: {"id": bucket_id, "content": "synthetic"}},
        }
        return add_local_candidates(service, request, query, [])

    role_only = add_candidate("问题", "question", [])
    specific_question = add_candidate("这段安排是否过于紧凑？", "question", ["这段安排是否过于紧凑？"])
    assert role_only[0]["score"] == pytest.approx(0.15)
    assert specific_question[0]["score"] == pytest.approx(1.0)

    role_disabled = add_candidate("问题", "question", [], {
        "cognition_role_match": 0.0,
    })
    question_enabled = add_candidate("这段安排是否过于紧凑？", "question",
                                     ["这段安排是否过于紧凑？"], {
        "cognition_role_match": 0.0, "cognition_question_match": 1.0,
    })
    assert role_disabled == []
    assert question_enabled[0]["score"] == pytest.approx(1.0)

    role_enabled = add_candidate("问题", "question", [], {
        "cognition_role_match": 1.0,
    })
    question_disabled = add_candidate("这段安排是否过于紧凑？", "question",
                                      ["这段安排是否过于紧凑？"], {
        "cognition_role_match": 0.0, "cognition_question_match": 0.0,
    })
    assert role_enabled[0]["score"] == pytest.approx(1.0)
    assert question_disabled == []
