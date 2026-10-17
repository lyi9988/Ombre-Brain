"""Focused Memory Recall contracts over real temporary Authority/Commit data."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore, MemoryIngestionPolicy
from memory_authority_view import MemoryAuthorityRecallView
from memory_commit_service import BucketMemoryProjection, MemoryCommitService, body_sha256
from memory_diary_l1 import DiaryL1Service
from memory_l0_sources import McpDiarySourceAdapter, MemoryL0Sources, initialize as initialize_source_copies
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
from memory_system_config import (
    ConfigurationError,
    revision,
    save_configuration,
    validate_values,
    values_from_config,
)


DAY = "2026-10-14"
CATALOG_REVISION = "recall-contract-r2"


def run(coro):
    return asyncio.run(coro)


def test_old_values_and_old_client_cas_save_preserve_recall_settings(tmp_path):
    old_values = values_from_config({})
    old_values.pop("recall")
    assert validate_values(old_values)["recall"]["mode"] == "inherit"

    base = {}
    path = str(tmp_path / "runtime.yaml")
    first = values_from_config(base)
    first["recall"].update(mode="fast_hybrid", formula_id="final", local_threshold=0.33)
    saved, effective = save_configuration(path, base, {
        "expected_revision": revision(values_from_config(base)),
        "request_id": "set-recall-r2",
        "values": first,
    })
    assert saved["validation"]["valid"] is True

    legacy_values = copy.deepcopy(saved["values"])
    legacy_values.pop("recall")
    legacy_saved, after_legacy_save = save_configuration(path, base, {
        "expected_revision": saved["revision"],
        "request_id": "legacy-client-r2",
        "values": legacy_values,
    })
    assert legacy_saved["values"]["recall"] == {
        **values_from_config(effective)["recall"],
        "mode": "fast_hybrid",
        "formula_id": "final",
        "local_threshold": 0.33,
    }
    assert after_legacy_save["memory_system"]["recall"] == legacy_saved["values"]["recall"]


def test_recall_feature_input_binding_and_formula_validation():
    values = values_from_config({})
    values["features"][0]["input_key"] = "runtime_relevance"
    assert validate_values(values)["features"][0]["input_key"] == "runtime_relevance"

    computed = copy.deepcopy(values)
    computed["features"].append({
        "id": "derived", "value_type": "number", "source": "computed",
        "missing": "unknown", "expression": "relevance", "input_key": "raw_derived",
    })
    with pytest.raises(ConfigurationError) as exc:
        validate_values(computed)
    assert exc.value.code == "computed_input_binding"

    bad_formula = values_from_config({})
    bad_formula["formulas"][0]["expression"] = "__import__('os')"
    with pytest.raises(ConfigurationError) as exc:
        validate_values(bad_formula)
    assert exc.value.code == "unsupported_expression_call"

    unknown_formula = values_from_config({})
    unknown_formula["recall"]["formula_id"] = "missing_formula"
    with pytest.raises(ConfigurationError) as exc:
        validate_values(unknown_formula)
    assert exc.value.code == "unknown_recall_formula"


def setup_diary(tmp_path, *, record_id="synthetic-recall-source", body="合成日记：今天散步后回家。"):
    config = {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
    }
    authority = MemoryAuthorityStore(config)
    buckets = BucketManager(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, buckets))
    catalog = catalog_with_defaults()
    revisions = MemoryL1RevisionService(
        authority, commit, buckets, lambda: (catalog, CATALOG_REVISION)
    )
    state = {"record": {"id": record_id, "content": body, "revision_id": "source-r1"}}

    async def read_diary(_day):
        return copy.deepcopy(state["record"])

    adapter = McpDiarySourceAdapter("synthetic-recall-provider", read_diary)
    sources = MemoryL0Sources(authority, [adapter])
    return {
        "authority": authority,
        "buckets": buckets,
        "catalog": catalog,
        "commit": commit,
        "diary": DiaryL1Service(revisions, sources, MemoryIngestionPolicy()),
        "reads": MemoryL1Reads(revisions, sources),
        "revisions": revisions,
        "source_state": state,
        "sources": sources,
        "adapter": adapter,
        "config": config,
    }


def make_committed_diary(env, *, request_id="diary-r2", evidence_binding_id="diary-source"):
    source = run(env["sources"].capture_diary(DAY, binding_id="fixture-capture"))["source"]
    scene_id = "new:recall-scene"
    statement_id = "new:recall-statement"
    scene = scene_template(env["catalog"], CATALOG_REVISION)
    scene.update(
        scene_id=scene_id,
        title="散步回家",
        anchors=[make_text_anchor(
            source["text"], target_namespace="l0", target_id=source["source_id"],
            target_revision=source["source_revision"], text_view_id=source["text_view_id"],
            start=0, end=len(source["text"]),
        )],
        statement_refs=[statement_id],
    )
    statement = statement_template()
    statement.update(
        statement_id=statement_id,
        content="主人散步后回家。",
        evidence_refs=[evidence_binding_id],
    )
    document = document_template(
        "整理后的散步记录。", "合成日记",
        catalog=env["catalog"], catalog_revision=CATALOG_REVISION,
    )
    document["entry"]["form"] = "diary_full"
    document["scenes"] = [scene]
    document["statements"] = [statement]
    candidate = run(env["diary"].prepare(
        day=DAY,
        document=document,
        scene_bodies={scene_id: "场景正文：散步后回家。"},
        generation={
            "prompt_revision": "synthetic-prompt-r1",
            "model_route_revision": "synthetic-route-r1",
            "task_id": request_id,
            "source_revision": source["source_revision"],
        },
        input_ranges=[{"start": 0, "end": len(source["text"])}],
        request_id=request_id,
        confidence=0.99,
        requested_mode="auto",
    ))
    assert candidate["status"] == "accepted"
    adopted = run(env["diary"].adopt(candidate["candidate_id"]))
    assert adopted["status"] == "complete"
    memory_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    root = run(env["revisions"].detail(memory_id, 1))
    bound_scene = root["l1"]["document"]["scenes"][0]
    child_id = scene_memory_id(memory_id, bound_scene["scene_id"])
    child = env["authority"].get_memory(child_id)
    assert child and child["active_revision"] == 1
    return {
        "candidate": candidate,
        "child_id": child_id,
        "memory_id": memory_id,
        "root": root,
        "binding_id": root["l1"]["document"]["evidence_bindings"][0]["binding_id"],
        "source": source,
    }


def test_sql_recall_projection_keeps_current_l1_and_legacy_filters_policy_and_root_dependency(tmp_path):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env)
    authority = env["authority"]
    legacy_body = "保留的旧记忆正文"
    authority.import_legacy_memory(
        memory_id="legacy-recall-memory", bucket_id="legacy-recall-memory",
        body_sha256=body_sha256(legacy_body),
        snapshot_path=str(tmp_path / "legacy-snapshot.md"), metadata={}, source_refs=[],
    )
    view = MemoryAuthorityRecallView({
        **env["config"], "memory_authority": {"enabled": True},
    })
    rows = view.memory_l1_recall_map([committed["memory_id"], committed["child_id"], "legacy-recall-memory"])
    assert rows[committed["memory_id"]]["state"] == "l1"
    assert rows[committed["memory_id"]]["revision"] == 1
    assert rows[committed["child_id"]]["state"] == "l1"
    assert rows["legacy-recall-memory"]["state"] == "legacy"

    authority.change_memory_state(
        memory_id=committed["memory_id"], state="active", recall_policy="manual_only",
        idempotency_key="make-recall-manual-only", actor="test", expected_revision=1,
        expected_state="active",
    )
    filtered = view.memory_l1_recall_map([committed["memory_id"], committed["child_id"]])
    assert committed["memory_id"] not in filtered
    assert filtered[committed["child_id"]]["state"] == "dependency_changed"
    assert run(env["reads"].recall_source_status([{
        "memory_id": committed["memory_id"], "revision": 1,
        "binding_ids": [committed["binding_id"]],
    }]))["items"][0]["state"] == "memory_not_current_or_eligible"


def test_recall_source_verification_is_ephemeral_copy_is_ineligible_and_binding_is_revision_local(tmp_path):
    env = setup_diary(tmp_path)
    # Build a binding without retaining a snapshot; verify_only must not create one.
    snapshot = run(env["adapter"].capture(DAY))
    binding = {
        "binding_id": "ephemeral-source",
        "source_namespace": snapshot["source_namespace"],
        "source_id": snapshot["source_id"],
        "source_revision": snapshot["source_revision"],
        "author_ref": None,
        "speaker_ref": None,
        "anchors": [make_text_anchor(
            snapshot["text"], target_namespace="l0", target_id=snapshot["source_id"],
            target_revision=snapshot["source_revision"], text_view_id=snapshot["text_view_id"],
            start=0, end=len(snapshot["text"]),
        )],
    }
    conn = env["authority"]._connect()
    try:
        initialize_source_copies(conn)
        conn.commit()
    finally:
        conn.close()
    assert env["sources"].copy(binding) is None
    verified = run(env["sources"].locate(binding, verify_only=True))
    assert verified["eligible_source"] is True
    assert "text" not in verified
    assert env["sources"].copy(binding) is None

    # A previously retained copy remains a display fallback, never current evidence.
    env["sources"].retain_copy(snapshot)
    env["source_state"]["record"]["revoked"] = True
    withdrawn = run(env["sources"].locate(binding, verify_only=True))
    assert withdrawn["state"] == "withdrawn"
    assert withdrawn["eligible_source"] is False
    display = run(env["sources"].locate(binding))
    assert display["eligible_source"] is False
    assert display["view"] == "source_copy"
    assert display["text"] == snapshot["text"]

    # A binding from another committed memory cannot be selected by memory ID.
    env["source_state"]["record"].pop("revoked")
    first = make_committed_diary(env, request_id="first-binding-memory")
    env["source_state"]["record"] = {
        "id": "second-synthetic-recall-source", "content": "另一份合成来源。", "revision_id": "source-r1",
    }
    original_capture = env["sources"].capture_diary
    async def capture_second_with_local_id(day, *, binding_id):
        result = await original_capture(day, binding_id=binding_id)
        if result["binding"] is not None:
            result["binding"]["binding_id"] = "only-second-memory-binding"
        return result

    env["sources"].capture_diary = capture_second_with_local_id
    second = make_committed_diary(
        env, request_id="second-binding-memory",
        evidence_binding_id="only-second-memory-binding",
    )
    env["sources"].capture_diary = original_capture
    first_root = run(env["revisions"].detail(first["memory_id"], 1))["l1"]["document"]
    second_binding_id = run(env["revisions"].detail(second["memory_id"], 1))["l1"]["document"]["evidence_bindings"][0]["binding_id"]
    first_binding_id = first_root["evidence_bindings"][0]["binding_id"]
    assert second_binding_id != first_binding_id
    status = run(env["reads"].recall_source_status([{
        "memory_id": first["memory_id"], "revision": 1,
        "binding_ids": [second_binding_id],
    }]))["items"][0]
    assert status["state"] == "ok"
    assert status["sources"][second_binding_id] == {
        "state": "unavailable", "eligible_source": False,
        "reason": "binding_not_in_revision",
    }


def _revise_root(env, memory_id, *, request_id="revise-root-r2"):
    detail = run(env["revisions"].detail(memory_id, 1))
    document = copy.deepcopy(detail["l1"]["document"])
    document["narrative"]["title"] = "新的 root 修订"
    return run(env["revisions"].save(memory_id, {
        "expected_revision": 1,
        "request_id": request_id,
        "document": document,
        "edit_note": "synthetic root dependency change",
    }))


@pytest.mark.parametrize("root_change", ["revision", "manual_only", "archived"])
def test_current_child_source_status_rejects_changed_or_ineligible_root_without_read(tmp_path, root_change):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env)
    if root_change == "revision":
        _revise_root(env, committed["memory_id"], request_id="root-revision-before-source-check")
    else:
        env["authority"].change_memory_state(
            memory_id=committed["memory_id"], state="archived" if root_change == "archived" else "active",
            recall_policy="manual_only" if root_change == "manual_only" else "enabled",
            idempotency_key="root-state-before-source-check:" + root_change,
            actor="test", expected_revision=1, expected_state="active",
        )

    calls = []
    async def must_not_read(_binding, **_kwargs):
        calls.append(True)
        raise AssertionError("ineligible child dependency must stop before L0 read")

    env["sources"].locate = must_not_read
    child = run(env["reads"].recall_source_status([{
        "memory_id": committed["child_id"], "revision": 1,
        "binding_ids": [committed["binding_id"]],
    }]))["items"][0]
    assert child["state"] == "dependency_changed"
    assert child["sources"] == {}
    assert calls == []


@pytest.mark.parametrize("change_kind", [
    "child_withdrawal", "child_revision", "root_withdrawal", "root_revision",
])
def test_recall_source_wait_rechecks_current_state_and_revision(tmp_path, change_kind):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env)
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
            "memory_id": committed["child_id"], "revision": 1,
            "binding_ids": [committed["binding_id"]],
        }]))
        await entered.wait()
        if change_kind == "child_withdrawal":
            env["authority"].change_memory_state(
                memory_id=committed["child_id"], state="archived", recall_policy="enabled",
                idempotency_key="archive-during-source-check", actor="test", expected_revision=1,
                expected_state="active",
            )
        elif change_kind == "child_revision":
            detail = await env["revisions"].detail(committed["child_id"], 1)
            document = copy.deepcopy(detail["l1"]["document"])
            document["narrative"]["title"] = "并发期间的新 child 修订"
            await env["revisions"].save(committed["child_id"], {
                "expected_revision": 1,
                "request_id": "revise-child-during-source-check",
                "document": document,
                "edit_note": "synthetic concurrent child revision",
            })
        elif change_kind == "root_withdrawal":
            env["authority"].change_memory_state(
                memory_id=committed["memory_id"], state="archived", recall_policy="enabled",
                idempotency_key="archive-root-during-source-check", actor="test", expected_revision=1,
                expected_state="active",
            )
        else:
            detail = await env["revisions"].detail(committed["memory_id"], 1)
            document = copy.deepcopy(detail["l1"]["document"])
            document["narrative"]["title"] = "并发期间的新 root 修订"
            await env["revisions"].save(committed["memory_id"], {
                "expected_revision": 1,
                "request_id": "revise-root-during-source-check",
                "document": document,
                "edit_note": "synthetic concurrent root revision",
            })
        release.set()
        return await request

    result = run(scenario())["items"][0]
    if change_kind.startswith("root_"):
        assert result["state"] == "dependency_changed"
    else:
        assert result["state"] == "memory_not_current_or_eligible"
    assert result["revision"] == 1


def _request(path, payload, *, headers=None):
    from starlette.requests import Request

    body = json.dumps(payload).encode("utf-8")
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    raw_headers = [(key.lower().encode("ascii"), value.encode("latin-1"))
                   for key, value in (headers or {}).items()]
    return Request({
        "type": "http", "method": "POST", "path": path, "headers": raw_headers,
        "query_string": b"", "path_params": {}, "server": ("testserver", 80),
        "client": ("testclient", 123), "scheme": "http",
    }, receive)


def test_recall_sources_api_keeps_existing_internal_auth_and_error_shape(monkeypatch):
    import server

    async def invalid_items(items):
        raise L1OwnerError("array_required", field_path="items")

    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "synthetic-recall-read-token")
    monkeypatch.setattr(server, "memory_authority_store", object())
    monkeypatch.setattr(server, "memory_commit_service", object())
    monkeypatch.setattr(server, "_memory_l1_reads", lambda: SimpleNamespace(recall_source_status=invalid_items))

    async def scenario():
        unauthorized = await server.api_memory_recall_sources(_request(
            "/api/memory-authority/recall-sources", {"items": []}
        ))
        authorized = await server.api_memory_recall_sources(_request(
            "/api/memory-authority/recall-sources", {"items": {}},
            headers={"Authorization": "Bearer synthetic-recall-read-token"},
        ))
        return unauthorized, authorized

    unauthorized, authorized = run(scenario())
    assert unauthorized.status_code == 401
    assert json.loads(unauthorized.body)["error"] == "unauthorized"
    assert authorized.status_code == 400
    assert json.loads(authorized.body) == {
        "error": {"code": "array_required", "field_path": "items"},
    }
