import asyncio
import copy

import pytest

from bucket_manager import BucketManager
from memory_authority import CommitStateError, IdempotencyConflict, MemoryAuthorityStore, RevisionConflict
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_l1_contract import L1ContractError
from memory_l1_schema import catalog_with_defaults, statement_template, scene_template, unknown
from memory_l1_service import L1OwnerError, MemoryL1RevisionService


def setup_service(tmp_path):
    config = {"buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state")}
    authority = MemoryAuthorityStore(config)
    manager = BucketManager(config)
    projection = BucketMemoryProjection(config, manager)
    commit = MemoryCommitService(authority, projection)
    catalog = catalog_with_defaults()
    service = MemoryL1RevisionService(authority, commit, manager, lambda: (catalog, "mc-test"))
    asyncio.run(commit.commit_memory(memory_id="m1", bucket_id="b1", expected_revision=0,
        body="完整的合成回忆。", metadata={"name": "标题", "owner_unrelated": {"keep": 7}},
        source_refs=["synthetic-source"], decision_source="synthetic", idempotency_key="seed", actor="test"))
    return service, authority, manager


def request_for(service, request_id="edit-1"):
    detail = asyncio.run(service.detail("m1"))
    return {"expected_revision": detail["memory"]["active_revision"], "request_id": request_id,
            "document": copy.deepcopy(detail["l1"]["document"]), "edit_note": "合成纠正"}


def test_legacy_can_be_edited_and_metadata_only_versions_do_not_overwrite_history(tmp_path):
    service, authority, _ = setup_service(tmp_path)
    original = asyncio.run(service.detail("m1"))
    assert original["l1"]["state"] == "legacy"
    payload = request_for(service)
    payload["document"]["retrieval"]["keywords"] = ["合成关键词"]
    saved = asyncio.run(service.save("m1", payload))
    assert saved["revision"] == 2 and saved["result"]["l1"]["state"] == "current"
    row = authority.get_memory_revision("m1", 2)
    assert row["metadata"]["owner_unrelated"] == {"keep": 7}
    assert row["body_sha256"] == authority.get_memory_revision("m1", 1)["body_sha256"]
    payload2 = request_for(service, "edit-2")
    payload2["document"]["retrieval"]["keywords"] = ["新的合成关键词"]
    second = asyncio.run(service.save("m1", payload2))
    assert second["l1_content_sha256"] != saved["l1_content_sha256"]
    assert asyncio.run(service.detail("m1", 2))["l1"]["document"]["retrieval"]["keywords"] == ["合成关键词"]
    assert asyncio.run(service.detail("m1", 1))["body"] == original["body"]
    assert authority.pending_outbox()[-1]["payload"]["memory_l1_content_sha256"] == second["l1_content_sha256"]


def test_old_request_replay_does_not_reapply_over_newer_version_and_bad_replay_conflicts(tmp_path):
    service, authority, _ = setup_service(tmp_path)
    first = request_for(service)
    saved = asyncio.run(service.save("m1", first))
    stale = {**first, "request_id": "stale"}
    with pytest.raises(RevisionConflict): asyncio.run(service.save("m1", stale))
    second = request_for(service, "edit-2")
    second["document"]["narrative"]["body"] += " 后来的补充。"
    asyncio.run(service.save("m1", second))
    replay = asyncio.run(service.save("m1", first))
    assert replay["replayed"] and replay["revision"] == saved["revision"]
    assert replay["result"]["current_revision"] == 3
    assert authority.get_memory("m1")["active_revision"] == 3
    different = copy.deepcopy(first)
    different["document"]["narrative"]["title"] = "不同请求内容"
    with pytest.raises(IdempotencyConflict): asyncio.run(service.save("m1", different))


def test_program_assigns_new_id_and_only_changed_statement_gets_new_revision(tmp_path):
    service, _, _ = setup_service(tmp_path)
    payload = request_for(service)
    stmt = statement_template(); stmt["content"] = "那次提出想抱抱。"
    stmt["classification"]["kind"] = unknown("known", "intention")
    stmt["classification"]["modality"] = unknown("known", "request")
    payload["document"]["statements"] = [stmt]
    first = asyncio.run(service.save("m1", payload))
    item = first["result"]["l1"]["document"]["statements"][0]
    assert item["statement_id"].startswith("st_") and item["statement_revision_id"]
    edit_title = request_for(service, "title")
    edit_title["document"]["narrative"]["title"] = "改标题"
    second = asyncio.run(service.save("m1", edit_title))
    assert second["result"]["l1"]["document"]["statements"][0]["statement_revision_id"] == item["statement_revision_id"]
    edit_statement = request_for(service, "change")
    edit_statement["document"]["statements"][0]["content"] = "那次请求暂停。"
    third = asyncio.run(service.save("m1", edit_statement))
    new = third["result"]["l1"]["document"]["statements"][0]
    assert new["statement_id"] == item["statement_id"] and new["statement_revision_id"] != item["statement_revision_id"]


def test_owner_cannot_forge_source_or_generation_bindings(tmp_path):
    service, authority, _ = setup_service(tmp_path)
    payload = request_for(service)
    payload["document"]["generation"]["origin"] = "first_pass_model"
    with pytest.raises(L1OwnerError, match="managed_field_changed"): asyncio.run(service.save("m1", payload))
    assert authority.get_memory("m1")["active_revision"] == 1


def test_projected_metadata_mismatch_prevents_finalize_and_same_request_can_recover(tmp_path):
    service, authority, manager = setup_service(tmp_path)
    payload = request_for(service)
    real_get = manager.get
    async def corrupted(bucket_id):
        result = await real_get(bucket_id)
        if result and "memory_l1" in result["metadata"]:
            result = copy.deepcopy(result)
            result["metadata"]["memory_l1"]["retrieval"]["keywords"] = ["tampered"]
        return result
    manager.get = corrupted
    with pytest.raises(L1ContractError, match="projected_semantic_mismatch"):
        asyncio.run(service.save("m1", payload))
    assert authority.get_memory("m1")["active_revision"] == 1
    manager.get = real_get
    recovered = asyncio.run(service.save("m1", payload))
    assert recovered["revision"] == 2 and recovered["replayed"]


def test_legacy_writer_cannot_rebind_old_semantics_to_a_changed_body(tmp_path):
    service, authority, _ = setup_service(tmp_path)
    asyncio.run(service.save("m1", request_for(service)))
    before = authority.get_memory_revision("m1", 2)
    with pytest.raises(CommitStateError, match="coordinated"):
        asyncio.run(service.commit.commit_memory(memory_id="m1", bucket_id="b1", expected_revision=2,
            body="Changed by a legacy body-only writer", metadata=before["metadata"], source_refs=before["source_refs"],
            decision_source="legacy", idempotency_key="legacy-write", actor="test"))
    assert authority.get_memory("m1")["active_revision"] == 2
    assert asyncio.run(service.detail("m1"))["body"] == "完整的合成回忆。"


def test_entity_references_are_bound_and_historical_attribute_meaning_is_preserved(tmp_path):
    service, _, _ = setup_service(tmp_path)
    catalog, config_revision = service.catalog_provider()
    catalog["attributes"] = [{"id": "companion", "value_type": "entity_ref", "scope": "any"}]
    payload = request_for(service)
    doc = payload["document"]
    doc["entities"] = [{"entity_id": "new:person", "entity_type": unknown("known", "assistant"), "name": "合成人物", "aliases": []}]
    doc["attributes"] = [{"definition_id": "companion", "value": unknown("known", "new:person"), "evidence_refs": []}]
    item = statement_template(); item["content"] = "这次和合成人物相处。"
    item["property"] = {"key": "companion", "value": unknown("known", "new:person"), "polarity": "affirmed", "evidence_refs": []}
    item["conditions"].update({"state": "known", "predicate": {"op": "eq", "field": "companion", "value": "new:person"}})
    doc["statements"] = [item]
    saved = asyncio.run(service.save("m1", payload))["result"]["l1"]["document"]
    ident = saved["entities"][0]["entity_id"]
    assert saved["attributes"][0]["value"]["value"] == ident
    assert saved["statements"][0]["property"]["value"]["value"] == ident
    assert saved["statements"][0]["conditions"]["predicate"]["value"] == ident
    # Changing the current directory must not reinterpret this saved record.
    catalog["attributes"][0] = {"id": "companion", "value_type": "boolean", "scope": "statement"}
    second = request_for(service, "keep-historical-meaning")
    second["document"]["narrative"]["title"] = "仅改标题"
    result = asyncio.run(service.save("m1", second))["result"]["l1"]["document"]
    assert result["generation"]["catalog_snapshot"]["attributes"][0]["value_type"] == "entity_ref"


def test_existing_child_machine_fields_are_rejected_not_silently_replaced(tmp_path):
    service, authority, _ = setup_service(tmp_path)
    payload = request_for(service)
    stmt = statement_template(); stmt["content"] = "合成陈述。"
    scene = scene_template(); scene["title"] = "合成场景"
    scene["statement_refs"] = [stmt["statement_id"]]
    payload["document"]["statements"] = [stmt]
    payload["document"]["scenes"] = [scene]
    payload["document"]["relations"] = [{"relation_id": "new:relation",
        "from": {"memory_id": None, "revision": "this_revision", "object_type": "statement", "object_id": stmt["statement_id"]},
        "to": {"memory_id": None, "revision": "this_revision", "object_type": "scene", "object_id": scene["scene_id"]},
        "kind": "same_event", "evidence_refs": [], "origin": "owner", "confirmation_state": "confirmed"}]
    asyncio.run(service.save("m1", payload))
    for module, field in (("statements", "statement_revision_id"), ("scenes", "scene_revision_id"), ("scenes", "generation"), ("relations", "origin")):
        edit = request_for(service, f"bad-{module}-{field}")
        if field == "generation": edit["document"][module][0][field]["actor"] = "forged"
        else: edit["document"][module][0][field] = "model" if field == "origin" else "forged"
        with pytest.raises(L1OwnerError) as exc:
            asyncio.run(service.save("m1", edit))
        assert exc.value.field_path == f"document.{module}[0].{field}"
        assert authority.get_memory("m1")["active_revision"] == 2
    malformed = request_for(service, "bad-entry")
    malformed["document"]["entry"] = []
    with pytest.raises(L1OwnerError) as exc:
        asyncio.run(service.save("m1", malformed))
    assert exc.value.field_path == "document.entry"
