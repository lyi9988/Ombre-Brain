import asyncio
import copy
from types import SimpleNamespace

import pytest

from memory_candidate_edit import save_owner_candidate
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_l1_ingestion import model_template
from memory_l1_schema import join_document, scene_template, statement_template, unknown
from memory_authority import RevisionConflict
from test_chat_memory_tool import envelope, service as chat_memory_service


def test_review_l1_edit_save_adopt_commit_and_revision_roundtrip(tmp_path):
    tool, _unused_projection = chat_memory_service(tmp_path, "review")
    engine = tool.engine
    authority = tool.store
    bucket_manager = __import__("bucket_manager").BucketManager(engine.config)
    tool.bucket_mgr = bucket_manager
    commit = MemoryCommitService(
        authority, BucketMemoryProjection(engine.config, bucket_manager=bucket_manager)
    )
    engine._memory_authority_service = lambda _bucket_manager: commit
    revisions = engine._memory_l1_revisions(bucket_manager)

    source_text = "主人希望把这段认真商量的合成经历记下来。"
    located_bindings = []

    async def locate(binding):
        located_bindings.append(copy.deepcopy(binding))
        return {"state": "available", "eligible_source": True, "reason": None}

    engine._memory_l1_sources = lambda: SimpleNamespace(locate=locate)

    document = model_template()
    document["narrative"].update(title="合成经历标题", body="完整的合成初始回忆正文。")
    document["entities"] = [{
        "entity_id": "new:entity",
        "entity_type": unknown("known", "assistant"),
        "name": "顾衍",
        "aliases": ["衍衍"],
    }]
    statement = statement_template()
    statement.pop("statement_revision_id")
    statement.update(content="主人希望保留这次合成经历。", evidence_refs=["source:current"])
    document["statements"] = [statement]
    scene = scene_template()
    for managed in ("scene_revision_id", "anchors", "generation"):
        scene.pop(managed)
    scene.update({
        "title": "合成经历场景",
        "statement_refs": ["new:statement"],
        "source_refs": ["source:current"],
        "source_ranges": [{"source_ref": "source:current", "start": 0, "end": len(source_text)}],
    })
    document["scenes"] = [scene]
    document["retrieval"] = {"keywords": ["合成经历"], "tags": ["初始标签"]}

    request = envelope()
    request["context"]["sources"][0]["text"] = source_text
    request["arguments"] = {"document": document, "kind": "key_event"}
    submitted = asyncio.run(tool.submit(request))
    candidate_id = submitted["candidate_id"]
    assert submitted["memory_status"] == "pending"
    assert authority.get_memory(candidate_id) is None

    original = authority.get_candidate(candidate_id)
    original_proposal = copy.deepcopy(original["proposal"])
    first_pass = original_proposal["metadata"]["memory_l1"]
    owner_document = join_document(original_proposal["proposed_body"], first_pass)
    owner_document["narrative"].update(title="主人审核后的标题", body="主人完整编辑后的第一版正文。")
    owner_document["entities"][0]["name"] = "主人确认的顾衍"
    owner_document["scenes"][0]["title"] = "主人审核后的场景"
    owner_document["statements"][0]["content"] = "主人审核后希望保留这段经历。"
    owner_document["retrieval"]["keywords"].append("主人保存的检索词")
    owner_document["retrieval"]["tags"].append("主人确认")

    edit = {"document": copy.deepcopy(owner_document)}
    saved = save_owner_candidate(
        authority, candidate_id, edit=edit, expected_revision=original["revision"],
        request_id="p2-owner-save-1", revision_service=revisions,
    )
    saved_again = save_owner_candidate(
        authority, candidate_id, edit=edit, expected_revision=original["revision"],
        request_id="p2-owner-save-1", revision_service=revisions,
    )
    assert saved_again == saved
    assert saved["revision"] == original["revision"] + 1
    assert saved["status"] == "pending"
    assert authority.get_memory(candidate_id) is None

    with pytest.raises(RevisionConflict):
        save_owner_candidate(
            authority, candidate_id, edit=edit, expected_revision=original["revision"],
            request_id="p2-owner-save-stale", revision_service=revisions,
        )
    readback_candidate = authority.get_candidate(candidate_id)
    saved_document = join_document(
        readback_candidate["proposal"]["proposed_body"],
        readback_candidate["proposal"]["metadata"]["memory_l1"],
    )
    assert readback_candidate["revision"] == saved["revision"]
    assert readback_candidate["status"] == "pending"
    assert saved_document == join_document(saved["proposal"]["proposed_body"], saved["proposal"]["metadata"]["memory_l1"])
    assert saved_document["generation"]["origin"] == "owner_edit"
    assert saved_document["evidence_bindings"] == first_pass["evidence_bindings"]
    for key in ("source_refs", "source_status", "original_excerpt", "owner_explicit"):
        assert readback_candidate["proposal"][key] == original_proposal[key]
    assert readback_candidate["proposal"]["metadata"]["runtime_context"] == original_proposal["metadata"]["runtime_context"]

    async def adopt_and_read():
        accepted = await engine.confirm_daily_chat_memory(
            [candidate_id], bucket_manager, request_id="p2-review-accept-1",
            expected_revisions={candidate_id: readback_candidate["revision"]},
        )
        assert accepted["created"] == 1
        assert accepted["results"][0]["status"] == "created"
        committed_candidate = authority.get_candidate(candidate_id)
        assert committed_candidate["status"] == "committed"
        assert committed_candidate["proposal"]["proposed_body"] == saved_document["narrative"]["body"]

        detail = await revisions.detail(candidate_id)
        assert detail["memory"]["active_revision"] == 1
        assert detail["body"] == saved_document["narrative"]["body"]
        assert detail["l1"]["state"] == "current"
        assert detail["l1"]["document"] == saved_document
        record = authority.get_memory_revision(candidate_id, 1)
        assert record["metadata"]["memory_l1"] == readback_candidate["proposal"]["metadata"]["memory_l1"]
        assert record["source_refs"] == original_proposal["source_refs"]
        assert record["metadata"]["runtime_context"] == original_proposal["metadata"]["runtime_context"]
        assert located_bindings == saved_document["evidence_bindings"]

        accepted_replay = await engine.confirm_daily_chat_memory(
            [candidate_id], bucket_manager, request_id="p2-review-accept-1",
            expected_revisions={candidate_id: readback_candidate["revision"]},
        )
        assert accepted_replay["idempotent_replay"] is True
        assert authority.get_candidate(candidate_id)["revision"] == committed_candidate["revision"]
        assert len(authority.list_memory_revisions(candidate_id)) == 1
        return detail

    first_detail = asyncio.run(adopt_and_read())

    second_document = copy.deepcopy(first_detail["l1"]["document"])
    second_document["narrative"]["body"] = "主人再次修订后的完整回忆正文。"
    second_document["retrieval"]["keywords"].append("二次修订词")
    second_request = {
        "expected_revision": 1,
        "request_id": "p2-memory-revision-2",
        "document": second_document,
        "edit_note": "合成二次修订",
    }
    second_saved = asyncio.run(revisions.save(candidate_id, copy.deepcopy(second_request)))
    assert second_saved["revision"] == 2
    second_replay = asyncio.run(revisions.save(candidate_id, copy.deepcopy(second_request)))
    assert second_replay["replayed"] is True
    assert second_replay["revision"] == 2

    stale_request = copy.deepcopy(second_request)
    stale_request["request_id"] = "p2-memory-revision-stale"
    stale_request["document"]["narrative"]["body"] = "过期页面不应覆盖新版本。"
    with pytest.raises(RevisionConflict):
        asyncio.run(revisions.save(candidate_id, stale_request))

    latest = asyncio.run(revisions.detail(candidate_id))
    historical = asyncio.run(revisions.detail(candidate_id, 1))
    latest_document = latest["l1"]["document"]
    assert latest["memory"]["active_revision"] == 2
    assert latest["body"] == "主人再次修订后的完整回忆正文。"
    assert latest["l1"]["state"] == "current"
    assert latest_document == second_saved["result"]["l1"]["document"]
    assert historical["body"] == saved_document["narrative"]["body"]
    assert historical["l1"]["document"] == saved_document
    for key in ("schema_version", "entry", "entities", "scenes", "statements", "evidence_bindings", "relations", "attributes"):
        assert latest_document[key] == saved_document[key]
    assert latest_document["narrative"]["title"] == saved_document["narrative"]["title"]
    assert latest_document["retrieval"]["tags"] == saved_document["retrieval"]["tags"]
    assert latest_document["retrieval"]["keywords"] == saved_document["retrieval"]["keywords"] + ["二次修订词"]
    assert authority.get_memory_revision(candidate_id, 2)["source_refs"] == original_proposal["source_refs"]
    assert len(authority.list_memory_revisions(candidate_id)) == 2
