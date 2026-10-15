import asyncio
import copy

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore, MemoryNotFound
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_l1_contract import L1ContractError
from memory_l0_sources import MemoryL0Sources
from memory_l1_index import MemoryL1Index, scene_memory_id
from memory_l1_reads import MemoryL1Reads, read_query
from memory_l1_schema import catalog_with_defaults, document_template, scene_template, split_document, statement_template
from memory_l1_service import L1OwnerError, MemoryL1RevisionService


def setup_memory(tmp_path):
    config = {"buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state")}
    authority = MemoryAuthorityStore(config)
    manager = BucketManager(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, manager))
    catalog = catalog_with_defaults()
    revisions = MemoryL1RevisionService(authority, commit, manager, lambda: (catalog, "catalog-p2b"))
    reads = MemoryL1Reads(revisions, MemoryL0Sources(authority))
    return authority, manager, commit, revisions, reads, catalog


def statement(statement_id, content, revision_id):
    item = statement_template()
    item.update(statement_id=statement_id, statement_revision_id=revision_id, content=content)
    return item


def root_document(catalog):
    document = document_template("完整的合成日记全文。", "合成日记", catalog=catalog, catalog_revision="catalog-p2b")
    document["entry"]["form"] = "diary_full"
    document["statements"] = [
        statement("stmt-a", "陈述 A 原文。", "stmt-a-r1"),
        statement("stmt-b", "陈述 B 原文。", "stmt-b-r1"),
        statement("stmt-unlinked", "未关联到场景的陈述。", "stmt-unlinked-r1"),
    ]
    scene = scene_template(catalog, "catalog-p2b")
    scene.update(scene_id="scene-main", scene_revision_id="scene-r1", title="主场景",
                 statement_refs=["stmt-b", "stmt-a"])
    document["scenes"] = [scene]
    return document


def commit_l1_fixture(commit, memory_id, bucket_id, expected_revision, document, idempotency_key):
    """Synthetic fixture helper: commits a fully validated L1 through real temp Authority/Commit/Bucket."""
    body, semantic = split_document(document)
    return asyncio.run(commit.commit_memory(
        memory_id=memory_id, bucket_id=bucket_id, expected_revision=expected_revision,
        body=body, metadata={"name": semantic["narrative"]["title"], "memory_l1": semantic},
        source_refs=["synthetic:p2b-fixture"], decision_source="synthetic_p2b_fixture",
        idempotency_key=idempotency_key, actor="test", l1_write_intent=True,
    ))


def test_json_only_revisions_are_indexed_isolated_and_rebuildable(tmp_path):
    authority, _, commit, revisions, reads, catalog = setup_memory(tmp_path)
    document = root_document(catalog)
    first = commit_l1_fixture(commit, "root", "bucket-root", 0, document, "root-r1")
    assert first["revision"] == 1

    index = MemoryL1Index(authority)
    before_memory = authority.get_memory("root")
    before_revision = authority.get_memory_revision("root", 1)
    before_outbox = authority.pending_outbox()
    conn = authority._connect()
    try:
        conn.execute("DELETE FROM memory_l1_index_revisions WHERE memory_id=? AND revision=?", ("root", 1))
        conn.commit()
    finally:
        conn.close()

    # Rebuilding is pinned to the requested authoritative revision and must not create a fact revision.
    assert index.ensure("root", 1) is True
    assert index.identity("root", 1)["content_digest"] == before_revision["metadata"]["memory_l1_content_sha256"]
    assert authority.get_memory("root") == before_memory
    assert authority.get_memory_revision("root", 1) == before_revision
    assert authority.pending_outbox() == before_outbox

    page_v1 = reads.page("root", 1, "statements", offset=0, limit=2)
    assert [item["statement_id"] for item in page_v1["items"]] == ["stmt-a", "stmt-b"]
    filtered = reads.page("root", 1, "statements", offset=0, limit=10, scene_id="scene-main")
    assert [item["statement_id"] for item in filtered["items"]] == ["stmt-b", "stmt-a"]
    assert filtered["page"]["total"] == 2
    with pytest.raises(MemoryNotFound):
        reads.page("root", 1, "statements", scene_id="scene-does-not-exist")

    # This is a JSON-only root revision: statement semantics change while the full narrative body stays fixed.
    draft = copy.deepcopy(asyncio.run(revisions.detail("root", 1))["l1"]["document"])
    draft["statements"][0]["content"] = "陈述 A 的新语义版本。"
    draft["retrieval"]["keywords"] = ["仅JSON更新"]
    saved = asyncio.run(revisions.save("root", {
        "expected_revision": 1, "request_id": "root-json-only-r2", "document": draft,
    }))
    assert saved["revision"] == 2
    second = authority.get_memory_revision("root", 2)
    assert second["body_sha256"] == before_revision["body_sha256"]
    assert second["metadata"]["memory_l1_semantic_sha256"] != before_revision["metadata"]["memory_l1_semantic_sha256"]
    assert index.identity("root", 2)["semantic_digest"] == second["metadata"]["memory_l1_semantic_sha256"]

    page_v2 = reads.page("root", 2, "statements", offset=0, limit=2)
    assert [item["content"] for item in page_v1["items"]] == ["陈述 A 原文。", "陈述 B 原文。"]
    assert [item["content"] for item in page_v2["items"]] == ["陈述 A 的新语义版本。", "陈述 B 原文。"]
    later_v2 = reads.page("root", 2, "statements", offset=2, limit=2)
    assert [item["statement_id"] for item in later_v2["items"]] == ["stmt-unlinked"]
    assert later_v2["page"]["total"] == 3


def test_child_references_root_statements_and_stays_bound_to_its_root_revision(tmp_path):
    authority, _, commit, revisions, reads, catalog = setup_memory(tmp_path)
    root = root_document(catalog)
    commit_l1_fixture(commit, "root", "bucket-root", 0, root, "root-r1")

    # Synthetic child fixture commits a legal L1 directly through temp Authority/Commit/Bucket.
    # It stands in for a future scene adapter and does not claim that production generation is implemented.
    child = document_template("只属于场景的连贯正文。", "主场景正文", catalog=catalog, catalog_revision="catalog-p2b")
    child["entry"].update(
        form="diary_scene", parent_ref={"memory_id": "root", "revision": 1},
        derived_from_revision=1, scene_ref={"scene_id": "scene-main", "scene_revision_id": "scene-r1"},
        shared_statement_refs=["stmt-b", "stmt-a"],
    )
    child_id = scene_memory_id("root", "scene-main")
    commit_l1_fixture(commit, child_id, "bucket-child", 0, child, "child-r1")
    child_record = authority.get_memory_revision(child_id, 1)
    assert child_record["metadata"]["memory_l1"]["statements"] == []
    assert child_record["metadata"]["memory_l1"]["entry"]["shared_statement_refs"] == ["stmt-b", "stmt-a"]

    old_scene = asyncio.run(reads.scene("root", 1, "scene-main"))
    assert old_scene["child"]["state"] == "ready"
    assert [item["content"] for item in old_scene["shared_statements"]] == ["陈述 B 原文。", "陈述 A 原文。"]
    assert old_scene["child"]["detail"]["requested_revision"] == 1

    draft = copy.deepcopy(asyncio.run(revisions.detail("root", 1))["l1"]["document"])
    draft["scenes"][0]["title"] = "更新后的主场景"
    unprepared = scene_template(catalog, "catalog-p2b")
    unprepared.update(scene_id="new:scene", title="尚未准备的场景", statement_refs=["stmt-unlinked"])
    draft["scenes"].append(unprepared)
    saved = asyncio.run(revisions.save("root", {
        "expected_revision": 1, "request_id": "root-scenes-r2", "document": draft,
    }))
    assert saved["revision"] == 2
    new_root = asyncio.run(revisions.detail("root", 2))["l1"]["document"]
    old_scene_r2 = next(item for item in new_root["scenes"] if item["scene_id"] == "scene-main")
    assert old_scene_r2["scene_revision_id"] != "scene-r1"
    new_scene = next(item for item in new_root["scenes"] if item["title"] == "尚未准备的场景")

    # The old child continues to read its immutable root revision; it is never rebound to the latest root.
    still_old = asyncio.run(reads.scene("root", 1, "scene-main"))
    assert still_old["child"]["state"] == "ready"
    assert still_old["child"]["detail"]["requested_revision"] == 1
    revised = reads.page("root", 2, "scenes", limit=10)
    states = {item["scene_id"]: item["child"]["state"] for item in revised["items"]}
    assert states["scene-main"] == "stale"
    assert states[new_scene["scene_id"]] == "not_prepared"
    assert authority.get_memory(child_id)["active_revision"] == 1


def test_scene_with_no_initial_claims_still_routes_new_claims_to_its_root(tmp_path):
    authority, _, commit, revisions, _, catalog = setup_memory(tmp_path)
    root = root_document(catalog)
    root["statements"] = []
    root["scenes"][0]["statement_refs"] = []
    commit_l1_fixture(commit, "root", "root-bucket", 0, root, "root-empty-claims")
    child = document_template("完整的场景叙述仍可保存和编辑。", "场景", catalog=catalog)
    child["entry"].update(form="diary_scene", parent_ref={"memory_id": "root", "revision": 1},
        scene_ref={"scene_id": "scene-main", "scene_revision_id": "scene-r1"}, shared_statement_refs=[])
    child_id = scene_memory_id("root", "scene-main")
    commit_l1_fixture(commit, child_id, "child-bucket", 0, child, "child-empty-claims")
    draft = copy.deepcopy(asyncio.run(revisions.detail(child_id))["l1"]["document"])
    draft["statements"] = [statement("new:claim", "新的陈述应加入整篇共享陈述。", None)]
    with pytest.raises(L1ContractError, match="shared_statements_must_not_be_duplicated"):
        asyncio.run(revisions.save(child_id, {"expected_revision": 1, "request_id": "child-claim", "document": draft}))
    assert authority.get_memory(child_id)["active_revision"] == 1
    assert authority.get_memory("root")["active_revision"] == 1


class DuplicateQuery:
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __iter__(self):
        return iter(dict.fromkeys(key for key, _ in self._pairs))

    def get(self, key):
        values = [value for name, value in self._pairs if name == key]
        return values[-1] if values else None

    def getlist(self, key):
        return [value for name, value in self._pairs if name == key]


@pytest.mark.parametrize("query,resource,code", [
    ({"revision": "1", "unknown": "x"}, "scenes", "unknown_query_field"),
    ({"revision": "0"}, "scenes", "query_out_of_range"),
    ({"revision": "one"}, "scenes", "integer_required"),
    ({"revision": "１"}, "scenes", "integer_required"),
    ({"revision": "1", "limit": "201"}, "scenes", "query_out_of_range"),
    ({"revision": "1", "start": "0", "end": "0"}, "source", "invalid_codepoint_range"),
    ({"revision": "1", "start": "4", "end": "3"}, "source", "invalid_codepoint_range"),
    ({"revision": "1", "start": "1"}, "source", "required_query_field"),
])
def test_read_query_rejects_unknown_bad_revision_ranges_and_unclamped_page_size(query, resource, code):
    with pytest.raises(L1OwnerError) as exc:
        read_query(query, resource)
    assert exc.value.code == code


def test_read_query_rejects_duplicate_fields():
    with pytest.raises(L1OwnerError) as exc:
        read_query(DuplicateQuery([("revision", "1"), ("revision", "2")]), "scenes")
    assert exc.value.code == "duplicate_query_field"


def test_owner_read_page_does_not_silently_clamp_limits(tmp_path):
    authority, _, commit, _, reads, catalog = setup_memory(tmp_path)
    commit_l1_fixture(commit, "root", "bucket-root", 0, root_document(catalog), "root-r1")
    with pytest.raises(L1OwnerError) as exc:
        reads.page("root", 1, "statements", limit=201)
    assert exc.value.code == "invalid_pagination"
