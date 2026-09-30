import asyncio
import hashlib
import sqlite3

import pytest

from memory_authority import MemoryAuthorityStore
from memory_projection_worker import MemoryProjectionWorker


class FakeBucketManager:
    def __init__(self, bucket):
        self.bucket = dict(bucket)
        self.get_calls = []
        self.permanent_dir = ""
        self.dynamic_dir = ""

    async def get(self, bucket_id):
        self.get_calls.append(str(bucket_id))
        if str(bucket_id) != str(self.bucket.get("id")):
            return None
        return dict(self.bucket)


class FakeEmbedding:
    enabled = True

    def __init__(self):
        self.generate_calls = []
        self.delete_calls = []
        self.fail_generate = False
        self.return_false = False
        self.fail_delete = False
        self.stored = {}

    async def generate_and_store(self, memory_id, text):
        if self.fail_generate:
            raise RuntimeError("embedding unavailable")
        if self.return_false:
            return False
        self.generate_calls.append((memory_id, text))
        self.stored[memory_id] = [0.1, 0.2]
        return {"memory_id": memory_id, "text": text}

    async def get_embedding(self, bucket_id):
        return self.stored.get(bucket_id)

    def delete_embedding(self, memory_id):
        if self.fail_delete:
            raise RuntimeError("embedding delete unavailable")
        self.delete_calls.append(memory_id)


class RevisionAwareFakeEmbedding(FakeEmbedding):
    provider = "test-provider"
    model = "test-embedding-model"
    dimension = 2
    base_url = "https://private.example.test/v1?key=do-not-store"

    def __init__(self):
        super().__init__()
        self.revision_calls = []
        self.config_snapshots = []

    def preparation_hash(self, *, kind, instruction=None):
        assert kind == "document"
        if instruction is None:
            return "d" * 64
        return hashlib.sha256(instruction.encode("utf-8")).hexdigest()

    async def generate_and_store(
        self, memory_id, text, *, source_revision=None, source_sha256=None,
        source_unit_id=None, parent_memory_id=None, document_instruction=None,
        config_snapshot=None,
    ):
        self.config_snapshots.append(config_snapshot)
        self.revision_calls.append({
            "legacy_bucket_id": memory_id,
            "text": text,
            "source_revision": source_revision,
            "source_sha256": source_sha256,
            "source_unit_id": source_unit_id,
            "parent_memory_id": parent_memory_id,
            "document_instruction": document_instruction,
            "config_snapshot": config_snapshot,
        })
        if self.fail_generate:
            raise RuntimeError("embedding unavailable")
        if self.return_false:
            return False
        self.generate_calls.append((source_unit_id, text))
        self.stored[source_unit_id] = [0.1, 0.2]
        return True


class SnapshotRevisionAwareFakeEmbedding(RevisionAwareFakeEmbedding):
    def __init__(self):
        super().__init__()
        self.snapshot_factory_calls = []
        self.created_snapshot = None

    def _query_config_snapshot(self, *, document_instruction=None):
        self.snapshot_factory_calls.append(document_instruction)
        self.created_snapshot = {
            "base_url": "https://snapshot-provider.example/v1?api_key=fake-only",
            "model": "snapshot-embedding-model",
            "api_key": "fake-only-key",
            "enabled": True,
            "max_chars": 6000,
            "document_instruction": document_instruction,
            "document_preparation": "e" * 64,
        }
        return self.created_snapshot


class FakePromptPlanMirror:
    def __init__(self, body):
        self.body = body
        self.resolve_calls = []

    def resolve_text(
        self, *, scope, source_id, live_body, identity_id, conversation_id,
    ):
        self.resolve_calls.append({
            "scope": scope,
            "source_id": source_id,
            "live_body": live_body,
            "identity_id": identity_id,
            "conversation_id": conversation_id,
        })
        return self.body, {"source_revision": "test-revision"}


class FakeMoment:
    def __init__(self):
        self.upsert_calls = []
        self.delete_calls = []
        self.fail_upsert = False
        self.fail_delete = False

    def upsert_bucket(self, bucket):
        if self.fail_upsert:
            raise RuntimeError("moment store unavailable")
        self.upsert_calls.append(dict(bucket))
        return [{"moment_id": f"moment:{bucket['id']}"}]

    def delete_bucket(self, memory_id):
        if self.fail_delete:
            raise RuntimeError("moment delete unavailable")
        self.delete_calls.append(memory_id)


class FakeNode:
    def __init__(self):
        self.upsert_calls = []
        self.delete_calls = []
        self.fail_upsert = False
        self.fail_delete = False

    def upsert_bucket(self, bucket):
        if self.fail_upsert:
            raise RuntimeError("node store unavailable")
        self.upsert_calls.append(dict(bucket))
        return {"node_id": bucket["id"]}

    def delete(self, memory_id):
        if self.fail_delete:
            raise RuntimeError("node delete unavailable")
        self.delete_calls.append(memory_id)


class FakeEntity:
    def __init__(self):
        self.replace_calls = []
        self.delete_calls = []
        self.fail_replace = False
        self.fail_delete = False

    def replace_bucket_edges(self, memory_id, edges):
        if self.fail_replace:
            raise RuntimeError("entity edge store unavailable")
        self.replace_calls.append((memory_id, list(edges)))
        return list(edges)

    def delete_for_bucket(self, memory_id):
        if self.fail_delete:
            raise RuntimeError("entity edge delete unavailable")
        self.delete_calls.append(memory_id)


class FakeWordMap:
    enabled = True

    def __init__(self):
        self.upsert_calls = []
        self.fail_upsert = False

    def upsert_bucket(self, bucket):
        if self.fail_upsert:
            raise RuntimeError("word map unavailable")
        self.upsert_calls.append(dict(bucket))
        return {"bucket_id": bucket["id"]}


def _authority(tmp_path):
    return MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})


def _bucket(memory_id, body):
    return {
        "id": memory_id,
        "content": body,
        "metadata": {
            "name": "咖啡偏好",
            "domain": ["偏好"],
            "tags": ["preference"],
            "comments": [],
        },
    }


def _commit_memory(
    authority,
    memory_id="memory-1",
    *,
    body="主人喜欢清淡的手冲咖啡。",
    memory_state="active",
    recall_policy="enabled",
):
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id=memory_id,
        bucket_id=memory_id,
        expected_revision=0,
        body_sha256=body_sha,
        snapshot_path=f"revisions/{memory_id}/1.md",
        metadata={"name": "咖啡偏好"},
        source_refs=[f"test:{memory_id}"],
        decision_source="test",
        idempotency_key=f"commit:{memory_id}",
        actor="test",
        memory_state=memory_state,
        recall_policy=recall_policy,
    )
    authority.record_body_written(
        prepared["operation_id"], observed_body_sha256=body_sha
    )
    return authority.finalize_memory_commit(prepared["operation_id"])


def _worker(
    authority, tmp_path, memory_id="memory-1",
    body="主人喜欢清淡的手冲咖啡。", embedding=None,
    prompt_plan_mirror=None,
):
    bucket_manager = FakeBucketManager(_bucket(memory_id, body))
    bucket_manager.dynamic_dir = str(tmp_path / "dynamic")
    bucket_manager.permanent_dir = str(tmp_path / "permanent")
    (tmp_path / "dynamic").mkdir(exist_ok=True)
    (tmp_path / "permanent").mkdir(exist_ok=True)
    bucket_manager.bucket["path"] = str(tmp_path / "dynamic" / f"{memory_id}.md")
    embedding = embedding or FakeEmbedding()
    moments = FakeMoment()
    node = FakeNode()
    entity = FakeEntity()
    word_map = FakeWordMap()
    worker = MemoryProjectionWorker(
        config={
            "state_dir": str(tmp_path / "state"),
            "identity": {
                "ai_name": "顾衍",
                "user_name": "ZJR",
                "user_aliases": ["主人"],
            },
        },
        authority=authority,
        bucket_manager=bucket_manager,
        embedding_engine=embedding,
        moment_store=moments,
        node_store=node,
        entity_edge_store=entity,
        word_map_store=word_map,
        prompt_plan_mirror=prompt_plan_mirror,
    )
    return worker, bucket_manager, embedding, moments, node, entity, word_map


def _outbox_row(authority, event_id):
    rows = [row for row in authority.pending_outbox() if row["event_id"] == event_id]
    if rows:
        return rows[0]
    conn = sqlite3.connect(authority.path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM outbox WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def test_revision_outbox_all_projectors_projected_and_status_has_exact_revision_and_hash(tmp_path):
    authority = _authority(tmp_path)
    commit = _commit_memory(authority)
    worker, bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority, tmp_path
    )

    result = asyncio.run(worker.run_once())

    assert result["claimed"] == 1
    assert result["projected"] == 1
    assert result["degraded"] == 0
    assert result["results"][0]["status"] == "projected"
    assert bucket_manager.get_calls == ["memory-1"]
    assert len(embedding.generate_calls) == 1
    assert [item["id"] for item in moments.upsert_calls] == ["memory-1"]
    assert [item["id"] for item in node.upsert_calls] == ["memory-1"]
    assert [item[0] for item in entity.replace_calls] == ["memory-1"]
    assert [item["id"] for item in word_map.upsert_calls] == ["memory-1"]

    outbox = _outbox_row(authority, commit["outbox_event_id"])
    assert outbox["status"] == "projected"
    assert outbox["attempts"] == 1

    expected_sha = commit["body_sha256"]
    statuses = authority.list_projection_status("memory-1", revision=1)
    assert {item["projector"] for item in statuses} == {
        "embedding",
        "moments",
        "memory_node",
        "entity_edges",
        "word_map",
        "identity_semantics",
        "memory_edges",
    }
    assert all(item["memory_revision"] == 1 for item in statuses)
    assert all(item["source_sha256"] == expected_sha for item in statuses)
    assert {item["status"] for item in statuses if item["projector"] in {
        "embedding", "moments", "memory_node", "entity_edges", "word_map"
    }} == {"projected"}
    identity_status = next(
        item for item in statuses if item["projector"] == "identity_semantics"
    )
    assert identity_status["status"] == "projected"
    assert identity_status["details"]["authority"] == "memory_authority"


def test_embedding_projection_carries_committed_revision_and_content_unit_metadata(tmp_path):
    authority = _authority(tmp_path)
    commit = _commit_memory(authority)
    embedding = RevisionAwareFakeEmbedding()
    worker, _bucket_manager, _embedding, *_rest = _worker(
        authority, tmp_path, embedding=embedding,
    )

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert len(embedding.revision_calls) == 2
    assert all(call["legacy_bucket_id"] == "memory-1" for call in embedding.revision_calls)
    assert all(call["source_revision"] == 1 for call in embedding.revision_calls)
    assert all(call["source_sha256"] == commit["body_sha256"] for call in embedding.revision_calls)
    assert all(call["parent_memory_id"] == "memory-1" for call in embedding.revision_calls)
    unit_ids = {call["source_unit_id"] for call in embedding.revision_calls}
    assert "memory-1" in unit_ids
    assert len(unit_ids) == 2
    status = next(item for item in authority.list_projection_status(
        "memory-1", revision=1
    ) if item["projector"] == "embedding")
    details = status["details"]
    assert details["source_memory_id"] == "memory-1"
    assert details["source_revision"] == 1
    assert details["source_sha256"] == commit["body_sha256"]
    assert details["source_unit_count"] == 2
    assert details["provider"] == "test-provider"
    assert details["model"] == "test-embedding-model"
    assert details["dimension"] == 2
    assert details["preparation_sha256"] == "d" * 64
    assert details["metadata_complete"] is True
    assert details["coverage_status"] == "current"
    assert details["outdated"] is False
    assert "do-not-store" not in str(details)
    assert "清淡的手冲咖啡" not in str(details)


def test_embedding_document_preparation_uses_prompt_mirror_snapshot(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    document_prompt = "Encode committed memory passages consistently."
    prompt_mirror = FakePromptPlanMirror(document_prompt)
    embedding = RevisionAwareFakeEmbedding()
    worker, *_ = _worker(
        authority,
        tmp_path,
        embedding=embedding,
        prompt_plan_mirror=prompt_mirror,
    )

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert prompt_mirror.resolve_calls == [{
        "scope": "memory.embedding_document",
        "source_id": "ombre.memory_embedding_document_prep_prompt",
        "live_body": "",
        "identity_id": "jiajia-main",
        "conversation_id": "",
    }]
    assert len(embedding.revision_calls) == 2
    assert all(
        call["document_instruction"] == document_prompt
        for call in embedding.revision_calls
    )
    embedding_status = next(item for item in authority.list_projection_status(
        "memory-1", revision=1
    ) if item["projector"] == "embedding")
    assert embedding_status["details"]["preparation_sha256"] == hashlib.sha256(
        document_prompt.encode("utf-8")
    ).hexdigest()


def test_projection_freezes_engine_snapshot_for_all_units_without_persisting_key(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    document_prompt = "Use the same document preparation for this memory."
    prompt_mirror = FakePromptPlanMirror(document_prompt)
    embedding = SnapshotRevisionAwareFakeEmbedding()
    worker, *_ = _worker(
        authority,
        tmp_path,
        embedding=embedding,
        prompt_plan_mirror=prompt_mirror,
    )

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert embedding.snapshot_factory_calls == [document_prompt]
    assert len(embedding.config_snapshots) == 2
    assert embedding.config_snapshots[0] is embedding.config_snapshots[1]
    assert all(
        call["document_instruction"] == document_prompt
        for call in embedding.revision_calls
    )
    details = next(item for item in authority.list_projection_status(
        "memory-1", revision=1
    ) if item["projector"] == "embedding")["details"]
    assert details["provider"] == "snapshot-provider.example"
    assert details["model"] == "snapshot-embedding-model"
    assert details["preparation_sha256"] == "e" * 64
    assert "api_key" not in str(details)
    assert "fake-only-key" not in str(details)


def test_legacy_embedding_double_keeps_two_argument_contract(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    worker, _bucket_manager, embedding, *_rest = _worker(authority, tmp_path)

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert len(embedding.generate_calls) == 1
    status = next(item for item in authority.list_projection_status(
        "memory-1", revision=1
    ) if item["projector"] == "embedding")
    assert status["details"]["revision_metadata_supported"] is False
    assert status["details"]["coverage_status"] == "incomplete"
    assert status["details"]["outdated"] is True


def test_one_projector_failure_degrades_event_but_other_projectors_continue(tmp_path):
    authority = _authority(tmp_path)
    commit = _commit_memory(authority)
    worker, _bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority, tmp_path
    )
    node.fail_upsert = True

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 0
    assert result["degraded"] == 1
    projection = result["results"][0]
    assert projection["status"] == "degraded"
    by_projector = {item["projector"]: item for item in projection["projections"]}
    assert by_projector["memory_node"]["status"] == "degraded"
    assert by_projector["memory_node"]["details"]["error"] == "RuntimeError"
    assert by_projector["embedding"]["status"] == "projected"
    assert by_projector["moments"]["status"] == "projected"
    assert by_projector["entity_edges"]["status"] == "projected"
    assert by_projector["word_map"]["status"] == "projected"
    assert len(embedding.generate_calls) == 1
    assert len(moments.upsert_calls) == 1
    assert len(node.upsert_calls) == 0
    assert len(entity.replace_calls) == 1
    assert len(word_map.upsert_calls) == 1

    outbox = _outbox_row(authority, commit["outbox_event_id"])
    assert outbox["status"] == "degraded"
    assert outbox["attempts"] == 1
    assert outbox["last_error"] == "memory_node:failed"


def test_degraded_event_is_claimed_again_and_projects_after_retry(tmp_path):
    authority = _authority(tmp_path)
    commit = _commit_memory(authority)
    worker, _bucket_manager, _embedding, _moments, node, _entity, _word_map = _worker(
        authority, tmp_path
    )
    node.fail_upsert = True
    first = asyncio.run(worker.run_once())
    assert first["degraded"] == 1

    node.fail_upsert = False
    second = asyncio.run(worker.run_once())

    assert second["claimed"] == 1
    assert second["projected"] == 1
    assert second["degraded"] == 0
    assert second["results"][0]["status"] == "projected"
    outbox = _outbox_row(authority, commit["outbox_event_id"])
    assert outbox["status"] == "projected"
    assert outbox["attempts"] == 2
    statuses = authority.list_projection_status("memory-1", revision=1)
    assert next(item for item in statuses if item["projector"] == "memory_node")["status"] == "projected"


@pytest.mark.parametrize(
    ("memory_state", "recall_policy"),
    [("tombstoned", "enabled"), ("active", "disabled")],
    ids=["tombstoned", "recall-disabled"],
)
def test_tombstoned_or_disabled_memory_deletes_indexes(
    tmp_path, memory_state, recall_policy
):
    authority = _authority(tmp_path)
    commit = _commit_memory(
        authority,
        memory_state=memory_state,
        recall_policy=recall_policy,
    )
    worker, bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority, tmp_path
    )

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert result["degraded"] == 0
    assert bucket_manager.get_calls == []
    assert embedding.delete_calls == ["memory-1"]
    assert moments.delete_calls == ["memory-1"]
    assert node.delete_calls == ["memory-1"]
    assert entity.delete_calls == ["memory-1"]
    assert word_map.upsert_calls == []
    by_projector = {item["projector"]: item for item in result["results"][0]["projections"]}
    for projector in ("embedding", "moments", "memory_node", "entity_edges"):
        assert by_projector[projector]["status"] == "deleted"
    assert by_projector["word_map"]["status"] == "pending_rebuild"
    assert by_projector["identity_semantics"]["status"] == "projected"
    outbox = _outbox_row(authority, commit["outbox_event_id"])
    assert outbox["status"] == "projected"


def test_stale_processing_event_is_recovered_then_projected(tmp_path):
    authority = _authority(tmp_path)
    commit = _commit_memory(authority)
    claimed = authority.claim_outbox(limit=1)
    assert claimed[0]["event_id"] == commit["outbox_event_id"]
    assert claimed[0]["status"] == "processing"

    conn = sqlite3.connect(authority.path)
    conn.execute(
        "UPDATE outbox SET updated_at=? WHERE event_id=?",
        ("2000-01-01T00:00:00+00:00", commit["outbox_event_id"]),
    )
    conn.commit()
    conn.close()

    worker, _bucket_manager, _embedding, _moments, _node, _entity, _word_map = _worker(
        authority, tmp_path
    )
    result = asyncio.run(worker.run_once())

    assert result["recovered_stale"] == 1
    assert result["claimed"] == 1
    assert result["projected"] == 1
    outbox = _outbox_row(authority, commit["outbox_event_id"])
    assert outbox["status"] == "projected"
    assert outbox["attempts"] == 1


def test_pending_repair_is_bounded_and_marks_verified_indexes_once(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    memory = authority.get_memory("memory-1")
    for projector in ("embedding", "entity_edges"):
        authority.set_projection_status(
            memory_id="memory-1", memory_revision=1, projector=projector,
            status="pending_rebuild", source_sha256=memory["body_sha256"],
            details={"reason": "legacy_index_missing"},
        )
    worker, _bucket_manager, embedding, _moments, _node, entity, _word_map = _worker(
        authority, tmp_path
    )

    first = asyncio.run(worker.repair_pending_once(limit=1))
    second = asyncio.run(worker.repair_pending_once(limit=1))

    assert first["attempted"] == 1
    assert first["projected"] == 2
    assert second["attempted"] == 0
    assert len(embedding.generate_calls) == 1
    assert len(entity.replace_calls) == 1
    status = {row["projector"]: row["status"] for row in authority.list_projection_status(
        "memory-1", revision=1
    )}
    assert status["embedding"] == status["entity_edges"] == "projected"


def test_legacy_embedding_upgrade_marks_projected_pending_and_obeys_all_or_none_unit_budget(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    memory = authority.get_memory("memory-1")
    authority.set_projection_status(
        memory_id="memory-1", memory_revision=1, projector="embedding",
        status="projected", source_sha256=memory["body_sha256"],
        details={"metadata_complete": False, "index_unchanged": True},
    )
    authority.set_projection_status(
        memory_id="memory-1", memory_revision=1, projector="entity_edges",
        status="pending_rebuild", source_sha256=memory["body_sha256"],
        details={"reason": "separate_projection"},
    )
    embedding = RevisionAwareFakeEmbedding()
    worker, _bucket_manager, _embedding, *_rest = _worker(
        authority, tmp_path, embedding=embedding,
    )

    too_small = asyncio.run(worker.repair_pending_once(
        limit=1, upgrade_legacy_indexes=True, max_embedding_units=1,
    ))

    assert too_small["attempted"] == 1
    assert too_small["projected"] == 0
    assert too_small["deferred"] == 1
    assert too_small["embedding_units_completed"] == 0
    assert too_small["embedding_units_remaining"] == 2
    assert too_small["embedding_units_budget_remaining"] == 1
    assert embedding.generate_calls == []
    statuses = {item["projector"]: item for item in authority.list_projection_status(
        "memory-1", revision=1
    )}
    assert statuses["embedding"]["status"] == "pending_rebuild"
    assert statuses["embedding"]["details"]["reason"] == "embedding_unit_budget_insufficient"
    assert statuses["entity_edges"]["status"] == "pending_rebuild"

    enough = asyncio.run(worker.repair_pending_once(
        limit=1, upgrade_legacy_indexes=True, max_embedding_units=2,
    ))

    assert enough["attempted"] == 1
    assert enough["projected"] == 1
    assert enough["embedding_units_completed"] == 2
    assert enough["embedding_units_remaining"] == 0
    assert enough["embedding_units_budget_remaining"] == 0
    assert len(embedding.generate_calls) == 2
    statuses = {item["projector"]: item for item in authority.list_projection_status(
        "memory-1", revision=1
    )}
    assert statuses["embedding"]["status"] == "projected"
    assert statuses["embedding"]["details"]["metadata_complete"] is True
    assert statuses["entity_edges"]["status"] == "pending_rebuild"


def test_projection_uses_bucket_identity_when_authority_id_differs(tmp_path):
    authority = _authority(tmp_path)
    body = "主人喜欢清淡的手冲咖啡。"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="authority-memory", bucket_id="bucket-projection",
        expected_revision=0, body_sha256=digest,
        snapshot_path="revisions/authority-memory/1.md",
        metadata={}, source_refs=["test:distinct-id"],
        decision_source="test", idempotency_key="distinct-id", actor="test",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
    authority.finalize_memory_commit(prepared["operation_id"])
    worker, _bucket_manager, embedding, _moments, _node, entity, _word_map = _worker(
        authority, tmp_path, memory_id="bucket-projection", body=body
    )

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert embedding.generate_calls[0][0] == "bucket-projection"
    assert entity.replace_calls[0][0] == "bucket-projection"


def test_embedding_false_result_does_not_claim_projected(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    worker, _bucket_manager, embedding, *_rest = _worker(authority, tmp_path)
    embedding.return_false = True

    result = asyncio.run(worker.run_once())

    assert result["degraded"] == 1
    status = {row["projector"]: row["status"] for row in authority.list_projection_status(
        "memory-1", revision=1
    )}
    assert status["embedding"] == "degraded"


def test_pending_repair_never_sends_manual_only_memory_to_provider(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority, recall_policy="manual_only")
    memory = authority.get_memory("memory-1")
    authority.set_projection_status(
        memory_id="memory-1", memory_revision=1, projector="embedding",
        status="pending_rebuild", source_sha256=memory["body_sha256"],
    )
    worker, _bucket_manager, embedding, *_rest = _worker(authority, tmp_path)

    result = asyncio.run(worker.repair_pending_once(limit=1))

    assert result["attempted"] == 0
    assert embedding.generate_calls == []


def test_pending_repair_rejects_bucket_archived_outside_live_recall(tmp_path):
    authority = _authority(tmp_path)
    _commit_memory(authority)
    memory = authority.get_memory("memory-1")
    authority.set_projection_status(
        memory_id="memory-1", memory_revision=1, projector="embedding",
        status="pending_rebuild", source_sha256=memory["body_sha256"],
    )
    worker, bucket_manager, embedding, *_rest = _worker(authority, tmp_path)
    bucket_manager.bucket["path"] = str(tmp_path / "archive" / "memory-1.md")

    result = asyncio.run(worker.repair_pending_once(limit=1))

    assert result["attempted"] == 1
    assert result["degraded"] == 1
    assert embedding.generate_calls == []
