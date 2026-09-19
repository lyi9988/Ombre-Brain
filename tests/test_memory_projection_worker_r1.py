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
        self.fail_delete = False

    async def generate_and_store(self, memory_id, text):
        if self.fail_generate:
            raise RuntimeError("embedding unavailable")
        self.generate_calls.append((memory_id, text))
        return {"memory_id": memory_id, "text": text}

    def delete_embedding(self, memory_id):
        if self.fail_delete:
            raise RuntimeError("embedding delete unavailable")
        self.delete_calls.append(memory_id)


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


def _worker(authority, tmp_path, memory_id="memory-1", body="主人喜欢清淡的手冲咖啡。"):
    bucket_manager = FakeBucketManager(_bucket(memory_id, body))
    embedding = FakeEmbedding()
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
    assert by_projector["identity_semantics"]["status"] == "pending_rebuild"
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
