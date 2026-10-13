from __future__ import annotations

import asyncio
import hashlib

from memory_authority import MemoryAuthorityStore
from memory_projection_worker import MemoryProjectionWorker


class _FailOnCallEmbedding:
    enabled = True

    def __init__(self):
        self.generate_calls = []
        self.get_calls = []
        self.delete_calls = []

    async def generate_and_store(self, *args, **kwargs):
        self.generate_calls.append((args, kwargs))
        raise AssertionError("archived state must not call embedding projection")

    async def get_embedding(self, bucket_id):
        self.get_calls.append(str(bucket_id))
        raise AssertionError("archived state must not inspect or rebuild embeddings")

    def delete_embedding(self, memory_id):
        self.delete_calls.append(str(memory_id))


class _FakeBucketManager:
    def __init__(self, bucket):
        self.bucket = dict(bucket)
        self.get_calls = []
        self.updates = []
        self.archives = []
        self.permanent_dir = ""
        self.dynamic_dir = ""

    async def get(self, bucket_id):
        self.get_calls.append(str(bucket_id))
        return dict(self.bucket) if str(bucket_id) == self.bucket["id"] else None

    async def update(self, bucket_id, **kwargs):
        self.updates.append((str(bucket_id), dict(kwargs)))
        return True

    async def archive(self, bucket_id):
        self.archives.append(str(bucket_id))
        return True


class _FakeDerivedStore:
    def __init__(self):
        self.upserts = []
        self.deletes = []
        self.enabled = True
        self.relevance_options = None
        self.annotation_options = None

    def upsert_bucket(self, bucket):
        self.upserts.append(dict(bucket))
        return {"id": bucket["id"]}

    def delete_bucket(self, memory_id):
        self.deletes.append(str(memory_id))

    def delete(self, memory_id):
        self.deletes.append(str(memory_id))

    def replace_bucket_edges(self, bucket_id, edges):
        self.upserts.append((str(bucket_id), list(edges)))
        return list(edges)

    def delete_for_bucket(self, memory_id):
        self.deletes.append(str(memory_id))


def _authority(tmp_path, *, recall_policy="enabled"):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    body = "one stable archived body"
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="memory:archive-gate",
        bucket_id="bucket-archive-gate",
        expected_revision=0,
        body_sha256=body_sha,
        snapshot_path="revisions/memory:archive-gate/1.md",
        metadata={},
        source_refs=["test:archive-gate"],
        decision_source="test",
        idempotency_key="commit:archive-gate",
        actor="test",
        recall_policy=recall_policy,
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=body_sha)
    authority.finalize_memory_commit(prepared["operation_id"])
    return authority, body, body_sha


def _worker(authority, body, embedding):
    bucket_manager = _FakeBucketManager({
        "id": "bucket-archive-gate",
        "content": body,
        "metadata": {"type": "archived"},
        "path": "/data/archive/Personal/bucket-archive-gate.md",
    })
    moment_store = _FakeDerivedStore()
    node_store = _FakeDerivedStore()
    entity_store = _FakeDerivedStore()
    word_map_store = _FakeDerivedStore()
    worker = MemoryProjectionWorker(
        config={"identity": {"ai_name": "顾衍", "user_name": "ZJR"}},
        authority=authority,
        bucket_manager=bucket_manager,
        embedding_engine=embedding,
        moment_store=moment_store,
        node_store=node_store,
        entity_edge_store=entity_store,
        word_map_store=word_map_store,
    )
    return worker, bucket_manager, moment_store, node_store, entity_store, word_map_store


def _pending_event(authority, event_type):
    return next(
        event for event in authority.pending_outbox(limit=20)
        if event["event_type"] == event_type
    )


def _state_event_count(authority):
    conn = authority._connect()
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM outbox WHERE event_type='MemoryStateChanged'"
        ).fetchone()[0])
    finally:
        conn.close()


def test_archived_enabled_state_only_event_deletes_derived_indexes_without_provider_or_bucket_write(tmp_path):
    authority, body, body_sha = _authority(tmp_path)
    authority.change_memory_state(
        memory_id="memory:archive-gate",
        state="archived",
        recall_policy="enabled",
        idempotency_key="archive:gate:r1",
        actor="owner_repair",
        expected_revision=1,
        expected_state="active",
    )
    event = _pending_event(authority, "MemoryStateChanged")
    embedding = _FailOnCallEmbedding()
    worker, bucket_manager, moments, nodes, edges, word_map = _worker(
        authority, body, embedding,
    )
    upsert_calls = []
    original_upsert = worker._upsert_indexes

    async def counted_upsert(*args, **kwargs):
        upsert_calls.append(True)
        return await original_upsert(*args, **kwargs)

    worker._upsert_indexes = counted_upsert
    result = asyncio.run(worker.project_event(event))

    assert result["status"] == "projected"
    assert bucket_manager.get_calls == []
    assert bucket_manager.updates == bucket_manager.archives == []
    assert upsert_calls == []
    assert embedding.generate_calls == embedding.get_calls == []
    assert embedding.delete_calls == ["bucket-archive-gate"]
    assert moments.deletes == nodes.deletes == edges.deletes == ["bucket-archive-gate"]
    assert word_map.upserts == []
    assert authority.get_memory("memory:archive-gate")["body_sha256"] == body_sha
    assert authority.get_memory("memory:archive-gate")["active_revision"] == 1
    assert authority.get_memory("memory:archive-gate")["recall_policy"] == "enabled"
    assert _state_event_count(authority) == 1

    # A duplicate outbox delivery is idempotent for state history and still
    # cannot re-enable a source or launch an embedding projection.
    replay = asyncio.run(worker.project_event(event))
    assert replay["status"] == "projected"
    assert _state_event_count(authority) == 1
    assert upsert_calls == []
    assert embedding.generate_calls == embedding.get_calls == []


def test_active_manual_only_memory_remains_on_existing_projection_path(tmp_path):
    authority, body, body_sha = _authority(tmp_path, recall_policy="manual_only")
    event = _pending_event(authority, "MemoryRevisionCommitted")
    embedding = _FailOnCallEmbedding()
    worker, bucket_manager, _moments, _nodes, _edges, _word_map = _worker(
        authority, body, embedding,
    )
    upsert_calls = []

    async def record_upsert(bucket, memory_id, revision, source_sha):
        upsert_calls.append((bucket["id"], memory_id, revision, source_sha))
        return []

    worker._upsert_indexes = record_upsert
    result = asyncio.run(worker.project_event(event))

    assert result["status"] == "projected"
    assert bucket_manager.get_calls == ["bucket-archive-gate"]
    assert upsert_calls == [("bucket-archive-gate", "memory:archive-gate", 1, body_sha)]
    assert embedding.generate_calls == embedding.get_calls == embedding.delete_calls == []
    assert authority.get_memory("memory:archive-gate")["state"] == "active"
