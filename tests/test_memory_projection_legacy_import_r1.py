"""Synthetic, provider-free coverage for legacy Recall projection inspection.

Legacy imports are an inventory/reconciliation event.  They may read existing
derived rows, but must not regenerate, upsert, or delete those rows.  A normal
``MemoryRevisionCommitted`` event keeps the ordinary incremental projection
path, so these tests exercise both branches against only in-memory fakes and a
temporary Word Map SQLite file.
"""

import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest

from memory_authority import MemoryAuthorityStore
from memory_projection_worker import MemoryProjectionWorker


MEMORY_ID = "legacy-memory-1"
BUCKET_ID = "legacy-bucket-1"
BODY = "主人喜欢清淡的手冲咖啡。"


class RecordingBucketManager:
    def __init__(self, bucket=None):
        self.bucket = dict(bucket) if bucket else None
        self.get_calls = []

    async def get(self, bucket_id):
        self.get_calls.append(str(bucket_id))
        if self.bucket is None or str(bucket_id) != str(self.bucket.get("id")):
            return None
        return dict(self.bucket)


class RecordingEmbedding:
    enabled = True

    def __init__(self, *, present=True):
        self.present = bool(present)
        self.read_calls = []
        self.generate_calls = []
        self.delete_calls = []

    async def get_embedding(self, bucket_id):
        self.read_calls.append(str(bucket_id))
        return [0.1, 0.2] if self.present else None

    async def generate_and_store(self, memory_id, text):
        self.generate_calls.append((str(memory_id), str(text)))
        return {"memory_id": memory_id}

    def delete_embedding(self, bucket_id):
        self.delete_calls.append(str(bucket_id))


class RecordingMoments:
    def __init__(self, *, present=True):
        self.present = bool(present)
        self.read_calls = []
        self.upsert_calls = []
        self.delete_calls = []

    def list_for_bucket(self, bucket_id, limit=100):
        self.read_calls.append((str(bucket_id), int(limit)))
        return [{"moment_id": f"moment:{bucket_id}"}] if self.present else []

    def upsert_bucket(self, bucket):
        self.upsert_calls.append(dict(bucket))
        return [{"moment_id": f"moment:{bucket['id']}"}]

    def delete_bucket(self, bucket_id):
        self.delete_calls.append(str(bucket_id))


class RecordingNode:
    def __init__(self, *, present=True):
        self.present = bool(present)
        self.read_calls = []
        self.upsert_calls = []
        self.delete_calls = []

    def get(self, bucket_id):
        self.read_calls.append(str(bucket_id))
        return {"bucket_id": str(bucket_id)} if self.present else None

    def upsert_bucket(self, bucket):
        self.upsert_calls.append(dict(bucket))
        return {"bucket_id": bucket["id"]}

    def delete(self, bucket_id):
        self.delete_calls.append(str(bucket_id))


class RecordingEntityEdges:
    def __init__(self, *, present=True):
        self.present = bool(present)
        self.read_calls = 0
        self.replace_calls = []
        self.delete_calls = []

    def list_edges(self):
        self.read_calls += 1
        if not self.present:
            return []
        return [{
            "bucket_id": BUCKET_ID,
            "subject": "ZJR",
            "relation": "likes",
            "object_text": "清淡的手冲咖啡",
        }]

    def replace_bucket_edges(self, bucket_id, edges):
        self.replace_calls.append((str(bucket_id), list(edges)))
        return list(edges)

    def delete_for_bucket(self, bucket_id):
        self.delete_calls.append(str(bucket_id))


class RecordingWordMap:
    enabled = True

    def __init__(self, db_path, *, present=True):
        self.db_path = str(db_path)
        self.extract_calls = []
        self.upsert_calls = []
        self._init_db(present=present)

    def _init_db(self, *, present):
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE word_card_nodes ("
            "bucket_id TEXT NOT NULL, term TEXT NOT NULL, source TEXT NOT NULL, "
            "kind TEXT NOT NULL, weight REAL NOT NULL, updated_at TEXT NOT NULL, "
            "PRIMARY KEY(bucket_id, term))"
        )
        if present:
            conn.execute(
                "INSERT INTO word_card_nodes "
                "(bucket_id,term,source,kind,weight,updated_at) VALUES(?,?,?,?,?,?)",
                (BUCKET_ID, "咖啡", "content", "keyword", 0.8, "2026-01-01T00:00:00Z"),
            )
        conn.commit()
        conn.close()

    def extract_bucket_terms(self, bucket):
        self.extract_calls.append(str(bucket.get("id")))
        return ["咖啡"]

    def upsert_bucket(self, bucket):
        self.upsert_calls.append(dict(bucket))
        return {"bucket_id": bucket["id"]}

    def rows(self):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                "SELECT bucket_id,term,source,kind,weight,updated_at "
                "FROM word_card_nodes ORDER BY bucket_id,term"
            ).fetchall()
        finally:
            conn.close()


def _bucket(body=BODY, bucket_id=BUCKET_ID):
    return {
        "id": bucket_id,
        "content": body,
        "metadata": {
            "name": "咖啡偏好",
            "domain": ["偏好"],
            "tags": ["preference"],
            "comments": [],
        },
    }


def _authority(tmp_path):
    return MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})


def _import_legacy(authority, *, memory_id=MEMORY_ID, bucket_id=BUCKET_ID,
                   body=BODY, state="active", recall_policy="enabled"):
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return authority.import_legacy_memory(
        memory_id=memory_id,
        bucket_id=bucket_id,
        body_sha256=body_sha,
        snapshot_path=f"revisions/{memory_id}/1.md",
        metadata={"name": "咖啡偏好", "domain": ["偏好"]},
        source_refs=[f"legacy:{memory_id}"],
        state=state,
        recall_policy=recall_policy,
    )


def _commit_revision(authority, *, memory_id, bucket_id, body, expected_revision=0):
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id=memory_id,
        bucket_id=bucket_id,
        expected_revision=expected_revision,
        body_sha256=body_sha,
        snapshot_path=f"revisions/{memory_id}/{expected_revision + 1}.md",
        metadata={"name": "咖啡偏好", "domain": ["偏好"]},
        source_refs=[f"test:{memory_id}:{expected_revision + 1}"],
        decision_source="test",
        idempotency_key=f"revision:{memory_id}:{expected_revision + 1}",
        actor="test",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=body_sha)
    return authority.finalize_memory_commit(prepared["operation_id"])


def _worker(authority, tmp_path, *, bucket=None, present=True):
    bucket_manager = RecordingBucketManager(bucket or _bucket())
    embedding = RecordingEmbedding(present=present)
    moments = RecordingMoments(present=present)
    node = RecordingNode(present=present)
    entity = RecordingEntityEdges(present=present)
    word_map = RecordingWordMap(tmp_path / "word-map.sqlite", present=present)
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


def _by_projector(result):
    return {item["projector"]: item for item in result["results"][0]["projections"]}


def _assert_no_legacy_writes(embedding, moments, node, entity, word_map):
    assert embedding.generate_calls == []
    assert embedding.delete_calls == []
    assert moments.upsert_calls == []
    assert moments.delete_calls == []
    assert node.upsert_calls == []
    assert node.delete_calls == []
    assert entity.replace_calls == []
    assert entity.delete_calls == []
    assert word_map.upsert_calls == []


def test_legacy_import_only_reads_existing_derived_indexes(tmp_path):
    authority = _authority(tmp_path)
    imported = _import_legacy(authority)
    worker, bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority, tmp_path, present=True
    )
    word_rows_before = word_map.rows()

    result = asyncio.run(worker.run_once())

    assert result["claimed"] == 1
    assert result["projected"] == 1
    assert result["degraded"] == 0
    event_result = result["results"][0]
    assert event_result["event_id"] == imported["outbox_event_id"]
    assert event_result["reason"] == "legacy_indexes_inspected_without_provider_calls"
    assert bucket_manager.get_calls == [BUCKET_ID]
    assert embedding.read_calls == [BUCKET_ID]
    assert moments.read_calls == [(BUCKET_ID, 1)]
    assert node.read_calls == [BUCKET_ID]
    assert entity.read_calls == 1
    assert word_map.extract_calls == [BUCKET_ID]
    assert word_map.rows() == word_rows_before
    _assert_no_legacy_writes(embedding, moments, node, entity, word_map)

    statuses = _by_projector(result)
    for projector in ("embedding", "moments", "memory_node", "entity_edges", "word_map"):
        assert statuses[projector]["status"] == "projected"
        assert statuses[projector]["details"]["index_unchanged"] is True


def test_legacy_import_marks_missing_indexes_without_model_or_index_writes(tmp_path):
    authority = _authority(tmp_path)
    _import_legacy(authority)
    worker, _bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority, tmp_path, present=False
    )

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert result["degraded"] == 0
    _assert_no_legacy_writes(embedding, moments, node, entity, word_map)
    statuses = _by_projector(result)
    for projector in ("embedding", "moments", "memory_node", "entity_edges", "word_map"):
        assert statuses[projector]["status"] in {"pending_rebuild", "degraded"}
        assert statuses[projector]["details"]["index_unchanged"] is True


def test_new_memory_revision_committed_keeps_incremental_projection_path(tmp_path):
    authority = _authority(tmp_path)
    commit = _commit_revision(
        authority,
        memory_id="memory-new-1",
        bucket_id="memory-new-1",
        body="主人喜欢新的手冲咖啡。",
    )
    worker, _bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority,
        tmp_path,
        bucket=_bucket("主人喜欢新的手冲咖啡。", "memory-new-1"),
        present=False,
    )

    result = asyncio.run(worker.run_once())

    assert result["claimed"] == 1
    assert result["projected"] == 1
    assert result["degraded"] == 0
    assert result["results"][0]["event_id"] == commit["outbox_event_id"]
    assert len(embedding.generate_calls) == 1
    assert len(moments.upsert_calls) == 1
    assert len(node.upsert_calls) == 1
    assert len(entity.replace_calls) == 1
    assert len(word_map.upsert_calls) == 1
    assert embedding.delete_calls == []
    assert moments.delete_calls == []
    assert node.delete_calls == []
    assert entity.delete_calls == []


@pytest.mark.parametrize(
    ("state", "recall_policy"),
    [("tombstoned", "enabled"), ("active", "disabled")],
    ids=["tombstoned", "disabled"],
)
def test_tombstoned_or_disabled_legacy_import_does_not_revive_or_delete_indexes(
    tmp_path, state, recall_policy
):
    authority = _authority(tmp_path)
    _import_legacy(authority, state=state, recall_policy=recall_policy)
    worker, bucket_manager, embedding, moments, node, entity, word_map = _worker(
        authority, tmp_path, present=True
    )
    word_rows_before = word_map.rows()

    result = asyncio.run(worker.run_once())

    assert result["projected"] == 1
    assert result["degraded"] == 0
    assert bucket_manager.get_calls == []
    assert embedding.read_calls == []
    assert moments.read_calls == []
    assert node.read_calls == []
    assert entity.read_calls == 0
    assert word_map.extract_calls == []
    assert word_map.rows() == word_rows_before
    _assert_no_legacy_writes(embedding, moments, node, entity, word_map)
    statuses = _by_projector(result)
    assert all(
        statuses[projector]["status"] == "disabled"
        for projector in (
            "embedding", "moments", "memory_node", "entity_edges", "word_map",
            "identity_semantics", "memory_edges",
        )
    )
