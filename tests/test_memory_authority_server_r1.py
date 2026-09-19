import asyncio

import server
from memory_authority import MemoryAuthorityStore
from memory_commit_service import BucketProjectionResult, MemoryCommitService


class FakeAuthority:
    def get_memory(self, memory_id):
        return {"memory_id": memory_id, "bucket_id": memory_id, "active_revision": 1}


class FakeBucketManager:
    def __init__(self):
        self.comments = []

    async def get(self, bucket_id):
        return {
            "id": bucket_id,
            "content": "正文",
            "metadata": {"comments": list(self.comments)},
        }

    async def add_comment(self, *args, **kwargs):
        raise AssertionError("authority-enabled server must not call BucketManager.add_comment directly")

    async def delete_comment(self, *args, **kwargs):
        raise AssertionError("authority-enabled server must not call BucketManager.delete_comment directly")

    async def search(self, *args, **kwargs):
        return []


class FakeCommitService:
    def __init__(self, bucket_manager):
        self.bucket_manager = bucket_manager
        self.append_calls = []
        self.retract_calls = []

    async def append_ring(self, **kwargs):
        self.append_calls.append(dict(kwargs))
        ring_id = "ring-authority-1"
        if not self.bucket_manager.comments:
            self.bucket_manager.comments.append({
                "id": ring_id,
                "content": kwargs["content"],
                "kind": kwargs["kind"],
                "author": kwargs["actor"],
                "source": kwargs["metadata"]["source"],
            })
        return {
            "memory_id": kwargs["memory_id"],
            "ring_id": ring_id,
            "outbox_event_id": "outbox-ring-1",
        }

    async def retract_ring(self, **kwargs):
        self.retract_calls.append(dict(kwargs))
        self.bucket_manager.comments = [
            item for item in self.bucket_manager.comments if item["id"] != kwargs["ring_id"]
        ]
        return {
            "memory_id": kwargs["memory_id"],
            "ring_id": kwargs["ring_id"],
            "status": "retracted",
            "outbox_event_id": "outbox-retract-1",
        }


def install_authority(monkeypatch):
    bucket_manager = FakeBucketManager()
    service = FakeCommitService(bucket_manager)
    monkeypatch.setattr(server, "bucket_mgr", bucket_manager)
    monkeypatch.setattr(server, "memory_authority_store", FakeAuthority())
    monkeypatch.setattr(server, "memory_commit_service", service)
    monkeypatch.setattr(server, "_queue_embedding_refresh", lambda _bucket_id: True)
    return bucket_manager, service


def test_comment_and_delete_route_through_ring_authority(monkeypatch):
    _bucket_manager, service = install_authority(monkeypatch)
    commented = asyncio.run(server.comment_bucket(
        bucket_id="memory-1",
        content="后来产生的新感受。",
        kind="feel",
        valence=0.8,
        idempotency_key="tool-operation-1",
    ))
    assert commented["status"] == "commented"
    assert commented["comment"]["id"] == "ring-authority-1"
    assert service.append_calls[0]["idempotency_key"] == "tool-operation-1"
    assert service.append_calls[0]["metadata"]["source"] == "comment_bucket"

    deleted = asyncio.run(server.delete_bucket_comment(
        "memory-1", "ring-authority-1", idempotency_key="tool-operation-2"
    ))
    assert deleted["status"] == "deleted"
    assert service.retract_calls[0]["idempotency_key"] == "tool-operation-2"


def test_hold_feel_uses_same_ring_authority(monkeypatch):
    _bucket_manager, service = install_authority(monkeypatch)
    result = asyncio.run(server.hold(
        content="我现在更理解她当时的感受。",
        feel=True,
        source_bucket="memory-1",
        valence=0.7,
        idempotency_key="hold-feel-1",
    ))
    assert result == "年轮→memory-1#ring-authority-1"
    assert service.append_calls[0]["kind"] == "feel"
    assert service.append_calls[0]["metadata"]["source"] == "hold(feel=True)"


class ToolMemoryProjection:
    def __init__(self):
        self.revisions = []
        self.deleted = []

    async def write_revision(self, **kwargs):
        self.revisions.append(dict(kwargs))
        return BucketProjectionResult(
            bucket_id=kwargs["bucket_id"], revision=kwargs["revision"],
            operation_id=kwargs["operation_id"],
            body_sha256=server.hashlib.sha256(kwargs["body"].encode("utf-8")).hexdigest(),
            snapshot_path=kwargs["snapshot_path"],
        )

    async def append_ring(self, **kwargs):
        raise AssertionError("not used")

    async def retract_ring(self, **kwargs):
        raise AssertionError("not used")

    async def delete_memory(self, **kwargs):
        self.deleted.append(kwargs["memory_id"])


def test_normal_hold_commits_new_memory_through_authority(monkeypatch, tmp_path):
    authority = MemoryAuthorityStore(str(tmp_path / "memory-authority.sqlite3"))
    projection = ToolMemoryProjection()
    service = MemoryCommitService(authority, projection)
    bucket_manager = FakeBucketManager()
    monkeypatch.setattr(server, "bucket_mgr", bucket_manager)
    monkeypatch.setattr(server, "memory_authority_store", authority)
    monkeypatch.setattr(server, "memory_commit_service", service)

    async def analyze(_content):
        return {
            "domain": ["生活"], "valence": 0.5, "arousal": 0.3,
            "tags": ["preference"], "suggested_name": "咖啡偏好",
            "memory_subject": "user", "memory_layer": "relationship",
        }

    async def unchanged(content, *_args, **_kwargs):
        return content

    async def no_related(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server.dehydrator, "analyze", analyze)
    monkeypatch.setattr(server, "_auto_generate_write_moment_if_needed", unchanged)
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda _bucket_id: True)

    result = asyncio.run(server.hold(
        content="主人不喜欢纯黑咖啡。",
        idempotency_key="hold-tool-call-1",
    ))
    assert result.startswith("新建→")
    assert len(projection.revisions) == 1
    memory_id = projection.revisions[0]["memory_id"]
    assert authority.get_memory(memory_id)["active_revision"] == 1
    assert authority.get_candidate(memory_id)["status"] == "committed"

    repeated = asyncio.run(server.hold(
        content="主人不喜欢纯黑咖啡。",
        idempotency_key="hold-tool-call-1",
    ))
    assert repeated.startswith("新建→")
    assert len(projection.revisions) == 1


def test_trace_creates_revision_then_tombstones_without_direct_bucket_update(monkeypatch, tmp_path):
    authority = MemoryAuthorityStore(str(tmp_path / "memory-authority.sqlite3"))
    projection = ToolMemoryProjection()
    service = MemoryCommitService(authority, projection)
    bucket_manager = FakeBucketManager()
    monkeypatch.setattr(server, "bucket_mgr", bucket_manager)
    monkeypatch.setattr(server, "memory_authority_store", authority)
    monkeypatch.setattr(server, "memory_commit_service", service)
    monkeypatch.setattr(server, "_queue_embedding_refresh_if_changed", lambda *_args: True)
    monkeypatch.setattr(server, "_delete_bucket_indexes", lambda _bucket_id: ({"embedding": True}, []))

    memory_id = asyncio.run(server._commit_tool_memory(
        content="旧正文",
        metadata={"name": "旧记忆", "domain": ["测试"], "tags": []},
        memory_type="durable_fact",
        source="hold",
        actor="guyan",
        idempotency_key="trace-seed",
    ))
    revised = asyncio.run(server.trace(
        bucket_id=memory_id,
        content="新正文",
        idempotency_key="trace-revision-1",
    ))
    assert "content=已替换" in revised
    assert authority.get_memory(memory_id)["active_revision"] == 2
    assert len(projection.revisions) == 2

    deleted = asyncio.run(server.trace(
        bucket_id=memory_id,
        delete=True,
        idempotency_key="trace-delete-1",
    ))
    assert deleted == f"已遗忘记忆桶: {memory_id}"
    memory = authority.get_memory(memory_id)
    assert memory["state"] == "tombstoned"
    assert memory["recall_policy"] == "disabled"
    assert projection.deleted == [memory_id]


def test_profile_fact_commits_evidence_refs_through_authority(monkeypatch, tmp_path):
    authority = MemoryAuthorityStore(str(tmp_path / "memory-authority.sqlite3"))
    projection = ToolMemoryProjection()
    service = MemoryCommitService(authority, projection)
    bucket_manager = FakeBucketManager()
    monkeypatch.setattr(server, "bucket_mgr", bucket_manager)
    monkeypatch.setattr(server, "memory_authority_store", authority)
    monkeypatch.setattr(server, "memory_commit_service", service)
    monkeypatch.setattr(
        server.memory_moment_store,
        "upsert_bucket",
        lambda _bucket: [{"moment_id": "moment-evidence-1"}],
    )
    monkeypatch.setattr(server.memory_edge_store, "add_edge", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(server, "_refresh_entity_edges_for_bucket", lambda _bucket: 0)
    monkeypatch.setattr(server, "_queue_embedding_refresh", lambda _bucket_id: True)

    result = asyncio.run(server.profile_fact(
        fact="主人不喜欢纯黑咖啡。",
        evidence_bucket_id="evidence-1",
        profile_kind="preference",
        idempotency_key="profile-fact-1",
    ))
    assert result.startswith("profile_fact→")
    memory_id = projection.revisions[0]["memory_id"]
    revision = authority.get_memory_revision(memory_id, 1)
    assert set(revision["source_refs"]) == {
        "tool_operation:profile-fact-1",
        "bucket:evidence-1",
        "moment:moment-evidence-1",
    }
    assert authority.get_candidate(memory_id)["status"] == "committed"
