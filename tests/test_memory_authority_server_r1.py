import asyncio

import server


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
