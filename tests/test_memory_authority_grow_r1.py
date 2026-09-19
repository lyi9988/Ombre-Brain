"""M2 grow-path contracts for the RECALL-R1 Memory Authority cutover.

The fixtures below use synthetic payloads and a recording projection.  They do
not load production Buckets or invoke a model; the assertions only inspect
authority metadata, operation keys, and projection counts.
"""

from __future__ import annotations

import asyncio
import hashlib

import server
from memory_authority import MemoryAuthorityStore
from memory_commit_service import (
    BucketProjectionResult,
    MemoryCommitService,
    body_sha256,
)


class RecordingProjection:
    """A no-filesystem projection that records commit identity and revisions."""

    def __init__(self):
        self.revisions: list[dict] = []

    async def write_revision(self, **kwargs):
        self.revisions.append(
            {
                key: kwargs[key]
                for key in (
                    "memory_id",
                    "bucket_id",
                    "revision",
                    "expected_revision",
                    "operation_id",
                    "snapshot_path",
                )
            }
        )
        return BucketProjectionResult(
            bucket_id=kwargs["bucket_id"],
            revision=kwargs["revision"],
            operation_id=kwargs["operation_id"],
            body_sha256=body_sha256(kwargs["body"]),
            snapshot_path=kwargs["snapshot_path"],
        )

    async def append_ring(self, **_kwargs):
        raise AssertionError("grow tests must not append a ring")

    async def retract_ring(self, **_kwargs):
        raise AssertionError("grow tests must not retract a ring")


class RecordingLegacyBucketManager:
    """Minimal pre-Authority BucketManager surface used by the compatibility test."""

    def __init__(self):
        self.created: list[dict] = []

    async def create(self, content, **kwargs):
        bucket_id = f"legacy-grow-{len(self.created) + 1}"
        self.created.append({"id": bucket_id, "content": content, "metadata": kwargs})
        return bucket_id

    async def search(self, *_args, **_kwargs):
        return []


def _install_common_grow_stubs(monkeypatch):
    async def no_start():
        return None

    async def no_related(*_args, **_kwargs):
        return None

    async def passthrough(content, *_args, **_kwargs):
        return content

    async def analyze(_content):
        return {
            "domain": ["synthetic"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": ["synthetic"],
            "suggested_name": "synthetic grow",
            "memory_subject": "user",
            "memory_layer": "relationship",
        }

    monkeypatch.setattr(server.decay_engine, "ensure_started", no_start)
    monkeypatch.setattr(server.memory_write_gate, "should_gate", lambda **_kwargs: False)
    monkeypatch.setattr(server, "_find_readonly_related_bucket", no_related)
    monkeypatch.setattr(server, "_auto_generate_write_moment_if_needed", passthrough)
    monkeypatch.setattr(server, "_queue_embedding_refresh", lambda _bucket_id: None)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda _bucket_id: None)
    monkeypatch.setattr(server.dehydrator, "analyze", analyze)


def _install_authority(monkeypatch, tmp_path):
    authority = MemoryAuthorityStore(str(tmp_path / "memory-authority.sqlite3"))
    projection = RecordingProjection()
    service = MemoryCommitService(authority, projection)
    monkeypatch.setattr(server, "memory_authority_store", authority)
    monkeypatch.setattr(server, "memory_commit_service", service)
    _install_common_grow_stubs(monkeypatch)
    return authority, projection


def _capture_commit_keys(monkeypatch):
    original = server._commit_tool_memory
    keys: list[str] = []

    async def capture(**kwargs):
        keys.append(str(kwargs["idempotency_key"]))
        return await original(**kwargs)

    monkeypatch.setattr(server, "_commit_tool_memory", capture)
    return keys


def test_grow_direct_structured_authority_retry_is_one_memory(monkeypatch, tmp_path):
    authority, projection = _install_authority(monkeypatch, tmp_path)
    keys = _capture_commit_keys(monkeypatch)
    direct = "# Synthetic direct\n### moment\nA curated fixture for direct grow."

    first = asyncio.run(server.grow(direct, idempotency_key="grow-direct-r1"))
    second = asyncio.run(server.grow(direct, idempotency_key="grow-direct-r1"))

    assert first == second
    assert keys == ["grow-direct-r1:direct", "grow-direct-r1:direct"]
    assert len(projection.revisions) == 1
    assert len(authority.list_candidates(status="committed")) == 1
    memory_id = projection.revisions[0]["memory_id"]
    assert authority.get_memory(memory_id)["active_revision"] == 1


def test_grow_short_authority_retry_is_one_memory(monkeypatch, tmp_path):
    authority, projection = _install_authority(monkeypatch, tmp_path)
    keys = _capture_commit_keys(monkeypatch)

    first = asyncio.run(server.grow("short synthetic fixture", idempotency_key="grow-short-r1"))
    second = asyncio.run(server.grow("short synthetic fixture", idempotency_key="grow-short-r1"))

    assert first == second
    assert keys == ["grow-short-r1:short", "grow-short-r1:short"]
    assert len(projection.revisions) == 1
    assert len(authority.list_candidates(status="committed")) == 1
    memory_id = projection.revisions[0]["memory_id"]
    assert authority.get_memory(memory_id)["active_revision"] == 1


def test_grow_multi_item_authority_keys_are_independent_stable_and_idempotent(
    monkeypatch, tmp_path
):
    authority, projection = _install_authority(monkeypatch, tmp_path)
    keys = _capture_commit_keys(monkeypatch)
    items = [
        "Synthetic split item zero for grow.",
        "Synthetic split item one for grow.",
    ]

    async def digest(_content):
        return [
            {
                "content": item,
                "tags": ["synthetic"],
                "importance": 5,
                "domain": ["synthetic"],
                "valence": 0.5,
                "arousal": 0.3,
                "name": f"item-{index}",
                "memory_subject": "user",
                "memory_layer": "relationship",
            }
            for index, item in enumerate(items)
        ]

    monkeypatch.setattr(server.dehydrator, "digest", digest)
    source = "A synthetic long grow input that deterministically splits into two items."
    first = asyncio.run(server.grow(source, idempotency_key="grow-multi-r1"))
    first_keys = list(keys)
    second = asyncio.run(server.grow(source, idempotency_key="grow-multi-r1"))

    expected_keys = [
        f"grow-multi-r1:item:{index}:{hashlib.sha256(item.encode('utf-8')).hexdigest()[:12]}"
        for index, item in enumerate(items)
    ]
    assert first == second
    assert first_keys == expected_keys
    assert first_keys[0] != first_keys[1]
    assert keys == expected_keys + expected_keys
    assert len(projection.revisions) == 2
    assert len({item["memory_id"] for item in projection.revisions}) == 2
    assert len(authority.list_candidates(status="committed")) == 2
    assert all(
        authority.get_memory(item["memory_id"])["active_revision"] == 1
        for item in projection.revisions
    )


def test_grow_preserves_legacy_bucket_writes_when_authority_disabled(monkeypatch):
    _install_common_grow_stubs(monkeypatch)
    manager = RecordingLegacyBucketManager()
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "memory_authority_store", None)
    monkeypatch.setattr(server, "memory_commit_service", None)

    async def digest(_content):
        return [
            {
                "content": "Synthetic legacy split zero.",
                "tags": ["synthetic"],
                "importance": 5,
                "domain": ["synthetic"],
                "valence": 0.5,
                "arousal": 0.3,
                "name": "legacy-zero",
                "memory_subject": "user",
                "memory_layer": "relationship",
            },
            {
                "content": "Synthetic legacy split one.",
                "tags": ["synthetic"],
                "importance": 5,
                "domain": ["synthetic"],
                "valence": 0.5,
                "arousal": 0.3,
                "name": "legacy-one",
                "memory_subject": "user",
                "memory_layer": "relationship",
            },
        ]

    monkeypatch.setattr(server.dehydrator, "digest", digest)
    asyncio.run(
        server.grow(
            "# Legacy direct\n### moment\nSynthetic direct fixture.",
            idempotency_key="legacy-direct-r1",
        )
    )
    asyncio.run(server.grow("legacy short fixture", idempotency_key="legacy-short-r1"))
    asyncio.run(
        server.grow(
            "A synthetic legacy input long enough for the digest path.",
            idempotency_key="legacy-multi-r1",
        )
    )

    assert len(manager.created) == 4
    assert [item["id"] for item in manager.created] == [
        "legacy-grow-1",
        "legacy-grow-2",
        "legacy-grow-3",
        "legacy-grow-4",
    ]
