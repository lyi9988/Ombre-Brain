"""Synthetic admin API coverage for the RECALL-R1 memory authority seam.

These tests call the Starlette handlers directly.  The Bucket manager and the
projection are deliberately in-memory so that no network, model, production
Bucket, or real project state is involved.  The authority itself uses a
temporary SQLite file, which exercises the real revision/idempotency/outbox
contracts.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from starlette.responses import JSONResponse

import server
from memory_authority import MemoryAuthorityStore
from memory_commit_service import BucketProjectionResult, MemoryCommitService, body_sha256


class FakeRequest:
    """Small request object exposing only what the handlers read."""

    def __init__(self, body=None, *, path_params=None, headers=None, query_params=None):
        self._body = copy.deepcopy(body) if body is not None else {}
        self.path_params = dict(path_params or {})
        self.headers = dict(headers or {})
        self.query_params = dict(query_params or {})
        self.cookies = {}

    async def json(self):
        return copy.deepcopy(self._body)


class FakeBucketManager:
    """Synthetic Bucket store with explicit legacy-call counters."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.created: list[dict] = []
        self.update_calls: list[tuple[str, dict]] = []
        self.archive_calls: list[str] = []
        self.activate_calls: list[str] = []
        self.delete_calls: list[str] = []

    def seed(self, bucket_id: str, *, content: str = "旧正文", metadata: dict | None = None):
        base = {
            "name": bucket_id,
            "type": "dynamic",
            "domain": ["测试"],
            "tags": [],
            "facets": [],
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
            "confidence": 0.9,
            "source": "synthetic",
            "created": "2026-09-20T00:00:00+00:00",
            "updated_at": "2026-09-20T00:00:00+00:00",
            "last_active": "2026-09-20T00:00:00+00:00",
            "active": True,
            "resolved": False,
            "deprecated": False,
        }
        base.update(copy.deepcopy(metadata or {}))
        self.items[bucket_id] = {
            "id": bucket_id,
            "content": content,
            "metadata": base,
        }

    async def get(self, bucket_id: str):
        item = self.items.get(str(bucket_id))
        return copy.deepcopy(item) if item is not None else None

    async def list_all(self, include_archive: bool = False):
        values = list(self.items.values())
        if not include_archive:
            values = [item for item in values if item.get("metadata", {}).get("type") != "archived"]
        return copy.deepcopy(values)

    async def create(self, content: str, **kwargs):
        bucket_id = str(kwargs.get("bucket_id") or f"legacy-{len(self.created) + 1}")
        metadata = {
            key: copy.deepcopy(value)
            for key, value in kwargs.items()
            if key not in {"bucket_id", "extra_metadata"}
        }
        metadata["type"] = metadata.pop("bucket_type", metadata.get("type", "dynamic"))
        metadata.update(copy.deepcopy(kwargs.get("extra_metadata") or {}))
        metadata.setdefault("name", bucket_id)
        self.items[bucket_id] = {"id": bucket_id, "content": content, "metadata": metadata}
        self.created.append({"id": bucket_id, "content": content, "kwargs": copy.deepcopy(kwargs)})
        return bucket_id

    async def update(self, bucket_id: str, *, content=None, extra_metadata=None, **kwargs):
        bucket_id = str(bucket_id)
        item = self.items.get(bucket_id)
        if item is None:
            return False
        payload = {"content": content, "extra_metadata": extra_metadata, **kwargs}
        self.update_calls.append((bucket_id, copy.deepcopy(payload)))
        if content is not None:
            item["content"] = content
        metadata = item.setdefault("metadata", {})
        if extra_metadata:
            metadata.update(copy.deepcopy(extra_metadata))
        for key, value in kwargs.items():
            if key == "bucket_type":
                metadata["type"] = value
            elif value is not None:
                metadata[key] = copy.deepcopy(value)
            else:
                metadata.pop(key, None)
        return True

    async def archive(self, bucket_id: str):
        self.archive_calls.append(str(bucket_id))
        item = self.items.get(str(bucket_id))
        if item is None:
            return False
        item["metadata"].update({"type": "archived", "active": False})
        return True

    async def activate(self, bucket_id: str):
        self.activate_calls.append(str(bucket_id))
        item = self.items.get(str(bucket_id))
        if item is None:
            return False
        item["metadata"].update({"type": "dynamic", "active": True, "resolved": False})
        return True

    async def delete(self, bucket_id: str):
        bucket_id = str(bucket_id)
        self.delete_calls.append(bucket_id)
        return self.items.pop(bucket_id, None) is not None


class RecordingProjection:
    """Memory projection that updates only the synthetic Bucket store."""

    def __init__(self, buckets: FakeBucketManager):
        self.buckets = buckets
        self.revisions: list[dict] = []
        self.states: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def write_revision(self, **kwargs):
        self.revisions.append(copy.deepcopy(kwargs))
        memory_id = str(kwargs["memory_id"])
        bucket_id = str(kwargs["bucket_id"])
        metadata = copy.deepcopy(dict(kwargs.get("metadata") or {}))
        metadata.update({
            "type": metadata.get("bucket_type", metadata.get("type", "dynamic")),
            "memory_id": memory_id,
            "memory_revision": int(kwargs["revision"]),
            "memory_body_sha256": body_sha256(kwargs["body"]),
            "memory_operation_id": kwargs["operation_id"],
            "memory_authority_schema": "memory-authority-v1",
        })
        self.buckets.items[bucket_id] = {
            "id": bucket_id,
            "content": str(kwargs["body"]),
            "metadata": metadata,
        }
        return BucketProjectionResult(
            bucket_id=bucket_id,
            revision=int(kwargs["revision"]),
            operation_id=str(kwargs["operation_id"]),
            body_sha256=body_sha256(kwargs["body"]),
            snapshot_path=str(kwargs["snapshot_path"]),
        )

    async def append_ring(self, **_kwargs):
        raise AssertionError("ring projection is outside this admin API fixture")

    async def retract_ring(self, **_kwargs):
        raise AssertionError("ring projection is outside this admin API fixture")

    async def delete_memory(self, *, memory_id: str, bucket_id: str):
        self.deleted.append(str(memory_id))
        self.buckets.items.pop(str(bucket_id), None)

    async def set_memory_state(self, *, memory_id: str, bucket_id: str, state: str):
        self.states.append((str(memory_id), str(state)))
        bucket = self.buckets.items.get(str(bucket_id))
        if bucket is None:
            raise AssertionError(f"missing synthetic bucket {bucket_id}")
        metadata = bucket.setdefault("metadata", {})
        if state == "archived":
            metadata.update({"type": "archived", "active": False})
        elif state == "active":
            metadata.update({"type": "dynamic", "active": True, "resolved": False, "deprecated": False})


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


def outbox_event_types(authority):
    """Read synthetic outbox rows, including events already projected."""
    connection = authority._connect()
    try:
        rows = connection.execute(
            "SELECT event_type FROM outbox ORDER BY created_at, event_id"
        ).fetchall()
        return [str(row["event_type"]) for row in rows]
    finally:
        connection.close()


def install(monkeypatch, tmp_path, *, authority_enabled: bool):
    buckets = FakeBucketManager()
    authority = MemoryAuthorityStore(str(tmp_path / "memory-authority.sqlite3")) if authority_enabled else None
    projection = RecordingProjection(buckets)
    service = MemoryCommitService(authority, projection) if authority is not None else None

    monkeypatch.setattr(server, "bucket_mgr", buckets)
    monkeypatch.setattr(server, "memory_authority_store", authority)
    monkeypatch.setattr(server, "memory_commit_service", service)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda _request: None)
    monkeypatch.setattr(server, "_memory_write_token", lambda: "synthetic-token")
    monkeypatch.setattr(server, "_authorized_memory_write", lambda _request: True)
    monkeypatch.setattr(server, "embedding_engine", SimpleNamespace(enabled=False))
    monkeypatch.setattr(server, "_queue_embedding_refresh", lambda _bucket_id: True)
    monkeypatch.setattr(server, "_queue_embedding_refresh_if_changed", lambda *_args: False)
    monkeypatch.setattr(server, "_queue_memory_enrichment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_delete_bucket_indexes", lambda _bucket_id: ({}, []))
    monkeypatch.setattr(server.decay_engine, "calculate_score", lambda _metadata: 0.0)
    return buckets, authority, service, projection


def seed_authority_memory(buckets, authority, service, bucket_id="memory-r1"):
    buckets.seed(bucket_id)
    result = asyncio.run(service.commit_memory(
        memory_id=bucket_id,
        bucket_id=bucket_id,
        expected_revision=0,
        body="旧正文",
        metadata=buckets.items[bucket_id]["metadata"],
        source_refs=["synthetic:seed"],
        decision_source="synthetic",
        idempotency_key=f"seed:{bucket_id}",
        actor="synthetic",
    ))
    assert result["revision"] == 1
    return bucket_id


def test_owner_memory_authority_read_models_are_revisioned_and_collapsible(monkeypatch, tmp_path):
    buckets, authority, service, _projection = install(
        monkeypatch, tmp_path, authority_enabled=True
    )
    memory_id = seed_authority_memory(buckets, authority, service)
    authority.upsert_alias(
        entity_id="person:yanyan",
        alias="晏晏",
        trust="owner",
        source_refs=[f"memory:{memory_id}"],
    )

    overview_response = asyncio.run(server.api_memory_authority_overview(FakeRequest()))
    overview = response_json(overview_response)
    assert overview_response.status_code == 200
    assert overview["overview"]["memories"] == {"active": 1}
    assert overview["overview"]["aliases"] == {"owner": 1}

    list_response = asyncio.run(server.api_memory_authority_memories(FakeRequest(
        query_params={"include_body": "true"}
    )))
    listed = response_json(list_response)
    assert listed["items"][0]["memory_id"] == memory_id
    assert listed["items"][0]["active_revision"] == 1
    assert listed["items"][0]["body"] == "旧正文"

    detail_response = asyncio.run(server.api_memory_authority_memory_detail(FakeRequest(
        path_params={"memory_id": memory_id}
    )))
    detail = response_json(detail_response)
    assert detail["memory"]["active_revision"] == 1
    assert detail["body"] == "旧正文"
    assert [row["revision"] for row in detail["revisions"]] == [1]
    assert detail["rings"] == []

    aliases_response = asyncio.run(server.api_memory_authority_aliases(FakeRequest()))
    aliases = response_json(aliases_response)
    assert aliases["items"][0]["alias"] == "晏晏"
    assert aliases["items"][0]["source_refs"] == [f"memory:{memory_id}"]

    compatibility = server._identity_semantics_payload(100)
    assert compatibility["authority"] == "memory_authority"
    assert compatibility["aliases"][0]["alias"] == "晏晏"

    rebuild_response = asyncio.run(server.api_identity_semantics_rebuild(FakeRequest()))
    assert rebuild_response.status_code == 409
    assert response_json(rebuild_response)["rebuild_allowed"] is False


def test_memory_authority_read_accepts_internal_bearer_without_dashboard_cookie(monkeypatch, tmp_path):
    _buckets, authority, _service, _projection = install(
        monkeypatch, tmp_path, authority_enabled=True
    )
    monkeypatch.setattr(server, "_dashboard_authenticated", lambda _request: False)
    monkeypatch.setattr(
        server,
        "_authorized_memory_write",
        lambda request: request.headers.get("authorization") == "Bearer internal-token",
    )
    monkeypatch.setattr(
        server,
        "_require_dashboard_auth",
        lambda _request: JSONResponse({"error": "unauthorized"}, status_code=401),
    )

    unauthorized = asyncio.run(server.api_memory_authority_overview(FakeRequest()))
    authorized = asyncio.run(server.api_memory_authority_overview(FakeRequest(
        headers={"authorization": "Bearer internal-token"}
    )))

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert response_json(authorized)["enabled"] is True


def test_api_memories_authority_create_update_revision_and_idempotency(monkeypatch, tmp_path):
    buckets, authority, _service, projection = install(monkeypatch, tmp_path, authority_enabled=True)

    create_request = FakeRequest({
        "title": "合成记忆",
        "content": "第一版正文",
        "type": "dynamic",
        "idempotency_key": "api-create-r1",
    })
    created = asyncio.run(server.api_create_memory(create_request))
    payload = response_json(created)
    assert created.status_code == 200
    assert payload["status"] == "created"
    memory_id = payload["id"]
    assert authority.get_memory(memory_id)["active_revision"] == 1
    assert len(projection.revisions) == 1

    # Repeating the create request resolves to the same explicit-tool candidate.
    repeated_create = asyncio.run(server.api_create_memory(create_request))
    assert response_json(repeated_create)["id"] == memory_id
    assert len(projection.revisions) == 1

    update_body = {
        "id": memory_id,
        "title": "合成记忆修订",
        "content": "第二版正文",
        "idempotency_key": "api-update-r1",
    }
    updated = asyncio.run(server.api_create_memory(FakeRequest(update_body)))
    assert updated.status_code == 200
    assert response_json(updated)["status"] == "updated"
    assert authority.get_memory(memory_id)["active_revision"] == 2
    assert authority.get_memory_revision(memory_id, 2)["body_sha256"] == body_sha256("第二版正文")
    assert len(projection.revisions) == 2

    # The same update key must return the already committed revision, not make #3.
    repeated_update = asyncio.run(server.api_create_memory(FakeRequest(update_body)))
    assert repeated_update.status_code == 200
    assert authority.get_memory(memory_id)["active_revision"] == 2
    assert len(projection.revisions) == 2


def test_api_bucket_update_authority_forms_revision(monkeypatch, tmp_path):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "bucket-update-r1")

    response = asyncio.run(server.api_bucket_update(FakeRequest(
        {"content": "管理端修订正文", "name": "修订名称", "expected_revision": 1,
         "idempotency_key": "bucket-update-r1"},
        path_params={"bucket_id": bucket_id},
    )))
    payload = response_json(response)

    assert response.status_code == 200
    assert payload["status"] == "updated"
    assert payload["revision"] == 2
    assert authority.get_memory(bucket_id)["active_revision"] == 2
    revision = authority.get_memory_revision(bucket_id, 2)
    assert revision["metadata"]["name"] == "修订名称"
    assert revision["body_sha256"] == body_sha256("管理端修订正文")
    assert len(projection.revisions) == 2
    assert buckets.update_calls == []


def commit_background_revision(service, buckets, memory_id, *, body, key):
    current_bucket = asyncio.run(buckets.get(memory_id))
    return asyncio.run(service.commit_memory(
        memory_id=memory_id,
        bucket_id=memory_id,
        expected_revision=1,
        body=body,
        metadata=current_bucket["metadata"],
        source_refs=[f"synthetic:{key}"],
        decision_source="synthetic",
        idempotency_key=key,
        actor="synthetic_background",
    ))


def test_api_bucket_update_stale_page_conflicts_without_losing_callers_draft(monkeypatch, tmp_path):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "bucket-update-stale-r1")
    old_page_draft = "旧页面 r1 上仍由调用方保留的草稿"

    background = commit_background_revision(
        service, buckets, bucket_id, body="后台先提交的 r2 正文", key="background-r2"
    )
    assert background["revision"] == 2
    request = FakeRequest(
        {"content": old_page_draft, "expected_revision": 1, "idempotency_key": "stale-page-r1"},
        path_params={"bucket_id": bucket_id},
    )

    response = asyncio.run(server.api_bucket_update(request))
    payload = response_json(response)

    assert response.status_code == 409
    assert payload["error"] == "revision_conflict"
    assert payload["current_revision"] == 2
    assert authority.get_memory(bucket_id)["active_revision"] == 2
    assert authority.get_memory_revision(bucket_id, 3) is None
    assert len(projection.revisions) == 2
    assert buckets.items[bucket_id]["content"] == "后台先提交的 r2 正文"
    assert old_page_draft == "旧页面 r1 上仍由调用方保留的草稿"
    assert request._body["content"] == old_page_draft


def test_api_bucket_update_requires_baseline_when_authority_is_enabled(monkeypatch, tmp_path):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "bucket-update-missing-r1")

    response = asyncio.run(server.api_bucket_update(FakeRequest(
        {"content": "缺少编辑基线的正文", "idempotency_key": "missing-baseline"},
        path_params={"bucket_id": bucket_id},
    )))

    assert response.status_code == 400
    assert authority.get_memory(bucket_id)["active_revision"] == 1
    assert authority.get_memory_revision(bucket_id, 2) is None
    assert len(projection.revisions) == 1
    assert buckets.items[bucket_id]["content"] == "旧正文"


def test_api_bucket_update_latest_baseline_revision_and_idempotent_replay(monkeypatch, tmp_path):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "bucket-update-replay-r1")
    commit_background_revision(
        service, buckets, bucket_id, body="后台提交的 r2 正文", key="background-replay-r2"
    )
    request_body = {
        "content": "主人基于 r2 提交的 r3 正文",
        "expected_revision": 2,
        "idempotency_key": "latest-page-r2",
    }

    first = asyncio.run(server.api_bucket_update(FakeRequest(
        request_body, path_params={"bucket_id": bucket_id}
    )))
    replay = asyncio.run(server.api_bucket_update(FakeRequest(
        request_body, path_params={"bucket_id": bucket_id}
    )))

    assert first.status_code == replay.status_code == 200
    first_payload, replay_payload = response_json(first), response_json(replay)
    assert first_payload["revision"] == replay_payload["revision"] == 3
    assert first_payload == replay_payload
    assert authority.get_memory(bucket_id)["active_revision"] == 3
    assert authority.get_memory_revision(bucket_id, 3)["body_sha256"] == body_sha256(
        "主人基于 r2 提交的 r3 正文"
    )
    assert authority.get_memory_revision(bucket_id, 4) is None
    assert len(projection.revisions) == 3


def test_api_bucket_update_rejects_same_operation_id_with_different_content(monkeypatch, tmp_path):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "bucket-update-payload-conflict-r1")
    operation_id = "dashboard-operation-r2"

    first = asyncio.run(server.api_bucket_update(FakeRequest(
        {
            "content": "正文 r2 属于首次提交",
            "expected_revision": 1,
            "idempotency_key": operation_id,
        },
        path_params={"bucket_id": bucket_id},
    )))
    conflicting_replay = asyncio.run(server.api_bucket_update(FakeRequest(
        {
            "content": "同一操作号不可提交另一正文",
            "expected_revision": 1,
            "idempotency_key": operation_id,
        },
        path_params={"bucket_id": bucket_id},
    )))

    assert first.status_code == 200
    assert response_json(first)["revision"] == 2
    assert conflicting_replay.status_code == 409
    assert response_json(conflicting_replay)["error"] == "operation_payload_conflict"
    assert authority.get_memory(bucket_id)["active_revision"] == 2
    assert authority.get_memory_revision(bucket_id, 2)["body_sha256"] == body_sha256(
        "正文 r2 属于首次提交"
    )
    assert authority.get_memory_revision(bucket_id, 3) is None
    assert buckets.items[bucket_id]["content"] == "正文 r2 属于首次提交"
    assert len(projection.revisions) == 2


def test_bulk_expected_revisions_conflict_per_row_and_allow_other_rows_to_continue(
    monkeypatch, tmp_path
):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    stale_id = seed_authority_memory(buckets, authority, service, "bulk-stale-r1")
    fresh_id = seed_authority_memory(buckets, authority, service, "bulk-fresh-r1")
    commit_background_revision(
        service, buckets, stale_id, body="后台已有 r2", key="bulk-background-r2"
    )

    response = asyncio.run(server.api_buckets_bulk_update(FakeRequest({
        "bucket_ids": [stale_id, fresh_id],
        "expected_revisions": {stale_id: 1, fresh_id: 1},
        "tags_add": ["批量标签"],
        "idempotency_key": "bulk-baseline-r1",
    })))
    payload = response_json(response)
    results = {row["id"]: row for row in payload["results"]}

    assert response.status_code == 200
    assert payload["failed"] == 1
    assert payload["changed"] == 1
    assert results[stale_id]["status"] == "conflict"
    assert results[fresh_id]["status"] == "changed"
    assert authority.get_memory(stale_id)["active_revision"] == 2
    assert authority.get_memory_revision(stale_id, 3) is None
    assert "批量标签" not in buckets.items[stale_id]["metadata"].get("tags", [])
    assert authority.get_memory(fresh_id)["active_revision"] == 2
    assert "批量标签" in buckets.items[fresh_id]["metadata"]["tags"]
    assert len(projection.revisions) == 4


def test_bucket_update_preserves_activation_time_until_touch_and_does_not_infer_event_time(
    tmp_path,
):
    from bucket_manager import BucketManager

    manager = BucketManager({"buckets_dir": str(tmp_path / "buckets")})
    bucket_id = "bucket-time-fields-r1"
    created = "2026-09-20T00:00:00+00:00"

    async def exercise():
        await manager.create(
            "原始正文",
            bucket_id=bucket_id,
            created=created,
            last_active=created,
        )
        before = await manager.get(bucket_id)
        updated = await manager.update(
            bucket_id,
            content="普通正文编辑",
            name="普通元数据编辑",
            tags=["编辑标签"],
        )
        after_update = await manager.get(bucket_id)
        await manager.touch(bucket_id)
        after_touch = await manager.get(bucket_id)
        return before, updated, after_update, after_touch

    before, update_ok, after_update, after_touch = asyncio.run(exercise())
    assert update_ok is True
    before_meta = before["metadata"]
    updated_meta = after_update["metadata"]
    touched_meta = after_touch["metadata"]
    assert after_update["content"] == "普通正文编辑"
    assert updated_meta["name"] == "普通元数据编辑"
    assert updated_meta["tags"] == ["编辑标签"]
    assert updated_meta["last_active"] == before_meta["last_active"] == created
    assert updated_meta["activation_count"] == before_meta["activation_count"] == 0
    assert touched_meta["last_active"] != created
    assert touched_meta["activation_count"] == 1
    for metadata in (updated_meta, touched_meta):
        assert "date" not in metadata
        assert "event_date" not in metadata
        assert "event_time" not in metadata


def test_bulk_archive_activate_emit_memory_state_changed_without_direct_bucket_update(
    monkeypatch, tmp_path
):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "bulk-state-r1")

    archived = asyncio.run(server.api_buckets_bulk_update(FakeRequest(
        {"bucket_ids": [bucket_id], "status": "archived", "idempotency_key": "bulk-archive-r1"}
    )))
    assert archived.status_code == 200
    assert response_json(archived)["changed_ids"] == [bucket_id]
    assert authority.get_memory(bucket_id)["state"] == "archived"
    assert projection.states == [(bucket_id, "archived")]
    assert buckets.update_calls == []
    assert buckets.archive_calls == []
    assert outbox_event_types(authority).count("MemoryStateChanged") == 1

    activated = asyncio.run(server.api_buckets_bulk_update(FakeRequest(
        {"bucket_ids": [bucket_id], "status": "active", "idempotency_key": "bulk-activate-r1"}
    )))
    assert activated.status_code == 200
    assert response_json(activated)["changed_ids"] == [bucket_id]
    assert authority.get_memory(bucket_id)["state"] == "active"
    assert projection.states == [(bucket_id, "archived"), (bucket_id, "active")]
    assert buckets.update_calls == []
    assert buckets.activate_calls == []
    assert outbox_event_types(authority).count("MemoryStateChanged") == 2


def test_import_review_metadata_revision_and_delete_tombstone(monkeypatch, tmp_path):
    buckets, authority, service, projection = install(monkeypatch, tmp_path, authority_enabled=True)
    bucket_id = seed_authority_memory(buckets, authority, service, "import-review-r1")

    reviewed = asyncio.run(server.api_import_review(FakeRequest({
        "idempotency_key": "import-review-r1",
        "decisions": [{"bucket_id": bucket_id, "action": "important"}],
    })))
    assert reviewed.status_code == 200
    assert response_json(reviewed) == {"applied": 1, "errors": 0}
    assert authority.get_memory(bucket_id)["active_revision"] == 2
    assert authority.get_memory_revision(bucket_id, 2)["metadata"]["importance"] == 9
    assert len(projection.revisions) == 2
    assert buckets.update_calls == []

    deleted = asyncio.run(server.api_import_review(FakeRequest({
        "idempotency_key": "import-delete-r1",
        "decisions": [{"bucket_id": bucket_id, "action": "delete"}],
    })))
    assert deleted.status_code == 200
    assert response_json(deleted) == {"applied": 1, "errors": 0}
    memory = authority.get_memory(bucket_id)
    assert memory["state"] == "tombstoned"
    assert memory["recall_policy"] == "disabled"
    assert projection.deleted == [bucket_id]
    assert bucket_id not in buckets.items
    assert buckets.delete_calls == []
    assert outbox_event_types(authority).count("MemoryStateChanged") == 1


def test_authority_off_preserves_legacy_admin_bucket_operations(monkeypatch, tmp_path):
    buckets, authority, _service, projection = install(monkeypatch, tmp_path, authority_enabled=False)

    created = asyncio.run(server.api_create_memory(FakeRequest({
        "id": "legacy-api-r1",
        "title": "旧行为",
        "content": "legacy create",
        "idempotency_key": "legacy-create-r1",
    })))
    assert created.status_code == 200
    assert buckets.created and buckets.created[0]["id"] == "legacy-api-r1"
    assert authority is None
    assert projection.revisions == []

    updated = asyncio.run(server.api_create_memory(FakeRequest({
        "id": "legacy-api-r1",
        "title": "旧行为修订",
        "content": "legacy update",
        "idempotency_key": "legacy-update-r1",
    })))
    assert updated.status_code == 200
    assert buckets.update_calls

    bucket_update = asyncio.run(server.api_bucket_update(FakeRequest(
        {"content": "legacy dashboard update", "idempotency_key": "legacy-dashboard-r1"},
        path_params={"bucket_id": "legacy-api-r1"},
    )))
    assert bucket_update.status_code == 200
    assert len(buckets.update_calls) >= 2

    bulk_archive = asyncio.run(server.api_buckets_bulk_update(FakeRequest({
        "bucket_ids": ["legacy-api-r1"], "status": "archived",
    })))
    assert bulk_archive.status_code == 200
    assert buckets.archive_calls == ["legacy-api-r1"]
    assert projection.states == []

    bulk_activate = asyncio.run(server.api_buckets_bulk_update(FakeRequest({
        "bucket_ids": ["legacy-api-r1"], "status": "active",
    })))
    assert bulk_activate.status_code == 200
    assert buckets.activate_calls == ["legacy-api-r1"]
    assert projection.states == []

    reviewed = asyncio.run(server.api_import_review(FakeRequest({
        "decisions": [{"bucket_id": "legacy-api-r1", "action": "important"}],
    })))
    assert reviewed.status_code == 200
    assert response_json(reviewed) == {"applied": 1, "errors": 0}
    assert buckets.update_calls[-1][1]["importance"] == 9

    deleted = asyncio.run(server.api_import_review(FakeRequest({
        "decisions": [{"bucket_id": "legacy-api-r1", "action": "delete"}],
    })))
    assert deleted.status_code == 200
    assert response_json(deleted) == {"applied": 1, "errors": 0}
    assert buckets.delete_calls == ["legacy-api-r1"]
    assert "legacy-api-r1" not in buckets.items
