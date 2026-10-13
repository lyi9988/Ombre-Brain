import asyncio
import json
from pathlib import Path

import memory_authority as authority_module
from bucket_manager import BucketManager
from decay_engine import DecayEngine
from memory_authority import MemoryAuthorityStore, RevisionConflict
from memory_commit_service import (
    BucketMemoryProjection,
    MemoryCommitService,
    body_sha256,
)


BODY = "主人把这段内容作为一条可保留正文。"
UPDATED_BODY = "主人后来修订了这条记忆正文。"
MEMORY_ID = "managed-memory-id"
BUCKET_ID = "physical-bucket-id"


def _stack(tmp_path, *, authority_enabled=True):
    config = {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
        "memory_authority": {"enabled": authority_enabled},
        "decay": {"threshold": 10000},
    }
    bucket_manager = BucketManager(config)
    authority = MemoryAuthorityStore(config)
    projection = BucketMemoryProjection(config, bucket_manager=bucket_manager)
    service = MemoryCommitService(authority, projection)
    return config, bucket_manager, authority, service


def _seed_managed(service):
    return asyncio.run(service.commit_memory(
        memory_id=MEMORY_ID,
        bucket_id=BUCKET_ID,
        expected_revision=0,
        body=BODY,
        metadata={
            "name": "secret bucket name",
            "importance": 5,
            "domain": ["private"],
            "bucket_type": "dynamic",
        },
        source_refs=["test:source"],
        decision_source="test",
        idempotency_key="seed-managed-decay-test",
        actor="test",
    ))


def _seed_authority_only(authority):
    prepared = authority.prepare_memory_commit(
        memory_id=MEMORY_ID,
        bucket_id=BUCKET_ID,
        expected_revision=0,
        body_sha256=body_sha256(BODY),
        snapshot_path=f"memory-revisions/{MEMORY_ID}/revision-00000001.md",
        metadata={"name": "secret bucket name"},
        source_refs=["test:source"],
        decision_source="test",
        idempotency_key="seed-authority-only-decay-test",
        actor="test",
    )
    authority.record_body_written(
        prepared["operation_id"], observed_body_sha256=body_sha256(BODY)
    )
    return authority.finalize_memory_commit(prepared["operation_id"])


def _count_archive_calls(bucket_manager):
    calls = []
    original = bucket_manager.archive

    async def counted(bucket_id):
        calls.append(str(bucket_id))
        return await original(bucket_id)

    bucket_manager.archive = counted
    return calls


def _state_rows(authority, memory_id):
    conn = authority._connect()
    try:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM outbox WHERE aggregate_id=? AND event_type='MemoryStateChanged' "
            "ORDER BY rowid",
            (memory_id,),
        ).fetchall()]
    finally:
        conn.close()


def test_decay_archives_managed_bucket_through_authority_projection_once(tmp_path):
    config, bucket_manager, authority, service = _stack(tmp_path)
    _seed_managed(service)
    original_revision = authority.get_memory(MEMORY_ID)["active_revision"]
    archive_calls = _count_archive_calls(bucket_manager)

    result = asyncio.run(DecayEngine(config, bucket_manager, service).run_decay_cycle())

    current = authority.get_memory(MEMORY_ID)
    archived_bucket = asyncio.run(bucket_manager.get(BUCKET_ID))
    assert result["archived"] == 1
    assert current["state"] == "archived"
    assert current["recall_policy"] == "enabled"
    assert current["active_revision"] == original_revision
    assert archived_bucket["content"] == BODY
    assert Path(archived_bucket["path"]).is_relative_to(Path(bucket_manager.archive_dir))
    assert archive_calls == [BUCKET_ID]
    events = _state_rows(authority, MEMORY_ID)
    assert len(events) == 1
    assert json.loads(events[0]["payload_json"])["state"] == "archived"


def test_decay_cas_rejects_owner_revision_race_without_archiving_or_leaking_text(
    tmp_path, caplog,
):
    config, bucket_manager, authority, service = _stack(tmp_path)
    _seed_managed(service)
    archive_calls = _count_archive_calls(bucket_manager)

    class RevisionRaceService:
        def __init__(self, inner):
            self.inner = inner
            self.authority = inner.authority
            self.raced = False

        async def change_memory_state(self, **kwargs):
            if not self.raced:
                self.raced = True
                await self.inner.commit_memory(
                    memory_id=MEMORY_ID,
                    bucket_id=BUCKET_ID,
                    expected_revision=1,
                    body=UPDATED_BODY,
                    metadata={
                        "name": "secret bucket name",
                        "importance": 5,
                        "domain": ["private"],
                        "bucket_type": "dynamic",
                    },
                    source_refs=["test:owner-edit"],
                    decision_source="owner",
                    idempotency_key="owner-revision-race",
                    actor="owner",
                )
            return await self.inner.change_memory_state(**kwargs)

    decay = DecayEngine(config, bucket_manager, RevisionRaceService(service))
    result = asyncio.run(decay.run_decay_cycle())

    current = authority.get_memory(MEMORY_ID)
    bucket = asyncio.run(bucket_manager.get(BUCKET_ID))
    assert result["archived"] == 0
    assert current["state"] == "active"
    assert current["active_revision"] == 2
    assert bucket["content"] == UPDATED_BODY
    assert Path(bucket["path"]).is_relative_to(Path(bucket_manager.dynamic_dir))
    assert archive_calls == []
    assert _state_rows(authority, MEMORY_ID) == []
    assert "secret bucket name" not in caplog.text
    assert BODY not in caplog.text
    assert UPDATED_BODY not in caplog.text


def test_decay_rearchives_after_owner_restore_at_same_revision(tmp_path):
    config, bucket_manager, authority, service = _stack(tmp_path)
    _seed_managed(service)
    archive_calls = _count_archive_calls(bucket_manager)
    decay = DecayEngine(config, bucket_manager, service)

    first = asyncio.run(decay.run_decay_cycle())
    archived = authority.get_memory(MEMORY_ID)
    restore = asyncio.run(service.change_memory_state(
        memory_id=MEMORY_ID,
        state="active",
        recall_policy=archived["recall_policy"],
        idempotency_key="owner-restore-same-revision",
        actor="owner",
        expected_revision=archived["active_revision"],
        expected_state="archived",
    ))
    second = asyncio.run(decay.run_decay_cycle())

    current = authority.get_memory(MEMORY_ID)
    bucket = asyncio.run(bucket_manager.get(BUCKET_ID))
    events = _state_rows(authority, MEMORY_ID)
    decay_keys = [
        row["idempotency_key"]
        for row in _commit_rows(authority, MEMORY_ID)
        if str(row["idempotency_key"]).startswith("decay-archive:")
    ]
    assert first["archived"] == 1
    assert restore["state"] == "active"
    assert second["archived"] == 1
    assert current["state"] == "archived"
    assert current["active_revision"] == 1
    assert bucket["content"] == BODY
    assert Path(bucket["path"]).is_relative_to(Path(bucket_manager.archive_dir))
    assert archive_calls == [BUCKET_ID, BUCKET_ID]
    assert len(events) == 3
    assert [json.loads(row["payload_json"])["state"] for row in events] == [
        "archived", "active", "archived",
    ]
    assert len(decay_keys) == 2
    assert len(set(decay_keys)) == 2


def _commit_rows(authority, memory_id):
    conn = authority._connect()
    try:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM commit_log WHERE aggregate_id=? ORDER BY rowid",
            (memory_id,),
        ).fetchall()]
    finally:
        conn.close()


def test_decay_rejects_owner_archive_restore_aba_at_same_body_revision(tmp_path):
    config, bucket_manager, authority, service = _stack(tmp_path)
    _seed_managed(service)
    archive_calls = _count_archive_calls(bucket_manager)

    class OwnerRestoreRace:
        def __init__(self):
            self.authority = authority

        async def change_memory_state(self, **kwargs):
            await service.change_memory_state(
                memory_id=MEMORY_ID, state="archived", recall_policy="enabled",
                idempotency_key="owner-aba-archive", actor="owner",
            )
            await service.change_memory_state(
                memory_id=MEMORY_ID, state="active", recall_policy="enabled",
                idempotency_key="owner-aba-restore", actor="owner",
            )
            return await service.change_memory_state(**kwargs)

    result = asyncio.run(DecayEngine(config, bucket_manager, OwnerRestoreRace()).run_decay_cycle())
    current = authority.get_memory(MEMORY_ID)
    bucket = asyncio.run(bucket_manager.get(BUCKET_ID))
    assert result["archived"] == 0
    assert current["state"] == "active"
    assert current["active_revision"] == 1
    assert bucket["content"] == BODY
    assert Path(bucket["path"]).is_relative_to(Path(bucket_manager.dynamic_dir))
    assert archive_calls == [BUCKET_ID]
    assert [json.loads(row["payload_json"])["state"] for row in _state_rows(authority, MEMORY_ID)] == [
        "archived", "active",
    ]


def test_legacy_bucket_uses_direct_archive_when_authority_has_no_row(tmp_path):
    config, bucket_manager, _authority, service = _stack(tmp_path)
    legacy_id = "legacy-decay-bucket"
    asyncio.run(bucket_manager.create(
        BODY,
        bucket_id=legacy_id,
        name="legacy bucket",
        domain=["private"],
    ))
    archive_calls = _count_archive_calls(bucket_manager)

    result = asyncio.run(DecayEngine(config, bucket_manager, service).run_decay_cycle())

    bucket = asyncio.run(bucket_manager.get(legacy_id))
    assert result["archived"] == 1
    assert archive_calls == [legacy_id]
    assert bucket["content"] == BODY
    assert Path(bucket["path"]).is_relative_to(Path(bucket_manager.archive_dir))


def test_enabled_authority_without_state_service_blocks_direct_archive(tmp_path):
    config, bucket_manager, _authority, _service = _stack(tmp_path)
    legacy_id = "cannot-classify-decay-bucket"
    asyncio.run(bucket_manager.create(
        BODY,
        bucket_id=legacy_id,
        name="secret bucket name",
        domain=["private"],
    ))
    archive_calls = _count_archive_calls(bucket_manager)

    result = asyncio.run(DecayEngine(config, bucket_manager).run_decay_cycle())

    bucket = asyncio.run(bucket_manager.get(legacy_id))
    assert result["archived"] == 0
    assert archive_calls == []
    assert Path(bucket["path"]).is_relative_to(Path(bucket_manager.dynamic_dir))


def test_authority_state_cas_conflict_writes_no_state_or_outbox_event(tmp_path):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    _seed_authority_only(authority)

    archived = authority.change_memory_state(
        memory_id=MEMORY_ID,
        state="archived",
        recall_policy="enabled",
        idempotency_key="authority-cas-archive",
        actor="decay",
        expected_revision=1,
        expected_state="active",
    )
    assert archived["state"] == "archived"
    before = _state_rows(authority, MEMORY_ID)

    try:
        authority.change_memory_state(
            memory_id=MEMORY_ID,
            state="active",
            recall_policy="enabled",
            idempotency_key="authority-cas-stale-revision",
            actor="owner",
            expected_revision=0,
            expected_state="archived",
        )
    except RevisionConflict:
        pass
    else:
        raise AssertionError("stale revision CAS should fail")

    current = authority.get_memory(MEMORY_ID)
    assert current["state"] == "archived"
    assert current["active_revision"] == 1
    assert _state_rows(authority, MEMORY_ID) == before


def test_state_generation_changes_when_owner_restores_same_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        authority_module, "_now", lambda: "2026-10-13T12:00:00+00:00"
    )
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    _seed_authority_only(authority)
    initial_updated_at = authority.get_memory(MEMORY_ID)["updated_at"]
    assert authority.get_memory_state_generation(MEMORY_ID) is None

    archived = authority.change_memory_state(
        memory_id=MEMORY_ID,
        state="archived",
        recall_policy="enabled",
        idempotency_key="generation-archive",
        actor="decay",
        expected_revision=1,
        expected_state="active",
    )
    archive_generation = authority.get_memory_state_generation(MEMORY_ID)
    restored = authority.change_memory_state(
        memory_id=MEMORY_ID,
        state="active",
        recall_policy="enabled",
        idempotency_key="generation-restore",
        actor="owner",
        expected_revision=1,
        expected_state="archived",
    )
    restore_generation = authority.get_memory_state_generation(MEMORY_ID)

    assert archived["state"] == "archived"
    assert restored["state"] == "active"
    assert archive_generation
    assert restore_generation
    assert restore_generation != archive_generation
    assert authority.get_memory(MEMORY_ID)["active_revision"] == 1
    assert initial_updated_at == authority.get_memory(MEMORY_ID)["updated_at"]


def test_memory_commit_service_forwards_cas_preconditions(tmp_path):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    _seed_authority_only(authority)
    service = MemoryCommitService(authority, projection=object())

    try:
        asyncio.run(service.change_memory_state(
            memory_id=MEMORY_ID,
            state="archived",
            recall_policy="enabled",
            idempotency_key="service-stale-cas",
            actor="decay",
            expected_revision=0,
            expected_state="active",
        ))
    except RevisionConflict:
        pass
    else:
        raise AssertionError("MemoryCommitService should forward stale CAS values")

    assert authority.get_memory(MEMORY_ID)["state"] == "active"
    assert _state_rows(authority, MEMORY_ID) == []
