import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
MEMORY_ID = "managed-memory-id"
BUCKET_ID = "physical-bucket-id"


def _stack(tmp_path, *, authority_enabled=True):
    config = {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
        "memory_authority": {"enabled": authority_enabled},
        "decay": {"threshold": 0.3, "check_interval_hours": 0},
    }
    bucket_manager = BucketManager(config)
    authority = MemoryAuthorityStore(config)
    projection = BucketMemoryProjection(config, bucket_manager=bucket_manager)
    service = MemoryCommitService(authority, projection)
    return config, bucket_manager, authority, service


def _seed_managed(service, *, memory_id=MEMORY_ID, last_active="2025-12-01T00:00:00+00:00"):
    return asyncio.run(service.commit_memory(
        memory_id=memory_id,
        bucket_id=BUCKET_ID,
        expected_revision=0,
        body=BODY,
        metadata={
            "name": "secret bucket name",
            "importance": 5,
            "domain": ["private"],
            "bucket_type": "dynamic",
            "last_active": last_active,
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


@pytest.mark.parametrize("authority_enabled", [True, False])
def test_retired_decay_never_scans_archives_or_resolves(tmp_path, monkeypatch, authority_enabled):
    config, bucket_manager, authority, service = _stack(tmp_path, authority_enabled=authority_enabled)
    old_date = "2025-10-03T00:00:00+00:00"
    if authority_enabled:
        _seed_managed(service, last_active=old_date)
    else:
        asyncio.run(bucket_manager.create(
            BODY, bucket_id=BUCKET_ID, domain=["private"], importance=5, last_active=old_date,
        ))
    resolve_id = "old-low-importance"
    asyncio.run(bucket_manager.create(
        BODY, bucket_id=resolve_id, domain=["private"], importance=4,
        last_active="2025-12-01T00:00:00+00:00",
    ))
    engine = DecayEngine(config, bucket_manager, service if authority_enabled else None)
    monkeypatch.setattr(engine, "_now_naive_utc", lambda: datetime(2026, 1, 1))
    original_archive_candidate = asyncio.run(bucket_manager.get(BUCKET_ID))
    original_resolve_candidate = asyncio.run(bucket_manager.get(resolve_id))

    # Distinct old triggers: one falls below the normal archive threshold;
    # the other is over 30 days and importance 4 but still above that threshold.
    threshold = config["decay"]["threshold"]
    assert engine.calculate_score(original_archive_candidate["metadata"]) < threshold
    assert engine.calculate_score(original_resolve_candidate["metadata"]) > threshold
    assert not original_resolve_candidate["metadata"].get("resolved", False)
    original_authority = authority.get_memory_state_snapshot(MEMORY_ID)
    list_buckets = AsyncMock(side_effect=AssertionError("a retired cycle must not list buckets"))
    monkeypatch.setattr(bucket_manager, "list_all", list_buckets)

    result = asyncio.run(engine.run_decay_cycle())

    assert result["status"] == "disabled"
    assert result["archived"] == result["auto_resolved"] == result["checked"] == 0
    list_buckets.assert_not_awaited()
    assert asyncio.run(bucket_manager.get(BUCKET_ID)) == original_archive_candidate
    assert asyncio.run(bucket_manager.get(resolve_id)) == original_resolve_candidate
    assert authority.get_memory_state_snapshot(MEMORY_ID) == original_authority
    assert _state_rows(authority, MEMORY_ID) == []


def test_legacy_start_entry_points_never_schedule_a_task(monkeypatch):
    def unexpected_task(coroutine, *args, **kwargs):
        coroutine.close()
        pytest.fail("retired decay scheduled a task")

    engine = DecayEngine({"decay": {"check_interval_hours": 0}})
    monkeypatch.setattr(asyncio, "create_task", unexpected_task)

    async def exercise_old_entry_points():
        await engine.ensure_started()
        await engine.start()
        assert engine.is_running is False
        await engine.stop()

    asyncio.run(exercise_old_entry_points())


def test_explicit_resolution_and_archive_still_work(tmp_path, monkeypatch):
    import server

    _config, bucket_manager, authority, service = _stack(tmp_path)
    _seed_managed(service, memory_id=BUCKET_ID)
    monkeypatch.setattr(server, "bucket_mgr", bucket_manager)
    monkeypatch.setattr(server, "memory_authority_store", authority)
    monkeypatch.setattr(server, "memory_commit_service", service)

    response = asyncio.run(server.trace(BUCKET_ID, resolved=1, idempotency_key="explicit-resolve"))

    assert "resolved=True" in response
    assert asyncio.run(bucket_manager.get(BUCKET_ID))["metadata"]["resolved"] is True
    current = authority.get_memory(BUCKET_ID)
    assert current["state"] == "active"

    archived = asyncio.run(service.change_memory_state(
        memory_id=BUCKET_ID, state="archived", recall_policy="enabled",
        idempotency_key="explicit-archive", actor="owner", expected_revision=current["active_revision"],
    ))

    assert archived["state"] == "archived"
    assert Path(asyncio.run(bucket_manager.get(BUCKET_ID))["path"]).is_relative_to(Path(bucket_manager.archive_dir))
    assert len(_state_rows(authority, BUCKET_ID)) == 1


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
