from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from memory_authority import (
    IdempotencyConflict,
    MemoryAuthorityStore,
    RevisionConflict,
)


@pytest.fixture
def authority(tmp_path) -> MemoryAuthorityStore:
    return MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})


def _commit(
    authority: MemoryAuthorityStore,
    *,
    memory_id: str = "memory:cas",
    bucket_id: str = "bucket-cas",
    expected_revision: int = 0,
    body: str = "stable body",
) -> str:
    digest = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
    revision = expected_revision + 1
    prepared = authority.prepare_memory_commit(
        memory_id=memory_id,
        bucket_id=bucket_id,
        expected_revision=expected_revision,
        body_sha256=digest,
        snapshot_path=f"revisions/{memory_id}/{revision}.md",
        metadata={},
        source_refs=[f"test:{memory_id}:{revision}"],
        decision_source="test",
        idempotency_key=f"cas-commit:{memory_id}:{revision}",
        actor="test",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
    authority.finalize_memory_commit(prepared["operation_id"])
    return digest


def _state_event_count(authority: MemoryAuthorityStore) -> int:
    conn = authority._connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM outbox WHERE event_type='MemoryStateChanged'"
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def test_change_memory_state_cas_aligns_lifecycle_without_changing_revision_or_policy(authority):
    body_hash = _commit(authority)

    result = authority.change_memory_state(
        memory_id="memory:cas",
        state="archived",
        recall_policy="enabled",
        idempotency_key="align:bucket-cas:r1",
        actor="owner_repair",
        expected_revision=1,
        expected_state="active",
    )

    current = authority.get_memory("memory:cas")
    assert current is not None
    assert current["state"] == "archived"
    assert current["recall_policy"] == "enabled"
    assert current["active_revision"] == 1
    assert current["body_sha256"] == body_hash
    assert result["outbox_event_id"]
    assert _state_event_count(authority) == 1
    outbox = authority.pending_outbox(limit=10)
    event = next(item for item in outbox if item["event_type"] == "MemoryStateChanged")
    assert event["aggregate_revision"] == 1
    assert event["payload"] == {
        "memory_id": "memory:cas",
        "state": "archived",
        "recall_policy": "enabled",
    }


def test_change_memory_state_cas_rejects_stale_revision_or_state_without_writes(authority):
    _commit(authority, expected_revision=0, body="revision one")
    _commit(authority, expected_revision=1, body="revision two")

    with pytest.raises(RevisionConflict, match="expected revision 1"):
        authority.change_memory_state(
            memory_id="memory:cas",
            state="archived",
            recall_policy="enabled",
            idempotency_key="align:stale-revision",
            actor="owner_repair",
            expected_revision=1,
            expected_state="active",
        )
    assert authority.get_memory("memory:cas")["state"] == "active"
    assert _state_event_count(authority) == 0

    authority.change_memory_state(
        memory_id="memory:cas",
        state="archived",
        recall_policy="enabled",
        idempotency_key="align:first-state-change",
        actor="owner_repair",
    )
    with pytest.raises(RevisionConflict, match="expected state active"):
        authority.change_memory_state(
            memory_id="memory:cas",
            state="active",
            recall_policy="enabled",
            idempotency_key="align:stale-state",
            actor="owner_repair",
            expected_revision=2,
            expected_state="active",
        )
    assert authority.get_memory("memory:cas")["state"] == "archived"
    assert _state_event_count(authority) == 1


def test_change_memory_state_cas_serializes_competing_updates_and_replays_idempotently(authority):
    _commit(authority)
    barrier = Barrier(2)

    def apply(key: str):
        barrier.wait(timeout=5)
        try:
            return authority.change_memory_state(
                memory_id="memory:cas",
                state="archived",
                recall_policy="enabled",
                idempotency_key=key,
                actor="owner_repair",
                expected_revision=1,
                expected_state="active",
            )
        except RevisionConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            key: pool.submit(apply, key)
            for key in ("align:concurrent-a", "align:concurrent-b")
        }
        results = {key: future.result(timeout=10) for key, future in futures.items()}

    winners = [key for key, result in results.items() if isinstance(result, dict)]
    losers = [result for result in results.values() if isinstance(result, RevisionConflict)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert _state_event_count(authority) == 1

    winning_key = winners[0]
    original = results[winning_key]
    replay = authority.change_memory_state(
        memory_id="memory:cas",
        state="archived",
        recall_policy="enabled",
        idempotency_key=winning_key,
        actor="owner_repair",
        expected_revision=1,
        expected_state="active",
    )
    assert replay == original
    assert _state_event_count(authority) == 1

    with pytest.raises(IdempotencyConflict):
        authority.change_memory_state(
            memory_id="memory:cas",
            state="archived",
            recall_policy="enabled",
            idempotency_key=winning_key,
            actor="owner_repair",
            expected_revision=1,
            expected_state="archived",
        )
    assert _state_event_count(authority) == 1
