import hashlib

import pytest

from memory_authority import (
    BodyHashMismatch,
    IdempotencyConflict,
    MemoryAuthorityStore,
    MemoryIngestionPolicy,
    MemoryProposal,
    RevisionConflict,
)


def proposal(**overrides):
    payload = {
        "proposal_id": "proposal-1",
        "source_type": "daily_chat",
        "proposed_body": "主人长期不喜欢苦味明显的黑咖啡。",
        "original_excerpt": "我真的不喜欢苦味明显的黑咖啡。",
        "source_refs": ["evt-1"],
        "source_status": "verified",
        "memory_type": "preference",
        "confidence": 0.91,
        "requested_mode": "auto",
    }
    payload.update(overrides)
    return MemoryProposal.from_mapping(payload)


def store(tmp_path):
    return MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})


def commit_first_memory(authority, *, idem="commit-1"):
    body = "主人长期不喜欢苦味明显的黑咖啡。"
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="memory-1",
        bucket_id="bucket-1",
        expected_revision=0,
        body_sha256=body_hash,
        snapshot_path="revisions/memory-1/1.md",
        metadata={"memory_type": "preference"},
        source_refs=["evt-1"],
        decision_source="auto",
        idempotency_key=idem,
        actor="policy",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=body_hash)
    return authority.finalize_memory_commit(prepared["operation_id"])


def test_policy_preserves_auto_and_routes_sensitive_types_to_review():
    policy = MemoryIngestionPolicy()

    automatic = policy.evaluate(proposal())
    assert automatic.action == "auto_accept"
    assert automatic.reason_codes == ("ACCEPT_AUTO_POLICY",)
    assert automatic.requires_owner_confirmation is False

    alias = policy.evaluate(proposal(memory_type="alias"))
    assert alias.action == "queue_review"
    assert alias.alias_trust == "weak"
    assert alias.requires_owner_confirmation is True


def test_explicit_owner_is_accepted_but_unverified_model_source_is_rejected():
    policy = MemoryIngestionPolicy()

    explicit = policy.evaluate(proposal(
        owner_explicit=True, source_refs=[], source_status="owner_direct", memory_type="alias"
    ))
    assert explicit.action == "auto_accept"
    assert explicit.alias_trust == "owner"

    unverified = policy.evaluate(proposal(source_status="unverified"))
    assert unverified.action == "reject"
    assert unverified.reason_codes == ("REJECT_SOURCE_UNVERIFIED",)


def test_candidate_identity_is_idempotent_and_payload_drift_conflicts(tmp_path):
    authority = store(tmp_path)
    policy = MemoryIngestionPolicy()
    item = proposal()
    decision = policy.evaluate(item)

    first = authority.put_candidate(item, decision)
    second = authority.put_candidate(item, decision)
    assert first["candidate_id"] == second["candidate_id"]
    assert first["status"] == "accepted"

    with pytest.raises(IdempotencyConflict):
        authority.put_candidate(proposal(proposed_body="内容被偷偷换掉。"), decision)


def test_candidate_decision_requires_expected_revision_and_request_is_idempotent(tmp_path):
    authority = store(tmp_path)
    item = proposal(requested_mode="review", confidence=0.4)
    created = authority.put_candidate(item, MemoryIngestionPolicy().evaluate(item))
    assert created["status"] == "pending"

    accepted = authority.decide_candidate(
        item.proposal_id,
        action="accept",
        expected_revision=1,
        request_id="decision-1",
        actor="owner",
        reason_codes=["OWNER_ACCEPTED"],
    )
    repeated = authority.decide_candidate(
        item.proposal_id,
        action="accept",
        expected_revision=1,
        request_id="decision-1",
        actor="owner",
        reason_codes=["OWNER_ACCEPTED"],
    )
    assert accepted["revision"] == 2
    assert repeated["revision"] == 2

    with pytest.raises(IdempotencyConflict):
        authority.decide_candidate(
            item.proposal_id,
            action="reject",
            expected_revision=2,
            request_id="decision-1",
            actor="owner",
        )


def test_idempotent_decision_returns_original_result_after_later_transition(tmp_path):
    authority = store(tmp_path)
    item = proposal(requested_mode="review", confidence=0.4)
    authority.put_candidate(item, MemoryIngestionPolicy().evaluate(item))

    rejected = authority.decide_candidate(
        item.proposal_id, action="reject", expected_revision=1,
        request_id="reject-1", actor="owner",
    )
    reopened = authority.decide_candidate(
        item.proposal_id, action="reopen", expected_revision=2,
        request_id="reopen-1", actor="owner",
    )
    repeated = authority.decide_candidate(
        item.proposal_id, action="reject", expected_revision=1,
        request_id="reject-1", actor="owner",
    )
    assert rejected["status"] == "rejected"
    assert reopened["status"] == "pending"
    assert repeated == rejected


def test_memory_commit_requires_observed_hash_then_activates_one_revision_and_outbox(tmp_path):
    authority = store(tmp_path)
    body_hash = hashlib.sha256("body".encode()).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="memory-1",
        bucket_id="bucket-1",
        expected_revision=0,
        body_sha256=body_hash,
        snapshot_path="revisions/memory-1/1.md",
        metadata={},
        source_refs=["evt-1"],
        decision_source="owner",
        idempotency_key="commit-1",
        actor="owner",
    )

    with pytest.raises(BodyHashMismatch):
        authority.record_body_written(prepared["operation_id"], observed_body_sha256="wrong")

    authority.record_body_written(prepared["operation_id"], observed_body_sha256=body_hash)
    result = authority.finalize_memory_commit(prepared["operation_id"])
    repeated = authority.finalize_memory_commit(prepared["operation_id"])
    assert result == repeated
    assert result["revision"] == 1
    assert authority.get_memory("memory-1")["active_revision"] == 1
    outbox = authority.pending_outbox()
    assert [item["event_type"] for item in outbox] == ["MemoryRevisionCommitted"]


def test_second_revision_checks_current_pointer(tmp_path):
    authority = store(tmp_path)
    commit_first_memory(authority)

    with pytest.raises(RevisionConflict):
        authority.prepare_memory_commit(
            memory_id="memory-1",
            bucket_id="bucket-1",
            expected_revision=0,
            body_sha256="sha",
            snapshot_path="revisions/memory-1/2.md",
            metadata={}, source_refs=["evt-2"], decision_source="owner",
            idempotency_key="commit-2", actor="owner",
        )


def test_ring_is_child_event_and_does_not_change_body_revision(tmp_path):
    authority = store(tmp_path)
    commit_first_memory(authority)
    before = authority.get_memory("memory-1")

    ring = authority.append_ring(
        memory_id="memory-1",
        content="现在回头看，我更理解主人当时的选择。",
        kind="feel",
        source_refs=["evt-2"],
        idempotency_key="ring-1",
        actor="guyan",
    )
    repeated = authority.append_ring(
        memory_id="memory-1",
        content="现在回头看，我更理解主人当时的选择。",
        kind="feel",
        source_refs=["evt-2"],
        idempotency_key="ring-1",
        actor="guyan",
    )
    after = authority.get_memory("memory-1")
    assert ring == repeated
    assert before["active_revision"] == after["active_revision"] == 1
    assert {item["event_type"] for item in authority.pending_outbox()} == {
        "MemoryRevisionCommitted", "MemoryRingAppended",
    }


def test_owner_alias_cannot_be_downgraded_by_automatic_rebuild(tmp_path):
    authority = store(tmp_path)
    owner = authority.upsert_alias(
        entity_id="guyan", alias="晏晏", trust="owner", source_refs=["owner-confirmation-1"]
    )
    rebuilt = authority.upsert_alias(
        entity_id="guyan", alias="晏晏", trust="auto", source_refs=["rebuild-1"]
    )
    assert owner["trust"] == rebuilt["trust"] == "owner"
    assert set(rebuilt["source_refs"]) == {"owner-confirmation-1", "rebuild-1"}
