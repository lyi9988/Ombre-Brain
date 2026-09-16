import hashlib
import asyncio

import pytest
import frontmatter

from bucket_manager import BucketManager
from memory_authority import (
    BodyHashMismatch,
    CommitStateError,
    IdempotencyConflict,
    MemoryAuthorityStore,
    MemoryIngestionPolicy,
    MemoryProposal,
    RevisionConflict,
)
from memory_commit_service import BucketProjectionResult, MemoryCommitService, ProjectionNotApplied
from memory_migration_audit import MemoryMigrationAuditor


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


def test_only_one_open_revision_commit_per_memory(tmp_path):
    authority = store(tmp_path)
    first = authority.prepare_memory_commit(
        memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
        body_sha256="sha-1", snapshot_path="revision-1.md", metadata={},
        source_refs=["evt-1"], decision_source="owner",
        idempotency_key="commit-open-1", actor="owner",
    )
    assert first["status"] == "prepared"
    with pytest.raises(CommitStateError):
        authority.prepare_memory_commit(
            memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
            body_sha256="sha-2", snapshot_path="revision-2.md", metadata={},
            source_refs=["evt-2"], decision_source="owner",
            idempotency_key="commit-open-2", actor="owner",
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


def test_bucket_writes_are_atomic_and_ring_id_can_follow_authority_id(tmp_path):
    if not hasattr(frontmatter, "Post"):
        pytest.skip("local environment has incompatible 'frontmatter' package")
    manager = BucketManager({"buckets_dir": str(tmp_path / "buckets")})
    bucket_id = asyncio.run(manager.create(
        "原始正文", bucket_id="bucket-1", name="测试记忆", domain=["测试"],
    ))
    assert bucket_id == "bucket-1"

    assert asyncio.run(manager.update(bucket_id, content="修订正文")) is True
    ring = asyncio.run(manager.add_comment(
        bucket_id,
        "后来产生的新感受。",
        kind="feel",
        comment_id="ring-authority-1",
        touch=False,
    ))
    assert ring["id"] == "ring-authority-1"

    path = manager._find_bucket_file(bucket_id)
    post = frontmatter.load(path)
    assert post.content == "修订正文"
    assert post["comments"][0]["id"] == "ring-authority-1"
    assert not list(tmp_path.rglob("*.tmp"))


class FakeProjection:
    def __init__(self):
        self.revisions = []
        self.rings = []
        self.fail_revision = False
        self.fail_ring = False

    async def write_revision(self, **kwargs):
        if self.fail_revision:
            raise ProjectionNotApplied("not written")
        self.revisions.append(dict(kwargs))
        return BucketProjectionResult(
            bucket_id=kwargs["bucket_id"], revision=kwargs["revision"],
            operation_id=kwargs["operation_id"],
            body_sha256=hashlib.sha256(kwargs["body"].encode("utf-8")).hexdigest(),
            snapshot_path=kwargs["snapshot_path"],
        )

    async def append_ring(self, **kwargs):
        if self.fail_ring:
            raise RuntimeError("ring projection failed")
        if not any(item["ring_id"] == kwargs["ring_id"] for item in self.rings):
            self.rings.append(dict(kwargs))


def test_commit_service_projects_once_and_idempotent_retry_returns_same_revision(tmp_path):
    authority = store(tmp_path)
    projection = FakeProjection()
    service = MemoryCommitService(authority, projection)
    kwargs = dict(
        memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
        body="正文", metadata={}, source_refs=["evt-1"], decision_source="owner",
        idempotency_key="service-commit-1", actor="owner",
    )
    first = asyncio.run(service.commit_memory(**kwargs))
    second = asyncio.run(service.commit_memory(**kwargs))
    assert first == second
    assert first["revision"] == 1
    assert len(projection.revisions) == 1


def test_definite_projection_failure_aborts_and_releases_open_commit(tmp_path):
    authority = store(tmp_path)
    projection = FakeProjection()
    projection.fail_revision = True
    service = MemoryCommitService(authority, projection)
    with pytest.raises(ProjectionNotApplied):
        asyncio.run(service.commit_memory(
            memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
            body="正文", metadata={}, source_refs=["evt-1"], decision_source="owner",
            idempotency_key="failed-service-1", actor="owner",
        ))
    assert authority.list_open_commits() == []


def test_ring_projection_failure_is_visible_as_degraded_and_not_a_second_ring(tmp_path):
    authority = store(tmp_path)
    projection = FakeProjection()
    service = MemoryCommitService(authority, projection)
    asyncio.run(service.commit_memory(
        memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
        body="正文", metadata={}, source_refs=["evt-1"], decision_source="owner",
        idempotency_key="service-commit-1", actor="owner",
    ))
    projection.fail_ring = True
    with pytest.raises(RuntimeError):
        asyncio.run(service.append_ring(
            memory_id="memory-1", content="年轮", kind="feel", source_refs=["evt-2"],
            idempotency_key="service-ring-1", actor="guyan",
        ))
    degraded = authority.pending_outbox()
    assert any(item["event_type"] == "MemoryRingAppended" and item["status"] == "degraded" for item in degraded)

    projection.fail_ring = False
    result = asyncio.run(service.append_ring(
        memory_id="memory-1", content="年轮", kind="feel", source_refs=["evt-2"],
        idempotency_key="service-ring-1", actor="guyan",
    ))
    assert len(projection.rings) == 1
    assert projection.rings[0]["ring_id"] == result["ring_id"]


def test_migration_audit_is_read_only_and_preserves_legacy_status_counts(tmp_path):
    candidates = tmp_path / "daily_chat_memory_candidates.json"
    candidates.write_text(
        """{
  "items": [
    {"status":"confirmed","bucket_id":"memory-1","candidate":{"id":"memory-1","proposed_memory":"正文一","source_verification":"verified","source_event_ids":["evt-1"]}},
    {"status":"pending","candidate":{"id":"memory-2","proposed_memory":"正文二","source_verification":"verified","source_event_ids":["evt-2"]}},
    {"status":"rejected","candidate":{"id":"memory-3","proposed_memory":"正文三","source_verification":"verified","source_event_ids":["evt-3"]}}
  ]
}""",
        encoding="utf-8",
    )
    bucket_dir = tmp_path / "buckets" / "dynamic" / "测试"
    bucket_dir.mkdir(parents=True)
    (bucket_dir / "memory-1.md").write_text(
        "---\nid: memory-1\ncomments:\n  - id: ring-1\n    content: 后来的感受\n---\n正文一\n",
        encoding="utf-8",
    )

    before = candidates.read_bytes()
    report = MemoryMigrationAuditor(
        candidates_path=candidates,
        buckets_dir=tmp_path / "buckets",
    ).run()
    assert report["mode"] == "read_only_dry_run"
    assert report["candidates"]["scanned"] == 3
    assert report["candidates"]["target_status_counts"] == {
        "accepted": 1, "pending": 1, "rejected": 1,
    }
    assert report["candidates"]["accepted_missing_bucket"] == []
    assert report["buckets"]["ring_count"] == 1
    assert all(value == 0 for value in report["side_effects"].values())
    assert candidates.read_bytes() == before
