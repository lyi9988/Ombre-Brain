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
from memory_authority_migrate import MemoryAuthorityMigrator, MigrationBlocked
from reflection_engine import ReflectionEngine


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

    commitment = policy.evaluate(proposal(memory_type="commitment"))
    assert commitment.action == "auto_accept"


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


def test_owner_edit_creates_candidate_revision_before_acceptance(tmp_path):
    authority = store(tmp_path)
    original = proposal(requested_mode="review", confidence=0.4)
    authority.put_candidate(original, MemoryIngestionPolicy().evaluate(original))
    edited = proposal(
        requested_mode="review", confidence=0.4,
        proposed_body="主人不喜欢纯黑咖啡，但可以接受加奶咖啡。",
    )
    revised = authority.revise_candidate(
        original.proposal_id,
        expected_revision=1,
        proposal=edited,
        request_id="edit-1",
        actor="owner",
    )
    accepted = authority.decide_candidate(
        original.proposal_id,
        action="accept",
        expected_revision=2,
        request_id="accept-after-edit",
        actor="owner",
    )
    assert revised["revision"] == 2
    assert revised["proposal"]["proposed_body"] == edited.proposed_body
    assert accepted["revision"] == 3


def test_candidate_acceptance_and_memory_commit_are_distinct_states(tmp_path):
    authority = store(tmp_path)
    item = proposal(requested_mode="review", confidence=0.4)
    authority.put_candidate(item, MemoryIngestionPolicy().evaluate(item))
    accepted = authority.decide_candidate(
        item.proposal_id, action="accept", expected_revision=1,
        request_id="accept-1", actor="owner",
    )
    committed = authority.decide_candidate(
        item.proposal_id, action="commit", expected_revision=2,
        request_id="candidate-commit-1", actor="commit_service",
    )
    assert accepted["status"] == "accepted"
    assert committed["status"] == "committed"


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
        self.retracted_rings = []
        self.states = []
        self.deleted_memories = []

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

    async def retract_ring(self, **kwargs):
        if kwargs["ring_id"] not in self.retracted_rings:
            self.retracted_rings.append(kwargs["ring_id"])

    async def delete_memory(self, **kwargs):
        self.deleted_memories.append(kwargs["memory_id"])

    async def set_memory_state(self, **kwargs):
        self.states.append((kwargs["memory_id"], kwargs["state"]))


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


def test_ring_retract_is_audited_and_idempotent(tmp_path):
    authority = store(tmp_path)
    projection = FakeProjection()
    service = MemoryCommitService(authority, projection)
    asyncio.run(service.commit_memory(
        memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
        body="正文", metadata={}, source_refs=["evt-1"], decision_source="owner",
        idempotency_key="commit-for-ring-retract", actor="owner",
    ))
    ring = asyncio.run(service.append_ring(
        memory_id="memory-1", content="年轮", kind="feel", source_refs=["evt-2"],
        idempotency_key="ring-before-retract", actor="guyan",
        metadata={"valence": 0.8, "source": "comment_bucket"},
    ))
    first = asyncio.run(service.retract_ring(
        memory_id="memory-1", ring_id=ring["ring_id"],
        idempotency_key="ring-retract-1", actor="guyan",
    ))
    second = asyncio.run(service.retract_ring(
        memory_id="memory-1", ring_id=ring["ring_id"],
        idempotency_key="ring-retract-1", actor="guyan",
    ))
    assert first == second
    assert first["status"] == "retracted"
    assert projection.retracted_rings == [ring["ring_id"]]


def test_memory_state_change_updates_authority_and_projection(tmp_path):
    authority = store(tmp_path)
    projection = FakeProjection()
    service = MemoryCommitService(authority, projection)
    asyncio.run(service.commit_memory(
        memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
        body="正文", metadata={}, source_refs=["evt-1"], decision_source="owner",
        idempotency_key="state-seed", actor="owner",
    ))
    archived = asyncio.run(service.change_memory_state(
        memory_id="memory-1", state="archived", recall_policy="disabled",
        idempotency_key="state-archive", actor="owner",
    ))
    assert archived["state"] == "archived"
    assert authority.get_memory("memory-1")["recall_policy"] == "disabled"
    assert projection.states == [("memory-1", "archived")]


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


def reflection_config(tmp_path, *, mode):
    return {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
        "gateway": {"prompt_plan_mirror_path": str(tmp_path / "prompt-plan.sqlite3")},
        "reflection": {
            "enabled": False,
            "daily_chat_memory_mode": mode,
            "daily_chat_memory_pending_path": str(tmp_path / "legacy-candidates.json"),
            "daily_chat_memory_requests_path": str(tmp_path / "requests.json"),
        },
        "memory_authority": {
            "enabled": True,
            "db_path": str(tmp_path / "memory-authority.sqlite3"),
        },
    }


def daily_candidate(candidate_id="daily-1"):
    return {
        "id": candidate_id,
        "date": "2026-09-16",
        "kind": "stable_preference",
        "title": "咖啡偏好",
        "content": "主人不喜欢纯黑咖啡。",
        "proposed_memory": "主人不喜欢纯黑咖啡。",
        "original_excerpt": "我不喜欢纯黑咖啡。",
        "source_verification": "verified",
        "source_hash": "abc123",
        "source_event_ids": [101],
        "source_turn_ids": [51],
        "confidence": 0.91,
        "tags": ["from_daily_chat", "stable_preference"],
        "domain": ["日常"],
        "importance": 6,
        "valence": 0.5,
        "arousal": 0.3,
        "soft_flags": [],
    }


def attach_fake_commit_service(engine):
    projection = FakeProjection()
    service = MemoryCommitService(engine.memory_authority_store, projection)
    engine._memory_authority_service = lambda _bucket_mgr: service
    return projection


def test_daily_auto_uses_authority_and_commits_without_legacy_json(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode="auto"))
    projection = attach_fake_commit_service(engine)
    result = asyncio.run(engine._write_daily_chat_memory_candidates(
        [daily_candidate()], object(), embedding_engine=None,
    ))
    row = engine.memory_authority_store.get_candidate("daily-1")
    assert result["created"] == 1
    assert row["status"] == "committed"
    assert len(projection.revisions) == 1
    assert not (tmp_path / "legacy-candidates.json").exists()


def test_daily_review_lists_and_confirms_through_same_authority(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode="review"))
    projection = attach_fake_commit_service(engine)
    pending = engine._store_daily_chat_memory_pending([{
        **daily_candidate(), "mode": "review", "status": "pending",
    }])
    listed = engine.list_daily_chat_memory_pending(status="pending")
    assert pending["added"] == 1
    assert listed[0]["id"] == "daily-1"
    assert listed[0]["status"] == "pending"

    result = asyncio.run(engine.confirm_daily_chat_memory(
        ["daily-1"], object(), action="confirm", request_id="owner-confirm-1",
    ))
    row = engine.memory_authority_store.get_candidate("daily-1")
    confirmed = engine.list_daily_chat_memory_pending(status="confirmed")
    assert result["created"] == 1
    assert row["status"] == "committed"
    assert confirmed[0]["id"] == "daily-1"
    assert len(projection.revisions) == 1


def test_daily_mode_conflict_fails_closed_when_authority_is_enabled(tmp_path):
    config = reflection_config(tmp_path, mode="review")
    config["memory_authority"]["ingestion_policy"] = {
        "source_modes": {"daily_chat": "auto"}
    }
    with pytest.raises(ValueError, match="daily_chat_memory_mode"):
        ReflectionEngine(config)


def test_background_reflection_uses_revisioned_memory_and_manual_recall_policy(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode="review"))
    projection = attach_fake_commit_service(engine)
    first = asyncio.run(engine._commit_background_memory(
        bucket_mgr=object(),
        memory_id="reflection_daily_2026-09-16",
        body="第一版日印象",
        metadata={"confidence": 0.9, "bucket_type": "feel"},
        source_type="reflection",
        memory_type="daily_impression",
        source_refs=["conversation_turn:1"],
        recall_policy="manual_only",
    ))
    second = asyncio.run(engine._commit_background_memory(
        bucket_mgr=object(),
        memory_id="reflection_daily_2026-09-16",
        body="修订后的日印象",
        metadata={"confidence": 0.9, "bucket_type": "feel"},
        source_type="reflection",
        memory_type="daily_impression",
        source_refs=["conversation_turn:1", "conversation_turn:2"],
        recall_policy="manual_only",
    ))
    memory = engine.memory_authority_store.get_memory("reflection_daily_2026-09-16")
    assert first["status"] == "created"
    assert second["status"] == "updated"
    assert memory["active_revision"] == 2
    assert memory["recall_policy"] == "manual_only"
    assert len(projection.revisions) == 2


def test_apply_migration_preserves_bucket_bytes_and_imports_revision_ring_and_candidate(tmp_path):
    candidates = tmp_path / "daily_chat_memory_candidates.json"
    candidates.write_text(
        """{
  "items": [
    {"status":"confirmed","bucket_id":"memory-1","candidate":{"id":"memory-1","mode":"review","proposed_memory":"正文一","original_excerpt":"原文","source_verification":"verified","source_event_ids":["evt-1"]}},
    {"status":"pending","candidate":{"id":"memory-2","mode":"review","proposed_memory":"正文二","original_excerpt":"原文二","source_verification":"verified","source_event_ids":["evt-2"]}}
  ],
  "cursor": {"raw_events": {"default": {"last_raw_event_id": 99}}}
}""",
        encoding="utf-8",
    )
    bucket_dir = tmp_path / "buckets" / "dynamic" / "测试"
    bucket_dir.mkdir(parents=True)
    bucket = bucket_dir / "memory-1.md"
    bucket.write_text(
        "---\nid: memory-1\ncomments:\n  - id: ring-1\n    kind: feel\n    author: 顾衍\n    created: '2026-09-16T01:00:00+00:00'\n    content: 后来的感受\n---\n正文一\n",
        encoding="utf-8",
    )
    before_bucket = bucket.read_bytes()
    migrator = MemoryAuthorityMigrator(
        candidates_path=candidates,
        buckets_dir=tmp_path / "buckets",
        state_dir=tmp_path / "state",
        authority_db_path=tmp_path / "state" / "memory_authority.sqlite3",
        backup_dir=tmp_path / "backup",
    )
    result = migrator.apply(expected_candidates=2, expected_buckets=1)
    authority = MemoryAuthorityStore(str(tmp_path / "state" / "memory_authority.sqlite3"))

    assert result["imported_memories"] == 1
    assert result["imported_rings"] == 1
    assert result["candidate_status_counts"] == {"committed": 1, "pending": 1}
    assert bucket.read_bytes() == before_bucket
    assert authority.get_memory("memory-1")["active_revision"] == 1
    assert authority.get_candidate("memory-1")["status"] == "committed"
    assert authority.get_candidate("memory-2")["status"] == "pending"
    assert authority.get_meta("daily_chat_memory_cursor")["raw_events"]["default"]["last_raw_event_id"] == 99
    assert (tmp_path / "backup" / "MANIFEST.json").exists()
    assert (tmp_path / "state" / "memory_revisions" / "memory-1" / "revision-00000001.md").exists()


def test_apply_migration_requires_explicit_counts_and_blocks_missing_confirmed_bucket(tmp_path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        '{"items":[{"status":"confirmed","candidate":{"id":"missing","proposed_memory":"正文"}}]}',
        encoding="utf-8",
    )
    (tmp_path / "buckets").mkdir()
    migrator = MemoryAuthorityMigrator(
        candidates_path=candidates,
        buckets_dir=tmp_path / "buckets",
        state_dir=tmp_path / "state",
        authority_db_path=tmp_path / "state" / "memory_authority.sqlite3",
        backup_dir=tmp_path / "backup",
    )
    with pytest.raises(MigrationBlocked, match="accepted_missing_bucket"):
        migrator.apply(expected_candidates=1, expected_buckets=0)
    assert not (tmp_path / "backup").exists()
    assert not (tmp_path / "state" / "memory_authority.sqlite3").exists()
