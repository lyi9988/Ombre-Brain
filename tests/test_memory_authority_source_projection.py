from __future__ import annotations

import hashlib

from memory_authority import MemoryAuthorityStore
from memory_authority_view import MemoryAuthorityRecallView


def _commit(authority: MemoryAuthorityStore, memory_id: str, bucket_id: str, body: str, revision: int = 1) -> str:
    digest = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id=memory_id,
        bucket_id=bucket_id,
        expected_revision=revision - 1,
        body_sha256=digest,
        snapshot_path=f"revisions/{memory_id}/{revision}.md",
        metadata={},
        source_refs=[f"test:{memory_id}:{revision}"],
        decision_source="test",
        idempotency_key=f"source-projection:{memory_id}:{revision}",
        actor="test",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
    authority.finalize_memory_commit(prepared["operation_id"])
    return digest


def _write_bucket(path, bucket_id: str, body: str, bucket_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nid: {bucket_id}\ntype: {bucket_type}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_source_projection_keeps_authority_claims_but_classifies_archive_and_missing(tmp_path):
    state_dir = tmp_path / "state"
    buckets_dir = tmp_path / "buckets"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    _commit(authority, "memory-active", "bucket-active", "current active body")
    _commit(authority, "memory-archived", "bucket-archived", "kept archived body")
    _commit(authority, "memory-missing", "bucket-missing", "unavailable body")
    active_path = buckets_dir / "dynamic" / "Personal" / "Active_bucket-active.md"
    _write_bucket(
        active_path,
        "bucket-active",
        "current active body",
        "dynamic",
    )
    _write_bucket(
        buckets_dir / "archive" / "Personal" / "Archived_bucket-archived.md",
        "bucket-archived",
        "kept archived body",
        "archived",
    )
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir),
        "buckets_dir": str(buckets_dir),
        "memory_authority": {"enabled": True},
    })

    snapshot = view.auto_recallable_source_projection()
    assert snapshot["authority_bucket_ids"] == frozenset({
        "bucket-active", "bucket-archived", "bucket-missing",
    })
    assert snapshot["recallable_bucket_ids"] == frozenset({"bucket-active"})
    assert snapshot["unavailable_bucket_ids_by_reason"] == {
        "archived": frozenset({"bucket-archived"}),
        "missing": frozenset({"bucket-missing"}),
        "ambiguous": frozenset(),
    }
    # The filesystem projection must not rewrite or hide authority facts.
    assert view.auto_recallable_bucket_ids() == snapshot["authority_bucket_ids"]

    archived_path = buckets_dir / "archive" / "Personal" / "Active_bucket-active.md"
    _write_bucket(archived_path, "bucket-active", "current active body", "archived")
    active_path.unlink()
    archived_snapshot = view.auto_recallable_source_projection(["bucket-active"])
    assert archived_snapshot["checked_bucket_ids"] == frozenset({"bucket-active"})
    assert archived_snapshot["recallable_bucket_ids"] == frozenset()
    assert archived_snapshot["unavailable_bucket_ids_by_reason"]["archived"] == frozenset({"bucket-active"})
    assert archived_snapshot["unavailable_bucket_ids_by_reason"]["missing"] == frozenset()


def test_source_projection_fails_closed_for_duplicate_active_archive_and_marker_conflict(tmp_path):
    state_dir = tmp_path / "state"
    buckets_dir = tmp_path / "buckets"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    _commit(authority, "memory-duplicate", "bucket-duplicate", "duplicate body")
    _commit(authority, "memory-marker", "bucket-marker", "marker body")
    _commit(authority, "memory-name-mismatch", "bucket-name-mismatch", "name mismatch body")
    _commit(authority, "memory-malformed", "bucket-malformed", "malformed body")
    _write_bucket(
        buckets_dir / "dynamic" / "Personal" / "One_bucket-duplicate.md",
        "bucket-duplicate",
        "duplicate body",
        "dynamic",
    )
    _write_bucket(
        buckets_dir / "archive" / "Personal" / "Two_bucket-duplicate.md",
        "bucket-duplicate",
        "duplicate body",
        "archived",
    )
    _write_bucket(
        buckets_dir / "dynamic" / "Personal" / "ArchivedMarker_bucket-marker.md",
        "bucket-marker",
        "marker body",
        "archived",
    )
    _write_bucket(
        buckets_dir / "dynamic" / "Personal" / "unexpected-filename.md",
        "bucket-name-mismatch",
        "name mismatch body",
        "dynamic",
    )
    malformed_path = buckets_dir / "dynamic" / "Personal" / "Broken_bucket-malformed.md"
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("not a Bucket frontmatter file", encoding="utf-8")
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir),
        "buckets_dir": str(buckets_dir),
        "memory_authority": {"enabled": True},
    })

    snapshot = view.auto_recallable_source_projection()
    assert snapshot["recallable_bucket_ids"] == frozenset()
    assert snapshot["unavailable_bucket_ids_by_reason"] == {
        "archived": frozenset({"bucket-marker"}),
        "missing": frozenset(),
        "ambiguous": frozenset({
            "bucket-duplicate", "bucket-name-mismatch", "bucket-malformed",
        }),
    }


def test_source_projection_preserves_active_revision_hash_without_falling_back_to_old_revision(tmp_path):
    state_dir = tmp_path / "state"
    buckets_dir = tmp_path / "buckets"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    old_hash = _commit(authority, "memory-revision", "bucket-revision", "old body", revision=1)
    current_hash = _commit(authority, "memory-revision", "bucket-revision", "current body", revision=2)
    _write_bucket(
        buckets_dir / "dynamic" / "Personal" / "Revision_bucket-revision.md",
        "bucket-revision",
        "old body",
        "dynamic",
    )
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir),
        "buckets_dir": str(buckets_dir),
        "memory_authority": {"enabled": True},
    })

    snapshot = view.auto_recallable_source_projection()
    index_metadata = view.memory_index_metadata_map(["bucket-revision"])
    assert snapshot["recallable_bucket_ids"] == frozenset({"bucket-revision"})
    assert index_metadata["bucket-revision"] == {
        "memory_id": "memory-revision",
        "revision": 2,
        "body_sha256": current_hash,
    }
    physical_sha = hashlib.sha256("old body".encode("utf-8")).hexdigest()
    assert old_hash == physical_sha
    assert physical_sha != index_metadata["bucket-revision"]["body_sha256"]
