"""File-backed Memory commit coordinator for RECALL-R1."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import frontmatter

from bucket_manager import BucketManager
from memory_authority import (
    BodyHashMismatch,
    CommitStateError,
    MemoryAuthorityStore,
    MemoryNotFound,
)


def canonical_memory_body(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def body_sha256(value: str) -> str:
    return hashlib.sha256(canonical_memory_body(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BucketProjectionResult:
    bucket_id: str
    revision: int
    operation_id: str
    body_sha256: str
    snapshot_path: str


class MemoryProjection(Protocol):
    async def write_revision(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        revision: int,
        expected_revision: int,
        body: str,
        metadata: Mapping[str, Any],
        operation_id: str,
        snapshot_path: str,
    ) -> BucketProjectionResult: ...

    async def append_ring(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        ring_id: str,
        content: str,
        kind: str,
        source_refs: Sequence[str],
        actor: str,
    ) -> None: ...


class ProjectionNotApplied(Exception):
    """A safe failure proving that the Bucket was not changed."""


class BucketMemoryProjection:
    """Compatibility projection preserving the existing Bucket format."""

    def __init__(self, config: Mapping[str, Any], bucket_manager: BucketManager | None = None):
        self.config = dict(config or {})
        self.bucket_manager = bucket_manager or BucketManager(self.config)
        state_dir = Path(str(self.config.get("state_dir") or "state"))
        self.revisions_dir = Path(
            str(self.config.get("memory_revision_dir") or state_dir / "memory_revisions")
        ).resolve()
        self.revisions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _metadata_fields(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        known_names = {
            "tags", "importance", "domain", "valence", "arousal", "bucket_type", "name",
            "pinned", "protected", "source", "created", "last_active", "updated_at", "anchor",
            "resolved", "digested", "confidence", "period", "date",
        }
        known = {key: value for key, value in dict(metadata or {}).items() if key in known_names}
        extra = {key: value for key, value in dict(metadata or {}).items() if key not in known_names}
        return known, extra

    def snapshot_path(self, memory_id: str, revision: int) -> Path:
        return self.revisions_dir / str(memory_id) / f"revision-{int(revision):08d}.md"

    async def write_revision(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        revision: int,
        expected_revision: int,
        body: str,
        metadata: Mapping[str, Any],
        operation_id: str,
        snapshot_path: str,
    ) -> BucketProjectionResult:
        canonical_body = canonical_memory_body(body)
        expected_hash = body_sha256(canonical_body)
        existing = await self.bucket_manager.get(bucket_id)
        if existing:
            current_metadata = dict(existing.get("metadata") or {})
            current_operation = str(current_metadata.get("memory_operation_id") or "")
            current_revision = int(current_metadata.get("memory_revision") or 0)
            if current_operation == operation_id:
                if body_sha256(existing.get("content") or "") != expected_hash:
                    raise BodyHashMismatch("idempotent Bucket projection body changed")
            else:
                if current_revision != int(expected_revision):
                    raise ProjectionNotApplied(
                        f"Bucket revision mismatch: expected {expected_revision}, current {current_revision}"
                    )
                known, extra = self._metadata_fields(metadata)
                extra.update({
                    "memory_id": memory_id,
                    "memory_revision": int(revision),
                    "memory_body_sha256": expected_hash,
                    "memory_operation_id": operation_id,
                    "memory_authority_schema": "memory-authority-v1",
                })
                updated = await self.bucket_manager.update(
                    bucket_id, content=canonical_body, extra_metadata=extra, **known
                )
                if not updated:
                    raise ProjectionNotApplied("Bucket update returned false")
        else:
            if int(expected_revision) != 0:
                raise ProjectionNotApplied("managed Bucket is missing for non-zero expected revision")
            known, extra = self._metadata_fields(metadata)
            extra.update({
                "memory_id": memory_id,
                "memory_revision": int(revision),
                "memory_body_sha256": expected_hash,
                "memory_operation_id": operation_id,
                "memory_authority_schema": "memory-authority-v1",
            })
            await self.bucket_manager.create(
                canonical_body, bucket_id=bucket_id, extra_metadata=extra, **known
            )

        current = await self.bucket_manager.get(bucket_id)
        if not current:
            raise CommitStateError("Bucket disappeared after projection")
        observed_hash = body_sha256(current.get("content") or "")
        current_metadata = dict(current.get("metadata") or {})
        if observed_hash != expected_hash:
            raise BodyHashMismatch("projected Bucket body hash differs")
        if str(current_metadata.get("memory_operation_id") or "") != operation_id:
            raise CommitStateError("projected Bucket operation marker differs")

        raw_path = Path(str(current["path"]))
        raw_text = raw_path.read_text(encoding="utf-8")
        target_snapshot = Path(snapshot_path).resolve()
        expected_snapshot = self.snapshot_path(memory_id, revision)
        if target_snapshot != expected_snapshot:
            raise CommitStateError("snapshot path is outside the canonical revision location")
        if target_snapshot.exists():
            if target_snapshot.read_text(encoding="utf-8") != raw_text:
                raise CommitStateError("immutable revision snapshot already exists with different bytes")
        else:
            target_snapshot.parent.mkdir(parents=True, exist_ok=True)
            BucketManager._atomic_write_text(str(target_snapshot), raw_text)

        return BucketProjectionResult(
            bucket_id=bucket_id,
            revision=int(revision),
            operation_id=operation_id,
            body_sha256=observed_hash,
            snapshot_path=str(target_snapshot),
        )

    async def append_ring(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        ring_id: str,
        content: str,
        kind: str,
        source_refs: Sequence[str],
        actor: str,
    ) -> None:
        entry = await self.bucket_manager.add_comment(
            bucket_id,
            canonical_memory_body(content),
            author=actor,
            kind=kind,
            source=(str(source_refs[0]) if source_refs else None),
            comment_id=ring_id,
        )
        if not entry or str(entry.get("id") or "") != ring_id:
            raise ProjectionNotApplied("Bucket ring projection failed")


class MemoryCommitService:
    def __init__(self, authority: MemoryAuthorityStore, projection: MemoryProjection):
        self.authority = authority
        self.projection = projection

    async def commit_memory(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        expected_revision: int,
        body: str,
        metadata: Mapping[str, Any],
        source_refs: Sequence[str],
        decision_source: str,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        canonical_body = canonical_memory_body(body)
        expected_hash = body_sha256(canonical_body)
        next_revision = int(expected_revision) + 1
        if isinstance(self.projection, BucketMemoryProjection):
            snapshot_path = str(self.projection.snapshot_path(memory_id, next_revision))
        else:
            snapshot_path = f"memory-revisions/{memory_id}/revision-{next_revision:08d}.md"
        prepared = self.authority.prepare_memory_commit(
            memory_id=memory_id,
            bucket_id=bucket_id,
            expected_revision=expected_revision,
            body_sha256=expected_hash,
            snapshot_path=snapshot_path,
            metadata=metadata,
            source_refs=source_refs,
            decision_source=decision_source,
            idempotency_key=idempotency_key,
            actor=actor,
        )
        if prepared.get("status") == "committed":
            return dict(prepared.get("result") or {})
        operation_id = str(prepared["operation_id"])
        commit = self.authority.get_commit(operation_id) or prepared
        if commit.get("status") == "body_written":
            return self.authority.finalize_memory_commit(operation_id)
        if commit.get("status") != "prepared":
            raise CommitStateError(f"cannot project commit from {commit.get('status')}")
        try:
            projected = await self.projection.write_revision(
                memory_id=memory_id,
                bucket_id=bucket_id,
                revision=next_revision,
                expected_revision=expected_revision,
                body=canonical_body,
                metadata=metadata,
                operation_id=operation_id,
                snapshot_path=snapshot_path,
            )
        except ProjectionNotApplied:
            self.authority.abort_memory_commit(operation_id, error_code="projection_not_applied")
            raise
        self.authority.record_body_written(
            operation_id, observed_body_sha256=projected.body_sha256
        )
        return self.authority.finalize_memory_commit(operation_id)

    async def append_ring(
        self,
        *,
        memory_id: str,
        content: str,
        kind: str,
        source_refs: Sequence[str],
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        memory = self.authority.get_memory(memory_id)
        if not memory:
            raise MemoryNotFound(memory_id)
        result = self.authority.append_ring(
            memory_id=memory_id,
            content=canonical_memory_body(content),
            kind=kind,
            source_refs=source_refs,
            idempotency_key=idempotency_key,
            actor=actor,
        )
        try:
            await self.projection.append_ring(
                memory_id=memory_id,
                bucket_id=str(memory["bucket_id"]),
                ring_id=str(result["ring_id"]),
                content=content,
                kind=kind,
                source_refs=source_refs,
                actor=actor,
            )
        except Exception as exc:
            self.authority.set_outbox_status(
                str(result["outbox_event_id"]), status="degraded", error=type(exc).__name__
            )
            raise
        self.authority.set_outbox_status(str(result["outbox_event_id"]), status="projected")
        return result
