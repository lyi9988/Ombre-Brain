"""Explicit, backup-first migration into the RECALL-R1 authority store."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from bucket_manager import BucketManager
from memory_authority import (
    MemoryAuthorityStore,
    MemoryProposal,
    PolicyDecision,
)
from memory_commit_service import body_sha256
from memory_migration_audit import (
    LEGACY_STATUS_MAP,
    MemoryMigrationAuditor,
    _candidate_items,
    _candidate_record,
    _frontmatter,
    _json_file,
)


class MigrationBlocked(RuntimeError):
    pass


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_refs(candidate: dict[str, Any]) -> list[str]:
    refs = []
    for value in candidate.get("source_event_ids") or []:
        if str(value).strip():
            refs.append(f"raw_event:{value}")
    for value in candidate.get("source_turn_ids") or []:
        if str(value).strip():
            refs.append(f"conversation_turn:{value}")
    return list(dict.fromkeys(refs))


def _legacy_proposal(item: dict[str, Any]) -> MemoryProposal:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
    return MemoryProposal.from_mapping({
        "proposal_id": str(candidate.get("id") or candidate.get("candidate_id") or item.get("id") or ""),
        "source_type": "daily_chat",
        "proposed_body": str(candidate.get("proposed_memory") or candidate.get("content") or ""),
        "original_excerpt": str(candidate.get("original_excerpt") or ""),
        "source_refs": _source_refs(candidate),
        "source_status": str(candidate.get("source_verification") or "legacy_unverified"),
        "memory_type": str(candidate.get("kind") or "key_event"),
        "confidence": candidate.get("confidence") or 0.0,
        "requested_mode": str(candidate.get("mode") or ("auto" if str(item.get("status")) == "applied" else "review")),
        "metadata": {"legacy_candidate": dict(candidate), "legacy_item": dict(item)},
    })


class MemoryAuthorityMigrator:
    def __init__(
        self,
        *,
        candidates_path: str | Path,
        buckets_dir: str | Path,
        state_dir: str | Path,
        authority_db_path: str | Path,
        backup_dir: str | Path,
        identity_semantics_db_path: str | Path | None = None,
    ):
        self.candidates_path = Path(candidates_path).resolve()
        self.buckets_dir = Path(buckets_dir).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.authority_db_path = Path(authority_db_path).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.identity_semantics_db_path = (
            Path(identity_semantics_db_path).resolve()
            if identity_semantics_db_path else None
        )
        self.revisions_dir = self.state_dir / "memory_revisions"

    def audit(self) -> dict[str, Any]:
        return MemoryMigrationAuditor(
            candidates_path=self.candidates_path,
            buckets_dir=self.buckets_dir,
            identity_semantics_db_path=self.identity_semantics_db_path,
        ).run()

    @staticmethod
    def _blocking_reasons(audit: dict[str, Any]) -> list[str]:
        candidates = audit["candidates"]
        buckets = audit["buckets"]
        reasons = []
        if candidates["duplicate_candidate_ids"]:
            reasons.append("duplicate_candidate_ids")
        if candidates["accepted_missing_bucket"]:
            reasons.append("accepted_missing_bucket")
        if buckets["duplicate_ids"]:
            reasons.append("duplicate_bucket_ids")
        if buckets["invalid_files"]:
            reasons.append("invalid_bucket_files")
        identity = audit.get("identity_semantics") or {}
        if identity.get("errors"):
            reasons.append("identity_semantics_unreadable")
        if identity.get("orphan_evidence_bucket_ids"):
            reasons.append("identity_semantics_orphan_evidence")
        return reasons

    def _backup(self, audit: dict[str, Any]) -> dict[str, Any]:
        self.backup_dir.mkdir(parents=True, exist_ok=False)
        try:
            os.chmod(self.backup_dir, 0o700)
        except OSError:
            pass
        candidate_backup = self.backup_dir / self.candidates_path.name
        shutil.copy2(self.candidates_path, candidate_backup)
        db_backup = self.backup_dir / "memory_authority.sqlite3.backup"
        db_absent = self.backup_dir / "memory_authority.sqlite3.absent"
        if self.authority_db_path.exists():
            source = sqlite3.connect(str(self.authority_db_path))
            target = sqlite3.connect(str(db_backup))
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        else:
            db_absent.write_text("absent before migration\n", encoding="utf-8")
        identity_backup = self.backup_dir / "identity_semantics.sqlite3.backup"
        if self.identity_semantics_db_path and self.identity_semantics_db_path.exists():
            source = sqlite3.connect(str(self.identity_semantics_db_path))
            target = sqlite3.connect(str(identity_backup))
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
        restore_map = self.backup_dir / "RESTORE-MAP.txt"
        restore_map.write_text(
            "Restore candidate JSON to:\n"
            f"  {self.candidates_path}\n"
            "Restore authority DB backup to:\n"
            f"  {self.authority_db_path}\n"
            "IdentitySemantic input is read-only; its optional backup is evidence only and is not rewritten.\n"
            "If the .absent marker exists, remove the newly-created authority DB instead.\n"
            "New immutable revision snapshots may be removed only after their paths are checked against MANIFEST.json.\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "memory-authority-migration-backup-v1",
            "candidates": {
                "source": str(self.candidates_path),
                "backup": str(candidate_backup),
                "sha256": _file_sha(candidate_backup),
            },
            "authority_db": {
                "source": str(self.authority_db_path),
                "backup": str(db_backup) if db_backup.exists() else "",
                "absent_marker": str(db_absent) if db_absent.exists() else "",
                "sha256": _file_sha(db_backup) if db_backup.exists() else "",
            },
            "identity_semantics_db": {
                "source": str(self.identity_semantics_db_path or ""),
                "backup": str(identity_backup) if identity_backup.exists() else "",
                "sha256": _file_sha(identity_backup) if identity_backup.exists() else "",
            },
            "restore_map": {"path": str(restore_map), "sha256": _file_sha(restore_map)},
            "audit": audit,
            "created_snapshots": [],
        }
        manifest_path = self.backup_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in (
            candidate_backup, db_backup, db_absent, identity_backup,
            restore_map, manifest_path,
        ):
            if path.exists():
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
        return manifest

    @staticmethod
    def _bucket_state(path: Path, metadata: dict[str, Any]) -> tuple[str, str]:
        state = "archived" if "archive" in path.parts or str(metadata.get("type")) == "archived" else "active"
        tags = {str(item) for item in (metadata.get("tags") or [])}
        if metadata.get("active") is False or metadata.get("deprecated"):
            recall_policy = "disabled"
        elif str(metadata.get("type") or "") == "feel" or tags & {"whisper", "daily_impression"}:
            recall_policy = "manual_only"
        else:
            recall_policy = "enabled"
        return state, recall_policy

    @staticmethod
    def _bucket_source_refs(metadata: dict[str, Any]) -> list[str]:
        refs = []
        fields = {
            "source_raw_event_ids": "raw_event",
            "source_conversation_turn_ids": "conversation_turn",
            "source_bucket_ids": "bucket",
            "source_persona_event_ids": "persona_event",
        }
        for field, prefix in fields.items():
            for value in metadata.get(field) or []:
                if str(value).strip():
                    refs.append(f"{prefix}:{value}")
        candidate_id = str(metadata.get("daily_chat_memory_candidate_id") or "").strip()
        if candidate_id:
            refs.append(f"candidate:{candidate_id}")
        return list(dict.fromkeys(refs))

    def apply(
        self,
        *,
        expected_candidates: int,
        expected_buckets: int,
        expected_aliases: int | None = None,
    ) -> dict[str, Any]:
        audit = self.audit()
        if int(audit["candidates"]["scanned"]) != int(expected_candidates):
            raise MigrationBlocked("candidate count differs from explicit expectation")
        if int(audit["buckets"]["count"]) != int(expected_buckets):
            raise MigrationBlocked("bucket count differs from explicit expectation")
        identity_audit = audit.get("identity_semantics") or {}
        if identity_audit.get("configured"):
            if expected_aliases is None:
                raise MigrationBlocked("identity alias count requires explicit expectation")
            if int(identity_audit.get("alias_count") or 0) != int(expected_aliases):
                raise MigrationBlocked("identity alias count differs from explicit expectation")
        reasons = self._blocking_reasons(audit)
        if reasons:
            raise MigrationBlocked(",".join(reasons))
        manifest = self._backup(audit)
        authority = MemoryAuthorityStore(str(self.authority_db_path))
        created_snapshots: list[str] = []
        imported_memories = imported_rings = 0
        imported_aliases = 0

        for path in sorted(self.buckets_dir.rglob("*.md")):
            metadata, body = _frontmatter(path)
            memory_id = str(metadata.get("id") or path.stem).strip()
            revision_path = self.revisions_dir / memory_id / "revision-00000001.md"
            raw_text = path.read_text(encoding="utf-8")
            if revision_path.exists():
                if revision_path.read_text(encoding="utf-8") != raw_text:
                    raise MigrationBlocked(f"revision snapshot conflict for {memory_id}")
            else:
                revision_path.parent.mkdir(parents=True, exist_ok=True)
                BucketManager._atomic_write_text(str(revision_path), raw_text)
                created_snapshots.append(str(revision_path))
            state, recall_policy = self._bucket_state(path.relative_to(self.buckets_dir), metadata)
            authority.import_legacy_memory(
                memory_id=memory_id,
                bucket_id=memory_id,
                body_sha256=body_sha256(body),
                snapshot_path=str(revision_path.resolve()),
                metadata=metadata,
                source_refs=self._bucket_source_refs(metadata),
                state=state,
                recall_policy=recall_policy,
            )
            imported_memories += 1
            comments = metadata.get("comments") if isinstance(metadata.get("comments"), list) else []
            for index, comment in enumerate(comments):
                if not isinstance(comment, dict) or not str(comment.get("content") or "").strip():
                    continue
                ring_id = str(comment.get("id") or f"legacy-ring-{memory_id}-{index + 1}")
                source_refs = [str(comment.get("source"))] if comment.get("source") else []
                authority.import_legacy_ring(
                    memory_id=memory_id,
                    ring_id=ring_id,
                    content=str(comment.get("content") or ""),
                    kind=str(comment.get("kind") or "comment"),
                    source_refs=source_refs,
                    actor=str(comment.get("author") or "legacy"),
                    created_at=str(comment.get("created") or ""),
                    metadata={
                        key: comment.get(key)
                        for key in ("valence", "arousal", "source")
                        if comment.get(key) is not None
                    },
                )
                imported_rings += 1

        if self.identity_semantics_db_path and self.identity_semantics_db_path.exists():
            conn = sqlite3.connect(
                f"file:{self.identity_semantics_db_path.as_posix()}?mode=ro",
                uri=True,
            )
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT a.canonical,a.alias,a.confidence,a.source,e.bucket_id "
                    "FROM identity_aliases a JOIN identity_alias_evidence e "
                    "ON e.canonical=a.canonical AND e.alias=a.alias "
                    "ORDER BY a.canonical,a.alias,e.bucket_id"
                ).fetchall()
            finally:
                conn.close()
            grouped: dict[tuple[str, str], list[str]] = {}
            for row in rows:
                grouped.setdefault(
                    (str(row["canonical"]), str(row["alias"])), []
                ).append(str(row["bucket_id"]))
            for (canonical, alias), bucket_ids in grouped.items():
                source_refs = [
                    f"memory:{bucket_id}"
                    for bucket_id in dict.fromkeys(bucket_ids)
                    if authority.get_memory(bucket_id)
                ]
                if not source_refs:
                    continue
                authority.upsert_alias(
                    entity_id=f"identity:{canonical}",
                    alias=alias,
                    trust="trusted_source",
                    source_refs=source_refs,
                )
                imported_aliases += 1

        candidate_payload = _json_file(self.candidates_path)
        imported_statuses: Counter[str] = Counter()
        for item in _candidate_items(candidate_payload):
            record = _candidate_record(item)
            proposal = _legacy_proposal(item)
            policy = PolicyDecision(
                action="queue_review",
                reason_codes=("LEGACY_IMPORT",),
                decision_policy="migration",
                alias_trust="weak",
                requires_owner_confirmation=True,
            )
            row = authority.put_candidate(proposal, policy)
            target = LEGACY_STATUS_MAP.get(record["legacy_status"], "pending")
            if target == "deferred":
                row = authority.decide_candidate(
                    proposal.proposal_id, action="defer", expected_revision=int(row["revision"]),
                    request_id=f"legacy-candidate:{proposal.proposal_id}:defer", actor="migration",
                )
            elif target == "rejected":
                row = authority.decide_candidate(
                    proposal.proposal_id, action="reject", expected_revision=int(row["revision"]),
                    request_id=f"legacy-candidate:{proposal.proposal_id}:reject", actor="migration",
                )
            elif target == "accepted":
                row = authority.decide_candidate(
                    proposal.proposal_id, action="accept", expected_revision=int(row["revision"]),
                    request_id=f"legacy-candidate:{proposal.proposal_id}:accept", actor="migration",
                )
                row = authority.decide_candidate(
                    proposal.proposal_id, action="commit", expected_revision=int(row["revision"]),
                    request_id=f"legacy-candidate:{proposal.proposal_id}:commit", actor="migration",
                )
            imported_statuses[str(row["status"])] += 1

        cursor = candidate_payload.get("cursor") if isinstance(candidate_payload, dict) else None
        if isinstance(cursor, dict):
            authority.set_meta("daily_chat_memory_cursor", cursor)

        manifest["created_snapshots"] = created_snapshots
        manifest_path = self.backup_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(manifest_path, 0o600)
        except OSError:
            pass
        return {
            "status": "applied",
            "authority_db": str(self.authority_db_path),
            "backup_dir": str(self.backup_dir),
            "imported_memories": imported_memories,
            "imported_rings": imported_rings,
            "imported_candidates": sum(imported_statuses.values()),
            "imported_aliases": imported_aliases,
            "candidate_status_counts": dict(sorted(imported_statuses.items())),
            "created_snapshots": len(created_snapshots),
            "bucket_writes": 0,
            "model_calls": 0,
            "embedding_requests": 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="RECALL-R1 authority migration")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--authority-db", required=True)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--identity-semantics-db")
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument("--expected-buckets", type=int)
    parser.add_argument("--expected-aliases", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    migrator = MemoryAuthorityMigrator(
        candidates_path=args.candidates,
        buckets_dir=args.buckets_dir,
        state_dir=args.state_dir,
        authority_db_path=args.authority_db,
        backup_dir=args.backup_dir,
        identity_semantics_db_path=args.identity_semantics_db,
    )
    if not args.apply:
        print(json.dumps(migrator.audit(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.expected_candidates is None or args.expected_buckets is None:
        parser.error("--apply requires --expected-candidates and --expected-buckets")
    result = migrator.apply(
            expected_candidates=args.expected_candidates,
            expected_buckets=args.expected_buckets,
            expected_aliases=args.expected_aliases,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
