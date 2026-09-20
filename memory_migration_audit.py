"""Read-only RECALL-R1 migration inventory.

The auditor intentionally has no apply mode.  It never initializes an authority
database, writes a Bucket, calls a model, or touches a derived index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from memory_commit_service import body_sha256, canonical_memory_body


LEGACY_STATUS_MAP = {
    "confirmed": "accepted",
    "applied": "accepted",
    "rejected": "rejected",
    "deferred": "deferred",
    "pending": "pending",
    "generated": "pending",
}


def _json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, canonical_memory_body(normalized)
    marker = normalized.find("\n---\n", 4)
    if marker < 0:
        return {}, canonical_memory_body(normalized)
    header = normalized[4:marker]
    body = normalized[marker + 5:]
    metadata = yaml.safe_load(header) or {}
    return (dict(metadata) if isinstance(metadata, dict) else {}), canonical_memory_body(body)


def _candidate_items(payload: Any) -> Iterable[dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else payload
    for item in items or []:
        if isinstance(item, dict):
            yield item


def _candidate_record(item: dict[str, Any]) -> dict[str, Any]:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
    candidate_id = str(candidate.get("id") or candidate.get("candidate_id") or item.get("id") or "").strip()
    legacy_status = str(item.get("status") or candidate.get("status") or "pending").strip().lower()
    status = LEGACY_STATUS_MAP.get(legacy_status, "pending")
    source_refs = candidate.get("source_event_ids") or candidate.get("source_turn_ids") or []
    if isinstance(source_refs, str):
        source_refs = [source_refs]
    expected_bucket_ids = []
    for value in (
        item.get("bucket_id"), candidate.get("target_bucket"),
        *(item.get("applied_bucket_ids") or []), *(candidate.get("applied_bucket_ids") or []),
    ):
        text = str(value or "").strip()
        if text and text not in expected_bucket_ids:
            expected_bucket_ids.append(text)
    if status == "accepted" and not expected_bucket_ids and candidate_id:
        expected_bucket_ids.append(candidate_id)
    body = str(candidate.get("proposed_memory") or candidate.get("content") or "").strip()
    return {
        "candidate_id": candidate_id,
        "legacy_status": legacy_status,
        "status": status,
        "body_sha256": body_sha256(body) if body else "",
        "source_status": str(candidate.get("source_verification") or candidate.get("source_status") or "legacy_unverified"),
        "source_ref_count": len([item for item in source_refs if str(item).strip()]),
        "expected_bucket_ids": expected_bucket_ids,
    }


class MemoryMigrationAuditor:
    def __init__(
        self,
        *,
        candidates_path: str | Path,
        buckets_dir: str | Path,
        identity_semantics_db_path: str | Path | None = None,
    ):
        self.candidates_path = Path(candidates_path).resolve()
        self.buckets_dir = Path(buckets_dir).resolve()
        self.identity_semantics_db_path = (
            Path(identity_semantics_db_path).resolve()
            if identity_semantics_db_path else None
        )

    def scan_identity_semantics(self, bucket_ids: set[str]) -> dict[str, Any]:
        path = self.identity_semantics_db_path
        if path is None:
            return {
                "configured": False,
                "path": "",
                "alias_count": 0,
                "evidence_count": 0,
                "eligible_alias_count": 0,
                "aliases_without_evidence": [],
                "orphan_evidence_bucket_ids": [],
                "errors": [],
            }
        if not path.exists():
            return {
                "configured": True,
                "path": str(path),
                "alias_count": 0,
                "evidence_count": 0,
                "eligible_alias_count": 0,
                "aliases_without_evidence": [],
                "orphan_evidence_bucket_ids": [],
                "errors": ["identity_semantics_db_missing"],
            }
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            aliases = conn.execute(
                "SELECT canonical,alias,scope,confidence,source FROM identity_aliases "
                "ORDER BY canonical,alias"
            ).fetchall()
            evidence = conn.execute(
                "SELECT canonical,alias,bucket_id FROM identity_alias_evidence "
                "ORDER BY canonical,alias,bucket_id"
            ).fetchall()
        except sqlite3.Error as exc:
            return {
                "configured": True,
                "path": str(path),
                "alias_count": 0,
                "evidence_count": 0,
                "eligible_alias_count": 0,
                "aliases_without_evidence": [],
                "orphan_evidence_bucket_ids": [],
                "errors": [f"identity_semantics_schema:{type(exc).__name__}"],
            }
        finally:
            conn.close()
        evidence_keys = {
            (str(row["canonical"]), str(row["alias"])) for row in evidence
        }
        aliases_without_evidence = [
            f"{row['canonical']}::{row['alias']}"
            for row in aliases
            if (str(row["canonical"]), str(row["alias"])) not in evidence_keys
        ]
        orphan_ids = sorted({
            str(row["bucket_id"])
            for row in evidence
            if str(row["bucket_id"]) not in bucket_ids
        })
        return {
            "configured": True,
            "path": str(path),
            "alias_count": len(aliases),
            "evidence_count": len(evidence),
            "eligible_alias_count": len(aliases) - len(aliases_without_evidence),
            "aliases_without_evidence": aliases_without_evidence,
            "orphan_evidence_bucket_ids": orphan_ids,
            "errors": [],
        }

    def scan_buckets(self) -> dict[str, Any]:
        records: dict[str, dict[str, Any]] = {}
        duplicate_ids: list[str] = []
        invalid_files: list[str] = []
        ring_count = 0
        for path in sorted(self.buckets_dir.rglob("*.md")):
            try:
                metadata, body = _frontmatter(path)
            except Exception:
                invalid_files.append(str(path.relative_to(self.buckets_dir)))
                continue
            bucket_id = str(metadata.get("id") or path.stem).strip()
            if bucket_id in records:
                duplicate_ids.append(bucket_id)
                continue
            comments = metadata.get("comments") if isinstance(metadata.get("comments"), list) else []
            ring_count += len(comments)
            records[bucket_id] = {
                "bucket_id": bucket_id,
                "relative_path": str(path.relative_to(self.buckets_dir)),
                "body_sha256": body_sha256(body),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "ring_count": len(comments),
                "source_candidate_id": str(metadata.get("daily_chat_memory_candidate_id") or ""),
                "memory_revision": int(metadata.get("memory_revision") or 0),
            }
        return {
            "count": len(records),
            "records": records,
            "duplicate_ids": sorted(set(duplicate_ids)),
            "invalid_files": invalid_files,
            "ring_count": ring_count,
        }

    def run(self) -> dict[str, Any]:
        payload = _json_file(self.candidates_path)
        candidates = [_candidate_record(item) for item in _candidate_items(payload)]
        bucket_scan = self.scan_buckets()
        buckets = bucket_scan.pop("records")
        identity_semantics = self.scan_identity_semantics(set(buckets))

        candidate_ids = [item["candidate_id"] for item in candidates if item["candidate_id"]]
        duplicate_candidates = sorted(
            candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1
        )
        statuses = Counter(item["legacy_status"] for item in candidates)
        mapped_statuses = Counter(item["status"] for item in candidates)
        accepted_missing_bucket = []
        accepted_body_mismatch = []
        unverified_sources = []
        mapped = 0
        for item in candidates:
            expected = item["expected_bucket_ids"]
            observed = [buckets[bucket_id] for bucket_id in expected if bucket_id in buckets]
            if observed:
                mapped += 1
            if item["status"] == "accepted" and not observed:
                accepted_missing_bucket.append(item["candidate_id"])
            if (
                item["status"] == "accepted"
                and item["body_sha256"]
                and observed
                and all(record["body_sha256"] != item["body_sha256"] for record in observed)
            ):
                accepted_body_mismatch.append(item["candidate_id"])
            if item["source_status"] != "verified" or item["source_ref_count"] == 0:
                unverified_sources.append(item["candidate_id"])

        inconsistencies = (
            len(duplicate_candidates)
            + len(bucket_scan["duplicate_ids"])
            + len(bucket_scan["invalid_files"])
            + len(accepted_missing_bucket)
            + len(accepted_body_mismatch)
            + len(identity_semantics["orphan_evidence_bucket_ids"])
            + len(identity_semantics["errors"])
        )
        return {
            "schema_version": "memory-migration-audit-v1",
            "mode": "read_only_dry_run",
            "candidates": {
                "scanned": len(candidates),
                "mapped_to_existing_bucket": mapped,
                "legacy_status_counts": dict(sorted(statuses.items())),
                "target_status_counts": dict(sorted(mapped_statuses.items())),
                "duplicate_candidate_ids": duplicate_candidates,
                "accepted_missing_bucket": accepted_missing_bucket,
                "accepted_body_mismatch": accepted_body_mismatch,
                "unverified_or_missing_source": unverified_sources,
            },
            "buckets": bucket_scan,
            "identity_semantics": identity_semantics,
            "inconsistency_count": inconsistencies,
            "side_effects": {
                "model_calls": 0,
                "bucket_writes": 0,
                "authority_db_writes": 0,
                "embedding_requests": 0,
                "canonical_memory_creates": 0,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RECALL-R1 migration audit")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--identity-semantics-db")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    report = MemoryMigrationAuditor(
        candidates_path=args.candidates,
        buckets_dir=args.buckets_dir,
        identity_semantics_db_path=args.identity_semantics_db,
    ).run()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
