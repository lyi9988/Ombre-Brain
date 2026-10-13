"""Read-only Gateway view of committed Memory authority facts."""
from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import yaml

from memory_authority import normalize_alias


class MemoryAuthorityRecallView:
    def __init__(self, config: dict[str, Any]):
        authority_cfg = config.get("memory_authority", {}) if isinstance(
            config.get("memory_authority", {}), dict) else {}
        state_dir = str(config.get("state_dir") or config.get("buckets_dir") or "state")
        buckets_dir = str(config.get("buckets_dir") or "").strip()
        self.buckets_dir = Path(buckets_dir).resolve() if buckets_dir else None
        self.enabled = bool(authority_cfg.get("enabled", False))
        self.path = Path(
            str(authority_cfg.get("db_path") or os.path.join(state_dir, "memory_authority.sqlite3"))
        ).resolve()
        self._stamp: tuple[int, int, int, int] | None = None
        self._trusted_aliases: list[dict[str, Any]] = []
        self._auto_bucket_stamp: tuple[int, int, int, int] | None = None
        self._auto_bucket_ids: frozenset[str] = frozenset()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=2.0)

    def _file_stamp(self) -> tuple[int, int, int, int] | None:
        try:
            stat = self.path.stat()
            wal_path = Path(str(self.path) + "-wal")
            try:
                wal = wal_path.stat()
                wal_stamp = (int(wal.st_mtime_ns), int(wal.st_size))
            except FileNotFoundError:
                wal_stamp = (0, 0)
            return int(stat.st_mtime_ns), int(stat.st_size), *wal_stamp
        except FileNotFoundError:
            return None

    def available(self) -> bool:
        return bool(self.enabled and self.path.exists())

    def auto_recallable_bucket_ids(self) -> frozenset[str]:
        """Only committed, active, auto-enabled Memory may enter chat recall.

        The Bucket files remain a readable projection, not a second policy
        authority. A missing/unreadable Authority fails closed when enabled.
        """
        stamp = self._file_stamp()
        if not self.available():
            self._auto_bucket_stamp = stamp
            self._auto_bucket_ids = frozenset()
            return self._auto_bucket_ids
        if stamp == self._auto_bucket_stamp:
            return self._auto_bucket_ids
        conn = None
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT bucket_id FROM memories "
                "WHERE state='active' AND recall_policy='enabled' "
                "AND bucket_id IS NOT NULL AND bucket_id!=''"
            ).fetchall()
        except sqlite3.Error:
            self._auto_bucket_ids = frozenset()
            self._auto_bucket_stamp = None
        else:
            self._auto_bucket_ids = frozenset(str(row[0]) for row in rows)
            self._auto_bucket_stamp = stamp
        finally:
            if conn is not None:
                conn.close()
        return self._auto_bucket_ids

    @staticmethod
    def _read_bucket_projection_markers(path: Path) -> dict[str, str] | None:
        """Read only Bucket frontmatter identity/type, never its body."""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                if handle.readline().strip() != "---":
                    return None
                header: list[str] = []
                header_bytes = 0
                for line in handle:
                    if line.strip() in {"---", "..."}:
                        value = yaml.safe_load("".join(header)) or {}
                        if not isinstance(value, dict):
                            return None
                        return {
                            "id": str(value.get("id") or "").strip(),
                            "type": str(value.get("type") or value.get("bucket_type") or "").strip().lower(),
                        }
                    header_bytes += len(line.encode("utf-8", errors="replace"))
                    if header_bytes > 64 * 1024:
                        return None
                    header.append(line)
        except (OSError, UnicodeError, yaml.YAMLError):
            return None
        return None

    def auto_recallable_source_projection(
        self, bucket_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Compare authoritative auto-recall claims with physical Bucket lifecycle.

        Authority remains the policy source. This read-only view qualifies a
        committed source only when one non-archived projection is present under
        permanent/dynamic/feel. It reports stale archived/missing claims instead
        of deleting them from the authority set, so callers can keep Recall
        incomplete visible without reviving archived files or rewriting facts.
        """
        authority_ids = frozenset(self.auto_recallable_bucket_ids())
        if bucket_ids is None:
            checked_ids = authority_ids
        else:
            requested_ids = frozenset(
                str(value or "").strip() for value in bucket_ids if str(value or "").strip()
            )
            checked_ids = authority_ids & requested_ids
        by_id: dict[str, list[tuple[str, str]]] = {bucket_id: [] for bucket_id in checked_ids}
        uncertain: set[str] = set()
        if not checked_ids or self.buckets_dir is None:
            return {
                "authority_bucket_ids": authority_ids,
                "checked_bucket_ids": checked_ids,
                "recallable_bucket_ids": frozenset(),
                "unavailable_bucket_ids_by_reason": {
                    "missing": checked_ids,
                    "archived": frozenset(),
                    "ambiguous": frozenset(),
                },
            }

        for category in ("permanent", "dynamic", "feel", "archive"):
            directory = self.buckets_dir / category
            if not directory.is_dir():
                continue
            try:
                files = directory.rglob("*.md")
                for path in files:
                    stem = path.stem
                    candidates = [
                        bucket_id for bucket_id in checked_ids
                        if stem == bucket_id or stem.endswith(f"_{bucket_id}")
                    ]
                    markers = self._read_bucket_projection_markers(path)
                    if markers is None or not markers.get("id"):
                        uncertain.update(candidates)
                        continue
                    actual_id = markers["id"]
                    if actual_id in checked_ids:
                        if actual_id not in candidates:
                            # BucketManager can list this file, but its lookup by
                            # ID cannot locate it from the filename convention.
                            uncertain.add(actual_id)
                        by_id[actual_id].append((category, markers.get("type", "")))
                    elif candidates:
                        uncertain.update(candidates)
            except OSError:
                # A partial walk cannot prove an ID is absent or active.
                uncertain.update(checked_ids)

        recallable: set[str] = set()
        archived: set[str] = set()
        ambiguous = set(uncertain)
        for bucket_id, locations in by_id.items():
            if bucket_id in uncertain:
                continue
            categories = {category for category, _ in locations}
            archived_marker = any(kind == "archived" for _, kind in locations)
            active_categories = categories & {"permanent", "dynamic", "feel"}
            if len(locations) > 1 or (active_categories and "archive" in categories):
                ambiguous.add(bucket_id)
            elif "archive" in categories or archived_marker:
                archived.add(bucket_id)
            elif len(active_categories) == 1 and len(locations) == 1:
                recallable.add(bucket_id)

        missing = set(checked_ids) - recallable - archived - ambiguous
        return {
            "authority_bucket_ids": authority_ids,
            "checked_bucket_ids": checked_ids,
            "recallable_bucket_ids": frozenset(recallable),
            "unavailable_bucket_ids_by_reason": {
                "archived": frozenset(archived),
                "missing": frozenset(missing),
                "ambiguous": frozenset(ambiguous),
            },
        }

    def trusted_aliases(self) -> list[dict[str, Any]]:
        stamp = self._file_stamp()
        if not self.available():
            self._stamp = stamp
            self._trusted_aliases = []
            return []
        if stamp == self._stamp:
            return [dict(item) for item in self._trusted_aliases]
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT alias_id,entity_id,alias,normalized_alias,trust,revision,"
                "source_refs_json,updated_at "
                "FROM entity_aliases WHERE state='active' AND trust IN ('owner','trusted_source') "
                "ORDER BY LENGTH(normalized_alias) DESC,alias_id"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            conn.close()
        self._stamp = stamp
        aliases = []
        for row in rows:
            item = dict(row)
            try:
                item["source_refs"] = list(json.loads(item.pop("source_refs_json") or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                item.pop("source_refs_json", None)
                item["source_refs"] = []
            aliases.append(item)
        self._trusted_aliases = aliases
        return [dict(item) for item in self._trusted_aliases]

    def match_aliases(self, query: str) -> list[dict[str, Any]]:
        normalized_query = normalize_alias(query).replace(" ", "")
        if not normalized_query:
            return []
        matches = []
        for row in self.trusted_aliases():
            key = normalize_alias(str(row.get("normalized_alias") or row.get("alias") or "")).replace(" ", "")
            if key and key in normalized_query:
                item = dict(row)
                memory_ids = [
                    str(ref)[len("memory:"):]
                    for ref in item.get("source_refs", []) or []
                    if str(ref or "").startswith("memory:")
                    and len(str(ref or "")) > len("memory:")
                ]
                resolved = self.resolve_memory_refs(memory_ids)
                item["memory_ids"] = list(resolved)
                item["bucket_ids"] = list(dict.fromkeys(
                    str(value.get("bucket_id") or "")
                    for value in resolved.values()
                    if str(value.get("bucket_id") or "")
                ))
                # A trusted name alone is not recall evidence.  If its
                # linked Memory was archived, deleted, or never committed,
                # it must not force a Fast route that cannot return a card.
                if not item["bucket_ids"]:
                    continue
                matches.append(item)
        return matches[:8]

    def watermark(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "memory_count": 0, "alias_revision": 0}
        conn = self._connect()
        try:
            memory_count = int(conn.execute(
                "SELECT COUNT(*) FROM memories WHERE state='active'"
            ).fetchone()[0])
            alias_revision = int(conn.execute(
                "SELECT COALESCE(MAX(revision),0) FROM entity_aliases WHERE state='active'"
            ).fetchone()[0])
            memory_revision = int(conn.execute(
                "SELECT COALESCE(SUM(active_revision),0) FROM memories WHERE state='active'"
            ).fetchone()[0])
        except sqlite3.Error:
            return {"available": False, "memory_count": 0, "alias_revision": 0}
        finally:
            conn.close()
        return {
            "available": True,
            "memory_count": memory_count,
            "alias_revision": alias_revision,
            "memory_revision_watermark": memory_revision,
            "db_stamp": self._file_stamp(),
        }

    @staticmethod
    def _decoded_row(row: sqlite3.Row, *json_fields: str) -> dict[str, Any]:
        value = dict(row)
        for field in json_fields:
            raw = value.pop(f"{field}_json", None)
            try:
                value[field] = json.loads(raw) if raw else ([] if field.endswith("refs") else {})
            except (TypeError, ValueError, json.JSONDecodeError):
                value[field] = [] if field.endswith("refs") else {}
        return value

    def overview(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "memories": {}, "candidates": {}, "aliases": {}, "outbox": {}}
        conn = self._connect()
        try:
            def grouped(table: str, field: str, where: str = "") -> dict[str, int]:
                rows = conn.execute(
                    f"SELECT {field},COUNT(*) FROM {table} {where} GROUP BY {field}"
                ).fetchall()
                return {str(row[0]): int(row[1]) for row in rows}
            return {
                "available": True,
                "memories": grouped("memories", "state"),
                "candidates": grouped("candidates", "status"),
                "aliases": grouped("entity_aliases", "trust", "WHERE state='active'"),
                "outbox": grouped("outbox", "status"),
                "watermark": self.watermark(),
            }
        except sqlite3.Error:
            return {"available": False, "memories": {}, "candidates": {}, "aliases": {}, "outbox": {}}
        finally:
            conn.close()

    def list_memories(self, *, state: str = "active", limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if not self.available():
            return []
        clauses = []
        params: list[Any] = []
        if state and state != "all":
            clauses.append("m.state=?")
            params.append(str(state))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT m.*,r.body_sha256,r.snapshot_path,r.metadata_json,r.source_refs_json,"
                "r.decision_source,r.created_at AS revision_created_at "
                "FROM memories m LEFT JOIN memory_revisions r "
                "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
                f"{where} ORDER BY m.updated_at DESC,m.memory_id LIMIT ? OFFSET ?",
                [*params, max(1, min(200, int(limit))), max(0, int(offset))],
            ).fetchall()
            return [self._decoded_row(row, "metadata", "source_refs") for row in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def memory_detail(self, memory_id: str) -> dict[str, Any] | None:
        if not self.available():
            return None
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            memory = conn.execute(
                "SELECT * FROM memories WHERE memory_id=?", (str(memory_id),)
            ).fetchone()
            if not memory:
                return None
            revisions = conn.execute(
                "SELECT * FROM memory_revisions WHERE memory_id=? ORDER BY revision DESC LIMIT 200",
                (str(memory_id),),
            ).fetchall()
            rings = conn.execute(
                "SELECT * FROM memory_rings WHERE memory_id=? ORDER BY created_at DESC,ring_id LIMIT 200",
                (str(memory_id),),
            ).fetchall()
            projections = conn.execute(
                "SELECT * FROM memory_projection_status WHERE memory_id=? "
                "ORDER BY memory_revision DESC,projector",
                (str(memory_id),),
            ).fetchall()
            return {
                "memory": dict(memory),
                "revisions": [
                    self._decoded_row(row, "metadata", "source_refs") for row in revisions
                ],
                "rings": [
                    self._decoded_row(row, "metadata", "source_refs") for row in rings
                ],
                "projection_status": [
                    self._decoded_row(row, "details") for row in projections
                ],
            }
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    def list_aliases(self, *, state: str = "active", trust: str = "all", limit: int = 200) -> list[dict[str, Any]]:
        if not self.available():
            return []
        clauses = []
        params: list[Any] = []
        if state and state != "all":
            clauses.append("state=?")
            params.append(str(state))
        if trust and trust != "all":
            clauses.append("trust=?")
            params.append(str(trust))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT * FROM entity_aliases {where} ORDER BY updated_at DESC,alias_id LIMIT ?",
                [*params, max(1, min(1000, int(limit)))],
            ).fetchall()
            output = []
            for row in rows:
                item = self._decoded_row(row, "source_refs")
                memory_ids = [
                    str(ref)[len("memory:"):]
                    for ref in item.get("source_refs", []) or []
                    if str(ref or "").startswith("memory:")
                ]
                item["memory_ids"] = memory_ids
                output.append(item)
            return output
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def memory_revision_map(self, memory_ids: list[str]) -> dict[str, int]:
        ids = list(dict.fromkeys(
            str(memory_id or "").strip()
            for memory_id in memory_ids
            if str(memory_id or "").strip()
        ))[:64]
        if not ids or not self.available():
            return {}
        resolved = self.resolve_memory_refs(ids)
        return {
            memory_id: int(item.get("active_revision") or 0)
            for memory_id, item in resolved.items()
        }

    def memory_index_metadata_map(
        self, bucket_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Resolve eligible bucket projections to their committed revision.

        This is a read-only lookup for derived Recall indexes.  It deliberately
        includes only active, auto-enabled Memory so manual-only/private rows
        cannot become searchable through an index metadata join.
        """
        ids = list(dict.fromkeys(
            str(bucket_id or "").strip()
            for bucket_id in bucket_ids
            if str(bucket_id or "").strip()
        ))
        if not ids or not self.available():
            return {}
        conn = self._connect()
        try:
            # This is the index eligibility pool, not a 64-result owner tool.
            # Batch bind variables without silently losing the tail of Memory.
            conn.execute("BEGIN")
            rows = []
            for offset in range(0, len(ids), 400):
                chunk = ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(conn.execute(
                    "SELECT m.memory_id,COALESCE(NULLIF(m.bucket_id,''),m.memory_id),"
                    "m.active_revision,COALESCE(r.body_sha256,'') "
                    "FROM memories AS m LEFT JOIN memory_revisions AS r "
                    "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
                    "WHERE m.state='active' AND m.recall_policy='enabled' "
                    f"AND (m.bucket_id IN ({placeholders}) "
                    f"OR m.memory_id IN ({placeholders}))",
                    [*chunk, *chunk],
                ).fetchall())
        except sqlite3.Error:
            return {}
        finally:
            conn.close()

        result: dict[str, dict[str, Any]] = {}
        ambiguous: set[str] = set()
        seen_rows: set[tuple] = set()
        for row in rows:
            signature = tuple(row)
            if signature in seen_rows:
                continue
            seen_rows.add(signature)
            bucket_id = str(row[1] or row[0])
            if not bucket_id or bucket_id in ambiguous:
                continue
            if bucket_id in result:
                # A bucket projected from multiple active authorities is not
                # safe to associate with either revision.
                result.pop(bucket_id, None)
                ambiguous.add(bucket_id)
                continue
            result[bucket_id] = {
                "memory_id": str(row[0]),
                "revision": int(row[2] or 0),
                "body_sha256": str(row[3] or ""),
            }
        return result

    def resolve_memory_refs(self, memory_or_bucket_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(
            str(memory_id or "").strip()
            for memory_id in memory_or_bucket_ids
            if str(memory_id or "").strip()
        ))[:64]
        if not ids or not self.available():
            return {}
        placeholders = ",".join("?" for _ in ids)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT memory_id,bucket_id,active_revision FROM memories "
                f"WHERE state='active' AND recall_policy='enabled' "
                f"AND (memory_id IN ({placeholders}) "
                f"OR bucket_id IN ({placeholders}))",
                [*ids, *ids],
            ).fetchall()
        except sqlite3.Error:
            return {}
        finally:
            conn.close()
        return {
            str(row[0]): {
                "memory_id": str(row[0]),
                "bucket_id": str(row[1]),
                "active_revision": int(row[2]),
            }
            for row in rows
        }
