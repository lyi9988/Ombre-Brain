"""Read-only Gateway view of committed Memory authority facts."""
from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Any

from memory_authority import normalize_alias


class MemoryAuthorityRecallView:
    def __init__(self, config: dict[str, Any]):
        authority_cfg = config.get("memory_authority", {}) if isinstance(
            config.get("memory_authority", {}), dict) else {}
        state_dir = str(config.get("state_dir") or config.get("buckets_dir") or "state")
        self.enabled = bool(authority_cfg.get("enabled", False))
        self.path = Path(
            str(authority_cfg.get("db_path") or os.path.join(state_dir, "memory_authority.sqlite3"))
        ).resolve()
        self._stamp: tuple[int, int, int, int] | None = None
        self._trusted_aliases: list[dict[str, Any]] = []

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
                f"WHERE state='active' AND (memory_id IN ({placeholders}) "
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
