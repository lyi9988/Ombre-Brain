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
                matches.append(row)
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

    def memory_revision_map(self, memory_ids: list[str]) -> dict[str, int]:
        ids = list(dict.fromkeys(
            str(memory_id or "").strip()
            for memory_id in memory_ids
            if str(memory_id or "").strip()
        ))[:64]
        if not ids or not self.available():
            return {}
        placeholders = ",".join("?" for _ in ids)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT memory_id,active_revision FROM memories "
                f"WHERE state='active' AND memory_id IN ({placeholders})",
                ids,
            ).fetchall()
        except sqlite3.Error:
            return {}
        finally:
            conn.close()
        return {str(row[0]): int(row[1]) for row in rows}
