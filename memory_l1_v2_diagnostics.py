"""Bounded operational receipts for natural-memory generation.

Candidate bodies and source versions remain in their original authority stores.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from memory_l1_contract import canonical_json


RETENTION_DAYS = 7
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
_BODY_KEYS = {"body", "text", "content", "messages", "raw_response", "response", "prompt",
              "system", "user", "arguments", "authorization", "api_key", "token", "secret"}


def _metadata(value):
    if isinstance(value, dict):
        return {str(key): _metadata(item) for key, item in value.items()
                if str(key).lower() not in _BODY_KEYS}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value[:200]]
    if isinstance(value, str):
        return value[:512]
    if value is None or type(value) in (int, float, bool):
        return value
    return type(value).__name__


class MemoryV2Diagnostics:
    def __init__(self, authority, *, max_bytes=DEFAULT_MAX_BYTES):
        self.authority = authority
        if type(max_bytes) is not int or max_bytes < 65536:
            raise ValueError("invalid_memory_diagnostics_capacity")
        self.max_bytes = max_bytes

    @staticmethod
    def _initialize(conn):
        conn.execute("""CREATE TABLE IF NOT EXISTS memory_v2_diagnostics (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, stage TEXT NOT NULL,
            created_at TEXT NOT NULL, payload_json TEXT NOT NULL, byte_count INTEGER NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS memory_v2_diag_run ON memory_v2_diagnostics(run_id,id)")

    def record(self, run_id, stage, data):
        if not isinstance(run_id, str) or not run_id or not isinstance(stage, str) or not stage:
            raise ValueError("diagnostic_identity_required")
        now = datetime.now(timezone.utc)
        payload = canonical_json(_metadata(data))
        conn = self.authority._connect()
        try:
            self._initialize(conn)
            conn.execute("DELETE FROM memory_v2_diagnostics WHERE created_at<?",
                         ((now - timedelta(days=RETENTION_DAYS)).isoformat(),))
            conn.execute("INSERT INTO memory_v2_diagnostics(run_id,stage,created_at,payload_json,byte_count) VALUES(?,?,?,?,?)",
                         (run_id, stage, now.isoformat(), payload, len(payload.encode("utf-8"))))
            total = conn.execute("SELECT COALESCE(SUM(byte_count),0) FROM memory_v2_diagnostics").fetchone()[0]
            if total > self.max_bytes:
                excess = total - self.max_bytes
                removed = 0
                through = None
                for row in conn.execute("SELECT id,byte_count FROM memory_v2_diagnostics ORDER BY id"):
                    removed += row[1]
                    through = row[0]
                    if removed >= excess:
                        break
                if through is not None:
                    conn.execute("DELETE FROM memory_v2_diagnostics WHERE id<=?", (through,))
            conn.commit()
        finally:
            conn.close()

    def list(self, run_id=None, *, limit=100):
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("invalid_diagnostic_limit")
        conn = self.authority._connect()
        try:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_v2_diagnostics'").fetchone():
                return []
            cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
            query = "SELECT run_id,stage,created_at,payload_json FROM memory_v2_diagnostics WHERE created_at>=?"
            params = [cutoff]
            if run_id:
                query += " AND run_id=?"
                params.append(run_id)
            rows = conn.execute(query + " ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
            return [{"run_id": row[0], "stage": row[1], "created_at": row[2],
                     "data": json.loads(row[3])} for row in rows]
        finally:
            conn.close()
