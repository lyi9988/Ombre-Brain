"""Owner-safe logical/physical model request trace storage for Ombre Gateway.

The Gateway is the only place that sees the post-injection provider payload.
Raw capture is therefore a sanitized deep copy taken immediately before the
HTTP request; it is never reconstructed from injection metadata. Metadata and
body rows are separate so disabling body capture leaves the inspector useful.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit


# Match credential-bearing fields without swallowing legitimate model controls
# such as ``max_tokens``, ``prompt_tokens`` and ``token_count``.  Raw View must
# show the actual resolved request parameters while still redacting secrets.
SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|api[_-]?key|(?:access|refresh|auth)[_-]?token|"
    r"cookie|secret|password|credential|^token$)",
    re.I,
)
SIGNED_QUERY_RE = re.compile(r"(?:sig|signature|token|key|expires|credential)", re.I)
BODY_MODES = {"metadata", "resolved", "raw_redacted", "full_owner_body"}
REASONING_VISIBILITY = {"hidden", "metadata_only", "provider_exposed"}
REASONING_BODY_KEYS = {
    "reasoning_content", "reasoning_text", "thinking_content", "reasoning_trace",
}


class _TraceConnection(sqlite3.Connection):
    """Connection context that also closes on Windows file-backed SQLite."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except Exception:
        return value
    if not parts.scheme or not parts.netloc or not parts.query:
        return value
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if not any(SIGNED_QUERY_RE.search(key) for key, _ in pairs):
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "[REDACTED]", ""))


def sanitize_for_owner(value: Any, *, reasoning_visibility: str = "hidden",
                       include_reasoning_body: bool = False) -> Any:
    """Sanitize recursive provider payload/headers without mutating input."""
    visibility = reasoning_visibility if reasoning_visibility in REASONING_VISIBILITY else "hidden"
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            lower = name.lower()
            if SENSITIVE_KEY_RE.search(lower):
                result[name] = "[REDACTED]"
                continue
            # Keep scalar request controls visible.  Only body-shaped or
            # provider-returned reasoning transport is hidden by default;
            # otherwise Raw View could not prove the configured outbound
            # thinking semantics.
            body_like_reasoning = lower in REASONING_BODY_KEYS or (
                lower in {"reasoning", "thinking"}
                and isinstance(item, (dict, list, tuple))
            )
            if body_like_reasoning and (
                visibility != "provider_exposed" or not include_reasoning_body
            ):
                continue
            result[name] = sanitize_for_owner(
                item, reasoning_visibility=visibility,
                include_reasoning_body=include_reasoning_body)
        return result
    if isinstance(value, list):
        return [sanitize_for_owner(item, reasoning_visibility=visibility,
                                   include_reasoning_body=include_reasoning_body)
                for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_owner(item, reasoning_visibility=visibility,
                                   include_reasoning_body=include_reasoning_body)
                for item in value]
    if isinstance(value, str):
        return _redact_url(value)
    return value


def _usage_from_body(body: Any) -> dict | None:
    if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
        return None
    usage = body["usage"]
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = int(value)
    return out or None


class ModelRequestTraceStore:
    """Durable Gateway trace ledger; trace failures never escape to callers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=5, factory=_TraceConnection)
        db.row_factory = sqlite3.Row
        return db

    def _init_db(self):
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS logical_requests (
                trace_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL DEFAULT '',
                turn_id TEXT NOT NULL DEFAULT '', request_id TEXT NOT NULL DEFAULT '',
                logical_request_id TEXT NOT NULL DEFAULT '',
                request_ordinal INTEGER NOT NULL DEFAULT 1,
                parent_request_id TEXT NOT NULL DEFAULT '', tool_round INTEGER,
                request_type TEXT NOT NULL DEFAULT 'initial', outcome TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
                client_id TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
            )""")
            columns = {
                row[1] for row in db.execute(
                    "PRAGMA table_info(logical_requests)").fetchall()
            }
            if "logical_request_id" not in columns:
                db.execute(
                    "ALTER TABLE logical_requests ADD COLUMN logical_request_id TEXT NOT NULL DEFAULT ''"
                )
            db.execute("""CREATE TABLE IF NOT EXISTS physical_attempts (
                attempt_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
                attempt_ordinal INTEGER NOT NULL, provider TEXT NOT NULL DEFAULT '',
                upstream TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
                alias TEXT NOT NULL DEFAULT '', retry_reason TEXT NOT NULL DEFAULT '',
                started_at_ms INTEGER NOT NULL, duration_ms INTEGER,
                http_status INTEGER, result_status TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '', usage_json TEXT NOT NULL DEFAULT '{}',
                raw_id TEXT NOT NULL DEFAULT '', created_at_ms INTEGER NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS raw_requests (
                raw_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL, view_mode TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at_ms INTEGER NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS trace_settings (
                id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL DEFAULT 1,
                capture_mode TEXT NOT NULL DEFAULT 'raw_redacted',
                body_visibility TEXT NOT NULL DEFAULT 'metadata_only',
                retention_days INTEGER NOT NULL DEFAULT 7,
                request_limit INTEGER NOT NULL DEFAULT 2000,
                disk_budget_mb INTEGER NOT NULL DEFAULT 256,
                capture_next INTEGER NOT NULL DEFAULT 0,
                revision_no INTEGER NOT NULL DEFAULT 1, updated_at_ms INTEGER NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL,
                event_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS trace_recent ON logical_requests(created_at_ms DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS trace_attempts ON physical_attempts(trace_id, attempt_ordinal)")
            db.execute("CREATE INDEX IF NOT EXISTS trace_events_recent ON trace_events(id DESC)")
            if db.execute("SELECT 1 FROM trace_settings WHERE id=1").fetchone() is None:
                db.execute("INSERT INTO trace_settings(id, updated_at_ms) VALUES(1, ?)", (_now_ms(),))

    def settings(self) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM trace_settings WHERE id=1").fetchone()
        return {"enabled": bool(row["enabled"]), "capture_mode": row["capture_mode"],
                "body_visibility": row["body_visibility"],
                "retention_days": int(row["retention_days"]),
                "request_limit": int(row["request_limit"]),
                "disk_budget_mb": int(row["disk_budget_mb"]),
                "capture_next": bool(row["capture_next"]),
                "revision_no": int(row["revision_no"]),
                "updated_at_ms": int(row["updated_at_ms"])}

    def update_settings(self, values: dict[str, Any], *, expected_revision: int | None = None) -> dict:
        current = self.settings()
        if expected_revision is not None and int(expected_revision) != current["revision_no"]:
            raise ValueError("trace settings revision conflict")
        merged = dict(current)
        merged.update({key: value for key, value in values.items() if value is not None})
        mode = str(merged["capture_mode"] or "raw_redacted")
        visibility = str(merged["body_visibility"] or "metadata_only")
        if mode not in BODY_MODES or visibility not in REASONING_VISIBILITY:
            raise ValueError("invalid trace capture/body visibility")
        # ``0`` is an owner-configurable no-time-expiry value.  It must not
        # make an in-flight request disappear between begin_logical() and its
        # first physical attempt; request_limit and disk_budget still bound
        # the retained set.
        merged["retention_days"] = max(0, min(3650, int(merged["retention_days"])))
        merged["request_limit"] = max(1, min(100000, int(merged["request_limit"])))
        merged["disk_budget_mb"] = max(1, min(102400, int(merged["disk_budget_mb"])))
        merged["revision_no"] = current["revision_no"] + 1
        with self._connect() as db:
            db.execute("""UPDATE trace_settings SET enabled=?, capture_mode=?,
                body_visibility=?, retention_days=?, request_limit=?, disk_budget_mb=?,
                capture_next=?, revision_no=?, updated_at_ms=? WHERE id=1""",
                (int(bool(merged["enabled"])), mode, visibility,
                 merged["retention_days"], merged["request_limit"],
                 merged["disk_budget_mb"], int(bool(merged["capture_next"])),
                 merged["revision_no"], _now_ms()))
        return self.settings()

    def begin_logical(self, metadata: dict[str, Any]) -> str:
        settings = self.settings()
        if not settings["enabled"] and not settings["capture_next"]:
            return ""
        trace_id = str(metadata.get("trace_id") or uuid.uuid4().hex).strip()[:120]
        now = _now_ms()
        safe = {key: metadata.get(key) for key in (
            "conversation_id", "turn_id", "request_id", "logical_request_id", "request_ordinal",
            "parent_request_id", "tool_round", "request_type", "provider",
            "model", "client_id")}
        # logical metadata may contain provenance/debug state, never bodies.
        safe["metadata"] = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        with self._connect() as db:
            db.execute("""INSERT INTO logical_requests(
                trace_id, conversation_id, turn_id, request_id, logical_request_id, request_ordinal,
                parent_request_id, tool_round, request_type, outcome, provider,
                model, client_id, metadata_json, created_at_ms, updated_at_ms)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(trace_id) DO UPDATE SET
                logical_request_id=excluded.logical_request_id,
                metadata_json=excluded.metadata_json, outcome=excluded.outcome,
                updated_at_ms=excluded.updated_at_ms""", (
                trace_id, str(safe.get("conversation_id") or "")[:200],
                str(safe.get("turn_id") or "")[:200], str(safe.get("request_id") or "")[:200],
                str(safe.get("logical_request_id") or "")[:240],
                int(safe.get("request_ordinal") or 1), str(safe.get("parent_request_id") or "")[:200],
                safe.get("tool_round"), str(safe.get("request_type") or "initial")[:40],
                str(metadata.get("outcome") or "")[:40], str(safe.get("provider") or "")[:120],
                str(safe.get("model") or "")[:160], str(safe.get("client_id") or "")[:80],
                _json(safe["metadata"]), now, now))
            self._event(db, trace_id, "request.started", safe)
        return trace_id

    def _event(self, db, trace_id: str, event_type: str, payload: Any):
        db.execute("INSERT INTO trace_events(trace_id,event_type,payload_json,created_at_ms) VALUES(?,?,?,?)",
                   (trace_id, event_type, _json(payload), _now_ms()))

    def record_attempt(self, *, trace_id: str, ordinal: int, provider: str,
                       upstream: str, model: str, alias: str = "",
                       retry_reason: str = "", started_at_ms: int | None = None,
                       duration_ms: int | None = None, http_status: int | None = None,
                       result_status: str = "", outcome: str = "",
                       usage: dict | None = None, payload: dict | None = None) -> str | None:
        try:
            settings = self.settings()
            attempt_id = f"attempt-{uuid.uuid4().hex}"
            trace_id = str(trace_id or "")[:120]
            with self._connect() as db:
                logical_exists = db.execute(
                    "SELECT 1 FROM logical_requests WHERE trace_id=?", (trace_id,)
                ).fetchone() is not None
            # Once a logical request has started, finish observing all of its
            # physical retries even if capture_next was consumed by attempt 1
            # or the owner toggled tracing while the request was in flight.
            if not settings["enabled"] and not settings["capture_next"] and not logical_exists:
                return None
            if not trace_id:
                return None
            with self._connect() as db:
                previous = db.execute(
                    "SELECT MAX(attempt_ordinal) AS value FROM physical_attempts WHERE trace_id=?",
                    (trace_id,),
                ).fetchone()
            # A Gateway retry helper may call a second forwarding function with
            # the same request ordinal. Keep one logical request and make the
            # physical order unambiguous instead of overwriting/duplicating it.
            ordinal = max(1, int(ordinal))
            if previous and previous["value"] is not None and ordinal <= int(previous["value"]):
                ordinal = int(previous["value"]) + 1
            raw_id = ""
            mode = settings["capture_mode"]
            if payload is not None and mode in {"raw_redacted", "full_owner_body"}:
                raw_id = f"raw-{uuid.uuid4().hex}"
                visibility = settings["body_visibility"]
                sanitized = sanitize_for_owner(
                    deepcopy(payload), reasoning_visibility=visibility,
                    include_reasoning_body=(visibility == "provider_exposed" and mode == "full_owner_body"))
                with self._connect() as db:
                    db.execute("INSERT INTO raw_requests VALUES(?,?,?,?,?,?)",
                               (raw_id, trace_id, attempt_id, mode, _json(sanitized), _now_ms()))
            with self._connect() as db:
                db.execute("""INSERT INTO physical_attempts(
                    attempt_id, trace_id, attempt_ordinal, provider, upstream, model,
                    alias, retry_reason, started_at_ms, duration_ms, http_status,
                    result_status, outcome, usage_json, raw_id, created_at_ms)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    attempt_id, trace_id, int(ordinal), str(provider or "")[:120],
                    str(upstream or "")[:160], str(model or "")[:160], str(alias or "")[:160],
                    str(retry_reason or "")[:200], int(started_at_ms or _now_ms()),
                    duration_ms, http_status, str(result_status or "")[:80],
                    str(outcome or "")[:40], _json(usage or {}), raw_id, _now_ms()))
                self._event(db, trace_id, "payload.ready", {
                    "attempt_id": attempt_id, "ordinal": ordinal,
                    "raw_available": bool(raw_id),
                })
                self._event(db, trace_id, "attempt", {"attempt_id": attempt_id, "ordinal": ordinal,
                                                       "status": http_status, "outcome": outcome})
            if settings["capture_next"]:
                self.update_settings({"capture_next": False}, expected_revision=settings["revision_no"])
            self.purge()
            return attempt_id
        except Exception:
            return None

    def update_attempt(self, attempt_id: str, *, duration_ms: int | None = None,
                       http_status: int | None = None, result_status: str | None = None,
                       outcome: str | None = None, usage: dict | None = None):
        try:
            fields, values = [], []
            for name, value in (("duration_ms", duration_ms), ("http_status", http_status),
                                ("result_status", result_status), ("outcome", outcome)):
                if value is not None:
                    fields.append(f"{name}=?"); values.append(value)
            if usage is not None:
                fields.append("usage_json=?"); values.append(_json(usage))
            if not fields:
                return
            values.append(attempt_id)
            with self._connect() as db:
                db.execute(f"UPDATE physical_attempts SET {', '.join(fields)} WHERE attempt_id=?", values)
                row = db.execute("SELECT trace_id FROM physical_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
                if row:
                    if usage is not None:
                        self._event(db, row["trace_id"], "usage", {
                            "attempt_id": attempt_id, "usage": usage,
                        })
                    # A physical attempt is not the logical request. The
                    # logical terminal event is emitted exactly once by
                    # set_outcome(); emitting ``completed`` here made every
                    # attempt update look like another model completion.
        except Exception:
            return

    def set_outcome(self, trace_id: str, outcome: str) -> None:
        try:
            with self._connect() as db:
                result = db.execute(
                    "UPDATE logical_requests SET outcome=?, updated_at_ms=? WHERE trace_id=?",
                    (str(outcome or "")[:40], _now_ms(), trace_id),
                )
                # A retention/budget purge may have removed an old request.
                # Never recreate an event-only phantom trace in that case.
                if result.rowcount:
                    terminal_payload = {"outcome": str(outcome or "")[:40]}
                    existing = db.execute(
                        "SELECT id FROM trace_events WHERE trace_id=? "
                        "AND event_type='completed' ORDER BY id DESC LIMIT 1",
                        (trace_id,),
                    ).fetchone()
                    if existing:
                        # A late cleanup/error path may call set_outcome after
                        # the normal response path. Keep one terminal event
                        # identity and refresh its payload instead of
                        # appending a second completion to the terminal.
                        db.execute(
                            "UPDATE trace_events SET payload_json=?, created_at_ms=? WHERE id=?",
                            (_json(terminal_payload), _now_ms(), existing["id"]),
                        )
                    else:
                        self._event(db, trace_id, "completed", terminal_payload)
        except Exception:
            return

    def record_event(self, trace_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append owner-safe activity telemetry without affecting the request."""
        try:
            with self._connect() as db:
                key = str(trace_id or "")[:120]
                if not db.execute("SELECT 1 FROM logical_requests WHERE trace_id=?", (key,)).fetchone():
                    return
                self._event(db, key, str(event_type or "event")[:80], payload or {})
        except Exception:
            return

    def update_metadata(self, trace_id: str, patch: dict[str, Any]) -> None:
        """Merge body-free resolved metadata after a provider response arrives.

        The logical row remains the authority for request identity; this is an
        additive observation update used for returned/replay-required status and
        exact provider outcome.  It never stores provider body text.
        """
        if not isinstance(patch, dict):
            return
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT metadata_json FROM logical_requests WHERE trace_id=?",
                    (str(trace_id or "")[:120],),
                ).fetchone()
                if not row:
                    return
                metadata = json.loads(row["metadata_json"] or "{}")
                if not isinstance(metadata, dict):
                    metadata = {}
                for key, value in patch.items():
                    if isinstance(value, dict) and isinstance(metadata.get(key), dict):
                        merged = dict(metadata[key])
                        merged.update(value)
                        metadata[key] = merged
                    else:
                        metadata[key] = value
                db.execute(
                    "UPDATE logical_requests SET metadata_json=?, updated_at_ms=? WHERE trace_id=?",
                    (_json(metadata), _now_ms(), str(trace_id or "")[:120]),
                )
                self._event(db, str(trace_id or "")[:120], "metadata.updated", patch)
        except Exception:
            return

    def record_tool_calls(self, trace_id: str, tool_calls: list[dict] | None) -> None:
        """Expose tool activity under a model request without making it a request."""
        for index, call in enumerate(tool_calls or []):
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            self.record_event(str(trace_id or ""), "tool.called", {
                "index": index,
                "tool_call_id": str(call.get("id") or "")[:160],
                "tool_name": str(function.get("name") or call.get("name") or "")[:160],
            })

    def update_latest_attempt(self, trace_id: str, *, duration_ms: int | None = None,
                              http_status: int | None = None, result_status: str | None = None,
                              outcome: str | None = None, usage: dict | None = None) -> None:
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT attempt_id FROM physical_attempts WHERE trace_id=? "
                    "ORDER BY attempt_ordinal DESC LIMIT 1", (trace_id,)
                ).fetchone()
            if row:
                self.update_attempt(row["attempt_id"], duration_ms=duration_ms,
                                    http_status=http_status, result_status=result_status,
                                    outcome=outcome, usage=usage)
        except Exception:
            return

    def _public(self, row, *, include_raw: bool = False, reasoning_visibility: str = "hidden") -> dict:
        item = {"attempt_id": row["attempt_id"], "trace_id": row["trace_id"],
                "attempt_ordinal": int(row["attempt_ordinal"]), "provider": row["provider"],
                "upstream": row["upstream"], "model": row["model"], "alias": row["alias"],
                "retry_reason": row["retry_reason"], "started_at_ms": int(row["started_at_ms"]),
                "duration_ms": row["duration_ms"], "http_status": row["http_status"],
                "result_status": row["result_status"], "outcome": row["outcome"],
                "usage": json.loads(row["usage_json"] or "{}"), "raw_available": bool(row["raw_id"])}
        if include_raw and row["raw_id"]:
            with self._connect() as db:
                raw = db.execute(
                    "SELECT payload_json, view_mode FROM raw_requests WHERE raw_id=?",
                    (row["raw_id"],)).fetchone()
            if raw:
                item["raw_view_mode"] = raw["view_mode"]
                stored = json.loads(raw["payload_json"])
                # A body captured with the owner-visible mode must still obey
                # the current visibility setting when it is read later.  This
                # prevents switching the Inspector back to hidden/metadata
                # from exposing a previously captured provider reasoning body.
                item["payload"] = sanitize_for_owner(
                    stored,
                    reasoning_visibility=reasoning_visibility,
                    include_reasoning_body=(
                        raw["view_mode"] == "full_owner_body"
                        and reasoning_visibility == "provider_exposed"
                    ),
                )
        return item

    def get(self, trace_id: str, *, view: str = "metadata") -> dict | None:
        settings = self.settings()
        view = view if view in BODY_MODES else "metadata"
        # A later settings change must not hide a body that was already
        # captured.  capture_mode controls future writes; raw_id controls
        # availability of this historical attempt.
        include_raw = view in {"raw_redacted", "full_owner_body"}
        with self._connect() as db:
            logical = db.execute("SELECT * FROM logical_requests WHERE trace_id=?", (trace_id,)).fetchone()
            if logical is None:
                return None
            attempts = db.execute("SELECT * FROM physical_attempts WHERE trace_id=? ORDER BY attempt_ordinal", (trace_id,)).fetchall()
            events = db.execute("SELECT event_type,payload_json,created_at_ms FROM trace_events WHERE trace_id=? ORDER BY id", (trace_id,)).fetchall()
        metadata = json.loads(logical["metadata_json"] or "{}")
        event_items = [{"type": row["event_type"], "payload": json.loads(row["payload_json"] or "{}"),
                        "created_at_ms": row["created_at_ms"]} for row in events]
        logical_request_id = str(logical["logical_request_id"] or "")
        if not logical_request_id:
            logical_request_id = next(
                (str(item["payload"].get("logical_request_id") or "")
                 for item in event_items if isinstance(item.get("payload"), dict)
                 and item["payload"].get("logical_request_id")),
                f"{logical['request_id']}:model:{int(logical['request_ordinal'] or 1)}",
            )
        result = {"trace_id": trace_id, "conversation_id": logical["conversation_id"],
                  "turn_id": logical["turn_id"], "request_id": logical["request_id"],
                  "logical_request_id": logical_request_id,
                  "request_ordinal": logical["request_ordinal"],
                  "parent_request_id": logical["parent_request_id"], "tool_round": logical["tool_round"],
                  "request_type": logical["request_type"], "outcome": logical["outcome"],
                  "provider": logical["provider"], "model": logical["model"],
                  "client_id": logical["client_id"], "metadata": metadata,
                  "attempts": [self._public(row, include_raw=include_raw,
                                             reasoning_visibility=settings["body_visibility"])
                               for row in attempts],
                  "events": event_items,
                  "view": view, "settings_revision": settings["revision_no"]}
        if view == "resolved":
            result["resolved"] = metadata.get("resolved") or metadata
        return result

    def list_recent(self, *, limit: int = 100, conversation_id: str = "", turn_id: str = "") -> list[dict]:
        limit = max(1, min(1000, int(limit)))
        clauses, args = [], []
        if conversation_id:
            clauses.append("conversation_id=?"); args.append(conversation_id)
        if turn_id:
            clauses.append("turn_id=?"); args.append(turn_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        with self._connect() as db:
            rows = db.execute("SELECT * FROM logical_requests" + where + " ORDER BY created_at_ms DESC LIMIT ?", args).fetchall()
        return [{"trace_id": row["trace_id"], "conversation_id": row["conversation_id"],
                 "turn_id": row["turn_id"], "request_id": row["request_id"],
                 "logical_request_id": row["logical_request_id"],
                 "request_ordinal": row["request_ordinal"], "request_type": row["request_type"],
                 "outcome": row["outcome"], "model": row["model"],
                 "provider": row["provider"], "created_at_ms": row["created_at_ms"]} for row in rows]

    def events_recent(self, *, limit: int = 200) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT trace_id,event_type,payload_json,created_at_ms
                FROM trace_events
                WHERE EXISTS (SELECT 1 FROM logical_requests l WHERE l.trace_id=trace_events.trace_id)
                ORDER BY id DESC LIMIT ?""",
                              (max(1, min(2000, int(limit))),)).fetchall()
        return [{"trace_id": row["trace_id"], "type": row["event_type"],
                 "payload": json.loads(row["payload_json"] or "{}"),
                 "created_at_ms": row["created_at_ms"]} for row in rows]

    def clear(self):
        with self._connect() as db:
            db.execute("DELETE FROM raw_requests")
            db.execute("DELETE FROM physical_attempts")
            db.execute("DELETE FROM logical_requests")
            db.execute("DELETE FROM trace_events")

    @staticmethod
    def _delete_trace_ids(db: sqlite3.Connection, trace_ids: list[str]) -> None:
        if not trace_ids:
            return
        marks = ",".join("?" for _ in trace_ids)
        db.execute(f"DELETE FROM raw_requests WHERE trace_id IN ({marks})", trace_ids)
        db.execute(f"DELETE FROM physical_attempts WHERE trace_id IN ({marks})", trace_ids)
        db.execute(f"DELETE FROM trace_events WHERE trace_id IN ({marks})", trace_ids)
        db.execute(f"DELETE FROM logical_requests WHERE trace_id IN ({marks})", trace_ids)

    @staticmethod
    def _stored_bytes(db: sqlite3.Connection) -> int:
        """Conservative persisted JSON byte estimate for budget eviction.

        SQLite pages are not immediately returned after DELETE, so the file
        size is not a useful per-trace budget signal.  Counting the durable
        JSON columns gives deterministic oldest-trace eviction and includes
        raw bodies, metadata, attempts and event payloads.
        """
        total = 0
        for table, column in (
            ("raw_requests", "payload_json"),
            ("logical_requests", "metadata_json"),
            ("physical_attempts", "usage_json"),
            ("trace_events", "payload_json"),
        ):
            value = db.execute(
                f"SELECT COALESCE(SUM(length({column})), 0) FROM {table}"
            ).fetchone()[0]
            total += int(value or 0)
        return total

    def purge(self):
        try:
            settings = self.settings()
            with self._connect() as db:
                retention_days = int(settings["retention_days"])
                if retention_days > 0:
                    cutoff = _now_ms() - retention_days * 86400000
                    db.execute("DELETE FROM raw_requests WHERE created_at_ms < ?", (cutoff,))
                    db.execute("DELETE FROM physical_attempts WHERE created_at_ms < ?", (cutoff,))
                    db.execute("DELETE FROM logical_requests WHERE created_at_ms < ?", (cutoff,))
                    db.execute("DELETE FROM trace_events WHERE created_at_ms < ?", (cutoff,))
                # Clean historical orphan rows left by older versions.  Trace
                # events are derived data and must never outlive their logical
                # request authority.
                db.execute("DELETE FROM trace_events WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests)")
                db.execute("DELETE FROM physical_attempts WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests)")
                db.execute("DELETE FROM raw_requests WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests)")
                db.execute("DELETE FROM logical_requests WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests ORDER BY created_at_ms DESC LIMIT ?)",
                           (settings["request_limit"],))
                db.execute("DELETE FROM physical_attempts WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests)")
                db.execute("DELETE FROM raw_requests WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests)")
                db.execute("DELETE FROM trace_events WHERE trace_id NOT IN (SELECT trace_id FROM logical_requests)")
                budget = max(1, int(settings["disk_budget_mb"])) * 1024 * 1024
                while self._stored_bytes(db) > budget:
                    row = db.execute(
                        "SELECT trace_id FROM logical_requests ORDER BY created_at_ms ASC LIMIT 1"
                    ).fetchone()
                    if row is None:
                        break
                    # Keep the newest logical request observable even when a
                    # single raw body exceeds the owner's budget.  The budget
                    # remains an eviction target for older traces, but it must
                    # not manufacture an empty Inspector for the live turn.
                    count = db.execute("SELECT COUNT(*) FROM logical_requests").fetchone()[0]
                    if int(count or 0) <= 1:
                        break
                    self._delete_trace_ids(db, [str(row["trace_id"])])
        except Exception:
            return


__all__ = ["ModelRequestTraceStore", "sanitize_for_owner"]
