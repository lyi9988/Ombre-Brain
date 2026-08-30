"""Canonical cross-channel continuation without prompt instructions."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class ContinuationBatch:
    events: tuple[dict[str, Any], ...]
    through_seq: int
    catch_up: dict[str, Any] = field(default_factory=dict)


class CanonicalContinuationAdapter:
    """Pull and ingest natural messages for one fixed canonical channel.

    Transport metadata stays in code/storage. Cross-channel context is merged as
    ordinary user/assistant messages; no system instruction is generated here.
    """

    def __init__(
        self, *, enabled: bool, base_url: str, token: str, device_id: str,
        conversation_id: str, channel_id: str, state_db_path: str | Path,
        http_client: httpx.AsyncClient, max_events: int = 40,
    ):
        self.enabled = bool(enabled)
        self.base_url = str(base_url or "").rstrip("/")
        self.token = str(token or "")
        self.device_id = str(device_id or "")
        self.conversation_id = str(conversation_id or "")
        self.channel_id = str(channel_id or "operit")
        self.state_db_path = str(state_db_path)
        self.http_client = http_client
        self.max_events = max(1, min(200, int(max_events)))
        configured = all((self.base_url, self.token, self.device_id, self.conversation_id, self.channel_id))
        if self.enabled and not configured:
            raise ValueError("canonical continuation enabled with incomplete configuration")
        if self.enabled:
            self._init_db()

    def _connect(self):
        Path(self.state_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.state_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS canonical_channel_cursors (
                conversation_id TEXT NOT NULL, channel_id TEXT NOT NULL,
                last_seq INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (conversation_id, channel_id))"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS canonical_outbox (
                source_event_id TEXT PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
                correlation_id TEXT NOT NULL DEFAULT "", attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT "", created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
            )

    def _channel(self, channel_id: str | None = None) -> str:
        """Resolve the effective channel; defaults to the configured one."""
        value = str(channel_id or "").strip()
        return value or self.channel_id

    def own_event_prefix(self, channel_id: str | None = None) -> str:
        return self._channel(channel_id) + ":"

    def cursor_or_none(self, channel_id: str | None = None) -> int | None:
        if not self.enabled:
            return 0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_seq FROM canonical_channel_cursors WHERE conversation_id=? AND channel_id=?",
                (self.conversation_id, self._channel(channel_id)),
            ).fetchone()
        return int(row["last_seq"]) if row else None

    def cursor(self, channel_id: str | None = None) -> int:
        value = self.cursor_or_none(channel_id)
        return int(value or 0)

    def commit_cursor(self, seq: int, channel_id: str | None = None) -> int:
        if not self.enabled:
            return 0
        value = max(0, int(seq))
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO canonical_channel_cursors
                (conversation_id,channel_id,last_seq,updated_at) VALUES(?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_id,channel_id) DO UPDATE SET
                last_seq=excluded.last_seq,updated_at=CURRENT_TIMESTAMP
                WHERE excluded.last_seq>canonical_channel_cursors.last_seq""",
                (self.conversation_id, self._channel(channel_id), value),
            )
        return self.cursor(channel_id)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Guyan-Device-ID": self.device_id,
        }

    async def verify(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        response = await self.http_client.get(
            self.base_url + "/bridge/v1/conversation", headers=self._headers()
        )
        response.raise_for_status()
        body = response.json()
        if body.get("conversation_id") != self.conversation_id:
            raise RuntimeError("canonical bridge conversation mismatch")
        if body.get("generation") is not False:
            raise RuntimeError("canonical bridge must be no-generation")
        return body

    async def pull(self, channel_id: str | None = None) -> ContinuationBatch:
        current_value = self.cursor_or_none(channel_id)
        current = int(current_value or 0)
        if not self.enabled:
            return ContinuationBatch((), current)
        if current_value is None:
            state = await self.verify()
            baseline = max(0, int(state.get("max_seq") or 0))
            self.commit_cursor(baseline, channel_id)
            return ContinuationBatch((), baseline)
        async def fetch(after_seq: int) -> dict[str, Any]:
            response = await self.http_client.get(
                self.base_url + "/bridge/v1/events",
                params={"after_seq": int(after_seq), "limit": self.max_events},
                headers=self._headers(),
            )
            response.raise_for_status()
            body = response.json()
            if body.get("conversation_id") != self.conversation_id:
                raise RuntimeError("canonical event conversation mismatch")
            return body

        body = await fetch(current)
        max_seq = max(current, int(body.get("max_seq") or current))
        catch_up: dict[str, Any] = {}
        # A per-client cursor is intentionally retained for exact ownership,
        # but an offline client must not spend its next request replaying an
        # arbitrarily old page while the other client has already advanced the
        # conversation.  Jump to the latest bounded page and expose the exact
        # skipped range in the request trace; this is a coverage policy, not
        # text deduplication and does not alter canonical storage.
        if max_seq - current > self.max_events:
            tail_after = max(0, max_seq - self.max_events)
            if tail_after > current:
                catch_up = {
                    "mode": "latest_bounded_page",
                    "cursor_before": current,
                    "max_seq_before": max_seq,
                    "requested_after_seq": tail_after,
                    "skipped_before_seq": tail_after,
                    "skipped_through_seq": tail_after,
                }
                body = await fetch(tail_after)
        through_seq = max(current, int(body.get("next_after_seq") or current))
        own_prefix = self.own_event_prefix(channel_id)
        selected = []
        for event in body.get("items") or []:
            if event.get("event_type") != "message":
                continue
            if event.get("role") not in {"user", "assistant"}:
                continue
            # Own-echo guard. Every channel now writes through the same bridge, so
            # source is "ombre-gateway" for all of them; identify our own writes by
            # the channel prefix we stamp onto source_event_id instead.
            if str(event.get("source_event_id") or "").startswith(own_prefix):
                continue
            content = str(event.get("content") or "").strip()
            if not content:
                continue
            selected.append({
                "seq": int(event.get("seq") or 0),
                "event_id": str(event.get("event_id") or event.get("id") or ""),
                "version_id": str(event.get("version_id") or ""),
                "role": event["role"],
                "content": content,
                "source_event_id": str(event.get("source_event_id") or ""),
            })
        if catch_up:
            catch_up["returned_after_seq"] = max(0, through_seq - self.max_events)
            catch_up["through_seq"] = through_seq
            catch_up["returned_event_count"] = len(body.get("items") or [])
        return ContinuationBatch(tuple(selected), through_seq, catch_up)

    @staticmethod
    def merge_messages(messages: list[dict[str, Any]], batch: ContinuationBatch) -> list[dict[str, Any]]:
        if not batch.events:
            return [dict(item) for item in messages]
        result = [dict(item) for item in messages]
        current_user = None
        for index in range(len(result) - 1, -1, -1):
            message = result[index]
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role in {"system", "developer", "tool", "function"}:
                # Tool-loop continuations end with tool protocol items.  They
                # are not a new user turn and must not become the insertion
                # anchor for canonical cross-client events.
                continue
            if role == "assistant":
                if message.get("tool_calls") or not str(message.get("content") or "").strip():
                    continue
                # A completed textual assistant reply means there is no
                # current user tail to augment in this request.
                break
            if role == "user":
                current_user = index
            break
        insertion = [{"role": event["role"], "content": event["content"]} for event in batch.events]
        if current_user is None:
            return result + insertion
        return result[:current_user] + insertion + result[current_user:]

    def queue_outbox(self, *, source_event_id: str, role: str, content: str, correlation_id: str = "") -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO canonical_outbox
                (source_event_id,role,content,correlation_id) VALUES(?,?,?,?)""",
                (source_event_id, role, content, correlation_id),
            )

    def outbox_size(self) -> int:
        if not self.enabled:
            return 0
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM canonical_outbox").fetchone()[0])

    async def flush_outbox(self, limit: int = 20) -> dict[str, int]:
        if not self.enabled:
            return {"delivered": 0, "pending": 0}
        with self._connect() as conn:
            rows = conn.execute(
                # A permanently failing/oversized legacy row must not sit in
                # front of fresh cross-client events forever.  Keep those
                # rows in the derived outbox for inspection, but stop retrying
                # them after a bounded number of attempts.
                "SELECT source_event_id,role,content,correlation_id,attempts "
                # Newest events are prioritised so a historical poison row
                # cannot delay the owner's next cross-client continuation.
                "FROM canonical_outbox WHERE attempts < ? "
                "ORDER BY created_at DESC, source_event_id DESC LIMIT ?",
                (3, max(1, min(100, int(limit)))),
            ).fetchall()
        delivered = 0
        failed = 0
        for row in rows:
            try:
                await self.ingest(
                    source_event_id=row["source_event_id"], role=row["role"],
                    content=row["content"], correlation_id=row["correlation_id"],
                )
            except Exception as exc:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE canonical_outbox SET attempts=attempts+1,last_error=? WHERE source_event_id=?",
                        (type(exc).__name__, row["source_event_id"]),
                    )
                # Skip this row and keep draining. A single undeliverable event
                # (oversized summary, malformed content) must never stall
                # every later event behind it. ``attempts < 3`` above makes a
                # repeatedly failing row self-quarantine on later cycles.
                failed += 1
                continue
            with self._connect() as conn:
                conn.execute("DELETE FROM canonical_outbox WHERE source_event_id=?", (row["source_event_id"],))
            delivered += 1
        return {"delivered": delivered, "pending": self.outbox_size()}

    async def ingest_or_queue(self, *, source_event_id: str, role: str, content: str, correlation_id: str = "") -> dict[str, Any]:
        try:
            return await self.ingest(
                source_event_id=source_event_id, role=role, content=content, correlation_id=correlation_id,
            )
        except Exception as exc:
            self.queue_outbox(
                source_event_id=source_event_id, role=role, content=content, correlation_id=correlation_id,
            )
            return {"created": False, "queued": True, "error_type": type(exc).__name__}

    async def ingest(self, *, source_event_id: str, role: str, content: str, correlation_id: str = "") -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "created": False}
        if role not in {"user", "assistant"}:
            raise ValueError("canonical ingest role must be user or assistant")
        body = {"source_event_id": source_event_id, "role": role, "content": content}
        if correlation_id:
            body["correlation_id"] = correlation_id
        response = await self.http_client.post(
            self.base_url + "/bridge/v1/events", json=body, headers=self._headers()
        )
        # 409 means the bridge already holds this source_event_id. That is the
        # idempotency guarantee doing its job, not a delivery failure -- treat it
        # as success so the outbox can retire the row instead of retrying forever.
        if response.status_code == 409:
            return {"created": False, "duplicate": True}
        response.raise_for_status()
        return response.json()
