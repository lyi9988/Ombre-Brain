"""Qualified source readers and explicitly labelled historical source copies."""
from __future__ import annotations

from datetime import date
import asyncio
import hashlib
import json

from memory_l1_contract import L1ContractError, canonical_json, make_text_anchor, verify_text_anchor

DIARY_NAMESPACE = "snapshot.ombre.diary_mcp"
DIARY_VIEW = "diary_mcp.content.exact-v1"


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS memory_l0_source_copies (
        namespace TEXT NOT NULL, source_id TEXT NOT NULL, revision_json TEXT NOT NULL,
        snapshot_json TEXT NOT NULL, PRIMARY KEY(namespace,source_id,revision_json)
    )""")


class McpDiarySourceAdapter:
    """A configured MCP view, deliberately not an Aiz canonical revision bridge."""
    namespace = DIARY_NAMESPACE

    def __init__(self, provider_identity, read_diary):
        self.provider_id = hashlib.sha256(provider_identity.encode("utf-8")).hexdigest()
        self.read_diary = read_diary

    async def capture(self, day):
        try:
            valid_date = isinstance(day, str) and date.fromisoformat(day).isoformat() == day
        except ValueError:
            valid_date = False
        if not valid_date:
            raise L1ContractError("invalid_diary_date", "date")
        raw = await self.read_diary(day)
        if not isinstance(raw, dict) or not isinstance(raw.get("content"), str) or not raw["content"].strip():
            return {"state": "unavailable", "reason": "diary_not_readable"}
        canonical_json(raw)
        state = "withdrawn" if raw.get("revoked") is True or raw.get("status") in ("revoked", "withdrawn", "deleted") else "available"
        ident = canonical_json({"provider": self.provider_id, "date": day, "diary_id": raw.get("id")})
        identity = {"text": raw["content"], "title": raw.get("title"), "author": raw.get("author"),
                    "speaker": raw.get("speaker"), "upstream_revision": raw.get("revision_id", raw.get("version")),
                    "kind": raw.get("kind")}
        revision = "view_sha256:" + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        return {"state": state, "source_namespace": self.namespace, "source_id": ident,
                "source_revision": revision, "text_view_id": DIARY_VIEW, "text": raw["content"],
                "title": raw.get("title"), "author": raw.get("author"), "speaker": raw.get("speaker"),
                "upstream_revision": identity["upstream_revision"], "kind": raw.get("kind"),
                "recorded_at": raw.get("created_at"), "date": day,
                "limitations": ["MCP返回视图的版本快照；未声明Aiz canonical/private/shared来源桥。",
                                "未提供的作者、发生时间和上游修订保持未知。"]}

    async def read(self, binding):
        try:
            locator = json.loads(binding["source_id"])
        except (TypeError, ValueError):
            return {"state": "unavailable", "reason": "unsupported_source_locator"}
        if not isinstance(locator, dict) or locator.get("provider") != self.provider_id:
            return {"state": "unavailable", "reason": "configured_source_changed"}
        result = await self.capture(locator.get("date"))
        if result.get("state") == "available" and result.get("source_id") != binding["source_id"]:
            return {"state": "changed", "reason": "diary_identity_changed"}
        return result


class MemoryL0Sources:
    def __init__(self, authority, adapters=(), *, read_timeout_seconds=20.0, diary_namespace=DIARY_NAMESPACE):
        self.authority = authority
        self.adapters = {adapter.namespace: adapter for adapter in adapters}
        self.read_timeout_seconds = read_timeout_seconds
        self.diary_namespace = diary_namespace

    def retain_copy(self, snapshot):
        """Called only with a program reader result, not owner/model JSON."""
        conn = self.authority._connect()
        try:
            initialize(conn)
            key = (snapshot["source_namespace"], snapshot["source_id"], canonical_json(snapshot["source_revision"]))
            existing = conn.execute("SELECT snapshot_json FROM memory_l0_source_copies WHERE namespace=? AND source_id=? AND revision_json=?", key).fetchone()
            value = canonical_json(snapshot)
            if existing:
                prior = json.loads(existing[0])
                # Live withdrawal is not an immutable text-view revision.
                if prior.get("text") != snapshot.get("text") or prior.get("text_view_id") != snapshot.get("text_view_id"):
                    raise L1ContractError("source_copy_identity_collision", "source")
            else:
                conn.execute("INSERT INTO memory_l0_source_copies VALUES(?,?,?,?)", (*key, value))
            conn.commit()
        finally:
            conn.close()

    def copy(self, binding):
        conn = self.authority._connect()
        try:
            row = conn.execute("SELECT snapshot_json FROM memory_l0_source_copies WHERE namespace=? AND source_id=? AND revision_json=?",
                               (binding["source_namespace"], binding["source_id"], canonical_json(binding["source_revision"]))).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    async def capture_diary(self, day, *, binding_id):
        adapter = self.adapters.get(self.diary_namespace)
        if adapter is None:
            raise L1ContractError("diary_reader_unconfigured", "source")
        snapshot = await asyncio.wait_for(adapter.capture(day), timeout=self.read_timeout_seconds)
        if snapshot.get("state") != "available":
            return {"source": snapshot, "binding": None}
        self.retain_copy(snapshot)
        anchor = make_text_anchor(snapshot["text"], target_namespace="l0", target_id=snapshot["source_id"],
                                  target_revision=snapshot["source_revision"], text_view_id=snapshot["text_view_id"],
                                  start=0, end=len(snapshot["text"]))
        binding = {"binding_id": binding_id, "source_namespace": snapshot["source_namespace"],
                   "source_id": snapshot["source_id"], "source_revision": snapshot["source_revision"],
                   "author_ref": None, "speaker_ref": None, "anchors": [anchor]}
        return {"source": snapshot, "binding": binding}

    async def locate(self, binding, *, start=None, end=None):
        adapter = self.adapters.get(binding["source_namespace"])
        try:
            current = await asyncio.wait_for(adapter.read(binding), timeout=self.read_timeout_seconds) if adapter else {"state": "unavailable", "reason": "source_adapter_unconfigured"}
        except Exception:
            current = {"state": "unavailable", "reason": "source_read_failed"}
        eligible = (current.get("state") == "available" and current.get("source_namespace") == binding["source_namespace"]
                    and current.get("source_id") == binding["source_id"]
                    and type(current.get("source_revision")) is type(binding["source_revision"])
                    and current.get("source_revision") == binding["source_revision"])
        if eligible:
            try:
                for anchor in binding["anchors"]:
                    verify_text_anchor(current["text"], anchor, target_namespace="l0", target_id=current["source_id"],
                                       target_revision=current["source_revision"], text_view_id=current["text_view_id"])
            except (L1ContractError, KeyError, TypeError):
                eligible = False
                current = {"state": "mismatch", "reason": "source_anchor_mismatch"}
        if eligible:
            self.retain_copy(current)
        snapshot = current if eligible else self.copy(binding)
        result = {"binding": binding, "state": ("changed" if current.get("state") == "available" else current.get("state")) if not eligible else "verified_snapshot",
                  "eligible_source": eligible, "reason": current.get("reason"), "view": "source" if eligible else "source_copy",
                  "text": None, "limitations": []}
        if snapshot:
            result.update({key: snapshot.get(key) for key in ("text", "title", "author", "speaker", "upstream_revision", "recorded_at", "limitations")})
            if not eligible: result["limitations"] = [*(result.get("limitations") or []), "历史原文副本；当前来源未核验通过，不作当前资格依据。"]
            if start is not None:
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(snapshot["text"]):
                    raise L1ContractError("invalid_codepoint_range", "source.range")
                result["text"] = snapshot["text"][start:end]
                result["range"] = {"start": start, "end": end, "coordinate_unit": "unicode_codepoint"}
        return result
