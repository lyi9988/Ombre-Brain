"""Derived Prompt Composer plan mirror for Ombre Gateway.

The owner authority remains Aizizhu.  This store accepts immutable,
authenticated gateway slices and derived background-scope bindings; it does
not expose an independent editing model.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_ROLES = {"system", "developer", "user", "assistant"}
_LANES = {"instruction", "message"}
_MODES = {"live_source", "frozen_snapshot", "owner_override", "off"}
_MAX_BLOCKS = 128
_MAX_TEXT = 100_000
_SCHEMA_VERSION = 1
_ANCHORS = {
    "instructions.start", "instructions.end", "gateway.after_first_system",
    "history.start", "history.depth", "current_user.before",
    "gateway.current_user_prefix", "current_user.prefix",
    "current_user.after", "history", "current_user", "tool_exchange",
    "response_schema",
}
_GATEWAY_SOURCES = frozenset({
    "ombre.core_memory", "ombre.portrait_memory", "ombre.just_now_context",
    "ombre.date_recall", "ombre.context_mode", "ombre.active_reminders",
    "ombre.memory_detail_request", "ombre.memory_reading_policy",
    "ombre.recalled_memory", "ombre.targeted_memory_detail",
    "ombre.diffused_memory", "ombre.recent_context",
    "ombre.date_persona_trace", "ombre.handoff_hint",
    "ombre.persona_state", "ombre.relationship_weather",
    "ombre.favorite_memory", "ombre.dream_context",
    "ombre.persona_post_reply_prompt", "ombre.memory_query_planner_prompt",
    "ombre.semantic_rescue_prompt", "ombre.reflection_prompt",
    "ombre.memory_classify_prompt", "ombre.diary_memory_prompt",
    "ombre.daily_chat_memory_prompt", "ombre.daily_chat_summary_prompt",
    "ombre.daily_activity_summary_prompt", "custom.gateway_block",
})


class PromptPlanMirrorError(RuntimeError):
    code = "prompt_plan_mirror_error"


class PromptPlanMirrorValidationError(PromptPlanMirrorError):
    code = "prompt_plan_invalid"


class PromptPlanMirrorConflict(PromptPlanMirrorError):
    code = "prompt_plan_conflict"


class PromptPlanMirrorNotFound(PromptPlanMirrorError):
    code = "prompt_plan_not_found"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _identifier(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result or not _ID_RE.fullmatch(result):
        raise PromptPlanMirrorValidationError(f"{field} is invalid")
    return result


def _hash_value(value: Any, field: str) -> str:
    result = str(value or "").strip().lower()
    if not _SHA_RE.fullmatch(result):
        raise PromptPlanMirrorValidationError(f"{field} must be sha256")
    return result


def normalize_gateway_slice(value: Any) -> dict:
    if not isinstance(value, dict):
        raise PromptPlanMirrorValidationError(
            "gateway_slice must be an object")
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or len(blocks) > _MAX_BLOCKS:
        raise PromptPlanMirrorValidationError(
            f"gateway_slice.blocks must contain at most {_MAX_BLOCKS} items")
    normalized = []
    seen = set()
    for index, raw in enumerate(blocks):
        if not isinstance(raw, dict):
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}] must be an object")
        block_id = _identifier(raw.get("block_id"), f"blocks[{index}].block_id")
        source_id = _identifier(
            raw.get("source_id"), f"blocks[{index}].source_id")
        if source_id not in _GATEWAY_SOURCES:
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}] is not a Gateway source")
        if block_id in seen:
            raise PromptPlanMirrorValidationError("block_id must be unique")
        seen.add(block_id)
        if str(raw.get("stage")) != "ombre_post_injection":
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}].stage is invalid")
        role = str(raw.get("role") or "system")
        lane = str(raw.get("lane") or "instruction")
        mode = str(raw.get("mode") or "live_source")
        if role not in _ROLES or lane not in _LANES or mode not in _MODES:
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}] role/lane/mode is invalid")
        anchor = str(raw.get("anchor") or "instructions.end")
        if anchor not in _ANCHORS:
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}].anchor is invalid")
        owner_body = str(raw.get("owner_body") or "")
        frozen_body = str(raw.get("frozen_body") or "")
        wrapper_text = str(raw.get("wrapper_text") or "")
        if (len(owner_body) > _MAX_TEXT or len(frozen_body) > _MAX_TEXT
                or len(wrapper_text) > _MAX_TEXT):
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}] body exceeds limit")
        if mode == "owner_override" and not owner_body.strip():
            raise PromptPlanMirrorValidationError(
                f"blocks[{index}] owner override body is empty")
        frozen_revision = raw.get("frozen_source_revision")
        frozen_sha = str(raw.get("frozen_sha256") or "").strip().lower()
        if mode == "frozen_snapshot":
            if not frozen_body.strip() or not str(frozen_revision or "").strip():
                raise PromptPlanMirrorValidationError(
                    f"blocks[{index}] frozen snapshot is incomplete")
            expected = hashlib.sha256(frozen_body.encode("utf-8")).hexdigest()
            if frozen_sha != expected:
                raise PromptPlanMirrorValidationError(
                    f"blocks[{index}] frozen snapshot SHA does not match")
        if source_id == "custom.gateway_block" and mode == "live_source":
            raise PromptPlanMirrorValidationError(
                "custom.gateway_block has no live source authority")
        normalized.append({
            "block_id": block_id,
            "source_id": source_id,
            "scope": _identifier(raw.get("scope"), f"blocks[{index}].scope"),
            "stage": "ombre_post_injection",
            "mode": mode,
            "role": role,
            "lane": lane,
            "anchor": anchor,
            "order": int(raw.get("order") or 0),
            "history_depth": max(0, int(raw.get("history_depth") or 0)),
            "enabled": bool(raw.get("enabled", True)),
            "condition": (
                raw.get("condition")
                if isinstance(raw.get("condition"), dict) else {}
            ),
            "token_budget": raw.get("token_budget"),
            "priority": int(raw.get("priority") or 0),
            "wrapper_mode": str(
                raw.get("wrapper_mode") or "source_default")[:40],
            "wrapper_text": wrapper_text,
            "owner_body": owner_body,
            "frozen_body": frozen_body,
            "frozen_source_revision": frozen_revision,
            "frozen_sha256": frozen_sha,
            "source_options": (
                raw.get("source_options")
                if isinstance(raw.get("source_options"), dict) else {}
            ),
        })
    settings = value.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    inheritance = settings.get("scope_inheritance")
    inheritance = inheritance if isinstance(inheritance, dict) else {}
    normalized_inheritance = {}
    for child, parent in inheritance.items():
        normalized_inheritance[_identifier(child, "scope_inheritance child")] = (
            _identifier(parent, "scope_inheritance parent"))
    normalized.sort(key=lambda item: (
        item["scope"], item["anchor"], item["order"],
        item["priority"], item["block_id"]))
    return {
        "schema": "guyan.gateway-prompt-slice.v1",
        "blocks": normalized,
        "settings": {"scope_inheritance": normalized_inheritance},
    }


class PromptPlanMirrorStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _open(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=5.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        return db

    @contextmanager
    def _connect(self):
        db = self._open()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _create_schema(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS prompt_plan_mirrors (
                    preset_id TEXT NOT NULL,
                    preset_revision INTEGER NOT NULL,
                    plan_sha256 TEXT NOT NULL,
                    slice_sha256 TEXT NOT NULL,
                    slice_json TEXT NOT NULL,
                    source_authority TEXT NOT NULL,
                    received_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(preset_id, preset_revision)
                );
                CREATE TABLE IF NOT EXISTS prompt_binding_mirrors (
                    identity_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL,
                    preset_id TEXT NOT NULL,
                    preset_revision INTEGER NOT NULL,
                    plan_sha256 TEXT NOT NULL,
                    aiz_binding_revision INTEGER NOT NULL,
                    source_authority TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(identity_id, conversation_id, scope)
                );
                CREATE TABLE IF NOT EXISTS prompt_mirror_actions (
                    request_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prompt_mirror_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            db.execute(
                """INSERT INTO prompt_mirror_meta(key,value)
                   VALUES ('schema_version',?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (str(_SCHEMA_VERSION),))

    @staticmethod
    def _action(db: sqlite3.Connection, request_id: str, action: str,
                fingerprint: str) -> dict | None:
        row = db.execute(
            "SELECT * FROM prompt_mirror_actions WHERE request_id=?",
            (request_id,)).fetchone()
        if row is None:
            return None
        if row["action"] != action or row["fingerprint"] != fingerprint:
            raise PromptPlanMirrorConflict(
                "request_id was already used with another mirror action")
        return json.loads(row["result_json"])

    @staticmethod
    def _record_action(db: sqlite3.Connection, request_id: str, action: str,
                       fingerprint: str, result: dict) -> None:
        db.execute(
            """INSERT INTO prompt_mirror_actions
               (request_id,action,fingerprint,result_json,created_at_ms)
               VALUES (?,?,?,?,?)""",
            (request_id, action, fingerprint, _json(result), _now_ms()))

    @staticmethod
    def _plan(row: sqlite3.Row, *, include_slice: bool) -> dict:
        result = {
            "preset_id": row["preset_id"],
            "preset_revision": int(row["preset_revision"]),
            "plan_sha256": row["plan_sha256"],
            "slice_sha256": row["slice_sha256"],
            "source_authority": row["source_authority"],
            "received_at_ms": int(row["received_at_ms"]),
            "status": "verified",
        }
        if include_slice:
            result["gateway_slice"] = json.loads(row["slice_json"])
        return result

    def put_plan(self, preset_id: str, revision: int, *,
                 plan_sha256: str, gateway_slice: dict,
                 source_authority: str = "aizizhu.prompt_composer",
                 request_id: str | None = None) -> dict:
        preset_id = _identifier(preset_id, "preset_id")
        revision = int(revision)
        if revision < 1:
            raise PromptPlanMirrorValidationError(
                "preset_revision must be positive")
        plan_sha256 = _hash_value(plan_sha256, "plan_sha256")
        normalized = normalize_gateway_slice(gateway_slice)
        slice_sha = _sha(normalized)
        source_authority = str(source_authority or "").strip()
        if source_authority != "aizizhu.prompt_composer":
            raise PromptPlanMirrorValidationError(
                "source_authority is not accepted")
        fingerprint = _sha({
            "preset_id": preset_id, "preset_revision": revision,
            "plan_sha256": plan_sha256, "gateway_slice": normalized,
            "source_authority": source_authority,
        })
        request_id = _identifier(
            request_id or f"auto:put-plan:{fingerprint}", "request_id")
        with self._connect() as db:
            replay = self._action(db, request_id, "put_plan", fingerprint)
            if replay is not None:
                return replay
            existing = db.execute(
                """SELECT * FROM prompt_plan_mirrors
                   WHERE preset_id=? AND preset_revision=?""",
                (preset_id, revision)).fetchone()
            if existing is not None:
                if (existing["plan_sha256"] != plan_sha256
                        or existing["slice_sha256"] != slice_sha):
                    raise PromptPlanMirrorConflict(
                        "immutable plan revision already has other content")
                result = self._plan(existing, include_slice=False)
                self._record_action(
                    db, request_id, "put_plan", fingerprint, result)
                return result
            db.execute(
                """INSERT INTO prompt_plan_mirrors
                   (preset_id,preset_revision,plan_sha256,slice_sha256,
                    slice_json,source_authority,received_at_ms)
                   VALUES (?,?,?,?,?,?,?)""",
                (preset_id, revision, plan_sha256, slice_sha,
                 _json(normalized), source_authority, _now_ms()))
            row = db.execute(
                """SELECT * FROM prompt_plan_mirrors
                   WHERE preset_id=? AND preset_revision=?""",
                (preset_id, revision)).fetchone()
            result = self._plan(row, include_slice=False)
            self._record_action(
                db, request_id, "put_plan", fingerprint, result)
            return result

    def get_plan(self, preset_id: str, revision: int, *,
                 include_slice: bool = False) -> dict:
        preset_id = _identifier(preset_id, "preset_id")
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM prompt_plan_mirrors
                   WHERE preset_id=? AND preset_revision=?""",
                (preset_id, int(revision))).fetchone()
            if row is None:
                raise PromptPlanMirrorNotFound("prompt plan was not found")
            return self._plan(row, include_slice=include_slice)

    def put_binding(self, scope: str, *, preset_id: str,
                    preset_revision: int, plan_sha256: str,
                    aiz_binding_revision: int,
                    source_authority: str = "aizizhu.prompt_composer",
                    request_id: str | None = None,
                    identity_id: str = "jiajia-main",
                    conversation_id: str = "") -> dict:
        scope = _identifier(scope, "scope")
        identity_id = _identifier(identity_id, "identity_id")
        conversation_id = str(conversation_id or "").strip()
        if conversation_id:
            conversation_id = _identifier(conversation_id, "conversation_id")
        plan = self.get_plan(preset_id, preset_revision)
        plan_sha256 = _hash_value(plan_sha256, "plan_sha256")
        if plan["plan_sha256"] != plan_sha256:
            raise PromptPlanMirrorConflict("binding plan SHA does not match")
        if source_authority != "aizizhu.prompt_composer":
            raise PromptPlanMirrorValidationError(
                "source_authority is not accepted")
        revision = int(aiz_binding_revision)
        if revision < 1:
            raise PromptPlanMirrorValidationError(
                "aiz_binding_revision must be positive")
        fingerprint = _sha({
            "scope": scope, "preset_id": preset_id,
            "identity_id": identity_id, "conversation_id": conversation_id,
            "preset_revision": int(preset_revision),
            "plan_sha256": plan_sha256,
            "aiz_binding_revision": revision,
            "source_authority": source_authority,
        })
        request_id = _identifier(
            request_id or f"auto:put-binding:{fingerprint}", "request_id")
        with self._connect() as db:
            replay = self._action(db, request_id, "put_binding", fingerprint)
            if replay is not None:
                return replay
            current = db.execute(
                """SELECT * FROM prompt_binding_mirrors
                   WHERE identity_id=? AND conversation_id=? AND scope=?""",
                (identity_id, conversation_id, scope)).fetchone()
            if current is not None:
                current_revision = int(current["aiz_binding_revision"])
                if revision < current_revision:
                    raise PromptPlanMirrorConflict(
                        "stale binding revision")
                if revision == current_revision:
                    same = (
                        current["preset_id"] == preset_id
                        and int(current["preset_revision"]) == int(
                            preset_revision)
                        and current["plan_sha256"] == plan_sha256
                    )
                    if not same:
                        raise PromptPlanMirrorConflict(
                            "binding revision already has other content")
            now = _now_ms()
            db.execute(
                """INSERT INTO prompt_binding_mirrors
                   (identity_id,conversation_id,scope,preset_id,
                    preset_revision,plan_sha256,
                    aiz_binding_revision,source_authority,updated_at_ms)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(identity_id,conversation_id,scope) DO UPDATE SET
                   preset_id=excluded.preset_id,
                   preset_revision=excluded.preset_revision,
                   plan_sha256=excluded.plan_sha256,
                   aiz_binding_revision=excluded.aiz_binding_revision,
                   source_authority=excluded.source_authority,
                   updated_at_ms=excluded.updated_at_ms""",
                (identity_id, conversation_id, scope, preset_id,
                 int(preset_revision), plan_sha256,
                 revision, source_authority, now))
            row = db.execute(
                """SELECT * FROM prompt_binding_mirrors
                   WHERE identity_id=? AND conversation_id=? AND scope=?""",
                (identity_id, conversation_id, scope)).fetchone()
            result = {
                "identity_id": row["identity_id"],
                "conversation_id": row["conversation_id"],
                "scope": row["scope"],
                "preset_id": row["preset_id"],
                "preset_revision": int(row["preset_revision"]),
                "plan_sha256": row["plan_sha256"],
                "aiz_binding_revision": int(row["aiz_binding_revision"]),
                "source_authority": row["source_authority"],
                "updated_at_ms": int(row["updated_at_ms"]),
                "status": "mirrored",
                "source": (
                    "conversation_binding" if conversation_id
                    else "identity_binding"
                ),
                "detail": "",
            }
            self._record_action(
                db, request_id, "put_binding", fingerprint, result)
            return result

    def get_binding(self, scope: str, *, identity_id: str = "jiajia-main",
                    conversation_id: str = "") -> dict:
        scope = _identifier(scope, "scope")
        identity_id = _identifier(identity_id, "identity_id")
        conversation_id = str(conversation_id or "").strip()
        if conversation_id:
            conversation_id = _identifier(conversation_id, "conversation_id")
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM prompt_binding_mirrors
                   WHERE identity_id=? AND conversation_id=? AND scope=?""",
                (identity_id, conversation_id, scope)).fetchone()
            source = "conversation_binding" if row is not None and conversation_id else (
                "identity_binding" if row is not None else "")
            if row is None and conversation_id:
                row = db.execute(
                    """SELECT * FROM prompt_binding_mirrors
                       WHERE identity_id=? AND conversation_id='' AND scope=?""",
                    (identity_id, scope)).fetchone()
                if row is not None:
                    source = "identity_binding"
            if row is None:
                return {
                    "identity_id": identity_id,
                    "conversation_id": conversation_id,
                    "scope": scope, "status": "legacy_default",
                    "source": "legacy_default",
                    "source_authority": "gateway.legacy",
                }
            plan = db.execute(
                """SELECT plan_sha256 FROM prompt_plan_mirrors
                   WHERE preset_id=? AND preset_revision=?""",
                (row["preset_id"], int(row["preset_revision"]))).fetchone()
            status = (
                "mirrored" if plan is not None
                and plan["plan_sha256"] == row["plan_sha256"]
                else "degraded"
            )
            return {
                "identity_id": row["identity_id"],
                "conversation_id": row["conversation_id"],
                "scope": row["scope"],
                "preset_id": row["preset_id"],
                "preset_revision": int(row["preset_revision"]),
                "plan_sha256": row["plan_sha256"],
                "aiz_binding_revision": int(row["aiz_binding_revision"]),
                "source_authority": row["source_authority"],
                "updated_at_ms": int(row["updated_at_ms"]),
                "status": status,
                "source": source,
                "detail": (
                    "" if status == "mirrored"
                    else "bound plan is missing or has a mismatched SHA"
                ),
            }


__all__ = [
    "PromptPlanMirrorConflict", "PromptPlanMirrorError",
    "PromptPlanMirrorNotFound", "PromptPlanMirrorStore",
    "PromptPlanMirrorValidationError", "normalize_gateway_slice",
]
