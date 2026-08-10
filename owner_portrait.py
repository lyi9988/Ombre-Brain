"""Owner-only whitelisted snapshot of the daily portrait state.

Portrait Viewer v0: this module renders the owner view of
``portrait_engine.load_state()``. It is a pure transform -- it never writes
state, never calls maintain/edit/lock/rollback, and never exposes internal
identifiers (state_path, bucket/session/profile ids, tokens, urls).

Three scopes (user/persona/relationship) use the same field structure.
Missing fields are rendered as empty strings/lists -- never fabricated text.
"""
from __future__ import annotations

from typing import Any

PORTRAIT_SCOPES = ("user", "persona", "relationship")
SCHEMA_VERSION = "owner-portrait-v0"

# Keys that must never appear anywhere in the owner response (recursive).
HIDDEN_KEY_FRAGMENTS = (
    "state_path",
    "bucket_id",
    "session_id",
    "profile_id",
    "moment_id",
    "token",
    "path",
)

# Source values allowed through for stable blocks.
ALLOWED_STABLE_SOURCES = {"model", "manual", "rollback"}


def _str(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bool(value: Any) -> bool:
    return bool(value)


def _float_or_none(value: Any):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _evidence_count(evidence: Any) -> int:
    if not isinstance(evidence, list):
        return 0
    return len([row for row in evidence if isinstance(row, dict)])


def _stable_block(scope: dict) -> dict:
    source = str(scope.get("stable_source") or "").strip()
    return {
        "text": _str(scope.get("stable")),
        "revision": _int(scope.get("stable_revision")),
        "source": source if source in ALLOWED_STABLE_SOURCES else "",
        "locked": _bool(scope.get("stable_locked")),
        "updated_at": _str(scope.get("stable_updated_at")),
        "source_dates": _string_list(scope.get("stable_source_dates")),
        "evidence_count": _evidence_count(scope.get("stable_evidence")),
    }


def _mid_term_block(scope: dict) -> dict:
    return {
        "text": _str(scope.get("mid_term")),
        "updated_at": _str(scope.get("mid_term_updated_at")),
        "source_dates": _string_list(scope.get("mid_term_source_dates")),
        "evidence_count": _evidence_count(scope.get("mid_term_evidence")),
    }


def _buffer_block(rows: Any) -> list[dict]:
    """recent_buffer / staging_pool rows: text plus owner-safe metadata."""
    result = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "text": _str(row.get("text")),
                "updated_at": _str(row.get("updated_at")),
                "source_date": _str(row.get("source_date")),
                "source_dates": _string_list(row.get("source_dates")),
                "confidence": _float_or_none(row.get("confidence")),
                "evidence_count": _evidence_count(row.get("evidence")),
            }
        )
    return result


def _history_block(rows: Any) -> list[dict]:
    """stable_history rows: revision plus owner-safe metadata."""
    result = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "").strip()
        result.append(
            {
                "revision": _int(row.get("revision")),
                "text": _str(row.get("text")),
                "source": source if source in ALLOWED_STABLE_SOURCES else "",
                "updated_at": _str(row.get("updated_at")),
                "source_dates": _string_list(row.get("source_dates")),
                "evidence_count": _evidence_count(row.get("evidence")),
            }
        )
    return result


def _scope_block(scope: Any) -> dict:
    if not isinstance(scope, dict):
        scope = {}
    return {
        "stable": _stable_block(scope),
        "mid_term": _mid_term_block(scope),
        "recent": _buffer_block(scope.get("recent_buffer")),
        "staging": _buffer_block(scope.get("staging_pool")),
        "history": _history_block(scope.get("stable_history")),
    }


def build_owner_portrait_snapshot(
    state: dict,
    *,
    enabled: bool = True,
    auto_enabled: bool = True,
    auto_initial_enabled: bool = False,
    daily_enabled: bool = True,
    self_anchor: dict | None = None,
    pending: dict | None = None,
) -> dict:
    """Whitelisted owner view of a portrait state dict (read-only transform)."""
    portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
    anchor = self_anchor if isinstance(self_anchor, dict) else {}
    pending = pending if isinstance(pending, dict) else {}
    by_layer = pending.get("by_layer", {}) if isinstance(pending.get("by_layer", {}), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": _bool(enabled),
        "auto_enabled": _bool(auto_enabled),
        "auto_initial_enabled": _bool(auto_initial_enabled),
        "daily_enabled": _bool(daily_enabled),
        "updated_at": _str(state.get("updated_at")),
        "last_run_date": _str(state.get("last_run_date")),
        "scopes": {
            scope: _scope_block(portrait.get(scope))
            for scope in PORTRAIT_SCOPES
        },
        "self_anchor": {
            "text": _str(anchor.get("text")),
            "updated_at": _str(anchor.get("updated_at")),
            "configured": _bool(anchor.get("configured")),
        },
        "pending": {
            "total": _int(pending.get("total")),
            "by_layer": {
                str(layer): _int(count) for layer, count in by_layer.items()
            },
        },
    }
