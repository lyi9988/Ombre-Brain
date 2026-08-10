"""Tests for the owner-only portrait snapshot (Portrait Viewer v0).

All fixtures are synthetic -- no production text, ids or config values.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from owner_portrait import build_owner_portrait_snapshot


def _scope_fixture(*, empty: bool = False) -> dict:
    if empty:
        return {
            "recent_buffer": [], "staging_pool": [], "mid_term": "",
            "mid_term_evidence": [], "mid_term_source_dates": [],
            "mid_term_source_date": "", "mid_term_updated_at": "",
            "stable": "", "stable_evidence": [], "stable_source_dates": [],
            "stable_source_date": "", "stable_updated_at": "",
            "stable_locked": False, "stable_revision": 0,
            "stable_source": "", "stable_history": [],
        }
    return {
        "recent_buffer": [
            {
                "text": "最近观察：润润这周专注梳理证据边界",
                "evidence": [{"bucket_id": "b-1", "role": "source"}],
                "source_dates": ["2026-08-09", "2026-08-10"],
                "source_date": "2026-08-10",
                "confidence": 0.72,
                "first_seen_date": "2026-08-09",
                "last_seen_date": "2026-08-10",
                "created_at": "2026-08-10T04:00:00Z",
                "updated_at": "2026-08-10T04:00:00Z",
                "count": 2,
            }
        ],
        "staging_pool": [
            {
                "text": "staging 候选：长期偏好保守证据口径",
                "evidence": [{"bucket_id": "b-2"}],
                "source_dates": ["2026-08-08"],
                "source_date": "2026-08-08",
                "confidence": 0.6,
                "updated_at": "2026-08-10T04:00:00Z",
            }
        ],
        "mid_term": "中期画像：最近在推进换窗连续性核对",
        "mid_term_evidence": [{"bucket_id": "b-3"}],
        "mid_term_source_dates": ["2026-08-08", "2026-08-09"],
        "mid_term_source_date": "2026-08-09",
        "mid_term_updated_at": "2026-08-10T04:00:00Z",
        "stable": "长期画像：顾衍重视稳定记忆边界与第一人称锚点。",
        "stable_evidence": [{"bucket_id": "b-4"}, {"bucket_id": "b-5"}],
        "stable_source_dates": ["2026-08-01", "2026-08-05"],
        "stable_source_date": "2026-08-05",
        "stable_updated_at": "2026-08-10T04:00:00Z",
        "stable_locked": False,
        "stable_revision": 2,
        "stable_source": "model",
        "stable_history": [
            {
                "revision": 1,
                "text": "历史版本：重视记忆边界。",
                "evidence": [{"bucket_id": "b-old"}],
                "source_dates": ["2026-07-28"],
                "updated_at": "2026-07-28T04:00:00Z",
                "source": "model",
            }
        ],
    }


def _state_fixture() -> dict:
    return {
        "version": "portrait-state-v1",
        "updated_at": "2026-08-10T04:00:00Z",
        "last_run_date": "2026-08-10",
        "portrait": {
            "user": _scope_fixture(),
            "persona": _scope_fixture(),
            "relationship": _scope_fixture(),
        },
        "recent_activities": [],
        "recent_timeline": [],
        "stable_candidates": [],
        "profile_fact_candidates": [],
        "skipped": [],
        "runs": [],
    }


def _state_path() -> str:
    return "/srv/ombre-brain/state/portrait_state.json"


def _anchor_fixture():
    return {
        "text": "我是顾衍，新窗口仍记得我是谁。",
        "updated_at": "2026-08-01T00:00:00Z",
        "configured": True,
    }


def _pending_fixture():
    return {"total": 1, "by_layer": {"identity_core": 1}}


def _build(scope_state=None, *, anchor=None, pending=None):
    state = _state_fixture()
    if scope_state is not None:
        for scope in ("user", "persona", "relationship"):
            state["portrait"][scope] = scope_state
    return build_owner_portrait_snapshot(
        state,
        enabled=True,
        auto_enabled=True,
        auto_initial_enabled=False,
        daily_enabled=True,
        self_anchor=anchor if anchor is not None else _anchor_fixture(),
        pending=pending if pending is not None else _pending_fixture(),
    )


def test_three_scopes_return_real_text_and_metadata():
    result = _build()
    for scope in ("user", "persona", "relationship"):
        block = result["scopes"][scope]
        assert "长期画像" in block["stable"]["text"]
        assert block["stable"]["revision"] == 2
        assert block["stable"]["source"] == "model"
        assert block["stable"]["locked"] is False
        assert block["stable"]["updated_at"] == "2026-08-10T04:00:00Z"
        assert block["stable"]["source_dates"] == ["2026-08-01", "2026-08-05"]
        assert block["stable"]["evidence_count"] == 2
        assert "中期画像" in block["mid_term"]["text"]
        assert block["mid_term"]["evidence_count"] == 1
        assert block["mid_term"]["source_dates"] == ["2026-08-08", "2026-08-09"]
        assert block["recent"][0]["text"].startswith("最近观察")
        assert block["recent"][0]["confidence"] == 0.72
        assert block["recent"][0]["evidence_count"] == 1
        assert block["staging"][0]["text"].startswith("staging 候选")
        assert block["history"][0]["revision"] == 1
        assert "历史版本" in block["history"][0]["text"]


def test_empty_fields_render_as_empty_not_fabricated():
    result = _build(_scope_fixture(empty=True))
    for scope in ("user", "persona", "relationship"):
        block = result["scopes"][scope]
        assert block["stable"]["text"] == ""
        assert block["stable"]["revision"] == 0
        assert block["stable"]["source"] == ""
        assert block["mid_term"]["text"] == ""
        assert block["recent"] == []
        assert block["staging"] == []
        assert block["history"] == []
    assert result["self_anchor"]["text"] == "我是顾衍，新窗口仍记得我是谁。"


def test_stable_history_keeps_revision_and_text():
    result = _build()
    history = result["scopes"]["user"]["history"]
    assert history[0]["revision"] == 1
    assert "历史版本" in history[0]["text"]
    assert history[0]["source"] == "model"
    assert history[0]["source_dates"] == ["2026-07-28"]
    assert history[0]["evidence_count"] == 1


def test_self_anchor_strips_internal_ids():
    result = _build()
    anchor = result["self_anchor"]
    assert anchor["text"] == "我是顾衍，新窗口仍记得我是谁。"
    assert anchor["configured"] is True
    serialized = json.dumps(result, ensure_ascii=False)
    assert "bucket_id" not in serialized
    assert "self-anchor-id" not in serialized


def test_pending_only_returns_counts():
    result = _build()
    assert result["pending"] == {"total": 1, "by_layer": {"identity_core": 1}}
    serialized = json.dumps(result, ensure_ascii=False)
    # no proposal body / no narrative content
    assert "proposal正文" not in serialized


def test_response_has_no_internal_keys_or_paths():
    result = _build()
    serialized = json.dumps(result, ensure_ascii=False)
    for fragment in ("state_path", "bucket_id", "session_id", "profile_id",
                     "moment_id", "token", _state_path(), "b-1", "b-4"):
        assert fragment not in serialized
    assert "state_dir" not in serialized


def test_snapshot_flags_and_top_level_fields():
    result = _build()
    assert result["schema_version"] == "owner-portrait-v0"
    assert result["enabled"] is True
    assert result["auto_enabled"] is True
    assert result["auto_initial_enabled"] is False
    assert result["daily_enabled"] is True
    assert result["updated_at"] == "2026-08-10T04:00:00Z"
    assert result["last_run_date"] == "2026-08-10"


def test_tool_read_does_not_modify_state_file(tmp_path):
    """The snapshot pipeline (load_state + transform) never writes state."""
    from portrait_engine import DailyPortraitMaintainer
    state_path = tmp_path / "portrait_state.json"
    maintainer = DailyPortraitMaintainer(
        {
            "state_dir": str(tmp_path),
            "portrait": {
                "state_path": str(state_path),
                "enabled": False,
            },
        }
    )
    maintainer.save_state(_state_fixture())
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    loaded = maintainer.load_state()
    build_owner_portrait_snapshot(
        loaded,
        enabled=True,
        auto_enabled=True,
        auto_initial_enabled=False,
        daily_enabled=True,
        self_anchor=_anchor_fixture(),
        pending=_pending_fixture(),
    )
    after = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert before == after


def test_malformed_scopes_do_not_crash():
    state = _state_fixture()
    state["portrait"]["user"] = None
    state["portrait"]["persona"] = "not-a-dict"
    result = build_owner_portrait_snapshot(
        state,
        self_anchor=None,
        pending=None,
    )
    assert result["scopes"]["user"]["stable"]["text"] == ""
    assert result["scopes"]["persona"]["recent"] == []
    assert result["self_anchor"] == {"text": "", "updated_at": "", "configured": False}
    assert result["pending"] == {"total": 0, "by_layer": {}}
