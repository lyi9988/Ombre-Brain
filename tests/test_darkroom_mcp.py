"""Darkroom MCP plumbing tests (Darkroom MCP Plumbing v1).

All fixtures are synthetic and confined to tmp_path. The module-level
``server`` import only instantiates stores (no writes); every tool-level test
replaces ``server.darkroom_store`` with a tmp_path-backed store so nothing
touches the repository-local or production ``state/darkroom`` directory.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

import server
from darkroom import DarkroomStore

DARKROOM_TOOLS = {
    "darkroom_enter",
    "darkroom_rooms",
    "darkroom_view",
    "darkroom_status",
    "darkroom_release",
}

FORBIDDEN_FRAGMENTS = (
    "token",
    "state_dir",
    "state_path",
    "bucket_id",
    "session_id",
    "profile_id",
    "entries.jsonl",
    "releases.jsonl",
)


def _repo_state_darkroom() -> Path:
    return Path(os.path.dirname(os.path.abspath(server.__file__))) / "state" / "darkroom"


def _make_store(tmp_path: Path) -> DarkroomStore:
    return DarkroomStore({"state_dir": str(tmp_path)})


@pytest.fixture()
def darkroom(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(server, "darkroom_store", store)
    return store


def _run(coro):
    return asyncio.run(coro)


def _assert_clean_response(payload: dict, tmp_path: Path):
    serialized = json.dumps(payload, ensure_ascii=False)
    for fragment in FORBIDDEN_FRAGMENTS:
        assert fragment not in serialized, f"leaked internal fragment: {fragment}"
    assert str(tmp_path) not in serialized


# ---------------------------------------------------------------------------
# tools/list registration
# ---------------------------------------------------------------------------


def test_tools_list_contains_five_darkroom_tools():
    tools = server.mcp._tool_manager.list_tools()
    names = {tool.name for tool in tools}
    assert DARKROOM_TOOLS <= names


# ---------------------------------------------------------------------------
# normal paths
# ---------------------------------------------------------------------------


def test_enter_then_rooms_view_status_release(darkroom, tmp_path):
    entered = _run(server.darkroom_enter(note="未显影的反思：今天想清楚了一件事"))
    assert entered["status"] == "entered"
    assert entered["entry_id"]
    assert entered["room_id"]
    _assert_clean_response(entered, tmp_path)

    entry_id = entered["entry_id"]

    rooms = _run(server.darkroom_rooms())
    assert rooms["status"] == "ok"
    assert rooms["count"] == 1
    assert rooms["rooms"][0]["room_id"] == entered["room_id"]
    _assert_clean_response(rooms, tmp_path)

    viewed = _run(server.darkroom_view(entry_id=entry_id))
    assert viewed["status"] == "visible"
    assert viewed["content"] == "未显影的反思：今天想清楚了一件事"
    _assert_clean_response(viewed, tmp_path)

    status = _run(server.darkroom_status())
    assert status["status"] == "ok"
    assert status["count"] == 1
    assert status["last_entry_id"] == entry_id
    _assert_clean_response(status, tmp_path)

    released = _run(server.darkroom_release(entry_id=entry_id, reason="显影"))
    assert released["status"] == "released"
    assert released["entry_id"] == entry_id
    _assert_clean_response(released, tmp_path)


def test_release_latest_works_without_entry_id(darkroom):
    entered = _run(server.darkroom_enter(note="另一条反思"))
    released = _run(server.darkroom_release())
    assert released["status"] == "released"
    assert released["entry_id"] == entered["entry_id"]


# ---------------------------------------------------------------------------
# error taxonomy
# ---------------------------------------------------------------------------


def test_view_latest_without_active_room_returns_no_active_room(darkroom):
    result = _run(server.darkroom_view(entry_id="latest"))
    assert result["status"] == "error"
    assert result["error_code"] == "no_active_room"


def test_view_explicit_missing_returns_room_not_found(darkroom):
    result = _run(server.darkroom_view(entry_id="dr_does_not_exist"))
    assert result["status"] == "error"
    assert result["error_code"] == "room_not_found"


def test_release_latest_without_active_room_returns_no_active_room(darkroom):
    result = _run(server.darkroom_release())
    assert result["status"] == "error"
    assert result["error_code"] == "no_active_room"


def test_release_explicit_missing_returns_room_not_found(darkroom):
    result = _run(server.darkroom_release(entry_id="dr_does_not_exist"))
    assert result["status"] == "error"
    assert result["error_code"] == "room_not_found"


def test_enter_invalid_visibility_returns_invalid_state(darkroom):
    result = _run(server.darkroom_enter(note="x", visibility="bogus"))
    assert result["status"] == "error"
    assert result["error_code"] == "invalid_state"


def test_release_locked_returns_permission_denied(darkroom, tmp_path):
    entered = _run(server.darkroom_enter(note="锁住的内容", lock_for="1h"))
    result = _run(server.darkroom_release(entry_id=entered["entry_id"]))
    assert result["status"] == "error"
    assert result["error_code"] == "permission_denied"
    assert result["unlock_at"]
    _assert_clean_response(result, tmp_path)


# ---------------------------------------------------------------------------
# repeat release safety
# ---------------------------------------------------------------------------


def test_repeat_release_returns_already_released_and_does_not_duplicate(darkroom):
    entered = _run(server.darkroom_enter(note="要显影的反思"))
    entry_id = entered["entry_id"]

    first = _run(server.darkroom_release(entry_id=entry_id))
    assert first["status"] == "released"

    second = _run(server.darkroom_release(entry_id=entry_id))
    assert second["status"] == "already_released"
    assert second["entry_id"] == entry_id
    assert second["released_at"]

    status = _run(server.darkroom_status())
    assert status["released_count"] == 1


# ---------------------------------------------------------------------------
# whitelist / no internal leakage
# ---------------------------------------------------------------------------


def test_status_response_has_no_internal_fields(darkroom, tmp_path):
    _run(server.darkroom_enter(note="一条反思", mood="平静", tags="a,b"))
    status = _run(server.darkroom_status())
    _assert_clean_response(status, tmp_path)
    serialized = json.dumps(status, ensure_ascii=False)
    assert "note" not in serialized
    assert "content" not in serialized


def test_rooms_response_has_no_content(darkroom, tmp_path):
    _run(server.darkroom_enter(note="正文不该出现在门口"))
    rooms = _run(server.darkroom_rooms())
    _assert_clean_response(rooms, tmp_path)
    serialized = json.dumps(rooms, ensure_ascii=False)
    assert "正文不该出现在门口" not in serialized


def test_enter_response_does_not_echo_note(darkroom, tmp_path):
    result = _run(server.darkroom_enter(note="这条正文不能被回显"))
    _assert_clean_response(result, tmp_path)
    serialized = json.dumps(result, ensure_ascii=False)
    assert "这条正文不能被回显" not in serialized


def test_tools_list_descriptions_do_not_leak_internal_fields():
    tools = server.mcp._tool_manager.list_tools()
    darkroom_tools = [tool for tool in tools if tool.name in DARKROOM_TOOLS]
    assert len(darkroom_tools) == 5
    for tool in darkroom_tools:
        input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
        serialized = json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": input_schema,
            },
            ensure_ascii=False,
        )
        for fragment in ("state_dir", "state_path", "token", "profile_id"):
            assert fragment not in serialized


# ---------------------------------------------------------------------------
# fixtures never touch production / repo-local state
# ---------------------------------------------------------------------------


def test_fixture_flow_does_not_create_repo_state_darkroom(darkroom, tmp_path):
    repo_dir = _repo_state_darkroom()
    assert not repo_dir.exists()

    _run(server.darkroom_enter(note="fixture 反思"))
    _run(server.darkroom_rooms())
    _run(server.darkroom_status())
    entered = _run(server.darkroom_enter(note="第二条"))
    _run(server.darkroom_release(entry_id=entered["entry_id"]))

    assert not repo_dir.exists()
    assert (Path(tmp_path) / "darkroom" / "entries.jsonl").exists()


def test_store_reads_and_writes_only_inside_tmp_path(darkroom, tmp_path):
    repo_dir = _repo_state_darkroom()
    before = _repo_state_darkroom().exists()

    store = darkroom
    store.enter(note="direct store fixture")
    store.status()
    store.rooms()

    assert not repo_dir.exists()
    assert before is False
    assert (Path(tmp_path) / "darkroom" / "state.json").exists()
