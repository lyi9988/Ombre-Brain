"""Exercise the Dashboard portrait routes without starting services or providers."""
import ast
import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers


class Request:
    def __init__(self, cookie=None, marker=None, body=None):
        self.cookies = {"ombre_session": cookie} if cookie else {}
        self.headers = Headers({"x-portrait-review-session": marker} if marker else {})
        self.body = body

    async def json(self):
        return self.body


@pytest.fixture
def routes():
    names = {
        "_dashboard_authenticated", "_require_dashboard_auth",
        "_portrait_review_session", "_portrait_review_session_conflict",
        "api_portrait_state", "api_portrait_state_item_delete",
        "_portrait_expected_revision", "_portrait_mutation_response", "api_portrait_stable_edit",
    }
    source = Path(__file__).resolve().parents[1] / "server.py"
    nodes = [
        node for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    calls = []

    async def payload():
        return {"portrait": {"user": {"stable_revision": 7}}}

    async def approve(candidate_id, _bucket_mgr, *, expected_revision, display_text):
        calls.append(("approve", candidate_id, expected_revision, display_text))
        return {"status": "updated", "candidate_id": candidate_id}

    def delete(**kwargs):
        calls.append(("delete", kwargs["candidate_id"]))
        return {"status": "deleted"}

    scope = {
        "hashlib": hashlib, "time": time,
        "_dashboard_sessions": {"synthetic-cookie-a": time.time() + 60, "synthetic-cookie-b": time.time() + 60},
        "_dashboard_setup_needed": lambda: False,
        "_portrait_state_payload": payload,
        "portrait_engine": SimpleNamespace(approve_l1_v2_portrait_candidate=approve, delete_state_item=delete),
        "bucket_mgr": object(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope, calls


def session_marker(routes, cookie):
    response = asyncio.run(routes["api_portrait_state"](Request(cookie)))
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    marker = json.loads(response.body)["review_session"]
    assert len(marker) == 64 and cookie.encode() not in response.body
    return marker


def test_portrait_session_marker_requires_auth_and_changes_with_cookie(routes):
    api, calls = routes
    marker = session_marker(api, "synthetic-cookie-a")
    assert session_marker(api, "synthetic-cookie-a") == marker
    assert session_marker(api, "synthetic-cookie-b") != marker
    response = asyncio.run(api["api_portrait_state"](Request(marker=marker)))
    assert response.status_code == 401
    assert "review_session" not in json.loads(response.body) and not calls


def test_stale_browser_session_cannot_adopt_or_delete_in_new_session(routes):
    api, calls = routes
    body = {"scope": "user", "candidate_id": "candidate-a", "expected_revision": 7, "display_text": "已编辑的句子"}
    delete_body = {"area": "profile_fact_candidates", "candidate_id": "candidate-a", "confirm": "DELETE"}
    marker = session_marker(api, "synthetic-cookie-a")
    for route, request_body in (
        ("api_portrait_stable_edit", body), ("api_portrait_state_item_delete", delete_body),
    ):
        response = asyncio.run(api[route](Request("synthetic-cookie-b", marker, request_body)))
        assert response.status_code == 409
        assert json.loads(response.body)["reason"] == "portrait_review_session_changed"
        assert not calls
        response = asyncio.run(api[route](Request(marker=marker, body=request_body)))
        assert response.status_code == 401
        assert not calls
    response = asyncio.run(api["api_portrait_stable_edit"](Request("synthetic-cookie-a", marker, body)))
    assert response.status_code == 200
    response = asyncio.run(api["api_portrait_state_item_delete"](Request("synthetic-cookie-a", marker, delete_body)))
    assert response.status_code == 200
    assert calls == [("approve", "candidate-a", 7, "已编辑的句子"), ("delete", "candidate-a")]


def test_sign_out_during_portrait_read_does_not_return_the_old_payload(routes):
    api, _ = routes

    async def payload_then_expire():
        api["_dashboard_sessions"].pop("synthetic-cookie-a")
        return {"profile_fact_candidates": [{"display_text": "旧会话的候选"}]}

    api["_portrait_state_payload"] = payload_then_expire
    response = asyncio.run(api["api_portrait_state"](Request("synthetic-cookie-a")))
    assert response.status_code == 401
    assert "profile_fact_candidates" not in json.loads(response.body)
