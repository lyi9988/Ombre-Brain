"""Retirement contract; synthetic history only, never production drafts."""
import asyncio
import json
import pytest
import server

TOOLS = {"darkroom_enter", "darkroom_rooms", "darkroom_view", "darkroom_status", "darkroom_release"}

class UntouchableStore:
    def __getattr__(self, name):
        raise AssertionError("Retired route touched store: " + name)

@pytest.fixture
def retired(monkeypatch, tmp_path):
    history = tmp_path / "entries.jsonl"
    history.write_text('{"note":"synthetic private draft"}\n', encoding="utf-8")
    before = history.read_bytes()
    monkeypatch.setattr(server, "darkroom_store", UntouchableStore())
    yield
    assert history.read_bytes() == before


def test_registration_and_cached_calls(retired):
    names = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert not TOOLS & names
    assert {"breath", "hold", "grow"} <= names
    async def check():
        for name in TOOLS:
            with pytest.raises(Exception, match="Unknown tool|not found"):
                await server.mcp.call_tool(name, {"note": "synthetic"} if name == "darkroom_enter" else {})
            kwargs = {"note": "synthetic"} if name == "darkroom_enter" else {}
            assert (await getattr(server, name)(**kwargs))["status"] == "retired"
    asyncio.run(check())


def test_api_auth_and_retired_no_data(retired, monkeypatch):
    monkeypatch.setattr(server, "_dashboard_authenticated", lambda request: False)
    monkeypatch.setattr(server, "_dashboard_setup_needed", lambda: False)
    assert asyncio.run(server.api_darkroom_status(object())).status_code == 401
    monkeypatch.setattr(server, "_dashboard_authenticated", lambda request: True)
    response = asyncio.run(server.api_darkroom_status(object()))
    assert response.status_code == 410
    assert json.loads(response.body) == server._darkroom_retired()
    assert server._format_handoff_darkroom_door() == ""
