import tempfile
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prompt_plan_mirror import (
    PromptPlanMirrorConflict,
    PromptPlanMirrorStore,
    PromptPlanMirrorValidationError,
)
from gateway import GatewayService


def _slice(body=""):
    return {
        "blocks": [{
            "block_id": "block:memory",
            "source_id": "ombre.recalled_memory",
            "scope": "talk.initial",
            "stage": "ombre_post_injection",
            "mode": "owner_override" if body else "live_source",
            "role": "system",
            "lane": "instruction",
            "anchor": "instructions.end",
            "order": 10,
            "priority": 10,
            "enabled": True,
            "owner_body": body,
        }]
    }


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield PromptPlanMirrorStore(Path(tmp) / "mirror.sqlite3")


def test_plan_revision_is_immutable_and_idempotent(store):
    sha = "a" * 64
    first = store.put_plan("preset:test", 1, plan_sha256=sha,
                           gateway_slice=_slice())
    replay = store.put_plan("preset:test", 1, plan_sha256=sha,
                            gateway_slice=_slice())
    assert replay == first
    assert first["status"] == "verified"
    with pytest.raises(PromptPlanMirrorConflict):
        store.put_plan("preset:test", 1, plan_sha256=sha,
                       gateway_slice=_slice("override"))


def test_plan_rejects_non_gateway_source(store):
    payload = _slice()
    payload["blocks"][0]["source_id"] = "conversation.history"
    with pytest.raises(PromptPlanMirrorValidationError):
        store.put_plan("preset:test", 1, plan_sha256="b" * 64,
                       gateway_slice=payload)


def test_plan_rejects_unknown_ombre_prefix_and_request_id_conflict(store):
    payload = _slice()
    payload["blocks"][0]["source_id"] = "ombre.unregistered_private_prompt"
    with pytest.raises(PromptPlanMirrorValidationError):
        store.put_plan("preset:test", 1, plan_sha256="b" * 64,
                       gateway_slice=payload, request_id="req:unknown")
    store.put_plan("preset:test", 1, plan_sha256="b" * 64,
                   gateway_slice=_slice(), request_id="req:fixed")
    with pytest.raises(PromptPlanMirrorConflict):
        store.put_plan("preset:test", 2, plan_sha256="c" * 64,
                       gateway_slice=_slice(), request_id="req:fixed")


def test_binding_requires_existing_matching_plan(store):
    with pytest.raises(Exception):
        store.put_binding(
            "memory.query_planner", preset_id="preset:missing",
            preset_revision=1, plan_sha256="c" * 64,
            aiz_binding_revision=1)
    store.put_plan("preset:test", 1, plan_sha256="c" * 64,
                   gateway_slice=_slice())
    binding = store.put_binding(
        "memory.query_planner", preset_id="preset:test",
        preset_revision=1, plan_sha256="c" * 64,
        aiz_binding_revision=1)
    assert binding["status"] == "mirrored"
    assert store.get_binding("memory.query_planner") == binding


def test_binding_rejects_stale_or_conflicting_revision(store):
    store.put_plan("preset:test", 1, plan_sha256="d" * 64,
                   gateway_slice=_slice())
    store.put_binding(
        "memory.query_planner", preset_id="preset:test",
        preset_revision=1, plan_sha256="d" * 64,
        aiz_binding_revision=2)
    with pytest.raises(PromptPlanMirrorConflict):
        store.put_binding(
            "memory.query_planner", preset_id="preset:test",
            preset_revision=1, plan_sha256="d" * 64,
            aiz_binding_revision=1)


def test_binding_reports_degraded_if_mirrored_plan_disappears(store):
    store.put_plan("preset:test", 1, plan_sha256="d" * 64,
                   gateway_slice=_slice())
    store.put_binding(
        "memory.query_planner", preset_id="preset:test",
        preset_revision=1, plan_sha256="d" * 64,
        aiz_binding_revision=1)
    with store._connect() as db:
        db.execute("DELETE FROM prompt_plan_mirrors")
    binding = store.get_binding("memory.query_planner")
    assert binding["status"] == "degraded"
    assert binding["detail"]


def test_owner_override_body_is_bounded(store):
    with pytest.raises(PromptPlanMirrorValidationError):
        store.put_plan("preset:test", 1, plan_sha256="e" * 64,
                       gateway_slice=_slice("x" * 100_001))


class _Request:
    def __init__(self, *, method="PUT", body=None, path_params=None,
                 query_params=None, token="secret", headers=None):
        self.method = method
        self._body = body
        self.path_params = path_params or {}
        self.query_params = query_params or {}
        self.headers = {"Authorization": f"Bearer {token}", **(headers or {})}

    async def json(self):
        return self._body


def _service(store):
    service = object.__new__(GatewayService)
    service.gateway_token = "secret"
    service.prompt_plan_mirror = store
    return service


def _response_body(response):
    return json.loads(response.body.decode("utf-8"))


def test_internal_mirror_api_is_authenticated_and_owner_safe(store):
    service = _service(store)
    denied = asyncio.run(service.handle_prompt_plan_mirror(_Request(
        token="wrong", path_params={"preset_id": "preset:test", "revision": 1},
        body={})))
    assert denied.status_code == 401
    plan = asyncio.run(service.handle_prompt_plan_mirror(_Request(
        path_params={"preset_id": "preset:test", "revision": 1},
        body={"plan_sha256": "f" * 64, "gateway_slice": _slice(),
              "request_id": "req:plan:1"})))
    assert plan.status_code == 200
    binding = asyncio.run(service.handle_prompt_binding_mirror(_Request(
        path_params={"scope": "talk.initial"},
        body={
            "identity_id": "jiajia-main", "conversation_id": "conv-main",
            "preset_id": "preset:test", "preset_revision": 1,
            "plan_sha256": "f" * 64, "aiz_binding_revision": 1,
            "request_id": "req:binding:1",
        })))
    payload = _response_body(binding)["binding"]
    assert payload["status"] == "mirrored"
    assert payload["conversation_id"] == "conv-main"


def test_request_plan_headers_require_verified_binding_and_phase(store):
    service = _service(store)
    store.put_plan("preset:test", 1, plan_sha256="f" * 64,
                   gateway_slice=_legacy_gateway_slice())
    store.put_binding(
        "talk.initial", identity_id="jiajia-main",
        conversation_id="conv-main", preset_id="preset:test",
        preset_revision=1, plan_sha256="f" * 64,
        aiz_binding_revision=3)
    base_headers = {
        "X-Guyan-Prompt-Preset-Id": "preset:test",
        "X-Guyan-Prompt-Preset-Revision": "1",
        "X-Guyan-Prompt-Plan-Sha256": "f" * 64,
        "X-Guyan-Prompt-Binding-Revision": "3",
        "X-Guyan-Conversation-Id": "conv-main",
    }
    initial = service._prompt_plan_for_request(_Request(headers={
        **base_headers, "X-Guyan-Prompt-Scope": "talk.initial",
    }), session_id="jiajia-main", continuation_phase=False)
    assert initial["scope"] == "talk.initial"
    continuation = service._prompt_plan_for_request(_Request(headers={
        **base_headers, "X-Guyan-Prompt-Scope": "talk.continuation",
    }), session_id="jiajia-main", continuation_phase=True)
    assert continuation["binding"]["scope"] == "talk.initial"
    with pytest.raises(ValueError):
        service._prompt_plan_for_request(_Request(headers={
            "X-Guyan-Prompt-Preset-Id": "preset:test",
        }), session_id="jiajia-main", continuation_phase=False)


def _gateway_service_for_composer():
    service = object.__new__(GatewayService)
    service.identity = {"ai_name": "顾衍"}
    service.inject_total_budget = 10000
    return service


def _legacy_gateway_slice():
    sources = [
        (100, "ombre.core_memory", "system", "instruction", "gateway.after_first_system"),
        (110, "ombre.portrait_memory", "system", "instruction", "gateway.after_first_system"),
        (200, "ombre.just_now_context", "user", "message", "gateway.current_user_prefix"),
        (210, "ombre.date_recall", "user", "message", "gateway.current_user_prefix"),
        (220, "ombre.context_mode", "user", "message", "gateway.current_user_prefix"),
        (230, "ombre.active_reminders", "user", "message", "gateway.current_user_prefix"),
        (240, "ombre.memory_detail_request", "user", "message", "gateway.current_user_prefix"),
        (250, "ombre.memory_reading_policy", "user", "message", "gateway.current_user_prefix"),
        (300, "ombre.recalled_memory", "user", "message", "gateway.current_user_prefix"),
        (310, "ombre.targeted_memory_detail", "user", "message", "gateway.current_user_prefix"),
        (320, "ombre.diffused_memory", "user", "message", "gateway.current_user_prefix"),
        (400, "ombre.recent_context", "user", "message", "gateway.current_user_prefix"),
        (410, "ombre.date_persona_trace", "user", "message", "gateway.current_user_prefix"),
        (420, "ombre.handoff_hint", "user", "message", "gateway.current_user_prefix"),
        (500, "ombre.persona_state", "user", "message", "gateway.current_user_prefix"),
        (510, "ombre.relationship_weather", "user", "message", "gateway.current_user_prefix"),
        (520, "ombre.favorite_memory", "user", "message", "gateway.current_user_prefix"),
        (600, "ombre.dream_context", "user", "message", "gateway.current_user_prefix"),
    ]
    return {
        "schema": "guyan.gateway-prompt-slice.v1",
        "settings": {"scope_inheritance": {
            "talk.continuation": "talk.initial"}},
        "blocks": [{
            "block_id": f"block:{source}", "source_id": source,
            "scope": "talk.initial", "stage": "ombre_post_injection",
            "mode": "live_source", "role": role, "lane": lane,
            "anchor": anchor, "order": order, "priority": order,
            "enabled": True, "owner_body": "", "frozen_body": "",
            "wrapper_text": "", "token_budget": None,
        } for order, source, role, lane, anchor in sources],
    }


def test_composed_legacy_gateway_slice_is_payload_equivalent():
    service = _gateway_service_for_composer()
    values = {
        "persona_block": "persona", "core_memory": "core",
        "portrait_memory": "portrait", "just_now_context": "just-now",
        "recent_context": "recent", "recalled_memory": "recalled",
        "relationship_weather": "weather", "favorite_memory": "favorite",
        "related_memory": "diffused", "targeted_memory_detail": "targeted",
        "dream_context": "dream", "active_reminders": "reminders",
        "memory_detail_recall_instruction": "detail", "handoff_tool_hint": "handoff",
        "context_mode": "standard", "date_persona_trace": "trace",
        "date_recall": "date",
    }
    legacy = service._build_injected_context_messages(**values)
    composed = service._build_composed_context_messages({
        "preset_id": "preset:test", "preset_revision": 1,
        "plan_sha256": "f" * 64, "slice_sha256": "e" * 64,
        "scope": "talk.initial", "gateway_slice": _legacy_gateway_slice(),
        "binding": {"aiz_binding_revision": 1},
    }, **values)
    assert composed[:2] == legacy
    assert len(composed[2]["resolved_blocks"]) == 18


def test_composer_can_disable_and_override_gateway_sources():
    service = _gateway_service_for_composer()
    gateway_slice = _legacy_gateway_slice()
    recalled = next(item for item in gateway_slice["blocks"]
                    if item["source_id"] == "ombre.recalled_memory")
    recalled["mode"] = "owner_override"
    recalled["owner_body"] = "owner memory rule"
    portrait = next(item for item in gateway_slice["blocks"]
                    if item["source_id"] == "ombre.portrait_memory")
    portrait["mode"] = "off"
    stable, dynamic, debug = service._build_composed_context_messages({
        "preset_id": "preset:test", "preset_revision": 2,
        "plan_sha256": "f" * 64, "slice_sha256": "e" * 64,
        "scope": "talk.initial", "gateway_slice": gateway_slice,
        "binding": {"aiz_binding_revision": 2},
    }, persona_block="", core_memory="core", portrait_memory="portrait",
       just_now_context="", recent_context="", recalled_memory="live memory",
       relationship_weather="", favorite_memory="", related_memory="",
       targeted_memory_detail="", dream_context="", active_reminders="",
       memory_detail_recall_instruction="", handoff_tool_hint="",
       context_mode="", date_persona_trace="", date_recall="")
    assert "portrait" not in stable
    assert "owner memory rule" in dynamic
    assert "live memory" not in dynamic
    assert debug["status"] == "applied"
