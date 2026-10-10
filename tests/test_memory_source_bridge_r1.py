"""Synthetic live-base Gateway wiring: coverage to existing raw archive only."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from gateway import CANONICAL_SOURCE_EVENT_ID_HEADER, CANONICAL_TURN_PHASE_HEADER
from memory_source_provenance import canonical_source, raw_event_source, digest
from test_gateway_canonical_integration import request, service

TEXT = "🙂我喜欢喝茶，晚上不喝咖啡。"


def prepared_gateway(*, phase="initial", scope="chat", bad=None, active=True):
    gateway = service()
    gateway._canonical_continuation_active = lambda session: active
    coverage = {"conversation_id": "c", "current_user_source_event_id": "app:new", "items": [
        {"event_id": "evt-1", "version_id": "v1", "source_event_id": "app:new", "turn_member": True,
         "message_index": 1, "source_text_sha256": digest(TEXT), "recorded_at_ms": 1791471600000}]}
    headers = {CANONICAL_SOURCE_EVENT_ID_HEADER: "app:new", "X-Guyan-Conversation-Id": "c",
               "X-Guyan-Runtime-Scope": scope}
    if phase != "initial":
        headers[CANONICAL_TURN_PHASE_HEADER] = "continuation"
    if bad == "origin":
        headers[CANONICAL_SOURCE_EVENT_ID_HEADER] = "app:other"
    elif bad == "conversation":
        headers["X-Guyan-Conversation-Id"] = "other"
    elif bad == "position":
        coverage["items"][0]["message_index"] = 0
    elif bad == "hash":
        coverage["items"][0]["source_text_sha256"] = "0" * 64
    elif bad == "version":
        coverage["items"][0]["version_id"] = ""
    headers["X-Guyan-Canonical-Coverage"] = json.dumps(coverage)
    req = request(headers)
    payload = {"messages": [{"role": "system", "content": "synthetic rules"}, {"role": "user", "content": TEXT}]}
    if phase != "initial":
        payload["messages"] += [{"role": "assistant", "content": "", "tool_calls": [{"id": "call"}]},
                                {"role": "tool", "tool_call_id": "call", "content": "synthetic result"}]
    payload["_ombre_trace_context"] = gateway._trace_context_from_request(req, session_id="main", request_type=phase, client_id="reality")
    before = copy.deepcopy(payload)
    prepared, state = asyncio.run(gateway._prepare_canonical_turn(req, payload, "main", TEXT))
    return gateway, prepared, state, before


@pytest.mark.parametrize("phase", ["initial", "continuation"])
@pytest.mark.parametrize("active", [False, True])
def test_actual_prepare_header_and_successful_round_keep_source_not_duplicate_chat(phase, active):
    gateway, payload, state, before = prepared_gateway(phase=phase, active=active)
    expected = canonical_source({"event_id": "evt-1", "version_id": "v1", "role": "user", "text": TEXT},
                                profile_id="jiajia-main", conversation_id="c")
    assert state["memory_source_bridge"][0]["source_key"] == expected["source_key"]
    assert "text" not in state["memory_source_bridge"][0]
    assert sum(m.get("content") == TEXT for m in payload["messages"]) == 1
    assert gateway.canonical_adapter.ingested == []
    if not active:
        assert payload == before
    events = []
    recorded = set()
    gateway.state_store = SimpleNamespace(record_success=lambda *a: 301, record_injection_debug=lambda *a: None,
        record_conversation_turn=lambda **kw: recorded.add(kw["canonical_key"]) or 301)
    gateway._canonical_turn_key_already_recorded = lambda **kw: kw["canonical_key"] in recorded
    gateway._is_recent_duplicate_conversation_turn = lambda **kw: False
    gateway._clean_conversation_turn_text = lambda t: t
    gateway._conversation_turn_original_text = lambda t, **kw: t
    gateway.conversation_turns_max_entries = 100
    gateway.raw_event_store = SimpleNamespace(ingest=lambda rows, **kw: events.extend(rows) or {})
    debug = {"post_injection_presence": {}}
    gateway._attach_canonical_trace_debug(debug, state)
    for _ in range(2):
        asyncio.run(gateway._record_successful_round("main", [], debug, user_message=TEXT,
            assistant_message={"role": "assistant", "content": "新的合成回答。"}, model="test", client="reality", route="test", canonical_key="one-operation"))
    assert len(events) == 2
    assert events[0]["metadata"]["canonical_source"]["source_key"] == expected["source_key"]
    assert events[1]["metadata"]["canonical_source"] == {}
    assert raw_event_source({**events[0], "id": 701}, profile_id="jiajia-main")["source_key"] == expected["source_key"]


@pytest.mark.parametrize("bad", ["origin", "conversation", "position", "hash", "version"])
def test_unproven_headers_cannot_bridge(bad):
    _, _, state, _ = prepared_gateway(bad=bad, active=False)
    assert not state.get("memory_source_bridge")


def test_background_scope_cannot_attach_canonical_memory_source():
    _, payload, state, before = prepared_gateway(scope="internal:worker")
    assert not state.get("memory_source_bridge") and state["status"] == "internal_scope"
    assert payload == before
