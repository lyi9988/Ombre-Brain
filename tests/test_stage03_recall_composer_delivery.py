"""Stage 03 regression contracts for Composer memory placement and delivery."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from gateway import (
    PROMPT_BINDING_REVISION_HEADER,
    PROMPT_PRESET_HEADER,
    PROMPT_REVISION_HEADER,
    PROMPT_SCOPE_HEADER,
    PROMPT_SHA_HEADER,
)
from memory_delivery import MemoryDeliveryStore
from prompt_plan_mirror import PromptPlanMirrorStore
from test_memory_recall_projection_r2 import (
    _block,
    _compose,
    _fixture,
    _generated_sections,
    _projection,
    _request,
)


def _headers(plan, *, scope=None, sha=None):
    binding = plan.get("binding") or {}
    return {
        PROMPT_PRESET_HEADER: plan["preset_id"],
        PROMPT_REVISION_HEADER: str(plan["preset_revision"]),
        PROMPT_SHA_HEADER: sha or plan["plan_sha256"],
        PROMPT_BINDING_REVISION_HEADER: str(binding["aiz_binding_revision"]),
        PROMPT_SCOPE_HEADER: scope or plan["scope"],
    }


def _new_plan(service, tmp_path, *, scope="talk.initial", memory_mode="live_source",
              memory_budget=4000, wrapper="OWNER-WRAPPER[{{content}}]",
              bind=True):
    preset_id, revision, plan_sha = "preset:stage03-recall", 1, "e" * 64
    blocks = [
        _block("ombre.date_recall", 100, scope=scope),
        _block("ombre.memory_recall", 200, scope=scope,
               mode=memory_mode, enabled=memory_mode != "off",
               token_budget=memory_budget, wrapper_text=wrapper),
        _block("ombre.recent_context", 300, scope=scope),
        # The unified carrier must suppress these old aliases even when they
        # appear in a valid mirrored plan.
        _block("ombre.recalled_memory", 310, scope=scope),
        _block("ombre.targeted_memory_detail", 320, scope=scope),
        _block("ombre.diffused_memory", 330, scope=scope),
    ]
    mirror = PromptPlanMirrorStore(Path(service.config["state_dir"]) / "stage03-recall-plan.sqlite")
    mirror.put_plan(preset_id, revision, plan_sha256=plan_sha,
                    gateway_slice={"schema": "guyan.gateway-prompt-slice.v1",
                                   "settings": {}, "blocks": blocks})
    if bind:
        mirror.put_binding(scope, preset_id=preset_id, preset_revision=revision,
            plan_sha256=plan_sha, aiz_binding_revision=7, identity_id="synthetic-session")
    service.prompt_plan_mirror = mirror
    request = SimpleNamespace(headers={
        PROMPT_PRESET_HEADER: preset_id,
        PROMPT_REVISION_HEADER: str(revision),
        PROMPT_SHA_HEADER: plan_sha,
        PROMPT_BINDING_REVISION_HEADER: "7",
        PROMPT_SCOPE_HEADER: scope,
    })
    return service._prompt_plan_for_request(
        request,
        session_id="synthetic-session",
        continuation_phase=scope == "talk.continuation",
    ), mirror, request


def _fallback_plan(service, tmp_path, valid_off_plan, failure):
    headers = _headers(valid_off_plan)
    if failure == "missing_mirror":
        service.prompt_plan_mirror = PromptPlanMirrorStore(
            Path(tmp_path) / "missing-mirror.sqlite",
        )
    elif failure == "invalid_sha":
        headers[PROMPT_SHA_HEADER] = "f" * 64
    elif failure == "unavailable_binding":
        original = service.prompt_plan_mirror.get_plan(
            valid_off_plan["preset_id"], valid_off_plan["preset_revision"],
            include_slice=True,
        )
        mirror = PromptPlanMirrorStore(Path(tmp_path) / "unbound-mirror.sqlite")
        mirror.put_plan(valid_off_plan["preset_id"], valid_off_plan["preset_revision"],
            plan_sha256=valid_off_plan["plan_sha256"],
            gateway_slice=original["gateway_slice"])
        service.prompt_plan_mirror = mirror
    else:
        raise AssertionError(failure)
    request = SimpleNamespace(headers=headers)
    return service._prompt_plan_for_request(
        request, session_id="synthetic-session", continuation_phase=False,
    )


def _legacy_wire(service, projection, sections, *, memory_enabled):
    # This mirrors the existing Gateway legacy branch after its Composer gate:
    # non-Memory context stays available while Memory fields are omitted.
    memory = sections if memory_enabled else {"direct": "", "targeted": "", "diffused": ""}
    stable, dynamic = service._build_injected_context_messages(
        persona_block="", core_memory="CORE-ALWAYS-ON", portrait_memory="",
        just_now_context="", recent_context="", recalled_memory=memory.get("direct", ""),
        relationship_weather="", favorite_memory="", related_memory=memory.get("diffused", ""),
        targeted_memory_detail=memory.get("targeted", ""), dream_context="",
        active_reminders="", memory_detail_recall_instruction="", handoff_tool_hint="",
        context_mode="", date_persona_trace="", date_recall="",
        memory_recall_projection=projection,
    )
    messages = service._inject_context_messages(
        [{"role": "system", "content": "synthetic system"},
         {"role": "user", "content": "synthetic current question"}],
        stable, dynamic,
    )
    service._reconcile_memory_recall_injection(
        projection, stable, dynamic, enabled=memory_enabled,
    )
    return messages


def _compiled_wire(service, plan, projection, sections, messages, *, date="ORDER-A", recent="ORDER-C"):
    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        persona_block="", core_memory="CORE-ALWAYS-ON", portrait_memory="",
        just_now_context="", recent_context=recent,
        recalled_memory=sections.get("direct", ""), relationship_weather="",
        favorite_memory="", related_memory=sections.get("diffused", ""),
        targeted_memory_detail=sections.get("targeted", ""), dream_context="",
        active_reminders="", memory_detail_recall_instruction="", handoff_tool_hint="",
        context_mode="", date_persona_trace="", date_recall=date,
        memory_recall_projection=projection,
    )
    wire = service._inject_context_messages(
        messages, stable, dynamic,
        composed_messages=debug.pop("_placement_messages"),
    )
    service._reconcile_memory_recall_injection(
        projection, stable, dynamic,
        enabled=service._composer_live_recall_enabled(plan),
    )
    return wire, stable, dynamic, debug


@pytest.mark.parametrize("failure", ["missing_mirror", "invalid_sha", "unavailable_binding"])
def test_explicit_memory_off_does_not_reappear_on_gateway_legacy_fallback(
    tmp_path, monkeypatch, failure,
):
    _authority, service, view, buckets, off_plan = _fixture(
        tmp_path, monkeypatch, memory_mode="off",
    )
    request = _request(view, ["bucket-l1", "bucket-legacy"])
    sections = _generated_sections(service, request, buckets)
    projection = _projection(service, request, sections)
    assert sections["direct"] and projection["fact_body"]

    fallback = _fallback_plan(service, tmp_path, off_plan, failure)
    assert fallback["fallback_legacy"] is True
    assert fallback["reason"] in {
        "plan_unavailable", "plan_sha_mismatch", "binding_unavailable",
    }
    memory_enabled = service._composer_live_recall_enabled(fallback)
    assert memory_enabled is False

    wire = _legacy_wire(service, projection, sections, memory_enabled=memory_enabled)
    payload_text = "\n".join(str(row.get("content") or "") for row in wire)
    assert "source-record-l1: current synthetic address evidence" not in payload_text
    assert "Legacy archive narrative" not in payload_text
    assert "CORE-ALWAYS-ON" in payload_text
    assert projection["recall_status"]["status"] == "disabled"
    assert projection["recall_status"]["injected_count"] == 0


def test_no_composer_headers_keep_the_original_legacy_memory_path(tmp_path, monkeypatch):
    _authority, service, view, buckets, _plan = _fixture(tmp_path, monkeypatch)
    request = _request(view, ["bucket-l1", "bucket-legacy"])
    sections = _generated_sections(service, request, buckets)
    projection = _projection(service, request, sections)

    plan = service._prompt_plan_for_request(
        SimpleNamespace(headers={}),
        session_id="synthetic-session",
        continuation_phase=False,
    )
    assert plan is None
    assert service._composer_live_recall_enabled(plan) is True
    wire = _legacy_wire(service, projection, sections, memory_enabled=True)
    payload_text = "\n".join(str(row.get("content") or "") for row in wire)
    assert "source-record-l1: current synthetic address evidence" in payload_text
    assert "Legacy archive narrative" in payload_text
    assert projection["recall_status"]["injected_count"] == 2


@pytest.mark.parametrize("scope", ["talk.initial", "talk.continuation"])
def test_owner_wrapper_and_a_b_c_order_survive_gateway_tool_continuation(
    tmp_path, monkeypatch, scope,
):
    _authority, service, view, buckets, _initial_plan = _fixture(tmp_path, monkeypatch)
    plan, _mirror, _request_obj = _new_plan(
        service, tmp_path, scope=scope,
        wrapper="OWNER-WRAPPER-BEGIN {{content}} OWNER-WRAPPER-END",
    )
    request = _request(view, ["bucket-l1", "bucket-legacy"])
    sections = _generated_sections(service, request, buckets)
    request["_semantic_records"] = view.memory_l1_recall_map(["bucket-l1", "bucket-legacy"])
    projection = _projection(service, request, sections)

    messages = [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "earlier question"},
    ]
    if scope == "talk.continuation":
        messages.extend([
            {"role": "assistant", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "synthetic_tool", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "synthetic tool result"},
            {"role": "assistant", "content": "tool result received"},
        ])
    messages.append({"role": "user", "content": "current question"})
    wire, _stable, _dynamic, debug = _compiled_wire(
        service, plan, projection, sections, messages,
    )
    wire_text = "\n".join(str(row.get("content") or "") for row in wire)
    user_text = next(row["content"] for row in wire if row["role"] == "user" and "current question" in row["content"])
    assert user_text.index("ORDER-A") < user_text.index("OWNER-WRAPPER-BEGIN")
    assert user_text.index("OWNER-WRAPPER-BEGIN") < user_text.index("source-record-l1: current synthetic address evidence")
    assert user_text.index("source-record-l1: current synthetic address evidence") < user_text.index("OWNER-WRAPPER-END")
    assert user_text.index("OWNER-WRAPPER-END") < user_text.index("ORDER-C")
    assert user_text.count("OWNER-WRAPPER-BEGIN") == user_text.count("OWNER-WRAPPER-END") == 1
    if scope == "talk.continuation":
        call_index = next(i for i, row in enumerate(wire) if row.get("tool_calls"))
        assert wire[call_index + 1]["role"] == "tool"
        assert wire[call_index + 1]["tool_call_id"] == "call-1"
        assert "synthetic tool result" in wire_text
    assert debug["resolved_blocks"]
    assert projection["recall_status"]["injected_count"] == 2

    service.memory_delivery = MemoryDeliveryStore(Path(tmp_path) / f"delivery-{scope}.sqlite")
    forward = {"messages": wire, "_ombre_trace_context": {
        "runtime_scope": "chat", "trace_id": "stage03-wrapper-trace",
        "conversation_id": "synthetic-conversation",
        "user_source_event_id": "user-event-1",
        "assistant_source_event_id": "assistant-event-1",
    }}
    service._prepare_memory_delivery(forward, {
        "memory_recall_projection": projection,
        "canonical_continuation": {},
    })
    manifest = forward["_ombre_trace_context"]["memory_delivery_manifest"]
    l1_fragments = [row for row in manifest["fragments"] if row["bucket_id"] == "bucket-l1"]
    assert len(l1_fragments) == 1
    assert l1_fragments[0]["l1_revision"] == 1
    assert l1_fragments[0]["source_refs"][0]["binding_id"] == "source:bucket-l1"
    assert l1_fragments[0]["text"] in wire_text


@pytest.mark.parametrize(
    "memory_mode,memory_budget",
    [("off", 4000), ("live_source", 1)],
    ids=["off", "composer-budget-omitted"],
)
def test_off_or_budget_omitted_memory_is_not_a_delivery_fragment(
    tmp_path, monkeypatch, memory_mode, memory_budget,
):
    _authority, service, view, buckets, _initial_plan = _fixture(
        tmp_path, monkeypatch, memory_mode=memory_mode, memory_budget=memory_budget,
    )
    request = _request(view, ["bucket-l1", "bucket-legacy"])
    request["_semantic_records"] = view.memory_l1_recall_map(["bucket-l1", "bucket-legacy"])
    sections = _generated_sections(service, request, buckets)
    projection = _projection(service, request, sections)
    plan, _mirror, _request_obj = _new_plan(
        service, tmp_path, memory_mode=memory_mode, memory_budget=memory_budget,
        wrapper="OWNER-WRAPPER[{{content}}]",
    )
    wire, _stable, _dynamic, _debug = _compiled_wire(
        service, plan, projection, sections,
        [{"role": "system", "content": "synthetic system"},
         {"role": "user", "content": "synthetic current question"}],
    )
    final_text = "\n".join(str(row.get("content") or "") for row in wire)
    assert "source-record-l1: current synthetic address evidence" not in final_text
    assert "Legacy archive narrative" not in final_text
    assert projection["recall_status"]["injected_count"] == 0

    service.memory_delivery = MemoryDeliveryStore(Path(tmp_path) / "memory-delivery.sqlite")
    forward = {"messages": wire, "_ombre_trace_context": {
        "runtime_scope": "chat", "trace_id": "stage03-delivery-trace",
        "conversation_id": "synthetic-conversation",
        "user_source_event_id": "user-event-1",
        "assistant_source_event_id": "assistant-event-1",
    }}
    service._prepare_memory_delivery(forward, {
        "memory_recall_projection": projection,
        "canonical_continuation": {},
    })
    manifest = forward["_ombre_trace_context"]["memory_delivery_manifest"]
    assert manifest["fragments"] == []
    service._begin_memory_delivery(forward, forward)
    attempt_id = manifest["request"]["attempt_id"]
    with service.memory_delivery._connect() as db:
        row = db.execute(
            "SELECT fragments_json,successful FROM memory_delivery_attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
    assert json.loads(row["fragments_json"]) == []
    assert row["successful"] == 0
    assert service.memory_delivery.complete(manifest["scope"]) is None
