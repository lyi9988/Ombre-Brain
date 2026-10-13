from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import GatewayService
from prompt_source_registry import fixed_prompt_source
from utils import count_tokens_approx


def _service() -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.identity = {"ai_name": "Test"}
    service.config = {}
    service.inject_total_budget = 10000
    return service


def _block(source_id: str, *, role: str, anchor: str, order: int,
           lane: str | None = None, mode: str = "live_source",
           owner_body: str = "", wrapper_text: str = "",
           token_budget: int | None = None, history_depth: int = 0,
           enabled: bool = True) -> dict:
    return {
        "block_id": f"r1:{order}:{source_id}",
        "source_id": source_id,
        "scope": "talk.initial",
        "stage": "ombre_post_injection",
        "mode": mode,
        "role": role,
        "lane": lane or ("instruction" if role in {"system", "developer"} else "message"),
        "anchor": anchor,
        "history_depth": history_depth,
        "order": order,
        "priority": order,
        "enabled": enabled,
        "owner_body": owner_body,
        "frozen_body": "",
        "wrapper_text": wrapper_text,
        "token_budget": token_budget,
        "condition": {},
    }


def _plan(*blocks: dict, scope: str = "talk.initial") -> dict:
    return {
        "preset_id": "preset:context-placement-r1",
        "preset_revision": 17,
        "plan_sha256": "a" * 64,
        "slice_sha256": "b" * 64,
        "scope": scope,
        "binding": {"aiz_binding_revision": 4},
        "gateway_slice": {
            "settings": {"scope_inheritance": {"talk.continuation": "talk.initial"}},
            "blocks": list(blocks),
        },
    }


def _context_values() -> dict:
    return {
        "persona_block": "",
        "core_memory": "CORE_BLOCK_R1",
        "portrait_memory": "PORTRAIT_BLOCK_R1",
        "just_now_context": "JUST_NOW_R1",
        "recent_context": "",
        "recalled_memory": "",
        "relationship_weather": "RELATIONSHIP_R1",
        "favorite_memory": "FAVORITE_R1",
        "related_memory": "",
        "targeted_memory_detail": "",
        "dream_context": "",
        "active_reminders": "",
        "memory_detail_recall_instruction": "",
        "handoff_tool_hint": "",
        "context_mode": "",
        "date_persona_trace": "",
        "date_recall": "DATE_RECALL_R1",
    }


def _compose_and_inject(service, plan, messages, *, coverage=None,
                        memory_recall_projection=None, values=None):
    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        **(values or _context_values()),
        memory_recall_projection=memory_recall_projection or {},
    )
    # The builder exposes this request-local value only to its caller.  The
    # normal prepare path pops it before composing owner-visible debug data.
    records = debug.pop("_placement_messages")
    prepared = service._inject_context_messages(
        messages,
        stable,
        dynamic,
        composed_messages=records,
        coverage=coverage,
    )
    assert "_placement_messages" not in debug
    return prepared, stable, dynamic, records, debug


def _index(messages: list[dict], text: str) -> int:
    return next(i for i, message in enumerate(messages)
                if text in str(message.get("content") or ""))


def _gateway_wire_messages(messages: list[dict]) -> list[dict]:
    """Model the public chat body: local projection/event metadata is header-only."""
    return [
        {key: copy.deepcopy(value) for key, value in row.items()
         if key not in {"_prompt_projection", "_guyan_event"}}
        for row in messages
    ]


def _worldbook_boundary_signature(messages: list[dict]) -> list[tuple]:
    canonical = [
        (str(row.get("content") or ""), index)
        for index, row in enumerate(messages)
        if row.get("content") in {"HISTORY_OLD", "HISTORY_NEW", "CURRENT_USER_R1"}
    ]
    return [
        (str(row.get("role") or ""), str(row.get("content") or ""), tuple(
            (body, index < event_index)
            for body, event_index in canonical
        ))
        for index, row in enumerate(messages)
        if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}
    ]


def test_gateway_resolver_places_blocks_at_four_instruction_and_turn_positions():
    service = _service()
    blocks = [
        _block("ombre.gateway_stable_preface_prompt", role="developer",
               anchor="instructions.start", order=10,
               mode="owner_override", owner_body="FIXED_OWNER_A"),
        _block("ombre.gateway_dynamic_preface_prompt", role="system",
               anchor="gateway.after_first_system", order=20,
               mode="owner_override", owner_body="AFTER_FIRST_SYSTEM_C"),
        _block("ombre.core_memory", role="system", anchor="instructions.end",
               order=30),
        _block("ombre.date_recall", role="user", anchor="history.start",
               order=40),
        _block("ombre.memory_recall", role="assistant", anchor="history.depth",
               order=50, history_depth=1),
        _block("ombre.just_now_context", role="user", anchor="current_user.before",
               order=60),
        _block("ombre.favorite_memory", role="user",
               anchor="gateway.current_user_prefix", order=70),
        _block("ombre.relationship_weather", role="developer",
               anchor="current_user.after", order=80),
    ]
    messages = [
        {"role": "system", "content": "SYS_0"},
        {"role": "developer", "content": "DEV_0"},
        {"role": "system", "content": "SYS_1"},
        {"role": "user", "content": "HISTORY_OLD"},
        {"role": "assistant", "content": "HISTORY_NEW"},
        {"role": "user", "content": "CURRENT_USER"},
    ]
    coverage = {
        "history_message_ranges": [[3, 5]],
        "current_user_message_ranges": [[5, 6]],
    }

    prepared, stable, dynamic, records, debug = _compose_and_inject(
        service,
        _plan(*blocks),
        messages,
        coverage=coverage,
        memory_recall_projection={"body": "MEMORY_RECALL_C"},
    )

    assert len(records) == 8
    assert [row["source_id"] for row in debug["resolved_blocks"]] == [
        block["source_id"] for block in blocks
    ]
    assert "FIXED_OWNER_A" in stable
    assert "MEMORY_RECALL_C" in dynamic
    assert _index(prepared, "FIXED_OWNER_A") < _index(prepared, "SYS_0")
    assert _index(prepared, "SYS_0") < _index(prepared, "AFTER_FIRST_SYSTEM_C")
    assert _index(prepared, "AFTER_FIRST_SYSTEM_C") < _index(prepared, "DEV_0")
    assert _index(prepared, "DEV_0") < _index(prepared, "SYS_1")
    assert _index(prepared, "SYS_1") < _index(prepared, "CORE_BLOCK_R1")
    assert _index(prepared, "CORE_BLOCK_R1") < _index(prepared, "DATE_RECALL_R1")
    assert _index(prepared, "DATE_RECALL_R1") < _index(prepared, "HISTORY_OLD")
    assert _index(prepared, "HISTORY_OLD") < _index(prepared, "MEMORY_RECALL_C")
    assert _index(prepared, "MEMORY_RECALL_C") < _index(prepared, "HISTORY_NEW")
    assert _index(prepared, "JUST_NOW_R1") < _index(prepared, "CURRENT_USER")
    assert _index(prepared, "FAVORITE_R1") == _index(prepared, "CURRENT_USER")
    assert _index(prepared, "CURRENT_USER") < _index(prepared, "RELATIONSHIP_R1")
    memory_message = next(row for row in prepared
                          if "MEMORY_RECALL_C" in str(row.get("content") or ""))
    assert memory_message["role"] == "assistant"
    assert prepared[-1]["role"] == "developer"


def test_fixed_wrapper_override_is_once_off_does_not_restore_factory_and_budgets_apply():
    service = _service()
    fixed_id = "ombre.gateway_stable_preface_prompt"
    original = [{"role": "system", "content": "BASE"},
                {"role": "user", "content": "CURRENT"}]
    wrapped = _block(
        fixed_id,
        role="system",
        anchor="instructions.end",
        order=1,
        mode="owner_override",
        owner_body="OWNER_FIXED_BODY_R1",
        wrapper_text="OWNER_WRAP[{{content}}]",
    )
    prepared, _stable, _dynamic, records, _debug = _compose_and_inject(
        service, _plan(wrapped), original,
    )
    assert len(records) == 1
    assert records[0]["content"] == "OWNER_WRAP[OWNER_FIXED_BODY_R1]"
    assert sum("OWNER_FIXED_BODY_R1" in str(row.get("content") or "")
               for row in prepared) == 1

    off = _block(fixed_id, role="system", anchor="instructions.start",
                 order=1, mode="off", enabled=False)
    off_prepared, stable, dynamic, records, _debug = _compose_and_inject(
        service, _plan(off), original,
    )
    factory_body = fixed_prompt_source(fixed_id).factory_body()
    assert stable == dynamic == ""
    assert records == []
    assert off_prepared == original
    assert factory_body not in "\n".join(str(row.get("content") or "")
                                         for row in off_prepared)

    first_cost = count_tokens_approx("FIRST_BUDGETED_BLOCK")
    first = _block(fixed_id, role="system", anchor="instructions.start",
                   order=1, mode="owner_override",
                   owner_body="FIRST_BUDGETED_BLOCK")
    second = _block("ombre.gateway_dynamic_preface_prompt", role="system",
                    anchor="instructions.end", order=2,
                    mode="owner_override", owner_body="SECOND_BUDGETED_BLOCK")
    service.inject_total_budget = first_cost
    _prepared, _stable, _dynamic, records, _debug = _compose_and_inject(
        service, _plan(first, second), original,
    )
    assert [row["content"] for row in records] == ["FIRST_BUDGETED_BLOCK"]

    first["token_budget"] = 0
    service.inject_total_budget = 10000
    _prepared, _stable, _dynamic, records, _debug = _compose_and_inject(
        service, _plan(first, second), original,
    )
    assert [row["content"] for row in records] == ["SECOND_BUDGETED_BLOCK"]

    service.inject_total_budget = 0
    _prepared, stable, dynamic, records, _debug = _compose_and_inject(
        service, _plan(second), original,
    )
    assert stable == dynamic == ""
    assert records == []


def test_staged_two_message_current_range_and_multipart_prefix_are_preserved():
    service = _service()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "HISTORY_USER"},
        {"role": "assistant", "content": "HISTORY_ASSISTANT"},
        {"role": "user", "content": "STAGED_CURRENT_1"},
        {"role": "user", "content": [
            {"type": "text", "text": "STAGED_CURRENT_2"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,synthetic"}},
        ]},
    ]
    original_last_content = copy.deepcopy(messages[-1]["content"])
    coverage = {
        "history_message_ranges": [[1, 3]],
        "current_user_message_ranges": [[3, 5]],
    }
    output = __import__("prompt_placement").place_context_messages(
        messages,
        [
            {"role": "system", "anchor": "current_user.before",
             "content": "BEFORE_STAGED_RANGE"},
            {"role": "user", "anchor": "gateway.current_user_prefix",
             "content": "STAGED_PREFIX"},
            {"role": "assistant", "anchor": "current_user.after",
             "content": "AFTER_STAGED_RANGE"},
        ],
        coverage,
    )

    assert _index(output, "BEFORE_STAGED_RANGE") < _index(output, "STAGED_CURRENT_1")
    assert _index(output, "BEFORE_STAGED_RANGE") < _index(output, "STAGED_CURRENT_2")
    assert _index(output, "STAGED_CURRENT_1") < _index(output, "AFTER_STAGED_RANGE")
    assert _index(output, "STAGED_CURRENT_2") < _index(output, "AFTER_STAGED_RANGE")
    last = next(row for row in output if isinstance(row.get("content"), list)
                and any("STAGED_CURRENT_2" in str(part) for part in row["content"]))
    assert last["content"][0]["text"].startswith("<ombre_live_context>")
    assert "STAGED_PREFIX" in last["content"][0]["text"]
    assert last["content"][1:] == original_last_content


def test_history_depth_cannot_split_a_tool_call_from_two_results():
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "HISTORY_OLD"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-a", "type": "function"},
            {"id": "call-b", "type": "function"},
        ]},
        {"role": "tool", "tool_call_id": "call-a", "content": "RESULT_A"},
        {"role": "tool", "tool_call_id": "call-b", "content": "RESULT_B"},
        {"role": "assistant", "content": "HISTORY_AFTER_TOOLS"},
        {"role": "user", "content": "CURRENT"},
    ]
    coverage = {
        "history_message_ranges": [[2, 5]],
        "current_user_message_ranges": [[6, 7]],
    }
    output = __import__("prompt_placement").place_context_messages(
        messages,
        [{"role": "system", "anchor": "history.depth",
          "history_depth": 1, "content": "ATOMIC_GROUP_MARKER"}],
        coverage,
    )
    marker = _index(output, "ATOMIC_GROUP_MARKER")
    call = next(i for i, row in enumerate(output)
                if row.get("tool_calls"))
    result_a = next(i for i, row in enumerate(output)
                    if row.get("tool_call_id") == "call-a")
    result_b = next(i for i, row in enumerate(output)
                    if row.get("tool_call_id") == "call-b")
    assert marker < call
    assert result_a == call + 1
    assert result_b == result_a + 1


def test_legacy_none_injection_keeps_the_existing_message_text_exactly():
    service = _service()
    messages = [
        {"role": "system", "content": "BASE_SYSTEM"},
        {"role": "user", "content": "CURRENT_TEXT"},
    ]
    expected = [
        {"role": "system", "content": "BASE_SYSTEM"},
        {"role": "system", "content": "LEGACY_STABLE_TEXT"},
        {"role": "user", "content": (
            "<ombre_live_context>\nLEGACY_DYNAMIC_TEXT\n</ombre_live_context>\n\n"
            "Current user message:\nCURRENT_TEXT")},
    ]
    assert service._inject_context_messages(
        messages, "LEGACY_STABLE_TEXT", "LEGACY_DYNAMIC_TEXT",
        composed_messages=None,
    ) == expected
    assert messages[1]["content"] == "CURRENT_TEXT"


def test_snapshot_replay_reuses_typed_placements_without_flattening_or_duplication():
    service = _service()
    service.prepare_snapshot_ttl_seconds = 300
    service._restore_cached_reasoning_content = lambda *_args, **_kwargs: None
    service._apply_prompt_cache_hints = lambda *_args, **_kwargs: None
    base_messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "HISTORY_OLD"},
        {"role": "assistant", "content": "HISTORY_NEW"},
        {"role": "user", "content": "CURRENT"},
    ]
    tool_tail = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-a", "type": "function"},
            {"id": "call-b", "type": "function"},
        ]},
        {"role": "tool", "tool_call_id": "call-a", "content": "RESULT_A"},
        {"role": "tool", "tool_call_id": "call-b", "content": "RESULT_B"},
    ]
    placements = [{
        "block_id": "memory:cached",
        "source_id": "ombre.memory_recall",
        "role": "assistant",
        "anchor": "history.depth",
        "history_depth": 1,
        "content": "CACHED_MEMORY_R1",
    }]
    identity = {
        "key_hash": "snapshot-context-placement-r1",
        "snapshot_parent_key": "snapshot-parent-context-placement-r1",
        "source_phase": "initial",
        "transport_session_id": "local-no-provider",
        "baseline_messages": base_messages,
        "tool_tail": tool_tail,
    }
    coverage = {
        "history_message_ranges": [[1, 3]],
        "current_user_message_ranges": [[3, 4]],
    }
    service._prepare_snapshot_store(
        identity,
        prepared_payload={"model": "local-no-provider", "messages": base_messages},
        base_forward_messages=base_messages,
        stable_context="",
        dynamic_context="CACHED_MEMORY_R1",
        injected_ids=["memory-r1"],
        debug_payload=None,
        composed_messages=placements,
    )
    entry = service._prepare_snapshot_get(identity["key_hash"])
    assert entry["composed_messages"] == placements

    replayed, injected_ids = service._prepare_snapshot_replay(
        entry,
        identity,
        {
            "model": "local-no-provider",
            "stream": False,
            "messages": [*base_messages, *tool_tail],
            "_ombre_trace_context": {"coverage": coverage},
        },
        continuation_phase=True,
        started_at=time.perf_counter(),
        include_debug=False,
    )
    final = replayed["messages"]
    assert injected_ids == ["memory-r1"]
    assert sum("CACHED_MEMORY_R1" in str(row.get("content") or "")
               for row in final) == 1
    memory = next(row for row in final
                  if "CACHED_MEMORY_R1" in str(row.get("content") or ""))
    assert memory["role"] == "assistant"
    assert _index(final, "HISTORY_OLD") < _index(final, "CACHED_MEMORY_R1")
    assert _index(final, "CACHED_MEMORY_R1") < _index(final, "HISTORY_NEW")
    call = next(i for i, row in enumerate(final) if row.get("tool_calls"))
    assert final[call + 1:call + 3] == tool_tail[1:]


def _prepare_harness() -> GatewayService:
    service = _service()
    service.upstream_default_model = "local-no-provider"
    service.operit_context_rewrite_enabled = False
    service.just_now_context_enabled = False
    service.date_recall_enabled = False
    service.automatic_recall_enabled = True
    service.recall_recent_turns = 4
    service.retrieval_mode = "graph"
    service.state_store = SimpleNamespace(get_last_success_at=lambda _session: None)
    service._maybe_reload_runtime_overlay = lambda: None
    service._get_upstream_for_model = lambda _model: None
    service._extract_current_turn_user_query = lambda _messages: ""
    service._messages_contain_handoff_context = lambda _messages: False
    service._query_is_handoff_trigger = lambda _query: False
    service._query_prefers_session_start_handoff = lambda _query: False
    service._query_requests_date_recall = lambda _query: False
    service._auto_recall_low_signal_query = lambda _query: False
    service._should_inject_interval = lambda *_args: False
    service._strip_external_context_from_user_text = lambda text: str(text or "")
    service._list_gateway_buckets = _no_buckets
    service._portrait_memory_debug_base = lambda: {}
    service._just_now_context_debug_base = lambda _query: {}
    service._date_recall_debug_base = lambda _query: {}
    service._date_persona_trace_debug_base = lambda _query: {}
    service._targeted_memory_detail_debug_base = lambda: {}
    service._query_planner_debug_base = lambda _query: {}
    service._memory_sentinel_debug_base = lambda _query: {}
    service._domain_sentinel_rule_plan = lambda _query: {}
    service._project_legacy_recall_status = lambda *_args, **_kwargs: None
    service._build_memory_recall_projection = lambda **_kwargs: {
        "body": "PRIVATE_DEBUG_BODY_R1",
        "fact_body": "PRIVATE_FACT_BODY_R1",
        "selected_memory_ids": [],
        "selected_bucket_ids": [],
        "injected_bucket_ids": [],
        "recall_status": {"status": "disabled", "injected_count": 0},
    }
    service._reconcile_memory_recall_injection = lambda *_args, **_kwargs: None
    service._effective_config_metadata = lambda: {"sha256": "local-config"}
    service._build_injection_debug_payload = lambda **_kwargs: {}
    service._rejected_bucket_evidence = lambda _buckets: {"reason_counts": {}}
    service._restore_cached_reasoning_content = lambda *_args, **_kwargs: None
    service._apply_prompt_cache_hints = lambda *_args, **_kwargs: None
    service._prepare_snapshot_identity = lambda *_args, **_kwargs: {
        "key_hash": "prepare-debug-context-placement-r1",
        "snapshot_parent_key": "prepare-debug-parent-r1",
        "source_phase": "initial",
        "transport_session_id": "local-no-provider",
    }
    return service


async def _no_buckets(*_args, **_kwargs):
    return []


def test_prepare_debug_and_snapshot_debug_drop_private_placement_bodies():
    service = _prepare_harness()
    plan = _plan(_block(
        "ombre.gateway_stable_preface_prompt",
        role="system",
        anchor="instructions.end",
        order=1,
        mode="owner_override",
        owner_body="PRIVATE_DEBUG_BODY_R1",
    ))
    payload = {
        "model": "local-no-provider",
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "CURRENT_USER"},
        ],
        "_ombre_trace_context": {
            "runtime_scope": "chat",
            "coverage": {"current_user_message_ranges": [[1, 2]]},
        },
    }
    forwarded, _injected_ids, debug = __import__("asyncio").run(
        service.prepare_payload(
            payload,
            "local-no-provider",
            include_debug=True,
            prompt_plan=plan,
        )
    )

    assert sum("PRIVATE_DEBUG_BODY_R1" in str(row.get("content") or "")
               for row in forwarded["messages"]) == 1
    assert "_placement_messages" not in debug["prompt_composer"]
    assert "PRIVATE_DEBUG_BODY_R1" not in json.dumps(debug, ensure_ascii=False)
    entry = service._prepare_snapshots[
        "prepare-debug-context-placement-r1"]
    assert entry["composed_messages"][0]["content"] == "PRIVATE_DEBUG_BODY_R1"
    assert "_placement_messages" not in entry["debug_payload"]["prompt_composer"]
    assert "PRIVATE_DEBUG_BODY_R1" not in json.dumps(
        entry["debug_payload"], ensure_ascii=False)


_AIZ_NO_PROVIDER_SCRIPT = r'''\
import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from core.emotion_behavior import EmotionBehaviorAdapter, DEFAULT_BEHAVIOR_PROMPT
from core.prompt_composer import PromptComposerStore, legacy_plan
from core.prompt_compiler import PromptCompiler
from core.sdk_runtime import OmbreSdkModel, SdkRuntimeRouter

IDENTITY = "context-placement-r1"
CONVERSATION = "context-placement-r1"
SCOPE = "talk.initial"
NEEDS_TIME = "2026-10-13T10:00:00+08:00"
AFFECT_TIME = "2026-10-13T09:59:00+08:00"

class AffectReader:
    async def fetch(self, _loop, *, consumer="reply_behavior"):
        assert consumer == "reply_behavior"
        return {
            "affect": {"label": "mixed-playful", "intensity": "medium"},
            "drive": {"labels": ["companionship"]},
            "relationship": {"weather": "close", "trust": "high"},
            "updated_at": AFFECT_TIME,
        }, "cached"

plan = legacy_plan()
wrapper = next(row for row in plan["blocks"]
               if row["source_id"] == "emotion.behavior_wrapper")
wrapper.update(mode="owner_override", owner_body="OWNER_EMOTION_WRAPPER_R17",
               role="user", lane="message", anchor="history.depth",
               history_depth=1, order=10)
state = next(row for row in plan["blocks"]
             if row["source_id"] == "emotion.current_state")
state.update(role="assistant", lane="message", anchor="history.depth",
             history_depth=1, order=30)
worldbook = next(row for row in plan["blocks"]
                 if row["source_id"] == "worldbook.entries")
worldbook.update(history_depth=1, order=25)

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
store = PromptComposerStore(root / "composer-authority.sqlite3")
created = store.create_preset(
    name="Gateway placement local validation",
    plan=plan,
    request_id="context-placement-r1:create",
)
store.record_gateway_mirror(
    created["preset_id"], 1,
    plan_sha256=created["revision"]["plan_sha256"], status="verified")
binding = store.binding(IDENTITY, SCOPE, conversation_id=CONVERSATION)
store.apply_binding(
    IDENTITY, SCOPE, conversation_id=CONVERSATION,
    preset_id=created["preset_id"], preset_revision=1,
    expected_binding_revision=binding["binding_revision"],
    request_id="context-placement-r1:apply",
)

messages = [
    {"role": "system", "content": "AIZ_FIXED_INSTRUCTION_R1",
     "_prompt_projection": {"source": "guyan.identity", "revision": "local-r1"}},
    {"role": "user", "content": "HISTORY_OLD",
     "_guyan_event": {"event_id": "history-1", "event_seq": 1,
                      "turn_member": False}},
    {"role": "assistant", "content": "HISTORY_NEW",
     "_guyan_event": {"event_id": "history-2", "event_seq": 2,
                      "turn_member": False}},
    {"role": "system", "content": "WORLD_BOOK_ALPHA",
     "_prompt_projection": {"source": "worldbook.entry",
                            "source_refs": {"entry_id": "local-wb-1"}}},
    {"role": "system", "content": "WORLD_BOOK_BETA",
     "_prompt_projection": {"source": "worldbook.entry",
                            "source_refs": {"entry_id": "local-wb-2"}}},
    {"role": "user", "content": "CURRENT_USER_R1",
     "_guyan_event": {"event_id": "current-1", "event_seq": 3,
                      "turn_member": True}},
]
needs = {
    "values": {"energy": 58, "rest": 31, "curiosity": 47,
               "novelty": 52, "expression": 63, "creativity": 71,
               "autonomy": 44, "competence": 38, "solitude": 29},
    "rest_session": {}, "updated_at": NEEDS_TIME,
}
prepared, _decision = asyncio.run(EmotionBehaviorAdapter(
    {"mode": "apply"}, affect_projection=AffectReader()).prepare(
        SimpleNamespace(emotion_experience_reducer=None, motivation_engine=None),
        messages, needs_snapshot=needs))
compiled, compile_debug = PromptCompiler(store).compile(
    prepared, identity_id=IDENTITY, conversation_id=CONVERSATION, scope=SCOPE)

sdk_model = object.__new__(OmbreSdkModel)
sdk_model._composer_tool_blocks = lambda: (False, {})
sdk_model.tool_prompt_resolver = None
sdk_model.catalog = None
sdk_model.turn_context = SimpleNamespace(
    conversation_id=CONVERSATION,
    context_revision="local-context-r1",
    canonical_user_source_event_id="current-1",
)
legacy_instruction = {"role": "system", "content": "LEGACY_SDK_INSTRUCTION_R1"}
sdk_source_messages = [legacy_instruction, *compiled]
sdk_router = object.__new__(SdkRuntimeRouter)
sdk_router._composer_tool_blocks = lambda: (False, {})
sdk_router.tool_prompt_resolver = None
sdk_router.catalog = None
instruction_blocks = sdk_router._instruction_blocks(sdk_source_messages)
sdk_system_instructions = "\n\n".join(text for text, _meta in instruction_blocks)
sdk_input = SdkRuntimeRouter._sdk_input_messages(sdk_source_messages)
message_offset = OmbreSdkModel._coverage_message_offset(sdk_system_instructions)
coverage = json.loads(sdk_model._coverage_header(
    sdk_input, message_offset=message_offset))
model_input = {
    "model": "local-no-provider",
    "messages": [{"role": "system", "content": sdk_system_instructions}, *sdk_input],
}
discovery_input = SdkRuntimeRouter._discovery_input_messages(
    sdk_source_messages, owner_events=[], recent_history_limit=0)
Path(sys.argv[2]).write_text(json.dumps({
    "model_input": model_input,
    "coverage": coverage,
    "compile_debug": compile_debug,
    "sdk_system_instructions": sdk_system_instructions,
    "compiled_messages": compiled,
    "discovery_input": discovery_input,
    "default_behavior_prompt": DEFAULT_BEHAVIOR_PROMPT.strip(),
    "expected_source_times": {
        "needs_updated_at": NEEDS_TIME,
        "affect_updated_at": AFFECT_TIME,
    },
}, ensure_ascii=False), encoding="utf-8")
'''


@pytest.fixture(scope="module")
def _real_aiz_sdk_artifact(request):
    project_root = Path(__file__).resolve().parents[3]
    aiz_root = project_root / "worktrees" / "aizizhu-prompt-context-contract-r1"
    aiz_python = project_root / "worktrees" / "aizizhu-recall-r1" / ".venv" / "Scripts" / "python.exe"
    assert aiz_root.is_dir()
    assert aiz_python.is_file()
    temporary = tempfile.TemporaryDirectory(
        prefix="prompt-context-placement-r1-",
        dir=str(Path(__file__).resolve().parents[1]),
    )
    request.addfinalizer(temporary.cleanup)
    tmp_path = Path(temporary.name)
    artifact_dir = tmp_path / "aiz-temporary-authority"
    artifact_file = tmp_path / "aiz-model-input.json"
    result = subprocess.run(
        [str(aiz_python), "-c", _AIZ_NO_PROVIDER_SCRIPT,
         str(artifact_dir), str(artifact_file)],
        cwd=aiz_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (
        f"Aiz SDK no-provider preparation failed ({result.returncode}): "
        f"stdout={result.stdout[-2000:]} stderr={result.stderr[-4000:]}"
    )
    return json.loads(artifact_file.read_text(encoding="utf-8"))


def test_real_aiz_sdk_input_flows_through_gateway_without_provider_call(
        _real_aiz_sdk_artifact):
    artifact = _real_aiz_sdk_artifact
    assert artifact["compile_debug"]["status"] == "applied"
    model_input = artifact["model_input"]
    messages = model_input["messages"]
    coverage = artifact["coverage"]
    worldbooks_before = [copy.deepcopy(row) for row in messages
                         if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}]
    worldbook_position_before = _worldbook_boundary_signature(messages)
    assert [row["content"] for row in worldbooks_before] == [
        "WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA",
    ]

    states = [row for row in messages
              if (row.get("_prompt_projection") or {}).get("source")
              == "emotion.current_state"]
    assert len(states) == 1
    state_body = states[0]["content"]
    state = json.loads(state_body.removeprefix("CURRENT_STATE="))
    assert state["source_times"] == artifact["expected_source_times"]
    all_text_before = "\n".join(str(row.get("content") or "") for row in messages)
    assert all_text_before.count("OWNER_EMOTION_WRAPPER_R17") == 1
    assert artifact["default_behavior_prompt"] not in all_text_before
    assert "LEGACY_SDK_INSTRUCTION_R1" in artifact["sdk_system_instructions"]
    assert "AIZ_FIXED_INSTRUCTION_R1" not in artifact["sdk_system_instructions"]
    fixed_instruction = next(row for row in messages
                             if row.get("content") == "AIZ_FIXED_INSTRUCTION_R1")
    assert fixed_instruction["role"] == "system"
    assert all("LEGACY_SDK_INSTRUCTION_R1" not in str(row.get("content") or "")
               for row in messages[1:])
    assert "AIZ_FIXED_INSTRUCTION_R1" in "\n".join(
        str(row.get("content") or "") for row in artifact["discovery_input"])
    assert "OWNER_EMOTION_WRAPPER_R17" in "\n".join(
        str(row.get("content") or "") for row in artifact["discovery_input"])
    assert "CURRENT_STATE=" in "\n".join(
        str(row.get("content") or "") for row in artifact["discovery_input"])
    assert "WORLD_BOOK_ALPHA" in "\n".join(
        str(row.get("content") or "") for row in artifact["discovery_input"])
    assert "HISTORY_OLD" not in "\n".join(
        str(row.get("content") or "") for row in artifact["discovery_input"])
    assert "HISTORY_NEW" not in "\n".join(
        str(row.get("content") or "") for row in artifact["discovery_input"])

    prompt_blocks = coverage["prompt_blocks"]
    assert all(isinstance(row, list) and len(row) == 6 for row in prompt_blocks)
    assert "OWNER_EMOTION_WRAPPER_R17" not in json.dumps(coverage, ensure_ascii=False)
    assert "CURRENT_STATE=" not in json.dumps(coverage, ensure_ascii=False)
    assert "AIZ_FIXED_INSTRUCTION_R1" not in json.dumps(coverage, ensure_ascii=False)
    assert "WORLD_BOOK_ALPHA" not in json.dumps(coverage, ensure_ascii=False)
    for index, _anchor, _order, _block_id, digest, _depth in prompt_blocks:
        body = messages[index].get("content")
        assert isinstance(body, str)
        assert hashlib.sha256(body.encode("utf-8")).hexdigest() == digest


def test_cross_stage_same_anchor_aiz_gateway_a_b_c_off_worldbook_and_snapshot(
        _real_aiz_sdk_artifact):
    artifact = _real_aiz_sdk_artifact
    model_input = artifact["model_input"]
    messages = model_input["messages"]
    coverage = artifact["coverage"]
    local_worldbooks = [row for row in messages
                        if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}]
    assert len(local_worldbooks) == 2
    refs = [row["_prompt_projection"]["source_refs"] for row in local_worldbooks]
    worldbook_block_ids = {row.get("block_id") for row in refs}
    descriptors = {row[3]: row for row in coverage["prompt_blocks"]}
    wrapper = next(row for row in messages
                   if (row.get("_prompt_projection") or {}).get("source")
                   == "emotion.behavior_wrapper")
    state = next(row for row in messages
                 if (row.get("_prompt_projection") or {}).get("source")
                 == "emotion.current_state")
    wrapper_refs = wrapper["_prompt_projection"]["source_refs"]
    state_refs = state["_prompt_projection"]["source_refs"]
    wrapper_row = descriptors[wrapper_refs["block_id"]]
    state_row = descriptors[state_refs["block_id"]]
    assert wrapper_row[1:3] == ["history.depth", 10]
    assert wrapper_row[5] == 1
    assert state_row[1:3] == ["history.depth", 30]
    assert state_row[5] == 1
    assert not (worldbook_block_ids & set(descriptors))
    wrapper_body, wrapper_role = wrapper["content"], wrapper["role"]
    state_body, state_role = state["content"], state["role"]

    # The SDK sends role/content plus the coverage header. Private source refs
    # are local compiler metadata and are deliberately absent at Gateway.
    messages = _gateway_wire_messages(messages)
    worldbooks_before = [
        (row.get("role"), row.get("content")) for row in messages
        if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}
    ]
    assert len(worldbooks_before) == 2
    assert all("_prompt_projection" not in row and "_guyan_event" not in row
               for row in messages)
    worldbook_position_before = _worldbook_boundary_signature(messages)

    service = _service()
    gateway_block = _block(
        "ombre.memory_recall",
        role="assistant",
        anchor="history.depth",
        order=20,
        history_depth=1,
    )
    plan = _plan(gateway_block)
    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        **{**_context_values(), "core_memory": "", "date_recall": "",
           "just_now_context": "", "relationship_weather": "",
           "favorite_memory": ""},
        memory_recall_projection={"body": "SELECTED_MEMORY_R1"},
    )
    placements = debug.pop("_placement_messages")
    assert len(placements) == 1
    assert placements[0]["order"] == 20
    outgoing_model_input = copy.deepcopy(model_input)
    outgoing_model_input["messages"] = service._inject_context_messages(
        messages,
        stable,
        dynamic,
        composed_messages=placements,
        coverage=coverage,
    )

    final_messages = outgoing_model_input["messages"]
    all_text_after = "\n".join(str(row.get("content") or "")
                                for row in final_messages)
    assert outgoing_model_input["model"] == "local-no-provider"
    assert all_text_after.count("OWNER_EMOTION_WRAPPER_R17") == 1
    assert all_text_after.count("SELECTED_MEMORY_R1") == 1
    memory = next(row for row in final_messages
                  if "SELECTED_MEMORY_R1" in str(row.get("content") or ""))
    assert memory["role"] == "assistant"
    assert _index(final_messages, wrapper_body) < _index(
        final_messages, "SELECTED_MEMORY_R1")
    assert _index(final_messages, "SELECTED_MEMORY_R1") < _index(
        final_messages, state_body)
    assert final_messages.index(memory) < _index(final_messages, "HISTORY_NEW")
    assert _index(final_messages, "WORLD_BOOK_ALPHA") < _index(final_messages, "WORLD_BOOK_BETA")
    worldbooks_after = [
        (row.get("role"), row.get("content")) for row in final_messages
        if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}
    ]
    assert worldbooks_after == worldbooks_before
    assert _worldbook_boundary_signature(final_messages) == worldbook_position_before
    final_state = next(row for row in final_messages
                       if row.get("content") == state_body)
    assert final_state["content"] == state_body
    assert final_state["role"] == state_role
    final_wrapper = next(row for row in final_messages
                         if row.get("content") == wrapper_body)
    assert final_wrapper["role"] == wrapper_role
    assert all("_prompt_projection" not in row and "_guyan_event" not in row
               for row in final_messages)

    off_block = _block(
        "ombre.memory_recall", role="assistant", anchor="history.depth",
        order=20, history_depth=1, mode="off", enabled=False,
    )
    off_stable, off_dynamic, off_debug = service._build_composed_context_messages(
        _plan(off_block),
        **{**_context_values(), "core_memory": "", "date_recall": "",
           "just_now_context": "", "relationship_weather": "",
           "favorite_memory": ""},
        memory_recall_projection={"body": "OFF_MEMORY_MUST_NOT_RETURN"},
    )
    off_records = off_debug.pop("_placement_messages")
    assert off_records == []
    off_messages = service._inject_context_messages(
        messages,
        off_stable,
        off_dynamic,
        composed_messages=off_records,
        coverage=coverage,
    )
    off_text = "\n".join(str(row.get("content") or "") for row in off_messages)
    assert "OFF_MEMORY_MUST_NOT_RETURN" not in off_text
    assert _index(off_messages, "OWNER_EMOTION_WRAPPER_R17") < _index(
        off_messages, "CURRENT_STATE=")
    assert ([(row.get("role"), row.get("content")) for row in off_messages
             if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}]
            == worldbooks_before)
    assert _worldbook_boundary_signature(off_messages) == worldbook_position_before

    service.prepare_snapshot_ttl_seconds = 300
    service._restore_cached_reasoning_content = lambda *_args, **_kwargs: None
    service._apply_prompt_cache_hints = lambda *_args, **_kwargs: None
    tool_tail = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "crossstage-call-a", "type": "function"},
            {"id": "crossstage-call-b", "type": "function"},
        ]},
        {"role": "tool", "tool_call_id": "crossstage-call-a", "content": "RESULT_A"},
        {"role": "tool", "tool_call_id": "crossstage-call-b", "content": "RESULT_B"},
    ]
    identity = {
        "key_hash": "crossstage-placement-snapshot-r1",
        "snapshot_parent_key": "crossstage-placement-parent-r1",
        "source_phase": "initial",
        "transport_session_id": "local-no-provider",
        "baseline_messages": messages,
        "tool_tail": tool_tail,
    }
    service._prepare_snapshot_store(
        identity,
        prepared_payload=outgoing_model_input,
        base_forward_messages=messages,
        stable_context=stable,
        dynamic_context=dynamic,
        injected_ids=["memory-r1"],
        debug_payload=None,
        composed_messages=placements,
    )
    entry = service._prepare_snapshot_get(identity["key_hash"])
    assert entry["composed_messages"] == placements
    replayed, injected_ids = service._prepare_snapshot_replay(
        entry,
        identity,
        {
            "model": "local-no-provider",
            "stream": False,
            "messages": [*messages, *tool_tail],
            "_ombre_trace_context": {"coverage": coverage},
        },
        continuation_phase=True,
        started_at=time.perf_counter(),
        include_debug=False,
    )
    replay_messages = replayed["messages"]
    assert injected_ids == ["memory-r1"]
    assert _index(replay_messages, wrapper_body) < _index(
        replay_messages, "SELECTED_MEMORY_R1")
    assert _index(replay_messages, "SELECTED_MEMORY_R1") < _index(
        replay_messages, state_body)
    assert sum("SELECTED_MEMORY_R1" in str(row.get("content") or "")
               for row in replay_messages) == 1
    assert ([(row.get("role"), row.get("content")) for row in replay_messages
             if row.get("content") in {"WORLD_BOOK_ALPHA", "WORLD_BOOK_BETA"}]
            == worldbooks_before)
    assert _worldbook_boundary_signature(replay_messages) == worldbook_position_before
    replay_wrapper = next(row for row in replay_messages
                          if row.get("content") == wrapper_body)
    replay_state = next(row for row in replay_messages
                        if row.get("content") == state_body)
    assert replay_wrapper["role"] == wrapper_role
    assert replay_state["role"] == state_role
    call_index = next(i for i, row in enumerate(replay_messages)
                      if row.get("tool_calls"))
    assert replay_messages[call_index + 1:call_index + 3] == tool_tail[1:]


def test_legacy_sdk_coverage_items_still_position_history_depth():
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "HISTORY_OLD"},
        {"role": "assistant", "content": "HISTORY_NEW"},
        {"role": "user", "content": "CURRENT"},
    ]
    coverage = {"items": [
        {"message_index": 1, "turn_member": False},
        {"message_index": 2, "turn_member": False},
        {"message_index": 3, "turn_member": True},
    ]}
    output = __import__("prompt_placement").place_context_messages(
        messages,
        [{"role": "system", "anchor": "history.depth",
          "history_depth": 1, "content": "LEGACY_ITEMS_MARKER"}],
        coverage,
    )
    assert _index(output, "HISTORY_OLD") < _index(output, "LEGACY_ITEMS_MARKER")
    assert _index(output, "LEGACY_ITEMS_MARKER") < _index(output, "HISTORY_NEW")
