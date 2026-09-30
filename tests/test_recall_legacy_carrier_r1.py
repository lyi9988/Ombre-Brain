"""Focused compatibility checks for pre-unified Recall Composer plans."""

from __future__ import annotations

from copy import deepcopy

from gateway import GatewayService


LEGACY_MEMORY_SOURCES = (
    "ombre.recalled_memory",
    "ombre.targeted_memory_detail",
    "ombre.diffused_memory",
)


def _legacy_plan(*, modes=None):
    modes = modes or {}
    blocks = []
    for order, source_id in enumerate(LEGACY_MEMORY_SOURCES, start=100):
        blocks.append({
            "block_id": f"block:{source_id}",
            "source_id": source_id,
            "scope": "talk.initial",
            "stage": "ombre_post_injection",
            "mode": modes.get(source_id, "live_source"),
            "role": "user",
            "lane": "message",
            "anchor": "gateway.current_user_prefix",
            "order": order,
            "priority": order,
            "enabled": True,
            "owner_body": "",
            "frozen_body": "",
            "wrapper_text": "",
            "token_budget": None,
        })
    return {
        "scope": "talk.initial",
        "gateway_slice": {
            "settings": {"scope_inheritance": {}},
            "blocks": blocks,
        },
        "binding": {"aiz_binding_revision": 1},
    }


def _service_and_context_args(recalled_memory="", *, related_memory="", targeted=""):
    service = GatewayService.__new__(GatewayService)
    service.identity = {"ai_name": "Synthetic companion"}
    service.inject_total_budget = 10000
    context_args = {
        "persona_block": "",
        "core_memory": "",
        "portrait_memory": "",
        "just_now_context": "",
        "recent_context": "",
        "recalled_memory": recalled_memory,
        "relationship_weather": "",
        "favorite_memory": "",
        "related_memory": related_memory,
        "targeted_memory_detail": targeted,
        "dream_context": "",
        "active_reminders": "",
        "memory_detail_recall_instruction": "",
        "handoff_tool_hint": "",
        "context_mode": "",
        "date_persona_trace": "",
        "date_recall": "",
    }
    return service, context_args


def _compile(service, plan, context_args, *, status):
    service._project_legacy_recall_status(plan, context_args, status)
    projection = {
        "body": "",
        "fact_body": "",
        "selected_bucket_ids": [],
        "recall_status": status,
    }
    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        **context_args,
        memory_recall_projection=projection,
    )
    enabled = service._composer_live_recall_enabled(plan)
    service._reconcile_memory_recall_injection(
        projection, stable, dynamic, enabled=enabled,
    )
    return stable, dynamic, debug, projection


def test_legacy_empty_no_match_status_reaches_actual_carrier_without_injecting_facts():
    service, context_args = _service_and_context_args()
    plan = _legacy_plan()
    status = {"status": "no_match", "selected_count": 0}

    stable, dynamic, debug, projection = _compile(
        service, plan, context_args, status=status,
    )

    actual_context = "\n\n".join(part for part in (stable, dynamic) if part)
    recalled_block = next(
        row for row in debug["resolved_blocks"]
        if row["source_id"] == "ombre.recalled_memory"
    )
    assert "Recall status: no_match." in actual_context
    assert "Status text by itself is not a selected Memory item." in actual_context
    assert recalled_block["role"] == "user"
    assert recalled_block["order"] == 100
    assert projection["injected_bucket_ids"] == []
    assert projection["recall_status"]["injected_count"] == 0
    assert projection["recall_status"]["status_only"] is True


def test_legacy_empty_incomplete_status_is_not_rendered_as_no_match():
    service, context_args = _service_and_context_args()
    status = {
        "status": "incomplete",
        "selected_count": 0,
        "incomplete_reasons": ["synthetic_index_unavailable"],
    }

    stable, dynamic, _debug, projection = _compile(
        service, _legacy_plan(), context_args, status=status,
    )

    actual_context = "\n\n".join(part for part in (stable, dynamic) if part)
    assert "Recall status: incomplete." in actual_context
    assert "do not treat it as `no_match`" in actual_context
    assert "Recall status: no_match." not in actual_context
    assert projection["recall_status"]["injected_count"] == 0
    assert projection["recall_status"]["status_only"] is True


def test_configured_canonical_source_is_used_once_without_legacy_wrapper():
    service, context_args = _service_and_context_args(
        "legacy-direct-marker",
        related_memory="legacy-diffused-marker",
        targeted="legacy-targeted-marker",
    )
    plan = _legacy_plan()
    plan["gateway_slice"]["blocks"].append({
        "block_id": "block:ombre.memory_recall",
        "source_id": "ombre.memory_recall",
        "scope": "talk.initial",
        "stage": "ombre_post_injection",
        "mode": "live_source",
        "role": "user",
        "lane": "message",
        "anchor": "gateway.current_user_prefix",
        "order": 295,
        "priority": 295,
        "enabled": True,
        "owner_body": "",
        "frozen_body": "",
        "wrapper_text": "",
        "token_budget": None,
    })
    status = {"status": "no_match", "selected_count": 0}
    context_before_projection = deepcopy(context_args)
    service._project_legacy_recall_status(plan, context_args, status)
    canonical_body = service._resolve_gateway_fixed_prompt(
        "ombre.memory_recall_status_wrapper_prompt",
        runtime_values={"status": "no_match", "content": ""},
    )
    projection = {
        "body": canonical_body,
        "fact_body": "",
        "selected_bucket_ids": [],
        "recall_status": status,
    }

    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        **context_args,
        memory_recall_projection=projection,
    )
    actual_context = "\n\n".join(part for part in (stable, dynamic) if part)

    assert context_args == context_before_projection
    assert "legacy_compatibility_projection" not in status
    assert actual_context.count("Recall status: no_match.") == 1
    assert "legacy-direct-marker" not in actual_context
    assert "legacy-diffused-marker" not in actual_context
    assert "legacy-targeted-marker" not in actual_context
    resolved_sources = [row["source_id"] for row in debug["resolved_blocks"]]
    assert resolved_sources.count("ombre.memory_recall") == 1
    assert not set(LEGACY_MEMORY_SOURCES) & set(resolved_sources)


def test_disabled_status_and_all_off_legacy_blocks_do_not_bypass_composer():
    off_modes = {source_id: "off" for source_id in LEGACY_MEMORY_SOURCES}
    service, context_args = _service_and_context_args(
        "legacy-direct-marker",
        related_memory="legacy-diffused-marker",
        targeted="legacy-targeted-marker",
    )
    status = {"status": "disabled", "selected_count": 0}
    plan = _legacy_plan(modes=off_modes)

    stable, dynamic, debug, projection = _compile(
        service, plan, context_args, status=status,
    )
    actual_context = "\n\n".join(part for part in (stable, dynamic) if part)

    assert service._composer_live_recall_enabled(plan) is False
    assert "Recall status: disabled." not in actual_context
    assert "legacy-direct-marker" not in actual_context
    assert "legacy-diffused-marker" not in actual_context
    assert "legacy-targeted-marker" not in actual_context
    assert not debug["resolved_blocks"]
    assert projection["recall_status"]["status"] == "disabled"
    assert projection["recall_status"]["injected_count"] == 0


def test_legacy_projection_and_composer_leave_plan_unchanged():
    service, context_args = _service_and_context_args()
    plan = _legacy_plan()
    before = deepcopy(plan)

    _compile(
        service, plan, context_args,
        status={"status": "no_match", "selected_count": 0},
    )

    assert plan == before
