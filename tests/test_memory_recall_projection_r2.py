"""Final request-local Memory projection and Composer contracts for R2."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from gateway import (
    PROMPT_BINDING_REVISION_HEADER,
    PROMPT_PRESET_HEADER,
    PROMPT_REVISION_HEADER,
    PROMPT_SCOPE_HEADER,
    PROMPT_SHA_HEADER,
    GatewayService,
    memory_use,
)
from memory_authority import MemoryAuthorityStore
from memory_authority_view import MemoryAuthorityRecallView
from memory_commit_service import body_sha256
from prompt_plan_mirror import PromptPlanMirrorStore
from test_memory_recall_gateway_r2 import _base_config, _commit_l1, _gateway_service, _install_source_http
from utils import count_tokens_approx


FIXED_NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)


def _commit_legacy(authority, bucket_id: str, body: str) -> None:
    prepared = authority.prepare_memory_commit(
        memory_id=f"memory:{bucket_id}",
        bucket_id=bucket_id,
        expected_revision=0,
        body_sha256=body_sha256(body),
        snapshot_path=f"revisions/memory:{bucket_id}/1.md",
        metadata={"name": f"Synthetic legacy {bucket_id}"},
        source_refs=[f"synthetic:{bucket_id}"],
        decision_source="gateway-projection-test",
        idempotency_key=f"gateway-projection:{bucket_id}",
        actor="test",
    )
    authority.record_body_written(
        prepared["operation_id"], observed_body_sha256=body_sha256(body),
    )
    authority.finalize_memory_commit(prepared["operation_id"])


def _block(source_id: str, order: int, *, scope: str = "talk.initial",
           anchor: str = "gateway.current_user_prefix", role: str = "user",
           lane: str = "message", mode: str = "live_source",
           enabled: bool = True, token_budget: int | None = None,
           wrapper_text: str = "") -> dict:
    return {
        "block_id": f"block:{scope}:{source_id}",
        "source_id": source_id,
        "scope": scope,
        "stage": "ombre_post_injection",
        "mode": mode,
        "role": role,
        "lane": lane,
        "anchor": anchor,
        "order": order,
        "priority": order,
        "enabled": enabled,
        "owner_body": "",
        "frozen_body": "",
        "wrapper_text": wrapper_text,
        "token_budget": token_budget,
    }


def _make_plan(service: GatewayService, *, memory_mode: str = "live_source",
               memory_budget: int = 4000, usage_mode: str | None = None) -> dict:
    blocks = [
        _block("ombre.core_memory", 100, anchor="gateway.after_first_system",
               role="system", lane="instruction"),
        _block("ombre.date_recall", 290),
        _block(
            "ombre.memory_recall", 300, mode=memory_mode,
            enabled=memory_mode != "off", token_budget=memory_budget,
            wrapper_text="CANONICAL-MEMORY-BEGIN {{content}} CANONICAL-MEMORY-END",
        ),
        _block("ombre.recalled_memory", 310),
        _block("ombre.targeted_memory_detail", 320),
        _block("ombre.diffused_memory", 330),
        _block("ombre.recent_context", 340),
    ]
    if usage_mode is not None:
        blocks.append(_block(
            "ombre.memory_usage_card_prompt", 10,
            scope="memory.usage_card", anchor="instructions.end",
            role="system", lane="instruction", mode=usage_mode,
            enabled=usage_mode != "off",
        ))
    gateway_slice = {
        "schema": "guyan.gateway-prompt-slice.v1",
        "settings": {},
        "blocks": blocks,
    }
    store = PromptPlanMirrorStore(service.config["state_dir"] + "/prompt-plan.sqlite3")
    service.prompt_plan_mirror = store
    plan_sha = "a" * 64
    preset_id = "preset:memory-projection-r2"
    store.put_plan(preset_id, 1, plan_sha256=plan_sha, gateway_slice=gateway_slice)
    store.put_binding(
        "talk.initial", preset_id=preset_id, preset_revision=1,
        plan_sha256=plan_sha, aiz_binding_revision=1,
        identity_id="synthetic-session",
    )
    if usage_mode is not None:
        store.put_binding(
            "memory.usage_card", preset_id=preset_id, preset_revision=1,
            plan_sha256=plan_sha, aiz_binding_revision=1,
            identity_id="jiajia-main",
        )
    request = SimpleNamespace(headers={
        PROMPT_PRESET_HEADER: preset_id,
        PROMPT_REVISION_HEADER: "1",
        PROMPT_SHA_HEADER: plan_sha,
        PROMPT_BINDING_REVISION_HEADER: "1",
        PROMPT_SCOPE_HEADER: "talk.initial",
    })
    return service._prompt_plan_for_request(
        request, session_id="synthetic-session", continuation_phase=False,
    )


def _fixture(tmp_path, monkeypatch, *, l1_body: str | None = None,
             legacy_body: str = "Legacy archive narrative for the current request.",
             memory_mode: str = "live_source", memory_budget: int = 4000,
             usage_mode: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    l1_body = l1_body or (
        "source-record-l1: current synthetic address evidence. "
        "The complete original record remains available for narrative context."
    )
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})
    _commit_l1(authority, "bucket-l1", l1_body)
    _commit_legacy(authority, "bucket-legacy", legacy_body)
    buckets = [
        {
            "id": "bucket-l1", "content": l1_body,
            "metadata": {"name": "source-record-l1", "type": "source"},
        },
        {
            "id": "bucket-legacy", "content": legacy_body,
            "metadata": {"name": "legacy-record", "type": "dynamic"},
        },
    ]
    view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path), "memory_authority": {"enabled": True},
    })
    _install_source_http(monkeypatch)
    service, _calls = _gateway_service(
        tmp_path, monkeypatch, view, buckets, [], _base_config(),
    )
    service.identity = {"ai_name": "顾衍"}
    service.inject_total_budget = 6000
    # GatewayService exposes a request-local copy of the configured policy;
    # install the synthetic term extractor on its fixture-owned base instance.
    service._recall_policy.specific_query_terms = lambda query: [str(query)]

    async def capsule(_body, _metadata):
        return "Synthetic source-record capsule."

    service.dehydrator = SimpleNamespace(dehydrate_direct_capsule=capsule)
    service._bucket_metadata_for_dehydration = lambda _bucket: {}
    plan = _make_plan(
        service, memory_mode=memory_mode, memory_budget=memory_budget,
        usage_mode=usage_mode,
    )
    return authority, service, view, buckets, plan


def _request(view, bucket_ids: list[str]) -> dict:
    return {
        "q_current": "source-record-l1",
        "q_context": "source-record-l1",
        "metadata": {"query_views": []},
        "_use_context": {
            "now": FIXED_NOW, "target_time": None, "query": "source-record-l1",
            "purpose": "current", "overview": False,
            "facts": {"hour": {"state": "known", "value": 12},
                      "weekday": {"state": "known", "value": 5}},
        },
        "selected_buckets": [{"id": bucket_id} for bucket_id in bucket_ids],
        "index_metadata": view.memory_index_metadata_map(bucket_ids),
        "route": "ordinary",
    }


def _raw_bucket(service, bucket: dict) -> str:
    bucket_id = str(bucket["id"])
    moment = {
        "bucket_id": bucket_id,
        "moment_id": f"moment:{bucket_id}",
        "section": "body",
    }
    return f"{service._direct_bucket_header(bucket, moment)} bucket_original\n{bucket['content']}"


def _source_record_raw(service, bucket: dict) -> str:
    moment = service._source_record_synthetic_moment_for_bucket(
        bucket, "source-record-l1", selected_reason="synthetic_explicit_match",
    )
    assert moment is not None
    header = service._direct_bucket_header(bucket, moment)
    return asyncio.run(service._format_source_record_direct_bucket(
        bucket, moment, header, service._rendered_bucket_content(bucket), 4000,
    ))


def _generated_sections(service, request, buckets, *, section_budget: int = 4000):
    source_record = _source_record_raw(service, buckets[0])
    legacy = _raw_bucket(service, buckets[1])
    return asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {
            "direct": (source_record + "\n" + legacy,
                       ["bucket-l1", "bucket-legacy"], section_budget),
            "targeted": (source_record, ["bucket-l1"], section_budget),
            "diffused": (source_record, ["bucket-l1"], section_budget),
        },
    ))


def _projection(service, request, sections):
    return service._build_memory_recall_projection(
        recalled_memory=sections.get("direct", ""),
        targeted_memory_detail=sections.get("targeted", ""),
        related_memory=sections.get("diffused", ""),
        recalled_moments=[],
        targeted_memory_detail_debug={
            "accepted_ids": ["bucket-l1"] if sections.get("targeted") else [],
        },
        memory_sentinel_debug={"route": "ordinary", "route_reason_codes": []},
        natural_input=request,
    )


def _compose(service, plan, projection, sections, *, core="CORE-BYPASS",
             date="ORDER-A", recent="ORDER-C"):
    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        persona_block="", core_memory=core, portrait_memory="",
        just_now_context="", recent_context=recent,
        recalled_memory=sections.get("direct", ""),
        relationship_weather="", favorite_memory="",
        related_memory=sections.get("diffused", ""),
        targeted_memory_detail=sections.get("targeted", ""),
        dream_context="", active_reminders="",
        memory_detail_recall_instruction="", handoff_tool_hint="",
        context_mode="", date_persona_trace="", date_recall=date,
        memory_recall_projection=projection,
    )
    placement_messages = debug.pop("_placement_messages")
    messages = service._inject_context_messages(
        [{"role": "system", "content": "synthetic system"},
         {"role": "user", "content": "synthetic current question"}],
        stable, dynamic, composed_messages=placement_messages,
    )
    service._reconcile_memory_recall_injection(
        projection, stable, dynamic,
        enabled=service._composer_live_recall_enabled(plan),
    )
    return messages, stable, dynamic, debug


def _user_content(messages: list[dict]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


def test_l1_legacy_source_record_and_repeated_sections_reach_one_ordered_composer_block(
    tmp_path, monkeypatch,
):
    _authority, service, view, buckets, plan = _fixture(tmp_path, monkeypatch)
    request = _request(view, ["bucket-l1", "bucket-legacy"])
    sections = _generated_sections(service, request, buckets)

    # The real source-record direct formatter fed the shared generator. Targeted
    # wins; the same L1 bucket in direct/diffused is projected only once.
    assert "Synthetic source-record capsule." in sections["targeted"]
    assert "current_basis" in sections["targeted"]
    assert "historical_evidence" in sections["targeted"]
    assert "historical_evidence" in sections["direct"]
    assert sections["diffused"] == ""
    assert sections["targeted"].count("Historical narrative (record date is not event date):") == 1
    assert sections["direct"].count("Historical narrative (record date is not event date):") == 1
    assert "Legacy archive narrative" in sections["direct"]
    assert request["injected_usage_bucket_ids"] == ["bucket-l1", "bucket-legacy"]
    trace_by_bucket = {
        item["bucket_id"]: item for item in request["presentation_trace"]
    }
    assert trace_by_bucket["bucket-l1"]["source"] == "l1_targeted_original"
    assert trace_by_bucket["bucket-l1"]["reason"] == "explicit_targeted_original"
    assert trace_by_bucket["bucket-legacy"]["source"] == "legacy"
    assert trace_by_bucket["bucket-legacy"]["fallback_reason"] == "no_l1_authority"

    projection = _projection(service, request, sections)
    assert projection["selected_bucket_ids"] == ["bucket-l1", "bucket-legacy"]
    assert projection["recall_status"]["natural_selection_count"] == 2
    assert projection["recall_status"]["selected_count"] == 2
    assert projection["recall_status"]["injected_count"] == 2
    assert projection["recall_status"]["status"] == "selected"

    messages, stable, dynamic, debug = _compose(service, plan, projection, sections)
    user = _user_content(messages)
    assert user.index("ORDER-A") < user.index("Recall status: selected.") < user.index("ORDER-C")
    assert user.count("CANONICAL-MEMORY-BEGIN") == 1
    assert user.count("CANONICAL-MEMORY-END") == 1
    assert user.count("Recall status: selected.") == 1
    assert user.count("Legacy archive narrative") == 1
    assert "CORE-BYPASS" in "\n".join(message["content"] for message in messages)
    assert projection["recall_status"]["injection_measured_after_compile"] is True
    assert projection["recall_status"]["injected_count"] == 2
    resolved_sources = [row["source_id"] for row in debug["resolved_blocks"]]
    assert resolved_sources.count("ombre.memory_recall") == 1
    assert not ({"ombre.recalled_memory", "ombre.targeted_memory_detail",
                 "ombre.diffused_memory"} & set(resolved_sources))
    assert "ORDER-A" in dynamic and "ORDER-C" in dynamic and "Recall status: selected." in dynamic
    assert stable and "CORE-BYPASS" in stable


def test_section_and_composer_budgets_control_actual_injection_and_counts(
    tmp_path, monkeypatch,
):
    long_body = "预算边界合成记忆。" * 600
    _authority, service, view, buckets, plan = _fixture(
        tmp_path, monkeypatch, l1_body=long_body, memory_budget=1,
    )
    request = _request(view, ["bucket-l1"])
    raw = _raw_bucket(service, buckets[0])
    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, ["bucket-l1"], 180),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))
    assert 0 < count_tokens_approx(sections["direct"]) <= 180
    assert "预算边界合成记忆。" in sections["direct"]
    assert request["injected_usage_bucket_ids"] == ["bucket-l1"]

    projection = service._build_memory_recall_projection(
        recalled_memory=sections["direct"], targeted_memory_detail="",
        related_memory="", recalled_moments=[],
        targeted_memory_detail_debug={"accepted_ids": []},
        memory_sentinel_debug={"route": "ordinary", "route_reason_codes": []},
        natural_input=request,
    )
    assert projection["recall_status"]["selected_count"] == 1
    assert projection["recall_status"]["injected_count"] == 1

    messages, _stable, dynamic, debug = _compose(service, plan, projection, {
        "direct": sections["direct"], "targeted": "", "diffused": "",
    })
    assert "Legacy archive narrative" not in dynamic
    assert "预算边界合成记忆。" not in _user_content(messages)
    assert projection["recall_status"]["selected_count"] == 1
    assert projection["recall_status"]["injected_count"] == 0
    assert projection["recall_status"]["status"] == "incomplete"
    assert projection["recall_status"]["injection_measured_after_compile"] is True
    assert "ombre.memory_recall" not in [row["source_id"] for row in debug["resolved_blocks"]]


def test_composer_off_and_automatic_disabled_do_not_restore_raw_memory_sections(
    tmp_path, monkeypatch,
):
    _authority, service, view, buckets, off_plan = _fixture(
        tmp_path, monkeypatch, memory_mode="off",
    )
    request = _request(view, ["bucket-l1", "bucket-legacy"])
    sections = _generated_sections(service, request, buckets)
    projection = _projection(service, request, sections)
    assert projection["fact_body"]
    messages, _stable, _dynamic, _debug = _compose(service, off_plan, projection, sections)
    compiled = "\n".join(message["content"] for message in messages)
    assert "source-record-l1: current synthetic address evidence" not in compiled
    assert "Legacy archive narrative" not in compiled
    assert projection["recall_status"]["status"] == "disabled"
    assert projection["recall_status"]["selected_count"] == 2
    assert projection["recall_status"]["injected_count"] == 0

    _authority2, disabled_service, disabled_view, _buckets2, live_plan = _fixture(
        tmp_path / "disabled", monkeypatch,
    )
    disabled_service.automatic_recall_enabled = False
    disabled_request = _request(disabled_view, [])
    disabled_request["skip_reason"] = "automatic_recall_disabled"
    empty_sections = {"direct": "", "targeted": "", "diffused": ""}
    disabled_projection = _projection(disabled_service, disabled_request, empty_sections)
    messages, _stable, _dynamic, _debug = _compose(
        disabled_service, live_plan, disabled_projection, empty_sections,
        core="CORE-BYPASS", date="ORDER-A", recent="ORDER-C",
    )
    compiled = "\n".join(message["content"] for message in messages)
    assert "Recall status: disabled." not in compiled
    assert "source-record-l1: current synthetic address evidence" not in compiled
    assert "Legacy archive narrative" not in compiled
    assert disabled_projection["recall_status"]["status"] == "disabled"
    assert disabled_projection["recall_status"]["selected_count"] == 0
    assert disabled_projection["recall_status"]["injected_count"] == 0
    assert "CORE-BYPASS" in compiled and "ORDER-A" in compiled and "ORDER-C" in compiled
