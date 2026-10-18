"""Readable Memory projections stay separate from retrieval custody metadata."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

import gateway
from gateway import (
    PROMPT_BINDING_REVISION_HEADER,
    PROMPT_PRESET_HEADER,
    PROMPT_REVISION_HEADER,
    PROMPT_SCOPE_HEADER,
    PROMPT_SHA_HEADER,
    memory_use,
)
from prompt_plan_mirror import PromptPlanMirrorStore
from test_memory_recall_projection_r2 import (
    _compose,
    _fixture,
    _projection,
    _raw_bucket,
    _request,
    _user_content,
)
from test_memory_recall_gateway_r2 import _commit_l1


L2_BODY = "Readable owner memory: search distinguishes original from rendered text."
PRESENTATION_REVISION = 7


def _request_with_records(view, bucket_ids: list[str]) -> dict:
    request = _request(view, bucket_ids)
    request["_semantic_records"] = view.memory_l1_recall_map(bucket_ids)
    return request


def _resolved_l2_stub(monkeypatch, text: str = L2_BODY):
    calls = []

    async def resolve(_service, candidates):
        calls.append([
            {
                "bucket_id": item["bucket_id"],
                "memory_id": item["memory_id"],
                "revision": item["revision"],
            }
            for item in candidates
        ])
        presentations = {}
        trace = {}
        for item in candidates:
            bucket_id = str(item["bucket_id"])
            presentations[bucket_id] = {
                "presentation_revision": PRESENTATION_REVISION,
                "text": text,
                "units": [{"unit_id": "owner-readable-unit", "text": text}],
            }
            trace[bucket_id] = {
                "bucket_id": bucket_id,
                "memory_id": item["memory_id"],
                "l1_revision": item["revision"],
                "presentation_revision": PRESENTATION_REVISION,
                "source": "l2",
                "resolver_state": "ready",
                "fallback_reason": None,
            }
        return presentations, trace

    monkeypatch.setattr(memory_use, "resolve_presentations", resolve)
    return calls


def _recent_carrier(service, bucket: dict, monkeypatch) -> str:
    """Exercise the recent selector without dehydration or model work."""
    bucket.setdefault("metadata", {})["created"] = datetime.now().isoformat()
    service.head_recent_hours = 48
    service._auto_query_too_vague = lambda _query: False
    service._recent_context_requires_topic_evidence = lambda _query: False
    service._query_wants_body_chain = lambda _query: False
    service._query_requests_recent_context = lambda _query: True
    service._is_self_anchor_recall_excluded_bucket = lambda _bucket: False
    service._parse_iso = lambda _value: datetime.now()
    service._summarize_buckets = lambda *_args: pytest.fail(
        "recent Memory must use the shared readable projection, not dehydration"
    )
    service.dehydrator = SimpleNamespace(
        dehydrate_direct_capsule=lambda *_args, **_kwargs: pytest.fail(
            "recent Memory must not dehydrate a second copy"
        )
    )
    monkeypatch.setattr(
        gateway, "can_bucket_be_recent_context",
        lambda *_args, **_kwargs: True,
    )
    return asyncio.run(service._build_recent_context_block(
        [bucket], "recent memory query", allow_vague=True,
    ))


def _composer_block(
    source_id: str,
    order: int,
    *,
    scope: str = "talk.initial",
    anchor: str = "gateway.current_user_prefix",
    role: str = "user",
    lane: str = "message",
    mode: str = "live_source",
    enabled: bool = True,
    wrapper_text: str = "",
) -> dict:
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
        "token_budget": None,
    }


def _composer_plan(
    service,
    tmp_path,
    *,
    memory_mode: str = "live_source",
    recent_mode: str = "live_source",
    usage_card_mode: str = "owner_override",
) -> dict:
    plan_sha = "d" * 64
    preset_id = "preset:readable-memory-contract"
    blocks = [
        _composer_block(
            "ombre.core_memory", 100, anchor="gateway.after_first_system",
            role="system", lane="instruction",
        ),
        _composer_block("ombre.date_recall", 290),
        _composer_block(
            "ombre.memory_recall", 300, mode=memory_mode,
            enabled=memory_mode != "off",
            wrapper_text="OWNER_RECALL[{{content}}]",
        ),
        _composer_block(
            "ombre.recent_context", 340, mode=recent_mode,
            enabled=recent_mode != "off",
        ),
    ]
    usage_card = _composer_block(
        "ombre.memory_usage_card_prompt", 10,
        scope="memory.usage_card", anchor="instructions.end",
        role="system", lane="instruction", mode=usage_card_mode,
        enabled=usage_card_mode != "off",
    )
    if usage_card_mode == "owner_override":
        usage_card["owner_body"] = "OWNER_CARD[{{narrative}}]"
    blocks.append(usage_card)

    store = PromptPlanMirrorStore(tmp_path / "readable-memory-prompt-plan.sqlite3")
    service.prompt_plan_mirror = store
    store.put_plan(
        preset_id, 1, plan_sha256=plan_sha,
        gateway_slice={
            "schema": "guyan.gateway-prompt-slice.v1",
            "settings": {},
            "blocks": blocks,
        },
    )
    store.put_binding(
        "talk.initial", preset_id=preset_id, preset_revision=1,
        plan_sha256=plan_sha, aiz_binding_revision=1,
        identity_id="synthetic-session",
    )
    persona = getattr(service, "persona_engine", None)
    owner_identity = getattr(persona, "canonical_session_id", "") or "jiajia-main"
    store.put_binding(
        "memory.usage_card", preset_id=preset_id, preset_revision=1,
        plan_sha256=plan_sha, aiz_binding_revision=1,
        identity_id=owner_identity,
    )
    request = SimpleNamespace(headers={
        PROMPT_PRESET_HEADER: preset_id,
        PROMPT_REVISION_HEADER: "1",
        PROMPT_SHA_HEADER: plan_sha,
        PROMPT_BINDING_REVISION_HEADER: "1",
        PROMPT_SCOPE_HEADER: "talk.initial",
    })
    plan = service._prompt_plan_for_request(
        request, session_id="synthetic-session", continuation_phase=False,
    )
    assert plan is not None and not plan.get("fallback_legacy")
    return plan


def _project(service, request: dict, buckets: list[dict], sections: dict) -> dict:
    return asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets, sections,
    ))


def _assert_no_runtime_labels(text: str) -> None:
    lowered = text.casefold()
    for marker in (
        "bucket_id:", "bucket-l1", "moment_id", "moment:", "score:",
        "source:", "l1_revision", "presentation_revision",
        "allowed memory uses:", "recall status:", "historical narrative",
        "[created:",
    ):
        assert marker.casefold() not in lowered


def _compose_with_history(
    service, plan, projection, sections, history_body: str, *, date: str = "", recent: str = "",
):
    stable, dynamic, debug = service._build_composed_context_messages(
        plan,
        persona_block="", core_memory="", portrait_memory="",
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
        [
            {"role": "system", "content": "synthetic system"},
            {"role": "assistant", "content": history_body},
            {"role": "user", "content": "synthetic current question"},
        ],
        stable, dynamic, composed_messages=placement_messages,
    )
    service._reconcile_memory_recall_injection(
        projection, stable, dynamic,
        enabled=service._composer_live_recall_enabled(plan),
    )
    return messages, stable, dynamic, debug


def test_direct_and_recent_same_source_resolve_once_and_compile_readable_l2(
    tmp_path, monkeypatch,
):
    _authority, service, view, buckets, _old_plan = _fixture(tmp_path, monkeypatch)
    plan = _composer_plan(service, tmp_path)
    bucket = buckets[0]
    request = _request_with_records(view, ["bucket-l1"])
    recent_carrier = _recent_carrier(service, bucket, monkeypatch)
    direct_carrier = _raw_bucket(service, bucket)
    calls = _resolved_l2_stub(monkeypatch)

    rendered = _project(service, request, [bucket], {
        "targeted": ("", [], 0),
        "direct": (direct_carrier, ["bucket-l1"], 4000),
        "diffused": ("", [], 0),
        "recent": (recent_carrier, ["bucket-l1"], 4000),
    })

    assert len(calls) == 1
    assert [item["bucket_id"] for item in calls[0]] == ["bucket-l1"]
    assert rendered["direct"].count(L2_BODY) == 1
    assert rendered["recent"] == ""
    assert request["requested_usage_bucket_ids"] == ["bucket-l1"]
    assert request["injected_usage_bucket_ids"] == ["bucket-l1"]
    spans = request["rendered_memory_spans"]
    assert len(spans) == 1
    assert spans[0] == {
        "bucket_id": "bucket-l1",
        "section": "direct",
        "text": rendered["direct"],
    }
    trace = next(row for row in request["presentation_trace"]
                 if row["bucket_id"] == "bucket-l1")
    assert trace["section"] == "direct"
    assert trace["source"] == "l2"
    assert trace["presentation_revision"] == PRESENTATION_REVISION
    assert trace["l1_revision"] == 1
    assert trace["memory_id"] == request["_semantic_records"]["bucket-l1"]["memory_id"]
    _assert_no_runtime_labels(rendered["direct"])

    projection = _projection(service, request, rendered)
    messages, _stable, _dynamic, _debug = _compose(
        service, plan, projection, rendered,
        core="CORE_STABLE", date="ORDER_BEFORE", recent="ORDER_AFTER",
    )
    user = _user_content(messages)
    assert user.count(L2_BODY) == 1
    assert "OWNER_CARD[" in user
    assert "OWNER_RECALL[" in user
    assert user.index("ORDER_BEFORE") < user.index("OWNER_RECALL[")
    assert user.index("OWNER_RECALL[") < user.index(L2_BODY) < user.index("ORDER_AFTER")
    assert projection["recall_status"]["injected_count"] == 1
    assert projection["injected_bucket_ids"] == ["bucket-l1"]
    _assert_no_runtime_labels(user)


def test_recent_only_uses_shared_l2_renderer_and_compiled_span(tmp_path, monkeypatch):
    _authority, service, view, buckets, _old_plan = _fixture(tmp_path, monkeypatch)
    plan = _composer_plan(service, tmp_path)
    bucket = buckets[0]
    request = _request_with_records(view, ["bucket-l1"])
    recent_carrier = _recent_carrier(service, bucket, monkeypatch)
    calls = _resolved_l2_stub(monkeypatch)

    rendered = _project(service, request, [bucket], {
        "targeted": ("", [], 0),
        "direct": ("", [], 0),
        "diffused": ("", [], 0),
        "recent": (recent_carrier, ["bucket-l1"], 4000),
    })

    assert len(calls) == 1
    assert rendered["direct"] == ""
    assert L2_BODY in rendered["recent"]
    assert "bucket-l1" not in rendered["recent"]
    assert request["rendered_memory_spans"][0]["section"] == "recent"
    trace = next(row for row in request["presentation_trace"]
                 if row["bucket_id"] == "bucket-l1")
    assert trace["source"] == "l2"
    assert trace["presentation_revision"] == PRESENTATION_REVISION
    _assert_no_runtime_labels(rendered["recent"])

    projection = _projection(service, request, rendered)
    recent_context = rendered["recent"] + "\n\nRECENT_ORDER_AFTER"
    messages, _stable, _dynamic, _debug = _compose(
        service, plan, projection, rendered,
        core="CORE_STABLE", date="DATE_ORDER_BEFORE", recent=recent_context,
    )
    user = _user_content(messages)
    assert user.count(L2_BODY) == 1
    assert user.index("DATE_ORDER_BEFORE") < user.index(L2_BODY)
    assert user.index(L2_BODY) < user.index("RECENT_ORDER_AFTER")
    assert "OWNER_CARD[" in user
    assert projection["recall_status"]["injected_count"] == 1
    assert projection["injected_bucket_ids"] == ["bucket-l1"]
    _assert_no_runtime_labels(user)


def test_equal_l2_text_in_direct_and_recent_counts_only_compiled_direct_source(
    tmp_path, monkeypatch,
):
    authority, service, view, buckets, _old_plan = _fixture(tmp_path, monkeypatch)
    direct_bucket = buckets[0]
    recent_body = direct_bucket["content"]
    _commit_l1(authority, "bucket-l1-recent", recent_body)
    recent_bucket = {
        "id": "bucket-l1-recent", "content": recent_body,
        "metadata": {"name": "source-record-recent", "type": "source"},
    }
    all_buckets = [direct_bucket, recent_bucket]
    request = _request_with_records(view, ["bucket-l1", "bucket-l1-recent"])
    direct_carrier = _raw_bucket(service, direct_bucket)
    recent_carrier = _recent_carrier(service, recent_bucket, monkeypatch)
    calls = _resolved_l2_stub(monkeypatch)
    plan = _composer_plan(service, tmp_path, recent_mode="off")

    rendered = _project(service, request, all_buckets, {
        "targeted": ("", [], 0),
        "direct": (direct_carrier, ["bucket-l1"], 4000),
        "diffused": ("", [], 0),
        "recent": (recent_carrier, ["bucket-l1-recent"], 4000),
    })

    assert len(calls) == 1
    assert {item["bucket_id"] for item in calls[0]} == {
        "bucket-l1", "bucket-l1-recent",
    }
    assert rendered["direct"].count(L2_BODY) == 1
    assert rendered["recent"].count(L2_BODY) == 1
    spans = request["rendered_memory_spans"]
    assert [(span["bucket_id"], span["section"]) for span in spans] == [
        ("bucket-l1", "direct"), ("bucket-l1-recent", "recent"),
    ]

    projection = _projection(service, request, rendered)
    messages, _stable, _dynamic, _debug = _compose(
        service, plan, projection, rendered, recent=rendered["recent"],
    )
    model_text = _user_content(messages)
    blocks = projection["compiled_memory_blocks"]
    assert any(block["source_id"] == "ombre.memory_recall"
               and L2_BODY in block["content"] for block in blocks)
    assert not any(block["source_id"] == "ombre.recent_context" for block in blocks)
    assert model_text.count(L2_BODY) == 1
    assert projection["recall_status"]["injected_count"] == 1
    assert projection["injected_bucket_ids"] == ["bucket-l1"]
    _assert_no_runtime_labels(model_text)


def test_source_record_moment_carrier_skips_capsule_and_reaches_shared_l2(
    tmp_path, monkeypatch,
):
    _authority, service, view, buckets, _old_plan = _fixture(tmp_path, monkeypatch)
    bucket = buckets[0]
    request = _request_with_records(view, ["bucket-l1"])
    plan = _composer_plan(service, tmp_path)
    service._recall_policy.has_strong_score = lambda **_kwargs: False
    service.dehydrator = SimpleNamespace(
        dehydrate_direct_capsule=lambda *_args, **_kwargs: pytest.fail(
            "source-record candidate carrier must not call the legacy capsule provider"
        ),
    )
    moment = service._source_record_synthetic_moment_for_bucket(
        bucket, "source-record-l1", selected_reason="synthetic_explicit_match",
    )
    assert moment is not None
    carrier = asyncio.run(service._format_recalled_moments(
        [moment], {"bucket-l1": [moment]}, [bucket], 4000,
        query_text="source-record-l1", context_mode="",
    ))
    assert carrier == "[bucket_id:bucket-l1]"
    assert bucket["content"] not in carrier
    assert "moment_id" not in carrier

    calls = _resolved_l2_stub(monkeypatch)
    rendered = _project(service, request, [bucket], {
        "targeted": ("", [], 0),
        "direct": (carrier, ["bucket-l1"], 4000),
        "diffused": ("", [], 0),
        "recent": ("", [], 0),
    })

    assert len(calls) == 1
    assert calls[0][0]["bucket_id"] == "bucket-l1"
    assert rendered["direct"].count(L2_BODY) == 1
    assert request["rendered_memory_spans"][0]["section"] == "direct"
    assert request["presentation_trace"][0]["source"] == "l2"
    projection = _projection(service, request, rendered)
    messages, _stable, _dynamic, _debug = _compose(
        service, plan, projection, rendered,
    )
    model_text = _user_content(messages)
    assert model_text.count(L2_BODY) == 1
    _assert_no_runtime_labels(model_text)


@pytest.mark.parametrize(
    ("legacy_body", "expected_body"),
    [
        ("Natural legacy paragraph remains readable.",
         "Natural legacy paragraph remains readable."),
        (json.dumps({"narrative": {"body": "NARRATIVE_BODY_CANARY"},
                     "keywords": ["KEYWORD_CANARY"],
                     "corefacts": ["COFACT_CANARY"],
                     "todos": ["TODO_CANARY"]}, ensure_ascii=False),
         "NARRATIVE_BODY_CANARY"),
        (json.dumps({"full_injection_text": "FULL_TEXT_CANARY",
                     "keywords": ["KEYWORD_CANARY"]}, ensure_ascii=False),
         "FULL_TEXT_CANARY"),
        (json.dumps({"summary": "OLD_JSON_SUMMARY_CANARY",
                     "keywords": ["KEYWORD_CANARY"],
                     "corefacts": ["COFACT_CANARY"],
                     "todos": ["TODO_CANARY"]}, ensure_ascii=False),
         ""),
        ('{"narrative":{"body":"TRUNCATED_JSON_CANARY"', ""),
    ],
    ids=["natural-legacy", "narrative-body", "full-text", "summary-only", "partial-json"],
)
def test_legacy_natural_prose_survives_but_json_metadata_is_not_projected(
    tmp_path, monkeypatch, legacy_body: str, expected_body: str,
):
    _authority, service, view, buckets, _plan = _fixture(
        tmp_path, monkeypatch, legacy_body=legacy_body,
    )
    legacy_bucket = buckets[1]
    request = _request_with_records(view, ["bucket-legacy"])

    async def no_l2_for_legacy(_service, candidates):
        assert candidates == []
        return {}, {}

    monkeypatch.setattr(memory_use, "resolve_presentations", no_l2_for_legacy)
    rendered = _project(service, request, [legacy_bucket], {
        "targeted": ("", [], 0),
        "direct": (_raw_bucket(service, legacy_bucket), ["bucket-legacy"], 4000),
        "diffused": ("", [], 0),
        "recent": ("", [], 0),
    })
    model_text = rendered["direct"]

    if expected_body:
        assert expected_body in model_text
        assert request["injected_usage_bucket_ids"] == ["bucket-legacy"]
        _assert_no_runtime_labels(model_text)
    else:
        assert model_text == ""
        assert request["injected_usage_bucket_ids"] == []
    if legacy_body != expected_body:
        assert legacy_body not in model_text
    for marker in (
        "OLD_JSON_SUMMARY_CANARY", "KEYWORD_CANARY", "COFACT_CANARY",
        "TODO_CANARY", "TRUNCATED_JSON_CANARY",
    ):
        assert marker not in model_text


@pytest.mark.parametrize(
    ("memory_mode", "total_budget", "expected_status"),
    [
        ("off", 6000, "disabled"),
        ("live_source", 0, "incomplete"),
    ],
    ids=["composer-off", "compiled-out-by-budget"],
)
def test_off_or_budget_discard_does_not_count_matching_history_as_injection(
    tmp_path, monkeypatch, memory_mode: str, total_budget: int, expected_status: str,
):
    _authority, service, view, buckets, _old_plan = _fixture(tmp_path, monkeypatch)
    bucket = buckets[0]
    request = _request_with_records(view, ["bucket-l1"])
    calls = _resolved_l2_stub(monkeypatch)
    rendered = _project(service, request, [bucket], {
        "targeted": ("", [], 0),
        "direct": (_raw_bucket(service, bucket), ["bucket-l1"], 4000),
        "diffused": ("", [], 0),
        "recent": ("", [], 0),
    })
    assert len(calls) == 1
    assert request["rendered_memory_spans"]

    service.inject_total_budget = total_budget
    plan = _composer_plan(service, tmp_path, memory_mode=memory_mode)
    projection = _projection(service, request, rendered)
    messages, stable, dynamic, _debug = _compose_with_history(
        service, plan, projection, rendered, L2_BODY,
    )
    compiled = "\n".join(str(row.get("content") or "") for row in messages)

    # The old conversation still contains this prose, but reconciliation only
    # measures newly compiled stable/dynamic context and therefore counts zero.
    assert L2_BODY in compiled
    assert projection["recall_status"]["injected_count"] == 0
    assert projection["injected_bucket_ids"] == []
    assert projection["recall_status"]["status"] == expected_status
    assert L2_BODY not in "\n\n".join(part for part in (stable, dynamic) if part)
    assert "OWNER_CARD[" not in "\n\n".join(part for part in (stable, dynamic) if part)
    assert request["presentation_trace"][0]["presentation_revision"] == PRESENTATION_REVISION
