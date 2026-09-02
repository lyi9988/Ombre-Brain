import asyncio
import hashlib
import json

import pytest

from gateway import GatewayService
from prompt_plan_mirror import (
    PromptPlanMirrorStore,
    PromptPlanMirrorValidationError,
)
from prompt_source_registry import (
    FIXED_PROMPT_SOURCES,
    RUNTIME_MECHANIC_SOURCES,
    DYNAMIC_CONTEXT_SOURCES,
    resolve_fixed_prompt,
    source_detail,
)


EXPECTED_FIXED_SOURCE_IDS = {
    "ombre.persona_post_reply_prompt",
    "ombre.memory_query_planner_prompt",
    "ombre.semantic_rescue_prompt",
    "ombre.memory_classify_prompt",
    "ombre.reflection_prompt",
    "ombre.diary_memory_prompt",
    "ombre.daily_chat_memory_prompt",
    "ombre.daily_chat_summary_prompt",
    "ombre.daily_activity_summary_prompt",
    "ombre.domain_sentinel_prompt",
    "ombre.dehydrate_prompt",
    "ombre.direct_capsule_prompt",
    "ombre.memory_merge_prompt",
    "ombre.memory_analyze_prompt",
    "ombre.memory_digest_prompt",
    "ombre.memory_moment_prompt",
    "ombre.dream_generation_prompt",
    "ombre.dream_wrapper_prompt",
    "ombre.portrait_patch_prompt",
    "ombre.portrait_stable_prompt",
    "ombre.profile_fact_proposal_prompt",
    "ombre.anchor_proposal_prompt",
    "ombre.gateway_stable_preface_prompt",
    "ombre.gateway_dynamic_preface_prompt",
    "ombre.memory_reading_policy_prompt",
    "ombre.date_boundary_prompt",
    "ombre.memory_detail_request_prompt",
    "ombre.handoff_hint_prompt",
    "ombre.handoff_date_hint_prompt",
    "ombre.active_reminder_prompt",
    "ombre.import_extract_prompt",
    "ombre.reclassify_prompt",
}


def test_registry_contains_every_included_fixed_source():
    assert set(FIXED_PROMPT_SOURCES) == EXPECTED_FIXED_SOURCE_IDS
    assert not EXPECTED_FIXED_SOURCE_IDS & DYNAMIC_CONTEXT_SOURCES
    assert not EXPECTED_FIXED_SOURCE_IDS & RUNTIME_MECHANIC_SOURCES


def test_fixed_detail_has_factory_and_live_body_only():
    for source_id in EXPECTED_FIXED_SOURCE_IDS:
        detail = source_detail(source_id, {})
        assert set(detail) == {
            "source_id",
            "body_kind",
            "factory_body",
            "live_body",
            "source_revision",
            "source_sha256",
            "authority",
        }
        assert detail["body_kind"] == "fixed_prompt"
        assert detail["factory_body"]
        assert detail["live_body"]
        assert detail["source_revision"]
        assert detail["authority"]
        assert detail["source_sha256"] == hashlib.sha256(
            detail["factory_body"].encode("utf-8")
        ).hexdigest()


@pytest.mark.parametrize(
    "source_id,body_kind",
    [
        ("ombre.core_memory", "dynamic_context"),
        ("ombre.active_reminders", "dynamic_context"),
        ("ombre.dream_context", "dynamic_context"),
        ("ombre.runtime.phase_marker", "runtime_mechanics"),
        ("ombre.not_registered", "unknown"),
    ],
)
def test_non_fixed_detail_never_returns_body(source_id, body_kind):
    detail = source_detail(source_id, {})
    assert detail["body_kind"] == body_kind
    assert set(detail) == {"source_id", "body_kind", "reason"}
    assert detail["reason"]
    assert "body" not in detail
    assert "sha256" not in json.dumps(detail, ensure_ascii=False)


def test_owner_authored_gateway_block_is_fixed_without_factory_body():
    detail = source_detail("custom.gateway_block", {})
    assert detail == {
        "source_id": "custom.gateway_block",
        "body_kind": "fixed_prompt",
        "reason": "owner-authored prompt has no factory body",
        "authority": "aizizhu.prompt_composer",
    }


def test_owner_authored_gateway_block_can_be_mirrored(tmp_path):
    body = {
        "blocks": [{
            "block_id": "block:custom",
            "source_id": "custom.gateway_block",
            "scope": "talk.initial",
            "stage": "ombre_post_injection",
            "mode": "owner_override",
            "role": "system",
            "lane": "instruction",
            "anchor": "gateway.after_first_system",
            "order": 1,
            "owner_body": "主人自定义的 Gateway 固定提示词",
        }],
    }
    result = PromptPlanMirrorStore(tmp_path / "custom.sqlite3").put_plan(
        "preset:test", 1, plan_sha256="c" * 64,
        gateway_slice=body,
    )
    assert result["status"] == "verified"


class _Request:
    def __init__(self, *, token="secret", source_id="ombre.dehydrate_prompt"):
        self.path_params = {"source_id": source_id}
        self.query_params = {}
        self.headers = {"Authorization": f"Bearer {token}"}


def _service():
    service = object.__new__(GatewayService)
    service.gateway_token = "secret"
    return service


def _response_body(response):
    return json.loads(response.body.decode("utf-8"))


def test_source_detail_endpoint_is_bearer_authenticated_and_no_store():
    service = _service()
    denied = asyncio.run(service.handle_prompt_source_detail(_Request(token="wrong")))
    assert denied.status_code == 401
    assert denied.headers["cache-control"] == "no-store"

    fixed = asyncio.run(service.handle_prompt_source_detail(_Request()))
    assert fixed.status_code == 200
    assert fixed.headers["cache-control"] == "no-store"
    assert set(_response_body(fixed)) == {
        "source_id",
        "body_kind",
        "factory_body",
        "live_body",
        "source_revision",
        "source_sha256",
        "authority",
    }

    dynamic = asyncio.run(service.handle_prompt_source_detail(
        _Request(source_id="ombre.core_memory")))
    assert dynamic.status_code == 200
    assert _response_body(dynamic)["body_kind"] == "dynamic_context"
    assert "factory_body" not in _response_body(dynamic)

    unknown = asyncio.run(service.handle_prompt_source_detail(
        _Request(source_id="ombre.not_registered")))
    assert unknown.status_code == 404
    assert _response_body(unknown)["body_kind"] == "unknown"
    assert "factory_body" not in _response_body(unknown)


def test_every_fixed_source_resolves_through_store(tmp_path):
    store = PromptPlanMirrorStore(tmp_path / "mirror.sqlite3")
    runtime_values = {
        "max_candidates": 4,
        "max_ids": 3,
        "content": "live runtime evidence",
    }
    for source_id, spec in FIXED_PROMPT_SOURCES.items():
        resolved, meta = resolve_fixed_prompt(
            store,
            source_id,
            config={"buckets_dir": str(tmp_path)},
            identity_id="jiajia-main",
            runtime_values=runtime_values,
        )
        assert resolved
        assert meta["status"] == "legacy_default"
        assert "{max_candidates}" not in resolved
        assert "{{max_ids}}" not in resolved
        assert "{{content}}" not in resolved


@pytest.mark.parametrize(
    "source_id,field",
    [
        ("ombre.dream_wrapper_prompt", "owner_body"),
        ("ombre.active_reminder_prompt", "owner_body"),
        ("ombre.memory_detail_request_prompt", "owner_body"),
    ],
)
def test_fixed_override_must_preserve_runtime_placeholders(
    tmp_path, source_id, field
):
    spec = FIXED_PROMPT_SOURCES[source_id]
    body = {
        "blocks": [{
            "block_id": "block:fixed",
            "source_id": source_id,
            "scope": spec.scope,
            "stage": "ombre_post_injection",
            "mode": "owner_override",
            "role": "system",
            "lane": "instruction",
            "anchor": "instructions.end",
            "order": 1,
            field: "owner text without runtime slot",
        }],
    }
    with pytest.raises(PromptPlanMirrorValidationError):
        PromptPlanMirrorStore(tmp_path / "mirror.sqlite3").put_plan(
            "preset:test", 1, plan_sha256="a" * 64,
            gateway_slice=body,
        )


def test_dynamic_context_rejects_owner_and_frozen_body(tmp_path):
    for mode, body_key in (("owner_override", "owner_body"),
                           ("frozen_snapshot", "frozen_body")):
        body = {
            "blocks": [{
                "block_id": "block:dynamic",
                "source_id": "ombre.recalled_memory",
                "scope": "talk.initial",
                "stage": "ombre_post_injection",
                "mode": mode,
                "role": "user",
                "lane": "message",
                "anchor": "gateway.current_user_prefix",
                body_key: "private dynamic text",
                "frozen_source_revision": "factory-v1"
                if mode == "frozen_snapshot" else None,
                "frozen_sha256": hashlib.sha256(
                    b"private dynamic text"
                ).hexdigest() if mode == "frozen_snapshot" else None,
            }],
        }
        with pytest.raises(PromptPlanMirrorValidationError):
            PromptPlanMirrorStore(tmp_path / f"{mode}.sqlite3").put_plan(
                "preset:test", 1, plan_sha256="b" * 64,
                gateway_slice=body,
            )


def test_resolve_fixed_prompt_uses_legacy_when_no_binding(tmp_path):
    store = PromptPlanMirrorStore(tmp_path / "mirror.sqlite3")
    resolved, meta = resolve_fixed_prompt(
        store,
        "ombre.memory_moment_prompt",
        config={"buckets_dir": str(tmp_path)},
    )
    assert resolved == FIXED_PROMPT_SOURCES[
        "ombre.memory_moment_prompt"
    ].factory_body()
    assert meta["payload_unchanged"] is True
