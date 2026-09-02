"""Authoritative registry for owner-editable fixed Ombre prompts.

The registry exposes factory defaults and classifications only.  Runtime
facts (memory, persona state, reminders, chat history, tool results, etc.)
remain owned by their original stores and are never returned as prompt body.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FixedPromptSource:
    source_id: str
    scope: str
    module: str
    attribute: str
    render_mode: str = "plain"
    required_placeholders: tuple[str, ...] = ()
    source_revision: str = "factory-v1"

    @property
    def authority(self) -> str:
        return f"ombre.factory:{self.module}.{self.attribute}"

    def factory_body(self) -> str:
        module = importlib.import_module(self.module)
        return str(getattr(module, self.attribute))


def _spec(source_id: str, scope: str, ref: str, *, render: str = "plain",
          required: tuple[str, ...] = ()) -> FixedPromptSource:
    module, attribute = ref.split(":", 1)
    return FixedPromptSource(
        source_id=source_id, scope=scope, module=module,
        attribute=attribute, render_mode=render,
        required_placeholders=required,
    )


FIXED_PROMPT_SOURCES = {
    item.source_id: item for item in (
        _spec("ombre.persona_post_reply_prompt", "persona.post_reply_evaluation",
              "persona_engine:POST_REPLY_EVALUATION_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.memory_query_planner_prompt", "memory.query_planner",
              "query_prompts:QUERY_PLANNER_SYSTEM_PROMPT"),
        _spec("ombre.semantic_rescue_prompt", "memory.semantic_rescue",
              "gateway_prompt_factories:SEMANTIC_RESCUE_SYSTEM_PROMPT"),
        _spec("ombre.memory_classify_prompt", "memory.reflection",
              "reflection_engine:CLASSIFY_PROMPT"),
        _spec("ombre.reflection_prompt", "memory.reflection",
              "reflection_engine:REFLECT_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.diary_memory_prompt", "memory.reflection",
              "reflection_engine:DIARY_MEMORY_PROMPT_TEMPLATE", render="domain_identity"),
        _spec("ombre.daily_chat_memory_prompt", "memory.daily_chat_review",
              "reflection_engine:DAILY_CHAT_MEMORY_PROMPT_TEMPLATE",
              render="domain_identity", required=("{max_candidates}",)),
        _spec("ombre.daily_chat_summary_prompt", "memory.daily_chat_review",
              "reflection_engine:DAILY_CHAT_MEMORY_SUMMARY_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.daily_activity_summary_prompt", "memory.daily_chat_review",
              "reflection_engine:DAILY_ACTIVITY_SUMMARY_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.domain_sentinel_prompt", "memory.domain_sentinel",
              "gateway_prompt_factories:DOMAIN_SENTINEL_SYSTEM_PROMPT"),
        _spec("ombre.dehydrate_prompt", "memory.dehydrate",
              "dehydrator:DEHYDRATE_PROMPT"),
        _spec("ombre.direct_capsule_prompt", "memory.direct_capsule",
              "dehydrator:DIRECT_BUCKET_CAPSULE_PROMPT"),
        _spec("ombre.memory_merge_prompt", "memory.merge",
              "dehydrator:MERGE_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.memory_analyze_prompt", "memory.analyze",
              "dehydrator:ANALYZE_PROMPT_TEMPLATE", render="domain"),
        _spec("ombre.memory_digest_prompt", "memory.digest",
              "dehydrator:DIGEST_PROMPT_TEMPLATE", render="domain_identity"),
        _spec("ombre.memory_moment_prompt", "memory.moment",
              "dehydrator:MOMENT_PROMPT"),
        _spec("ombre.dream_generation_prompt", "dream.generate",
              "dream_engine:DREAM_PROMPT"),
        _spec("ombre.dream_wrapper_prompt", "talk.initial",
              "gateway_prompt_factories:DREAM_WRAPPER_PROMPT", required=("{{content}}",)),
        _spec("ombre.portrait_patch_prompt", "portrait.patch",
              "portrait_engine:PORTRAIT_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.portrait_stable_prompt", "portrait.stable",
              "portrait_engine:STABLE_MAINTENANCE_PROMPT_TEMPLATE", render="identity"),
        _spec("ombre.profile_fact_proposal_prompt", "profile.fact_proposal",
              "profile_prompts:PROFILE_FACT_PROPOSAL_PROMPT_TEMPLATE", render="identity_format"),
        _spec("ombre.anchor_proposal_prompt", "profile.anchor_proposal",
              "profile_prompts:ANCHOR_PROPOSAL_PROMPT_TEMPLATE", render="identity_format"),
        _spec("ombre.gateway_stable_preface_prompt", "talk.initial",
              "gateway_prompt_factories:GATEWAY_STABLE_PREFACE_PROMPT"),
        _spec("ombre.gateway_dynamic_preface_prompt", "talk.initial",
              "gateway_prompt_factories:GATEWAY_DYNAMIC_PREFACE_PROMPT"),
        _spec("ombre.memory_reading_policy_prompt", "talk.initial",
              "gateway_prompt_factories:MEMORY_READING_POLICY_PROMPT"),
        _spec("ombre.date_boundary_prompt", "talk.initial",
              "gateway_prompt_factories:DATE_BOUNDARY_PROMPT"),
        _spec("ombre.memory_detail_request_prompt", "talk.initial",
              "gateway_prompt_factories:MEMORY_DETAIL_REQUEST_PROMPT", required=("{{max_ids}}",)),
        _spec("ombre.handoff_hint_prompt", "talk.initial",
              "gateway_prompt_factories:HANDOFF_HINT_PROMPT"),
        _spec("ombre.handoff_date_hint_prompt", "talk.initial",
              "gateway_prompt_factories:HANDOFF_DATE_HINT_PROMPT"),
        _spec("ombre.active_reminder_prompt", "talk.initial",
              "gateway_prompt_factories:ACTIVE_REMINDER_PROMPT", required=("{{content}}",)),
        _spec("ombre.import_extract_prompt", "memory.import",
              "import_memory:IMPORT_EXTRACT_PROMPT"),
        _spec("ombre.reclassify_prompt", "memory.reclassify",
              "reclassify_api:ANALYZE_PROMPT"),
    )
}


DYNAMIC_CONTEXT_SOURCES = frozenset({
    "ombre.core_memory", "ombre.portrait_memory", "ombre.just_now_context",
    "ombre.date_recall", "ombre.context_mode", "ombre.active_reminders",
    "ombre.memory_detail_request", "ombre.memory_reading_policy",
    "ombre.recalled_memory", "ombre.targeted_memory_detail",
    "ombre.diffused_memory", "ombre.recent_context",
    "ombre.date_persona_trace", "ombre.handoff_hint", "ombre.persona_state",
    "ombre.relationship_weather", "ombre.favorite_memory",
    "ombre.dream_context",
})


OWNER_AUTHORED_PROMPT_SOURCES = frozenset({"custom.gateway_block"})


# These identifiers describe transport/runtime mechanics for owner-visible
# classification only.  They are deliberately not accepted in a mirrored
# prompt plan because they are not model-facing natural-language authority.
RUNTIME_MECHANIC_SOURCES = frozenset({"ombre.runtime.phase_marker"})


def fixed_prompt_source(source_id: str) -> FixedPromptSource | None:
    return FIXED_PROMPT_SOURCES.get(str(source_id or "").strip())


def factory_body(source_id: str) -> str:
    spec = fixed_prompt_source(source_id)
    if spec is None:
        raise KeyError(source_id)
    return spec.factory_body()


def prompt_source_kind(source_id: str) -> str:
    source_id = str(source_id or "").strip()
    if source_id in FIXED_PROMPT_SOURCES:
        return "fixed_prompt"
    if source_id in OWNER_AUTHORED_PROMPT_SOURCES:
        return "fixed_prompt"
    if source_id in DYNAMIC_CONTEXT_SOURCES:
        return "dynamic_context"
    if source_id in RUNTIME_MECHANIC_SOURCES:
        return "runtime_mechanics"
    return "unknown"


def _identity_values(config: dict | None) -> dict[str, str]:
    from identity import identity_names
    return identity_names(config or {})


def render_factory_body(spec: FixedPromptSource, body: str,
                        config: dict | None = None) -> str:
    rendered = str(body or "")
    if spec.render_mode == "identity_format":
        # Profile proposal factories were historically rendered with
        # ``str.format`` and therefore use doubled JSON braces.  Unescape the
        # factory syntax before substituting identity values so braces inside
        # an owner-configured name remain untouched.
        rendered = rendered.replace("{{", "{").replace("}}", "}")
    if spec.render_mode in {"domain", "domain_identity"}:
        from memory_metadata import domain_prompt_options_text
        rendered = rendered.replace("{domain_options_text}", domain_prompt_options_text())
    if spec.render_mode in {"identity", "domain_identity", "identity_format"}:
        from identity import render_identity_template
        rendered = render_identity_template(rendered, _identity_values(config))
    return rendered


def validate_required_placeholders(source_id: str, body: str) -> None:
    spec = fixed_prompt_source(source_id)
    if not spec:
        return
    missing = [token for token in spec.required_placeholders if token not in str(body or "")]
    if missing:
        raise ValueError(
            f"{source_id} override is missing required placeholders: {', '.join(missing)}")


def render_runtime_placeholders(source_id: str, body: str,
                                values: dict[str, Any] | None = None) -> str:
    rendered = str(body or "")
    values = values if isinstance(values, dict) else {}
    for key, value in values.items():
        rendered = rendered.replace("{{" + str(key) + "}}", str(value))
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered


def source_detail(source_id: str, config: dict | None = None) -> dict[str, Any]:
    source_id = str(source_id or "").strip()
    kind = prompt_source_kind(source_id)
    if kind != "fixed_prompt":
        return {
            "source_id": source_id,
            "body_kind": kind,
            "reason": (
                "runtime facts stay in their source authority and have no editable fixed body"
                if kind == "dynamic_context" else
                "runtime protocol is not natural-language prompt authority"
                if kind == "runtime_mechanics" else
                "source_id is not registered"
            ),
        }
    if source_id in OWNER_AUTHORED_PROMPT_SOURCES:
        return {
            "source_id": source_id,
            "body_kind": "fixed_prompt",
            "reason": "owner-authored prompt has no factory body",
            "authority": "aizizhu.prompt_composer",
        }
    spec = FIXED_PROMPT_SOURCES[source_id]
    factory_body = spec.factory_body()
    live_body = render_factory_body(spec, factory_body, config)
    return {
        "source_id": source_id,
        "body_kind": "fixed_prompt",
        "factory_body": factory_body,
        "live_body": live_body,
        "source_revision": spec.source_revision,
        "source_sha256": hashlib.sha256(factory_body.encode("utf-8")).hexdigest(),
        "authority": spec.authority,
    }


def resolve_fixed_prompt(store: Any, source_id: str, *,
                         config: dict | None = None,
                         identity_id: str = "jiajia-main",
                         conversation_id: str = "",
                         runtime_values: dict[str, Any] | None = None) -> tuple[str, dict]:
    spec = fixed_prompt_source(source_id)
    if not spec:
        raise KeyError(source_id)
    factory_body = spec.factory_body()
    resolved, meta = store.resolve_text(
        scope=spec.scope, source_id=source_id, live_body=factory_body,
        identity_id=identity_id, conversation_id=conversation_id)
    validate_required_placeholders(source_id, resolved)
    rendered = render_factory_body(spec, resolved, config)
    rendered = render_runtime_placeholders(source_id, rendered, runtime_values)
    return rendered, meta


__all__ = [
    "DYNAMIC_CONTEXT_SOURCES", "FIXED_PROMPT_SOURCES",
    "OWNER_AUTHORED_PROMPT_SOURCES", "RUNTIME_MECHANIC_SOURCES",
    "FixedPromptSource", "factory_body",
    "fixed_prompt_source",
    "prompt_source_kind", "render_factory_body", "render_runtime_placeholders",
    "resolve_fixed_prompt", "source_detail", "validate_required_placeholders",
]
