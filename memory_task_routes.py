"""Stage-specific choices over existing Reflection clients, with no new credentials."""
from copy import deepcopy
import hashlib
import json

STAGES = ("auto_l1", "chat_l1", "action_l1", "diary_l1", "chat_summary", "activity_summary")
DEFAULT_ROUTE = {"connection": "legacy", "model": "", "thinking_mode": "inherit",
                 "reasoning_effort": "", "max_tokens": None}


def normalize_task_routes(raw=None):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(STAGES):
        raise ValueError("invalid_memory_task_routes")
    result = {}
    for stage in STAGES:
        value = raw.get(stage, {})
        if not isinstance(value, dict) or set(value) - set(DEFAULT_ROUTE):
            raise ValueError("invalid_memory_task_route:" + stage)
        row = {**DEFAULT_ROUTE, **value}
        if row["connection"] not in ("legacy", "reflection", "dehydration", "daily"):
            raise ValueError("invalid_memory_task_connection:" + stage)
        if not isinstance(row["model"], str) or len(row["model"]) > 200:
            raise ValueError("invalid_memory_task_model:" + stage)
        row["model"] = row["model"].strip()
        if row["thinking_mode"] not in ("inherit", "default", "enabled", "disabled"):
            raise ValueError("invalid_memory_task_thinking:" + stage)
        if row["reasoning_effort"] not in ("", "low", "medium", "high"):
            raise ValueError("invalid_memory_task_effort:" + stage)
        tokens = row["max_tokens"]
        if tokens is not None and (type(tokens) is not int or not 1 <= tokens <= 1000000):
            raise ValueError("invalid_memory_task_max_tokens:" + stage)
        result[stage] = row
    return result


def resolve_task_route(engine, stage, default_max_tokens=None):
    if stage not in STAGES:
        raise ValueError("unknown_memory_task_stage")
    config = getattr(engine, "config", {}) or {}
    reflection = config.get("reflection") or {}
    row = normalize_task_routes(reflection.get("memory_task_routes"))[stage]
    source = row["connection"]
    daily = getattr(engine, "daily_chat_memory_client", None)
    reflection_client = getattr(engine, "client", None)
    dehydration = getattr(engine, "dehydration_client", None)
    if source == "legacy":
        if stage == "diary_l1":
            source = "reflection"
        elif stage == "activity_summary":
            source = "daily" if daily else "reflection"
        else:
            source = "daily" if daily else "dehydration" if dehydration and getattr(engine, "dehydration_model", "") else "reflection"
    client = {"daily": daily, "reflection": reflection_client, "dehydration": dehydration}[source]
    if row["connection"] == "reflection" and not getattr(engine, "reflection_connection_explicit", True):
        client = None
    source_model = {
        "reflection": getattr(engine, "model", ""),
        "dehydration": getattr(engine, "dehydration_model", ""),
        "daily": getattr(engine, "daily_chat_memory_summary_model" if stage in
                         ("chat_summary", "activity_summary") else "daily_chat_memory_candidate_model", ""),
    }[source]
    model = row["model"] or source_model
    if source == "reflection":
        inherited_thinking = getattr(engine, "thinking_mode", "")
    elif source == "dehydration":
        inherited_thinking = (config.get("dehydration") or {}).get("thinking_mode", "")
    else:
        inherited_thinking = reflection.get("daily_chat_memory_thinking_mode", "")
    thinking = inherited_thinking if row["thinking_mode"] == "inherit" else row["thinking_mode"]
    if thinking == "default":
        thinking = ""
    if default_max_tokens is None:
        if stage == "diary_l1":
            from memory_l1_runtime import task_options
            default_max_tokens = task_options(config)["diary_l1_max_tokens"]
        else:
            attribute = {"chat_summary": "daily_chat_memory_summary_max_tokens",
                         "activity_summary": "daily_activity_summary_max_tokens"}.get(
                             stage, "daily_chat_memory_candidate_max_tokens")
            default_max_tokens = getattr(engine, attribute, 16000 if stage.endswith("l1") else 2200)
    configured = client is not None and bool(model)
    return {"stage": stage, "connection": source, "client": client, "model": str(model or ""),
            "thinking_mode": str(thinking or ""), "reasoning_effort": row["reasoning_effort"],
            "max_tokens": row["max_tokens"] if row["max_tokens"] is not None else default_max_tokens,
            "configured": configured,
            "unavailable_reason": "" if configured else "connection_unconfigured" if client is None else "model_unconfigured",
            "model_source": "stage" if row["model"] else source,
            "thinking_source": source if row["thinking_mode"] == "inherit" else "stage",
            "max_tokens_source": "stage" if row["max_tokens"] is not None else "task_default"}


def public_task_models(engine):
    routes = normalize_task_routes((getattr(engine, "config", {}).get("reflection") or {}).get("memory_task_routes"))
    effective = {}
    for stage in STAGES:
        row = resolve_task_route(engine, stage)
        row.pop("client")
        effective[stage] = row
    return {"routes": deepcopy(routes), "effective": effective, "revision": task_routes_revision(routes)}


def task_routes_revision(routes):
    return "mt-" + hashlib.sha256(json.dumps(normalize_task_routes(routes), sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def completion_options(route, *, temperature):
    options = {"max_tokens": route["max_tokens"], "temperature": temperature}
    if route["thinking_mode"]:
        options["extra_body"] = {"thinking": {"type": route["thinking_mode"]}}
    if route["reasoning_effort"]:
        options["reasoning_effort"] = route["reasoning_effort"]
    return options
