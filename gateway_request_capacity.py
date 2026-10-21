"""Resolve explicit capacity metadata for the actual Gateway route, without bodies."""
from __future__ import annotations

import hashlib
import json


def positive_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def explicit_limits(value):
    if not isinstance(value, dict):
        return {}
    return {key: positive_int(value[key]) for key in ("context_window", "max_output_tokens")
            if positive_int(value.get(key)) is not None}


def route_limits(config, route):
    """Use only this upstream and matching alias/provider model's explicit fields.

    Context conflicts take the smaller window; output reservation uses the larger
    explicit maximum. No inference from model names or unrelated provider rows.
    """
    config = config if isinstance(config, dict) else {}
    upstream = route["upstream"]
    raw_upstreams = config.get("upstreams")
    if isinstance(raw_upstreams, list) and raw_upstreams:
        raw = next((item for index, item in enumerate(raw_upstreams, 1)
                    if isinstance(item, dict) and str(item.get("name") or f"upstream-{index}").strip()
                    == upstream.get("name")), {})
        models = raw.get("models", [])
    else:
        raw = config
        models = raw.get("upstream_models", [])
    sources = [("upstream", explicit_limits(upstream)), ("upstream_config", explicit_limits(raw))]
    aliases = {str(route.get("public_model") or ""), str(route.get("upstream_model") or "")} - {""}
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            public = str(item.get("id") or item.get("alias") or item.get("name") or
                         item.get("model") or item.get("upstream_model") or "").strip()
            target = str(item.get("upstream_model") or item.get("provider_model") or
                         item.get("target_model") or item.get("model") or public).strip()
            if public in aliases or target == route.get("upstream_model"):
                sources.append(("model_config", explicit_limits(item)))
    windows = [value["context_window"] for _source, value in sources if "context_window" in value]
    reserves = [value["max_output_tokens"] for _source, value in sources if "max_output_tokens" in value]
    return {"context_window": min(windows) if windows else None,
            "reserved_output_tokens": max(reserves) if reserves else None,
            "sources": [{"source": source, **value} for source, value in sources if value]}


def has_full_attachment(payload, trace):
    manifest = trace.get("attachment_adoption") or {}
    image_descriptors = set()
    # Legacy callers may omit the transport manifest. Inspect the actual source
    # wrapper rather than mistaking a filename or attachment mention for a body.
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        image_present = isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"image_url", "image"}
            for part in content)
        blocks = [content] if isinstance(content, str) else (
            [part.get("text") for part in content if isinstance(part, dict)]
            if isinstance(content, list) else [])
        for text in blocks:
            if not isinstance(text, str) or "attachment_source" not in text:
                continue
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                # Legacy string injection can prefix the exact JSON block
                # with recall context. Decode only a structurally valid
                # wrapper; mentions of attachments are never sufficient.
                marker = text.find('"attachment_source"')
                start = text.rfind("{", 0, marker)
                if marker < 0 or start < 0:
                    continue
                try:
                    value, _end = json.JSONDecoder().raw_decode(text[start:])
                except (ValueError, TypeError):
                    continue
            if (isinstance(value, dict) and isinstance(value.get("attachment_source"), dict)
                    and isinstance(value.get("content"), str)):
                return True
            if (image_present and isinstance(value, dict)
                    and isinstance(value.get("attachment_source"), dict)
                    and value["attachment_source"].get("source_kind") == "image"
                    and value.get("adoption") == "native_image"
                    and value.get("media_available") is True
                    and value.get("content_available") is False
                    and value.get("content") is None):
                image_descriptors.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    # Owner policy: original native images may dispatch with explicitly unknown
    # capacity. Full text and unaccounted manifest entries retain the prior gate.
    # This classifies input only; adoption still verifies the actual image bytes.
    if manifest:
        wanted = manifest.get("body_sha256s")
        # This optional unavailable-only manifest declares no adopted full text.
        # Actual text detected above still wins; capacity estimation is unchanged.
        from attachment_adoption import projection_only_capacity_exempt
        if projection_only_capacity_exempt(payload, trace):
            return False
        return not (isinstance(wanted, list) and wanted
                    and all(isinstance(item, str) for item in wanted)
                    and set(wanted) <= image_descriptors)
    return False
