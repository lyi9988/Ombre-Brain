"""Body-free final-request capacity checks shared by SDK and Gateway.

The UTF-8/JSON measure is a conservative text-token upper estimate, not
provider usage or a model-specific tokenizer. Opaque media/server tools have
unknown cost. Unknown capacity does not mean overflow: ordinary chat remains
available; full attachment requests wait for verifiable capacity.
"""
from __future__ import annotations

import json
from typing import Any

HEADER = "X-Guyan-Context-Capacity"
_FIELDS = ("version", "model", "context_window", "reserved_output_tokens",
           "provider_id", "registry_revision", "capacity_source")
_OPAQUE_TYPES = {"image", "image_url", "input_image", "input_audio", "audio",
                 "video", "video_url", "file", "input_file", "document"}


def _integer(value: Any, *, zero: bool = False) -> int | None:
    return value if (isinstance(value, int) and not isinstance(value, bool)
                     and value >= (0 if zero else 1)) else None


def safe_contract(value: dict) -> dict:
    """Keep only bounded scalar routing metadata; never echo nested input."""
    result = {"version": 1}
    if value.get("contract_error"):
        result["contract_error"] = "invalid_capacity_contract"
    for key in ("model", "provider_id", "capacity_source", "registry_revision"):
        item = value.get(key)
        if item is None:
            continue
        if key == "registry_revision" and isinstance(item, int) and not isinstance(item, bool):
            item = str(item)
        if not isinstance(item, str) or len(item) > 300:
            result["contract_error"] = "invalid_capacity_contract"
        else:
            result[key] = item
    for key in ("context_window", "reserved_output_tokens"):
        item = value.get(key)
        result[key] = _integer(item, zero=key == "reserved_output_tokens")
        if item is not None and result[key] is None:
            result["contract_error"] = "invalid_capacity_contract"
    return result


def read_header(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        if len(raw.encode("utf-8")) > 4096:
            raise ValueError("capacity_header_too_large")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("invalid_capacity_contract")
        return safe_contract(value)
    except (ValueError, TypeError, UnicodeError):
        return {"contract_error": "invalid_capacity_contract"}


def encode_header(contract: dict) -> str:
    value = safe_contract(contract)
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def contains_attachment_text(payload: dict) -> bool:
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            text = (block.get("text") if isinstance(block, dict)
                    and block.get("type") in {"text", "input_text"} else block)
            if not isinstance(text, str) or not text.lstrip().startswith("{"):
                continue
            try:
                view = json.loads(text)
                if (isinstance(view, dict) and isinstance(view.get("attachment_source"), dict)
                        and view.get("content_available") is True
                        and isinstance(view.get("content"), str)):
                    return True
            except (ValueError, TypeError):
                pass
    return False


def _opaque(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("type"), str) and value["type"] in _OPAQUE_TYPES:
            return True
        return any(_opaque(item) for item in value.values())
    if isinstance(value, list):
        return any(_opaque(item) for item in value)
    return False


def _bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8"))


def assess_capacity(payload: dict, contract: dict | None = None, *,
                    stage: str = "final", has_attachments: bool | None = None) -> dict:
    contract = safe_contract(contract) if isinstance(contract, dict) else {}
    capacity = _integer(contract.get("context_window"))
    configured_reserve = _integer(contract.get("reserved_output_tokens"), zero=True)
    actual_reserves = [_integer(payload.get(key), zero=True)
                       for key in ("max_tokens", "max_completion_tokens") if key in payload]
    valid_reserves = [n for n in [configured_reserve, *actual_reserves] if n is not None]
    reserve = max(valid_reserves) if valid_reserves else None
    attachments = bool(has_attachments) or contains_attachment_text(payload)
    model = str(payload.get("model") or "")
    report = {
        "version": 1, "stage": stage, "model": model,
        "context_window": capacity, "reserved_output_tokens": reserve,
        "calculation_method": "json_utf8_upper_estimate_v1",
        "measurement": "conservative_estimate", "provider_usage_verified": False,
        "capacity_source": contract.get("capacity_source") or "unavailable",
        "registry_revision": contract.get("registry_revision"),
        "has_attachment_text": bool(attachments),
        "input_tokens_upper_bound": None, "required_tokens_upper_bound": None,
        "components": {},
    }
    unknown = contract.get("contract_error")
    if contract.get("model") and contract["model"] != model:
        unknown = "capacity_model_mismatch"
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    response_schema = payload.get("response_format")
    try:
        # Counts the actual serialized wire structures, including call/result
        # tails, native tools/handoffs and strict response schemas. Internal
        # metadata is removed by the caller before reaching this boundary.
        components = {"messages_utf8": _bytes(messages), "tools_utf8": _bytes(tools),
                      "response_format_utf8": _bytes(response_schema) if response_schema else 0}
        other = {key: value for key, value in payload.items()
                 if key not in {"messages", "tools", "response_format", "model", "stream",
                                "stream_options", "max_tokens", "max_completion_tokens"}
                 and not key.startswith("_")}
        components["other_input_utf8"] = _bytes(other)
        # Explicit framing allowance is part of the estimate, not a change to
        # the owner's context/output budget and not a claimed exact usage.
        components["framing_tokens"] = 256 + 32 * len(messages) + 64 * len(tools)
        report["components"] = components
        report["input_tokens_upper_bound"] = sum(components.values())
        if reserve is not None:
            report["required_tokens_upper_bound"] = report["input_tokens_upper_bound"] + reserve
    except (ValueError, TypeError, UnicodeError, OverflowError):
        unknown = unknown or "input_token_cost_unknown"
    if _opaque(messages) or any(isinstance(tool, dict) and tool.get("type") is not None
                                and tool.get("type") != "function" for tool in tools):
        unknown = unknown or "opaque_input_token_cost_unknown"
        # Serialized media bytes are wire-size observations, not a visual or
        # audio token estimate. Preserve them in *_utf8 components, but never
        # label base64 length as a model input/combined token upper bound.
        report.update(opaque_input_present=True, input_tokens_upper_bound=None,
                      required_tokens_upper_bound=None)
    if capacity is None:
        unknown = unknown or "model_context_capacity_unknown"
    if reserve is None or any(n is None for n in actual_reserves):
        unknown = unknown or "output_reservation_unknown"
    # Unknown input cost does not erase a separately proven output-only
    # overflow. Do not use a malformed or different-model contract as proof.
    trusted_window = (capacity is not None and not contract.get("contract_error")
                      and (not contract.get("model") or contract["model"] == model))
    if trusted_window and reserve is not None and reserve > capacity:
        report.update(state="exceeded", code="context_capacity_exceeded",
                      reason="output_reservation_exceeds_context_window",
                      dispatch_allowed=False)
    elif unknown:
        report.update(state="capacity_unverified", code="context_capacity_unknown",
                      reason=unknown, dispatch_allowed=not attachments)
    elif report["required_tokens_upper_bound"] > capacity:
        report.update(state="exceeded", code="context_capacity_exceeded",
                      reason="complete_request_and_output_reservation_exceed_capacity",
                      dispatch_allowed=False)
    else:
        report.update(state="within_capacity", code=None, reason=None, dispatch_allowed=True)
    return report


def blocked(report: dict) -> bool:
    return report.get("dispatch_allowed") is False


def error_body(report: dict) -> dict:
    unknown = report.get("code") == "context_capacity_unknown"
    return {
        "error": {
            "type": "context_capacity_error", "code": report.get("code"),
            "message": ("附件全文请求的容量尚无法核实，未发送模型。原文件与正文保持完整。"
                        if unknown else
                        "完整请求加输出预留的保守估算超过已配置容量，未发送模型，正文未裁剪、预算未降低。"),
        },
        "context_capacity": report,
        "available_actions": (["核实实际路由的上下文容量与输出预留", "取消本轮附件全文采用后继续纯聊天"]
                              if unknown else
                              ["自行减少本轮附件或上下文", "选择实际容量更大的已配置模型"]),
    }
