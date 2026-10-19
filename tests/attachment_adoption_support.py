"""Small fake-upstream harness for the integrated Gateway receipt tests."""
import base64
import hashlib
import json
from types import SimpleNamespace

import httpx

from attachment_adoption import AdoptionReceipts, canonical, parse_header, sha
from gateway import GatewayService, create_gateway_app


def attachment_payload(*, logical_request_id="logical-1", text="附件正文：海边骑行"):
    locator = {
        "owner_id": "owner-1",
        "conversation_id": "conversation-1",
        "event_id": "event-1",
        "event_version_id": "event-version-1",
        "resource_id": "resource-1",
        "raw_sha256": "a" * 64,
        "text_sha256": sha(text),
        "text_view_id": "attachment.utf8-sig.exact-v1",
    }
    source_id = canonical(locator)
    wrapper = {
        "attachment_source": {
            "source_id": source_id,
            "source_revision": "attachment_sha256:" + sha(source_id),
            "locator": locator,
        },
        "content": text,
    }
    block = json.dumps(wrapper, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    header = base64.urlsafe_b64encode(
        json.dumps({
            "version": 1,
            "profile_id": "profile-1",
            "body_sha256s": [hashlib.sha256(block.encode("utf-8")).hexdigest()],
        }, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    trace = {
        "trace_id": "trace-1",
        "request_id": "request-1",
        "turn_id": "turn-1",
        "conversation_id": "conversation-1",
        "logical_request_id": logical_request_id,
        "attachment_adoption": parse_header(header),
    }
    payload = {
        "model": "public-model",
        "messages": [{"role": "user", "content": block}],
        "_ombre_trace_context": trace,
    }
    expected_ref = {
        "source_id": source_id,
        "source_revision": wrapper["attachment_source"]["source_revision"],
        "body_sha256": sha(block),
    }
    return payload, block, text, expected_ref


class _NoTraceStore:
    def record_attempt(self, **_kwargs):
        return None


def gateway_service(handler):
    upstream = {
        "name": "fake-upstream",
        "base_url": "https://upstream.invalid/v1",
        "api_keys": [{"label": "test-key", "value": "provider-secret"}],
    }
    route = {
        "upstream": upstream,
        "public_model": "public-model",
        "upstream_model": "upstream-model",
    }
    service = GatewayService.__new__(GatewayService)
    service.gateway_token = "gateway-test-token"
    service.attachment_adoption = AdoptionReceipts()
    service.model_request_trace = _NoTraceStore()
    service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service._resolve_upstream_for_model = lambda _model: route
    service._available_upstream_api_keys = lambda _upstream: upstream["api_keys"]
    service._clear_upstream_key_cooldown = lambda *_args: None
    service._cool_down_upstream_key = lambda *_args: None
    return service, route


async def lookup(service, query, *, authorized=True):
    app = create_gateway_app(config={"gateway": {}}, service=service)
    app.state.gateway_service = service
    headers = {"Authorization": "Bearer gateway-test-token"} if authorized else {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        response = await client.post(
            "/internal/attachment-adoption/lookup", json=query, headers=headers
        )
    return response


def r1_tool_lookup_query(**overrides):
    query = {
        "request_id": "request-1",
        "turn_id": "turn-1",
        "conversation_id": "conversation-1",
        "profile_id": "profile-1",
        "tool_call_id": "call-1",
    }
    query.update(overrides)
    return query


def r2_response_lookup_query(**overrides):
    query = {
        "lookup_type": "response",
        "request_id": "request-1",
        "turn_id": "turn-1",
        "conversation_id": "conversation-1",
        "profile_id": "profile-1",
        "logical_request_id": "logical-1",
    }
    query.update(overrides)
    return query


def openai_response(*, finish_reason="tool_calls", tool_call_id="call-1"):
    tool_calls = []
    if finish_reason == "tool_calls":
        tool_calls = [{
            "id": tool_call_id,
            "type": "function",
            "function": {"name": "memory.write", "arguments": "{}"},
        }]
    return {
        "choices": [{
            "index": 0,
            "message": {"content": "完成", "tool_calls": tool_calls},
            "finish_reason": finish_reason,
        }]
    }
