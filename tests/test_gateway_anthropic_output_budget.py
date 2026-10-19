"""Offline Anthropic budget conversion and local error propagation contracts."""
import asyncio
import copy
import json

import httpx
import pytest

from attachment_adoption_support import gateway_service
from gateway import AnthropicOutputBudgetError, GatewayService
from starlette.requests import Request


def harness():
    sent = []

    def provider(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"type": "message", "content": []})

    service, route = gateway_service(provider)
    route["upstream"]["protocol"] = "anthropic"
    return service, route, sent


def payload(**budgets):
    return {"model": "public-model", "messages": [{"role": "user", "content": "完整问题"}],
            "_ombre_trace_context": {}, **budgets}


async def send(service, route, body, stream):
    if stream:
        return await service._open_anthropic_upstream_stream(route, body)
    return await service._forward_anthropic_upstream(body, route)


@pytest.mark.parametrize("budgets", [
    {"max_tokens": 32768}, {"max_completion_tokens": 32768},
    {"max_tokens": 32768, "max_completion_tokens": 32768},
])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_budget_reaches_anthropic_and_final_capacity_unchanged(budgets, stream):
    service, route, sent = harness()
    body = payload(**budgets)
    original = copy.deepcopy(body)

    async def run():
        response = await send(service, route, body, stream)
        assert response.status_code == 200
        assert len(sent) == 1
        assert sent[0]["max_tokens"] == 32768
        assert "max_completion_tokens" not in sent[0]
        assert sent[0]["messages"][0]["content"] == "完整问题"
        report = body["_ombre_trace_context"]["context_capacity"]
        assert report["reserved_output_tokens"] == 32768
        assert report["route"]["stream"] is stream
        assert body["messages"] == original["messages"]
        for field, value in budgets.items():
            assert body[field] == value
        await response.aclose()
        await service.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("budgets, expected", [
    ({"max_tokens": 4000, "max_completion_tokens": 8000}, "conflict"),
    ({"max_tokens": None}, "max_tokens"),
    ({"max_completion_tokens": 0}, "max_completion_tokens"),
    ({"max_completion_tokens": -4}, "max_completion_tokens"),
    ({"max_completion_tokens": True}, "max_completion_tokens"),
    ({"max_completion_tokens": 1.5}, "max_completion_tokens"),
    ({"max_completion_tokens": "8192"}, "max_completion_tokens"),
    ({"max_tokens": 8192, "max_completion_tokens": None}, "max_completion_tokens"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_or_conflicting_budgets_reject_before_provider_or_adoption(budgets, expected, stream):
    service, route, sent = harness()
    service.gateway_cfg["anthropic_max_tokens"] = 9999
    body = payload(**budgets)
    original = copy.deepcopy(body)
    service.attachment_adoption.begin = lambda *_args: pytest.fail("invalid budget began adoption")

    async def run():
        response = await send(service, route, body, stream)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_output_budget"
        assert expected in error["message"]
        assert not sent
        assert body == original
        for proxy in (service._proxy_response, service._proxy_anthropic_error_response):
            public = proxy(response)
            assert public.status_code == 400
            assert json.loads(public.body)["error"] == error
        await service.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("native", [False, True])
def test_stream_public_wrapper_preserves_budget_error_and_never_opens_provider(native):
    service, route, sent = harness()
    body = payload(max_tokens=10000, max_completion_tokens=20000)

    async def run():
        wrapper = (service._stream_native_anthropic_upstream if native
                   else service._stream_anthropic_upstream_as_openai)
        response = await wrapper(route, body, "session", [], "question")
        assert response.status_code == 400
        assert json.loads(response.body)["error"]["code"] == "invalid_output_budget"
        assert "conflict" in json.loads(response.body)["error"]["message"]
        assert not sent
        await service.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("config, expected", [({}, 1024), ({"anthropic_max_tokens": 32000}, 32000),
    ({"anthropic_max_tokens": "32000"}, 32000), ({"anthropic_max_tokens": " +032000 "}, 32000)])
def test_omitted_budgets_keep_existing_default_or_configured_budget(config, expected):
    service = GatewayService.__new__(GatewayService)
    service.gateway_cfg = config
    assert service._anthropic_max_tokens({}) == expected
    assert service._anthropic_max_tokens({"max_completion_tokens": 64000}) == 64000


def test_native_anthropic_input_cannot_silently_drop_openai_budget_alias():
    service = GatewayService.__new__(GatewayService)
    body = payload(max_tokens=4000, max_completion_tokens=8000)
    with pytest.raises(ValueError, match="max_completion_tokens is not supported by /v1/messages"):
        service._anthropic_request_to_openai(body)


def test_converter_itself_exposes_conflict_without_fallback():
    service, route, _sent = harness()
    with pytest.raises(AnthropicOutputBudgetError, match="conflict"):
        service._anthropic_payload_for_upstream(payload(max_tokens=4000, max_completion_tokens=8000), route)
    asyncio.run(service.http_client.aclose())


@pytest.mark.parametrize("stream", [False, True])
def test_converted_completion_budget_can_block_capacity_before_dispatch(stream):
    service, route, sent = harness()
    service.gateway_cfg["upstreams"][0]["context_window"] = 10000
    body = payload(max_completion_tokens=32000)

    async def run():
        response = await send(service, route, body, stream)
        assert response.status_code == 413
        assert response.json()["context_capacity"]["reserved_output_tokens"] == 32000
        assert body["max_completion_tokens"] == 32000
        assert not sent
        await service.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_native_http_handler_reports_unsupported_budget_before_preparation(stream):
    service, _route, sent = harness()
    service._session_id_for_request = lambda _request: "session"
    service._client_label_from_request = lambda *_args: "test"
    encoded = json.dumps(payload(max_completion_tokens=32000, stream=stream)).encode()

    async def receive():
        return {"type": "http.request", "body": encoded, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                       "headers": [(b"x-api-key", b"gateway-test-token")]}, receive)

    async def run():
        response = await service.handle_anthropic_messages(request)
        assert response.status_code == 400
        assert "max_completion_tokens is not supported" in json.loads(response.body)["error"]["message"]
        assert not sent
        await service.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("configured", [None, 0, -1, True, 1.5, "invalid", "0", "1.5"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_configured_budget_rejects_without_fallback_or_provider(configured, stream):
    service, route, sent = harness()
    service.gateway_cfg["anthropic_max_tokens"] = configured

    async def run():
        response = await send(service, route, payload(), stream)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_output_budget"
        assert "gateway.anthropic_max_tokens" in response.json()["error"]["message"]
        assert not sent
        # A valid request budget takes precedence over an unused bad default.
        assert service._anthropic_max_tokens({"max_tokens": 32000}) == 32000
        await service.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("invalid", [None, 0, -1, True, 1.5, "32000"])
@pytest.mark.parametrize("stream", [False, True])
def test_native_invalid_max_tokens_rejects_before_preparation_or_openai_routing(invalid, stream):
    service, route, sent = harness()
    route["upstream"]["protocol"] = "openai"
    service._session_id_for_request = lambda _request: "session"
    service._client_label_from_request = lambda *_args: "test"
    service._resolve_upstream_for_model = lambda *_args: pytest.fail("invalid native budget reached routing")
    encoded = json.dumps(payload(max_tokens=invalid, stream=stream)).encode()

    async def receive():
        return {"type": "http.request", "body": encoded, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                       "headers": [(b"x-api-key", b"gateway-test-token")]}, receive)

    async def run():
        response = await service.handle_anthropic_messages(request)
        assert response.status_code == 400
        assert "max_tokens must be a positive integer" in json.loads(response.body)["error"]["message"]
        assert not sent
        await service.http_client.aclose()

    asyncio.run(run())


def test_native_valid_max_tokens_preserves_value_without_consulting_anthropic_default():
    service = GatewayService.__new__(GatewayService)
    converted = service._anthropic_request_to_openai(payload(max_tokens=32000))
    assert converted["max_tokens"] == 32000
