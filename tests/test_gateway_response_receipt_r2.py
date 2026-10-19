import asyncio
import json

import httpx

from attachment_adoption_support import (
    attachment_payload,
    gateway_service,
    lookup,
    openai_response,
    r2_response_lookup_query,
)


def test_successful_plain_response_receipt_is_exact_and_duplicate_success_is_ambiguous():
    calls = 0

    def fake_upstream(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=openai_response(finish_reason="stop"))

    service, _route = gateway_service(fake_upstream)
    payload, block, private_text, expected_ref = attachment_payload()

    async def run():
        first = await service._forward_upstream(payload)
        assert first.status_code == 200
        available = await lookup(service, r2_response_lookup_query())
        assert available.json()["state"] == "available"
        assert available.json()["lookup_type"] == "response"
        assert available.json()["logical_request_id"] == "logical-1"
        assert available.json()["attachments"] == [expected_ref]
        assert private_text not in available.text
        wrong_scope = await lookup(
            service, r2_response_lookup_query(request_id="another-request")
        )
        assert wrong_scope.json()["state"] == "unavailable"

        second = await service._forward_upstream(payload)
        assert second.status_code == 200
        ambiguous = await lookup(service, r2_response_lookup_query())
        assert ambiguous.json()["state"] == "unknown"
        assert ambiguous.json()["reason"] == "ambiguous_physical_attempt"
        assert "attachments" not in ambiguous.json()
        assert calls == 2
        await service.http_client.aclose()

    asyncio.run(run())


def test_length_and_http_failure_do_not_create_ordinary_response_receipts():
    status = 200
    finish_reason = "length"

    def fake_upstream(_request):
        if status == 500:
            return httpx.Response(500, json={"error": "upstream failure"})
        return httpx.Response(200, json=openai_response(finish_reason=finish_reason))

    service, _route = gateway_service(fake_upstream)

    async def run():
        payload, _block, _text, _ref = attachment_payload(logical_request_id="length-request")
        length = await service._forward_upstream(payload)
        assert length.status_code == 200
        length_receipt = await lookup(
            service,
            r2_response_lookup_query(logical_request_id="length-request"),
        )
        assert length_receipt.json()["state"] == "unavailable"

        nonlocal status
        status = 500
        payload, _block, _text, _ref = attachment_payload(logical_request_id="http-error")
        failed = await service._forward_upstream(payload)
        assert failed.status_code == 500
        failed_receipt = await lookup(
            service,
            r2_response_lookup_query(logical_request_id="http-error"),
        )
        assert failed_receipt.json()["state"] == "unavailable"
        await service.http_client.aclose()

    asyncio.run(run())
