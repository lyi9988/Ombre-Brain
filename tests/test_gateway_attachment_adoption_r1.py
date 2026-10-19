import asyncio
import gzip
import json
import time

import httpx

from attachment_adoption_support import (
    attachment_payload,
    gateway_service,
    lookup,
    openai_response,
    r1_tool_lookup_query,
)


def test_physical_tool_completion_uses_exact_scope_and_authenticated_lookup():
    sent = {}

    def fake_upstream(request):
        sent["payload"] = json.loads(request.content)
        return httpx.Response(200, json=openai_response())

    service, _route = gateway_service(fake_upstream)
    payload, block, private_text, expected_ref = attachment_payload()

    async def run():
        physical = await service._forward_upstream(payload)
        assert physical.status_code == 200
        # Trace/manifest bookkeeping stays inside Gateway and does not alter
        # the final chat body sent to the provider.
        assert "_ombre_trace_context" not in sent["payload"]
        assert sent["payload"]["messages"][0]["content"] == block
        denied = await lookup(service, r1_tool_lookup_query(), authorized=False)
        assert denied.status_code == 401
        valid = await lookup(service, r1_tool_lookup_query())
        assert valid.status_code == 200
        receipt = valid.json()
        assert receipt["state"] == "available"
        assert receipt["attempt_id"].startswith("adopt-")
        assert receipt["attachments"] == [expected_ref]
        assert private_text not in valid.text
        wrong_turn = await lookup(
            service, r1_tool_lookup_query(turn_id="another-turn")
        )
        assert wrong_turn.json()["state"] == "unavailable"
        await service.http_client.aclose()

    asyncio.run(run())


def test_actual_gateway_stream_hook_decodes_gzip_and_records_before_yield():
    payload, _block, _private_text, _expected_ref = attachment_payload()
    sse = (
        'data: {"choices":[{"index":0,"delta":{"tool_calls":'
        '[{"index":0,"id":"call-1"}]},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    def fake_upstream(_request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream", "Content-Encoding": "gzip"},
            content=gzip.compress(sse),
        )

    service, route = gateway_service(fake_upstream)

    async def run():
        upstream = await service._open_upstream_stream(route, payload)
        timing = {}
        stream = service._iter_upstream_bytes(
            upstream,
            timing,
            stream_started_at=time.perf_counter(),
            body_started_at=time.perf_counter(),
        )
        first_downstream_chunk = await anext(stream)
        assert first_downstream_chunk == sse
        receipt = service.attachment_adoption.lookup(r1_tool_lookup_query())
        assert receipt["state"] == "available"
        assert [item["body_sha256"] for item in receipt["attachments"]]
        await stream.aclose()
        await upstream.aclose()
        await service.http_client.aclose()

    asyncio.run(run())
