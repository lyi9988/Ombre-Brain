import asyncio

import httpx
import pytest

from canonical_continuation import CanonicalContinuationAdapter


def adapter(tmp_path, handler, **overrides):
    values = dict(
        enabled=True, base_url="http://canonical.local", token="bridge-token",
        device_id="ombre-gateway-operit", conversation_id="conv-jiajia-main",
        channel_id="operit", state_db_path=tmp_path / "gateway.db",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_events=20,
    )
    values.update(overrides)
    return CanonicalContinuationAdapter(**values)


def test_pull_filters_non_natural_and_own_events_then_merges_before_current_user(tmp_path):
    async def scenario():
        async def handler(request):
            assert request.headers["X-Guyan-Device-ID"] == "ombre-gateway-operit"
            return httpx.Response(200, json={
                "conversation_id": "conv-jiajia-main", "next_after_seq": 8,
                "items": [
                    {"seq": 4, "event_type": "turn.started", "role": "system"},
                    {"seq": 5, "event_type": "message", "role": "user", "content": "那我下午去。", "source": "app"},
                    {"seq": 6, "event_type": "message", "role": "assistant", "content": "别盯着针。", "source": "conversation-service"},
                    {"seq": 7, "event_type": "message", "role": "assistant", "content": "own", "source": "ombre-gateway"},
                    {"seq": 8, "event_type": "message", "role": "tool", "content": "secret", "source": "app"},
                ],
            })
        item = adapter(tmp_path, handler)
        item.commit_cursor(3)
        batch = await item.pull()
        assert [(event["role"], event["content"]) for event in batch.events] == [
            ("user", "那我下午去。"), ("assistant", "别盯着针。"),
        ]
        messages = [{"role": "system", "content": "role card"}, {"role": "user", "content": "但不会很疼吧？"}]
        merged = item.merge_messages(messages, batch)
        assert merged == [
            {"role": "system", "content": "role card"},
            {"role": "user", "content": "那我下午去。"},
            {"role": "assistant", "content": "别盯着针。"},
            {"role": "user", "content": "但不会很疼吧？"},
        ]
        assert item.cursor() == 3  # pull does not advance before a successful reply
        assert item.commit_cursor(batch.through_seq) == 8
        assert item.commit_cursor(3) == 8
        await item.http_client.aclose()
    asyncio.run(scenario())

def test_ingest_uses_fixed_endpoint_and_no_prompt_metadata(tmp_path):
    async def scenario():
        seen = {}
        async def handler(request):
            seen["path"] = request.url.path
            seen["body"] = __import__("json").loads(request.content)
            return httpx.Response(201, json={"created": True})
        item = adapter(tmp_path, handler)
        result = await item.ingest(source_event_id="operit:r1:user", role="user", content="你好")
        assert result["created"] is True
        assert seen == {"path": "/bridge/v1/events", "body": {"source_event_id": "operit:r1:user", "role": "user", "content": "你好"}}
        await item.http_client.aclose()
    asyncio.run(scenario())

def test_enabled_configuration_is_fail_closed(tmp_path):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    with pytest.raises(ValueError):
        CanonicalContinuationAdapter(
            enabled=True, base_url="", token="", device_id="", conversation_id="",
            channel_id="operit", state_db_path=tmp_path / "x.db", http_client=client,
        )


def test_first_pull_bootstraps_to_current_max_seq_without_replaying_history(tmp_path):
    async def scenario():
        calls = []
        async def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={
                "conversation_id": "conv-jiajia-main",
                "generation": False, "max_seq": 309,
            })
        item = adapter(tmp_path, handler)
        batch = await item.pull()
        assert batch.events == ()
        assert batch.through_seq == 309
        assert item.cursor() == 309
        assert calls == ["/bridge/v1/conversation"]
        await item.http_client.aclose()
    asyncio.run(scenario())


def test_failed_ingest_is_queued_and_later_flushed(tmp_path):
    async def scenario():
        failing = adapter(tmp_path, lambda request: httpx.Response(503))
        result = await failing.ingest_or_queue(
            source_event_id="operit:r2:user", role="user", content="稍后同步",
        )
        assert result["queued"] is True
        assert failing.outbox_size() == 1
        await failing.http_client.aclose()

        async def healthy_handler(request):
            return httpx.Response(201, json={"created": True})
        healthy = adapter(tmp_path, healthy_handler)
        status = await healthy.flush_outbox()
        assert status == {"delivered": 1, "pending": 0}
        await healthy.http_client.aclose()
    asyncio.run(scenario())
