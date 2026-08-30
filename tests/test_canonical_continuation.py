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
                    {"seq": 5, "id": "evt-reality-user", "event_type": "message", "role": "user", "content": "那我下午去。", "source": "app", "source_event_id": "reality:r5:user"},
                    {"seq": 6, "id": "evt-reality-assistant", "event_type": "message", "role": "assistant", "content": "别盯着针。", "source": "conversation-service", "source_event_id": "reality:r5:assistant"},
                    {"seq": 7, "event_type": "message", "role": "assistant", "content": "own",
                     "source": "ombre-gateway", "source_event_id": "operit:r9:assistant"},
                    {"seq": 8, "event_type": "message", "role": "tool", "content": "secret", "source": "app"},
                ],
            })
        item = adapter(tmp_path, handler)
        item.commit_cursor(3)
        batch = await item.pull()
        assert [(event["role"], event["content"]) for event in batch.events] == [
            ("user", "那我下午去。"), ("assistant", "别盯着针。"),
        ]
        assert [event["event_id"] for event in batch.events] == [
            "evt-reality-user", "evt-reality-assistant",
        ]
        assert [event["source_event_id"] for event in batch.events] == [
            "reality:r5:user", "reality:r5:assistant",
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


def test_pull_keeps_events_written_by_other_channels_through_same_bridge(tmp_path):
    """Every channel writes via the same bridge, so `source` is "ombre-gateway" for
    all of them. Only our own channel prefix may be filtered, otherwise cross-channel
    continuation silently degrades to an empty batch."""
    async def scenario():
        async def handler(request):
            return httpx.Response(200, json={
                "conversation_id": "conv-jiajia-main", "next_after_seq": 12,
                "items": [
                    {"seq": 11, "event_type": "message", "role": "user", "content": "Reality 那边说的",
                     "source": "ombre-gateway", "source_event_id": "reality:r1:user"},
                    {"seq": 12, "event_type": "message", "role": "assistant", "content": "Operit 自己说的",
                     "source": "ombre-gateway", "source_event_id": "operit:r1:assistant"},
                ],
            })
        item = adapter(tmp_path, handler, channel_id="operit")
        item.commit_cursor(10)
        batch = await item.pull()
        assert [(event["role"], event["content"]) for event in batch.events] == [
            ("user", "Reality 那边说的"),
        ]
        await item.http_client.aclose()
    asyncio.run(scenario())


def test_cursors_are_isolated_per_channel(tmp_path):
    """Each channel tracks its own read position in the shared ledger; a single
    shared cursor would let one channel consume events the other never sees."""
    async def scenario():
        item = adapter(tmp_path, lambda request: httpx.Response(200))
        assert item.commit_cursor(40, "operit") == 40
        assert item.commit_cursor(7, "reality") == 7
        assert item.cursor("operit") == 40
        assert item.cursor("reality") == 7
        assert item.cursor() == 40  # default channel is the configured one
        await item.http_client.aclose()
    asyncio.run(scenario())


def test_ingest_treats_409_duplicate_as_success(tmp_path):
    """409 is the bridge's idempotency guard, not a delivery failure.

    Treating it as an error made already-delivered outbox rows retry forever.
    """
    async def scenario():
        async def handler(request):
            return httpx.Response(409, json={"error": "duplicate_event"})
        item = adapter(tmp_path, handler)
        result = await item.ingest(
            source_event_id="operit:r3:user", role="user", content="hi",
        )
        assert result["duplicate"] is True
        assert result["created"] is False
        await item.http_client.aclose()
    asyncio.run(scenario())


def test_flush_outbox_skips_poisoned_row_and_keeps_draining(tmp_path):
    """One undeliverable event must not stall every event queued behind it."""
    async def scenario():
        async def handler(request):
            body = __import__("json").loads(request.content)
            if body["source_event_id"] == "operit:bad:user":
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"created": True})
        item = adapter(tmp_path, handler)
        item.queue_outbox(source_event_id="operit:bad:user", role="user", content="poison")
        item.queue_outbox(source_event_id="operit:good1:user", role="user", content="behind it")
        item.queue_outbox(source_event_id="operit:good2:user", role="user", content="also behind")
        assert item.outbox_size() == 3
        result = await item.flush_outbox()
        # The two healthy rows drain despite the poisoned one being first.
        assert result["delivered"] == 2
        assert result["pending"] == 1
        await item.http_client.aclose()
    asyncio.run(scenario())


def test_flush_outbox_skips_exhausted_legacy_rows_for_fresh_events(tmp_path):
    async def scenario():
        seen = []

        async def handler(request):
            body = __import__("json").loads(request.content)
            seen.append(body["source_event_id"])
            return httpx.Response(201, json={"created": True})

        item = adapter(tmp_path, handler)
        item.queue_outbox(
            source_event_id="reality:legacy-poison:user",
            role="user", content="old",
        )
        with item._connect() as conn:
            conn.execute(
                "UPDATE canonical_outbox SET attempts=2603,last_error=? "
                "WHERE source_event_id=?",
                ("HTTPStatusError", "reality:legacy-poison:user"),
            )
        item.queue_outbox(
            source_event_id="operit:fresh:user", role="user", content="fresh",
        )
        result = await item.flush_outbox()
        assert result["delivered"] == 1
        assert seen == ["operit:fresh:user"]
        assert item.outbox_size() == 1
        await item.http_client.aclose()

    asyncio.run(scenario())
