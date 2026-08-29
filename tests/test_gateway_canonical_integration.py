import asyncio
from types import SimpleNamespace

from starlette.requests import Request

from canonical_continuation import ContinuationBatch
from gateway import GatewayService


class FakeAdapter:
    def __init__(self):
        self.enabled = True
        self.ingested = []
        self.committed = []
        self.pull_calls = 0
        self.flush_calls = 0
        self.pull_channels = []

    def cursor(self, channel_id=None):
        return 10

    async def flush_outbox(self):
        self.flush_calls += 1
        return {"delivered": 0, "pending": 0}

    async def pull(self, channel_id=None):
        self.pull_calls += 1
        self.pull_channels.append(channel_id)
        return ContinuationBatch((
            {"seq": 11, "role": "user", "content": "那我下午去。", "source_event_id": "app:11"},
            {"seq": 12, "role": "assistant", "content": "别盯着针。", "source_event_id": "app:12"},
        ), 12)

    @staticmethod
    def merge_messages(messages, batch):
        return GatewayMerge.merge(messages, batch)

    async def ingest_or_queue(self, **values):
        self.ingested.append(values)
        return {"created": True}

    def commit_cursor(self, seq, channel_id=None):
        self.committed.append((seq, channel_id) if channel_id else seq)
        return seq


class GatewayMerge:
    @staticmethod
    def merge(messages, batch):
        result = [dict(item) for item in messages]
        index = len(result) - 1
        inserted = [{"role": event["role"], "content": event["content"]} for event in batch.events]
        return result[:index] + inserted + result[index:]


def request(headers=None):
    pairs = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": pairs})


def service(profile="jiajia-main"):
    item = GatewayService.__new__(GatewayService)
    item.persona_engine = SimpleNamespace(profile_id=profile)
    item.canonical_target_session_id = "jiajia"
    item.canonical_target_profile_id = "jiajia-main"
    item.canonical_session_channels = {"jiajia": "operit", "main": "reality"}
    item.canonical_client_channels = {
        "operit": "operit", "reality": "reality", "telegram": "telegram",
    }
    item.canonical_adapter = FakeAdapter()
    return item


def test_target_turn_merges_real_roles_and_ingests_only_current_operit_user():
    async def scenario():
        item = service()
        payload = {
            "model": "guyan",
            "messages": [
                {"role": "system", "content": "editable role card"},
                {"role": "user", "content": "但不会很疼吧？"},
            ],
        }
        prepared, state = await item._prepare_canonical_turn(
            request({"X-Request-ID": "req-123"}), payload, "jiajia", "但不会很疼吧？",
        )
        assert prepared["messages"] == [
            {"role": "system", "content": "editable role card"},
            {"role": "user", "content": "那我下午去。"},
            {"role": "assistant", "content": "别盯着针。"},
            {"role": "user", "content": "但不会很疼吧？"},
        ]
        assert [message for message in prepared["messages"] if message["role"] == "system"] == [
            {"role": "system", "content": "editable role card"}
        ]
        assert item.canonical_adapter.ingested == [{
            "source_event_id": "operit:req-123:user", "role": "user",
            "content": "但不会很疼吧？",
        }]
        assert state["status"] == "injected"
        assert state["through_seq"] == 12
        assert item.canonical_adapter.committed == []
    asyncio.run(scenario())


def test_non_target_session_does_not_touch_adapter():
    async def scenario():
        item = service()
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        prepared, state = await item._prepare_canonical_turn(request(), payload, "other", "hello")
        assert prepared is payload
        assert state == {"enabled": False, "status": "not_target"}
        assert item.canonical_adapter.pull_calls == 0
        assert item.canonical_adapter.ingested == []
    asyncio.run(scenario())


def test_successful_assistant_is_ingested_then_cursor_commits():
    async def scenario():
        item = service()
        state = {
            "enabled": True, "status": "injected", "request_id": "req-123",
            "user_source_event_id": "operit:req-123:user", "through_seq": 12,
            "channel_id": "operit",
        }
        await item._finalize_canonical_turn(state, {"role": "assistant", "content": "通常就一下。"})
        assert item.canonical_adapter.ingested == [{
            "source_event_id": "operit:req-123:assistant", "role": "assistant",
            "content": "通常就一下。", "correlation_id": "operit:req-123:user",
        }]
        assert item.canonical_adapter.committed == [(12, "operit")]
        assert state["assistant_write_status"] == "created"
    asyncio.run(scenario())


def test_no_assistant_output_does_not_commit_cursor():
    async def scenario():
        item = service()
        state = {"enabled": True, "status": "injected", "request_id": "req", "through_seq": 12}
        await item._finalize_canonical_turn(state, None)
        assert item.canonical_adapter.ingested == []
        assert item.canonical_adapter.committed == []
        assert state["assistant_write_status"] == "skipped_no_output"
    asyncio.run(scenario())


def test_second_channel_uses_its_own_prefix_and_cursor():
    """A client that falls back to default_session_id ("main") must still take part,
    writing under its own channel prefix so the other channel can read it back."""
    async def scenario():
        item = service()
        payload = {"messages": [{"role": "user", "content": "在网页这边说的"}]}
        prepared, state = await item._prepare_canonical_turn(
            request({"X-Request-ID": "req-9"}), payload, "main", "在网页这边说的",
        )
        assert state["enabled"] is True
        assert state["channel_id"] == "reality"
        assert item.canonical_adapter.pull_channels == ["reality"]
        assert item.canonical_adapter.ingested == [{
            "source_event_id": "reality:req-9:user", "role": "user",
            "content": "在网页这边说的",
        }]
        await item._finalize_canonical_turn(state, {"role": "assistant", "content": "嗯。"})
        assert item.canonical_adapter.ingested[-1]["source_event_id"] == "reality:req-9:assistant"
        assert item.canonical_adapter.committed == [(12, "reality")]
    asyncio.run(scenario())


def test_shared_session_uses_client_marker_for_a_separate_continuation_cursor():
    """Reality and Operit share jiajia-main, but must not share a read cursor.

    The marker is cursor selection only: the session remains the caller's
    canonical identity and is still what activates the continuation bridge.
    """
    async def scenario():
        item = service()
        payload = {"messages": [{"role": "user", "content": "从 Reality 继续"}]}
        _prepared, state = await item._prepare_canonical_turn(
            request({"X-Request-ID": "reality-1", "X-Ombre-Client-Id": "reality"}),
            payload, "jiajia", "从 Reality 继续",
        )
        assert state["channel_id"] == "reality"
        assert state["channel_selection"] == "client_marker"
        assert state["client_marker"] == "reality"
        assert item.canonical_adapter.pull_channels == ["reality"]
        assert item.canonical_adapter.ingested[0]["source_event_id"] == "reality:reality-1:user"
    asyncio.run(scenario())


def test_unmapped_session_is_still_not_target():
    async def scenario():
        item = service()
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        prepared, state = await item._prepare_canonical_turn(request(), payload, "stranger", "hello")
        assert state == {"enabled": False, "status": "not_target"}
        assert item.canonical_adapter.pull_calls == 0
    asyncio.run(scenario())


def test_continuation_presence_is_based_on_inserted_canonical_events():
    debug = {"post_injection_presence": {"canonical_continuation": True}}

    GatewayService._attach_canonical_trace_debug(
        debug, {"enabled": True, "status": "deduped", "event_count": 0}
    )
    assert debug["post_injection_presence"]["canonical_continuation"] is False

    GatewayService._attach_canonical_trace_debug(
        debug, {"enabled": True, "status": "injected", "event_count": 1}
    )
    assert debug["post_injection_presence"]["canonical_continuation"] is True


def test_coverage_dedup_is_exact_and_same_text_different_event_survives():
    batch = ContinuationBatch((
        {"seq": 11, "event_id": "evt-covered", "version_id": "v1",
         "source_event_id": "app:covered", "role": "user", "content": "相同文字"},
        {"seq": 12, "event_id": "evt-distinct", "version_id": "v2",
         "source_event_id": "telegram:distinct", "role": "user", "content": "相同文字"},
    ), 12)
    filtered, debug = GatewayService._dedupe_canonical_batch(batch, {
        "conversation_id": "conv-1",
        "context_revision": 7,
        "items": [{
            "event_id": "evt-covered", "version_id": "v1",
            "source_event_id": "app:covered",
        }],
        "fingerprint": "coverage-fp",
    })
    assert [event["event_id"] for event in filtered.events] == ["evt-distinct"]
    assert debug["coverage_fingerprint"] == "coverage-fp"
    assert debug["deduped"][0]["dedup_reason"] == "coverage_exact_key"
    assert debug["inserted_missing"][0]["event_id"] == "evt-distinct"
    assert debug["legacy_text_fallback_used"] is False


def test_client_marker_is_diagnostic_and_does_not_replace_session_identity():
    item = service()
    trace = item._trace_context_from_request(
        request({"X-Ombre-Client-Id": "telegram"}),
        session_id="jiajia-main", request_type="initial", client_id="telegram",
    )
    assert trace["client_id"] == "telegram"
    assert trace["session_id"] == "jiajia-main"
