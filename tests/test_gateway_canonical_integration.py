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

    def cursor(self):
        return 10

    async def flush_outbox(self):
        self.flush_calls += 1
        return {"delivered": 0, "pending": 0}

    async def pull(self):
        self.pull_calls += 1
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

    def commit_cursor(self, seq):
        self.committed.append(seq)
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
        }
        await item._finalize_canonical_turn(state, {"role": "assistant", "content": "通常就一下。"})
        assert item.canonical_adapter.ingested == [{
            "source_event_id": "operit:req-123:assistant", "role": "assistant",
            "content": "通常就一下。", "correlation_id": "operit:req-123:user",
        }]
        assert item.canonical_adapter.committed == [12]
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
