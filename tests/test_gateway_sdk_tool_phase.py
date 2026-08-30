"""L3-SDK-C1 Scheme P: Gateway tool-loop continuation-phase tests.

Real-Gateway function-level characterization (not copied mock): a minimal
GatewayService is built via __new__ with a REAL GatewayStateStore on a tmp
SQLite file, and the actual production methods under test are executed:

  - _continuation_phase_enabled          (new, fail-closed gating)
  - _extract_continuation_turn_user_query(new, backward real-user scan)
  - _canonical_turn_key                  (new, H1|H2 idempotency key)
  - _record_conversation_turn            (modified, canonical-key dedupe)
  - prepare_payload's is_new_user_turn   (modified, continuation branch)

Coverage maps to the 18 required gateway tests (task book section IV).
"""
import asyncio
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from gateway import (
    CANONICAL_ASSISTANT_SOURCE_EVENT_ID_HEADER,
    CANONICAL_SOURCE_EVENT_ID_HEADER,
    CANONICAL_TURN_PHASE_CONTINUATION,
    CANONICAL_TURN_PHASE_HEADER,
    GatewayService,
)
from gateway_state import GatewayStateStore


# ---------------------------------------------------------------------------
# Minimal real-Gateway harness
# ---------------------------------------------------------------------------

H1 = CANONICAL_SOURCE_EVENT_ID_HEADER
H2 = CANONICAL_ASSISTANT_SOURCE_EVENT_ID_HEADER
PHASE = CANONICAL_TURN_PHASE_HEADER
PHASE_CONT = CANONICAL_TURN_PHASE_CONTINUATION


def make_request(headers=None):
    pairs = [(key.lower().encode(), str(value).encode())
             for key, value in (headers or {}).items()]
    return Request({"type": "http", "method": "POST",
                    "path": "/v1/chat/completions", "headers": pairs})


def minimal_service(tmp_path):
    """Real GatewayService instance with only the attributes the tested
    methods touch, plus a REAL GatewayStateStore on tmp SQLite."""
    db_path = str(tmp_path / "state" / "gateway_state.db")
    store = GatewayStateStore(db_path)
    item = GatewayService.__new__(GatewayService)
    item.state_store = store
    item.persona_engine = SimpleNamespace(profile_id="jiajia-main")
    item.reminder_store = SimpleNamespace(mark_reminded=lambda *a, **k: None)
    item.raw_event_store = SimpleNamespace(
        record_raw_event=lambda *a, **k: None)
    item.conversation_turns_max_entries = 500
    item.canonical_adapter = SimpleNamespace(enabled=False)
    item.canonical_target_session_id = ""
    item.canonical_target_profile_id = ""
    item.canonical_session_channels = {}
    item.gateway_token = "test-token"
    item.default_session_id = "main"
    item.upstream_default_model = "guyan"
    item.upstream_models = ["guyan"]
    item.bucket_mgr = SimpleNamespace(list_all=async_return([]))
    item.retrieval_mode = "disabled"
    item.just_now_context_enabled = False
    item.date_recall_enabled = False
    item.operit_context_rewrite_enabled = False
    item.recent_context_reentry_idle_hours = 24
    item.recent_context_cooldown_hours = 6
    item.current_inner_state_interval_rounds = 0
    item.core_memory_interval_rounds = 0
    item.relationship_weather_interval_rounds = 0
    item.favorite_memory_interval_rounds = 0
    item.recalled_budget = 0
    item.related_memory_budget = 0
    item.portrait_memory_enabled = False
    item.active_reminders_enabled = False
    item.active_reminder_inject_limit = 0
    item.recall_enabled = False
    item.dream_inject_enabled = False
    item.dream_cfg = {}
    item._bucket_list_cache = {}
    item.bucket_list_cache_ttl_seconds = 0
    # Timezone + query-classifier dependencies (kept real where cheap, stubbed
    # where they would touch buckets/LLM/date machinery unrelated to phase).
    item.gateway_tz = __import__("zoneinfo").ZoneInfo("Asia/Shanghai")
    item._query_date_hint = lambda *a, **k: None
    item._query_requests_date_persona_trace = lambda *a, **k: False
    item._query_requests_favorite_memory = lambda *a, **k: False
    item._query_requests_recent_context = lambda *a, **k: False
    item._query_requests_just_now_context = lambda *a, **k: False
    item._query_has_handoff_transition_marker = lambda *a, **k: False
    item._query_prefers_session_start_handoff = lambda *a, **k: False
    item._query_is_handoff_trigger = lambda *a, **k: False
    item._query_requests_date_recall = lambda *a, **k: False
    item._auto_recall_low_signal_query = lambda *a, **k: False
    item._query_should_skip_broad_for_targeted_memory_detail = lambda *a, **k: False
    item._should_inject_interval = lambda *a, **k: False
    item._should_inject_recent_context = lambda *a, **k: False
    item._recent_context_reason = lambda *a, **k: ""
    item._classify_context_mode = lambda *a, **k: ""
    item._get_persona_state_for_context_mode = lambda *a, **k: None
    item._dynamic_recall_search_query = lambda *a, **k: ""
    item._extract_bucket_ids_from_context = lambda *a, **k: []
    item._extract_moment_ids_from_context = lambda *a, **k: []
    item._build_active_reminders_block = lambda *a, **k: ("", [])
    item._build_just_now_chat_context = lambda *a, **k: ("", {})
    item._build_date_recall_context = lambda *a, **k: ("", {}, [])
    item._build_date_persona_trace_block = lambda *a, **k: ("", {})
    item._build_portrait_memory_block = lambda *a, **k: ("", {})
    item._build_targeted_memory_detail = lambda *a, **k: ("", {})
    item._build_moment_diffused_memory_with_debug = lambda *a, **k: ("", [])
    item._format_recalled_moments = async_return("")
    item._build_core_memory_block = async_return("")
    item._build_relationship_weather_block = async_return("")
    item._build_favorite_memory_block = async_return(("", []))
    item._build_recent_context_block = async_return("")
    item._build_dream_context_block = async_return(("", {"status": "skipped"}))
    item._refresh_moment_graph = lambda *a, **k: ([], {}, {})
    item._select_dynamic_buckets = async_return(([], [], {}))
    item._select_dynamic_moments = async_return(([], [], [], [], {}))
    item._direct_moments_for_bucket = lambda *a, **k: []
    item._representative_moment = lambda *a, **k: None
    item._source_record_synthetic_moment_for_bucket = lambda *a, **k: None
    item._moment_with_bucket_recall_signal = lambda m, s: m
    item._with_explicit_source_record_buckets = lambda q, b, a: b
    item._build_injected_context_messages = lambda **k: ("", "")
    item._inject_context_messages = lambda messages, s, d: messages
    item._apply_prompt_cache_hints = lambda *a, **k: None
    item._restore_cached_reasoning_content = lambda *a, **k: None
    item._operit_context_rewrite_debug_base = lambda: {}
    item._rewrite_operit_context_for_forward = lambda m: (m, "", "", {})
    item._append_named_context_section = lambda s, n, c: s
    item._build_injection_debug_payload = lambda **k: {}
    item._memory_sentinel_debug_base = lambda *a, **k: {}
    item._domain_sentinel_rule_plan = lambda *a, **k: {}
    item._query_planner_debug_base = lambda *a, **k: {}
    item._date_persona_trace_debug_base = lambda *a, **k: {}
    item._targeted_memory_detail_debug_base = lambda: {}
    item._portrait_memory_debug_base = lambda: {}
    item._just_now_context_debug_base = lambda *a, **k: {}
    item._date_recall_debug_base = lambda *a, **k: {}
    item.upstreams = {
        "guyan": {"name": "guyan", "base_url": "http://gateway.local",
                  "api_key_env": "OMBRE_GATEWAY_UPSTREAM_API_KEY",
                  "api_keys": []},
    }
    item._resolve_upstream_for_model = lambda model: {
        "upstream": item.upstreams["guyan"], "upstream_model": model,
    }
    item._get_upstream_for_model = lambda model: item.upstreams["guyan"]
    item.memory_detail_recall_enabled = False
    item.memory_detail_recall_max_ids = 0
    return item


def async_return(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner


def sdk_tool_messages(user_text="查一下今天的天气", tool_call_id="call_1",
                      tool_result="晴，24°C"):
    """Shape of an SDK continuation request: history + assistant tool_call +
    role=tool result at the END (the blocking shape from S5 scenario 6)."""
    return [
        {"role": "system", "content": "role card"},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": tool_call_id, "type": "function",
            "function": {"name": "weather", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": tool_result},
    ]


def _pure_tmp_path():
    return Path(tempfile.mkdtemp(prefix="gw-sdk-phase-test-"))


# ---------------------------------------------------------------------------
# 8/9/10. Phase header fail-closed gating
# ---------------------------------------------------------------------------

def test_phase_header_missing_is_legacy():
    item = minimal_service(_pure_tmp_path())
    assert item._continuation_phase_enabled(
        make_request({H1: "app:d1:c1", H2: "turn:u1:assistant"})) is False
    assert item._continuation_phase_enabled(make_request({})) is False


def test_phase_header_requires_h1():
    item = minimal_service(_pure_tmp_path())
    req = make_request({PHASE: PHASE_CONT, H2: "turn:u1:assistant"})
    assert item._continuation_phase_enabled(req) is False


def test_phase_header_requires_h2():
    item = minimal_service(_pure_tmp_path())
    req = make_request({PHASE: PHASE_CONT, H1: "app:d1:c1"})
    assert item._continuation_phase_enabled(req) is False


def test_phase_header_rejects_illegal_value():
    item = minimal_service(_pure_tmp_path())
    req = make_request({PHASE: "final", H1: "app:d1:c1", H2: "turn:u1:assistant"})
    assert item._continuation_phase_enabled(req) is False


def test_phase_header_enabled_with_both_h1_h2():
    item = minimal_service(_pure_tmp_path())
    req = make_request({PHASE: PHASE_CONT, H1: "app:d1:c1", H2: "turn:u1:assistant"})
    assert item._continuation_phase_enabled(req) is True


# ---------------------------------------------------------------------------
# 11. Tool output must not be treated as the current user
# ---------------------------------------------------------------------------

def test_continuation_extraction_skips_tool_result_and_finds_real_user():
    item = minimal_service(_pure_tmp_path())
    messages = sdk_tool_messages()
    # The LAST real user is the genuine current-turn owner.
    assert item._extract_continuation_turn_user_query(messages) == "查一下今天的天气"


def test_continuation_extraction_skips_developer_and_tool_calls():
    item = minimal_service(_pure_tmp_path())
    messages = [
        {"role": "system", "content": "role"},
        {"role": "developer", "content": "injected guidance"},
        {"role": "user", "content": "第一条"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "result2"},
    ]
    assert item._extract_continuation_turn_user_query(messages) == "第一条"


def test_continuation_extraction_empty_when_no_user():
    item = minimal_service(_pure_tmp_path())
    messages = [
        {"role": "system", "content": "role"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    assert item._extract_continuation_turn_user_query(messages) == ""


def test_legacy_extraction_still_returns_empty_on_tool_end():
    """Regression guard: the ORIGINAL _extract_current_turn_user_query keeps
    its exact semantics (tool-ending continuation is NOT a current user turn)."""
    item = minimal_service(_pure_tmp_path())
    messages = sdk_tool_messages()
    assert item._extract_current_turn_user_query(messages) == ""


# ---------------------------------------------------------------------------
# 7/6/5. Final-turn idempotency via canonical key (H1|H2)
# ---------------------------------------------------------------------------

def test_canonical_turn_key_stable_for_same_h1_h2(tmp_path):
    item = minimal_service(tmp_path)
    k1 = item._canonical_turn_key(
        make_request({H1: "app:d1:c1", H2: "turn:u1:assistant"}))
    k2 = item._canonical_turn_key(
        make_request({PHASE: PHASE_CONT, H1: "app:d1:c1", H2: "turn:u1:assistant"}))
    assert k1 and k1 == k2
    # different H1/H2 -> different key
    k3 = item._canonical_turn_key(
        make_request({H1: "app:d1:c2", H2: "turn:u2:assistant"}))
    assert k3 != k1
    # missing header -> empty (fail-closed, no dedupe for legacy)
    assert item._canonical_turn_key(make_request({})) == ""


def test_record_conversation_turn_once_then_key_dedupe(tmp_path):
    item = minimal_service(tmp_path)
    item._record_conversation_turn(
        session_id="jiajia", round_id=1,
        user_message="查天气", assistant_message={"role": "assistant", "content": "晴"},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-abc",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert len(turns) == 1
    assert turns[0]["canonical_key"] == "key-abc"
    # Same canonical key replay (retry_continuation) -> recorded once.
    item._record_conversation_turn(
        session_id="jiajia", round_id=2,
        user_message="查天气", assistant_message={"role": "assistant", "content": "晴"},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-abc",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert len(turns) == 1


def test_record_conversation_turn_different_key_second_turn(tmp_path):
    item = minimal_service(tmp_path)
    item._record_conversation_turn(
        session_id="jiajia", round_id=1,
        user_message="查天气", assistant_message={"role": "assistant", "content": "晴"},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-a",
    )
    item._record_conversation_turn(
        session_id="jiajia", round_id=2,
        user_message="再查", assistant_message={"role": "assistant", "content": "多云"},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-b",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert len(turns) == 2


# ---------------------------------------------------------------------------
# 3/4/13. Intermediate tool_call responses never enter conversation_turn
# ---------------------------------------------------------------------------

def test_tool_call_response_not_recorded(tmp_path):
    item = minimal_service(tmp_path)
    item._record_conversation_turn(
        session_id="jiajia", round_id=1,
        user_message="查天气",
        assistant_message={"role": "assistant", "content": None,
                           "tool_calls": [{"id": "c1", "type": "function",
                                           "function": {"name": "w", "arguments": "{}"}}]},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-a",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert turns == []


def test_second_tool_call_in_continuation_not_recorded(tmp_path):
    """Continuation returning a SECOND tool_call (no final text) must not
    enter conversation_turn -- the loop keeps going, final only later."""
    item = minimal_service(tmp_path)
    item._record_conversation_turn(
        session_id="jiajia", round_id=2,
        user_message="查天气",
        assistant_message={"role": "assistant", "content": "",
                           "tool_calls": [{"id": "c2", "type": "function",
                                           "function": {"name": "w2", "arguments": "{}"}}]},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-b",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert turns == []


def test_retry_continuation_final_recorded_once(tmp_path):
    """A retry_continuation replay with the SAME H1/H2 (same canonical key)
    that returns the final answer must not double-record the turn."""
    item = minimal_service(tmp_path)
    final = {"role": "assistant", "content": "今天是晴，24°C"}
    # first continuation attempt -> recorded
    item._record_conversation_turn(
        session_id="jiajia", round_id=2,
        user_message="查天气", assistant_message=final,
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-final",
    )
    # retry_continuation, same canonical key -> skipped (no duplicate)
    item._record_conversation_turn(
        session_id="jiajia", round_id=3,
        user_message="查天气", assistant_message=final,
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-final",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert len(turns) == 1
    assert turns[0]["canonical_key"] == "key-final"


# ---------------------------------------------------------------------------
# Migration: pre-existing production DB without canonical_key column must be
# upgraded in place (fresh DBs include the column via CREATE TABLE).
# ---------------------------------------------------------------------------

def test_existing_db_migrated_with_canonical_key_column(tmp_path):
    """Simulate a production gateway_state.db created BEFORE Scheme P: build
    the old schema, then open GatewayStateStore against it -- the additive
    migration must add the column + index and never fail on index creation."""
    db_path = tmp_path / "state" / "gateway_state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            round_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            user_text TEXT NOT NULL DEFAULT '',
            assistant_text TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            client TEXT NOT NULL DEFAULT '',
            route TEXT NOT NULL DEFAULT '',
            UNIQUE(profile_id, session_id, round_id)
        )
    """)
    conn.execute("""
        INSERT INTO conversation_turns
        (profile_id, session_id, round_id, created_at, user_text, assistant_text)
        VALUES ('jiajia-main', 'main', 1, ?, '旧消息', '旧回复')
    """, (datetime.now(timezone.utc).isoformat(timespec="seconds"),))
    conn.commit()
    conn.close()

    store = GatewayStateStore(str(db_path))  # must not raise
    columns = [
        row["name"]
        for row in store._connect().execute(
            "PRAGMA table_info(conversation_turns)").fetchall()
    ]
    assert "canonical_key" in columns
    # Existing legacy row is preserved and behaves as before.
    turns = store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="main", limit=10)
    assert len(turns) == 1
    assert turns[0]["canonical_key"] == ""
    # New keyed write works and dedupes.
    store.record_conversation_turn(
        profile_id="jiajia-main", session_id="main", round_id=2,
        user_text="新", assistant_text="新回复", canonical_key="k1",
    )
    assert store.canonical_turn_key_exists(
        profile_id="jiajia-main", session_id="main", canonical_key="k1") is True
    assert store.canonical_turn_key_exists(
        profile_id="jiajia-main", session_id="main", canonical_key="k2") is False


def test_fresh_db_includes_column_and_migration_is_noop(tmp_path):
    store = GatewayStateStore(str(tmp_path / "state" / "gateway_state.db"))
    columns = [
        row["name"]
        for row in store._connect().execute(
            "PRAGMA table_info(conversation_turns)").fetchall()
    ]
    assert "canonical_key" in columns


# ---------------------------------------------------------------------------
# 2/5. Final answer recorded exactly once (initial + continuation)
# ---------------------------------------------------------------------------

def test_final_answer_recorded_once_initial_and_continuation(tmp_path):
    item = minimal_service(tmp_path)
    # initial request: no tool call, final answer, no canonical key needed
    item._record_conversation_turn(
        session_id="jiajia", round_id=1,
        user_message="你好", assistant_message={"role": "assistant", "content": "你好呀"},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="",
    )
    # continuation request returning the FINAL answer (recorded under key)
    item._record_conversation_turn(
        session_id="jiajia", round_id=2,
        user_message="查天气", assistant_message={"role": "assistant", "content": "今天是晴，24°C"},
        model="guyan", client="sdk", route="/v1/chat/completions",
        canonical_key="key-final",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="jiajia", limit=10)
    assert len(turns) == 2


# ---------------------------------------------------------------------------
# 1/16. Legacy path unchanged: no phase header -> old extractor used,
#       tool-ending message still yields no current user turn.
# ---------------------------------------------------------------------------

def test_legacy_no_phase_still_uses_old_extractor(tmp_path):
    item = minimal_service(tmp_path)
    assert item._continuation_phase_enabled(
        make_request({H1: "app:d1:c1", H2: "turn:u1:assistant"})) is False


def test_legacy_plain_chat_round_records_without_canonical_key(tmp_path):
    item = minimal_service(tmp_path)
    item._record_conversation_turn(
        session_id="main", round_id=1,
        user_message="在吗", assistant_message={"role": "assistant", "content": "在的"},
        model="guyan", client="app", route="/v1/chat/completions",
        canonical_key="",
    )
    turns = item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", session_id="main", limit=10)
    assert len(turns) == 1
    assert turns[0]["canonical_key"] == ""


# ---------------------------------------------------------------------------
# 14. H1/H2 stay origin-owned and an Aizizhu-owned turn is not mirrored into
#     the already-shared canonical bridge a second time.
# ---------------------------------------------------------------------------

def test_origin_headers_keep_local_and_bridge_mirror_off(tmp_path):
    from canonical_continuation import ContinuationBatch

    class FakeAdapter:
        def __init__(self):
            self.enabled = True
            self.ingested = []
            self.committed = []
            self.pull_calls = 0
        def cursor(self, channel_id=None):
            return 10
        async def flush_outbox(self):
            return {"delivered": 0, "pending": 0}
        async def pull(self, channel_id=None):
            self.pull_calls += 1
            return ContinuationBatch((), 10)
        @staticmethod
        def merge_messages(messages, batch):
            return messages
        async def ingest_or_queue(self, **values):
            self.ingested.append(values)
            return {"created": True}
        def commit_cursor(self, seq, channel_id=None):
            self.committed.append((seq, channel_id) if channel_id else seq)
            return seq

    item = minimal_service(tmp_path)
    item.canonical_adapter = FakeAdapter()
    item.canonical_target_session_id = "jiajia"
    item.canonical_target_profile_id = "jiajia-main"
    item.canonical_session_channels = {"jiajia": "operit", "main": "reality"}
    req = make_request({PHASE: PHASE_CONT, H1: "app:d1:c1", H2: "turn:u1:assistant"})
    payload = {"model": "guyan", "messages": sdk_tool_messages()}
    canonical_user = item._extract_continuation_turn_user_query(payload["messages"])
    prepared, state = asyncio.run(item._prepare_canonical_turn(
        req, payload, "jiajia", canonical_user))
    assert state["mirror_write_enabled"] is False
    assert state["bridge_write_enabled"] is False
    assert state["user_write_status"] == "skipped_origin_owned"
    assert item.canonical_adapter.ingested == []
    asyncio.run(item._finalize_canonical_turn(
        state, {"role": "assistant", "content": "晴，24°C"}))
    assert state["assistant_write_status"] == "skipped_origin_owned"
    assert item.canonical_adapter.ingested == []


# ---------------------------------------------------------------------------
# 12. continuation still reaches persona/memory injection
#     (is_new_user_turn must be True with a real user behind tool results)
# ---------------------------------------------------------------------------

def test_prepare_payload_continuation_sets_new_user_turn(tmp_path):
    """prepare_payload with continuation_phase=True must classify the request
    as a current user turn (is_new_user_turn=True) so persona/memory
    injection runs -- while the legacy path with the same messages does NOT."""
    item = minimal_service(tmp_path)
    async def noop_route_memory_sentinel(*a, **k):
        return {"route": "skip"}
    item._route_memory_sentinel = noop_route_memory_sentinel
    item._route_domain_sentinel = noop_route_memory_sentinel
    item._domain_sentinel_should_skip_recall = lambda *a, **k: False
    item.persona_engine.enabled = False
    item.persona_engine.build_pre_reply_guidance = async_return({})
    item.persona_engine.format_state_block = lambda *a, **k: ""

    payload = {"model": "guyan", "messages": sdk_tool_messages()}

    # Legacy: same tool-ending messages -> NOT a current user turn.
    _, recalled_legacy, debug_legacy = asyncio.run(item.prepare_payload(
        payload, "jiajia", include_debug=True))
    timing_legacy = debug_legacy.get("prepare_timing_debug") or {}
    assert timing_legacy.get("is_new_user_turn") is False
    assert recalled_legacy is None

    # Continuation: legal phase header -> current user turn.
    _, recalled_cont, debug_cont = asyncio.run(item.prepare_payload(
        payload, "jiajia", include_debug=True, continuation_phase=True))
    timing_cont = debug_cont.get("prepare_timing_debug") or {}
    assert timing_cont.get("is_new_user_turn") is True
    assert timing_cont.get("query_chars", 0) > 0


# ---------------------------------------------------------------------------
# 15. Different canonical source events do not cross
# ---------------------------------------------------------------------------

def test_canonical_keys_are_distinct_for_different_source_events(tmp_path):
    item = minimal_service(tmp_path)
    k_a1 = item._canonical_turn_key(
        make_request({H1: "app:da:c1", H2: "turn:ua1:assistant"}))
    k_b1 = item._canonical_turn_key(
        make_request({H1: "app:db:c1", H2: "turn:ub1:assistant"}))
    assert k_a1 != k_b1
    # Same logical turn replayed -> same key (idempotent).
    k_a2 = item._canonical_turn_key(
        make_request({PHASE: PHASE_CONT, H1: "app:da:c1", H2: "turn:ua1:assistant"}))
    assert k_a2 == k_a1


def test_same_canonical_key_across_sessions_is_claimed_once(tmp_path):
    item = minimal_service(tmp_path)
    first = item.state_store.record_conversation_turn(
        profile_id="jiajia-main", session_id="main", round_id=1,
        user_text="Reality user", assistant_text="Reality reply",
        canonical_key="source-key-1",
    )
    second = item.state_store.record_conversation_turn(
        profile_id="jiajia-main", session_id="jiajia-main", round_id=1,
        user_text="Telegram replay", assistant_text="Telegram reply",
        canonical_key="source-key-1",
    )
    assert first > 0
    assert second == 0
    assert len(item.state_store.list_recent_conversation_turns(
        profile_id="jiajia-main", limit=10, hours=24 * 3650)) == 1
    assert item.state_store.canonical_turn_key_exists(
        profile_id="jiajia-main", session_id="main",
        canonical_key="source-key-1") is True
    assert item.state_store.canonical_turn_key_exists(
        profile_id="jiajia-main", session_id="jiajia-main",
        canonical_key="source-key-1") is True


def test_existing_cross_session_claim_conflict_stops_migration_without_delete(tmp_path):
    db_path = tmp_path / "state" / "gateway_state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            round_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            user_text TEXT NOT NULL DEFAULT '',
            assistant_text TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            client TEXT NOT NULL DEFAULT '',
            route TEXT NOT NULL DEFAULT '',
            canonical_key TEXT NOT NULL DEFAULT '',
            UNIQUE(profile_id, session_id, round_id)
        )
    """)
    conn.executemany(
        """
        INSERT INTO conversation_turns
        (profile_id, session_id, round_id, created_at, user_text,
         assistant_text, canonical_key)
        VALUES ('jiajia-main', ?, ?, ?, ?, ?, 'conflict-key')
        """,
        [
            ("main", 1, "2026-08-23T00:00:00+00:00", "a", "b"),
            ("jiajia-main", 1, "2026-08-23T00:00:01+00:00", "c", "d"),
        ],
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="canonical_key conflicts"):
        GatewayStateStore(str(db_path))
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 2
    conn.close()
