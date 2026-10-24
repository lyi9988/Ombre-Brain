"""Offline real Gateway prepare/Composer and canonical bridge consumer tests.

The supplied validation venv has unittest but no pytest. Reuse only the existing
network-free fixture functions through AST loading; production methods and
assertions below execute normally, without a fake pytest module.
"""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest

import httpx
from starlette.requests import Request

from canonical_continuation import CanonicalContinuationAdapter
from gateway import GatewayService, INNER_STATE_SCHEMA_HEADER
from gateway_state import GatewayStateStore
from memory_affinity import inner_state_snapshots, read_transport, resolve_signals, statement_affinity
from prompt_plan_mirror import PromptPlanMirrorStore


def _load_fixture_functions(filename, names):
    source = Path(__file__).with_name(filename)
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("required existing fixture changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), globals())


_load_fixture_functions("test_gateway_sdk_tool_phase.py", {"minimal_service", "_snapshot_service", "make_request", "async_return"})
_load_fixture_functions("test_prompt_context_placements_r1.py", {"_block", "_plan"})
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "inner-state-context-v1.json").read_text(encoding="utf-8"))


def _binding_settings():
    return {"recall": {"signal_bindings": [
        {"id": "care", "producer": "ombre.persona", "key": "protective_drive", "max_age_seconds": 3600},
        {"id": "energy", "producer": "aiz.needs", "key": "energy", "max_age_seconds": 3600}],
        "affinity_mappings": [{"id": "care_topic", "selector": {"topic": "care"},
            "signal_id": "care", "target": "drive_affinity", "weight": .8}]},
        "catalog": {"topics": ["care"], "definitions": []}}


class UnifiedProjectionTests(unittest.TestCase):
    def test_direct_projection_matches_actual_aiz_transport_and_unchanged_affinity(self):
        raw = FIXTURE["bridge"]["inner_state"]
        before = deepcopy(raw)
        projected = inner_state_snapshots(raw)
        expected = [item for item in json.loads(FIXTURE["recall_header"])["snapshots"]
                    if item["producer"] in {"aiz.needs", "ombre.persona"}]
        self.assertEqual(projected, expected)
        self.assertEqual(raw, before)
        self.assertEqual(projected[0]["checkpoint"], projected[1]["checkpoint"])
        now = datetime.fromisoformat(raw["settled_at"]) + timedelta(minutes=5)
        resolved = resolve_signals(_binding_settings(), projected, now)
        self.assertEqual(resolved["energy"]["value"], 0)
        features, _ = statement_affinity({"classification": {"topics": ["care"]}}, _binding_settings(), resolved)
        self.assertAlmostEqual(features["drive_affinity"], .4 * .8)
        self.assertNotIn("PRIVATE SOURCE", json.dumps(projected))

    def test_stale_keeps_values_and_as_of_unknown_and_zero_stay_distinct(self):
        raw = deepcopy(FIXTURE["bridge"]["inner_state"])
        raw["availability"] = "stale"
        raw["dimensions"]["protective_drive"].update(value=None, status="unknown", reason="not_observed")
        projected = inner_state_snapshots(raw)
        self.assertEqual(projected[0]["signals"]["energy"]["value"], 0)
        self.assertEqual(projected[1]["signals"]["protective_drive"]["state"], "unknown")
        self.assertIsNone(projected[1]["signals"]["protective_drive"]["value"])
        self.assertEqual(projected[0]["as_of"], raw["settled_at"])
        self.assertEqual(projected[0]["checkpoint"]["availability"], "stale")
        old = datetime.fromisoformat(raw["settled_at"]) + timedelta(hours=2)
        self.assertEqual(resolve_signals(_binding_settings(), projected, old)["energy"]["reason"], "stale")
        future = datetime.fromisoformat(raw["settled_at"]) - timedelta(seconds=1)
        self.assertEqual(resolve_signals(_binding_settings(), projected, future)["energy"]["reason"], "future_snapshot")

    def test_transport_accepts_four_bound_projections_and_rejects_mixed_checkpoints(self):
        scope = json.loads(FIXTURE["recall_header"])["scope"]
        received = read_transport(FIXTURE["recall_header"], **scope)
        self.assertEqual(received["state"], "received")
        self.assertEqual(len(received["snapshots"]), 4)
        mixed = json.loads(FIXTURE["recall_header"])
        mixed["snapshots"][3]["checkpoint"]["revision"]["brain"] += 1
        self.assertEqual(read_transport(json.dumps(mixed), **scope)["state"], "unavailable")
        self.assertEqual(read_transport(FIXTURE["recall_header"], **{**scope, "turn_id": "different"})["reason"], "scope_mismatch")

    def test_complete_registry_and_declared_units_are_required(self):
        for change in ("missing", "unit", "range", "unknown", "owner"):
            with self.subTest(change=change):
                raw = deepcopy(FIXTURE["bridge"]["inner_state"])
                if change == "missing": raw["dimensions"].pop("libido")
                if change == "unit": raw["dimensions"]["energy"]["unit"] = "0..100"
                if change == "range": raw["dimensions"]["slow_mood"]["value"] = -1
                if change == "unknown": raw["dimensions"]["fatigue"].update(value=None, status="known")
                if change == "owner": raw["dimensions"]["fatigue"]["owner"] = "aiz"
                with self.assertRaises(ValueError): inner_state_snapshots(raw)

    def test_unavailable_source_cannot_become_zero_or_midpoint(self):
        projected = inner_state_snapshots({"schema": "inner-state-v1", "availability": "unavailable", "dimensions": {}})
        self.assertEqual(len(projected), 2)
        self.assertTrue(all(item["state"] == "unavailable" and item["revision"] is None for item in projected))
        self.assertTrue(all(row["state"] == "unknown" and row["value"] is None
                            for item in projected for row in item["signals"].values()))


class GatewayUnifiedContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.service = _snapshot_service(self.root)
        self.service.config = {"gateway": {}}
        self.service.prompt_plan_mirror = PromptPlanMirrorStore(self.root / "prompt-mirror.sqlite")
        self.service.identity = {"ai_name": "Fixture", "user_name": "Owner", "user_display_name": "Owner"}
        self.service.inject_total_budget = 10000
        self.service.current_inner_state_interval_rounds = 1
        self.service.relationship_weather_interval_rounds = 1
        self.service._should_inject_interval = lambda session, interval: interval == 1
        self.service._build_relationship_weather_block = async_return("OWNER RELATIONSHIP RECORD")
        self.legacy_calls = []
        def forbidden(*args, **kwargs):
            self.legacy_calls.append(True)
            raise AssertionError("local Persona short-state path must not run")
        self.service.persona_engine = SimpleNamespace(profile_id="jiajia-main", enabled=True,
            get_current_state=forbidden, build_pre_reply_guidance=forbidden,
            format_state_block=forbidden, update_from_exchange=forbidden)
        for name in ("_get_persona_state_for_context_mode", "_classify_context_mode",
                     "_build_injected_context_messages", "_inject_context_messages", "_append_named_context_section"):
            del self.service.__dict__[name]
        self.requests, self.posts = [], []
        self.response = deepcopy(FIXTURE["bridge"])
        self.response_status = 200
        async def transport(request):
            self.requests.append(request)
            if request.url.path == "/bridge/v1/inner-state":
                return httpx.Response(self.response_status, json=self.response)
            if request.url.path == "/bridge/v1/conversation":
                return httpx.Response(200, json={"conversation_id": "fixture-conversation", "generation": False, "max_seq": 0})
            if request.method == "GET" and request.url.path == "/bridge/v1/events":
                return httpx.Response(200, json={"conversation_id": "fixture-conversation", "items": [], "next_after_seq": 0, "max_seq": 0})
            if request.method == "POST" and request.url.path == "/bridge/v1/events":
                self.posts.append(json.loads(request.content))
                return httpx.Response(201, json={"created": True, "event": {"id": f"fixture-event-{len(self.posts)}"}})
            raise AssertionError("unexpected route")
        client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.addAsyncCleanup(client.aclose)
        self.service.canonical_adapter = CanonicalContinuationAdapter(enabled=True,
            base_url="http://aiz.fixture", token="fixture-token", device_id="fixture-device",
            conversation_id="fixture-conversation", channel_id="operit",
            state_db_path=self.root / "canonical.sqlite", http_client=client)
        self.service.canonical_target_session_id = "jiajia-main"
        self.service.canonical_target_profile_id = "jiajia-main"
        self.service.canonical_session_channels = {"jiajia-main": "operit"}
        self.service.canonical_client_channels = {"operit": "operit"}
        self.last_signals = None
        original = self.service._build_memory_recall_projection
        def capture(**kwargs):
            self.last_signals = deepcopy(kwargs["natural_input"]["_signal_snapshots"])
            return original(**kwargs)
        self.service._build_memory_recall_projection = capture

    def request(self, *, marked=False, extra=None):
        headers = {"X-Request-ID": "stable-direct-request", **(extra or {})}
        if marked:
            headers.update({INNER_STATE_SCHEMA_HEADER: "inner-state-v1",
                "X-Guyan-Conversation-Id": "fixture-conversation", "X-Guyan-Turn-Id": "fixture-turn",
                "X-Ombre-Canonical-Source-Event-Id": "fixture-user-source",
                "X-Ombre-Canonical-Assistant-Source-Event-Id": "fixture-assistant-source",
                "X-Guyan-Recall-Signals": FIXTURE["recall_header"]})
        return make_request(headers)

    def payload(self, request, *, upstream_state=False):
        messages = [{"role": "system", "content": "OWNER SYSTEM"},
            {"role": "user", "content": "HISTORY USER"}, {"role": "assistant", "content": "HISTORY ASSISTANT"},
            {"role": "user", "content": "我们待一会儿"}]
        if upstream_state:
            messages.insert(1, deepcopy(FIXTURE["aiz_message"]))
        trace = self.service._trace_context_from_request(request, session_id="jiajia-main", request_type="initial", client_id="operit")
        return {"model": "guyan", "messages": messages, "_ombre_trace_context": trace}

    async def prepare(self, *, marked=False, plan=None, upstream_state=False, request=None):
        request = request or self.request(marked=marked)
        return await self.service.prepare_payload(self.payload(request, upstream_state=upstream_state),
            "jiajia-main", request=request, include_debug=True, prompt_plan=plan)

    def body(self, prepared):
        return "\n".join(str(row.get("content") or "") for row in prepared[0]["messages"])

    async def test_actual_direct_prepare_uses_single_full_snapshot_and_no_legacy_prompt(self):
        result = await self.prepare()
        text = FIXTURE["bridge"]["expression"]["text"]
        self.assertEqual(self.body(result).count(text), 1)
        self.assertIn("OWNER RELATIONSHIP RECORD", self.body(result))
        self.assertEqual(result[2]["inner_state"]["state"], "read")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].headers["X-Guyan-Device-ID"], "fixture-device")
        self.assertEqual(self.requests[0].headers["Authorization"], "Bearer fixture-token")
        self.assertEqual(self.requests[0].url.query, b"")
        self.assertEqual(self.last_signals, inner_state_snapshots(self.response["inner_state"]))
        for prohibited in ("B1_INNER_STATE", "Current Private Affect", "PRIVATE SOURCE BODY", "private-fixture-checkpoint", '"dimensions"'):
            self.assertNotIn(prohibited, self.body(result))
        for required in ("占有", "身体", "被偏爱", "黏", "疲劳"):
            self.assertIn(required, self.body(result))
        self.assertEqual(self.legacy_calls, [])

    async def test_actual_aiz_bridge_handler_output_is_adopted_by_gateway(self):
        from core.canonical_bridge_api import CanonicalBridgeHandlers
        from core.conversation_store import ConversationStore
        from core.inner_state import DIMENSIONS, make_host_snapshot, assemble_snapshot
        original = FIXTURE['bridge']['inner_state']
        hosts = [make_host_snapshot(owner=owner, subject_id=original['subject_id'],
            checkpoint_id=original['checkpoint_id'], revision=original['revision'][owner],
            rules_version=original['rules_version'], settled_at=original['settled_at'],
            values={key:{'value':original['dimensions'][key]['value'],'reason':original['dimensions'][key].get('reason')}
                    for key,spec in DIMENSIONS.items() if spec.owner==owner},
            rest=original.get('rest') if owner=='aiz' else None,
            feelings=original.get('feelings',[]) if owner=='brain' else []) for owner in ('aiz','brain')]
        full = assemble_snapshot(*hosts, subject_id=original['subject_id'])
        reads = []
        loop = SimpleNamespace(get_inner_state_snapshot=lambda: (reads.append('read') or full))
        bridge = CanonicalBridgeHandlers(store=ConversationStore(self.root/'actual-aiz-canonical.sqlite'),
            identity_id='fixture-owner', conversation_id='fixture-conversation', device_id='fixture-device', agent_loop=loop)
        actual_body = []
        async def transport(request):
            self.assertEqual(request.url.path, '/bridge/v1/inner-state')
            response = await bridge.inner_state(SimpleNamespace(query={}))
            body = json.loads(response.text)
            actual_body.append(body)
            return httpx.Response(response.status, json=body)
        client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.addAsyncCleanup(client.aclose)
        self.service.canonical_adapter.http_client = client
        result = await self.prepare()
        self.assertEqual(reads, ['read'])
        self.assertEqual(self.body(result).count(actual_body[0]['expression']['text']), 1)
        self.assertEqual(self.last_signals, inner_state_snapshots(full))
        self.assertEqual(self.legacy_calls, [])

    async def test_marked_request_keeps_aiz_body_and_its_affinity_without_second_read(self):
        result = await self.prepare(marked=True, upstream_state=True)
        text = FIXTURE["bridge"]["expression"]["text"]
        self.assertEqual(self.body(result).count(text), 1)
        self.assertIn("OWNER RELATIONSHIP RECORD", self.body(result))
        self.assertEqual(result[2]["inner_state"]["state"], "upstream_owned")
        self.assertEqual(self.requests, [])
        self.assertEqual(self.last_signals, json.loads(FIXTURE["recall_header"])["snapshots"])
        self.assertEqual(self.legacy_calls, [])

    async def test_marked_state_off_does_not_resurrect_but_other_owner_source_remains(self):
        plan = _plan(_block("ombre.persona_state", role="system", anchor="instructions.end", order=1,
                            mode="off", owner_body="OLD LOCAL STATE", wrapper_text="OWNER WRAPPER"),
                     _block("ombre.relationship_weather", role="developer", anchor="current_user.after", order=2))
        result = await self.prepare(marked=True, plan=plan)
        for prohibited in (FIXTURE["bridge"]["expression"]["text"], "OLD LOCAL STATE", "OWNER WRAPPER"):
            self.assertNotIn(prohibited, self.body(result))
        self.assertIn("OWNER RELATIONSHIP RECORD", self.body(result))
        self.assertEqual(self.requests, [])

    async def test_direct_composer_preserves_role_wrapper_depth_position_and_sha(self):
        plan = _plan(_block("ombre.persona_state", role="assistant", anchor="history.depth", history_depth=1,
                            order=1, wrapper_text="OWNER PREFIX\n{{content}}\nOWNER SUFFIX"),
                     _block("ombre.relationship_weather", role="developer", anchor="current_user.after", order=2))
        result = await self.prepare(plan=plan)
        messages = result[0]["messages"]
        index = next(i for i, row in enumerate(messages) if "OWNER PREFIX" in str(row.get("content")))
        self.assertEqual(messages[index]["role"], "assistant")
        self.assertEqual(messages[index]["content"], "OWNER PREFIX\n" + self.response["expression"]["text"] + "\nOWNER SUFFIX")
        self.assertGreater(index, next(i for i,row in enumerate(messages) if row.get("content") == "HISTORY USER"))
        self.assertLess(index, next(i for i,row in enumerate(messages) if row.get("content") == "HISTORY ASSISTANT"))
        resolved = result[2]["prompt_composer"]["resolved_blocks"][0]
        self.assertEqual(resolved["history_depth"], 1)
        self.assertEqual(resolved["model_message_sha256"], hashlib.sha256(messages[index]["content"].encode()).hexdigest())

    async def test_direct_off_and_owner_override_follow_applied_composer(self):
        for mode, body in (("off", ""), ("owner_override", "OWNER EXACT BODY")):
            with self.subTest(mode=mode):
                plan = _plan(_block("ombre.persona_state", role="system", anchor="instructions.end", order=1,
                    mode=mode, owner_body="OWNER EXACT BODY"),
                    _block("ombre.relationship_weather", role="developer", anchor="current_user.after", order=2))
                result = await self.prepare(plan=plan)
                self.assertNotIn(self.response["expression"]["text"], self.body(result))
                self.assertEqual("OWNER EXACT BODY" in self.body(result), bool(body))
                self.assertIn("OWNER RELATIONSHIP RECORD", self.body(result))

    async def test_mismatched_or_unavailable_api_cannot_fall_back_to_brain_only_state(self):
        for fault in ("conversation", "expression", "partial", "generation", "http"):
            with self.subTest(fault=fault):
                self.response = deepcopy(FIXTURE["bridge"])
                self.response_status = 503 if fault == "http" else 200
                if fault == "conversation": self.response["conversation_id"] = "different"
                if fault == "expression": self.response["expression"]["revision"]["brain"] += 1
                if fault == "partial": self.response["inner_state"]["dimensions"].pop("energy")
                if fault == "generation": self.response["generation"] = True
                result = await self.prepare()
                self.assertNotIn(FIXTURE["bridge"]["expression"]["text"], self.body(result))
                self.assertEqual(result[2]["inner_state"]["state"], "unavailable")
                self.assertEqual(self.last_signals, [])
                self.assertEqual(self.legacy_calls, [])

    async def test_stale_and_unavailable_bridge_keep_actual_renderer_scope_and_unknowns(self):
        for availability in ("stale", "unavailable"):
            with self.subTest(availability=availability):
                self.response = deepcopy(FIXTURE["bridge_" + availability])
                result = await self.prepare()
                self.assertEqual(self.body(result).count(self.response["expression"]["text"]), 1)
                self.assertEqual(result[2]["inner_state"]["availability"], availability)
                if availability == "stale":
                    self.assertIn("最近一次已结算的内态", self.body(result))
                    self.assertEqual(self.last_signals[0]["as_of"], FIXTURE["bridge"]["inner_state"]["settled_at"])
                    self.assertEqual(self.last_signals[0]["signals"]["energy"]["value"], 0)
                else:
                    self.assertTrue(all(item["state"] == "unavailable" for item in self.last_signals))
                    self.assertTrue(all(row["value"] is None for item in self.last_signals for row in item["signals"].values()))
                self.assertEqual(self.legacy_calls, [])

    async def test_context_mode_uses_transported_affect_and_only_persisted_long_relationship(self):
        db_path = self.root / "long-relationship.sqlite"
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE persona_global_state(profile_id TEXT,affinity REAL,dominance REAL,defensiveness REAL,trust REAL)")
            db.execute("INSERT INTO persona_global_state VALUES('jiajia-main',.8,.2,.3,.9)")
        self.service.persona_engine.db_path = str(db_path)
        signals = inner_state_snapshots(self.response["inner_state"])
        state = self.service._get_persona_state_for_context_mode("jiajia-main", signal_snapshots=signals)
        self.assertEqual(state["affect"]["longing"], .6)
        self.assertEqual(state["relationship"]["trust"], .9)
        self.assertEqual(self.service._classify_context_mode("我们待一会儿", state), "intimate")
        unknown = self.service._get_persona_state_for_context_mode("jiajia-main", signal_snapshots=[])
        self.assertTrue(all(value is None for value in unknown["affect"].values()))
        self.assertEqual(self.legacy_calls, [])

    async def test_internal_scope_and_unconnected_conversation_do_not_read_or_use_local_state(self):
        request = self.request(extra={"X-Guyan-Runtime-Scope": "internal:recall"})
        payload = self.payload(request)
        result = await self.service.prepare_payload(payload, "jiajia-main", request=request, include_debug=True)
        self.assertEqual(result[0]["messages"], payload["messages"])
        self.assertTrue(result[2]["internal_scope"])
        disconnected = await self.service._read_unified_inner_state_for_context(self.request(), "different-session")
        self.assertEqual(disconnected["diagnostics"]["reason"], "canonical_scope_not_connected")
        self.assertEqual(self.requests, [])
        self.assertEqual(self.legacy_calls, [])

    async def test_disabled_persona_source_does_not_reopen_the_state_reader(self):
        self.service.persona_engine.enabled = False
        result = await self.prepare()
        self.assertNotIn(FIXTURE["bridge"]["expression"]["text"], self.body(result))
        self.assertIn("OWNER RELATIONSHIP RECORD", self.body(result))
        self.assertEqual(self.requests, [])
        self.assertEqual(self.legacy_calls, [])

    async def test_cache_identity_separates_upstream_ownership_and_reuses_same_request(self):
        self.service.prepare_snapshot_ttl_seconds = 60
        # Existing source guards deliberately re-read relationship material.
        # Test state cache identity with no independent memory source present.
        self.service._build_relationship_weather_block = async_return("")
        request = self.request(extra={"X-Ombre-Canonical-Source-Event-Id": "fixture-user-source",
            "X-Ombre-Canonical-Assistant-Source-Event-Id": "fixture-assistant-source"})
        payload = self.payload(request)
        first = await self.service.prepare_payload(payload, "jiajia-main", request=request, include_debug=True)
        second = await self.service.prepare_payload(payload, "jiajia-main", request=request, include_debug=True)
        self.assertEqual(first[0]["messages"], second[0]["messages"])
        self.assertEqual(len(self.requests), 1)
        marked_request = make_request({**dict(request.headers), INNER_STATE_SCHEMA_HEADER: "inner-state-v1"})
        marked = await self.service.prepare_payload(payload, "jiajia-main", request=marked_request, include_debug=True)
        self.assertEqual(marked[2]["inner_state"]["state"], "upstream_owned")
        self.assertNotIn(self.response["expression"]["text"], self.body(marked))
        self.assertEqual(len(self.requests), 1)

    async def test_live_postreply_has_no_local_evaluator_and_direct_wire_pairs_original_user(self):
        request = self.request()
        payload, state = await self.service._prepare_canonical_turn(request, self.payload(request), "jiajia-main", "original user")
        await self.service._finalize_canonical_turn(state, {"role": "assistant", "content": "original assistant"})
        self.assertEqual([row["role"] for row in self.posts], ["user", "assistant"])
        self.assertEqual(self.posts[1]["correlation_id"], self.posts[0]["source_event_id"])
        await self.service._update_persona_after_assistant_message("jiajia-main", "original user",
            {"role": "assistant", "content": "original assistant"}, [])
        self.assertEqual(self.legacy_calls, [])
        self.posts.clear()
        marked = self.request(marked=True)
        _, state = await self.service._prepare_canonical_turn(marked, self.payload(marked), "jiajia-main", "original user")
        await self.service._finalize_canonical_turn(state, {"role": "assistant", "content": "original assistant"})
        self.assertEqual(self.posts, [])
        self.assertEqual(state["assistant_write_status"], "skipped_origin_owned")


if __name__ == "__main__":
    unittest.main()
