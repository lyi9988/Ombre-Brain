import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from gateway import GatewayService
from model_request_trace import BODY_MODES, ModelRequestTraceStore, sanitize_for_owner


class ModelRequestTraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="model_request_trace_test_")
        self.store = ModelRequestTraceStore(Path(self.tmp.name) / "trace.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def _logical(self, trace_id="trace-1"):
        return self.store.begin_logical({
            "trace_id": trace_id,
            "conversation_id": "conversation-1",
            "turn_id": "event-1",
            "request_id": "request-1",
            "logical_request_id": "request-1:model:1",
            "request_ordinal": 1,
            "request_type": "initial",
            "metadata": {"coverage": {"items": [{"event_id": "event-1"}]}},
        })

    def test_logical_request_id_is_durable_and_listed(self):
        trace_id = self._logical("trace-logical-id")
        item = self.store.get(trace_id, view="metadata")
        self.assertEqual(item["logical_request_id"], "request-1:model:1")
        self.assertEqual(self.store.list_recent(limit=1)[0]["logical_request_id"], "request-1:model:1")

    def test_raw_is_sanitized_deep_copy_and_preserves_scalar_reasoning_controls(self):
        trace_id = self._logical()
        payload = {
            "model": "deepseek-reasoner",
            "thinking": True,
            "reasoning_effort": "high",
            "max_tokens": 512,
            "token_count": 7,
            "messages": [{
                "role": "assistant",
                "reasoning_content": "provider returned body",
                "content": "answer",
            }],
            "url": "https://example.test/result?signature=secret&x=1",
            "headers": {"Authorization": "Bearer secret", "X-Trace": "ok"},
        }
        attempt_id = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="deepseek", upstream="gw",
            model="deepseek-reasoner", payload=payload,
        )
        payload["messages"][0]["content"] = "mutated after send"
        item = self.store.get(trace_id, view="raw_redacted")
        raw = item["attempts"][0]["payload"]
        self.assertEqual(raw["messages"][0]["content"], "answer")
        self.assertTrue(raw["thinking"])
        self.assertEqual(raw["reasoning_effort"], "high")
        self.assertEqual(raw["max_tokens"], 512)
        self.assertEqual(raw["token_count"], 7)
        self.assertNotIn("reasoning_content", raw["messages"][0])
        self.assertEqual(raw["headers"]["Authorization"], "[REDACTED]")
        self.assertIn("[REDACTED]", raw["url"])
        self.assertTrue(attempt_id)

    def test_provider_exposed_full_body_and_later_capture_setting_keeps_existing_raw(self):
        self.store.update_settings({
            "capture_mode": "full_owner_body",
            "body_visibility": "provider_exposed",
        })
        trace_id = self._logical("trace-2")
        self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="deepseek", upstream="gw",
            model="deepseek-reasoner", payload={
                "thinking": True,
                "messages": [{"role": "assistant", "reasoning_content": "public body"}],
            },
        )
        self.store.update_settings({"capture_mode": "metadata"})
        item = self.store.get(trace_id, view="full_owner_body")
        self.assertEqual(
            item["attempts"][0]["payload"]["messages"][0]["reasoning_content"],
            "public body",
        )
        self.store.update_settings({"body_visibility": "hidden"})
        hidden = self.store.get(trace_id, view="full_owner_body")
        self.assertNotIn(
            "reasoning_content", hidden["attempts"][0]["payload"]["messages"][0]
        )

    def test_capture_next_keeps_all_attempts_in_one_logical_request(self):
        self.store.update_settings({
            "enabled": False,
            "capture_mode": "raw_redacted",
            "capture_next": True,
        })
        trace_id = self._logical("trace-capture-next")
        first = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="p", upstream="u", model="m",
            payload={"messages": [{"role": "user", "content": "first"}]},
        )
        second = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="p", upstream="u", model="m",
            retry_reason="provider_retry",
            payload={"messages": [{"role": "user", "content": "retry"}]},
        )
        self.assertTrue(first)
        self.assertTrue(second)
        item = self.store.get(trace_id, view="metadata")
        self.assertEqual([row["attempt_ordinal"] for row in item["attempts"]], [1, 2])

    def test_retries_are_attempts_under_one_logical_request_with_exact_usage(self):
        trace_id = self._logical("trace-3")
        first = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="p", upstream="u", model="m",
            retry_reason="", payload={"messages": [{"role": "user", "content": "x"}]},
        )
        second = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="p", upstream="u", model="m",
            retry_reason="provider_retry", payload={"messages": [{"role": "user", "content": "x"}]},
        )
        self.store.update_attempt(first, http_status=429, outcome="error", usage={"prompt_tokens": 10})
        self.store.update_attempt(second, http_status=200, outcome="final_answer", usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
        self.store.set_outcome(trace_id, "final_answer")
        item = self.store.get(trace_id, view="metadata")
        self.assertEqual(item["outcome"], "final_answer")
        self.assertEqual([row["attempt_ordinal"] for row in item["attempts"]], [1, 2])
        self.assertEqual(item["attempts"][0]["usage"]["prompt_tokens"], 10)
        self.assertEqual(item["attempts"][1]["usage"]["total_tokens"], 14)
        completed = [event for event in item["events"] if event["type"] == "completed"]
        usage = [event for event in item["events"] if event["type"] == "usage"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["payload"]["outcome"], "final_answer")
        self.assertEqual(len(usage), 2)

    def test_owner_safe_provider_timing_is_persisted_per_attempt(self):
        trace_id = self._logical("trace-timing")
        attempt_id = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="bing", upstream="bing",
            model="m", retry_reason="candidate_fallback",
            timing={"header_ms": 8, "first_byte_ms": 12, "total_ms": 30},
        )
        self.store.update_attempt(
            attempt_id, duration_ms=30, http_status=200,
            result_status="stream_completed", outcome="final_answer",
            timing={"header_ms": 8, "first_byte_ms": 12, "total_ms": 30,
                    "timeout_category": ""},
        )
        item = self.store.get(trace_id, view="metadata")
        self.assertEqual(item["attempts"][0]["retry_reason"], "candidate_fallback")
        self.assertEqual(item["attempts"][0]["timing"]["first_byte_ms"], 12)
        self.assertEqual(item["attempts"][0]["timing"]["total_ms"], 30)

    def test_sanitizer_does_not_mutate_input(self):
        value = {"thinking": False, "reasoning_effort": "low", "token": "secret"}
        safe = sanitize_for_owner(value)
        self.assertEqual(value["token"], "secret")
        self.assertFalse(safe["thinking"])
        self.assertEqual(safe["reasoning_effort"], "low")
        self.assertEqual(safe["token"], "[REDACTED]")

    def test_text_owner_body_view_preserves_text_roles_order_and_parameters(self):
        self.store.update_settings({
            "capture_mode": "full_owner_body",
            "body_visibility": "metadata_only",
        })
        trace_id = self._logical("trace-text-owner-body")
        image_data = "A" * 4096
        anthropic_data = "B" * 2048
        gemini_data = "C" * 1024
        input_image_data = "E" * 512
        payload = {
            "model": "synthetic-model",
            "temperature": 0.25,
            "max_tokens": 256,
            "system": "Keep this exact system instruction.",
            "messages": [
                {"role": "system", "content": "First instruction."},
                {"role": "user", "content": [
                    {"type": "text", "text": "before image"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + image_data, "detail": "high"}},
                    {"type": "text", "text": "between images"},
                    {"type": "input_image", "image_url": "https://media.example/old.jpg"},
                    {"type": "input_image", "data": input_image_data},
                    {"type": "text", "text": "after image"},
                ]},
                {"role": "assistant", "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
                {"role": "assistant", "content": "final response"},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "anthropic": {"content": [{"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": anthropic_data,
            }}]},
            "contents": [{"parts": [
                {"text": "Gemini instruction."},
                {"inline_data": {"mime_type": "image/png", "data": gemini_data}},
                {"inlineData": {"mimeType": "image/jpeg", "data": "D" * 1024}},
                {"file_data": {"mime_type": "image/png", "file_uri": "gs://private-bucket/image.png"}},
            ]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 256},
        }
        attempt_id = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="synthetic", upstream="fixture",
            model="synthetic-model", payload=payload,
        )

        item = self.store.get(trace_id, view="text_owner_body")
        projected = item["attempts"][0]["payload"]
        user_content = projected["messages"][1]["content"]
        self.assertEqual([message["role"] for message in projected["messages"]],
                         [message["role"] for message in payload["messages"]])
        self.assertEqual([block.get("text") for block in user_content if block.get("type") == "text"],
                         ["before image", "between images", "after image"])
        self.assertEqual([block["type"] for block in user_content],
                         [block["type"] for block in payload["messages"][1]["content"]])
        self.assertEqual(user_content[1]["image_url"]["detail"], "high")
        self.assertIn("MIME image/png", user_content[1]["image_url"]["url"])
        self.assertIn("base64 4096 chars", user_content[1]["image_url"]["url"])
        self.assertIn("image URI omitted", user_content[3]["image_url"])
        self.assertIn("image/unknown", user_content[4]["data"])
        self.assertEqual(projected["system"], payload["system"])
        self.assertEqual(projected["tools"], payload["tools"])
        self.assertEqual(projected["generationConfig"], payload["generationConfig"])
        self.assertIn("image/jpeg", projected["anthropic"]["content"][0]["source"]["data"])
        self.assertIn("image/png", projected["contents"][0]["parts"][1]["inline_data"]["data"])
        encoded = json.dumps(projected, ensure_ascii=False)
        for image_body in (image_data, anthropic_data, gemini_data, input_image_data, "D" * 1024,
                           "gs://private-bucket/image.png", "https://media.example/old.jpg"):
            self.assertNotIn(image_body, encoded)
        self.assertNotIn("data:image/", encoded)
        self.assertTrue(attempt_id)

    def test_text_owner_body_is_read_only_keeps_full_raw_and_filters_one_attempt(self):
        self.store.update_settings({
            "capture_mode": "full_owner_body",
            "body_visibility": "metadata_only",
        })
        trace_id = self._logical("trace-text-view-read-only")
        first_payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "original words"},
            {"type": "image_url", "image_url": "data:image/png;base64,ORIGINAL-BYTES"},
        ]}], "temperature": 0.5}
        first = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="synthetic", upstream="fixture",
            model="synthetic-model", payload=first_payload,
        )
        second = self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="synthetic", upstream="fixture",
            model="synthetic-model", payload={"messages": [{"role": "user", "content": "retry"}]},
        )
        with self.store._connect() as db:
            raw_id = db.execute("SELECT raw_id FROM physical_attempts WHERE attempt_id=?", (first,)).fetchone()[0]
            saved_before = db.execute("SELECT payload_json FROM raw_requests WHERE raw_id=?", (raw_id,)).fetchone()[0]

        self.store.record_event(trace_id, "diagnostic", {
            "attempt_id": first, "outcome": "completed", "metadata": {"large_detail": "X" * 2000},
        })
        projected = self.store.get(trace_id, view="text_owner_body", attempt_id=first)
        self.assertEqual([row["attempt_id"] for row in projected["attempts"]], [first])
        self.assertEqual(projected["attempts"][0]["payload"]["messages"][0]["content"][0]["text"],
                         "original words")
        self.assertNotIn("ORIGINAL-BYTES", json.dumps(projected))
        self.assertEqual(projected["metadata"]["coverage"], {"items": [{"event_id": "event-1"}]})
        diagnostic = next(event for event in projected["events"] if event["type"] == "diagnostic")
        self.assertEqual(diagnostic["payload"]["outcome"], "completed")
        self.assertNotIn("metadata", diagnostic["payload"])

        with self.store._connect() as db:
            saved_after = db.execute("SELECT payload_json FROM raw_requests WHERE raw_id=?", (raw_id,)).fetchone()[0]
        self.assertEqual(saved_before, saved_after)
        full = self.store.get(trace_id, view="full_owner_body", attempt_id=first)
        self.assertEqual(full["attempts"][0]["payload"], first_payload)
        self.assertEqual(
            [(event["type"], event["created_at_ms"]) for event in projected["events"]],
            [(event["type"], event["created_at_ms"]) for event in full["events"]],
        )
        self.assertLess(len(json.dumps(projected)), len(json.dumps(full)))
        self.assertEqual(len(self.store.get(trace_id, view="full_owner_body")["attempts"]), 2)
        self.assertIsNone(self.store.get(trace_id, view="text_owner_body", attempt_id="not-this-trace"))
        self.assertEqual(self.store.settings()["capture_mode"], "full_owner_body")
        self.assertNotIn("text_owner_body", BODY_MODES)
        with self.assertRaises(ValueError):
            self.store.update_settings({"capture_mode": "text_owner_body"})
        self.assertNotEqual(first, second)

    def test_zero_retention_keeps_current_logical_request_and_no_orphan_completion(self):
        self.store.update_settings({
            "retention_days": 0,
            "request_limit": 1,
            "disk_budget_mb": 1,
        })
        trace_id = self._logical("trace-zero-retention")
        self.store.purge()
        self.assertIsNotNone(self.store.get(trace_id, view="metadata"))
        self.store.set_outcome(trace_id, "final_answer")
        self.store.set_outcome(trace_id, "final_answer")
        item = self.store.get(trace_id, view="metadata")
        self.assertEqual(item["outcome"], "final_answer")
        self.assertEqual(
            [event["type"] for event in item["events"]].count("completed"), 1
        )

    def test_too_small_budget_does_not_delete_the_only_current_request(self):
        self.store.update_settings({
            "retention_days": 0,
            "request_limit": 1,
            "disk_budget_mb": 1,
        })
        trace_id = self._logical("trace-budget")
        self.store.record_attempt(
            trace_id=trace_id, ordinal=1, provider="p", upstream="u", model="m",
            payload={"messages": [{"role": "user", "content": "x" * (1024 * 1024 + 1)}]},
        )
        self.assertIsNotNone(self.store.get(trace_id, view="metadata"))

    def test_set_outcome_for_purged_trace_does_not_recreate_event_only_trace(self):
        trace_id = self._logical("trace-purged")
        with self.store._connect() as db:
            db.execute("DELETE FROM logical_requests WHERE trace_id=?", (trace_id,))
        self.store.set_outcome(trace_id, "final_answer")
        self.assertIsNone(self.store.get(trace_id, view="metadata"))
        self.assertEqual(self.store.events_recent(), [])

class ModelRequestTraceReadRouteTest(unittest.TestCase):
    def test_text_view_and_attempt_filter_still_pass_through_owner_auth(self):
        calls = []

        def get_trace(trace_id, *, view, attempt_id):
            calls.append((trace_id, view, attempt_id))
            return {"trace_id": trace_id, "view": view, "attempt_id": attempt_id}

        def authorize(header):
            if header != "Bearer owner-token":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return None

        service = SimpleNamespace(
            _authorize=authorize,
            model_request_trace=SimpleNamespace(get=get_trace),
        )

        def request(header):
            return SimpleNamespace(
                headers={"Authorization": header},
                path_params={"trace_id": "trace-route"},
                query_params={"view": "text_owner_body", "attempt_id": "attempt-1"},
            )

        denied = asyncio.run(GatewayService.handle_model_request_trace(service, request("")))
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(calls, [])

        allowed = asyncio.run(GatewayService.handle_model_request_trace(
            service, request("Bearer owner-token")))
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(calls, [("trace-route", "text_owner_body", "attempt-1")])
        self.assertEqual(json.loads(allowed.body)["attempt_id"], "attempt-1")


if __name__ == "__main__":
    unittest.main()
