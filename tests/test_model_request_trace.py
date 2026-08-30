import tempfile
import unittest
from pathlib import Path

from model_request_trace import ModelRequestTraceStore, sanitize_for_owner


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

    def test_sanitizer_does_not_mutate_input(self):
        value = {"thinking": False, "reasoning_effort": "low", "token": "secret"}
        safe = sanitize_for_owner(value)
        self.assertEqual(value["token"], "secret")
        self.assertFalse(safe["thinking"])
        self.assertEqual(safe["reasoning_effort"], "low")
        self.assertEqual(safe["token"], "[REDACTED]")

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


if __name__ == "__main__":
    unittest.main()
