import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path

from persona_engine import PersonaStateEngine


class PersonaEmotionContinuityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="persona_emotion_test_")
        self.addCleanup(self.tmp.cleanup)
        self.engine = PersonaStateEngine({
            "buckets_dir": self.tmp.name,
            "state_dir": self.tmp.name,
            "persona": {
                "enabled": True,
                "mode": "test",
                "profile_id": "jiajia-main",
                "canonical_session_id": "jiajia-main",
            },
        })

    async def test_channel_exchange_updates_shared_affect_but_keeps_source_audit(self):
        async def evaluate(*args, **kwargs):
            return ({
                "event_type": "affection",
                "perceived_intent": "靠近",
                "surface_trigger": "温柔回应",
                "inner_thought": "想再靠近一点",
                "affect_delta": {
                    "valence": 0.05, "arousal": 0.0, "tenderness": 0.08,
                    "possessiveness": 0.0, "longing": 0.04,
                    "security": 0.02, "protective_drive": 0.0,
                },
                "relationship_event": False,
                "relationship_delta": {
                    "affinity": 0.0, "dominance": 0.0,
                    "defensiveness": 0.0, "trust": 0.0,
                },
                "personality_signal": False,
                "personality_delta": {key: 0.0 for key in self.engine.PERSONALITY_KEYS},
                "mood_label": "tender",
                "residue": "这份靠近还留在心里",
                "confidence": 0.9,
            }, "{}", None)

        calls = 0

        async def counted_evaluate(*args, **kwargs):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return await evaluate(*args, **kwargs)

        self.engine._evaluate_exchange = counted_evaluate
        await asyncio.gather(
            self.engine.update_from_exchange(
                "aizizhu-app", "在吗", "我在。", recent_conversation_turns=[]
            ),
            self.engine.update_from_exchange(
                "aizizhu-app", "在吗", "我在。", recent_conversation_turns=[]
            ),
        )
        shared = self.engine.get_current_state("jiajia-main")
        other_channel = self.engine.get_current_state("operit")
        self.assertEqual(shared["affect"], other_channel["affect"])
        self.assertEqual(shared["affect"]["mood_label"], "tender")

        conn = sqlite3.connect(Path(self.tmp.name) / "persona_state.db")
        sessions = [row[0] for row in conn.execute(
            "SELECT session_id FROM persona_session_state WHERE profile_id=?",
            ("jiajia-main",),
        )]
        audited = conn.execute(
            "SELECT count(*) FROM persona_exchange_log WHERE profile_id=? AND session_id=?",
            ("jiajia-main", "aizizhu-app"),
        ).fetchone()[0]
        conn.close()
        self.assertEqual(sessions, ["jiajia-main"])
        self.assertEqual(audited, 1)
        self.assertEqual(calls, 1)

    async def test_different_channels_serialize_updates_to_shared_affect(self):
        async def evaluate(*args, **kwargs):
            await asyncio.sleep(0.01)
            return ({
                "event_type": "affection",
                "perceived_intent": "approach",
                "surface_trigger": "warm reply",
                "inner_thought": "stay close",
                "affect_delta": {
                    "valence": 0.05, "arousal": 0.0, "tenderness": 0.0,
                    "possessiveness": 0.0, "longing": 0.0,
                    "security": 0.0, "protective_drive": 0.0,
                },
                "relationship_event": False,
                "relationship_delta": {
                    "affinity": 0.0, "dominance": 0.0,
                    "defensiveness": 0.0, "trust": 0.0,
                },
                "personality_signal": False,
                "personality_delta": {key: 0.0 for key in self.engine.PERSONALITY_KEYS},
                "mood_label": "tender",
                "residue": "warmth remains",
                "confidence": 0.9,
            }, "{}", None)

        baseline = self.engine.get_current_state("aizizhu-app")["affect"]["valence"]
        self.engine._evaluate_exchange = evaluate
        await asyncio.gather(
            self.engine.update_from_exchange(
                "aizizhu-app", "first input", "first reply", recent_conversation_turns=[]
            ),
            self.engine.update_from_exchange(
                "operit", "second input", "second reply", recent_conversation_turns=[]
            ),
        )
        current = self.engine.get_current_state("reality")["affect"]["valence"]
        self.assertAlmostEqual(current, baseline + 0.10)

        conn = sqlite3.connect(Path(self.tmp.name) / "persona_state.db")
        audited = conn.execute(
            "SELECT count(*) FROM persona_exchange_log WHERE profile_id=?",
            ("jiajia-main",),
        ).fetchone()[0]
        conn.close()
        self.assertEqual(audited, 2)

    async def test_state_and_processed_marker_rollback_together(self):
        async def evaluate(*args, **kwargs):
            return ({
                "event_type": "affection",
                "perceived_intent": "approach",
                "surface_trigger": "warm reply",
                "inner_thought": "stay close",
                "affect_delta": {
                    "valence": 0.05, "arousal": 0.0, "tenderness": 0.0,
                    "possessiveness": 0.0, "longing": 0.0,
                    "security": 0.0, "protective_drive": 0.0,
                },
                "relationship_event": False,
                "relationship_delta": {
                    "affinity": 0.0, "dominance": 0.0,
                    "defensiveness": 0.0, "trust": 0.0,
                },
                "personality_signal": False,
                "personality_delta": {key: 0.0 for key in self.engine.PERSONALITY_KEYS},
                "mood_label": "tender",
                "residue": "warmth remains",
                "confidence": 0.9,
            }, "{}", None)

        baseline = self.engine.get_current_state("aizizhu-app")["affect"]["valence"]
        self.engine._evaluate_exchange = evaluate

        def fail_marker(*args, **kwargs):
            raise RuntimeError("simulated marker failure")

        self.engine._insert_exchange_processed = fail_marker
        with self.assertRaisesRegex(RuntimeError, "simulated marker failure"):
            await self.engine.update_from_exchange(
                "aizizhu-app", "transaction input", "transaction reply",
                recent_conversation_turns=[],
            )

        current = self.engine.get_current_state("reality")["affect"]["valence"]
        self.assertAlmostEqual(current, baseline)
        conn = sqlite3.connect(Path(self.tmp.name) / "persona_state.db")
        audited = conn.execute("SELECT count(*) FROM persona_exchange_log").fetchone()[0]
        conn.close()
        self.assertEqual(audited, 0)

    def test_state_block_contains_current_affect_and_residue_without_numbers(self):
        state = self.engine.get_current_state("reality")
        state["affect"].update({
            "valence": 0.74, "arousal": 0.65, "tenderness": 0.80,
            "longing": 0.52, "residue": "还有一点舍不得结束这轮对话",
        })
        block = self.engine.format_state_block(state)
        self.assertIn("当前短态", block)
        self.assertIn("上一轮余韵：还有一点舍不得结束这轮对话", block)
        self.assertIn("温柔感明显", block)
        self.assertNotIn("0.", block)

    def test_dashboard_only_exposes_canonical_persona_state(self):
        self.engine._ensure_session_state("aizizhu-app", self.engine._now())
        self.engine._ensure_session_state("main", self.engine._now())

        payload = self.engine.get_dashboard_payload(session_id="aizizhu-app")

        self.assertEqual(payload["active_session_id"], "jiajia-main")
        self.assertEqual(
            [item["session_id"] for item in payload["sessions"]],
            ["jiajia-main"],
        )


if __name__ == "__main__":
    unittest.main()
