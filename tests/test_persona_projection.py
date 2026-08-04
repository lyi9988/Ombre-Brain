import unittest

from persona_projection import safe_persona_projection


class PersonaProjectionTest(unittest.TestCase):
    def test_safe_persona_projection_excludes_private_and_personality_fields(self):
        state = {
        "profile_id": "jiajia-main",
        "personality": {"openness": 0.9},
        "affect": {
            "mood_label": "bright_warm",
            "valence": 0.72,
            "arousal": 0.7,
            "longing": 0.5,
            "protective_drive": 0.7,
            "residue": "private residue",
            "inner_thought": "private thought",
        },
        "relationship": {
            "affinity": 0.82, "trust": 0.8, "defensiveness": 0.1,
        },
        "reply_guidance": "hidden guidance",
        }

        result = safe_persona_projection(state, "2026-08-04T00:00:00Z")

        self.assertEqual(result["profile_id"], "jiajia-main")
        self.assertEqual(result["affect"]["label"], "bright_warm")
        self.assertEqual(result["affect"]["intensity"], "high")
        self.assertEqual(result["drive"]["labels"], ["connection", "protection", "expression"])
        self.assertEqual(result["relationship"]["weather"], "close_and_secure")
        self.assertNotIn("personality", result)
        serialized = str(result)
        for private in ("private residue", "private thought", "hidden guidance"):
            self.assertNotIn(private, serialized)


    def test_safe_persona_projection_uses_honest_ranges(self):
        result = safe_persona_projection({
            "affect": {"mood_label": "quiet", "valence": 0.4, "arousal": 0.2},
            "relationship": {"affinity": 0.5, "trust": 0.5, "defensiveness": 0.5},
        })

        self.assertEqual(result["affect"]["intensity"], "low")
        self.assertEqual(result["drive"]["labels"], ["companionship"])
        self.assertEqual(result["relationship"]["weather"], "careful")


if __name__ == "__main__":
    unittest.main()
