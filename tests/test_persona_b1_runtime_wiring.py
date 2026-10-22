import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from persona_engine import PersonaStateEngine
from persona_b1_runtime import preflight, prepare, activate_prepared, reopen, make_engine
from test_persona_b1_layers import layers_policy


class RuntimeWiring(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.old = root / 'old.db'; self.candidate = root / 'candidate.db'
        self.config = {'buckets_dir': str(root),
                       'persona': {'profile_id': 'guyan', 'canonical_session_id': 'chat',
                                   'mode': 'disabled', 'db_path': str(self.old)},
                       'b1': {'mode': 'candidate_authority', 'legacy_db_path': str(self.old),
                              'candidate_db_path': str(self.candidate), 'profile_id': 'guyan',
                              'scope': 'guyan', 'epoch': 1, 'legacy_session_id': 'chat',
                              'handoff_at': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                              'global_snapshot_path': str(root / 'source.json'),
                              'policy_path': str(root / 'policy.json')}}
        engine = PersonaStateEngine(self.config, str(self.old))
        engine.get_current_state('chat')
        Path(self.config['b1']['policy_path']).write_text(json.dumps(layers_policy()), encoding='utf-8')

    def test_prepare_activate_reopen_and_preserve_global(self):
        before = preflight(self.config)
        self.assertEqual(before['scope'], 'guyan')
        self.assertEqual(prepare(self.config)['source_sha256'], before['source_sha256'])
        self.assertEqual(prepare(self.config)['source_sha256'], before['source_sha256'])
        with self.assertRaises(Exception):
            reopen(self.config)
        activate_prepared(self.config)
        first = make_engine(self.config).b1_candidate.snapshot()
        self.assertEqual(first['revision'], 0)
        with sqlite3.connect(self.old) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE persona_session_state SET valence=.1 WHERE profile_id='guyan'")
        self.assertEqual(reopen(self.config).snapshot()['revision'], 0)
        self.assertEqual(first['personality']['openness'],
                         reopen(self.config).snapshot()['personality']['openness'])
        wrong = {**self.config, 'persona': {**self.config['persona'], 'db_path': str(self.old.parent / 'wrong.db')}}
        with self.assertRaisesRegex(ValueError, 'b1_running_legacy_path_mismatch'):
            make_engine(wrong)

    def test_wrong_binding_and_changed_source(self):
        prepare(self.config)
        with sqlite3.connect(self.old) as db:
            db.execute("UPDATE persona_session_state SET valence=.2 WHERE profile_id='guyan'")
        with self.assertRaisesRegex(ValueError, 'b1_legacy_source_changed'):
            activate_prepared(self.config)
        # An unfenced candidate never masquerades as an installed owner.
        with self.assertRaises(Exception):
            reopen(self.config)

    def test_only_absent_legacy_fields_can_receive_explicit_initial_values(self):
        path = self.old.parent / 'initial.json'
        self.config['b1']['initial_values_path'] = str(path)
        path.write_text(json.dumps({'slow_confidence': .42}), encoding='utf-8')
        self.assertEqual(preflight(self.config)['initial']['slow_confidence'], .42)
        path.write_text(json.dumps({'valence': .1}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'b1_original_field_cannot_be_overridden'):
            preflight(self.config)
