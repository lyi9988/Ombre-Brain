"""Real v2 writer regression: missing integration input cannot erase a checkpoint."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

import persona_b1 as writer
from b1_growth_protocol import derive_frame
from b1_pressure_protocol import KEYS, build_pressure_frame
from b1_protocol import digest
from persona_inner_state import DECISION_SCHEMA, observation_id


START = datetime(2026, 10, 23, tzinfo=timezone.utc)
SCOPE = 'retention-fixture'


def config():
    return {'revision': 'retention-v2',
        'laws': {key: {'half_life_hours': None, 'target': 0, 'growth_per_hour': 0}
                 for key in writer.FIELDS},
        'events': {'affection': {'tenderness': {'delta': .1, 'satisfaction': 0,
                    'floor': 0, 'plateau_hours': 0}}},
        'repeat_hours': 1, 'repeat_factor': .5, 'couplings': {},
        'subtype_policy': {}, 'mixed_policy': {}, 'cling_relief': {},
        'sleep_multipliers': {key: 1 for key in writer.FIELDS}, 'initial_sleeping': False,
        'mood_policy': {}, 'presentation': {}, 'slow_axis_events': {}, 'proof_policy': {}}


def envelope(policy, event_id='retention-event', minutes=15):
    # This accepted observation carries actual source text/span hashes, rather
    # than using a historical v1 input or an unbound hand-written label.
    source = {'user': 'I care about you.', 'assistant': 'I am here.'}
    source_hash = digest(source)
    spans = [{'role': 'user', 'start': 0, 'end': len(source['user']),
              'evidence_hash': digest(source['user'])}]
    observation = {'type': 'affection', 'actor': 'owner', 'target': 'assistant',
        'context': 'direct', 'assertion': 'occurred', 'certainty': 'supported',
        'evidence_spans': spans, 'subject_binding': None}
    observation['evidence_hash'] = digest({'source_hash': source_hash, 'evidence_spans': spans})
    observation['observation_id'] = observation_id(source_hash, observation)
    return {'namespace': 'b1-completed-turn-v2', 'scope': SCOPE, 'event_id': event_id,
        'occurred_at': (START + timedelta(minutes=minutes)).isoformat(),
        'policy_revision': policy['revision'], 'labels': [],
        'decision': {'schema': DECISION_SCHEMA, 'accepted': [observation], 'rejected': [],
            'source_hash': source_hash, 'interpretation_is_not_ground_truth': True}}


def growth_policy():
    return {'revision': 'growth-v2', 'grid_seconds': 900, 'anchor_utc': START.isoformat(),
        'source_scope': SCOPE, 'source_policy_revision': 'retention-v2',
        'targets': {'longing': {'emotion': None, 'drives': {'anger': -.2},
                                'floor': .4, 'ceiling': 1.8}}}


def growth_frame(policy, known):
    return derive_frame(policy, START, {'scope': SCOPE, 'policy_revision': 'retention-v2',
        'revision': 0 if known else None, 'as_of': START.isoformat() if known else None,
        'values': {'anger': .2} if known else {},
        'availability': 'available' if known else 'unavailable'})


def pressure_policy():
    return {'revision': 'pressure-v2', 'grid_seconds': 900, 'anchor_utc': START.isoformat(),
        'hosts': {'brain': {'scope': SCOPE, 'policy_revision': 'retention-v2'},
                  'aiz': {'scope': 'aiz-fixture', 'policy_revision': 'aiz-v2'}},
        'bindings': {key: {'host': 'aiz', 'key': key, 'unit': '0..100'} for key in KEYS}}


def pressure_frame(policy, known):
    aiz = {'scope': 'aiz-fixture', 'policy_revision': 'aiz-v2', 'revision': 0,
           'as_of': START.isoformat(), 'values': {key: 80 for key in KEYS}}
    brain = {'scope': SCOPE, 'policy_revision': 'retention-v2', 'revision': 0,
             'as_of': START.isoformat(), 'values': {}}
    return build_pressure_frame(policy, START, aiz if known else None, brain)


class ModulationRetentionV2(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b1-modulation-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'candidate.db'

    def store(self):
        return writer.PersonaB1Store(self.path, legacy_path=Path(self.temp.name) / 'legacy.db')

    def initialize(self, policy, unknown=()):
        initial = {key: None if key in unknown else .2 for key in writer.FIELDS}
        self.store().initialize(SCOPE, initial, START.isoformat(), policy)

    def rows(self):
        with sqlite3.connect(self.path) as db:
            return {table: db.execute('SELECT * FROM ' + table + ' ORDER BY 1,2').fetchall()
                    for table in ('b1_state', 'b1_numeric', 'b1_events',
                                  'b1_growth_frames', 'b1_pressure_frames')}

    def assert_rejected_preserves(self, event, source_kind, coordinate):
        before = self.rows()
        checkpoint = self.store().read(SCOPE)
        with self.assertRaises(writer.ModulationSourceUnavailable) as raised:
            self.store().settle(event, expected_revision=0)
        self.assertEqual(raised.exception.code, 'modulation_source_unavailable')
        self.assertEqual(raised.exception.source_kind, source_kind)
        self.assertEqual(raised.exception.coordinate, coordinate)
        self.assertEqual(before, self.rows())
        self.assertEqual(checkpoint, self.store().read(SCOPE))  # reopened connection/store
        self.assertIsNone(self.store().receipt(SCOPE, event['event_id']))

    def assert_recovers_once(self, event):
        result = self.store().settle(event, expected_revision=0)
        self.assertFalse(result['replayed'])
        after = self.rows()
        self.assertEqual(self.store().read(SCOPE)['revision'], 1)
        self.assertAlmostEqual(self.store().read(SCOPE)['values']['tenderness'], .3)
        retry = self.store().settle(event, expected_revision=0)
        self.assertTrue(retry['replayed'])
        self.assertEqual(result['receipt'], retry['receipt'])
        self.assertEqual(after, self.rows())

    def test_growth_missing_source_rolls_back_then_same_id_recovers(self):
        policy = config(); policy['growth_policy'] = growth_policy()
        policy['laws']['longing']['growth_per_hour'] = .1
        self.initialize(policy)
        event = envelope(policy)
        event['growth_frames'] = [growth_frame(policy['growth_policy'], False)]
        self.assert_rejected_preserves(event, 'growth', 'longing')
        event['growth_frames'] = [growth_frame(policy['growth_policy'], True)]
        self.assert_recovers_once(event)
        self.assertGreater(self.store().read(SCOPE)['values']['longing'], .2)

    def test_sleep_phase_missing_rolls_back_then_same_id_recovers(self):
        policy = config(); policy['initial_sleeping'] = None
        policy['sleep_multipliers']['longing'] = 2
        policy['laws']['longing']['growth_per_hour'] = .1
        self.initialize(policy)
        event = envelope(policy)
        self.assert_rejected_preserves(event, 'sleep_phase', 'longing')
        # A trusted lifecycle input at the original checkpoint supplies the
        # missing state, preserving the failed event identity for the retry.
        self.store().phase_event(SCOPE, 'known-wake', START.isoformat(), policy['revision'],
                                 'wake', source_kind='internal_owner_request')
        event_result = self.store().settle(event, expected_revision=1)
        self.assertFalse(event_result['replayed'])
        self.assertEqual(self.store().read(SCOPE)['revision'], 2)
        self.assertTrue(self.store().settle(event, expected_revision=1)['replayed'])
        self.assertAlmostEqual(self.store().read(SCOPE)['values']['tenderness'], .3)

    def test_fatigue_pressure_missing_rolls_back_then_same_id_recovers(self):
        policy = config(); policy['fatigue_policy'] = {'initial': .2, 'maximum': 1,
            'build_per_hour': .1, 'recover_per_hour': .2, 'pressure_threshold': .5}
        policy['pressure_policy'] = pressure_policy()
        self.initialize(policy)
        event = envelope(policy)
        event['pressure_frames'] = [pressure_frame(policy['pressure_policy'], False)]
        self.assert_rejected_preserves(event, 'pressure', 'fatigue')
        event['pressure_frames'] = [pressure_frame(policy['pressure_policy'], True)]
        self.assert_recovers_once(event)
        self.assertAlmostEqual(self.store().read(SCOPE)['fatigue']['value'], .225)

    def test_fatigue_sleep_missing_preserves_known_fatigue(self):
        policy = config(); policy['initial_sleeping'] = None
        policy['fatigue_policy'] = {'initial': .2, 'maximum': 1,
            'build_per_hour': .1, 'recover_per_hour': .2, 'pressure_threshold': .5}
        self.initialize(policy)
        self.assert_rejected_preserves(envelope(policy), 'sleep_phase', 'fatigue')

    def test_genuine_unknown_values_are_not_seeded(self):
        policy = config(); policy['growth_policy'] = growth_policy()
        policy['laws']['longing']['growth_per_hour'] = .1
        policy['fatigue_policy'] = {'initial': None, 'maximum': 1,
            'build_per_hour': .1, 'recover_per_hour': .2, 'pressure_threshold': .5}
        self.initialize(policy, unknown=('longing',))
        event = envelope(policy)
        event['growth_frames'] = [growth_frame(policy['growth_policy'], True)]
        self.assert_recovers_once(event)
        state = self.store().read(SCOPE)
        self.assertIsNone(state['values']['longing'])
        self.assertIsNone(state['fatigue']['value'])

    def test_unused_growth_factor_and_unit_sleep_multiplier_do_not_block(self):
        policy = config(); policy['growth_policy'] = growth_policy()
        policy['initial_sleeping'] = None
        self.initialize(policy)
        event = envelope(policy)
        event['growth_frames'] = [growth_frame(policy['growth_policy'], False)]
        self.assert_recovers_once(event)
        self.assertEqual(self.store().read(SCOPE)['values']['longing'], .2)


if __name__ == '__main__':
    unittest.main()
