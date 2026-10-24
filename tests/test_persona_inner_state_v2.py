"""Offline behavioral checks; all databases and calibrations are fixtures."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from b1_protocol import digest, canonical
from persona_b1 import (FIELDS, Conflict, PersonaB1Adapter, PersonaB1Store,
                        decision_check, decision_check_v1)
from persona_b1_layers import record_mood
from persona_inner_state import (STATE_SCHEMA, TENDENCIES, migrate_state,
                                 observation_id, upgrade_policy_v2)
from persona_projection import safe_persona_projection


WHEN = '2026-10-23T00:00:00.000000+00:00'


def effect(delta=0, relief=0):
    return {'delta': delta, 'satisfaction': relief, 'floor': 0, 'plateau_hours': 0}


def fixture_policy():
    """Explicit fixture parameters, never a production or user calibration."""
    return {
        'revision': 'fixture',
        'laws': {key: {'half_life_hours': None, 'target': 0, 'growth_per_hour': 0} for key in FIELDS},
        'events': {'affection': {'valence': effect(.1)},
                   'intimacy': {'libido': effect(.2), 'possessiveness': effect(.1)},
                   'conflict': {'anger': effect(.25), 'grieve': effect(.12)},
                   'reconciliation': {'anger': effect(relief=.65)},
                   # This must never add an unbound generic empathy reward.
                   'empathy': {'grieve': effect(.8)}},
        'repeat_hours': 1, 'repeat_factor': .5, 'couplings': {},
        'subtype_policy': {'bump': .2, 'visible_weight': .01, 'clear_below': .01},
        'mixed_policy': {'negative_min': .3, 'positive_min': .3, 'hold_hours': 0},
        'cling_relief': {}, 'sleep_multipliers': {key: 1 for key in FIELDS},
        'initial_sleeping': False, 'mood_policy': {}, 'presentation': {},
        'slow_axis_events': {'affection': {'slow_security': .01}},
        'proof_policy': {}, 'correction_confidence_delta': -.04,
        'conflict_policy': {'anger_to_grief': .3, 'max_fade_hours': 8, 'soothed_fade_hours': 3,
                            'clear_below': .03, 'soothe_types': ['affection'],
                            'soothe_first': .1, 'soothe_step': .05, 'soothe_max': .25, 'soothe_floor': .25},
        'empathy_policy': {
            'base': .1, 'weights': {'owner': 1, 'family': .8, 'known': .5, 'stranger': .25},
            'caps': {name: None for name in ('owner', 'family', 'known', 'stranger')},
            'daily_caps': {name: None for name in ('owner', 'family', 'known', 'stranger')},
            'dedupe_hours': 6, 'helped_relief': .6, 'offset_minutes': 0,
            'impulses': {sub: {'valence': 0, 'arousal': 0} for sub in ('心疼', '不平', '想帮忙', '替人高兴')}},
        'light_policy': {'minutes': {'心动': 60, '害羞': 30, '得意': 60, '骄傲': 60},
                         'light_needed': 1, 'window_minutes': 30, 'repeat_factor': .6,
                         'min_minutes': 15, 'offset_minutes': 0, 'proof_confidence_min': .35}}


def source(user='抱抱你。', assistant='好，我在。', event_id='user-1'):
    return {'conversation_key': digest('conversation-1'), 'user_id': event_id,
            'assistant_id': 'assistant-' + event_id, 'user': user, 'assistant': assistant}


def span(src, role, text=None):
    start = 0 if text is None else src[role].index(text)
    end = len(src[role]) if text is None else start + len(text)
    return {'role': role, 'start': start, 'end': end,
            'evidence_hash': digest(src[role][start:end])}


def mention(src, role, text):
    start = src[role].index(text)
    return {'source_hash': digest(src), 'role': role, 'start': start,
            'end': start + len(text), 'mention_hash': digest(text)}


def observation(src, kind='affection', *, actor='owner', target='assistant',
                spans=None, subject=None, context='direct'):
    row = {'type': kind, 'actor': actor, 'target': target, 'context': context,
           'assertion': 'occurred', 'certainty': 'supported',
           'evidence_spans': spans or [span(src, 'user' if actor == 'owner' else 'assistant')],
           'subject_binding': subject}
    row['evidence_hash'] = digest({'source_hash': digest(src), 'evidence_spans': row['evidence_spans']})
    row['observation_id'] = observation_id(digest(src), row)
    return row


def decision(src, rows):
    return {'schema': 'b1-decision-v2', 'accepted': rows, 'rejected': [],
            'source_hash': digest(src), 'interpretation_is_not_ground_truth': True}


def envelope(src, rows, **details):
    return {'namespace': 'b1-completed-turn-v2', 'scope': 'fixture',
            'event_id': src['user_id'], 'occurred_at': WHEN,
            'policy_revision': 'fixture-inner-state-v2',
            'decision': decision(src, rows), 'labels': [], **details}


def turn(src):
    return {'conversation_id': 'conversation-1',
            'user_event': {'id': src['user_id'], 'role': 'user', 'content': src['user']},
            'assistant_event': {'id': src['assistant_id'], 'role': 'assistant', 'content': src['assistant']}}


def empathy(row, subject_ref, incident_ref, closeness='known', sub='心疼'):
    return {'observation_id': row['observation_id'], 'type': 'empathy', 'sub': sub,
            'subject_ref': subject_ref, 'incident_ref': incident_ref,
            'closeness': closeness, 'identity_status': 'resolved',
            'evidence_hash': row['evidence_hash'], 'subject_binding': row['subject_binding']}


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'candidate.sqlite'
        self.store = PersonaB1Store(self.path, legacy_path=Path(self.tmp.name) / 'legacy.sqlite')
        self.initial = {key: .6 for key in FIELDS}
        self.policy = upgrade_policy_v2(fixture_policy())
        self.store.initialize('fixture', self.initial, WHEN, self.policy)
        self.adapter = PersonaB1Adapter(self.store, 'fixture', {})

    def raw_state(self):
        with sqlite3.connect(self.path) as db:
            return json.loads(db.execute('SELECT state FROM b1_state').fetchone()[0])

    def write_legacy_fixture(self, *, mood=-.8, tendencies=None):
        state = self.raw_state()
        for key in ('state_schema', 'tendency_sources', 'tendency_evidence', 'state_migrations', 'migration_issues'):
            state.pop(key, None)
        state['slow_mood'] = {'value': mood, 'unit': '-1..1', 'covered_hours': 1, 'unknown_hours': 0}
        state['tendencies'] = tendencies or {}
        state['grudge'] = {'event_id': 'old-conflict', 'at': WHEN, 'evidence_hash': digest('old')}
        state['conflict_episode'] = {'active': True, 'soothes': 2, 'conflicts': 3, 'reconciled': False}
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE b1_state SET state=?,policy=?', (canonical(state), canonical(fixture_policy())))
        return state

    def test_same_kind_many_observations_get_one_base_reward(self):
        src = source('先抱抱你，再亲亲你。')
        a = observation(src, spans=[span(src, 'user', '先抱抱你')])
        b = observation(src, spans=[span(src, 'user', '再亲亲你')])
        self.adapter.settle(envelope(src, [a, b, deepcopy(a)]), turn(src))
        state = self.store.read('fixture')
        self.assertAlmostEqual(state['values']['valence'], .64)
        self.assertAlmostEqual(state['values']['slow_security'], .61)
        self.assertEqual(len(state['causes']), 2)
        self.assertEqual({row['observation_id'] for row in state['causes']}, {a['observation_id'], b['observation_id']})

    def test_each_subject_in_shared_span_gets_own_incident(self):
        src = source('小林和小王都受伤了。')
        rows = [observation(src, 'empathy', target='third_party', subject=mention(src, 'user', who)) for who in ('小林', '小王')]
        self.assertNotEqual(rows[0]['observation_id'], rows[1]['observation_id'])
        self.assertEqual(rows[0]['evidence_hash'], rows[1]['evidence_hash'])
        details = [empathy(row, who, 'same-incident') for row, who in zip(rows, ('person-lin', 'person-wang'))]
        self.adapter.settle(envelope(src, rows, empathy_details=details), turn(src))
        state = self.store.read('fixture')
        self.assertAlmostEqual(state['values']['grieve'], .7)
        self.assertEqual(len(state['empathy_loads']), 2)

    def test_third_party_does_not_contaminate_owner_observation(self):
        src = source('小林很难过。', '主人今天也很委屈。')
        owner = observation(src, 'empathy', actor='assistant', target='owner', subject=mention(src, 'assistant', '主人'))
        external = observation(src, 'empathy', target='third_party', subject=mention(src, 'user', '小林'))
        self.adapter.settle(envelope(src, [owner, external], empathy_details=[
            empathy(owner, 'owner', 'owner-incident', 'owner'), empathy(external, 'person-lin', 'other-incident')]), turn(src))
        self.assertAlmostEqual(self.store.read('fixture')['values']['grieve'], .75)

    def test_subject_from_other_observation_cannot_be_borrowed(self):
        src = source('小林和小王都受伤了。')
        rows = [observation(src, 'empathy', target='third_party', subject=mention(src, 'user', who)) for who in ('小林', '小王')]
        detail = empathy(rows[0], 'person-lin', 'incident')
        detail['subject_binding'] = rows[1]['subject_binding']
        with self.assertRaises((ValueError, Conflict)):
            self.adapter.settle(envelope(src, rows, empathy_details=[detail]), turn(src))
        self.assertEqual(self.store.read('fixture')['revision'], 0)

    def test_bound_third_party_still_cannot_become_owner(self):
        src = source('小林很难过。')
        row = observation(src, 'empathy', target='third_party', subject=mention(src, 'user', '小林'))
        with self.assertRaises(ValueError):
            self.adapter.settle(envelope(src, [row], empathy_details=[empathy(row, 'owner', 'incident', 'owner')]), turn(src))
        self.assertEqual(self.store.read('fixture')['revision'], 0)

    def test_generic_unresolved_empathy_does_not_fabricate_subject_effect(self):
        src = source('有人很难过。')
        row = observation(src, 'empathy', target='third_party')
        self.adapter.settle(envelope(src, [row]), turn(src))
        self.assertEqual(self.store.read('fixture')['values'], self.initial)

    def test_adapter_checks_actual_original_not_self_consistent_forged_hash(self):
        src = source()
        row = observation(src)
        row['evidence_spans'][0]['evidence_hash'] = digest('fabricated')
        row['evidence_hash'] = digest({'source_hash': digest(src), 'evidence_spans': row['evidence_spans']})
        row['observation_id'] = observation_id(digest(src), row)
        with self.assertRaises(Conflict):
            self.adapter.settle(envelope(src, [row]), turn(src))
        self.assertIsNone(self.store.receipt('fixture', src['user_id']))
        self.assertEqual(self.store.read('fixture')['revision'], 0)

    def test_actual_subject_mention_hash_is_verified(self):
        src = source('小林很难过。')
        subject = mention(src, 'user', '小林')
        subject['mention_hash'] = digest('小王')
        row = observation(src, 'empathy', target='third_party', subject=subject)
        with self.assertRaises(Conflict):
            self.adapter.settle(envelope(src, [row], empathy_details=[empathy(row, 'person-wang', 'incident')]), turn(src))
        self.assertEqual(self.store.read('fixture')['revision'], 0)

    def test_modes_and_feelings_bind_individual_observation(self):
        src = source('抱抱你，我也有点委屈。')
        a = observation(src, spans=[span(src, 'user', '抱抱你')])
        b = observation(src, spans=[span(src, 'user', '我也有点委屈')])
        bindings = [{'observation_id': r['observation_id'], 'type': r['type'], 'label': label,
                     'evidence_hash': r['evidence_hash']} for r, label in ((a, 'happy'), (b, 'hurt'))]
        env = envelope(src, [a, b], labels=['happy', 'hurt'], feeling_sources=bindings,
                       interaction_modes={a['observation_id']: 'mark_only'},
                       nuances=[{'observation_id': a['observation_id'], 'type': 'affection',
                                 'key': 'grieve', 'label': '委屈', 'evidence_hash': a['evidence_hash']}])
        self.adapter.settle(env, turn(src))
        state = self.store.read('fixture')
        self.assertEqual(state['labels'], ['hurt'])
        self.assertEqual(state['subtypes'], {})
        self.assertAlmostEqual(state['values']['valence'], .64)

    def test_label_keeps_its_actual_old_source_when_another_event_arrives(self):
        with sqlite3.connect(self.path) as db:
            p=deepcopy(self.policy)
            p['presentation']={'base_thresholds':{'pleasant_high':.7,'active_high':.5,
                'pleasant_low':.3,'relaxed_valence':.6,'relaxed_arousal':.2,
                'tense_arousal':.7,'tired_arousal':.1},'base_margin':.1,
                'base_hold_hours':0,'cause_hold_hours':0,'cause_priorities':{},'label_rules':{},'tones':{}}
            db.execute('UPDATE b1_state SET policy=?',(canonical(p),))
        refs={}
        for event_id,label in (('event-A','happy'),('event-B','hurt')):
            src=source(event_id=event_id);row=observation(src)
            binding={'observation_id':row['observation_id'],'label':label,'type':row['type'],'evidence_hash':row['evidence_hash']}
            refs[label]={'kind':'conversation','event_id':event_id,'occurred_at':WHEN,'observation_id':row['observation_id']}
            self.adapter.settle(envelope(src,[row],labels=[label],feeling_sources=[binding]),turn(src))
        snapshot=self.adapter.snapshot()
        self.assertEqual(snapshot['affect']['label_sources']['happy'],[refs['happy']])
        self.assertEqual(snapshot['affect']['label_sources']['hurt'],[refs['hurt']])
        self.assertEqual(set(snapshot['affect']['labels']),{'happy','hurt'})
        self.store.advance_clock('fixture','unrelated-clock',WHEN,self.policy['revision'])
        self.assertEqual(self.adapter.snapshot()['affect']['label_sources']['happy'],[refs['happy']])

    def test_unattributed_history_stays_diagnostic_and_does_not_claim_current_feeling(self):
        state=self.raw_state();state['labels']=['sad'];state['label_times']={'sad':WHEN}
        state['lit']={'word':'心动','at':WHEN,'until':'2026-10-23T01:00:00+00:00','raw_excerpt':'private-old-text'}
        with sqlite3.connect(self.path) as db:db.execute('UPDATE b1_state SET state=?',(canonical(state),))
        snapshot=self.adapter.snapshot()
        self.assertEqual(snapshot['affect']['labels'],[])
        self.assertEqual(snapshot['affect']['unattributed_labels'],['sad'])
        self.assertEqual(snapshot['affect']['label_sources'],{})
        self.assertIsNone(snapshot['affect']['lit'])
        self.assertEqual(snapshot['affect']['unattributed_lit']['word'],'心动')
        self.assertNotIn('private-old-text',canonical(snapshot))
        self.assertEqual(self.raw_state(),state)
        projected=safe_persona_projection(snapshot)
        self.assertEqual(projected['affect']['unattributed_labels'],['sad'])
        self.assertIsNone(projected['affect']['label'])

    def test_conversation_light_retains_its_verified_source_reference(self):
        src=source();row=observation(src)
        detail={'observation_id':row['observation_id'],'type':'affection','sub':'心动',
                'strength':'heavy','evidence_hash':row['evidence_hash']}
        self.adapter.settle(envelope(src,[row],light_details=[detail]),turn(src))
        lit=self.adapter.snapshot()['affect']['lit']
        self.assertEqual(lit['source_refs'],[{'kind':'conversation','event_id':src['user_id'],
                         'occurred_at':WHEN,'observation_id':row['observation_id']}])
        self.assertEqual(datetime.fromisoformat(lit['at']),datetime.fromisoformat(WHEN))
        self.assertEqual(datetime.fromisoformat(lit['until'])-datetime.fromisoformat(lit['at']),timedelta(minutes=60))
        self.assertEqual(safe_persona_projection(self.adapter.snapshot())['affect']['lit']['source_refs'],lit['source_refs'])

    def test_verified_operation_light_uses_operation_reference(self):
        with sqlite3.connect(self.path) as db:
            p=deepcopy(self.policy)
            p['proof_policy']={'gains':{key:.03 for key in ('her_ok','first','hard','helped_ack')},
                               'daily_factor':.6,'high_threshold':.75,'high_factor':.5}
            db.execute('UPDATE b1_state SET policy=?',(canonical(p),))
        raw={'scope':'fixture','operation_id':'canonical-operation','status':'succeeded',
             'proof':'her_ok','revision':1,'result_hash':digest('canonical-result'),'occurred_at':WHEN}
        self.adapter.result_resolver=lambda _:raw
        result=self.adapter.record_operation('canonical-operation',self.policy['revision'])
        self.assertEqual(result['affect']['lit']['word'],'得意')
        self.assertEqual(result['affect']['lit']['source_refs'],[{'kind':'operation','event_id':'canonical-operation','occurred_at':WHEN}])

    def test_type_only_mode_and_detail_are_not_v2_fallbacks(self):
        src = source()
        row = observation(src)
        with self.assertRaises(ValueError):
            self.adapter.settle(envelope(src, [row], interaction_modes={'affection': 'mark_only'}), turn(src))
        with self.assertRaises(ValueError):
            self.adapter.settle(envelope(src, [row], nuances=[{'type': 'affection', 'key': 'grieve', 'label': '委屈', 'evidence_hash': row['evidence_hash']}]), turn(src))

    def test_many_same_kind_observations_do_not_multiply_light_taps(self):
        with sqlite3.connect(self.path) as db:
            p=deepcopy(self.policy);p['light_policy']['light_needed']=3
            db.execute('UPDATE b1_state SET policy=?',(canonical(p),))
        src=source('抱抱你，再亲亲你。')
        rows=[observation(src,spans=[span(src,'user',word)]) for word in ('抱抱你','再亲亲你')]
        details=[{'observation_id':row['observation_id'],'type':'affection','sub':'心动',
                  'strength':'light','evidence_hash':row['evidence_hash']} for row in rows]
        self.adapter.settle(envelope(src,rows,light_details=details),turn(src))
        state=self.store.read('fixture')
        self.assertEqual(len(state['light_taps']),1)
        self.assertIsNone(state['lit'])

    def test_light_must_bind_its_own_observation(self):
        src=source('抱抱你，再亲亲你。')
        rows=[observation(src,spans=[span(src,'user',word)]) for word in ('抱抱你','再亲亲你')]
        detail={'observation_id':rows[0]['observation_id'],'type':'affection','sub':'心动',
                'strength':'heavy','evidence_hash':rows[1]['evidence_hash']}
        with self.assertRaises(ValueError):
            self.adapter.settle(envelope(src,rows,light_details=[detail]),turn(src))
        self.assertEqual(self.store.read('fixture')['revision'],0)

    def test_correction_preserves_both_sources_and_applies_once(self):
        src = source('我说的是小林。', '你说的是小王。')
        row = observation(src, 'correction', spans=[span(src, 'user'), span(src, 'assistant')])
        detail = {key: row[key] for key in ('observation_id', 'type', 'evidence_hash')}
        self.adapter.settle(envelope(src, [row, deepcopy(row)], correction_details=[detail, deepcopy(detail)]), turn(src))
        self.assertAlmostEqual(self.store.read('fixture')['values']['slow_confidence'], .56)
        incomplete = observation(src, 'correction')
        with self.assertRaises(ValueError):
            decision_check(decision(src, [incomplete]))

    def test_soothing_is_explicit_and_has_approved_independent_repeat(self):
        src = source(event_id='soothe-1')
        row = observation(src, 'soothing')
        self.adapter.settle(envelope(src, [row]), turn(src))
        state = self.store.read('fixture')
        self.assertAlmostEqual(state['values']['anger'], .6 * .85)
        self.assertAlmostEqual(state['values']['grieve'], .6 * .90)
        src2 = source(event_id='soothe-2')
        self.adapter.settle(envelope(src2, [observation(src2, 'soothing')]), turn(src2))
        state = self.store.read('fixture')
        self.assertAlmostEqual(state['values']['anger'], .6 * .85 * .925)
        self.assertAlmostEqual(state['values']['grieve'], .6 * .90 * .95)
        self.assertIsNone(state['grudge'])
        self.assertIsNone(state['conflict_episode'])

    def test_affection_does_not_implicitly_claim_soothing(self):
        src = source()
        self.adapter.settle(envelope(src, [observation(src)]), turn(src))
        values = self.store.read('fixture')['values']
        self.assertEqual(values['anger'], .6)
        self.assertEqual(values['grieve'], .6)

    def test_soothing_owner_source_may_have_assistant_corroboration(self):
        src=source('先抱抱你，别委屈。','嗯，我被你安抚到了。')
        row=observation(src,'soothing',spans=[span(src,'user'),span(src,'assistant')])
        self.adapter.settle(envelope(src,[row]),turn(src))
        self.assertAlmostEqual(self.store.read('fixture')['values']['anger'],.51)

    def test_assistant_soothing_owner_does_not_relieve_own_state(self):
        src=source('我很难过。','主人别难过，我抱抱你。')
        row=observation(src,'soothing',actor='assistant',target='owner')
        with self.assertRaises(ValueError):self.adapter.settle(envelope(src,[row]),turn(src))
        self.assertEqual(self.store.read('fixture')['values'],self.initial)
        self.assertEqual(self.store.read('fixture')['revision'],0)

    def test_soothing_cannot_borrow_only_assistant_claim_as_owner_source(self):
        src=source('继续吧。','主人安抚我了。')
        row=observation(src,'soothing',spans=[span(src,'assistant')])
        with self.assertRaises(ValueError):self.adapter.settle(envelope(src,[row]),turn(src))
        self.assertEqual(self.store.read('fixture')['values'],self.initial)

    def test_ordinary_conflict_and_intimacy_still_coexist(self):
        src = source('我有点生气，但还是想亲亲你。')
        rows = [observation(src, kind) for kind in ('conflict', 'intimacy')]
        self.adapter.settle(envelope(src, rows), turn(src))
        state = self.store.read('fixture')
        self.assertAlmostEqual(state['values']['anger'], .85)
        self.assertAlmostEqual(state['values']['libido'], .8)
        self.assertAlmostEqual(state['values']['possessiveness'], .7)
        self.assertIn('anger_closeness', state['mixed'])
        self.assertIsNone(state['conflict_episode'])

    def test_old_reconciliation_cannot_execute_in_new_input(self):
        src = source()
        with self.assertRaises(ValueError):
            decision_check(decision(src, [observation(src, 'reconciliation')]))

    def test_read_does_not_write_and_explicit_migration_runs_once(self):
        original = self.write_legacy_fixture()
        preview = self.store.read('fixture')
        self.assertAlmostEqual(preview['slow_mood']['value'], .1)
        self.assertEqual(self.raw_state(), original)
        migrated = self.store.migrate_scope('fixture', expected_revision=0)
        self.assertTrue(migrated['migrated'])
        persisted = self.raw_state()
        self.assertEqual(persisted['state_schema'], STATE_SCHEMA)
        self.assertEqual(persisted['values'], self.initial)
        self.assertAlmostEqual(persisted['slow_mood']['value'], .1)
        self.assertEqual(persisted['state_migrations'][0]['retired_conflict']['episode'], original['conflict_episode'])
        self.assertEqual(persisted['state_migrations'][0]['prior_slow_mood'], original['slow_mood'])
        self.assertEqual(persisted['state_migrations'][0]['retired_policy']['conflict_policy'],fixture_policy()['conflict_policy'])
        repeated = self.store.migrate_scope('fixture', expected_revision=0)
        self.assertFalse(repeated['migrated'])
        self.assertEqual(repeated['state']['revision'], 1)
        self.assertEqual(self.raw_state(), persisted)

    def test_migration_failure_rolls_back_checkpoint_policy_and_receipt(self):
        original=self.write_legacy_fixture()
        def fail(_):raise RuntimeError('fixture commit failure')
        self.store._before_commit=fail
        with self.assertRaises(RuntimeError):self.store.migrate_scope('fixture',expected_revision=0)
        self.assertEqual(self.raw_state(),original)
        self.assertIsNone(self.store.receipt('fixture','migration:'+STATE_SCHEMA))
        with sqlite3.connect(self.path) as db:
            self.assertEqual(json.loads(db.execute('SELECT policy FROM b1_state').fetchone()[0]),fixture_policy())
            self.assertEqual(db.execute('SELECT COUNT(*) FROM b1_numeric').fetchone()[0],1)

    def test_new_input_requires_explicit_legacy_policy_migration(self):
        original=self.write_legacy_fixture()
        src=source()
        with self.assertRaises(Conflict):self.adapter.settle(envelope(src,[observation(src)]),turn(src))
        self.assertEqual(self.raw_state(),original)

    def test_old_outer_envelope_cannot_become_new_write(self):
        src=source()
        env=envelope(src,[observation(src)]);env['namespace']='b1-completed-turn-v1'
        with self.assertRaises(ValueError):self.adapter.settle(env,turn(src))
        self.assertEqual(self.store.read('fixture')['revision'],0)

    def test_different_later_migration_policy_has_explicit_conflict(self):
        self.write_legacy_fixture()
        self.store.migrate_scope('fixture')
        with self.assertRaises(Conflict):self.store.migrate_scope('fixture',policy_revision='unrelated-next-policy')
        self.assertEqual(self.store.read('fixture')['revision'],1)

    def test_migration_accepts_explicit_coordinated_nested_policy(self):
        self.write_legacy_fixture()
        upgraded=upgrade_policy_v2(fixture_policy(),revision='stage01-fixture')
        upgraded['growth_policy']={'revision':'growth-stage01','grid_seconds':900,
                                  'anchor_utc':datetime.fromisoformat(WHEN).isoformat(),'source_scope':'fixture',
                                  'source_policy_revision':'stage01-fixture','targets':{}}
        self.store.migrate_scope('fixture',upgraded_policy=upgraded)
        with sqlite3.connect(self.path) as db:
            saved=json.loads(db.execute('SELECT policy FROM b1_state').fetchone()[0])
        self.assertEqual(saved['growth_policy'],upgraded['growth_policy'])
        self.assertEqual(saved['revision'],'stage01-fixture')
        self.assertEqual(self.store.read('fixture')['values'],self.initial)

    def test_unknown_tendency_scale_keeps_evidence_and_other_state(self):
        self.write_legacy_fixture(tendencies={TENDENCIES[0]: 125, TENDENCIES[1]: .7})
        result = self.store.migrate_scope('fixture', expected_revision=0)
        state = result['state']
        self.assertTrue(all(value is None for value in state['tendencies'].values()))
        self.assertEqual(state['tendency_evidence'][TENDENCIES[0]]['value'], 125)
        self.assertEqual(len(state['migration_issues']), 2)
        self.assertEqual(state['values'], self.initial)

    def test_explicit_tendency_source_imports_normalized_values_once(self):
        self.write_legacy_fixture(tendencies={TENDENCIES[0]: .2})
        sources = {TENDENCIES[0]: {'unit': '0..1', 'source_ref': 'canonical-fixture'}}
        state = self.store.migrate_scope('fixture', expected_revision=0, tendency_sources=sources)['state']
        self.assertEqual(state['tendencies'][TENDENCIES[0]], .2)
        self.store.migrate_scope('fixture', tendencies={TENDENCIES[0]: .9}, tendency_sources=sources)
        self.assertEqual(self.store.read('fixture')['tendencies'][TENDENCIES[0]], .2)

    def test_seed_and_projection_keep_absent_tendencies_unknown(self):
        snapshot = self.adapter.snapshot()
        self.assertTrue(all(value is None for value in snapshot['tendencies'].values()))
        projected = safe_persona_projection(snapshot)
        self.assertTrue(all(value is None for value in projected['tendencies'].values()))
        self.assertEqual(projected['slow_axes']['mood']['unit'], '0..1')
        self.assertEqual(snapshot['state_schema'],STATE_SCHEMA)
        self.assertEqual(snapshot['policy_revision'],self.policy['revision'])
        self.assertEqual(projected['policy_revision'],self.policy['revision'])
        self.assertEqual(projected['drive']['longing'],.6)
        self.assertEqual(projected['drive']['protective'],.6)
        self.assertEqual(projected['drive']['longing_level'],'high')

    def test_natural_decay_does_not_run_retired_episode_transfer(self):
        self.write_legacy_fixture()
        with sqlite3.connect(self.path) as db:
            p = fixture_policy()
            p['laws']['anger']['half_life_hours'] = 1
            p['laws']['grieve']['half_life_hours'] = 1
            db.execute('UPDATE b1_state SET policy=?', (canonical(p),))
        self.store.migrate_scope('fixture')
        self.store.advance_clock('fixture', 'hour-1', '2026-10-23T01:00:00+00:00', self.policy['revision'])
        state = self.store.read('fixture')
        self.assertAlmostEqual(state['values']['anger'], .3)
        self.assertAlmostEqual(state['values']['grieve'], .3)


class PureCase(unittest.TestCase):
    def test_legacy_is_an_explicit_reader_only(self):
        legacy = {'accepted': [], 'rejected': [], 'source_hash': digest('old'), 'interpretation_is_not_ground_truth': True}
        self.assertEqual(decision_check_v1(legacy), legacy)
        with self.assertRaises(ValueError):
            decision_check(legacy)

    def test_mood_generation_is_bounded_normalized_and_unknown_stays_unknown(self):
        start = datetime(2026, 10, 23, tzinfo=timezone.utc)
        p = {'window_days': 3, 'offset_minutes': 0, 'midpoint': .55, 'span': .25}
        for value, expected in ((0, 0), (.55, .5), (1, 1), (None, None)):
            state = {'mood_days': {}}
            record_mood(state, start, start + timedelta(hours=1), lambda _: value, p)
            if expected is None:self.assertIsNone(state['slow_mood']['value'])
            else:self.assertAlmostEqual(state['slow_mood']['value'], expected)
            self.assertEqual(state['slow_mood']['unit'], '0..1')

    def test_normalized_mood_is_not_transformed_again(self):
        raw = {'state_schema': STATE_SCHEMA, 'slow_mood': {'value': .3, 'unit': '0..1'}, 'tendencies': {}}
        self.assertEqual(migrate_state(migrate_state(raw))['slow_mood']['value'], .3)

    def test_disjoint_evidence_has_no_new_sixteen_span_cap(self):
        src=source('abcdefghijklmnopqrstuvwxyz0123456789')
        spans=[{'role':'user','start':i,'end':i+1,'evidence_hash':digest(src['user'][i:i+1])}
               for i in range(0,36,2)]
        row=observation(src,spans=spans)
        self.assertEqual(decision_check(decision(src,[row]))['accepted'][0]['evidence_spans'],spans)


if __name__ == '__main__':
    unittest.main()
