"""New typed receipt sources and one-transaction snapshot, temporary data only."""
from datetime import datetime,timedelta
import json
import sqlite3
import unittest
from unittest.mock import patch

import test_persona_inner_state_v2 as base
from b1_protocol import canonical,digest


class ChangeSourceTests(unittest.TestCase):
    def setUp(self):
        fixture=base.StoreCase();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.store=fixture.store;self.adapter=fixture.adapter

    def test_negative_and_intimate_changes_keep_their_own_observation(self):
        src=base.source('有点生气，但还是想亲近。')
        rows=[base.observation(src,kind) for kind in ('conflict','intimacy')]
        result=self.adapter.settle(base.envelope(src,rows),base.turn(src))
        changes={row['key']:row for row in result['recent_changes']}
        self.assertEqual(changes['anger']['reason_code'],'conflict')
        self.assertEqual(changes['anger']['source_refs'][0]['observation_id'],rows[0]['observation_id'])
        self.assertEqual(changes['libido']['reason_code'],'intimacy')
        self.assertEqual(changes['libido']['source_refs'][0]['observation_id'],rows[1]['observation_id'])
        self.assertNotIn(src['user'],canonical(result))

    def test_time_and_event_deltas_use_the_actual_middle_state(self):
        with sqlite3.connect(self.fixture.path) as db:
            p=self.fixture.policy.copy();p['laws']=dict(p['laws']);p['laws']['anger']={'half_life_hours':1,'target':0,'growth_per_hour':0}
            db.execute('UPDATE b1_state SET policy=?',(canonical(p),))
        src=base.source();env=base.envelope(src,[base.observation(src,'conflict')])
        env['occurred_at']='2026-10-23T01:00:00+00:00'
        result=self.adapter.settle(env,base.turn(src));receipt=result['b1_receipt']['receipt']
        self.assertEqual(receipt['schema'],'persona-inner-receipt-v1');self.assertEqual(receipt['unit'],'0..1')
        self.assertAlmostEqual(receipt['state_before']['anger'],.6)
        self.assertAlmostEqual(receipt['time_state']['anger'],.3)
        self.assertAlmostEqual(receipt['state_after']['anger'],.55)
        self.assertAlmostEqual(receipt['time_deltas']['anger'],-.3)
        self.assertAlmostEqual(receipt['event_deltas']['anger'],.25)
        changes=[row for row in result['recent_changes'] if row['key']=='anger']
        self.assertEqual([row['reason_code'] for row in changes],['time_passed','conflict'])
        self.assertEqual(changes[0]['source_refs'][0]['kind'],'clock')
        self.assertEqual(changes[1]['source_refs'][0]['kind'],'conversation')

    def test_slow_security_and_operation_confidence_have_real_sources(self):
        src=base.source();row=base.observation(src)
        result=self.adapter.settle(base.envelope(src,[row]),base.turn(src))
        security=next(change for change in result['recent_changes'] if change['key']=='slow_security')
        self.assertEqual(security['reason_code'],'affection')
        self.assertEqual(security['source_refs'][0]['observation_id'],row['observation_id'])
        with sqlite3.connect(self.fixture.path) as db:
            p=self.fixture.policy.copy();p['proof_policy']={'gains':dict.fromkeys(('her_ok','first','hard','helped_ack'),.03),
                'daily_factor':.6,'high_threshold':.75,'high_factor':.5}
            db.execute('UPDATE b1_state SET policy=?',(canonical(p),))
        self.adapter.result_resolver=lambda _:dict(scope='fixture',operation_id='canonical-op',status='succeeded',proof='her_ok',
            revision=1,result_hash=digest('result'),occurred_at=base.WHEN)
        result=self.adapter.record_operation('canonical-op',self.fixture.policy['revision'])
        confidence=next(change for change in result['recent_changes'] if change['key']=='slow_confidence')
        self.assertEqual(confidence['reason_code'],'verified_operation')
        self.assertEqual(confidence['source_refs'],[{'kind':'operation','event_id':'canonical-op','occurred_at':base.WHEN}])

    def test_auxiliary_changes_require_known_before_and_after(self):
        with sqlite3.connect(self.fixture.path) as db:
            p=self.fixture.policy.copy()
            p['fatigue_policy']={'initial':.4,'maximum':1,'build_per_hour':.1,'recover_per_hour':.1,'pressure_threshold':.5}
            p['mood_policy']={'window_days':3,'offset_minutes':0,'midpoint':.5,'span':.5}
            p['laws']=dict(p['laws']);p['laws']['valence']={'half_life_hours':1,'target':.2,'growth_per_hour':0}
            state=self.fixture.raw_state();state['fatigue']['value']=.4
            db.execute('UPDATE b1_state SET policy=?,state=?',(canonical(p),canonical(state)))
        self.adapter.set_phase('actual-sleep',base.WHEN,p['revision'],'sleep','internal_owner_request')
        first=self.adapter.advance_clock('hour-1','2026-10-23T01:00:00+00:00',p['revision'])
        self.assertTrue(any(row['key']=='fatigue' for row in first['recent_changes']))
        self.assertFalse(any(row['key']=='slow_mood' for row in first['recent_changes']))
        second=self.adapter.advance_clock('hour-2','2026-10-23T02:00:00+00:00',p['revision'])
        mood=next(row for row in second['recent_changes'] if row['key']=='slow_mood')
        self.assertIsNotNone(mood['before']);self.assertIsNotNone(mood['after'])
        self.assertEqual(mood['reason_code'],'time_passed')

    def test_new_history_is_bounded_and_old_units_are_never_inferred(self):
        for index in range(20):self.store.advance_clock('fixture','zero-'+str(index),base.WHEN,self.fixture.policy['revision'])
        with sqlite3.connect(self.fixture.path) as db:
            for index,receipt in enumerate(({'after_revision':20,'private_body':'legacy-private'},
                {'schema':'persona-inner-receipt-v1','unit':'-1..1','after_revision':20,'private_body':'signed-private'})):
                db.execute('INSERT INTO b1_events VALUES(?,?,?,?)',('fixture','legacy-'+str(index),digest(receipt),canonical(receipt)))
        state,_,history=self.store.snapshot_components('fixture')
        self.assertEqual(len(history),16);self.assertEqual(history[0]['after_revision'],state['revision'])
        self.assertTrue(all(row['unit']=='0..1' for row in history))
        public=canonical(self.adapter.snapshot())
        self.assertNotIn('legacy-private',public);self.assertNotIn('signed-private',public)

    def test_snapshot_state_and_history_share_one_transaction(self):
        with sqlite3.connect(self.fixture.path) as db:db.execute('PRAGMA journal_mode=WAL')
        original=self.store._state;written=False
        def interleaved(row):
            nonlocal written
            state=original(row)
            if not written:
                written=True
                self.store.update_tendencies('fixture','during-read',{'social_recovery_rate':.9},
                    {'social_recovery_rate':{'unit':'0..1','source_ref':'review:during-read'}},
                    expected_revision=0,occurred_at=base.WHEN)
            return state
        with patch.object(self.store,'_state',side_effect=interleaved):snapshot=self.adapter.snapshot()
        self.assertEqual(snapshot['revision'],0)
        self.assertIsNone(snapshot['tendencies']['social_recovery_rate'])
        self.assertEqual(snapshot['source_refs'],[])
        self.assertEqual(self.adapter.snapshot()['revision'],1)

    def test_tendency_review_is_not_relabelled_as_learning_or_operation(self):
        for index,value in enumerate((.3,.7)):
            self.store.update_tendencies('fixture','review-'+str(index),{'social_recovery_rate':value},
                {'social_recovery_rate':{'unit':'0..1','source_ref':'canonical-review-'+str(index)}},
                expected_revision=index,occurred_at=base.WHEN)
        change=next(row for row in self.adapter.snapshot()['recent_changes'] if row['key']=='social_recovery_rate')
        self.assertEqual(change['before'],.3);self.assertEqual(change['after'],.7)
        self.assertEqual(change['reason_code'],'tendency_review_update')
        self.assertEqual(change['source_refs'],[{'kind':'tendency_review','event_id':'tendency:review-1',
            'occurred_at':base.WHEN,'source_ref':'canonical-review-1'}])


if __name__=='__main__':unittest.main()
