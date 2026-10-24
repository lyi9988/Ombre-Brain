"""Source-backed tendency CAS; no automatic learning or guessed values."""
from copy import deepcopy
import unittest

import test_persona_inner_state_v2 as base
from persona_b1 import Conflict
from persona_inner_state import TENDENCIES, tendency_factor


class TendencyTests(unittest.TestCase):
    def setUp(self):
        fixture=base.StoreCase();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.store=fixture.store;self.adapter=fixture.adapter

    def update(self,event='review-1',changes=None,revision=0,when=base.WHEN,sources=None):
        changes={'social_recovery_rate':.8,'creative_satisfaction':.2} if changes is None else changes
        sources={key:{'unit':'0..1','source_ref':'review:'+event+':'+key} for key in changes} if sources is None else sources
        return self.store.update_tendencies('fixture',event,changes,sources,
                                           expected_revision=revision,occurred_at=when)

    def test_cas_and_numeric_are_one_checkpoint_with_explicit_sources(self):
        before=self.store.read('fixture');result=self.update();state=self.store.read('fixture')
        self.assertFalse(result['replayed']);self.assertEqual(state['revision'],1)
        self.assertEqual(state['values'],before['values'])
        self.assertEqual(state['at'],before['at'])
        self.assertEqual(state['tendencies']['social_recovery_rate'],.8)
        self.assertEqual(state['tendency_sources']['social_recovery_rate']['source_ref'],'review:review-1:social_recovery_rate')
        numeric=self.store.numeric_at('fixture',base.WHEN)
        self.assertEqual(numeric['revision'],state['revision'])
        self.assertEqual(numeric['policy_revision'],self.fixture.policy['revision'])
        self.assertEqual({key:numeric['values'][key] for key in TENDENCIES},state['tendencies'])
        self.assertFalse(result['receipt']['automatic_learning'])

    def test_original_identity_replays_but_changed_source_conflicts(self):
        first=self.update();replayed=self.update()
        self.assertTrue(replayed['replayed']);self.assertEqual(replayed['receipt'],first['receipt'])
        with self.assertRaises(Conflict):self.update(changes={'social_recovery_rate':.3})
        self.assertEqual(self.store.read('fixture')['revision'],1)

    def test_stale_revision_never_overwrites_other_update(self):
        self.update()
        with self.assertRaises(Conflict):self.update('another',{'creative_satisfaction':1},revision=0)
        self.assertEqual(self.store.read('fixture')['tendencies']['creative_satisfaction'],.2)
        self.assertIsNone(self.store.receipt('fixture','tendency:another'))

    def test_update_does_not_advance_or_backdate_the_shared_checkpoint(self):
        for when in ('2026-10-22T23:59:59+00:00','2026-10-23T00:00:01+00:00'):
            with self.assertRaises(Conflict):self.update(when=when)
        self.assertEqual(self.store.read('fixture')['revision'],0)

    def test_unknown_is_explicit_and_retains_its_review_source(self):
        self.update()
        self.update('clear',{'social_recovery_rate':None},revision=1)
        state=self.store.read('fixture')
        self.assertIsNone(state['tendencies']['social_recovery_rate'])
        self.assertEqual(state['tendency_sources']['social_recovery_rate'],
                         {'unit':'0..1','source_ref':'review:clear:social_recovery_rate','status':'unknown'})
        self.assertIsNone(self.store.numeric_at('fixture')['values']['social_recovery_rate'])

    def test_undefined_or_missing_scale_and_invalid_values_are_rejected(self):
        for source in ({'source_ref':'source'}, {'unit':'source_native','source_ref':'source'},
                       {'unit':'0..1','source_ref':''}):
            with self.assertRaises(ValueError):self.update(changes={'social_recovery_rate':.7},sources={'social_recovery_rate':source})
        for value in (True,1.1,-.1,float('nan')):
            with self.assertRaises(ValueError):self.update(changes={'social_recovery_rate':value})
        self.assertEqual(self.store.read('fixture')['revision'],0)

    def test_transaction_failure_rolls_back_tendency_receipt_and_numeric(self):
        before=self.store.read('fixture');numeric=self.store.numeric_at('fixture')
        def fail(_):raise OSError('fixture before commit')
        self.store._before_commit=fail
        with self.assertRaises(OSError):self.update()
        self.assertEqual(self.store.read('fixture'),before)
        self.assertEqual(self.store.numeric_at('fixture'),numeric)
        self.assertIsNone(self.store.receipt('fixture','tendency:review-1'))

    def test_only_approved_consumers_have_a_factor(self):
        for name in ('social_recovery_rate','creative_satisfaction'):
            for value,expected in ((None,1),(0,.8),(.5,1),(1,1.2)):
                self.assertEqual(tendency_factor(name,value),expected)
        for name in set(TENDENCIES)-{'social_recovery_rate','creative_satisfaction'}:
            with self.assertRaises(ValueError):tendency_factor(name,.5)
        self.assertTrue(all(value is None for value in self.store.read('fixture')['tendencies'].values()))


if __name__=='__main__':unittest.main()
