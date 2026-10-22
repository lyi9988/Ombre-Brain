import unittest,json
from datetime import timedelta
import test_persona_b1_layers as fixtures
from test_persona_b1 import NOW,turn,event
from persona_b1 import FIELDS
from b1_growth_protocol import derive_frame

def growth_policy():
    p=fixtures.layers_policy();p['presentation']={};p['mood_policy']={}
    p['growth_policy']={'revision':'growth-1','grid_seconds':3600,'source_scope':'guyan','source_policy_revision':'source-1','targets':{'libido':{'emotion':None,'drives':{'grieve':1},'floor':0,'ceiling':2}}}
    p['events']['intimacy']={'libido':{'delta':.1,'satisfaction':0,'floor':0,'plateau_hours':0}}
    p['laws']['libido']={'half_life_hours':None,'target':0,'growth_per_hour':.05,'rest_ceiling':.8,'relax_per_hour':.2}
    return p
class Growth(unittest.TestCase):
    setUp=fixtures.Layers.setUp
    make=fixtures.Layers.make
    def loader(self,p,calls,unknown=False):
        def load(at):
            calls.append(at)
            return derive_frame(p['growth_policy'],at,{'scope':'guyan','policy_revision':'source-1','revision':1,'as_of':at.isoformat(),'values':{'valence':.5,'arousal':.5,'grieve':None if unknown else .5,'anger':0,'favored':0}})
        return load
    def test_factor_only_growth_and_persisted_grid_not_reread(self):
        p=growth_policy();calls=[];s,a=self.make(p=p,growth_frame_loader=self.loader(p,calls))
        a.advance_clock('half',NOW+timedelta(minutes=30),p['revision']);a.advance_clock('whole',NOW+timedelta(hours=1),p['revision'])
        self.assertEqual(len(calls),1);self.assertAlmostEqual(s.read('guyan')['values']['libido'],.575)
        before=s.read('guyan');a.advance_clock('whole',NOW+timedelta(hours=1),p['revision']);self.assertEqual(s.read('guyan'),before);self.assertEqual(len(calls),1)
    def test_decay_above_ceiling_not_scaled_by_factor(self):
        p=growth_policy();seed={k:.5 for k in FIELDS};seed['libido']=1;s,a=self.make(p=p,seed=seed,growth_frame_loader=self.loader(p,[]))
        a.advance_clock('one',NOW+timedelta(hours=1),p['revision'])
        import math
        self.assertAlmostEqual(s.read('guyan')['values']['libido'],.8+.2*math.exp(-.2))
    def test_unknown_only_target_preserves_last_known_not_current(self):
        p=growth_policy();s,a=self.make(p=p,growth_frame_loader=self.loader(p,[],True));a.advance_clock('one',NOW+timedelta(hours=1),p['revision']);r=s.read('guyan')
        self.assertIsNone(r['values']['libido']);self.assertIsNotNone(r['values']['longing']);self.assertEqual(r['growth_coverage']['libido']['last_known'],.5);self.assertEqual(r['growth_coverage']['libido']['status'],'unknown')
        from persona_projection import safe_persona_projection
        projected=safe_persona_projection(a.snapshot())
        self.assertEqual(projected['affect']['growth_coverage']['libido']['last_known'],.5);self.assertIsNone(projected['drive']['libido'])
    def test_tampered_factor_rejected_even_rehashed(self):
        p=growth_policy();calls=[];load=self.loader(p,calls)
        from b1_growth_protocol import digest
        def bad(at):
            f=load(at);f['targets']['libido']['factor']=9;f['frame_hash']=digest({k:v for k,v in f.items() if k!='frame_hash'});return f
        s,a=self.make(p=p,growth_frame_loader=bad)
        with self.assertRaises(ValueError):a.advance_clock('bad',NOW+timedelta(hours=1),p['revision'])
        self.assertEqual(s.read('guyan')['revision'],0)
    def test_whole_split_and_new_event_delta_not_multiplied(self):
        p=growth_policy();sa,a=self.make('whole',p,growth_frame_loader=self.loader(p,[]));sb,b=self.make('split',p,growth_frame_loader=self.loader(p,[]))
        a.advance_clock('whole',NOW+timedelta(hours=2),p['revision'])
        for i in range(1,9):b.advance_clock(str(i),NOW+timedelta(minutes=15*i),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['libido'],sb.read('guyan')['values']['libido'],places=12)
        t=turn();e=event(t,'intimacy',NOW+timedelta(hours=2));before=sa.read('guyan')['values']['libido'];a.settle(e,t)
        r=p['events']['intimacy']['libido'];expected=before-max(0,before-r['floor'])*r['satisfaction']+r['delta']
        self.assertAlmostEqual(sa.read('guyan')['values']['libido'],min(1,max(0,expected)))
