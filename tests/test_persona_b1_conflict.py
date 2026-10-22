import unittest
from datetime import timedelta
import test_persona_b1_layers as fixtures
from test_persona_b1 import NOW,turn,event
from persona_b1 import FIELDS
from persona_b1_ceiling import ceiling_advance
from persona_b1_conflict import transferred_grief

def conflict_policy():
    p=fixtures.layers_policy();p['presentation']={};p['mood_policy']={};p['couplings']={}
    p['conflict_policy']={'anger_to_grief':.3,'max_fade_hours':10,'soothed_fade_hours':4,'clear_below':.03,'soothe_types':['affection','intimacy'],'soothe_first':.1,'soothe_step':.05,'soothe_max':.4,'soothe_floor':.02}
    p['laws']['anger']={'half_life_hours':2,'target':0,'growth_per_hour':0};p['laws']['grieve']={'half_life_hours':3,'target':0,'growth_per_hour':0}
    return p
class Episodes(unittest.TestCase):
    setUp=fixtures.Layers.setUp
    make=fixtures.Layers.make
    def start(self,name='a',p=None):
        s,a=self.make(name,p or conflict_policy());t=turn();e=event(t,'conflict',labels=['hurt','angry']);a.settle(e,t);return s,a
    def test_first_source_preserved_and_retry_no_repeat(self):
        s,a=self.start();before=s.read('guyan');t=turn();a.settle(event(t,'conflict',labels=['hurt','angry']),t)
        self.assertEqual(before,s.read('guyan'))
        t=turn('next');a.settle(event(t,'conflict',NOW+timedelta(minutes=1)),t)
        after=s.read('guyan');self.assertEqual(after['grudge'],before['grudge']);self.assertEqual(after['conflict_episode']['conflicts'],2)
    def test_soothing_not_reconciliation(self):
        s,a=self.start();t=turn('next');a.settle(event(t,'affection',NOW+timedelta(minutes=1)),t)
        q=s.read('guyan');self.assertFalse(q['conflict_episode']['reconciled']);self.assertIsNotNone(q['grudge']);self.assertEqual(q['conflict_episode']['soothes'],1)
    def test_reconciliation_stops_transfer_not_erases_grief(self):
        s,a=self.start();t=turn('next');a.settle(event(t,'reconciliation',NOW+timedelta(hours=1)),t);before=s.read('guyan')
        self.assertTrue(before['conflict_episode']['reconciled']);self.assertIsNone(before['grudge'])
        a.advance_clock('after',NOW+timedelta(hours=2),conflict_policy()['revision'])
        self.assertAlmostEqual(s.read('guyan')['values']['grieve'],before['values']['grieve']*2**(-1/3))
    def test_transfer_expiry_whole_split(self):
        sa,a=self.start('whole');sb,b=self.start('split');p=conflict_policy()
        a.advance_clock('whole',NOW+timedelta(hours=14),p['revision'])
        for i in range(1,29):b.advance_clock(str(i),NOW+timedelta(hours=i/2),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['grieve'],sb.read('guyan')['values']['grieve'],places=10)
        self.assertIsNone(sa.read('guyan')['grudge']);self.assertEqual(sa.read('guyan')['conflict_episode']['close_reason'],'faded')
    def test_transfer_equal_rate_and_saturation_partition(self):
        for ka,kg,g,a in ((.5,.5,.2,.8),(2,.1,.9,1)):
            whole=transferred_grief(g,a,ka,kg,.8,5);gg=g;aa=a
            import math
            for i in range(50):gg=transferred_grief(gg,aa,ka,kg,.8,.1);aa*=math.exp(-ka*.1)
            self.assertAlmostEqual(whole,gg,places=11)
    def test_invalid_assertion_does_not_open_episode(self):
        for field,value in (('context','quoted'),('assertion','negated'),('actor','third_party')):
            s,a=self.make(field,conflict_policy());t=turn();e=event(t,'conflict');e['decision']['accepted'][0][field]=value
            with self.assertRaises(ValueError):a.settle(e,t)
            self.assertIsNone(s.read('guyan')['conflict_episode'])
    def test_ceiling_preserves_peak_and_partition(self):
        for initial in (.2,.8,1):
            whole=ceiling_advance(initial,10,.6,.05,.2);split=initial
            for i in range(100):split=ceiling_advance(split,.1,.6,.05,.2)
            self.assertAlmostEqual(whole,split,places=12)
        self.assertGreater(ceiling_advance(1,1,.6,.05,.2),.6)
        self.assertEqual(ceiling_advance(1,0,.6,.05,.2),1)
        self.assertEqual(ceiling_advance(.2,1,.6,.05,.2,True),.2)
        self.assertLess(ceiling_advance(1,1,.6,.05,.2,True),1)
    def test_ceiling_real_store_not_tick_delta(self):
        p=conflict_policy();p['laws']['libido']={'half_life_hours':None,'target':0,'growth_per_hour':.05,'rest_ceiling':.6,'relax_per_hour':.2}
        seed={k:.5 for k in FIELDS};seed['libido']=1
        sa,a=self.make('whole',p,seed);sb,b=self.make('split',p,seed)
        a.advance_clock('whole',NOW+timedelta(hours=10),p['revision'])
        for i in range(1,21):b.advance_clock(str(i),NOW+timedelta(hours=i/2),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['libido'],sb.read('guyan')['values']['libido'],places=12)
    def test_transfer_feeds_emotion_same_trajectory_whole_split(self):
        p=conflict_policy();p['couplings']={'valence':{'grieve':-.3,'anger':-.2}};p['laws']['valence']={'half_life_hours':2,'target':.6,'growth_per_hour':0}
        sa,a=self.start('whole',p);sb,b=self.start('split',p)
        a.advance_clock('whole',NOW+timedelta(hours=12),p['revision'])
        for i in range(1,25):b.advance_clock(str(i),NOW+timedelta(hours=i/2),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['valence'],sb.read('guyan')['values']['valence'],places=7)
    def test_ceiling_source_emotion_does_not_assume_one_exponential(self):
        p=conflict_policy();p['conflict_policy']={};p['laws']['favored']={'half_life_hours':None,'target':0,'growth_per_hour':.1,'rest_ceiling':.7,'relax_per_hour':.2};p['couplings']={'valence':{'favored':-.3}};p['laws']['valence']={'half_life_hours':2,'target':.6,'growth_per_hour':0}
        seed={k:.3 for k in FIELDS};sa,a=self.make('whole',p,seed);sb,b=self.make('split',p,seed)
        a.advance_clock('whole',NOW+timedelta(hours=12),p['revision'])
        for i in range(1,25):b.advance_clock(str(i),NOW+timedelta(hours=i/2),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['valence'],sb.read('guyan')['values']['valence'],places=9)
    def test_explicit_night_dawn_boundaries_not_sleep_guess(self):
        p=conflict_policy();p['conflict_policy']={};p['laws']['libido']={'half_life_hours':None,'target':0,'growth_per_hour':.01}
        p['rhythm_policy']={'offset_minutes':0,'night_start':22*60,'night_end':6*60,'dawn_start':60,'dawn_end':8*60,'night_multipliers':{'libido':.4},'dawn_freeze':['libido']}
        sa,a=self.make('whole',p);sb,b=self.make('split',p)
        a.advance_clock('whole',NOW+timedelta(hours=24),p['revision'])
        for i in range(1,49):b.advance_clock(str(i),NOW+timedelta(minutes=30*i),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['libido'],sb.read('guyan')['values']['libido'],places=12)
        self.assertAlmostEqual(sa.read('guyan')['values']['libido'],.5+.01*(14+3*.4))
    def test_episode_security_drift_stops_at_exact_fade(self):
        p=conflict_policy();p['conflict_policy']['security_drift_per_hour']=-.01;p['conflict_policy']['max_fade_hours']=3;p['laws']['slow_security']={'half_life_hours':None,'target':0,'growth_per_hour':0};p['slow_axis_events']={}
        sa,a=self.start('whole',p);sb,b=self.start('split',p)
        a.advance_clock('whole',NOW+timedelta(hours=5),p['revision'])
        for i in range(1,11):b.advance_clock(str(i),NOW+timedelta(hours=i/2),p['revision'])
        self.assertAlmostEqual(sa.read('guyan')['values']['slow_security'],.47,places=8)
        self.assertAlmostEqual(sa.read('guyan')['values']['slow_security'],sb.read('guyan')['values']['slow_security'],places=8)
