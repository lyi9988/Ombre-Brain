import unittest
from datetime import timedelta
import test_persona_b1_layers as fixtures
layers_policy=fixtures.layers_policy
from test_persona_b1 import NOW
from persona_b1 import FIELDS
class TimeLayers(unittest.TestCase):
    setUp=fixtures.Layers.setUp
    make=fixtures.Layers.make
    # Run via explicit test names below; imported prior class is not new coverage.
    def test_base_hold_same_time_uneven_clock_calls(self):
        p=layers_policy();p['mood_policy']={};p['laws']['valence']={'half_life_hours':1,'target':.9,'growth_per_hour':0};p['laws']['arousal']={'half_life_hours':None,'target':0,'growth_per_hour':0}
        seed={k:.5 for k in FIELDS};seed['valence']=.65;seed['arousal']=.4
        sa,a=self.make('whole',p,seed);sb,b=self.make('parts',p,seed)
        a.advance_clock('init',NOW,p['revision']);b.advance_clock('init',NOW,p['revision'])
        a.advance_clock('end',NOW+timedelta(minutes=19,seconds=37),p['revision'])
        for i,seconds in enumerate((17,71,219,460,619,1001,1177)):
            b.advance_clock(str(i),NOW+timedelta(seconds=seconds),p['revision'])
        self.assertEqual(sa.read('guyan')['base_label'],sb.read('guyan')['base_label'])
        self.assertEqual(sa.read('guyan')['pending_base'],sb.read('guyan')['pending_base'])
        self.assertEqual(sa.read('guyan')['base_label'],'secure')
    def test_observation_budget_failure_rolls_back_all(self):
        p=layers_policy();s,a=self.make(p=p);before=s.read('guyan')
        with self.assertRaisesRegex(RuntimeError,'observation budget'):
            a.advance_clock('too-long',NOW+timedelta(days=7),p['revision'])
        self.assertEqual(s.read('guyan'),before);self.assertIsNone(s.receipt('guyan','clock:too-long'))
    def test_coupled_mood_interval_partition_tolerance(self):
        p=layers_policy();p['presentation']={};p['couplings']={'valence':{'anger':1,'grieve':-1,'favored':-1}}
        p['laws']['valence']={'half_life_hours':.6931471805599453/.001,'target':0,'growth_per_hour':0}
        for key,rate in [('anger',.2),('grieve',1),('favored',.05)]:p['laws'][key]={'half_life_hours':.6931471805599453/rate,'target':0,'growth_per_hour':0}
        seed={k:.5 for k in FIELDS};seed.update(anger=.8,grieve=.5,favored=.4)
        sa,a=self.make('whole',p,seed);sb,b=self.make('parts',p,seed)
        a.advance_clock('end',NOW+timedelta(hours=24),p['revision'])
        for i in range(1,25):b.advance_clock(str(i),NOW+timedelta(hours=i),p['revision'])
        self.assertAlmostEqual(a.snapshot()['slow_axes']['mood']['value'],b.snapshot()['slow_axes']['mood']['value'],places=7)

    def test_optional_zero_clock_does_not_change_first_minute_base(self):
        p=layers_policy();p['mood_policy']={};p['laws']['valence']={'half_life_hours':1,'target':.9,'growth_per_hour':0};p['laws']['arousal']={'half_life_hours':None,'target':0,'growth_per_hour':0}
        seed={k:.5 for k in FIELDS};seed.update(valence=.679,arousal=.4)
        sa,a=self.make('whole',p,seed);sb,b=self.make('zero',p,seed)
        b.advance_clock('zero',NOW,p['revision'])
        a.advance_clock('end',NOW+timedelta(minutes=1),p['revision']);b.advance_clock('end',NOW+timedelta(minutes=1),p['revision'])
        self.assertEqual(sa.read('guyan')['base_label'],sb.read('guyan')['base_label'])
        self.assertEqual(sa.read('guyan')['pending_base'],sb.read('guyan')['pending_base'])
        self.assertEqual(sa.read('guyan')['base_label'],'calm')
