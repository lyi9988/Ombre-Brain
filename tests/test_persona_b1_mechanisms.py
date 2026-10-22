import math,sqlite3,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch
from persona_projection import safe_persona_projection
from persona_b1 import PersonaB1Store,PersonaB1Adapter,FIELDS
from persona_b1_coupling import coupled,IntegrationBudgetError
from test_persona_b1 import config,event,turn,NOW

def mechanism_policy():
    p=config();p['couplings']={'valence':{'grieve':-.35,'anger':-.25,'favored':-.15},'arousal':{'anger':.3,'grieve':-.05}}
    p['subtype_policy']={'bump':.4,'visible_weight':.1,'clear_below':.1}
    p['mixed_policy']={'negative_min':.3,'positive_min':.6,'hold_hours':.25}
    p['cling_relief']={'threshold':.1,'factor':.5}
    return p
class Mechanisms(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.primary=self.root/'legacy';self.primary.write_text('untouched')
    def make(self,name='one',p=None,seed=None):
        p=p or mechanism_policy();seed=seed or {k:.5 for k in FIELDS}
        s=PersonaB1Store(self.root/(name+'.db'),legacy_path=self.primary);s.initialize('guyan',seed,NOW,p)
        return s,PersonaB1Adapter(s,'guyan',{})
    def test_analytic_convolution_equal_decay_limit(self):
        law={'half_life_hours':2,'target':.5,'growth_per_hour':0};sources={'anger':.8};laws={'anger':{**law,'target':0}}
        y,m=coupled(.5,law,3,sources,{'anger':-.2},laws)
        k=math.log(2)/2;expected=.5-.2*.8*k*3*math.exp(-k*3)
        self.assertAlmostEqual(y,expected,places=14);self.assertEqual(m['method'],'analytic_convolution')
    def test_coupled_whole_split_same_and_no_false_happy_reset(self):
        s,a=self.make();t=turn();a.settle(event(t,'conflict',labels=['angry','hurt']),t)
        a.advance_clock('1',NOW+timedelta(hours=1),config()['revision']);a.advance_clock('2',NOW+timedelta(hours=3),config()['revision'])
        whole,b=self.make('whole');b.settle(event(t,'conflict',labels=['angry','hurt']),t);b.advance_clock('whole',NOW+timedelta(hours=3),config()['revision'])
        for key in FIELDS:self.assertAlmostEqual(s.read('guyan')['values'][key],whole.read('guyan')['values'][key],places=11)
        self.assertLess(s.read('guyan')['values']['valence'],.5)
    def test_clipped_targets_numeric_convolution_partition(self):
        p=mechanism_policy();p['couplings']['valence']={'anger':-1,'grieve':-1}
        p['laws']['anger']={'half_life_hours':1,'target':0,'growth_per_hour':0};p['laws']['grieve']={'half_life_hours':3,'target':0,'growth_per_hour':0}
        seed={k:.8 for k in FIELDS}
        s,a=self.make(p=p,seed=seed);r=a.advance_clock('all',NOW+timedelta(hours=6),p['revision'])
        whole=s.read('guyan')['values']['valence'];other,b=self.make('split',p,seed)
        for i in range(1,7):b.advance_clock(str(i),NOW+timedelta(hours=i),p['revision'])
        self.assertAlmostEqual(whole,other.read('guyan')['values']['valence'],places=8)
        self.assertIn('bounded_convolution',[m['method'] for m in r['b1_receipt']['receipt']['time_methods']])
    def test_narrow_positive_lobe_not_missed(self):
        law={'half_life_hours':math.log(2)/.001,'target':0,'growth_per_hour':0}
        source={'anger':.8,'grieve':.5,'favored':.4};rates={'anger':.2,'grieve':1,'favored':.05}
        laws={key:{'half_life_hours':math.log(2)/q,'target':0,'growth_per_hour':0} for key,q in rates.items()}
        coefficients={'anger':1,'grieve':-1,'favored':-1}
        whole,method=coupled(.5,law,1000,source,coefficients,laws)
        state=dict(source);split=.5
        for _ in range(1000):
            split,_=coupled(split,law,1,state,coefficients,laws)
            state={key:value*math.exp(-rates[key]) for key,value in state.items()}
        self.assertAlmostEqual(whole,split,places=9)
        self.assertGreater(whole,.184);self.assertGreater(method['evaluations'],5)
        self.assertLessEqual(method['absolute_enclosure_bound'],1e-10)
    def test_unknown_source_does_not_become_zero(self):
        seed={k:.5 for k in FIELDS};seed['anger']=None;s,a=self.make(seed=seed)
        r=a.advance_clock('clock',NOW+timedelta(hours=1),config()['revision'])
        self.assertIsNone(r['affect']['valence']);self.assertIsNone(r['affect']['arousal'])
        self.assertIn('unknown_input',[m['method'] for m in r['b1_receipt']['receipt']['time_methods']])
    def test_precision_failure_rolls_back_not_applied(self):
        s,a=self.make()
        with patch('persona_b1_coupling.coupled',side_effect=IntegrationBudgetError('budget')):
            with self.assertRaises(IntegrationBudgetError):a.advance_clock('fail',NOW+timedelta(hours=1),config()['revision'])
        self.assertEqual(s.read('guyan')['revision'],0);self.assertIsNone(s.receipt('guyan','clock:fail'))
    def test_subtype_requires_evidence_and_decays(self):
        s,a=self.make();t=turn();e=event(t,'loss',labels=['hurt'])
        e['nuances']=[dict(type='loss',key='grieve',label='委屈',evidence_hash=e['decision']['accepted'][0]['evidence_hash'])]
        a.settle(e,t);self.assertEqual(a.snapshot()['affect']['subtypes'][0]['label'],'委屈')
        self.assertEqual(safe_persona_projection(a.snapshot())['affect']['subtypes'][0]['label'],'委屈')
        a.advance_clock('after',NOW+timedelta(hours=12),config()['revision'])
        self.assertEqual(s.read('guyan')['causes'][0]['kind'],'loss')
        self.assertEqual(a.snapshot()['affect']['subtypes'],[])
        self.assertEqual(safe_persona_projection(a.snapshot())['affect']['subtypes'],[])
        t=turn('u2');bad=event(t,'loss',when=NOW+timedelta(hours=12));bad['nuances']=[dict(type='loss',key='grieve',label='委屈',evidence_hash='0'*64)]
        with self.assertRaises(ValueError):a.settle(bad,t)
    def test_heartache_and_longing_not_forced_into_conflict(self):
        seed={k:.8 for k in FIELDS};s,a=self.make(seed=seed);t=turn();e=event(t,'empathy',labels=['tender','longing'])
        e['nuances']=[dict(type='empathy',key='grieve',label='心疼',evidence_hash=e['decision']['accepted'][0]['evidence_hash'])]
        a.settle(e,t);a.advance_clock('wait',NOW+timedelta(hours=.5),config()['revision'])
        self.assertNotIn('grieve_closeness',s.read('guyan')['mixed'])
    def test_mixed_crossing_time_independent_of_tick(self):
        p=mechanism_policy();p['couplings']={};seed={k:.7 for k in FIELDS};seed['anger']=.1
        p['laws']['anger']={'half_life_hours':None,'target':0,'growth_per_hour':.2}
        for k in ('longing','possessiveness','protective_drive'):p['laws'][k]={'half_life_hours':None,'target':0,'growth_per_hour':0}
        s,a=self.make(p=p,seed=seed);a.advance_clock('all',NOW+timedelta(hours=2),p['revision'])
        other,b=self.make('split',p,seed)
        for i in range(1,9):b.advance_clock(str(i),NOW+timedelta(hours=i/4),p['revision'])
        self.assertEqual(s.read('guyan')['mixed']['anger_closeness'],other.read('guyan')['mixed']['anger_closeness'])
        self.assertEqual(s.read('guyan')['mixed']['anger_closeness']['status'],'visible')
    def test_possessiveness_not_lost_when_longing_low(self):
        seed={k:.7 for k in FIELDS};seed['longing']=.05;s,a=self.make(seed=seed)
        a.advance_clock('1',NOW+timedelta(hours=.5),config()['revision'])
        self.assertEqual(s.read('guyan')['mixed']['anger_closeness']['status'],'visible')
    def test_cling_relief_keeps_more_want_without_penalizing_intimacy(self):
        seed={k:.5 for k in FIELDS};seed['cling']=.4;s,a=self.make(seed=seed)
        t=turn();r=a.settle(event(t),t)
        self.assertAlmostEqual(r['affect']['longing'],.5*(1-.3*.5))
        self.assertGreater(r['affect']['possessiveness'],.5);self.assertGreater(r['affect']['cling'],.4)
    def test_unknown_mixed_does_not_mean_absent(self):
        seed={k:.7 for k in FIELDS};seed['possessiveness']=None;seed['longing']=.05;s,a=self.make(seed=seed)
        a.advance_clock('clock',NOW+timedelta(minutes=1),config()['revision'])
        self.assertEqual(s.read('guyan')['mixed']['anger_closeness']['status'],'unknown')

if __name__=='__main__':unittest.main()
