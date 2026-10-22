import tempfile,unittest
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
from persona_b1 import PersonaB1Store,PersonaB1Adapter,FIELDS,Conflict
from persona_projection import safe_persona_projection
from test_persona_b1 import NOW,turn,event
from test_persona_b1_mechanisms import mechanism_policy

def layers_policy():
    p=mechanism_policy();p['couplings']={}
    p['laws']['slow_security']={'half_life_hours':72,'target':.6,'growth_per_hour':0}
    p['laws']['slow_confidence']={'half_life_hours':36,'target':.5,'growth_per_hour':0}
    p['slow_axis_events']={'affection':{'slow_security':.01},'conflict':{'slow_security':-.05},'reconciliation':{'slow_security':.08}}
    p['proof_policy']={'gains':{'her_ok':.03,'first':.03,'hard':.04,'helped_ack':.02},'daily_factor':.6,'high_threshold':.75,'high_factor':.5}
    p['mood_policy']={'window_days':3,'offset_minutes':0,'midpoint':.5,'span':.5}
    p['presentation']={'base_thresholds':{'pleasant_high':.68,'active_high':.5,'pleasant_low':.38,'relaxed_valence':.58,'relaxed_arousal':.22,'tense_arousal':.62,'tired_arousal':.18},'base_margin':.1,'base_hold_hours':.1,'cause_hold_hours':.25,'cause_priorities':{'hurt':3,'angry':2,'tender':1,'desire':2},'label_rules':{'tender':{'key':'tenderness','minimum':.2,'maximum':1},'hurt':{'key':'grieve','minimum':.2,'maximum':1},'angry':{'key':'anger','minimum':.2,'maximum':1},'clingy':{'key':'cling','minimum':.1,'maximum':1},'desire':{'key':'libido','minimum':.1,'maximum':1}},'tones':{'warm':{'valence':.8,'arousal':.4,'weight':.5}}}
    p['sleep_multipliers']['valence']=2;p['sleep_multipliers']['arousal']=2
    return p

class Layers(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.old=self.root/'old';self.old.write_text('unchanged')
    def make(self,name='a',p=None,seed=None,**kwargs):
        p=p or layers_policy();seed=seed or {k:.5 for k in FIELDS}
        s=PersonaB1Store(self.root/(name+'.db'),legacy_path=self.old);s.initialize('guyan',seed,NOW,p)
        g={'relationship':{'affinity':.86,'trust':.82},'tendencies':{'exploration_affinity':.7,'reflection_preference':.4}}
        return s,PersonaB1Adapter(s,'guyan',g,**kwargs)
    def test_slow_axes_distinct_and_six_tendencies_retained(self):
        s,a=self.make();t=turn();r=a.settle(event(t),t)
        self.assertEqual(r['slow_axes']['security'],.51);self.assertEqual(r['affect']['security'],.5)
        self.assertEqual(r['slow_axes']['confidence'],.5)
        self.assertEqual(r['tendencies']['exploration_affinity'],.7);self.assertEqual(len(r['tendencies']),6)
        self.assertIsNone(r['tendencies']['social_recovery_rate']);self.assertEqual(r['relationship']['affinity'],.86)
        self.assertEqual(self.old.read_text(),'unchanged')
    def test_mood_no_log_unknown_then_time_weighted(self):
        p=layers_policy();p['laws']['valence']={'half_life_hours':None,'target':0,'growth_per_hour':0};seed={k:.8 for k in FIELDS}
        s,a=self.make(p=p,seed=seed);self.assertIsNone(a.snapshot()['slow_axes']['mood']['value'])
        a.advance_clock('clock',NOW+timedelta(hours=6),p['revision']);self.assertAlmostEqual(a.snapshot()['slow_axes']['mood']['value'],.6)
        b,ba=self.make('b',p,seed)
        for i in range(1,4):ba.advance_clock(str(i),NOW+timedelta(hours=2*i),p['revision'])
        self.assertAlmostEqual(a.snapshot()['slow_axes']['mood']['value'],ba.snapshot()['slow_axes']['mood']['value'])
        self.assertEqual(a.snapshot()['slow_axes']['mood']['covered_hours'],6)
    def test_slow_security_can_follow_continuing_drive_not_global_identity(self):
        p=layers_policy();p['couplings']['slow_security']={'favored':-.3,'anger':-.2};seed={k:.8 for k in FIELDS}
        s,a=self.make(p=p,seed=seed);r=a.advance_clock('one',NOW+timedelta(hours=12),p['revision'])
        self.assertLess(r['slow_axes']['security'],.8)
        self.assertEqual(r['relationship']['affinity'],.86);self.assertEqual(r['tendencies']['exploration_affinity'],.7)
    def test_base_label_waits_without_refresh_side_effect(self):
        p=layers_policy();p['laws']['valence']={'half_life_hours':None,'target':0,'growth_per_hour':0};p['laws']['arousal']={'half_life_hours':None,'target':0,'growth_per_hour':0}
        p['presentation']['tones']['warm']={'valence':.69,'arousal':.4,'weight':1};seed={k:.5 for k in FIELDS};seed['arousal']=.4;seed['valence']=.67
        s,a=self.make(p=p,seed=seed);a.advance_clock('initial',NOW,p['revision'])
        self.assertEqual(a.snapshot()['affect']['base_label'],'calm')
        t=turn();e=event(t,when=NOW+timedelta(minutes=1));e['tone']='warm';a.settle(e,t)
        self.assertEqual(a.snapshot()['affect']['base_label'],'calm');before=s.read('guyan');a.snapshot();self.assertEqual(before,s.read('guyan'))
        a.advance_clock('later',NOW+timedelta(minutes=8),p['revision']);self.assertEqual(a.snapshot()['affect']['base_label'],'secure')
    def test_unknown_valence_keeps_mood_unknown(self):
        seed={k:.5 for k in FIELDS};seed['valence']=None;s,a=self.make(seed=seed)
        a.advance_clock('clock',NOW+timedelta(hours=1),layers_policy()['revision'])
        self.assertIsNone(a.snapshot()['slow_axes']['mood']['value']);self.assertEqual(a.snapshot()['slow_axes']['mood']['unknown_hours'],1)
    def test_explicit_sleep_wake_changes_only_following_interval(self):
        p=layers_policy();p['laws']['valence']={'half_life_hours':2,'target':.2,'growth_per_hour':0};seed={k:.8 for k in FIELDS}
        s,a=self.make(p=p,seed=seed);a.set_phase('sleep',NOW,p['revision'],'sleep')
        r=a.advance_clock('one',NOW+timedelta(hours=1),p['revision']);self.assertAlmostEqual(r['affect']['valence'],.5)
        a.set_phase('wake',NOW+timedelta(hours=1),p['revision'],'wake')
        r=a.advance_clock('two',NOW+timedelta(hours=3),p['revision']);self.assertAlmostEqual(r['affect']['valence'],.35)
        self.assertFalse(r['sleeping'])
    def test_chat_cannot_claim_sleep_or_operation(self):
        s,a=self.make();e=event(turn());e['phase']='sleep'
        with self.assertRaises(ValueError):a.settle(e,turn())
        e=event(turn());e['operation_result']={}
        with self.assertRaises(ValueError):a.settle(e,turn())
    def test_unknown_sleep_not_assumed_awake(self):
        p=layers_policy();p['initial_sleeping']=None;s,a=self.make(p=p)
        a.advance_clock('clock',NOW+timedelta(hours=1),p['revision'])
        self.assertIsNone(a.snapshot()['affect']['valence']);self.assertIsNone(a.snapshot()['slow_axes']['mood']['value'])
    def test_real_result_resolver_confidence_and_replay(self):
        records={str(i):{'scope':'guyan','operation_id':str(i),'status':'succeeded','proof':'first','revision':1,'result_hash':str(i)*64,'occurred_at':(NOW+timedelta(seconds=i)).isoformat()} for i in (1,2)}
        s,a=self.make(result_resolver=lambda ref:deepcopy(records[ref]));first=a.record_operation('1',layers_policy()['revision'])
        self.assertAlmostEqual(first['slow_axes']['confidence'],.53)
        again=a.record_operation('1',layers_policy()['revision']);self.assertTrue(again['b1_receipt']['replayed'])
        second=a.record_operation('2',layers_policy()['revision']);self.assertGreater(second['slow_axes']['confidence'],.5479);self.assertLess(second['slow_axes']['confidence'],.5481)
    def test_failed_or_routine_result_no_credit(self):
        raw={'scope':'guyan','operation_id':'1','status':'failed','proof':'first','revision':1,'result_hash':'1'*64,'occurred_at':NOW.isoformat()}
        s,a=self.make(result_resolver=lambda ref:deepcopy(raw))
        with self.assertRaises(ValueError):a.record_operation('1',layers_policy()['revision'])
        raw.update(status='succeeded',proof='routine')
        with self.assertRaises(ValueError):a.record_operation('1',layers_policy()['revision'])
        self.assertEqual(s.read('guyan')['revision'],0)
    def test_source_tone_and_cause_do_not_erase_hurt(self):
        s,a=self.make();t=turn();e=event(t,'conflict',labels=['hurt','angry']);r=a.settle(e,t)
        self.assertEqual(r['affect']['cause_label']['label'],'hurt')
        t=turn('u2');e=event(t,'affection',NOW+timedelta(minutes=1),['tender']);e['tone']='warm';r=a.settle(e,t)
        self.assertEqual(r['affect']['cause_label']['label'],'hurt');self.assertIn('hurt',r['affect']['labels']);self.assertIn('tender',r['affect']['labels'])
        self.assertEqual(safe_persona_projection(r)['affect']['cause_label']['label'],'hurt')
    def test_source_label_retires_by_configured_state_not_ttl(self):
        p=layers_policy();p['laws']['libido']={'half_life_hours':1,'target':0,'growth_per_hour':0};s,a=self.make(p=p)
        t=turn();r=a.settle(event(t,'intimacy',labels=['desire']),t);self.assertIn('desire',r['affect']['labels'])
        r=a.advance_clock('late',NOW+timedelta(hours=4),p['revision']);self.assertNotIn('desire',r['affect']['labels']);self.assertIsNone(r['affect']['mood_label'])
        self.assertGreater(r['affect']['libido'],0)
        self.assertEqual(r['affect']['label_times']['desire'],NOW.isoformat(timespec='microseconds'))
    def test_missing_lifecycle_does_not_silently_delete_source_label(self):
        s,a=self.make();t=turn();r=a.settle(event(t,'affection',labels=['possessive']),t)
        self.assertIn('possessive',r['affect']['uncertain_labels']);self.assertIn('possessive',r['affect']['labels'])
        self.assertGreater(r['affect']['possessiveness'],.5)


    def test_sleep_phase_does_not_accept_numeric_boolean(self):
        for value in (0,1,"false"):
            p=layers_policy();p["initial_sleeping"]=value
            with self.assertRaises(ValueError):self.make(str(value),p=p)

    def test_zero_elapsed_phase_confirmation_preserves_known_seed(self):
        for phase in ('wake','sleep'):
            p=layers_policy();p['initial_sleeping']=None
            s,a=self.make(phase,p=p)
            r=a.set_phase('phase',NOW,p['revision'],phase)
            self.assertEqual(r['affect']['valence'],.5)
            self.assertIs(r['sleeping'],phase=='sleep')
