"""Real SQLite and actual PersonaStateEngine entries; synthetic assessments only."""
import asyncio
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sqlite3,tempfile,unittest
from unittest.mock import patch
from persona_b1 import PersonaB1Store,PersonaB1Adapter,FIELDS,Conflict,digest,advance
from persona_engine import PersonaStateEngine
from persona_projection import safe_persona_projection
NOW=datetime(2026,10,22,12,tzinfo=timezone.utc)
def config():
    def r(delta=0,satisfaction=0,plateau=0):return dict(delta=delta,satisfaction=satisfaction,floor=0,plateau_hours=plateau)
    laws={k:dict(half_life_hours=4 if k in ('anger','grieve','favored') else 2,target=.5,growth_per_hour=0) for k in FIELDS}
    laws['cling']=dict(half_life_hours=.5,target=0,growth_per_hour=0)
    return dict(revision='fixture-not-user-calibration',laws=laws,repeat_hours=2,repeat_factor=.5,couplings={},subtype_policy={},mixed_policy={},cling_relief={},sleep_multipliers={k:1 for k in FIELDS},initial_sleeping=False,mood_policy={},presentation={},slow_axis_events={},proof_policy={},events={
      'affection':{'longing':r(satisfaction=.3,plateau=1),'tenderness':r(.2),'cling':r(.3),'possessiveness':r(.1)},
      'conflict':{'anger':r(.4),'grieve':r(.3),'valence':r(-.3),'security':r(-.2)},
      'empathy':{'anger':r(satisfaction=.2),'grieve':r(satisfaction=.15),'tenderness':r(.1)},
      'reconciliation':{'anger':r(satisfaction=.5),'grieve':r(satisfaction=.35),'security':r(.2)}})
def turn(id='u1'):
    return dict(conversation_id='conversation',user_event=dict(id=id,role='user',content='合成对话，不是用户经历。'),assistant_event=dict(id='a'+id,role='assistant',content='合成回应。'))
def event(t,kind='affection',when=NOW,labels=None):
    u,a=t['user_event'],t['assistant_event'];src={'conversation_key':digest(t['conversation_id']),'user_id':u['id'],'assistant_id':a['id'],'user':u['content'],'assistant':a['content']}
    obs=dict(type=kind,actor='owner',target='assistant',context='direct',assertion='occurred',certainty='supported',role='user',start=0,end=len(u['content']),evidence_hash=digest(u['content']))
    return dict(namespace='b1-completed-turn-v1',scope='guyan',event_id=u['id'],occurred_at=when.isoformat(),policy_revision=config()['revision'],decision=dict(accepted=[obs],rejected=[],source_hash=digest(src),interpretation_is_not_ground_truth=True),labels=labels or ['tender','clingy'])
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.legacy=self.root/'legacy.db';self.legacy.write_bytes(b'not touched')
        self.store=PersonaB1Store(self.root/'candidate.db',legacy_path=self.legacy)
        self.initial={k:.5 for k in FIELDS};self.initial['cling']=0
        self.store.initialize('guyan',self.initial,NOW,config())
        self.adapter=PersonaB1Adapter(self.store,'guyan',dict(relationship=dict(affinity=.86,trust=.82,defensiveness=.12),personality=dict(openness=.56)))
        self.engine=PersonaStateEngine(dict(buckets_dir=str(self.root),persona=dict(mode='disabled',profile_id='guyan')),str(self.legacy),b1_candidate=self.adapter)
    def run_event(self,t=None,e=None):
        t=t or turn();e=e or event(t)
        return asyncio.run(self.engine.update_from_exchange(t['conversation_id'],t['user_event']['content'],t['assistant_event']['content'],b1_event=e,b1_turn=t))
    def test_actual_engine_one_owner_projection(self):
        with patch.object(self.engine,'_apply_session_decay',side_effect=AssertionError('old decay')),patch.object(self.engine,'_apply_session_delta',side_effect=AssertionError('old delta')):
            s=self.run_event();p=safe_persona_projection(s)
        self.assertLess(s['affect']['longing'],.5);self.assertGreater(s['affect']['cling'],0);self.assertGreater(s['affect']['possessiveness'],.5)
        self.assertEqual(p['affect']['labels'],['tender','clingy']);self.assertEqual(p['affect']['tenderness'],s['affect']['tenderness']);self.assertEqual(p['drive']['possessiveness'],s['affect']['possessiveness'])
        self.assertEqual(self.legacy.read_bytes(),b'not touched');self.assertFalse(s['b1_receipt']['receipt']['global_personality_changed'])
    def test_repeated_read_does_not_advance_or_write(self):
        self.run_event();a=self.store.read('guyan')
        for _ in range(3):self.engine.get_current_state('other');asyncio.run(self.engine.build_pre_reply_guidance('conversation'))
        self.assertEqual(a,self.store.read('guyan'))
    def test_retry_receipt_and_source_conflict(self):
        a=self.run_event();b=self.run_event();self.assertTrue(b['b1_receipt']['replayed']);self.assertEqual(a['b1_receipt']['receipt'],b['b1_receipt']['receipt'])
        t=turn();t['user_event']['content']='changed'
        with self.assertRaises(Conflict):self.run_event(t,event(t))
    def test_source_mismatch_and_span_reject(self):
        for mutation in ('hash','span','assistant'):
            t=turn();e=event(t)
            if mutation=='hash':e['decision']['source_hash']='0'*64
            elif mutation=='span':e['decision']['accepted'][0]['evidence_hash']='0'*64
            else:t['assistant_event']['content']='changed'
            with self.subTest(mutation=mutation),self.assertRaises(Conflict):self.run_event(t,e)
        self.assertEqual(self.store.read('guyan')['revision'],0)
    def test_semantic_rejection_not_misread_conflict(self):
        for field,v in [('context','reported'),('context','ordinary_absence'),('context','connection_loss'),('assertion','negated'),('certainty','uncertain'),('actor','third_party')]:
            e=event(turn(),'conflict');e['decision']['accepted'][0][field]=v
            with self.subTest(field=field,v=v),self.assertRaises(ValueError):self.run_event(e=e)
        self.assertEqual(self.store.read('guyan')['revision'],0)
    def test_conflict_comfort_repair_keeps_residue(self):
        e=event(turn(),'conflict',labels=['angry','hurt']);a=self.run_event(e=e)
        t=turn('u2');b=self.run_event(t,event(t,'empathy',NOW+timedelta(minutes=1),['tender','hurt']))
        self.assertLess(b['affect']['anger'],a['affect']['anger']);self.assertGreater(b['affect']['grieve'],0)
        self.assertEqual(b['affect']['labels'],['tender','hurt'])
        t=turn('u3');c=self.run_event(t,event(t,'reconciliation',NOW+timedelta(minutes=2),['secure','tender']))
        self.assertLess(c['affect']['anger'],b['affect']['anger']);self.assertGreater(c['affect']['grieve'],0)
    def test_reconciliation_requires_owner(self):
        e=event(turn(),'reconciliation');r=e['decision']['accepted'][0];r.update(actor='assistant',target='owner',role='assistant')
        with self.assertRaises(ValueError):self.run_event(e=e)
    def test_unknown_stays_unknown_and_no_fake_calm(self):
        unknown={k:None for k in FIELDS};s=PersonaB1Store(self.root/'unknown.db',legacy_path=self.legacy);s.initialize('guyan',unknown,NOW,config())
        a=PersonaB1Adapter(s,'guyan',{});p=safe_persona_projection(a.snapshot())
        self.assertIsNone(p['affect']['label']);self.assertIsNone(p['affect']['valence']);self.assertIsNone(p['drive']['longing']);self.assertEqual(p['drive']['labels'],[])
        a.settle(event(turn()),turn());self.assertTrue(all(v is None for v in s.read('guyan')['values'].values()))
    def test_new_dimensions_distinct(self):
        self.assertTrue({'libido','grieve','anger','favored','arousal','possessiveness'}<=set(self.store.read('guyan')['values']))
        p=safe_persona_projection(self.adapter.snapshot());self.assertEqual(p['drive']['libido'],.5);self.assertNotIn('inner_thought',str(p))
    def test_global_snapshot_allowlist_and_session_defense_separate(self):
        a=PersonaB1Adapter(self.store,'guyan',{'relationship':{'affinity':.8,'private':'SECRET'},'personality':{'openness':.4,'private':'SECRET'},'private':'SECRET'})
        s=a.snapshot();self.assertNotIn('SECRET',str(s))
        self.assertEqual(s['affect']['session_defensiveness'],.5)
        self.assertIsNone(s['relationship']['defensiveness'])
    def test_private_text_and_receipts_not_projection(self):
        s=self.run_event();s['affect']['inner_thought']='PRIVATE_SENTINEL';s['private']='PRIVATE_SENTINEL';s['b1_receipt']['raw']='PRIVATE_SENTINEL'
        p=safe_persona_projection(s);self.assertNotIn('PRIVATE_SENTINEL',str(p));self.assertNotIn('source_hash',str(p));self.assertNotIn('b1_receipt',p)
    def test_finite_and_policy_strict(self):
        for value in (float('nan'),float('inf'),True):
            cfg=config();cfg['laws']['longing']['target']=value
            with self.assertRaises(ValueError):self.store.initialize('bad',self.initial,NOW,cfg)
        with self.assertRaises(KeyError):self.store.read('bad')
    def test_crash_rolls_back_event_and_checkpoint(self):
        with patch.object(self.store,'_before_commit',side_effect=RuntimeError('crash')):
            with self.assertRaises(RuntimeError):self.run_event()
        self.assertEqual(self.store.read('guyan')['revision'],0);self.assertIsNone(self.store.receipt('guyan','u1'))
    def test_restart_and_repeat_discount(self):
        self.run_event();self.store=PersonaB1Store(self.root/'candidate.db',legacy_path=self.legacy);self.adapter.store=self.store
        t=turn('u2');s=self.run_event(t,event(t,when=NOW+timedelta(minutes=1)))
        self.assertTrue(all(r['repeat_factor']==.5 for r in s['b1_receipt']['receipt']['effects']))
    def test_time_semigroup_zero_and_platform_boundary(self):
        law=dict(half_life_hours=3,target=.2,growth_per_hour=.01)
        self.assertEqual(advance(.7,law,0),.7)
        self.assertAlmostEqual(advance(.7,law,5),advance(advance(.7,law,2),law,3),places=14)
        self.assertLess(advance(.7,law,1,True),.7)
    def test_missing_protocol_does_not_fallback_legacy(self):
        with self.assertRaises(ValueError):asyncio.run(self.engine.update_from_exchange('conversation','u','a'))
        self.assertEqual(self.legacy.read_bytes(),b'not touched')
    def test_intimacy_no_hidden_attenuation_or_ceiling(self):
        p=config();p['events']['intimacy']={k:dict(delta=.4,satisfaction=0,floor=0,plateau_hours=0) for k in ('libido','possessiveness','cling')}
        store=PersonaB1Store(self.root/'intimacy.db',legacy_path=self.legacy);seed={k:.6 for k in FIELDS};store.initialize('guyan',seed,NOW,p)
        adapter=PersonaB1Adapter(store,'guyan',{});e=event(turn(),'intimacy',labels=['desire','possessive','clingy'])
        state=adapter.settle(e,turn());projection=safe_persona_projection(state)
        for key in ('libido','possessiveness','cling'):
            self.assertEqual(state['affect'][key],1);self.assertEqual(projection['drive'][key],1)
        self.assertEqual(projection['affect']['labels'],['desire','possessive','clingy'])
    def test_other_read_routes_do_not_open_legacy(self):
        self.run_event()
        self.engine.get_self_model_state('conversation')
        p=self.engine.get_dashboard_payload('conversation')
        self.assertEqual(p['source'],'b1_candidate');self.assertEqual(p['events'],[])
        with self.assertRaises(ValueError):self.engine._connect()
        self.assertEqual(self.legacy.read_bytes(),b'not touched')
    def test_explicit_clock_restart_and_label_age(self):
        first=self.run_event();stamp=first['affect']['labels_updated_at']
        advanced=self.engine.advance_b1_clock('tick1',NOW+timedelta(hours=1),config()['revision'])
        self.assertLess(advanced['affect']['cling'],first['affect']['cling'])
        self.assertEqual(advanced['affect']['labels_updated_at'],stamp)
        again=self.engine.advance_b1_clock('tick1',NOW+timedelta(hours=1),config()['revision'])
        self.assertTrue(again['b1_receipt']['replayed'])
        state=self.store.read('guyan')
        self.engine.get_current_state('conversation');self.assertEqual(state,self.store.read('guyan'))
    def test_clock_partition_does_not_change_dynamics(self):
        self.run_event();self.engine.advance_b1_clock('one',NOW+timedelta(minutes=30),config()['revision'])
        self.engine.advance_b1_clock('two',NOW+timedelta(hours=2),config()['revision'])
        whole=PersonaB1Store(self.root/'whole.db',legacy_path=self.legacy);whole.initialize('guyan',self.initial,NOW,config())
        a=PersonaB1Adapter(whole,'guyan',{});a.settle(event(turn()),turn());a.advance_clock('whole',NOW+timedelta(hours=2),config()['revision'])
        for k in FIELDS:self.assertAlmostEqual(self.store.read('guyan')['values'][k],whole.read('guyan')['values'][k],places=14)
    def test_alias_future_schema_and_incomplete_refused(self):
        with self.assertRaises(ValueError):PersonaB1Store(self.legacy,legacy_path=self.legacy)
        with sqlite3.connect(self.store.path) as db:db.execute('DROP TABLE b1_events')
        with self.assertRaises(ValueError):PersonaB1Store(self.store.path,legacy_path=self.legacy)
    def test_no_private_labels_or_chat_success_claims(self):
        e=event(turn());e['labels']=['PRIVATE SENTENCE']
        with self.assertRaises(ValueError):self.run_event(e=e)
        e=event(turn(),'task_progress')
        with self.assertRaises(ValueError):self.run_event(e=e)

if __name__=='__main__':unittest.main()
