import unittest
from datetime import timedelta
import test_persona_b1_layers as fixtures
from test_persona_b1 import NOW,turn,event
from persona_b1 import FIELDS
from persona_projection import safe_persona_projection

def social_policy():
    p=fixtures.layers_policy();p['presentation']={};p['mood_policy']={};p['couplings']={}
    near=('owner','family','known','stranger')
    p['empathy_policy']={'base':.2,'weights':{k:1 for k in near},'caps':{k:None for k in near},'daily_caps':{k:None for k in near},'dedupe_hours':6,'helped_relief':.6,'offset_minutes':0,'impulses':{k:{'valence':0,'arousal':0} for k in ('心疼','不平','想帮忙','替人高兴')}}
    p['light_policy']={'minutes':{k:60 for k in ('心动','害羞','得意','骄傲')},'light_needed':3,'window_minutes':30,'repeat_factor':1,'min_minutes':15,'offset_minutes':0,'proof_confidence_min':.3}
    return p

def empathy(t,sub='心疼',near='known',subject='person:1',incident='incident:1',when=NOW):
    e=event(t,'empathy',when);e['empathy_details']=[{'type':'empathy','sub':sub,'subject_ref':subject,'incident_ref':incident,'closeness':near,'identity_status':'resolved' if near!='unknown' else 'unresolved','evidence_hash':e['decision']['accepted'][0]['evidence_hash']}];return e

def light(t,kind='affection',sub='心动',strength='light',when=NOW):
    e=event(t,kind,when);e['light_details']=[{'type':kind,'sub':sub,'strength':strength,'evidence_hash':e['decision']['accepted'][0]['evidence_hash']}];return e
class Social(unittest.TestCase):
    setUp=fixtures.Layers.setUp
    make=fixtures.Layers.make
    def test_same_incident_different_turn_dedupes_but_new_incident_not(self):
        s,a=self.make(p=social_policy());t=turn();a.settle(empathy(t),t);v=s.read('guyan')['values']['grieve']
        t=turn('second');a.settle(empathy(t),t);self.assertEqual(s.read('guyan')['values']['grieve'],v)
        t=turn('third');a.settle(empathy(t,incident='incident:2'),t);self.assertGreater(s.read('guyan')['values']['grieve'],v)
    def test_unknown_person_does_not_become_owner(self):
        s,a=self.make(p=social_policy());t=turn();e=empathy(t,near='unknown');r=a.settle(e,t)
        self.assertEqual(s.read('guyan')['values']['grieve'],.5);self.assertFalse(s.read('guyan')['empathy_loads'])
        self.assertIn('unknown_closeness',[x.get('status') for x in r['b1_receipt']['receipt']['effects']])
    def test_helped_only_corresponding_share_not_relationship_or_other_person(self):
        raw={'scope':'guyan','operation_id':'done','status':'succeeded','proof':'helped_ack','revision':1,'result_hash':'a'*64,'occurred_at':NOW.isoformat(),'helped':{'subject_ref':'person:1','incident_ref':'incident:1','sub':'心疼'}}
        s,a=self.make(p=social_policy(),result_resolver=lambda _:raw)
        for i,person in enumerate(('person:1','person:2')):
            t=turn(str(i));a.settle(empathy(t,subject=person),t)
        before=s.read('guyan');a.record_operation('done',social_policy()['revision']);after=s.read('guyan')
        self.assertAlmostEqual(before['values']['grieve']-after['values']['grieve'],.12)
        self.assertAlmostEqual(sum(x['value'] for x in after['empathy_loads'].values() if x['subject_ref']=='person:2'),.2)
        self.assertAlmostEqual(after['values']['grieve']-sum(x['value'] for x in after['empathy_loads'].values()),.5)
        a.record_operation('done',social_policy()['revision']);self.assertEqual(after,s.read('guyan'))
    def test_owner_empathy_not_external_share(self):
        s,a=self.make(p=social_policy());t=turn();a.settle(empathy(t,near='owner',subject='owner'),t)
        self.assertAlmostEqual(s.read('guyan')['values']['grieve'],.7);self.assertFalse(s.read('guyan')['empathy_loads'])
    def test_share_decays_same_time_as_total(self):
        p=social_policy();p['laws']['grieve']={'half_life_hours':2,'target':0,'growth_per_hour':0};s,a=self.make(p=p);t=turn();a.settle(empathy(t),t);a.advance_clock('later',NOW+timedelta(hours=2),p['revision'])
        self.assertAlmostEqual(sum(x['value'] for x in s.read('guyan')['empathy_loads'].values()),.1)
    def test_light_accumulates_once_not_retry_and_expires_by_time(self):
        s,a=self.make(p=social_policy());t=turn();e=light(t);a.settle(e,t);a.settle(e,t);self.assertIsNone(a.snapshot()['affect']['lit'])
        for i in (2,3):
            t=turn(str(i));a.settle(light(t,when=NOW+timedelta(minutes=i)),t)
        self.assertEqual(a.snapshot()['affect']['lit']['word'],'心动')
        a.advance_clock('end',NOW+timedelta(hours=2),social_policy()['revision']);self.assertIsNone(a.snapshot()['affect']['lit'])
        self.assertEqual(len(s.read('guyan')['light_marks']),1)
    def test_heavy_intimacy_not_forced_three_taps(self):
        s,a=self.make(p=social_policy());t=turn();a.settle(light(t,'intimacy','心动','heavy'),t)
        self.assertEqual(a.snapshot()['affect']['lit']['word'],'心动')
    def test_mark_only_light_does_not_settle_drive(self):
        s,a=self.make(p=social_policy());t=turn();e=light(t,'intimacy','心动','heavy');e['interaction_modes']={'intimacy':'mark_only'};before=s.read('guyan')['values'];a.settle(e,t)
        self.assertEqual(s.read('guyan')['values'],before);self.assertEqual(a.snapshot()['affect']['lit']['word'],'心动')
    def test_pride_requires_actual_owner_subject_not_third_party(self):
        s,a=self.make(p=social_policy());t=turn();e=empathy(t,'替人高兴');e['light_details']=[{'type':'empathy','sub':'骄傲','strength':'heavy','evidence_hash':e['decision']['accepted'][0]['evidence_hash']}]
        with self.assertRaises(ValueError):a.settle(e,t)
        e['empathy_details'][0].update(closeness='owner',subject_ref='owner');a.settle(e,t)
        self.assertEqual(a.snapshot()['affect']['lit']['word'],'骄傲')
    def test_desired_help_routes_to_aiz_not_new_share_axis(self):
        s,a=self.make(p=social_policy());t=turn();r=a.settle(empathy(t,'想帮忙'),t)
        self.assertNotIn('expression',s.read('guyan')['values']);self.assertIn('aiz_owned',[x.get('status') for x in r['b1_receipt']['receipt']['effects']])
    def test_reported_third_party_empathy_not_conflict(self):
        s,a=self.make(p=social_policy());t=turn();e=empathy(t);e['decision']['accepted'][0].update(target='third_party',context='reported');a.settle(e,t)
        self.assertIsNone(s.read('guyan')['conflict_episode']);self.assertGreater(s.read('guyan')['values']['grieve'],.5)
    def test_presence_is_real_completed_turn_not_read_or_disconnect_story(self):
        p=social_policy();p['presence_policy']={'possessiveness_min':.2,'cling_min':.8,'absence_hours':1};s,a=self.make(p=p)
        self.assertIsNone(a.snapshot()['affect']['last_conversation_at']);self.assertIsNone(a.snapshot()['affect']['connection_phase'])
        t=turn();a.settle(event(t,'companionship'),t);a.advance_clock('later',NOW+timedelta(hours=2),p['revision'])
        self.assertEqual(a.snapshot()['affect']['connection_phase'],'想念');before=s.read('guyan');a.snapshot();self.assertEqual(before,s.read('guyan'))
    def test_correction_dual_spans_bound_even_mark_only_and_retry(self):
        from b1_protocol import digest
        p=social_policy();p['correction_confidence_delta']=-.04;s,a=self.make(p=p);t=turn();e=event(t,'affection');e['interaction_modes']={'affection':'mark_only'}
        user=t['user_event']['content'];assistant=t['assistant_event']['content'];e['correction_details']=[{'type':'correction','actor':'owner','target':'assistant','context':'direct','assertion':'occurred','certainty':'supported','user_start':0,'user_end':len(user),'assistant_start':0,'assistant_end':len(assistant),'evidence_hash':digest({'user':user,'assistant':assistant})}]
        a.settle(e,t);self.assertAlmostEqual(s.read('guyan')['values']['slow_confidence'],.46);before=s.read('guyan');a.settle(e,t);self.assertEqual(before,s.read('guyan'))
        bad=event(turn('bad'),'affection');bad['correction_details']=e['correction_details'];bad['correction_details'][0]['evidence_hash']='a'*64
        with self.assertRaises(ValueError):a.settle(bad,turn('bad'))
    def test_mixed_modes_only_bound_apply_feelings_enter_state(self):
        p=social_policy();s,a=self.make(p=p);t=turn();e=event(t,'affection',labels=['tender','angry']);row=dict(e['decision']['accepted'][0]);row['type']='conflict';e['decision']['accepted'].append(row);e['interaction_modes']={'conflict':'mark_only','affection':'apply'}
        e['feeling_sources']=[{'label':'tender','type':'affection','evidence_hash':row['evidence_hash']},{'label':'angry','type':'conflict','evidence_hash':row['evidence_hash']}]
        a.settle(e,t);self.assertEqual(a.snapshot()['affect']['labels'],['tender'])
    def test_private_subjects_and_evidence_never_enter_projection(self):
        s,a=self.make(p=social_policy());t=turn();e=empathy(t,subject='PRIVATE-SUBJECT-SENTINEL',incident='PRIVATE-INCIDENT-SENTINEL');a.settle(e,t)
        import json
        raw=json.dumps(safe_persona_projection(a.snapshot()),ensure_ascii=False)
        self.assertNotIn('PRIVATE-',raw);self.assertNotIn('evidence_hash',raw);self.assertIn('empathy_load',raw)
