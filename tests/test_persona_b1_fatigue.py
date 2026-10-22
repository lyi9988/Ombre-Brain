import unittest
from datetime import timedelta
from copy import deepcopy
import test_persona_b1_layers as fixtures
from test_persona_b1 import NOW
from persona_b1 import PersonaB1Store,Conflict
from b1_pressure_protocol import build_pressure_frame

def config():
 p=fixtures.layers_policy();p['presentation']={};p['mood_policy']={};p['initial_sleeping']=False
 pairs={'possess':('brain','possessiveness'),'monitor':('brain','protective_drive'),'share':('aiz','expression'),'libido':('brain','libido'),'curiosity':('aiz','curiosity'),'boredom':('aiz','boredom'),'duty':('aiz','duty'),'reflection':('aiz','reflection'),'grieve':('brain','grieve'),'anger':('brain','anger'),'favored':('brain','favored')}
 p['pressure_policy']={'revision':'test-pressure','grid_seconds':3600,'anchor_utc':NOW.isoformat(),'hosts':{'aiz':{'scope':'guyan','policy_revision':'a1'},'brain':{'scope':'guyan','policy_revision':p['revision']}},'bindings':{k:{'host':h,'key':f,'unit':'0..100' if h=='aiz' else '0..1'} for k,(h,f) in pairs.items()}}
 p['fatigue_policy']={'initial':.2,'maximum':1,'build_per_hour':.1,'recover_per_hour':.2,'pressure_threshold':.7}
 return p
class Fatigue(unittest.TestCase):
 setUp=fixtures.Layers.setUp
 make=fixtures.Layers.make
 def loader(self,p,calls,unknown=False):
  def load(at):
   calls.append(at);hosts=[]
   for host in ('aiz','brain'):
    h=p['pressure_policy']['hosts'][host];vals={b['key']:(80 if host=='aiz' else .8) for b in p['pressure_policy']['bindings'].values() if b['host']==host}
    if unknown and host=='aiz':vals['expression']=None
    hosts.append({**h,'revision':1,'as_of':at.isoformat(),'values':vals})
   return build_pressure_frame(p['pressure_policy'],at,*hosts)
  return load
 def test_whole_split_retry_and_restart_same_pressure(self):
  p=config();calls=[];s,a=self.make('whole',p,pressure_frame_loader=self.loader(p,calls));t,b=self.make('split',p,pressure_frame_loader=self.loader(p,[]))
  a.advance_clock('whole',NOW+timedelta(hours=1),p['revision'])
  b.advance_clock('half',NOW+timedelta(minutes=30),p['revision']);b.advance_clock('end',NOW+timedelta(hours=1),p['revision'])
  self.assertAlmostEqual(s.read('guyan')['fatigue']['value'],.3);self.assertAlmostEqual(s.read('guyan')['fatigue']['value'],t.read('guyan')['fatigue']['value'],places=14)
  a.advance_clock('whole',NOW+timedelta(hours=1),p['revision']);self.assertEqual(len(calls),1)
  reopened=PersonaB1Store(s.path,legacy_path=self.old);self.assertEqual(reopened.read('guyan'),s.read('guyan'))
 def test_unknown_not_energy_or_zero_and_sleep_requires_no_pressure(self):
  p=config();s,a=self.make(p=p,pressure_frame_loader=self.loader(p,[],True));a.advance_clock('x',NOW+timedelta(hours=1),p['revision'])
  self.assertIsNone(s.read('guyan')['fatigue']['value']);self.assertEqual(s.read('guyan')['fatigue']['last_known'],.2)
  p['initial_sleeping']=True;t,b=self.make('sleep',p,pressure_frame_loader=self.loader(p,[],True));b.advance_clock('x',NOW+timedelta(minutes=30),p['revision']);self.assertAlmostEqual(t.read('guyan')['fatigue']['value'],.1)
 def test_changed_frame_rejected_before_state_change(self):
  p=config();load=self.loader(p,[]);s,a=self.make(p=p,pressure_frame_loader=load);f=load(NOW)
  a.advance_clock('half',NOW+timedelta(minutes=30),p['revision'],pressure_frames=[f]);before=s.read('guyan')
  bad=deepcopy(f);bad['sources']['aiz']['values']['expression']=20
  bad=build_pressure_frame(p['pressure_policy'],NOW,bad['sources']['aiz'],bad['sources']['brain'])
  with self.assertRaises(Conflict):a.advance_clock('end',NOW+timedelta(hours=1),p['revision'],pressure_frames=[bad])
  self.assertEqual(s.read('guyan'),before)
 def test_historical_numeric_does_not_use_future_or_advance(self):
  p=config();s,a=self.make(p=p,pressure_frame_loader=self.loader(p,[]));a.advance_clock('one',NOW+timedelta(hours=1),p['revision']);before=s.read('guyan')
  self.assertIsNone(s.numeric_at('guyan',NOW-timedelta(seconds=1)))
  old=s.numeric_at('guyan',NOW+timedelta(minutes=30));self.assertEqual(old['revision'],0);self.assertEqual(old['values']['fatigue'],.2)
  self.assertEqual(s.numeric_at('guyan')['revision'],1);self.assertEqual(s.read('guyan'),before)

 def test_mean_pressure_not_all_high_and_strict_threshold(self):
  from persona_b1_layers import fatigue_step,PRESSURE_KEYS
  p=config()['fatigue_policy'];v={k:.8 for k in PRESSURE_KEYS};v['anger']=0
  self.assertAlmostEqual(fatigue_step(.2,False,v,1,p)[0],.3)
  p['pressure_threshold']=.5;v={k:.5 for k in PRESSURE_KEYS}
  self.assertEqual(fatigue_step(.2,False,v,1,p),(.2,'known_pressure_below_or_equal'))
