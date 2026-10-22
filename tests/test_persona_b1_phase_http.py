import unittest
from datetime import timedelta
import test_persona_b1_http as fixtures
from test_persona_b1 import NOW
from test_persona_b1_layers import layers_policy
class PhaseHTTP(unittest.IsolatedAsyncioTestCase):
 asyncSetUp=fixtures.HTTP.asyncSetUp
 def payload(self,**extra):return {'scope':'guyan','phase_id':'internal-rest-1','occurred_at':NOW.isoformat(),'policy_revision':layers_policy()['revision'],'phase':'sleep','source_kind':'internal_owner_request',**extra}
 async def phase(self,p,auth=True):return await self.client.post('/internal/persona/b1/phase',json=p,headers={'Authorization':'Bearer fixture-only'} if auth else {})
 async def test_same_time_phase_preserves_values_and_not_physical_claim(self):
  before=self.store.read('guyan')['values'];r=await self.phase(self.payload());self.assertEqual(r.status_code,200,r.text)
  self.assertFalse(r.json()['physical_sleep_claim']);self.assertTrue(self.store.read('guyan')['sleeping']);self.assertEqual(self.store.read('guyan')['values'],before)
 async def test_replay_after_later_clock_and_changed_source_conflict(self):
  p=self.payload();first=await self.phase(p);self.engine.b1_candidate.advance_clock('later',NOW+timedelta(minutes=1),p['policy_revision'])
  replay=await self.phase(p);self.assertEqual(replay.status_code,200,replay.text);self.assertTrue(replay.json()['b1_receipt']['replayed']);self.assertEqual(first.json()['b1_receipt']['receipt'],replay.json()['b1_receipt']['receipt'])
  bad=await self.phase({**p,'source_kind':'internal_rest_plan'});self.assertEqual(bad.status_code,409)
 async def test_future_rejected_then_clock_and_phase_at_checkpoint(self):
  p=self.payload(occurred_at=(NOW+timedelta(minutes=1)).isoformat());r=await self.phase(p);self.assertEqual(r.status_code,409);self.assertEqual(self.store.read('guyan')['revision'],0)
  self.engine.b1_candidate.advance_clock('first',p['occurred_at'],p['policy_revision']);r=await self.phase(p);self.assertEqual(r.status_code,200,r.text)
 async def test_auth_scope_source_kind_policy_default_fail_closed(self):
  self.assertEqual((await self.phase(self.payload(),False)).status_code,401)
  for change in ({'scope':'other'},{'source_kind':'quiet_hours'},{'phase':'real_sleep'}):self.assertEqual((await self.phase(self.payload(**change))).status_code,400)
  self.assertEqual((await self.phase(self.payload(policy_revision='wrong'))).status_code,409)
  self.assertEqual(self.store.read('guyan')['revision'],0);self.engine.b1_candidate=None
  self.assertEqual((await self.phase(self.payload())).status_code,503)
