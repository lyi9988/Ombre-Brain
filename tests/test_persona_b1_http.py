"""Real Starlette routing and HTTP serialization, fake internal credential only.
Full production server module is not booted (its unrelated services need config).
"""
import hmac,tempfile,unittest
from pathlib import Path
import httpx
from starlette.applications import Starlette
from starlette.routing import Route
from persona_b1 import PersonaB1Store,PersonaB1Adapter,FIELDS
from persona_engine import PersonaStateEngine
from persona_b1_http import register_candidate_route
from test_persona_b1 import NOW,turn,event
from test_persona_b1_layers import layers_policy
class RouteRegistry:
    def __init__(self):self.routes=[]
    def custom_route(self,path,methods):
        def add(fn):self.routes.append(Route(path,fn,methods=methods));return fn
        return add
class HTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        self.store=PersonaB1Store(root/'b1.db',legacy_path=root/'old.db');self.store.initialize('guyan',{k:.5 for k in FIELDS},NOW,layers_policy())
        adapter=PersonaB1Adapter(self.store,'guyan',{})
        self.engine=PersonaStateEngine({'buckets_dir':str(root),'persona':{'profile_id':'guyan','mode':'disabled'}},str(root/'old.db'),b1_candidate=adapter)
        registry=RouteRegistry()
        # Same existing server credential seam; test-only in-memory value.
        def auth(req):return hmac.compare_digest(req.headers.get('authorization',''),'Bearer fixture-only')
        register_candidate_route(registry,lambda:self.engine,auth)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=registry.routes)),base_url='http://candidate.test')
        self.addAsyncCleanup(self.client.aclose)
    async def post(self,data,headers=None):return await self.client.post('/internal/persona/b1/candidate',json=data,headers=headers or {'Authorization':'Bearer fixture-only'})
    async def test_http_authenticated_source_bound_replay(self):
        payload={'envelope':event(turn()),'turn':turn()};a=await self.post(payload);b=await self.post(payload)
        self.assertEqual(a.status_code,200);self.assertEqual(b.status_code,200);self.assertTrue(b.json()['replayed']);self.assertEqual(self.store.read('guyan')['revision'],1)
        self.assertIn('possessiveness',a.json()['projection']['drive']);self.assertNotIn('合成',a.text)
    async def test_missing_or_wrong_auth_no_write(self):
        for headers in ({'Authorization':'Bearer wrong'},{'Cookie':'ombre_session=owner-cookie'}):
            r=await self.post({'envelope':event(turn()),'turn':turn()},headers);self.assertEqual(r.status_code,401)
        self.assertEqual(self.store.read('guyan')['revision'],0)
    async def test_default_not_configured_cannot_fallback(self):
        self.engine.b1_candidate=None;r=await self.post({'envelope':event(turn()),'turn':turn()})
        self.assertEqual(r.status_code,503);self.assertEqual(self.store.read('guyan')['revision'],0)
    async def test_changed_source_conflict_no_private_error(self):
        e=event(turn());e['decision']['source_hash']='0'*64;r=await self.post({'envelope':e,'turn':turn()})
        self.assertEqual(r.status_code,409);self.assertNotIn('合成',r.text);self.assertEqual(self.store.read('guyan')['revision'],0)
    async def test_bounded_body_and_schema(self):
        r=await self.client.post('/internal/persona/b1/candidate',content=b'x'*131073,headers={'Authorization':'Bearer fixture-only'})
        self.assertEqual(r.status_code,413)
        r=await self.post({'private':'PRIVATE_SENTINEL'});self.assertEqual(r.status_code,400);self.assertNotIn('PRIVATE_SENTINEL',r.text)
    async def test_no_get_or_public_tool_write(self):
        r=await self.client.get('/internal/persona/b1/candidate');self.assertEqual(r.status_code,405)
        self.assertEqual(self.store.read('guyan')['revision'],0)
    async def test_clock_actual_authenticated_route_no_fake_turn(self):
        from datetime import timedelta
        payload={'scope':'guyan','occurred_at':(NOW+timedelta(hours=1)).isoformat(),'policy_revision':layers_policy()['revision'],'growth_frames':[]}
        headers={'Authorization':'Bearer fixture-only'}
        r=await self.client.post('/internal/persona/b1/clock',json=payload,headers=headers)
        self.assertEqual(r.status_code,200,r.text);first=r.json()['b1_receipt'];self.assertFalse(first['replayed'])
        again=await self.client.post('/internal/persona/b1/clock',json=payload,headers=headers);self.assertTrue(again.json()['b1_receipt']['replayed'])
        self.assertEqual(first['receipt']['after_revision'],again.json()['b1_receipt']['receipt']['after_revision'])
        bad=await self.client.post('/internal/persona/b1/clock',json=payload);self.assertEqual(bad.status_code,401)
    async def test_clock_scope_policy_no_anonymous_or_legacy_fallback(self):
        payload={'scope':'wrong','occurred_at':NOW.isoformat(),'policy_revision':layers_policy()['revision'],'growth_frames':[]};headers={'Authorization':'Bearer fixture-only'}
        r=await self.client.post('/internal/persona/b1/clock',json=payload,headers=headers);self.assertEqual(r.status_code,400)
        self.engine.b1_candidate=None
        r=await self.client.post('/internal/persona/b1/clock',json=payload,headers=headers);self.assertEqual(r.status_code,503)
    async def test_result_actual_route_auth_idempotent_no_chat_proof(self):
        result={'scope':'guyan','operation_id':'confirmed-work','status':'succeeded','proof':'her_ok','revision':1,'result_hash':'a'*64,'occurred_at':NOW.isoformat()}
        payload={'result':result,'policy_revision':layers_policy()['revision']};headers={'Authorization':'Bearer fixture-only'}
        r=await self.client.post('/internal/persona/b1/result',json=payload,headers=headers);self.assertEqual(r.status_code,200,r.text)
        again=await self.client.post('/internal/persona/b1/result',json=payload,headers=headers);self.assertTrue(again.json()['b1_receipt']['replayed'])
        noauth=await self.client.post('/internal/persona/b1/result',json=payload);self.assertEqual(noauth.status_code,401)
        result['status']='failed';bad=await self.client.post('/internal/persona/b1/result',json=payload,headers=headers);self.assertEqual(bad.status_code,400)
    async def test_numeric_history_auth_scope_and_no_write(self):
        from datetime import timedelta
        headers={'Authorization':'Bearer fixture-only'}
        await self.post({'envelope':event(turn()),'turn':turn()})
        before=self.store.read('guyan')
        r=await self.client.get('/internal/persona/b1/numeric',params={'scope':'guyan','at':(NOW-timedelta(seconds=1)).isoformat()},headers=headers)
        self.assertEqual(r.status_code,200);self.assertIsNone(r.json()['snapshot'])
        r=await self.client.get('/internal/persona/b1/numeric',params={'scope':'guyan','at':NOW.isoformat()},headers=headers)
        self.assertEqual(r.status_code,200);self.assertLessEqual(__import__('datetime').datetime.fromisoformat(r.json()['snapshot']['as_of']),NOW);self.assertNotIn('合成',r.text)
        self.assertEqual((await self.client.get('/internal/persona/b1/numeric',params={'scope':'guyan'})).status_code,401)
        self.assertEqual((await self.client.get('/internal/persona/b1/numeric',params={'scope':'other'},headers=headers)).status_code,400)
        self.assertEqual(self.store.read('guyan'),before)
