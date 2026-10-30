"""Reviewed tendency ingress preserves the original B1 writer and boundaries."""
import unittest

import test_persona_b1_http as fixtures
from test_persona_b1 import NOW


class TendencyHTTP(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.HTTP.asyncSetUp

    def payload(self, **changes):
        return {'scope': 'guyan', 'event_id': 'self-review:real-run',
            'changes': {'exploration_affinity': 0.0},
            'sources': {'exploration_affinity': {'unit': '0..1', 'status': 'known',
                                               'source_ref': 'self-review:real-run:exploration_affinity'}},
            'expected_revision': 0, 'occurred_at': NOW.isoformat(), **changes}

    async def post(self, payload, authenticated=True):
        return await self.client.post('/internal/persona/b1/tendencies', json=payload,
            headers={'Authorization': 'Bearer fixture-only'} if authenticated else {})

    async def test_exact_payload_replays_after_another_checkpoint_and_keeps_explicit_zero(self):
        before = self.store.read('guyan')
        missing = await self.client.get('/internal/persona/b1/tendencies/receipt',
            params={'scope': 'guyan', 'event_id': self.payload()['event_id']},
            headers={'Authorization': 'Bearer fixture-only'})
        self.assertEqual(missing.json()['status'], 'not_found')
        first = await self.post(self.payload())
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(self.store.read('guyan')['tendencies']['exploration_affinity'], 0)
        self.assertEqual(self.store.read('guyan')['values'], before['values'])
        replay = await self.post(self.payload())
        self.assertTrue(replay.json()['tendency_receipt']['replayed'])
        self.assertEqual(first.json()['tendency_receipt']['receipt'], replay.json()['tendency_receipt']['receipt'])
        self.assertEqual(self.store.read('guyan')['revision'], 1)
        original = await self.client.get('/internal/persona/b1/tendencies/receipt',
            params={'scope': 'guyan', 'event_id': self.payload()['event_id']},
            headers={'Authorization': 'Bearer fixture-only'})
        self.assertEqual(original.json()['receipt'], first.json()['tendency_receipt']['receipt'])
        anonymous = await self.client.get('/internal/persona/b1/tendencies/receipt',
            params={'scope': 'guyan', 'event_id': self.payload()['event_id']})
        self.assertEqual(anonymous.status_code, 401)
        changed = await self.post(self.payload(changes={'exploration_affinity': .8}))
        self.assertEqual(changed.status_code, 409)

    async def test_auth_scope_schema_revision_and_off_adapter_fail_without_writes(self):
        self.assertEqual((await self.post(self.payload(), False)).status_code, 401)
        for change, status in (({'scope': 'other'}, 400), ({'expected_revision': 9}, 409),
                               ({'occurred_at': '2026-10-29T12:00:00+00:00'}, 409),
                               ({'changes': {'exploration_affinity': True}}, 400)):
            self.assertEqual((await self.post(self.payload(**change))).status_code, status)
        self.assertEqual(self.store.read('guyan')['revision'], 0)
        self.engine.b1_candidate = None
        self.assertEqual((await self.post(self.payload())).status_code, 503)
