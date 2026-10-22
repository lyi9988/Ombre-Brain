import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx

from gateway import GatewayService, create_gateway_app
from persona_engine import PersonaStateEngine
from persona_b1_runtime import prepare, activate_prepared
from test_persona_b1_layers import layers_policy


class GatewayFactoryWiring(unittest.IsolatedAsyncioTestCase):
    async def test_actual_gateway_constructor_reopens_persisted_candidate(self):
        temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp.cleanup)
        root = Path(temp.name); old = root / 'old.db'
        cfg = {'buckets_dir': str(root),
            'persona': {'profile_id': 'guyan', 'canonical_session_id': 'chat',
                        'db_path': str(old), 'mode': 'disabled'},
            'gateway': {'b1_persona_writer': {'mode': 'candidate_authority',
                'scopes': [{'profile_id': 'guyan', 'scope': 'guyan', 'epoch': 1}]}},
            'b1': {'mode': 'candidate_authority', 'legacy_db_path': str(old),
                   'candidate_db_path': str(root / 'candidate.db'),
                   'profile_id': 'guyan', 'scope': 'guyan', 'epoch': 1,
                   'legacy_session_id': 'chat',
                   'handoff_at': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                   'global_snapshot_path': str(root / 'source.json'),
                   'policy_path': str(root / 'policy.json')}}
        PersonaStateEngine(cfg, str(old)).get_current_state('chat')
        Path(cfg['b1']['policy_path']).write_text(json.dumps(layers_policy()), encoding='utf-8')
        prepare(cfg); activate_prepared(cfg)
        service = GatewayService(cfg)
        self.addAsyncCleanup(service.close)
        self.assertIsNotNone(service.persona_engine.b1_candidate)
        self.assertTrue(service._b1_persona_authority_active())
        wrong = {**cfg, 'b1': {**cfg['b1'], 'epoch': 2}}
        with self.assertRaises(Exception):
            GatewayService(wrong)

        env_names = ('OMBRE_PERSONA_API_KEY', 'OMBRE_PERSONA_MODEL', 'OMBRE_PERSONA_BASE_URL')
        old_env = {name: os.environ.get(name) for name in env_names}
        def restore_env():
            for name, value in old_env.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name] = value
        self.addCleanup(restore_env)
        old_candidate_path = service.persona_engine.b1_candidate.store.path
        updated = service._apply_persona_config({'model': 'synthetic-next',
            'base_url': 'https://fixture.invalid/v1', 'api_key': 'synthetic-key',
            'event_recording_enabled': False, 'enabled': True})
        self.assertEqual(updated, ['persona.enabled', 'persona.event_recording_enabled',
                                   'persona.model', 'persona.base_url', 'persona.api_key'])
        self.assertEqual(service.persona_engine.model, 'synthetic-next')
        self.assertEqual(service.persona_engine.api_key, 'synthetic-key')
        self.assertFalse(service.persona_engine.event_recording_enabled)
        self.assertEqual(service.persona_engine.b1_candidate.store.path, old_candidate_path)
        self.assertIs(service.persona_engine.config, service.config)
        self.assertIs(service.persona_engine.prompt_resolver.__self__, service)
        self.assertTrue(service._b1_persona_authority_active())
        retained_engine = service.persona_engine
        retained_config = dict(service.config['persona'])
        for change in ({'enabled': False}, {'mode': 'llm'}, {'profile_id': 'other'}):
            with self.assertRaisesRegex(ValueError, 'planned_reconfiguration'):
                service._apply_persona_config(change)
            self.assertIs(service.persona_engine, retained_engine)
            self.assertEqual(service.config['persona'], retained_config)

        service.gateway_token = 'synthetic-gateway'
        app = create_gateway_app(config=cfg, service=service)
        app.state.gateway_service = service
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://fixture.invalid') as client:
            response = await client.post('/api/config', json={'persona': {'enabled': False}},
                headers={'Authorization': 'Bearer synthetic-gateway'})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['error'],
                             'b1_writer_authority_requires_planned_reconfiguration')
            self.assertIs(service.persona_engine, retained_engine)
            response = await client.post('/api/config', json={'persona': {'model': 'synthetic-http'}},
                headers={'Authorization': 'Bearer synthetic-gateway'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('persona.model', response.json()['updated'])
            self.assertEqual(service.persona_engine.model, 'synthetic-http')
            self.assertTrue(service._b1_persona_authority_active())
