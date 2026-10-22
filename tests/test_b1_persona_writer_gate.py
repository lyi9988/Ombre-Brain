"""Actual Gateway method, fake engine only; no external/model requests."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from gateway import GatewayService
from b1_persona_writer_gate import parse_writer_gate, B1WriterConfigurationError

BINDING = {'profile_id': 'guyan', 'scope': 'guyan', 'epoch': 1}

class WriterGateTests(unittest.IsolatedAsyncioTestCase):
    def service(self, gate=None, profile='guyan'):
        service = GatewayService.__new__(GatewayService)
        service.config = {'gateway': {}}
        if gate is not None:
            service.config['gateway']['b1_persona_writer'] = gate
        service.persona_engine = SimpleNamespace(enabled=True, profile_id=profile,
                                                update_from_exchange=AsyncMock())
        service._coerce_message_text = lambda content: content
        service._summarize_assistant_tool_calls = lambda message: ''
        service._recent_persona_conversation_turns = lambda *args: []
        return service

    async def call(self, service, session='main'):
        await service._update_persona_after_assistant_message(session, 'hello', {'content': 'hi'}, [])

    async def test_default_preserves_original_writer(self):
        service = self.service()
        await self.call(service)
        service.persona_engine.update_from_exchange.assert_awaited_once()

    async def test_off_and_shadow_preserve_all_sessions(self):
        for mode in ('off', 'shadow'):
            for session, profile in (('main','guyan'), ('another','guyan'), ('','guyan'), ('main','other')):
                with self.subTest(mode=mode, session=session, profile=profile):
                    service = self.service({'mode': mode, 'scopes': [BINDING]}, profile)
                    await self.call(service, session)
                    service.persona_engine.update_from_exchange.assert_awaited_once()

    async def test_authority_rejected_before_old_writer_all_scopes(self):
        for session in ('main', 'another', ''):
            service = self.service({'mode': 'candidate_authority', 'scopes': [BINDING]})
            with self.assertRaisesRegex(B1WriterConfigurationError, 'coverage_incomplete'):
                await self.call(service, session)
            service.persona_engine.update_from_exchange.assert_not_awaited()

    async def test_authority_binding_does_not_disable_other_profile(self):
        service = self.service({'mode': 'candidate_authority', 'scopes': [BINDING]}, profile='other')
        await self.call(service, 'main')
        service.persona_engine.update_from_exchange.assert_awaited_once()

    async def test_authority_rejected_at_start_before_store_construction(self):
        # Deliberately no buckets_dir or other construction dependencies.
        with self.assertRaisesRegex(B1WriterConfigurationError, 'candidate_engine_required'):
            GatewayService({'persona': {'profile_id':'guyan'}, 'gateway': {'b1_persona_writer': {'mode': 'candidate_authority', 'scopes': [BINDING]}}})

    async def test_runtime_invalid_config_cannot_silently_fallback(self):
        service = self.service({'mode': 'unknown'})
        with self.assertRaises(B1WriterConfigurationError):
            await self.call(service)
        service.persona_engine.update_from_exchange.assert_not_awaited()

    async def test_original_tool_call_skip_preserved(self):
        service = self.service({'mode': 'shadow', 'scopes': [BINDING]})
        await service._update_persona_after_assistant_message('main', 'hello', {'content':'hi', 'tool_calls':[{}]}, [])
        service.persona_engine.update_from_exchange.assert_not_awaited()

    async def test_strict_configuration_no_claimed_coverage_bypass(self):
        invalid = [[], {'mode':True}, {'mode':'shadow', 'scopes':'all'},
                   {'mode':'shadow', 'scopes':[dict(BINDING, profile_id=' guyan')]},
                   {'mode':'shadow', 'scopes':[BINDING,BINDING]},
                   {'mode':'candidate_authority','coverage_complete':True,'scopes':[BINDING]},
                   {'mode':'candidate_authority','scopes':[]}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(B1WriterConfigurationError):
                parse_writer_gate(config)

    async def test_hot_edit_cannot_remove_candidate_adapter(self):
        service = self.service({'mode': 'candidate_authority', 'scopes': [BINDING]})
        with self.assertRaisesRegex(ValueError, 'planned_reconfiguration'):
            service._apply_persona_config({'enabled': False, 'api_key':'synthetic'})
        self.assertNotIn('persona', service.config)

    async def test_config_and_injected_profile_must_agree(self):
        gate=parse_writer_gate({'mode':'candidate_authority','scopes':[BINDING]})
        with self.assertRaisesRegex(B1WriterConfigurationError,'profile_mismatch'):
            gate.verify_engine(SimpleNamespace(profile_id='guyan'),'other')

    async def test_config_snapshot_immutable(self):
        config={'mode':'shadow','scopes':[dict(BINDING)]}
        gate=parse_writer_gate(config)
        config['scopes'][0]['scope']='changed'
        self.assertEqual(gate.scopes, (('guyan','guyan',1),))

if __name__ == '__main__': unittest.main()
