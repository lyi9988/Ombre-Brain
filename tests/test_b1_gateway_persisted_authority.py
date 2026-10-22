"""Actual Gateway + Persona engine + temporary persisted SQLite authority.

Requires the sibling Brain v4 authority changes on this same Ombre source tree.
No production DB, network or model request is used.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from gateway import GatewayService
from persona_engine import PersonaStateEngine
from persona_b1 import PersonaB1Store, PersonaB1Adapter, FIELDS
from persona_b1_authority import activate, AuthorityError
from test_persona_b1 import NOW
from test_persona_b1_layers import layers_policy


class GatewayPersistedAuthority(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name); self.old=root/'old.db'
        self.cfg={'buckets_dir':str(root),'persona':{'profile_id':'guyan','mode':'disabled'},
                  'gateway':{'b1_persona_writer':{'mode':'candidate_authority','scopes':[{'profile_id':'guyan','scope':'guyan','epoch':1}]}}}
        self.legacy=PersonaStateEngine(self.cfg,str(self.old)); self.legacy.get_current_state('chat')
        store=PersonaB1Store(root/'candidate.db',legacy_path=self.old)
        store.initialize('guyan',{k:.5 for k in FIELDS},NOW,layers_policy())
        self.adapter=PersonaB1Adapter(store,'guyan',{})
        self.engine=PersonaStateEngine(self.cfg,str(self.old),b1_candidate=self.adapter)
        self.service=GatewayService.__new__(GatewayService)
        self.service.config=self.cfg; self.service.persona_engine=self.engine

    def activate(self):
        return activate(self.old,self.adapter,profile_id='guyan',scope='guyan',epoch=1)

    async def test_missing_persisted_authority_rejects_not_config_grant(self):
        with self.assertRaises(AuthorityError):
            self.service._b1_persona_authority_active()

    async def test_real_fence_suppresses_all_target_profile_aliases(self):
        self.activate()
        # Real engine would reject legacy exchange without bound B1 turn if
        # called. All aliases must return before that old evaluation path.
        for session in ('chat','another-channel','',None):
            await self.service._update_persona_after_assistant_message(session,'hello',{'content':'hi'},[])
        self.assertEqual(self.adapter.store.read('guyan')['revision'],0)

    async def test_actual_candidate_pre_reply_read_is_readonly(self):
        self.activate(); before=self.adapter.store.read('guyan')
        self.assertTrue(self.service._b1_persona_authority_active())
        current=await self.engine.build_pre_reply_guidance('another-channel','hello')
        self.assertEqual(current['source'],'b1_candidate')
        self.assertEqual(self.adapter.store.read('guyan'),before)
        with sqlite3.connect(self.old) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE persona_session_state SET valence=.9 WHERE profile_id='guyan'")

    async def test_wrong_epoch_and_removed_trigger_fail_before_postreply(self):
        self.activate()
        self.cfg['gateway']['b1_persona_writer']['scopes'][0]['epoch']=2
        with self.assertRaises(AuthorityError):
            await self.service._update_persona_after_assistant_message('chat','hello',{'content':'hi'},[])
        self.cfg['gateway']['b1_persona_writer']['scopes'][0]['epoch']=1
        with sqlite3.connect(self.old) as db: db.execute('DROP TRIGGER b1_fence_persona_session_state_update')
        with self.assertRaises(AuthorityError):
            self.service._get_persona_state_for_context_mode('chat')

    async def test_other_real_profile_keeps_legacy_postreply(self):
        other_cfg={'buckets_dir':self.cfg['buckets_dir'],'persona':{'profile_id':'other','mode':'disabled'}}
        other=PersonaStateEngine(other_cfg,str(self.old)); other.get_current_state('chat')
        self.activate(); self.service.persona_engine=other
        self.service._coerce_message_text=lambda x:x
        self.service._summarize_assistant_tool_calls=lambda x:''
        self.service._recent_persona_conversation_turns=lambda *x:[]
        other.update_from_exchange=AsyncMock(wraps=other.update_from_exchange)
        await self.service._update_persona_after_assistant_message('chat','hello',{'content':'hi'},[])
        other.update_from_exchange.assert_awaited_once()

    async def test_candidate_fence_is_persistent_across_new_engine(self):
        self.activate()
        restarted=PersonaStateEngine(self.cfg,str(self.old),b1_candidate=self.adapter)
        self.service.persona_engine=restarted
        self.assertTrue(self.service._b1_persona_authority_active())
        self.assertEqual(restarted.verify_b1_authority('guyan',1)['epoch'],1)

if __name__=='__main__': unittest.main()
