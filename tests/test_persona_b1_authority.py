import sqlite3,tempfile,unittest
from pathlib import Path
from persona_engine import PersonaStateEngine
from persona_b1 import PersonaB1Store,PersonaB1Adapter,FIELDS
from persona_b1_authority import activate,AuthorityError,TABLES
from test_persona_b1 import NOW
from test_persona_b1_layers import layers_policy
class Authority(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.old=self.root/'old.db'
        self.cfg={'buckets_dir':str(self.root),'persona':{'profile_id':'guyan','mode':'disabled'}}
        self.legacy=PersonaStateEngine(self.cfg,str(self.old))
        self.legacy.get_current_state('chat')
        other={'buckets_dir':str(self.root),'persona':{'profile_id':'other','mode':'disabled'}}
        self.other=PersonaStateEngine(other,str(self.old));self.other.get_current_state('chat')
        self.store=PersonaB1Store(self.root/'candidate.db',legacy_path=self.old);self.store.initialize('guyan',{k:.5 for k in FIELDS},NOW,layers_policy())
        self.adapter=PersonaB1Adapter(self.store,'guyan',{})
        self.engine=PersonaStateEngine(self.cfg,str(self.old),b1_candidate=self.adapter)
    def activate(self):return activate(self.old,self.adapter,profile_id='guyan',scope='guyan',epoch=1)
    def test_default_no_marker_no_activation(self):
        with self.assertRaises(AuthorityError):self.engine.verify_b1_authority('guyan',1)
        with sqlite3.connect(self.old) as c:self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE name='persona_b1_authority'").fetchone()[0],0)
    def test_activate_verify_replay(self):
        self.assertEqual(self.activate(),self.activate())
        self.assertEqual(self.engine.verify_b1_authority('guyan',1),{'profile_id':'guyan','scope':'guyan','epoch':1,'mode':'candidate'})
        with self.assertRaises(AuthorityError):self.engine.verify_b1_authority('guyan',2)
    def test_stale_legacy_connection_cannot_write(self):
        c=sqlite3.connect(self.old);self.addCleanup(c.close)
        c.execute("SELECT * FROM persona_session_state WHERE profile_id='guyan'").fetchall()
        self.activate()
        with self.assertRaises(sqlite3.IntegrityError):c.execute("UPDATE persona_session_state SET valence=.01 WHERE profile_id='guyan'")
        c.rollback()
        with self.assertRaises(sqlite3.IntegrityError):c.execute("DELETE FROM persona_global_state WHERE profile_id='guyan'")
    def test_all_legacy_tables_insert_update_delete_fenced(self):
        self.activate()
        with sqlite3.connect(self.old) as c:
            for table in TABLES:
                for op in ('INSERT','UPDATE','DELETE'):
                    trigger=c.execute("SELECT sql FROM sqlite_master WHERE name=?",('b1_fence_'+table+'_'+op.lower(),)).fetchone()
                    self.assertIsNotNone(trigger)
            with self.assertRaises(sqlite3.IntegrityError):c.execute("INSERT INTO persona_exchange_log(profile_id,session_id,exchange_hash,created_at) VALUES('guyan','x','hash','now')")
            with self.assertRaises(sqlite3.IntegrityError):c.execute("INSERT INTO persona_events(profile_id,session_id,message_hash,created_at) VALUES('guyan','x','hash','now')")
    def test_other_profile_and_cross_profile_rename(self):
        self.activate()
        with sqlite3.connect(self.old) as c:
            c.execute("UPDATE persona_session_state SET valence=.1 WHERE profile_id='other'")
            self.assertEqual(c.execute("SELECT valence FROM persona_session_state WHERE profile_id='other'").fetchone()[0],.1)
            with self.assertRaises(sqlite3.IntegrityError):c.execute("UPDATE persona_session_state SET profile_id='guyan',session_id='fresh' WHERE profile_id='other'")
    def test_missing_trigger_invalidates_evidence(self):
        self.activate()
        with sqlite3.connect(self.old) as c:c.execute('DROP TRIGGER b1_fence_persona_session_state_update')
        with self.assertRaises(AuthorityError):self.engine.verify_b1_authority('guyan',1)
        with self.assertRaises(AuthorityError):self.activate()
    def test_activation_transaction_rollback(self):
        with sqlite3.connect(self.old) as c:c.execute("CREATE TRIGGER b1_fence_persona_session_state_update BEFORE UPDATE ON persona_session_state BEGIN SELECT 1; END")
        with self.assertRaises(AuthorityError):self.activate()
        with sqlite3.connect(self.old) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE name='persona_b1_authority'").fetchone()[0],0)
            c.execute("UPDATE persona_session_state SET valence=.3 WHERE profile_id='guyan'")
    def test_same_scope_other_candidate_rejected(self):
        self.activate();s=PersonaB1Store(self.root/'other.db',legacy_path=self.old);s.initialize('guyan',{k:.5 for k in FIELDS},NOW,layers_policy())
        wrong=PersonaStateEngine(self.cfg,str(self.old),b1_candidate=PersonaB1Adapter(s,'guyan',{}))
        with self.assertRaises(AuthorityError):wrong.verify_b1_authority('guyan',1)
    def test_same_path_replaced_seed_rejected(self):
        self.activate()
        with sqlite3.connect(self.store.path) as c:c.execute("UPDATE b1_state SET seed_hash='replaced' WHERE scope='guyan'")
        with self.assertRaises(AuthorityError):self.engine.verify_b1_authority('guyan',1)

    def test_real_legacy_read_decay_does_not_hold_lock_after_fence(self):
        self.activate()
        with self.assertRaises(sqlite3.IntegrityError):self.legacy.get_current_state('chat')
        other=self.other.get_current_state('chat')
        self.assertIsInstance(other,dict)
        with sqlite3.connect(self.old,timeout=.01) as c:
            c.execute("UPDATE persona_session_state SET valence=.2 WHERE profile_id='other'")
            self.assertEqual(c.execute("SELECT valence FROM persona_session_state WHERE profile_id='other'").fetchone()[0],.2)
