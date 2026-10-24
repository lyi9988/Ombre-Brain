"""Later explicit policy cutovers retain Brain state and recover authority fences."""
from copy import deepcopy
import json
import sqlite3
import unittest
from unittest.mock import patch

from persona_b1 import Conflict
from persona_b1_runtime import upgrade, activate_prepared, make_engine, reopen
from test_persona_b1_v2_runtime import RuntimeFixture


class PolicyUpgradeV2(unittest.TestCase):
    def setUp(self):
        self.fixture = RuntimeFixture();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.old_fixture();self.fixture.migrate();activate_prepared(self.fixture.config)

    def new_policy(self):
        row = self.fixture.checkpoint(); p = json.loads(row['policy'])
        p['revision'] = 'fixture-later-policy-v3'
        p['growth_policy'] = {'revision':'fixture-later-growth-v3','grid_seconds':900,
            'anchor_utc':row['at'],'source_scope':'fixture','source_policy_revision':p['revision'],
            'targets':{'novelty':{'emotion':{'valence_slope':.25,'arousal_slope':.20,
                'valence_center':.5,'valence_span':.5,'arousal_center':.30,'arousal_span':.7},
                'drives':{},'floor':.4,'ceiling':1.8}}}
        path = self.fixture.root/'later-policy.json';path.write_text(json.dumps(p),encoding='utf-8')
        return p,path

    def test_existing_v2_policy_upgrade_backs_up_and_reopens_once(self):
        before = self.fixture.checkpoint();p,path=self.new_policy()
        backup = self.fixture.root/'before-later.db'
        result=upgrade(self.fixture.config,upgraded_policy_path=path,
                       expected_revision=before['revision'],backup_path=backup)
        self.assertTrue(result['migrated']);self.assertTrue(result['authority_rebound'])
        after=self.fixture.checkpoint()
        self.assertEqual(after['state'],before['state']);self.assertEqual(after['at'],before['at'])
        self.assertEqual(self.fixture.checkpoint(backup),before)
        self.assertEqual(json.loads(after['policy']),p)
        self.assertEqual(make_engine(self.fixture.config).b1_candidate.snapshot()['revision'],after['revision'])
        replay=upgrade(self.fixture.config,upgraded_policy_path=path,
                       expected_revision=before['revision'],backup_path=backup)
        self.assertFalse(replay['migrated']);self.assertEqual(self.fixture.checkpoint(),after)
        with sqlite3.connect(self.fixture.config['b1']['candidate_db_path']) as db:
            receipt=json.loads(db.execute("SELECT receipt FROM b1_events WHERE event_id=?",('policy-upgrade:'+p['revision'],)).fetchone()[0])
            self.assertEqual(receipt['before_policy'],json.loads(before['policy']))
            self.assertEqual(db.execute('SELECT count(*) FROM b1_numeric').fetchone()[0],3)

    def test_rebind_interruption_recovers_original_policy_receipt_without_reapply(self):
        before=self.fixture.checkpoint();p,path=self.new_policy()
        import persona_b1_authority
        original=persona_b1_authority._check_triggers;calls=[]
        def interrupt(db):
            calls.append(True)
            if len(calls)==2:raise RuntimeError('policy rebind interrupted')
            return original(db)
        backup=self.fixture.root/'rebind-preimage.db'
        with patch.object(persona_b1_authority,'_check_triggers',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                upgrade(self.fixture.config,upgraded_policy_path=path,
                        expected_revision=before['revision'],backup_path=backup)
        after=self.fixture.checkpoint()
        self.assertEqual(after['revision'],before['revision']+1)
        with self.assertRaises(ValueError):reopen(self.fixture.config)
        replay=upgrade(self.fixture.config,upgraded_policy_path=path,
                       expected_revision=before['revision'],backup_path=backup)
        self.assertFalse(replay['migrated']);self.assertTrue(replay['authority_rebound'])
        self.assertEqual(self.fixture.checkpoint(),after)
        self.assertEqual(make_engine(self.fixture.config).b1_candidate.snapshot()['policy_revision'],p['revision'])

    def test_reusing_brain_or_frame_revision_is_rejected(self):
        before=self.fixture.checkpoint();p,path=self.new_policy()
        p['revision']=json.loads(before['policy'])['revision']
        path.write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaisesRegex(Conflict,'new revision'):
            upgrade(self.fixture.config,upgraded_policy_path=path,
                expected_revision=before['revision'],backup_path=self.fixture.root/'same-revision.db')
        self.assertEqual(self.fixture.checkpoint(),before)

    def test_first_v1_migration_can_adopt_new_shared_growth_artifact(self):
        fresh=RuntimeFixture();fresh.setUp();self.addCleanup(fresh.doCleanups)
        before=fresh.old_fixture()
        p=deepcopy(fresh.approved);p['revision']='first-shared-brain-v2'
        p['growth_policy']={'revision':'first-shared-growth-v2','grid_seconds':900,
            'anchor_utc':before['at'],'source_scope':'fixture','source_policy_revision':p['revision'],
            'targets':{'novelty':{'emotion':{'valence_slope':.25,'arousal_slope':.20,
                'valence_center':.5,'valence_span':.5,'arousal_center':.30,'arousal_span':.7},
                'drives':{},'floor':.4,'ceiling':1.8}}}
        path=fresh.root/'first-shared.json';path.write_text(json.dumps(p),encoding='utf-8')
        result=upgrade(fresh.config,upgraded_policy_path=path,expected_revision=0,backup_path=fresh.backup)
        self.assertTrue(result['migrated']);activate_prepared(fresh.config)
        engine=make_engine(fresh.config)
        self.assertEqual(engine.b1_candidate.snapshot()['policy_revision'],p['revision'])
        self.assertEqual(json.loads(fresh.checkpoint()['policy'])['growth_policy'],p['growth_policy'])
        self.assertEqual(json.loads(fresh.checkpoint()['state'])['values'],json.loads(before['state'])['values'])

    def test_reusing_shared_frame_revision_under_new_brain_revision_is_rejected(self):
        before=self.fixture.checkpoint();p,path=self.new_policy()
        upgrade(self.fixture.config,upgraded_policy_path=path,expected_revision=before['revision'],
                backup_path=self.fixture.root/'first-preimage.db')
        before=self.fixture.checkpoint()
        p['revision']='another-brain-policy'
        p['growth_policy']['source_policy_revision']=p['revision']
        path.write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaisesRegex(Conflict,'frame policy requires a new revision'):
            upgrade(self.fixture.config,upgraded_policy_path=path,expected_revision=before['revision'],
                    backup_path=self.fixture.root/'second-preimage.db')
        self.assertEqual(self.fixture.checkpoint(),before)


if __name__=='__main__':unittest.main()
