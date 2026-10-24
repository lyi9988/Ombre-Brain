"""Explicit local v2 factory/upgrade and real authenticated HTTP fixtures."""
from copy import deepcopy
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hmac
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from persona_b1 import PersonaB1Store, PersonaB1Adapter, Conflict
from persona_b1_authority import activate
from persona_b1_http import register_candidate_route
from persona_b1_runtime import prepare, upgrade, activate_prepared, reopen, make_engine
from persona_engine import PersonaStateEngine
from persona_inner_state import STATE_SCHEMA, TENDENCIES, upgrade_policy_v2
from test_persona_inner_state_v2 import fixture_policy, source, observation, decision, turn


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.when = datetime.now(timezone.utc) + timedelta(days=1)
        self.config = {'buckets_dir': str(self.root),
            'persona': {'profile_id': 'fixture', 'canonical_session_id': 'chat', 'mode': 'disabled', 'db_path': str(self.root / 'legacy.db')},
            'b1': {'mode': 'candidate_authority', 'legacy_db_path': str(self.root / 'legacy.db'),
                'candidate_db_path': str(self.root / 'candidate.db'), 'profile_id': 'fixture', 'scope': 'fixture',
                'epoch': 1, 'legacy_session_id': 'chat', 'handoff_at': self.when.isoformat(),
                'global_snapshot_path': str(self.root / 'global.json'), 'policy_path': str(self.root / 'policy.json')}}
        engine = PersonaStateEngine(self.config)
        engine.get_current_state('chat')
        with sqlite3.connect(self.config['b1']['legacy_db_path']) as db:
            db.execute("UPDATE persona_session_state SET valence=.8 WHERE profile_id='fixture'")
        self.approved = upgrade_policy_v2(fixture_policy())
        self.policy_path = self.root / 'approved-v2.json'
        self.policy_path.write_text(json.dumps(self.approved), encoding='utf-8')
        Path(self.config['b1']['policy_path']).write_text(json.dumps(fixture_policy()), encoding='utf-8')
        self.backup = self.root / 'before-v2.sqlite'

    def checkpoint(self, path=None):
        with sqlite3.connect(path or self.config['b1']['candidate_db_path']) as db:
            db.row_factory = sqlite3.Row
            return dict(db.execute('SELECT * FROM b1_state WHERE scope=?', ('fixture',)).fetchone())

    def old_fixture(self):
        prepare(self.config)
        row = self.checkpoint()
        state = json.loads(row['state'])
        for key in ('state_schema', 'state_migrations', 'tendency_sources', 'tendency_evidence', 'migration_issues'):
            state.pop(key, None)
        state['slow_mood'] = {'value': -.6, 'unit': '-1..1', 'covered_hours': 1, 'unknown_hours': 0}
        state['tendencies'] = {'exploration_affinity': 12.5}
        with sqlite3.connect(self.config['b1']['candidate_db_path']) as db:
            db.execute('UPDATE b1_state SET state=? WHERE scope=?', (json.dumps(state), 'fixture'))
        return self.checkpoint()

    def migrate(self, **kwargs):
        return upgrade(self.config, upgraded_policy_path=self.policy_path, expected_revision=0, backup_path=self.backup, **kwargs)

    def test_startup_rejects_unmigrated_disk_without_write_or_implicit_upgrade(self):
        before = self.old_fixture()
        with patch.object(PersonaB1Store, 'migrate_scope', side_effect=AssertionError('startup may not migrate')):
            with self.assertRaisesRegex(ValueError, 'explicit_inner_state_upgrade_required'):
                reopen(self.config, require_fence=False)
        self.assertEqual(self.checkpoint(), before)
        self.assertFalse(self.backup.exists())

    def test_explicit_upgrade_backup_activate_and_restart_use_v2_exactly_once(self):
        before = self.old_fixture()
        result = self.migrate()
        self.assertTrue(result['migrated'])
        self.assertTrue(result['activation_required'])
        self.assertFalse(result['production_applied'])
        self.assertEqual(self.checkpoint(self.backup), before)
        activate_prepared(self.config)
        with patch.object(PersonaB1Store, 'migrate_scope', side_effect=AssertionError('startup may not migrate')):
            engine = make_engine(self.config)
            a, b = engine.b1_candidate.snapshot(), reopen(self.config).snapshot()
        self.assertEqual(a['state_schema'], STATE_SCHEMA)
        self.assertAlmostEqual(a['slow_axes']['mood']['value'], .2)
        self.assertEqual(a['slow_axes']['mood']['unit'], '0..1')
        self.assertEqual(a['affect']['valence'], .8)
        self.assertEqual(a['revision'], b['revision'])
        self.assertTrue(all(value is None for value in a['tendencies'].values()))
        self.assertEqual(self.migrate()['migrated'], False)
        self.assertEqual(self.checkpoint(self.backup), before)

    def test_prepare_with_explicit_v2_policy_can_activate_without_a_synthetic_legacy_migration(self):
        Path(self.config['b1']['policy_path']).write_text(json.dumps(self.approved), encoding='utf-8')
        prepare(self.config)
        activate_prepared(self.config)
        self.assertEqual(make_engine(self.config).b1_candidate.snapshot()['state_schema'], STATE_SCHEMA)

    def test_tendencies_require_explicit_versioned_unit_source_and_never_reset_at_startup(self):
        path = self.root / 'tendencies.json'
        data = {'schema': 'persona-tendency-source-v1', 'scope': 'fixture', 'revision': 'fixture-source-1',
            'tendencies': {'exploration_affinity': {'value': .7, 'unit': '0..1', 'status': 'known', 'source_ref': 'source:declared'},
                           'reflection_preference': {'value': 12.5, 'unit': 'source_native', 'status': 'known', 'source_ref': 'source:unknown-scale'}}}
        path.write_text(json.dumps(data), encoding='utf-8')
        self.config['b1']['tendency_source_path'] = str(path)
        Path(self.config['b1']['policy_path']).write_text(json.dumps(self.approved), encoding='utf-8')
        prepare(self.config)
        activate_prepared(self.config)
        first = reopen(self.config).snapshot()
        self.assertEqual(first['tendencies']['exploration_affinity'], .7)
        self.assertIsNone(first['tendencies']['reflection_preference'])
        self.assertIsNone(first['tendencies']['social_recovery_rate'])
        path.write_text('no longer a valid source file', encoding='utf-8')
        later = reopen(self.config).snapshot()
        self.assertEqual(later['tendencies'], first['tendencies'])
        self.assertEqual(later['revision'], first['revision'])

    def test_active_fence_policy_rebind_retains_same_identity_and_business_data(self):
        self.old_fixture()
        store = PersonaB1Store(self.config['b1']['candidate_db_path'], legacy_path=self.config['b1']['legacy_db_path'])
        activate(self.config['b1']['legacy_db_path'], PersonaB1Adapter(store, 'fixture', {}), profile_id='fixture', scope='fixture', epoch=1)
        with sqlite3.connect(self.config['b1']['legacy_db_path']) as db:
            before = db.execute('SELECT * FROM persona_global_state').fetchall()
        result = self.migrate()
        self.assertTrue(result['authority_rebound'])
        self.assertFalse(result['activation_required'])
        self.assertEqual(make_engine(self.config).b1_candidate.snapshot()['policy_revision'], self.approved['revision'])
        with sqlite3.connect(self.config['b1']['legacy_db_path']) as db:
            self.assertEqual(db.execute('SELECT * FROM persona_global_state').fetchall(), before)

    def test_unknown_outcome_between_candidate_and_fence_can_reconcile_without_replaying_migration(self):
        self.old_fixture()
        store = PersonaB1Store(self.config['b1']['candidate_db_path'], legacy_path=self.config['b1']['legacy_db_path'])
        activate(self.config['b1']['legacy_db_path'], PersonaB1Adapter(store, 'fixture', {}), profile_id='fixture', scope='fixture', epoch=1)
        import persona_b1_authority
        original = persona_b1_authority._check_triggers
        calls = []
        def stop_before_rebind(db):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError('fixture interrupted before second database commit')
            return original(db)
        with patch.object(persona_b1_authority, '_check_triggers', side_effect=stop_before_rebind):
            with self.assertRaisesRegex(ValueError, 'fixture interrupted'):
                self.migrate()
        self.assertEqual(self.checkpoint()['revision'], 1)
        with self.assertRaisesRegex(ValueError, 'authority_binding_mismatch'):
            reopen(self.config)
        replay = self.migrate()
        self.assertFalse(replay['migrated'])
        self.assertTrue(replay['authority_rebound'])
        self.assertEqual(make_engine(self.config).b1_candidate.snapshot()['revision'], 1)

    def test_stale_revision_wrong_policy_and_backup_alias_never_write(self):
        before = self.old_fixture()
        with self.assertRaises(Conflict):
            upgrade(self.config, upgraded_policy_path=self.policy_path, expected_revision=9, backup_path=self.backup)
        self.assertFalse(self.backup.exists())
        with self.assertRaisesRegex(ValueError, 'explicit_v2_policy_required'):
            upgrade(self.config, upgraded_policy_path=self.config['b1']['policy_path'], expected_revision=0, backup_path=self.backup)
        with self.assertRaisesRegex(ValueError, 'backup_alias'):
            upgrade(self.config, upgraded_policy_path=self.policy_path, expected_revision=0, backup_path=self.config['b1']['candidate_db_path'])
        self.assertEqual(self.checkpoint(), before)

    def test_hardlinked_backup_is_rejected_and_off_mode_never_reopens_candidate(self):
        self.old_fixture()
        try:
            os.link(self.config['b1']['candidate_db_path'], self.backup)
        except OSError:
            self.skipTest('filesystem does not support a hardlink fixture')
        with self.assertRaisesRegex(ValueError, 'backup_alias'):
            self.migrate()
        disabled = deepcopy(self.config)
        disabled['b1']['mode'] = 'off'
        self.assertIsNone(reopen(disabled))


class Routes:
    def __init__(self): self.routes = []
    def custom_route(self, path, methods):
        def register(fn):
            self.routes.append(Route(path, fn, methods=methods))
            return fn
        return register


class V2HTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = RuntimeFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.old_fixture()
        self.fixture.migrate()
        activate_prepared(self.fixture.config)
        self.engine = make_engine(self.fixture.config)
        registry = Routes()
        register_candidate_route(registry, lambda: self.engine,
            lambda request: hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer fixture-only'))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=registry.routes)), base_url='http://fixture.invalid')
        self.addAsyncCleanup(self.client.aclose)
        self.headers = {'Authorization': 'Bearer fixture-only'}

    async def test_real_v2_turn_and_numeric_http_keep_normalized_values_and_replay(self):
        src = source('抱抱你。')
        row = observation(src)
        envelope = {'namespace': 'b1-completed-turn-v2', 'scope': 'fixture', 'event_id': src['user_id'],
            'occurred_at': (self.fixture.when + timedelta(seconds=1)).isoformat(),
            'policy_revision': self.fixture.approved['revision'], 'decision': decision(src, [row]), 'labels': []}
        payload = {'envelope': envelope, 'turn': turn(src)}
        first = await self.client.post('/internal/persona/b1/candidate', json=payload, headers=self.headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['projection']['state_schema'], STATE_SCHEMA)
        self.assertAlmostEqual(first.json()['projection']['slow_axes']['mood']['value'], .2)
        self.assertEqual(first.json()['projection']['slow_axes']['mood']['unit'], '0..1')
        replay = await self.client.post('/internal/persona/b1/candidate', json=payload, headers=self.headers)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertTrue(replay.json()['replayed'])
        before = self.engine.b1_candidate.store.read('fixture')
        numeric = await self.client.get('/internal/persona/b1/numeric', params={'scope': 'fixture'}, headers=self.headers)
        self.assertEqual(numeric.status_code, 200)
        self.assertEqual(numeric.json()['snapshot']['values']['valence'], before['values']['valence'])
        self.assertGreater(numeric.json()['snapshot']['values']['valence'], .7)
        history = await self.client.get('/internal/persona/b1/numeric', params={'scope': 'fixture', 'at': self.fixture.when.isoformat()}, headers=self.headers)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()['snapshot']['revision'], 1)
        self.assertEqual(self.engine.b1_candidate.store.read('fixture'), before)

    async def test_old_live_turn_is_rejected_without_private_error_or_legacy_write(self):
        before = self.engine.b1_candidate.store.read('fixture')
        data = {'envelope': {'namespace': 'b1-completed-turn-v1', 'private': 'PRIVATE_SENTINEL'}, 'turn': {}}
        response = await self.client.post('/internal/persona/b1/candidate', json=data, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('PRIVATE_SENTINEL', response.text)
        self.assertEqual(self.engine.b1_candidate.store.read('fixture'), before)
        unauthorized = await self.client.get('/internal/persona/b1/numeric', params={'scope': 'fixture'})
        self.assertEqual(unauthorized.status_code, 401)


if __name__ == '__main__': unittest.main()
