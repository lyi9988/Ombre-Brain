"""Explicit Persona B1 preparation and read-only startup binding.

The CLI is an operator action.  Importing this module never seeds a candidate.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sqlite3
from contextlib import closing

from persona_b1 import FIELDS, PersonaB1Adapter, PersonaB1Store, Conflict, number, at, policy
from persona_b1_authority import activate, verify
from persona_inner_state import STATE_SCHEMA, TENDENCIES, normalize_tendencies


def _settings(config):
    b1 = config.get('b1') or {}
    mode = b1.get('mode', 'off')
    if mode == 'off':
        return None
    if mode != 'candidate_authority':
        raise ValueError('unsupported_b1_mode')
    required = ('candidate_db_path', 'legacy_db_path', 'profile_id', 'scope',
                'epoch', 'global_snapshot_path', 'handoff_at')
    if any(not b1.get(k) for k in required):
        raise ValueError('b1_explicit_binding_required')
    if b1['profile_id'] != b1['scope'] or b1['profile_id'] != config.get('persona', {}).get('profile_id'):
        raise ValueError('b1_profile_scope_mismatch')
    if b1.get('legacy_session_id') != config.get('persona', {}).get('canonical_session_id') or not b1.get('legacy_session_id'):
        raise ValueError('b1_canonical_session_mismatch')
    if type(b1['epoch']) is not int or b1['epoch'] < 1:
        raise ValueError('b1_epoch_invalid')
    candidate = Path(b1['candidate_db_path'])
    legacy = Path(b1['legacy_db_path'])
    if candidate.resolve() == legacy.resolve():
        raise ValueError('b1_database_alias')
    return b1


def _read_legacy(b1):
    path = Path(b1['legacy_db_path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('b1_legacy_missing_or_alias')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        profile = b1['profile_id']
        global_row = db.execute('SELECT * FROM persona_global_state WHERE profile_id=?', (profile,)).fetchone()
        if global_row is None:
            raise ValueError('b1_global_source_missing')
        session_id = b1.get('legacy_session_id')
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('b1_explicit_legacy_session_required')
        session = db.execute('SELECT * FROM persona_session_state WHERE profile_id=? AND session_id=?', (profile, session_id)).fetchone()
        if session is None:
            raise ValueError('b1_session_source_missing')
        global_row, session = dict(global_row), dict(session)
    return global_row, session


def _current_global(b1):
    path = Path(b1['legacy_db_path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('b1_legacy_global_unavailable')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT * FROM persona_global_state WHERE profile_id=?',
                         (b1['profile_id'],)).fetchone()
    if row is None:
        raise ValueError('b1_legacy_global_unavailable')
    return dict(row)


def _candidate_checkpoint(b1, path=None):
    path = Path(path or b1['candidate_db_path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('b1_candidate_not_prepared')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT * FROM b1_state WHERE scope=?', (b1['scope'],)).fetchone()
    if row is None:
        raise ValueError('b1_candidate_scope_missing')
    return dict(row)


def _v2_policy_ready(config):
    return (isinstance(config, dict) and not config.get('conflict_policy')
            and 'reconciliation' not in config.get('events', {})
            and 'reconciliation' not in config.get('slow_axis_events', {})
            and bool(config.get('soothing_policy')) and 'soothing' in config.get('events', {}))


def _require_v2_checkpoint(b1):
    # store.read intentionally supports historical v1 inspection in memory.
    # Only the raw persisted marker can authorize a live v2 factory binding.
    row = _candidate_checkpoint(b1)
    if json.loads(row['state']).get('state_schema') != STATE_SCHEMA or not _v2_policy_ready(json.loads(row['policy'])):
        raise ValueError('b1_explicit_inner_state_upgrade_required')
    return row


def _tendency_initial(b1):
    path = b1.get('tendency_source_path')
    if not path:
        return None, None, {'status': 'source_not_configured', 'known_keys': [], 'unknown_keys': list(TENDENCIES)}
    source = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(source, dict) or set(source) != {'schema', 'scope', 'revision', 'tendencies'}
            or source['schema'] != 'persona-tendency-source-v1' or source['scope'] != b1['scope']
            or not isinstance(source['revision'], str) or not source['revision'].strip()
            or not isinstance(source['tendencies'], dict) or set(source['tendencies']) - set(TENDENCIES)):
        raise ValueError('b1_tendency_source_binding_invalid')
    values, sources = {}, {}
    for key, item in source['tendencies'].items():
        if not isinstance(item, dict) or set(item) != {'value', 'unit', 'status', 'source_ref'}:
            raise ValueError('b1_tendency_source_row_invalid')
        values[key] = item['value']
        sources[key] = {field: item[field] for field in ('unit', 'status', 'source_ref')}
    values, sources, _, issues = normalize_tendencies(values, sources)
    return values, sources, {'status': 'explicit_versioned_source', 'revision': source['revision'],
        'known_keys': [key for key, value in values.items() if value is not None],
        'unknown_keys': [key for key, value in values.items() if value is None], 'issues': issues}


class LiveGlobalPersonaB1Adapter(PersonaB1Adapter):
    """B1 owns short affect; original global table remains the sole long-state read."""
    def __init__(self, store, scope, global_snapshot, *, binding):
        super().__init__(store, scope, global_snapshot)
        self._global_binding = binding

    def snapshot(self):
        result = super().snapshot()
        row = _current_global(self._global_binding)
        result['relationship'] = {k: number(row[k]) for k in
            ('affinity', 'dominance', 'defensiveness', 'trust')}
        result['personality'] = {k: number(row[k]) for k in
            ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')}
        return result


def preflight(config):
    b1 = _settings(config)
    if b1 is None:
        return {'mode': 'off'}
    global_row, session = _read_legacy(b1)
    initial = {key: session.get(key) if key in session.keys() else None for key in FIELDS}
    operator_initial = {}
    if b1.get('initial_values_path'):
        operator_initial = json.loads(Path(b1['initial_values_path']).read_text(encoding='utf-8'))
        if not isinstance(operator_initial, dict) or set(operator_initial) - set(FIELDS):
            raise ValueError('b1_initial_values_invalid')
        if set(operator_initial) & set(session):
            raise ValueError('b1_original_field_cannot_be_overridden')
        for key, value in operator_initial.items():
            initial[key] = number(value) if value is not None else None
    handoff_at = at(b1['handoff_at'])
    if datetime.fromisoformat(handoff_at) < datetime.fromisoformat(at(session['updated_at'])):
        raise ValueError('b1_handoff_precedes_source')
    # Existing session defensiveness is a different, directly named coordinate.
    snapshot = {
        'relationship': {k: global_row[k] for k in ('affinity', 'dominance', 'defensiveness', 'trust')},
        'personality': {k: global_row[k] for k in ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')},
        'tendencies': {},
    }
    source = {'profile_id': b1['profile_id'], 'session_id': b1['legacy_session_id'],
              'global': global_row, 'session': session}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    tendencies, tendency_sources, tendency_diagnostic = _tendency_initial(b1)
    return {'mode': 'candidate_authority', 'scope': b1['scope'],
            'source_sha256': source_hash, 'original_at': session['updated_at'],
            'handoff_at': handoff_at, 'gap_coverage': 'unknown',
            'operator_initial_keys': sorted(operator_initial),
            'initial': initial, 'global_snapshot': snapshot,
            'tendency_initial': tendencies, 'tendency_sources': tendency_sources,
            'tendency_source': tendency_diagnostic}


def _snapshot_file(b1, result):
    path = Path(b1['global_snapshot_path'])
    payload = {'scope': b1['scope'], 'source_sha256': result['source_sha256'],
               'original_at': result['original_at'], 'handoff_at': result['handoff_at'],
               'gap_coverage': 'unknown', 'operator_initial_keys': result['operator_initial_keys'],
               'global_snapshot': result['global_snapshot']}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('b1_legacy_source_changed')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
    return payload


def prepare(config):
    b1 = _settings(config)
    if b1 is None:
        raise ValueError('b1_prepare_requires_candidate_mode')
    result = preflight(config)
    _snapshot_file(b1, result)
    policy_path = b1.get('policy_path')
    if not policy_path:
        raise ValueError('b1_policy_path_required')
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    # A complete explicit policy is validated by PersonaB1Store.initialize.
    candidate = Path(b1['candidate_db_path'])
    if not candidate.exists():
        candidate.parent.mkdir(parents=True, exist_ok=True)
    store = PersonaB1Store(candidate, legacy_path=b1['legacy_db_path'])
    store.initialize(b1['scope'], result['initial'], result['handoff_at'], policy,
                     tendencies=result['tendency_initial'], tendency_sources=result['tendency_sources'])
    if preflight(config)['source_sha256'] != result['source_sha256']:
        raise ValueError('b1_legacy_source_changed_during_prepare')
    return {'scope': b1['scope'], 'source_sha256': result['source_sha256'],
            'candidate_db_path': str(candidate.resolve()), 'state': 'prepared'}


def reopen(config, *, require_fence=True):
    b1 = _settings(config)
    if b1 is None:
        return None
    candidate = Path(b1['candidate_db_path'])
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError('b1_candidate_not_prepared')
    _require_v2_checkpoint(b1)
    snapshot_path = Path(b1['global_snapshot_path'])
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ValueError('b1_global_snapshot_missing')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    if snapshot.get('scope') != b1['scope'] or not snapshot.get('source_sha256'):
        raise ValueError('b1_snapshot_binding_invalid')
    if snapshot.get('handoff_at') != at(b1['handoff_at']) or snapshot.get('gap_coverage') != 'unknown':
        raise ValueError('b1_handoff_binding_invalid')
    store = PersonaB1Store(candidate, legacy_path=b1['legacy_db_path'])
    adapter = LiveGlobalPersonaB1Adapter(store, b1['scope'], snapshot['global_snapshot'], binding=b1)
    store.read(b1['scope'])
    if require_fence:
        verify(b1['legacy_db_path'], adapter, profile_id=b1['profile_id'],
               scope=b1['scope'], epoch=b1['epoch'])
    return adapter


def _backup_checkpoint(b1, destination, before):
    destination = Path(destination)
    if destination.is_symlink() or destination.resolve() in {Path(b1['candidate_db_path']).resolve(), Path(b1['legacy_db_path']).resolve()}:
        raise ValueError('b1_upgrade_backup_alias')
    if destination.exists() and any(destination.samefile(Path(b1[key])) for key in ('candidate_db_path', 'legacy_db_path')):
        raise ValueError('b1_upgrade_backup_alias')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        with closing(sqlite3.connect(Path(b1['candidate_db_path']).resolve().as_uri() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)
    if _candidate_checkpoint(b1, destination) != before:
        raise Conflict('b1_upgrade_backup_preimage_mismatch')
    return str(destination.resolve())


def _authority_binding(b1, before):
    from persona_b1_authority import _check_triggers
    with closing(sqlite3.connect(Path(b1['legacy_db_path']).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='persona_b1_authority'").fetchone():
            return None
        row = db.execute('SELECT * FROM persona_b1_authority WHERE profile_id=?', (b1['profile_id'],)).fetchone()
        if row is None:
            return None
        _check_triggers(db)
    binding = dict(row)
    expected = (b1['profile_id'], b1['scope'], b1['epoch'], 'candidate', str(Path(b1['candidate_db_path']).resolve()), before['seed_hash'])
    if tuple(binding[key] for key in ('profile_id', 'scope', 'epoch', 'mode', 'candidate_path', 'seed_hash')) != expected:
        raise ValueError('b1_upgrade_authority_identity_mismatch')
    allowed_revisions = {json.loads(before['policy'])['revision']}
    with closing(sqlite3.connect(Path(b1['candidate_db_path']).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        migrations = db.execute("SELECT receipt FROM b1_events WHERE scope=? AND (event_id=? OR event_id LIKE 'policy-upgrade:%')", (b1['scope'], 'migration:' + STATE_SCHEMA)).fetchall()
        for migrated in migrations:
            receipt = json.loads(migrated[0])
            prior = db.execute('SELECT snapshot FROM b1_numeric WHERE scope=? AND revision=?', (b1['scope'], receipt['before_revision'])).fetchone()
            if prior:
                allowed_revisions.add(json.loads(prior[0])['policy_revision'])
    if binding['policy_revision'] not in allowed_revisions:
        raise ValueError('b1_upgrade_authority_policy_mismatch')
    return binding


def upgrade(config, *, upgraded_policy_path, expected_revision, backup_path):
    """Explicit local maintenance, never invoked by startup or an HTTP request.

    The operator supplies a complete approved v2 policy and a backup path. Both
    databases retain their existing identities; an installed fence is rebound
    only by comparing its exact preimage. A failed second-database rebind blocks
    startup and the same approved upgrade can reconcile the recorded migration.
    Serving processes must be stopped or otherwise controlled by the operator.
    """
    b1 = _settings(config)
    if b1 is None or type(expected_revision) is not int or expected_revision < 0:
        raise ValueError('b1_explicit_upgrade_binding_and_revision_required')
    approved = policy(json.loads(Path(upgraded_policy_path).read_text(encoding='utf-8')))
    if not _v2_policy_ready(approved):
        raise ValueError('b1_explicit_v2_policy_required')
    before = _candidate_checkpoint(b1)
    binding = _authority_binding(b1, before)
    unchanged = json.loads(before['state']).get('state_schema') == STATE_SCHEMA and json.loads(before['policy']) == approved
    if not unchanged and before['revision'] != expected_revision:
        raise Conflict('stale upgrade revision')
    backup = _backup_checkpoint(b1, backup_path, before) if not unchanged else None
    tendencies, sources, tendency_diagnostic = _tendency_initial(b1)
    store = PersonaB1Store(b1['candidate_db_path'], legacy_path=b1['legacy_db_path'])
    if json.loads(before['state']).get('state_schema') == STATE_SCHEMA and _v2_policy_ready(json.loads(before['policy'])):
        result = store.upgrade_policy(b1['scope'], approved, expected_revision=expected_revision)
    else:
        result = store.migrate_scope(b1['scope'], expected_revision=expected_revision,
                                    upgraded_policy=approved, tendencies=tendencies, tendency_sources=sources)
    _require_v2_checkpoint(b1)
    rebound = False
    if binding is not None and binding['policy_revision'] != approved['revision']:
        from persona_b1_authority import _check_triggers
        with closing(sqlite3.connect(Path(b1['legacy_db_path']).resolve().as_uri() + '?mode=rw', uri=True)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN IMMEDIATE')
            _check_triggers(db)
            current = db.execute('SELECT * FROM persona_b1_authority WHERE profile_id=?', (b1['profile_id'],)).fetchone()
            if current is None or dict(current) != binding:
                raise Conflict('b1_upgrade_authority_preimage_changed')
            db.execute('UPDATE persona_b1_authority SET policy_revision=? WHERE profile_id=?', (approved['revision'], b1['profile_id']))
            rebound = True
    if binding is not None:
        verify(b1['legacy_db_path'], PersonaB1Adapter(store, b1['scope'], {}),
               profile_id=b1['profile_id'], scope=b1['scope'], epoch=b1['epoch'])
    return {'scope': b1['scope'], 'state': 'upgraded', 'state_schema': STATE_SCHEMA,
            'migrated': result['migrated'], 'revision': result['state']['revision'],
            'policy_revision': approved['revision'], 'backup_path': backup,
            'authority_rebound': rebound, 'activation_required': binding is None,
            'tendency_source': tendency_diagnostic, 'production_applied': False}


def make_engine(config):
    """The actual server/Gateway factory, including engine path verification."""
    from persona_engine import PersonaStateEngine
    b1 = _settings(config)
    adapter = reopen(config) if b1 is not None else None
    engine = PersonaStateEngine(config, b1_candidate=adapter)
    if b1 is not None:
        if Path(engine.db_path).resolve() != Path(b1['legacy_db_path']).resolve():
            raise ValueError('b1_running_legacy_path_mismatch')
        engine.verify_b1_authority(b1['scope'], b1['epoch'])
    return engine


def activate_prepared(config):
    b1 = _settings(config)
    current = preflight(config)
    _snapshot_file(b1, current)
    adapter = reopen(config, require_fence=False)
    return activate(b1['legacy_db_path'], adapter, profile_id=b1['profile_id'],
                    scope=b1['scope'], epoch=b1['epoch'])


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'prepare', 'upgrade', 'activate', 'verify'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--upgraded-policy')
    parser.add_argument('--expected-revision', type=int)
    parser.add_argument('--backup')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.action == 'preflight':
        value = preflight(config)
        value.pop('initial', None); value.pop('global_snapshot', None)
        value.pop('tendency_initial', None); value.pop('tendency_sources', None)
    elif args.action == 'prepare':
        value = prepare(config)
    elif args.action == 'activate':
        value = activate_prepared(config)
    elif args.action == 'upgrade':
        if not args.upgraded_policy or args.expected_revision is None or not args.backup:
            parser.error('upgrade requires --upgraded-policy, --expected-revision and --backup')
        value = upgrade(config, upgraded_policy_path=args.upgraded_policy,
                        expected_revision=args.expected_revision, backup_path=args.backup)
    else:
        reopen(config); value = {'state': 'verified'}
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
