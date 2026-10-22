"""Explicit offline authority fencing. No constructor activation or auto migration.

Triggers cover legacy writers even if their Python version lacks this module.
This is write exclusion, not protection against an administrator dropping SQL
triggers, and not a claim that old processes keep serving without errors.
"""
import json,sqlite3
from contextlib import closing
from pathlib import Path

TABLES=('persona_global_state','persona_session_state','persona_events','persona_exchange_log')
class AuthorityError(ValueError):pass

def _path(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise AuthorityError('authority_database_missing_or_alias')
    return p.resolve()

def _connect(path,write=False):
    p=_path(path)
    c=sqlite3.connect(p.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=.1)
    c.row_factory=sqlite3.Row
    return c

def _trigger(table,action):
    # UPDATE must protect both its old and its new identity.
    aliases=('NEW','OLD') if action=='UPDATE' else (('OLD',) if action=='DELETE' else ('NEW',))
    guard=' OR '.join("EXISTS(SELECT 1 FROM persona_b1_authority WHERE profile_id="+a+".profile_id AND mode='candidate')" for a in aliases)
    name='b1_fence_'+table+'_'+action.lower()
    sql=f"CREATE TRIGGER {name} BEFORE {action} ON {table} WHEN {guard} BEGIN SELECT RAISE(ROLLBACK, 'b1_profile_authority_fenced'); END"
    return name,sql

def _schema(c):
    for table in TABLES:
        if 'profile_id' not in {r['name'] for r in c.execute('PRAGMA table_info('+table+')')}:raise AuthorityError('legacy_schema_incomplete')

def _record(c,profile):
    try:r=c.execute('SELECT * FROM persona_b1_authority WHERE profile_id=?',(profile,)).fetchone()
    except sqlite3.OperationalError as e:raise AuthorityError('authority_not_installed') from e
    if r is None:raise AuthorityError('profile_not_owned')
    return dict(r)

def _check_triggers(c):
    for table in TABLES:
        for action in ('INSERT','UPDATE','DELETE'):
            name,sql=_trigger(table,action)
            row=c.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",(name,)).fetchone()
            if row is None or row['sql']!=sql:raise AuthorityError('authority_fence_missing_or_changed')

def activate(legacy_path,adapter,*,profile_id,scope,epoch):
    """Explicit maintenance action on a supplied DB, never called at startup.

Callers must separately establish state handoff and stop/upgrade old serving
processes. No state is copied or guessed; this only installs exclusion.
"""
    if not isinstance(epoch,int) or isinstance(epoch,bool) or epoch<1:raise AuthorityError('invalid_epoch')
    if profile_id!=scope or adapter.scope!=scope:raise AuthorityError('authority_identity_mismatch')
    legacy=_path(legacy_path);candidate=_path(adapter.store.path)
    if legacy==candidate or legacy.samefile(candidate):raise AuthorityError('authority_database_alias')
    adapter.store.read(scope)
    with closing(adapter.store._connect(True)) as candidate_db:
        binding=candidate_db.execute('SELECT policy,seed_hash FROM b1_state WHERE scope=?',(scope,)).fetchone();policy=json.loads(binding[0])['revision'];seed_hash=binding[1]
    c=_connect(legacy,True)
    try:
        c.execute('BEGIN IMMEDIATE');_schema(c)
        c.execute("CREATE TABLE IF NOT EXISTS persona_b1_authority(profile_id TEXT PRIMARY KEY,scope TEXT NOT NULL,epoch INTEGER NOT NULL,mode TEXT NOT NULL,candidate_path TEXT NOT NULL,policy_revision TEXT NOT NULL,seed_hash TEXT NOT NULL)")
        old=c.execute('SELECT * FROM persona_b1_authority WHERE profile_id=?',(profile_id,)).fetchone()
        values=(profile_id,scope,epoch,'candidate',str(candidate),policy,seed_hash)
        if old is not None:
            if tuple(old)!=values:raise AuthorityError('authority_already_owned')
            _check_triggers(c)
        else:
            for table in TABLES:
                for action in ('INSERT','UPDATE','DELETE'):
                    name,sql=_trigger(table,action)
                    row=c.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",(name,)).fetchone()
                    if row is None:c.execute(sql)
                    elif row['sql']!=sql:raise AuthorityError('authority_fence_name_conflict')
            c.execute('INSERT INTO persona_b1_authority VALUES(?,?,?,?,?,?,?)',values)
        c.commit()
    except BaseException:c.rollback();raise
    finally:c.close()
    return verify(legacy,adapter,profile_id=profile_id,scope=scope,epoch=epoch)

def verify(legacy_path,adapter,*,profile_id,scope,epoch):
    if not isinstance(epoch,int) or isinstance(epoch,bool) or epoch<1:raise AuthorityError('invalid_epoch')
    if adapter is None or adapter.scope!=scope or scope!=profile_id:raise AuthorityError('authority_identity_mismatch')
    candidate=_path(adapter.store.path);c=_connect(legacy_path)
    try:
        c.execute('BEGIN');_schema(c);r=_record(c,profile_id);_check_triggers(c)
        adapter.store.read(scope)
        with closing(adapter.store._connect(True)) as candidate_db:
            binding=candidate_db.execute('SELECT policy,seed_hash FROM b1_state WHERE scope=?',(scope,)).fetchone();policy=json.loads(binding[0])['revision'];seed_hash=binding[1]
        if (r['scope'],r['epoch'],r['mode'],r['candidate_path'],r['policy_revision'],r['seed_hash'])!=(scope,epoch,'candidate',str(candidate),policy,seed_hash):raise AuthorityError('authority_binding_mismatch')
        return {'profile_id':profile_id,'scope':scope,'epoch':epoch,'mode':'candidate'}
    finally:c.close()
