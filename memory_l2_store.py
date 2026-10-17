"""Versioned second-pass presentations in the existing Memory Authority database.

These are editable products of L1, not another source of adopted memory facts.
All head changes and task claims use the Authority transaction boundary.
"""
from __future__ import annotations

import json
import time
import uuid

from memory_authority import IdempotencyConflict, MemoryNotFound, RevisionConflict
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_service import L1OwnerError


def initialize(conn):
    for sql in (
        """CREATE TABLE IF NOT EXISTS memory_l2_heads (
            memory_id TEXT PRIMARY KEY, presentation_revision INTEGER NOT NULL DEFAULT 0,
            owner_protected INTEGER NOT NULL DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS memory_l2_revisions (
            memory_id TEXT NOT NULL, presentation_revision INTEGER NOT NULL,
            first_pass_revision INTEGER NOT NULL, dependency_json TEXT NOT NULL,
            document_json TEXT NOT NULL, actor TEXT NOT NULL, created_at REAL NOT NULL,
            request_id TEXT NOT NULL, request_sha256 TEXT NOT NULL, sync_state TEXT NOT NULL,
            PRIMARY KEY(memory_id,presentation_revision), UNIQUE(memory_id,request_id))""",
        """CREATE TABLE IF NOT EXISTS memory_l2_jobs (
            task_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, first_pass_revision INTEGER NOT NULL,
            state TEXT NOT NULL, dependency_json TEXT NOT NULL DEFAULT '{}',
            base_presentation_revision INTEGER NOT NULL DEFAULT 0, lease_token TEXT NOT NULL DEFAULT '',
            lease_until REAL NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS memory_l2_job_queue ON memory_l2_jobs(state,created_at)",
        """CREATE TABLE IF NOT EXISTS memory_l2_sync_proposals (
            memory_id TEXT NOT NULL, presentation_revision INTEGER NOT NULL,
            first_pass_revision INTEGER NOT NULL, state TEXT NOT NULL,
            PRIMARY KEY(memory_id,presentation_revision))""",
    ):
        conn.execute(sql)
    if 'edit_note' not in {row[1] for row in conn.execute('PRAGMA table_info(memory_l2_revisions)')}:
        conn.execute("ALTER TABLE memory_l2_revisions ADD COLUMN edit_note TEXT NOT NULL DEFAULT ''")
    if 'publication' not in {row[1] for row in conn.execute('PRAGMA table_info(memory_l2_revisions)')}:
        conn.execute("ALTER TABLE memory_l2_revisions ADD COLUMN publication TEXT NOT NULL DEFAULT 'active'")
    if 'units_state' not in {row[1] for row in conn.execute('PRAGMA table_info(memory_l2_revisions)')}:
        conn.execute("ALTER TABLE memory_l2_revisions ADD COLUMN units_state TEXT NOT NULL DEFAULT 'current'")
    if 'proposal_json' not in {row[1] for row in conn.execute('PRAGMA table_info(memory_l2_sync_proposals)')}:
        conn.execute("ALTER TABLE memory_l2_sync_proposals ADD COLUMN proposal_json TEXT NOT NULL DEFAULT '{}'")


def enqueue_in_transaction(conn, memory_id, revision, *, request_id=None):
    """Called by L1 commit before its outbox transaction is committed."""
    task_id = 'l2-' + semantic_sha256({'memory_id':memory_id, 'revision':revision, 'request_id':request_id or 'commit'})
    stamp = time.time()
    conn.execute("INSERT OR IGNORE INTO memory_l2_heads(memory_id) VALUES(?)", (memory_id,))
    conn.execute("""INSERT OR IGNORE INTO memory_l2_jobs
        (task_id,memory_id,first_pass_revision,state,created_at,updated_at)
        VALUES(?,?,?,'pending',?,?)""", (task_id, memory_id, revision, stamp, stamp))
    return task_id


def validate_document(document, statement_ids):
    required = {'full_injection_text', 'coherent_sections', 'optional_use_variants'}
    if not isinstance(document, dict) or not required <= set(document) or set(document) - required - {'review'}:
        raise L1OwnerError('invalid_l2_document', field_path='document')
    if not isinstance(document['full_injection_text'], str) or not document['full_injection_text'].strip():
        raise L1OwnerError('l2_full_text_required', field_path='document.full_injection_text')
    if 'review' in document:
        review = document['review']
        if (not isinstance(review, dict) or set(review) != {'required','reasons'} or type(review['required']) is not bool
                or not isinstance(review['reasons'], list) or any(not isinstance(s,str) for s in review['reasons'])):
            raise L1OwnerError('invalid_l2_review', field_path='document.review')
    known = set(statement_ids)
    for module, key, extra in (('coherent_sections', 'section_id', set()),
                               ('optional_use_variants', 'variant_id', {'intended_use'})):
        rows = document[module]
        if not isinstance(rows, list): raise L1OwnerError('array_required', field_path='document.' + module)
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {key, 'text', 'statement_refs'} | extra
                    or not isinstance(row[key], str) or not row[key].strip() or row[key] in seen
                    or not isinstance(row['text'], str) or not row['text'].strip()
                    or not isinstance(row['statement_refs'], list) or not row['statement_refs']
                    or any(not isinstance(ref, str) or ref not in known for ref in row['statement_refs'])
                    or (extra and (not isinstance(row['intended_use'], str) or not row['intended_use']))):
                raise L1OwnerError('invalid_l2_unit', field_path='document.' + module)
            if extra and row['intended_use'] not in {'current_basis', 'conditional_basis', 'relationship_reference', 'historical_evidence', 'unresolved_evidence'}:
                raise L1OwnerError('invalid_l2_intended_use', field_path='document.' + module)
            seen.add(row[key])
    canonical_json(document)
    return document


class MemoryL2Store:
    def __init__(self, authority): self.authority = authority

    def request_receipt(self, memory_id, request_id, fingerprint):
        conn = self.authority._connect()
        try:
            row = conn.execute('SELECT presentation_revision,sync_state,request_sha256 FROM memory_l2_revisions WHERE memory_id=? AND request_id=?', (memory_id, request_id)).fetchone()
            if row is None: return None
            if row['request_sha256'] != fingerprint: raise IdempotencyConflict('L2 request changed')
            return {'presentation_revision':row['presentation_revision'], 'sync_state':row['sync_state'], 'replayed':True}
        finally: conn.close()

    @staticmethod
    def dependency(conn, memory_id, generation):
        row = conn.execute("""SELECT m.*,r.metadata_json FROM memories m JOIN memory_revisions r
            ON r.memory_id=m.memory_id AND r.revision=m.active_revision WHERE m.memory_id=?""", (memory_id,)).fetchone()
        if row is None: raise MemoryNotFound(memory_id)
        meta = json.loads(row['metadata_json'])
        rings = conn.execute("SELECT ring_id,content_sha256,state FROM memory_rings WHERE memory_id=? ORDER BY ring_id", (memory_id,)).fetchall()
        parent = (meta.get('memory_l1') or {}).get('entry', {}).get('parent_ref')
        parent_current = None
        if parent:
            root = conn.execute('SELECT active_revision,body_sha256,state,recall_policy FROM memories WHERE memory_id=?', (parent['memory_id'],)).fetchone()
            parent_current = dict(root) if root else {'state': 'missing'}
        return {'memory_id': memory_id, 'first_pass_revision': row['active_revision'], 'body_hash': row['body_sha256'],
                'l1_content_sha256': meta.get('memory_l1_content_sha256'),
                'l1_semantic_sha256': meta.get('memory_l1_semantic_sha256'),
                'memory_state': row['state'], 'recall_policy': row['recall_policy'],
                'rings': [list(r) for r in rings], 'parent_ref': parent, 'parent_current': parent_current, **generation}

    def current_dependency(self, memory_id, generation):
        conn = self.authority._connect()
        try: return self.dependency(conn, memory_id, generation)
        finally: conn.close()

    @staticmethod
    def _revision(row):
        if row is None: return None
        item = dict(row)
        item['dependency_versions'] = json.loads(item.pop('dependency_json'))
        item['document'] = json.loads(item.pop('document_json'))
        item.pop('request_sha256', None)
        item['presentation_id'] = 'presentation:' + item['memory_id']
        return item

    def detail(self, memory_id, dependency, revision=None):
        conn = self.authority._connect()
        try:
            head = conn.execute('SELECT * FROM memory_l2_heads WHERE memory_id=?', (memory_id,)).fetchone()
            selected = revision if revision is not None else (head['presentation_revision'] if head else 0)
            row = conn.execute('SELECT * FROM memory_l2_revisions WHERE memory_id=? AND presentation_revision=?', (memory_id, selected)).fetchone()
            if revision is not None and row is None: raise MemoryNotFound('presentation_revision')
            product = self._revision(row)
            jobs = [dict(r) for r in conn.execute('SELECT task_id,state,reason,first_pass_revision,created_at,updated_at FROM memory_l2_jobs WHERE memory_id=? ORDER BY created_at DESC LIMIT 50', (memory_id,))]
            state, reason = 'not_prepared', 'l2_missing'
            if product:
                if product['sync_state'] == 'pending_l1': state, reason = 'pending_sync', 'l1_sync_required'
                elif product['dependency_versions'] != dependency: state, reason = 'stale', 'dependency_changed'
                elif product['publication'] == 'candidate': state, reason = 'pending_review', 'owner_l2_preserved'
                elif product['actor'] == 'model' and product['document'].get('review', {}).get('required', True):
                    state, reason = 'pending_review', 'l2_semantic_review_required'
                else: state, reason = 'ready', ''
            elif jobs:
                state, reason = jobs[0]['state'], jobs[0]['reason']
            if dependency['memory_state'] != 'active' or dependency['recall_policy'] == 'disabled':
                state, reason = 'stale', 'memory_not_eligible'
            if not dependency.get('l1_content_sha256'): state, reason = 'not_prepared', 'l1_required'
            revisions = [dict(r) for r in conn.execute('SELECT presentation_revision,first_pass_revision,actor,created_at,sync_state,publication FROM memory_l2_revisions WHERE memory_id=? ORDER BY presentation_revision DESC', (memory_id,))]
            proposals = [dict(r) for r in conn.execute('SELECT * FROM memory_l2_sync_proposals WHERE memory_id=? ORDER BY presentation_revision DESC', (memory_id,))]
            for proposal in proposals:
                proposal['proposal'] = json.loads(proposal.pop('proposal_json'))
            return {'state': state, 'reason': reason, 'presentation': product,
                    'current_presentation_revision': head['presentation_revision'] if head else 0,
                    'owner_protected': bool(head and head['owner_protected']),
                    'revisions': revisions, 'jobs': jobs, 'sync_proposals': proposals,
                    'current_dependencies': dependency}
        finally: conn.close()

    def enqueue(self, memory_id, dependency, request_id):
        conn = self.authority._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            current = self.dependency(conn, memory_id, {k:v for k,v in dependency.items() if k not in
                {'memory_id','first_pass_revision','body_hash','l1_content_sha256','l1_semantic_sha256','memory_state','recall_policy','rings','parent_ref','parent_current'}})
            if current != dependency: raise RevisionConflict('L1 changed')
            if not current.get('l1_content_sha256'): raise L1OwnerError('l1_required', status=409)
            request_key = 'l2-prepare:' + semantic_sha256({'memory_id':memory_id,'request_id':request_id})
            previous = conn.execute('SELECT value FROM authority_meta WHERE key=?', (request_key,)).fetchone()
            if previous and int(previous['value']) != dependency['first_pass_revision']:
                raise IdempotencyConflict('L2 preparation request changed')
            conn.execute('INSERT OR IGNORE INTO authority_meta VALUES(?,?)', (request_key, str(dependency['first_pass_revision'])))
            ident = enqueue_in_transaction(conn, memory_id, dependency['first_pass_revision'], request_id=request_id)
            if not request_id.startswith('dependency:'):
                head = conn.execute('SELECT presentation_revision FROM memory_l2_heads WHERE memory_id=?', (memory_id,)).fetchone()
                conn.execute("UPDATE memory_l2_jobs SET reason='owner_requested',base_presentation_revision=? WHERE task_id=? AND state='pending' AND reason!='owner_requested'", (head['presentation_revision'], ident))
            conn.commit()
            return {'task_id': ident, 'state': 'pending'}
        except Exception:
            conn.rollback(); raise
        finally: conn.close()

    def pending(self, limit):
        conn = self.authority._connect()
        try:
            # A lost lease is surfaced, never blindly re-issued to a provider.
            conn.execute("UPDATE memory_l2_jobs SET state='failed',reason='worker_interrupted',updated_at=? WHERE state='running' AND lease_until<?", (time.time(), time.time()))
            rows = conn.execute("SELECT * FROM memory_l2_jobs WHERE state='pending' ORDER BY created_at,task_id LIMIT ?", (limit,)).fetchall()
            conn.commit()
            return [dict(r) for r in rows]
        finally: conn.close()

    def tracked_ids(self):
        conn = self.authority._connect()
        try: return [r[0] for r in conn.execute('SELECT memory_id FROM memory_l2_heads')]
        finally: conn.close()

    def retire_missing(self, task_id):
        conn = self.authority._connect()
        try:
            conn.execute("UPDATE memory_l2_jobs SET state='stale',reason='memory_missing',updated_at=? WHERE task_id=? AND state='pending'", (time.time(), task_id))
            conn.commit()
        finally: conn.close()

    def claim(self, task_id, dependency):
        conn = self.authority._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            job = conn.execute('SELECT * FROM memory_l2_jobs WHERE task_id=?', (task_id,)).fetchone()
            if not job or job['state'] != 'pending': return None
            if conn.execute("SELECT 1 FROM memory_l2_jobs WHERE memory_id=? AND state='running' AND lease_until>=? LIMIT 1",
                            (job['memory_id'], time.time())).fetchone(): return None
            head = conn.execute('SELECT * FROM memory_l2_heads WHERE memory_id=?', (job['memory_id'],)).fetchone()
            completed = conn.execute('SELECT dependency_json,created_at FROM memory_l2_revisions WHERE memory_id=? ORDER BY presentation_revision DESC LIMIT 1', (job['memory_id'],)).fetchone()
            if completed and json.loads(completed['dependency_json']) == dependency and completed['created_at'] >= job['created_at']:
                conn.execute("UPDATE memory_l2_jobs SET state='ready',reason='already_current',updated_at=? WHERE task_id=?", (time.time(), task_id))
                conn.commit(); return None
            ready = conn.execute('SELECT dependency_json,sync_state FROM memory_l2_revisions WHERE memory_id=? AND presentation_revision=?',
                                 (job['memory_id'], head['presentation_revision'])).fetchone()
            if ready and json.loads(ready['dependency_json']) == dependency and ready['sync_state'] != 'pending_l1' and (job['reason'] != 'owner_requested' or head['presentation_revision'] != job['base_presentation_revision']):
                conn.execute("UPDATE memory_l2_jobs SET state='ready',reason='already_current',updated_at=? WHERE task_id=?", (time.time(), task_id))
                conn.commit(); return None
            if (job['first_pass_revision'] != dependency['first_pass_revision'] or not dependency.get('l1_content_sha256')
                    or dependency['memory_state'] != 'active' or dependency['recall_policy'] == 'disabled'):
                conn.execute("UPDATE memory_l2_jobs SET state='stale',reason='dependency_changed',updated_at=? WHERE task_id=?", (time.time(), task_id))
                conn.commit(); return None
            token = uuid.uuid4().hex
            conn.execute("UPDATE memory_l2_jobs SET state='running',dependency_json=?,base_presentation_revision=?,lease_token=?,lease_until=?,updated_at=? WHERE task_id=?",
                         (canonical_json(dependency), head['presentation_revision'], token, time.time()+1800, time.time(), task_id))
            conn.commit()
            return {**dict(job), 'dependency': dependency, 'base_presentation_revision': head['presentation_revision'], 'lease_token': token,
                    'owner_protected': bool(head['owner_protected'])}
        except Exception:
            conn.rollback(); raise
        finally: conn.close()

    def finish(self, job, state, reason=''):
        conn = self.authority._connect()
        try:
            conn.execute("UPDATE memory_l2_jobs SET state=?,reason=?,updated_at=? WHERE task_id=? AND state='running' AND lease_token=?",
                         (state, reason, time.time(), job['task_id'], job['lease_token']))
            conn.commit()
        finally: conn.close()

    def save(self, memory_id, document, dependency, *, request_id, expected_revision, actor, change_kind='wording', job=None,
             request_fingerprint=None, edit_note='', confirm_sync_proposals=()):
        fingerprint = request_fingerprint or semantic_sha256({'document':document, 'dependency':dependency, 'expected_revision':expected_revision,
                                       'actor':actor, 'change_kind':change_kind})
        conn = self.authority._connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            prior = conn.execute('SELECT * FROM memory_l2_revisions WHERE memory_id=? AND request_id=?', (memory_id, request_id)).fetchone()
            if prior:
                if prior['request_sha256'] != fingerprint: raise IdempotencyConflict('L2 request changed')
                return {'presentation_revision': prior['presentation_revision'], 'sync_state':prior['sync_state'], 'replayed': True}
            current = self.dependency(conn, memory_id, {k:v for k,v in dependency.items() if k not in
                {'memory_id','first_pass_revision','body_hash','l1_content_sha256','l1_semantic_sha256','memory_state','recall_policy','rings','parent_ref','parent_current'}})
            if current != dependency: raise RevisionConflict('L1 changed during L2 generation')
            head = conn.execute('SELECT * FROM memory_l2_heads WHERE memory_id=?', (memory_id,)).fetchone()
            current_revision = head['presentation_revision'] if head else 0
            if expected_revision != current_revision: raise RevisionConflict('L2 draft is stale')
            if job:
                lease = conn.execute('SELECT state,lease_token FROM memory_l2_jobs WHERE task_id=?', (job['task_id'],)).fetchone()
                if not lease or lease['state'] != 'running' or lease['lease_token'] != job['lease_token']:
                    raise RevisionConflict('L2 job lease changed')
            candidate = bool(job and head and head['owner_protected'])
            revision = conn.execute('SELECT COALESCE(MAX(presentation_revision),0)+1 FROM memory_l2_revisions WHERE memory_id=?', (memory_id,)).fetchone()[0]
            pending = {r[0] for r in conn.execute("SELECT presentation_revision FROM memory_l2_sync_proposals WHERE memory_id=? AND state='pending'", (memory_id,))}
            confirmed = set(confirm_sync_proposals)
            if confirmed - pending: raise L1OwnerError('sync_proposal_not_pending', status=409, field_path='confirm_sync_proposals')
            sync = 'pending_l1' if actor == 'owner' and (change_kind == 'facts' or pending - confirmed) else 'not_required'
            before = conn.execute('SELECT document_json,units_state FROM memory_l2_revisions WHERE memory_id=? AND presentation_revision=?', (memory_id, current_revision)).fetchone()
            units_state = 'current'
            if actor == 'owner' and before:
                previous = json.loads(before['document_json'])
                if all(previous[key] == document[key] for key in ('coherent_sections','optional_use_variants')):
                    units_state = ('stale_after_owner_text_edit' if previous['full_injection_text'] != document['full_injection_text']
                                   and any(document[key] for key in ('coherent_sections','optional_use_variants')) else before['units_state'])
            conn.execute('''INSERT INTO memory_l2_revisions
                (memory_id,presentation_revision,first_pass_revision,dependency_json,document_json,actor,created_at,request_id,request_sha256,sync_state,edit_note,publication,units_state)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                         (memory_id, revision, dependency['first_pass_revision'], canonical_json(dependency), canonical_json(document), actor, time.time(), request_id, fingerprint, sync, edit_note, 'candidate' if candidate else 'active', units_state))
            if not candidate:
                conn.execute('INSERT INTO memory_l2_heads VALUES(?,?,?) ON CONFLICT(memory_id) DO UPDATE SET presentation_revision=excluded.presentation_revision,owner_protected=excluded.owner_protected',
                             (memory_id, revision, int(actor == 'owner')))
            if actor == 'owner' and change_kind == 'facts':
                proposal = {'kind':'l2_fact_sync', 'l1_target':{'memory_id':memory_id,'revision':dependency['first_pass_revision']},
                            'before_text':json.loads(before[0])['full_injection_text'] if before else '',
                            'proposed_text':document['full_injection_text'], 'edit_note':edit_note}
                conn.execute('INSERT INTO memory_l2_sync_proposals(memory_id,presentation_revision,first_pass_revision,state,proposal_json) VALUES(?,?,?,?,?)',
                             (memory_id, revision, dependency['first_pass_revision'], 'pending', canonical_json(proposal)))
            if actor == 'owner' and confirmed:
                conn.execute("""UPDATE memory_l2_sync_proposals SET state='owner_confirmed_after_l1_revision',
                    proposal_json=json_set(proposal_json,'$.synchronized_l1_revision',?,'$.confirmed_presentation_revision',?)
                    WHERE memory_id=? AND state='pending' AND presentation_revision IN (""" + ','.join('?' for _ in confirmed) + ')',
                             (dependency['first_pass_revision'], revision, memory_id, *sorted(confirmed)))
            if job:
                state = 'pending_review' if candidate or document.get('review', {}).get('required', True) else 'ready'
                reason = 'owner_l2_preserved' if candidate else 'l2_semantic_review_required' if state == 'pending_review' else ''
                conn.execute("UPDATE memory_l2_jobs SET state=?,reason=?,updated_at=? WHERE task_id=?", (state, reason, time.time(), job['task_id']))
            conn.commit()
            return {'presentation_revision': revision, 'sync_state': sync, 'replayed': False,
                    'state': 'pending_sync' if sync == 'pending_l1' else
                    'pending_review' if candidate or (actor == 'model' and document.get('review', {}).get('required', True)) else 'ready'}
        except Exception:
            conn.rollback(); raise
        finally: conn.close()
