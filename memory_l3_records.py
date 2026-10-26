"""L3 evolution views and versioned supplementary notes on Memory Authority.

Notes are not new facts, do not carry fabricated L0 bindings, and do not start
another memory writer. Adopted semantic changes still use the existing L1 path.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json

from memory_authority import MemoryNotFound, RevisionConflict, IdempotencyConflict
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_service import L1OwnerError


def decoded_ring(row):
    result = dict(row)
    for key in ('metadata', 'source_refs'):
        if key not in result:
            result[key] = json.loads(result.pop(key + '_json', '{}' if key == 'metadata' else '[]'))
    return result


def relevant_rings(rings, document):
    """Only sourced semantic additions belong in the L2 writing dependency.

    Keep legacy sourced additions without rewriting history. Entry/author names
    such as dashboard are not L0 evidence. New notes remain fully visible in L3.
    """
    bindings = {row['binding_id'] for row in (document or {}).get('evidence_bindings', [])}
    result = []
    for value in rings:
        row = decoded_ring(value)
        role = row['metadata'].get('l3_role')
        refs = row['source_refs']
        if (row.get('state') != 'active' or role in {'note', 'proposal', 'usage'}
                or not refs or not set(refs) <= bindings):
            continue
        if role == 'adopted_update' and not row['metadata'].get('adoption_receipt'):
            continue
        result.append(row)
    return result


def ring_dependency_rows(conn, memory_id, document):
    rows = conn.execute('SELECT * FROM memory_rings WHERE memory_id=? ORDER BY ring_id', (memory_id,)).fetchall()
    return [[row['ring_id'], row['content_sha256'], row['state']]
            for row in relevant_rings(rows, document)]


def note_projection(row):
    row = decoded_ring(row)
    meta = row['metadata']
    return {'event_id': row['ring_id'], 'kind': 'followup_proposal' if meta.get('l3_role') == 'proposal' else 'note', 'state': row['state'],
        'created_at': row['created_at'], 'author': row['created_by'], 'content': row['content'],
        'source_refs': row['source_refs'], 'details': {
            'content_sha256': row['content_sha256'], 'source_entry': meta.get('source_entry', meta.get('source')),
            'l3_role': meta.get('l3_role', 'legacy_supplement'), 'original_kind': row['kind'],
            'replaces_ring_id': meta.get('replaces_ring_id'), 'adoption_receipt': meta.get('adoption_receipt')}}


class MemoryL3Records:
    def __init__(self, authority, commit_service):
        self.authority, self.commit = authority, commit_service

    def _ring(self, memory_id, ring_id):
        with closing(self.authority._connect()) as conn:
            row = conn.execute('SELECT * FROM memory_rings WHERE memory_id=? AND ring_id=?',
                               (memory_id, ring_id)).fetchone()
        if row is None:
            raise MemoryNotFound(ring_id)
        return decoded_ring(row)

    @staticmethod
    def _validate(payload, *, retract=False):
        required = {'expected_revision', 'request_id'} | ({'expected_content_sha256'} if retract else {'content'})
        optional = set() if retract else {'replaces_ring_id', 'expected_content_sha256'}
        if (not isinstance(payload, dict) or not required <= set(payload) or set(payload) - required - optional
                or type(payload['expected_revision']) is not int or payload['expected_revision'] < 1
                or not isinstance(payload['request_id'], str) or not 1 <= len(payload['request_id'].strip()) <= 128
                or (not retract and (not isinstance(payload['content'], str) or not payload['content'].strip()))):
            raise L1OwnerError('invalid_l3_note_request')
        if retract or payload.get('replaces_ring_id'):
            value = payload.get('expected_content_sha256')
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise L1OwnerError('note_source_revision_required', field_path='expected_content_sha256')
        if payload.get('replaces_ring_id') and not isinstance(payload['replaces_ring_id'], str):
            raise L1OwnerError('invalid_ring_identity')

    def _reserve(self, memory_id, payload, kind, ring_id=None):
        key = 'l3-note-request:' + semantic_sha256({'memory_id': memory_id, 'kind': kind, 'request_id': payload['request_id']})
        fingerprint = semantic_sha256({'memory_id': memory_id, 'kind': kind, 'ring_id': ring_id, 'payload': payload})
        with closing(self.authority._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            prior = conn.execute('SELECT value FROM authority_meta WHERE key=?', (key,)).fetchone()
            if prior:
                row = json.loads(prior[0])
                if row['fingerprint'] != fingerprint:
                    raise IdempotencyConflict('L3 note request content changed')
                conn.commit()
                return key, row
            memory = conn.execute('SELECT * FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
            if memory is None:
                raise MemoryNotFound(memory_id)
            if memory['active_revision'] != payload['expected_revision']:
                raise RevisionConflict('L3 note target changed')
            if memory['state'] == 'tombstoned':
                raise L1OwnerError('memory_withdrawn', status=409)
            row = {'fingerprint': fingerprint, 'state': 'pending', 'request_id': payload['request_id']}
            conn.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (key, canonical_json(row)))
            conn.commit()
            return key, row

    def _done(self, key, row, result):
        self.authority.set_meta(key, {**row, 'state': 'completed', 'receipt': result})
        return result

    async def save_note(self, memory_id, payload):
        self._validate(payload)
        key, row = self._reserve(memory_id, payload, 'save')
        if row['state'] == 'completed':
            return {**deepcopy(row['receipt']), 'replayed': True}
        original_commit = self.authority.get_commit_by_idempotency(key + ':append')
        if original_commit is None:
            current = self.authority.get_memory(memory_id)
            if not current or current['active_revision'] != payload['expected_revision'] or current['state'] == 'tombstoned':
                raise RevisionConflict('L3 note target changed before commit')
        replacement = payload.get('replaces_ring_id')
        if replacement:
            original = self._ring(memory_id, replacement)
            if (original['content_sha256'] != payload['expected_content_sha256']
                    or (original['state'] != 'active' and original_commit is None)):
                raise RevisionConflict('original L3 supplement changed')
        metadata = {'l3_role': 'note', 'source_entry': 'owner_memory_details', 'touch': False,
            'target_revision': payload['expected_revision'], 'request_fingerprint': row['fingerprint']}
        if replacement:
            metadata['replaces_ring_id'] = replacement
        receipt = await self.commit.append_ring(memory_id=memory_id, content=payload['content'], kind='comment',
            source_refs=[], idempotency_key=key + ':append', actor='owner', metadata=metadata)
        if replacement:
            await self.commit.retract_ring(memory_id=memory_id, ring_id=replacement,
                idempotency_key=key + ':replace', actor='owner')
        result = {'status': 'ok', 'memory_id': memory_id, 'ring_id': receipt['ring_id'],
            'request_id': payload['request_id'], 'committed': True, 'replayed': False,
            'note': note_projection(self._ring(memory_id, receipt['ring_id']))}
        if replacement:
            result['replaced_ring_id'] = replacement
        return self._done(key, row, result)

    async def retract_note(self, memory_id, ring_id, payload):
        self._validate(payload, retract=True)
        key, row = self._reserve(memory_id, payload, 'retract', ring_id)
        if row['state'] == 'completed':
            return {**deepcopy(row['receipt']), 'replayed': True}
        original = self._ring(memory_id, ring_id)
        if original['content_sha256'] != payload['expected_content_sha256']:
            raise RevisionConflict('original L3 supplement changed')
        await self.commit.retract_ring(memory_id=memory_id, ring_id=ring_id,
            idempotency_key=key + ':retract', actor='owner')
        return self._done(key, row, {'status': 'ok', 'memory_id': memory_id, 'ring_id': ring_id,
            'request_id': payload['request_id'], 'committed': True, 'replayed': False,
            'note': note_projection(self._ring(memory_id, ring_id))})

    def timeline(self, memory_id, *, offset=0, limit=50):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise L1OwnerError('invalid_evolution_pagination')
        take = offset + limit
        with closing(self.authority._connect()) as conn:
            conn.execute('BEGIN')
            memory = conn.execute('SELECT * FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
            if memory is None:
                raise MemoryNotFound(memory_id)
            revisions = conn.execute('SELECT * FROM memory_revisions WHERE memory_id=? ORDER BY revision DESC LIMIT ?',
                                     (memory_id, take)).fetchall()
            presentations = conn.execute('SELECT * FROM memory_l2_revisions WHERE memory_id=? ORDER BY presentation_revision DESC LIMIT ?',
                                         (memory_id, take)).fetchall()
            rings = conn.execute('SELECT * FROM memory_rings WHERE memory_id=? ORDER BY created_at DESC,ring_id DESC LIMIT ?',
                                 (memory_id, take)).fetchall()
            proposals = conn.execute('SELECT * FROM memory_l2_sync_proposals WHERE memory_id=?', (memory_id,)).fetchall()
            followups = conn.execute("SELECT value FROM authority_meta WHERE key LIKE 'memory_l3_followup:%' "
                "AND json_extract(value,'$.memory_id')=? ORDER BY json_extract(value,'$.created_at') DESC LIMIT ?",
                (memory_id, take)).fetchall()
            total = sum(conn.execute('SELECT count(*) FROM ' + table + ' WHERE memory_id=?', (memory_id,)).fetchone()[0]
                        for table in ('memory_revisions', 'memory_l2_revisions', 'memory_rings'))
            total += conn.execute("SELECT count(*) FROM authority_meta WHERE key LIKE 'memory_l3_followup:%' "
                "AND json_extract(value,'$.memory_id')=?", (memory_id,)).fetchone()[0]
        sync = {r['presentation_revision']: {'state': r['state'], 'proposal': json.loads(r['proposal_json'])} for r in proposals}
        items = []
        for row in revisions:
            meta = json.loads(row['metadata_json'])
            items.append({'event_id': 'l1:' + str(row['revision']), 'kind': 'l1_revision',
                'revision': row['revision'], 'state': 'current' if row['revision'] == memory['active_revision'] else 'history',
                'created_at': row['created_at'], 'author': row['created_by'], 'source_refs': json.loads(row['source_refs_json']),
                'details': {'decision_source': row['decision_source'], 'operation_id': row['operation_id'],
                    'edit_note': (meta.get('memory_l1_owner_request') or {}).get('edit_note', '')}})
        for row in presentations:
            items.append({'event_id': 'l2:' + str(row['presentation_revision']), 'kind': 'l2_revision',
                'presentation_revision': row['presentation_revision'], 'revision': row['first_pass_revision'],
                'state': row['sync_state'], 'created_at': datetime.fromtimestamp(row['created_at'], timezone.utc).isoformat(),
                'author': row['actor'], 'source_refs': [], 'details': {'edit_note': row['edit_note'],
                    'publication': row['publication'], 'sync_proposal': sync.get(row['presentation_revision'])}})
        items.extend(note_projection(row) for row in rings)
        for value in followups:
            row = json.loads(value['value'])
            candidates = [self.authority.get_candidate(cid) for cid in row['candidate_ids']]
            update = row['envelope']['updates'][0]
            receipt = row.get('receipt')
            if not receipt and candidates and candidates[0] and candidates[0]['status'] == 'committed':
                adopted = self.authority.get_commit_by_idempotency('program-l1:followup:' + memory_id + ':' + row['task_id'])
                if adopted and adopted['status'] == 'committed':
                    receipt = {'operation_id': adopted['operation_id'], 'result': adopted['result']}
            source_refs = ['canonical_event:' + s['event_id'] + '@' + s['version_id']
                for s in row['envelope']['sources'] if s['role'] in {span['role'] for span in update['evidence_spans']}]
            items.append({'event_id': row['task_id'], 'kind': 'followup_proposal',
                'state': (candidates[0]['status'] if candidates and candidates[0] else row['phase']),
                'revision': row['revision'], 'created_at': row['created_at'], 'author': 'memory_followup',
                'content': row.get('draft_content') or (candidates[0]['proposal']['proposed_body'] if candidates and candidates[0] else update['content']), 'source_refs': source_refs, 'details': {
                    'task_id': row['task_id'], 'task_revision': row['revision'], 'phase': row['phase'],
                    'issue': row.get('issue'), 'candidate_ids': row['candidate_ids'], 'receipt': receipt,
                    'allowed_actions': ['retry'] if row['phase'] == 'failed' else [],
                    'kind': update['kind'], 'long_term_value': update['long_term_value']}})
        def order(item):
            return (datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')).timestamp(), item['event_id'])
        items.sort(key=order, reverse=True)
        return {'status': 'ok', 'memory_id': memory_id, 'current_revision': memory['active_revision'],
            'total': total, 'offset': offset, 'limit': limit, 'items': items[offset:offset+limit]}
