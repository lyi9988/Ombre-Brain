"""Identifier-only source preparation jobs in the existing Authority metadata.

These records are task receipts, never adopted facts or another memory store.
The original Proposal/Policy/Commit still decides and commits every result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

from memory_authority import IdempotencyConflict
from memory_l1_contract import canonical_json
from memory_l1_service import L1OwnerError

PREFIX = 'memory_l1_source_task:'
_ID = re.compile(r'^[A-Za-z0-9_.:-]{1,200}$')


def _stamp(): return datetime.now(timezone.utc).isoformat(timespec='seconds')


class MemoryL1SourceTasks:
    def __init__(self, authority): self.authority = authority

    def get(self, task_id):
        db = self.authority._connect()
        try:
            row = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + task_id,)).fetchone()
            return json.loads(row['value']) if row else None
        finally: db.close()

    async def retain_action(self, task_id, reader, *, profile_id, captured_source=None):
        """A program read plus committed L0 copy; queueing alone proves neither."""
        from memory_l0_sources import MemoryL0Sources
        from memory_l1_ingestion import bind_sources
        task = self.get(task_id)
        if task is None: raise L1OwnerError('task_not_found', status=404, field_path='task_id')
        source = captured_source if captured_source is not None else await reader.capture(task['action_id'], task['artifact_id'])
        if (source.get('state') != 'available' or source.get('eligible_source') is not True
                or source.get('source_id') != task['action_id']
                or source.get('source_revision') != task['artifact_id']
                or not isinstance(source.get('text'), str) or not source['text'].strip()):
            return {'retained': False, 'state': source.get('state', 'unavailable'),
                'reason': source.get('reason', 'action_source_not_available')}
        raw = {'namespace': 'action_receipt', 'system': 'aizizhu',
            'conversation_id': reader.options.get('conversation_id', 'conv-jiajia-main'),
            'profile_id': profile_id, 'event_id': task['action_id'], 'version_id': task['artifact_id'],
            'role': 'tool', 'text': source['text'], 'recorded_at': source.get('recorded_at')}
        snapshots, _ = bind_sources([raw])
        store = MemoryL0Sources(self.authority)
        store.retain_copy(snapshots[0])
        if store.copy(snapshots[0]) != snapshots[0]:
            raise L1OwnerError('retained_source_readback_failed', status=503, field_path='source')
        receipt = {'schema': 'action-source-retention-v1', 'retained': True,
            'source_namespace': 'snapshot.aiz.action_receipt', 'source_id': task['action_id'],
            'source_revision': task['artifact_id'],
            'content_sha256': hashlib.sha256(source['text'].encode('utf-8')).hexdigest(),
            'l0_source_id': snapshots[0]['source_id'], 'l0_source_revision': snapshots[0]['source_revision'],
            'adopted': False}
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            current = json.loads(db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + task_id,)).fetchone()['value'])
            prior = current.get('retention')
            if prior and prior != receipt:
                raise IdempotencyConflict('retained action source changed')
            current['retention'] = receipt
            db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(current), PREFIX + task_id))
            db.commit()
        except Exception:
            db.rollback(); raise
        finally: db.close()
        return receipt

    def enqueue_action(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'action_id', 'artifact_id', 'request_id'}:
            raise L1OwnerError('invalid_action_source_request', field_path='source_task')
        for key, value in payload.items():
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise L1OwnerError('invalid_source_identity', field_path='source_task.' + key)
        source = {'action_id': payload['action_id'], 'artifact_id': payload['artifact_id']}
        digest = hashlib.sha256(canonical_json(source).encode()).hexdigest()
        task_id = 'action_source_' + digest[:40]
        key = PREFIX + task_id
        request_key = PREFIX + 'request:' + hashlib.sha256(payload['request_id'].encode()).hexdigest()
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            prior_request = db.execute('SELECT value FROM authority_meta WHERE key=?', (request_key,)).fetchone()
            if prior_request and json.loads(prior_request['value']) != digest:
                raise IdempotencyConflict('action source request_id payload changed')
            prior = db.execute('SELECT value FROM authority_meta WHERE key=?', (key,)).fetchone()
            status = 'exists' if prior else 'queued'
            if not prior:
                value = {**source, 'task_id': task_id, 'request_id': payload['request_id'],
                    'phase': 'queued', 'created_at': _stamp(), 'updated_at': _stamp(), 'candidate_ids': []}
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (key, canonical_json(value)))
            if not prior_request:
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (request_key, canonical_json(digest)))
            db.commit()
            return {'status': status, 'receipt_status': 'source_queued', 'adopted': False,
                'task_id': task_id, 'source_id': payload['action_id'], 'source_revision': payload['artifact_id']}
        except Exception:
            db.rollback()
            raise
        finally: db.close()

    def list(self, *, limit=100, phase=None):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise L1OwnerError('invalid_task_page_limit', field_path='limit')
        db = self.authority._connect()
        try:
            if phase is not None and phase not in ('queued', 'generating', 'prepared', 'failed'):
                raise L1OwnerError('invalid_task_phase', field_path='phase')
            where = ' AND json_extract(value,\'$.phase\')=?' if phase else ''
            params = [PREFIX + 'action_source_%', *([phase] if phase else []), limit]
            rows = db.execute('SELECT value FROM authority_meta WHERE key LIKE ?' + where
                + ' ORDER BY json_extract(value,\'$.created_at\'),key LIMIT ?', params).fetchall()
            return [json.loads(row['value']) for row in rows]
        finally: db.close()

    def page_with_chat(self, *, limit=100, offset=0):
        """One ordered SQL page across the two existing task families."""
        if type(limit) is not int or not 1 <= limit <= 200 or type(offset) is not int or offset < 0:
            raise L1OwnerError('invalid_task_page_limit', field_path='limit')
        from memory_chat_preparation import PREFIX as CHAT_PREFIX, ChatMemoryPreparation
        from memory_browse_preparation import PREFIX as BROWSE_PREFIX, BrowseMemoryPreparation
        action_prefix = PREFIX + 'action_source_'
        where = '((key>=? AND key<?) OR (key>=? AND key<?) OR (key>=? AND key<?))'
        params = (action_prefix, action_prefix + '\uffff', CHAT_PREFIX, CHAT_PREFIX + '\uffff', BROWSE_PREFIX, BROWSE_PREFIX + '\uffff')
        db = self.authority._connect()
        try:
            total = db.execute('SELECT count(*) FROM authority_meta WHERE ' + where, params).fetchone()[0]
            rows = db.execute('SELECT key,value FROM authority_meta WHERE ' + where
                + " ORDER BY json_extract(value,'$.created_at') DESC,json_extract(value,'$.task_id') DESC,key DESC LIMIT ? OFFSET ?",
                (*params, limit, offset)).fetchall()
            values = [ChatMemoryPreparation.public(json.loads(row['value'])) if row['key'].startswith(CHAT_PREFIX)
                      else BrowseMemoryPreparation.public(json.loads(row['value'])) if row['key'].startswith(BROWSE_PREFIX)
                      else json.loads(row['value']) for row in rows]
        finally:
            db.close()
        next_offset = offset + len(values)
        return {'tasks': values, 'limit': limit, 'offset': offset, 'total': total,
                'has_more': next_offset < total, 'next_offset': next_offset if next_offset < total else None}

    def runnable(self, *, limit=100):
        """Select waiting work in SQL, independent of owner history pages."""
        if type(limit) is not int or not 1 <= limit <= 200:
            raise L1OwnerError('invalid_task_page_limit', field_path='limit')
        db = self.authority._connect()
        try:
            rows = db.execute("SELECT value FROM authority_meta WHERE key LIKE ? AND "
                "(json_extract(value,'$.phase')='queued' OR (json_extract(value,'$.phase')='prepared' "
                "AND json_array_length(value,'$.remaining_candidate_ids')>0)) "
                "ORDER BY CASE json_extract(value,'$.phase') WHEN 'prepared' THEN 0 ELSE 1 END,"
                "json_extract(value,'$.created_at'),key LIMIT ?", (PREFIX + 'action_source_%', limit)).fetchall()
            return [json.loads(row['value']) for row in rows]
        finally: db.close()

    def transition(self, task_id, expected_phase, phase, **result):
        if phase not in ('queued', 'generating', 'prepared', 'failed'):
            raise L1OwnerError('invalid_task_phase', field_path='phase')
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + task_id,)).fetchone()
            if row is None: raise L1OwnerError('task_not_found', status=404, field_path='task_id')
            current = json.loads(row['value'])
            if current['phase'] != expected_phase:
                db.rollback()
                return None
            current.update(result, phase=phase, updated_at=_stamp())
            db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(current), PREFIX + task_id))
            db.commit()
            return current
        except Exception:
            db.rollback()
            raise
        finally: db.close()
