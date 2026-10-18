"""Owner-readable model wire evidence in the existing Authority metadata store.

Only explicitly supplied messages, safe completion options and visible output
are retained. This is call evidence, never another memory or source authority.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import uuid

from memory_l1_contract import canonical_json
from memory_l1_service import L1OwnerError

PREFIX = 'memory_model_io:'
STAGES = ('chat_l1', 'presentation_l2')


def _stamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def completion_parameters(route, temperature):
    """Use the same option builder as the SDK boundary, never retain a client."""
    from memory_task_routes import completion_options
    options = completion_options({**route, 'thinking_mode': route.get('thinking_mode', ''),
                                  'reasoning_effort': route.get('reasoning_effort', '')},
                                 temperature=temperature)
    return {'model': route['model'], **options}


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def response_usage(response):
    """Preserve numeric provider accounting, including reasoning/cache details."""
    usage = _field(response, 'usage')
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = vars(usage) if hasattr(usage, '__dict__') else {}
    def numeric(value):
        if type(value) in (int, float):
            return value
        if not isinstance(value, dict) and hasattr(value, '__dict__'):
            value = vars(value)
        if isinstance(value, dict):
            return {k: item for k, v in value.items() if (item := numeric(v)) is not None}
        return None
    result = numeric(usage) or {}
    details = result.get('completion_tokens_details', {})
    if 'reasoning_tokens' in details:
        result['reasoning_tokens'] = details['reasoning_tokens']
    return result


class MemoryModelIO:
    def __init__(self, authority):
        self.authority = authority

    @staticmethod
    def _prefix(memory_id):
        return PREFIX + hashlib.sha256(memory_id.encode('utf-8')).hexdigest() + ':'

    def start(self, memory_id, *, stage, task_id, attempt, messages, parameters,
              generation=None, draft_revision=None, source_revisions=None, attempt_id=None):
        if stage not in STAGES:
            raise ValueError('unsupported model IO stage')
        started = _stamp()
        io_id = self._prefix(memory_id) + started + ':' + uuid.uuid4().hex
        # This whitelist accepts SDK options only, not route/client/headers/key.
        allowed = {'model', 'max_tokens', 'max_completion_tokens', 'temperature',
                   'reasoning_effort', 'thinking', 'effort'}
        safe = {k: copy.deepcopy(v) for k, v in parameters.items() if k in allowed}
        if isinstance(parameters.get('extra_body'), dict):
            safe['extra_body'] = {k: copy.deepcopy(v) for k, v in parameters['extra_body'].items()
                                  if k in ('thinking', 'reasoning_effort', 'effort')}
        row = {'io_id': io_id, 'memory_id': memory_id, 'stage': stage, 'task_id': task_id,
               'attempt': attempt, 'attempt_id': attempt_id or uuid.uuid4().hex,
               'draft_revision': draft_revision, 'source_revisions': copy.deepcopy(source_revisions or []),
               'generation': copy.deepcopy(generation or {}), 'messages': copy.deepcopy(messages),
               'completion_parameters': safe, 'started_at': started, 'ended_at': None,
               'status': 'running', 'usage': {}, 'finish_reason': None, 'raw_output': None,
               'validation': {'status': 'pending'}}
        self.authority.set_meta(io_id, row)
        return io_id

    def _update(self, io_id, **changes):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            found = db.execute('SELECT value FROM authority_meta WHERE key=?', (io_id,)).fetchone()
            if not found:
                raise ValueError('model IO record missing')
            row = json.loads(found['value'])
            row.update(copy.deepcopy(changes))
            db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(row), io_id))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def returned(self, io_id, *, response, raw_output, finish_reason):
        wire = getattr(response, 'native_request', None)
        actual = {}
        if isinstance(wire, dict):
            allowed = {'model', 'input', 'messages', 'max_output_tokens', 'max_tokens',
                       'temperature', 'reasoning', 'text', 'response_format', 'reasoning_effort', 'extra_body'}
            request = {k: copy.deepcopy(v) for k, v in wire.items() if k in allowed}
            actual = {'native_request': request,
                      'messages': request.get('input', request.get('messages', [])),
                      'completion_parameters': {k: v for k, v in request.items() if k not in ('input', 'messages')}}
        self._update(io_id, status='validating', ended_at=_stamp(), raw_output=raw_output,
                     finish_reason=finish_reason, usage=response_usage(response), **actual)

    def requested(self, io_id, request, *, timeout_seconds):
        """Capture the real native wire before dispatch, including on timeout."""
        allowed = {'model', 'input', 'messages', 'max_output_tokens', 'max_tokens',
                   'temperature', 'reasoning', 'text', 'response_format', 'reasoning_effort', 'extra_body'}
        wire = {key: copy.deepcopy(value) for key, value in request.items() if key in allowed}
        self._update(io_id, native_request=wire,
            messages=wire.get('input', wire.get('messages', [])),
            completion_parameters={key: value for key, value in wire.items() if key not in ('input', 'messages')},
            transport={'timeout_seconds': timeout_seconds, 'max_retries': 0})

    def finish(self, io_id, *, error=None):
        changes = {'status': 'failed' if error else 'succeeded',
                   'validation': {'status': 'failed' if error else 'passed'}}
        if error:
            # Program-defined type/code/path; provider exception text may contain secrets.
            safe = {'code': getattr(error, 'code', type(error).__name__),
                    'field_path': getattr(error, 'field_path', getattr(error, 'path', ''))}
            changes.update(error=safe, validation={'status': 'failed', 'issue': safe})
        # Do not clear visible output or response accounting on either outcome.
        row = self.authority.get_meta(io_id)
        if not row.get('ended_at'):
            changes['ended_at'] = _stamp()
        self._update(io_id, **changes)

    def list(self, memory_id, limit=20, offset=0):
        if type(limit) is not int or not 1 <= limit <= 200 or type(offset) is not int or offset < 0:
            raise L1OwnerError('invalid_model_io_page', field_path='query')
        prefix = self._prefix(memory_id)
        db = self.authority._connect()
        try:
            # Primary-key range avoids scanning other memories or metadata families.
            rows = db.execute('SELECT value FROM authority_meta WHERE key>=? AND key<? '
                              'ORDER BY key DESC LIMIT ? OFFSET ?',
                              (prefix, prefix + '\uffff', limit + 1, offset)).fetchall()
            exists = bool(rows) or bool(db.execute('SELECT 1 FROM authority_meta WHERE key>=? AND key<? LIMIT 1',
                                                   (prefix, prefix + '\uffff')).fetchone())
        finally:
            db.close()
        items = [json.loads(row['value']) for row in rows[:limit]]
        more = len(rows) > limit
        return {'status': 'ok', 'memory_id': memory_id, 'items': items, 'limit': limit,
                'offset': offset, 'has_more': more, 'next_offset': offset + len(items) if more else None,
                'capture_state': 'recorded' if exists else 'missing', 'supported_stages': list(STAGES)}
