"""Owner-readable model wire evidence in the existing Authority metadata store.

Only explicitly supplied messages, safe completion options and visible output
are retained. This is call evidence, never another memory or source authority.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid

from memory_l1_contract import canonical_json
from memory_l1_service import L1OwnerError
from memory_l1_v2_diagnostics import DEFAULT_MAX_BYTES, RETENTION_DAYS

PREFIX = 'memory_model_io:'
V2_PREFIX = 'memory_model_io_v2:'
STAGES = ('chat_l1', 'auto_l1', 'window_compression', 'browse_l1', 'action_l1',
          'presentation_l2', 'memory_l3_sync')
REQUEST_FIELDS = {'model', 'input', 'messages', 'max_output_tokens', 'max_tokens',
                  'temperature', 'reasoning', 'text', 'response_format', 'reasoning_effort', 'extra_body'}


def _stamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def _v2_cutoff():
    return (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat(timespec='microseconds')


def completion_parameters(route, temperature):
    """Use the same option builder as the SDK boundary, never retain a client."""
    from memory_task_routes import completion_options
    options = completion_options({**route, 'thinking_mode': route.get('thinking_mode', ''),
                                  'reasoning_effort': route.get('reasoning_effort', '')},
                                 temperature=temperature)
    return {'model': route['model'], **options}


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def model_error_details(error):
    """Keep structured diagnostics without copying exception text or provider payloads."""
    error_type = type(error).__name__
    code = getattr(error, 'code', None)
    if type(code) is not int and (type(code) is not str or not re.fullmatch(r'[A-Za-z0-9_]{1,160}', code)):
        code = error_type
    path = getattr(error, 'field_path', None) or getattr(error, 'path', None)
    schema_path = r'[A-Za-z_]\w*(?:\.\w+|\[[0-9]+\])*'
    patch_pointer = r'(?:/(?:[-\w.:]|~[01])+)+'
    if type(path) is not str or len(path) > 160 or not re.fullmatch(
            schema_path + '|' + patch_pointer, path, flags=re.ASCII):
        path = ''
    details = {'code': code, 'error_type': error_type, 'field_path': path}
    status = getattr(error, 'status_code', None)
    if status is None:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
    if type(status) is int and 100 <= status <= 599:
        details['http_status'] = status
        body = getattr(error, 'body', None)
        provider = body.get('error', body) if isinstance(body, dict) else {}
        if isinstance(provider, dict):
            provider_code = provider.get('code')
            if (type(provider_code) is str and re.fullmatch(r'[A-Z][A-Z0-9_]{2,79}', provider_code)
                    and provider_code != code):
                details['provider_code'] = provider_code
            trace = provider.get('traceId')
            if type(trace) is str and re.fullmatch(r'trace_[0-9a-fA-F-]{16,50}', trace):
                details['provider_trace_id'] = trace
    return details


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
    def _prefix(memory_id, *, v2=False):
        return (V2_PREFIX if v2 else PREFIX) + hashlib.sha256(memory_id.encode('utf-8')).hexdigest() + ':'

    def start(self, memory_id, *, stage, task_id, attempt, messages, parameters,
              generation=None, draft_revision=None, source_revisions=None, attempt_id=None, part=None,
              task_attempt=None):
        if stage not in STAGES:
            raise ValueError('unsupported model IO stage')
        started = _stamp()
        io_id = self._prefix(memory_id, v2=part is not None) + started + ':' + uuid.uuid4().hex
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
        if part is not None:
            row['part'] = part
        if task_attempt is not None:
            row['task_attempt'] = task_attempt
        self.authority.set_meta(io_id, row)
        return io_id

    @staticmethod
    def _rotate_v2(db):
        lower, upper = V2_PREFIX, V2_PREFIX + '\uffff'
        time_start = len(V2_PREFIX) + 66  # SQLite offsets are one-based.
        db.execute('DELETE FROM authority_meta WHERE key>=? AND key<? AND substr(key,?)<?',
                   (lower, upper, time_start, _v2_cutoff()))
        rows = db.execute(
            "SELECT key,length(CAST(value AS BLOB)),json_extract(value,'$.status') "
            "FROM authority_meta WHERE key>=? AND key<? ORDER BY substr(key,?)",
            (lower, upper, time_start)).fetchall()
        total = sum(row[1] for row in rows)
        expired = []
        for key, size, status in rows:
            if total <= DEFAULT_MAX_BYTES:
                break
            if status in ('succeeded', 'failed'):
                expired.append((key,))
                total -= size
        db.executemany('DELETE FROM authority_meta WHERE key=?', expired)

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
            if io_id.startswith(V2_PREFIX) and changes.get('status') in ('succeeded', 'failed'):
                self._rotate_v2(db)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def returned(self, io_id, *, response, raw_output, finish_reason):
        wire = getattr(response, 'native_request', None)
        actual = {}
        if isinstance(wire, dict) and not io_id.startswith(V2_PREFIX):
            request = {k: copy.deepcopy(v) for k, v in wire.items() if k in REQUEST_FIELDS}
            actual = {'native_request': request,
                      'messages': request.get('input', request.get('messages', [])),
                      'completion_parameters': {k: v for k, v in request.items() if k not in ('input', 'messages')}}
        self._update(io_id, status='validating', ended_at=_stamp(), raw_output=raw_output,
                     finish_reason=finish_reason, usage=response_usage(response), **actual)

    def requested(self, io_id, request, *, timeout_seconds):
        """Capture the real native wire before dispatch, including on timeout."""
        wire = {key: copy.deepcopy(value) for key, value in request.items() if key in REQUEST_FIELDS}
        transport = {'timeout_seconds': timeout_seconds, 'max_retries': 0}
        if io_id.startswith(V2_PREFIX):
            # The owner API derives messages and options from this one exact copy.
            self._update(io_id, native_request=wire, transport=transport)
            return
        self._update(io_id, native_request=wire,
            messages=wire.get('input', wire.get('messages', [])),
            completion_parameters={key: value for key, value in wire.items() if key not in ('input', 'messages')},
            transport=transport)

    def finish(self, io_id, *, error=None):
        row = self.authority.get_meta(io_id)
        failure = None
        changes = {'status': 'succeeded', 'validation': {'status': 'passed'}}
        if error is not None:
            failure = model_error_details(error)
            failure['failure_phase'] = 'response_processing' if row['status'] == 'validating' else 'request'
            changes.update(status='failed', error=failure, validation={'status': 'failed', 'issue': failure})
        # Do not clear visible output or response accounting on either outcome.
        if not row.get('ended_at'):
            changes['ended_at'] = _stamp()
        self._update(io_id, **changes)
        return failure

    def _generation_task(self, memory_id):
        from memory_l1_v2 import is_v2

        candidate = self.authority.get_candidate(memory_id)
        if candidate:
            metadata = candidate['proposal'].get('metadata') or {}
        else:
            memory = self.authority.get_memory(memory_id)
            if not memory or not memory['active_revision']:
                return None
            revision = self.authority.get_memory_revision(memory_id, memory['active_revision'])
            metadata = (revision or {}).get('metadata') or {}
        document = metadata.get('memory_l1')
        return (document.get('generation') or {}).get('task_id') if is_v2(document) else None

    def list(self, memory_id, limit=20, offset=0):
        if type(limit) is not int or not 1 <= limit <= 200 or type(offset) is not int or offset < 0:
            raise L1OwnerError('invalid_model_io_page', field_path='query')
        direct = self._prefix(memory_id)
        cutoff = _v2_cutoff()
        scopes = [(direct, direct + '\uffff', len(direct) + 1)]
        task_id = self._generation_task(memory_id)
        for identity in dict.fromkeys((memory_id, task_id)):
            if isinstance(identity, str) and identity:
                prefix = self._prefix(identity, v2=True)
                scopes.append((prefix + cutoff, prefix + '\uffff', len(prefix) + 1))
        condition = 'key>=? AND key<?'
        query = ' UNION ALL '.join(
            f'SELECT value, substr(key, ?) AS started FROM authority_meta WHERE {condition}' for _ in scopes)
        arguments = [value for lower, upper, offset_start in scopes for value in (offset_start, lower, upper)]
        db = self.authority._connect()
        try:
            rows = db.execute(query + ' ORDER BY started DESC LIMIT ? OFFSET ?',
                              (*arguments, limit + 1, offset)).fetchall()
            exists = bool(rows) or any(db.execute(
                f'SELECT 1 FROM authority_meta WHERE {condition} LIMIT 1', (lower, upper)).fetchone()
                for lower, upper, _ in scopes)
        finally:
            db.close()
        items = [json.loads(row['value']) for row in rows[:limit]]
        for item in items:
            wire = item.get('native_request')
            if item['io_id'].startswith(V2_PREFIX) and wire:
                item['messages'] = wire.get('input', wire.get('messages', []))
                item['completion_parameters'] = {k: v for k, v in wire.items() if k not in ('input', 'messages')}
        more = len(rows) > limit
        return {'status': 'ok', 'memory_id': memory_id, 'items': items, 'limit': limit,
                'offset': offset, 'has_more': more, 'next_offset': offset + len(items) if more else None,
                'capture_state': 'recorded' if exists else 'missing', 'supported_stages': list(STAGES)}
