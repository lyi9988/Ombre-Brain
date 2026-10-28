"""Durable evidence of Memory fragments on successful model wires.

This is a delivery/usage receipt, never a memory authority or proof that a
model understood a fragment. Canonical source IDs scope a whole logical turn.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

CARRIER_SPANS = ContextVar('memory_delivery_carrier_spans', default=None)


@contextmanager
def capture_carriers():
    rows = []
    token = CARRIER_SPANS.set(rows)
    try:
        yield rows
    finally:
        CARRIER_SPANS.reset(token)


def record_carrier(bucket, text, section):
    rows = CARRIER_SPANS.get()
    if rows is not None and bucket.get('id') and text.strip():
        rows.append({'bucket_id': bucket['id'], 'section': section, 'text': text,
            'legacy_body_sha256': sha(str(bucket.get('content') or ''))})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def digest(value):
    return sha(canonical(value))


def scope_from(query):
    if query.get('consumer_kind') == 'autonomy':
        from autonomy_memory import scope_from as autonomy_scope
        return autonomy_scope(query)
    scope = {key: query.get(key) for key in
        ('conversation_id', 'user_source_event_id', 'assistant_source_event_id')}
    if any(not isinstance(v, str) or not v.strip() or len(v) > 400 for v in scope.values()):
        raise ValueError('canonical_scope_required')
    return scope


def payload_texts(payload):
    system = payload.get('system')
    if isinstance(system, str):
        yield system
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get('text'), str):
                yield block['text']
    for message in payload.get('messages') or []:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str):
                    yield block['text']


def candidates(projection, records=None):
    """Freeze source versions/coverage alongside the actual rendered snippets."""
    records = records or {}
    traces = {row.get('bucket_id'): row for row in projection.get('presentation_trace') or []}
    result = []
    for span in [*(projection.get('rendered_memory_spans') or []), *(projection.get('carrier_memory_spans') or [])]:
        bid, text = span.get('bucket_id'), span.get('text')
        if not bid or not isinstance(text, str) or not text.strip():
            continue
        record = records.get(span.get('record_key') or bid) or {}
        trace = span.get('source_trace') or traces.get(bid) or {}
        document = record.get('document') or {}
        refs = [{key: row.get(key) for key in
                 ('binding_id', 'source_namespace', 'source_id', 'source_revision')}
                for row in document.get('evidence_bindings') or []]
        memory_id = trace.get('memory_id') or record.get('memory_id') or bid
        identities = sorted((row['source_namespace'], row['source_id']) for row in refs)
        group = digest(identities or [['memory', memory_id]])
        result.append({'memory_id': memory_id, 'bucket_id': bid,
            'l1_revision': (trace.get('l1_revision') or record.get('revision')) if document else None,
            'l2_revision': trace.get('presentation_revision'), 'source_group_id': 'source_' + group,
            'text': text, 'text_sha256': sha(text), 'source_refs': refs,
            'coverage': {'section': span.get('section'),
                'statement_ids': trace.get('injected_statement_ids') or
                    (trace.get('statement_selection') or {}).get('direct_ids') or [],
                'unit_ids': trace.get('selected_unit_ids') or [],
                'partial_presentation': bool(trace.get('partial_presentation')),
                'legacy_projection_body_sha256': span.get('legacy_body_sha256') if not document else None,
                'legacy_authority_revision': record.get('revision') if not document else None},
            'legacy_body_sha256': None if document else (record.get('body_sha256') or span.get('legacy_body_sha256'))})
    return result


def compiled_candidates(projection, frozen):
    """A generated block must contain the snippet; incoming history alone is not evidence.

    A clipped card has no complete semantic coverage, so it remains delivery
    evidence of exactly that prefix and is explicitly marked partial.
    """
    sources = {'direct': {'ombre.recalled_memory', 'ombre.memory_recall'},
        'targeted': {'ombre.targeted_memory_detail', 'ombre.memory_recall'},
        'diffused': {'ombre.diffused_memory', 'ombre.memory_recall'}, 'recent': {'ombre.recent_context'},
        'core': {'ombre.core_memory'}, 'portrait': {'ombre.portrait_memory'},
        'favorite': {'ombre.favorite_memory'}, 'date': {'ombre.date_recall'},
        'date_persona': {'ombre.date_persona_trace'}, 'dream': {'ombre.dream_context', 'ombre.dream_wrapper_prompt'}}
    result = []
    for row in frozen:
        text = row['text']
        for block in projection.get('compiled_memory_blocks') or []:
            if block.get('source_id') not in sources.get(row['coverage']['section'], set()):
                continue
            body = block.get('content') or ''
            offset = body.find(text)
            shown = text if offset >= 0 else ''
            if not shown:
                # Match a surviving prefix at the end of a compiler-trimmed block.
                start = body.find(text[:8])
                while start >= 0:
                    tail = body[start:]
                    if len(tail.strip()) >= 8 and text.startswith(tail):
                        shown = tail
                        break
                    start = body.find(text[:8], start + 1)
            if shown:
                value = copy.deepcopy(row)
                value['text'], value['text_sha256'] = shown, sha(shown)
                value['coverage']['wire_partial'] = shown != text
                if shown != text:
                    value['coverage']['statement_ids'] = []
                    value['coverage']['unit_ids'] = []
                value['_compiled_block'] = body
                result.append(value)
                break
    return result


def final_fragments(frozen, payload):
    texts = list(payload_texts(payload))
    result = []
    for candidate in frozen:
        block = candidate.get('_compiled_block')
        text = candidate['text']
        if not block or not any(block in wire and text in block for wire in texts):
            continue
        row = {k: copy.deepcopy(v) for k, v in candidate.items() if not k.startswith('_')}
        row['fragment_id'] = 'fragment_' + digest(row)[:32]
        if row not in result:
            result.append(row)
    return result


class MemoryDeliveryStore:
    def __init__(self, path):
        self.path = str(path)
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS memory_delivery_attempts (
                  attempt_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, trace_id TEXT NOT NULL,
                  scope_json TEXT NOT NULL, fragments_json TEXT NOT NULL,
                  successful INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                  request_json TEXT NOT NULL DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS memory_delivery_attempt_scope ON memory_delivery_attempts(scope_key);
                CREATE TABLE IF NOT EXISTS memory_delivery_turns (
                  receipt_id TEXT PRIMARY KEY, scope_key TEXT UNIQUE NOT NULL,
                  context_json TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_activation_uses (
                  use_id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                  result_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_activation_groups (
                  conversation_id TEXT PRIMARY KEY, receipt_id TEXT NOT NULL, groups_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_activation_group_receipts (
                  receipt_id TEXT NOT NULL, group_id TEXT NOT NULL, activation_id TEXT NOT NULL,
                  PRIMARY KEY(receipt_id,group_id));
                CREATE TABLE IF NOT EXISTS memory_activation_assessments (
                  receipt_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, result_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_autonomy_preparations (
                  recall_id TEXT PRIMARY KEY, scope_json TEXT NOT NULL,
                  fragments_json TEXT NOT NULL, messages_json TEXT NOT NULL);
            ''')
            if 'request_json' not in {row[1] for row in db.execute('PRAGMA table_info(memory_delivery_attempts)')}:
                db.execute("ALTER TABLE memory_delivery_attempts ADD COLUMN request_json TEXT NOT NULL DEFAULT '{}'")

    def _connect(self):
        class Connection(sqlite3.Connection):
            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    self.close()
        db = sqlite3.connect(self.path, timeout=10, factory=Connection)
        db.row_factory = sqlite3.Row
        return db

    def begin(self, scope, trace_id, fragments, *, request=None):
        scope = scope_from(scope)
        attempt = 'memory_attempt_' + uuid.uuid4().hex
        with self._connect() as db:
            db.execute('INSERT INTO memory_delivery_attempts VALUES(?,?,?,?,?,?,?,?)',
                (attempt, digest(scope), trace_id, canonical(scope), canonical(fragments), 0,
                 datetime.now(timezone.utc).isoformat(), canonical(request or {})))
        return attempt

    def prepare_autonomy(self, scope, fragments, messages):
        scope = scope_from(scope)
        if scope.get('consumer_kind') != 'autonomy': raise ValueError('autonomy_scope_required')
        recall_id = 'autonomy_recall_' + uuid.uuid4().hex
        with self._connect() as db:
            db.execute('INSERT INTO memory_autonomy_preparations VALUES(?,?,?,?)',
                (recall_id, canonical(scope), canonical(fragments), canonical(messages)))
        return recall_id

    def autonomy_preparation(self, recall_id):
        with self._connect() as db:
            row = db.execute('SELECT * FROM memory_autonomy_preparations WHERE recall_id=?', (recall_id,)).fetchone()
        return ({'scope': json.loads(row['scope_json']), 'fragments': json.loads(row['fragments_json']),
            'memory_messages': json.loads(row['messages_json'])} if row else None)

    def succeed(self, attempt):
        with self._connect() as db:
            db.execute('UPDATE memory_delivery_attempts SET successful=1 WHERE attempt_id=?', (attempt,))

    def complete(self, scope):
        scope = scope_from(scope)
        key = digest(scope)
        receipt = 'delivery_' + key[:32]
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT context_json FROM memory_delivery_turns WHERE scope_key=?', (key,)).fetchone()
            if old:
                return json.loads(old['context_json'])
            rows = db.execute('SELECT fragments_json FROM memory_delivery_attempts '
                'WHERE scope_key=? AND successful=1 ORDER BY created_at,attempt_id', (key,)).fetchall()
            if not rows:
                return None
            fragments = {}
            for row in rows:
                for fragment in json.loads(row['fragments_json']):
                    fragments.setdefault(fragment['fragment_id'], fragment)
            context = {'schema': 'memory-context-v1', 'scope': scope,
                'fragments': sorted(fragments.values(), key=lambda row: row['fragment_id'])}
            context.update(delivery_receipt_id=receipt, delivery_sha256=digest(context))
            db.execute('INSERT INTO memory_delivery_turns VALUES(?,?,?,?)',
                (receipt, key, canonical(context), datetime.now(timezone.utc).isoformat()))
        return context

    def lookup(self, query):
        scope = scope_from(query)
        with self._connect() as db:
            row = db.execute('SELECT context_json FROM memory_delivery_turns WHERE scope_key=?', (digest(scope),)).fetchone()
            requests = db.execute('SELECT attempt_id,trace_id,request_json,fragments_json '
                'FROM memory_delivery_attempts WHERE scope_key=? AND successful=1 ORDER BY created_at,attempt_id',
                (digest(scope),)).fetchall()
        if not row:
            return {'state': 'unavailable', 'reason': 'completed_delivery_not_found'}
        context = json.loads(row['context_json'])
        for key in ('delivery_receipt_id', 'delivery_sha256'):
            if key in query and query[key] != context[key]:
                return {'state': 'unavailable', 'reason': 'delivery_identity_mismatch'}
        return {'state': 'available', 'context': context, 'requests': [
            {**json.loads(row['request_json']), 'attempt_id': row['attempt_id'], 'trace_id': row['trace_id'],
             'fragment_ids': [f['fragment_id'] for f in json.loads(row['fragments_json'])]}
            for row in requests]}

    @staticmethod
    def _activation_result(db, query):
        row = db.execute('SELECT * FROM memory_activation_assessments WHERE receipt_id=?',
            (query.get('delivery_receipt_id'),)).fetchone()
        if row:
            if row['payload_sha256'] != digest(query):
                raise ValueError('memory_use_idempotency_conflict')
            return json.loads(row['result_json'])
        return None

    def activation_result(self, query):
        with self._connect() as db:
            return self._activation_result(db, query)

    async def activate(self, query, context, bucket_manager, *, fragment_states=None):
        """Trust only resolved current-turn evidence from the authenticated evaluator.

        A bucket-level marker makes recovery after its write but before this
        SQLite transaction idempotent. No time-neighbour ripple is involved.
        """
        uses = query.get('uses')
        if not isinstance(uses, list):
            raise ValueError('uses_required')
        native = context['scope'].get('consumer_kind') == 'autonomy'
        pair = (set((query.get('source_events') or {}).values()) if native else
            {query.get('user_event_id'), query.get('assistant_event_id')} - {None, ''})
        def evidence(refs):
            return bool(refs) and isinstance(refs, list) and all(isinstance(ref, dict)
                and ref.get('event_id') in pair and type(ref.get('start')) is int and ref['start'] >= 0
                and type(ref.get('end')) is int and ref['end'] > ref['start']
                and isinstance(ref.get('text_sha256'), str) and len(ref['text_sha256']) == 64 for ref in refs)
        fragments = {f['fragment_id']: f for f in context['fragments']}
        eligibility = {row['fragment_id']: row for row in fragment_states or []}
        seen = set()
        for use in uses:
            if (not isinstance(use, dict) or not isinstance(use.get('use_id'), str) or not use['use_id']
                    or use['use_id'] in seen or use.get('fragment_ref') not in fragments
                    or use.get('status') not in {'new_use', 'continuing', 'not_used', 'uncertain'}):
                raise ValueError('invalid_memory_use')
            seen.add(use['use_id'])
        results, active = [], {}
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old_batch = self._activation_result(db, query)
            if old_batch is not None:
                return old_batch
            # Autonomy and chat keep separate continuity cursors; both touch the
            # same original memory activation authority and source groups.
            continuity_scope = ('autonomy:' + context['scope']['conversation_id'] if native else context['scope']['conversation_id'])
            previous = db.execute('SELECT * FROM memory_activation_groups WHERE conversation_id=?',
                (continuity_scope,)).fetchone()
            prior_groups = json.loads(previous['groups_json']) if previous else {}
            unavailable_groups = {fragment['source_group_id'] for fragment in context['fragments']
                if eligibility.get(fragment['fragment_id'], {}).get('state') == 'unavailable'}
            active.update({group: value for group, value in prior_groups.items() if group in unavailable_groups})
            current_order = db.execute('SELECT rowid FROM memory_delivery_turns WHERE receipt_id=?',
                (context['delivery_receipt_id'],)).fetchone()[0]
            previous_order = db.execute('SELECT rowid FROM memory_delivery_turns WHERE receipt_id=?',
                (previous['receipt_id'],)).fetchone()[0] if previous else 0
            late = current_order < previous_order
            for use in uses:
                if (not isinstance(use, dict) or not isinstance(use.get('use_id'), str) or not use['use_id']
                        or use.get('status') not in {'new_use', 'continuing', 'not_used', 'uncertain'}):
                    raise ValueError('invalid_memory_use')
                use_id, fingerprint = use['use_id'], digest({'scope': context['scope'], 'use': use})
                old = db.execute('SELECT * FROM memory_activation_uses WHERE use_id=?', (use_id,)).fetchone()
                if old:
                    if old['receipt_id'] != context['delivery_receipt_id'] or old['payload_sha256'] != fingerprint:
                        raise ValueError('memory_use_idempotency_conflict')
                    results.append(json.loads(old['result_json']))
                    continue
                fragment = fragments.get(use.get('fragment_ref'))
                if fragment is None:
                    raise ValueError('fragment_not_delivered')
                group, status = fragment['source_group_id'], use['status']
                episode = str(use.get('episode_id') or '')
                has_evidence = evidence(use.get('evidence_refs'))
                prior = prior_groups.get(group)
                continuous = bool(prior and previous['receipt_id'] != context['delivery_receipt_id'])
                new_episode = bool(episode and prior and episode != prior.get('episode_id')
                    and evidence(use.get('new_episode_evidence_refs')))
                result = {'use_id': use_id, 'status': 'not_activated', 'reason': status}
                if prior and (status == 'uncertain' or not has_evidence):
                    # An unverified response is not evidence that continuity ended.
                    active[group] = prior
                eligible = eligibility.get(fragment['fragment_id'], {'state': 'available'})
                if eligible['state'] != 'available':
                    result['reason'] = eligible.get('reason') or eligible['state']
                    if prior:
                        active[group] = prior
                    db.execute('INSERT INTO memory_activation_uses VALUES(?,?,?,?)',
                        (use_id, context['delivery_receipt_id'], fingerprint, canonical(result)))
                    results.append(result)
                    continue
                already = db.execute('SELECT 1 FROM memory_activation_group_receipts WHERE receipt_id=? AND group_id=?',
                    (context['delivery_receipt_id'], group)).fetchone()
                if status in {'new_use', 'continuing'} and has_evidence:
                    active[group] = {'episode_id': episode or (prior or {}).get('episode_id', '')}
                if status == 'new_use' and has_evidence and not late and not already and (not continuous or new_episode):
                    activation_id = 'activation_' + digest([context['delivery_receipt_id'], group])
                    ok = await bucket_manager.touch(fragment['bucket_id'], activation_id=activation_id)
                    if ok is not True:
                        raise RuntimeError('memory_activation_write_unconfirmed')
                    db.execute('INSERT INTO memory_activation_group_receipts VALUES(?,?,?)',
                        (context['delivery_receipt_id'], group, activation_id))
                    result = {'use_id': use_id, 'status': 'activated'}
                elif status == 'new_use' and has_evidence and (late or already):
                    result = {'use_id': use_id, 'status': 'not_activated',
                        'reason': 'late_evaluation' if late else 'same_turn_source_group'}
                elif status == 'new_use' and has_evidence and continuous:
                    result = {'use_id': use_id, 'status': 'continuing', 'reason': 'continuous_source_group'}
                elif status == 'new_use' and not has_evidence:
                    result['reason'] = 'current_evidence_required'
                db.execute('INSERT INTO memory_activation_uses VALUES(?,?,?,?)',
                    (use_id, context['delivery_receipt_id'], fingerprint, canonical(result)))
                results.append(result)
            if not late and (not previous or previous['receipt_id'] != context['delivery_receipt_id']):
                db.execute('INSERT OR REPLACE INTO memory_activation_groups VALUES(?,?,?)',
                    (continuity_scope, context['delivery_receipt_id'], canonical(active)))
            result = {'state': 'recorded', 'results': results}
            db.execute('INSERT INTO memory_activation_assessments VALUES(?,?,?)',
                (context['delivery_receipt_id'], digest(query), canonical(result)))
        return result
