"""Bounded views and durable dispatch receipts inside existing daily windows."""
from __future__ import annotations

import copy
import json
from contextlib import closing

from memory_authority import RevisionConflict
from memory_l1_contract import semantic_sha256
from memory_l1_service import L1OwnerError


def segment_plan(sources, budget):
    if type(budget) is not int or budget < 256:
        raise ValueError('invalid_segment_budget')
    context = min(128, budget // 8)
    width = budget - 4 * context
    result = []
    for index, source in enumerate(sources):
        text = source['text']
        for start in range(0, len(text), width):
            end = min(start + width, len(text))
            views = {str(index): {'start': max(0, start-context),
                                 'end': min(len(text), end+context)}}
            if index:
                size = len(sources[index-1]['text'])
                views[str(index-1)] = {'start': max(0, size-context), 'end': size}
            if index + 1 < len(sources):
                views[str(index+1)] = {'start': 0, 'end': min(context, len(sources[index+1]['text']))}
            result.append({'core_source': f'source:{index+1}',
                           'core_range': {'start': start, 'end': end}, 'views': views})
    return result


class ChatSegmentProgress:
    """Working state in Authority, never a parallel memory/candidate authority."""
    def __init__(self, authority):
        self.authority = authority
        with closing(authority._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS memory_chat_segment_progress ('
                       'task_id TEXT PRIMARY KEY, scope TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL)')

    def get(self, task_id):
        with closing(self.authority._connect()) as db:
            row = db.execute('SELECT * FROM memory_chat_segment_progress WHERE task_id=?', (task_id,)).fetchone()
            return {'revision': row['revision'], 'progress': json.loads(row['payload'])} if row else None

    def save(self, task_id, scope, progress, expected_revision, *, dispatch=False):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM memory_chat_segment_progress WHERE task_id=?', (task_id,)).fetchone()
            if (row['revision'] if row else 0) != expected_revision:
                raise RevisionConflict('chat segment progress changed')
            if dispatch:
                uncertain = db.execute("SELECT task_id FROM memory_chat_segment_progress "
                    "WHERE scope=? AND json_extract(payload,'$.phase')='dispatched'", (scope,)).fetchone()
                if uncertain:
                    raise L1OwnerError('model_outcome_unknown_requires_retry_decision', status=409)
            revision = expected_revision + 1
            db.execute('INSERT INTO memory_chat_segment_progress VALUES(?,?,?,?) '
                       'ON CONFLICT(task_id) DO UPDATE SET revision=excluded.revision,payload=excluded.payload',
                       (task_id, scope, revision, json.dumps(progress, ensure_ascii=False)))
            db.commit()
            return {'revision': revision, 'progress': copy.deepcopy(progress)}
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()


async def extract_segments(engine, day, window, *, mode, min_confidence,
                           candidate_budget, call_budget, self_context, audit):
    """Prepare once, then drain through the unchanged daily Authority writer."""
    from memory_candidate_preparation import MemoryCandidatePreparation
    from memory_source_provenance import daily_sources

    sources = daily_sources(window, profile_id=engine.memory_semantic_profile_id)
    plan = segment_plan(sources, engine.daily_chat_memory_window_max_input_chars)
    prompt_limit = max(1, engine.daily_chat_memory_max_per_day)
    prompt, prompt_context = engine._memory_l1_prompt('ombre.daily_chat_memory_prompt',
        runtime_values={'max_candidates': prompt_limit})
    if not prompt.strip():
        return [], {'partial': True, 'reason': 'prompt_disabled', 'model_call_count': 0}
    route = engine._memory_task_route('auto_l1')
    _, catalog_revision = engine._memory_l1_revisions().catalog_provider()
    source_identity = [{k: s.get(k) for k in ('source_key', 'snapshot_sha256', 'version_id')} for s in sources]
    scope = 'chat-segment-source:' + semantic_sha256({'day': day, 'sources': source_identity})
    # Scheduling/candidate quotas do not change already generated segment identity.
    identity = {'protocol': 'chat-segment-v1', 'scope': scope, 'plan': plan,
        'prompt': prompt_context['prompt_revision'], 'catalog': catalog_revision,
        'route': {k: v for k, v in route.items() if k != 'client'},
        'temperature': engine.temperature,
        'source_identity': engine.config.get('identity', {})}
    task_id = 'chat-segments:' + semantic_sha256(identity)
    store = ChatSegmentProgress(engine.memory_authority_store)
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    record = store.get(task_id)
    state = record['progress'] if record else {'cursor': 0, 'phase': 'ready', 'receipts': []}
    revision = record['revision'] if record else 0
    calls = 0
    output = []
    meta = {'task_id': task_id, 'segment_count': len(plan), 'partial': True,
            'model_call_count': 0, 'model_candidate_count': 0, 'reason': 'window_budget_exhausted'}

    def save(value, *, dispatch=False):
        nonlocal state, revision
        saved = store.save(task_id, scope, value, revision, dispatch=dispatch)
        state, revision = saved['progress'], saved['revision']

    def collect(receipt):
        rows = queue.remaining(receipt)
        remaining = max(0, candidate_budget - len(output))
        output.extend(engine._authority_candidate_legacy_item(row)['candidate'] for row in rows[:remaining])
        return len(rows) > remaining

    # Old prepared proposals remain canonical, including owner revisions.
    for key in state['receipts']:
        receipt = queue.get(key)
        if receipt is None:
            raise L1OwnerError('segment_receipt_missing', status=409)
        if collect(receipt):
            meta['reason'] = 'candidate_budget_exhausted_prepared'
            return output, meta
        if receipt.get('issue'):
            meta.update(reason=receipt['issue']['code'], requires_manual_retry=True)
            return output, meta
    while state['cursor'] < len(plan):
        if state['phase'] == 'dispatched':
            meta.update(reason='model_outcome_unknown_requires_retry_decision', requires_manual_retry=True)
            break
        if len(output) >= candidate_budget:
            meta['reason'] = 'candidate_budget_exhausted'
            break
        index = state['cursor']
        segment = plan[index]
        preparation_key = task_id + ':' + str(index)
        receipt = queue.get(preparation_key)
        if receipt is None:
            if state['phase'] != 'received' and calls >= call_budget:
                break
            if state['phase'] != 'received':
                save({**state, 'phase': 'dispatched'}, dispatch=True)
                calls += 1
            # Persist the provider result BEFORE parsing. Reuse on restart, even
            # if normalization or candidate storage crashed after receipt.
            def received(response):
                save({**state, 'phase': 'received', 'response': {
                    'content': engine._completion_content(response),
                    'finish_reason': engine._completion_finish_reason(response),
                    'native_schema': getattr(response, 'native_schema', None),
                    'native_status': getattr(response, 'native_status', 'completed')}})
            try:
                candidates, stats = await engine._extract_window_candidates(day, window,
                    self_context=self_context, max_candidates=prompt_limit,
                    window_index=index+1, window_total=len(plan),
                    segment_context={**segment, 'task_id': task_id},
                    saved_response=state.get('response'), response_checkpoint=received)
                meta['model_candidate_count'] += len(candidates)
                if stats.get('prompt_disabled'):
                    raise L1OwnerError('prompt_disabled', field_path='generation')
                if stats.get('empty_output') or stats.get('parse_failure'):
                    raise L1OwnerError('segment_invalid_response', field_path='generation')
                local_audit = {}
                normalized = engine._normalize_daily_chat_memory_candidates(day, candidates, window,
                    max_candidates=max(1, len(candidates)+1), min_confidence=min_confidence,
                    mode=mode, audit=local_audit)
                invalid = local_audit.get('hard_rejects', {}).get('invalid_l1_document', 0)
                issue = {'code': 'invalid_l1_document', 'rejected_count': invalid} if invalid else None
                receipt = queue.prepare(preparation_key,
                    [engine._daily_chat_memory_proposal(item, mode=mode) for item in normalized],
                    engine.memory_ingestion_policy, scope=scope, source_fingerprint=preparation_key,
                    generated_count=len(candidates), issue=issue)
            except Exception as exc:
                meta.update(reason=('model_outcome_unknown_requires_retry_decision'
                    if state['phase'] == 'dispatched' else getattr(exc, 'code', type(exc).__name__)),
                    requires_manual_retry=True, model_call_count=calls)
                return output, meta
        # A queue receipt committed just before a crash wins over the cursor.
        save({'cursor': index+1, 'phase': 'ready',
              'receipts': state['receipts'] + [preparation_key]})
        if collect(receipt):
            meta['reason'] = 'candidate_budget_exhausted_prepared'
            break
        if receipt.get('issue'):
            meta.update(reason=receipt['issue']['code'], requires_manual_retry=True)
            break
    else:
        if state['phase'] != 'complete':
            save({**state, 'phase': 'complete'})
        meta.update(partial=False, reason='complete')
    meta.update(model_call_count=calls, cursor=state['cursor'])
    if audit is not None:
        audit.setdefault('segmented_windows', []).append(copy.deepcopy(meta))
    return output, meta
