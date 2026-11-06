"""Event-first ordinary chat extraction using the existing Authority preparation queue."""
from __future__ import annotations

import hashlib
import json
from contextlib import closing
from types import SimpleNamespace

from memory_candidate_preparation import MemoryCandidatePreparation
from memory_chat_event_spans import parse_event_selection
from memory_chat_segments import ChatSegmentProgress, segment_plan
from memory_l1_api import validate_response
from memory_l1_contract import semantic_sha256
from memory_l1_service import L1OwnerError
from memory_source_provenance import daily_sources


def window_method(engine, day, window, oversized):
    """Pin each original window once, also respecting work prepared before this option existed."""
    authority = engine.memory_authority_store
    fingerprint = hashlib.sha256(json.dumps(
        {'date': day, 'profile': engine.memory_semantic_profile_id, 'window': window},
        ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    key = 'daily_chat_window_method:' + fingerprint
    existing = authority.get_meta(key)
    if existing:
        return existing, fingerprint
    method = engine.config.get('reflection', {}).get('daily_chat_memory_extraction_mode', 'single_pass')
    if method != 'event_first' or MemoryCandidatePreparation(authority).get('daily_chat:' + fingerprint):
        method = 'single_pass'
    with closing(authority._connect()) as db, db:
        if method == 'event_first' and oversized and db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_chat_segment_progress'").fetchone():
            sources = daily_sources(window, profile_id=engine.memory_semantic_profile_id)
            ids = [{k: s.get(k) for k in ('source_key', 'snapshot_sha256', 'version_id')} for s in sources]
            old_scope = 'chat-segment-source:' + semantic_sha256({'day': day, 'sources': ids})
            if db.execute('SELECT 1 FROM memory_chat_segment_progress WHERE scope=? LIMIT 1', (old_scope,)).fetchone():
                method = 'single_pass'
        db.execute('INSERT OR IGNORE INTO authority_meta(key,value) VALUES(?,?)', (key, json.dumps(method)))
        pinned = json.loads(db.execute('SELECT value FROM authority_meta WHERE key=?', (key,)).fetchone()['value'])
    return pinned, fingerprint


def _reply(snapshot):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=snapshot['content']),
                           finish_reason=snapshot['finish_reason'])], native_schema=snapshot.get('native_schema'),
                           native_status=snapshot.get('native_status', 'completed'))


def _selection_envelope(engine, day, window, segment):
    material, _, _ = engine._memory_l1_daily_context(window, segment_context=segment)
    visible = material['source_texts']
    sources = [{'source': row['binding_id'], 'speaker': row['participant']['display_name'],
                'speaker_kind': row['participant']['kind'],
                'attribution': row['participant']['resolution'], 'when': row.get('source_recorded_at'),
                'text': row['text']} for row in visible]
    payload = {'input_kind': 'daily_chat_event_selection', 'date': day, 'sources': sources}
    return visible, payload


def _owned_by_segment(event, segment):
    if segment is None:
        return True
    core = segment['core_range']
    return any(a['source'] == segment['core_source'] and a['start'] < core['end'] and core['start'] < a['end']
               for a in event['anchors'])


def _event_receipt(day, sources, event):
    anchors = []
    for a in event['anchors']:
        src = sources[int(a['source'].split(':')[1]) - 1]
        anchors.append([src['source_key'], src.get('version_id'), src.get('snapshot_sha256'), a['start'], a['end']])
    return 'daily_chat_event:' + semantic_sha256({'day': day, 'anchors': anchors, 'point': event['point']})


async def extract_event_window(engine, day, window, *, fingerprint, oversized, mode, min_confidence,
                               candidate_budget, call_budget, window_index, window_total, audit, force=False):
    store = ChatSegmentProgress(engine.memory_authority_store)
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    originals = daily_sources(window, profile_id=engine.memory_semantic_profile_id)
    task_id = 'chat-events:' + fingerprint
    logical_sources = [{k: s.get(k) for k in ('system','namespace','profile_id','conversation_id','event_id','role')}
                       for s in originals]
    scope = 'chat-event-source:' + semantic_sha256({'day': day, 'sources': logical_sources})
    record = store.get(task_id)
    state = record['progress'] if record else {'unit': 0, 'index': 0, 'phase': 'ready', 'receipts': []}
    revision = record['revision'] if record else 0
    segment_budget = (state.get('segment_budget') if record else
                      engine.daily_chat_memory_window_max_input_chars if oversized else None)
    plan = segment_plan(originals, segment_budget) if segment_budget is not None else [None]
    calls = generated = 0
    output = {}

    def save(value, *, dispatch=False):
        nonlocal revision, state
        updated = store.save(task_id, scope, {**value, 'segment_budget': segment_budget}, revision, dispatch=dispatch)
        revision, state = updated['revision'], updated['progress']

    def finish(reason, *, manual=False):
        return list(output.values()), {'task_id':task_id, 'unit':state['unit'], 'index':state['index'],
            'phase':state['phase'], 'revision':revision, 'model_call_count':calls, 'model_candidate_count':generated,
            'step': ('complete' if state['phase'] == 'complete' else 'assembly' if 'events' in state else 'selection'),
            **({'error_field': state['error_field']} if state.get('error_field') and state['phase'] == 'failed' else {}),
            'partial':reason != 'complete', 'reason':reason, 'requires_manual_retry':manual}

    def collect(receipt):
        for row in queue.remaining(receipt):
            cid = row['candidate_id']
            if cid not in output:
                if len(output) >= candidate_budget:
                    return False
                output[cid] = engine._authority_candidate_legacy_item(row)['candidate']
        return True

    def advance(receipt_key):
        receipts = list(state['receipts'])
        if receipt_key not in receipts:
            receipts.append(receipt_key)
        save({'unit':state['unit'], 'index':state['index']+1, 'phase':'ready',
              'events':state['events'], 'receipts':receipts})

    master, _ = engine._memory_l1_prompt('ombre.daily_chat_memory_prompt',
        runtime_values={'max_candidates': max(1, engine.daily_chat_memory_max_per_day)})
    selection_prompt, selector_meta = engine._memory_l1_prompt('ombre.daily_chat_event_selection_prompt')
    assembly_prompt, _ = engine._memory_l1_prompt('ombre.daily_chat_event_assembly_prompt')
    route = engine._memory_task_route('auto_l1')
    if not master.strip() or not selection_prompt.strip() or not assembly_prompt.strip():
        return finish('prompt_disabled', manual=True)
    if not route.get('client') or not route.get('configured', True):
        return finish('memory_model_unconfigured', manual=True)
    for previous in state['receipts']:
        receipt = queue.get(previous)
        if not receipt or receipt.get('stale'):
            return finish('source_preparation_stale', manual=True)
        if not collect(receipt):
            return finish('candidate_budget_exhausted_prepared')
    while state['unit'] < len(plan):
        segment = plan[state['unit']]
        selecting = 'events' not in state
        if not selecting and state['index'] == len(state['events']):
            save({'unit':state['unit']+1, 'index':0, 'phase':'ready', 'receipts':state['receipts']})
            continue
        event = None if selecting else state['events'][state['index']]
        receipt_key = None if selecting else _event_receipt(day, originals, event)
        receipt = queue.get(receipt_key) if receipt_key else None
        if receipt:
            if receipt.get('stale'):
                return finish('source_preparation_stale', manual=True)
            advance(receipt_key)
            if not collect(receipt):
                return finish('candidate_budget_exhausted_prepared')
            continue
        if state['phase'] == 'dispatched':
            return finish('model_outcome_unknown_requires_retry_decision', manual=True)
        if state['phase'] == 'failed' and not force:
            return finish(state['error'], manual=True)
        if len(output) >= candidate_budget:
            return finish('candidate_budget_exhausted')
        cached = state.get('response') if state['phase'] == 'received' else None
        if not cached and calls >= call_budget:
            return finish('window_budget_exhausted')
        def start_call(prompt_meta, effective_route):
            nonlocal calls
            _, catalog_revision = engine._memory_l1_revisions().catalog_provider()
            provenance = {'prompt_revision': prompt_meta['prompt_revision'], 'model_route_revision':
                          'route_sha256:' + semantic_sha256({k:v for k,v in effective_route.items() if k != 'client'})}
            save({**state, 'phase':'dispatched', 'error':None, 'catalog_revision':catalog_revision,
                  'generation_context':provenance, 'response':None}, dispatch=True)
            calls += 1

        def received(response):
            save({**state, 'phase':'received', 'response':{
                'content':engine._completion_content(response), 'finish_reason':engine._completion_finish_reason(response),
                'native_schema':getattr(response, 'native_schema', None), 'native_status':getattr(response,'native_status','completed'),
                'generation_context':state['generation_context']}})

        try:
            if selecting:
                visible, payload = _selection_envelope(engine, day, window, segment)
                if cached:
                    response = _reply(cached)
                else:
                    start_call(selector_meta, route)
                    response = await engine._daily_chat_memory_create_completion(route['client'], model=route['model'],
                        messages=[{'role':'system', 'content':selection_prompt},
                                  {'role':'user', 'content':json.dumps(payload, ensure_ascii=False)}],
                        task_stage='auto_l1', task_route=route, max_tokens=engine.daily_chat_memory_candidate_max_tokens,
                        temperature=engine.temperature, use_daily_client=route['connection']=='daily')
                    received(response)
                validate_response(response)
                if engine._completion_finish_reason(response) != 'stop':
                    raise L1OwnerError('event_selection_incomplete', field_path='generation')
                raw = json.loads(engine._completion_content(response))
                events = [e for e in parse_event_selection(raw, visible) if _owned_by_segment(e, segment)]
                # The transient whole-conversation explanation is never a candidate or L2 source.
                save({'unit':state['unit'], 'index':0, 'phase':'ready', 'events':events, 'receipts':state['receipts']})
                continue
            if cached and engine._memory_l1_revisions().catalog_provider()[1] != state['catalog_revision']:
                raise L1OwnerError('event_assembly_catalog_changed', field_path='catalog_revision')
            candidates, stats = await engine._extract_window_candidates(day, window, mode=mode,
                selected_event=event, segment_context=segment, window_index=window_index,
                window_total=window_total, saved_response=cached, response_checkpoint=received, before_model_call=start_call)
            if stats.get('prompt_disabled'):
                raise L1OwnerError('prompt_disabled', field_path='assembly')
            generated += len(candidates)
            local_audit = {}
            normalized = engine._normalize_daily_chat_memory_candidates(day, candidates, window, max_candidates=1,
                min_confidence=min_confidence, mode=mode, audit=local_audit)
            if audit is not None:
                for key in ('hard_rejects', 'soft_flags'):
                    counts = audit.setdefault(key, {})
                    for reason, count in local_audit.get(key, {}).items():
                        counts[reason] = counts.get(reason, 0) + count
                if local_audit.get('l1_issues'):
                    audit.setdefault('l1_issues', []).extend(local_audit['l1_issues'])
            if local_audit.get('hard_rejects',{}).get('invalid_l1_document'):
                raise L1OwnerError('invalid_l1_document', field_path='document')
            proposals = [engine._daily_chat_memory_proposal(item,mode=mode) for item in normalized]
            receipt = queue.prepare(receipt_key, proposals, engine.memory_ingestion_policy, scope=scope,
                source_fingerprint=receipt_key, generated_count=len(candidates))
        except Exception as exc:
            from memory_authority import RevisionConflict
            if isinstance(exc, RevisionConflict):
                raise
            code = getattr(exc, 'code', None) or type(exc).__name__
            status = getattr(exc, 'status_code', None)
            rejected = isinstance(status, int) and 400 <= status < 500 and status not in (408,409)
            if state['phase'] == 'dispatched' and not rejected and not isinstance(exc, L1OwnerError):
                return finish('model_outcome_unknown_requires_retry_decision', manual=True)
            if state['phase'] == 'received' and not isinstance(exc, (L1OwnerError, json.JSONDecodeError)):
                # An interrupted write can be resumed from the completed response without a new charge.
                return finish('event_preparation_interrupted', manual=True)
            # A received but malformed answer is a known failure. An explicit retry can
            # dispatch this step alone; durable selections and earlier receipts remain.
            save({**state, 'phase':'failed', 'error':code, 'error_field':getattr(exc, 'field_path', '')})
            return finish(code, manual=True)
        advance(receipt_key)
        if not collect(receipt):
            return finish('candidate_budget_exhausted_prepared')
    if state['phase'] != 'complete':
        save({**state,'phase':'complete'})
    return finish('complete')
