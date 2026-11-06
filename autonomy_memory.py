"""Native activity recall consumer; uses the original Recall and delivery store."""
from copy import deepcopy
import math
import os
import sqlite3
import time
from urllib.parse import quote

import httpx

from memory_delivery import digest, compiled_candidates

FIELDS = {'consumer_kind', 'conversation_id', 'identity_id', 'activity_id',
          'activity_revision', 'source_scope', 'cycle_id', 'operation_id'}
HEADER = 'X-Guyan-Autonomy-Memory-Recall'


async def feedback_evidence(service, context):
    """Resolve annotated replies from the authenticated daily planning record."""
    from memory_delivery import scope_from as delivery_scope
    origins, fragments = {}, {}
    if context.get('model_runtime_scope') != 'internal:self_adjustment':
        return origins, []
    for key, original in (context.get('feedback_turns') or {}).items():
        origin = origins[key] = {'status': 'unavailable', 'targets': []}
        if not isinstance(original, dict) or original.get('consumer_kind'):
            continue
        try:
            scope = delivery_scope(original)
            result = service.memory_delivery.lookup(scope)
        except (sqlite3.Error, ValueError, KeyError, TypeError):
            continue
        if result.get('state') != 'available':
            continue
        receipt = result['context']
        if (receipt.get('scope') != scope or not receipt.get('delivery_receipt_id')
                or receipt.get('delivery_sha256') != digest({k: receipt[k] for k in ('schema', 'scope', 'fragments')})):
            continue
        rows = receipt['fragments']
        origin.update(status='available' if rows else 'empty', fragment_count=len(rows),
                      receipt_id=receipt['delivery_receipt_id'],
                      requests=[{k: req[k] for k in ('trace_id', 'logical_request_id', 'request_id', 'request_ordinal',
                          'fragment_ids', 'configuration_revision', 'prompt_preset_revision', 'provider_payload_sha256')
                          if k in req} for req in result.get('requests', [])])
        for row in rows:
            fragment_id = row['fragment_id']
            origin['targets'].append({'fragment_id': fragment_id, 'target_key': 'memory-target:' + digest({
                k: row.get(k) for k in ('memory_id','bucket_id','l1_revision','l2_revision','source_group_id',
                                        'text_sha256','source_refs','coverage','legacy_body_sha256')}),
                'memory_id': row['memory_id'],
                'l1_revision': row.get('l1_revision'), 'l2_revision': row.get('l2_revision')})
            fragments[fragment_id] = row
    if not fragments:
        return origins, []
    formal = [f for f in fragments.values() if type(f.get('l1_revision')) is int and f['l1_revision'] > 0]
    try:
        states = {r['fragment_id']: r['state'] for r in
                  await service._validate_memory_delivery({'fragments': formal})} if formal else {}
    except (httpx.HTTPError, sqlite3.Error, ValueError, TypeError, KeyError, TimeoutError):
        states = {}  # Original delivery survives; present eligibility remains unknown.
    for origin in origins.values():
        for target in origin['targets']:
            target['source_state'] = states.get(target['fragment_id'], 'unavailable')
    return origins, [row for row in formal if states.get(row['fragment_id']) == 'available'
                     and not row['coverage'].get('wire_partial')]


async def source_excerpt(service, fragment, binding, token_budget):
    """Read just the bound L0 range. A historical fallback is not evidence."""
    from memory_delivery import sha
    anchor = next((a for a in binding.get('anchors', []) if isinstance(a, dict)
                   and type(a.get('start')) is int and type(a.get('end')) is int
                   and 0 < a['end'] - a['start'] <= 4 * token_budget), None)
    client = getattr(service, 'http_client', None)
    if not anchor or client is None:
        return None
    base = os.environ.get('OMBRE_MEMORY_INDEX_ADMIN_URL', 'http://ombre-brain:8000').rstrip('/')
    token = os.environ.get('OMBRE_MEMORY_WRITE_TOKEN') or getattr(service, 'gateway_token', '')
    url = base + '/api/memory-authority/memories/' + quote(fragment['memory_id'], safe='') + '/sources/' + quote(binding['binding_id'], safe='')
    try:
        response = await client.get(url, headers={'Authorization': 'Bearer ' + token},
            params={'revision': fragment['l1_revision'], 'start': anchor['start'], 'end': anchor['end']})
        response.raise_for_status()
        source = response.json()['source']
        if not isinstance(source, dict):
            return None
        text = source.get('text')
        if (source.get('eligible_source') is True and source.get('binding') == binding
                and isinstance(text, str) and sha(text) == anchor.get('quote_sha256')):
            return text
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        pass
    return None


async def add_feedback_evidence(service, original, sections, normal_spans):
    """Use the original Recall carriers and their whole-section budgets."""
    from utils import count_tokens_approx
    view = getattr(service, 'memory_authority_view', None)
    reader = getattr(view, 'memory_l1_recall_map', None)
    records = reader([f['bucket_id'] for f in original]) if callable(reader) and original else {}
    selected, excerpts = [], {}
    for kind, (normal, limit) in list(sections.items()):
        parts = []
        for f in original:
            if f['coverage'].get('section') != kind:
                continue
            text = f['text']
            record = records.get(f['bucket_id']) or {}
            statements = (record.get('document') or {}).get('statements') or []
            ids = set(f['coverage'].get('statement_ids') or [])
            facts = [s for s in statements if s.get('statement_id') in ids and isinstance(s.get('content'), str)]
            label = '被批注的原回复当时收到的记忆片段：\n'
            excerpt = label + text
            if count_tokens_approx('\n\n'.join([*parts, excerpt])) > limit:
                continue
            if facts and record.get('revision') == f['l1_revision']:
                compared = excerpt + '\n同版 L1 对应陈述：\n' + '\n'.join(s['content'] for s in facts)
                if count_tokens_approx('\n\n'.join([*parts, compared])) <= limit:
                    excerpt = compared
                    binding_ids = {bid for s in facts for bid in s.get('evidence_refs') or []}
                    refs = {r['binding_id'] for r in f.get('source_refs', [])}
                    for binding in (record['document'].get('evidence_bindings') or []):
                        if binding.get('binding_id') not in binding_ids & refs:
                            continue
                        key = (f['memory_id'], f['l1_revision'], binding['binding_id'])
                        if key not in excerpts:
                            excerpts[key] = await source_excerpt(service, f, binding, limit)
                        text_view = excerpts[key]
                        if text_view:
                            compared = excerpt + '\n原始记录中的对应文字：\n' + text_view
                            if count_tokens_approx('\n\n'.join([*parts, compared])) <= limit:
                                excerpt = compared
            parts.append(excerpt)
            # final_fragments creates its own content identity. Copy only the
            # original semantic candidate, never re-hash an existing id into it.
            selected.append({k: deepcopy(v) for k,v in f.items() if k != 'fragment_id'})
        if parts:
            if normal and count_tokens_approx('\n\n'.join([*parts, normal])) <= limit:
                if not any(normal in part for part in parts):
                    parts.append(normal)
            else:
                for span in normal_spans:
                    text = span.get('text')
                    if (span.get('section') == kind and text and not any(text in part for part in parts)
                            and count_tokens_approx('\n\n'.join([*parts, text])) <= limit):
                        parts.append(text)
            sections[kind] = ('\n\n'.join(parts), limit)
    return selected


def preparation_contract(service):
    """Bound the whole operation without shortening its existing selection IO.

    Source read and the surrounding preparation use the existing Gateway read
    allowance. Reserve the existing selectable maximum (30s), so a cached
    contract also covers an owner changing Recall from 15s to 30s at runtime.
    This is an operation/transport deadline, not a new retrieval quality budget.
    """
    from gateway import DEFAULT_GATEWAY_UPSTREAM_READ_TIMEOUT_SECONDS
    read = getattr(getattr(getattr(service, 'http_client', None), 'timeout', None), 'read', None)
    if isinstance(read, bool) or not isinstance(read, (int, float)) or not math.isfinite(read) or read <= 0:
        read = DEFAULT_GATEWAY_UPSTREAM_READ_TIMEOUT_SECONDS
    selection = getattr(service, 'recall_timeout_seconds', 15.0)
    selection_limit = max(30.0, selection)
    return {'schema': 'autonomy-recall-contract-v1', 'state': 'available',
        'source_read_seconds': read, 'preparation_seconds': read,
        'selection_seconds': selection, 'selection_limit_seconds': selection_limit,
        'total_seconds': read * 2 + selection_limit}


def recall_input(context):
    """The authenticated planning record is complete input, not a user turn."""
    from memory_delivery import sha
    scope = scope_from(context.get('scope'))
    query = context.get('query')
    if context.get('status') != 'prepared' or not isinstance(query, str) or not query.strip():
        raise ValueError('autonomy_prepared_source_required')
    return {'q_current': query, 'q_context': query, 'query_views': [query],
        'current_indices': [], 'recent_indices': [], 'current_messages': [], 'recent_messages': [],
        'metadata': {'input_source': 'autonomy_planning_record', 'consumer_kind': 'autonomy',
            'current_input_complete': True, 'source_scope': scope,
            'conversation_id': scope['conversation_id'], 'turn_id': '',
            'current_message_indices': [], 'current_message_count': 0, 'staged_user_event_ids': [],
            'recent_message_indices': [], 'recent_turns_requested': 0, 'recent_turns_selected': 0,
            'recent_context_truncated_chars': 0, 'current_chars': len(query), 'context_chars': 0,
            'query_views': [{'kind': 'activity', 'chars': len(query), 'sha256': sha(query)}]}}


def scope_from(value):
    if not isinstance(value, dict) or not FIELDS <= set(value):
        raise ValueError('autonomy_scope_required')
    scope = {k: value[k] for k in FIELDS}
    if scope['consumer_kind'] != 'autonomy': raise ValueError('autonomy_consumer_required')
    for key in FIELDS - {'activity_id', 'activity_revision'}:
        if not isinstance(scope[key], str) or not scope[key].strip() or len(scope[key]) > 400:
            raise ValueError('autonomy_identity_required')
    if scope['activity_id'] is None:
        if scope['activity_revision'] is not None: raise ValueError('autonomy_activity_revision_mismatch')
    elif (not isinstance(scope['activity_id'], str) or not scope['activity_id']
          or len(scope['activity_id']) > 400 or type(scope['activity_revision']) is not int
          or scope['activity_revision'] < 1):
        raise ValueError('autonomy_activity_revision_required')
    return scope


async def read_context(service, scope):
    scope = scope_from(scope)
    adapter = getattr(service, 'canonical_adapter', None)
    if (not adapter or not adapter.enabled or not adapter.base_url or not adapter.token
            or adapter.conversation_id != scope['conversation_id']):
        raise ValueError('autonomy_source_reader_unconfigured_or_scope_mismatch')
    response = await adapter.http_client.get(adapter.base_url + '/bridge/v1/autonomy-memory-context',
        params={'cycle_id': scope['cycle_id'], 'operation_id': scope['operation_id']},
        headers={'Authorization': 'Bearer ' + adapter.token, 'X-Guyan-Device-ID': adapter.device_id})
    response.raise_for_status()
    result = response.json()
    if result.get('state') != 'available' or scope_from(result.get('scope')) != scope:
        raise ValueError('autonomy_source_not_current_or_exact')
    return result


async def historical_input_states(service, context):
    """Qualify immutable delivered inputs, not permission to revise a target.

    L1/L2 may have advanced since this completed review. Its original L1
    bindings must still exist and pass the existing live source reader. A
    withdrawn memory, disabled recall policy or revoked original source still
    fails. No historical presentation body is reconstructed or returned here.
    """
    import asyncio
    import httpx
    import memory_recall_runtime as memory_use
    modern = [f for f in context['fragments'] if type(f.get('l1_revision')) is int and f['l1_revision'] > 0]
    legacy = [f for f in context['fragments'] if f not in modern]
    states = {f['fragment_id']: {'fragment_id': f['fragment_id'], 'state': 'unavailable',
        'reason': 'historical_source_validation_unavailable'} for f in modern}
    if legacy:
        for row in await service._validate_memory_delivery({'fragments': legacy}):
            states[row['fragment_id']] = row
    if not modern:
        return list(states.values())
    items = {}
    for fragment in modern:
        key = (fragment['memory_id'], fragment['l1_revision'])
        item = items.setdefault(key, {'memory_id': key[0], 'revision': key[1],
            'binding_ids': [], 'allow_historical': True})
        item['binding_ids'] = sorted(set(item['binding_ids']) | {ref['binding_id'] for ref in fragment['source_refs']})
    try:
        result = await memory_use.read_batch(service, {}, 'historical_delivery_source_validation',
            '/api/memory-authority/recall-sources', {'items': list(items.values())})
        rows = {(row['memory_id'], row['revision']): row for row in result['items']}
        for fragment in modern:
            row = rows.get((fragment['memory_id'], fragment['l1_revision'])) or {}
            state = states[fragment['fragment_id']]
            if row.get('state') != 'ok':
                state.update(state='withdrawn' if row.get('state') == 'memory_not_current_or_eligible' else 'changed',
                    reason='historical_memory_source_not_eligible')
                continue
            refs = fragment['source_refs']
            if refs and all((row.get('sources') or {}).get(ref['binding_id'], {}).get('eligible_source') is True for ref in refs):
                state.clear()
                state.update(fragment_id=fragment['fragment_id'], state='available')
            else:
                state.update(state='withdrawn', reason='historical_memory_source_no_longer_eligible')
    except (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError, TypeError):
        pass  # Unknown source availability remains explicitly unavailable.
    return list(states.values())


async def prepare(service, scope, *, model, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    started = time.perf_counter()
    diagnostics['failure_stage'] = 'source_context'
    context = await read_context(service, scope)
    diagnostics['source_context_ms'] = round((time.perf_counter() - started) * 1000)
    if context.get('status') != 'prepared' or not isinstance(context.get('query'), str) or not context['query'].strip():
        raise ValueError('autonomy_prepared_source_required')
    origins, original_fragments = await feedback_evidence(service, context)
    if original_fragments:
        context = {**context, '_feedback_evidence': original_fragments}
    # This is a real activity search cue, never a fabricated user/chat event.
    payload = {'model': model, 'messages': [{'role': 'system', 'content': context['query']}],
        '_ombre_trace_context': {'runtime_scope': 'internal:autonomy_chain',
            'conversation_id': scope['conversation_id'], 'request_id': context.get('request_id'),
            'recall_signals': context.get('recall_signals') or {}}}
    diagnostics['failure_stage'] = 'recall_prepare'
    prepare_started = time.perf_counter()
    # This consumer needs the exact compiled projection, not UI-only full
    # candidate prose/re-scoring. Compact keeps projection/timings intact.
    _, _, debug = await service.prepare_payload(payload, scope['conversation_id'],
        include_debug=True, debug_detail='compact', autonomy_context=context)
    diagnostics['recall_prepare_ms'] = round((time.perf_counter() - prepare_started) * 1000)
    projection = debug.get('memory_recall_projection') or {}
    frozen = compiled_candidates(projection, projection.get('delivery_candidates') or [])
    messages = []
    for block in projection.get('compiled_memory_blocks') or []:
        text = block.get('content')
        if not text: continue
        refs = [f for f in frozen if f.get('_compiled_block') == text]
        if not refs: continue
        messages.append({'role': 'system', 'content': text, '_prompt_projection': {
            'source': block['source_id'], 'revision': 'recall:' + digest(refs),
            'source_refs': {'scope': deepcopy(scope), 'fragments': [
                {k: deepcopy(v) for k, v in f.items() if not k.startswith('_')} for f in refs]}}})
    recall_id = service.memory_delivery.prepare_autonomy(scope, frozen, messages)
    diagnostics.update(failure_stage='completed', duration_ms=round((time.perf_counter() - started) * 1000))
    return {'state': 'available', 'scope': scope, 'recall_id': recall_id,
        'memory_messages': messages, 'recall_status': projection.get('recall_status'), 'feedback_origins': origins,
        'preparation': deepcopy(diagnostics),
        'assessment': {'uses': 'not_assessed', 'reactions': 'not_assessed'}}


async def attach(service, payload, debug):
    trace = service._trace_from_payload(payload)
    recall_id = trace.get('autonomy_memory_recall_id')
    if not recall_id: return
    purpose = trace.get('runtime_scope')
    if purpose not in {'internal:autonomy_chain', 'internal:self_adjustment'}:
        raise ValueError('autonomy_memory_consumer_scope_mismatch')
    prepared = service.memory_delivery.autonomy_preparation(recall_id)
    if prepared is None: raise ValueError('autonomy_memory_preparation_not_found')
    current = await read_context(service, prepared['scope'])
    if current.get('status') != 'prepared':
        raise ValueError('autonomy_memory_preparation_no_longer_current')
    if current.get('model_runtime_scope', 'internal:autonomy_chain') != purpose:
        raise ValueError('autonomy_memory_consumer_scope_mismatch')
    if purpose == 'internal:self_adjustment' and (trace.get('request_id') != prepared['scope']['cycle_id']
            or current.get('request_id') != prepared['scope']['cycle_id']):
        raise ValueError('daily_review_memory_cycle_mismatch')
    ref = {'scope': prepared['scope'], 'trace_id': trace.get('trace_id', '')}
    trace['memory_delivery_manifest'] = {'scope': prepared['scope'],
        'fragments': prepared['fragments'], 'request': ref}
    payload['_ombre_trace_context'] = trace
    debug['_memory_delivery_request'] = ref


def validate_usage_source(query, actual):
    from memory_delivery import sha
    source = actual.get('source') or {}
    events = source.get('source_events')
    if actual.get('status') != 'completed' or not isinstance(events, dict) or query.get('source_events') != events:
        raise ValueError('original_autonomy_result_required')
    by_id = {value: source[key] for key, value in events.items()}
    for use in query.get('uses') or []:
        for ref in [*(use.get('evidence_refs') or []), *(use.get('new_episode_evidence_refs') or [])]:
            text = by_id.get(ref.get('event_id'))
            start, end = ref.get('start'), ref.get('end')
            if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text) or sha(text[start:end]) != ref.get('text_sha256')):
                raise ValueError('autonomy_memory_use_source_mismatch')
