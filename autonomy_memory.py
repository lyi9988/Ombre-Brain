"""Native activity recall consumer; uses the original Recall and delivery store."""
from copy import deepcopy

from memory_delivery import digest, compiled_candidates

FIELDS = {'consumer_kind', 'conversation_id', 'identity_id', 'activity_id',
          'activity_revision', 'source_scope', 'cycle_id', 'operation_id'}
HEADER = 'X-Guyan-Autonomy-Memory-Recall'


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


async def prepare(service, scope, *, model):
    context = await read_context(service, scope)
    if context.get('status') != 'prepared' or not isinstance(context.get('query'), str) or not context['query'].strip():
        raise ValueError('autonomy_prepared_source_required')
    # This is a real activity search cue, never a fabricated user/chat event.
    payload = {'model': model, 'messages': [{'role': 'system', 'content': context['query']}],
        '_ombre_trace_context': {'runtime_scope': 'internal:autonomy_chain',
            'conversation_id': scope['conversation_id'], 'request_id': context.get('request_id'),
            'recall_signals': context.get('recall_signals') or {}}}
    _, _, debug = await service.prepare_payload(payload, scope['conversation_id'],
        include_debug=True, debug_detail='full', autonomy_context=context)
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
    return {'state': 'available', 'scope': scope, 'recall_id': recall_id,
        'memory_messages': messages, 'recall_status': projection.get('recall_status'),
        'assessment': {'uses': 'not_assessed', 'reactions': 'not_assessed'}}


async def attach(service, payload, debug):
    trace = service._trace_from_payload(payload)
    recall_id = trace.get('autonomy_memory_recall_id')
    if not recall_id: return
    if trace.get('runtime_scope') != 'internal:autonomy_chain':
        raise ValueError('autonomy_memory_consumer_scope_mismatch')
    prepared = service.memory_delivery.autonomy_preparation(recall_id)
    if prepared is None: raise ValueError('autonomy_memory_preparation_not_found')
    current = await read_context(service, prepared['scope'])
    if current.get('status') != 'prepared':
        raise ValueError('autonomy_memory_preparation_no_longer_current')
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
