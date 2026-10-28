"""Read and verify native activity result sources, never canonical chat pairs."""
import os
import httpx
from b1_protocol import digest

SCOPE_FIELDS = {'consumer_kind', 'conversation_id', 'identity_id', 'activity_id',
                'activity_revision', 'source_scope', 'cycle_id', 'operation_id'}
ROLES = ('planning_output', 'step_outcome')


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != SCOPE_FIELDS or scope['consumer_kind'] != 'autonomy':
        raise ValueError('native autonomy scope required')
    for key in SCOPE_FIELDS - {'activity_id', 'activity_revision'}:
        if not isinstance(scope[key], str) or not scope[key] or len(scope[key]) > 400:
            raise ValueError('native autonomy identity required')
    if scope['activity_id'] is None:
        if scope['activity_revision'] is not None: raise ValueError('activity revision mismatch')
    elif (not isinstance(scope['activity_id'], str) or not scope['activity_id']
          or type(scope['activity_revision']) is not int or scope['activity_revision'] < 1):
        raise ValueError('activity revision required')
    return scope


def source_hash(source):
    if (not isinstance(source, dict) or set(source) != {'schema', 'scope', 'request_id',
            'input_sha256', 'planning_output', 'step_outcome', 'source_events'}
            or source['schema'] != 'autonomy-memory-source-v1'
            or not isinstance(source.get('scope'), dict)
            or source.get('request_id') != source['scope'].get('cycle_id')
            or not isinstance(source.get('input_sha256'), str)
            or len(source['input_sha256']) != 64
            or any(char not in '0123456789abcdef' for char in source['input_sha256'])
            or any(not isinstance(source[r], str) for r in ROLES)
            or not isinstance(source['source_events'], dict)
            or not {'planning_output'} <= set(source['source_events']) <= set(ROLES)
            or len(set(source['source_events'].values())) != len(source['source_events'])):
        raise ValueError('native autonomy memory source required')
    validate_scope(source['scope'])
    for role in ROLES:
        if not source[role] and role not in source['source_events']: continue
        if not isinstance(source['source_events'].get(role), str) or not source['source_events'][role]:
            raise ValueError('original planning result reference required')
    return digest(source)


async def read_source(config, scope, *, client=None):
    validate_scope(scope)
    options = (config.get('gateway') or {}).get('canonical_continuation') or {}
    token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
    if (not options.get('enabled') or not options.get('base_url') or not token
            or options.get('conversation_id') != scope['conversation_id']):
        raise ValueError('autonomy source reader not configured for scope')
    params = {'cycle_id': scope['cycle_id'], 'operation_id': scope['operation_id']}
    headers = {'Authorization': 'Bearer ' + token,
        'X-Guyan-Device-ID': str(options.get('device_id') or 'ombre-gateway-operit')}
    url = str(options['base_url']).rstrip('/') + '/bridge/v1/autonomy-memory-context'
    if client is not None:
        response = await client.get(url, params=params, headers=headers)
    else:
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as reader:
            response = await reader.get(url, params=params, headers=headers)
    response.raise_for_status()
    result = response.json()
    if (result.get('state') != 'available' or result.get('scope') != scope
            or result.get('status') != 'completed' or result.get('source', {}).get('scope') != scope
            or result.get('source_hash') != source_hash(result.get('source'))):
        raise ValueError('autonomy source not current or exact')
    return result['source']
