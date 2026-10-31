"""Owner-safe task/connection metadata over the existing Reflection resolver.

No new routing or fallback, no credential material and no model/body reads.
Configuration is not call evidence: missing observations stay unobserved.
"""
from urllib.parse import urlsplit, urlunsplit

from starlette.responses import JSONResponse


def safe_address(value):
    try:
        parts = urlsplit(str(value or ''))
        if parts.scheme not in {'http', 'https'} or not parts.hostname:
            return ''
        host = '[' + parts.hostname + ']' if ':' in parts.hostname else parts.hostname
        if parts.port:
            host += ':' + str(parts.port)
        return urlunsplit((parts.scheme, host, parts.path, '', ''))
    except (TypeError, ValueError):
        return ''


def public_view(engine):
    resolved = engine.memory_task_models_public()
    labels = {'reflection': '自动记忆连接', 'dehydration': '脱水连接', 'daily': '每日记忆任务连接'}
    clients = {'reflection': getattr(engine, 'client', None),
        'dehydration': getattr(engine, 'dehydration_client', None),
        'daily': getattr(engine, 'daily_chat_memory_client', None)}
    profiles = {}
    for name, client in clients.items():
        profiles[name] = {'connection_id': 'brain.memory:' + name, 'label': labels[name],
            'base_url': safe_address(getattr(client, 'base_url', '')) if client else '',
            'credential_configured': bool(getattr(client, 'api_key', '')) if client else None,
            'client_available': client is not None}
    fields = {'stage', 'connection', 'model', 'thinking_mode', 'reasoning_effort', 'max_tokens',
        'timeout_seconds', 'configured', 'unavailable_reason', 'model_source', 'thinking_source', 'max_tokens_source'}
    effective = {stage: {key: value for key, value in row.items() if key in fields}
                 for stage, row in resolved.get('effective', {}).items()}
    choices = {stage: {key: value for key, value in row.items() if key in
        {'connection', 'model', 'thinking_mode', 'reasoning_effort', 'max_tokens'}}
        for stage, row in resolved.get('routes', {}).items()}
    return {'schema': 'brain-model-task-view07-v1', 'authority': 'reflection.memory_task_routes',
        'revision': resolved.get('revision'), 'routes': choices, 'effective': effective, 'connections': profiles,
        'last_observed': {stage: None for stage in effective},
        'observation_scope': 'No connection-bound task-level recent receipt is supplied by this metadata getter.',
        'editor': {'path': '/dashboard', 'fragment_prefix': 'memory-task-', 'section': 'model-config'}}


def owner_response(request, engine, authorize):
    denied = authorize(request)
    if denied is not None:
        return denied
    if request.query_params:
        return JSONResponse({'error': {'code': 'invalid_model_task_query'}}, status_code=400)
    try:
        return JSONResponse(public_view(engine), headers={'Cache-Control': 'no-store'})
    except (ValueError, TypeError, AttributeError, KeyError):
        return JSONResponse({'error': {'code': 'model_task_view_unavailable'}}, status_code=503)
