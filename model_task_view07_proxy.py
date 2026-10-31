"""Fixed read-only task metadata on the existing Gateway Memory owner chain."""
import os

import httpx
from starlette.responses import JSONResponse


async def read_model_task_metadata(service, request):
    headers = {'Cache-Control': 'no-store'}
    if request.method != 'GET':
        return JSONResponse({'error': {'code': 'method_not_allowed'}}, status_code=405, headers=headers)
    if request.query_params:
        return JSONResponse({'error': {'code': 'invalid_model_task_query'}}, status_code=400, headers=headers)
    # Same destination and credential precedence as the existing Memory owner
    # reads/maintenance handlers. Never borrow a B1 or provider credential.
    base = os.environ.get('OMBRE_MEMORY_INDEX_ADMIN_URL', 'http://ombre-brain:8000').rstrip('/')
    token = os.environ.get('OMBRE_MEMORY_WRITE_TOKEN') or service.gateway_token
    if not base or not token:
        return JSONResponse({'error': {'code': 'model_task_view_not_configured'}}, status_code=503, headers=headers)
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.get(base + '/internal/memory-task-models', headers={'Authorization': 'Bearer ' + token})
        status = response.status_code
        if status != 200:
            code = ('model_task_view_auth_failed' if status in (401, 403)
                else 'model_task_view_not_deployed' if status == 404 else 'model_task_view_upstream_error')
            return JSONResponse({'error': {'code': code}}, status_code=status if status in (401, 403, 404) else 503, headers=headers)
        value = response.json()
        if not isinstance(value, dict) or value.get('schema') != 'brain-model-task-view07-v1':
            raise ValueError('invalid_task_metadata')
        allowed = {'schema', 'authority', 'revision', 'routes', 'effective', 'connections',
                   'last_observed', 'observation_scope', 'editor'}
        return JSONResponse({key: item for key, item in value.items() if key in allowed}, headers=headers)
    except httpx.TimeoutException:
        return JSONResponse({'error': {'code': 'model_task_view_timeout'}}, status_code=504, headers=headers)
    except (httpx.NetworkError, OSError):
        return JSONResponse({'error': {'code': 'model_task_view_connection_failed'}}, status_code=502, headers=headers)
    except (ValueError, TypeError):
        return JSONResponse({'error': {'code': 'model_task_view_invalid_response'}}, status_code=502, headers=headers)
