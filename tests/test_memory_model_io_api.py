"""The owner route exposes complete captured calls, never to writer/read tokens."""
import ast
import asyncio
import json
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from memory_authority import MemoryAuthorityStore
from memory_model_io import MemoryModelIO


def app_for(authority):
    path = Path(__file__).resolve().parents[1] / 'server.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == 'api_memory_model_io')
    route = ast.literal_eval(node.decorator_list[0].args[0])
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    def owner(request):
        if request.headers.get('authorization') != 'Bearer owner':
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
    scope = {'memory_authority_store': authority, '_require_dashboard_auth': owner}
    exec(compile(module, str(path), 'exec'), scope)
    return Starlette(routes=[Route(route, scope['api_memory_model_io'], methods=['GET'])])


def get(app, path, token='owner'):
    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            return await client.get(path, headers={'Authorization': 'Bearer ' + token})
    return asyncio.run(call())


def test_owner_route_full_call_pending_memory_and_other_tokens_denied(tmp_path):
    authority = MemoryAuthorityStore({'state_dir': str(tmp_path)})
    recorder = MemoryModelIO(authority)
    text = 'Complete visible content. ' * 2000
    recorder.start('pending-task', stage='chat_l1', task_id='pending-task', attempt=1,
                   messages=[{'role': 'user', 'content': text}], parameters={'model': 'test-model'})
    recorder.start('other-memory', stage='chat_l1', task_id='other-memory', attempt=1,
                   messages=[{'role': 'user', 'content': 'other private body'}], parameters={'model': 'test-model'})
    app = app_for(authority)
    url = '/api/memory-authority/memories/pending-task/model-io'
    for token in ('', 'writer', 'internal-read'):
        assert get(app, url, token).status_code == 401
    result = get(app, url)
    assert result.status_code == 200
    assert result.headers['cache-control'] == 'no-store'
    assert len(result.json()['items']) == 1
    assert result.json()['items'][0]['messages'][0]['content'] == text
    assert 'other private body' not in result.text


def test_model_io_route_empty_legacy_and_pagination_validation(tmp_path):
    app = app_for(MemoryAuthorityStore({'state_dir': str(tmp_path)}))
    url = '/api/memory-authority/memories/legacy/model-io'
    result = get(app, url)
    assert result.status_code == 200
    assert result.json()['capture_state'] == 'missing'
    assert result.json()['items'] == []
    for query in ('limit=201', 'limit=0', 'offset=-1', 'limit=x', 'limit=2&limit=3', 'secret=1'):
        assert get(app, url + '?' + query).status_code == 400
