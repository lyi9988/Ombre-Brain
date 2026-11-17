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
from memory_l1_contract import L1ContractError
from memory_l1_service import L1OwnerError
import memory_model_io
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
    io_id = recorder.start('pending-task', stage='chat_l1', task_id='pending-task', attempt=1,
                          messages=[{'role': 'user', 'content': text}], parameters={'model': 'test-model'})
    recorder.returned(io_id, response={}, raw_output='{}', finish_reason='stop')
    recorder.finish(io_id, error=L1ContractError('nonempty_string_required', 'document.statements[0].content'))
    other_id = recorder.start('other-memory', stage='memory_l3_sync', task_id='other-memory', attempt=1,
                              messages=[{'role': 'user', 'content': 'other private body'}],
                              parameters={'model': 'test-model'})
    recorder.returned(other_id, response={}, raw_output='{}', finish_reason='stop')
    pointer = '/statements/st-test/content'
    recorder.finish(other_id, error=L1OwnerError('l3_patch_member_not_found', field_path=pointer))
    app = app_for(authority)
    url = '/api/memory-authority/memories/pending-task/model-io'
    for token in ('', 'writer', 'internal-read'):
        assert get(app, url, token).status_code == 401
    result = get(app, url)
    assert result.status_code == 200
    assert result.headers['cache-control'] == 'no-store'
    assert len(result.json()['items']) == 1
    record = result.json()['items'][0]
    assert record['messages'][0]['content'] == text
    assert record['stage'] == 'chat_l1' and record['status'] == 'failed'
    assert record['error'] == record['validation']['issue'] == {
        'code': 'nonempty_string_required', 'error_type': 'L1ContractError',
        'field_path': 'document.statements[0].content', 'failure_phase': 'response_processing',
    }
    assert 'other private body' not in result.text
    other = get(app, '/api/memory-authority/memories/other-memory/model-io').json()['items'][0]
    assert other['error']['field_path'] == pointer


def test_model_io_route_empty_legacy_and_pagination_validation(tmp_path):
    app = app_for(MemoryAuthorityStore({'state_dir': str(tmp_path)}))
    url = '/api/memory-authority/memories/legacy/model-io'
    result = get(app, url)
    assert result.status_code == 200
    assert result.json()['capture_state'] == 'missing'
    assert result.json()['items'] == []
    for query in ('limit=201', 'limit=0', 'offset=-1', 'limit=x', 'limit=2&limit=3', 'secret=1'):
        assert get(app, url + '?' + query).status_code == 400


def test_model_io_v2_expiry_and_order_preserve_legacy_records(tmp_path, monkeypatch):
    authority = MemoryAuthorityStore({'state_dir': str(tmp_path)})
    recorder = MemoryModelIO(authority)
    monkeypatch.setattr(memory_model_io, '_v2_cutoff', lambda: '2026-11-11T00:00:00.000000+00:00')
    stamps = ('2026-11-10T23:59:59.000000+00:00', '2026-11-17T10:00:00.000000+00:00',
              '2026-11-17T11:00:00.000000+00:00', '2026-11-17T12:00:00.000000+00:00')
    times = iter(stamps)
    monkeypatch.setattr(memory_model_io, '_stamp', lambda: next(times))
    params = {'stage': 'action_l1', 'task_id': 'task', 'attempt': 1, 'messages': [], 'parameters': {'model': 'x'}}
    expired = recorder.start('task', part=1, **params)
    older = recorder.start('task', **{**params, 'stage': 'presentation_l2'})
    newer = recorder.start('task', part=1, **params)
    recorder.requested(newer, {'model': 'x', 'input': [{'role': 'user', 'content': 'reading input'}],
                               'text': {'format': {'type': 'json_object'}}, 'api_key': 'secret'}, timeout_seconds=5)
    recorder.finish(newer, error=TimeoutError('private provider details'))
    assert authority.get_meta(expired) is None and authority.get_meta(older) is not None
    app = app_for(authority)
    uri = '/api/memory-authority/memories/task/model-io'
    first = get(app, uri + '?limit=1').json()
    assert first['next_offset'] == 1 and len(first['items']) == 1
    record = first['items'][0]
    assert record['io_id'] == newer and record['status'] == 'failed'
    assert record['messages'] == record['native_request']['input']
    assert 'api_key' not in record['native_request']
    assert record['raw_output'] is None and record['error']['code'] == 'TimeoutError'
    assert get(app, uri + '?limit=1&offset=1').json()['items'][0]['io_id'] == older
