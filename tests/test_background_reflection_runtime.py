"""Actual SDK wire and durable task recovery; no external provider requests."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI
import pytest

from background_reflection_runtime import BackgroundReflectionRuntime, ReflectionConfigurationSource
from chat_memory_tool import candidate_id
from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from memory_diary_progress import DiaryProgress
from reflection_engine import ReflectionEngine
from test_chat_memory_preparation import setup as setup_chat
from test_memory_authority_r1 import reflection_config
from test_memory_write_quality_routes import config_api, FakeRequest


def connection(config, tag):
    config['reflection'].update(enabled=True, base_url='https://provider-' + tag.lower() + '.invalid/v1',
        model='model-' + tag, api_key='', memory_task_routes={'chat_l1': {'connection': 'reflection'}})


def sdk_transport(monkeypatch, *, handler=None):
    import reflection_engine
    requests, clients = [], []
    for name in ('OMBRE_REFLECTION_API_KEY', 'OMBRE_EMBEDDING_API_KEY', 'OMBRE_PERSONA_API_KEY', 'OMBRE_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    async def send(request):
        requests.append({'url': str(request.url), 'authorization': request.headers.get('authorization'),
            'body': json.loads(request.content)})
        if handler:
            await handler(request)
        content = {'document': {'format': 'semantic-v1', 'entities': [],
            'statements': [{'content': '合成的当次请求', 'subject_ref': None,
                'classification': {'kind': None, 'stability': None, 'modality': None, 'basis': None},
                'time': None, 'conditions': None, 'evidence_refs': ['source:2']}],
            'retrieval': {'keywords': ['合成关键词'], 'tags': []}}, 'kind': 'shared_experience', 'confidence': .95}
        try:
            material = json.loads(json.loads(request.content)['messages'][-1]['content'])
        except (ValueError, TypeError, KeyError):
            material = {}
        if 'l1_document' in material:
            refs = [row['statement_id'] for row in material['l1_document']['statements']]
            content = {'full_injection_text': '完整合成第二遍正文。', 'coherent_sections': [
                {'section_id': 'synthetic-section', 'text': '完整合成第二遍正文。', 'statement_refs': refs}],
                'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
        return httpx.Response(200, json={'id': 'synthetic-sdk-response', 'object': 'chat.completion',
            'created': 1, 'model': json.loads(request.content)['model'], 'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(content)}, 'finish_reason': 'stop'}]})
    def client(**kwargs):
        value = AsyncOpenAI(**kwargs, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)))
        clients.append(value)
        return value
    monkeypatch.setattr(reflection_engine, 'AsyncOpenAI', client)
    return requests, clients


async def request(engine):
    route = engine._memory_task_route('chat_l1')
    assert route['configured']
    return await route['client'].chat.completions.create(model=route['model'],
        messages=[{'role': 'user', 'content': 'synthetic offline request'}])


def test_original_durable_task_uses_new_applied_connection_once_and_reuses_stable_clients(tmp_path, monkeypatch):
    env, fixture, envelope, _ = setup_chat(tmp_path)
    config = deepcopy(fixture.config)
    connection(config, 'A')
    requests, clients = sdk_transport(monkeypatch)
    monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-A')
    made = []
    def factory(config, mirror, **kwargs):
        engine = ReflectionEngine(config, mirror, **kwargs)
        engine.memory_authority_store = env['authority']
        engine._memory_l1_revisions = lambda manager=None: env['revisions']
        engine._memory_l1_sources = fixture._memory_l1_sources
        made.append(engine)
        return engine
    async def scenario():
        runtime = BackgroundReflectionRuntime(engine_factory=factory)
        tasks = ChatMemoryPreparation(env['authority'])
        original = tasks.enqueue(envelope)
        key = candidate_id(envelope['operation_id'])
        async with runtime.batch(config) as first:
            first._memory_preparation_cursor = 2
            assert tasks.get(key)['phase'] == 'queued' and not requests
            first._store_daily_chat_memory_run_audit({'date': '2026-10-25', 'status': 'completed',
                'model_call_count': 1, 'processed_until_seq': 12})
            audit_bytes = Path(first.daily_chat_memory_run_audit_path).read_bytes()
            checkpoint = DiaryProgress(first.memory_authority_store).save('original-diary-task',
                {'previous_revision': 1, 'prose': ['原已完成窗口'], 'input_ranges': [{'start': 0, 'end': 7}]},
                expected_revision=0)
        connection(config, 'B')
        monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-B')
        async with runtime.batch(config) as second:
            assert second is not first and first.client.is_closed()
            assert second._memory_preparation_cursor == 2
            assert Path(second.daily_chat_memory_run_audit_path).read_bytes() == audit_bytes
            assert DiaryProgress(second.memory_authority_store).get('original-diary-task') == checkpoint
            result = await prepare_chat_tasks(second, env['buckets'])
            assert result['results'][0]['memory_status'] == 'pending', result
            assert tasks.get(key)['task_id'] == original['task_id'] and tasks.get(key)['attempts'] == 1
        config.setdefault('gateway', {})['memory_injection_max_chars'] = 17000
        config['portrait'] = {'model': 'unrelated-portrait'}
        config['reflection']['chat_memory_jobs_per_run'] = 7
        async with runtime.batch(config) as third:
            assert third is second and not third.client.is_closed()
            await prepare_chat_tasks(third, env['buckets'])
            assert third.config['reflection']['chat_memory_jobs_per_run'] == 7
        assert len(made) == 2 and len(requests) == 1
        assert requests[0]['url'].startswith('https://provider-b.invalid/v1/')
        assert requests[0]['authorization'] == 'Bearer synthetic-key-B'
        assert requests[0]['body']['model'] == 'model-B'
        assert tasks.get(key)['attempts'] == 1
        await runtime.close()
        assert all(client.is_closed() for client in clients)
    asyncio.run(scenario())


def test_inflight_request_keeps_old_snapshot_and_next_batch_waits_before_closing(tmp_path, monkeypatch):
    async def scenario():
        began, release = asyncio.Event(), asyncio.Event()
        async def handler(_request):
            if not began.is_set():
                began.set()
                await release.wait()
        requests, clients = sdk_transport(monkeypatch, handler=handler)
        config = reflection_config(tmp_path, mode='review')
        connection(config, 'A')
        monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-A')
        runtime = BackgroundReflectionRuntime()
        engines = []
        async def batch():
            async with runtime.batch(config) as engine:
                engines.append(engine)
                await request(engine)
        first = asyncio.create_task(batch())
        await began.wait()
        connection(config, 'B')
        monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-B')
        assert engines[0]._memory_task_route('chat_l1')['model'] == 'model-A'
        second = asyncio.create_task(batch())
        await asyncio.sleep(0)
        assert len(engines) == 1 and not clients[0].is_closed()
        release.set()
        await asyncio.gather(first, second)
        assert clients[0].is_closed() and engines[0] is not engines[1]
        assert [(r['body']['model'], r['authorization']) for r in requests] == [
            ('model-A', 'Bearer synthetic-key-A'), ('model-B', 'Bearer synthetic-key-B')]
        await runtime.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('connection_name', ['reflection', 'daily'])
def test_removed_named_key_leaves_explicit_route_unavailable_without_borrowing(tmp_path, monkeypatch, connection_name):
    requests, _clients = sdk_transport(monkeypatch)
    config = reflection_config(tmp_path, mode='review')
    connection(config, 'A')
    named = 'OMBRE_REFLECTION_API_KEY' if connection_name == 'reflection' else 'SYNTHETIC_DAILY_KEY'
    monkeypatch.setenv(named, 'synthetic-key-A')
    config['embedding'] = {'base_url': 'https://fallback.invalid/v1', 'api_key': 'synthetic-unrelated-key'}
    if connection_name == 'daily':
        config['reflection'].update(daily_chat_memory_api_key_env=named, daily_chat_memory_base_url='https://daily.invalid/v1',
            daily_chat_memory_candidate_model='daily-model', memory_task_routes={'chat_l1': {'connection': 'daily'}})
    async def scenario():
        runtime = BackgroundReflectionRuntime()
        async with runtime.batch(config) as first:
            assert first._memory_task_route('chat_l1')['configured']
        monkeypatch.delenv(named)
        async with runtime.batch(config) as second:
            assert first is not second
            route = second._memory_task_route('chat_l1')
            assert not route['configured'] and route['unavailable_reason'] == 'connection_unconfigured'
            with pytest.raises(RuntimeError, match='connection_unconfigured'):
                await second._daily_chat_memory_create_completion(None, model='', messages=[], max_tokens=1,
                    temperature=.1, use_daily_client=False, task_stage='chat_l1', task_route=route)
        assert not requests
        await runtime.close()
    asyncio.run(scenario())


def test_config_save_midpoint_never_combines_new_url_with_old_named_key(tmp_path, monkeypatch):
    async def scenario():
        requests, _clients = sdk_transport(monkeypatch)
        config = reflection_config(tmp_path, mode='review')
        connection(config, 'A')
        monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-A')
        source = ReflectionConfigurationSource()
        midpoint, finish = asyncio.Event(), asyncio.Event()
        @source.applied_update(lambda: config)
        async def owner_save():
            config['reflection']['base_url'] = 'https://provider-b.invalid/v1'
            midpoint.set()
            await finish.wait()
            config['reflection']['model'] = 'model-B'
            monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-B')
        runtime = BackgroundReflectionRuntime(configuration_source=source)
        saving = asyncio.create_task(owner_save())
        await midpoint.wait()
        async with runtime.batch(config) as first:
            await request(first)
        finish.set()
        await saving
        async with runtime.batch(config) as second:
            await request(second)
        assert [(r['url'].split('/')[2], r['body']['model'], r['authorization']) for r in requests] == [
            ('provider-a.invalid', 'model-A', 'Bearer synthetic-key-A'),
            ('provider-b.invalid', 'model-B', 'Bearer synthetic-key-B')]
        await runtime.close()
    asyncio.run(scenario())


def test_all_three_background_entrypoints_use_loop_owned_batch_and_teardown():
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'server.py').read_text(encoding='utf-8'))
    for name in ('_reflection_loop', '_chat_memory_preparation_loop', '_memory_l2_preparation_loop'):
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
        assert any(isinstance(node, ast.AsyncWith) and any(isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == 'batch'
            for item in node.items) for node in ast.walk(function)), name
        assert any(isinstance(node, ast.Try) and node.finalbody and 'close' in ast.unparse(node.finalbody)
            for node in ast.walk(function)), name
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'ReflectionEngine' for node in ast.walk(function)), name


def test_actual_owner_config_save_reaches_next_background_sdk_request(config_api, monkeypatch):
    server, config = config_api.server, config_api.config
    requests, clients = sdk_transport(monkeypatch)
    connection(config, 'A')
    monkeypatch.setenv('OMBRE_REFLECTION_API_KEY', 'synthetic-key-A')
    async def no_gateway(_payload, **_kwargs):
        return server.activation_receipt(state='not_configured', reason='synthetic-no-gateway')
    monkeypatch.setattr(server, '_hot_update_gateway_config', no_gateway)
    async def scenario():
        runtime = BackgroundReflectionRuntime(server.prompt_plan_mirror,
            configuration_source=server.reflection_configuration_source)
        async with runtime.batch(config) as first:
            assert first._memory_task_route('chat_l1')['model'] == 'model-A'
        saved = await server.api_config_update(FakeRequest({'reflection': {
            'base_url': 'https://provider-b.invalid/v1', 'model': 'model-B',
            'api_key': 'synthetic-key-B'}, 'persist': True}))
        assert saved.status_code == 200, saved.body
        assert 'provider-b.invalid' in config_api.runtime_config_path.read_text(encoding='utf-8')
        async with runtime.batch(config) as second:
            assert second is not first and second is not server.reflection_engine
            await request(second)
        assert len(requests) == 1
        assert requests[0]['url'].startswith('https://provider-b.invalid/v1/')
        assert requests[0]['authorization'] == 'Bearer synthetic-key-B'
        assert requests[0]['body']['model'] == 'model-B'
        await runtime.close()
        for client in clients:
            await client.close()
    asyncio.run(scenario())


def test_l2_worker_reuses_durable_queue_and_new_dehydration_client_after_first_batch(tmp_path, monkeypatch):
    from test_memory_l2_dependency_scope_r2 import _persist_source_bound_l1
    from memory_l2_service import MemoryL2Service
    service, env, _old_calls = _persist_source_bound_l1(tmp_path / 'source')
    config = reflection_config(tmp_path / 'worker', mode='review')
    config['reflection'].update(enabled=True, memory_task_routes={'presentation_l2': {'connection': 'dehydration'}})
    config['dehydration'] = {'base_url': 'https://provider-a.invalid/v1', 'model': 'l2-A', 'api_key': 'synthetic-key-A'}
    requests, clients = sdk_transport(monkeypatch)
    def factory(config, mirror, **kwargs):
        engine = ReflectionEngine(config, mirror, **kwargs)
        engine.memory_authority_store = env['authority']
        engine._memory_l1_sources = lambda: env['sources']
        return engine
    async def scenario():
        runtime = BackgroundReflectionRuntime(engine_factory=factory)
        async with runtime.batch(config) as first:
            first_l2 = MemoryL2Service(first, env['revisions'])
            assert (await first_l2.run_once(1))['results'][0]['state'] == 'ready'
            original = first_l2.detail('m1')['l2']['presentation']
        detail = await env['revisions'].detail('m1')
        edit = {'expected_revision': detail['current_revision'], 'request_id': 'actual-next-l1',
                'document': deepcopy(detail['l1']['document'])}
        edit['document']['narrative']['body'] += ' 本次局部修订。'
        await env['revisions'].save('m1', edit)
        queued = service.store.pending(10)
        assert queued
        config['dehydration'].update(base_url='https://provider-b.invalid/v1', model='l2-B', api_key='synthetic-key-B')
        async with runtime.batch(config) as second:
            assert second is not first and first.dehydration_client.is_closed()
            second_l2 = MemoryL2Service(second, env['revisions'])
            assert second_l2.store.pending(10)[0]['task_id'] == queued[0]['task_id']
            result = await second_l2.run_once(1)
            assert result['results'][0]['state'] == 'ready', result
            assert second_l2.detail('m1')['l2']['presentation']['first_pass_revision'] == 3
            assert second_l2.detail('m1', original['presentation_revision'])['l2']['presentation'] == original
            await second_l2.run_once(1)
        assert [(r['body']['model'], r['authorization']) for r in requests] == [
            ('l2-A', 'Bearer synthetic-key-A'), ('l2-B', 'Bearer synthetic-key-B')]
        assert requests[1]['url'].startswith('https://provider-b.invalid/v1/')
        await runtime.close()
        assert all(client.is_closed() for client in clients)
    asyncio.run(scenario())
