"""Native planning recall requires a successful provider wire and an exact source."""
import asyncio
from types import SimpleNamespace

import pytest
import httpx

import autonomy_memory
from gateway import GatewayService
from memory_delivery import MemoryDeliveryStore, candidates, final_fragments, sha
from attachment_adoption_support import gateway_service


SCOPE = {'consumer_kind': 'autonomy', 'conversation_id': 'conversation-1',
    'identity_id': 'owner-1', 'activity_id': None, 'activity_revision': None,
    'source_scope': '["owner-1","conversation-1"]', 'cycle_id': 'cycle-1',
    'operation_id': 'operation-1'}
TEXT = 'Prior research found a sample-size limitation.'


def projection():
    row = {'memory_id': 'memory-1', 'revision': 1,
        'document': {'evidence_bindings': [{'binding_id': 'binding-1',
            'source_namespace': 'snapshot.aiz.action_receipt', 'source_id': 'action-1',
            'source_revision': 'revision-1'}]}}
    result = {'rendered_memory_spans': [{'bucket_id': 'bucket-1', 'section': 'direct', 'text': TEXT}],
        'presentation_trace': [{'bucket_id': 'bucket-1', 'memory_id': 'memory-1',
            'l1_revision': 1, 'presentation_revision': 1}],
        'compiled_memory_blocks': [{'source_id': 'ombre.memory_recall',
            'content': 'Memory: ' + TEXT}]}
    result['delivery_candidates'] = candidates(result, {'bucket-1': row})
    return result


def run(coro):
    return asyncio.run(coro)


def test_native_recall_preparation_is_not_delivery_and_successful_wire_is(tmp_path, monkeypatch):
    store = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    async def source(_service, scope):
        return {'state': 'available', 'scope': scope, 'status': 'prepared',
            'query': 'Continue the sample-size analysis', 'request_id': 'cycle-1'}
    async def prepare_payload(payload, _conversation, **kwargs):
        assert kwargs['autonomy_context']['scope']['conversation_id'] == SCOPE['conversation_id']
        return payload, None, {'memory_recall_projection': projection()}
    monkeypatch.setattr(autonomy_memory, 'read_context', source)
    service = SimpleNamespace(prepare_payload=prepare_payload, memory_delivery=store,
        _trace_from_payload=lambda payload: payload['_ombre_trace_context'])
    prepared = run(autonomy_memory.prepare(service, SCOPE, model='owner-model'))
    assert prepared['memory_messages'] and prepared['recall_id']
    assert store.lookup(SCOPE)['state'] == 'unavailable'

    trace = {'runtime_scope': 'internal:autonomy_chain',
        'autonomy_memory_recall_id': prepared['recall_id'], 'trace_id': 'wire-1'}
    payload = {'messages': prepared['memory_messages'], '_ombre_trace_context': trace}
    debug = {}
    run(autonomy_memory.attach(service, payload, debug))
    GatewayService._begin_memory_delivery(service, payload, {'messages': []})
    assert store.lookup(SCOPE)['state'] == 'unavailable'
    # A successful provider response without a memory block proves no delivery.
    GatewayService._succeed_memory_delivery(service, payload)
    assert store.lookup(SCOPE)['context']['fragments'] == []

    other_scope = {**SCOPE, 'cycle_id': 'cycle-2', 'operation_id': 'operation-2'}
    prepared2 = run(autonomy_memory.prepare(service, other_scope, model='owner-model'))
    payload2 = {'messages': prepared2['memory_messages'], '_ombre_trace_context': {
        'runtime_scope': 'internal:autonomy_chain',
        'autonomy_memory_recall_id': prepared2['recall_id'], 'trace_id': 'wire-2'}}
    run(autonomy_memory.attach(service, payload2, {}))
    GatewayService._begin_memory_delivery(service, payload2, {'messages': [
        {'role': 'system', 'content': prepared2['memory_messages'][0]['content']}]})
    assert store.lookup(other_scope)['state'] == 'unavailable'
    GatewayService._succeed_memory_delivery(service, payload2)
    delivered = store.lookup(other_scope)['context']
    assert len(delivered['fragments']) == 1
    assert delivered['fragments'][0]['text_sha256'] == sha(TEXT)
    assert store.complete(other_scope) == delivered


def test_scope_rejects_missing_or_mismatched_activity_revision():
    with pytest.raises(ValueError):
        autonomy_memory.scope_from({**SCOPE, 'activity_id': 'activity-1'})
    with pytest.raises(ValueError):
        autonomy_memory.scope_from({**SCOPE, 'activity_revision': 1})


def test_internal_http_recall_route_requires_auth_and_server_prepared_scope(tmp_path, monkeypatch):
    service, _ = gateway_service(lambda _: httpx.Response(200, json={}))
    service.memory_delivery = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    async def source(_service, scope):
        return {'state': 'available', 'scope': scope, 'status': 'prepared',
            'query': 'Continue the actual activity', 'request_id': scope['cycle_id']}
    async def prepare_payload(payload, _conversation, **kwargs):
        return payload, None, {'memory_recall_projection': projection()}
    monkeypatch.setattr(autonomy_memory, 'read_context', source)
    monkeypatch.setattr(service, 'prepare_payload', prepare_payload)
    from gateway import create_gateway_app
    app = create_gateway_app(config={'gateway': {}}, service=service)
    app.state.gateway_service = service
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                base_url='http://local.test') as client:
            route = '/internal/autonomy/memory-recall'
            request = {'scope': SCOPE, 'model': 'owner-model'}
            assert (await client.post(route, json=request)).status_code == 401
            response = await client.post(route, json=request,
                headers={'Authorization': 'Bearer gateway-test-token'})
            assert response.status_code == 200
            assert response.json()['memory_messages']
            assert service.memory_delivery.lookup(SCOPE)['state'] == 'unavailable'
        await service.http_client.aclose()
    run(scenario())

def test_real_prepare_native_query_alias_smoke(tmp_path, monkeypatch):
    from test_gateway_sdk_tool_phase import minimal_service
    real_client = httpx.AsyncClient
    service = minimal_service(tmp_path)
    from test_memory_recall_projection_r2 import _fixture
    _, recall_service, view, buckets, _ = _fixture(tmp_path / 'memory', monkeypatch)
    service.__dict__.update(recall_service.__dict__)
    from recall_policy import RecallPolicy
    service.recall_policy = RecallPolicy()
    import jieba
    if not hasattr(jieba, 'lcut'):
        monkeypatch.setattr(jieba, 'lcut', lambda text, **kw: list(jieba.cut(text)), raising=False)
    service.persona_engine.enabled = False
    service.recalled_budget = 4000
    service.memory_detail_recall_budget = 4000
    service.recent_budget = 0
    service.retrieval_mode = 'bucket'
    async def listed(**kwargs): return buckets
    service._list_gateway_buckets = listed
    from gateway import memory_use
    monkeypatch.setattr(memory_use.httpx, 'AsyncClient', real_client)
    from test_memory_recall_gateway_r2 import _install_source_http
    _install_source_http(monkeypatch, presentations={'memory:bucket-l1': {'state': 'ready', 'text': 'L2_ACTUAL_BODY', 'presentation_revision': 1}})
    async def select(*args, **kwargs):
        return [buckets[0]], [], {}
    service._select_dynamic_buckets = select
    for name in ('_build_injected_context_messages', '_format_recalled_moments', '_direct_moments_for_bucket', '_source_record_synthetic_moment_for_bucket'):
        setattr(service, name, getattr(GatewayService, name).__get__(service))
    service._rejected_bucket_evidence = GatewayService._rejected_bucket_evidence
    service.memory_delivery = MemoryDeliveryStore(tmp_path / 'delivery.sqlite')
    resolved = []
    service._get_upstream_for_model = lambda model: resolved.append(model)
    async def source(_service, scope):
        return {'state': 'available', 'scope': scope, 'status': 'prepared', 'query': 'source-record-l1', 'request_id': scope['cycle_id']}
    monkeypatch.setattr(autonomy_memory, 'read_context', source)
    original_prepare = service.prepare_payload
    captured = {}
    async def observe(*args, **kwargs):
        result = await original_prepare(*args, **kwargs)
        captured.update(result[2])
        return result
    service.prepare_payload = observe
    result = run(autonomy_memory.prepare(service, SCOPE, model='synthetic-alias'))
    assert resolved == ['synthetic-alias']
    assert result['memory_messages'], __import__('json').dumps(captured['memory_recall_projection'], ensure_ascii=False, default=str)
    assert service.memory_delivery.lookup(SCOPE)['state'] == 'unavailable'


    assert captured['prepare_timing_debug']['recall_input']['staged_user_event_ids'] == []
    assert captured['prepare_timing_debug']['recall_input']['current_message_count'] == 0
    assert captured['prepare_timing_debug']['recall_input']['input_source'] == 'autonomy_planning_record'
    assert captured['prepare_timing_debug']['recall_input']['current_input_complete'] is True
    assert result['scope'] == SCOPE
    body = '\n'.join(message['content'] for message in result['memory_messages'])
    assert 'L2_ACTUAL_BODY' in body
    payload = {'messages': result['memory_messages'], '_ombre_trace_context': {
        'runtime_scope': 'internal:autonomy_chain', 'autonomy_memory_recall_id': result['recall_id'], 'trace_id': 'native-wire'}}
    run(autonomy_memory.attach(service, payload, {}))
    GatewayService._begin_memory_delivery(service, payload, {'messages': []})
    GatewayService._succeed_memory_delivery(service, payload)
    assert service.memory_delivery.lookup(SCOPE)['context']['fragments'] == []
    # A second final-wire attempt with actual compiled content proves delivery.
    second_scope = {**SCOPE, 'cycle_id': 'native-second', 'operation_id': 'native-second'}
    second = run(autonomy_memory.prepare(service, second_scope, model='synthetic-alias'))
    payload['_ombre_trace_context'].update(trace_id='native-wire-content', autonomy_memory_recall_id=second['recall_id'])
    run(autonomy_memory.attach(service, payload, {}))
    GatewayService._begin_memory_delivery(service, payload, {'messages': [{'role': 'system', 'content': body}]})
    GatewayService._succeed_memory_delivery(service, payload)
    assert service.memory_delivery.lookup(second_scope)['context']['fragments']


def test_native_recall_input_never_constructs_a_user_message():
    context = {'scope': SCOPE, 'status': 'prepared', 'query': 'Continue the actual study'}
    value = autonomy_memory.recall_input(context)
    assert value['current_messages'] == [] and value['recent_messages'] == []
    assert value['metadata']['source_scope'] == SCOPE
    assert value['metadata']['current_input_complete'] is True
    assert value['q_current'] == context['query']
    with pytest.raises(ValueError):
        autonomy_memory.recall_input({**context, 'status': 'completed'})
