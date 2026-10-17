import copy
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import httpx
from starlette.requests import Request

from gateway import GatewayService, CANONICAL_SOURCE_EVENT_ID_HEADER, create_gateway_app
import memory_recall_runtime as runtime
from memory_affinity import resolve_signals
from memory_system_config import REQUEST_CONFIGURATION
from persona_engine import PersonaStateEngine
from test_memory_affinity import NOW, settings, snapshot, statement
from memory_l1_schema import document_template
from test_memory_configuration_api import service_at


def test_real_persona_read_exposes_only_sampled_signals_with_source_time(tmp_path):
    engine = PersonaStateEngine({'state_dir': str(tmp_path), 'buckets_dir': str(tmp_path),
        'persona': {'enabled': True, 'mode': 'test', 'profile_id': 'p', 'canonical_session_id': 's'}})
    state = engine.get_current_state('s')
    source = state['recall_signal_source']
    assert source['producer'] == 'ombre.persona' and source['state'] == 'available'
    assert 'inner_thought' not in str(source) and 'residue' not in str(source)
    bound = resolve_signals(settings(), [source], datetime.now(timezone.utc))
    assert bound['care']['state'] == 'known'
    assert bound['care']['value'] == pytest.approx(state['affect']['protective_drive'], abs=.0005)


def test_real_trace_parser_checks_scope_and_upstream_never_receives_signals():
    service = GatewayService.__new__(GatewayService)
    scope = dict(conversation_id='conv1', turn_id='turn1', source_event_id='event1')
    raw = json.dumps({'schema_version': 'recall-signals-v1', 'scope': scope,
                     'snapshots': [snapshot(.9, producer='aiz.needs', key='curiosity')]})
    headers = {'X-Guyan-Recall-Signals': raw, 'X-Guyan-Conversation-Id': 'conv1',
               'X-Guyan-Turn-Id': 'turn1', CANONICAL_SOURCE_EVENT_ID_HEADER: 'event1'}
    request = Request({'type': 'http', 'headers': [(key.lower().encode(), value.encode()) for key, value in headers.items()]})
    trace = service._trace_context_from_request(request, session_id='s', request_type='initial', client_id='reality')
    assert trace['recall_signals']['state'] == 'received'
    forwarded = service._payload_for_upstream_model({'messages': [{'role': 'user', 'content': 'ordinary chat'}],
        '_ombre_trace_context': trace}, 'provider-model')
    assert 'recall_signals' not in str(forwarded)
    assert forwarded['messages'] == [{'role': 'user', 'content': 'ordinary chat'}]


def test_request_context_freezes_source_and_owner_formula_uses_it_before_and_after_rerank():
    values = settings()
    service = SimpleNamespace(config={'memory_system': values}, recall_policy=SimpleNamespace(),
                              memory_authority_view=None)
    snapshots = [snapshot()]
    request = {'_signal_snapshots': snapshots, '_signal_sampled_at': NOW, 'allow_semantic': False}
    document = document_template('synthetic mixed memory', catalog=values['catalog'])
    document['statements'] = [statement(['health'])]
    candidate = {'bucket': {'id': 'm1'}, 'score': .5, 'keyword_score': .5}
    request['_semantic_records'] = {'m1': {'document': document}}
    context = runtime.context(service, request, 'ordinary chat')
    snapshots[0]['signals']['protective_drive']['value'] = 1
    runtime.score_pool(service, request, 'ordinary chat', [candidate])
    assert candidate['score'] == pytest.approx(.7)
    candidate['_raw_fused_score'] = .4
    runtime.score_pool(service, request, 'ordinary chat', [candidate], after_rerank=True)
    assert candidate['score'] == pytest.approx(.56)
    assert context['signals']['care']['value'] == .5
    assert candidate['_configured_score']['statements'][0]['affinity']['matched_rules'] == ['health_care']


def test_hot_updated_mapping_applies_next_request_without_mutating_frozen_request():
    values = settings()
    service = SimpleNamespace(config={'memory_system': values}, recall_policy=SimpleNamespace())
    request = {'_signal_snapshots': [snapshot()], '_signal_sampled_at': NOW}
    document = document_template('synthetic memory', catalog=values['catalog'])
    document['statements'] = [statement(['health'])]
    request['_semantic_records'] = {'m1': {'document': document}}
    candidate = lambda: {'bucket': {'id': 'm1'}, 'score': .5}
    token = REQUEST_CONFIGURATION.set({'service': id(service), 'values': copy.deepcopy(values), 'revision': 'r1'})
    try:
        original = candidate()
        runtime.score_pool(service, request, 'q', [original])
        values['recall']['affinity_mappings'][0]['weight'] = .2
        frozen = candidate()
        runtime.score_pool(service, request, 'q', [frozen])
        assert frozen['score'] == original['score'] == pytest.approx(.7)
    finally:
        REQUEST_CONFIGURATION.reset(token)
    next_request = {key: copy.deepcopy(value) for key, value in request.items() if key != '_use_context'}
    changed = candidate()
    runtime.score_pool(service, next_request, 'q', [changed])
    assert changed['score'] == pytest.approx(.55)


def test_owner_configuration_routes_save_readback_and_preview_actual_affinity(tmp_path):
    service = service_at(tmp_path / 'overlay.yaml')
    service.gateway_tz = timezone.utc
    app = create_gateway_app(config=service.config, service=service)
    app.state.gateway_service = service
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            path = '/api/memory-authority/configuration'
            assert (await client.get(path)).status_code == 401
            headers = {'Authorization': 'Bearer owner'}
            before = (await client.get(path, headers=headers)).json()
            body = {'expected_revision': before['revision'], 'request_id': 'affinity-route-save', 'values': settings()}
            saved = await client.put(path, json=body, headers=headers)
            assert saved.status_code == 200
            after = (await client.get(path, headers=headers)).json()
            assert saved.json()['saved_revision'] == after['revision']
            assert after['values']['recall']['affinity_mappings'] == settings()['recall']['affinity_mappings']
            raw_overlay = (tmp_path / 'overlay.yaml').read_bytes()
            result = await client.post(path + '/preview', headers=headers, json={
                'values': after['values'], 'inputs': {'relevance': .5},
                'scenario': {'now': NOW.isoformat(), 'topics': ['health'], 'signal_snapshots': [snapshot()]}})
            assert result.status_code == 200
            assert result.json()['result']['pipeline']['score'] == pytest.approx(.7)
            assert result.json()['result']['provider_requests'] == 0
            assert (tmp_path / 'overlay.yaml').read_bytes() == raw_overlay
            assert (await client.put(path, json={**body, 'request_id': 'old-revision'}, headers=headers)).status_code == 409
    asyncio.run(run())
