"""Native planning evidence settles memory without a fabricated chat turn."""
import asyncio
from copy import deepcopy

import pytest
import httpx
from starlette.applications import Starlette

from autonomy_memory import source_hash
from b1_protocol import digest
from memory_authority import MemoryAuthorityStore
from memory_l1_source_tasks import MemoryL1SourceTasks
from persona_memory_reaction_context import memory_ref, reaction_id
from test_stage03_memory_reactions import delivered, prepare
from test_persona_inner_state_v2 import source as chat_source, WHEN


SCOPE = {'consumer_kind': 'autonomy', 'conversation_id': 'conversation-1',
    'identity_id': 'owner-1', 'activity_id': None, 'activity_revision': None,
    'source_scope': '["owner-1","conversation-1"]', 'cycle_id': 'cycle-1',
    'operation_id': 'operation-1'}


def fixture():
    text = 'I will compare the two studies by sample size.'
    source = {'schema': 'autonomy-memory-source-v1', 'scope': deepcopy(SCOPE),
        'request_id': 'cycle-1', 'input_sha256': 'a' * 64,
        'planning_output': text, 'step_outcome': '',
        'source_events': {'planning_output': 'planning:cycle-1:3'}}
    context, _ = delivered(chat_source())
    context['scope'] = deepcopy(SCOPE)
    context['delivery_sha256'] = digest({k: context[k] for k in ('schema', 'scope', 'fragments')})
    span = {'role': 'planning_output', 'start': 0, 'end': len(text),
        'evidence_hash': digest(text)}
    ref = memory_ref(context['fragments'][0])
    row = {'reaction': 'new_understanding', 'dimension': 'longing',
        'effect': 'evoke', 'direction': 'increase', 'strength': 'clear',
        'memory_ref': ref, 'evidence_spans': [span], 'interaction_observation_ids': []}
    row['evidence_hash'] = digest({'source_hash': source_hash(source),
        'evidence_spans': [span], 'memory_ref': ref})
    row['reaction_id'] = reaction_id(source_hash(source), row)
    envelope = {'namespace': 'b1-autonomy-memory-v1', 'scope': 'fixture',
        'event_id': 'autonomy-memory:cycle-1', 'occurred_at': WHEN,
        'policy_revision': 'fixture-inner-state-v2',
        'decision': {'schema': 'b1-decision-v2', 'accepted': [], 'rejected': [],
            'source_hash': source_hash(source), 'interpretation_is_not_ground_truth': True},
        'labels': [], 'memory_context': context,
        'memory_reactions': {'schema': 'memory-reactions-v1', 'observations': [row]}}
    return source, envelope


def test_real_native_reaction_and_replay_use_original_ledger(tmp_path):
    store, adapter, _ = prepare(tmp_path)
    source, envelope = fixture()
    async def reader(scope):
        assert scope == SCOPE
        return deepcopy(source)
    async def resolver(context, actual):
        assert actual == source
        return {'state': 'available', 'delivery_receipt_id': context['delivery_receipt_id'],
            'delivery_sha256': context['delivery_sha256'],
            'eligible_fragment_ids': [context['fragments'][0]['fragment_id']]}
    initial = store.read('fixture')['values']['longing']
    first = asyncio.run(adapter.settle_autonomy_memory(envelope, source,
        source_reader=reader, resolver=resolver))['b1_receipt']
    assert first['receipt']['after']['longing'] == pytest.approx(initial + .05)
    assert first['receipt']['source_kind'] == 'autonomy'
    assert first['receipt']['source_refs'][0]['kind'] == 'operation'
    again = asyncio.run(adapter.settle_autonomy_memory(envelope, source,
        source_reader=reader, resolver=resolver))['b1_receipt']
    assert again['replayed'] is True


def test_rejects_chat_role_and_changed_original_source(tmp_path):
    _, adapter, _ = prepare(tmp_path)
    source, envelope = fixture()
    async def reader(_scope): return deepcopy(source)
    async def resolver(context, _source):
        return {'state': 'available', 'delivery_receipt_id': context['delivery_receipt_id'],
            'delivery_sha256': context['delivery_sha256'], 'eligible_fragment_ids': []}
    forged = deepcopy(envelope)
    forged['memory_reactions']['observations'][0]['evidence_spans'][0]['role'] = 'assistant'
    with pytest.raises(ValueError):
        asyncio.run(adapter.settle_autonomy_memory(forged, source,
            source_reader=reader, resolver=resolver))
    async def changed(_scope): return {**source, 'planning_output': 'changed'}
    with pytest.raises(Exception):
        asyncio.run(adapter.settle_autonomy_memory(envelope, source,
            source_reader=changed, resolver=resolver))


def test_action_queue_requires_actual_l0_copy_before_retention_receipt(tmp_path):
    authority = MemoryAuthorityStore(str(tmp_path / 'memory.sqlite'))
    tasks = MemoryL1SourceTasks(authority)
    queued = tasks.enqueue_action({'action_id': 'action-1', 'artifact_id': 'artifact-1',
        'request_id': 'request-1'})
    assert queued['receipt_status'] == 'source_queued'
    assert tasks.get(queued['task_id']).get('retention') is None
    reader = type('Reader', (), {'options': {'conversation_id': 'conversation-1'}})()
    text = 'Actual action result and source text.'
    captured = {'state': 'available', 'eligible_source': True, 'source_id': 'action-1',
        'source_revision': 'artifact-1', 'text': text, 'recorded_at': WHEN}
    bad = asyncio.run(tasks.retain_action(queued['task_id'], reader,
        profile_id='owner-1', captured_source={**captured, 'source_revision': 'other'}))
    assert bad['retained'] is False
    assert tasks.get(queued['task_id']).get('retention') is None
    receipt = asyncio.run(tasks.retain_action(queued['task_id'], reader,
        profile_id='owner-1', captured_source=captured))
    assert receipt['retained'] is True and receipt['adopted'] is False
    assert receipt['source_namespace'] == 'snapshot.aiz.action_receipt'
    assert receipt['source_id'] == 'action-1' and receipt['source_revision'] == 'artifact-1'
    assert tasks.get(queued['task_id'])['retention'] == receipt
    assert asyncio.run(tasks.retain_action(queued['task_id'], reader,
        profile_id='owner-1', captured_source=captured)) == receipt


def test_authenticated_native_http_ingress_binds_original_source(tmp_path):
    from persona_b1_http import register_candidate_route
    from test_persona_b1_http import RouteRegistry
    store, adapter, _ = prepare(tmp_path)
    source, envelope = fixture()
    async def reader(_scope): return deepcopy(source)
    async def resolver(context, _source):
        return {'state': 'available', 'delivery_receipt_id': context['delivery_receipt_id'],
            'delivery_sha256': context['delivery_sha256'],
            'eligible_fragment_ids': [context['fragments'][0]['fragment_id']]}
    routes = RouteRegistry()
    engine = type('Engine', (), {'b1_candidate': adapter})()
    register_candidate_route(routes, lambda: engine,
        lambda request: request.headers.get('Authorization') == 'Bearer fixture',
        memory_context_resolver=resolver, autonomy_source_reader=reader)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=routes.routes)),
                base_url='https://brain.invalid') as client:
            path = '/internal/persona/b1/autonomy-memory'
            body = {'envelope': envelope, 'source': source}
            assert (await client.post(path, json=body)).status_code == 401
            first = await client.post(path, json=body, headers={'Authorization': 'Bearer fixture'})
            assert first.status_code == 200, first.text
            assert first.json()['b1_receipt']['receipt']['source_kind'] == 'autonomy'
            replay = await client.post(path, json=body, headers={'Authorization': 'Bearer fixture'})
            assert replay.status_code == 200 and replay.json()['b1_receipt']['replayed'] is True
            forged = {'envelope': envelope, 'source': {**source, 'planning_output': 'forged'}}
            assert (await client.post(path, json=forged,
                headers={'Authorization': 'Bearer fixture'})).status_code == 409
    asyncio.run(scenario())
    assert store.read('fixture')['values']['longing'] == pytest.approx(.65)
