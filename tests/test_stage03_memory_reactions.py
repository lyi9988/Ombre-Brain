"""Memory-triggered reactions share the existing exact-source B1 ledger."""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from b1_protocol import canonical, digest
from persona_b1 import FIELDS, PersonaB1Store, PersonaB1Adapter, Conflict
from persona_inner_state import upgrade_policy_v2
from persona_memory_reactions import apply_dimension, repeat_factor, RULES_VERSION
from persona_memory_reaction_context import memory_ref, reaction_id, validate_reactions, MemoryReactionResolver
from test_persona_inner_state_v2 import fixture_policy, source, span, observation, envelope, turn, WHEN


def delivered(src, *, group='source-group-1', text='那次旧事中的亲近与委屈都仍在。'):
    events = turn(src)
    events['user_event']['source_event_id'] = 'gateway-user-' + src['user_id']
    events['assistant_event']['source_event_id'] = 'gateway-assistant-' + src['user_id']
    fragment = {'memory_id': 'memory-test', 'bucket_id': 'memory-test', 'l1_revision': 1, 'l2_revision': None,
        'source_group_id': group, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'source_refs': [], 'coverage': {'section': 'direct', 'statement_ids': [], 'unit_ids': [],
            'partial_presentation': False, 'wire_partial': False}, 'legacy_body_sha256': None}
    fragment['fragment_id'] = 'fragment_' + digest(fragment)[:32]
    scope = {'conversation_id': events['conversation_id'], 'user_source_event_id': events['user_event']['source_event_id'],
        'assistant_source_event_id': events['assistant_event']['source_event_id']}
    context = {'schema': 'memory-context-v1', 'scope': scope, 'fragments': [fragment]}
    context.update(delivery_receipt_id='delivery_' + digest(scope)[:32], delivery_sha256=digest(context))
    return context, events


def reaction(src, context, *, dimension='longing', direction='increase', effect='evoke', strength='clear',
             category='new_reaction', linked=()):
    ref = memory_ref(context['fragments'][0])
    spans = [span(src, 'assistant')]
    row = {'reaction': category, 'dimension': dimension, 'effect': effect, 'direction': direction, 'strength': strength,
        'memory_ref': ref, 'evidence_spans': spans, 'interaction_observation_ids': sorted(linked)}
    row.update(evidence_hash=digest({'source_hash': digest(src), 'evidence_spans': spans, 'memory_ref': ref}))
    row['reaction_id'] = reaction_id(digest(src), row)
    return row


def prepare(tmp_path, *, initial=None, policy=None):
    config = upgrade_policy_v2(policy or fixture_policy())
    store = PersonaB1Store(tmp_path / 'candidate.sqlite', legacy_path=tmp_path / 'untouched-legacy.sqlite')
    values = {key: .6 for key in FIELDS}
    values.update(initial or {})
    store.initialize('fixture', values, WHEN, config)
    return store, PersonaB1Adapter(store, 'fixture', {}), config


def submit(adapter, *, event='u1', category='new_reaction', dimension='longing', direction='increase', effect='evoke',
           strength='clear', when=WHEN, interactions=(), linked=(), context_group='same-source', eligible=True):
    src = source('我们继续说那件旧事。', '这次想起时，我有了新的感受。', event_id=event)
    context, events = delivered(src, group=context_group)
    rows = [reaction(src, context, category=category, dimension=dimension, direction=direction, effect=effect,
        strength=strength, linked=linked)]
    env = envelope(src, list(interactions), memory_context=context,
        memory_reactions={'schema': 'memory-reactions-v1', 'observations': rows})
    env['occurred_at'] = when
    proof = {'state': 'available', 'eligible_fragment_ids': [context['fragments'][0]['fragment_id']] if eligible else [],
        'delivery_receipt_id': context['delivery_receipt_id'], 'delivery_sha256': context['delivery_sha256'],
        'reason': 'source_withdrawn' if not eligible else ''}
    return adapter.settle(env, events, memory_validation=proof), env, events, rows[0]


@pytest.mark.parametrize('strength,amount', [('light', .02), ('clear', .05), ('strong', .08)])
def test_approved_pure_rule_keeps_body_dimensions_and_emotion_inertia(strength, amount):
    assert apply_dimension(.6, effect='evoke', direction='increase', strength=strength)['after'] == pytest.approx(.6 + amount)
    assert apply_dimension(.6, effect='evoke', direction='increase', strength=strength, emotion_inertia=True)['after'] == pytest.approx(.6 + amount * .4)
    assert apply_dimension(.6, effect='evoke', direction='decrease', strength=strength, emotion_inertia=True)['after'] == pytest.approx(.6 - amount * .6)


def test_memory_only_event_same_receipt_replay_and_restart_repeat_modulation(tmp_path):
    store, adapter, policy = prepare(tmp_path)
    first, env, events, row = submit(adapter)
    receipt = first['b1_receipt']['receipt']
    assert receipt['after']['longing'] == pytest.approx(.65)
    assert receipt['memory_reactions']['rules_version'] == RULES_VERSION
    assert receipt['memory_reactions']['observations'][0]['memory_ref'] == row['memory_ref']
    assert receipt['source_refs'] == [{'kind': 'conversation', 'event_id': 'u1', 'occurred_at': WHEN}]
    assert adapter.settle(env, events)['b1_receipt']['replayed'] is True
    restarted = PersonaB1Adapter(PersonaB1Store(store.path, legacy_path=tmp_path / 'untouched-legacy.sqlite'), 'fixture', {})
    second = submit(restarted, event='u2', when=(datetime.fromisoformat(WHEN) + timedelta(minutes=20)).isoformat())[0]
    assert second['b1_receipt']['receipt']['after']['longing'] == pytest.approx(.675)
    assert second['b1_receipt']['receipt']['memory_reactions']['observations'][0]['repeat_factor'] == .5
    assert store.read('fixture')['values']['slow_security'] == .6
    assert store.read('fixture')['values']['slow_confidence'] == .6


def test_continuing_has_no_pulse_but_new_understanding_can_change_same_source(tmp_path):
    store, adapter, _ = prepare(tmp_path)
    submit(adapter)
    continuing = submit(adapter, event='u2', category='continuing')[0]['b1_receipt']['receipt']
    assert continuing['after']['longing'] == pytest.approx(.65)
    assert continuing['memory_reactions']['observations'][0]['status'] == 'no_new_pulse'
    changed = submit(adapter, event='u3', category='new_understanding')[0]['b1_receipt']['receipt']
    assert changed['after']['longing'] == pytest.approx(.675)
    independent = submit(adapter, event='u4', context_group='independent-source')[0]['b1_receipt']['receipt']
    assert independent['after']['longing'] == pytest.approx(.725)


def test_source_revocation_unknown_and_no_resolver_preserve_known_state(tmp_path):
    store, adapter, _ = prepare(tmp_path, initial={'libido': None})
    withdrawn = submit(adapter, eligible=False)[0]['b1_receipt']['receipt']
    assert withdrawn['after']['longing'] == .6
    assert withdrawn['memory_reactions']['observations'][0]['status'] == 'deferred'
    unknown = submit(adapter, event='u2', dimension='libido')[0]['b1_receipt']['receipt']
    assert unknown['after']['libido'] is None
    assert unknown['memory_reactions']['observations'][0]['status'] == 'unknown_not_updated'
    _, env, events, _ = submit(adapter, event='u3', category='continuing')
    env['event_id'] = 'u4'
    # Use a fresh original envelope rather than relabel the old source IDs.
    src = source(event_id='no-resolver')
    context, events = delivered(src)
    env = envelope(src, [], memory_context=context,
        memory_reactions={'schema': 'memory-reactions-v1', 'observations': [reaction(src, context)]})
    result = asyncio.run(adapter.settle_async(env, events))['b1_receipt']['receipt']
    assert result['after']['longing'] == .6
    assert result['memory_reactions']['observations'][0]['status'] == 'deferred'


def test_ease_uses_original_lower_bound_and_no_energy_or_slow_axis(tmp_path):
    store, adapter, _ = prepare(tmp_path)
    result = submit(adapter, dimension='libido', effect='ease', direction='decrease', strength='strong')[0]['b1_receipt']['receipt']
    assert result['after']['libido'] == pytest.approx(.51)
    assert result['after']['slow_security'] == .6
    src = source(event_id='bad-axis')
    context, events = delivered(src)
    row = reaction(src, context, dimension='slow_security')
    with pytest.raises(ValueError, match='approved memory reaction semantics'):
        adapter.settle(envelope(src, [], memory_context=context,
            memory_reactions={'schema': 'memory-reactions-v1', 'observations': [row]}), events)
    row = reaction(src, context, dimension='energy')
    with pytest.raises(ValueError, match='approved memory reaction semantics'):
        validate_reactions({'schema': 'memory-reactions-v1', 'observations': [row]}, context, src, [])


def test_unknown_security_modulation_defers_grief_without_erasing_known_value(tmp_path):
    config = fixture_policy()
    config['security_grief_policy'] = {'security_below': .4, 'grief_event_multiplier': 1.5}
    store, adapter, _ = prepare(tmp_path, initial={'slow_security': None}, policy=config)
    result = submit(adapter, dimension='grieve')[0]['b1_receipt']['receipt']
    assert result['after']['grieve'] == .6 and result['after']['slow_security'] is None
    assert result['memory_reactions']['observations'][0]['reason'] == 'unknown_security_modulation'


def test_real_soothing_same_direction_suppresses_only_overlapping_dimension(tmp_path):
    store, adapter, _ = prepare(tmp_path)
    src = source('我现在抱抱你，刚才那些我明白。', '这次被理解，旧委屈也缓和了。', event_id='soothing-turn')
    context, events = delivered(src)
    obs = observation(src, 'soothing')
    rows = [reaction(src, context, dimension='anger', direction='decrease', effect='ease', linked=[obs['observation_id']]),
        reaction(src, context, dimension='libido', direction='increase', linked=[obs['observation_id']])]
    env = envelope(src, [obs], memory_context=context, memory_reactions={'schema': 'memory-reactions-v1', 'observations': rows})
    proof = {'state': 'available', 'eligible_fragment_ids': [context['fragments'][0]['fragment_id']],
        'delivery_receipt_id': context['delivery_receipt_id'], 'delivery_sha256': context['delivery_sha256']}
    receipt = adapter.settle(env, events, memory_validation=proof)['b1_receipt']['receipt']
    results = receipt['memory_reactions']['observations']
    assert results[0]['status'] == 'same_cause_already_settled'
    assert receipt['after']['anger'] == pytest.approx(.51)
    assert results[1]['status'] == 'applied' and receipt['after']['libido'] == pytest.approx(.65)


def qualified_context(tmp_path, src):
    from test_memory_l2_dependency_scope_r2 import _persist_source_bound_l1
    service, env, _ = _persist_source_bound_l1(tmp_path / 'memory')
    asyncio.run(env['commit'].commit_memory(memory_id='legacy', bucket_id='legacy', expected_revision=0,
        body='历史原记忆正文仍完整存在。', metadata={'name': '历史稿'}, source_refs=['legacy-source'],
        decision_source='synthetic', idempotency_key='legacy', actor='test'))
    context, events = delivered(src)
    l1 = env['authority'].get_memory_revision('m1', 2)['metadata']['memory_l1']
    fragment = context['fragments'][0]
    fragment.update(memory_id='m1', bucket_id='b1', l1_revision=2, text='只由 A 支持的完整合成记忆。',
        source_refs=[{key: binding[key] for key in ('binding_id', 'source_namespace', 'source_id', 'source_revision')}
            for binding in l1['evidence_bindings'] if binding['binding_id'] == 'source:A'])
    fragment['text_sha256'] = hashlib.sha256(fragment['text'].encode()).hexdigest()
    legacy = deepcopy(fragment)
    legacy.update(memory_id='legacy', bucket_id='legacy', l1_revision=None, text='历史原记忆正文仍完整存在。',
        source_group_id='legacy-group', source_refs=['legacy-source'])
    legacy['text_sha256'] = hashlib.sha256(legacy['text'].encode()).hexdigest()
    legacy['legacy_body_sha256'] = env['authority'].get_memory_revision('legacy', 1)['body_sha256']
    legacy['coverage']['legacy_authority_revision'] = 1
    unknown = deepcopy(legacy)
    unknown.update(memory_id='legacy-unknown', bucket_id='legacy-unknown', source_group_id='unknown-group')
    unknown['coverage']['legacy_authority_revision'] = None
    context['fragments'] = [fragment, legacy, unknown]
    for value in context['fragments']:
        value['fragment_id'] = 'fragment_' + digest({key: item for key, item in value.items() if key != 'fragment_id'})[:32]
    context['delivery_sha256'] = digest({key: context[key] for key in ('schema', 'scope', 'fragments')})
    return context, events, service, env


def test_real_authority_resolver_supports_legacy_mixed_and_current_source_withdrawal(tmp_path):
    from memory_l1_reads import MemoryL1Reads
    src = source(event_id='resolver-turn')
    context, events, l2, env = qualified_context(tmp_path, src)
    methods = []
    async def scenario():
        async def gateway(request):
            methods.append(request.method)
            assert request.url.path == '/internal/memory-delivery/lookup'
            body = json.loads(request.content)
            assert body['user_source_event_id'] == events['user_event']['source_event_id']
            assert body['delivery_sha256'] == context['delivery_sha256']
            return httpx.Response(200, json={'state': 'available', 'context': context,
                'fragment_states': [{'fragment_id': f['fragment_id'], 'state': 'available'} for f in context['fragments']]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as client:
            resolver = MemoryReactionResolver(lambda: ('https://gateway.invalid/api/config', 'fixture-token'),
                lambda: MemoryL1Reads(env['revisions'], env['sources']), lambda: l2, client=client)
            proof = await resolver(context, events)
            modern, legacy, unknown = [value['fragment_id'] for value in context['fragments']]
            assert proof['eligible_fragment_ids'] == [modern, legacy]
            assert proof['qualifications'][legacy] == 'legacy_authority_current'
            assert unknown in proof['reasons']
            env['source_state']['record']['content'] = '来源已更正，旧版本不能继续产生反应。'
            changed = await resolver(context, events)
            assert changed['eligible_fragment_ids'] == [legacy]
            assert changed['reasons'][modern] == 'memory_source_withdrawn_or_changed'
            wrong_pair = deepcopy(events)
            wrong_pair['user_event']['source_event_id'] = 'unrelated'
            denied = await resolver(context, wrong_pair)
            assert denied['state'] == 'unavailable'
            assert len(methods) == 2
    asyncio.run(scenario())


def test_real_http_memory_only_uses_resolver_and_replay_does_not_lookup_again(tmp_path):
    from memory_l1_reads import MemoryL1Reads
    from persona_engine import PersonaStateEngine
    from persona_b1_http import register_candidate_route
    from test_persona_b1_http import RouteRegistry
    from starlette.applications import Starlette
    src = source('今天又想起那件事。', '这次我仍会想靠近你。', event_id='http-memory')
    context, events, l2, env = qualified_context(tmp_path, src)
    # Use one known formal memory, keeping the exact original delivered scope.
    context['fragments'] = context['fragments'][:1]
    context['delivery_sha256'] = digest({key: context[key] for key in ('schema', 'scope', 'fragments')})
    store, adapter, _ = prepare(tmp_path / 'inner-state')
    engine = PersonaStateEngine({'buckets_dir': str(tmp_path / 'inner-state'),
        'persona': {'profile_id': 'fixture', 'mode': 'disabled'}}, str(tmp_path / 'inner-state' / 'legacy.sqlite'), b1_candidate=adapter)
    envelope_value = envelope(src, [], memory_context=context,
        memory_reactions={'schema': 'memory-reactions-v1', 'observations': [reaction(src, context)]})
    lookups = []
    current_context = [context]
    async def scenario():
        async def gateway(request):
            lookups.append(request)
            return httpx.Response(200, json={'state': 'available', 'context': current_context[0],
                'fragment_states': [{'fragment_id': current_context[0]['fragments'][0]['fragment_id'], 'state': 'available'}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as gateway_client:
            resolver = MemoryReactionResolver(lambda: ('https://gateway.invalid/api/config', 'fixture'),
                lambda: MemoryL1Reads(env['revisions'], env['sources']), lambda: l2, client=gateway_client)
            routes = RouteRegistry()
            register_candidate_route(routes, lambda: engine, lambda request: request.headers.get('Authorization') == 'Bearer fixture',
                memory_context_resolver=resolver)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=routes.routes)), base_url='https://brain.invalid') as client:
                payload = {'envelope': envelope_value, 'turn': events}
                denied = await client.post('/internal/persona/b1/candidate', json=payload)
                assert denied.status_code == 401 and not lookups
                first = await client.post('/internal/persona/b1/candidate', json=payload, headers={'Authorization': 'Bearer fixture'})
                assert first.status_code == 200, first.text
                actual_receipt = first.json()['b1_receipt']['receipt']['memory_reactions']
                assert actual_receipt['rules_version'] == RULES_VERSION
                assert actual_receipt['observations'][0]['status'] == 'applied'
                assert context['fragments'][0]['text'] not in first.text
                replay = await client.post('/internal/persona/b1/candidate', json=payload, headers={'Authorization': 'Bearer fixture'})
                assert replay.status_code == 200 and replay.json()['replayed'] is True
                assert replay.json()['b1_receipt']['receipt']['memory_reactions'] == actual_receipt
                assert len(lookups) == 1
                assert store.read('fixture')['values']['longing'] == pytest.approx(.65)
                assert store.read('fixture')['revision'] == 1
                env['source_state']['record']['content'] = '原记忆来源已更正。'
                next_source = source('现在抱抱你。', '现在的相处仍是真的。', event_id='deferred-memory')
                next_context, next_events = delivered(next_source)
                next_context['fragments'] = deepcopy(context['fragments'])
                next_context['delivery_sha256'] = digest({key: next_context[key] for key in ('schema', 'scope', 'fragments')})
                current_context[0] = next_context
                next_payload = {'envelope': envelope(next_source, [observation(next_source)], memory_context=next_context,
                    memory_reactions={'schema': 'memory-reactions-v1', 'observations': [reaction(next_source, next_context)]}), 'turn': next_events}
                deferred = await client.post('/internal/persona/b1/candidate', json=next_payload, headers={'Authorization': 'Bearer fixture'})
                assert deferred.status_code == 200, deferred.text
                assert deferred.json()['b1_receipt']['receipt']['memory_reactions']['observations'][0]['status'] == 'deferred'
                assert store.read('fixture')['values']['longing'] == pytest.approx(.65)
                assert store.read('fixture')['values']['valence'] == pytest.approx(.64)
    asyncio.run(scenario())


def test_same_current_trigger_multiple_cards_one_strongest_pulse_and_shared_repeat_history(tmp_path):
    store, adapter, _ = prepare(tmp_path)
    src = source(event_id='multi-card')
    context, events = delivered(src, group='first-group')
    other = deepcopy(context['fragments'][0])
    other['source_group_id'] = 'second-group'
    other['fragment_id'] = 'fragment_' + digest({key: value for key, value in other.items() if key != 'fragment_id'})[:32]
    context['fragments'].append(other)
    context['delivery_sha256'] = digest({key: context[key] for key in ('schema', 'scope', 'fragments')})
    first = reaction(src, context, strength='strong')
    second_context = {**context, 'fragments': [other]}
    second = reaction(src, second_context, strength='clear')
    env = envelope(src, [], memory_context=context, memory_reactions={'schema': 'memory-reactions-v1', 'observations': [second, first]})
    proof = {'state': 'available', 'eligible_fragment_ids': [f['fragment_id'] for f in context['fragments']],
        'delivery_receipt_id': context['delivery_receipt_id'], 'delivery_sha256': context['delivery_sha256']}
    receipt = adapter.settle(env, events, memory_validation=proof)['b1_receipt']['receipt']
    assert receipt['after']['longing'] == pytest.approx(.68)
    assert receipt['memory_reactions']['observations'][0]['status'] == 'same_trigger_coalesced'
    assert len(store.read('fixture')['memory_reaction_history']) == 2
    later = submit(adapter, event='later', context_group='second-group')[0]['b1_receipt']['receipt']
    assert later['after']['longing'] == pytest.approx(.705)
    assert later['memory_reactions']['observations'][0]['repeat_factor'] == .5


def test_memory_reaction_updates_mixed_state_in_the_same_turn(tmp_path):
    store, adapter, _ = prepare(tmp_path, initial={'anger': .28, 'grieve': .28, 'longing': .28,
        'tenderness': .28, 'libido': .28, 'possessiveness': .28})
    src = source(event_id='mixed-memory')
    context, events = delivered(src)
    rows = [reaction(src, context, dimension=key, strength='strong') for key in ('anger', 'longing')]
    env = envelope(src, [], memory_context=context, memory_reactions={'schema': 'memory-reactions-v1', 'observations': rows})
    proof = {'state': 'available', 'eligible_fragment_ids': [context['fragments'][0]['fragment_id']],
        'delivery_receipt_id': context['delivery_receipt_id'], 'delivery_sha256': context['delivery_sha256']}
    adapter.settle(env, events, memory_validation=proof)
    state = store.read('fixture')
    assert state['values']['anger'] == pytest.approx(.36) and state['values']['longing'] == pytest.approx(.36)
    assert state['mixed']['anger_closeness']['status'] == 'visible'


def test_known_security_modulation_and_higher_existing_floor_are_preserved(tmp_path):
    config = fixture_policy()
    config['security_grief_policy'] = {'security_below': .4, 'grief_event_multiplier': 1.5}
    config['events']['soothing'] = {'libido': {'delta': 0, 'satisfaction': .8, 'floor': .4, 'plateau_hours': 0}}
    store, adapter, _ = prepare(tmp_path, initial={'slow_security': .2}, policy=config)
    first = submit(adapter, dimension='grieve')[0]['b1_receipt']['receipt']
    assert first['after']['grieve'] == pytest.approx(.675)
    second = submit(adapter, event='u2', dimension='libido', direction='decrease', effect='ease', strength='strong')[0]['b1_receipt']['receipt']
    assert second['after']['libido'] == pytest.approx(.57)
    assert second['after']['slow_security'] == .2


def test_actual_gateway_delivery_ledger_final_wire_and_original_lookup_drive_brain(tmp_path):
    import importlib.util
    from pathlib import Path
    from memory_l1_reads import MemoryL1Reads
    gateway_path = Path(__file__).resolve().parents[2] / 'stage03-confirmed-gateway' / 'memory_delivery.py'
    assert gateway_path.is_file(), 'local cross-service fixture requires the prepared Gateway worktree'
    spec = importlib.util.spec_from_file_location('stage03_gateway_delivery_fixture', gateway_path)
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    src = source('又想起以前那件事。', '这次回忆让我有了新的感受。', event_id='actual-wire')
    original_context, events, l2, env = qualified_context(tmp_path, src)
    source_fragment = original_context['fragments'][0]
    block = '主人可编排的记忆内容：' + source_fragment['text']
    candidate = {key: value for key, value in source_fragment.items() if key != 'fragment_id'}
    frozen = [{**candidate, '_compiled_block': block}]
    assert gateway.final_fragments(frozen, {'messages': [{'role': 'system', 'content': 'Composer已关闭记忆块。'}]}) == []
    fragments = gateway.final_fragments(frozen, {'messages': [{'role': 'system', 'content': block}]})
    ledger = gateway.MemoryDeliveryStore(tmp_path / 'gateway-delivery.sqlite3')
    attempt = ledger.begin(original_context['scope'], 'actual-outbound-attempt', fragments)
    assert ledger.complete(original_context['scope']) is None
    ledger.succeed(attempt)
    context = ledger.complete(original_context['scope'])
    assert context['fragments'] == fragments
    store, adapter, _ = prepare(tmp_path / 'brain-state')
    env_value = envelope(src, [], memory_context=context,
        memory_reactions={'schema': 'memory-reactions-v1', 'observations': [reaction(src, context)]})
    async def scenario():
        async def lookup(request):
            result = ledger.lookup(json.loads(request.content))
            result['fragment_states'] = [{'fragment_id': f['fragment_id'], 'state': 'available'} for f in result['context']['fragments']]
            return httpx.Response(200, json=result)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lookup)) as client:
            resolver = MemoryReactionResolver(lambda: ('https://gateway.invalid/api/config', 'fake-only'),
                lambda: MemoryL1Reads(env['revisions'], env['sources']), lambda: l2, client=client)
            result = await adapter.settle_async(env_value, events, resolver=resolver)
            assert result['b1_receipt']['receipt']['after']['longing'] == pytest.approx(.65)
            assert result['b1_receipt']['receipt']['memory_reactions']['delivery_receipt_id'] == context['delivery_receipt_id']
            assert ledger.lookup({**context['scope'], 'delivery_sha256': '0' * 64})['state'] == 'unavailable'
    asyncio.run(scenario())


def test_candidate_transport_preserves_large_delivered_memory_under_existing_source_ceiling(tmp_path):
    from memory_l1_reads import MemoryL1Reads
    from persona_engine import PersonaStateEngine
    from persona_b1_http import register_candidate_route, MAX_CANDIDATE_BODY, MAX_BODY
    from test_persona_b1_http import RouteRegistry
    from starlette.applications import Starlette
    src = source('完整记忆材料保持。', '我读到的这段记忆让我有新的感受。', event_id='large-context')
    context, events, l2, env = qualified_context(tmp_path, src)
    text = '这是一段完整投递且不应截断的历史回忆。' * 5000
    asyncio.run(env['commit'].commit_memory(memory_id='large-legacy', bucket_id='large-legacy', expected_revision=0,
        body=text, metadata={'name': '完整历史稿'}, source_refs=['legacy-source'], decision_source='synthetic',
        idempotency_key='large-legacy', actor='test'))
    fragment = deepcopy(context['fragments'][1])
    fragment.update(memory_id='large-legacy', bucket_id='large-legacy', text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(), legacy_body_sha256=hashlib.sha256(text.encode()).hexdigest())
    fragment['fragment_id'] = 'fragment_' + digest({key: value for key, value in fragment.items() if key != 'fragment_id'})[:32]
    context['fragments'] = [fragment]
    context['delivery_sha256'] = digest({key: context[key] for key in ('schema', 'scope', 'fragments')})
    store, adapter, _ = prepare(tmp_path / 'large-brain')
    engine = PersonaStateEngine({'buckets_dir': str(tmp_path / 'large-brain'), 'persona': {'profile_id': 'fixture', 'mode': 'disabled'}},
        str(tmp_path / 'large-brain' / 'legacy.sqlite'), b1_candidate=adapter)
    payload = {'envelope': envelope(src, [], memory_context=context,
        memory_reactions={'schema': 'memory-reactions-v1', 'observations': [reaction(src, context)]}), 'turn': events}
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    assert MAX_BODY < len(encoded) < MAX_CANDIDATE_BODY
    calls = []
    async def scenario():
        async def lookup(request):
            calls.append(request)
            return httpx.Response(200, json={'state': 'available', 'context': context,
                'fragment_states': [{'fragment_id': fragment['fragment_id'], 'state': 'available'}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(lookup)) as gateway_client:
            resolver = MemoryReactionResolver(lambda: ('https://gateway.invalid/api/config', 'fake-only'),
                lambda: MemoryL1Reads(env['revisions'], env['sources']), lambda: l2, client=gateway_client)
            routes = RouteRegistry()
            register_candidate_route(routes, lambda: engine, lambda request: request.headers.get('Authorization') == 'Bearer fixture',
                memory_context_resolver=resolver)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=routes.routes)), base_url='https://brain.invalid') as client:
                denied = await client.post('/internal/persona/b1/candidate', content=encoded)
                assert denied.status_code == 401 and calls == []
                accepted = await client.post('/internal/persona/b1/candidate', content=encoded,
                    headers={'Authorization': 'Bearer fixture', 'Content-Type': 'application/json'})
                assert accepted.status_code == 200, accepted.text
                assert store.read('fixture')['values']['longing'] == pytest.approx(.65)
                assert len(calls) == 1
                async def chunks():
                    yield b'x' * MAX_CANDIDATE_BODY
                    yield b'x'
                too_large = await client.post('/internal/persona/b1/candidate', content=chunks(), headers={'Authorization': 'Bearer fixture'})
                assert too_large.status_code == 413 and len(calls) == 1
                clock_large = await client.post('/internal/persona/b1/clock', content=b'x' * (MAX_BODY + 1),
                    headers={'Authorization': 'Bearer fixture'})
                assert clock_large.status_code == 413
    asyncio.run(scenario())


@pytest.mark.parametrize('value,direction,requested,actual', [(0, 'decrease', -.08, 0),
    (1, 'increase', .08, 0), (.99, 'increase', .08, .01), (.01, 'decrease', -.08, -.01)])
def test_saturation_receipt_reports_actual_delta_and_retains_requested(value, direction, requested, actual):
    result = apply_dimension(value, effect='evoke', direction=direction, strength='strong')
    assert result['requested_delta'] == pytest.approx(requested)
    assert result['delta'] == pytest.approx(actual)
    assert result['delta'] == result['after'] - result['before']
    assert result['status'] == ('applied' if actual else 'no_change')


@pytest.mark.parametrize('value,multiplier,requested,actual', [(.99, 1.5, .12, .01), (.96, .5, .04, .04),
    (1, 1.5, .12, 0)])
def test_modulated_saturation_uses_requested_pulse_then_reports_actual_receipt(tmp_path, value, multiplier, requested, actual):
    config = fixture_policy()
    config['security_grief_policy'] = {'security_below': .4, 'grief_event_multiplier': multiplier}
    store, adapter, _ = prepare(tmp_path, initial={'grieve': value, 'slow_security': .2}, policy=config)
    receipt = submit(adapter, dimension='grieve', strength='strong')[0]['b1_receipt']['receipt']
    result = receipt['memory_reactions']['observations'][0]
    assert result['requested_delta'] == pytest.approx(requested)
    assert result['delta'] == pytest.approx(actual)
    assert result['delta'] == result['after'] - result['before']
    if actual:
        assert receipt['event_deltas']['grieve'] == pytest.approx(actual)
    assert store.read('fixture')['values']['grieve'] == pytest.approx(min(1, value + requested))


def test_saturated_known_reaction_keeps_repeat_history_for_later_same_direction(tmp_path):
    store, adapter, _ = prepare(tmp_path, initial={'longing': 1})
    first = submit(adapter, strength='strong')[0]['b1_receipt']['receipt']
    assert first['memory_reactions']['observations'][0]['status'] == 'no_change'
    assert first['memory_reactions']['observations'][0]['delta'] == 0
    assert len(store.read('fixture')['memory_reaction_history']) == 1
    submit(adapter, event='u2', direction='decrease')
    later = submit(adapter, event='u3')[0]['b1_receipt']['receipt']
    assert later['after']['longing'] == pytest.approx(.975)
    assert later['memory_reactions']['observations'][0]['repeat_factor'] == .5
