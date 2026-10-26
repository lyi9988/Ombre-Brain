"""Offline final-wire delivery, recovery, eligibility and activation checks."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from bucket_manager import BucketManager
from gateway import GatewayService, create_gateway_app
import memory_delivery
from memory_delivery import (MemoryDeliveryStore, candidates, compiled_candidates,
                             final_fragments, digest, sha, capture_carriers)
from attachment_adoption_support import gateway_service, openai_response

SCOPE = {'conversation_id': 'conversation-1', 'user_source_event_id': 'user-1',
         'assistant_source_event_id': 'turn:user-event-1:assistant'}


def run(value):
    return asyncio.run(value)


def source_fixture(*, bid='bucket-1', text='The supplied past memory has its original meaning.', legacy=False):
    binding = {'binding_id': 'binding-1', 'source_namespace': 'snapshot.canonical',
               'source_id': 'source-1', 'source_revision': 'source-r1'}
    record = {'memory_id': 'memory-' + bid, 'revision': 3, 'body_sha256': sha('original body'),
              'document': None if legacy else {'evidence_bindings': [binding]}}
    projection = {'rendered_memory_spans': [{'bucket_id': bid, 'section': 'direct', 'text': text}],
        'presentation_trace': [{'bucket_id': bid, 'memory_id': record['memory_id'],
            'l1_revision': None if legacy else 3, 'presentation_revision': None if legacy else 2,
            'injected_statement_ids': ['statement-1'], 'selected_unit_ids': ['unit-1']}],
        'compiled_memory_blocks': [{'source_id': 'ombre.memory_recall', 'content': 'OWNER WRAPPER\n' + text}]}
    projection['delivery_candidates'] = candidates(projection, {bid: record})
    frozen = compiled_candidates(projection, projection['delivery_candidates'])
    wire = {'messages': [{'role': 'system', 'content': 'OWNER WRAPPER\n' + text}]}
    return projection, frozen, wire


def completed(store, scope=SCOPE, *, bid='bucket-1', legacy=False):
    _, frozen, wire = source_fixture(bid=bid, legacy=legacy)
    attempt = store.begin(scope, 'trace-1', final_fragments(frozen, wire))
    store.succeed(attempt)
    return store.complete(scope)


def use_query(context, *, use_id='use-1', status='new_use', episode=None, fresh=False):
    evidence = {'event_id': 'assistant-event', 'start': 0, 'end': 5, 'text_sha256': sha('proof')}
    use = {'use_id': use_id, 'fragment_ref': context['fragments'][0]['fragment_id'],
        'status': status, 'evidence_refs': [evidence]}
    if episode:
        use['episode_id'] = episode
    if fresh:
        use['new_episode_evidence_refs'] = [evidence]
    return {**context['scope'], 'delivery_receipt_id': context['delivery_receipt_id'],
        'delivery_sha256': context['delivery_sha256'], 'user_event_id': 'user-event',
        'assistant_event_id': 'assistant-event', 'uses': [use]}


class Buckets:
    def __init__(self):
        self.calls = []
    async def touch(self, bid, *, activation_id=None):
        self.calls.append((bid, activation_id))
        return True


def test_isolated_module_and_actual_wire_not_selected_ids_or_status_wrappers():
    assert Path(memory_delivery.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    projection, frozen, wire = source_fixture()
    fragments = final_fragments(frozen, wire)
    assert len(fragments) == 1 and fragments[0]['l1_revision'] == 3 and fragments[0]['l2_revision'] == 2
    assert fragments[0]['coverage']['statement_ids'] == ['statement-1']
    assert final_fragments(frozen, {'messages': [{'role': 'system', 'content': 'Recall selected 1.'}]}) == []
    projection['compiled_memory_blocks'] = []
    assert compiled_candidates(projection, projection['delivery_candidates']) == []
    # Incoming history containing the original memory cannot revive an off block.
    assert final_fragments([], wire) == []


def test_physical_success_union_survives_restart_and_retry_deduplication(tmp_path):
    path = tmp_path / 'gateway.sqlite'
    store = MemoryDeliveryStore(path)
    _, frozen, wire = source_fixture()
    fragments = final_fragments(frozen, wire)
    store.begin(SCOPE, 'failed-transport', fragments)
    assert store.complete(SCOPE) is None
    for _ in range(2):
        attempt = store.begin(SCOPE, 'tool-call-or-retry', fragments)
        store.succeed(attempt)
    context = store.complete(SCOPE)
    assert len(context['fragments']) == 1
    assert context['delivery_sha256'] == digest({key: context[key] for key in ('schema', 'scope', 'fragments')})
    restored = MemoryDeliveryStore(path)
    assert restored.lookup(SCOPE)['context'] == context
    assert restored.complete(SCOPE) == context
    assert restored.lookup({**SCOPE, 'assistant_source_event_id': 'other'})['state'] == 'unavailable'
    assert restored.lookup({**SCOPE, 'delivery_sha256': 'wrong'})['reason'] == 'delivery_identity_mismatch'


def test_empty_success_receipt_is_explicit_and_not_an_activation(tmp_path):
    store = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    attempt = store.begin(SCOPE, 'trace-empty', [])
    store.succeed(attempt)
    context = store.complete(SCOPE)
    assert store.lookup(SCOPE)['state'] == 'available'
    assert context['fragments'] == []


def test_activation_evidence_idempotency_continuity_and_new_episode(tmp_path):
    store, buckets = MemoryDeliveryStore(tmp_path / 'gateway.sqlite'), Buckets()
    first = completed(store)
    query = use_query(first, episode='episode-1')
    assert run(store.activate(query, first, buckets))['results'][0]['status'] == 'activated'
    assert run(store.activate(query, first, buckets))['results'][0]['status'] == 'activated'
    assert len(buckets.calls) == 1
    with pytest.raises(ValueError, match='conflict'):
        run(store.activate({**query, 'uses': []}, first, buckets))
    second = completed(store, {**SCOPE, 'user_source_event_id': 'user-2', 'assistant_source_event_id': 'assistant-2'})
    again = use_query(second, use_id='use-2', episode='episode-2')
    assert run(store.activate(again, second, buckets))['results'][0]['status'] == 'continuing'
    assert len(buckets.calls) == 1
    third = completed(store, {**SCOPE, 'user_source_event_id': 'user-3', 'assistant_source_event_id': 'assistant-3'})
    fresh = use_query(third, use_id='use-3', episode='episode-3', fresh=True)
    assert run(store.activate(fresh, third, buckets))['results'][0]['status'] == 'activated'
    assert len(buckets.calls) == 2


@pytest.mark.parametrize('status', ['not_used', 'uncertain', 'continuing'])
def test_delivery_or_non_new_use_never_strengthens(tmp_path, status):
    store, buckets = MemoryDeliveryStore(tmp_path / 'gateway.sqlite'), Buckets()
    context = completed(store)
    result = run(store.activate(use_query(context, status=status), context, buckets))
    assert result['results'][0]['status'] == 'not_activated'
    assert buckets.calls == []


def test_missing_or_wrong_turn_evidence_does_not_activate(tmp_path):
    store, buckets = MemoryDeliveryStore(tmp_path / 'gateway.sqlite'), Buckets()
    context = completed(store)
    query = use_query(context)
    query['uses'][0]['evidence_refs'][0]['event_id'] = 'unrelated-event'
    result = run(store.activate(query, context, buckets))
    assert result['results'][0]['reason'] == 'current_evidence_required'
    assert buckets.calls == []


def test_same_source_group_in_one_turn_and_late_evaluation_do_not_strengthen_twice(tmp_path):
    store, buckets = MemoryDeliveryStore(tmp_path / 'gateway.sqlite'), Buckets()
    first = completed(store)
    query = use_query(first)
    query['uses'].append({**copy.deepcopy(query['uses'][0]), 'use_id': 'second-use'})
    result = run(store.activate(query, first, buckets))
    assert [row['status'] for row in result['results']] == ['activated', 'not_activated']
    assert len(buckets.calls) == 1
    late = completed(store, {**SCOPE, 'user_source_event_id': 'late-u', 'assistant_source_event_id': 'late-a'})
    newer = completed(store, {**SCOPE, 'user_source_event_id': 'new-u', 'assistant_source_event_id': 'new-a'})
    run(store.activate(use_query(newer, use_id='new-use', status='not_used'), newer, buckets))
    result = run(store.activate(use_query(late, use_id='late-use'), late, buckets))
    assert result['results'][0]['reason'] == 'late_evaluation'
    assert len(buckets.calls) == 1


def test_bucket_activation_recovery_preserves_old_count_and_does_not_ripple(tmp_path):
    mgr = BucketManager({'buckets_dir': str(tmp_path / 'buckets')})
    async def scenario():
        first = await mgr.create(content='original body', bucket_id='bucket-1')
        second = await mgr.create(content='temporal neighbour', bucket_id='bucket-2')
        import frontmatter
        path = mgr._find_bucket_file(first)
        post = frontmatter.load(path)
        post['activation_count'] = 4.3
        mgr._write_post_atomic(path, post)
        original_bytes = Path(path).read_bytes()
        before = (await mgr.get(second))['metadata']['activation_count']
        await mgr.touch(first, activation_id='activation-1')
        await mgr.touch(first, activation_id='activation-1')
        assert Path(path).read_bytes() == original_bytes, 'activation never rewrites the memory body or base metadata'
        assert (await mgr.get(first))['metadata']['activation_count'] == 5.3
        await mgr.add_comment(first, 'actual new comment', comment_id='comment-1')
        await mgr.add_comment(first, 'actual new comment', comment_id='comment-1')
        assert (await mgr.get(first))['metadata']['activation_count'] == 6.3
        assert (await mgr.get(second))['metadata']['activation_count'] == before
    run(scenario())


def test_owner_edit_does_not_restart_activation_clock_without_explicit_value(tmp_path):
    mgr = BucketManager({'buckets_dir': str(tmp_path / 'buckets')})
    async def scenario():
        original = '2026-01-01T00:00:00+00:00'
        bid = await mgr.create(content='original body', bucket_id='edited-memory', last_active=original)
        assert await mgr.update(bid, content='owner revised body')
        edited = await mgr.get(bid)
        assert edited['content'] == 'owner revised body'
        assert edited['metadata']['last_active'] == original
        assert edited['metadata']['activation_count'] == 0
        explicit = '2026-02-01T00:00:00+00:00'
        assert await mgr.update(bid, last_active=explicit)
        assert (await mgr.get(bid))['metadata']['last_active'] == explicit
    run(scenario())


def test_activation_projection_survives_owner_edit_roundtrip_restart_decay_and_node_consumers(tmp_path):
    import frontmatter
    from decay_engine import DecayEngine
    from memory_nodes import MemoryNodeStore

    config = {'buckets_dir': str(tmp_path / 'buckets'), 'state_dir': str(tmp_path / 'state'),
              'decay': {'threshold': -1}}
    mgr = BucketManager(config)
    async def scenario():
        bid = await mgr.create(content='original body', bucket_id='owner-edited')
        path = mgr._find_bucket_file(bid)
        original_record = mgr.activation_ledger.record
        def owner_edit_between_read_and_activation(*args):
            post = frontmatter.load(path)
            post.content = 'owner concurrent revision must remain'
            mgr._write_post_atomic(path, post)
            original_record(*args)
        mgr.activation_ledger.record = owner_edit_between_read_and_activation
        assert await mgr.touch(bid, activation_id='owner-race') is True
        mgr.activation_ledger.record = original_record
        current = await mgr.get(bid)
        assert current['content'] == 'owner concurrent revision must remain'
        assert current['metadata']['activation_count'] == 1
        await mgr.update(bid, extra_metadata=current['metadata'])
        assert frontmatter.load(path)['activation_count'] == 0, 'projected count cannot be written back as new base'
        assert '_memory_activation_projection' not in frontmatter.load(path).metadata
        restored = BucketManager(config)
        single = await restored.get(bid)
        listing = next(row for row in await restored.list_all() if row['id'] == bid)
        assert single['metadata']['activation_count'] == listing['metadata']['activation_count'] == 1
        nodes = MemoryNodeStore(config)
        nodes.bulk_upsert([listing])
        assert nodes.get(bid)['activation_count'] == 1
        assert nodes.get(bid)['last_active'] == single['metadata']['last_active']
        decay = DecayEngine(config, restored)
        seen = []
        calculate = decay.calculate_score
        def record(metadata):
            seen.append(metadata.copy())
            return calculate(metadata)
        decay.calculate_score = record
        await decay.run_decay_cycle()
        assert seen[0]['activation_count'] == 1 and seen[0]['last_active'] == single['metadata']['last_active']
    run(scenario())


def test_legacy_revision_and_body_are_independently_eligible_without_fabricating_l1(tmp_path):
    service = GatewayService.__new__(GatewayService)
    store = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    context = completed(store, legacy=True)
    fragment = context['fragments'][0]
    assert fragment['l1_revision'] is None and fragment['l2_revision'] is None
    service.memory_authority_enabled = True
    service.memory_authority_view = SimpleNamespace(memory_index_metadata_map=lambda _: {
        'bucket-1': {'memory_id': fragment['memory_id'], 'revision': 3, 'body_sha256': sha('original body')}})
    async def get(_):
        return {'content': 'original body', 'metadata': {}}
    service.bucket_mgr = SimpleNamespace(get=get)
    assert run(service._validate_memory_delivery(context))[0]['state'] == 'available'
    service.memory_authority_view.memory_index_metadata_map = lambda _: {
        'bucket-1': {'memory_id': fragment['memory_id'], 'revision': 4, 'body_sha256': sha('original body')}}
    assert run(service._validate_memory_delivery(context))[0]['state'] == 'changed'


def test_crash_after_activation_receipt_before_feedback_commit_recovers_once(tmp_path):
    mgr = BucketManager({'buckets_dir': str(tmp_path / 'buckets')})
    store = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    context = completed(store)
    query = use_query(context)
    async def scenario():
        await mgr.create(content='original body', bucket_id='bucket-1')
        original = mgr.touch
        async def interrupted(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError('simulated_after_receipt_write')
        mgr.touch = interrupted
        with pytest.raises(RuntimeError, match='simulated_after_receipt_write'):
            await store.activate(query, context, mgr)
        assert (await mgr.get('bucket-1'))['metadata']['activation_count'] == 1
        mgr.touch = original
        restored = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
        assert (await restored.activate(query, context, mgr))['results'][0]['status'] == 'activated'
        assert (await mgr.get('bucket-1'))['metadata']['activation_count'] == 1
    run(scenario())


def test_unknown_source_does_not_clear_continuity_but_known_empty_assessment_does(tmp_path):
    store, buckets = MemoryDeliveryStore(tmp_path / 'gateway.sqlite'), Buckets()
    first = completed(store)
    run(store.activate(use_query(first), first, buckets))
    second = completed(store, {**SCOPE, 'user_source_event_id': 'u2', 'assistant_source_event_id': 'a2'})
    query = use_query(second, use_id='u2')
    query['uses'] = []
    run(store.activate(query, second, buckets, fragment_states=[{
        'fragment_id': second['fragments'][0]['fragment_id'], 'state': 'unavailable'}]))
    third = completed(store, {**SCOPE, 'user_source_event_id': 'u3', 'assistant_source_event_id': 'a3'})
    assert run(store.activate(use_query(third, use_id='u3'), third, buckets))['results'][0]['status'] == 'continuing'
    fourth = completed(store, {**SCOPE, 'user_source_event_id': 'u4', 'assistant_source_event_id': 'a4'})
    query = use_query(fourth, use_id='u4')
    query['uses'] = []
    run(store.activate(query, fourth, buckets))
    fifth = completed(store, {**SCOPE, 'user_source_event_id': 'u5', 'assistant_source_event_id': 'a5'})
    assert run(store.activate(use_query(fifth, use_id='u5'), fifth, buckets))['results'][0]['status'] == 'activated'


def test_tool_call_with_text_cannot_finalize_canonical_assistant_before_final_answer():
    from test_gateway_canonical_integration import service
    item = service()
    state = {'enabled': True, 'bridge_write_enabled': True, 'request_id': 'logical-1', 'channel_id': 'operit'}
    run(item._finalize_canonical_turn(state, {'content': 'Working on the tool.', 'tool_calls': [{'id': 'tool-1'}]}))
    assert item.canonical_adapter.ingested == []
    assert state['assistant_write_status'] == 'skipped_tool_continuation'


@pytest.mark.parametrize('path', ['openai', 'anthropic', 'openai_stream', 'anthropic_stream'])
def test_real_forward_hook_uses_serialized_provider_payload_and_no_internal_manifest_leaks(tmp_path, path):
    sent = []
    def upstream(request):
        sent.append(__import__('json').loads(request.content))
        if path.endswith('_stream'):
            return httpx.Response(200, content=b'data: [DONE]\n\n', headers={'Content-Type': 'text/event-stream'})
        if path == 'anthropic':
            return httpx.Response(200, json={'content': [{'type': 'text', 'text': 'completed'}], 'stop_reason': 'end_turn'})
        return httpx.Response(200, json=openai_response(finish_reason='stop'))
    service, route = gateway_service(upstream)
    service.memory_delivery = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    projection, _, wire = source_fixture()
    payload = {**wire, 'model': 'public-model', '_ombre_trace_context': {**SCOPE, 'trace_id': 'trace-1'}}
    debug = {'memory_recall_projection': projection}
    service._prepare_memory_delivery(payload, debug)
    async def scenario():
        if path == 'openai': response = await service._forward_upstream(payload)
        elif path == 'anthropic': response = await service._forward_anthropic_upstream(payload, route)
        elif path == 'openai_stream': response = await service._open_upstream_stream(route, payload)
        else: response = await service._open_anthropic_upstream_stream(route, payload)
        assert response.status_code == 200
        assert '_ombre_trace_context' not in sent[0]
        if path.endswith('_stream'):
            assert service.memory_delivery.complete(SCOPE) is None, 'headers alone cannot prove successful completion'
            await response.aread()
            # The existing successful-final-output handler is the completion gate.
            service.state_store = SimpleNamespace(record_success=lambda *_: 1, record_injection_debug=lambda *_: None)
            service._record_conversation_turn = lambda **_: None
            await service._record_successful_round('session', [], debug, assistant_message={'content': 'completed'})
        assert service.memory_delivery.complete(SCOPE)['fragments'][0]['text'] == projection['delivery_candidates'][0]['text']
        await service.http_client.aclose()
    run(scenario())


def test_mixed_unavailable_fragment_does_not_block_known_usage(tmp_path):
    store, buckets = MemoryDeliveryStore(tmp_path / 'gateway.sqlite'), Buckets()
    _, modern, modern_wire = source_fixture()
    _, legacy, legacy_wire = source_fixture(bid='unknown-legacy', legacy=True)
    attempt = store.begin(SCOPE, 'mixed', [*final_fragments(modern, modern_wire), *final_fragments(legacy, legacy_wire)])
    store.succeed(attempt)
    context = store.complete(SCOPE)
    query = use_query(context)
    query['uses'] = [{**query['uses'][0], 'use_id': 'use-' + str(index), 'fragment_ref': fragment['fragment_id']}
        for index, fragment in enumerate(context['fragments'])]
    states = [{'fragment_id': f['fragment_id'], 'state': 'available' if f['l1_revision'] else 'unavailable',
               'reason': 'source_unknown'} for f in context['fragments']]
    result = run(store.activate(query, context, buckets, fragment_states=states))
    assert sorted(row['status'] for row in result['results']) == ['activated', 'not_activated']
    assert buckets.calls[0][0] == 'bucket-1' and len(buckets.calls) == 1


def test_authenticated_lookup_keeps_exact_receipt_and_validates_mixed_sources_separately(tmp_path, monkeypatch):
    service, _ = gateway_service(lambda _: httpx.Response(200, json={}))
    service.memory_delivery = MemoryDeliveryStore(tmp_path / 'gateway.sqlite')
    _, frozen, wire = source_fixture()
    _, legacy, legacy_wire = source_fixture(bid='legacy', legacy=True)
    attempt = service.memory_delivery.begin(SCOPE, 'trace-mixed', [*final_fragments(frozen, wire), *final_fragments(legacy, legacy_wire)])
    service.memory_delivery.succeed(attempt)
    context = service.memory_delivery.complete(SCOPE)
    async def get(_):
        return {'content': 'changed old body', 'metadata': {}}
    service.bucket_mgr = SimpleNamespace(get=get)
    async def read_batch(_service, _request, stage, _path, payload):
        if stage == 'delivery_l2_validation':
            return {'items': [{'memory_id': 'memory-bucket-1', 'state': 'ready', 'presentation_revision': 2}]}
        return {'items': [{'memory_id': 'memory-bucket-1', 'revision': 3, 'state': 'ok',
            'sources': {'binding-1': {'state': 'available', 'eligible_source': True}}}]}
    monkeypatch.setattr('memory_recall_runtime.read_batch', read_batch)
    app = create_gateway_app(config={'gateway': {}}, service=service)
    app.state.gateway_service = service
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local.test') as client:
            path = '/internal/memory-delivery/lookup'
            assert (await client.post(path, json=SCOPE)).status_code == 401
            response = await client.post(path, json=SCOPE, headers={'Authorization': 'Bearer gateway-test-token'})
            result = response.json()
            assert result['context'] == context and result['state'] == 'available'
            assert sorted(row['state'] for row in result['fragment_states']) == ['available', 'changed']
        await service.http_client.aclose()
    run(scenario())


def test_core_and_favorite_collect_exact_generated_summary_not_fresh_model_call():
    service = GatewayService.__new__(GatewayService)
    service._summarize_bucket = lambda _: asyncio.sleep(0, result='the already-generated summary')
    service._trim_text = lambda text, _: text
    with capture_carriers() as rows:
        text = run(service._summarize_buckets([{'id': 'pinned', 'content': 'original body'}], 100))
    assert rows[0]['text'] == text and rows[0]['section'] == 'core'
    assert rows[0]['legacy_body_sha256'] == sha('original body')


@pytest.mark.parametrize('field,section', [('core_memory', 'core'), ('favorite_memory', 'favorite'),
    ('date_recall', 'date'), ('date_persona_trace', 'date_persona'), ('dream_context', 'dream')])
def test_actual_legacy_carrier_compilation_keeps_source_fragment_and_final_wire_proof(tmp_path, monkeypatch, field, section):
    from test_memory_recall_projection_r2 import _fixture
    _authority, service, _view, _buckets, _plan = _fixture(tmp_path, monkeypatch)
    sections = {'persona_block': '', 'core_memory': '', 'portrait_memory': ''}
    text = 'A real source-backed legacy memory, retaining the complete surviving phrase.'
    sections[field] = text
    projection = {'rendered_memory_spans': [], 'carrier_memory_spans': [
        {'bucket_id': 'legacy-source', 'section': section, 'text': text, 'legacy_body_sha256': sha('old source body')}],
        'presentation_trace': []}
    stable, dynamic = service._build_injected_context_messages(**sections, memory_recall_projection=projection)
    wire = service._inject_context_messages([{'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'current question'}], stable, dynamic)
    frozen = compiled_candidates(projection, candidates(projection))
    fragments = final_fragments(frozen, {'messages': wire})
    assert len(fragments) == 1
    assert fragments[0]['text'] == text and fragments[0]['coverage']['section'] == section
    assert fragments[0]['l1_revision'] is None and fragments[0]['l2_revision'] is None
    assert fragments[0]['legacy_body_sha256'] == sha('old source body')
