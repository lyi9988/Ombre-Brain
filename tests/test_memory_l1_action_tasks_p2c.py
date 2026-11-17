import asyncio
import copy
import json
from datetime import datetime
from types import SimpleNamespace as NS

import httpx
import pytest

from bucket_manager import BucketManager
from memory_authority import IdempotencyConflict
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_ingestion import model_template
from memory_l1_schema import statement_template
from memory_l1_source_tasks import MemoryL1SourceTasks
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config


def run(coro): return asyncio.run(coro)


class FakeCompletionClient:
    """Capture the actual routed chat.completions.create request offline."""

    def __init__(self, content):
        self.content = content
        self.calls = []
        self.chat = NS(completions=self)
        self.base_url = 'https://offline.example/v1'

    def with_options(self, **kwargs):
        return self

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        await asyncio.sleep(0)
        return NS(choices=[NS(
            message=NS(content=self.content),
            finish_reason='stop',
        )])


def setup(tmp_path, monkeypatch, mode='review', *, mock_action_reader=True):
    config = reflection_config(tmp_path, mode=mode)
    engine = ReflectionEngine(config)
    buckets = BucketManager(config)
    tasks = MemoryL1SourceTasks(engine.memory_authority_store)
    source = {'state': 'available', 'eligible_source': True, 'source_id': 'action-a',
        'source_revision': 'artifact-a', 'text': '执行完成：浏览了天文学资料。', 'author': {'role': 'tool'},
        'coverage': {'complete': True}}
    async def capture(self, action_id, artifact_id):
        assert action_id == 'action-a' and artifact_id == 'artifact-a'
        return copy.deepcopy(source)
    if mock_action_reader:
        monkeypatch.setattr('memory_l1_source_readers.AizActionSourceReader.capture', capture)
    draft = {'format': 'semantic-v1', 'entities': [],
        'narrative': {'title': '浏览天文学之后', 'body': '我完成浏览，留下这次经历。' * 600},
        'retrieval': {'keywords': ['天文学']}, 'statements': [{
            'content': '这次浏览了天文学资料。', 'subject_ref': None,
            'classification': {'kind': 'experience', 'stability': 'episode', 'modality': 'assertion', 'basis': 'record'},
            'time': None, 'conditions': None, 'evidence_refs': ['source:1']}]}
    client = FakeCompletionClient(json.dumps({'candidates': [
        {'document': draft, 'kind': 'key_event', 'confidence': 0.9, 'source_refs': ['source:1'], 'reason': '实际回执'}]}))
    engine.config['reflection']['memory_task_routes'] = {
        'action_l1': {
            'connection': 'dehydration',
            'model': 'synthetic-action-l1-v1',
            'thinking_mode': 'enabled',
            'reasoning_effort': 'medium',
            'max_tokens': 4321,
        },
    }
    engine.config['dehydration'] = {'thinking_mode': 'disabled'}
    engine.dehydration_client = client
    engine.dehydration_model = 'synthetic-dehydration-default'
    calls = client.calls
    payload = {'action_id': 'action-a', 'artifact_id': 'artifact-a', 'request_id': 'run-a'}
    return engine, buckets, tasks, payload, calls, source


def test_queue_is_identifiers_only_idempotent_and_conflicting_request_is_not_repurposed(tmp_path, monkeypatch):
    engine, _, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    first = tasks.enqueue_action(payload)
    replay = tasks.enqueue_action(payload)
    assert first['task_id'] == replay['task_id'] and replay['status'] == 'exists'
    assert first['adopted'] is False and calls == []
    assert len(tasks.list()) == 1
    assert 'text' not in tasks.list()[0] and 'body' not in tasks.list()[0]
    with pytest.raises(IdempotencyConflict): tasks.enqueue_action({**payload, 'artifact_id': 'artifact-other'})
    assert len(tasks.list()) == 1 and engine.memory_authority_store.list_memories() == []


@pytest.mark.parametrize('action_mode', ['review', 'off'])
def test_daily_due_respects_v2_action_mode_when_chat_memory_is_off(tmp_path, monkeypatch, action_mode):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch, mode='off')
    engine.enabled = engine.auto_enabled = True
    engine.config['reflection']['daily_chat_memory_extraction_mode'] = 'whole_window_v2'
    engine.memory_ingestion_policy.source_modes['action_receipt'] = action_mode
    engine.daily_chat_memory_review_max_per_day = 0
    engine.daily_chat_memory_hour = 0
    engine._local_now = lambda: datetime(2026, 11, 14, 8, tzinfo=engine.tz)
    legacy = tasks.enqueue_action(payload)['task_id']
    action = tasks.enqueue_action({**payload, 'request_id': 'v2-action', 'memory_intent': {
        'activity_id': 'activity-a', 'activity_revision': 1,
        'decision': 'remember', 'draft': '我读完了那份资料。',
        'sources': [{'action_id': 'action-a', 'artifact_id': 'artifact-a'}],
    }}, generation_protocol='whole_window_v2')['task_id']
    prepared = []

    async def prepare(_engine, _buckets, _tasks, claimed, _reader, mode, *, force):
        prepared.append((claimed['task_id'], mode))
        return {'candidate_ids': ['fixture-action-candidate']}

    monkeypatch.setattr('memory_l1_runtime._prepare_v2_action', prepare)
    run(engine.run_due(buckets))
    assert prepared == ([(action, 'review')] if action_mode == 'review' else [])
    assert tasks.get(legacy)['phase'] == 'queued'
    assert calls == []


def test_action_semantic_draft_assembles_full_l1_with_original_receipt_and_route(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    engine.dehydration_client.content = json.dumps({'candidates': [{
        'kind': 'key_event', 'confidence': 0.9, 'source_refs': ['source:1'], 'reason': '实际回执', 'document': {
            'format': 'semantic-v1', 'narrative': {'title': '行动经验', 'body': '我完成浏览并留下完整经验。'},
            'entities': [], 'retrieval': {'keywords': ['天文学']}, 'statements': [{'content': '这次浏览了天文学资料。', 'subject_ref': None,
                'classification': {'kind': 'experience', 'stability': 'episode', 'modality': 'assertion', 'basis': 'record'},
                'time': None, 'conditions': None, 'evidence_refs': ['source:1']}]}}]})
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    assert result['results'][0]['status'] == 'prepared'
    proposal = engine.memory_authority_store.list_candidates(status='all')[0]['proposal']
    assert proposal['proposed_body'] == '我完成浏览并留下完整经验。'
    semantic = proposal['metadata']['memory_l1']
    assert semantic['statements'][0]['classification']['basis'] == {
        'state': 'known', 'value': 'record', 'evidence_refs': ['source:1']}
    assert semantic['evidence_bindings'][0]['source_namespace'] == 'snapshot.aiz.action_receipt'
    request = json.loads(calls[0]['messages'][1]['content'])
    contract = request['l1_contract']
    assert contract['source_texts'][0]['text'] == source['text']
    assert 'semantic_contract' in contract
    assert not {'document_schema', 'catalog', 'document_template'}.intersection(contract)
    assert calls[0]['model'] == 'synthetic-action-l1-v1'
    assert calls[0]['max_tokens'] == 4321
    assert calls[0]['reasoning_effort'] == 'medium'


def test_review_worker_keeps_full_l1_and_does_not_claim_adoption(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    record = result['results'][0]
    assert record['status'] == 'prepared' and record['adopted'] is False
    assert len(calls) == 1
    assert calls[0]['model'] == 'synthetic-action-l1-v1'
    assert calls[0]['max_tokens'] == 4321
    assert calls[0]['extra_body'] == {'thinking': {'type': 'enabled'}}
    assert calls[0]['reasoning_effort'] == 'medium'
    item = engine.memory_authority_store.get_candidate(record['candidate_ids'][0])
    assert item['status'] == 'pending' and len(item['proposal']['proposed_body']) > 4000
    semantic = item['proposal']['metadata']['memory_l1']
    assert semantic['evidence_bindings'][0]['source_namespace'] == 'snapshot.aiz.action_receipt'
    assert semantic['statements'][0]['subject_ref']['state'] == 'not_stated'
    assert semantic['statements'][0]['time']['event']['start']['state'] == 'not_stated'
    assert engine.memory_authority_store.list_memories() == []


def test_auto_worker_uses_original_policy_and_commits_complete_semantics(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch, mode='auto')
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    record = result['results'][0]
    assert record['adopted'] is True
    ident = record['candidate_ids'][0]
    assert engine.memory_authority_store.get_candidate(ident)['status'] == 'committed'
    revision = engine.memory_authority_store.get_memory_revision(ident, 1)
    assert revision['metadata']['memory_l1']['generation']['actor'] == 'action_first_pass'
    assert revision['metadata']['memory_l1_content_sha256']
    assert len(run(buckets.get(ident))['content']) > 4000
    assert run(engine._memory_l1_action_tasks(buckets))['results'] == [] and len(calls) == 1


def test_same_task_claim_does_not_make_two_model_calls(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    async def concurrent(): return await asyncio.gather(engine._memory_l1_action_tasks(buckets), engine._memory_l1_action_tasks(buckets))
    result = run(concurrent())
    assert sum(len(row['results']) for row in result) == 1 and len(calls) == 1


def test_nonfinal_source_fails_visibly_without_generating_or_adopting(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    source.update(state='unknown', eligible_source=False)
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    assert result['results'][0]['status'] == 'failed' and calls == []
    assert tasks.list()[0]['phase'] == 'failed'
    assert engine.memory_authority_store.list_candidates(status='all') == []


def test_generation_zero_disables_processing_without_discarding_queue(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    engine.config['reflection']['action_l1_jobs_per_run'] = 0
    tasks.enqueue_action(payload)
    assert run(engine._memory_l1_action_tasks(buckets))['status'] == 'disabled'
    assert tasks.list()[0]['phase'] == 'queued' and calls == []


def test_source_failure_does_not_accept_or_lock_pending_owner_draft(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    made = run(engine._memory_l1_action_tasks(buckets))['results'][0]
    ident = made['candidate_ids'][0]
    source.update(state='unknown', eligible_source=False)
    before = engine.memory_authority_store.get_candidate(ident)
    from memory_l1_service import L1OwnerError
    with pytest.raises(L1OwnerError, match='source_not_currently_readable'):
        run(engine._memory_l1_adopt_candidate(before, buckets, request_id='owner-retry', owner=True))
    assert engine.memory_authority_store.get_candidate(ident) == before
    assert before['status'] == 'pending' and engine.memory_authority_store.list_memories() == []


def test_unneeded_source_does_not_block_valid_evidence(tmp_path, monkeypatch):
    from memory_authority import MemoryProposal
    from memory_l1_service import L1OwnerError
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    ident = run(engine._memory_l1_action_tasks(buckets))['results'][0]['candidate_ids'][0]
    row = engine.memory_authority_store.get_candidate(ident)
    proposal = copy.deepcopy(row['proposal'])
    semantic = proposal['metadata']['memory_l1']
    extra = copy.deepcopy(semantic['evidence_bindings'][0]); extra['binding_id'] = 'source:unused'
    locator = json.loads(extra['source_id']); locator['event_id'] = 'action-other'
    from memory_l1_contract import canonical_json
    extra['source_id'] = canonical_json(locator)
    extra['anchors'][0]['target_id'] = extra['source_id']
    semantic['evidence_bindings'].append(extra)
    row = engine.memory_authority_store.revise_candidate(ident, expected_revision=row['revision'],
        proposal=MemoryProposal.from_mapping(proposal), request_id='fixture-with-extra-source', actor='test')
    class Readers:
        async def locate(self, binding):
            available = binding['binding_id'] == 'source:1'
            return {'eligible_source': available, 'state': 'verified_snapshot' if available else 'withdrawn'}
    engine._memory_l1_sources = lambda: Readers()
    result = run(engine._memory_l1_adopt_candidate(row, buckets, request_id='owner-valid', owner=True))
    assert result['status'] == 'created'
    record = engine.memory_authority_store.get_memory_revision(ident, 1)
    assert record['metadata']['source_observations_at_adoption']['source:unused']['eligible_source'] is False


def test_owner_task_api_auth_cas_retry_and_no_model_side_effect(tmp_path, monkeypatch):
    import server
    from starlette.requests import Request
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(server, 'memory_authority_store', engine.memory_authority_store)
    monkeypatch.setattr(server, 'reflection_engine', engine)
    monkeypatch.setattr(server, '_memory_write_token', lambda: 'task-token')
    def request(method, body=None, *, authorized=True, task_id=None):
        async def receive(): return {'type': 'http.request', 'body': json.dumps(body or {}).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': method, 'path': '/api/memory-authority/action-source',
            'query_string': b'', 'path_params': {'task_id': task_id} if task_id else {},
            'headers': [(b'authorization', b'Bearer task-token')] if authorized else []}, receive)
    async def check():
        # Auth helper accepts owner sessions too; force neither credential for
        # the unauthorized request without changing the real bearer path.
        assert (await server.api_memory_l1_source_tasks(request('POST', payload, authorized=False))).status_code == 401
        posted = await server.api_memory_l1_source_tasks(request('POST', payload))
        assert posted.status_code == 202 and not calls
        ident = json.loads(posted.body)['task_id']
        assert json.loads((await server.api_memory_l1_source_tasks(request('GET'))).body)['tasks'][0]['task_id'] == ident
        tasks.transition(ident, 'queued', 'failed', issue={'code': 'synthetic'})
        engine._daily_chat_memory_run_lock = True
        assert (await server.api_memory_l1_source_tasks(request('POST', {'expected_phase':'failed'}, task_id=ident))).status_code == 409
        engine._daily_chat_memory_run_lock = False
        retried = await server.api_memory_l1_source_tasks(request('POST', {'expected_phase':'failed'}, task_id=ident))
        assert retried.status_code == 200 and json.loads(retried.body)['action_reexecuted'] is False
        stale = await server.api_memory_l1_source_tasks(request('POST', {'expected_phase':'failed'}, task_id=ident))
        assert stale.status_code == 409 and tasks.list()[0]['phase'] == 'queued' and not calls
    run(check())


def test_source_withdrawn_during_model_generation_is_not_auto_adopted(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch, mode='auto')
    original = engine._daily_chat_memory_create_completion
    async def completed_then_withdrawn(*args, **kwargs):
        result = await original(*args, **kwargs)
        source.update(state='withdrawn', eligible_source=False)
        return result
    engine._daily_chat_memory_create_completion = completed_then_withdrawn
    tasks.enqueue_action(payload)
    row = run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert row['status'] == 'prepared' and row['adopted'] is False
    assert row['adoption_issues'][0]['code'] == 'source_not_currently_readable'
    assert engine.memory_authority_store.list_memories() == []
    assert engine.memory_authority_store.get_candidate(row['candidate_ids'][0])['status'] == 'accepted'


@pytest.fixture
def active_reading(tmp_path, monkeypatch):
    import server
    from starlette.requests import Request

    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch, mock_action_reader=False)
    engine.config['reflection']['daily_chat_memory_extraction_mode'] = 'whole_window_v2'
    engine.config['gateway']['canonical_continuation'] = {
        'enabled': True, 'conversation_id': 'conv-jiajia-main', 'device_id': 'fixture-brain',
        'base_url': 'https://fixture-aiz.invalid',
    }
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'fixture-bridge-token')
    body = '我想以后找得到笔记的原始出处，读了教程的两节。\r\n我打算把作者的话和自己的理解分开记。'
    full_article = '先记下出处。这里是没有读到的一节。再区分引文。'
    fragments = ['先记下出处。', '再区分引文。']
    skipped = '这里是没有读到的一节。'
    first_length, second_length = map(len, fragments)
    reading = {'material_id': 'material-a', 'source_revision': 'article-revision-a',
        'read_ranges': [[0, first_length], [first_length + len(skipped), len(full_article)]]}
    identity_hash = semantic_sha256({key: reading[key] for key in ('material_id', 'source_revision')})
    payload.update(request_id='action-memory-intent:' + identity_hash, memory_intent={
        'decision': 'remember', 'draft': body, 'activity_id': 'activity-a', 'activity_revision': 'stage-1',
        'sources': [{'action_id': payload['action_id'], 'artifact_id': payload['artifact_id']}],
        'reading': copy.deepcopy(reading),
    })
    receipt_result = {'ok': True, 'reading': {**copy.deepcopy(reading), 'reading_actor': '顾衍'},
        'evidence_bundle': {'summary': '试着分别记来源与理解。'}}
    receipt_time = '2026-11-15T17:07:13+00:00'
    receipt = {'state': 'available', 'eligible_source': True, 'source_id': payload['action_id'],
        'source_namespace': 'snapshot.aiz.action_receipt', 'source_revision': payload['artifact_id'],
        'text': canonical_json(receipt_result), 'recorded_at': receipt_time, 'action_status': 'completed',
        'completion_kind': 'executor', 'tool_id': 'browse.fetch_page_text', 'coverage': {'complete': True}}
    article = {'state': 'available', 'eligible_source': True, 'source_revision': 'browse_sha256:' + 'a' * 64,
        'source_namespace': 'snapshot.aiz.browse.article', 'material_id': reading['material_id'],
        'source_id': canonical_json({**reading, 'view': 'article'}), 'content_revision': reading['source_revision'],
        'text': '\n\n'.join(fragments), 'read_ranges': [[0, first_length], [first_length + 2, first_length + 2 + second_length]],
        'coverage': {'read_ranges': copy.deepcopy(reading['read_ranges'])}, 'text_view_id': 'browse.article.exact-v1',
        'title': '笔记教程', 'author': '教程作者', 'recorded_at': receipt_time}
    bridge_calls = []

    def bridge(request):
        assert request.method == 'GET' and request.url.path == '/bridge/v1/memory-source'
        assert request.headers['Authorization'] == 'Bearer fixture-bridge-token'
        assert request.headers['X-Guyan-Device-ID'] == 'fixture-brain'
        params = dict(request.url.params)
        bridge_calls.append(params)
        if params['kind'] == 'action_receipt':
            assert params['source_id'] == receipt['source_id'] and params['version'] == receipt['source_revision']
            source = receipt
        else:
            assert params['kind'] == 'browse_material'
            assert params['source_id'] == article['source_id']
            assert params['version'] in {'current', article['source_revision']}
            source = article
        return httpx.Response(200, json={'source': source})

    client = httpx.AsyncClient
    monkeypatch.setattr('memory_l1_source_readers.httpx.AsyncClient',
        lambda **kwargs: client(transport=httpx.MockTransport(bridge), **kwargs))
    monkeypatch.setattr(server, 'config', engine.config)
    monkeypatch.setattr(server, 'memory_authority_store', engine.memory_authority_store)
    monkeypatch.setattr(server, 'reflection_engine', engine)
    monkeypatch.setattr(server, '_memory_write_token', lambda: 'fixture-writer')
    card = {'title': '给阅读笔记留出处', 'kind': 'knowledge_discovery',
        'stability': 'episode', 'primary_topic': 'learning_career', 'keywords': ['原文出处', '阅读笔记'],
        'people': ['p1'], 'sources': ['m1', 'm2'],
        'event_time': {'start': '2026-11-16', 'precision': 'day', 'sources': ['m2']}}

    def model_returns(cards):
        engine.dehydration_client.content = canonical_json({'cards': cards})

    model_returns([card])

    def post(value=payload, *, authorized=True):
        async def receive():
            return {'type': 'http.request', 'body': canonical_json(value).encode(), 'more_body': False}
        request = Request({'type': 'http', 'method': 'POST', 'path': '/api/memory-authority/action-source',
            'query_string': b'', 'headers': [(b'authorization', b'Bearer fixture-writer')] if authorized else []}, receive)
        response = run(server.api_memory_l1_source_tasks(request))
        return response.status_code, json.loads(response.body)

    return NS(engine=engine, buckets=buckets, tasks=tasks, post=post, payload=payload, reading=reading,
        task_id='action_source_' + identity_hash[:40], calls=calls, bridge_calls=bridge_calls,
        card=card, model_returns=model_returns, body=body, article=article, receipt=receipt,
        receipt_result=receipt_result, fragments=fragments, skipped=skipped)


@pytest.mark.parametrize('unsolicited_body', [False, True])
def test_active_reading_prepares_and_adopts_one_card_with_the_original_draft_and_exact_sources(active_reading, unsolicited_body):
    from memory_l1_v2_generation import ACTIVE_PROMPT_ID
    from memory_l1_v2_prompts import L1_V2_ACTIVE_PROMPT

    f = active_reading
    assert f.post(authorized=False)[0] == 401
    assert f.bridge_calls == []
    if unsolicited_body:
        f.model_returns([{**f.card, 'body': '我做了原稿中没有提及的事情。'}])
    status, posted = f.post()
    assert status == 202 and posted['task_id'] == f.task_id and not posted['adopted']
    assert posted['retention']['retained'] and not f.calls
    assert posted['memory_intent_sha256'] == semantic_sha256({
        key: f.payload[key] for key in ('action_id', 'artifact_id', 'memory_intent')})
    prepared = run(f.engine._memory_l1_action_tasks(f.buckets))['results'][0]
    assert prepared['status'] == 'prepared' and not prepared['adopted'], prepared
    candidate_id, = prepared['candidate_ids']
    candidate = f.engine.memory_authority_store.get_candidate(candidate_id)
    assert candidate['status'] == 'pending'
    body = '2026年11月16日\n\n' + f.body.replace('\r\n', '\n')
    assert candidate['proposal']['proposed_body'] == body
    assert f.tasks.get(f.task_id)['memory_intent']['draft'] == f.body

    request, = f.calls
    system, material = request['messages']
    assert system['content'] == L1_V2_ACTIVE_PROMPT
    assert f.skipped not in material['content']
    payload = json.loads(material['content'])
    assert payload['input_kind'] == 'action_l1'
    assert payload['agent_draft']['source_ref'] == 'm1'
    assert payload['messages'][0][3] == f.body
    assert [row[3] for row in payload['messages'] if row[0] == 'm2'] == f.fragments
    assert payload['read_coverage'][0]['coverage']['read_ranges'] == f.reading['read_ranges']
    assert payload['reading_event'] == {'source_ref': 'm3', 'occurred_at': '2026-11-16T01:07:13+08:00'}
    cards = request['response_format']['json_schema']['schema']['properties']['cards']
    assert cards['minItems'] == cards['maxItems'] == 1
    assert 'body' not in cards['items']['required']

    metadata = candidate['proposal']['metadata']['memory_l1']
    assert metadata['kind'] == 'knowledge_discovery'
    assert metadata['generation']['prompt_source_id'] == ACTIVE_PROMPT_ID
    assert metadata['event_time']['sources'] == ['source:2', 'source:3']
    draft, article, receipt = metadata['evidence_bindings']
    assert draft['source_namespace'] == 'snapshot.ombre.agent_memory_draft'
    assert article['source_namespace'] == 'snapshot.aiz.browse.article'
    assert receipt['source_namespace'] == 'snapshot.aiz.action_receipt'
    sources = f.engine._memory_l1_sources()
    assert sources.copy(draft)['text'] == f.body
    assert sources.copy(article)['text'] == '\n\n'.join(f.fragments)
    assert any(call['version'] == f.article['source_revision'] for call in f.bridge_calls)

    adopted = run(f.engine._memory_l1_adopt_candidate(candidate, f.buckets, request_id='owner-accept-fixture', owner=True))
    assert adopted['status'] == 'created'
    revision = f.engine.memory_authority_store.get_memory_revision(candidate_id, 1)
    assert revision['metadata']['memory_l1']['evidence_bindings'] == metadata['evidence_bindings']
    assert run(f.buckets.get(candidate_id))['content'].strip() == body


def test_active_reading_replay_keeps_original_task_and_candidate(active_reading):
    f = active_reading
    assert f.post()[0] == 202
    replay_status, replay = f.post()
    assert replay_status == 202 and replay['status'] == 'exists' and replay['task_id'] == f.task_id
    continued = copy.deepcopy(f.payload)
    continued['memory_intent']['draft'] = '读过以后又改变了措辞。'
    continued['memory_intent']['activity_revision'] = 'stage-2'
    continued['memory_intent']['reading']['read_ranges'] = [[0, 1]]
    assert f.post(continued) == (409, {'error': 'idempotency_conflict'})
    assert len(f.tasks.list()) == 1 and f.tasks.get(f.task_id)['memory_intent'] == f.payload['memory_intent']

    async def concurrent():
        return await asyncio.gather(f.engine._memory_l1_action_tasks(f.buckets), f.engine._memory_l1_action_tasks(f.buckets))

    result = run(concurrent())
    assert sum(len(item['results']) for item in result) == 1
    assert len(f.calls) == 1 and len(f.engine.memory_authority_store.list_candidates(status='all')) == 1


@pytest.mark.parametrize('invalid', ['shape', 'range', 'request_id'])
def test_active_reading_rejects_invalid_submission_shape(active_reading, invalid):
    f = active_reading
    payload = copy.deepcopy(f.payload)
    if invalid == 'shape':
        payload['memory_intent']['reading']['extra'] = True
    elif invalid == 'range':
        payload['memory_intent']['reading']['read_ranges'] = [[False, 2], [2, 1]]
    else:
        payload['request_id'] = 'new-request-but-same-material'
    status, issue = f.post(payload)
    assert status == 400
    assert issue['error'] == ('action_reading_request_id_mismatch' if invalid == 'request_id' else 'invalid_action_reading')
    assert f.tasks.list() == [] and f.calls == [] and f.bridge_calls == []


@pytest.mark.parametrize('problem,reason', [
    ('not_completed', 'reading_action_receipt_mismatch'),
    ('unsuccessful', 'reading_action_receipt_mismatch'),
    ('receipt_ranges', 'reading_action_receipt_mismatch'),
    ('new_revision', 'reading_content_revision_changed'),
    ('article_ranges', 'reading_footprint_changed'),
    ('unread_text', 'reading_footprint_changed'),
])
def test_active_reading_rejects_unverified_reading_before_generation(active_reading, problem, reason):
    f = active_reading
    if problem == 'not_completed':
        f.receipt['action_status'] = 'failed'
    elif problem == 'unsuccessful':
        f.receipt['text'] = canonical_json({**f.receipt_result, 'ok': False})
    elif problem == 'receipt_ranges':
        f.receipt_result['reading']['read_ranges'] = [[0, 1]]
        f.receipt['text'] = canonical_json(f.receipt_result)
    elif problem == 'new_revision':
        f.article['content_revision'] = 'other-revision'
    elif problem == 'article_ranges':
        f.article['coverage']['read_ranges'] = [[0, 1]]
    else:
        f.article['text'] += f.skipped
    assert f.post()[0] == 202
    row = run(f.engine._memory_l1_action_tasks(f.buckets))['results'][0]
    assert row['status'] == 'failed' and row['issue']['code'] == reason
    if problem in {'not_completed', 'unsuccessful', 'receipt_ranges'}:
        assert all(call['kind'] == 'action_receipt' for call in f.bridge_calls)
    assert not f.calls and f.engine.memory_authority_store.list_candidates(status='all') == []
    assert f.tasks.get(f.task_id)['memory_intent']['draft'] == f.body


@pytest.mark.parametrize('problem,reason', [
    ('zero', 'native_l1_schema_mismatch'), ('two', 'native_l1_schema_mismatch'),
    ('missing_draft', 'l1_v2_primary_source_required'), ('missing_article', 'l1_v2_result_source_required'),
    ('wrong_date', 'l1_v2_reading_time_unsupported'),
    ('date_from_draft_only', 'l1_v2_reading_time_unsupported'),
])
def test_active_reading_rejects_invalid_model_card_without_rewriting_original(active_reading, problem, reason):
    f = active_reading
    card = copy.deepcopy(f.card)
    if problem == 'missing_draft':
        card['sources'] = ['m2', 'm3']
    if problem == 'missing_article':
        card['sources'] = ['m1', 'm3']
    if problem == 'wrong_date':
        card['event_time']['start'] = '2026-11-15'
    if problem == 'date_from_draft_only':
        card['event_time']['sources'] = ['m1']
    f.model_returns([] if problem == 'zero' else [card, card] if problem == 'two' else [card])
    assert f.post()[0] == 202
    row = run(f.engine._memory_l1_action_tasks(f.buckets))['results'][0]
    assert row['status'] == 'failed' and row['issue']['code'] == reason, row
    assert len(f.calls) == 1 and f.engine.memory_authority_store.list_candidates(status='all') == []
    assert f.tasks.get(f.task_id)['memory_intent']['draft'] == f.body


@pytest.mark.parametrize('known', [False, True])
def test_active_reading_keeps_already_dated_or_undated_original(active_reading, known):
    f = active_reading
    card = copy.deepcopy(f.card)
    if known:
        f.payload['memory_intent']['draft'] = '2026-11-16 我读了教程的两节。'
    else:
        card['event_time'] = None
        f.receipt['recorded_at'] = None
    f.model_returns([card])
    assert f.post()[0] == 202
    row = run(f.engine._memory_l1_action_tasks(f.buckets))['results'][0]
    assert row['status'] == 'prepared', row
    proposal = f.engine.memory_authority_store.get_candidate(row['candidate_ids'][0])['proposal']
    assert proposal['proposed_body'] == f.payload['memory_intent']['draft'].replace('\r\n', '\n')


def test_active_reading_review_checks_original_article_revision_again(active_reading):
    from memory_l1_service import L1OwnerError

    f = active_reading
    assert f.post()[0] == 202
    prepared = run(f.engine._memory_l1_action_tasks(f.buckets))['results'][0]
    candidate_id, = prepared['candidate_ids']
    candidate = f.engine.memory_authority_store.get_candidate(candidate_id)
    f.article['content_revision'] = 'new-revision'
    with pytest.raises(L1OwnerError, match='card_source_not_currently_readable'):
        run(f.engine._memory_l1_adopt_candidate(candidate, f.buckets, request_id='owner-fixture', owner=True))
    assert f.engine.memory_authority_store.get_candidate(candidate_id)['status'] == 'pending'
    assert f.engine.memory_authority_store.list_memories() == []


@pytest.mark.parametrize('placement', ['instructions.end', 'current_user.after', 'off', 'condition'])
def test_active_reading_uses_actual_composer_placement_and_switch(active_reading, placement):
    from memory_l1_v2_generation import ACTIVE_PROMPT_ID
    from memory_l1_v2_prompts import L1_V2_ACTIVE_PROMPT

    f = active_reading
    scope = 'memory.l1_v2_active'
    stage = 'ombre_post_injection'
    block = {'block_id': 'active-reading', 'scope': scope, 'stage': stage, 'source_id': ACTIVE_PROMPT_ID,
        'order': 20, 'wrapper_text': 'B\n{{content}}', 'mode': 'live_source', 'anchor': 'instructions.end'}
    if placement == 'off':
        block['mode'] = 'off'
    elif placement == 'condition':
        block['condition'] = {'rules': [{'source': 'gateway.scope', 'op': 'equals', 'value': 'memory.other'}]}
    elif placement == 'current_user.after':
        block.update(anchor=placement, lane='message', role='user')
    custom = [
        {'block_id': 'static-' + text, 'scope': scope, 'stage': stage,
         'source_id': 'custom.gateway_block', 'owner_body': text,
         'mode': 'owner_override', 'anchor': 'instructions.end', 'order': order}
        for order, text in ((30, 'C'), (10, 'A'))
    ]
    mirror = f.engine.prompt_plan_mirror
    sha = 'c' * 64
    mirror.put_plan('preset:reading', 1, plan_sha256=sha, gateway_slice={'blocks': [*custom, block]})
    mirror.put_binding(scope, preset_id='preset:reading', preset_revision=1, plan_sha256=sha, aiz_binding_revision=1)
    assert f.post()[0] == 202
    prepared = run(f.engine._memory_l1_action_tasks(f.buckets))['results'][0]
    if placement in {'off', 'condition'}:
        assert prepared['status'] == 'failed' and prepared['issue']['code'] == 'prompt_disabled'
        assert f.calls == []
        return
    assert prepared['status'] == 'prepared', prepared
    messages = f.calls[0]['messages']
    assert sum(message['content'].count(L1_V2_ACTIVE_PROMPT) for message in messages) == 1
    if placement == 'current_user.after':
        assert [message['content'] for message in messages[:2]] == ['A', 'C']
        assert json.loads(messages[2]['content'])['input_kind'] == 'action_l1'
        assert messages[3] == {'role': 'user', 'content': 'B\n' + L1_V2_ACTIVE_PROMPT}
    else:
        assert [message['content'] for message in messages[:3]] == ['A', 'B\n' + L1_V2_ACTIVE_PROMPT, 'C']
        assert json.loads(messages[3]['content'])['input_kind'] == 'action_l1'
