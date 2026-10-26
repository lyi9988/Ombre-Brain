"""Isolated cross-service reading store -> full L1 -> persistent L2 queue."""
import asyncio
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from memory_browse_preparation import BrowseMemoryPreparation, prepare_browse_tasks
from memory_l1_source_readers import AizBrowseSourceReader
from memory_l0_sources import MemoryL0Sources
from memory_l1_contract import canonical_json
from memory_l1_ingestion import bind_sources
from memory_browse_source import model_source, semantic_views
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_diary_l1_p2b import setup_diary


def load_aiz(name):
    path = Path(__file__).resolve().parents[2] / 'stage03-confirmed-aiz' / 'core' / (name + '.py')
    if not path.is_file():
        pytest.skip('cross-service fixture needs adjacent stage03-confirmed-aiz source tree')
    spec = importlib.util.spec_from_file_location('browse_fixture_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(tmp_path, monkeypatch, *, during=None, full_format=False, article_evidence=False):
    recent = load_aiz('recent_materials')
    bridge = load_aiz('browse_memory_source')
    now = NS(value=datetime(2026, 10, 25, 1, tzinfo=timezone.utc))
    store = recent.RecentMaterialStore(tmp_path / 'presentation.sqlite3', clock=lambda: now.value)
    body = '读到的内容。未读的秘密结尾。'
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    coverage = {'read_ranges': [[0, 6]], 'read_complete': False, 'extracted_complete': True,
        'understanding_status': 'completed', 'format': 'structured', 'quality_score': 2,
        'long_term_reasons': [{'automatic': True, 'kind': 'sustained_value', 'text': '这个解释可用于后续比较。'}]}
    row = store.record(scope='conv-jiajia-main', source_id='browse:original:urlhash', source_revision=content_hash,
        occurred_at=now.value, kind='reading', title='外部材料', body=body, source_refs=[], outcome='partial', coverage=coverage,
        metadata={'actor': 'guyan', 'reading_actor': '顾衍', 'owner_reading_evidence': False,
            'content_item_id': 'item-test', 'source_event_id': 'browse:original', 'reported_author': '外部署名'})
    store.publish(row['material_id'], row['revision'], facts=[{'text': '读到的内容', 'sources': ['C1']}],
        appraisal=[{'text': '这是一个待比较的解释', 'sources': ['C1']}], open_questions=[], coverage=coverage)
    store.hold(row['material_id'], 'longterm_pending', 'browse:original:key')
    env = setup_diary(tmp_path / 'memory', '日记不在本次范围')
    config = reflection_config(tmp_path / 'engine', mode='off')
    config['gateway'] = {'canonical_continuation': {'enabled': True, 'conversation_id': 'conv-jiajia-main'}}
    engine = ReflectionEngine(config)
    engine.memory_authority_store = env['authority']
    engine._memory_l1_revisions = lambda manager=None: env['revisions']
    readers = MemoryL0Sources(env['authority'], [AizBrowseSourceReader(config, view) for view in ('article', 'reading')])
    for reader in readers.adapters.values():
        reader.upstream_version = lambda binding: (readers.copy(binding) or {}).get('upstream_revision')
    engine._memory_l1_sources = lambda: readers
    async def capture(self, content_item_id, version='current'):
        return bridge.read(store, 'conv-jiajia-main', canonical_json({'content_item_id': content_item_id, 'view': self.view}), version)
    monkeypatch.setattr(AizBrowseSourceReader, 'capture', capture)
    calls = []
    async def create(client, **kwargs):
        calls.append(json.loads(kwargs['messages'][1]['content']))
        if during:
            during(store, row, coverage)
        doc = {'format': 'semantic-v1', 'entities': [], 'retrieval': {'keywords': ['阅读解释']},
            'statements': [{'content': '这次读到的解释仍需比较。', 'subject_ref': None,
                'classification': {'kind': None, 'stability': None, 'modality': None, 'basis': None},
                'time': None, 'conditions': None, 'evidence_refs': ['source:1', 'source:2'] if article_evidence else ['source:2']}]}
        if full_format:
            from memory_l1_semantic_wire import assemble_document
            doc = assemble_document(doc, narrative={'title': '模型题', 'body': '模型正文不能覆盖原稿'})
        return NS(choices=[NS(message=NS(content=json.dumps({'document': doc})), finish_reason='stop')])
    engine._memory_task_route = lambda stage: {'stage': stage, 'client': object(), 'model': 'fake-only',
        'connection': 'reflection', 'configured': True, 'thinking_mode': '', 'reasoning_effort': '', 'max_tokens': 16000}
    engine._daily_chat_memory_create_completion = create
    narrative = '我读了这份外部材料的一部分，这次理解仍需比较。'
    sha = hashlib.sha256(narrative.encode()).hexdigest()
    descriptor = {'schema': 'aiz.memory-source-ref-v1', 'namespace': 'snapshot.aiz.browse.summary',
        'source_id': 'item-test', 'source_revision': sha, 'content_sha256': sha,
        'author': {'role': 'assistant_self', 'producer': 'skill.web_browse'},
        'coverage': {'kind': 'external_source_summary', 'original_page_stored': False},
        'source_scope': {'schema': 'browse-experience-v1', 'source_kind': 'web_browse', 'producer': 'skill.web_browse',
            'actor_role': 'assistant_self', 'source_event_id': 'browse:original', 'reported_author': '外部署名',
            'claim_kind': 'external_source_summary', 'source_ref': {'content_item_id': 'item-test', 'source_domain': 'example.test',
                'content_hash': content_hash, 'summary_hash': 'summary-hash', 'canonical_url': 'https://example.test/article'}}}
    tasks = BrowseMemoryPreparation(env['authority'])
    kwargs = {'operation_key': 'browse:original:key', 'content': narrative, 'descriptor': descriptor,
        'metadata': {'name': '', 'tags': ['browse_note'], 'importance': 7}}
    receipt = tasks.enqueue(**kwargs)
    return NS(env=env, engine=engine, store=store, row=row, coverage=coverage, tasks=tasks, calls=calls,
        receipt=receipt, kwargs=kwargs, now=now, bridge=bridge, readers=readers)


@pytest.mark.parametrize('full_format', [False, True])
def test_real_store_to_formal_l1_and_l2_queue_without_chat_mode_dependency(tmp_path, monkeypatch, full_format):
    f = setup(tmp_path, monkeypatch, full_format=full_format)
    assert f.receipt['adopted'] is False and f.receipt['preparation_status'] == 'queued'
    assert f.env['authority'].list_memories() == [] and not f.calls
    assert f.tasks.enqueue(**f.kwargs) == f.receipt
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['adopted'] is True, result
    mid = f.receipt['task_id']
    revision = f.env['authority'].get_memory_revision(mid, 1)
    semantic = revision['metadata']['memory_l1']
    assert revision['metadata']['source'] == 'hold'
    assert revision['metadata']['aiz_source_descriptor']['descriptor'] == f.kwargs['descriptor']
    assert 'tool_operation:browse:original:key' in revision['source_refs']
    assert semantic['statements'] and semantic['retrieval']['keywords'] == ['阅读解释']
    assert semantic['evidence_bindings'][0]['source_namespace'] == 'snapshot.aiz.browse.article'
    assert semantic['evidence_bindings'][0]['anchors'][0]['end'] == 6
    article = f.readers.copy(semantic['evidence_bindings'][0])
    reading = f.readers.copy(semantic['evidence_bindings'][1])
    assert article['text'] == '读到的内容。未读的秘密结尾。'
    assert article['participant']['kind'] == 'unknown' and reading['participant']['kind'] == 'ai'
    assert '未读的秘密结尾' not in json.dumps(f.calls[0], ensure_ascii=False)
    assert f.calls[0]['draft']['narrative']['body'] == f.kwargs['content']
    detail = asyncio.run(f.env['revisions'].detail(mid))
    assert detail['body'] == f.kwargs['content'] and detail['l1']['state'] == 'current'
    with f.env['authority']._connect() as db:
        assert db.execute('SELECT count(*) FROM memory_l2_jobs WHERE memory_id=?', (mid,)).fetchone()[0] == 1
    asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert len(f.calls) == 1


@pytest.mark.parametrize('change', ['withdraw', 'understanding'])
def test_source_change_during_model_cannot_create_or_adopt_candidate(tmp_path, monkeypatch, change):
    def during(store, row, coverage):
        if change == 'withdraw':
            store.withdraw(row['material_id'], row['revision'])
        else:
            store.publish(row['material_id'], row['revision'], facts=[{'text': '已更正的不同解释'}],
                appraisal=[], open_questions=[], coverage=coverage)
    f = setup(tmp_path, monkeypatch, during=during)
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['status'] == 'failed'
    assert f.tasks.receipt(f.receipt['task_id'])['preparation_status'] == 'failed'
    assert f.env['authority'].list_memories() == []
    assert f.env['authority'].list_candidates(status='all') == []
    assert f.tasks.detail(f.receipt['task_id'])['draft']['content'] == f.kwargs['content']


def test_pending_retains_real_source_beyond_recent_window_but_not_revocation(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    f.now.value += timedelta(hours=25)
    assert f.store.cleanup() == 0
    assert f.store.list('conv-jiajia-main') == []
    assert asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['adopted']
    f.store.withdraw(f.row['material_id'], f.row['revision'])
    semantic = f.env['authority'].get_memory_revision(f.receipt['task_id'], 1)['metadata']['memory_l1']
    status = asyncio.run(f.readers.locate(semantic['evidence_bindings'][0]))
    assert status['eligible_source'] is False and status['view'] == 'source_copy'


def test_body_only_model_output_stays_failed_without_fabricated_l1(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    async def incomplete(*args, **kwargs):
        return NS(choices=[NS(message=NS(content=json.dumps({'document': {'narrative': {'body': '只有正文'}}})), finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion = incomplete
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['status'] == 'failed'
    assert f.env['authority'].list_memories() == []


def test_native_schema_keeps_full_contract_and_real_source_ids(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    from memory_l1_api import prepare_request
    observed = []
    original = f.engine._daily_chat_memory_create_completion
    async def inspect(*args, **kwargs):
        request, schema = prepare_request(kwargs['messages'], stage='chat_l1')
        observed.append(schema)
        return await original(*args, **kwargs)
    f.engine._daily_chat_memory_create_completion = inspect
    assert asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['adopted']
    assert observed[0]['$defs']['EvidenceRef']['enum'] == ['source:1', 'source:2']
    assert {'entities', 'statements', 'retrieval'}.issubset(observed[0]['$defs']['Document']['properties'])


def test_legacy_accepted_unknown_receipt_does_not_enqueue_or_replay(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    from memory_authority import MemoryProposal, PolicyDecision
    mid = 'memory_' + hashlib.sha256(b'old-key').hexdigest()[:16]
    f.env['authority'].put_candidate(MemoryProposal.from_mapping({'proposal_id': mid, 'proposed_body': '旧正文',
        'source_type': 'hold'}), PolicyDecision('auto_accept', ('OLD',), 'explicit_tool'))
    receipt = f.tasks.enqueue(**{**f.kwargs, 'operation_key': 'old-key'})
    assert receipt['memory_status'] == 'accepted' and receipt['adopted'] is False
    assert f.tasks.get(mid) is None


@pytest.mark.parametrize('phase', ['prepared', 'body_written', 'committed'])
def test_original_commit_receipt_is_not_replaced_or_retried_even_without_candidate(tmp_path, monkeypatch, phase):
    f = setup(tmp_path, monkeypatch)
    key = 'old-original-key'
    mid = 'memory_' + hashlib.sha256(key.encode()).hexdigest()[:16]
    digest = hashlib.sha256(f.kwargs['content'].encode()).hexdigest()
    authority = f.env['authority']
    operation = authority.prepare_memory_commit(memory_id=mid, bucket_id=mid, expected_revision=0,
        body_sha256=digest, snapshot_path='old-snapshot', metadata={}, source_refs=[],
        decision_source='tool', idempotency_key=key + ':revision', actor='hold')
    if phase in ('body_written', 'committed'):
        authority.record_body_written(operation['operation_id'], observed_body_sha256=digest)
    if phase == 'committed':
        authority.finalize_memory_commit(operation['operation_id'])
    prior = authority.get_commit_by_idempotency(key + ':revision')
    receipt = f.tasks.enqueue(**{**f.kwargs, 'operation_key': key})
    assert receipt['adopted'] is False
    assert receipt['preparation_status'] == 'legacy_receipt_requires_reconciliation'
    assert f.tasks.get(mid) is None
    assert authority.get_commit_by_idempotency(key + ':revision') == prior


@pytest.mark.parametrize('owner_edit', [False, True])
def test_real_aiz_receipt_reader_confirms_formal_revision_and_owner_lineage(tmp_path, monkeypatch, owner_edit):
    import httpx
    receipt_module = load_aiz('material_reconciliation')
    f = setup(tmp_path, monkeypatch)
    mid = f.receipt['task_id']
    f.store.release(f.row['material_id'], 'longterm_pending', f.kwargs['operation_key'])
    target = 'https://original-brain.invalid'
    receipt_module.begin_longterm_receipt(f.store, material_id=f.row['material_id'], scope='conv-jiajia-main',
        source_revision=f.row['revision'], idempotency_key=f.kwargs['operation_key'], content=f.kwargs['content'],
        source_metadata=f.kwargs['descriptor'], brain_url=target)
    intent = receipt_module._read_intent(f.store, f.row['material_id'], f.kwargs['operation_key'])
    if owner_edit:
        edit = {'expected_revision': 1, 'request_id': 'owner-correction',
            'draft': {'narrative': {'title': '主人纠正的标题', 'body': '我只读了前文，这个解释还要比较，不能当成定论。'}}}
        changed = f.tasks.edit(mid, edit)
        assert changed['task']['draft_revision'] == 2
        assert changed['history'][0]['draft']['content'] == f.kwargs['content']
        assert f.tasks.edit(mid, edit) == changed
        assert f.tasks.enqueue(**f.kwargs)['adopted'] is False
    payload = f.tasks.receipt(mid)
    async def scenario():
        nonlocal payload
        async def transport(request):
            assert request.method == 'GET'
            return httpx.Response(200, json=payload)
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            reader = receipt_module.BrainHoldReceiptReader(lambda: (target, 'synthetic'), client=client)
            assert (await reader.read(intent))['status'] == 'unknown'
            assert (await prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['adopted']
            payload = await f.env['revisions'].detail(mid)
            assert payload['revisions'][0]['body_sha256'] == hashlib.sha256(payload['body'].encode()).hexdigest()
            confirmed = await reader.read(intent)
            assert confirmed['status'] == 'confirmed', confirmed
            assert confirmed['body_matches_original_submission'] is (not owner_edit)
            assert receipt_module.confirm_longterm_receipt(f.store, material_id=f.row['material_id'], idempotency_key=f.kwargs['operation_key'])
            assert not f.store.pending_longterm_receipts()
            # Tampered new provenance cannot fall back to a legacy success.
            payload['revisions'][0]['metadata']['browse_preparation']['draft_body_sha256'] = '0' * 64
            assert (await reader.read(intent))['status'] == 'unknown'
    asyncio.run(scenario())


def test_successful_model_output_survives_failed_adoption_and_retry_reuses_it(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    original = f.engine._memory_l1_adopt_candidate
    first = True
    async def failed_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise RuntimeError('synthetic temporary commit outage')
        return await original(*args, **kwargs)
    f.engine._memory_l1_adopt_candidate = failed_once
    first_result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert first_result['results'][0]['status'] == 'failed'
    mid = f.receipt['task_id']
    row = f.tasks.get(mid)
    assert row['preparation_output']['text']
    def retry(row):
        row.update(phase='queued', lease_until=0, claim_token=None, issue=None)
        return row
    f.tasks.update(mid, expected_revision=row['revision'], transform=retry)
    second_result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert second_result['results'][0]['adopted'] is True
    assert len(f.calls) == 1
    assert f.env['authority'].get_memory(mid)['active_revision'] == 1


def test_shared_allowance_does_not_double_and_alternates_queues(monkeypatch):
    import memory_browse_preparation as browse
    import memory_chat_preparation as chat
    engine = NS(config={'reflection': {'chat_memory_jobs_per_run': 1}})
    calls = []
    async def chat_worker(engine, bucket_mgr, *, max_jobs):
        calls.append(('chat', max_jobs))
        return {'results': [{}]}
    async def browse_worker(engine, bucket_mgr, *, max_jobs):
        calls.append(('browse', max_jobs))
        return {'results': [{}]}
    monkeypatch.setattr(chat, 'prepare_chat_tasks', chat_worker)
    monkeypatch.setattr(browse, 'prepare_browse_tasks', browse_worker)
    asyncio.run(browse.prepare_memory_tasks(engine, None))
    asyncio.run(browse.prepare_memory_tasks(engine, None))
    assert calls == [('chat', 1), ('browse', 1)]


def test_checkpoint_reused_after_transient_source_read_failure_without_second_model(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    real_locate = f.readers.locate
    fail_once = True
    async def locate(binding):
        nonlocal fail_once
        if f.calls and fail_once:
            fail_once = False
            return {'eligible_source': False, 'state': 'unavailable'}
        return await real_locate(binding)
    f.readers.locate = locate
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['status'] == 'failed'
    mid = f.receipt['task_id']
    row = f.tasks.get(mid)
    raw = row['preparation_output']['text']
    from memory_model_io import MemoryModelIO
    model_io = MemoryModelIO(f.env['authority']).list(mid)['items'][0]
    assert model_io['raw_output'] == raw and model_io['status'] == 'failed'
    assert f.env['authority'].list_candidates(status='all') == []
    def retry(row):
        row.update(phase='queued', lease_until=0, claim_token=None, issue=None)
        return row
    f.tasks.update(mid, expected_revision=row['revision'], transform=retry)
    assert asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['adopted']
    assert len(f.calls) == 1
    assert f.tasks.get(mid)['preparation_output']['text'] == raw


def test_checkpoint_cannot_be_assigned_to_new_understanding_on_retry(tmp_path, monkeypatch):
    def during(store, row, coverage):
        store.publish(row['material_id'], row['revision'], facts=[{'text': '已修订的理解'}],
            appraisal=[], open_questions=[], coverage=coverage)
    f = setup(tmp_path, monkeypatch, during=during)
    assert asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['status'] == 'failed'
    mid = f.receipt['task_id']
    row = f.tasks.get(mid)
    def retry(row):
        row.update(phase='queued', lease_until=0, claim_token=None, issue=None)
        return row
    f.tasks.update(mid, expected_revision=row['revision'], transform=retry)
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['issue']['code'] == 'browse_prepared_source_changed'
    assert len(f.calls) == 1
    assert f.env['authority'].list_memories() == []


def test_owner_edit_while_model_runs_keeps_new_draft_queued_without_stale_candidate(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    original = f.engine._daily_chat_memory_create_completion
    async def create(*args, **kwargs):
        row = f.tasks.get(f.receipt['task_id'])
        f.tasks.edit(row['task_id'], {'expected_revision': row['revision'], 'request_id': 'during-model',
            'draft': {'narrative': {'body': '主人纠正后的待准备稿。'}}})
        return await original(*args, **kwargs)
    f.engine._daily_chat_memory_create_completion = create
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['status'] == 'failed'
    row = f.tasks.get(f.receipt['task_id'])
    assert row['phase'] == 'queued' and row['envelope']['content'] == '主人纠正后的待准备稿。'
    assert f.env['authority'].list_candidates(status='all') == []


def test_owner_edit_at_candidate_publish_boundary_prevents_stale_commit(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    publish = BrowseMemoryPreparation.put_prepared
    def interleaved(tasks, claimed, proposal, policy):
        tasks.edit(claimed['task_id'], {'expected_revision': claimed['revision'], 'request_id': 'at-publish-boundary',
            'draft': {'narrative': {'body': '发布候选前主人纠正的最新稿。'}}})
        return publish(tasks, claimed, proposal, policy)
    monkeypatch.setattr(BrowseMemoryPreparation, 'put_prepared', interleaved)
    result = asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))
    assert result['results'][0]['issue']['code'] == 'submission_revision_conflict'
    current = f.tasks.get(f.receipt['task_id'])
    assert current['phase'] == 'queued' and current['draft_revision'] == 2
    assert current['envelope']['content'] == '发布候选前主人纠正的最新稿。'
    assert f.env['authority'].list_candidates(status='all') == []
    assert f.env['authority'].list_memories() == []


def test_candidate_publish_between_edit_precheck_and_update_fences_stale_edit(tmp_path, monkeypatch):
    from memory_authority import MemoryProposal, PolicyDecision
    from memory_l1_service import L1OwnerError
    f = setup(tmp_path, monkeypatch)
    mid = f.receipt['task_id']
    claimed = f.tasks.claim(f.tasks.get(mid), lease_seconds=60)
    proposal = MemoryProposal.from_mapping({'proposal_id': mid, 'source_type': 'hold', 'proposed_body': f.kwargs['content']})
    policy = PolicyDecision('auto_accept', ('TEST',), 'explicit_tool')
    get_candidate = f.env['authority'].get_candidate
    fired = False
    def interleaved(memory_id):
        nonlocal fired
        value = get_candidate(memory_id)
        if memory_id == mid and value is None and not fired:
            fired = True
            f.tasks.put_prepared(claimed, proposal, policy)
        return value
    monkeypatch.setattr(f.env['authority'], 'get_candidate', interleaved)
    with pytest.raises(L1OwnerError, match='submission_revision_conflict'):
        f.tasks.edit(mid, {'expected_revision': claimed['revision'], 'request_id': 'stale-edit',
            'draft': {'narrative': {'body': '候选采用后不能绕过候选审核编辑。'}}})
    current = f.tasks.get(mid)
    assert current['revision'] == claimed['revision'] + 1
    assert current['candidate_ids'] == [mid] and current['draft_revision'] == 1
    assert current['envelope']['content'] == f.kwargs['content']
    assert get_candidate(mid)['proposal']['proposed_body'] == f.kwargs['content']


def test_actual_hold_queues_browse_without_synchronous_model_or_whisper_reroute(tmp_path, monkeypatch):
    import ast
    f = setup(tmp_path, monkeypatch)
    path = Path(__file__).resolve().parents[1] / 'server.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {'hold', '_memory_operation_key'}]
    for node in nodes:
        node.decorator_list = []
    commits = []
    async def started():
        return None
    async def commit(**kwargs):
        commits.append(kwargs)
        return 'private-whisper'
    def forbidden(*args, **kwargs):
        raise AssertionError('browse queue must precede model/tag/body mutation')
    namespace = {'json': json, 'memory_authority_store': f.env['authority'], 'memory_commit_service': f.env['commit'],
        'decay_engine': NS(ensure_started=started), '_normalize_memory_sections_for_write': forbidden,
        '_commit_tool_memory': commit, '_ai_author_name': lambda: '顾衍'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    result = json.loads(asyncio.run(namespace['hold'](f.kwargs['content'], source_metadata=f.kwargs['descriptor'],
        idempotency_key='actual-hold:key')))
    assert result['preparation_status'] == 'queued' and result['adopted'] is False
    assert not commits and not f.calls and f.env['authority'].list_memories() == []
    whisper = asyncio.run(namespace['hold']('保留原私密路径。', whisper=True, idempotency_key='private-key'))
    assert whisper == '🫧whisper→private-whisper'
    assert commits[0]['source'] == 'hold_whisper' and commits[0]['recall_policy'] == 'manual_only'


def test_browse_independent_l2_persists_and_resolves_using_only_read_article_segments(tmp_path, monkeypatch):
    from memory_l2_service import MemoryL2Service
    f = setup(tmp_path, monkeypatch, article_evidence=True)
    f.coverage['read_ranges'] = [[0, 2], [4, 6]]
    f.store.publish(f.row['material_id'], f.row['revision'], facts=[{'text': '读到的内容', 'sources': ['C1']}],
        appraisal=[{'text': '这是一个待比较的解释', 'sources': ['C1']}], open_questions=[], coverage=f.coverage)
    assert asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['adopted']
    mid = f.receipt['task_id']
    l1 = asyncio.run(f.env['revisions'].detail(mid))['l1']['document']
    statement = l1['statements'][0]['statement_id']
    l2_calls = []
    async def complete_l2(client, **kwargs):
        assert kwargs['task_stage'] == 'presentation_l2'
        payload = json.loads(kwargs['messages'][1]['content'])
        l2_calls.append(payload)
        source = next(s for s in payload['sources'] if s['binding_id'] == 'source:1')
        assert 'text' not in source
        assert source['read_ranges'] == [{'start': 0, 'end': 2, 'text': '读到'}, {'start': 4, 'end': 6, 'text': '容。'}]
        assert '未读的秘密结尾' not in kwargs['messages'][1]['content']
        doc = {'full_injection_text': '我那次只读了材料中的两小段，留下的解释仍需比较。',
            'coherent_sections': [{'section_id': 'reading', 'text': '那次理解还没有成为定论。', 'statement_refs': [statement]}],
            'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
        return NS(choices=[NS(message=NS(content=json.dumps(doc)), finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion = complete_l2
    service = MemoryL2Service(f.engine, f.env['revisions'])
    result = asyncio.run(service.run_once(1))
    assert result['processed'] == 1 and result['results'][0]['state'] == 'ready', result
    persisted = service.detail(mid)['l2']
    assert persisted['state'] == 'ready'
    assert persisted['presentation']['document']['full_injection_text'].startswith('我那次只读了')
    resolved = asyncio.run(service.resolve({'items': [{'memory_id': mid, 'revision': 1, 'statement_ids': [statement],
        'allowed_uses': ['historical_evidence'], 'narrative_allowed': True,
        'statement_uses': {statement: 'historical_evidence'}}]}))
    assert resolved['items'][0]['state'] == 'ready', resolved
    assert resolved['items'][0]['text']
    assert len(f.calls) == 1 and len(l2_calls) == 1


def test_brain_formal_adoption_edit_and_reopen_keep_shared_activation_receipts(tmp_path, monkeypatch):
    import frontmatter
    from bucket_manager import BucketManager
    from memory_activation import PROJECTION
    f = setup(tmp_path, monkeypatch)
    assert asyncio.run(prepare_browse_tasks(f.engine, f.env['buckets']))['results'][0]['adopted']
    mid = f.receipt['task_id']
    manager = f.env['buckets']
    file_path = Path(manager._find_bucket_file(mid))
    original_bytes = file_path.read_bytes()
    assert asyncio.run(manager.touch(mid, activation_id='shared-use-once')) is True
    assert file_path.read_bytes() == original_bytes  # Activation never overwrites owner Markdown.
    detail = asyncio.run(f.env['revisions'].detail(mid))
    document = detail['l1']['document']
    document['narrative']['body'] += '\n\n主人补充的本地编辑说明。'
    saved = asyncio.run(f.env['revisions'].save(mid, {'expected_revision': 1, 'request_id': 'owner-edit-with-activation',
        'document': document, 'edit_note': '临时库编辑保护验证'}))
    assert saved['revision'] == 2
    config = {'buckets_dir': str(tmp_path / 'memory' / 'buckets'), 'state_dir': str(tmp_path / 'memory' / 'state')}
    reopened = BucketManager(config)
    assert asyncio.run(reopened.touch(mid, activation_id='shared-use-once')) is True
    bucket = asyncio.run(reopened.get(mid))
    assert bucket['metadata']['activation_count'] == 1
    assert bucket['content'].endswith('主人补充的本地编辑说明。')
    assert PROJECTION not in frontmatter.load(reopened._find_bucket_file(mid)).metadata
    assert f.env['authority'].get_memory(mid)['active_revision'] == 2
