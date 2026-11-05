"""Temporary Authority + actual Review/Commit/L2 queue, with no provider calls."""
import asyncio
import copy
import hashlib
import json
import time
from types import SimpleNamespace as NS

import pytest
from starlette.requests import Request

from memory_authority import IdempotencyConflict
from memory_l0_sources import MemoryL0Sources
from memory_l1_ingestion import source_snapshot
from memory_l1_source_readers import configured_source_readers
from memory_owner_summary import (OwnerSummaryPreparation, OwnerSummarySourceReader,
    owner_summary_response, prepare_owner_summary_tasks, task_id, windows)
from memory_source_provenance import canonical_source
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_diary_l1_p2b import setup_diary


def setup(tmp_path, monkeypatch, *, mode='review', count=1, body=None, draft=None, during=None, image_only=False):
    env = setup_diary(tmp_path / 'memory', 'unrelated diary')
    config = reflection_config(tmp_path / 'engine', mode=mode)
    config['gateway'] = {'canonical_continuation': {'enabled':True, 'base_url':'http://aiz.test', 'conversation_id':'other-fixed-conversation'}}
    config['reflection']['daily_chat_memory_window_max_input_chars'] = 256
    engine = ReflectionEngine(config)
    engine.memory_authority_store = env['authority']
    engine._memory_l1_revisions = lambda manager=None: env['revisions']
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'synthetic-private-bridge')
    readers = MemoryL0Sources(env['authority'], configured_source_readers(config, env['authority']))
    engine._memory_l1_sources = lambda: readers
    summary_body = draft or '希望留下这次原话中的意思。'
    summary = {'id':'summary-1', 'revision':1, 'body':summary_body,
        'body_sha256':hashlib.sha256(summary_body.encode()).hexdigest()}
    sources = [{'event_id':'event-' + str(i), 'version_id':'v-' + str(i), 'role':'user' if i % 2 == 0 else 'assistant',
        'text':body or '计划仍要等预约确认。' + str(i), 'created_at':'2026-11-01T00:00:00+00:00', 'seq':i} for i in range(count)]
    if image_only:
        sources[0].update(text='', attachment_refs=[{'resource_id':'fixture-image','mime':'image/png','text_status':'not_applicable'}])
    envelope = {'operation_id':'owner-summary.fixture', 'summary':summary,
        'context':{'conversation_id':'real-app-conversation', 'branch_id':'real-branch', 'profile_id':'real-profile', 'sources':sources}}
    current = NS(valid=True)
    async def capture(self, operation_id, locator, version):
        expected = next(s for s in sources if s['event_id'] == locator['event_id'])
        if not current.valid: return {'state':'changed', 'eligible_source':False}
        return {'state':'available','eligible_source':True, 'source_revision':expected['version_id'],
            'text':expected['text'], 'role':expected['role'], 'recorded_at':expected['created_at'],
            'conversation_id':envelope['context']['conversation_id'], 'branch_id':'real-branch', 'profile_id':'real-profile',
            'summary':{k:summary[k] for k in ('id','revision','body_sha256')}}
    monkeypatch.setattr(OwnerSummarySourceReader, 'capture', capture)
    calls = []
    async def create(client, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        calls.append(payload)
        if during: during(current)
        selected = payload['l1_contract']['source_texts']
        ref = selected[-1]['binding_id']
        text = selected[-1]['text']
        document = {'format':'semantic-v1', 'entities':[],
            'narrative':{'title':'原话中的计划', 'body':'原话中的计划仍须等待预约确认。'},
            'retrieval':{'keywords':['预约计划']},
            'statements':[{'content':text, 'subject_ref':None,
                'classification':{'kind':None,'stability':None,'modality':None,'basis':None},
                'time':None,'conditions':None,'evidence_refs':[ref]}]}
        result = {'candidates':[{'document':document, 'kind':'key_event', 'confidence':0.95,
            'source_refs':[ref], 'reason':'原始资料支持'}]}
        return NS(choices=[NS(message=NS(content=json.dumps(result)),finish_reason='stop')])
    engine._daily_chat_memory_create_completion = create
    engine._memory_task_route = lambda stage: {'stage':stage, 'client':object(), 'model':'synthetic-only',
        'connection':'reflection', 'configured':True, 'thinking_mode':'', 'reasoning_effort':'', 'max_tokens':16000}
    tasks = OwnerSummaryPreparation(env['authority'], engine)
    receipt = tasks.enqueue(envelope, budget=256)
    return NS(env=env,engine=engine,tasks=tasks,envelope=envelope,receipt=receipt,calls=calls,current=current,readers=readers)


def run(f):
    return asyncio.run(prepare_owner_summary_tasks(f.engine, f.env['buckets']))


def test_image_only_without_text_is_a_real_reference_but_no_invented_long_term_memory(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch, image_only=True)
    receipt = run(f)['results'][0]
    assert receipt['status'] == 'no_new_memory'
    assert receipt['candidate_ids'] == [] and not f.calls


def test_owner_selection_queues_then_original_review_and_commit_l2(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    assert f.receipt['status'] == 'queued' and not f.calls
    assert f.tasks.enqueue(f.envelope, budget=256) == f.receipt
    result = run(f)
    assert result['results'][0]['status'] == 'review', result
    cid = result['results'][0]['candidate_ids'][0]
    candidate = f.env['authority'].get_candidate(cid)
    assert candidate['proposal']['source_type'] == 'owner_summary'
    assert not candidate['proposal']['owner_explicit']
    binding = candidate['proposal']['metadata']['memory_l1']['evidence_bindings'][0]
    assert binding['source_namespace'] == 'snapshot.aiz.canonical_event'
    original = f.envelope['context']['sources'][0]
    expected = source_snapshot(canonical_source(original, conversation_id='real-app-conversation', profile_id='real-profile'), config=f.engine.config)
    assert binding['source_id'] == expected['source_id'] and binding['source_revision'] == expected['source_revision']
    assert asyncio.run(f.readers.locate(binding))['eligible_source'] is True
    asyncio.run(f.engine._memory_l1_adopt_candidate(candidate,f.env['buckets'],owner=True,request_id='synthetic-owner-review'))
    receipt = f.tasks.receipt(f.receipt['task_id'])
    assert receipt['status'] == 'l1_accepted_l2_pending'
    assert receipt['memories'][0]['l2_state'] == 'pending'
    assert len(f.calls) == 1
    run(f)
    assert len(f.calls) == 1


def test_auto_uses_existing_policy_but_off_cannot_bypass(tmp_path, monkeypatch):
    f = setup(tmp_path / 'auto', monkeypatch, mode='auto')
    result = run(f)
    assert result['results'][0]['status'] == 'l1_accepted_l2_pending',result
    g = setup(tmp_path / 'off', monkeypatch, mode='off')
    assert run(g)['status'] == 'disabled'
    assert g.tasks.receipt(g.receipt['task_id'])['status'] == 'disabled'
    assert g.tasks.receipt(g.receipt['task_id'])['issue']['code'] == 'memory_source_disabled'
    assert not g.calls and not g.env['authority'].list_memories()


@pytest.mark.parametrize('count,body', [(19,None), (1,'起'+ '正文'*350 +'尾')])
def test_all_sources_and_long_single_source_suffix_are_processed(tmp_path, monkeypatch, count, body):
    f = setup(tmp_path, monkeypatch, count=count, body=body)
    while f.tasks.get(f.receipt['task_id'])['phase'] == 'queued': run(f)
    row = f.tasks.get(f.receipt['task_id'])
    assert row['phase'] == 'prepared', row.get('issue')
    if body: assert len(f.calls) > 1
    else: assert len(f.calls[0]['l1_contract']['source_texts']) == count
    texts = {}
    for payload in f.calls:
        assert sum(len(s['text']) for s in payload['l1_contract']['source_texts']) <= 256
        for source in payload['l1_contract']['source_texts']:
            # Source group/position are retained in each original range.
            texts.setdefault(source['source_recorded_at'], []).append(source['text'])
    ranges = [r for w in row['windows'] for r in w['ranges']]
    for index, source in enumerate(f.envelope['context']['sources']):
        spans = [r for r in ranges if r['source_index'] == index]
        assert spans[0]['start'] == 0 and spans[-1]['end'] == len(source['text'])
        assert all(a['end'] == b['start'] for a,b in zip(spans,spans[1:]))
    for cid in row['candidate_ids']:
        semantic = f.env['authority'].get_candidate(cid)['proposal']['metadata']['memory_l1']
        assert all(b['anchors'][0]['end'] <= len(f.readers.copy(b)['text']) for b in semantic['evidence_bindings'])


def test_long_summary_draft_ranges_cover_every_character(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch, draft='头'+ '完整摘要'*180 +'尾')
    while f.tasks.get(f.receipt['task_id'])['phase'] == 'queued': run(f)
    delivered = ''.join(p['owner_selected_summary']['body'] for p in f.calls)
    assert delivered == f.envelope['summary']['body']
    assert f.tasks.get(f.receipt['task_id'])['envelope'] == f.envelope


def test_source_changes_during_generation_never_publish_or_commit(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch, during=lambda current:setattr(current,'valid',False))
    result = run(f)
    assert result['results'][0]['status'] == 'source_changed', result
    assert not f.env['authority'].list_memories()
    assert f.tasks.get(f.receipt['task_id'])['candidate_ids'] == []
    run(f)
    assert len(f.calls) == 1


def test_operation_replay_is_stable_and_changed_payload_conflicts(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    changed = copy.deepcopy(f.envelope)
    changed['summary']['title'] = '修改的选择'
    with pytest.raises(IdempotencyConflict): f.tasks.enqueue(changed,budget=256)
    assert f.tasks.get(f.receipt['task_id'])['envelope'] == f.envelope


def request(method, token='', payload=None, operation_id='owner-summary.fixture'):
    raw = json.dumps(payload).encode()
    async def receive(): return {'type':'http.request','body':raw,'more_body':False}
    return Request({'type':'http','method':method,'path':'/api/memory-authority/owner-summary',
        'headers':[(b'authorization',('Bearer '+token).encode())], 'path_params':{'operation_id':operation_id}},receive)


def test_private_api_auth_no_store_queued_conflict_and_source_changed(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    invoke = lambda req: asyncio.run(owner_summary_response(req,f.engine,f.env['authority']))
    assert invoke(request('POST','ordinary-memory-writer',f.envelope)).status_code == 401
    response = invoke(request('POST','synthetic-private-bridge',f.envelope))
    assert response.status_code == 202 and json.loads(response.body)['status'] == 'queued'
    assert response.headers['cache-control'] == 'no-store' and not f.calls
    changed = copy.deepcopy(f.envelope); changed['context']['branch_id'] = 'another'
    assert invoke(request('POST','synthetic-private-bridge',changed)).status_code == 409
    f.current.valid = False
    response = invoke(request('GET','synthetic-private-bridge'))
    assert response.status_code == 200 and json.loads(response.body)['status'] == 'source_changed'


def test_lost_worker_unknown_outcome_never_implicitly_repeats_model(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    row = f.tasks.get(f.receipt['task_id'])
    raw_sources = asyncio.run(OwnerSummarySourceReader(f.engine.config).sources(f.envelope))
    row = f.tasks.update(row['task_id'],expected_revision=row['revision'],transform=lambda value:{**value,'windows':windows(raw_sources,256)})
    claimed = f.tasks.claim(row, lease_seconds=-1)
    result = run(f)
    assert result['results'][0]['issue']['code'] == 'execution_outcome_unknown'
    assert not f.calls


def test_saved_window_output_survives_crash_without_second_model(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    original = f.tasks.put_window
    def fail(*args,**kwargs): raise RuntimeError('synthetic process interruption after completion checkpoint')
    monkeypatch.setattr(OwnerSummaryPreparation,'put_window',fail)
    assert run(f)['results'][0]['status'] == 'failed'
    assert len(f.calls) == 1
    row = f.tasks.get(f.receipt['task_id'])
    assert row['windows'][0]['output']['text']
    monkeypatch.setattr(OwnerSummaryPreparation,'put_window',lambda self,*a,**kw:original(*a,**kw))
    f.tasks.update(row['task_id'],expected_revision=row['revision'],transform=lambda value:{**value,'phase':'queued','issue':None})
    result = run(f)
    assert result['results'][0]['status'] == 'review',result
    assert len(f.calls) == 1


def test_known_failed_retry_uses_same_operation_and_window_cas(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    original = f.engine._daily_chat_memory_create_completion
    async def bad(client, **kwargs):
        return NS(choices=[NS(message=NS(content='{"unexpected":true}'),finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion = bad
    receipt = run(f)['results'][0]
    assert receipt['status'] == 'failed' and receipt['retryable'] is True
    payload = {'operation_id':f.envelope['operation_id'],'retry':True,'expected_task_revision':receipt['task_revision']}
    invoke = lambda: asyncio.run(owner_summary_response(request('POST','synthetic-private-bridge',payload),f.engine,f.env['authority']))
    assert json.loads(invoke().body)['status'] == 'queued'
    assert json.loads(invoke().body)['status'] == 'queued'
    f.engine._daily_chat_memory_create_completion = original
    result = run(f)['results'][0]
    assert result['status'] == 'review' and result['retryable'] is False
    assert len(f.calls) == 1
    assert invoke().status_code == 409


def test_original_l2_worker_ready_then_owner_text_remains_protected(tmp_path, monkeypatch):
    from memory_l2_service import MemoryL2Service
    f = setup(tmp_path, monkeypatch, mode='auto')
    receipt = run(f)['results'][0]
    mid = receipt['memory_ids'][0]
    semantic = f.env['authority'].get_memory_revision(mid,1)['metadata']['memory_l1']
    sid = semantic['statements'][0]['statement_id']
    async def l2_completion(client, **kwargs):
        assert kwargs['task_stage'] == 'presentation_l2'
        document = {'full_injection_text':'这件计划仍须等预约确认。',
            'coherent_sections':[{'section_id':'plan','text':'这件计划仍须等预约确认。','statement_refs':[sid]}],
            'optional_use_variants':[], 'review':{'required':False,'reasons':[]}}
        return NS(choices=[NS(message=NS(content=json.dumps(document)),finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion = l2_completion
    service = MemoryL2Service(f.engine,f.env['revisions'])
    result = asyncio.run(service.run_once(1))
    assert result['processed'] == 1 and result['results'][0]['state'] == 'ready',result
    assert f.tasks.receipt(f.receipt['task_id'])['status'] == 'ready'
    document = {'full_injection_text':'主人保留的可读措辞。','coherent_sections':[], 'optional_use_variants':[]}
    asyncio.run(service.save(mid,{'expected_l1_revision':1,'expected_presentation_revision':1,'request_id':'owner-wording',
        'document':document,'change_kind':'wording'}))
    service.prepare(mid,{'expected_l1_revision':1,'request_id':'after-owner'})
    asyncio.run(service.run_once(1))
    detail = service.detail(mid)
    assert detail['owner_protected'] is True
    assert detail['l2']['presentation']['document']['full_injection_text'] == document['full_injection_text']
    assert f.tasks.receipt(f.receipt['task_id'])['status'] == 'ready'


def test_same_source_coverage_links_original_memory_without_second_commit(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch,mode='auto')
    original = run(f)['results'][0]['memory_ids'][0]
    second = copy.deepcopy(f.envelope); second['operation_id'] = 'owner-summary.second-selection'
    receipt = f.tasks.enqueue(second,budget=256)
    result = run(f)['results'][0]
    assert result['task_id'] == receipt['task_id']
    assert result['status'] == 'linked',result
    assert result['linked_memories'][0]['memory_id'] == original
    assert result['candidate_ids'] == []
    assert len(f.env['authority'].list_memories()) == 1
    # Changing the second draft alone cannot revoke the first owner's valid
    # access to the same original event. Withdrawing that event affects both.
    original_capture = OwnerSummarySourceReader.capture
    async def changed_second(self, operation_id, locator, version):
        if operation_id == second['operation_id']:
            return {'state': 'changed', 'eligible_source': False}
        return await original_capture(self, operation_id, locator, version)
    monkeypatch.setattr(OwnerSummarySourceReader, 'capture', changed_second)
    binding = f.env['authority'].get_candidate(original)['proposal']['metadata']['memory_l1']['evidence_bindings'][0]
    assert asyncio.run(f.readers.locate(binding))['eligible_source']
    f.current.valid = False
    assert not asyncio.run(f.readers.locate(binding))['eligible_source']


@pytest.mark.parametrize('block', ['prompt','model'])
def test_configuration_block_is_visible_and_original_queue_resumes(tmp_path, monkeypatch, block):
    f = setup(tmp_path, monkeypatch)
    prompt, route = f.engine._memory_l1_prompt, f.engine._memory_task_route
    if block == 'prompt':
        f.engine._memory_l1_prompt = lambda *a,**kw: ('',{'prompt_revision':'disabled'})
    else:
        f.engine._memory_task_route = lambda stage:{**route(stage),'configured':False}
    result = run(f)
    assert result['status'] == 'disabled' and not f.calls
    receipt = f.tasks.receipt(f.receipt['task_id'])
    assert receipt['status'] == 'blocked' and receipt['retryable'] is False
    assert receipt['issue']['code'] == ('memory_prompt_disabled' if block == 'prompt' else 'memory_model_unconfigured')
    assert f.tasks.get(f.receipt['task_id'])['phase'] == 'queued'
    f.engine._memory_l1_prompt, f.engine._memory_task_route = prompt, route
    assert run(f)['results'][0]['status'] == 'review'
    assert len(f.calls) == 1
