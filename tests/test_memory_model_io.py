import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

from memory_authority import MemoryAuthorityStore
from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from memory_l1_schema import document_template, statement_template
from memory_l2_service import semantic_input_projection
from memory_model_io import MemoryModelIO
from test_chat_memory_preparation import setup


@pytest.mark.parametrize('bad', [False, True])
def test_actual_chat_wire_visible_output_and_accounting_survive_validation(tmp_path, bad):
    env, engine, envelope, _ = setup(tmp_path, bad=bad)
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    captured = {}
    original = engine._daily_chat_memory_create_completion
    async def create(*args, **kwargs):
        captured['messages'] = copy.deepcopy(kwargs['messages'])
        # Capture must precede the actual provider boundary.
        assert MemoryModelIO(env['authority']).list(key)['items'][0]['status'] == 'running'
        response = await original(*args, **kwargs)
        captured['raw'] = engine._completion_content(response)
        response.usage = NS(prompt_tokens=400, completion_tokens=500, total_tokens=900,
            completion_tokens_details=NS(reasoning_tokens=300),
            prompt_tokens_details=NS(cached_tokens=123), prompt_cache_hit_tokens=123)
        return response
    engine._daily_chat_memory_create_completion = create
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    record = MemoryModelIO(env['authority']).list(key)['items'][0]
    assert record['messages'] == captured['messages']
    assert record['raw_output'] == captured['raw']
    assert record['status'] == ('failed' if bad else 'succeeded')
    assert record['validation']['status'] == ('failed' if bad else 'passed')
    assert record['usage']['reasoning_tokens'] == 300
    assert record['usage']['prompt_tokens_details']['cached_tokens'] == 123
    assert record['finish_reason'] == 'stop'
    assert record['draft_revision'] == 1 and record['attempt'] == 1
    assert len(record['source_revisions']) == 3
    assert record['ended_at'] >= record['started_at']
    assert tasks.detail(key)['preparation_output']['text'] == captured['raw']


def test_provider_failure_preserves_actual_input_without_exception_secret(tmp_path):
    env, engine, envelope, _ = setup(tmp_path)
    key = ChatMemoryPreparation(env['authority']).enqueue(envelope)['task_id']
    async def fail(*args, **kwargs):
        raise RuntimeError('secret-provider-credential')
    engine._daily_chat_memory_create_completion = fail
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    record = MemoryModelIO(env['authority']).list(key)['items'][0]
    assert record['status'] == 'failed' and record['raw_output'] is None
    assert json.loads(record['messages'][1]['content'])['draft'] == envelope['arguments']
    assert 'secret-provider-credential' not in json.dumps(record)


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_l2_actual_wire_and_unmodified_visible_json_survive_success_or_length(tmp_path, finish):
    from test_memory_l2_p4 import setup as setup_l2
    service, authority, _ = setup_l2(tmp_path)
    original = service.engine._daily_chat_memory_create_completion
    captured = {}
    async def complete(*args, **kwargs):
        captured['messages'] = copy.deepcopy(kwargs['messages'])
        response = await original(*args, **kwargs)
        response.text = '```json\n' + response.text + '\n```'
        response.finish = finish
        captured['raw'] = response.text
        return response
    service.engine._daily_chat_memory_create_completion = complete
    asyncio.run(service.run_once(1))
    record = MemoryModelIO(authority).list('m1')['items'][0]
    assert record['stage'] == 'presentation_l2'
    assert record['messages'] == captured['messages']
    assert record['raw_output'] == captured['raw']
    assert record['status'] == ('succeeded' if finish == 'stop' else 'failed')
    material = json.loads(record['messages'][1]['content'])
    assert 'generation' not in material['l1_document']
    assert set(material['output_schema']['properties']) >= {'full_injection_text', 'coherent_sections', 'optional_use_variants', 'review'}
    assert record['completion_parameters']['max_tokens'] == 16000


def test_metadata_pagination_is_scoped_complete_and_legacy_is_missing(tmp_path):
    authority = MemoryAuthorityStore(str(tmp_path / 'authority.db'))
    store = MemoryModelIO(authority)
    complete = '完整正文🙂\n' * 16000
    for memory_id in ('one', 'two'):
        for attempt in range(3):
            ident = store.start(memory_id, stage='chat_l1', task_id='task', attempt=attempt,
                messages=[{'role':'user', 'content': complete}],
                parameters={'model':'test', 'max_tokens':16000, 'client':'secret', 'headers':{'key':'secret'}})
            store.returned(ident, response={}, raw_output=complete, finish_reason='stop')
            store.finish(ident)
    first = store.list('one', limit=2)
    assert first['has_more'] and first['next_offset'] == 2
    second = store.list('one', limit=2, offset=2)
    assert not second['has_more'] and second['next_offset'] is None
    assert len({row['io_id'] for row in first['items'] + second['items']}) == 3
    for row in first['items'] + second['items']:
        assert row['memory_id'] == 'one' and row['messages'][0]['content'] == complete
        assert row['raw_output'] == complete
        assert row['completion_parameters'] == {'model':'test', 'max_tokens':16000}
    assert store.list('legacy')['capture_state'] == 'missing'
    assert store.list('legacy')['items'] == []


def test_l2_projection_preserves_complex_semantics_full_sources_and_local_refs():
    body = '我当时没有承诺；只有她明确愿意时才继续，时间不确定。' * 600
    doc = document_template(body, '完整叙事')
    stmt = statement_template()
    stmt.update(statement_id='st-real', statement_revision_id='rev-machine', content=body,
        evidence_refs=['source:1'],
        property={'key':'preference', 'polarity':'negative', 'value':{'state':'unresolved','value':None,'evidence_refs':['source:1']}},
        cognition={'role':{'state':'known','value':'hypothesis','evidence_refs':[]},
            'epistemic_status':{'state':'known','value':'tentative','evidence_refs':[]},
            'about_refs':['st-shared'],'open_questions':['她是否仍愿意？']},
        attributes=[{'definition_id':'custom-context','value':{'state':'known','value':'见面','evidence_refs':['source:1']},'evidence_refs':['source:1']}])
    stmt['conditions'].update(state='known', text='只有她明确愿意',
        predicate={'op':'not','args':[{'op':'eq','field':'custom-context','value':'拒绝'}]},
        exceptions=[{'state':'known','text':'若她说不要就停止','predicate':None,'evidence_refs':['source:1']}])
    doc['statements'] = [stmt]
    # Projection must never recursively strip a semantic key that happens to
    # share the name of custody metadata (including extensible owner values).
    doc['attributes'] = [{'definition_id':'custom', 'value':{'generation':'保留', 'source_revision':'引用语义'},
                          'evidence_refs':['source:1']}]
    doc['scenes'] = [{'scene_id':'scene1','scene_revision_id':'machine','title':'那次见面',
        'statement_refs':['st-real'],'attributes':stmt['attributes'],'generation':{'prompt_revision':'machine'}}]
    doc['relations'] = [{'from':'st-real','to':'st-shared','kind':'limits','evidence_refs':['source:1']}]
    doc['entry']['shared_statement_refs'] = ['st-shared']
    binding = {'binding_id':'source:1','source_revision':23,'source_id':'canonical-event',
        'author_ref':'user','speaker_ref':'user','anchors':[{'start':3,'end':15,'text_sha256':'machine',
            'target_id':'canonical-event-locator', 'target_namespace':'l0'}]}
    doc['evidence_bindings'] = [binding]
    doc['generation']['catalog_snapshot']['attributes'] = [
        {'id':'custom-context','value_type':'string','description':'具体会面情境'}]
    shared = {**stmt,'statement_id':'st-shared','content':'原共享陈述，不代表现在的意愿。'}
    material = {'l1_document':doc, 'shared_statements':[shared], 'sources':[{'binding':binding,
        'text':body,'author':'她','speaker':'她','recorded_at':'2026-10-18T12:00:00Z',
        'eligible_source':True,'state':'verified_snapshot','view':'source','limitations':[]}],
        'rings':[{'content':'后来她明确否定了这个推测。','state':'active','content_sha256':'machine'}]}
    before = copy.deepcopy(material)
    projected = semantic_input_projection(material)
    assert material == before
    p = projected['l1_document']
    assert p['narrative']['body'] == body and p['statements'][0]['content'] == body
    for key in ('conditions','property','cognition','attributes','time','evidence_refs'):
        assert p['statements'][0][key] == stmt[key]
    assert p['scenes'][0]['statement_refs'] == ['st-real']
    assert p['relations'] == doc['relations']
    assert p['attributes'] == doc['attributes']
    assert projected['shared_statements'][0]['content'] == shared['content']
    assert projected['sources'][0]['text'] == body
    assert projected['sources'][0]['binding_id'] == stmt['evidence_refs'][0]
    assert projected['sources'][0]['author'] == '她'
    assert not {'eligible_source', 'state', 'view', 'limitations'} & projected['sources'][0].keys()
    assert projected['sources'][0]['source_recorded_at'] == material['sources'][0]['recorded_at']
    assert projected['rings'][0]['content'] == material['rings'][0]['content']
    assert projected['semantic_definitions']['attributes']['custom-context']['description'] == '具体会面情境'
    assert 'generation' not in p and 'statement_revision_id' not in p['statements'][0]
    assert 'source_revision' not in p['evidence_bindings'][0]
    assert p['evidence_bindings'][0]['anchors'] == [{'start':3, 'end':15}]


def test_l2_projection_keeps_source_content_restrictions_without_empty_or_technical_audit():
    from memory_l1_ingestion import source_snapshot
    doc = document_template('没有承诺；她是否愿意仍未说明。', '条件未知')
    source = source_snapshot({'namespace':'canonical_event', 'system':'aiz', 'profile_id':'owner',
        'conversation_id':'conversation', 'event_id':'event', 'role':'user',
        'text':'仅限这次会面；之后是否适用仍未知。', 'recorded_at':None})
    source['binding'] = {'binding_id':'source:1'}
    source.update(state='verified_snapshot', eligible_source=True, view='source')
    source['limitations'].extend(['', '原文引用的是过去看法，不表示当下认同。'])
    material = {'l1_document':doc, 'shared_statements':[], 'sources':[source], 'rings':[]}
    projected = semantic_input_projection(material)
    assert projected['sources'][0]['text'] == source['text']
    assert projected['sources'][0]['source_recorded_at'] is None
    assert projected['sources'][0]['limitations'] == ['原文引用的是过去看法，不表示当下认同。']
    assert 'rings' not in projected
    assert not {'eligible_source', 'state', 'view'} & projected['sources'][0].keys()
