import copy
import json

import pytest

from memory_l1_catalog_schema import model_document_schema
from memory_l1_contract import L1ContractError
from memory_l1_ingestion import prepare_model_document
from memory_l1_schema import catalog_with_defaults, unknown
from memory_l1_semantic_wire import assemble_document, semantic_contract, semantic_sources
from memory_l1_service import L1OwnerError
from test_memory_l1_ingestion_p2c import setup


def compact():
    return {'format': 'semantic-v1', 'entities': [{'name': '主人', 'entity_type': 'user'}],
            'statements': [{'content': '那次想要抱抱。', 'subject_ref': 'e1',
                'classification': {'kind': 'preference', 'stability': 'episode',
                                   'modality': 'request', 'basis': 'direct_expression'},
                'time': None, 'conditions': None, 'evidence_refs': ['source:current']}]}


def test_compact_enters_real_l1_binding_without_promoting_past_state(tmp_path):
    service, source, _, generation = setup(tmp_path)
    source_copy = copy.deepcopy(source)
    semantic = compact()
    original = copy.deepcopy(semantic)
    body = '完整第一遍稿。' * 1000
    assembled = assemble_document(semantic, narrative={'body': body})
    document = prepare_model_document(service, memory_id='m1', document=assembled,
        sources=[source], generation=generation, request_id='task1')
    assert semantic == original and source == source_copy
    assert document['narrative']['body'] == body
    statement = document['statements'][0]
    assert statement['subject_ref']['value'] == document['entities'][0]['entity_id']
    assert statement['classification']['stability']['value'] == 'episode'
    assert statement['time']['applicability']['temporal_scope'] == 'unknown'
    assert statement['time']['event']['start']['state'] == 'not_stated'
    assert statement['evidence_refs'] == ['source:1']
    assert statement['statement_revision_id']
    assert document['evidence_bindings'][0]['source_revision']


@pytest.mark.parametrize('path', [('time',), ('conditions',), ('subject_ref',),
                                 ('evidence_refs',), ('classification', 'stability')])
def test_missing_semantic_judgment_is_not_silently_filled(path):
    draft = compact()
    target = draft['statements'][0]
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(L1OwnerError, match='semantic_fields_required'):
        assemble_document(draft, narrative={'body': '正文'})


def test_narrative_ownership_and_full_format_compatibility(tmp_path):
    draft = compact()
    with pytest.raises(L1OwnerError):
        assemble_document(draft)
    draft['narrative'] = {'title': '模型拟题', 'body': '不应覆盖原稿'}
    assert assemble_document(draft, narrative={'body': '原稿'})['narrative'] == {
        'title': '模型拟题', 'body': '原稿'}
    assert assemble_document(draft, narrative={'title': '主人题', 'body': '原稿'})['narrative'] == {
        'title': '主人题', 'body': '原稿'}
    full = setup(tmp_path)[2]
    assert assemble_document(full) == full
    assert assemble_document(full) is not full


def test_full_semantic_modules_survive_program_assembly_and_validation(tmp_path):
    service, source, full, generation = setup(tmp_path)
    catalog = catalog_with_defaults({'attributes': [
        {'id': 'temperature', 'value_type': 'number'}, {'id': 'allowed', 'value_type': 'boolean'}]})
    service.catalog_provider = lambda: (catalog, 'custom-r1')
    draft = {'format': 'semantic-v1', **copy.deepcopy(full)}
    claim = draft['statements'][0]
    claim['property'] = {'key': 'allowed', 'value': False, 'polarity': 'negated', 'evidence_refs': ['source:current']}
    claim['attributes'] = [{'definition_id': 'temperature', 'value': 0}]
    claim['conditions'] = {'state': 'known', 'text': '温度低且没有许可时；具体例外仍未解决。',
        'predicate': {'op': 'all', 'args': [
            {'op': 'lt', 'field': 'temperature', 'value': 10},
            {'op': 'not', 'args': [{'op': 'eq', 'field': 'allowed', 'value': True}]}]},
        'exceptions': [{'state': 'unresolved', 'text': '尚未确定具体例外'}]}
    claim['cognition'] = {'role': 'hypothesis', 'epistemic_status': 'tentative',
        'about_refs': ['new:user'], 'open_questions': ['这个条件是否仍适用？']}
    claim['time']['event'] = {'start': '2026-08-15', 'end': None, 'precision': 'day',
                              'interpretation': 'instant', 'timezone': None}
    claim['time']['applicability'] = {'temporal_scope': 'episode', 'start': None, 'end': None,
                                      'scene_ref': 'new:scene'}
    draft['relations'] = [{'from': {'object_type': 'statement', 'object_id': 'new:comfort'},
        'to': {'object_type': 'statement', 'object_id': 'new:symptom'},
        'kind': 'supplements', 'evidence_refs': ['source:current']}]
    assembled = assemble_document(draft)
    result = prepare_model_document(service, memory_id='m1', document=assembled,
        sources=[source], generation=generation, request_id='rich')
    s = result['statements'][0]
    assert s['property']['value'] == unknown('known', False, ['source:1'])
    assert s['attributes'][0]['value'] == unknown('known', 0, ['source:1'])
    assert s['conditions']['exceptions'][0]['state'] == 'unresolved'
    assert s['conditions']['predicate'] == claim['conditions']['predicate']
    assert s['cognition']['open_questions'] == claim['cognition']['open_questions']
    assert s['time']['applicability']['scene_ref'] == result['scenes'][0]['scene_id']
    assert result['relations'][0]['confirmation_state'] == 'proposed'
    assert result['scenes'][0]['anchors'][0]['end'] == 9
    assert result['narrative'] == full['narrative']


@pytest.mark.parametrize('mutation', ['catalog', 'source', 'unbounded'])
def test_assembler_cannot_bypass_canonical_semantic_validation(tmp_path, mutation):
    service, source, _, generation = setup(tmp_path)
    draft = compact()
    claim = draft['statements'][0]
    if mutation == 'catalog':
        claim['classification']['kind'] = 'invented_category'
    elif mutation == 'source':
        claim['evidence_refs'] = ['source:999']
    else:
        claim['time'] = {'event': None, 'applicability': {'temporal_scope': 'time_range',
            'start': None, 'end': {'state': 'unbounded', 'evidence_refs': []}}}
    with pytest.raises((L1OwnerError, L1ContractError)):
        prepare_model_document(service, memory_id='m1', document=assemble_document(draft, narrative={'body': '正文'}),
            sources=[source], generation=generation, request_id='invalid')


def test_single_catalog_view_retains_owner_semantics_without_repeated_machine_schema():
    catalog = catalog_with_defaults({'topics': ['custom'], 'definitions': [
        {'axis': 'topics', 'id': 'custom', 'label': '自定义主题', 'description': '主人定义，不可删除。',
         'aliases': ['别名'], 'examples': ['例子'], 'parent_id': None}]})
    before = copy.deepcopy(catalog)
    contract = semantic_contract(['source:1'], catalog=catalog)
    assert catalog == before
    assert contract['catalog']['topics']['custom']['aliases'] == ['别名']
    assert contract['catalog']['topics']['custom']['description'] == '主人定义，不可删除。'
    assert set(contract['catalog']['kinds']) == set(catalog['kinds'])
    assert '$defs' not in json.dumps(contract)
    old = {'document_schema': model_document_schema(['source:1'], catalog=catalog), 'catalog': catalog}
    assert len(json.dumps(contract, ensure_ascii=False)) < len(json.dumps(old, ensure_ascii=False)) * .65


def test_explicit_unknown_scope_expands_without_inventing_event_or_validity(tmp_path):
    service, source, _, generation = setup(tmp_path)
    source['recorded_at'] = '2026-10-18T06:08:33Z'
    draft = compact()
    draft['statements'][0]['time'] = {'event': None,
        'applicability': {'temporal_scope': 'unknown'}}
    assembled = assemble_document(draft, narrative={'body': '完整回忆'})
    result = prepare_model_document(service, memory_id='m1', document=assembled,
        sources=[source], generation=generation, request_id='unknown-scope')
    time = result['statements'][0]['time']
    assert time['event']['start'] == unknown()
    assert time['event']['end'] == unknown()
    assert time['applicability']['start'] == unknown()
    assert time['applicability']['end'] == unknown()
    assert result['statements'][0]['conditions']['state'] == 'not_stated'


def test_sparse_known_event_preserves_supplied_time_and_special_unknown_states(tmp_path):
    service, source, _, generation = setup(tmp_path)
    draft = compact()
    draft['statements'][0]['time'] = {
        'event': {'start': '2026-09-10', 'precision': 'day', 'interpretation': 'instant'},
        'applicability': {'temporal_scope': 'unknown', 'end': {'state': 'unresolved'}}}
    result = prepare_model_document(service, memory_id='m1',
        document=assemble_document(draft, narrative={'body': '那次的完整回忆'}),
        sources=[source], generation=generation, request_id='explicit-date')
    time = result['statements'][0]['time']
    assert time['event']['start']['value'] == '2026-09-10'
    assert time['event']['precision'] == 'day'
    assert time['event']['end'] == unknown() and time['event']['timezone'] == unknown()
    assert time['applicability']['end']['state'] == 'unresolved'


@pytest.mark.parametrize('time', [{}, {'event': None}, {'event': {}, 'applicability': None},
                                  {'event': None, 'applicability': {}}])
def test_time_semantic_decisions_are_still_required(time):
    draft = compact()
    draft['statements'][0]['time'] = time
    with pytest.raises(L1OwnerError, match='semantic_fields_required'):
        assemble_document(draft, narrative={'body': '正文'})


def test_model_source_view_preserves_evidence_without_managed_identity_or_time_alias():
    snapshots = [{'text': '未裁剪的原文。' * 1500, 'author': 'user',
                  'recorded_at': '2026-10-18T06:08:33Z', 'source_revision': 'program-only'}]
    bindings = [{'binding_id': 'source:1', 'source_revision': 'program-only', 'anchors': [{'sha': 'program-only'}]}]
    original = copy.deepcopy((snapshots, bindings))
    projected = semantic_sources(snapshots, bindings)
    assert (snapshots, bindings) == original
    assert projected == [{'binding_id': 'source:1', 'text': snapshots[0]['text'], 'author': 'user',
                          'source_recorded_at': '2026-10-18T06:08:33Z'}]
    with pytest.raises(L1OwnerError, match='source_binding_count_mismatch'):
        semantic_sources(snapshots, [])
