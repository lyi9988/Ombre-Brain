"""Review may relocate an intact keyword object; all schema rules still apply."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

from jsonschema import Draft202012Validator
import pytest

from memory_l1_api import build_schema, validate_candidate_items, validate_response
from memory_l1_service import L1OwnerError
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_l1_native_api import SOURCE_IDS, catalog, wrapped


FLAG = 'retrieval_relocated_from_narrative'


def response_for(value, schema=None):
    return NS(choices=[NS(message=NS(content=json.dumps(value, ensure_ascii=False)), finish_reason='stop')],
              native_schema=schema or build_schema(SOURCE_IDS, catalog(), stage='auto_l1'))


def misplaced():
    value = wrapped('auto_l1')
    doc = value['candidates'][0]['document']
    doc['narrative']['retrieval'] = doc.pop('retrieval')
    return value


def test_layout_opt_in_preserves_original_error_object_and_valid_sibling():
    value = misplaced()
    correct = wrapped('auto_l1')['candidates'][0]
    value['candidates'].append(deepcopy(correct))
    response = response_for(value)
    original_raw = response.choices[0].message.content
    strict, original_issue = validate_candidate_items(response)
    assert '_candidate_validation_issue' in strict[0]
    assert strict[1] == correct

    repairs, rejected = {}, {}
    items, issue = validate_candidate_items(response, review_layout_repairs=repairs, rejected_drafts=rejected)
    assert issue == original_issue and issue['code'] == 'native_l1_schema_mismatch'
    assert items[0] == correct
    assert items[0]['document']['retrieval'] == value['candidates'][0]['document']['narrative']['retrieval']
    assert json.dumps(items[1], ensure_ascii=False) == json.dumps(value['candidates'][1], ensure_ascii=False)
    assert not rejected
    assert repairs == {'item_0': {'issue': strict[0]['_candidate_validation_issue'],
        'from': 'document.narrative.retrieval', 'to': 'document.retrieval'}}
    Draft202012Validator(response.native_schema).validate({'candidates': items})
    assert response.choices[0].message.content == original_raw
    with pytest.raises(L1OwnerError, match='native_l1_schema_mismatch'):
        validate_response(response)

    # Re-validating a corrected draft is a normal strict success, with no second repair.
    again = {}
    same, same_issue = validate_candidate_items(response_for({'candidates': items}), review_layout_repairs=again)
    assert same == items and same_issue is None and again == {}


@pytest.mark.parametrize('failure', ['conflict', 'present_null', 'missing', 'not_object',
    'empty_keywords', 'invalid_cognition', 'narrative_entry', 'candidate_extra', 'source_ref'])
def test_layout_rejects_ambiguity_absence_and_every_other_candidate_failure(failure):
    value = misplaced()
    item = value['candidates'][0]
    doc = item['document']
    if failure == 'conflict':
        doc['retrieval'] = deepcopy(doc['narrative']['retrieval'])
    elif failure == 'present_null':
        doc['retrieval'] = None
    elif failure == 'missing':
        doc['narrative'].pop('retrieval')
    elif failure == 'not_object':
        doc['narrative']['retrieval'] = ['keyword']
    elif failure == 'empty_keywords':
        doc['narrative']['retrieval']['keywords'] = []
    elif failure == 'invalid_cognition':
        doc['statements'][0]['cognition']['role'] = 'fact'
    elif failure == 'narrative_entry':
        doc['narrative']['entry'] = deepcopy(doc['entry'])
    elif failure == 'candidate_extra':
        item['entry'] = deepcopy(doc['entry'])
    else:
        item['source_refs'] = ['source:999']
    repairs, rejected = {}, {}
    response = response_for(value)
    result, issue = validate_candidate_items(response, review_layout_repairs=repairs, rejected_drafts=rejected)
    assert result[0].get('_candidate_validation_issue') and issue
    assert repairs == {} and rejected['item_0']['draft'] == item
    assert json.loads(response.choices[0].message.content) == value


@pytest.mark.parametrize('extra', ['candidates_meta', 'entry'])
def test_layout_never_repairs_an_invalid_root(extra):
    value = misplaced()
    value[extra] = {}
    repairs = {}
    with pytest.raises(L1OwnerError, match='native_l1_schema_mismatch'):
        validate_candidate_items(response_for(value), review_layout_repairs=repairs)
    assert repairs == {}


def extraction_fixture(tmp_path, mode):
    from memory_l1_ingestion import model_template
    from memory_l1_schema import statement_template
    engine = ReflectionEngine(reflection_config(tmp_path, mode=mode or 'review'))
    engine._memory_task_route = lambda stage: {'stage': stage, 'client': object(),
        'model': 'fixture', 'connection': 'reflection', 'max_tokens': 16000}
    engine._memory_l1_prompt = lambda *_args, **_kwargs: ('fixture prompt', {'prompt_revision': 'fixture-p1'})
    configured_catalog = engine._memory_l1_revisions().catalog_provider()[0]
    doc = model_template(catalog=configured_catalog)
    doc['format'] = 'semantic-v1'
    doc['narrative'] = {'title': '苏州探亲安排',
        'body': '十一月的苏州探亲已列入近期安排，出发时间需等待公司的排班结果。'}
    doc['retrieval']['keywords'] = ['苏州探亲', '等待排班']
    statement = statement_template()
    statement.pop('statement_revision_id')
    statement.update(content='主人计划十一月赴苏州探亲，具体出发日期等待排班。', evidence_refs=['source:1'])
    doc['statements'] = [statement]
    doc['narrative']['retrieval'] = doc.pop('retrieval')
    response = response_for({'candidates': [{'kind': 'key_event', 'confidence': .99,
        'reason': '主人明确表达近期探亲安排。', 'source_refs': ['source:1'], 'document': doc}]},
        build_schema(['source:1'], configured_catalog, stage='auto_l1'))
    async def complete(*_args, **_kwargs):
        return response
    engine._daily_chat_memory_create_completion = complete
    turns = [{'id': 11, 'session_id': 's', 'user_text':
        '我已经决定十一月初去苏州看家人，预计十一月三日出发，但日期要以公司排班为准。',
        'assistant_text': '', 'created_at': '2026-10-25T09:00:00+08:00'}]
    return engine, turns, response


@pytest.mark.parametrize('mode', ['review', 'auto', None, 'unknown'])
def test_actual_extraction_normalization_and_pending_metadata_are_review_only(tmp_path, mode):
    engine, turns, response = extraction_fixture(tmp_path, mode)
    raw = response.choices[0].message.content
    extracted, stats = asyncio.run(engine._extract_window_candidates('2026-10-25', turns, mode=mode))
    assert bool(stats.get('review_layout_repairs')) is (mode == 'review')
    audit = {}
    normalized = engine._normalize_daily_chat_memory_candidates('2026-10-25', extracted, turns,
        mode=mode or 'review', min_confidence=.9, audit=audit)
    assert response.choices[0].message.content == raw
    with pytest.raises(L1OwnerError, match='native_l1_schema_mismatch'):
        validate_response(response)
    if mode != 'review':
        assert normalized == [] and audit['hard_rejects']['invalid_l1_document'] == 1
        assert not engine.memory_authority_store.list_candidates(status='all')
        return
    assert len(normalized) == 1, audit
    candidate = normalized[0]
    assert FLAG in candidate['soft_flags'] and candidate['l1_document']['retrieval']['keywords'] == ['苏州探亲', '等待排班']
    engine._store_daily_chat_memory_pending(normalized)
    saved = engine.memory_authority_store.get_candidate(candidate['id'])
    assert saved['status'] == 'pending'
    legacy = saved['proposal']['metadata']['legacy_candidate']
    assert FLAG in legacy['soft_flags'] and legacy['review_layout_repair'] == extracted[0]['_review_layout_repair']
    assert engine.memory_authority_store.get_memory(candidate['id']) is None
    engine._store_daily_chat_memory_pending(normalized)
    assert engine.memory_authority_store.get_candidate(candidate['id'])['revision'] == saved['revision']


def test_daily_window_passes_explicit_review_mode_to_adapter(tmp_path):
    engine, turns, _response = extraction_fixture(tmp_path, 'review')
    candidates, meta = asyncio.run(engine._extract_daily_chat_memory_candidates('2026-10-25', turns,
        mode='review', min_confidence=.9))
    assert len(candidates) == 1, meta
    assert FLAG in candidates[0]['soft_flags']
