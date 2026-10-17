import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from memory_affinity import (persona_snapshot, read_transport, resolve_signals,
                             sample_persona, statement_affinity)
from memory_l1_schema import document_template, statement_template
from memory_recall_semantics import preview_pipeline, resolve_document, score_document
from memory_system_config import validate_values, values_from_config

NOW = datetime(2026, 10, 18, 6, tzinfo=timezone.utc)


def settings():
    values = values_from_config({})
    values['catalog']['topics'] = ['health', 'care', 'code']
    values['recall']['signal_bindings'] = [
        {'id': 'care', 'producer': 'ombre.persona', 'key': 'protective_drive', 'max_age_seconds': 3600},
        {'id': 'curiosity', 'producer': 'aiz.needs', 'key': 'curiosity', 'max_age_seconds': 3600},
    ]
    values['recall']['affinity_mappings'] = [
        {'id': 'health_care', 'selector': {'topic': 'health'}, 'signal_id': 'care', 'target': 'drive_affinity', 'weight': .8},
        {'id': 'care_again', 'selector': {'topic': 'care'}, 'signal_id': 'care', 'target': 'drive_affinity', 'weight': .6},
        {'id': 'code_curious', 'selector': {'topic': 'code'}, 'signal_id': 'curiosity', 'target': 'interest_affinity', 'weight': .5},
    ]
    values['features'].append({'id': 'drive', 'value_type': 'number', 'source': 'runtime', 'missing': 'unknown', 'input_key': 'drive_affinity'})
    values['formulas'].append({'id': 'affinity_score', 'expression': 'relevance * (1 + coalesce(drive, 0))', 'weights': {}})
    values['recall']['formula_id'] = 'affinity_score'
    return validate_values(values)


def snapshot(value=.5, *, producer='ombre.persona', key='protective_drive', **overrides):
    return {'producer': producer, 'state': 'available', 'revision': 'actual-source-r1',
            'as_of': NOW.isoformat(), 'signals': {key: {'state': 'known', 'value': value,
                                                       'evidence_refs': ['actual-source:r1']}}, **overrides}


def statement(topics):
    result = statement_template()
    result['content'] = 'A synthetic past experience, not the current user state.'
    result['classification']['topics'] = topics
    return result


@pytest.mark.parametrize('value', [0, .5, 1])
def test_actual_zero_is_known_and_topics_do_not_multiply_one_signal(value):
    values = settings()
    signals = resolve_signals(values, [snapshot(value)], NOW)
    features, trace = statement_affinity(statement(['health', 'health', 'care']), values, signals)
    assert signals['care']['state'] == 'known'
    assert features['drive_affinity'] == pytest.approx(value * .8)
    assert features['signal_care'] == value
    assert features['interest_affinity'] is None
    assert trace['targets']['drive_affinity']['contributions'][0]['revision'] == 'actual-source-r1'


@pytest.mark.parametrize('change,expected', [
    ({'state': 'default'}, 'producer_default'),
    ({'state': 'error'}, 'producer_error'),
    ({'revision': None}, 'revision_unknown'),
    ({'as_of': None}, 'time_unknown'),
    ({'as_of': (NOW - timedelta(hours=2)).isoformat()}, 'stale'),
    ({'as_of': (NOW + timedelta(seconds=1)).isoformat()}, 'future_snapshot'),
])
def test_unusable_source_is_unknown_not_zero(change, expected):
    signals = resolve_signals(settings(), [snapshot(**change)], NOW)
    assert signals['care']['state'] == 'unknown'
    assert signals['care']['value'] is None
    assert signals['care']['reason'] == expected


def test_one_unknown_matching_source_cannot_be_hidden_by_another_known_source():
    values = settings()
    values['recall']['affinity_mappings'].append({
        'id': 'unknown_care', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    signals = resolve_signals(values, [snapshot()], NOW)
    assert statement_affinity(statement(['health']), values, signals)[0]['drive_affinity'] is None
    values['recall']['affinity_mappings'][-1]['weight'] = 0
    assert statement_affinity(statement(['health']), values, signals)[0]['drive_affinity'] == pytest.approx(.4)


def test_domain_is_explicit_and_combined_selector_is_and():
    values = settings()
    values['recall']['affinity_mappings'][0]['selector']['domain'] = 'life'
    signals = resolve_signals(values, [snapshot()], NOW)
    assert statement_affinity(statement(['health']), values, signals)[0]['drive_affinity'] is None
    assert statement_affinity(statement(['health']), values, signals, domain='tech')[0]['drive_affinity'] is None
    assert statement_affinity(statement(['health']), values, signals, domain='life')[0]['drive_affinity'] == pytest.approx(.4)


def test_statement_granularity_and_memory_attributes_cannot_spoof_runtime_signals():
    values = settings()
    document = document_template('mixed historical memory', catalog=values['catalog'])
    document['statements'] = [statement(['health']), statement(['code'])]
    document['attributes'] = [{'definition_id': 'drive_affinity', 'value': {'state': 'known', 'value': 1}},
                              {'definition_id': 'signal_care', 'value': {'state': 'known', 'value': 1}}]
    context = {'now': NOW, 'facts': {}, 'purpose': 'current', 'signals': resolve_signals(values, [snapshot()], NOW)}
    result = score_document(document, values, context, {'relevance': .5})
    assert result['statements'][0]['score'] == pytest.approx(.7)
    assert result['statements'][1]['score'] == pytest.approx(.5)
    assert result['statements'][1]['features']['drive_affinity'] is None
    assert result['statements'][0]['features']['signal_care'] == .5
    # Ranking preference is not source/condition/current-fact authorization.
    usage = resolve_document(document, context, [])
    assert all(row['usage'] != 'current_basis' for row in usage)


def test_preview_and_runtime_use_identical_affinity_math_without_model_or_write():
    values = settings()
    inputs = {'relevance': .5}
    scenario = {'now': NOW.isoformat(), 'topics': ['health'], 'signal_snapshots': [snapshot()]}
    preview = preview_pipeline(values, inputs, scenario)
    document = document_template('simulation', catalog=values['catalog'])
    document['statements'] = [statement(['health'])]
    runtime = score_document(document, values, {'now': NOW, 'facts': {}, 'purpose': 'current',
        'signals': resolve_signals(values, scenario['signal_snapshots'], NOW)}, inputs)
    assert preview['pipeline']['score'] == runtime['score'] == pytest.approx(.7)
    assert preview['pipeline']['context']['simulation'] is True
    assert preview['provider_requests'] == 0 and preview['saved'] is False


def test_header_scope_is_exact_and_other_producers_cannot_be_forged():
    scope = {'conversation_id': 'c1', 'turn_id': 't1', 'source_event_id': 'h1'}
    envelope = {'schema_version': 'recall-signals-v1', 'scope': scope,
                'snapshots': [snapshot(producer='aiz.needs', key='curiosity')]}
    assert read_transport(json.dumps(envelope), **scope)['state'] == 'received'
    assert read_transport(json.dumps(envelope), **{**scope, 'turn_id': 't2'})['reason'] == 'scope_mismatch'
    envelope['snapshots'][0]['producer'] = 'ombre.persona'
    assert read_transport(json.dumps(envelope), **scope)['reason'] == 'invalid_transport'
    assert read_transport('invalid', **scope)['state'] == 'unavailable'


def test_persona_reuses_source_sample_and_does_not_invent_provenance_for_self_model():
    sample = sample_persona({'protective_drive': 0, 'longing': .3}, profile_id='p', session_id='s', updated_at=NOW.isoformat())
    state = {'recall_signal_source': sample, 'affect': {'protective_drive': 1}, 'inner_thought': 'never transmitted'}
    frozen = persona_snapshot(state)
    sample['signals']['protective_drive']['value'] = 1
    assert frozen['signals']['protective_drive']['value'] == 0
    assert 'inner_thought' not in str(frozen)
    assert persona_snapshot({'affect': {'longing': .9}})['state'] == 'unavailable'


def test_signal_unknown_time_does_not_borrow_collection_update_from_another_item():
    source = snapshot()
    source['signals']['protective_drive']['as_of'] = None
    bound = resolve_signals(settings(), [source], NOW)
    assert bound['care']['as_of'] is None
    assert bound['care']['reason'] == 'time_unknown'
    assert bound['care']['value'] is None


def test_removed_signal_binding_cannot_be_replaced_by_memory_attribute():
    values = settings()
    values['recall']['signal_bindings'] = []
    values['recall']['affinity_mappings'] = []
    values['features'].append({'id': 'removed', 'value_type': 'number', 'source': 'runtime',
                               'missing': 'unknown', 'input_key': 'signal_removed'})
    document = document_template('synthetic memory', catalog=values['catalog'])
    document['attributes'] = [{'definition_id': 'signal_removed', 'value': {'state': 'known', 'value': 1}}]
    result = score_document(document, values, {'now': NOW, 'signals': {}}, {'relevance': .5})
    assert result['statements'][0]['calculation']['features']['removed'] is None
