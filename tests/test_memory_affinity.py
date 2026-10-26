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


@pytest.mark.parametrize('aggregation', ['max', 'mean'])
def test_unknown_matching_source_is_visible_while_known_reward_remains_effective(aggregation):
    values = settings()
    values['recall']['affinity_aggregation'] = aggregation
    values['recall']['affinity_mappings'].append({
        'id': 'unknown_care', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    signals = resolve_signals(values, [snapshot()], NOW)
    features, trace = statement_affinity(statement(['health']), values, signals)
    target = trace['targets']['drive_affinity']
    assert features['drive_affinity'] == pytest.approx(.4)
    assert features['signal_curiosity'] is None
    assert trace['aggregation'] == aggregation
    assert target['state'] == 'partial' and target['complete'] is False
    assert target['reason'] == 'partial_signal_unknown'
    assert target['coverage'] == {'known': 1, 'unknown': 1, 'matched': 2, 'ratio': .5}
    unknown = target['contributions'][1]
    assert unknown['state'] == 'unknown' and unknown['value'] is None and unknown['intensity'] is None
    assert unknown['reason'] == 'producer_unavailable'
    values['recall']['affinity_mappings'][-1]['weight'] = 0
    features, trace = statement_affinity(statement(['health']), values, signals)
    assert features['drive_affinity'] == pytest.approx(.4)
    assert trace['targets']['drive_affinity']['contributions'][1]['value'] is None
    assert trace['targets']['drive_affinity']['state'] == 'partial'


@pytest.mark.parametrize('aggregation,expected', [('max', .4), ('mean', .35)])
def test_multiple_known_signals_use_configured_aggregation_without_duplicate_topics(aggregation, expected):
    values = settings()
    values['recall']['affinity_aggregation'] = aggregation
    values['recall']['affinity_mappings'].append({
        'id': 'curious_health', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    signals = resolve_signals(values, [snapshot(), snapshot(.6, producer='aiz.needs', key='curiosity')], NOW)
    features, trace = statement_affinity(statement(['health', 'care', 'health']), values, signals)
    target = trace['targets']['drive_affinity']
    assert features['drive_affinity'] == pytest.approx(expected)
    assert target['state'] == 'computed' and target['complete'] is True
    assert target['reason'] is None
    assert target['coverage'] == {'known': 2, 'unknown': 0, 'matched': 2, 'ratio': 1}
    assert [row['value'] for row in target['contributions']] == pytest.approx([.4, .3])


@pytest.mark.parametrize('aggregation,expected', [('max', .4), ('mean', .2)])
def test_partial_aggregation_uses_multiple_known_contributions_including_known_zero(aggregation, expected):
    values = settings()
    values['recall']['affinity_aggregation'] = aggregation
    values['recall']['affinity_mappings'].extend([
        {'id': 'curious_health', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
         'target': 'drive_affinity', 'weight': .5},
        {'id': 'unknown_health', 'selector': {'topic': 'health'}, 'signal_id': 'unavailable',
         'target': 'drive_affinity', 'weight': .9}])
    signals = resolve_signals(values, [snapshot(), snapshot(0, producer='aiz.needs', key='curiosity')], NOW)
    features, trace = statement_affinity(statement(['health']), values, signals)
    target = trace['targets']['drive_affinity']
    assert features['drive_affinity'] == pytest.approx(expected)
    assert target['state'] == 'partial' and target['coverage']['known'] == 2
    assert target['contributions'][1]['state'] == 'known'
    assert target['contributions'][1]['value'] == 0
    assert target['contributions'][2]['reason'] == 'binding_unknown'


@pytest.mark.parametrize('weight', [0, .8])
def test_all_unknown_stays_unknown_even_with_zero_mapping_weight(weight):
    values = settings()
    values['recall']['affinity_mappings'][0]['weight'] = weight
    features, trace = statement_affinity(statement(['health']), values, {})
    target = trace['targets']['drive_affinity']
    assert features['drive_affinity'] is None
    assert target['value'] is None and target['state'] == 'unknown' and target['complete'] is False
    assert target['reason'] == 'signal_unknown'
    assert target['coverage'] == {'known': 0, 'unknown': 1, 'matched': 1, 'ratio': 0}
    assert target['contributions'][0]['value'] is None


def test_known_zero_is_an_effective_partial_value_and_zero_weight_known_stays_known():
    values = settings()
    values['recall']['affinity_mappings'].append({
        'id': 'unknown_health', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    signals = resolve_signals(values, [snapshot(0)], NOW)
    features, trace = statement_affinity(statement(['health']), values, signals)
    assert features['drive_affinity'] == 0
    assert trace['targets']['drive_affinity']['state'] == 'partial'
    values['recall']['affinity_mappings'][0]['weight'] = 0
    signals = resolve_signals(values, [snapshot(.8)], NOW)
    _, trace = statement_affinity(statement(['health']), values, signals)
    assert trace['targets']['drive_affinity']['contributions'][0]['state'] == 'known'
    assert trace['targets']['drive_affinity']['contributions'][0]['value'] == 0


def test_no_mapping_has_no_reward_or_fictitious_coverage():
    features, trace = statement_affinity(statement([]), settings(), {})
    assert all(features[target] is None for target in ('drive_affinity', 'interest_affinity', 'goal_alignment'))
    assert trace['matched_rules'] == []
    assert trace['targets']['drive_affinity'] == {
        'value': None, 'reason': 'not_mapped', 'state': 'unknown', 'complete': False,
        'coverage': {'known': 0, 'unknown': 0, 'matched': 0, 'ratio': None}, 'contributions': []}


@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), -float('inf'), True, '0.5', -.1, 1.1, 10 ** 1000])
def test_invalid_direct_signal_cannot_become_known_or_leak_as_formula_input(invalid):
    features, trace = statement_affinity(statement(['health']), settings(), {
        'care': {'state': 'known', 'value': invalid, 'reason': None}})
    assert features['signal_care'] is None and features['drive_affinity'] is None
    contribution = trace['targets']['drive_affinity']['contributions'][0]
    assert contribution['state'] == 'unknown' and contribution['value'] is None
    assert contribution['reason'] == 'invalid_signal_value'
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize('source', [
    {'state': 'unknown', 'value': .9, 'reason': 'scope_mismatch'},
    {'state': 'known', 'value': .9, 'reason': 'stale'},
    'corrupt', None])
def test_unknown_or_corrupt_direct_sources_cannot_be_relabelled_known(source):
    features, trace = statement_affinity(statement(['health']), settings(), {'care': source})
    contribution = trace['targets']['drive_affinity']['contributions'][0]
    assert features['signal_care'] is None and features['drive_affinity'] is None
    assert contribution['state'] == 'unknown' and contribution['intensity'] is None
    if isinstance(source, dict):
        assert contribution['reason'] == source['reason']


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


def test_partial_reward_reaches_existing_scoring_formula_only_when_selected():
    values = settings()
    values['recall']['affinity_mappings'].append({
        'id': 'unknown_health', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    signals = resolve_signals(values, [snapshot()], NOW)

    def score(topics, base, *, bound=signals):
        document = document_template('synthetic memory', catalog=values['catalog'])
        document['statements'] = [statement(topics)]
        sid = document['statements'][0]['statement_id']
        return score_document(document, values, {'now': NOW, 'signals': bound}, {
            'relevance': base, 'statement_selection': {
                'direct_ids': [sid], 'support_ids': [], 'matches': {}}})

    rewarded = score(['health'], .5)
    other = score(['code'], .6)
    assert rewarded['score'] == pytest.approx(.7)
    assert rewarded['score'] > other['score']
    assert rewarded['statements'][0]['affinity']['targets']['drive_affinity']['state'] == 'partial'
    assert rewarded['statements'][0]['calculation']['features']['drive'] == pytest.approx(.4)

    # The optional affinity changes no default formula or base ordering.
    values['recall']['formula_id'] = 'final'
    assert score(['health'], .5)['score'] == .5 < score(['code'], .6)['score']
    values['recall']['formula_id'] = 'affinity_score'
    assert score(['health'], .5, bound={})['score'] == .5 < score(['code'], .6, bound={})['score']
    values['recall']['affinity_mappings'] = []
    assert score(['health'], .5)['score'] == .5 < score(['code'], .6)['score']


@pytest.mark.parametrize('aggregation', ['max', 'mean'])
def test_partial_preview_and_runtime_have_same_reward_and_incomplete_diagnostics(aggregation):
    values = settings()
    values['recall']['affinity_aggregation'] = aggregation
    values['recall']['affinity_mappings'].append({
        'id': 'unknown_health', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    preview = preview_pipeline(values, {'relevance': .5}, {
        'now': NOW.isoformat(), 'topics': ['health'], 'signal_snapshots': [snapshot()]})
    row = preview['pipeline']['statements'][0]
    assert row['score'] == pytest.approx(.7)
    assert row['features']['drive_affinity'] == pytest.approx(.4)
    assert row['affinity']['aggregation'] == aggregation
    assert row['affinity']['targets']['drive_affinity']['complete'] is False
    assert row['affinity']['targets']['drive_affinity']['contributions'][1]['value'] is None
    assert preview['provider_requests'] == 0 and preview['saved'] is False


def test_partial_reward_is_used_by_attributed_runtime_candidate_before_and_after_rerank():
    from types import SimpleNamespace
    from memory_recall_runtime import score_pool

    values = settings()
    values['recall']['affinity_mappings'].append({
        'id': 'unknown_health', 'selector': {'topic': 'health'}, 'signal_id': 'curiosity',
        'target': 'drive_affinity', 'weight': .5})
    service = SimpleNamespace(config={'memory_system': values}, recall_policy=SimpleNamespace(),
                              memory_authority_view=None)
    document = document_template('synthetic memory', catalog=values['catalog'])
    document['statements'] = [statement(['health'])]
    request = {'_signal_snapshots': [snapshot()], '_signal_sampled_at': NOW,
               'allow_semantic': False, '_semantic_records': {'m1': {'document': document}}}
    query = document['statements'][0]['content']
    candidate = {'bucket': {'id': 'm1'}, 'score': .5, 'keyword_score': .5}
    score_pool(service, request, query, [candidate])
    assert candidate['score'] == pytest.approx(.7)
    target = candidate['_configured_score']['statements'][0]['affinity']['targets']['drive_affinity']
    assert target['state'] == 'partial' and target['coverage']['unknown'] == 1
    candidate['_raw_fused_score'] = .4
    score_pool(service, request, query, [candidate], after_rerank=True)
    assert candidate['score'] == pytest.approx(.56)


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
