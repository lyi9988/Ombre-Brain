"""Offline semantic packing: a claim never outlives its subject or limits."""
from copy import deepcopy

import pytest

from memory_readable_projection import fit_units, qualified_prose, qualified_units
from utils import count_tokens_approx


def known(value):
    return {'state': 'known', 'value': value}


def unknown(state='not_stated'):
    return {'state': state, 'value': None}


def statement(ident='s1', content='我不喝咖啡。', **overrides):
    item = {'statement_id': ident, 'content': content, 'subject_ref': known('alice'),
        'usage': 'historical_evidence',
        'classification': {'modality': known('assertion'), 'basis': known('record')},
        'conditions': {'state': 'not_applicable', 'text': None, 'predicate': None, 'exceptions': []},
        'time': {'event': {'start': unknown(), 'end': unknown(), 'interpretation': 'unknown',
                          'precision': 'unknown', 'timezone': unknown()},
                 'applicability': {'temporal_scope': 'unknown', 'start': unknown(), 'end': unknown(),
                                   'scene_ref': None, 'recurrence_text': None}}}
    item.update(overrides)
    return item


def usage(*items, narrative_allowed=True, all_statements=None):
    return {'state': 'l1', 'narrative_allowed': narrative_allowed, 'items': list(items),
        'record': {'document': {'entities': [{'entity_id': 'alice', 'name': 'Alice', 'entity_type': known('third_party')}],
            'statements': deepcopy(list(items) if all_statements is None else all_statements), 'scenes': []}}}


def test_narrative_and_all_subject_time_condition_limits_are_one_indivisible_unit():
    body = 'Alice住在北京。'
    item = statement(content=body)
    item['time']['event'].update(start=known('2020-01-01'), interpretation='instant')
    item['time']['applicability'].update(temporal_scope='time_range',
        start=known('2020-01-01'), end=known('2020-12-31'))
    item['conditions'] = {'state': 'known', 'text': '仅限当年的借住期间。', 'predicate': None,
        'exceptions': [{'state': 'known', 'text': '出差时不住这里。', 'predicate': None}]}
    units = qualified_units(usage(item), body)
    assert len(units) == 2  # One full qualified narrative, then its safe statement alternative.
    for text in (body, 'Alice', '2020-01-01', '2020-12-31', '仅限当年的借住期间。', '出差时不住这里。'):
        assert text in units[0]
    assert fit_units(units, count_tokens_approx(body)) == ''
    assert fit_units(units, count_tokens_approx(units[0])) == units[0]
    assert qualified_prose(usage(item), body) == units[0]


def test_oversize_complete_narrative_can_fall_back_to_complete_statement_without_duplication():
    item = statement(content='不喝咖啡。')
    item['conditions'] = {'state': 'known', 'text': '仅限生病期间。', 'predicate': None, 'exceptions': []}
    units = qualified_units(usage(item), '完整经历中的其他细节。' * 100)
    assert len(units) == 2 and units[1] in units[0]
    assert fit_units(units, count_tokens_approx(units[1])) == units[1]
    assert 'Alice' in units[1] and '不喝咖啡。' in units[1] and '仅限生病期间。' in units[1]
    assert fit_units(units, count_tokens_approx(units[0])) == units[0]
    assert qualified_prose(usage(item), '完整经历中的其他细节。' * 100) == units[0]


@pytest.mark.parametrize('where', ['condition', 'exception'])
def test_predicate_without_supplied_natural_wording_never_leaks_original_narrative(where):
    item = statement(content='可以吃芒果。')
    predicate_only = {'state': 'known', 'text': None, 'predicate': {'op': 'eq', 'field': 'health', 'value': 'well'}}
    item['conditions'] = {**predicate_only, 'exceptions': []} if where == 'condition' else {
        'state': 'known', 'text': '只在饭后。', 'predicate': None, 'exceptions': [predicate_only]}
    assert qualified_units(usage(item), '可以吃芒果。') == []
    assert qualified_prose(usage(item), '可以吃芒果。') == ''


def test_unrenderable_claim_does_not_block_a_separate_faithful_statement():
    bad = statement('bad', '可以吃芒果。')
    bad['conditions'] = {'state': 'known', 'text': None,
        'predicate': {'op': 'eq', 'field': 'hour', 'value': 3}, 'exceptions': []}
    good = statement('good', '不喝咖啡。')
    units = qualified_units(usage(bad, good), '可以吃芒果，也不喝咖啡。')
    assert len(units) == 1 and '不喝咖啡。' in units[0] and 'Alice' in units[0]
    assert '芒果' not in units[0] and 'hour' not in units[0]


def test_third_party_first_person_is_quoted_with_resolved_subject_without_inventing_a_speaker():
    units = qualified_units(usage(statement()), '我不喝咖啡。')
    assert len(units) == 1 and '关于Alice：“我不喝咖啡。”' in units[0]
    assert 'Alice说' not in units[0] and 'assistant' not in units[0] and '主体尚未确定' not in units[0]
    assert not units[0].startswith('我不喝咖啡。')


def test_unresolvable_known_subject_cannot_be_silently_downgraded_to_unknown():
    item = statement(subject_ref=known('missing-person'))
    assert qualified_units(usage(item), '我不喝咖啡。') == []


def test_unknown_subject_and_unresolved_claim_remain_unknown_in_the_same_unit():
    item = statement(subject_ref=unknown('unresolved'), usage='unresolved_evidence')
    item['conditions'] = {'state': 'unresolved', 'text': None, 'predicate': None, 'exceptions': []}
    units = qualified_units(usage(item, narrative_allowed=False), '')
    assert len(units) == 1
    assert '主体尚未确定' in units[0] and '尚未确定的是' in units[0]
    assert '适用条件尚未确定' in units[0] and '我不喝咖啡。' in units[0]
    assert fit_units(units, count_tokens_approx('我不喝咖啡。')) == ''


def test_uncertainty_window_is_not_rewritten_as_duration_and_keeps_precision_timezone():
    item = statement()
    item['time']['event'].update(start=known('2020-01-01T01:02:03+08:00'),
        end=known('2020-01-03T04:05:06+08:00'), interpretation='uncertainty_window',
        precision='approximate', timezone=known('Asia/Shanghai'))
    unit = qualified_units(usage(item, narrative_allowed=False), '')[0]
    assert '可能发生在' in unit and '具体时间尚未确定' in unit
    assert '2020-01-01 01:02:03+08:00' in unit and '2020-01-03 04:05:06+08:00' in unit
    assert 'Asia/Shanghai' in unit and '约略时间' in unit and '这段经历的时间为' not in unit


@pytest.mark.parametrize(('end_state', 'expected'), [
    ('unbounded', '未限定结束时间'), ('not_stated', '结束时间未说明'),
    ('unresolved', '结束时间尚未确定'), ('not_applicable', '结束时间不适用')])
def test_missing_upper_bound_never_becomes_an_implicit_permanent_scope(end_state, expected):
    item = statement()
    item['time']['applicability'].update(temporal_scope='time_range', start=known('2020-01-01'), end=unknown(end_state))
    unit = qualified_units(usage(item, narrative_allowed=False), '')[0]
    assert '从2020-01-01开始适用' in unit and expected in unit
    assert fit_units([unit], count_tokens_approx('从2020-01-01开始适用')) == ''


def test_recurring_scope_keeps_its_actual_bounds_and_textual_schedule():
    item = statement()
    item['time']['applicability'].update(temporal_scope='recurring', start=known('2020-01-01'),
        end=known('2020-12-31'), recurrence_text='只在每周一，节假日不适用。')
    unit = qualified_units(usage(item, narrative_allowed=False), '')[0]
    assert '2020-01-01至2020-12-31' in unit and '只在每周一，节假日不适用。' in unit


def test_partial_reference_coverage_and_excluded_claim_never_admit_mixed_narrative():
    good, excluded = statement('good'), statement('excluded', '喜欢芒果。', usage='excluded')
    units = qualified_units(usage(good, excluded), '完整原稿中的芒果部分不能采用。')
    assert len(units) == 1 and '咖啡' in units[0] and '芒果' not in units[0]


def test_plan_and_negation_stay_together_instead_of_becoming_completed_action():
    item = statement(content='下周不去北京。')
    item['classification']['modality'] = known('plan')
    unit = qualified_units(usage(item, narrative_allowed=False), '')[0]
    assert '当时的计划是' in unit and '下周不去北京。' in unit and '已经' not in unit


def test_packing_skips_oversize_units_and_wraps_the_combined_body_once():
    units = ['很长的经历。' * 100, '关于Alice：不喝咖啡。\n仅限生病期间。', '关于Bob：喝茶。']
    wrap = lambda text: 'OWNER-BEGIN\n' + text + '\nOWNER-END'
    expected = wrap('\n\n'.join(units[1:]))
    result = fit_units(units, count_tokens_approx(expected), wrap=wrap)
    assert result == expected
    assert result.count('OWNER-BEGIN') == result.count('OWNER-END') == 1
    assert count_tokens_approx(result) <= count_tokens_approx(expected)
    assert fit_units(units[1:], 0, wrap=wrap) == ''
    assert fit_units(units[1:], 10000, wrap=lambda _text: '') == ''


def test_legacy_paragraphs_are_not_split_from_their_original_qualifiers():
    text = '我住北京。\n\n但那只是2020年借住，之后已经搬走。'
    units = qualified_units({'state': 'legacy'}, text)
    assert units == [text]
    assert fit_units(units, count_tokens_approx('我住北京。')) == ''
