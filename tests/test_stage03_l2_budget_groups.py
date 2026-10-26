"""Replay actual saved provider units; no picker, source or model call is mocked."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import pytest
from gateway import GatewayService
from memory_query_evidence import match_statements, correction_matches, query_purpose
from memory_recall_semantics import resolve_document, resolve_statement, statement_context, score_document
from memory_recall_runtime import render_presentation, query_temporal_contexts
from memory_delivery import candidates, compiled_candidates, final_fragments
from utils import count_tokens_approx

DATA = json.loads(Path(__file__).with_name('data_stage03_real_l2_units.json').read_text(encoding='utf-8'))


def service(total=2400):
    result = GatewayService.__new__(GatewayService)
    result.config = {'identity': {'ai_name': '顾衍'}, 'gateway': {}}
    result.identity = {'ai_name': '顾衍'}
    result.inject_total_budget = total
    result.memory_detail_recall_max_ids = 2
    return result


def replay(source, question):
    row = DATA['sources'][source]
    doc = row['document']
    context = {'now': datetime(2026, 10, 26, 20, tzinfo=timezone(timedelta(hours=8))),
        'target_time': None, 'query': question, 'purpose': query_purpose(question)['purpose'], 'facts': {},
        'memory_id': row['memory_id'], 'revision': 1}
    # This is an immutable replay of the two observed source-qualified inputs,
    # not authorization for fresh live source writes or an invented picker score.
    uses = resolve_document(doc, context, source_states={b['binding_id']: 'available' for b in doc['evidence_bindings']})
    selection = match_statements(doc, context)
    selection['required_support_ids'] = selection['support_ids'][:]
    usage = {'state': 'l1', 'record': {'document': doc}, 'items': uses,
        'query_purpose': context['purpose'], 'statement_selection': selection,
        'correction_statement_ids': correction_matches(doc, context, selection)}
    product = row['l2']
    presentation = {'text': product['full_injection_text'], 'presentation_revision': 1,
        'units': [{'unit_id': u['section_id'], 'text': u['text'], 'statement_refs': u['statement_refs']}
                  for u in product['coherent_sections']],
        'coverage': {'used_statement_ids': [s['statement_id'] for s in doc['statements']]}}
    return usage, presentation


@pytest.mark.parametrize('question', DATA['questions'])
def test_real_current_correction_does_not_require_old_plan_or_every_lexical_aside(question):
    usage, presentation = replay('auto', question)
    trace = {}
    text, reason, ids, partial = render_presentation(service(), usage, '', presentation, 400, trace=trace)
    assert reason is None and partial and ids == ['sec_correction']
    assert text == DATA['sources']['auto']['l2']['coherent_sections'][1]['text']
    assert '2026年11月3日晚上八点' in text and '归还银色录音笔' in text and '没有值班' in text
    assert '不能写成她已经赴约' in text and '不能写成陈屿已经归还' in text
    assert count_tokens_approx(text) <= 400
    assert set(trace['l2_budget']['required_statement_ids']) < set(trace['l2_budget']['matched_statement_ids'])


@pytest.mark.parametrize('question', DATA['questions'])
def test_real_book_correction_and_conditions_are_not_separated_to_force_a_fit(question):
    usage, presentation = replay('chat', question)
    trace = {}
    text, reason, ids, partial = render_presentation(service(), usage, '', presentation, 400, trace=trace)
    assert not text and not ids and not partial
    assert reason == 'no_complete_correction_unit_fits_remaining_budget'
    required = set(trace['l2_budget']['required_statement_ids'])
    assert required and not any(required <= set(unit['statement_refs']) for unit in presentation['units'])
    # The honest budget gap is recorded, rather than blamed on an owner-off template.
    assert trace['l2_budget']['available_tokens'] == 400
    assert trace['l2_budget']['full_text_tokens'] > 400


def test_change_history_keeps_the_before_and_after_complete_units():
    usage, presentation = replay('auto', '请回顾这次约定最初怎么定，后来怎么改，前后条件各是什么？')
    usage['query_purpose'] = 'history'
    text, reason, _ids, _partial = render_presentation(service(), usage, '', presentation, 400)
    assert not text and reason == 'no_complete_correction_unit_fits_remaining_budget'


def test_real_saved_full_l2_bodies_reach_composer_wire_and_delivery_without_rewriting():
    svc = service()
    parts, spans, traces, records = [], [], [], {}
    remaining = 1200
    for source in ('chat', 'auto'):
        row = DATA['sources'][source]
        usage, presentation = replay(source, DATA['questions'][1])
        text, reason, ids, partial = render_presentation(svc, usage, '', presentation, remaining)
        assert reason is None and not partial and text == row['l2']['full_injection_text']
        parts.append(text); remaining -= count_tokens_approx(text)
        spans.append({'bucket_id': row['memory_id'], 'section': 'direct', 'text': text})
        traces.append({'bucket_id': row['memory_id'], 'memory_id': row['memory_id'], 'l1_revision': 1,
            'presentation_revision': 1, 'selected_unit_ids': ids, 'partial_presentation': False,
            'injected_statement_ids': [s['statement_id'] for s in row['document']['statements']]})
        records[row['memory_id']] = {'memory_id': row['memory_id'], 'revision': 1, 'document': row['document']}
    body = '\n\n'.join(parts)
    projection = {'rendered_memory_spans': spans, 'presentation_trace': traces, 'fact_body': body,
        'body': body, 'carrier_memory_spans': []}
    plan = {'scope': 'talk.initial', 'gateway_slice': {'settings': {}, 'blocks': [{
        'block_id': 'replay-real-sources', 'source_id': 'ombre.recalled_memory', 'scope': 'talk.initial',
        'mode': 'live_source', 'enabled': True, 'order': 110, 'role': 'user',
        'anchor': 'gateway.current_user_prefix', 'token_budget': None, 'condition': {}}]}}
    stable, dynamic, debug = svc._build_composed_context_messages(plan,
        persona_block='', core_memory='', portrait_memory='', just_now_context='', recent_context='',
        recalled_memory=body, relationship_weather='', favorite_memory='', related_memory='',
        targeted_memory_detail='', dream_context='', active_reminders='', memory_detail_recall_instruction='',
        handoff_tool_hint='', context_mode='', date_persona_trace='', date_recall='', memory_recall_projection=projection)
    messages = svc._inject_context_messages([{'role': 'system', 'content': 'synthetic validation'},
        {'role': 'user', 'content': DATA['questions'][1]}], stable, dynamic,
        composed_messages=debug['_placement_messages'])
    fragments = final_fragments(compiled_candidates(projection, candidates(projection, records)), {'messages': messages})
    assert len(fragments) == 2
    assert {f['text'] for f in fragments} == set(parts)
    assert all(f['l2_revision'] == 1 and f['coverage']['partial_presentation'] is False for f in fragments)
    assert count_tokens_approx(body) <= 1200


@pytest.mark.parametrize("carrier", ["direct", "core"])
def test_legacy_global_cap_never_leaks_a_prefix_without_its_final_and_condition(carrier):
    svc = service(110)
    body = '完整计划：2026年11月3日晚上八点见陈屿。' * 18 + '仅在录音笔已归还且主人没有值班时成立。'
    stable, dynamic = svc._build_injected_context_messages(persona_block='', core_memory=body if carrier == 'core' else '', portrait_memory='',
        recalled_memory=body if carrier == 'direct' else '', memory_recall_projection={})
    assert body not in stable + dynamic
    assert '完整计划' not in stable + dynamic
    assert count_tokens_approx(stable) + count_tokens_approx(dynamic) <= 110


NOW = datetime(2026, 10, 26, 20, tzinfo=timezone(timedelta(hours=8)))


def temporal_context(question):
    temporal, parts = query_temporal_contexts(service(), {'q_current': question}, question, NOW)
    return {'now': NOW, 'query': question, 'facts': {}, **temporal, 'query_part_contexts': parts}


@pytest.mark.parametrize('question', DATA['questions'])
def test_original_multi_date_questions_do_not_apply_the_first_date_to_every_plan(question):
    context = temporal_context(question)
    assert context['target_time'] is None
    row = DATA['sources']['auto']
    uses = resolve_document(row['document'], context,
        source_states={b['binding_id']: 'available' for b in row['document']['evidence_bindings']})
    future = next(u for u in uses if u['statement_id'] == row['document']['statements'][2]['statement_id'])
    assert future['scope_match'] is False
    assert future['usage'] == 'relationship_reference'
    assert 'future_plan_reference_not_execution' in future['reason_codes']
    assert future['condition_match'] is None


def test_independent_named_questions_bind_dates_to_their_actual_subjects_and_time_features():
    question = '林岚在2026年10月30日去哪里还摄影册？陈屿在2026年11月3日见面的条件是什么？'
    context = temporal_context(question)
    assert context['target_time'] is None and len(context['query_part_contexts']) == 2
    examples = [('chat', 3, '2026-10-30', 0), ('auto', 2, '2026-11-03', 1)]
    from memory_system_config import values_from_config
    for source, index, expected_date, part in examples:
        doc = DATA['sources'][source]['document'];statement = doc['statements'][index]
        scoped = statement_context(statement, context, doc)
        assert scoped['target_time'].date().isoformat() == expected_date
        assert scoped['query_part_index'] == part
        assert scoped['statement_time_binding'] == 'evidenced_question_subject'
        calculation = score_document(doc, values_from_config({}), context, {'relevance': .7})
        row = next(row for row in calculation['statements'] if row['statement_id'] == statement['statement_id'])
        assert row['features']['event_age_days'] >= 0
        assert row['features']['scope_match'] == resolve_statement(statement, scoped, source_state='available')['scope_match']
        assert row['features']['time_factor'] is None


@pytest.mark.parametrize('question', [
    '2026年10月30日要做什么？2026年11月3日有什么条件？',
    '林岚在2026年10月30日有什么安排？林岚在2026年11月3日有什么安排？',
])
def test_dates_alone_or_repeated_subjects_do_not_guess_a_statement_binding(question):
    context = temporal_context(question)
    doc = DATA['sources']['chat']['document']
    scoped = statement_context(doc['statements'][3], context, doc)
    assert scoped['target_time'] is None and scoped['statement_time_binding'] == 'ambiguous'


def test_future_plan_never_overrides_withdrawal_supersession_or_explicit_history():
    statement = DATA['sources']['auto']['document']['statements'][2]
    context = temporal_context(DATA['questions'][1])
    assert resolve_statement(statement, context, source_state='revoked')['usage'] == 'excluded'
    assert resolve_statement(statement, context, source_state='available', superseded=True)['usage'] == 'historical_evidence'
    assert resolve_statement(statement, {**context, 'purpose': 'history'}, source_state='available')['usage'] == 'historical_evidence'
    review = {'id': 'existing-review', 'selector': {}, 'function': 'none', 'review_after_hours': 1}
    assert resolve_statement(statement, context, [review], source_state='available')['usage'] == 'historical_evidence'


def test_explicit_boundary_and_english_correction_cannot_be_dropped_for_shorter_old_premise():
    usage = {'state': 'l1', 'query_purpose': 'current',
        'statement_selection': {'direct_ids': ['old', 'new', 'boundary'], 'support_ids': []},
        'items': [
            {'statement_id': 'old', 'source_state': 'available', 'content': 'The earlier appointment was Tuesday.'},
            {'statement_id': 'new', 'source_state': 'available', 'content': 'The appointment was corrected to Friday.'},
            {'statement_id': 'boundary', 'source_state': 'available', 'content': 'I did not promise to attend.',
             'classification': {'kind': {'state': 'known', 'value': 'boundary'}}}]}
    presentation = {'text': 'a' * 500, 'units': [
        {'unit_id': 'old', 'text': 'Tuesday.', 'statement_refs': ['old']},
        {'unit_id': 'new', 'text': 'Corrected to Friday; no promise to attend.', 'statement_refs': ['new', 'boundary']}],
        'coverage': {'used_statement_ids': ['old', 'new', 'boundary']}}
    text, reason, ids, partial = render_presentation(service(), usage, '', presentation,
        count_tokens_approx(presentation['units'][1]['text']))
    assert reason is None and ids == ['new'] and partial
    assert text == presentation['units'][1]['text']
