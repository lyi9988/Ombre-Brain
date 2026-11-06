"""Normal event-first engine/Authority/Review/L2 flow; scripted provider, no network.

The provider's selection is scripted to test custody and continuation. These
tests do not claim that a real model judges the value of receipt/chat wording.
"""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from memory_candidate_preparation import MemoryCandidatePreparation
from memory_l1_contract import verify_text_anchor
from memory_l1_ingestion import source_snapshot
from memory_l0_sources import MemoryL0Sources
from memory_l2_service import MemoryL2Service
from memory_source_provenance import daily_sources
from reflection_engine import ReflectionEngine
from test_daily_chat_memory_quality_review_v4 import FakeRawEvents, FakeTurns
from test_memory_authority_r1 import reflection_config
from test_memory_l1_ingestion_p2c import setup

DAY = '2026-10-20'
SELECTION = 'daily_chat_event_selection'
ASSEMBLY = 'daily_chat_event_assembly'
SUMMARY_MARKER = '__FREE_SUMMARY_UNSUPPORTED_FUTURE__'
CONTEXT_MARKER = '__ONLY_ORGANIZATION_NEVER_L1_EVIDENCE__'


def harness(tmp_path, texts, groups, *, calls_per_run=100):
    service, _, _, _ = setup(tmp_path)
    config = reflection_config(tmp_path / 'engine', mode='review')
    config['reflection'].update(daily_chat_memory_extraction_mode='event_first',
        daily_chat_memory_window_max_input_chars=8000, daily_chat_memory_max_windows_per_run=calls_per_run)
    engine = ReflectionEngine(config)
    engine.memory_authority_store = service.authority
    engine._memory_l1_revisions = lambda *args: service
    engine._memory_authority_service = lambda *args: service.commit
    engine.daily_chat_memory_turn_limit = 0
    events = [{'id': 11 + index, 'session_id': 'session', 'role': role, 'text': text,
        'created_at': f'{DAY}T01:{index:02}:00Z'} for index, (role, text) in enumerate(texts)]
    window = [{'id': 1, 'session_id': 'session', 'created_at': events[0]['created_at'],
        'user_text': '\n'.join(text for role, text in texts if role == 'user'),
        'assistant_text': '\n'.join(text for role, text in texts if role == 'assistant'),
        'raw_event_ids': [event['id'] for event in events], '_semantic_sources': events}]
    calls, control = [], {'unknown': None, 'invalid': None}

    async def create(**kwargs):
        request = json.loads(kwargs['messages'][-1]['content'])
        kind = request.get('input_kind') or ('l2' if 'l1_document' in request else 'unexpected')
        calls.append({'kind': kind, 'payload': request, 'wire': deepcopy(kwargs)})
        if kind in (SELECTION, ASSEMBLY):
            assert kwargs['max_tokens'] == 32000
        if kind == ASSEMBLY:
            refs = [row['binding_id'] for row in request['l1_contract']['source_texts']]
            assert request['selection_focus'] == {'source_refs': refs, 'evidence': False}
            assembly_wire = json.dumps(kwargs['messages'], ensure_ascii=False)
            assert SUMMARY_MARKER not in assembly_wire and CONTEXT_MARKER not in assembly_wire
            assert '将来要重新明确' not in assembly_wire
        if control['unknown'] == kind:
            raise TimeoutError('synthetic uncertain provider outcome')
        if control['invalid'] == kind:
            value = ({'context': '已收到，但缺少合法选材。', 'memories': [{'point': '无证据', 'anchors': []}]}
                     if kind == SELECTION else {'candidates': 'not-an-array'})
        elif kind == SELECTION:
            assert all({'source', 'speaker', 'when', 'text'} <= set(source) for source in request['sources'])
            memories = []
            for point, quotes in groups:
                if control.get('visible_groups_only') and not all(
                        any(quote in row['text'] for row in request['sources']) for quote in quotes):
                    continue
                anchors = []
                for quote in quotes:
                    matching = [row for row in request['sources'] if quote in row['text']]
                    assert len(matching) == 1
                    anchors.append({'source': matching[0]['source'], 'quote': quote})
                memories.append({'point': point + '；' + SUMMARY_MARKER + '将来要重新明确', 'anchors': anchors})
            value = {'context': CONTEXT_MARKER, 'memories': memories}
        elif kind == ASSEMBLY:
            rows = request['l1_contract']['source_texts']
            # Output labels come from the LOCAL fixture table matched against
            # selected quotes, never from the selector's free summary/request.
            quotes = sorted(row['text'] for row in rows)
            matches = [point for point, group in groups if sorted(group) == quotes]
            assert len(matches) == 1
            point = matches[0]
            body = f'那次我们说起{point}；我记得原话是“' + '；'.join(row['text'] for row in rows) + '”。'
            statements = [{'statement_id': f's{index + 1}', 'content': row['text'], 'subject_ref': None,
                'classification': {'kind': 'experience', 'stability': 'episode', 'modality': 'assertion',
                    'basis': 'reported' if '主办方回复我' in row['text'] else 'direct_expression'},
                'time': None, 'conditions': None, 'evidence_refs': [row['binding_id']]}
                for index, row in enumerate(rows)]
            refs = [row['binding_id'] for row in rows]
            doc = {'format': 'semantic-v1', 'narrative': {'title': point, 'body': body}, 'entities': [],
                'statements': statements, 'retrieval': {'keywords': [row['text'] for row in rows]},
                'scenes': [{'title': point, 'statement_refs': [row['statement_id'] for row in statements],
                    'source_ranges': [{'source_ref': row['binding_id'], **row['input_range']} for row in rows]}]}
            value = {'candidates': [{'kind': 'key_event', 'confidence': .99, 'source_refs': refs,
                                    'reason': '这次有具体经过和结果。', 'document': doc}]}
        elif kind == 'l2':
            refs = [row['statement_id'] for row in request['l1_document']['statements']]
            value = {'full_injection_text': request['l1_document']['narrative']['body'],
                'coherent_sections': [{'section_id': 'event', 'text': '保留这件事的完整前后变化。', 'statement_refs': refs}],
                'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
        else:
            raise AssertionError('event_first used the legacy model input')
        return NS(choices=[NS(message=NS(content=json.dumps(value, ensure_ascii=False)), finish_reason='stop')])

    engine.model = 'offline-event-fixture'
    engine.client = NS(base_url='https://fixture.invalid/v1', chat=NS(completions=NS(create=create)))
    def options(**kwargs):
        assert kwargs['max_retries'] == 0
        return engine.client
    engine.client.with_options = options
    return NS(engine=engine, service=service, window=window, events=events, calls=calls, control=control)


def extract(h, *, force=False):
    return asyncio.run(h.engine._extract_daily_chat_memory_candidates(DAY, h.window,
        mode='review', max_candidates=10, audit={}, force=force))


def run_day(h, *, force=False):
    return asyncio.run(h.engine.run_daily_chat_memory(h.service.buckets,
        conversation_turn_store=FakeTurns([]), raw_event_store=FakeRawEvents(h.events),
        key=DAY, mode='review', force=force, chat_windows_only=True))


def kinds(h):
    return [call['kind'] for call in h.calls]


def approve_and_l2(h, ident):
    # This reader represents the SAME currently available synthetic raw events.
    # It is not an eligibility shortcut based only on a retained historical copy.
    current = [source_snapshot(source, config=h.engine.config) for source in
               daily_sources(h.window, profile_id=h.engine.memory_semantic_profile_id)]
    class Reader:
        async def locate(self, binding):
            snapshot, = [row for row in current if row['source_namespace'] == binding['source_namespace']
                and row['source_id'] == binding['source_id'] and row['source_revision'] == binding['source_revision']]
            for anchor in binding['anchors']:
                verify_text_anchor(snapshot['text'], anchor, **{key: anchor[key] for key in
                    ('target_namespace', 'target_id', 'target_revision', 'text_view_id')})
            return {**deepcopy(snapshot), 'binding': binding, 'state': 'verified_snapshot', 'eligible_source': True}
    h.engine._memory_l1_sources = lambda: Reader()
    row = h.service.authority.get_candidate(ident)
    assert row['status'] == 'pending'
    adopted = asyncio.run(h.engine._memory_l1_adopt_candidate(row, h.service.buckets,
        request_id='owner-accept-' + ident, owner=True))
    assert adopted['status'] == 'created'
    assert h.service.authority.get_candidate(ident)['status'] == 'committed'
    l2 = MemoryL2Service(h.engine, h.service)
    l2.prepare(ident, {'expected_l1_revision': 1, 'request_id': 'l2-' + ident})
    result = asyncio.run(l2.run_once(1))
    assert result['processed'] == 1 and result['results'][0]['state'] == 'ready', result
    return next(call['payload'] for call in reversed(h.calls) if call['kind'] == 'l2')


def test_public_runner_empty_selection_completes_cursor_without_l1(tmp_path):
    h = harness(tmp_path, [('user', '早上吃了普通早餐，今天照常上课。'), ('assistant', '嗯嗯，知道啦。')], [])
    result = run_day(h)
    assert result['status'] == 'zero_candidates' and result['cursor_updated'], result
    assert result['last_raw_event_id'] == h.events[-1]['id']
    assert kinds(h) == [SELECTION]
    assert not h.service.authority.list_candidates(status='all') and not h.service.authority.list_memories()
    assert run_day(h)['reason'] == 'no_new_raw_events'
    assert kinds(h) == [SELECTION]


def test_auto_run_acknowledges_a_full_candidate_routed_to_owner_review(tmp_path):
    text = '我们约好周六一起看展览，具体在美术馆南门见。'
    h = harness(tmp_path, [('user', text)], [('周六约定', [text])])
    def run():
        return asyncio.run(h.engine.run_daily_chat_memory(h.service.buckets,
            conversation_turn_store=FakeTurns([]), raw_event_store=FakeRawEvents(h.events),
            key=DAY, mode='auto', chat_windows_only=True))
    result = run()
    assert result['status'] == 'pending' and result['pending_count'] == 1 and result['cursor_updated']
    assert run()['reason'] == 'no_new_raw_events' and kinds(h) == [SELECTION, ASSEMBLY]
    candidate, = h.service.authority.list_candidates(status='all')
    assert candidate['status'] == 'pending' and candidate['policy']['action'] == 'queue_review'
    assert not h.service.authority.list_memories()


def test_public_runner_two_events_review_queue_then_owner_admission_and_l2(tmp_path):
    receipt = '主办方回复我：照片已经收齐，不用重新上传。'
    changed = '后来我把约定改为周六上午，周五不去了。'
    texts = [('assistant', '嗯嗯，知道啦。'), ('user', receipt), ('user', changed)]
    groups = [('主办方确认照片收齐且无需重传', [receipt]), ('约定从周五改到周六', [changed])]
    h = harness(tmp_path, texts, groups)
    result = run_day(h)
    assert result['status'] == 'pending' and result['cursor_updated'], result
    assert kinds(h) == [SELECTION, ASSEMBLY, ASSEMBLY]
    assert [row['text'] for row in h.calls[0]['payload']['sources']] == [text for _, text in texts]
    assert len(result['candidates']) == 2 and all(row['status'] == 'pending' for row in result['candidates'])
    assert len(h.service.authority.list_candidates(status='pending')) == 2
    assert not h.service.authority.list_memories()
    for call, (point, quotes) in zip(h.calls[1:], groups):
        rows = call['payload']['l1_contract']['source_texts']
        assert call['payload']['selection_focus'] == {'source_refs': [row['binding_id'] for row in rows], 'evidence': False}
        assert [row['text'] for row in rows] == quotes
        assert SUMMARY_MARKER not in json.dumps(call['payload'], ensure_ascii=False)
    assert [row['title'] for row in result['candidates']] == [point for point, _ in groups]
    first_id = result['candidates'][0]['id']
    approve_and_l2(h, first_id)
    assert len(h.service.authority.list_memories()) == 1
    assert len(h.service.authority.list_candidates(status='pending')) == 1
    before = len(h.calls)
    assert run_day(h)['reason'] == 'no_new_raw_events'
    assert len(h.calls) == before


def test_isolated_assistant_claim_is_not_mistaken_for_a_verified_tool_receipt(tmp_path):
    claim = '工具回执：草稿已导出为 plan.md，状态 completed。'
    h = harness(tmp_path, [('user', '嗯嗯，知道啦。'), ('assistant', claim)], [('草稿导出声称', [claim])])
    audits = []
    original = h.engine._normalize_daily_chat_memory_candidates
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        audits.append(deepcopy(kwargs.get('audit', {})))
        return result
    h.engine._normalize_daily_chat_memory_candidates = observe
    result = run_day(h)
    assert kinds(h) == [SELECTION, ASSEMBLY]
    assert result['status'] == 'zero_candidates' and result['cursor_updated'], result
    assert not h.service.authority.list_candidates(status='all') and not h.service.authority.list_memories()
    assert sum(audit.get('hard_rejects', {}).get('candidate_type_not_supported_by_source', 0)
               for audit in audits) == 1, audits


def test_partial_message_selection_stays_bounded_through_real_review_adoption_and_l2(tmp_path):
    text = '__未选前缀__先收货😀。__未选中间__后来改期🐈。__未选后缀__'
    h = harness(tmp_path, [('user', text)], [('收货后再改期这件事', ['先收货😀。', '后来改期🐈。'])])
    result = run_day(h)
    assert result['status'] == 'pending' and len(result['candidates']) == 1, result
    assembly = h.calls[1]['payload']
    assert [row['text'] for row in assembly['l1_contract']['source_texts']] == ['先收货😀。', '后来改期🐈。']
    assert '__未选' not in json.dumps(assembly, ensure_ascii=False)
    candidate = h.service.authority.get_candidate(result['candidates'][0]['id'])
    assert '__未选' not in candidate['proposal']['original_excerpt']
    bindings = candidate['proposal']['metadata']['memory_l1']['evidence_bindings']
    assert len(bindings) == 2 and bindings[0]['source_id'] == bindings[1]['source_id']
    assert all(MemoryL0Sources(h.service.authority).copy(binding)['text'] == text for binding in bindings)
    l2 = approve_and_l2(h, candidate['candidate_id'])
    assert '__未选' not in json.dumps(l2, ensure_ascii=False)
    assert all('text' not in source for source in l2['sources'])
    assert [row['text'] for source in l2['sources'] for row in source['read_ranges']] == ['先收货😀。', '后来改期🐈。']


def test_long_single_turn_uses_absolute_span_and_neighbour_context_cannot_write_twice(tmp_path):
    prefix = '普通背景😀。' * 18
    prefix += '甲' * (112 - len(prefix))
    quote = '后来约定改到周六😀。'
    text = prefix + quote + '后续闲聊。' * 60
    h = harness(tmp_path, [('user', text)], [('约定改到周六的完整事件', [quote])])
    h.engine.daily_chat_memory_window_max_input_chars = 256
    h.engine.config['reflection']['daily_chat_memory_window_max_input_chars'] = 256
    h.control['visible_groups_only'] = True
    result = run_day(h)
    assert result['status'] == 'pending' and result['cursor_updated'], result
    assert len(result['candidates']) == 1
    selectors = [call['payload'] for call in h.calls if call['kind'] == SELECTION]
    assert len(selectors) > 1
    # The same quote is offered once as core and once as the next unit's context.
    assert sum(any(quote in row['text'] for row in payload['sources']) for payload in selectors) == 2
    assert all(sum(len(row['text']) for row in payload['sources']) <= 256 for payload in selectors)
    assemblies = [call['payload'] for call in h.calls if call['kind'] == ASSEMBLY]
    assert len(assemblies) == 1
    view, = assemblies[0]['l1_contract']['source_texts']
    assert view['text'] == quote and view['input_range'] == {'start': 112, 'end': 112 + len(quote)}
    assert '普通背景' not in json.dumps(assemblies[0], ensure_ascii=False)
    assert '后续闲聊' not in json.dumps(assemblies[0], ensure_ascii=False)
    row = h.service.authority.get_candidate(result['candidates'][0]['id'])
    binding, = row['proposal']['metadata']['memory_l1']['evidence_bindings']
    assert binding['anchors'][0]['start'] == 112 and binding['anchors'][0]['end'] == 112 + len(quote)
    assert MemoryL0Sources(h.service.authority).copy(binding)['text'] == text
    assert len(h.service.authority.list_candidates(status='pending')) == 1
    before = len(h.calls)
    assert run_day(h)['reason'] == 'no_new_raw_events'
    assert len(h.calls) == before and not h.service.authority.list_memories()


def test_same_turn_cannot_borrow_unselected_owner_words_for_assistant_claim(tmp_path):
    owner = '我今天把课程笔记重新排好，下午去图书馆。'
    claim = '工具回执：草稿已导出为 plan.md，状态 completed。'
    h = harness(tmp_path, [('user', owner), ('assistant', claim)], [('助手声称已导出草稿', [claim])])
    # One actual turn holds BOTH messages, so whole-turn admission would borrow
    # the unrelated owner's words. Only the selected assistant passage qualifies.
    assert h.window[0]['user_text'] == owner and h.window[0]['assistant_text'] == claim
    audit = {}
    rows, meta = asyncio.run(h.engine._extract_daily_chat_memory_candidates(DAY, h.window,
        mode='review', max_candidates=10, audit=audit))
    assert not rows and not meta['partial'], meta
    assert kinds(h) == [SELECTION, ASSEMBLY]
    assembly = h.calls[-1]['payload']
    assert [row['text'] for row in assembly['l1_contract']['source_texts']] == [claim]
    assert owner not in json.dumps(assembly, ensure_ascii=False)
    assert audit['hard_rejects']['candidate_type_not_supported_by_source'] == 1
    assert not h.service.authority.list_candidates(status='all') and not h.service.authority.list_memories()


def test_one_model_call_budget_restarts_at_assembly_without_reselection_or_duplicate_write(tmp_path):
    texts = [('user', '我把约定改成周六上午，周五不去。')]
    groups = [('约定改到周六', [texts[0][1]])]
    h = harness(tmp_path, texts, groups, calls_per_run=1)
    first, first_meta = extract(h)
    assert first == [] and first_meta['partial'] and first_meta['last_processed_event_id'] == 0, first_meta
    assert kinds(h) == [SELECTION]
    resumed = harness(tmp_path, texts, groups, calls_per_run=1)
    # A default-mode change must not reselect or reinterpret an in-flight window.
    resumed.engine.config['reflection']['daily_chat_memory_extraction_mode'] = 'single_pass'
    rows, meta = extract(resumed)
    assert len(rows) == 1 and not meta['partial'], meta
    assert kinds(resumed) == [ASSEMBLY]
    written = asyncio.run(resumed.engine._write_daily_chat_memory_candidates_authority(rows, resumed.service.buckets))
    assert written['results'][0]['status'] == 'pending'
    queue = MemoryCandidatePreparation(resumed.service.authority)
    queue.acknowledge(rows[0]['preparation_key'], [rows[0]['id']])
    repeated, finished = extract(resumed)
    assert not repeated and not finished['partial'] and kinds(resumed) == [ASSEMBLY]
    assert len(resumed.service.authority.list_candidates(status='all')) == 1
    assert not resumed.service.authority.list_memories()


@pytest.mark.parametrize('stage', [SELECTION, ASSEMBLY])
def test_unknown_stage_outcome_does_not_dispatch_again_even_after_restart_force(tmp_path, stage):
    texts = [('user', '我把约定改到周六上午。')]
    groups = [('约定变化', [texts[0][1]])]
    h = harness(tmp_path, texts, groups)
    h.control['unknown'] = stage
    rows, meta = extract(h)
    assert not rows and meta['reason'] == 'model_outcome_unknown_requires_retry_decision', meta
    assert kinds(h) == ([SELECTION] if stage == SELECTION else [SELECTION, ASSEMBLY])
    resumed = harness(tmp_path, texts, groups)
    for force in (False, True):
        assert extract(resumed, force=force)[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    resumed.engine.temperature += .1
    assert extract(resumed, force=True)[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert not resumed.calls and not resumed.service.authority.list_candidates(status='all')


@pytest.mark.parametrize('stage', [SELECTION, ASSEMBLY])
def test_known_invalid_response_force_retries_only_failed_step(tmp_path, stage):
    texts = [('user', '我把约定改到周六上午。')]
    groups = [('约定变化', [texts[0][1]])]
    h = harness(tmp_path, texts, groups)
    h.control['invalid'] = stage
    rows, meta = extract(h)
    assert not rows and meta['partial'] and meta['last_processed_event_id'] == 0, meta
    before = kinds(h)
    h.control['invalid'] = None
    assert not extract(h)[0] and kinds(h) == before
    repaired, recovered = extract(h, force=True)
    assert len(repaired) == 1 and not recovered['partial'], recovered
    assert kinds(h) == ([SELECTION, SELECTION, ASSEMBLY] if stage == SELECTION
                        else [SELECTION, ASSEMBLY, ASSEMBLY])
    assert len(h.service.authority.list_candidates(status='pending')) == 1
    assert not h.service.authority.list_memories()
