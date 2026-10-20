"""Real ReflectionEngine/native Schema/temporary Authority, no network."""
import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from memory_candidate_preparation import MemoryCandidatePreparation
from memory_chat_segments import ChatSegmentProgress, segment_plan
from memory_l0_sources import MemoryL0Sources
from memory_l1_contract import verify_text_anchor
from memory_source_provenance import daily_sources
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_l1_ingestion_p2c import setup

DAY = '2026-10-20'


def fixture(tmp_path, *, calls_per_run=1, mode='review'):
    service, _, _, _ = setup(tmp_path)
    config = reflection_config(tmp_path/'engine', mode=mode)
    config['reflection'].update(daily_chat_memory_window_max_input_chars=256,
        daily_chat_memory_max_windows_per_run=calls_per_run)
    engine = ReflectionEngine(config)
    engine.memory_authority_store = service.authority
    engine._memory_l1_revisions = lambda *args: service
    engine._memory_authority_service = lambda *args: service.commit
    calls = []
    control = {'fail': False, 'cancel': False, 'bad': False, 'count': 1}
    async def create(**kwargs):
        assert kwargs['response_format']['type'] == 'json_schema'
        assert kwargs['max_tokens'] == 32000  # Existing route budget is intact.
        control['system_message'] = kwargs['messages'][0]['content']
        payload = json.loads(kwargs['messages'][-1]['content'])['l1_contract']
        calls.append(payload)
        if control['fail']: raise TimeoutError('synthetic unknown outcome')
        if control['cancel']: raise asyncio.CancelledError()
        segment = payload['segment']
        core = segment['core_range']
        ref = segment['core_source']
        candidates = []
        for n in range(control['count']):
            body = f"她向我说明了第{core['start']}段的第{n}个具体偏好，希望我记住这种选择。"
            doc = {'format': 'semantic-v1', 'narrative': {'title': '具体偏好', 'body': body},
                'entities': [], 'statements': [{'content': body, 'subject_ref': None,
                    'classification': {'kind': 'preference', 'stability': None, 'modality': None, 'basis': None},
                    'time': None, 'conditions': None, 'evidence_refs': [ref]}],
                'scenes': [{'title': '偏好来源', 'statement_refs': ['s1'],
                    'source_ranges': [{'source_ref': ref, **core}]}],
                'retrieval': {'keywords': ['偏好', str(core['start'])]}}
            if control['bad']:
                doc['scenes'][0]['source_ranges'][0]['end'] = 999999
            candidates.append({'kind': 'stable_preference', 'confidence': 0.99,
                'reason': '来源明确说明偏好', 'source_refs': [ref], 'document': doc})
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates': candidates}, ensure_ascii=False)),
                              finish_reason='stop')])
    engine.model = 'offline-fixture'
    def client_options(**kwargs):
        assert kwargs['max_retries'] == 0
        return engine.client
    engine.client = NS(base_url='https://fixture.invalid/v1', chat=NS(completions=NS(create=create)),
                       with_options=client_options)
    text = ('我不喜欢苦咖啡，喜欢清淡茶饮🙂。' * 40)
    window = [{'id': 1, 'session_id': 'session', 'created_at': '2026-10-20T01:00:00Z',
        'user_text': text, 'assistant_text': '', 'raw_event_ids': [11],
        '_semantic_sources': [{'id': 11, 'role': 'user', 'text': text, 'session_id': 'session',
                               'created_at': '2026-10-20T01:00:00Z'}]}]
    return engine, service, window, calls, control


def extract(engine, window, budget=10):
    return asyncio.run(engine._extract_daily_chat_memory_candidates(DAY, window,
        mode=engine.daily_chat_memory_mode, max_candidates=budget, audit={}))


def acknowledge(engine, candidates):
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    for item in candidates:
        queue.acknowledge(item['preparation_key'], [item['id']])


def test_unicode_plan_and_views_are_bounded_without_mutating_raw(tmp_path):
    e, s, window, calls, _ = fixture(tmp_path)
    original = copy.deepcopy(window)
    candidates, meta = extract(e, window)
    assert candidates and meta['partial'], meta
    assert window == original
    source = daily_sources(window, profile_id=e.memory_semantic_profile_id)[0]
    plan = segment_plan([source], 256)
    assert ''.join(source['text'][p['core_range']['start']:p['core_range']['end']] for p in plan) == source['text']
    assert sum(len(x['text']) for x in calls[0]['source_texts']) <= 256
    binding = candidates[0]['memory_l1']['evidence_bindings'][0]
    full_copy = MemoryL0Sources(s.authority).copy(binding)
    assert full_copy['text'] == source['text']
    assert calls[0]['source_texts'][0]['original_text_sha256'] == hashlib.sha256(source['text'].encode()).hexdigest()
    anchor = candidates[0]['memory_l1']['scenes'][0]['anchors'][0]
    verify_text_anchor(source['text'], anchor, **{k: anchor[k] for k in
        ('target_namespace', 'target_id', 'target_revision', 'text_view_id')})


def test_restart_coverage_no_repeat_and_final_receipt(tmp_path):
    e, s, window, calls, _ = fixture(tmp_path)
    first, meta = extract(e, window)
    acknowledge(e, first)
    e2, s2, _, calls2, _ = fixture(tmp_path)
    for _ in range(20):
        candidates, meta = extract(e2, window)
        acknowledge(e2, candidates)
        if not meta['partial']: break
    assert not meta['partial'] and meta['last_processed_event_id'] == 11, meta
    all_calls = calls+calls2
    cores = [p['segment']['core_range'] for p in all_calls]
    assert len(cores) == len({r['start'] for r in cores})
    assert cores[0]['start'] == 0 and cores[-1]['end'] == len(window[0]['user_text'])
    assert all(a['end'] == b['start'] for a,b in zip(cores, cores[1:]))
    before = len(calls2)
    assert extract(e2, window)[1]['partial'] is False
    assert len(calls2) == before
    record = ChatSegmentProgress(s2.authority).get(meta['segment_progress']['task_id'])
    assert record['progress']['phase'] == 'complete'


def test_unknown_timeout_blocks_restart_and_changed_prompt(tmp_path):
    e, s, window, calls, state = fixture(tmp_path)
    state['fail'] = True
    assert extract(e, window)[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    e2, _, _, calls2, _ = fixture(tmp_path)
    assert extract(e2, window)[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    forced = asyncio.run(e2._extract_daily_chat_memory_candidates(DAY, window,
        mode='review', max_candidates=10, force=True))
    assert forced[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    # A changed route/prompt cannot silently replay the same uncertain source.
    e2.temperature += .1
    assert extract(e2, window)[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert len(calls) == 1 and not calls2


def test_cancel_keeps_unknown_intent(tmp_path):
    e, _, window, calls, state = fixture(tmp_path)
    state['cancel'] = True
    with pytest.raises(asyncio.CancelledError): extract(e, window)
    state['cancel'] = False
    assert extract(e, window)[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert len(calls) == 1


def test_response_saved_before_prepare_crash_reused(tmp_path, monkeypatch):
    e, s, window, calls, _ = fixture(tmp_path)
    original = MemoryCandidatePreparation.prepare
    def fail(*a, **kw): raise RuntimeError('crash before preparation')
    monkeypatch.setattr(MemoryCandidatePreparation, 'prepare', fail)
    assert extract(e, window)[1]['partial']
    monkeypatch.setattr(MemoryCandidatePreparation, 'prepare', original)
    e2, _, _, calls2, _ = fixture(tmp_path)
    candidates, meta = extract(e2, window, budget=1)
    assert candidates and not calls2 and len(calls) == 1
    assert meta['partial']


def test_prepared_before_cursor_crash_reused(tmp_path, monkeypatch):
    e, s, window, calls, _ = fixture(tmp_path)
    original = ChatSegmentProgress.save
    def fail(self, task, scope, value, revision, **kw):
        if value['phase'] == 'ready' and value['cursor'] == 1:
            raise RuntimeError('crash after preparation')
        return original(self, task, scope, value, revision, **kw)
    monkeypatch.setattr(ChatSegmentProgress, 'save', fail)
    assert extract(e, window)[1]['partial']
    monkeypatch.setattr(ChatSegmentProgress, 'save', original)
    e2, _, _, calls2, _ = fixture(tmp_path)
    candidates, meta = extract(e2, window, budget=1)
    assert len(candidates) == 1 and not calls2 and len(calls) == 1


def test_candidate_quota_drains_saved_results_without_rebilling(tmp_path):
    e, _, window, calls, state = fixture(tmp_path)
    state['count'] = 3
    ids = []
    for _ in range(3):
        rows, meta = extract(e, window, budget=1)
        assert len(rows) == 1 and meta['partial'], meta
        ids.append(rows[0]['id'])
        acknowledge(e, rows)
    assert len(set(ids)) == 3 and len(calls) == 1
    extract(e, window, budget=1)
    assert len(calls) == 2


def test_changed_original_does_not_reuse_old_cursor(tmp_path):
    e, _, window, calls, _ = fixture(tmp_path)
    rows, a = extract(e, window)
    acknowledge(e, rows)
    window[0]['_semantic_sources'][0]['text'] += '新增的原文🙂'
    rows, b = extract(e, window)
    assert a['segment_progress']['task_id'] != b['segment_progress']['task_id']
    assert calls[-1]['segment']['core_range']['start'] == 0


def test_invalid_range_is_not_consumed_or_resent(tmp_path):
    e, _, window, calls, state = fixture(tmp_path)
    state['bad'] = True
    rows, meta = extract(e, window)
    assert rows == [] and meta['reason'] == 'invalid_l1_document'
    assert meta['last_processed_event_id'] == 0
    extract(e, window)
    assert len(calls) == 1


def test_sparse_source_labels_and_original_coordinates(tmp_path):
    e, s, window, calls, _ = fixture(tmp_path)
    long_source = window[0]['_semantic_sources'][0]
    window[0]['_semantic_sources'] = [
        {**long_source, 'id': 10, 'text': '无关短语。'}, long_source,
        {**long_source, 'id': 12, 'text': '另一条无关短语。'}]
    first, _ = extract(e, window)
    acknowledge(e, first)
    rows, meta = extract(e, window)
    assert calls[-1]['segment']['core_source'] == 'source:2'
    assert rows, meta
    bindings = rows[0]['memory_l1']['evidence_bindings']
    assert [b['binding_id'] for b in bindings] == ['source:2']
    assert MemoryL0Sources(s.authority).copy(bindings[0])['text'] == long_source['text']


def test_concurrent_engines_dispatch_only_once(tmp_path):
    e, s, window, calls, _ = fixture(tmp_path)
    e2, _, _, calls2, _ = fixture(tmp_path)
    original = e.client.chat.completions.create
    async def race():
        started, release = asyncio.Event(), asyncio.Event()
        async def blocked(**kwargs):
            started.set()
            await release.wait()
            return await original(**kwargs)
        e.client.chat.completions.create = blocked
        first = asyncio.create_task(e._extract_daily_chat_memory_candidates(
            DAY, window, mode='review', max_candidates=10))
        await started.wait()
        second = await e2._extract_daily_chat_memory_candidates(DAY, window, mode='review', max_candidates=10)
        release.set()
        return await first, second
    first, second = asyncio.run(race())
    assert first[0]
    assert second[1]['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert len(calls) == 1 and not calls2


@pytest.mark.parametrize('action', ['reject', 'defer', 'accept'])
def test_owner_decision_is_preserved_on_resume(tmp_path, action):
    e, s, window, calls, _ = fixture(tmp_path)
    rows, _ = extract(e, window)
    ident = rows[0]['id']
    row = s.authority.get_candidate(ident)
    decided = s.authority.decide_candidate(ident, action=action, expected_revision=row['revision'],
        request_id='owner-'+action, actor='owner')
    extract(e, window)
    current = s.authority.get_candidate(ident)
    assert current['revision'] == decided['revision']
    assert current['status'] == decided['status']
    assert [p['segment']['core_range']['start'] for p in calls] == [0, 128]


def test_owner_edit_and_review_writer_use_original_candidate(tmp_path):
    from dataclasses import replace
    from memory_authority import MemoryProposal
    e, s, window, calls, _ = fixture(tmp_path)
    rows, _ = extract(e, window)
    row = s.authority.get_candidate(rows[0]['id'])
    proposal = replace(MemoryProposal.from_mapping(row['proposal']), proposed_body='主人保留的手改稿。')
    s.authority.revise_candidate(row['candidate_id'], expected_revision=row['revision'],
        proposal=proposal, request_id='owner-edit', actor='owner')
    resumed, _ = extract(e, window, budget=1)
    assert resumed[0]['content'] == '主人保留的手改稿。'
    result = asyncio.run(e._write_daily_chat_memory_candidates_authority(resumed, s.buckets))
    assert result['results'][0]['status'] == 'pending'
    assert len(calls) == 1


def test_auto_preparation_and_real_bucket_commit(tmp_path):
    e, s, window, calls, _ = fixture(tmp_path, mode='auto')
    rows, meta = extract(e, window)
    assert rows, meta
    row = s.authority.get_candidate(rows[0]['id'])
    assert row['status'] == 'accepted'
    # An authorized synthetic reader of the same raw source, not an L0-copy
    # fallback: adoption still checks source availability via its real API.
    class Reader:
        async def locate(self, binding):
            return {'state': 'available', 'eligible_source': True}
    e._memory_l1_sources = lambda: Reader()
    result = asyncio.run(e._write_daily_chat_memory_candidates_authority(rows, s.buckets))
    assert result['created'] == 1 and result['failed'] == 0, result
    committed = s.authority.get_memory(rows[0]['id'])
    assert committed['active_revision'] == 1
    assert s.authority.get_candidate(rows[0]['id'])['status'] == 'committed'
    replay = asyncio.run(e._write_daily_chat_memory_candidates_authority(rows, s.buckets))
    assert replay['exists'] == 1 and len(calls) == 1


def test_fully_empty_segments_still_save_complete_receipts(tmp_path):
    e, s, window, calls, state = fixture(tmp_path, calls_per_run=100)
    state['count'] = 0
    rows, meta = extract(e, window)
    assert rows == [] and not meta['partial'] and meta['last_processed_event_id'] == 11
    count = len(calls)
    extract(e, window)
    assert len(calls) == count
    record = ChatSegmentProgress(s.authority).get(meta['segment_progress']['task_id'])
    assert len(record['progress']['receipts']) == count


def test_mode_switch_preserves_partial_review_without_regeneration(tmp_path):
    e, s, window, calls, _ = fixture(tmp_path)
    rows, a = extract(e, window)
    acknowledge(e, rows)
    e.daily_chat_memory_mode = 'auto'
    _, b = extract(e, window)
    assert a['segment_progress']['task_id'] == b['segment_progress']['task_id']
    assert [p['segment']['core_range']['start'] for p in calls] == [0, 128]
    assert s.authority.get_candidate(rows[0]['id'])['status'] == 'pending'


def test_daily_runner_restarts_then_advances_raw_watermark_only_at_tail(tmp_path):
    from test_daily_chat_memory_quality_review_v4 import FakeRawEvents, FakeTurns
    e, s, window, calls, _ = fixture(tmp_path)
    e.daily_chat_memory_turn_limit = 0
    events = copy.deepcopy(window[0]['_semantic_sources'])
    async def batch(engine, service):
        result = await engine._run_daily_chat_memory_impl(service.buckets,
            conversation_turn_store=FakeTurns([]), raw_event_store=FakeRawEvents(events),
            key=DAY, mode='review')
        return engine._ack_daily_chat_memory_preparation(result)
    first = asyncio.run(batch(e, s))
    assert first['status'] == 'partial' and not first['cursor_updated'], first
    assert first['candidates'][0]['status'] == 'pending'
    e2, s2, _, calls2, _ = fixture(tmp_path)
    e2.daily_chat_memory_turn_limit = 0
    for _ in range(20):
        result = asyncio.run(batch(e2, s2))
        if result['status'] != 'partial': break
        assert not result['cursor_updated']
    assert result['cursor_updated'] and result['last_raw_event_id'] == 11, result
    assert len(calls)+len(calls2) == len(segment_plan(daily_sources(window,
        profile_id=e.memory_semantic_profile_id), 256))
    count = len(calls2)
    assert asyncio.run(batch(e2, s2))['reason'] == 'no_new_raw_events'
    assert len(calls2) == count


def test_progress_cas_failure_preserves_original_receipt(tmp_path):
    from memory_authority import RevisionConflict
    e, s, _, _, _ = fixture(tmp_path)
    store = ChatSegmentProgress(s.authority)
    initial = {'cursor': 0, 'phase': 'ready', 'receipts': []}
    store.save('synthetic-task', 'synthetic-source', initial, 0)
    with pytest.raises(RevisionConflict):
        store.save('synthetic-task', 'synthetic-source',
                   {'cursor': 99, 'phase': 'complete', 'receipts': []}, 0)
    assert store.get('synthetic-task') == {'revision': 1, 'progress': initial}


def test_native_segment_context_cannot_create_unrelated_candidate(tmp_path):
    e, _, window, calls, _ = fixture(tmp_path)
    original = e.client.chat.completions.create
    async def unrelated(**kwargs):
        response = await original(**kwargs)
        parsed = json.loads(response.choices[0].message.content)
        candidate = parsed['candidates'][0]
        candidate['source_refs'] = ['source:2']
        candidate['document']['statements'][0]['evidence_refs'] = ['source:2']
        candidate['document']['scenes'] = []
        response.choices[0].message.content = json.dumps(parsed)
        return response
    e.client.chat.completions.create = unrelated
    source = window[0]['_semantic_sources'][0]
    window[0]['_semantic_sources'].append({**source, 'id': 12, 'text': '仅相邻语境。'})
    rows, meta = extract(e, window)
    assert rows == [] and meta['reason'] == 'invalid_l1_document'
    assert meta['last_processed_event_id'] == 0 and len(calls) == 1


def test_segment_path_honors_owner_prompt_wrapper_and_off(tmp_path):
    e, _, window, calls, state = fixture(tmp_path)
    def bind(mode, revision):
        block = {'block_id': 'daily-block', 'source_id': 'ombre.daily_chat_memory_prompt',
            'scope': 'memory.daily_chat_review', 'stage': 'ombre_post_injection', 'mode': mode,
            'role': 'system', 'lane': 'instruction', 'anchor': 'instructions.end', 'order': 0,
            'priority': 1, 'enabled': True, 'owner_body': 'B主人提取规则{max_candidates}', 'wrapper_text': 'A\n{{content}}\nC'}
        digest = str(revision)*64
        e.prompt_plan_mirror.put_plan('segmented-test', revision, plan_sha256=digest,
            gateway_slice={'blocks': [block]})
        e.prompt_plan_mirror.put_binding('memory.daily_chat_review', preset_id='segmented-test',
            preset_revision=revision, plan_sha256=digest, aiz_binding_revision=revision)
    bind('off', 1)
    rows, meta = extract(e, window)
    assert rows == [] and meta['reason'] == 'prompt_disabled' and not calls
    bind('owner_override', 2)
    rows, meta = extract(e, window)
    assert rows and state['system_message'] == f'A\nB主人提取规则{e.daily_chat_memory_max_per_day}\nC'
    assert calls[0]['segment']['candidate_requires_core_support'] is True
