"""Offline: generated drafts survive budgets in the original Authority."""
import asyncio
import copy
import json

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryProposal
from memory_candidate_preparation import MemoryCandidatePreparation, PREFIX
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config, proposal
from test_memory_daily_windows import raw_day
from test_daily_chat_memory_quality_review_v4 import FakeRawEvents, FakeTurns
from test_memory_l1_action_tasks_p2c import FakeCompletionClient, setup as action_setup


def run_day(engine, events, mode='review', buckets=None, force=False):
    # Exercise one real persistence/cursor batch; the public worker may drain
    # these batches continuously under the same task and concurrency lock.
    async def batch():
        result = await engine._run_daily_chat_memory_impl(buckets,
            raw_event_store=FakeRawEvents(events), conversation_turn_store=FakeTurns([]),
            key='2026-08-18', mode=mode, force=force)
        return engine._ack_daily_chat_memory_preparation(result)
    return asyncio.run(batch())


def generated(index, *, long=True):
    body = f'主人约定今后和我一起完成第{index}件散步计划。' * (250 if long else 1)
    return {'kind': 'commitment', 'confidence': 0.95, 'source_refs': [f'source:{index}'],
        'reason': '原文中的约定', 'document': {'format': 'semantic-v1',
            'narrative': {'title': f'散步计划{index}', 'body': body},
            'entities': [], 'retrieval': {'keywords': ['散步', f'计划{index}']},
            'statements': [{'content': f'主人约定完成第{index}件散步计划。', 'subject_ref': None,
                'classification': {'kind': 'experience', 'stability': 'episode', 'modality': 'assertion', 'basis': 'record'},
                'time': None, 'conditions': None, 'evidence_refs': [f'source:{index}']}]}}


def daily(tmp_path, mode='review'):
    config = reflection_config(tmp_path, mode=mode)
    config['reflection'].update(daily_chat_memory_max_per_day=1, daily_chat_memory_review_max_per_day=1,
        daily_chat_memory_summary_enabled=False, action_l1_jobs_per_run=0)
    engine = ReflectionEngine(config)
    client = FakeCompletionClient(json.dumps({'candidates': [generated(1), generated(2)]}, ensure_ascii=False))
    engine.config['reflection']['memory_task_routes'] = {'auto_l1': {'connection': 'daily', 'model': 'offline-l1'}}
    engine.daily_chat_memory_client = client
    class Sources:
        async def locate(self, binding): return {'eligible_source': True, 'state': 'verified_snapshot'}
    engine._memory_l1_sources = lambda: Sources()
    return engine, client, BucketManager(config)


def test_daily_review_overflow_keeps_complete_l1_and_resumes_without_generation(tmp_path):
    engine, client, buckets = daily(tmp_path)
    first = run_day(engine, raw_day(2), buckets=buckets)
    assert first['status'] == 'partial' and first['reason'] == 'candidate_budget_exhausted_prepared'
    assert len(first['candidates']) == 1 and first['last_raw_event_id'] == 0
    assert first['preparation_batches'][0] == {'generated': 2, 'validated': 2, 'queued': 1,
        'processed': 1, 'adopted': 0, 'rejected': 0, 'deferred': 0}
    rows = engine.memory_authority_store.list_candidates(status='all')
    assert len(rows) == 2 and all(r['status'] == 'pending' for r in rows)
    assert all(len(r['proposal']['proposed_body']) > 4000 for r in rows)
    assert all(r['proposal']['metadata']['memory_l1']['statements'] for r in rows)
    assert len(engine.list_daily_chat_memory_pending(status='all')) == 2
    assert engine.memory_authority_store.list_memories() == []
    second = run_day(engine, raw_day(2), buckets=buckets)
    assert second['status'] == 'pending' and second['last_raw_event_id'] == 2
    assert len(second['candidates']) == 1 and len(client.calls) == 1
    assert len(engine.memory_authority_store.list_candidates(status='all')) == 2
    assert run_day(engine, raw_day(2), buckets=buckets)['reason'] == 'no_new_raw_events'
    assert len(client.calls) == 1


def test_daily_auto_only_adopts_budgeted_share_and_reuses_original_candidates(tmp_path):
    engine, client, buckets = daily(tmp_path, 'auto')
    first = run_day(engine, raw_day(2), mode='auto', buckets=buckets)
    assert first['created'] == 1 and first['preparation_batches'][0]['queued'] == 1
    assert len(engine.memory_authority_store.list_memories()) == 1
    second = run_day(engine, raw_day(2), mode='auto', buckets=buckets)
    assert second['created'] == 1 and len(client.calls) == 1
    assert second['preparation_batches'][0]['adopted'] == 2
    assert len(engine.memory_authority_store.list_memories()) == 2
    assert all(r['status'] == 'committed' for r in engine.memory_authority_store.list_candidates(status='all'))


def test_second_window_overflow_reuses_after_impl_rereads_from_advanced_cursor(tmp_path):
    engine, client, buckets = daily(tmp_path)
    engine.daily_chat_memory_max_per_day = engine.daily_chat_memory_review_max_per_day = 2
    engine.daily_chat_memory_window_turns = 4
    engine.daily_chat_memory_window_stride_turns = 2
    async def create(**kwargs):
        client.calls.append(copy.deepcopy(kwargs))
        payload = json.loads(kwargs['messages'][-1]['content'])
        first = payload['l1_contract']['source_texts'][0]['text']
        selected = [generated(1)] if '第1件事' in first else [generated(1), generated(2)]
        # Distinct first-source IDs and grounding are bound by the real engine.
        for item in selected: item['document']['narrative']['body'] += first
        from types import SimpleNamespace as NS
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates': selected}, ensure_ascii=False)), finish_reason='stop')])
    client.create = create
    first = run_day(engine, raw_day(8), buckets=buckets)
    assert first['last_raw_event_id'] == 2 and len(first['candidates']) == 2
    assert len(client.calls) == 2 and any(batch['queued'] == 1 for batch in first['preparation_batches'])
    # Exactly one further window may be generated; the retained second window
    # must be read by ID from Authority even though its run index changed 2→1.
    second = run_day(engine, raw_day(8), buckets=buckets)
    assert len(second['candidates']) == 2 and second['last_raw_event_id'] == 4
    assert len(client.calls) == 3
    assert second['preparation_batches'][0]['queued'] in (0, 1)
    assert engine.list_daily_chat_memory_run_audit()[-1]['prepared_reused_count'] == 1


@pytest.mark.parametrize('decision', ['edit', 'reject', 'defer'])
def test_daily_continuation_keeps_owner_edit_or_decision(tmp_path, decision):
    engine, client, buckets = daily(tmp_path)
    first = run_day(engine, raw_day(2), buckets=buckets)
    key = first['candidates'][0]['preparation_key']
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    row = queue.remaining(queue.get(key))[0]
    ident = row['candidate_id']
    if decision == 'edit':
        payload = copy.deepcopy(row['proposal'])
        payload['proposed_body'] += '主人补充了审核备注。'
        edited = engine.memory_authority_store.revise_candidate(ident, expected_revision=row['revision'],
            proposal=MemoryProposal.from_mapping(payload), request_id='owner-edit', actor='owner')
    else:
        edited = engine.memory_authority_store.decide_candidate(ident, action=decision,
            expected_revision=row['revision'], request_id='owner-choice', actor='owner')
    second = run_day(engine, raw_day(2), buckets=buckets)
    assert second['last_raw_event_id'] == 2 and len(client.calls) == 1
    current = engine.memory_authority_store.get_candidate(ident)
    assert current == edited
    if decision == 'edit': assert second['candidates'][0]['content'] == edited['proposal']['proposed_body']
    else: assert queue.counts(queue.get(key))[decision + ('ed' if decision == 'reject' else 'red')] == 1


def test_changed_source_preserves_old_drafts_as_visible_stale(tmp_path):
    engine, client, buckets = daily(tmp_path)
    events = raw_day(2)
    first = run_day(engine, events, buckets=buckets)
    old_ids = {r['candidate_id'] for r in engine.memory_authority_store.list_candidates(status='all')}
    events[0]['text'] += '后来我更正了第一件事。'
    run_day(engine, events, buckets=buckets)
    assert len(client.calls) == 2
    old_rows = [engine._authority_candidate_legacy_item(engine.memory_authority_store.get_candidate(i)) for i in old_ids]
    assert all(row['preparation_status'] == 'stale_source' for row in old_rows)
    assert all(engine.memory_authority_store.get_candidate(i) for i in old_ids)


def test_candidate_batch_and_identifier_receipt_roll_back_together(tmp_path, monkeypatch):
    engine, _, _ = daily(tmp_path)
    store = engine.memory_authority_store
    queue = MemoryCandidatePreparation(store)
    original = store._put_candidate
    def fail_second(conn, draft, policy):
        if draft.proposal_id == 'second': raise RuntimeError('synthetic storage failure')
        return original(conn, draft, policy)
    monkeypatch.setattr(store, '_put_candidate', fail_second)
    with pytest.raises(RuntimeError):
        queue.prepare('offline-batch', [proposal(proposal_id='first'), proposal(proposal_id='second')],
            engine.memory_ingestion_policy, scope='offline', source_fingerprint='version-1')
    assert store.list_candidates(status='all') == [] and queue.get('offline-batch') is None


def test_action_overflow_saves_all_and_next_run_reuses_original_task(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = action_setup(tmp_path, monkeypatch, mode='auto')
    engine.daily_chat_memory_max_per_day = 1
    item = json.loads(engine.dehydration_client.content)['candidates'][0]
    other = copy.deepcopy(item)
    other['document']['narrative']['body'] += '这也是完整保留的另一份经历。'
    engine.dehydration_client.content = json.dumps({'candidates': [item, other]}, ensure_ascii=False)
    tasks.enqueue_action(payload)
    first = asyncio.run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert first['preparation_counts']['validated'] == 2 and first['preparation_counts']['queued'] == 1
    assert len(engine.memory_authority_store.list_memories()) == 1 and len(calls) == 1
    second = asyncio.run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert second['adopted'] is True and second['candidate_ids'] == first['candidate_ids']
    assert second['remaining_candidate_ids'] == [] and len(calls) == 1
    assert len(engine.memory_authority_store.list_memories()) == 2
    receipt = MemoryCandidatePreparation(engine.memory_authority_store).get(tasks.list()[0]['preparation_key'])
    encoded = json.dumps(receipt)
    assert 'proposed_body' not in encoded and 'document' not in encoded


def test_action_retained_draft_rechecks_source_without_generating_again(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = action_setup(tmp_path, monkeypatch, mode='auto')
    engine.daily_chat_memory_max_per_day = 1
    item = json.loads(engine.dehydration_client.content)['candidates'][0]
    other = copy.deepcopy(item)
    other['document']['narrative']['body'] += '另一份完整行动经历。'
    engine.dehydration_client.content = json.dumps({'candidates': [item, other]}, ensure_ascii=False)
    tasks.enqueue_action(payload)
    first = asyncio.run(engine._memory_l1_action_tasks(buckets))['results'][0]
    source.update(state='withdrawn', eligible_source=False)
    blocked = asyncio.run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert blocked['adopted'] is False and len(calls) == 1
    assert blocked['remaining_candidate_ids'] == first['remaining_candidate_ids']
    assert blocked['adoption_issues'][0]['code'] == 'source_not_currently_readable'
    assert len(engine.memory_authority_store.list_memories()) == 1


def test_stale_receipt_never_resumes_or_adopts_even_if_its_old_source_reappears(tmp_path):
    engine, client, buckets = daily(tmp_path)
    events = raw_day(2)
    first = run_day(engine, events, buckets=buckets)
    key = first['candidates'][0]['preparation_key']
    changed = copy.deepcopy(events)
    changed[0]['text'] += '来源版本已变。'
    run_day(engine, changed, buckets=buckets)
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    assert queue.remaining(queue.get(key)) == []
    reverted = run_day(engine, events, buckets=buckets)
    assert reverted['status'] == 'partial' and reverted['reason'] == 'source_preparation_stale'
    assert reverted['cursor_updated'] is False and len(client.calls) == 2


@pytest.mark.parametrize('invalid_kind', ['fact', '这段私人原文不属于分类标识。'])
def test_outer_kind_failure_keeps_the_original_candidate_position_and_safe_value(tmp_path, invalid_kind):
    from memory_l1_api import AUTO_KINDS

    engine, client, buckets = daily(tmp_path)
    good, bad = generated(1), generated(2)
    bad['kind'] = invalid_kind
    client.content = json.dumps({'candidates': [good, bad]}, ensure_ascii=False)
    result = run_day(engine, raw_day(2), buckets=buckets)
    assert result['status'] == 'partial' and len(result['candidates']) == 1
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    receipt = queue.get(result['candidates'][0]['preparation_key'])
    diagnostic = receipt['issue']['schema_error']
    assert diagnostic['field_path'] == 'candidates.1.kind'
    assert diagnostic['candidate_index'] == 1 and diagnostic['validator'] == 'enum'
    assert diagnostic['allowed_kinds'] == list(AUTO_KINDS)
    if invalid_kind == 'fact':
        assert diagnostic['actual_kind'] == invalid_kind
    else:
        assert diagnostic['actual_length'] == len(invalid_kind)
        assert invalid_kind not in json.dumps(diagnostic, ensure_ascii=False)
    assert good['document']['narrative']['body'] not in json.dumps(diagnostic, ensure_ascii=False)


def test_mixed_invalid_response_keeps_valid_l1_blocks_retry_and_force_can_correct(tmp_path):
    engine, client, buckets = daily(tmp_path)
    good, bad = generated(1), generated(2)
    del bad['document']['statements'][0]['classification']['basis']
    client.content = json.dumps({'candidates': [good, bad]}, ensure_ascii=False)
    first = run_day(engine, raw_day(2), buckets=buckets)
    assert first['status'] == 'partial' and first['reason'] == 'invalid_l1_document'
    assert first['last_raw_event_id'] == 0 and len(first['candidates']) == 1
    rows = engine.memory_authority_store.list_candidates(status='all')
    assert len(rows) == 1 and rows[0]['proposal']['proposed_body'] == good['document']['narrative']['body']
    key = first['candidates'][0]['preparation_key']
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    receipt = queue.get(key)
    assert receipt['issue']['schema_error']['code'] == 'native_l1_schema_mismatch'
    assert receipt['requires_manual_retry'] is True and receipt['validated_count'] == 1
    blocked = run_day(engine, raw_day(2), buckets=buckets)
    assert blocked['reason'] == 'invalid_l1_document' and len(client.calls) == 1
    client.content = json.dumps({'item_1': generated(2)}, ensure_ascii=False)
    corrected = run_day(engine, raw_day(2), buckets=buckets, force=True)
    assert corrected['status'] == 'pending' and corrected['last_raw_event_id'] == 2
    assert len(client.calls) == 2 and engine.memory_authority_store.count_candidates(status='all') == 2
    assert engine.memory_authority_store.get_candidate(rows[0]['candidate_id']) == rows[0]
    assert queue.get(key)['issue'] is None
    assert queue.rejected_drafts(queue.get(key)) == {}
    repair_input = json.loads(client.calls[-1]['messages'][-1]['content'])
    assert list(repair_input['candidate_repairs']) == ['item_1']


def test_category_retry_only_accepts_the_slot_and_keeps_the_entire_draft(tmp_path):
    engine, client, buckets = daily(tmp_path)
    good, original = generated(1), generated(2)
    original['kind'] = 'experience'
    client.content = json.dumps({'candidates': [good, original]}, ensure_ascii=False)
    first = run_day(engine, raw_day(2), buckets=buckets)
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    key = first['candidates'][0]['preparation_key']
    retained = engine.memory_authority_store.list_candidates(status='all')[0]
    assert queue.rejected_drafts(queue.get(key))['item_1']['kind_only']
    assert 'draft' not in queue.get(key)['issue']
    client.content = json.dumps({'item_1': 'fact'})
    assert first['reason'] == 'candidate_kind_retry_ready'
    failed = run_day(engine, raw_day(2), buckets=buckets)
    assert failed['status'] == 'partial' and not failed['cursor_updated']
    assert engine.daily_chat_memory_run_cursor()['last_raw_event_id'] == 0
    assert queue.rejected_drafts(queue.get(key))['item_1']['draft'] == original
    run_day(engine, raw_day(2), buckets=buckets)
    assert len(client.calls) == 2
    client.content = json.dumps({'item_1': 'commitment'})
    recovered = run_day(engine, raw_day(2), buckets=buckets, force=True)
    assert recovered['last_raw_event_id'] == 2 and recovered['status'] == 'pending'
    assert engine.memory_authority_store.get_candidate(retained['candidate_id']) == retained
    recovered_row = next(row for row in queue.rows(queue.get(key)) if row['candidate_id'] != retained['candidate_id'])
    assert recovered_row['proposal']['proposed_body'] == original['document']['narrative']['body']
    assert recovered_row['proposal']['metadata']['memory_l1']['statements']
    assert not engine.memory_authority_store.list_memories()


def test_pre_upgrade_receipt_recovers_only_missing_item_from_original_window(tmp_path):
    from memory_candidate_preparation import RETRY_PREFIX
    engine, client, buckets = daily(tmp_path)
    good, missing = generated(1), generated(2)
    client.content = json.dumps({'candidates': [good, {**missing, 'kind': 'fact'}]}, ensure_ascii=False)
    first = run_day(engine, raw_day(2), buckets=buckets)
    key = first['candidates'][0]['preparation_key']
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    receipt = queue.get(key)
    receipt.pop('kind_retry_ready')
    receipt.pop('kind_retry_attempted')
    engine.memory_authority_store.set_meta(PREFIX + key, receipt)
    engine.memory_authority_store.set_meta(RETRY_PREFIX + key, {})
    sibling = queue.rows(receipt)[0]
    client.content = json.dumps({'candidates': [missing]}, ensure_ascii=False)
    result = run_day(engine, raw_day(2), buckets=buckets, force=True)
    assert result['status'] == 'pending' and len(result['candidates']) == 1
    request = json.loads(client.calls[-1]['messages'][-1]['content'])
    assert request['recover_missing']['missing_count'] == 1
    assert request['recover_missing']['already_retained_ids'] == [sibling['candidate_id']]
    assert len(request['l1_contract']['source_texts']) == 2
    assert engine.memory_authority_store.get_candidate(sibling['candidate_id']) == sibling


def test_explicit_failed_window_retry_does_not_start_unrelated_preparation(tmp_path):
    engine, client, buckets = daily(tmp_path)
    client.content = json.dumps({'candidates': [generated(1)]}, ensure_ascii=False)
    async def unexpected(*args, **kwargs):
        pytest.fail('chat window retry started a different source task')
    engine._memory_l1_action_tasks = engine._memory_l1_attachment_tasks = unexpected
    result = asyncio.run(engine.run_daily_chat_memory(buckets, raw_event_store=FakeRawEvents(raw_day(1)),
        conversation_turn_store=FakeTurns([]), key='2026-08-18', mode='review', force=True,
        chat_windows_only=True))
    assert result['status'] == 'pending' and len(client.calls) == 1


def test_chat_letter_from_actual_sources_stays_in_review(tmp_path):
    engine, client, buckets = daily(tmp_path)
    events = raw_day(1)
    events[0]['text'] = '这是我写给顾衍的情书：一直喜欢你，想让你知道。'
    letter = generated(1, long=False)
    letter.update(kind='love_letter', reason='主人在当前消息写的情书。')
    letter['document']['narrative'] = {'title': '她写给我的信', 'body': '她写下一封信，告诉我她一直喜欢我。'}
    letter['document']['statements'][0]['content'] = '她在这条消息里写了一封情书。'
    client.content = json.dumps({'candidates': [letter]}, ensure_ascii=False)
    result = run_day(engine, events, buckets=buckets)
    assert result['status'] == 'pending'
    row = engine.memory_authority_store.list_candidates(status='all')[0]
    assert row['proposal']['memory_type'] == 'love_letter' and row['status'] == 'pending'
    assert row['proposal']['source_refs']


def test_action_mixed_invalid_response_retains_valid_draft_and_original_retry_can_correct(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = action_setup(tmp_path, monkeypatch)
    item = json.loads(engine.dehydration_client.content)['candidates'][0]
    bad = copy.deepcopy(item)
    del bad['document']['retrieval']
    engine.dehydration_client.content = json.dumps({'candidates': [item, bad]}, ensure_ascii=False)
    queued = tasks.enqueue_action(payload)
    first = asyncio.run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert first['status'] == 'partial' and len(first['candidate_ids']) == 1
    assert tasks.list()[0]['phase'] == 'failed'
    assert first['issue']['schema_error']['code'] == 'native_l1_schema_mismatch'
    assert asyncio.run(engine._memory_l1_action_tasks(buckets))['results'] == [] and len(calls) == 1
    tasks.transition(queued['task_id'], 'failed', 'queued')  # Existing owner retry; source action is never executed.
    engine.dehydration_client.content = json.dumps({'candidates': [item]}, ensure_ascii=False)
    corrected = asyncio.run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert corrected['status'] == 'prepared' and corrected['candidate_ids'] == first['candidate_ids']
    assert corrected['issue'] is None and len(calls) == 2
