"""Normal Reflection ingestion and native v2 requests with a scripted provider."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from chat_memory_tool import ChatMemoryTool
from memory_candidate_preparation import MemoryCandidatePreparation
from memory_chat_segments import ChatSegmentProgress
from memory_chat_preparation import prepare_chat_tasks
from memory_l1_v2_diagnostics import MemoryV2Diagnostics
from memory_l1_v2_generation import INPUT_TOKENS, PROMPT_ID
from raw_events import RawEventStore
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config


DAY = '2026-11-12'


def _engine(path, reply, *, mode='review'):
    config = reflection_config(path, mode=mode)
    config['reflection'].update(daily_chat_memory_extraction_mode='whole_window_v2',
                                daily_chat_memory_turn_limit=1, daily_chat_memory_window_turns=2)
    config['persona'] = {'canonical_session_id': 'main', 'profile_id': 'profile'}
    config['gateway']['canonical_continuation'] = {'enabled': True, 'conversation_id': 'canonical-main'}
    engine = ReflectionEngine(config)
    calls = []

    async def create(**request):
        payload = json.loads(request['messages'][-1]['content'])
        calls.append((request, payload))
        data = reply(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'cards': data}, ensure_ascii=False)), finish_reason='stop')],
            usage={'prompt_tokens': 100, 'completion_tokens': 200, 'total_tokens': 300})

    client = SimpleNamespace(base_url='https://fixture.invalid/v1',
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def options(**request_options):
        assert request_options['max_retries'] == 0
        return client

    client.with_options = options
    engine.client, engine.model = client, 'fixture-v2'
    return engine, calls


def _events(engine, texts):
    store = RawEventStore(engine.config)
    start = datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
    records = [{'source_event_id': f'event-{index}', 'role': role, 'text': text,
        'created_at': (start + timedelta(seconds=index)).isoformat(),
        'session_id': 'main', 'conversation_id': 'main', 'metadata': {'profile_id': 'profile', 'round_id': index // 2}}
        for index, (role, text) in enumerate(texts)]
    result = store.ingest(records, source='gateway')
    assert result['inserted'] == len(records), result
    return store


def _run(engine, store, *, force=False):
    return asyncio.run(engine.run_daily_chat_memory(None, raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=DAY, chat_windows_only=True, force=force))


def _card(sources):
    return {'title': '雨夜的约定', 'body': '2026年11月12日，我记得那晚我们一起等雨停，约好第二天再走。',
            'kind': 'commitment', 'stability': 'episode', 'keywords': ['雨夜', '等雨停'],
            'people': [], 'sources': sources}


def test_daily_window_delivers_every_original_then_queues_cards(tmp_path):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['messages'][0][0], payload['messages'][-1][0]])])
    texts = [('user' if index % 2 == 0 else 'assistant', f'第{index}条对话，在旧小窗限制之外仍需要看到原话。')
             for index in range(92)]
    store = _events(engine, texts)
    result = _run(engine, store)
    assert result['status'] == 'pending', result
    assert result['cursor_updated'] and result['model_call_count'] == 1
    assert len(calls) == 1 and [row[3] for row in calls[0][1]['messages']] == [text for _, text in texts]
    request, payload = calls[0]
    assert payload['window']['turns'] == 46
    assert request['max_tokens'] == 32000 and request['extra_body']['thinking']['type'] == 'enabled'
    assert 'statements' not in json.dumps(request['response_format'])
    assert result['candidates'][0]['l1_document']['generation']['prompt_source_id'] == PROMPT_ID
    assert engine.memory_authority_store.get_memory(result['candidates'][0]['id']) is None
    assert _run(engine, store)['status'] == 'skipped' and len(calls) == 1
    diagnostic = MemoryV2Diagnostics(engine.memory_authority_store).list(result['run_id'])
    returned = next(item['data'] for item in diagnostic if item['stage'] == 'model_return')
    assert returned['usage']['total_tokens'] == 300 and returned['input_tokens_estimate'] <= INPUT_TOKENS
    assert '在旧小窗限制之外' not in json.dumps(diagnostic, ensure_ascii=False)


def test_unclassified_experience_reaches_review_without_becoming_a_memory(tmp_path):
    from memory_l1_schema import join_document

    def reply(payload):
        card = _card([payload['messages'][0][0]])
        return [{**card, 'kind': None}, {**card, 'title': '真实生活经历', 'kind': 'life_experience'}]

    engine, calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '雨很大，我们留在屋里直到天亮。')])
    result = _run(engine, store)
    assert result['status'] == 'pending' and len(calls) == 1
    cards = {item['l1_document']['kind']: item for item in result['candidates']}
    assert cards[None]['classification_gap'] and not cards['life_experience']['classification_gap']
    candidate_id = cards[None]['id']
    row = engine.memory_authority_store.get_candidate(candidate_id)
    assert row['policy']['reason_codes'] == ['classification_gap']
    assert asyncio.run(engine._memory_l1_adopt_candidate(row, None, owner=True,
        request_id='try-unclassified'))['status'] == 'classification_gap'
    assert engine.memory_authority_store.get_candidate(candidate_id)['status'] == 'pending'
    assert engine.memory_authority_store.get_memory(candidate_id) is None
    proposal = row['proposal']
    document = join_document(proposal['proposed_body'], proposal['metadata']['memory_l1'],
                             allow_classification_gap=True)
    assert document['source_refs'] and document['kind'] is None


def test_v2_uses_saved_auto_policy_and_retains_sensitive_drafts(tmp_path):
    from memory_authority import MemoryIngestionPolicy

    def reply(payload):
        ref = payload['messages'][0][0]
        ready = {**_card([ref]), 'review': {'confidence': .9, 'sensitive': False,
                                           'transient': False, 'generic': False}}
        return [ready,
                {**ready, 'title': '需要主人决定的记录', 'review': {**ready['review'], 'sensitive': True}},
                {**ready, 'title': '未授权的类别', 'kind': 'personal_state'},
                {**ready, 'title': '不足以自动采用的记录', 'review': {**ready['review'], 'confidence': .3}}]

    engine, calls = _engine(tmp_path, reply, mode='auto')
    engine.memory_ingestion_policy = MemoryIngestionPolicy({'auto_types': ['commitment'], 'auto_confidence': .8})
    store = _events(engine, [('user', '雨很大，我们留在屋里直到天亮。')])
    from bucket_manager import BucketManager
    result = asyncio.run(engine.run_daily_chat_memory(BucketManager(engine.config), raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=DAY, chat_windows_only=True))
    assert len(calls) == 1 and result['auto_applied_count'] == 1 and result['pending_count'] == 3
    assert 'review' in calls[0][0]['response_format']['json_schema']['schema']['properties']['cards']['items']['properties']
    candidates = result['candidates']
    assert candidates[0]['status'] == 'applied'
    assert all(candidate['status'] == 'pending' for candidate in candidates[1:])
    assert engine.memory_authority_store.get_candidate(candidates[1]['id'])['policy']['reason_codes'] == ['REVIEW_SENSITIVE']
    assert 'review' not in candidates[0]['l1_document']


def test_overlong_message_is_read_in_continuous_ranges_and_empty_cards_advance(tmp_path):
    engine, calls = _engine(tmp_path, lambda _payload: [])
    # Insert an already-authenticated archive row so this tests v2's source
    # coverage independently of the public raw ingest size guard.
    text = '相遇而后长谈，雨落得很轻。' * 6000
    store = _events(engine, [('user', '前一句')])
    with store._connect() as db:
        db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', (text, 'event-0'))
    result = _run(engine, store)
    assert result['status'] == 'zero_candidates', result
    assert result['cursor_updated'] and len(calls) > 1
    covered = []
    for _, payload in calls:
        ranges = payload['source_ranges']['m1']
        for left, right in ranges:
            if not covered or left == covered[-1][1]:
                covered.append((left, right))
    assert covered[0][0] == 0 and covered[-1][1] == len(text)
    assert ''.join(text[left:right] for left, right in covered) == text


def test_card_from_one_segment_binds_only_that_observed_range(tmp_path):
    engine, calls = _engine(tmp_path, lambda payload: [_card(['m1'])] if not calls[:-1] else [])
    text = '同走夜路那晚，灯照进雨里。' * 6000
    store = _events(engine, [('user', '前一句')])
    with store._connect() as db:
        db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', (text, 'event-0'))
    result = _run(engine, store)
    assert result['status'] == 'pending', result
    assert len(calls) > 1
    _, first = calls[0]
    anchor = result['candidates'][0]['source_bindings'][0]['anchors'][0]
    assert [anchor['start'], anchor['end']] == first['source_ranges']['m1'][0]
    assert anchor['end'] < len(text)


def test_adjacent_day_context_records_matching_prior_card_revision(tmp_path):
    engine, calls = _engine(tmp_path, lambda payload: [_card(list(dict.fromkeys(
        row[0] for row in payload['messages'])))])
    store = _events(engine, [('user', '那晚我提议等雨停。'), ('assistant', '好，我陪你等。')])
    first = _run(engine, store)
    assert first['status'] == 'pending'
    tomorrow = '2026-11-13'
    store.ingest([{'source_event_id': 'next-day', 'role': 'user', 'text': '后来雨停了，我们一起走了。',
        'created_at': tomorrow + 'T00:00:00Z', 'session_id': 'main', 'conversation_id': 'main',
        'metadata': {'profile_id': 'profile', 'round_id': 1}}], source='gateway')
    second = asyncio.run(engine.run_daily_chat_memory(None, raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=tomorrow, chat_windows_only=True))
    assert second['status'] == 'pending' and len(calls) == 2, second
    assert [row[3] for row in calls[1][1]['messages']] == [
        '那晚我提议等雨停。', '好，我陪你等。', '后来雨停了，我们一起走了。']
    prior = calls[1][1]['continuation']['draft_cards'][0]
    assert prior['body'] == first['candidates'][0]['l1_document']['body']
    assert set(prior['sources']) == {'m1', 'm2'}
    row = engine.memory_authority_store.get_candidate(second['candidates'][0]['id'])
    link = row['proposal']['metadata']['l1_v2_continuation'][0]
    assert link['previous_candidate_id'] == first['candidates'][0]['id']
    assert link['previous_candidate_revision'] == 1


def test_unknown_dispatch_cannot_retry_from_scheduler(tmp_path):
    def unknown(_payload):
        raise TimeoutError('fixture provider outcome unknown')

    engine, calls = _engine(tmp_path, unknown)
    store = _events(engine, [('user', '那次我们等到了迟到的车。')])
    first = _run(engine, store)
    assert first['status'] == 'partial' and not first['cursor_updated']
    second = _run(engine, store)
    assert second['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert len(calls) == 1


@pytest.mark.parametrize('late_return', [False, True])
def test_daily_force_rescans_changed_source_while_original_dispatch_is_unknown(tmp_path, late_return):
    def reply(payload):
        return [_card([payload['messages'][-1][0]])]

    engine, original_calls = _engine(tmp_path, reply)
    recovery, recovery_calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    owner = SimpleNamespace(profile_id='profile')

    async def scenario():
        sent = asyncio.Event()
        release = asyncio.Event()
        original_create = engine.client.chat.completions.create

        async def create(**request):
            if not late_return:
                original_calls.append(request)
                raise TimeoutError('fixture provider outcome unknown')
            sent.set()
            await release.wait()
            return await original_create(**request)

        engine.client.chat.completions.create = create

        async def run(worker, *, force=False):
            return await worker.run_daily_chat_memory(None, raw_event_store=store, persona_engine=owner,
                key=DAY, force=force, chat_windows_only=True)

        original = asyncio.create_task(run(engine))
        if late_return:
            await asyncio.wait_for(sent.wait(), timeout=5)
        else:
            assert (await original)['status'] == 'partial'
        try:
            blocked = await run(recovery)
            assert blocked['reason'] == 'model_outcome_unknown_requires_retry_decision'
            original_key = 'l1-v2:' + blocked['generation_task_id']
            progress = ChatSegmentProgress(recovery.memory_authority_store)
            dispatched = progress.get(original_key)
            assert dispatched['progress']['phase'] == 'dispatched'

            with store._connect() as db:
                db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', ('更正，我们没有约好。', 'event-0'))
            changed = await run(recovery)
            assert changed['reason'] == 'l1_v2_daily_source_changed_requires_rescan'
            assert recovery_calls == [] and recovery._daily_chat_memory_last_raw_event_id('profile') == 0

            resumed = await run(recovery, force=True)
            assert resumed['status'] == 'pending', resumed
            assert resumed['supersedes_generation_task_id'] == blocked['generation_task_id']
            assert resumed['generation_task_id'] != blocked['generation_task_id']
            assert [row[3] for row in recovery_calls[0][1]['messages']] == ['更正，我们没有约好。']
            assert resumed['processed_until_seq'] == 1 and len(recovery_calls) == 1
        finally:
            release.set()
            original_result = await original

        if late_return:
            assert original_result['reason'] == 'l1_v2_daily_window_superseded'
        assert len(original_calls) == 1 and progress.get(original_key) == dispatched
        assert MemoryCandidatePreparation(recovery.memory_authority_store).get(original_key) is None
        assert len(recovery.memory_authority_store.list_candidates(status='pending')) == 1
        assert recovery._load_daily_chat_memory_run_audit()[-1]['status'] == 'pending'
        assert recovery._daily_chat_memory_last_raw_event_id('profile') == 1

    asyncio.run(scenario())


def _append_event(store, text):
    result = store.ingest([{'source_event_id': 'late-event', 'role': 'assistant', 'text': text,
        'created_at': DAY + 'T01:00:00Z', 'session_id': 'main', 'conversation_id': 'main',
        'metadata': {'profile_id': 'profile', 'round_id': 1}}], source='gateway')
    assert result['inserted'] == 1


@pytest.mark.parametrize('first_empty', [False, True])
def test_daily_resume_uses_prepared_watermark_then_reads_new_messages(tmp_path, monkeypatch, first_empty):
    def reply(payload):
        if len(payload['messages']) == 1 and first_empty:
            return []
        return [_card([payload['messages'][-1][0]])]

    engine, calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    watermarks = []
    update = engine._update_daily_chat_memory_raw_cursor

    def fail_first_write(profile, watermark, day):
        watermarks.append(watermark)
        if len(watermarks) == 1:
            raise OSError('cursor write interrupted after preparation')
        return update(profile, watermark, day)

    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', fail_first_write)
    first = _run(engine, store)
    assert first['reason'] == 'l1_v2_daily_cursor_not_persisted'
    assert first['processed_until_seq'] == 0
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 0
    before = engine.memory_authority_store.list_candidates(status='pending')
    assert len(before) == (0 if first_empty else 1)
    _append_event(store, '雨停后，我又说可以一起走。')

    resumed = _run(engine, store)
    assert resumed['status'] == 'pending', resumed
    assert watermarks == [1, 1, 2] and len(calls) == 2
    second = calls[1][1]
    assert [row[3] for row in second['messages']] == ['原来我们说好等雨停。', '雨停后，我又说可以一起走。']
    assert second['core_sources'] == [second['messages'][-1][0]]
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 2
    rows = engine.memory_authority_store.list_candidates(status='pending')
    assert len(rows) == (1 if first_empty else 2)
    if not first_empty:
        assert any(row['candidate_id'] == before[0]['candidate_id'] and row['revision'] == before[0]['revision'] for row in rows)
    assert _run(engine, store)['status'] == 'skipped' and len(calls) == 2


@pytest.mark.parametrize('withdraw', [False, True])
def test_daily_resume_refuses_changed_source_until_explicit_rescan(tmp_path, monkeypatch, withdraw):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['messages'][-1][0]])])
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    update = engine._update_daily_chat_memory_raw_cursor
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', lambda *_: False)
    first = _run(engine, store)
    assert first['reason'] == 'l1_v2_daily_cursor_not_persisted'
    old = engine.memory_authority_store.list_candidates(status='pending')[0]
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', update)
    with store._connect() as db:
        if withdraw:
            db.execute('DELETE FROM raw_events WHERE source_event_id=?', ('event-0',))
        else:
            db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', ('更正，我们没有约好。', 'event-0'))
    _append_event(store, '这是之后新增加的对话。')
    blocked = _run(engine, store)
    assert blocked['reason'] == 'l1_v2_daily_source_changed_requires_rescan'
    assert len(calls) == 1 and engine._daily_chat_memory_last_raw_event_id('profile') == 0

    resumed = _run(engine, store, force=True)
    assert resumed['status'] == 'pending', resumed
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 2
    sent = [[row[3] for row in payload['messages']] for _, payload in calls[1:]]
    if not withdraw:
        assert sent[0] == ['更正，我们没有约好。']
    assert sent[-1][-1] == '这是之后新增加的对话。'
    assert all('原来我们说好等雨停。' not in texts for texts in sent)
    assert engine.memory_authority_store.get_candidate(old['candidate_id'])['revision'] == old['revision']


def test_daily_resume_retries_only_incomplete_adoption_before_new_window(tmp_path, monkeypatch):
    def reply(payload):
        ref = payload['messages'][-1][0]
        card = {**_card([ref]), 'review': {'confidence': .9, 'sensitive': False,
                                          'transient': False, 'generic': False}}
        return [card, {**card, 'title': '雨停后的散步'}] if len(payload['messages']) == 1 else [card]

    engine, calls = _engine(tmp_path, reply, mode='auto')
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    adoption_attempts = []
    failed = set()

    async def adopt(row, _buckets, *, request_id):
        candidate = row['candidate_id']
        adoption_attempts.append(candidate)
        if len(adoption_attempts) == 2:
            failed.add(candidate)
            raise OSError('synthetic write failure for one candidate')
        # Simulate the original durable state transition while keeping this
        # test focused on the generation/receipt/cursor boundary.
        engine.memory_authority_store.decide_candidate(candidate, action='commit',
            expected_revision=row['revision'], request_id=request_id, actor='fixture')
        return {'status': 'created'}

    monkeypatch.setattr(engine, '_memory_l1_adopt_candidate', adopt)
    first = _run(engine, store)
    assert first['reason'] == 'candidate_persistence_failed', first
    assert first['processed_until_seq'] == 0 and len(calls) == 1
    _append_event(store, '雨停后，我又说可以一起走。')
    resumed = _run(engine, store)
    assert resumed['status'] == 'created', resumed
    assert len(calls) == 2 and len(adoption_attempts) == 4
    assert [candidate for candidate in adoption_attempts if candidate in failed] == [*failed, *failed]
    assert adoption_attempts.count(adoption_attempts[0]) == 1
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 2


def test_explicit_retry_keeps_unfinished_nonzero_window(tmp_path, monkeypatch):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['messages'][-1][0]])])
    store = _events(engine, [('user', '原来的对话。')])
    assert _run(engine, store)['status'] == 'pending'
    _append_event(store, '新追加的对话。')
    update = engine._update_daily_chat_memory_raw_cursor
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', lambda *_: False)
    assert _run(engine, store)['reason'] == 'l1_v2_daily_cursor_not_persisted'
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', update)

    resumed = _run(engine, store, force=True)
    assert resumed['source_start_seq'] == 1 and resumed['processed_until_seq'] == 2
    assert len(calls) == 2 and len(engine.memory_authority_store.list_candidates(status='pending')) == 2


def test_chat_draft_can_prepare_without_nearby_chat_sources(tmp_path):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['agent_draft']['source_ref']])])
    envelope = {'operation_id': 'op-' + '1' * 32, 'arguments': {'narrative': {
        'title': '雨夜', 'body': '我记得有一次自己在雨夜等车，想起她说可以晚点再走。'}},
        'context': {'conversation_id': 'canonical-main', 'request_id': 'r1', 'turn_id': 'e1', 'sources': []}}
    queued = asyncio.run(ChatMemoryTool(engine, None).submit(envelope))
    result = asyncio.run(prepare_chat_tasks(engine, None))
    assert result['results'][0]['memory_status'] == 'pending', result
    assert calls[0][1]['messages'][0][3] == envelope['arguments']['narrative']['body']
    row = engine.memory_authority_store.get_candidate(queued['task_id'])
    assert row['proposal']['metadata']['memory_l1']['evidence_bindings'][0]['source_namespace'] == (
        'snapshot.ombre.agent_memory_draft')
