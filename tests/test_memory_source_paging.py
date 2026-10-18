"""Real SQLite date/source continuation and owner list accessibility, offline."""
import asyncio
from datetime import datetime, timedelta
import json
from types import SimpleNamespace as NS

import pytest

from gateway_state import GatewayStateStore
from raw_events import RawEventStore
from test_memory_daily_windows import configured_engine, raw_day
from test_daily_chat_memory_quality_review_v4 import FakeBuckets, FakeTurns


def sources(tmp_path, count=200):
    raw = RawEventStore({'state_dir': str(tmp_path / 'raw')})
    state = GatewayStateStore(str(tmp_path / 'turns' / 'turns.sqlite'))
    events = raw_day(count)
    # More than the former latest-500 SQL sample is newer than the target day.
    for index in range(600):
        events.append({'id': count + index + 1, 'role': 'user', 'text': f'新日期素材{index}',
                       'created_at': '2026-08-20T10:00:00+08:00', 'metadata': {}})
    raw.ingest([{**event, 'source_event_id': str(index)} for index, event in enumerate(events)], source='test')
    for index, event in enumerate(events):
        state.record_conversation_turn(profile_id='default', session_id='s', round_id=index + 1,
            user_text=event['text'], created_at=datetime.fromisoformat(event['created_at']), max_entries=2000)
    return raw, state


def period():
    return {'start_at': datetime.fromisoformat('2026-08-18T00:00:00+08:00'),
            'end_at': datetime.fromisoformat('2026-08-19T00:00:00+08:00')}


def test_old_date_before_latest_500_and_positive_source_batches_cover_all_200(tmp_path):
    raw, state = sources(tmp_path)
    assert len(raw.list_events_between(**period(), limit=0, chronological=True)) == 200
    assert len(state.list_conversation_turns_between(profile_id='default', **period(), limit=0, chronological=True)) == 200
    for store, kind in ((raw, 'raw'), (state, 'turn')):
        seen, cursor = [], 0
        while True:
            kwargs = {'profile_id': 'default', 'after_turn_id': cursor} if kind == 'turn' else {'after_event_id': cursor}
            method = store.list_conversation_turns_between if kind == 'turn' else store.list_events_between
            batch = method(**period(), limit=23, chronological=True, **kwargs)
            if not batch:
                break
            seen.extend(row['id'] for row in batch)
            cursor = batch[-1]['id']
        assert seen == list(range(1, 201))


def test_sql_date_predicate_preserves_naive_query_timezone_and_z_offsets(tmp_path):
    raw = RawEventStore({'state_dir': str(tmp_path)})
    stamps = ['2026-08-17T15:59:59Z', '2026-08-17T16:00:00Z',
              '2026-08-18T00:00:00', '2026-08-18T23:59:59+08:00',
              '2026-08-18T16:00:00Z', '2026-08-17T23:59:59',
              '2026-08-18T00:00:00+09:00', '2026-08-18T01:00:00+09:00']
    db = raw._connect()
    try:
        for index, stamp in enumerate(stamps):
            db.execute("INSERT INTO raw_events(source,event_hash,role,text,created_at,ingested_at) VALUES(?,?,?,?,?,?)",
                       ('test', str(index), 'user', str(index), stamp, stamp))
        db.commit()
    finally:
        db.close()
    result = raw.list_events_between(**period(), limit=0, chronological=True)
    assert [row['text'] for row in result] == ['1', '2', '3', '7']


@pytest.mark.parametrize('fallback', [False, True])
def test_public_job_drains_positive_source_batches_without_losing_early_turns(tmp_path, fallback):
    raw, state = sources(tmp_path)
    engine = configured_engine(tmp_path / 'engine', daily_chat_memory_turn_limit=30,
                               daily_chat_memory_max_windows_per_run=2)
    seen = []
    async def extract(key, window, **kwargs):
        seen.extend(turn['id'] for turn in window)
        return [], {}
    engine._extract_window_candidates = extract
    result = asyncio.run(engine.run_daily_chat_memory(FakeBuckets(), conversation_turn_store=state,
        raw_event_store=None if fallback else raw, key='2026-08-18', mode='review'))
    assert result['status'] == 'zero_candidates'
    assert result['progress']['state'] == 'complete'
    assert result['progress']['batches_completed'] > 1
    assert set(seen) == set(range(1, 201))
    if fallback:
        assert engine._daily_chat_memory_last_conversation_turn_id('default') == 200
    else:
        assert engine.daily_chat_memory_run_cursor()['last_raw_event_id'] == 200


def test_public_cap_one_retains_three_generated_drafts_and_calls_provider_once(tmp_path):
    from test_memory_candidate_overflow import daily, generated
    from test_daily_chat_memory_quality_review_v4 import FakeRawEvents
    engine, client, buckets = daily(tmp_path)
    client.content = json.dumps({'candidates': [generated(1), generated(2), generated(3)]}, ensure_ascii=False)
    # The fixture client stores its response under result, preserving its real
    # capture boundary without replacing the public worker or extraction.
    if hasattr(client, 'result'):
        client.result = client.content
    result = asyncio.run(engine.run_daily_chat_memory(buckets, conversation_turn_store=FakeTurns([]),
        raw_event_store=FakeRawEvents(raw_day(3)), key='2026-08-18', mode='review'))
    assert result['status'] == 'pending'
    assert result['progress']['state'] == 'complete'
    assert result['progress']['remaining_prepared_candidates'] == 0
    assert len(client.calls) == 1
    assert len(engine.memory_authority_store.list_candidates(status='all')) == 3
    assert result['last_raw_event_id'] == 3


def test_review_mode_remains_enabled_when_auto_cap_zero_and_hotrefresh_uses_owner_values(tmp_path):
    engine = configured_engine(tmp_path, daily_chat_memory_max_per_day=0, daily_chat_memory_review_max_per_day=15)
    config = {'reflection': {'daily_chat_memory_max_per_day': 25, 'daily_chat_memory_review_max_per_day': 31,
        'daily_chat_memory_max_windows_per_run': 17, 'daily_chat_memory_window_max_input_chars': 24000}}
    engine.refresh_daily_chat_memory_options(config)
    assert engine.daily_chat_memory_max_per_day == 25
    assert engine.daily_chat_memory_review_max_per_day == 31
    assert engine.daily_chat_memory_max_windows_per_run == 17
    assert engine.daily_chat_memory_window_max_input_chars == 24000
    engine.daily_chat_memory_max_per_day = 0
    async def extract(key, window, **kwargs): return [], {}
    engine._extract_window_candidates = extract
    from test_daily_chat_memory_quality_review_v4 import FakeRawEvents
    result = asyncio.run(engine.run_daily_chat_memory(FakeBuckets(), conversation_turn_store=FakeTurns([]),
        raw_event_store=FakeRawEvents(raw_day(2)), key='2026-08-18', mode='review'))
    assert result['status'] == 'zero_candidates'


def test_run_due_prioritizes_old_partial_and_refuses_cursor_ahead(tmp_path):
    engine = configured_engine(tmp_path)
    engine.auto_enabled = True
    engine.daily_enabled = engine.weekly_enabled = False
    engine._local_now = lambda: datetime.fromisoformat('2026-08-21T10:00:00+08:00')
    calls = []
    async def run(*args, **kwargs):
        calls.append(kwargs['now'].date().isoformat())
        return {'status': 'zero_candidates'}
    engine.run_daily_chat_memory = run
    engine._store_daily_chat_memory_run_audit({'date': '2026-08-18', 'status': 'partial', 'source_start_seq': 0,
        'processed_until_seq': 0, 'error_category': 'window_budget_exhausted'})
    asyncio.run(engine.run_due(FakeBuckets()))
    assert calls == ['2026-08-18']
    engine._update_daily_chat_memory_raw_cursor('default', 20, '2026-08-20')
    calls.clear()
    result = asyncio.run(engine.run_due(FakeBuckets()))
    assert calls == ['2026-08-20']
    assert result[0]['unresolved_dates'][0]['status'] == 'needs_review'
    assert result[0]['unresolved_dates'][0]['reason'] == 'source_cursor_ahead_of_partial'
    calls.clear()
    asyncio.run(engine.run_due(FakeBuckets()))
    assert calls == ['2026-08-20']  # The persisted needs_review row cannot starve a new date.


def test_old_partial_candidates_do_not_enter_yesterdays_daily_reflection(tmp_path):
    engine = configured_engine(tmp_path)
    engine.auto_enabled = engine.daily_enabled = True
    engine.daily_hour = 0
    engine.weekly_enabled = False
    engine._local_now = lambda: datetime.fromisoformat('2026-08-21T10:00:00+08:00')
    engine._store_daily_chat_memory_run_audit({'date': '2026-08-18', 'status': 'partial', 'processed_until_seq': 0,
                                             'error_category': 'window_budget_exhausted'})
    async def run(*args, **kwargs):
        return {'status': 'pending', 'date': '2026-08-18', 'candidates': [{'id': 'old', 'date': '2026-08-18'}]}
    captured = []
    async def reflect(*args, **kwargs):
        captured.append(kwargs)
        return {'status': 'skipped'}
    engine.run_daily_chat_memory = run
    engine.reflect = reflect
    results = asyncio.run(engine.run_due(FakeBuckets()))
    assert results[0]['candidates'][0]['date'] == '2026-08-18'
    assert results[0]['next_date'] == '2026-08-20'
    assert captured[0]['now'].date().isoformat() == '2026-08-20'
    assert captured[0]['daily_chat_memory_candidates'] == []


def test_odd_raw_batch_boundary_preserves_complete_session_rounds(tmp_path):
    raw = RawEventStore({'state_dir': str(tmp_path / 'raw')})
    events = []
    for index in range(24):
        events.append({'source_event_id': str(index), 'role': 'user' if index % 2 == 0 else 'assistant',
            'text': f'完整一轮第{index // 2 + 1}条{index % 2}', 'session_id': 's',
            'created_at': '2026-08-18T09:00:00+08:00', 'metadata': {'round_id': index // 2 + 1}})
    raw.ingest(events, source='test')
    for size, expected in ((1, 2), (23, 24)):
        batch = raw.list_events_between(**period(), chronological=True, limit=size, complete_rounds=True)
        assert len(batch) == expected
        assert batch[-1]['role'] == 'assistant'
    engine = configured_engine(tmp_path / 'engine', daily_chat_memory_turn_limit=23)
    seen = []
    async def extract(key, window, **kwargs):
        seen.extend(window)
        return [], {}
    engine._extract_window_candidates = extract
    result = asyncio.run(engine.run_daily_chat_memory(FakeBuckets(), conversation_turn_store=FakeTurns([]),
        raw_event_store=raw, key='2026-08-18', mode='review'))
    assert result['status'] == 'zero_candidates'
    assert len(seen) == 12 and all(len(turn['_semantic_sources']) == 2 for turn in seen)
    assert result['last_raw_event_id'] == 24


def test_candidate_pagination_reaches_past_1000_and_mixed_tasks_past_100(tmp_path):
    from memory_authority import MemoryAuthorityStore, MemoryIngestionPolicy
    from memory_l1_source_tasks import MemoryL1SourceTasks, PREFIX as ACTION_PREFIX
    from memory_chat_preparation import PREFIX as CHAT_PREFIX
    from test_memory_authority_r1 import proposal
    authority = MemoryAuthorityStore({'state_dir': str(tmp_path)})
    policy = MemoryIngestionPolicy()
    db = authority._connect()
    try:
        db.execute('BEGIN IMMEDIATE')
        for index in range(1007):
            item = proposal(proposal_id=f'page-{index:04}', requested_mode='review')
            authority._put_candidate(db, item, policy.evaluate(item))
        for index in range(205):
            is_chat = index % 2 == 0
            task_id = ('chat_memory_' if is_chat else 'action_source_') + str(index)
            row = {'task_id': task_id, 'phase': 'queued', 'created_at': f'2026-08-{index // 24 + 1:02}T{index % 24:02}:00:00Z',
                   'envelope': {'private': 'not a public field'}}
            db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)',
                       ((CHAT_PREFIX if is_chat else ACTION_PREFIX) + task_id, json.dumps(row)))
        db.commit()
    finally:
        db.close()
    engine = configured_engine(tmp_path / 'engine')
    engine.memory_authority_enabled = True
    engine.memory_authority_store = authority
    ids = []
    for offset in range(0, 1100, 100):
        page = engine.daily_chat_memory_pending_page(status='review_visible', limit=100, offset=offset)
        ids.extend(item['id'] for item in page['items'])
        assert page['total'] == page['saved_total'] == 1007
        if not page['has_more']: break
    assert len(set(ids)) == len(ids) == 1007
    tasks, offset = [], 0
    while True:
        page = MemoryL1SourceTasks(authority).page_with_chat(limit=100, offset=offset)
        assert page['total'] == 205
        tasks.extend(page['tasks'])
        if not page['has_more']: break
        offset = page['next_offset']
    assert len({task['task_id'] for task in tasks}) == 205
    assert all('envelope' not in task for task in tasks if task.get('source_kind') == 'chat_submission')
