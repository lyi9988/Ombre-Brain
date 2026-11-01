"""Daily/weekly impression retirement preserves independent memory and portrait inputs."""
import asyncio
import copy
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from reflection_engine import ReflectionEngine
from portrait_engine import DailyPortraitMaintainer
from test_memory_authority_r1 import reflection_config


def engine(tmp_path):
    value = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    value.enabled = value.auto_enabled = True
    return value


@pytest.mark.parametrize('force', [False, True])
@pytest.mark.parametrize('period', ['daily', 'weekly'])
@pytest.mark.parametrize('saved_enabled', [False, True])
def test_explicit_impressions_are_retired_without_source_reads_models_or_writes(
        tmp_path, force, period, saved_enabled):
    value = engine(tmp_path)
    value.daily_enabled = value.weekly_enabled = saved_enabled
    forbidden = AsyncMock(side_effect=AssertionError('retired impression path ran'))
    value._reflection_materials = forbidden
    value._api_reflect = forbidden
    value._maybe_extract_diary_memory = forbidden
    bucket_mgr = SimpleNamespace(list_all=forbidden, get=forbidden,
                                 create=forbidden, update=forbidden, delete=forbidden)
    result = asyncio.run(value.reflect(period, bucket_mgr, force=force))
    assert result['status'] == 'retired'
    assert result['reason'] == 'relationship_impressions_retired'
    assert result['period'] == period
    assert result['diary_memory']['reason'] == 'separate_diary_task'
    forbidden.assert_not_called()


def test_due_keeps_auto_chat_and_diary_l1_but_never_schedules_impressions(tmp_path):
    value = engine(tmp_path)
    value.daily_enabled = False
    value.daily_hour = value.daily_chat_memory_hour = value.weekly_hour = 0
    value.weekly_enabled = True
    value.weekly_day = 0
    value.diary_memory_extract_enabled = True
    value.diary_memory_extract_max_per_day = 3
    value._local_now = lambda now=None: datetime(2026, 11, 2, 8, 0, tzinfo=value.tz)
    value._load_daily_chat_memory_run_audit = lambda: []
    value.run_daily_chat_memory = AsyncMock(return_value={'status': 'pending', 'candidates': []})
    value._memory_l1_diary_task = AsyncMock(return_value={'status': 'queued', 'adopted': False})
    # Real _maybe_extract_diary_memory keeps its enabled/mode/source-task gate.
    value.reflect = AsyncMock(side_effect=AssertionError('retired scheduled reflection ran'))
    value._reflection_materials = AsyncMock(side_effect=AssertionError('daily weather materials were read'))

    result = asyncio.run(value.run_due(object()))
    value.run_daily_chat_memory.assert_awaited_once()
    assert value._memory_l1_diary_task.await_args.args[0] == '2026-11-01'
    value.reflect.assert_not_called()
    assert next(row for row in result if row.get('kind') == 'diary_memory')['diary_memory']['adopted'] is False
    value._reflection_materials.assert_not_called()

    value.run_daily_chat_memory.reset_mock()
    value._memory_l1_diary_task.reset_mock()
    value.reflect.reset_mock()
    value.daily_chat_memory_mode = 'off'
    value.weekly_enabled = False
    asyncio.run(value.run_due(object()))
    value.run_daily_chat_memory.assert_not_called()
    value._memory_l1_diary_task.assert_not_called()
    value.reflect.assert_not_called()


@pytest.mark.parametrize('gate', ['before_diary_hour', 'diary_disabled', 'daily_limit_zero'])
def test_diary_l1_retains_its_own_hour_switch_and_limit(tmp_path, gate):
    value = engine(tmp_path)
    value.daily_chat_memory_hour = 0
    value.daily_hour = 9 if gate == 'before_diary_hour' else 0
    value.diary_memory_extract_enabled = gate != 'diary_disabled'
    value.diary_memory_extract_max_per_day = 0 if gate == 'daily_limit_zero' else 3
    value._local_now = lambda now=None: datetime(2026, 11, 2, 8, 0, tzinfo=value.tz)
    value._load_daily_chat_memory_run_audit = lambda: []
    value.run_daily_chat_memory = AsyncMock(return_value={'status': 'pending', 'candidates': []})
    value._memory_l1_diary_task = AsyncMock(side_effect=AssertionError('diary gate bypassed'))
    value.reflect = AsyncMock(side_effect=AssertionError('retired scheduled reflection ran'))

    result = asyncio.run(value.run_due(object()))
    value.run_daily_chat_memory.assert_awaited_once()
    value._memory_l1_diary_task.assert_not_called()
    value.reflect.assert_not_called()
    assert not any(row.get('kind') == 'diary_memory' for row in result)


@pytest.mark.parametrize('initial', [False, True])
def test_portrait_materials_drop_only_old_impressions_and_keep_original_state(tmp_path, initial):
    config = reflection_config(tmp_path, mode='review')
    config['portrait'] = {'enabled': False, 'state_path': str(tmp_path / 'portrait.json')}
    config['self_anchor'] = {'entry_bucket_id': 'self-anchor'}
    value = DailyPortraitMaintainer(config)
    created = '2026-11-02T07:00:00+08:00'

    def bucket(bucket_id, tags, content, **metadata):
        return {'id': bucket_id, 'content': content, 'metadata': {
            'type': 'feel', 'created': created, 'tags': tags, **metadata}}

    buckets = [
        bucket('old-daily', ['relationship_weather', 'daily_impression'], 'RETIRED_DAILY_ONLY'),
        bucket('old-weekly', ['relationship_weather', 'weekly_impression'], 'RETIRED_WEEKLY_ONLY'),
        bucket('ordinary-feel', ['feel'], 'ORDINARY_FEEL_PRESERVED'),
        bucket('new-state', ['inner_state'], 'UNIFIED_STATE_PRESERVED'),
        bucket('profile', ['profile_fact'], 'OWNER_PROFILE_PRESERVED'),
        bucket('self-anchor', ['self_anchor'], 'PERSONA_ANCHOR_PRESERVED', pinned=True),
    ]
    before = copy.deepcopy(buckets)
    bucket_mgr = SimpleNamespace(list_all=AsyncMock(return_value=buckets))
    event = {'id': 37, 'session_id': 'fixture-session', 'created_at': created,
             'event_type': 'relationship_event', 'user_excerpt': 'ORIGINAL_OWNER_TURN',
             'assistant_excerpt': 'ORIGINAL_AI_TURN', 'inner_thought': 'ORIGINAL_PERSONA_STATE'}
    persona = SimpleNamespace(get_dashboard_payload=Mock(return_value={'events': [event]}))
    state = {'portrait': {'persona': {'stable': 'EXISTING_PORTRAIT', 'stable_revision': 4}},
             'recent_activities': [{'text': 'EXISTING_ACTIVITY'}]}
    state_before = copy.deepcopy(state)

    result = asyncio.run(value._daily_materials(
        bucket_mgr, persona, datetime(2026, 11, 2, 8, tzinfo=value.tz), state, initial=initial))

    assert {row['bucket_id'] for row in result['buckets']} == {
        'ordinary-feel', 'new-state', 'profile', 'self-anchor'}
    assert result['evidence_scope_limits']['profile'] == ['user']
    assert result['evidence_scope_limits']['self-anchor'] == ['persona']
    assert not {'old-daily', 'old-weekly'} & result['evidence_scope_limits'].keys()
    assert result['persona_stable_evidence']['self_anchor_ids'] == ['self-anchor']
    assert result['persona_events'][0]['inner_thought'] == 'ORIGINAL_PERSONA_STATE'
    assert result['persona_events'][0]['user_excerpt'] == 'ORIGINAL_OWNER_TURN'
    assert result['previous_portrait']['persona']['stable'] == 'EXISTING_PORTRAIT'
    assert result['previous_portrait']['persona']['stable_revision'] == 4
    serialized = json.dumps(result)
    assert 'RETIRED_DAILY_ONLY' not in serialized and 'RETIRED_WEEKLY_ONLY' not in serialized
    assert buckets == before and state == state_before  # Read-only filtering retains history.
    bucket_mgr.list_all.assert_awaited_once_with(include_archive=False)


def test_handoff_recent_summary_ignores_old_flattened_impressions_but_keeps_original_turns():
    value = DailyPortraitMaintainer.__new__(DailyPortraitMaintainer)
    value.recent_continuity_days = 3
    value.identity = {'user_display_name': '主人', 'ai_name': '顾衍'}
    materials = {'buckets': [
        {'bucket_id': 'old-daily', 'source_date': '2026-11-02',
         'tags': ['relationship_weather', 'daily_impression'], 'text': 'RETIRED_DAILY_ONLY'},
        {'bucket_id': 'old-weekly', 'source_date': '2026-11-02',
         'tags': ['relationship_weather', 'weekly_impression'], 'text': 'RETIRED_WEEKLY_ONLY'},
    ], 'persona_events': [{'source_date': '2026-11-02',
                          'user_excerpt': 'ORIGINAL_OWNER_TURN',
                          'assistant_excerpt': 'ORIGINAL_AI_TURN'}]}

    result = value._build_handoff_recent_summaries(materials, {}, '2026-11-02')
    summary = result['2026-11-02']
    assert '主人说“ORIGINAL_OWNER_TURN”' in summary
    assert '顾衍回“ORIGINAL_AI_TURN”' in summary
    assert 'RETIRED_DAILY_ONLY' not in summary and 'RETIRED_WEEKLY_ONLY' not in summary
    assert '关系天气' not in summary
