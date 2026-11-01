"""Retired relationship weather cannot re-enter the actual Gateway model input."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from test_prompt_context_placements_r1 import (
    _block, _compose_and_inject, _context_values, _gateway_wire_messages, _plan, _service,
)


@pytest.mark.parametrize('mode', ['live_source', 'frozen_snapshot', 'owner_override'])
def test_retired_weather_all_modes_are_absent_from_final_input_and_keep_neighbor_sources(mode):
    service = _service()
    retired = _block('ombre.relationship_weather', role='developer',
                     anchor='turn.before_current_user', order=20, mode=mode,
                     owner_body='RETIRED_OWNER_COPY', wrapper_text='RETIRED_WRAPPER {{content}}')
    retired['frozen_body'] = 'RETIRED_FROZEN_COPY'
    plan = _plan(
        _block('ombre.core_memory', role='system', anchor='turn.before_current_user', order=10),
        retired,
        _block('ombre.just_now_context', role='developer', anchor='turn.before_current_user', order=30),
    )
    saved_plan = copy.deepcopy(plan)
    values = _context_values()
    values['relationship_weather'] = 'RETIRED_LIVE_COPY'
    messages = [
        {'role': 'system', 'content': 'OWNER_SYSTEM'},
        {'role': 'user', 'content': 'HISTORY_OWNER_TURN'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call-fixture', 'type': 'function',
             'function': {'name': 'memory.read', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call-fixture', 'content': 'ORIGINAL_TOOL_RESULT'},
        {'role': 'user', 'content': 'CURRENT_OWNER_TURN'},
    ]

    prepared, stable, dynamic, records, debug = _compose_and_inject(
        service, plan, messages, values=values)
    final_messages = _gateway_wire_messages(prepared)
    serialized = json.dumps(final_messages)

    assert 'RETIRED_' not in serialized
    assert all('RETIRED_' not in str(part) for part in (stable, dynamic, records, debug))
    assert 'CORE_BLOCK_R1' in serialized and 'JUST_NOW_R1' in serialized
    assert serialized.index('CORE_BLOCK_R1') < serialized.index('JUST_NOW_R1')
    assert serialized.index('JUST_NOW_R1') < serialized.index('CURRENT_OWNER_TURN')
    assert [row for row in final_messages if row.get('content') not in {
        message.get('content') for message in messages}]  # Neighbor blocks really compiled.
    assert [row for row in final_messages if row.get('role') in {'assistant', 'tool'}] == messages[2:4]
    assert plan == saved_plan  # Neither discard nor migrate the owner's saved preset.


@pytest.mark.parametrize('has_event', [False, True])
def test_date_persona_trace_reads_original_events_without_reviving_legacy_impression(has_event):
    service = _service()
    service.date_persona_trace_enabled = True
    service.date_persona_trace_include_daily = True  # Stale switch cannot restore it.
    service.date_persona_trace_budget = 1000
    service.date_persona_trace_max_events = 5
    service.gateway_tz = ZoneInfo('Asia/Shanghai')
    service._query_date_hint = lambda query: {'date': '2026-11-02', 'label': 'fixture day'}
    event = {'id': 37, 'created_at': '2026-11-02T07:00:00+08:00',
             'user_excerpt': 'ORIGINAL_OWNER_TURN', 'assistant_excerpt': 'ORIGINAL_AI_TURN'}
    service.persona_engine = SimpleNamespace(_list_events=Mock(return_value=[event] if has_event else []))
    service._daily_impression_bucket_for_date = Mock(
        side_effect=AssertionError('retired daily impression lookup ran'))
    buckets = [{'id': 'reflection_daily_2026-11-02', 'content': 'RETIRED_DAILY_ONLY',
                'metadata': {'type': 'feel', 'date': '2026-11-02',
                             'tags': ['relationship_weather', 'daily_impression']}}]
    before = copy.deepcopy(buckets)

    body, debug = service._build_date_persona_trace_block('fixture date query', buckets)

    service._daily_impression_bucket_for_date.assert_not_called()
    assert 'RETIRED_DAILY_ONLY' not in body
    assert debug['daily_bucket_id'] == ''
    if has_event:
        assert 'user: ORIGINAL_OWNER_TURN' in body and 'assistant: ORIGINAL_AI_TURN' in body
        assert debug['selected_event_ids'] == [37]
        assert debug['status'] == 'injected'
    else:
        assert body == '' and debug['skip_reason'] == 'no_material'
    assert buckets == before
