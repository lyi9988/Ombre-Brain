import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from test_chat_memory_preparation import setup


def response_adapter(engine, transform, *, finish_reason='stop'):
    original = engine._daily_chat_memory_create_completion
    async def create(*args, **kwargs):
        result = await original(*args, **kwargs)
        parsed = json.loads(result.choices[0].message.content)
        result.choices[0].message.content = json.dumps(transform(parsed))
        result.choices[0].finish_reason = finish_reason
        result.usage = NS(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        return result
    engine._daily_chat_memory_create_completion = create


def test_complete_document_root_is_wrapped_without_bypassing_l1_validation(tmp_path):
    env, engine, envelope, calls = setup(tmp_path)
    response_adapter(engine, lambda value: value['document'])
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    result = asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert result['results'][0]['memory_status'] == 'pending'
    row = env['authority'].get_candidate(key)
    assert row['status'] == 'pending'
    assert row['proposal']['proposed_body'] == envelope['arguments']['narrative']['body']
    assert tasks.get(key)['output_shape']['normalization'] == 'complete_document_root'
    assert tasks.get(key)['preparation_output'] is None
    assert env['authority'].list_memories() == [] and len(calls) == 1


def test_document_root_still_rejects_invalid_inner_semantics(tmp_path):
    env, engine, envelope, calls = setup(tmp_path, bad=True)
    response_adapter(engine, lambda value: value['document'])
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert tasks.get(key)['phase'] == 'failed'
    assert tasks.get(key)['issue']['code'] == 'unknown_catalog_value'
    assert env['authority'].list_candidates(status='all') == []
    assert len(calls) == 1


@pytest.mark.parametrize('finish_reason', ['stop', 'length'])
def test_failure_keeps_private_response_and_usage_without_public_list_leak(tmp_path, finish_reason):
    env, engine, envelope, calls = setup(tmp_path)
    response_adapter(engine, lambda value: {'unrecognized_wrapper': value['document']},
                     finish_reason=finish_reason)
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    row = tasks.get(key)
    assert row['phase'] == 'failed'
    assert row['usage'] == {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}
    assert row['generation']['actor'] == 'chat_memory_preparation'
    assert row['duration_ms'] >= 0
    assert json.loads(row['preparation_output']['text']).get('unrecognized_wrapper')
    assert row['preparation_output']['draft_revision'] == 1
    assert row['preparation_output']['attempt'] == 1
    assert tasks.detail(key)['preparation_output'] == row['preparation_output']
    public = tasks.list()
    assert 'preparation_output' not in public[0]
    assert '后台拟题' not in json.dumps(public, ensure_ascii=False)
    assert env['authority'].list_candidates(status='all') == [] and len(calls) == 1
