import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

from chat_memory_tool import ChatMemoryTool, candidate_id
from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from memory_l1_ingestion import model_template
from memory_l1_schema import statement_template
from memory_l1_service import L1OwnerError
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_diary_l1_p2b import setup_diary


def setup(tmp_path, *, mode='review', bad=False, during=None, compact=False):
    env = setup_diary(tmp_path / 'memory', '合成来源')
    config = reflection_config(tmp_path / 'engine', mode=mode)
    engine = ReflectionEngine(config)
    engine.memory_authority_store = env['authority']
    engine.memory_ingestion_policy = env['policy']
    engine._memory_l1_revisions = lambda manager=None: env['revisions']
    from memory_l0_sources import MemoryL0Sources
    retain_copy = MemoryL0Sources(env['authority']).retain_copy
    engine._memory_l1_sources = lambda: NS(locate=locate, retain_copy=retain_copy)
    calls = []
    async def create(client, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        calls.append(payload)
        if during:
            during()
        document = model_template(catalog=env['catalog'])
        document['narrative'] = {'title': '后台拟题', 'body': '模型尝试改写正文'}
        statement = statement_template()
        statement.pop('statement_revision_id')
        statement.update(statement_id='claim1', content='合成的当次请求', evidence_refs=['source:2'])
        if bad:
            statement['subject_ref'] = {'state': 'known', 'value': 'undeclared', 'evidence_refs': ['source:2']}
        document['statements'] = [statement]
        if compact:
            document = {'format': 'semantic-v1', 'entities': [], 'statements': [{
                'content': statement['content'], 'subject_ref': None,
                'classification': {'kind': None, 'stability': None, 'modality': None, 'basis': None},
                'time': None, 'conditions': None, 'evidence_refs': ['source:2']}]}
        return NS(choices=[NS(message=NS(content=json.dumps({'document': document,
            'kind': 'shared_experience', 'confidence': 0.95})), finish_reason='stop')])
    async def locate(binding):
        return {'state': 'available', 'eligible_source': True}
    # Preparation now consumes the explicit stage router, not the old daily
    # client helper. Keep this integration fixture on the real route boundary.
    engine._memory_task_route = lambda stage: {'stage':stage, 'client':object(), 'model':'synthetic-model',
        'connection':'reflection', 'configured':True, 'thinking_mode':'', 'reasoning_effort':'',
        'max_tokens':16000, 'unavailable_reason':''}
    engine._daily_chat_memory_create_completion = create
    sources = [{'event_id': 'evt-1', 'version_id': 'v1', 'role': 'user', 'text': '我今天有点累。', 'created_at': '2026-10-16T10:00:00Z'},
               {'event_id': 'evt-2', 'version_id': 'v2', 'role': 'user', 'text': '抱抱我。', 'created_at': '2026-10-16T10:00:01Z'},
               {'event_id': 'evt-3', 'version_id': 'v3', 'role': 'user', 'text': '记下来。', 'created_at': '2026-10-16T10:00:02Z'}]
    envelope = {'operation_id': 'op-' + 'b' * 32, 'arguments': {'narrative': {'body': '我记下她这一次说累了，希望我抱抱她。'}},
                'context': {'conversation_id': 'conv', 'request_id': 'request', 'turn_id': 'evt-3', 'sources': sources}}
    return env, engine, envelope, calls


def test_submit_is_durable_without_model_or_adopted_memory_and_replays_readonly(tmp_path):
    env, engine, envelope, calls = setup(tmp_path)
    writer = ChatMemoryTool(engine, env['buckets'])
    first = asyncio.run(writer.submit(envelope))
    second = asyncio.run(writer.submit(envelope))
    assert first == second and first['memory_status'] == 'preparing'
    assert first['submission_status'] == 'accepted'
    assert not calls and env['authority'].list_memories() == []
    task = ChatMemoryPreparation(env['authority']).get(candidate_id(envelope['operation_id']))
    assert task['envelope'] == envelope


@pytest.mark.parametrize('compact', [False, True])
def test_background_prepares_full_l1_preserves_body_and_uses_real_message_evidence(tmp_path, compact):
    env, engine, envelope, calls = setup(tmp_path, compact=compact)
    tasks = ChatMemoryPreparation(env['authority'])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert result['results'][0]['memory_status'] == 'pending'
    row = env['authority'].get_candidate(candidate_id(envelope['operation_id']))
    assert row['proposal']['proposed_body'] == envelope['arguments']['narrative']['body']
    l1 = row['proposal']['metadata']['memory_l1']
    assert l1['statements'][0]['statement_id'].startswith('st_')
    assert l1['statements'][0]['evidence_refs'] == ['source:2']
    assert len(l1['evidence_bindings']) == 3
    assert l1['generation']['actor'] == 'chat_memory_preparation'
    assert len(calls) == 1
    assert calls[0]['narrative_from_draft'] is True
    assert 'semantic_contract' in calls[0]
    assert not {'document_schema', 'document_template', 'catalog'}.intersection(calls[0])
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert len(calls) == 1


def test_auto_goes_through_original_commit_not_just_prepared(tmp_path):
    env, engine, envelope, calls = setup(tmp_path, mode='auto')
    tasks = ChatMemoryPreparation(env['authority'])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert result['results'][0]['memory_status'] == 'committed'
    assert tasks.receipt(candidate_id(envelope['operation_id']))['l1_state'] == 'current'


def test_invalid_semantics_keeps_original_draft_and_field_issue(tmp_path):
    env, engine, envelope, calls = setup(tmp_path, bad=True)
    tasks = ChatMemoryPreparation(env['authority'])
    tasks.enqueue(envelope)
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    result = tasks.receipt(candidate_id(envelope['operation_id']))
    assert result['memory_status'] == 'preparation_failed'
    assert result['issue']['code'] == 'unknown_catalog_value'
    assert result['issue']['field_path'] == 'statements[0].subject_ref.value'
    assert tasks.detail(result['task_id'])['draft'] == envelope['arguments']
    assert env['authority'].list_candidates(status='all') == []


def test_owner_edit_requeues_with_history_and_old_revision_cannot_overwrite(tmp_path):
    env, engine, envelope, calls = setup(tmp_path)
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    edit = {'expected_revision': 1, 'request_id': 'edit1', 'draft': {'narrative': {'body': '主人编辑的完整稿。'}}}
    changed = tasks.edit(key, edit)
    assert changed['task']['draft_revision'] == 2 and changed['history'][0]['draft'] == envelope['arguments']
    assert tasks.edit(key, edit) == changed
    with pytest.raises(L1OwnerError, match='submission_revision_conflict'):
        tasks.edit(key, {**edit, 'request_id': 'edit2'})


def test_owner_edit_while_model_runs_does_not_adopt_stale_output(tmp_path):
    env, engine, envelope, calls = setup(tmp_path)
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    original = engine._daily_chat_memory_create_completion
    async def create(*args, **kw):
        current = tasks.get(key)
        tasks.edit(key, {'expected_revision': current['revision'], 'request_id': 'edit-during-model',
                         'draft': {'narrative': {'body': '新的主人版本。'}}})
        return await original(*args, **kw)
    engine._daily_chat_memory_create_completion = create
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert tasks.get(key)['phase'] == 'queued'
    assert env['authority'].list_candidates(status='all') == []


def test_revoked_source_prevents_model_and_adoption(tmp_path):
    env, engine, envelope, calls = setup(tmp_path)
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    async def revoked(binding):
        return {'state': 'revoked', 'eligible_source': False}
    from memory_l0_sources import MemoryL0Sources
    engine._memory_l1_sources = lambda: NS(locate=revoked,
        retain_copy=MemoryL0Sources(env['authority']).retain_copy)
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert tasks.receipt(key)['issue']['code'] == 'source_not_currently_readable'
    assert not calls


def test_restart_can_recover_an_expired_generation_lease(tmp_path):
    env, engine, envelope, calls = setup(tmp_path)
    tasks = ChatMemoryPreparation(env['authority'])
    key = tasks.enqueue(envelope)['task_id']
    tasks.claim(tasks.get(key), lease_seconds=-1)
    asyncio.run(prepare_chat_tasks(engine, env['buckets']))
    assert tasks.receipt(key)['memory_status'] == 'pending'
    assert len(calls) == 1
