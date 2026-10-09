import asyncio
import copy

import pytest

from chat_memory_tool import ChatMemoryTool, candidate_id, receipt
from memory_authority import IdempotencyConflict
from memory_narrative import memory_context_labels
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config, attach_fake_commit_service


def envelope(**changes):
    value = {'operation_id': 'op-' + 'a' * 32,
             'arguments': {'title': '一次回忆', 'content': '我听她说，她在昨天搬了家，没有答应长期住那里。',
                           'kind': 'shared_experience', 'confidence': 0.9,
                           'event_time': {'expression': '昨天', 'evidence': '我昨天搬了家。'}},
             'context': {'conversation_id': 'c', 'request_id': 'r', 'turn_id': 'evt-1',
                         'sources': [{'event_id': 'evt-1', 'version_id': 'v1', 'role': 'user',
                                      'text': '我昨天搬了家。没有答应长期住那里。', 'created_at': '2026-10-08T17:00:00Z'}]}}
    value['arguments'].update(changes)
    return value


def service(tmp_path, mode='auto'):
    engine = ReflectionEngine(reflection_config(tmp_path, mode=mode))
    projection = attach_fake_commit_service(engine)
    return ChatMemoryTool(engine, object()), projection


def test_auto_body_time_provenance_and_same_operation_replay(tmp_path):
    tool, projection = service(tmp_path)
    async def run():
        value = envelope()
        result = await tool.submit(value)
        assert result['memory_status'] == 'committed'
        assert (await tool.submit(value))['memory_id'] == result['memory_id']
        assert len(projection.revisions) == 1
        row = tool.store.get_candidate(result['memory_id'])
        assert row['proposal']['owner_explicit'] is False
        assert row['proposal']['proposed_body'] == value['arguments']['content']
        narrative = row['proposal']['metadata']['narrative']
        assert narrative['generation_kind'] == 'chat_authored'
        assert narrative['event_time']['precision'] == 'expression'
        assert narrative['event_time']['reference_date'] == '2026-10-09'
        assert narrative['sources'][0]['version_id'] == 'v1'
        assert 'conversation_turn:' not in str(narrative)
        with pytest.raises(IdempotencyConflict):
            await tool.submit(envelope(content='different'))
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['review', 'off'])
def test_policy_not_bypassed(tmp_path, mode):
    tool, projection = service(tmp_path, mode)
    result = asyncio.run(tool.submit(envelope()))
    assert result['memory_status'] == ('pending' if mode == 'review' else 'rejected')
    assert result['memory_id'] is None and not projection.revisions


def test_review_existing_entry_confirms_without_truncating(tmp_path):
    tool, projection = service(tmp_path, 'review')
    content = '我保留完整经历，不把它变成一条短事实。' * 120
    async def run():
        result = await tool.submit(envelope(content=content))
        assert tool.engine._load_daily_chat_memory_pending()[0]['candidate']['content'] == content
        confirmed = await tool.engine.confirm_daily_chat_memory([result['candidate_id']], tool.bucket_mgr, request_id='owner-confirm')
        assert confirmed['created'] == 1
        assert receipt(tool.store, result['candidate_id'])['memory_status'] == 'committed'
        assert tool.store.get_candidate(result['candidate_id'])['proposal']['proposed_body'] == content
    asyncio.run(run())


def test_owner_edit_and_unknown_time(tmp_path):
    tool, projection = service(tmp_path, 'review')
    async def run():
        result = await tool.submit(envelope(event_time={'expression': '2099年1月1日', 'evidence': 'not in source'}))
        row = tool.store.get_candidate(result['candidate_id'])
        assert row['proposal']['metadata']['narrative']['event_time']['precision'] == 'unknown'
        await tool.confirm(row, edit={'content': '主人修改后的正文'}, request_id='edit')
        assert receipt(tool.store, result['candidate_id'])['body_preserved'] is False
        assert tool.store.get_candidate(result['candidate_id'])['proposal']['proposed_body'] == '主人修改后的正文'
    asyncio.run(run())


@pytest.mark.parametrize('field', ['owner_explicit', 'source_status', 'narrative', 'source_refs'])
def test_model_cannot_forge_authority(tmp_path, field):
    tool, projection = service(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(tool.submit(envelope(**{field: True})))
    assert not projection.revisions


def test_sensitive_and_identity_go_to_existing_review(tmp_path):
    tool, projection = service(tmp_path)
    assert asyncio.run(tool.submit(envelope(kind='identity')))['memory_status'] == 'pending'
    assert not projection.revisions


def test_review_display_source_preview_is_read_only_and_revisioned(tmp_path):
    tool, projection = service(tmp_path, 'review')
    async def run():
        result = await tool.submit(envelope())
        before = tool.store.get_candidate(result['candidate_id'])
        display = tool.engine.list_daily_chat_memory_pending()[0]['display']
        assert display['confirm_blocked'] is False and display['has_source_preview'] is True
        preview = await tool.engine.daily_chat_memory_source_preview(result['candidate_id'])
        assert preview['source_authority'] == 'aizizhu_canonical_snapshot'
        assert preview['events'][0]['version_id'] == 'v1'
        assert tool.store.get_candidate(result['candidate_id']) == before
    asyncio.run(run())


def test_crash_after_memory_commit_reconciles_same_revision_only_on_owner_action(tmp_path, monkeypatch):
    tool, projection = service(tmp_path)
    real = tool.store.decide_candidate
    def crash(*args, **kwargs):
        if kwargs.get('action') == 'commit':
            raise RuntimeError('simulated receipt crash')
        return real(*args, **kwargs)
    monkeypatch.setattr(tool.store, 'decide_candidate', crash)
    async def run():
        with pytest.raises(RuntimeError):
            await tool.submit(envelope())
        result = await tool.submit(envelope())
        assert result['memory_status'] == 'accepted'
        assert len(projection.revisions) == 1
        monkeypatch.setattr(tool.store, 'decide_candidate', real)
        await tool.confirm(tool.store.get_candidate(result['candidate_id']), edit=None, request_id='recover')
        assert receipt(tool.store, result['candidate_id'])['memory_status'] == 'committed'
        assert len(projection.revisions) == 1
    asyncio.run(run())


def test_server_auth_read_status_and_validation(tmp_path, monkeypatch):
    import json
    import server
    from starlette.requests import Request
    tool, projection = service(tmp_path, 'review')
    monkeypatch.setattr(server, 'reflection_engine', tool.engine)
    monkeypatch.setattr(server, 'memory_authority_store', tool.store)
    monkeypatch.setattr(server, 'bucket_mgr', tool.bucket_mgr)
    monkeypatch.setattr(server, '_memory_write_token', lambda: 'test-token')
    def request(method='POST', body=None, authorized=True):
        async def receive():
            return {'type': 'http.request', 'body': json.dumps(body or envelope()).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': method, 'path': '/',
                        'path_params': {'operation_id': envelope()['operation_id']},
                        'headers': [(b'authorization', b'Bearer test-token')] if authorized else []}, receive)
    async def run():
        assert (await server.api_chat_memory_tool(request(authorized=False))).status_code == 401
        assert (await server.api_chat_memory_tool(request(body={'owner_explicit': True}))).status_code == 400
        result = await server.api_chat_memory_tool(request())
        assert result.status_code == 200 and result.headers['Cache-Control'] == 'no-store'
        response = await server.api_chat_memory_tool(request('GET'))
        assert json.loads(response.body)['memory_status'] == 'pending'
        assert b'content' not in response.body and not projection.revisions
    asyncio.run(run())


def test_proven_not_applied_failure_can_be_explicitly_retried(tmp_path):
    tool, projection = service(tmp_path)
    projection.fail_revision = True
    async def run():
        result = await tool.submit(envelope())
        assert result['memory_status'] == 'commit_failed'
        assert not projection.revisions
        projection.fail_revision = False
        await tool.confirm(tool.store.get_candidate(result['candidate_id']), edit=None, request_id='owner-retry')
        assert receipt(tool.store, result['candidate_id'])['memory_status'] == 'committed'
        assert len(projection.revisions) == 1
    asyncio.run(run())
