import asyncio
import copy
import json

import pytest

from memory_authority import IdempotencyConflict, InvalidTransition, RevisionConflict
from memory_candidate_edit import save_owner_candidate
from reflection_engine import ReflectionEngine
from test_chat_memory_tool import envelope, service
from test_memory_authority_r1 import reflection_config, attach_fake_commit_service, daily_candidate


def daily(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    projection = attach_fake_commit_service(engine)
    engine._store_daily_chat_memory_pending([{**daily_candidate(), 'mode': 'review', 'status': 'pending'}])
    return engine, projection, 'daily-1'


def save(engine, ident, edit, revision=1, request='save-1'):
    return save_owner_candidate(engine.memory_authority_store, ident, edit=edit,
                                expected_revision=revision, request_id=request)


def test_save_keeps_pending_source_body_and_policy_then_explicitly_adopts(tmp_path):
    engine, projection, ident = daily(tmp_path)
    before = engine.memory_authority_store.get_candidate(ident)
    content = '主人自己的完整修订。' * 180
    row = save(engine, ident, {'content': content, 'tags': ['标签'], 'event_date': '2026-10-09'})
    assert row['status'] == 'pending' and row['revision'] == 2 and not projection.revisions
    assert row['proposal']['proposed_body'] == content
    for key in ('source_refs', 'original_excerpt', 'source_status', 'requested_mode', 'owner_explicit'):
        assert row['proposal'][key] == before['proposal'][key]
    result = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='adopt', expected_revisions={ident: 2}))
    assert result['created'] == 1
    assert engine.memory_authority_store.get_candidate(ident)['proposal']['proposed_body'] == content
    assert len(projection.revisions) == 1


def test_chat_save_uses_same_adapter_without_losing_body_kind_or_arrays(tmp_path):
    tool, projection = service(tmp_path, 'review')
    original = asyncio.run(tool.submit(envelope()))
    ident = original['candidate_id']
    row = save(tool.engine, ident, {'content': '主人保留的原文', 'kind': 'boundary', 'tags': ['不催'], 'domain': []})
    assert row['proposal']['memory_type'] == 'boundary'
    assert row['proposal']['metadata']['legacy_candidate']['kind'] == 'boundary'
    assert row['proposal']['metadata']['legacy_candidate']['tags'] == ['不催']
    assert not projection.revisions
    result = asyncio.run(tool.engine.confirm_daily_chat_memory([ident], object(), request_id='adopt', expected_revisions={ident: 2}))
    assert result['created'] == 1 and tool.store.get_candidate(ident)['status'] == 'committed'


def test_title_only_and_unchanged_body_preserve_time(tmp_path):
    tool, _ = service(tmp_path, 'review')
    original = envelope()
    ident = asyncio.run(tool.submit(original))['candidate_id']
    before = tool.store.get_candidate(ident)['proposal']['metadata']['narrative']['event_time']
    row = save(tool.engine, ident, {'title': '只改标题', 'content': original['arguments']['content']})
    assert row['proposal']['metadata']['narrative']['event_time'] == before
    row = save(tool.engine, ident, {'content': '正文发生变化'}, revision=2, request='save-2')
    assert row['proposal']['metadata']['narrative']['event_time']['precision'] == 'unknown'


def test_same_edit_receipt_replays_after_other_edits_and_rejects_changed_payload(tmp_path):
    engine, _, ident = daily(tmp_path)
    first = save(engine, ident, {'title': '第一版'})
    save(engine, ident, {'title': '第二版'}, revision=2, request='save-2')
    assert save(engine, ident, {'title': '第一版'}) == first
    assert engine.memory_authority_store.get_candidate(ident)['revision'] == 3
    with pytest.raises(IdempotencyConflict): save(engine, ident, {'title': '偷换请求'})
    with pytest.raises(RevisionConflict): save(engine, ident, {'title': '旧页面'}, request='old')


def test_stale_confirm_does_not_adopt_new_owner_revision(tmp_path):
    engine, projection, ident = daily(tmp_path)
    save(engine, ident, {'content': '另一个页面保存的修改'})
    result = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='stale', expected_revisions={ident: 1}))
    assert result['results'][0]['status'] == 'revision_conflict'
    assert engine.memory_authority_store.get_candidate(ident)['status'] == 'pending'
    assert not projection.revisions
    replay = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='stale', expected_revisions={ident: 2}))
    assert replay['reason'] == 'idempotency_conflict' and not projection.revisions


def test_concurrent_edit_during_confirm_returns_item_conflict(tmp_path, monkeypatch):
    engine, projection, ident = daily(tmp_path)
    def concurrent(*args, **kwargs):
        raise RevisionConflict('simulated CAS race')
    monkeypatch.setattr(engine.memory_authority_store, 'revise_candidate', concurrent)
    result = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='race',
        expected_revisions={ident: 1}, edits={ident: {'content': '主人本次修改'}}))
    assert result['results'] == [{'id': ident, 'status': 'revision_conflict'}]
    assert not projection.revisions


@pytest.mark.parametrize('edit', [
    {}, {'source_refs': ['fake']}, {'narrative': {}}, {'owner_explicit': True},
    {'title': ''}, {'title': 't' * 101}, {'content': 'c' * 12001}, {'content': '  '},
    {'kind': 'fake'}, {'tags': 'not-an-array'}, {'domain': ['']}, {'confidence': float('nan')},
    {'confidence': 1.1}, {'importance': True}, {'importance': 11}, {'event_date': '2026-02-30'},
])
def test_invalid_edits_do_not_change_candidate(tmp_path, edit):
    engine, projection, ident = daily(tmp_path)
    before = engine.memory_authority_store.get_candidate(ident)
    with pytest.raises((ValueError, TypeError)): save(engine, ident, edit)
    assert engine.memory_authority_store.get_candidate(ident) == before
    assert not projection.revisions


def test_committed_candidate_is_not_editable_as_pending(tmp_path):
    engine, _, ident = daily(tmp_path)
    asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='adopt'))
    row = engine.memory_authority_store.get_candidate(ident)
    with pytest.raises(InvalidTransition): save(engine, ident, {'content': 'wrong path'}, revision=row['revision'])


def test_revisioned_save_does_not_clear_needs_edit_when_body_unchanged(tmp_path):
    engine, _, ident = daily(tmp_path)
    row = engine.memory_authority_store.get_candidate(ident)
    from memory_candidate_edit import owner_edit_payload
    row['proposal']['metadata']['legacy_candidate']['soft_flags'] = ['needs_owner_edit', 'weak_source_support']
    result = owner_edit_payload(row, {'content': row['proposal']['proposed_body']})
    assert 'needs_owner_edit' in result['metadata']['legacy_candidate']['soft_flags']


def test_edit_api_auth_validation_and_saved_receipt(tmp_path, monkeypatch):
    server = pytest.importorskip('server')
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    engine, projection, ident = daily(tmp_path)
    monkeypatch.setattr(server, 'reflection_engine', engine)
    monkeypatch.setattr(server, '_require_dashboard_auth', lambda request:
        None if request.headers.get('x-test-owner') == 'yes' else JSONResponse({'error': 'unauthorized'}, status_code=401))
    def request(body, owner=True):
        async def receive():
            return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': 'POST', 'path': '/',
                        'headers': [(b'x-test-owner', b'yes')] if owner else []}, receive)
    body = {'candidate_id': ident, 'edit': {'title': '独立保存'}, 'expected_revision': 1, 'request_id': 'api-save'}
    async def run():
        assert (await server.api_daily_chat_memory_edit(request(body, False))).status_code == 401
        assert (await server.api_daily_chat_memory_edit(request({**body, 'owner': True}))).status_code == 400
        response = await server.api_daily_chat_memory_edit(request(body))
        assert response.status_code == 200 and response.headers['Cache-Control'] == 'no-store'
        data = json.loads(response.body)
        assert data['status'] == 'saved' and data['adopted'] is False
        assert data['item']['authority_revision'] == 2 and data['item']['status'] == 'pending'
        assert (await server.api_daily_chat_memory_edit(request({**body, 'request_id': 'different'}))).status_code == 409
    asyncio.run(run())
    assert not projection.revisions
