"""Owner diary recovery transport and independent read-wait configuration; offline only."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.responses import JSONResponse

from config_activation import activation_receipt, credential_edit_revision
from memory_l1_runtime import task_options
from memory_system_config import merge_config, read_overlay, update_overlay
from test_memory_write_quality_routes import (
    FakeRequest,
    config_api,
    install_temporary_config_runtime,
    response_json,
)


@pytest.fixture
def diary_api(config_api, monkeypatch, tmp_path):
    server, config = config_api.server, config_api.config
    config['reflection']['daily_chat_memory_timeout_seconds'] = 222
    engine, _authority, _commit, clients = install_temporary_config_runtime(config_api, monkeypatch, tmp_path)
    credential_edit_revision(config)

    async def no_gateway(_payload, **_kwargs):
        return activation_receipt(state='not_configured', reason='synthetic-no-gateway')

    monkeypatch.setattr(server, '_hot_update_gateway_config', no_gateway)
    return SimpleNamespace(**vars(config_api), engine=engine, clients=clients)


def test_diary_wait_is_inherited_then_saved_and_reloaded_without_changing_task_options(diary_api):
    server, config = diary_api.server, diary_api.config
    initial = response_json(asyncio.run(server.api_config_get(FakeRequest())))['reflection']
    assert initial['diary_l1_read_timeout_seconds'] == 222
    assert initial['diary_l1_read_timeout_inherited'] is True
    original_options = task_options(config)
    original_routes = deepcopy(initial['memory_task_routes'])
    update_overlay(diary_api.runtime_config_path, lambda _data: {'untouched': {'keep': 19}})

    for seconds in (1, 3600, 360):
        response = asyncio.run(server.api_config_update(FakeRequest({
            'reflection': {'diary_l1_read_timeout_seconds': seconds}, 'persist': True})))
        assert response.status_code == 200, response.body
        assert response_json(response)['persistence']['runtime_yaml'] == 'saved'
        public = response_json(asyncio.run(server.api_config_get(FakeRequest())))['reflection']
        assert public['diary_l1_read_timeout_seconds'] == seconds
        assert public['diary_l1_read_timeout_inherited'] is False
        assert public['memory_task_revision'] == initial['memory_task_revision']
        assert public['memory_task_routes'] == original_routes
        assert task_options(config) == original_options
        assert server.reflection_engine.daily_chat_memory_timeout_seconds == 222
        saved = read_overlay(diary_api.runtime_config_path)
        assert saved == {'untouched': {'keep': 19}, 'reflection': {'diary_l1_read_timeout_seconds': seconds}}
        reloaded = merge_config(config, saved)
        assert reloaded['reflection']['diary_l1_read_timeout_seconds'] == seconds
    assert all(not client.calls for client in diary_api.clients)


@pytest.mark.parametrize('value', [True, False, None, '360', 0, -1, 3601, float('nan'), float('inf')])
def test_invalid_diary_wait_is_rejected_before_any_config_mutation(diary_api, value):
    server, config = diary_api.server, diary_api.config
    before, engine = deepcopy(config), server.reflection_engine
    response = asyncio.run(server.api_config_update(FakeRequest({
        'dehydration': {'model': 'must-not-be-applied'},
        'reflection': {'diary_l1_read_timeout_seconds': value}, 'persist': True})))
    assert response.status_code == 400
    assert response_json(response)['field_path'] == 'reflection.diary_l1_read_timeout_seconds'
    assert config == before and server.reflection_engine is engine
    assert not diary_api.runtime_config_path.exists()


def test_diary_wait_persist_failure_restores_previous_runtime(diary_api, monkeypatch):
    import memory_system_config
    before = deepcopy(diary_api.config)
    engine = diary_api.server.reflection_engine

    def fail_overlay(*_args, **_kwargs):
        raise OSError('synthetic unavailable file')

    monkeypatch.setattr(memory_system_config, 'update_overlay', fail_overlay)
    response = asyncio.run(diary_api.server.api_config_update(FakeRequest({
        'reflection': {'diary_l1_read_timeout_seconds': 360}, 'persist': True,
        'expected_credential_revision': diary_api.config['_credential_edit_revision']})))
    assert response.status_code == 500
    assert response_json(response)['error'] == 'runtime_persist_failed'
    assert diary_api.config == before
    assert diary_api.server.reflection_engine is engine


def retry_engine(monkeypatch, server, result=None):
    call = AsyncMock(return_value=result or {'status': 'skipped', 'reason': 'model_outcome_unknown_requires_retry_decision'})
    engine = SimpleNamespace(memory_authority_enabled=True, memory_authority_store=object(),
        _local_now=lambda: datetime(2026, 11, 6, 12, tzinfo=timezone.utc),
        _maybe_extract_diary_memory=call)
    monkeypatch.setattr(server, 'reflection_engine', engine)
    return call


def test_plain_retry_keeps_unknown_decision_explicit_and_known_failure_retry_enabled(config_api, monkeypatch):
    call = retry_engine(monkeypatch, config_api.server)
    response = asyncio.run(config_api.server.api_diary_memory_retry(FakeRequest({'date': '2026-11-01'})))
    assert response.status_code == 200
    assert response_json(response)['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert call.await_args.kwargs == {'retry_failed': True, 'retry_timed_out': None}


def test_exact_timeout_retry_passes_identity_and_exposes_safe_diagnostics_only(config_api, monkeypatch):
    selection = {'task_id': 'diary-task-known', 'checkpoint_revision': 7}
    diagnostic = {'window': 2, 'reason': 'invalid_request', 'code': 'BadRequestError',
        'field_path': 'text.format', 'http_status': 400, 'provider_param': 'text.format',
        'provider_trace_id': 'trace-synthetic', 'upstream_request_id': 'req-synthetic',
        'provider_category': 'invalid_request_error', 'finish_reason': 'stop'}
    private = {'content': 'PRIVATE SOURCE TEXT', 'message': 'PRIVATE PROVIDER BODY', 'response_body': 'PRIVATE BODY'}
    call = retry_engine(monkeypatch, config_api.server, {
        'status': 'failed', 'reason': 'no_complete_first_pass_window',
        'task_id': selection['task_id'], 'checkpoint_revision': 8,
        'code': diagnostic['code'], 'field_path': diagnostic['field_path'],
        'issue': {**diagnostic, **private}, 'failures': [{**diagnostic, **private}], **private})
    response = asyncio.run(config_api.server.api_diary_memory_retry(FakeRequest({
        'date': '2026-11-01', 'retry_timed_out': selection})))
    receipt = response_json(response)
    assert call.await_args.kwargs == {'retry_failed': True, 'retry_timed_out': selection}
    assert receipt['task_id'] == selection['task_id'] and receipt['checkpoint_revision'] == 8
    assert receipt['issue'] == diagnostic and receipt['failures'] == [diagnostic]
    assert receipt['code'] == 'BadRequestError' and receipt['field_path'] == 'text.format'
    assert b'PRIVATE' not in response.body


@pytest.mark.parametrize('selection', [None, True, {}, {'task_id': 'x'},
    {'task_id': '', 'checkpoint_revision': 1}, {'task_id': 7, 'checkpoint_revision': 1},
    {'task_id': 'x', 'checkpoint_revision': True}, {'task_id': 'x', 'checkpoint_revision': 0},
    {'task_id': 'x', 'checkpoint_revision': '1'}, {'task_id': 'x', 'checkpoint_revision': 1.0},
    {'task_id': 'x', 'checkpoint_revision': 1, 'force': True}])
def test_timeout_retry_rejects_malformed_selection_without_extraction(config_api, monkeypatch, selection):
    call = retry_engine(monkeypatch, config_api.server)
    response = asyncio.run(config_api.server.api_diary_memory_retry(FakeRequest({
        'date': '2026-11-01', 'retry_timed_out': selection})))
    assert response.status_code == 400
    assert response_json(response)['code'] == 'invalid_retry_timed_out'
    assert response_json(response)['field_path'].startswith('retry_timed_out')
    call.assert_not_awaited()


@pytest.mark.parametrize('date', ['2026-11-06', '2026-11-07', '2026-11-31', '2026-1-1', None])
def test_retry_still_requires_a_completed_calendar_date(config_api, monkeypatch, date):
    call = retry_engine(monkeypatch, config_api.server)
    response = asyncio.run(config_api.server.api_diary_memory_retry(FakeRequest({'date': date})))
    assert response.status_code == 400
    call.assert_not_awaited()


def test_diary_retry_requires_owner_auth_before_reading_or_executing(config_api, monkeypatch):
    call = retry_engine(monkeypatch, config_api.server)
    monkeypatch.setattr(config_api.server, '_require_dashboard_auth',
                        lambda _request: JSONResponse({'error': 'unauthorized'}, status_code=401))
    response = asyncio.run(config_api.server.api_diary_memory_retry(FakeRequest({'date': '2026-11-01'})))
    assert response.status_code == 401
    call.assert_not_awaited()
