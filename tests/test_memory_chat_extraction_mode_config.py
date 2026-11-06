"""The existing owner settings route saves and reads the opt-in daily mode."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from config_activation import activation_receipt, credential_edit_revision
from memory_system_config import merge_config, read_overlay, update_overlay
from test_memory_write_quality_routes import (
    FakeRequest, config_api, install_temporary_config_runtime, response_json,
)


FIELD = 'daily_chat_memory_extraction_mode'


@pytest.fixture
def extraction_api(config_api, monkeypatch, tmp_path):
    _engine, _authority, _commit, clients = install_temporary_config_runtime(config_api, monkeypatch, tmp_path)
    async def no_gateway(_payload, **_kwargs):
        return activation_receipt(state='not_configured', reason='synthetic-no-gateway')
    monkeypatch.setattr(config_api.server, '_hot_update_gateway_config', no_gateway)
    credential_edit_revision(config_api.config)
    return SimpleNamespace(**vars(config_api), clients=clients)


def test_existing_install_default_then_mode_save_read_reload_and_runtime_config(extraction_api):
    server, config = extraction_api.server, extraction_api.config
    config['reflection'].pop(FIELD, None)
    initial = response_json(asyncio.run(server.api_config_get(FakeRequest())))['reflection']
    assert initial[FIELD] == 'single_pass'
    original_routes = deepcopy(initial['memory_task_routes'])
    original_mode = config['reflection']['daily_chat_memory_mode']
    update_overlay(extraction_api.runtime_config_path, lambda _data: {'untouched': {'keep': 19}})
    for mode in ('event_first', 'single_pass'):
        saved = asyncio.run(server.api_config_update(FakeRequest({'reflection': {FIELD: mode}, 'persist': True})))
        assert saved.status_code == 200, saved.body
        public = response_json(asyncio.run(server.api_config_get(FakeRequest())))['reflection']
        assert public[FIELD] == mode and public['memory_task_routes'] == original_routes
        assert config['reflection']['daily_chat_memory_mode'] == original_mode
        assert server.reflection_engine.config['reflection'][FIELD] == mode
        overlay = read_overlay(extraction_api.runtime_config_path)
        assert overlay == {'untouched': {'keep': 19}, 'reflection': {FIELD: mode}}
        assert merge_config(config, overlay)['reflection'][FIELD] == mode
        snapshot, _credentials, _signature = server.reflection_configuration_source.snapshot(config)
        assert snapshot['reflection'][FIELD] == mode
    assert all(not client.calls for client in extraction_api.clients)


@pytest.mark.parametrize('mode', [None, True, False, {}, [], 'event-first', 'unknown'])
def test_invalid_mode_cannot_partially_change_config_or_write_overlay(extraction_api, mode):
    server, config = extraction_api.server, extraction_api.config
    before, engine = deepcopy(config), server.reflection_engine
    response = asyncio.run(server.api_config_update(FakeRequest({
        'dehydration': {'model': 'must-not-change'}, 'reflection': {FIELD: mode}, 'persist': True})))
    assert response.status_code == 400 and response_json(response)['field_path'] == 'reflection.' + FIELD
    assert config == before and server.reflection_engine is engine
    assert not extraction_api.runtime_config_path.exists()


def test_mode_persistence_failure_restores_the_previous_runtime(extraction_api, monkeypatch):
    import memory_system_config
    before, engine = deepcopy(extraction_api.config), extraction_api.server.reflection_engine
    def fail_overlay(*_args, **_kwargs):
        raise OSError('synthetic unavailable overlay')
    monkeypatch.setattr(memory_system_config, 'update_overlay', fail_overlay)
    failed = asyncio.run(extraction_api.server.api_config_update(FakeRequest({
        'reflection': {FIELD: 'event_first'}, 'persist': True})))
    assert failed.status_code == 500
    assert extraction_api.config == before and extraction_api.server.reflection_engine is engine


def test_background_runtime_refresh_exposes_mode_at_the_next_batch():
    from background_reflection_runtime import BackgroundReflectionRuntime
    class Engine:
        def __init__(self, config, _mirror, **_kwargs):
            self.config = config
        def refresh_daily_chat_memory_options(self, config):
            self.config = config
    async def run():
        runtime = BackgroundReflectionRuntime(engine_factory=Engine)
        config = {'reflection': {FIELD: 'single_pass'}}
        async with runtime.batch(config) as first:
            config['reflection'][FIELD] = 'event_first'
            assert first.config['reflection'][FIELD] == 'single_pass'
        async with runtime.batch(config) as second:
            assert second is first and second.config['reflection'][FIELD] == 'event_first'
        await runtime.close()
    asyncio.run(run())
