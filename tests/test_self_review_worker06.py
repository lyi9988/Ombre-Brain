"""Original worker and config reload keep the real delivery validator."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_l3_followup import MemoryFollowups
from test_memory_authority_r1 import reflection_config
from test_background_reflection_runtime import sdk_transport
from test_memory_write_quality_routes import config_api, FakeRequest


def test_actual_worker_rebinds_resolver_to_each_loop_owned_replacement(tmp_path, monkeypatch):
    import memory_browse_preparation
    from bucket_manager import BucketManager
    config = reflection_config(tmp_path, mode='auto')
    requests, clients = sdk_transport(monkeypatch)
    seen, errors = [], []

    async def resolver(context, turn):
        return {'state': 'available', 'eligible_fragment_ids': ['synthetic-fragment']}

    async def prepare(engine, buckets):
        handler = MemoryFollowups(engine, buckets)
        assert handler.resolver is resolver
        assert (await handler.resolver({}, {}))['eligible_fragment_ids'] == ['synthetic-fragment']
        seen.append(engine)
        config.setdefault('reflection', {})['model'] = 'different-synthetic-model'

    async def next_batch(_):
        if len(seen) == 2 or errors:
            raise asyncio.CancelledError()

    monkeypatch.setattr(memory_browse_preparation, 'prepare_memory_tasks', prepare)
    # Run the unmodified production worker function, with only its process
    # surroundings isolated. BackgroundReflectionRuntime/engines are real.
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'server.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_chat_memory_preparation_loop')
    scope = {'config': config, 'bucket_mgr': BucketManager(config), 'prompt_plan_mirror': None,
        'reflection_configuration_source': None, '_resolve_memory_reaction_context': resolver,
        'logger': SimpleNamespace(warning=lambda *args: errors.append(args)),
        'asyncio': SimpleNamespace(sleep=next_batch),
        'task_options': lambda _: {'chat_memory_prepare_interval_seconds': 0}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original-worker>', 'exec'), scope)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await scope['_chat_memory_preparation_loop']()
        assert not errors and len(seen) == 2 and seen[0] is not seen[1]
        assert not requests
        for client in clients:
            await client.close()
    asyncio.run(run())


def test_owner_config_reload_keeps_followup_delivery_resolver(config_api, monkeypatch):
    server = config_api.server
    requests, clients = sdk_transport(monkeypatch)

    async def no_gateway(_payload, **_kwargs):
        return server.activation_receipt(state='not_configured', reason='synthetic-no-gateway')

    monkeypatch.setattr(server, '_hot_update_gateway_config', no_gateway)

    async def run():
        old = server.reflection_engine
        response = await server.api_config_update(FakeRequest({'reflection': {'model': 'synthetic-reloaded-model'}}))
        assert response.status_code == 200, response.body
        assert server.reflection_engine is not old
        assert server.reflection_engine.memory_followup_resolver is server._resolve_memory_reaction_context
        assert not requests
        for client in clients:
            await client.close()
    asyncio.run(run())
