"""Exercise the actual owner endpoint without importing server startup."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('payload,memory_force,summary_force', [
    ({'date': '2026-10-17', 'retry_failed_windows': True}, True, False),
    ({'date': '2026-10-17'}, False, False),
    ({'date': '2026-10-17', 'force': True}, True, True),
])
def test_failed_window_retry_targets_memory_without_forcing_summary(payload, memory_force, summary_force):
    tree = ast.parse((Path(__file__).parents[1] / 'server.py').read_text(encoding='utf-8'))
    endpoint = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == 'api_daily_chat_memory_run')
    endpoint.decorator_list = []
    calls = {}
    async def memory(*args, **kwargs):
        calls['memory'] = kwargs
        return {'status': 'pending', 'date': '2026-10-17', 'candidates': []}
    async def summary(**kwargs):
        calls['summary'] = kwargs
        return {'status': 'skipped'}
    async def impression(date): return None
    async def body(): return payload
    scope = {'_require_dashboard_auth': lambda request: None,
        '_bool_value': lambda value, default: value if isinstance(value, bool) else default,
        'reflection_engine': SimpleNamespace(run_daily_chat_memory=memory, run_daily_activity_summary=summary),
        'bucket_mgr': None, 'gateway_state_store': None, 'raw_event_store': None,
        'persona_engine': None, 'embedding_engine': None,
        '_daily_impression_material_for_date': impression,
        '_store_daily_activity_summary_result': lambda value: value,
        'logger': SimpleNamespace(warning=lambda *args: None)}
    exec(compile(ast.Module(body=[endpoint], type_ignores=[]), 'server-retry-route', 'exec'), scope)
    response = asyncio.run(scope['api_daily_chat_memory_run'](SimpleNamespace(json=body)))
    assert response.status_code == 200
    assert calls['memory']['key'] == '2026-10-17'
    assert calls['memory']['force'] is memory_force
    assert calls['summary']['force'] is summary_force
