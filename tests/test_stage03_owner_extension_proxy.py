"""Real owner routes with a local fake Brain; no provider or live writes."""
import asyncio
import ast
import json
from pathlib import Path

import httpx
import pytest

from gateway import GatewayService, create_gateway_app
from prompt_source_registry import fixed_prompt_source, _MEMORY_L3_SYNC_PROMPT


ROUTES = [
    ('GET', '/presentation/sync-proposals', None),
    ('POST', '/presentation/sync-proposals/2/prepare', {'expected_revision': 1, 'request_id': 'prepare'}),
    ('POST', '/presentation/sync-proposals/2/apply', {'expected_revision': 2, 'request_id': 'apply'}),
    ('POST', '/presentation/sync-proposals/2/retry', {'expected_revision': 3, 'request_id': 'retry'}),
    ('GET', '/evolution?offset=0&limit=50', None),
    ('POST', '/evolution/notes', {'expected_revision': 1, 'request_id': 'note', 'content': '完整合成说明。'}),
    ('POST', '/evolution/notes/ring-1/retract', {'expected_revision': 1, 'request_id': 'retract', 'expected_content_sha256': 'a' * 64}),
    ('POST', '/evolution/followups/task-1/retry', {'expected_revision': 1, 'request_id': 'followup-retry'}),
]


def harness(monkeypatch):
    original_client = httpx.AsyncClient
    service = GatewayService.__new__(GatewayService)
    service.gateway_token = 'synthetic-owner-token'
    app = create_gateway_app(config={'gateway': {}}, service=service)
    app.state.gateway_service = service
    incoming = original_client(transport=httpx.ASGITransport(app=app), base_url='http://gateway.test')
    seen = []
    def brain(request):
        seen.append(request)
        return httpx.Response(200, json={'status': 'ok', 'memory_id': 'memory-1', 'receipt': {'exact': True}})
    monkeypatch.setenv('OMBRE_MEMORY_INDEX_ADMIN_URL', 'http://brain.test')
    monkeypatch.setenv('OMBRE_MEMORY_WRITE_TOKEN', 'synthetic-internal-token')
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original_client(transport=httpx.MockTransport(brain), **kwargs))
    return incoming, seen


@pytest.mark.parametrize('method,suffix,payload', ROUTES)
def test_owner_routes_preserve_exact_receipts_and_fixed_paths(monkeypatch, method, suffix, payload):
    client, seen = harness(monkeypatch)
    path = '/api/memory-authority/memories/memory-1' + suffix
    async def scenario():
        async with client:
            denied = await client.request(method, path, json=payload)
            assert denied.status_code == 401 and seen == []
            response = await client.request(method, path, json=payload, headers={'Authorization': 'Bearer synthetic-owner-token'})
            assert response.status_code == 200 and response.json()['receipt'] == {'exact': True}
            assert response.headers['cache-control'] == 'no-store'
        assert len(seen) == 1 and seen[0].url.host == 'brain.test'
        assert seen[0].url.path == path.split('?', 1)[0]
        assert seen[0].headers['authorization'] == 'Bearer synthetic-internal-token'
        if payload is not None:
            assert json.loads(seen[0].content) == payload
    asyncio.run(scenario())


@pytest.mark.parametrize('suffix,payload', [
    ('/presentation/sync-proposals/2/apply', {'expected_revision': 1, 'request_id': 'x', 'actor': 'owner'}),
    ('/presentation/sync-proposals/2/not-allowed', {'expected_revision': 1, 'request_id': 'x'}),
    ('/evolution/notes', {'expected_revision': True, 'request_id': 'x', 'content': 'body'}),
    ('/evolution/notes', {'expected_revision': 1, 'request_id': 'x', 'content': 'body', 'arbitrary_url': 'http://elsewhere.test'}),
    ('/evolution/followups/task-1/retry', {'expected_revision': 1, 'request_id': 'x', 'content': 'cannot write here'}),
])
def test_unapproved_proxy_fields_never_leave_gateway(monkeypatch, suffix, payload):
    client, seen = harness(monkeypatch)
    async def scenario():
        async with client:
            response = await client.post('/api/memory-authority/memories/memory-1' + suffix,
                json=payload, headers={'Authorization': 'Bearer synthetic-owner-token'})
            assert response.status_code == 400 and seen == []
    asyncio.run(scenario())


def test_l3_source_spec_and_factory_body_are_exact_scoped_mirror():
    spec = fixed_prompt_source('ombre.memory_l3_sync_prompt')
    assert spec.scope == 'memory.reflection'
    source = Path(__file__).resolve().parents[2] / 'stage03-confirmed-brain' / 'prompt_source_registry.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == '_MEMORY_L3_SYNC_PROMPT' for target in node.targets))
    assert _MEMORY_L3_SYNC_PROMPT == ast.literal_eval(assignment.value)
