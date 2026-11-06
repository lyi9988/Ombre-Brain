"""Production handler/auth/Review dispatch over isolated real Authority state."""
import ast
import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace as NS

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from test_memory_legacy_backfill import setup


def run(task):
    return asyncio.run(task)


def app_for(service, config):
    base = Path(__file__).resolve().parents[1]
    engine_source = ast.parse((base / 'reflection_engine.py').read_text(encoding='utf-8'))
    engine_class = next(node for node in engine_source.body if isinstance(node, ast.ClassDef) and node.name == 'ReflectionEngine')
    names = {'confirm_daily_chat_memory', '_confirm_daily_chat_memory_authority', '_authority_candidate_legacy_item'}
    methods = [node for node in engine_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    engine_ns = {'datetime': datetime, 'timezone': timezone, 'hashlib': hashlib}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(base / 'reflection_engine.py'), 'exec'), engine_ns)
    engine = NS(config=config, memory_authority_enabled=True, memory_authority_store=service.authority,
        _memory_l1_revisions=lambda manager: service.revisions, _memory_l1_sources=service.source_reader_factory,
        _request_ledger_lookup=lambda rid: None, _request_ledger_record=lambda rid, result: None,
        _confirm_rate_limit_ok=lambda now: True)
    for name in names:
        setattr(engine, name, MethodType(engine_ns[name], engine))
    source = ast.parse((base / 'server.py').read_text(encoding='utf-8'))
    names = {'api_memory_legacy_backfill', 'api_daily_chat_memory_confirm', '_require_owner_or_internal_index_maintenance',
        '_dashboard_authenticated', '_require_dashboard_auth', '_memory_write_token', '_authorized_memory_write'}
    functions, routes = [], []
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            for dec in node.decorator_list:
                routes.append((ast.literal_eval(dec.args[0]), node.name,
                    ast.literal_eval(next(k.value for k in dec.keywords if k.arg == 'methods'))))
            node.decorator_list = []
            functions.append(node)
    namespace = {'config': config, 'reflection_engine': engine, 'memory_authority_store': service.authority,
        'memory_commit_service': service.revisions.commit, 'bucket_mgr': service.revisions.buckets,
        '_memory_l1_revision_service': lambda: service.revisions, '_dashboard_sessions': {'owner-cookie': time.time() + 60},
        '_dashboard_setup_needed': lambda: False, 'time': time, 'os': os, 'hmac': hmac, 'embedding_engine': None}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(base / 'server.py'), 'exec'), namespace)
    return Starlette(routes=[Route(path, namespace[name], methods=methods) for path, name, methods in routes])


def test_api_auth_stage_readback_and_original_review_confirm(tmp_path, monkeypatch):
    service, payload, config = setup(tmp_path)
    monkeypatch.setenv('OMBRE_MEMORY_WRITE_TOKEN', 'internal-test-token')
    app = app_for(service, config)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            url = '/api/memory-authority/memories/m1/legacy-backfill'
            assert (await client.post(url, json=payload)).status_code == 401
            assert (await client.post(url, json=payload, headers={'Authorization': 'Bearer wrong'})).status_code == 401
            staged = await client.post(url, json=payload, headers={'Authorization': 'Bearer internal-test-token'})
            assert staged.status_code == 200
            body = staged.json()
            assert body['adopted'] is False and body['base_revision'] == 1
            assert service.authority.get_memory('m1')['active_revision'] == 1
            readback = await client.get(url, headers={'Authorization': 'Bearer internal-test-token'})
            assert readback.json()['candidate']['candidate_id'] == body['candidate_id']
            confirm = {'candidate_ids': [body['candidate_id']], 'action': 'confirm', 'confirm': 'WRITE',
                'expected_revisions': {body['candidate_id']: body['candidate']['revision']}, 'request_id': 'approve-via-original-review'}
            assert (await client.post('/api/daily-chat-memory/confirm', json=confirm,
                headers={'Authorization': 'Bearer internal-test-token'})).status_code == 401
            client.cookies.set('ombre_session', 'owner-cookie')
            result = await client.post('/api/daily-chat-memory/confirm', json=confirm)
            assert result.status_code == 200
            assert result.json()['results'][0]['memory_id'] == 'm1'
            assert service.authority.get_memory('m1')['active_revision'] == 2
            assert service.authority.get_memory(body['candidate_id']) is None
    run(scenario())
