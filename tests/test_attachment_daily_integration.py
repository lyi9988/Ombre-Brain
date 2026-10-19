"""Actual local Aiz/Gateway -> ReflectionEngine -> Authority/Bucket/L2 chain.

Only completion responses are synthetic. No provider calls, production state,
or replacement extraction/commit methods are used. Requires the local candidate
worktrees and the documented staging archive fixture.
"""
from datetime import datetime, timezone
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
BRAIN = Path(__file__).resolve().parents[1]
AIZ = ROOT / 'worktrees/integration-aiz-foundations-r1'
GATEWAY = ROOT / 'worktrees/integration-gateway-foundations-r1'
sys.path[:0] = [str(BRAIN), str(AIZ), str(GATEWAY), str(ROOT / 'staging/p4-python-deps')]
spec = importlib.util.spec_from_file_location('attachment_archive_fixture', ROOT / 'staging/integration-foundations-20261019/aiz/run_archive_chain.py')
archive_fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_fixture)
sys.path[:0] = [str(BRAIN), str(AIZ), str(GATEWAY)]
gateway_spec = importlib.util.spec_from_file_location('local_integration_gateway', GATEWAY / 'gateway.py')
local_gateway = importlib.util.module_from_spec(gateway_spec)
sys.modules[gateway_spec.name] = local_gateway
gateway_spec.loader.exec_module(local_gateway)
assert Path(local_gateway.__file__).parent == GATEWAY

from bucket_manager import BucketManager
from gateway_state import GatewayStateStore
from memory_candidate_preparation import MemoryCandidatePreparation
from memory_l1_service import L1OwnerError
from memory_l2_service import MemoryL2Service
from raw_events import RawEventStore
from reflection_engine import ReflectionEngine
from core import chat_attachments
from core.attachment_adoption import AttachmentAdoptionClient


async def fixture(code=False):
    value = archive_fixture.ArchiveTests('test_completed_receipt_atomic_commit_listing_and_exact_bridge')
    # Brain contains its own historical gateway.py; bind only the setup import
    # to the explicitly loaded current Gateway source, then restore the module.
    with patch.dict(sys.modules, {'gateway': local_gateway}):
        await value.asyncSetUp()
    if code:
        block = chat_attachments.upload(value.store, 'owner', 'c', 'code', 'notes.txt', 'text/plain',
            b'External author prefers mystery stories.\n```python\nsecret_code()\n```\nEnd.')[0]
        value.event = value.store.append_event('c', role='user', content='Read external notes', source='app',
            source_event_id='code-message', metadata={'attachment_owner_id': 'owner', 'content_blocks': [block]})['event']
        value.messages = chat_attachments.project_messages(value.store, 'c', [{
            'role': 'user', 'content': value.event['content'], '_guyan_event': {'event_id': value.event['id']}}], budget_tokens=20000)
        value.scope.canonical_user_event_id = value.event['id']
        value.client = AttachmentAdoptionClient(str(value.server.make_url('/v1')), 'test-only', value.scope)
        value.raw_block = block
    value.assistant = value.commit(await value.completed_meta())
    return value


def engine_for(path, upstream, mode, *, fail_first=False):
    config = {
        'state_dir': str(path / 'state'), 'buckets_dir': str(path / 'buckets'),
        'identity': {'ai_name': 'Fixture AI', 'user_name': 'Fixture Owner'},
        'persona': {'profile_id': 'profile', 'canonical_session_id': 'profile'},
        'gateway': {'canonical_continuation': {'enabled': True, 'conversation_id': 'c',
            'base_url': str(upstream.server.make_url(''))}},
        'memory_authority': {'enabled': True},
        'reflection': {'enabled': True, 'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'local-fixture-only',
            'model': 'synthetic-local-response', 'daily_chat_memory_mode': mode,
            'daily_chat_memory_summary_enabled': False,
            'memory_task_routes': {stage: {'connection': 'reflection', 'model': 'synthetic-local-response'}
                for stage in ('auto_l1', 'presentation_l2')}},
    }
    engine = ReflectionEngine(config)
    calls = []

    async def completion(client, **kwargs):
        stage = kwargs['task_stage']
        payload = json.loads(kwargs['messages'][-1]['content'])
        calls.append({'stage': stage, 'payload': copy.deepcopy(payload)})
        if fail_first and len(calls) == 1:
            raise RuntimeError('synthetic completion interruption')
        if stage == 'auto_l1':
            source_id = payload['l1_contract']['semantic_contract']['source_ids'][0]
            result = {'candidates': [{'kind': 'key_event', 'confidence': 0.95,
                'source_refs': [source_id], 'reason': 'External author claim, not owner testimony.',
                'document': {'format': 'semantic-v1', 'entry': {'form': 'reference'},
                    'narrative': {'title': 'External reading', 'body': 'I read an external author describing a preference for mystery stories; the author is unknown.'},
                    'entities': [{'name': 'Unknown document author', 'entity_type': 'third_party'}],
                    'statements': [{'content': 'The external author describes a preference for mystery stories.',
                        'subject_ref': 'e1', 'classification': {'kind': 'preference', 'stability': None,
                            'modality': 'assertion', 'basis': 'reported'},
                        'time': None, 'conditions': None, 'evidence_refs': [source_id]}],
                    'retrieval': {'keywords': ['external reading', 'mystery stories']}}}]}
            from memory_l1_api import prepare_request
            _, schema = prepare_request(kwargs['messages'], stage=stage)
        elif stage == 'presentation_l2':
            sid = payload['l1_document']['statements'][0]['statement_id']
            result = {'full_injection_text': 'I read an external document whose unknown author described liking mystery stories.',
                'coherent_sections': [{'section_id': 'reading', 'text': 'The unknown author described liking mystery stories.', 'statement_refs': [sid]}],
                'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
            schema = None
        else:
            raise AssertionError('Unexpected completion stage: ' + stage)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)), finish_reason='stop')])
        if schema is not None:
            response.native_schema = schema
            response.native_status = 'completed'
        return response
    engine._daily_chat_memory_create_completion = completion
    manager = BucketManager(config)
    turns = GatewayStateStore(str(path / 'turns.sqlite'))
    raw = RawEventStore(config)
    # Persist a pre-existing chat cursor; the authority daily path must still
    # consume completed attachments when the actual raw DB contains no new row.
    engine._update_daily_chat_memory_raw_cursor('default', 7, '2026-10-19')
    return engine, manager, turns, raw, calls


async def daily(engine, manager, turns, raw):
    result = await engine.run_daily_chat_memory(manager, conversation_turn_store=turns,
        raw_event_store=raw, key='2026-10-19', now=datetime(2026, 10, 19, 3, tzinfo=timezone.utc))
    assert result['reason'] == 'no_new_raw_events'
    assert engine._daily_chat_memory_last_raw_event_id('default') == 7
    return result['attachment_memory_preparation']


@pytest.mark.parametrize('mode', ['auto', 'review'])
def test_no_new_raw_daily_actual_engine_commits_and_generates_ready_l2(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'test-only')
    async def run():
        upstream = await fixture(code=True)
        try:
            engine, manager, turns, raw, calls = engine_for(tmp_path, upstream, mode, fail_first=mode == 'auto')
            first = await daily(engine, manager, turns, raw)
            if mode == 'auto':
                assert first['results'][0]['status'] == 'blocked'
                assert not engine.memory_authority_store.list_candidates(status='all')
                first = await daily(engine, manager, turns, raw)
            assert first['status'] == 'completed', first
            rows = engine.memory_authority_store.list_candidates(status='all')
            assert len(rows) == 1
            candidate = rows[0]
            if mode == 'review':
                assert candidate['status'] == 'pending'
                assert await manager.list_all() == []
                adopted = await engine._memory_l1_adopt_candidate(candidate, manager, request_id='owner-review', owner=True)
                assert adopted['status'] == 'created'
            ident = candidate['candidate_id']
            assert engine.memory_authority_store.get_candidate(ident)['status'] == 'committed'
            assert (await manager.get(ident))['content']
            revisions = engine._memory_l1_revisions(manager)
            detail = await revisions.detail(ident)
            doc = detail['l1']['document']
            binding = doc['evidence_bindings'][0]
            assert binding['source_namespace'] == 'snapshot.aiz.adopted_attachment_text'
            source = await engine._memory_l1_sources().locate(binding)
            assert source['eligible_source']
            assert '（代码块）' in source['text'] and 'secret_code()' not in source['text']
            resource = chat_attachments.get(upstream.store, 'owner', 'c', upstream.raw_block['resource_id'])
            assert b'secret_code()' in b''.join(chat_attachments.iter_bytes(upstream.store, resource))
            model_material = next(call['payload'] for call in reversed(calls) if call['stage'] == 'auto_l1')
            assert '（代码块）' in json.dumps(model_material, ensure_ascii=False)
            assert 'secret_code()' not in json.dumps(model_material)
            queue = MemoryCandidatePreparation(engine.memory_authority_store)
            assert queue.get(first['results'][0]['key'])['candidate_ids'] == [ident]
            before = len(calls)
            await daily(engine, manager, turns, raw)
            assert len(calls) == before
            l2 = MemoryL2Service(engine, revisions)
            result = await l2.run_once(10)
            assert result['results'][0]['state'] == 'ready', result
            product = l2.detail(ident)['l2']
            assert product['state'] == 'ready'
            assert product['presentation']['first_pass_revision'] == detail['current_revision']
            assert product['presentation']['source_spans'][0]['source_revision'] == binding['source_revision']
            before = len(calls)
            await l2.run_once(10)
            assert len(calls) == before
        finally:
            await upstream.asyncTearDown()
    asyncio.run(run())


def test_review_adoption_blocked_after_success_event_withdrawal(tmp_path, monkeypatch):
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'test-only')
    async def run():
        upstream = await fixture()
        try:
            engine, manager, turns, raw, calls = engine_for(tmp_path, upstream, 'review')
            result = await daily(engine, manager, turns, raw)
            assert result['status'] == 'completed', result
            row = engine.memory_authority_store.list_candidates(status='all')[0]
            upstream.store.withdraw_message('c', target_event_id=upstream.assistant['id'], source_event_id='withdraw-adoption')
            with pytest.raises(L1OwnerError, match='source_not_currently_readable'):
                await engine._memory_l1_adopt_candidate(row, manager, request_id='withdrawn-review', owner=True)
            assert engine.memory_authority_store.get_candidate(row['candidate_id'])['status'] == 'pending'
            assert await manager.list_all() == []
            assert not engine.memory_authority_store.get_memory(row['candidate_id'])
            l2 = MemoryL2Service(engine, engine._memory_l1_revisions(manager))
            assert (await l2.run_once(10))['processed'] == 0
        finally:
            await upstream.asyncTearDown()
    asyncio.run(run())
