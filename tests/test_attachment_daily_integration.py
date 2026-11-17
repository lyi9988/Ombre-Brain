"""Local verification of attachment generation retirement and evidence compatibility.

Only completion responses are synthetic. No provider calls, production state,
or replacement extraction/commit methods are used. Requires the local candidate
worktrees and the documented staging archive fixture.
"""
from datetime import datetime, timezone
import asyncio
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
from memory_authority import MemoryProposal
from memory_candidate_preparation import MemoryCandidatePreparation
from memory_daily_attachments import AdoptedAttachmentReader
from memory_l1_schema import split_document
from memory_l1_semantic_wire import assemble_document
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
    metadata = await value.completed_meta()
    value.attachment_ref = metadata['attachment_adoption_refs'][0]
    value.assistant = value.commit(metadata)
    return value


def engine_for(path, upstream, mode):
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
        calls.append(stage)
        assert stage == 'presentation_l2', stage
        payload = json.loads(kwargs['messages'][-1]['content'])
        sid = payload['l1_document']['statements'][0]['statement_id']
        result = {'full_injection_text': 'I read an external document whose unknown author described liking mystery stories.',
            'coherent_sections': [{'section_id': 'reading', 'text': 'The unknown author described liking mystery stories.', 'statement_refs': [sid]}],
            'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)), finish_reason='stop')])
    engine._daily_chat_memory_create_completion = completion
    manager = BucketManager(config)
    turns = GatewayStateStore(str(path / 'turns.sqlite'))
    raw = RawEventStore(config)
    # The attachment was read and adopted upstream, while chat has no new row.
    engine._update_daily_chat_memory_raw_cursor('default', 7, '2026-10-19')
    return engine, manager, turns, raw, calls


async def daily(engine, manager, turns, raw):
    result = await engine.run_daily_chat_memory(manager, conversation_turn_store=turns,
        raw_event_store=raw, key='2026-10-19', now=datetime(2026, 10, 19, 3, tzinfo=timezone.utc))
    assert result['reason'] == 'no_new_raw_events'
    assert engine._daily_chat_memory_last_raw_event_id('default') == 7
    assert 'attachment_memory_preparation' not in result


async def seed_pending_attachment(engine, upstream):
    event_id = upstream.assistant['id']
    version = upstream.store.version_state_for_event('c', event_id)['version_id']
    source = await AdoptedAttachmentReader(engine.config).capture({
        'event_id': event_id, 'version_id': version, 'ref': upstream.attachment_ref,
    })
    assert source['namespace'] == 'attachment_text'
    ident = 'attachment_l1_existing_candidate'
    key = 'attachment_l1:existing_receipt'
    model_document = {
        'format': 'semantic-v1',
        'entry': {'form': 'reference'},
        'narrative': {'title': 'External reading', 'body': 'I read an author describing a preference for mystery stories.'},
        'entities': [{'name': 'Unknown document author', 'entity_type': 'third_party'}],
        'statements': [{'content': 'The external author describes a preference for mystery stories.',
            'subject_ref': 'e1', 'classification': {'kind': 'preference', 'stability': None,
                'modality': 'assertion', 'basis': 'reported'},
            'time': None, 'conditions': None, 'evidence_refs': ['source:1']}],
        'retrieval': {'keywords': ['external reading', 'mystery stories']},
    }
    doc = engine._memory_l1_daily_document({
        'document': assemble_document(model_document), 'source_refs': ['source:1'],
        '_l1_sources': [source], '_l1_generation': {
            'prompt_revision': None, 'model_route_revision': None,
            'task_id': key, 'actor': 'daily_chat_first_pass',
        },
    }, ident)
    body, semantic = split_document(doc)
    proposal = MemoryProposal.from_mapping({
        'proposal_id': ident, 'source_type': 'attachment_document',
        'proposed_body': body, 'source_refs': [source['source_key']], 'source_status': 'verified',
        'memory_type': 'key_event', 'confidence': 0.95, 'requested_mode': 'review',
        'owner_explicit': False, 'metadata': {'memory_l1': semantic, 'attachment_preparation_key': key},
    })
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    receipt = queue.prepare(key, [proposal], engine.memory_ingestion_policy, scope=key,
        source_fingerprint=source['source_key'], generated_count=1)
    assert receipt['candidate_ids'] == [ident]
    return engine.memory_authority_store.get_candidate(ident)


@pytest.mark.parametrize('mode', ['auto', 'review'])
def test_daily_skips_attachment_only_generation_and_preserves_old_review(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'test-only')
    async def run():
        upstream = await fixture(code=True)
        try:
            engine, manager, turns, raw, calls = engine_for(tmp_path, upstream, mode)
            await daily(engine, manager, turns, raw)
            assert not engine.memory_authority_store.list_candidates(status='all')
            assert not calls
            candidate = await seed_pending_attachment(engine, upstream)
            await daily(engine, manager, turns, raw)
            assert engine.memory_authority_store.list_candidates(status='all') == [candidate]
            assert candidate['status'] == 'pending'
            assert not calls
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


@pytest.mark.parametrize('opening,closing,newline', [
    ('```','```','\n'), ('```','````','\n'), ('~~~~','~~~~~','\n'), ('```','````','\r\n'),
])
def test_code_view_keeps_prose_after_valid_closing_fence(opening,closing,newline):
    from memory_attachment_source import semantic_text
    text=newline.join(['作者有一项偏好。',opening+'python','secret_code()',closing,'后来作者改变了想法。'])
    source={'namespace':'attachment_text','text':text,'attachment_format':{'name':'note.md','mime':'text/markdown'}}
    assert semantic_text(source)==newline.join(['作者有一项偏好。','（代码块）','后来作者改变了想法。'])
    assert source['text']==text


def test_review_adoption_blocked_after_success_event_withdrawal(tmp_path, monkeypatch):
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'test-only')
    async def run():
        upstream = await fixture()
        try:
            engine, manager, _, _, _ = engine_for(tmp_path, upstream, 'review')
            row = await seed_pending_attachment(engine, upstream)
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
