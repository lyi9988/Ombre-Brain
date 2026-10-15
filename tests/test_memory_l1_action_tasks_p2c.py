import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

from bucket_manager import BucketManager
from memory_authority import IdempotencyConflict
from memory_l1_ingestion import model_template
from memory_l1_schema import statement_template
from memory_l1_source_tasks import MemoryL1SourceTasks
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config


def run(coro): return asyncio.run(coro)


def setup(tmp_path, monkeypatch, mode='review'):
    config = reflection_config(tmp_path, mode=mode)
    engine = ReflectionEngine(config)
    buckets = BucketManager(config)
    tasks = MemoryL1SourceTasks(engine.memory_authority_store)
    calls = []
    source = {'state': 'available', 'eligible_source': True, 'source_id': 'action-a',
        'source_revision': 'artifact-a', 'text': '执行完成：浏览了天文学资料。', 'author': {'role': 'tool'},
        'coverage': {'complete': True}}
    async def capture(self, action_id, artifact_id):
        assert action_id == 'action-a' and artifact_id == 'artifact-a'
        return copy.deepcopy(source)
    monkeypatch.setattr('memory_l1_source_readers.AizActionSourceReader.capture', capture)
    draft = model_template()
    draft['narrative'] = {'title': '浏览天文学之后', 'body': '我完成浏览，留下这次经历。' * 600}
    statement = statement_template(); statement.pop('statement_revision_id')
    statement.update(content='这次浏览了天文学资料。', evidence_refs=['source:current'])
    draft['statements'] = [statement]
    async def completion(client, **kwargs):
        calls.append(kwargs)
        await asyncio.sleep(0)
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates': [
            {'document': draft, 'kind': 'key_event', 'confidence': 0.9}]})), finish_reason='stop')])
    engine._daily_chat_memory_model_client = lambda **kwargs: (object(), 'synthetic-route', False)
    engine._daily_chat_memory_create_completion = completion
    payload = {'action_id': 'action-a', 'artifact_id': 'artifact-a', 'request_id': 'run-a'}
    return engine, buckets, tasks, payload, calls, source


def test_queue_is_identifiers_only_idempotent_and_conflicting_request_is_not_repurposed(tmp_path, monkeypatch):
    engine, _, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    first = tasks.enqueue_action(payload)
    replay = tasks.enqueue_action(payload)
    assert first['task_id'] == replay['task_id'] and replay['status'] == 'exists'
    assert first['adopted'] is False and calls == []
    assert len(tasks.list()) == 1
    assert 'text' not in tasks.list()[0] and 'body' not in tasks.list()[0]
    with pytest.raises(IdempotencyConflict): tasks.enqueue_action({**payload, 'artifact_id': 'artifact-other'})
    assert len(tasks.list()) == 1 and engine.memory_authority_store.list_memories() == []


def test_review_worker_keeps_full_l1_and_does_not_claim_adoption(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    record = result['results'][0]
    assert record['status'] == 'prepared' and record['adopted'] is False
    assert len(calls) == 1
    item = engine.memory_authority_store.get_candidate(record['candidate_ids'][0])
    assert item['status'] == 'pending' and len(item['proposal']['proposed_body']) > 4000
    semantic = item['proposal']['metadata']['memory_l1']
    assert semantic['evidence_bindings'][0]['source_namespace'] == 'snapshot.aiz.action_receipt'
    assert semantic['statements'][0]['subject_ref']['state'] == 'not_stated'
    assert semantic['statements'][0]['time']['event']['start']['state'] == 'not_stated'
    assert engine.memory_authority_store.list_memories() == []


def test_auto_worker_uses_original_policy_and_commits_complete_semantics(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch, mode='auto')
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    record = result['results'][0]
    assert record['adopted'] is True
    ident = record['candidate_ids'][0]
    assert engine.memory_authority_store.get_candidate(ident)['status'] == 'committed'
    revision = engine.memory_authority_store.get_memory_revision(ident, 1)
    assert revision['metadata']['memory_l1']['generation']['actor'] == 'action_first_pass'
    assert revision['metadata']['memory_l1_content_sha256']
    assert len(run(buckets.get(ident))['content']) > 4000
    assert run(engine._memory_l1_action_tasks(buckets))['results'] == [] and len(calls) == 1


def test_same_task_claim_does_not_make_two_model_calls(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    async def concurrent(): return await asyncio.gather(engine._memory_l1_action_tasks(buckets), engine._memory_l1_action_tasks(buckets))
    result = run(concurrent())
    assert sum(len(row['results']) for row in result) == 1 and len(calls) == 1


def test_nonfinal_source_fails_visibly_without_generating_or_adopting(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    source.update(state='unknown', eligible_source=False)
    tasks.enqueue_action(payload)
    result = run(engine._memory_l1_action_tasks(buckets))
    assert result['results'][0]['status'] == 'failed' and calls == []
    assert tasks.list()[0]['phase'] == 'failed'
    assert engine.memory_authority_store.list_candidates(status='all') == []


def test_generation_zero_disables_processing_without_discarding_queue(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    engine.config['reflection']['action_l1_jobs_per_run'] = 0
    tasks.enqueue_action(payload)
    assert run(engine._memory_l1_action_tasks(buckets))['status'] == 'disabled'
    assert tasks.list()[0]['phase'] == 'queued' and calls == []


def test_source_failure_does_not_accept_or_lock_pending_owner_draft(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    made = run(engine._memory_l1_action_tasks(buckets))['results'][0]
    ident = made['candidate_ids'][0]
    source.update(state='unknown', eligible_source=False)
    before = engine.memory_authority_store.get_candidate(ident)
    from memory_l1_service import L1OwnerError
    with pytest.raises(L1OwnerError, match='source_not_currently_readable'):
        run(engine._memory_l1_adopt_candidate(before, buckets, request_id='owner-retry', owner=True))
    assert engine.memory_authority_store.get_candidate(ident) == before
    assert before['status'] == 'pending' and engine.memory_authority_store.list_memories() == []


def test_unneeded_source_does_not_block_valid_evidence(tmp_path, monkeypatch):
    from memory_authority import MemoryProposal
    from memory_l1_service import L1OwnerError
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch)
    tasks.enqueue_action(payload)
    ident = run(engine._memory_l1_action_tasks(buckets))['results'][0]['candidate_ids'][0]
    row = engine.memory_authority_store.get_candidate(ident)
    proposal = copy.deepcopy(row['proposal'])
    semantic = proposal['metadata']['memory_l1']
    extra = copy.deepcopy(semantic['evidence_bindings'][0]); extra['binding_id'] = 'source:unused'
    locator = json.loads(extra['source_id']); locator['event_id'] = 'action-other'
    from memory_l1_contract import canonical_json
    extra['source_id'] = canonical_json(locator)
    extra['anchors'][0]['target_id'] = extra['source_id']
    semantic['evidence_bindings'].append(extra)
    row = engine.memory_authority_store.revise_candidate(ident, expected_revision=row['revision'],
        proposal=MemoryProposal.from_mapping(proposal), request_id='fixture-with-extra-source', actor='test')
    class Readers:
        async def locate(self, binding):
            available = binding['binding_id'] == 'source:1'
            return {'eligible_source': available, 'state': 'verified_snapshot' if available else 'withdrawn'}
    engine._memory_l1_sources = lambda: Readers()
    result = run(engine._memory_l1_adopt_candidate(row, buckets, request_id='owner-valid', owner=True))
    assert result['status'] == 'created'
    record = engine.memory_authority_store.get_memory_revision(ident, 1)
    assert record['metadata']['source_observations_at_adoption']['source:unused']['eligible_source'] is False


def test_owner_task_api_auth_cas_retry_and_no_model_side_effect(tmp_path, monkeypatch):
    import server
    from starlette.requests import Request
    engine, buckets, tasks, payload, calls, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(server, 'memory_authority_store', engine.memory_authority_store)
    monkeypatch.setattr(server, 'reflection_engine', engine)
    monkeypatch.setattr(server, '_memory_write_token', lambda: 'task-token')
    def request(method, body=None, *, authorized=True, task_id=None):
        async def receive(): return {'type': 'http.request', 'body': json.dumps(body or {}).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': method, 'path': '/api/memory-authority/action-source',
            'query_string': b'', 'path_params': {'task_id': task_id} if task_id else {},
            'headers': [(b'authorization', b'Bearer task-token')] if authorized else []}, receive)
    async def check():
        # Auth helper accepts owner sessions too; force neither credential for
        # the unauthorized request without changing the real bearer path.
        assert (await server.api_memory_l1_source_tasks(request('POST', payload, authorized=False))).status_code == 401
        posted = await server.api_memory_l1_source_tasks(request('POST', payload))
        assert posted.status_code == 202 and not calls
        ident = json.loads(posted.body)['task_id']
        assert json.loads((await server.api_memory_l1_source_tasks(request('GET'))).body)['tasks'][0]['task_id'] == ident
        tasks.transition(ident, 'queued', 'failed', issue={'code': 'synthetic'})
        engine._daily_chat_memory_run_lock = True
        assert (await server.api_memory_l1_source_tasks(request('POST', {'expected_phase':'failed'}, task_id=ident))).status_code == 409
        engine._daily_chat_memory_run_lock = False
        retried = await server.api_memory_l1_source_tasks(request('POST', {'expected_phase':'failed'}, task_id=ident))
        assert retried.status_code == 200 and json.loads(retried.body)['action_reexecuted'] is False
        stale = await server.api_memory_l1_source_tasks(request('POST', {'expected_phase':'failed'}, task_id=ident))
        assert stale.status_code == 409 and tasks.list()[0]['phase'] == 'queued' and not calls
    run(check())


def test_source_withdrawn_during_model_generation_is_not_auto_adopted(tmp_path, monkeypatch):
    engine, buckets, tasks, payload, calls, source = setup(tmp_path, monkeypatch, mode='auto')
    original = engine._daily_chat_memory_create_completion
    async def completed_then_withdrawn(*args, **kwargs):
        result = await original(*args, **kwargs)
        source.update(state='withdrawn', eligible_source=False)
        return result
    engine._daily_chat_memory_create_completion = completed_then_withdrawn
    tasks.enqueue_action(payload)
    row = run(engine._memory_l1_action_tasks(buckets))['results'][0]
    assert row['status'] == 'prepared' and row['adopted'] is False
    assert row['adoption_issues'][0]['code'] == 'source_not_currently_readable'
    assert engine.memory_authority_store.list_memories() == []
    assert engine.memory_authority_store.get_candidate(row['candidate_ids'][0])['status'] == 'accepted'
