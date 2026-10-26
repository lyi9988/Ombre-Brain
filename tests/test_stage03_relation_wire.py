"""Scoped pending facts and adopted cross-memory material at the final wire."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from gateway import memory_use
from test_memory_recall_projection_r2 import _fixture, _request, _projection, _compose
from test_memory_recall_gateway_r2 import _install_source_http, _semantic_document
from memory_l1_contract import l1_content_sha256, semantic_sha256
from memory_delivery import candidates, compiled_candidates, final_fragments
from utils import count_tokens_approx


def project_direct(service, view, buckets, *, purpose='current'):
    bucket = buckets[0]
    request = _request(view, [bucket['id']])
    request['_use_context']['purpose'] = purpose
    raw = f"[bucket_id:{bucket['id']}] {bucket['content']}"
    sections = asyncio.run(memory_use.project_sections(service, request, request['q_context'], buckets,
        {'direct': (raw, [bucket['id']], 4000), 'targeted': ('', [], 0), 'diffused': ('', [], 0)}))
    return request, sections


def test_resolver_pending_fact_sync_never_falls_back_to_old_l1_current_fact(tmp_path, monkeypatch):
    real_client = memory_use.httpx.AsyncClient
    _authority, service, view, buckets, _plan = _fixture(tmp_path, monkeypatch)
    # _fixture already installs a transport. Restore the original client before
    # replacing it; nesting its installer would silently keep the old handler.
    monkeypatch.setattr(memory_use.httpx, 'AsyncClient', real_client)
    _install_source_http(monkeypatch, presentations={'memory:bucket-l1': {
        'state': 'pending_sync', 'reason': 'l1_sync_required', 'text': '', 'presentation_revision': None}})
    request, sections = project_direct(service, view, buckets)
    assert '现住址在白桦街' not in sections['direct']
    assert buckets[0]['content'] not in sections['direct']
    assert request['presentation_trace'][0]['source'] == 'none'
    assert request['presentation_trace'][0]['fallback_reason'] == 'l1_sync_required'


def test_known_pending_scope_excludes_only_affected_statement_and_never_uses_full_narrative(tmp_path, monkeypatch):
    _authority, service, view, buckets, _plan = _fixture(tmp_path, monkeypatch)
    original = view.memory_l1_recall_map
    def scoped(ids):
        records = deepcopy(original(ids))
        if 'bucket-l1' in records:
            records['bucket-l1']['sync_scope'] = {'statement_ids': ['statement:bucket-l1:current'], 'unknown_scope': False}
        return records
    monkeypatch.setattr(view, 'memory_l1_recall_map', scoped)
    request, sections = project_direct(service, view, buckets)
    uses = {item['statement_id']: item for item in request['usage_by_bucket']['bucket-l1']['items']}
    assert uses['statement:bucket-l1:current']['usage'] == 'excluded'
    assert uses['statement:bucket-l1:old']['usage'] != 'excluded'
    assert request['usage_by_bucket']['bucket-l1']['narrative_allowed'] is False
    assert '现住址在白桦街' not in sections['direct'] and buckets[0]['content'] not in sections['direct']
    assert '旧住址在青石路' in sections['direct']


def test_pending_scope_does_not_destroy_explicit_historical_reading(tmp_path, monkeypatch):
    _authority, service, view, buckets, _plan = _fixture(tmp_path, monkeypatch)
    original = view.memory_l1_recall_map
    def scoped(ids):
        records = deepcopy(original(ids))
        if 'bucket-l1' in records:
            records['bucket-l1']['sync_scope'] = {'statement_ids': [], 'unknown_scope': True}
        return records
    monkeypatch.setattr(view, 'memory_l1_recall_map', scoped)
    request, sections = project_direct(service, view, buckets, purpose='history')
    assert '现住址在白桦街' in sections['direct']
    assert request['usage_by_bucket']['bucket-l1']['query_purpose'] == 'history'
    assert not any('l1_sync_required' in item.get('reason_codes', []) for item in request['usage_by_bucket']['bucket-l1']['items'])


LIMIT_TEXT = '后续限定：白桦街只是临时住址，长期住址仍待确认。'


def commit_counterpart(authority, *, condition_false=False):
    body = 'B_PRIVATE_FULL_NARRATIVE_NOT_NEEDED_FOR_A'
    semantic, _binding = _semantic_document('bucket-B', body, current=LIMIT_TEXT)
    if condition_false:
        semantic['generation']['catalog_snapshot']['attributes'].append({'id': 'hour', 'value_type': 'integer', 'scope': 'any'})
        semantic['statements'][1]['conditions'] = {'state': 'known', 'text': '仅在凌晨零点适用',
            'predicate': {'op': 'eq', 'field': 'hour', 'value': 0}, 'exceptions': [], 'evidence_refs': ['source:bucket-B']}
    semantic['relations'] = [{'relation_id': 'relation-B-limits-A', 'kind': 'limits',
        'from': {'memory_id': None, 'revision': 'this_revision', 'object_type': 'statement', 'object_id': 'statement:bucket-B:current'},
        'to': {'memory_id': 'memory:bucket-l1', 'revision': 1, 'object_type': 'statement', 'object_id': 'statement:bucket-l1:current'},
        'origin': 'model', 'confirmation_state': 'confirmed', 'evidence_refs': ['source:bucket-B']}]
    metadata = {'name': 'B 的后续限定', 'memory_l1': semantic, 'memory_l1_semantic_sha256': semantic_sha256(semantic),
            'memory_l1_content_sha256': l1_content_sha256(body, semantic), 'memory_l1_owner_request': {
                'request_id': 'owner-adopt-B-limits-A', 'request_sha256': 'fixture',
                'owner_relation_adoptions': ['relation-B-limits-A']}}
    # Gateway's legacy test writer intentionally has no new Brain schema/hooks.
    # Use the actual Brain constructor, writer and index in a separate process
    # so identically named modules from these two services cannot shadow one another.
    state_dir = str(Path(authority.path).parent)
    script = '''import asyncio,json,sys
from memory_authority import MemoryAuthorityStore
from memory_commit_service import MemoryCommitService,BucketMemoryProjection
from bucket_manager import BucketManager
from memory_l1_index import MemoryL1Index
p=json.load(sys.stdin)
config={'state_dir':p['state_dir'],'buckets_dir':p['state_dir']+'/relation-test-buckets'}
a=MemoryAuthorityStore(config)
c=MemoryCommitService(a,BucketMemoryProjection(config,BucketManager(config)))
asyncio.run(c.commit_memory(memory_id='memory-bucket-B',bucket_id='bucket-B',expected_revision=0,
    body=p['body'],metadata=p['metadata'],source_refs=['synthetic:B'],decision_source='owner_l1_edit',
    idempotency_key='owner-B-confirmed-relation',actor='owner',l1_write_intent=True))
MemoryL1Index(a).ensure('memory:bucket-l1',1)
'''
    completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', script], input=json.dumps({'state_dir': state_dir, 'body': body, 'metadata': metadata}),
        text=True, encoding='utf-8', capture_output=True, cwd=Path(__file__).resolve().parents[2] / 'stage03-confirmed-brain')
    assert completed.returncode == 0, completed.stderr


def test_adopted_incoming_counterpart_reaches_final_composer_wire_and_delivery_fragment(tmp_path, monkeypatch):
    authority, service, view, buckets, plan = _fixture(tmp_path, monkeypatch)
    commit_counterpart(authority)
    graph = view.relation_context(['memory:bucket-l1'])
    assert len(graph['relations']) == 1 and any(node['memory_id'] == 'memory-bucket-B' for node in graph['nodes'])
    request, sections = project_direct(service, view, buckets)
    assert request['selected_buckets'] == [{'id': 'bucket-l1'}], 'B is a necessary counterpart, not an unrelated retrieved result'
    assert LIMIT_TEXT in sections['direct']
    assert 'B_PRIVATE_FULL_NARRATIVE_NOT_NEEDED_FOR_A' not in sections['direct']
    projection = _projection(service, request, sections)
    messages, _stable, _dynamic, _debug = _compose(service, plan, projection, sections)
    wire = {'messages': messages}
    wire_text = '\n'.join(message['content'] for message in messages)
    assert wire_text.count(LIMIT_TEXT) == 1
    assert wire_text.index('CANONICAL-MEMORY-BEGIN') < wire_text.index(LIMIT_TEXT) < wire_text.index('CANONICAL-MEMORY-END')
    assert count_tokens_approx(sections['direct']) <= 4000
    frozen = compiled_candidates(projection, candidates(projection, request['_semantic_records']))
    delivered = final_fragments(frozen, wire)
    counterpart = [row for row in delivered if row['memory_id'] == 'memory-bucket-B']
    assert counterpart and LIMIT_TEXT in counterpart[0]['text']
    assert counterpart[0]['l1_revision'] == 1
    assert counterpart[0]['coverage']['statement_ids'] == ['statement:bucket-B:current']


def test_withdrawn_counterpart_never_reappears_and_cannot_restore_original_current_basis(tmp_path, monkeypatch):
    authority, service, view, buckets, plan = _fixture(tmp_path, monkeypatch)
    commit_counterpart(authority)
    authority.change_memory_state(memory_id='memory-bucket-B', state='archived', recall_policy='disabled',
        expected_revision=1, idempotency_key='withdraw-B-before-final-wire', actor='owner')
    request, sections = project_direct(service, view, buckets)
    assert LIMIT_TEXT not in sections['direct']
    current = next(item for item in request['usage_by_bucket']['bucket-l1']['items'] if item['statement_id'] == 'statement:bucket-l1:current')
    assert current['usage'] != 'current_basis'
    assert 'adopted_relation_requires_current_counterpart' in current['reason_codes']
    assert request['usage_by_bucket']['bucket-l1']['narrative_allowed'] is False
    projection = _projection(service, request, sections)
    messages, _stable, _dynamic, _debug = _compose(service, plan, projection, sections)
    assert LIMIT_TEXT not in json.dumps(messages, ensure_ascii=False)


def test_composer_memory_off_cannot_revive_adopted_counterpart_through_a_second_carrier(tmp_path, monkeypatch):
    authority, service, view, buckets, plan = _fixture(tmp_path, monkeypatch, memory_mode='off')
    commit_counterpart(authority)
    request, sections = project_direct(service, view, buckets)
    assert LIMIT_TEXT in sections['direct'], 'the off gate is tested against real qualified counterpart material'
    projection = _projection(service, request, sections)
    messages, _stable, _dynamic, _debug = _compose(service, plan, projection, sections)
    assert LIMIT_TEXT not in json.dumps(messages, ensure_ascii=False)
    frozen = compiled_candidates(projection, candidates(projection, request['_semantic_records']))
    assert final_fragments(frozen, {'messages': messages}) == []


def test_verified_counterpart_condition_false_does_not_globally_downgrade_A_current_fact(tmp_path, monkeypatch):
    authority, service, view, buckets, _plan = _fixture(tmp_path, monkeypatch)
    commit_counterpart(authority, condition_false=True)
    request, sections = project_direct(service, view, buckets)
    current = next(item for item in request['usage_by_bucket']['bucket-l1']['items'] if item['statement_id'] == 'statement:bucket-l1:current')
    assert current['usage'] == 'current_basis'
    assert LIMIT_TEXT not in sections['direct']


def test_late_resolver_withdrawal_discards_prior_counterpart_qualification_and_keeps_unrelated_memory(tmp_path, monkeypatch):
    real_client = memory_use.httpx.AsyncClient
    authority, service, view, buckets, plan = _fixture(tmp_path, monkeypatch)
    commit_counterpart(authority)
    monkeypatch.setattr(memory_use.httpx, 'AsyncClient', real_client)
    calls = _install_source_http(monkeypatch, presentations={'memory:bucket-l1': {
        'state': 'unavailable', 'reason': 'related_memory_dependency_unavailable', 'text': '',
        'presentation_revision': None, 'relation_issues': [{'memory_id': 'memory-bucket-B', 'state': 'unavailable'}]}})
    ids = [bucket['id'] for bucket in buckets]
    request = _request(view, ids)
    raw = '\n\n'.join(f"[bucket_id:{bucket['id']}] {bucket['content']}" for bucket in buckets)
    sections = asyncio.run(memory_use.project_sections(service, request, request['q_context'], buckets,
        {'direct': (raw, ids, 4000), 'targeted': ('', [], 0), 'diffused': ('', [], 0)}))
    assert any(item['memory_id'] == 'memory-bucket-B' for call in calls for item in call['payload']['items']), 'B was qualified by the earlier actual source RPC'
    assert calls.presentation_requests, 'the later resolver response actually ran'
    assert LIMIT_TEXT not in sections['direct']
    assert buckets[1]['content'] in sections['direct'], 'unrelated memory is preserved'
    assert 'related_memory_dependency_unavailable' in request['incomplete_reasons']
    current = next(item for item in request['usage_by_bucket']['bucket-l1']['items'] if item['statement_id'] == 'statement:bucket-l1:current')
    assert current['usage'] != 'current_basis'
    projection = _projection(service, request, sections)
    messages, _stable, _dynamic, _debug = _compose(service, plan, projection, sections)
    assert LIMIT_TEXT not in json.dumps(messages, ensure_ascii=False)
    frozen = compiled_candidates(projection, candidates(projection, request['_semantic_records']))
    assert not any(row['memory_id'] == 'memory-bucket-B' for row in final_fragments(frozen, {'messages': messages}))
