import asyncio, copy, json
import pytest
from test_memory_l1_ingestion_p2c import setup
from memory_l1_ingestion import prepare_model_document, candidate_source_refs, bind_sources
from memory_l1_runtime import MemoryL1Runtime
from memory_l1_service import L1OwnerError
from memory_l1_schema import split_document
from memory_l0_sources import MemoryL0Sources
from memory_authority import MemoryProposal, MemoryIngestionPolicy
from memory_candidate_preparation import MemoryCandidatePreparation

def env(tmp_path, count=3):
    service, source, draft, generation = setup(tmp_path)
    sources = [dict(copy.deepcopy(source), event_id=f'e{i+1}', version_id=f'v{i+1}', source_key=f'key{i+1}') for i in range(count)]
    draft = json.loads(json.dumps(draft).replace('source:current', f'source:{count}'))
    return service, sources, draft, generation

def run(e, refs):
    service, sources, draft, generation = e
    runtime = MemoryL1Runtime()
    runtime.config = {}
    runtime._memory_l1_revisions = lambda: service
    return runtime._memory_l1_daily_document({'document': draft, 'source_refs': refs,
        '_l1_sources': sources, '_l1_generation': generation}, 'm1')

def ids(doc): return [b['binding_id'] for b in doc['evidence_bindings']]

def test_two_sources_only_candidate_support_is_retained_and_committed(tmp_path):
    e = env(tmp_path, 2)
    doc = run(e, ['source:2'])
    assert ids(doc) == ['source:2']
    service, sources, draft, generation = e
    snapshots, bindings = bind_sources(sources, config={})
    assert doc['evidence_bindings'] == [bindings[1]]
    assert MemoryL0Sources(service.authority).copy(bindings[1])['text'] == sources[1]['text']
    with service.authority._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM memory_l0_source_copies').fetchone()[0] == 1
    body, semantic = split_document(doc)
    asyncio.run(service.commit.commit_memory(memory_id='m1', bucket_id='bucket1', expected_revision=0,
        body=body, metadata={'memory_l1': semantic}, source_refs=['key2'], decision_source='test',
        idempotency_key='test-commit', actor='model', l1_write_intent=True))
    saved = service.authority.get_memory_revision('m1', 1)
    assert [b['binding_id'] for b in saved['metadata']['memory_l1']['evidence_bindings']] == ['source:2']
    assert saved['source_refs'] == ['key2']

def test_sparse_ids_not_renumbered_and_identity_hashes_unchanged(tmp_path):
    e = env(tmp_path)
    doc = run(e, ['source:3'])
    assert ids(doc) == ['source:3']
    assert doc['statements'][0]['evidence_refs'] == ['source:3']
    _, bindings = bind_sources(e[1], config={})
    assert doc['evidence_bindings'] == bindings[2:]
    assert doc['evidence_bindings'][0]['author_ref'] != 'owner'

def test_entity_and_statement_and_candidate_support_are_unioned(tmp_path):
    e = env(tmp_path)
    e[2]['entities'][0]['entity_type']['evidence_refs'] = ['source:1']
    doc = run(e, ['source:2'])
    assert ids(doc) == ['source:1', 'source:2', 'source:3']
    assert doc['entities'][0]['entity_type']['evidence_refs'] == ['source:1']

def test_scene_range_can_be_only_ref_to_other_source(tmp_path):
    e = env(tmp_path)
    e[2]['scenes'][0]['source_ranges'][0]['source_ref'] = 'source:1'
    doc = run(e, [])
    assert ids(doc) == ['source:1', 'source:3']
    assert doc['scenes'][0]['anchors'][0]['target_id'] == doc['evidence_bindings'][0]['source_id']

@pytest.mark.parametrize('refs,code', [(None,'candidate_source_refs_required'), ('source:3','candidate_source_refs_required'),
    (['source:999'],'unknown_source_binding'), ([1],'candidate_source_refs_required')])
def test_bad_explicit_refs_fail_without_persisting_sources(tmp_path, refs, code):
    e = env(tmp_path)
    with pytest.raises(L1OwnerError, match=code): run(e, refs)
    with e[0].authority._connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if 'memory_l0_source_copies' in tables:
            assert db.execute('SELECT COUNT(*) FROM memory_l0_source_copies').fetchone()[0] == 0

def test_no_refs_never_falls_back_to_whole_window():
    with pytest.raises(L1OwnerError, match='candidate_sources_required'):
        candidate_source_refs({'entities': [], 'statements': []}, [], 2)

def test_unscoped_existing_manual_path_unchanged(tmp_path):
    service, source, draft, generation = setup(tmp_path)
    second = dict(source, event_id='e2', version_id='v2')
    doc = prepare_model_document(service, memory_id='manual', document=draft, sources=[source, second],
        generation=generation, request_id='manual')
    assert ids(doc) == ['source:1', 'source:2']
    assert doc['statements'][0]['evidence_refs'] == ['source:2']

def test_valid_candidate_and_invalid_ref_issue_persist_without_losing_progress(tmp_path):
    e = env(tmp_path)
    doc = run(e, ['source:3'])
    try: candidate_source_refs(e[2], ['source:999'], 3)
    except L1OwnerError as exc: issue = {'code': exc.code, 'field_path': exc.field_path}
    proposal = MemoryProposal(proposal_id='valid', source_type='daily_chat', proposed_body=doc['narrative']['body'],
        source_refs=('key3',), metadata={'memory_l1': split_document(doc)[1]})
    queue = MemoryCandidatePreparation(e[0].authority)
    receipt = queue.prepare('batch', [proposal], MemoryIngestionPolicy({}), scope='day', source_fingerprint='sources',
        generated_count=2, issue=issue)
    assert receipt['requires_manual_retry'] and receipt['issue']['code'] == 'unknown_source_binding'
    assert receipt['candidate_ids'] == ['valid'] and receipt['generated_count'] == 2
    assert len(queue.remaining(queue.get('batch'))) == 1
    queue.acknowledge('batch', ['not-a-candidate'])
    assert queue.get('batch')['processed_ids'] == []

def test_explicit_peer_author_is_preserved_not_inferred_from_api_role(tmp_path):
    e = env(tmp_path)
    service, sources, draft, generation = e
    s = sources[2]
    actor = {'subject_id': 'peer:yanyan', 'kind': 'ai', 'display_name': '晏晏'}
    config = {'identity': {'source_participants': [{**{k: s[k] for k in ('system', 'namespace', 'profile_id', 'conversation_id', 'role')}, 'participant': actor}]}}
    doc = prepare_model_document(service, memory_id='peer', document=draft, sources=sources, generation=generation,
        request_id='peer', config=config, candidate_refs=['source:3'])
    assert doc['evidence_bindings'][0]['author_ref'] == 'peer:yanyan'
    snapshot = MemoryL0Sources(service.authority).copy(doc['evidence_bindings'][0])
    assert snapshot['participant']['kind'] == 'ai' and snapshot['participant']['subject_id'] == 'peer:yanyan'

def test_unknown_semantic_ref_rejected_even_when_candidate_ref_valid(tmp_path):
    e = env(tmp_path)
    e[2]['entities'][0]['entity_type']['evidence_refs'] = ['source:99']
    with pytest.raises(L1OwnerError, match='unknown_source_binding'): run(e, ['source:3'])
