import asyncio
import copy
import sqlite3
import pytest
from types import SimpleNamespace as NS
from pathlib import Path

from embedding_engine import EmbeddingEngine
from gateway import GatewayService
from memory_projection_worker import MemoryProjectionWorker, embedding_projection_needs_upgrade
from memory_retrieval_text import TEXT_PROJECTION_VERSION, l1_retrieval_text
from test_memory_recall_gateway_r2 import _authority_fixture


def test_shared_semantic_projection_keeps_negation_conditions_and_late_text_without_custody():
    body = '完整回忆中的经历。' * 1000 + '最后仍然没有承诺。'
    from test_memory_recall_gateway_r2 import _semantic_document
    doc, _ = _semantic_document('projection', body)
    doc['generation']['prompt_revision'] = 'DO-NOT-SEND-PROMPT-REVISION'
    statement = doc['statements'][0]
    statement['content'] = '只有她愿意才继续；不能把过去意愿当成现在同意。'
    statement['conditions'].update(state='known', text='只有她愿意',
        exceptions=[{'state':'known', 'text':'她拒绝就停止', 'predicate':None, 'evidence_refs':[]}])
    statement['cognition'] = {'role':{'state':'known','value':'hypothesis'},
        'epistemic_status':{'state':'known','value':'tentative'}, 'about_refs':['entity:user'],
        'open_questions':['她现在是否仍愿意？']}
    before = copy.deepcopy(doc)
    text = l1_retrieval_text(doc, body=body)
    assert body in text and len(text) > 4000
    for value in ('只有她愿意才继续', '她拒绝就停止', 'tentative', '她现在是否仍愿意？', '主人', '2019-01-01'):
        assert value in text
    for value in ('DO-NOT-SEND-PROMPT-REVISION', 'source:projection', 'statement:projection:',
                  'entity:user', 'text_sha256', 'catalog_snapshot', 'source_revision'):
        assert value not in text
    assert doc == before


def test_worker_and_rerank_use_same_committed_l1_text_and_version(tmp_path):
    authority, view, records, buckets = _authority_fixture(tmp_path, [('projection', '完整的原始正文。')])
    bucket = buckets[0]
    memory_id = records['projection']['memory_id']
    captured = []
    class Engine:
        enabled = True
        model = 'synthetic'
        base_url = 'https://synthetic.invalid'
        dimension = 2
        def preparation_hash(self, **kwargs): return 'd' * 64
        async def generate_and_store(self, parent, text, **metadata):
            captured.append((parent, text, metadata))
            metadata['result_metadata']['dimension'] = 2
            return True
    worker = MemoryProjectionWorker(config={}, authority=authority, bucket_manager=None, embedding_engine=Engine())
    units = worker._embedding_source_units(bucket, memory_id, revision=1)
    gateway = GatewayService.__new__(GatewayService)
    gateway.memory_authority_view = view
    assert units[0]['text'] == gateway._bucket_rerank_document(bucket)
    assert units[0]['text'] == l1_retrieval_text(records['projection']['document'], body=bucket['content'])
    rerank_calls = []
    gateway.reranker_engine = NS(enabled=True, candidate_limit=1, score_weight=0.65)
    gateway._bucket_rerank_candidate_priority = lambda query, item: (-item['score'],)
    gateway._bucket_reranked_candidate_rank = lambda query, item: (-item['score'],)
    async def rerank(**kwargs):
        rerank_calls.append(kwargs)
        return [NS(index=0, score=0.9)]
    gateway._rerank_cached = rerank
    asyncio.run(gateway._rerank_scored_bucket_candidates('住所', [{'bucket':bucket, 'score':0.5}]))
    assert rerank_calls[0]['documents'] == [units[0]['text']]
    assert rerank_calls[0]['top_n'] == 1
    result = asyncio.run(worker._project_embedding_units(bucket, memory_id, 1,
        authority.get_memory(memory_id)['body_sha256'], source_units=units))
    assert result['text_projection_version'] == TEXT_PROJECTION_VERSION
    assert all(call[2]['text_projection_version'] == TEXT_PROJECTION_VERSION for call in captured)
    assert all(call[2]['source_revision'] == 1 and call[2]['parent_memory_id'] == memory_id for call in captured)
    if len(units) > 1:
        moment = {'bucket_id':bucket['id'], 'text':'一个完整场景。', 'moment_id':'moment'}
        assert gateway._moment_rerank_document(moment) == l1_retrieval_text(records['projection']['document'], body=moment['text'])
    assert embedding_projection_needs_upgrade('projected', {'metadata_complete':True},
        {'text_projection_version':TEXT_PROJECTION_VERSION})
    assert not embedding_projection_needs_upgrade('projected', {'metadata_complete':True,
        'text_projection_version':TEXT_PROJECTION_VERSION}, {'text_projection_version':TEXT_PROJECTION_VERSION})


def test_old_l1_vector_version_is_stale_before_query_provider_but_legacy_stays_compatible(tmp_path):
    authority, view, records, buckets = _authority_fixture(tmp_path, [('projection', '原始正文。')])
    engine = EmbeddingEngine({'buckets_dir':str(tmp_path), 'embedding':{
        'enabled':True, 'api_key':'synthetic-test-key', 'base_url':'https://synthetic.invalid', 'model':'synthetic'}})
    record = authority.get_memory(records['projection']['memory_id'])
    kwargs = {'source_revision':1, 'source_sha256':record['body_sha256'],
              'parent_memory_id':record['memory_id'], 'input_text':'stored document'}
    engine._store_embedding('projection', [1.0, 0.0], **kwargs)
    queries = []
    async def query_embeddings(texts, **options):
        queries.append(texts)
        return [[1.0, 0.0] for _ in texts]
    engine.query_embeddings = query_embeddings
    debug = {}
    metadata = view.memory_index_metadata_map(['projection'])
    assert metadata['projection']['text_projection_version'] == TEXT_PROJECTION_VERSION
    result = asyncio.run(engine.search_similar_queries(['query'], eligible_ids={'projection'},
        top_k=1, index_metadata=metadata, cache_debug=debug))
    assert result == [] and queries == []
    assert debug['index_rejections'] == {'embedding_text_projection_mismatch':1}
    engine._store_embedding('projection', [1.0, 0.0], text_projection_version=TEXT_PROJECTION_VERSION, **kwargs)
    result = asyncio.run(engine.search_similar_queries(['query'], eligible_ids={'projection'},
        top_k=1, index_metadata=metadata))
    assert result[0]['index_status'] == 'verified'
    # Pre-L1 records have no new version expectation and retain their old path.
    engine._store_embedding('legacy', [1.0, 0.0], source_revision=1, source_sha256='legacy-body',
        parent_memory_id='legacy-memory', input_text='legacy body')
    result = asyncio.run(engine.search_similar_queries(['query'], eligible_ids={'legacy'}, top_k=1,
        index_metadata={'legacy':{'revision':1,'body_sha256':'legacy-body','memory_id':'legacy-memory'}}))
    assert result[0]['bucket_id'] == 'legacy'


def test_old_embedding_schema_adds_nullable_version_without_rewriting_rows(tmp_path):
    path = tmp_path / 'embeddings.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE embeddings(bucket_id TEXT PRIMARY KEY,embedding TEXT NOT NULL,model TEXT,dimension INTEGER,updated_at TEXT NOT NULL)')
        db.execute("INSERT INTO embeddings VALUES('legacy','[1,0]','synthetic',2,'original-stamp')")
    EmbeddingEngine({'buckets_dir':str(tmp_path)})
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT embedding,updated_at,text_projection_version FROM embeddings').fetchone() == ('[1,0]', 'original-stamp', None)


def test_legacy_rerank_documents_are_complete_after_former_4000_boundary():
    gateway = GatewayService.__new__(GatewayService)
    text = '保留完整旧库正文。' * 1000 + '最后一句保留。'
    assert text in gateway._bucket_rerank_document({'id':'legacy', 'metadata':{}, 'content':text})
    assert text in gateway._moment_rerank_document({'bucket_id':'legacy', 'text':text})


def test_semantic_projection_omits_empty_time_shells_but_keeps_explicit_unknowns_and_real_predicates():
    from memory_l1_schema import document_template, statement_template, unknown
    doc = document_template('一次未说明时间的经历。')
    statement = statement_template()
    statement['content'] = '是否继续仍未知，若她拒绝就停止。'
    doc['statements'] = [statement]
    text = l1_retrieval_text(doc)
    assert 'event:' not in text and 'applicability:' not in text
    assert 'start=' not in text and 'timezone=' not in text
    statement['time']['applicability']['end'] = unknown('unbounded')
    statement['conditions'].update(state='unresolved', predicate={'op':'all','args':[
        {'op':'not','args':[{'op':'eq','field':'choice','value':'refuse'}]},
        {'op':'any','args':[{'op':'eq','field':'ready','value':True}]}]})
    text = l1_retrieval_text(doc)
    assert 'end=unbounded' in text and 'conditions: unresolved' in text
    assert 'all (not (choice eq refuse); any (ready eq True))' in text


def test_existing_maintenance_queue_detects_l1_projection_stale_without_relabeling_legacy(tmp_path):
    from scripts import repair_recall_r1_embeddings as repair
    from test_memory_projection_worker_r1 import _commit_memory
    authority, _view, records, _buckets = _authority_fixture(tmp_path, [('projection', '正文。')])
    _commit_memory(authority, memory_id='legacy')
    for memory_id in ('memory:projection', 'legacy'):
        memory = authority.get_memory(memory_id)
        authority.set_projection_status(memory_id=memory_id, memory_revision=1, projector='embedding',
            status='projected', source_sha256=memory['body_sha256'], details={'metadata_complete':True})
    candidate = repair._next_candidate(Path(authority.path))
    assert candidate['memory_id'] == 'memory:projection' and candidate['has_l1']
    assert repair._upgrade_candidate_count(Path(authority.path)) == 1
    readonly = repair.ReadOnlyMemoryAuthority(authority.path)
    assert readonly.get_memory('legacy')['active_revision'] == 1
    assert readonly.get_memory_revision('memory:projection', 1)['metadata']['memory_l1']
    with readonly._connect() as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("UPDATE memories SET recall_policy='disabled'")
    assert authority.get_memory('legacy')['recall_policy'] == 'enabled'
