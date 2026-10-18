"""Offline actual Brain producer and SQLite vector-version contract."""
import asyncio
import sqlite3
import pytest
from pathlib import Path

from embedding_engine import EmbeddingEngine
from memory_l1_schema import statement_template
from memory_projection_worker import MemoryProjectionWorker
from memory_retrieval_text import TEXT_PROJECTION_VERSION, l1_retrieval_text
from test_memory_l1_revisions import request_for, setup_service


def test_brain_projector_writes_semantic_input_version_through_actual_engine(tmp_path, monkeypatch):
    revisions, authority, manager = setup_service(tmp_path)
    edit = request_for(revisions)
    statement = statement_template()
    statement['content'] = '只有她愿意才继续；并没有承诺已经实现。'
    edit['document']['statements'] = [statement]
    asyncio.run(revisions.save('m1', edit))
    engine = EmbeddingEngine({'buckets_dir':str(tmp_path), 'embedding':{
        'api_key':'synthetic-test-key', 'model':'synthetic', 'base_url':'https://synthetic.invalid'}})
    requests = []
    async def request(_endpoint, _key, _model, text, **kwargs):
        requests.append(text)
        return 200, {'data':[{'index':0,'embedding':[1.0,0.0]}]}
    monkeypatch.setattr(engine, '_request_embedding', request)
    worker = MemoryProjectionWorker(config={}, authority=authority, bucket_manager=manager, embedding_engine=engine)
    bucket = asyncio.run(manager.get('b1'))
    revision = authority.get_memory('m1')
    result = asyncio.run(worker._project_embedding_units(bucket, 'm1', 2, revision['body_sha256']))
    assert result['text_projection_version'] == TEXT_PROJECTION_VERSION
    assert result['metadata_complete'] is True
    semantic = authority.get_memory_revision('m1', 2)['metadata']['memory_l1']
    assert requests[0] == l1_retrieval_text(semantic, body=bucket['content'])
    assert statement['content'] in requests[0]
    with sqlite3.connect(engine.db_path) as db:
        rows = db.execute('SELECT source_revision,parent_memory_id,text_projection_version,input_truncated_chars FROM embeddings').fetchall()
    assert rows and all(row == (2, 'm1', TEXT_PROJECTION_VERSION, 0) for row in rows)


@pytest.mark.parametrize('oversized', [False, True])
def test_complete_paragraph_packing_obeys_actual_budget_and_preserves_origin(tmp_path, monkeypatch, oversized):
    revisions, authority, manager = setup_service(tmp_path)
    paragraphs = [('第' + str(index) + '段。' + '完整经历。' * (120 if oversized else 40)) for index in range(4)]
    edit = request_for(revisions)
    edit['document']['narrative']['body'] = '\n\n'.join(paragraphs)
    stmt = statement_template()
    stmt['content'] = '没有承诺；只有她愿意才继续。'
    stmt['conditions'].update(state='known', text='只有她愿意')
    edit['document']['statements'] = [stmt]
    asyncio.run(revisions.save('m1', edit))
    engine = EmbeddingEngine({'buckets_dir':str(tmp_path), 'embedding':{'api_key':'synthetic-test-key',
        'model':'synthetic', 'base_url':'https://synthetic.invalid', 'max_chars':500}})
    sent = []
    async def request(_endpoint, _key, _model, text, **kwargs):
        sent.append(text)
        return 200, {'data':[{'embedding':[1.0,0.0]}]}
    monkeypatch.setattr(engine, '_request_embedding', request)
    worker = MemoryProjectionWorker(config={}, authority=authority, bucket_manager=manager, embedding_engine=engine)
    bucket = asyncio.run(manager.get('b1'))
    revision = authority.get_memory('m1')
    async def project():
        return await worker._project_async('embedding', 'm1', 2, revision['body_sha256'], enabled=True,
            action=lambda:worker._project_embedding_units(bucket, 'm1', 2, revision['body_sha256']))
    result = asyncio.run(project())
    if oversized:
        assert result['status'] == 'pending_rebuild'
        assert result['details']['reason'] == 'embedding_semantic_unit_exceeds_character_budget'
        assert result['details']['max_chars'] == 500
        assert sent == []
        # The same complete material works when the owner raises the existing
        # adjustable bound; no source rewrite or automatic summary is needed.
        engine.max_chars = 1500
        result = asyncio.run(project())
    assert result['status'] == 'projected'
    assert all(len(text) <= engine.max_chars for text in sent)
    assert all(any(paragraph in text for text in sent) for paragraph in paragraphs)
    assert any(stmt['content'] in text and '只有她愿意' in text for text in sent)
    with sqlite3.connect(engine.db_path) as db:
        rows = db.execute('SELECT source_unit_id,source_origin_unit_id,input_truncated_chars FROM embeddings').fetchall()
    assert rows and any(row[0] != row[1] for row in rows)
    assert all(row[1] and row[2] == 0 for row in rows)
    # The existing maintenance verifier must reproduce the same packing and
    # hashes rather than checking the old single truncated seed.
    from scripts import repair_recall_r1_embeddings as repair
    monkeypatch.setattr(repair, '_resolve_prompts', lambda *args: ('', ''))
    verified = asyncio.run(repair._verify_projection_rows(config={}, paths={
        'authority':Path(authority.path), 'embeddings':Path(engine.db_path)},
        worker=worker, embedding=engine, mirror=object(), candidate={
            'memory_id':'m1','bucket_id':'b1','revision':2,'body_sha256':revision['body_sha256'],'has_l1':True}))
    assert verified['ok'] is True and verified['rows_checked'] == len(rows)
    async def queries(texts, **kwargs): return [[1.0,0.0] for _ in texts]
    monkeypatch.setattr(engine, 'query_embeddings', queries)
    result = asyncio.run(engine.search_similar_queries(['query'], eligible_ids={'b1'}, top_k=1,
        index_metadata={'b1':{'revision':2, 'memory_id':'m1','body_sha256':revision['body_sha256'],
                             'text_projection_version':TEXT_PROJECTION_VERSION}}))
    assert result[0]['unit_id'] in {row[1] for row in rows}
    assert result[0]['index_status'] == 'verified'


def test_expanded_unit_count_respects_original_all_or_none_repair_budget(tmp_path, monkeypatch):
    revisions, authority, manager = setup_service(tmp_path)
    edit = request_for(revisions)
    edit['document']['narrative']['body'] = '\n\n'.join('完整自然段。' * 40 for _ in range(6))
    asyncio.run(revisions.save('m1', edit))
    engine = EmbeddingEngine({'buckets_dir':str(tmp_path), 'embedding':{'api_key':'synthetic',
        'base_url':'https://synthetic.invalid','model':'synthetic','max_chars':500}})
    async def forbidden(*args, **kwargs): raise AssertionError('provider must remain untouched')
    monkeypatch.setattr(engine, '_request_embedding', forbidden)
    worker = MemoryProjectionWorker(config={}, authority=authority, bucket_manager=manager, embedding_engine=engine)
    bucket = asyncio.run(manager.get('b1'))
    revision = authority.get_memory('m1')
    result = asyncio.run(worker._project_async('embedding', 'm1', 2, revision['body_sha256'], enabled=True,
        action=lambda:worker._project_embedding_units(bucket, 'm1', 2, revision['body_sha256'], max_units=1)))
    assert result['status'] == 'pending_rebuild'
    assert result['details']['reason'] == 'embedding_unit_budget_exhausted'
    assert result['details']['required_units'] > 1
    with sqlite3.connect(engine.db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0] == 0
