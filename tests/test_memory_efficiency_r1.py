"""Offline production-seam checks: conservative routing and equivalent retrieval."""
import asyncio
import copy
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from dream_engine import DreamEngine, DreamRecord
from embedding_engine import EmbeddingEngine
from gateway import GatewayService
from recall_input import build_recall_input
from recall_preflight import clock_preflight
from test_gateway_sdk_tool_phase import minimal_service, make_request
from test_recall_natural_r1 import _embedding_engine
from turn_clock import TIME_HEADER, ZONE_HEADER, BODY_HEADER


def clock_payload(texts, *, history=None):
    clock = {'role': 'system', 'content': 'clock reference fixture'}
    messages = [clock, *(history or []), *[{'role': 'user', 'content': text} for text in texts]]
    start = len(messages) - len(texts)
    trace = {'coverage': {'context_revision': 2, 'items': [
        {'message_index': i, 'turn_member': i >= start, 'event_id': f'event-{i}'}
        for i in range(1, len(messages))]}}
    metadata = {'source': 'aizizhu_turn_snapshot', 'physical_verified': True,
                'physical_matches': [{'role': 'system', 'message_index': 0}]}
    def extract(value):
        if isinstance(value, str):
            return value
        return ''.join(p.get('text', '') for p in value if isinstance(p, dict))
    natural = build_recall_input(messages, current_query=extract(texts[-1]),
                                 text_extractor=extract, cleaner=lambda s: s,
                                 trace_context=trace)
    return natural, messages, metadata, trace


@pytest.mark.parametrize('text', ['现在几点了？', '请问现在是什么时间', '今天是几号？',
    '今天星期几', '当前时间是多少', 'What time is it now?', 'what is the date today?'])
def test_clock_only_is_a_verified_full_clause(text):
    natural, messages, clock, _ = clock_payload([text])
    assert clock_preflight(natural, messages, clock)['skip_dynamic_recall']


@pytest.mark.parametrize('text', ['几点？', '那天几点', '今天是什么日子', '今天是我的生日吗',
    '现在几点，记得我们约好了什么吗', '今天几号，我又难受了', '今天有约定吗', '又难受了',
    '嗯', '你记得吗', '上次这个时间我们做了什么', '把“现在几点”翻译一下',
    '现在几点，帮我设置提醒', '几点了，昨天那件事呢', '现在几点，现在我不舒服',
    [{'type': 'text', 'text': '现在几点'}, {'type': 'image_url', 'image_url': {'url': 'fixture'}}]])
def test_history_emotion_ambiguous_mixed_and_media_keep_recall(text):
    natural, messages, clock, _ = clock_payload([text])
    assert not clock_preflight(natural, messages, clock)['skip_dynamic_recall']


def test_entire_staged_batch_not_just_last_question():
    natural, messages, clock, _ = clock_payload(['又难受了', '现在几点？'])
    assert not clock_preflight(natural, messages, clock)['skip_dynamic_recall']
    natural, messages, clock, _ = clock_payload(['今天几号', '现在几点？'])
    assert clock_preflight(natural, messages, clock)['skip_dynamic_recall']


@pytest.mark.parametrize('change', ['missing-clock', 'duplicate-clock', 'no-coverage', 'partial', 'transformed'])
def test_missing_or_ambiguous_evidence_never_skips(change):
    natural, messages, clock, _ = clock_payload(['现在几点'])
    if change == 'missing-clock': clock = {}
    if change == 'duplicate-clock': clock['physical_verified'] = False
    if change == 'no-coverage': natural['metadata']['input_source'] = 'legacy_selected_messages'
    if change == 'partial': natural['metadata']['current_input_complete'] = False
    if change == 'transformed': messages[-1]['content'] += ' memory request'
    assert not clock_preflight(natural, messages, clock)['skip_dynamic_recall']


@pytest.mark.parametrize('scheduled', [False, True])
def test_gateway_clock_skips_before_buckets_but_preserves_configured_core_and_persona(tmp_path, scheduled):
    service = minimal_service(tmp_path)
    service.persona_engine = SimpleNamespace(enabled=True, profile_id='test',
        format_state_block=lambda state: 'persona kept',
        build_pre_reply_guidance=async_return({'enabled': True}))
    service.current_inner_state_interval_rounds = 1
    service.core_memory_interval_rounds = 2
    service._should_inject_interval = lambda sid, interval: interval == 1 or scheduled and interval == 2
    service.recalled_budget = 100
    service.retrieval_mode = 'graph'
    service.diffusion_inject_max_items = 2
    calls = []
    async def buckets(**kwargs):
        calls.append('buckets'); return [{'id': 'core'}]
    async def unexpected(*args, **kwargs):
        raise AssertionError('clock must not enter remote/dynamic selection')
    async def core(rows):
        assert rows == [{'id': 'core'}]; calls.append('core'); return 'core kept'
    service._list_gateway_buckets = buckets
    service._route_memory_sentinel = unexpected
    service._select_dynamic_moments = unexpected
    service._build_dream_context_block = unexpected
    service._build_core_memory_block = core
    captured = {}
    service._build_injected_context_messages = lambda **kw: (captured.update(kw) or '', '')
    natural, messages, _, trace = clock_payload(['现在几点？'])
    request = make_request({TIME_HEADER: '2026-10-10T00:00:00+00:00', ZONE_HEADER: 'Asia/Shanghai',
        BODY_HEADER: hashlib.sha256(messages[0]['content'].encode()).hexdigest(),
        'X-Ombre-Canonical-Source-Event-Id': 'event-current',
        'X-Ombre-Canonical-Assistant-Source-Event-Id': 'event-reply'})
    result = asyncio.run(service.prepare_payload({'model': 'guyan', 'messages': messages,
        '_ombre_trace_context': trace}, 'main', include_debug=True, request=request))
    assert calls == (['buckets', 'core'] if scheduled else [])
    assert captured['persona_block'] == 'persona kept'
    assert captured['core_memory'] == ('core kept' if scheduled else '')
    assert result[2]['prepare_timing_debug']['recall_preflight']['reason'] == 'pure_current_clock'
    assert result[2]['prepare_timing_debug']['recall_status']['embedding_request_count'] == 0
    assert result[0]['messages'] == messages


def async_return(value):
    async def run(*args, **kwargs): return value
    return run


@pytest.mark.parametrize('texts', [['那天几点'], ['又难受了', '现在几点'], ['今天几号，约定还记得吗']])
def test_gateway_mixed_and_emotional_input_still_enters_ordinary_retrieval(tmp_path, texts):
    service = minimal_service(tmp_path)
    service.persona_engine.enabled = False
    service.recalled_budget = 100
    service.retrieval_mode = 'graph'
    service.diffusion_inject_max_items = 2
    service._route_memory_sentinel = async_return({})
    service._route_domain_sentinel = async_return({})
    service.graph_bucket_rerank_enabled = False
    calls = []
    async def select(selector, query, *args, **kwargs):
        calls.append((query, kwargs['allow_semantic'])); return [], [], [], [], {}
    service._select_recall_with_fallback = select
    _, messages, _, trace = clock_payload(texts)
    request = make_request({TIME_HEADER: '2026-10-10T00:00:00+00:00', ZONE_HEADER: 'Asia/Shanghai',
        BODY_HEADER: hashlib.sha256(messages[0]['content'].encode()).hexdigest(),
        'X-Ombre-Canonical-Source-Event-Id': 'event-current',
        'X-Ombre-Canonical-Assistant-Source-Event-Id': 'event-reply'})
    result = asyncio.run(service.prepare_payload({'model': 'guyan', 'messages': messages,
        '_ombre_trace_context': trace}, 'main', include_debug=True, request=request))
    assert calls == [('\n'.join(texts), True)]
    assert not result[2]['prepare_timing_debug']['recall_preflight']['skip_dynamic_recall']


def dream_fixture(tmp_path, cues=()):
    engine = DreamEngine.__new__(DreamEngine)
    engine.enabled = engine.surface_enabled = True
    engine.retain_after_surface = True
    engine.min_surface_age_hours = 1
    engine.alpha_subordinate = .2
    engine.attempt_threshold = .7
    engine.surface_threshold = .8
    engine.spontaneous_surface_prob = 0
    engine.max_surface_attempts = 6
    engine.claim_ttl_minutes = 5
    engine.tz = ZoneInfo('Asia/Shanghai')
    engine.identity = {'ai_name': 'test'}
    engine._now = lambda now: now or datetime.now(timezone.utc)
    record = DreamRecord({'dream_id': 'd1', 'generated_at': (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                         'recall_cues': list(cues)}, 'dream fixture', tmp_path / 'd1.md')
    record.path.touch()
    rows = [record]; mutations = []
    engine.list_records = lambda: list(rows)
    def write(metadata, body):
        mutations.append(metadata); rows[0] = DreamRecord(metadata, body, record.path); return rows[0]
    engine._write_record = write
    engine._log_event = lambda *args: None
    engine._delete_record = lambda *args: mutations.append('deleted')
    return engine, rows, mutations


def test_dream_without_vectors_does_not_call_provider_but_keeps_text_and_state(tmp_path):
    engine, rows, mutations = dream_fixture(tmp_path)
    async def forbidden(*args, **kwargs): raise AssertionError('no comparison vectors')
    embedding = SimpleNamespace(enabled=True, get_embeddings=async_return({}), query_embedding=forbidden)
    result = asyncio.run(engine.surface_with_status(query='ordinary query', embedding_engine=embedding))
    assert result['reason'] == 'no_resonance'
    assert result['semantic_status'] == 'no_compatible_dream_vectors'
    assert not mutations
    rows[0].metadata['recall_cues'] = ['a complete matching cue']
    result = asyncio.run(engine.surface_with_status(query='a complete matching cue', embedding_engine=embedding))
    assert result['status'] == 'injected'
    assert rows[0].surfaced


def test_dream_semantic_vectors_loaded_once_and_snapshot_is_shared(tmp_path):
    engine, rows, mutations = dream_fixture(tmp_path)
    calls = []; snapshot = {'model': 'frozen'}
    async def vectors(ids, **kwargs):
        calls.append(('vectors', kwargs)); return {'d1': [1, 0]}
    async def query(text, **kwargs):
        calls.append(('query', kwargs)); return [1, 0]
    embedding = SimpleNamespace(enabled=True, _query_config_snapshot=lambda: snapshot,
        get_embeddings=vectors, query_embedding=query)
    result = asyncio.run(engine.surface_with_status(query='semantic-only paraphrase', embedding_engine=embedding))
    assert result['status'] == 'injected'
    assert [c[0] for c in calls] == ['vectors', 'query']
    assert all(c[1]['config_snapshot'] is snapshot for c in calls)


def test_dream_index_failure_or_cancel_does_not_consume(tmp_path):
    engine, rows, mutations = dream_fixture(tmp_path)
    async def failed(ids): raise OSError('fixture')
    result = asyncio.run(engine.surface_with_status(query='some query',
        embedding_engine=SimpleNamespace(enabled=True, get_embeddings=failed)))
    assert result['reason'] == 'dream_index_unavailable'
    async def cancelled(ids): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.surface_with_status(query='some query',
            embedding_engine=SimpleNamespace(enabled=True, get_embeddings=cancelled)))
    assert not mutations and not rows[0].surfaced


def test_dream_local_only_does_not_read_embeddings(tmp_path):
    engine, rows, mutations = dream_fixture(tmp_path, ['a matching cue'])
    async def forbidden(*args, **kwargs): raise AssertionError('local only')
    result = asyncio.run(engine.surface_with_status(query='a matching cue', allow_semantic=False,
        embedding_engine=SimpleNamespace(enabled=True, get_embeddings=forbidden, query_embedding=forbidden)))
    assert result['status'] == 'injected'


def test_eligible_vector_read_is_bounded_and_preserves_tie_order(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    connect = sqlite3.connect
    def limited_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', limited_connect)
    ids = {f'b-{i}' for i in range(405)}
    with sqlite3.connect(engine.db_path) as db:
        for i in range(410):
            db.execute('INSERT INTO embeddings(bucket_id, embedding, model, dimension, updated_at, parent_bucket_id) VALUES(?,?,?,?,?,?)',
                (f'unit-{i}', '[1,0]', engine.model, 2, 'now', f'b-{i}'))
        # A legacy vector with no parent is still eligible under its own id.
        db.execute('INSERT INTO embeddings(bucket_id, embedding, model, dimension, updated_at) VALUES(?,?,?,?,?)',
                   ('legacy', '[1,0]', engine.model, 2, 'now'))
    ids.add('legacy')
    async def query(texts, **kwargs): return [[1, 0] for _ in texts]
    engine.query_embeddings = query
    debug = {}
    hits = asyncio.run(engine.search_similar_queries(['text'], eligible_ids=ids, top_k=500, cache_debug=debug))
    assert [h['bucket_id'] for h in hits] == [f'b-{i}' for i in range(405)] + ['legacy']
    assert debug['index_rows_read'] == 406
    assert debug['valid_unit_count'] == 406
    assert debug['coverage_status'] == 'complete'
    for name in ('index_read_ms', 'index_filter_ms', 'similarity_ms'):
        assert debug[name] >= 0


def test_dream_vector_model_uses_frozen_snapshot(tmp_path):
    engine = _embedding_engine(tmp_path)
    with sqlite3.connect(engine.db_path) as db:
        db.execute('INSERT INTO embeddings(bucket_id, embedding, model, dimension, updated_at) VALUES(?,?,?,?,?)',
                   ('d1', '[1,0]', engine.model, 2, 'now'))
    snapshot = engine._query_config_snapshot()
    engine.model = 'changed-after-snapshot'
    assert asyncio.run(engine.get_embeddings(['d1'])) == {}
    assert asyncio.run(engine.get_embeddings(['d1'], config_snapshot=snapshot)) == {'d1': [1, 0]}


def test_dream_spontaneous_path_remains_available_without_vectors(tmp_path, monkeypatch):
    engine, rows, _ = dream_fixture(tmp_path)
    engine.spontaneous_surface_prob = .1
    monkeypatch.setattr('dream_engine.random.random', lambda: 0)
    result = asyncio.run(engine.surface_with_status(query='ordinary query',
        embedding_engine=SimpleNamespace(enabled=True, get_embeddings=async_return({}))))
    assert result['status'] == 'injected' and rows[0].surfaced


def test_dream_invalid_stored_vectors_are_not_used(tmp_path):
    engine = _embedding_engine(tmp_path)
    with sqlite3.connect(engine.db_path) as db:
        for name, value in [('object', '{}'), ('not-number', '["text",0]'), ('not-finite', '[NaN,0]')]:
            db.execute('INSERT INTO embeddings(bucket_id, embedding, model, dimension, updated_at) VALUES(?,?,?,?,?)',
                       (name, value, engine.model, 2, 'now'))
    assert asyncio.run(engine.get_embeddings(['object', 'not-number', 'not-finite'])) == {}
