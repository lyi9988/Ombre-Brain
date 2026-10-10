"""Offline reuse tests; loopback HTTP only, no provider or owner data."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sqlite3
import threading

import httpx
import pytest

from embedding_engine import EmbeddingEngine
from embedding_vector_cache import VectorDecodeCache


def test_decode_reuses_only_immutable_numeric_value():
    cache = VectorDecodeCache()
    cold, hot = {}, {}
    first, valid = cache.decode('[0.25, 1, -0.5]', diagnostics=cold)
    second, valid_again = cache.decode('[0.25, 1, -0.5]', diagnostics=hot)
    assert valid and valid_again and first is second
    assert first == (0.25, 1, -0.5)
    assert cold == {"misses": 1} and hot == {"hits": 1}
    with pytest.raises(TypeError):
        second[0] = 999


@pytest.mark.parametrize('payload', ['[]', '{}', 'null', 'false', '1',
                                     '[true,0]', '["1",0]', '[NaN,0]', '[Infinity,0]'])
def test_invalid_vectors_are_never_cached(payload):
    cache = VectorDecodeCache()
    for _ in range(2):
        debug = {}
        _, valid = cache.decode(payload, diagnostics=debug)
        assert not valid and debug == {"misses": 1, "invalid": 1}
        assert cache.snapshot()['entries'] == 0


def test_bad_json_keeps_decode_exception():
    cache = VectorDecodeCache()
    with pytest.raises(ValueError):
        cache.decode('[broken')
    assert cache.snapshot()['entries'] == 0


def test_lru_entry_and_byte_bounds_and_clear():
    cache = VectorDecodeCache(max_entries=1)
    cache.decode('[1,0]')
    debug = {}
    cache.decode('[0,1]', diagnostics=debug)
    assert debug == {'misses': 1, 'evictions': 1}
    assert cache.snapshot()['entries'] == 1
    assert cache.snapshot()['accounted_bytes'] <= cache.max_bytes
    cache.clear()
    assert cache.snapshot()['entries'] == cache.snapshot()['accounted_bytes'] == 0
    small = VectorDecodeCache(max_bytes=1)
    debug = {}
    assert small.decode('[1,0]', diagnostics=debug) == ((1, 0), True)
    assert debug == {'misses': 1, 'bypassed': 1}
    assert small.snapshot()['entries'] == 0


def test_concurrent_decode_does_not_double_account():
    cache = VectorDecodeCache()
    payload = json.dumps([0.25] * 1024)
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda _: cache.decode(payload), range(32)))
    assert len({id(value) for value, _valid in rows}) == 1
    assert cache.snapshot()['entries'] == 1
    single = VectorDecodeCache()
    single.decode(payload)
    assert cache.snapshot()['accounted_bytes'] == single.snapshot()['accounted_bytes']


@pytest.fixture
def search_engine(tmp_path, monkeypatch):
    engine = EmbeddingEngine({'buckets_dir': str(tmp_path / 'buckets'), 'embedding': {
        'api_key': 'offline-key', 'base_url': 'https://embedding.invalid/v1',
        'model': 'offline-model', 'dimensions': 2,
    }})
    engine._store_embedding('a', [1.0, 0.0], source_revision=1,
                            source_sha256='body-a', source_unit_id='unit-a',
                            parent_memory_id='memory-a', input_text='offline fixture')
    calls = []

    async def request(_endpoint, _key, _model, input_value, **_kwargs):
        calls.append(input_value)
        return 200, {'data': [{'index': 0, 'embedding': [1.0, 0.0]}]}

    monkeypatch.setattr(engine, '_request_embedding', request)
    metadata = {'a': {'revision': 1, 'body_sha256': 'body-a', 'memory_id': 'memory-a'}}
    yield engine, metadata, calls
    asyncio.run(engine.close())


def search(engine, metadata, *, query='offline query', eligible=None):
    debug = {}
    rows = asyncio.run(engine.search_similar_queries(
        [query], eligible_ids=set(metadata) if eligible is None else eligible,
        index_metadata=metadata, cache_debug=debug))
    return rows, debug


def test_warm_decode_preserves_scores_and_query_cache(search_engine):
    engine, metadata, calls = search_engine
    cold, cold_debug = search(engine, metadata)
    hot, hot_debug = search(engine, metadata)
    assert cold == hot and cold[0]['score'] == 1.0
    assert cold_debug['vector_decode_cache']['misses'] == 1
    assert hot_debug['vector_decode_cache']['hits'] == 1
    assert len(calls) == 1  # Existing query cache remains the only query cache.
    assert hot_debug['provider_requests'] == 0


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('revision', 2, 'stale_memory_revision'),
    ('body_sha256', 'new-body', 'stale_memory_revision'),
    ('memory_id', 'other-memory', 'stale_memory_revision'),
])
def test_warm_vectors_do_not_cache_authority_decisions(search_engine, field, value, reason):
    engine, metadata, calls = search_engine
    search(engine, metadata)
    metadata['a'][field] = value
    rows, debug = search(engine, metadata, query='different query')
    assert not rows and debug['index_rejections'][reason] == 1
    assert debug['vector_decode_cache']['hits'] == 1
    assert len(calls) == 1


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('model', 'different-model', 'embedding_space_mismatch'),
    ('dimension', 3, 'embedding_space_mismatch'),
    ('provider', 'https://other.invalid/v1', 'embedding_provider_mismatch'),
    ('preparation_sha256', 'different-preparation', 'embedding_preparation_mismatch'),
])
def test_warm_vectors_recheck_current_row_contract(search_engine, field, value, reason):
    engine, metadata, calls = search_engine
    search(engine, metadata)
    # Field names are fixed parameters of this offline test, never user input.
    with sqlite3.connect(engine.db_path) as db:
        db.execute(f'UPDATE embeddings SET {field}=?', (value,))
    rows, debug = search(engine, metadata, query='different query')
    assert not rows and debug['index_rejections'][reason] == 1
    assert debug['vector_decode_cache']['hits'] == 1
    assert len(calls) == 1


def test_warm_vectors_do_not_resurrect_revoked_or_deleted_sources(search_engine):
    engine, metadata, calls = search_engine
    search(engine, metadata)
    rows, _ = search(engine, metadata, eligible=set(), query='after revoke')
    assert not rows and len(calls) == 1
    with sqlite3.connect(engine.db_path) as db:
        db.execute('DELETE FROM embeddings')
    rows, debug = search(engine, metadata, query='after delete')
    assert not rows and len(calls) == 1
    assert debug['index_rows_read'] == 0 and debug['missing_bucket_count'] == 1


def test_serialized_value_change_cannot_hit_old_decoding(search_engine):
    engine, metadata, _calls = search_engine
    search(engine, metadata)
    with sqlite3.connect(engine.db_path) as db:
        db.execute('UPDATE embeddings SET embedding=?', ('[0.0,1.0]',))
    rows, debug = search(engine, metadata)
    assert rows[0]['score'] == 0.0
    assert debug['vector_decode_cache']['misses'] == 1
    assert debug['vector_decode_cache']['hits'] == 0


def test_runtime_and_diagnostics_do_not_expose_vectors_or_keys(search_engine):
    engine, metadata, _ = search_engine
    _rows, debug = search(engine, metadata)
    serialized = json.dumps({'runtime': engine.runtime_debug(), 'decode': debug['vector_decode_cache']})
    assert 'offline-key' not in serialized and 'body-a' not in serialized
    assert 'memory-a' not in serialized and '[1.0, 0.0]' not in serialized


@pytest.mark.parametrize('new_connection', [True, False, None])
def test_http_trace_reports_observed_reuse_not_guessed_reuse(tmp_path, new_connection):
    engine = EmbeddingEngine({'buckets_dir': str(tmp_path), 'embedding': {
        'api_key': 'offline-key', 'base_url': 'https://embedding.invalid/v1'}})

    async def run():
        async def handle(request):
            trace = request.extensions['trace']
            if new_connection is True:
                for phase in ('connect_tcp', 'start_tls'):
                    await trace('connection.' + phase + '.started', {'secret': 'do-not-retain'})
                    await trace('connection.' + phase + '.complete', {})
            if new_connection is not None:
                await trace('http11.send_request_headers.started', {'secret': 'do-not-retain'})
            return httpx.Response(200, json={'data': [{'embedding': [1, 0]}]})

        engine._new_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            _, response = await engine._request_embedding(
                'https://embedding.invalid/v1/embeddings', 'offline-key', 'offline-model', 'offline')
            timing = response['_local_http_timing']
            assert timing['connection_reused'] is (None if new_connection is None else not new_connection)
            assert timing['keepalive_expiry_seconds'] == 120.0
            assert 'do-not-retain' not in json.dumps(timing)
        finally:
            await engine.close()

    asyncio.run(run())


def test_real_loopback_http_pool_reuses_connection_without_probe_or_retry(tmp_path, monkeypatch):
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    state = {'connections': 0, 'requests': 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            state['connections'] += 1

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            state['requests'] += 1
            body = b'{"data":[{"embedding":[1,0]}]}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}/embeddings'
    engine = EmbeddingEngine({'buckets_dir': str(tmp_path), 'embedding': {
        'api_key': 'offline-key', 'base_url': endpoint}})

    async def run():
        try:
            _, first = await engine._request_embedding(endpoint, 'offline-key', 'offline-model', 'first')
            _, second = await engine._request_embedding(endpoint, 'offline-key', 'offline-model', 'second')
            assert first['_local_http_timing']['connection_reused'] is False
            assert second['_local_http_timing']['connection_reused'] is True
            assert state == {'connections': 1, 'requests': 2}
            assert engine.client._transport._pool._keepalive_expiry == 120.0
            assert engine.runtime_debug()['connection_fallback_count'] == 0
        finally:
            await engine.close()

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
