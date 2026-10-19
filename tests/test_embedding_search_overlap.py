"""Provider overlap and cancellation coverage for search_similar_queries."""
import asyncio
import json
import sqlite3
import time

import pytest

from embedding_engine import EmbeddingEngine


def _engine(tmp_path):
    return EmbeddingEngine({
        "buckets_dir": str(tmp_path / "buckets"),
        "embedding": {
            "enabled": True,
            "api_key": "local-test-key",
            "base_url": "https://embedding.example/v1",
            "model": "test-embedding-model",
            "query_cache_ttl_seconds": 300,
        },
    })


def _seed_rows(engine, count):
    snapshot = engine._query_config_snapshot()
    rows = []
    for index in range(count):
        bucket_id = f"candidate-{index:04}"
        rows.append((
            bucket_id,
            json.dumps([float(count - index), 1.0]),
            snapshot["model"],
            2,
            "synthetic",
            bucket_id,
            None,
            f"unit-{index:04}",
            1,
            f"source-hash-{index}",
            f"input-hash-{index}",
            snapshot["document_preparation"],
            snapshot["base_url"].rstrip("/"),
            0,
            None,
            None,
        ))
    with sqlite3.connect(engine.db_path) as db:
        db.executemany(
            """INSERT INTO embeddings
               (bucket_id, embedding, model, dimension, updated_at, parent_bucket_id,
                parent_memory_id, source_unit_id, source_revision, source_sha256,
                input_sha256, preparation_sha256, provider, input_truncated_chars,
                text_projection_version, source_origin_unit_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return {row[0] for row in rows}


def _query_response(input_value):
    count = len(input_value) if isinstance(input_value, list) else 1
    return 200, {
        "data": [
            {"index": index, "embedding": [1.0, 0.0]}
            for index in range(count)
        ],
        "_local_http_timing": {"provider_attempts": 1},
    }


def _no_pending_tasks():
    pending = [task for task in asyncio.all_tasks()
               if task is not asyncio.current_task() and not task.done()]
    assert pending == []


def test_search_starts_one_batched_query_before_all_rows_are_validated_and_reuses_cache(
    tmp_path, monkeypatch,
):
    engine = _engine(tmp_path)
    count = 130
    eligible = _seed_rows(engine, count)
    original_decode = engine._vector_decode_cache.decode
    decode_count = 0
    all_rows_decoded = asyncio.Event()
    provider_started_at_decode_count = []
    provider_calls = []
    release_provider = asyncio.Event()
    query_calls = []
    original_query_embeddings = engine.query_embeddings

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None, dimensions=None):
        provider_calls.append(input_value)
        provider_started_at_decode_count.append(decode_count)
        await release_provider.wait()
        return _query_response(input_value)

    def tracked_decode(payload, *, diagnostics=None):
        nonlocal decode_count
        decode_count += 1
        if decode_count == count:
            all_rows_decoded.set()
        return original_decode(payload, diagnostics=diagnostics)

    async def tracked_query_embeddings(*args, **kwargs):
        query_calls.append(True)
        return await original_query_embeddings(*args, **kwargs)

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    monkeypatch.setattr(engine, "query_embeddings", tracked_query_embeddings)
    monkeypatch.setattr(engine._vector_decode_cache, "decode", tracked_decode)

    async def run():
        first_debug = {}
        first_task = asyncio.create_task(engine.search_similar_queries(
            ["synthetic query"], top_k=3, eligible_ids=eligible, cache_debug=first_debug,
        ))
        await asyncio.wait_for(all_rows_decoded.wait(), timeout=2.0)
        assert len(provider_calls) == 1
        assert len(query_calls) == 1
        assert provider_started_at_decode_count[0] < count
        release_provider.set()
        first_result = await first_task
        assert first_debug["index_rows_read"] == count
        assert first_debug["valid_unit_count"] == count
        assert first_debug["provider_requests"] == 1

        second_debug = {}
        second_result = await engine.search_similar_queries(
            ["synthetic query"], top_k=3, eligible_ids=eligible, cache_debug=second_debug,
        )
        assert second_debug["status"] == "hit"
        assert second_debug["provider_requests"] == 0
        assert len(provider_calls) == 1
        assert len(query_calls) == 2
        assert second_result == first_result
        assert [row["bucket_id"] for row in first_result] == [
            "candidate-0000", "candidate-0001", "candidate-0002",
        ]
        assert decode_count == count * 2

    asyncio.run(run())


@pytest.mark.parametrize("with_stale_row", [False, True], ids=["empty-index", "all-stale"])
def test_search_without_compatible_vectors_never_starts_query_provider(
    tmp_path, monkeypatch, with_stale_row,
):
    engine = _engine(tmp_path)
    eligible = {"candidate-0000"}
    index_metadata = None
    if with_stale_row:
        engine._store_embedding(
            "candidate-0000", [1.0, 0.0], source_revision=1,
            source_sha256="old-body", source_unit_id="unit-0",
            parent_memory_id="candidate-0000", input_text="synthetic source",
        )
        index_metadata = {"candidate-0000": {
            "revision": 2, "body_sha256": "current-body", "memory_id": "candidate-0000",
        }}
    calls = []
    query_calls = []

    async def forbidden_request(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("provider must not run without a compatible stored vector")

    async def forbidden_query(*_args, **_kwargs):
        query_calls.append(True)
        raise AssertionError("query_embeddings must not run without a compatible stored vector")

    monkeypatch.setattr(engine, "_request_embedding", forbidden_request)
    monkeypatch.setattr(engine, "query_embeddings", forbidden_query)
    debug = {}
    result = asyncio.run(engine.search_similar_queries(
        ["synthetic query"], eligible_ids=eligible,
        index_metadata=index_metadata, cache_debug=debug,
    ))

    assert result == []
    assert calls == []
    assert query_calls == []
    assert debug["status"] == "no_compatible_index"
    assert debug["provider_requests"] == 0
    assert debug["valid_unit_count"] == 0
    assert not engine._query_batch_inflight


def test_external_cancel_during_index_filter_cancels_and_awaits_query_child(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    count = 160
    eligible = _seed_rows(engine, count)
    original_decode = engine._vector_decode_cache.decode
    decode_count = 0
    filter_progress = asyncio.Event()
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def fake_request(_endpoint, _api_key, _model, _input_value, *, deadline=None, dimensions=None):
        provider_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            provider_cancelled.set()

    def tracked_decode(payload, *, diagnostics=None):
        nonlocal decode_count
        decode_count += 1
        if decode_count == 40:
            filter_progress.set()
        return original_decode(payload, diagnostics=diagnostics)

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    monkeypatch.setattr(engine._vector_decode_cache, "decode", tracked_decode)

    async def run():
        debug = {}
        search_task = asyncio.create_task(engine.search_similar_queries(
            ["synthetic query"], eligible_ids=eligible, cache_debug=debug,
        ))
        await asyncio.wait_for(filter_progress.wait(), timeout=2.0)
        assert provider_started.is_set()
        assert decode_count < count
        search_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await search_task
        assert provider_cancelled.is_set()
        assert debug["status"] == "cancelled"
        assert debug["error_type"] == "CancelledError"
        assert not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


def test_deadline_cancels_and_awaits_query_child(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    eligible = _seed_rows(engine, 1)
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def fake_request(_endpoint, _api_key, _model, _input_value, *, deadline=None, dimensions=None):
        provider_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            provider_cancelled.set()

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        debug = {}
        with pytest.raises(asyncio.TimeoutError):
            await engine.search_similar_queries(
                ["synthetic query"], eligible_ids=eligible,
                deadline=time.monotonic() + 0.05, cache_debug=debug,
            )
        assert provider_started.is_set()
        assert provider_cancelled.is_set()
        assert debug["error_type"] == "TimeoutError"
        assert not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


def test_similarity_loop_yields_to_external_cancellation(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    count = 256
    eligible = _seed_rows(engine, count)
    similarity_progress = asyncio.Event()
    provider_calls = []
    original_cosine = engine._cosine_similarity
    comparison_count = 0

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None, dimensions=None):
        provider_calls.append(input_value)
        return _query_response(input_value)

    def tracked_cosine(left, right):
        nonlocal comparison_count
        comparison_count += 1
        if comparison_count == 70:
            similarity_progress.set()
        return original_cosine(left, right)

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    monkeypatch.setattr(engine, "_cosine_similarity", tracked_cosine)

    async def run():
        debug = {}
        search_task = asyncio.create_task(engine.search_similar_queries(
            ["synthetic query"], eligible_ids=eligible, cache_debug=debug,
        ))
        await asyncio.wait_for(similarity_progress.wait(), timeout=2.0)
        assert len(provider_calls) == 1
        assert comparison_count < count
        search_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await search_task
        assert debug["status"] == "cancelled"
        assert not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())
