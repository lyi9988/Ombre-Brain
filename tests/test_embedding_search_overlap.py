"""Provider overlap and cancellation coverage for search_similar_queries."""
import asyncio
import json
import sqlite3
import time

import httpx
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


@pytest.mark.parametrize("stopping,exit_kind", [
    ("starter", "cancel"), ("starter", "deadline"), ("follower", "cancel"), ("follower", "deadline"),
])
def test_shared_query_survives_one_waiter_stopping(tmp_path, monkeypatch, stopping, exit_kind):
    engine = _engine(tmp_path)
    provider_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        nonlocal calls
        calls += 1
        provider_started.set()
        await release.wait()
        return _query_response(input_value)

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        def deadline_for(name):
            return time.monotonic() + 0.2 if name == stopping and exit_kind == "deadline" else None

        starter = asyncio.create_task(engine.query_embeddings(
            ["shared query"], deadline=deadline_for("starter"),
        ))
        await asyncio.wait_for(provider_started.wait(), timeout=1)
        debug = {}
        follower = asyncio.create_task(engine.query_embeddings(
            ["shared query"], cache_debug=debug, deadline=deadline_for("follower"),
        ))
        await asyncio.sleep(0)
        assert debug["status"] == "singleflight"

        leaving, survivor = (starter, follower) if stopping == "starter" else (follower, starter)
        if exit_kind == "cancel":
            leaving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await leaving
        else:
            await asyncio.wait({leaving}, timeout=0.8)
            assert leaving.done()
            with pytest.raises(asyncio.TimeoutError):
                await leaving

        release.set()
        assert await survivor == [[1.0, 0.0]]
        assert calls == 1
        assert not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


def test_last_waiter_recancels_during_close_without_harming_new_flight(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.query_cache_ttl_seconds = 0
    started, closing, closed = (asyncio.Event() for _ in range(3))
    started_new, release_old, release_new = (asyncio.Event() for _ in range(3))
    calls = 0

    async def fake_request(_endpoint, _api_key, _model, input_value, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Future()
            finally:
                closing.set()
                await release_old.wait()
                closed.set()
        started_new.set()
        await release_new.wait()
        return _query_response(input_value)

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        old = asyncio.create_task(engine.query_embeddings(["same"]))
        await asyncio.wait_for(started.wait(), 1)
        old.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        old.cancel()
        new = asyncio.create_task(engine.query_embeddings(["same"]))
        await asyncio.wait_for(started_new.wait(), 1)
        assert not old.done() and not closed.is_set()
        release_old.set()
        with pytest.raises(asyncio.CancelledError):
            await old
        assert closed.is_set()
        debug = {}
        peer = asyncio.create_task(engine.query_embeddings(["same"], cache_debug=debug))
        await asyncio.sleep(0)
        assert debug["status"] == "singleflight"
        release_new.set()
        assert await asyncio.gather(new, peer) == [[[1.0, 0.0]], [[1.0, 0.0]]]
        assert calls == 2 and not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


def test_last_waiter_during_transport_deadline_does_not_cancel_close_twice(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.REQUEST_BUDGET_SECONDS = 0.05
    closing, release, closed = (asyncio.Event() for _ in range(3))

    async def fake_request(*_args, **_kwargs):
        try:
            await asyncio.Future()
        finally:
            closing.set()
            await release.wait()
            closed.set()

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        leaving = asyncio.create_task(engine.query_embeddings(["same"]))
        await asyncio.wait_for(closing.wait(), 1)
        leaving.cancel()  # The transport deadline has already started closing.
        await asyncio.sleep(0)
        assert not leaving.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await leaving
        assert closed.is_set() and not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


def test_late_waiter_hits_cache_when_completion_and_cancel_coincide(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    returned = asyncio.Event()
    calls = 0

    async def fake_request(_endpoint, _api_key, _model, input_value, **_kwargs):
        nonlocal calls
        calls += 1
        returned.set()
        return _query_response(input_value)

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        leaving = asyncio.create_task(engine.query_embeddings(["same"]))
        await returned.wait()
        assert not leaving.done()
        leaving.cancel()  # The validated vector is already published; the caller has not resumed.
        debug = {}
        assert await engine.query_embeddings(["same"], cache_debug=debug) == [[1.0, 0.0]]
        assert debug["status"] == "hit" and calls == 1
        with pytest.raises(asyncio.CancelledError):
            await leaving
        assert not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


def test_concurrent_query_batches_only_request_missing_vectors(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = []

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        inputs = input_value if isinstance(input_value, list) else [input_value]
        labels = [text.rsplit("Query: ", 1)[-1] for text in inputs]
        calls.append(set(labels))
        if len(calls) == 1:
            first_started.set()
            await release_first.wait()
        return 200, {"data": [
            {"index": i, "embedding": [float(sum(map(ord, label))), 1.0]}
            for i, label in enumerate(labels)
        ]}

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        first = asyncio.create_task(engine.query_embeddings(["shared", "first-only"]))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        debug = {}
        second = asyncio.create_task(engine.query_embeddings(
            ["second-only", "shared", "second-only"], cache_debug=debug,
        ))
        await asyncio.sleep(0)
        release_first.set()
        first_vectors, second_vectors = await asyncio.gather(first, second)
        assert calls == [{"shared", "first-only"}, {"second-only"}]
        assert second_vectors[1] == first_vectors[0]
        assert second_vectors[0] == second_vectors[2] != second_vectors[1]
        assert debug["shared_provider_request"] is True
        assert debug["provider_batch_size"] == 1
        assert not engine._query_batch_inflight
        _no_pending_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("cache_state", ["warm", "partial", "overlap"])
def test_query_rejects_mixed_dimensions_and_can_recover(tmp_path, monkeypatch, cache_state):
    engine = _engine(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()
    coherent = False
    calls = []

    async def fake_request(_endpoint, _api_key, _model, input_value, **_kwargs):
        inputs = input_value if isinstance(input_value, list) else [input_value]
        labels = [text.rsplit("Query: ", 1)[-1] for text in inputs]
        calls.append(set(labels))
        if cache_state == "overlap" and len(calls) == 1:
            started.set()
            await release.wait()
        return 200, {"data": [
            {"index": index, "embedding": [1.0] * (2 if coherent or label == "a" else 3)}
            for index, label in enumerate(labels)
        ]}

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        nonlocal coherent
        owner = None
        if cache_state == "overlap":
            owner = asyncio.create_task(engine.query_embeddings(["a"]))
            await asyncio.wait_for(started.wait(), 1)
        else:
            assert await engine.query_embeddings(["a"]) == [[1.0, 1.0]]
            if cache_state == "warm":
                assert await engine.query_embeddings(["b"]) == [[1.0, 1.0, 1.0]]

        pair = asyncio.create_task(engine.query_embeddings(["a", "b"]))
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(ValueError, match="embedding_batch_dimension_mismatch"):
            await pair
        if owner is not None:
            assert await owner == [[1.0, 1.0]]
        assert calls == [{"a"}, {"b"}]  # No repeat vector request before the mismatch.
        assert not engine._query_cache

        coherent = True
        assert await engine.query_embeddings(["a", "b"]) == [[1.0, 1.0], [1.0, 1.0]]
        assert calls[-1] == {"a", "b"}
        assert await engine.query_embeddings(["a", "b"]) == [[1.0, 1.0], [1.0, 1.0]]
        assert len(calls) == 3
        _no_pending_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("initial_failure", [httpx.ConnectError, httpx.PoolTimeout, httpx.ReadTimeout])
def test_query_connection_fallback_can_be_cancelled_and_still_recovers(
    tmp_path, monkeypatch, initial_failure,
):
    engine = _engine(tmp_path)
    engine.dimension = 2
    fallback_started, fallback_closed = asyncio.Event(), asyncio.Event()
    fallback_calls = 0
    pooled_requests = []

    class Pooled:
        async def post(self, endpoint, **kwargs):
            pooled_requests.append((endpoint, kwargs["headers"], kwargs["json"]))
            raise initial_failure("synthetic transport failure")

    class Fallback:
        def __init__(self, *, follow_redirects):
            assert follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            fallback_closed.set()

        async def post(self, endpoint, **kwargs):
            nonlocal fallback_calls
            fallback_calls += 1
            assert (endpoint, kwargs["headers"], kwargs["json"]) == pooled_requests[-1]
            assert kwargs["json"]["dimensions"] == 2
            if fallback_calls == 1:
                fallback_started.set()
                await asyncio.Future()
            return httpx.Response(200, request=httpx.Request("POST", endpoint),
                                  json=_query_response(kwargs["json"]["input"])[1])

    async def get_client():
        return Pooled()

    def forbidden_sync(*_args, **_kwargs):
        pytest.fail("shared queries must not leave a synchronous request running")

    monkeypatch.setattr(engine, "_ensure_client", get_client)
    monkeypatch.setattr(engine, "_request_embedding_sync", forbidden_sync)
    monkeypatch.setattr(httpx, "AsyncClient", Fallback)

    async def run():
        if initial_failure is httpx.ReadTimeout:
            with pytest.raises(httpx.ReadTimeout):
                await engine.query_embeddings(["same"])
            assert fallback_calls == 0
        else:
            leaving = asyncio.create_task(engine.query_embeddings(["same"]))
            await asyncio.wait_for(fallback_started.wait(), 1)
            leaving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await leaving
            assert fallback_closed.is_set() and not engine._query_cache

            debug = {}
            assert await engine.query_embeddings(["same"], cache_debug=debug) == [[1.0, 0.0]]
            assert fallback_calls == 2 and debug["provider_requests"] == 2
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
