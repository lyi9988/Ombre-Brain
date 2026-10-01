"""Offline regressions for the explicit embedding output-dimension contract.

All provider traffic is captured by HTTPX/urllib test doubles. These synthetic
checks are not provider smoke tests or natural owner acceptance.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import urllib.request
from types import SimpleNamespace

import httpx
import pytest

from embedding_engine import EmbeddingEngine
from gateway import GatewayService
from memory_projection_worker import MemoryProjectionWorker


def _engine(tmp_path, *, dimensions=...):
    embedding = {
        "enabled": True,
        "api_key": "local-test-key",
        "base_url": "https://embedding.invalid/v1",
        "model": "test-embedding-model",
        "query_cache_ttl_seconds": 300,
    }
    if dimensions is not ...:
        embedding["dimensions"] = dimensions
    return EmbeddingEngine({"buckets_dir": str(tmp_path / "buckets"), "embedding": embedding})


def _attach_httpx(engine, handler, monkeypatch):
    monkeypatch.setattr(
        engine,
        "_new_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _body(request):
    return json.loads(request.content.decode("utf-8"))


def _embedding_response(request, *, default_dimension=2):
    payload = _body(request)
    input_value = payload["input"]
    count = len(input_value) if isinstance(input_value, list) else 1
    dimension = payload.get("dimensions", default_dimension)
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": [float(index + 1)] * dimension}
                for index in range(count)
            ]
        },
    )


def test_requested_dimensions_validate_bounds_and_reconfigure_can_clear(tmp_path, monkeypatch):
    assert EmbeddingEngine._requested_dimension(None) is None
    assert EmbeddingEngine._requested_dimension(1) == 1
    assert EmbeddingEngine._requested_dimension(65536) == 65536
    assert EmbeddingEngine._requested_dimension("32") == 32

    for invalid in (True, False, 0, 65537, -1, 1.5, "1.5", "", "-1"):
        with pytest.raises(ValueError):
            _engine(tmp_path / f"invalid-{str(invalid).replace('/', '_')}", dimensions=invalid)

    engine = _engine(tmp_path, dimensions=16)
    assert engine.dimension == 16
    payloads = []

    def handler(request):
        payloads.append(_body(request))
        return _embedding_response(request)

    _attach_httpx(engine, handler, monkeypatch)

    async def clear():
        await engine.reconfigure({"embedding": {
            "enabled": True,
            "api_key": "local-test-key",
            "base_url": "https://embedding.invalid/v1",
            "model": "test-embedding-model",
            "dimensions": None,
        }})
        assert engine.dimension is None
        assert len(await engine.query_embedding("query after clearing dimensions")) == 2
        await engine.close()

    asyncio.run(clear())
    assert len(payloads) == 1
    assert "dimensions" not in payloads[0]


def test_async_http_payload_omits_default_and_includes_explicit_single_and_batch_dimensions(
    tmp_path, monkeypatch,
):
    default_payloads = []

    def default_handler(request):
        default_payloads.append(_body(request))
        return _embedding_response(request)

    default_engine = _engine(tmp_path / "default")
    _attach_httpx(default_engine, default_handler, monkeypatch)

    explicit_payloads = []

    def explicit_handler(request):
        explicit_payloads.append(_body(request))
        return _embedding_response(request)

    explicit_engine = _engine(tmp_path / "explicit", dimensions=3)
    _attach_httpx(explicit_engine, explicit_handler, monkeypatch)

    async def run():
        assert len(await default_engine.query_embedding("default single")) == 2
        assert len(await default_engine.query_embeddings(["default batch a", "default batch b"])) == 2
        assert len(await explicit_engine.query_embedding("explicit single")) == 3
        assert len(await explicit_engine.query_embeddings(["explicit batch a", "explicit batch b"])) == 2
        await default_engine.close()
        await explicit_engine.close()

    asyncio.run(run())

    assert len(default_payloads) == 2
    assert all("dimensions" not in payload for payload in default_payloads)
    assert isinstance(default_payloads[0]["input"], str)
    assert isinstance(default_payloads[1]["input"], list)
    assert len(explicit_payloads) == 2
    assert all(payload["dimensions"] == 3 for payload in explicit_payloads)
    assert isinstance(explicit_payloads[0]["input"], str)
    assert isinstance(explicit_payloads[1]["input"], list)


def test_urllib_connect_fallback_serializes_explicit_dimensions(tmp_path, monkeypatch):
    engine = _engine(tmp_path, dimensions=4)
    async_payloads = []
    fallback_payloads = []

    def fail_connect(request):
        async_payloads.append(_body(request))
        raise httpx.ConnectError("synthetic pre-send failure", request=request)

    _attach_httpx(engine, fail_connect, monkeypatch)

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"data": [{"embedding": [0.25] * 4}]}).encode("utf-8")

    def fake_urlopen(request, *, timeout):
        assert timeout > 0
        fallback_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    async def run():
        assert len(await engine.query_embedding("fallback query")) == 4
        await engine.close()

    asyncio.run(run())

    assert len(async_payloads) == 1
    assert async_payloads[0]["dimensions"] == 4
    assert len(fallback_payloads) == 1
    assert fallback_payloads[0]["dimensions"] == 4
    assert fallback_payloads[0]["input"] == async_payloads[0]["input"]


def test_dimension_separates_query_cache_and_concurrent_batch_singleflight(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    class GateTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.payloads = []

        async def handle_async_request(self, request):
            payload = _body(request)
            self.payloads.append(payload)
            if len(self.payloads) >= 2:
                self.started.set()
            await asyncio.wait_for(self.release.wait(), timeout=2)
            return _embedding_response(request)

    transport = GateTransport()
    monkeypatch.setattr(engine, "_new_client", lambda: httpx.AsyncClient(transport=transport))
    base_snapshot = engine._query_config_snapshot()
    two_dim = {**base_snapshot, "dimension": 2}
    three_dim = {**base_snapshot, "dimension": 3}
    query = "same query, separate vector spaces"
    assert engine._query_cache_key(query, snapshot=two_dim) != engine._query_cache_key(
        query, snapshot=three_dim,
    )

    async def run():
        left = asyncio.create_task(engine.query_embeddings([query], config_snapshot=two_dim))
        right = asyncio.create_task(engine.query_embeddings([query], config_snapshot=three_dim))
        await asyncio.wait_for(transport.started.wait(), timeout=1)
        transport.release.set()
        vectors = await asyncio.gather(left, right)
        assert [len(vectors[0][0]), len(vectors[1][0])] == [2, 3]
        assert len(await engine.query_embeddings([query], config_snapshot=two_dim)) == 1
        assert len(await engine.query_embeddings([query], config_snapshot=three_dim)) == 1
        await engine.close()

    asyncio.run(run())

    assert len(transport.payloads) == 2
    assert {payload["dimensions"] for payload in transport.payloads} == {2, 3}


def test_inflight_query_keeps_snapshot_dimension_across_runtime_reconfigure(tmp_path, monkeypatch):
    engine = _engine(tmp_path, dimensions=2)

    class GateTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.payloads = []

        async def handle_async_request(self, request):
            self.payloads.append(_body(request))
            self.started.set()
            await asyncio.wait_for(self.release.wait(), timeout=2)
            return _embedding_response(request)

    transport = GateTransport()
    monkeypatch.setattr(engine, "_new_client", lambda: httpx.AsyncClient(transport=transport))
    old_snapshot = engine._query_config_snapshot()

    async def run():
        task = asyncio.create_task(engine.query_embeddings(
            ["query during hot reload"], config_snapshot=old_snapshot,
        ))
        await asyncio.wait_for(transport.started.wait(), timeout=1)
        await engine.reconfigure({
            "embedding": {
                "enabled": True,
                "api_key": "local-test-key",
                "base_url": "https://embedding.invalid/v1",
                "model": "test-embedding-model",
                "dimensions": 3,
            }
        })
        assert engine.dimension == 3
        transport.release.set()
        vectors = await task
        assert len(vectors[0]) == 2
        await engine.close()

    asyncio.run(run())

    assert len(transport.payloads) == 1
    assert transport.payloads[0]["dimensions"] == 2


def test_wrong_shape_single_batch_and_document_vectors_are_not_cached_or_stored(
    tmp_path, monkeypatch,
):
    engine = _engine(tmp_path, dimensions=3)

    def wrong_shape(request):
        payload = _body(request)
        inputs = payload["input"]
        count = len(inputs) if isinstance(inputs, list) else 1
        return httpx.Response(200, json={
            "data": [
                {"index": index, "embedding": [0.1, 0.2]}
                for index in range(count)
            ]
        })

    _attach_httpx(engine, wrong_shape, monkeypatch)

    async def run():
        assert await engine.query_embedding("wrong single shape") == []
        assert engine._query_cache == {}
        with pytest.raises(ValueError, match="dimension_mismatch|requested_dimension_mismatch"):
            await engine.query_embeddings(["wrong batch shape"])
        assert engine._query_cache == {}
        assert await engine.generate_and_store("wrong-document", "wrong document shape") is False
        await engine.close()

    asyncio.run(run())

    conn = sqlite3.connect(engine.db_path)
    try:
        assert conn.execute(
            "SELECT 1 FROM embeddings WHERE bucket_id = ?", ("wrong-document",),
        ).fetchone() is None
    finally:
        conn.close()


def test_get_embedding_rejects_legacy_row_with_other_requested_dimension(tmp_path):
    engine = _engine(tmp_path, dimensions=4)
    conn = sqlite3.connect(engine.db_path)
    try:
        conn.execute(
            "INSERT INTO embeddings (bucket_id, embedding, model, dimension, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-short-vector", json.dumps([0.1, 0.2]), engine.model, 2, "synthetic"),
        )
        conn.commit()
    finally:
        conn.close()

    assert asyncio.run(engine.get_embedding("legacy-short-vector")) is None


def test_gateway_runtime_overlay_updates_and_clears_embedding_dimension():
    config = {
        "embedding": {
            "enabled": True,
            "model": "test-embedding-model",
            "base_url": "https://embedding.invalid/v1",
            "api_key": "local-test-key",
            "dimensions": 2,
        },
        "gateway": {},
    }
    service = GatewayService.__new__(GatewayService)
    service.config = config
    service.embedding_cfg = config["embedding"]
    service.embedding_engine = SimpleNamespace(
        model="test-embedding-model", dimension=2,
        base_url="https://embedding.invalid/v1", api_key="local-test-key", enabled=True,
    )
    service._runtime_credential_seen = {
        "OMBRE_EMBEDDING_API_KEY": "local-test-key",
        "OMBRE_RERANKER_API_KEY": None,
    }
    service._runtime_credential_managed = set()
    service._runtime_overlay_status = {}
    service._runtime_env_credentials = lambda: None

    service._apply_runtime_overlay({"embedding": {"dimensions": 5}}, env_changed=False)
    assert service.config["embedding"]["dimensions"] == 5
    assert service.embedding_engine.dimension == 5

    service._apply_runtime_overlay({"embedding": {"dimensions": None}}, env_changed=False)
    assert service.config["embedding"]["dimensions"] is None
    assert service.embedding_engine.dimension is None


def test_single_query_freezes_dimension_snapshot_before_key_and_provider_work(tmp_path, monkeypatch):
    engine = _engine(tmp_path, dimensions=2)
    payloads = []

    def handler(request):
        payloads.append(_body(request))
        return _embedding_response(request)

    _attach_httpx(engine, handler, monkeypatch)
    make_key = engine._query_cache_key

    def mutate_dimension_after_key(text, *, snapshot=None):
        # Deterministically place the hot update after query_embedding has
        # captured its request identity but before its provider coroutine runs.
        key = make_key(text, snapshot=snapshot)
        engine.dimension = 3
        return key

    monkeypatch.setattr(engine, "_query_cache_key", mutate_dimension_after_key)

    async def run():
        old_request = await engine.query_embedding("single query across hot reload")
        new_request = await engine.query_embedding("single query across hot reload")
        await engine.close()
        return old_request, new_request

    old_request, new_request = asyncio.run(run())

    assert len(old_request) == 2
    assert len(new_request) == 3
    assert [payload["dimensions"] for payload in payloads] == [2, 3]


def test_gateway_semantic_cache_keeps_old_model_dimension_flight_in_its_own_key():
    started = asyncio.Event()
    release_old = asyncio.Event()
    calls = []

    class SyntheticEmbedding:
        model = "model-old"
        dimension = 2
        base_url = "https://embedding.invalid/v1"
        db_path = ""

        def _query_config_snapshot(self):
            return {
                "model": self.model,
                "dimension": self.dimension,
                "base_url": self.base_url,
                "max_chars": 6000,
                "query_instruction": "synthetic query instruction",
                "document_preparation": "synthetic preparation identity",
            }

        async def search_similar(self, _query, *, top_k, config_snapshot):
            identity = (config_snapshot["model"], config_snapshot["dimension"])
            calls.append(identity)
            if identity == ("model-old", 2):
                started.set()
                await asyncio.wait_for(release_old.wait(), timeout=2)
            return [(f"{identity[0]}-{identity[1]}", 0.9)][:top_k]

    embedding = SyntheticEmbedding()
    service = GatewayService.__new__(GatewayService)
    service.embedding_engine = embedding
    service._embedding_store_cache_stamp = lambda: "unchanged-synthetic-store"
    service.semantic_candidate_top_k = 5
    service.embedding_query_timeout_seconds = 5.0
    service.semantic_query_cache_ttl_seconds = 300.0
    service.semantic_query_cache_max_entries = 8
    service._semantic_query_cache = {}
    service._semantic_query_inflight = {}

    async def run():
        old_debug, new_debug, cached_debug = {}, {}, {}
        old_task = asyncio.create_task(service._semantic_search_cached(
            "same semantic query", cache_debug=old_debug,
        ))
        await asyncio.wait_for(started.wait(), timeout=1)

        embedding.model = "model-new"
        embedding.dimension = 3
        new_result = await service._semantic_search_cached(
            "same semantic query", cache_debug=new_debug,
        )
        release_old.set()
        old_result = await old_task
        cached_new_result = await service._semantic_search_cached(
            "same semantic query", cache_debug=cached_debug,
        )
        return old_result, new_result, cached_new_result, old_debug, new_debug, cached_debug

    old_result, new_result, cached_new_result, old_debug, new_debug, cached_debug = asyncio.run(run())

    assert calls == [("model-old", 2), ("model-new", 3)]
    assert old_result == [("model-old-2", 0.9)]
    assert new_result == cached_new_result == [("model-new-3", 0.9)]
    assert old_debug["status"] == "miss"
    assert new_debug["status"] == "miss"
    assert cached_debug["status"] == "hit"


def test_worker_snapshot_none_uses_successful_unit_shapes_not_mutable_engine_health(
    tmp_path, monkeypatch,
):
    engine = _engine(tmp_path)
    mode = {"mixed": False}
    payloads = []

    def handler(request):
        payload = _body(request)
        payloads.append(payload)
        input_value = payload["input"]
        text = input_value if isinstance(input_value, str) else input_value[0]
        if not mode["mixed"]:
            # Simulate a runtime hot update and an unrelated last-call health
            # value while this request, whose snapshot.dimension is None, runs.
            engine.dimension = 3
            engine._runtime["last_vector_dimension"] = 4
            dimension = 2
        else:
            dimension = 3 if "mixed-second" in text else 2
        return httpx.Response(200, json={
            "data": [{"index": 0, "embedding": [0.25] * dimension}]
        })

    _attach_httpx(engine, handler, monkeypatch)
    real_generate_and_store = engine.generate_and_store
    observed_metadata = []

    async def capture_result_metadata(bucket_id, content, **kwargs):
        result = await real_generate_and_store(bucket_id, content, **kwargs)
        if kwargs.get("result_metadata") is not None:
            observed_metadata.append(dict(kwargs["result_metadata"]))
            # Another completion may overwrite shared health immediately after
            # this unit succeeds, before the worker finalizes Memory details.
            engine._runtime["last_vector_dimension"] = 4
        return result

    monkeypatch.setattr(engine, "generate_and_store", capture_result_metadata)
    worker = MemoryProjectionWorker(
        config={}, authority=None, bucket_manager=None, embedding_engine=engine,
    )

    async def run():
        single_projection = await worker._project_embedding_units(
            {"id": "memory-dimension-snapshot"},
            "memory-dimension-snapshot", 1, "a" * 64,
            source_units=[{
                "source_unit_id": "unit-snapshot",
                "source_kind": "committed_body",
                "text": "synthetic single projection",
            }],
        )
        assert engine.dimension == 3
        assert engine._runtime["last_vector_dimension"] == 4
        assert single_projection["dimension"] == 2
        assert single_projection["metadata_complete"] is True
        assert observed_metadata == [{"dimension": 2}]

        mode["mixed"] = True
        engine.dimension = None
        with pytest.raises(RuntimeError, match="embedding_projection_dimension_changed"):
            await worker._project_embedding_units(
                {"id": "memory-dimension-mixed"},
                "memory-dimension-mixed", 1, "b" * 64,
                source_units=[
                    {"source_unit_id": "unit-mixed-first", "source_kind": "message", "text": "mixed-first"},
                    {"source_unit_id": "unit-mixed-second", "source_kind": "message", "text": "mixed-second"},
                ],
            )
        await engine.close()

    asyncio.run(run())

    assert all("dimensions" not in payload for payload in payloads)
    assert observed_metadata == [{"dimension": 2}, {"dimension": 2}, {"dimension": 3}]
