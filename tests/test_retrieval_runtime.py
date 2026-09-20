import asyncio
import time
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime, timezone

import httpx

from embedding_engine import EmbeddingEngine
from reranker_engine import RerankerEngine
from gateway import GatewayService
from dream_engine import DreamEngine


def test_embedding_runtime_debug_records_success_without_body_or_credentials(tmp_path, monkeypatch):
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "embedding": {
            "enabled": True,
            "api_key": "owner-secret",
            "base_url": "https://embedding.example/v1",
            "model": "test-embedding",
        },
    })

    class ConnectClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection unavailable")

        async def aclose(self):
            return None

    monkeypatch.setattr(engine, "_new_client", lambda: ConnectClient())
    monkeypatch.setattr(
        engine,
        "_request_embedding_sync",
        lambda *args, **kwargs: (200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}),
    )
    vector = asyncio.run(engine._generate_embedding("private query", kind="query"))

    assert vector == [0.1, 0.2, 0.3]
    debug = engine.runtime_debug()
    assert debug["last_status"] == "ok"
    assert debug["last_operation"] == "query"
    assert debug["last_vector_dimension"] == 3
    assert debug["configured"] is True
    assert "owner-secret" not in str(debug)
    assert "private query" not in str(debug)


def test_query_embedding_cache_singleflights_exact_recall_and_dream_query(tmp_path, monkeypatch):
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "embedding": {
            "enabled": True,
            "api_key": "owner-secret",
            "base_url": "https://embedding.example/v1",
            "model": "test-embedding",
            "query_cache_ttl_seconds": 300,
        },
    })
    calls = []

    async def fake_generate(text, *, kind="document"):
        calls.append((text, kind))
        await asyncio.sleep(0.01)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(engine, "_generate_embedding", fake_generate)

    async def run():
        first, second = await asyncio.gather(
            engine.query_embedding("晏晏是谁"),
            engine.query_embedding("晏晏是谁"),
        )
        third = await engine.query_embedding("晏晏是谁")
        return first, second, third

    first, second, third = asyncio.run(run())

    assert first == second == third == [0.1, 0.2, 0.3]
    assert calls == [("晏晏是谁", "query")]
    assert engine.runtime_debug()["last_query_cache_status"] == "hit"
    assert "晏晏是谁" not in str(engine._query_cache)


def test_dream_surface_reuses_embedding_engine_query_cache_contract():
    calls = []

    class Engine:
        enabled = True

        async def query_embedding(self, query):
            calls.append(query)
            return [0.4, 0.5]

    dream = DreamEngine.__new__(DreamEngine)
    vector = asyncio.run(dream._query_embedding("共同经历", Engine()))

    assert vector == [0.4, 0.5]
    assert calls == ["共同经历"]


def test_dream_fast_route_keeps_local_cues_but_skips_remote_query_embedding(monkeypatch):
    dream = DreamEngine.__new__(DreamEngine)
    dream.enabled = True
    dream.surface_enabled = True
    dream.retain_after_surface = True
    dream.tz = timezone.utc
    dream.min_surface_age_hours = 0
    dream.attempt_threshold = 2.0
    dream.surface_threshold = 2.0
    dream.alpha_subordinate = 0.25
    dream.spontaneous_surface_prob = 0.0
    dream.max_surface_attempts = 4
    record = SimpleNamespace(
        surfaced=False,
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"surface_attempts": 0},
        dream_id="dream-1",
    )
    dream.list_records = lambda: [record]
    dream._eligible_context = lambda *_args: True

    async def forbidden_embedding(*_args, **_kwargs):
        raise AssertionError("Fast/tone routes must not block on dream embedding")

    async def local_cue_only(*_args, **_kwargs):
        return 0.0

    dream._query_embedding = forbidden_embedding
    dream._cue_score = local_cue_only

    result = asyncio.run(dream.surface_with_status(
        query="普通聊天",
        embedding_engine=SimpleNamespace(enabled=True),
        allow_semantic=False,
    ))

    assert result == {"status": "skipped", "reason": "no_resonance"}


def test_embedding_runtime_debug_records_failure_type(tmp_path, monkeypatch):
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "embedding": {"enabled": True, "api_key": "k", "base_url": "https://example/v1"},
    })

    class ConnectClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection unavailable")

        async def aclose(self):
            return None

    monkeypatch.setattr(engine, "_new_client", lambda: ConnectClient())
    monkeypatch.setattr(
        engine,
        "_request_embedding_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("provider timed out")
        ),
    )
    assert asyncio.run(engine._generate_embedding("query", kind="query")) == []
    debug = engine.runtime_debug()
    assert debug["last_status"] == "error"
    assert debug["last_error_type"] == "TimeoutError"
    assert debug["last_result_count"] is None


def test_embedding_runtime_debug_records_cancellation(tmp_path, monkeypatch):
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "embedding": {"enabled": True, "api_key": "k", "base_url": "https://example/v1"},
    })

    class ConnectClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection unavailable")

        async def aclose(self):
            return None

    monkeypatch.setattr(engine, "_new_client", lambda: ConnectClient())
    monkeypatch.setattr(
        engine,
        "_request_embedding_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            asyncio.CancelledError()
        ),
    )
    try:
        asyncio.run(engine._generate_embedding("query", kind="query"))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation must be propagated")
    debug = engine.runtime_debug()
    assert debug["last_status"] == "cancelled"
    assert debug["last_error_type"] == "CancelledError"


def test_reranker_runtime_debug_records_success_without_documents(monkeypatch):
    engine = RerankerEngine({
        "embedding": {"api_key": "embedding-secret", "base_url": "https://example/v1"},
        "reranker": {"enabled": True, "model": "test-reranker"},
    })

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.91}]}

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.aclose()
            return False

        async def post(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse()

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", lambda **kwargs: FakeClient())
    result = asyncio.run(engine.rerank("query", ["document"], top_n=1))

    assert result[0].index == 0
    debug = engine.runtime_debug()
    assert debug["last_status"] == "ok"
    assert debug["last_http_status"] == 200
    assert debug["last_result_count"] == 1
    assert "embedding-secret" not in str(debug)


def test_retrieval_clients_are_reused_and_closed(monkeypatch, tmp_path):
    created = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            self.calls = 0
            self.closed = False
            created.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse()

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr("embedding_engine.httpx.AsyncClient", FakeClient)
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path / "embedding"),
        "embedding": {"enabled": True, "api_key": "secret", "base_url": "https://example/v1"},
    })

    async def scenario():
        assert await engine._generate_embedding("one", kind="query") == [0.1, 0.2, 0.3]
        assert await engine._generate_embedding("two", kind="query") == [0.1, 0.2, 0.3]
        await engine.close()

    asyncio.run(scenario())
    assert len(created) == 1
    assert created[0].calls == 2
    assert created[0].closed is True


def test_embedding_fallback_only_retries_connection_setup_failures(monkeypatch, tmp_path):
    calls = {"fallback": 0}

    class FakeClient:
        is_closed = False

        def __init__(self, error):
            self.error = error

        async def post(self, *args, **kwargs):
            raise self.error

        async def aclose(self):
            return None

    async def run_case(error):
        engine = EmbeddingEngine({
            "buckets_dir": str(tmp_path / type(error).__name__),
            "embedding": {"enabled": True, "api_key": "secret", "base_url": "https://example/v1"},
        })
        client = FakeClient(error)
        engine._new_client = lambda: client

        def fallback(*args, **kwargs):
            calls["fallback"] += 1
            return 200, {"data": [{"embedding": [0.4, 0.5]}]}

        engine._request_embedding_sync = fallback
        vector = await engine._generate_embedding("query", kind="query")
        debug = engine.runtime_debug()
        await engine.close()
        return vector, debug

    vector, connect_debug = asyncio.run(run_case(httpx.ConnectError("connect")))
    assert vector == [0.4, 0.5]
    assert connect_debug["last_error_category"] == "connect"
    assert connect_debug["connection_fallback_count"] == 1

    vector, pool_debug = asyncio.run(run_case(httpx.PoolTimeout("pool unavailable")))
    assert vector == [0.4, 0.5]
    assert pool_debug["last_error_category"] == "pool"
    assert pool_debug["connection_fallback_count"] == 1

    for error, category in (
        (httpx.ReadTimeout("read stalled"), "read"),
        (httpx.WriteTimeout("write stalled"), "write"),
        (httpx.RemoteProtocolError("response interrupted"), "remote_protocol"),
    ):
        vector, debug = asyncio.run(run_case(error))
        assert vector == []
        assert debug["last_error_category"] == category
        assert debug["connection_fallback_count"] == 0
    assert calls["fallback"] == 2


def test_embedding_does_not_fallback_after_total_budget_is_exhausted(monkeypatch, tmp_path):
    fallback_calls = []

    class FakeClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("connect")

        async def aclose(self):
            return None

    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "embedding": {"enabled": True, "api_key": "secret", "base_url": "https://example/v1"},
    })
    engine._new_client = lambda: FakeClient()
    engine._request_embedding_sync = lambda *args, **kwargs: fallback_calls.append(True)

    async def scenario():
        try:
            await engine._request_embedding(
                "https://example/v1/embeddings", "secret", "model", "query",
                deadline=time.monotonic() - 1,
            )
        except httpx.ConnectError:
            return
        raise AssertionError("an exhausted budget must not start urllib fallback")

    asyncio.run(scenario())
    assert fallback_calls == []
    assert engine.runtime_debug()["connection_fallback_count"] == 0


def test_reranker_keeps_per_call_transport_after_live_provider_downgrade(monkeypatch):
    created = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.8}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            self.calls = 0
            self.closed = False
            created.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.aclose()
            return False

        async def post(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse()

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", FakeClient)
    engine = RerankerEngine({
        "reranker": {
            "enabled": True,
            "api_key": "secret",
            "base_url": "https://example/v1",
            "model": "reranker",
        },
    })

    async def scenario():
        assert await engine.rerank("one", ["document"], top_n=1)
        assert await engine.rerank("two", ["document"], top_n=1)

    asyncio.run(scenario())
    assert len(created) == 2
    assert [client.calls for client in created] == [1, 1]
    assert all(client.closed for client in created)


def test_gateway_reloads_runtime_overlay_without_rebuilding_brain(tmp_path):
    overlay = Path(tmp_path) / "config.runtime.yaml"
    overlay.write_text(
        "gateway:\n"
        "  current_inner_state_interval_rounds: 1\n"
        "  relationship_weather_interval_rounds: 0\n"
        "  embedding_query_timeout_seconds: 8\n"
        "  graph_bucket_rerank_enabled: false\n"
        "embedding:\n"
        "  model: gitee/qwen3-embedding-8b\n"
        "  base_url: https://api.pie-xian.com/v1\n"
        "  enabled: true\n"
        "reranker:\n"
        "  model: qwen3-reranker-8b\n"
        "  base_url: https://api.futureppo.top/v1\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    service = GatewayService.__new__(GatewayService)
    service.config = {
        "_runtime_config_path": str(overlay),
        "gateway": {},
        "embedding": {"model": "old-embedding", "base_url": "https://old.example/v1", "enabled": True},
        "reranker": {"model": "old-reranker", "base_url": "https://old.example/v1", "enabled": True},
    }
    service.gateway_cfg = service.config["gateway"]
    service.embedding_cfg = service.config["embedding"]
    service.embedding_engine = SimpleNamespace(
        model="old-embedding", base_url="https://old.example/v1", api_key="embedding-key", enabled=True,
    )
    service.reranker_engine = SimpleNamespace(
        model="old-reranker", base_url="https://old.example/v1", api_key="reranker-key", enabled=True,
        timeout=12.0, candidate_limit=20, score_weight=0.65,
    )
    service.current_inner_state_interval_rounds = 15
    service.relationship_weather_interval_rounds = 0
    service.embedding_query_timeout_seconds = 3.0
    service.graph_bucket_rerank_enabled = True
    service._runtime_overlay_path = str(overlay)
    service._runtime_overlay_signature = None
    service._runtime_overlay_lock = __import__("threading").RLock()
    service._runtime_overlay_status = {}

    service._maybe_reload_runtime_overlay(force=True)

    assert service.embedding_engine.model == "gitee/qwen3-embedding-8b"
    assert service.embedding_engine.base_url == "https://api.pie-xian.com/v1"
    assert service.reranker_engine.base_url == "https://api.futureppo.top/v1"
    assert service.current_inner_state_interval_rounds == 1
    assert service.embedding_query_timeout_seconds == 8.0
    assert service.graph_bucket_rerank_enabled is False
    assert service._runtime_overlay_status["last_reload_status"] == "reloaded"
    assert service._runtime_overlay_status["sha256"]


def test_graph_bucket_rerank_defaults_off_to_avoid_duplicate_remote_roundtrip():
    service = GatewayService.__new__(GatewayService)
    service.gateway_cfg = {}
    service.graph_bucket_rerank_enabled = GatewayService._bool_config_value(
        service.gateway_cfg.get("graph_bucket_rerank_enabled"), False)
    assert service.graph_bucket_rerank_enabled is False


def test_effective_retrieval_config_has_stable_hash_and_no_credentials():
    service = GatewayService.__new__(GatewayService)
    service.embedding_engine = SimpleNamespace(
        enabled=True,
        model="gitee/qwen3-embedding-8b",
        base_url="https://embedding.example/v1",
        max_chars=6000,
        api_key="embedding-secret",
    )
    service.reranker_engine = SimpleNamespace(
        enabled=True,
        model="qwen3-reranker-8b",
        base_url="https://reranker.example/v1",
        timeout=12.0,
        candidate_limit=9,
        score_weight=0.65,
        api_key="reranker-secret",
    )
    service.diffusion_options = SimpleNamespace(
        enabled=True,
        max_hops=2,
        top_k=4,
        min_activation=0.18,
        chain_walk_enabled=True,
        chain_max_hops=3,
        chain_min_strength=0.2,
        chain_min_confidence=0.72,
        chain_min_relation_priority=60,
        chain_max_frontier=10,
    )
    service.retrieval_mode = "graph"
    service.dynamic_top_k = 10
    service.semantic_candidate_top_k = 50
    service.moment_search_limit = 50
    service.graph_bucket_rerank_enabled = False
    service.recalled_budget = 400
    service.related_memory_budget = 110
    service.embedding_query_timeout_seconds = 8.0
    service.query_planner_enabled = True
    service.query_planner_model = "deepseek-v4-flash"
    service.query_planner_max_queries = 3
    service.query_planner_max_tokens = 360
    service.query_planner_supplemental_semantic = False
    service.memory_detail_recall_enabled = True
    service.memory_detail_recall_max_ids = 2
    service.memory_detail_recall_budget = 200
    service.current_inner_state_interval_rounds = 1
    service.operit_context_rewrite_enabled = True
    service._runtime_overlay_status = {
        "present": True,
        "sha256": "overlay-sha",
    }

    first = service._effective_config_metadata()
    second = service._effective_config_metadata()
    assert first == second
    assert first["revision"] == 1
    assert len(first["sha256"]) == 64
    assert "embedding-secret" not in str(first)
    assert "reranker-secret" not in str(first)

    service.semantic_candidate_top_k = 24
    changed = service._effective_config_metadata()
    assert changed["revision"] == 2
    assert changed["sha256"] != first["sha256"]


def test_candidate_stage_telemetry_keeps_provider_window_explicit():
    stages = []
    GatewayService._record_candidate_stage(
        stages,
        "moment_rerank",
        50,
        50,
        limit=9,
        provider_input_count=9,
        provider_output_count=9,
        duration_ms=123,
        reason="final moment rerank",
    )

    assert stages == [{
        "stage": "moment_rerank",
        "input_count": 50,
        "output_count": 50,
        "skipped": False,
        "limit": 9,
        "provider_input_count": 9,
        "provider_output_count": 9,
        "reason": "final moment rerank",
        "duration_ms": 123,
    }]


def test_memory_detail_recall_debug_reports_trigger_and_retry_timing():
    service = GatewayService.__new__(GatewayService)
    service.memory_detail_recall_enabled = True
    service.memory_detail_recall_max_ids = 2
    service.memory_detail_recall_budget = 200
    async def detail_context(_ids):
        return "detail body", []

    async def retry_response(_payload):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "retried"}}]},
        )

    service._build_memory_detail_recall_context = detail_context
    service._insert_memory_detail_context = lambda messages, detail: [
        *messages, {"role": "system", "content": detail}
    ]
    service._forward_upstream = retry_response

    response, debug = asyncio.run(service._maybe_retry_with_memory_detail(
        forward_payload={"messages": [{"role": "user", "content": "query"}], "stream": False},
        upstream_response=httpx.Response(
            200,
            json={"choices": [{"message": {
                "role": "assistant",
                "content": '[memory_detail ids="bucket-1"]need detail',
            }}]},
        ),
        injection_debug={"injected_bucket_ids": ["bucket-1"]},
    ))

    assert response.status_code == 200
    assert debug["trigger_count"] == 1
    assert debug["retry_request_count"] == 1
    assert debug["retried"] is True
    assert debug["retry_status_code"] == 200
    assert debug["elapsed_ms"] >= debug["retry_elapsed_ms"] >= 0
