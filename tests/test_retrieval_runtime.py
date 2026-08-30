import asyncio
from types import SimpleNamespace
from pathlib import Path

from embedding_engine import EmbeddingEngine
from reranker_engine import RerankerEngine
from gateway import GatewayService


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

    monkeypatch.setattr(
        engine,
        "_request_embedding_sync",
        lambda endpoint, api_key, model, input_value: (200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}),
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


def test_embedding_runtime_debug_records_failure_type(tmp_path, monkeypatch):
    engine = EmbeddingEngine({
        "buckets_dir": str(tmp_path),
        "embedding": {"enabled": True, "api_key": "k", "base_url": "https://example/v1"},
    })

    monkeypatch.setattr(
        engine,
        "_request_embedding_sync",
        lambda endpoint, api_key, model, input_value: (_ for _ in ()).throw(
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

    monkeypatch.setattr(
        engine,
        "_request_embedding_sync",
        lambda endpoint, api_key, model, input_value: (_ for _ in ()).throw(
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
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", lambda **kwargs: FakeClient())
    result = asyncio.run(engine.rerank("query", ["document"], top_n=1))

    assert result[0].index == 0
    debug = engine.runtime_debug()
    assert debug["last_status"] == "ok"
    assert debug["last_http_status"] == 200
    assert debug["last_result_count"] == 1
    assert "embedding-secret" not in str(debug)


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
