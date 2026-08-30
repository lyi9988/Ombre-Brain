import asyncio

from embedding_engine import EmbeddingEngine
from reranker_engine import RerankerEngine


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

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("embedding_engine.httpx.AsyncClient", lambda **kwargs: FakeClient())
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

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise TimeoutError("provider timed out")

    monkeypatch.setattr("embedding_engine.httpx.AsyncClient", lambda **kwargs: FailingClient())
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

    class CancelledClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise asyncio.CancelledError()

    monkeypatch.setattr("embedding_engine.httpx.AsyncClient", lambda **kwargs: CancelledClient())
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
