"""CHAT-RUNTIME-R5 retrieval/index contracts.

The tests use local SQLite and Gateway method doubles only.  They do not call
embedding/reranker providers, start the server, or write production state.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import GatewayService
from memory_moments import MemoryMomentStore


def _config(tmp_path):
    return {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
    }


def _bucket(bucket_id: str, content: str = "与主人约定周末一起散步。"):
    return {
        "id": bucket_id,
        "metadata": {"name": f"bucket-{bucket_id}"},
        "content": content,
    }


def test_memory_moment_bulk_upsert_commits_moments_and_signature_atomically(
    tmp_path, monkeypatch
):
    store = MemoryMomentStore(_config(tmp_path))
    first = _bucket("b1", "旧内容")
    second = _bucket("b2", "新内容")
    store.bulk_upsert([first], source_signature="sig-old")

    original_replace = store._replace_bucket

    def fail_on_second_bucket(conn, bucket_id, moments, bucket_title):
        if bucket_id == "b2":
            raise RuntimeError("synthetic rebuild failure")
        return original_replace(conn, bucket_id, moments, bucket_title)

    monkeypatch.setattr(store, "_replace_bucket", fail_on_second_bucket)
    with pytest.raises(RuntimeError, match="synthetic"):
        store.bulk_upsert(
            [_bucket("b1", "新版本"), second],
            source_signature="sig-new",
        )

    # The failed transaction must leave both the old derived rows and the old
    # source identity intact; a partially rebuilt index is not reusable.
    assert store.source_signature() == "sig-old"
    assert [item["text"] for item in store.list_for_bucket("b1")] == ["旧内容"]
    assert store.list_for_bucket("b2") == []


def test_memory_moment_signature_invalidates_only_on_real_delete_or_upsert(
    tmp_path,
):
    store = MemoryMomentStore(_config(tmp_path))
    original = _bucket("b1", "旧内容")
    changed = _bucket("b1", "更新后的内容")

    store.bulk_upsert([original], source_signature="sig-1")
    no_op_delete = store.delete_bucket("missing-bucket")
    assert no_op_delete == {"moments": 0, "edges": 0, "aliases": 0}
    assert store.source_signature() == "sig-1"

    store.upsert_bucket(changed)
    assert store.source_signature() == ""

    store.bulk_upsert([changed], source_signature="sig-2")
    deleted = store.delete_bucket("b1")
    assert deleted["moments"] > 0
    assert store.source_signature() == ""


def _graph_service(store):
    service = GatewayService.__new__(GatewayService)
    service.memory_moment_store = store
    service.memory_edge_store = SimpleNamespace(list_edges=lambda: [])
    service.self_anchor_entry_bucket_id = ""
    service._moment_graph_refresh_lock = threading.RLock()
    service._moment_graph_cache_signature = ""
    service._moment_graph_cache_value = None
    service._moment_graph_cache_bucket_list_id = 0
    service._moment_graph_cache_edge_stamp = (0, 0)
    service._moment_graph_cache_store_stamp = (0, 0)
    service._prune_self_anchor_moment_index = lambda _buckets: None
    service._memory_edge_store_stamp = lambda: (0, 0)
    service._memory_moment_store_stamp = lambda: (0, 0)
    service._moment_graph_signature = lambda _buckets, _edges: "sig-v1"
    service._recallable_moments = lambda moments: moments
    service._moments_by_bucket = lambda moments: {
        bucket_id: [item for item in moments if item.get("bucket_id") == bucket_id]
        for bucket_id in {item.get("bucket_id") for item in moments}
    }
    service._bucket_edges_as_moment_edges = lambda *_args: []
    return service


def test_new_gateway_with_same_persistent_signature_skips_bulk_upsert(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    bucket = _bucket("b1")
    first_store = MemoryMomentStore(config)
    first_store.bulk_upsert([bucket], source_signature="sig-v1")

    second_store = MemoryMomentStore(config)
    calls = []
    original_bulk_upsert = second_store.bulk_upsert

    def tracked_bulk_upsert(*args, **kwargs):
        calls.append((args, kwargs))
        return original_bulk_upsert(*args, **kwargs)

    monkeypatch.setattr(second_store, "bulk_upsert", tracked_bulk_upsert)
    moments, grouped, edges = _graph_service(second_store)._refresh_moment_graph(
        [bucket]
    )

    assert calls == []
    assert len(moments) == 1
    assert list(grouped) == ["b1"]
    assert edges == []


class _CountingReranker:
    enabled = True
    candidate_limit = 20
    score_weight = 0.65

    def __init__(self):
        self.calls = 0

    async def rerank(self, query, documents, *, top_n):
        self.calls += 1
        return [SimpleNamespace(index=0, score=0.91)]


def _selection_service(reranker):
    service = GatewayService.__new__(GatewayService)
    service.inject_max_cards = 2
    service.moment_search_limit = 4
    service.word_map_hint_moment_boost = 1.0
    service.reranker_engine = reranker
    service._query_planner_debug_base = lambda query: {
        "query": query, "timing_ms": {}, "candidate_stages": [],
    }
    service._auto_query_too_vague = lambda _query: False
    service._has_named_exact_anchor_candidate = lambda *_args: False
    service._query_anchor_plan = lambda _query: {}
    service._query_has_relevance_facet = lambda _query: False
    service._is_dynamic_candidate = lambda _bucket: True
    service._is_relevance_suppressed = lambda *_args: False
    service._is_relevance_candidate_bucket = lambda *_args: False
    service._is_semantic_candidate_bucket = lambda _bucket: False
    service._entity_priority_recall_search_query = lambda query: query
    service._with_explicit_source_record_buckets = lambda _q, selected, _all: selected
    service._session_hard_exclude_bucket_ids = lambda _session: set()
    service._query_explicitly_requests_caution_memory = lambda _query: False
    service._apply_relevance_to_moment_candidates = lambda _q, candidates: candidates
    service._moment_with_bucket_recall_signal = lambda moment, _signal: dict(moment)
    service._is_source_record_bucket = lambda _bucket: False
    service._anchor_plan_direct_rejection = lambda *_args: None
    service._admit_moment_for_recall = lambda *_args, **_kwargs: True
    service._promote_reliable_moment_hits_to_direct_seed = (
        lambda _q, selected, _candidates: selected
    )
    service._recall_rank = lambda _q, _moment: (0,)
    return service


def _selection_bucket_and_moment():
    bucket = _bucket("b1")
    moment = {
        "moment_id": "b1:m1",
        "bucket_id": "b1",
        "section": "body",
        "text": "与主人约定周末一起散步。",
        "score": 0.8,
        "metadata": {},
    }
    return bucket, moment


def test_no_admitted_bucket_skips_moment_search_and_provider_rerank():
    reranker = _CountingReranker()
    service = _selection_service(reranker)
    bucket, moment = _selection_bucket_and_moment()

    async def no_admitted_bucket(*_args, **_kwargs):
        return [], [], {"timing_ms": {}, "candidate_stages": []}

    service._select_dynamic_buckets = no_admitted_bucket
    service._with_explicit_source_record_buckets = lambda _q, _selected, _all: []

    selected, candidates, suppressed, suppressed_buckets, debug = asyncio.run(
        service._select_dynamic_moments(
            "周末散步",
            "session-1",
            [bucket],
            {"b1": [moment]},
            all_moments=[moment],
            search_query="周末散步",
            include_query_planner_debug=True,
        )
    )

    assert selected == []
    assert candidates == []
    assert suppressed == []
    assert suppressed_buckets == []
    assert reranker.calls == 0
    stages = {item["stage"]: item for item in debug["candidate_stages"]}
    assert debug["moment_skip_reason"] == "no_admitted_buckets"
    assert stages["moment.search_0"]["skipped"] is True
    assert stages["moment_rerank"]["skipped"] is True
    assert stages["moment.admit_candidates"]["skipped"] is True


def test_admitted_bucket_keeps_moment_search_and_provider_rerank_path():
    reranker = _CountingReranker()
    service = _selection_service(reranker)
    bucket, moment = _selection_bucket_and_moment()
    service.memory_moment_store = SimpleNamespace(
        search_moment_items=lambda *_args, **_kwargs: [dict(moment)],
    )

    async def one_admitted_bucket(*_args, **_kwargs):
        return [bucket], [], {"timing_ms": {}, "candidate_stages": []}

    service._select_dynamic_buckets = one_admitted_bucket

    selected, candidates, _suppressed, _suppressed_buckets, debug = asyncio.run(
        service._select_dynamic_moments(
            "周末散步",
            "session-1",
            [bucket],
            {"b1": [moment]},
            all_moments=[moment],
            search_query="周末散步",
            include_query_planner_debug=True,
        )
    )

    assert reranker.calls == 1
    assert len(candidates) == 1
    assert len(selected) == 1
    stages = {item["stage"]: item for item in debug["candidate_stages"]}
    assert stages["moment_rerank"]["skipped"] is False
    assert stages["moment_rerank"]["provider_input_count"] == 1
    assert stages["moment.final_output"]["output_count"] == 1
