"""Focused contracts for RECALL-R1 routing, authority reads, and exact caches.

The suite is synthetic: it uses temporary SQLite files and local doubles only.
It must never call a model, embedding provider, reranker provider, or live API.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from gateway import GatewayService
from memory_authority import MemoryAuthorityStore
from memory_authority_view import MemoryAuthorityRecallView


def test_authority_view_exposes_only_trusted_aliases_and_refreshes_wal(tmp_path):
    state_dir = tmp_path / "state"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir),
        "memory_authority": {"enabled": True},
    })

    authority.upsert_alias(
        entity_id="person:yanyan",
        alias="晏晏",
        trust="auto",
        source_refs=["memory-1"],
    )
    assert view.match_aliases("晏晏是谁") == []

    authority.upsert_alias(
        entity_id="person:yanyan",
        alias="晏晏",
        trust="owner",
        source_refs=["owner-confirmation-1"],
    )
    matches = view.match_aliases("你还记得晏晏吗")

    assert len(matches) == 1
    assert matches[0]["entity_id"] == "person:yanyan"
    assert matches[0]["trust"] == "owner"
    assert matches[0]["revision"] == 2
    assert matches[0]["memory_ids"] == []
    assert matches[0]["bucket_ids"] == []


def _sentinel_service(alias_matches=None):
    service = GatewayService.__new__(GatewayService)
    service.memory_sentinel_enabled = True
    service.memory_authority_view = SimpleNamespace(
        match_aliases=lambda _query: list(alias_matches or [])
    )
    service._memory_sentinel_searchable_residue_terms = lambda _query: []
    service._query_looks_emotional_reason_lookup = lambda _query: False
    return service


def test_owner_confirmed_alias_is_a_fast_hard_evidence_route():
    service = _sentinel_service([{
        "alias_id": "alias-1",
        "entity_id": "person:yanyan",
        "alias": "晏晏",
        "trust": "owner",
        "revision": 3,
    }])

    routed = asyncio.run(service._route_memory_sentinel(
        "晏晏是谁",
        "jiajia-main",
        [],
    ))

    assert routed["route"] == "fast"
    assert routed["route_reason_codes"] == ["FAST_OWNER_ALIAS"]
    assert routed["hard_bypass_reason"] == "owner_alias"
    assert routed["anchors"] == ["晏晏", "person:yanyan"]


def test_owner_alias_memory_link_becomes_first_class_hard_evidence():
    service = GatewayService.__new__(GatewayService)
    service.inject_max_cards = 2
    service.first_card_min_score = 0.55
    selected, forced = service._merge_owner_alias_memory_items(
        [{"bucket": {"id": "memory-other"}, "score": 0.9}],
        [
            {"id": "memory-1", "metadata": {"name": "晏晏"}, "content": "共同经历"},
            {"id": "memory-other", "metadata": {}, "content": "other"},
        ],
        ["memory-1"],
    )

    assert [item["bucket"]["id"] for item in selected] == [
        "memory-1",
        "memory-other",
    ]
    assert forced[0]["owner_alias_match"] is True
    assert forced[0]["admission_reason"] == "owner_alias"
    assert forced[0]["hard_evidence_labels"] == ["owner_alias"]


def test_only_explicit_memory_source_refs_can_force_alias_recall():
    debug = {
        "owner_alias_matches": [{
            "source_refs": [
                "raw_event:evt-1",
                "candidate:candidate-1",
                "memory:memory-1",
                "memory:memory-1",
            ]
        }]
    }
    assert GatewayService._owner_alias_memory_ids(debug) == ["memory-1"]


def test_explicit_or_causal_recall_routes_deep_but_specific_topic_routes_fast():
    service = _sentinel_service()
    service._memory_sentinel_hard_bypass_reason = (
        lambda query, *_args, **_kwargs:
        "explicit_recall_marker" if "记得" in query else "searchable_residue"
    )

    explicit = asyncio.run(service._route_memory_sentinel(
        "你还记得我们以前说过的那件事吗", "jiajia-main", []
    ))
    specific = asyncio.run(service._route_memory_sentinel(
        "我们做的 Prompt Composer", "jiajia-main", []
    ))

    service._query_looks_emotional_reason_lookup = lambda _query: True
    causal = asyncio.run(service._route_memory_sentinel(
        "我之前为什么会因为那件事难过", "jiajia-main", []
    ))

    assert explicit["route"] == "deep"
    assert explicit["route_reason_codes"] == ["DEEP_EXPLICIT_RECALL"]
    assert specific["route"] == "fast"
    assert specific["route_reason_codes"] == ["FAST_LOCAL_TOPIC"]
    assert causal["route"] == "deep"
    assert causal["route_reason_codes"] == ["DEEP_EMOTIONAL_REASON"]


def test_fast_route_never_runs_semantic_planner_or_reranker():
    assert GatewayService._recall_route_execution_options("fast") == {
        "allow_semantic": False,
        "allow_query_planner": False,
        "allow_rerank": False,
    }
    assert GatewayService._recall_route_execution_options("deep") == {
        "allow_semantic": True,
        "allow_query_planner": True,
        "allow_rerank": True,
    }
    assert GatewayService._recall_route_execution_options("skip") == {
        "allow_semantic": False,
        "allow_query_planner": False,
        "allow_rerank": False,
    }


def _moment_graph_service(*, authority_enabled: bool, rebuild_enabled: bool):
    calls = []
    store = SimpleNamespace(
        bulk_upsert=lambda *args, **kwargs: calls.append((args, kwargs)),
        source_signature=lambda: "legacy-signature",
        list_all=lambda: [],
        list_edges=lambda: [],
    )
    service = GatewayService.__new__(GatewayService)
    service.memory_moment_store = store
    service.memory_edge_store = SimpleNamespace(list_edges=lambda: [])
    service.memory_authority_enabled = authority_enabled
    service.moment_request_rebuild_enabled = rebuild_enabled
    service.self_anchor_entry_bucket_id = ""
    service._moment_graph_refresh_lock = threading.RLock()
    service._moment_graph_cache_signature = ""
    service._moment_graph_cache_value = None
    service._moment_graph_cache_bucket_list_id = 0
    service._moment_graph_cache_edge_stamp = (0, 0)
    service._moment_graph_cache_store_stamp = (0, 0)
    service._prune_self_anchor_moment_index = lambda _buckets: None
    service._memory_edge_store_stamp = lambda: (0, 0)
    service._memory_moment_store_stamp = lambda: (1, 1)
    service._moment_graph_signature = lambda _buckets, _edges: "fresh-signature"
    service._recallable_moments = lambda moments: moments
    service._moments_by_bucket = lambda _moments: {}
    service._bucket_edges_as_moment_edges = lambda *_args: []
    return service, calls


def test_authority_mode_never_rebuilds_moment_index_in_request_path():
    service, calls = _moment_graph_service(
        authority_enabled=True,
        rebuild_enabled=False,
    )
    service._refresh_moment_graph([{"id": "memory-1", "metadata": {}, "content": "body"}])
    assert calls == []


def test_legacy_mode_preserves_request_time_moment_rebuild():
    service, calls = _moment_graph_service(
        authority_enabled=False,
        rebuild_enabled=True,
    )
    service._refresh_moment_graph([{"id": "memory-1", "metadata": {}, "content": "body"}])
    assert len(calls) == 1


def test_domain_sentinel_local_prepare_never_calls_remote_client():
    service = GatewayService.__new__(GatewayService)
    service._domain_sentinel_rule_plan = lambda _query: {
        "source": "rules",
        "called": False,
        "message_type": "conversation",
        "should_recall": True,
        "recall_route": "search",
        "reason": "local_rule",
        "errors": [],
    }
    service._domain_sentinel_query_explicitly_needs_memory = lambda _query: False
    service._domain_sentinel_should_skip_recall = lambda _debug, _query: False
    service.http_client = SimpleNamespace(
        post=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("remote Domain Sentinel must not run in prepare")
        )
    )

    result = asyncio.run(service._route_domain_sentinel(
        "普通聊天问题",
        allow_remote=False,
    ))

    assert result["called"] is False
    assert result["remote_skip_reason"] == "prepare_uses_local_rules"


class _CountingEmbedding:
    model = "synthetic-embedding"
    base_url = "https://invalid.example/v1"
    db_path = ""

    def __init__(self):
        self.calls = 0

    async def search_similar(self, _query, *, top_k):
        self.calls += 1
        return [("memory-1", 0.91)][:top_k]


def test_semantic_exact_cache_reuses_query_result_without_provider_call():
    embedding = _CountingEmbedding()
    service = GatewayService.__new__(GatewayService)
    service.embedding_engine = embedding
    service.semantic_candidate_top_k = 24
    service.embedding_query_timeout_seconds = 8.0
    service.semantic_query_cache_ttl_seconds = 300.0
    service.semantic_query_cache_max_entries = 8
    service._semantic_query_cache = {}
    service._semantic_query_inflight = {}

    first_debug = {}
    second_debug = {}

    async def run():
        first = await service._semantic_search_cached("晏晏", cache_debug=first_debug)
        second = await service._semantic_search_cached("晏晏", cache_debug=second_debug)
        return first, second

    first, second = asyncio.run(run())

    assert first == second == [("memory-1", 0.91)]
    assert embedding.calls == 1
    assert first_debug["status"] == "miss"
    assert second_debug["status"] == "hit"


class _CountingReranker:
    model = "synthetic-reranker"
    base_url = "https://invalid.example/v1"

    def __init__(self):
        self.calls = 0

    async def rerank(self, _query, _documents, *, top_n):
        self.calls += 1
        return [SimpleNamespace(index=0, score=0.88)][:top_n]


def test_rerank_exact_cache_reuses_document_identity_without_provider_call():
    reranker = _CountingReranker()
    service = GatewayService.__new__(GatewayService)
    service.reranker_engine = reranker
    service.rerank_cache_ttl_seconds = 300.0
    service.rerank_cache_max_entries = 8
    service._rerank_cache = {}
    service._rerank_inflight = {}

    first_debug = {}
    second_debug = {}

    async def run():
        first = await service._rerank_cached(
            namespace="moment",
            query="晏晏",
            documents=["共同经历"],
            top_n=1,
            cache_debug=first_debug,
        )
        second = await service._rerank_cached(
            namespace="moment",
            query="晏晏",
            documents=["共同经历"],
            top_n=1,
            cache_debug=second_debug,
        )
        return first, second

    first, second = asyncio.run(run())

    assert [(row.index, row.score) for row in first] == [(0, 0.88)]
    assert [(row.index, row.score) for row in second] == [(0, 0.88)]
    assert reranker.calls == 1
    assert first_debug["status"] == "miss"
    assert second_debug["status"] == "hit"


def test_recall_why_uses_stable_owner_safe_reason_codes():
    service = GatewayService.__new__(GatewayService)
    item = {
        "bucket": {"id": "memory-1", "metadata": {}},
        "admission_reason": "semantic_only",
        "semantic_score": 0.82,
    }

    debug = service._recall_why_debug(item, status="rejected", stage="bucket")

    assert debug["admission"]["code"] == "REJECT_SEMANTIC_ONLY"
    assert debug["status"] == "rejected"
    assert "body" not in debug


def test_memory_recall_projection_records_authority_revision_and_body_hash():
    service = GatewayService.__new__(GatewayService)
    service.memory_authority_view = SimpleNamespace(
        watermark=lambda: {
            "available": True,
            "memory_count": 12,
            "memory_revision_watermark": 18,
            "alias_revision": 4,
        },
        memory_revision_map=lambda ids: {
            memory_id: 3 for memory_id in ids if memory_id == "memory-1"
        },
    )
    service._extract_bucket_ids_from_context = (
        lambda _body: ["memory-2"]
    )

    projection = service._build_memory_recall_projection(
        recalled_memory="direct evidence",
        targeted_memory_detail="",
        related_memory="related evidence [bucket_id:memory-2]",
        recalled_moments=[{
            "bucket_id": "memory-1",
            "moment_id": "memory-1:m1",
            "metadata": {"ring_id": "ring-1"},
        }],
        targeted_memory_detail_debug={"accepted_ids": []},
        memory_sentinel_debug={
            "route": "fast",
            "route_reason_codes": ["FAST_OWNER_ALIAS"],
        },
    )

    assert projection["source_id"] == "ombre.memory_recall"
    assert projection["route"] == "fast"
    assert projection["reason_codes"] == ["FAST_OWNER_ALIAS"]
    assert projection["selected_memory_ids"] == ["memory-1"]
    assert projection["selected_bucket_ids"] == ["memory-1", "memory-2"]
    assert projection["selected_memory_revisions"] == {"memory-1": 3}
    assert projection["selected_ring_ids"] == ["ring-1"]
    assert projection["status"] == "ready"
    assert projection["token_estimate"] > 0
    assert len(projection["body_sha256"]) == 64


def test_alias_resolves_memory_id_to_distinct_bucket_projection_id(tmp_path):
    state_dir = tmp_path / "state"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    body = "主人和晏晏的共同经历。"
    import hashlib
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="memory-canonical-1",
        bucket_id="bucket-projection-9",
        expected_revision=0,
        body_sha256=digest,
        snapshot_path="revisions/memory-canonical-1/1.md",
        metadata={"memory_type": "key_event"},
        source_refs=["evt-1"],
        decision_source="owner",
        idempotency_key="commit-distinct-id",
        actor="owner",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
    authority.finalize_memory_commit(prepared["operation_id"])
    authority.upsert_alias(
        entity_id="person:yanyan",
        alias="晏晏",
        trust="owner",
        source_refs=["memory:memory-canonical-1"],
    )
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir),
        "memory_authority": {"enabled": True},
    })

    match = view.match_aliases("晏晏是谁")[0]

    assert match["memory_ids"] == ["memory-canonical-1"]
    assert match["bucket_ids"] == ["bucket-projection-9"]
    assert view.memory_revision_map(["bucket-projection-9"]) == {
        "memory-canonical-1": 1
    }
