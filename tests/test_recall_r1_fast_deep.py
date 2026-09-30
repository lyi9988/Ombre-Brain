"""Focused contracts for RECALL-R1 routing, authority reads, and exact caches.

The suite is synthetic: it uses temporary SQLite files and local doubles only.
It must never call a model, embedding provider, reranker provider, or live API.
"""

from __future__ import annotations

import asyncio
import threading
import json
import hashlib
import pytest
from types import SimpleNamespace

import gateway as gateway_module
from gateway import GatewayService
from reranker_engine import RerankerEngine
from memory_authority import MemoryAuthorityStore
from memory_authority_view import MemoryAuthorityRecallView


def test_short_cjk_identity_title_requires_rare_corrobated_authority_evidence():
    service = GatewayService.__new__(GatewayService)
    service._dynamic_anchor_term_is_category = lambda _term: False
    plan = {
        "short_cjk_identity_axis_terms": ["晏晏"],
        "term_stats": [{
            "term": "晏晏", "document_frequency": 12, "document_count": 744,
        }],
    }
    bucket = {
        "id": "memory-bucket", "metadata": {"type": "dynamic", "name": "晏晏的故事"},
        "content": "晏晏与主人有共同经历。",
    }
    assert service._short_cjk_identity_title_terms(bucket, plan) == ["晏晏"]
    assert GatewayService._hard_bucket_evidence_labels(["short_cjk_identity_title"]) == [
        "short_cjk_identity_title"
    ]
    assert service._short_cjk_identity_title_terms(
        {**bucket, "content": "另一段没有目标称呼的故事。"}, plan
    ) == []
    assert service._short_cjk_identity_title_terms(bucket, {
        **plan, "term_stats": [{"term": "晏晏", "document_frequency": 80, "document_count": 744}],
    }) == []
    assert service._short_cjk_identity_title_terms(bucket, {
        **plan, "short_cjk_identity_axis_terms": [],
    }) == []


def test_short_cjk_identity_question_activates_axis_without_owner_name_marker():
    service = GatewayService.__new__(GatewayService)
    service.bucket_mgr = SimpleNamespace(lexical_term_specificity_stats=lambda _terms, _buckets: {
        "晏晏": {"document_frequency": 12, "document_count": 744},
    })
    service._recall_query_plan = lambda _query: SimpleNamespace(activated_axis_terms=("晏晏",))
    service._dynamic_anchor_query_terms = lambda _query: ["晏晏"]
    service._dynamic_anchor_term_is_category = lambda _term: False
    service._query_has_identity_name_intent = lambda _query: False
    plan = service._dynamic_anchor_plan("你还记得晏晏是谁吗？", [], [])
    assert plan["short_cjk_identity_axis_terms"] == ["晏晏"]


def test_verified_local_short_name_skips_remote_embedding():
    class AfterSemanticBranch(Exception):
        pass

    service = GatewayService.__new__(GatewayService)
    service.inject_max_cards = 2
    service.dynamic_top_k = 10
    service.semantic_candidate_top_k = 24
    service.memory_authority_view = None
    service._recall_query_plan = lambda _query: SimpleNamespace(
        skip_reason="", skip_long_term_recall=False,
    )
    service._query_has_relevance_facet = lambda _query: False
    service._is_dynamic_candidate = lambda _bucket: True
    service._is_relevance_suppressed = lambda _query, _bucket: False
    service._is_semantic_candidate_bucket = lambda _bucket: True
    service._is_identity_name_candidate_bucket = lambda _query, _bucket: False
    service._retrieval_alias_hits = lambda _query, _ids: []
    service._dynamic_anchor_plan = lambda *_args: {"short_cjk_identity_axis_terms": ["晏晏"]}
    service._normalized_recall_query = lambda _query: "晏晏"
    service._get_keyword_candidates = lambda _query, _buckets: {"bucket-1": 0.9}
    service._short_cjk_identity_title_terms = lambda _bucket, _plan: ["晏晏"]
    service._add_timing_ms = lambda *_args: None
    service._query_looks_emotional_reason_lookup = lambda _query: False

    async def forbidden_semantic(*_args, **_kwargs):
        raise AssertionError("remote embedding should not be called")

    service._get_semantic_candidates = forbidden_semantic
    service._get_exact_anchor_candidates = lambda *_args: (_ for _ in ()).throw(
        AfterSemanticBranch()
    )
    stages = []
    with pytest.raises(AfterSemanticBranch):
        asyncio.run(service._dynamic_bucket_candidate_items(
            "你还记得晏晏是谁吗？", "jiajia-main", [{"id": "bucket-1"}],
            candidate_stages=stages,
        ))
    embedding_stage = next(row for row in stages if row["stage"] == "candidate.embedding_candidates")
    assert embedding_stage["skipped"] is True
    assert embedding_stage["verified_local_anchor_count"] == 1

    calls = []
    service._short_cjk_identity_title_terms = lambda _bucket, _plan: []

    async def fallback_semantic(*_args, **_kwargs):
        calls.append(True)
        return {}

    service._get_semantic_candidates = fallback_semantic
    with pytest.raises(AfterSemanticBranch):
        asyncio.run(service._dynamic_bucket_candidate_items(
            "你还记得晏晏是谁吗？", "jiajia-main", [{"id": "bucket-1"}],
        ))
    assert calls == [True]


def test_authority_auto_bucket_gate_fails_closed_and_rejections_are_body_free():
    service = GatewayService.__new__(GatewayService)
    service.memory_authority_view = SimpleNamespace(
        enabled=True,
        auto_recallable_bucket_ids=lambda: frozenset({"enabled-bucket"}),
    )
    assert service._authority_auto_recall_allowed({"id": "enabled-bucket"})
    assert not service._authority_auto_recall_allowed({"id": "manual-bucket"})
    rejected = {"bucket": {"id": "manual-bucket", "content": "private body"}}
    assert service._admit_bucket_for_recall("test", rejected) is False
    assert rejected["admission_reason"] == "authority_auto_recall_denied"
    evidence = service._rejected_bucket_evidence([rejected])
    assert evidence["reason_counts"] == {"authority_auto_recall_denied": 1}
    assert "manual-bucket" not in json.dumps(evidence)
    assert "private body" not in json.dumps(evidence)


def test_authority_view_auto_bucket_ids_exclude_manual_only_and_refresh_wal(tmp_path):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})

    def commit(bucket_id, recall_policy):
        digest = hashlib.sha256(bucket_id.encode()).hexdigest()
        prepared = authority.prepare_memory_commit(
            memory_id=f"memory:{bucket_id}", bucket_id=bucket_id,
            expected_revision=0, body_sha256=digest,
            snapshot_path=f"revisions/{bucket_id}/1.md", metadata={},
            source_refs=["owner-test"], decision_source="owner",
            idempotency_key=f"auto-gate:{bucket_id}", actor="owner",
            recall_policy=recall_policy,
        )
        authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
        authority.finalize_memory_commit(prepared["operation_id"])

    commit("enabled-bucket", "enabled")
    commit("manual-bucket", "manual_only")
    view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path), "memory_authority": {"enabled": True},
    })
    assert view.auto_recallable_bucket_ids() == frozenset({"enabled-bucket"})
    commit("new-enabled-bucket", "enabled")
    assert view.auto_recallable_bucket_ids() == frozenset({
        "enabled-bucket", "new-enabled-bucket",
    })


def test_authority_view_index_metadata_uses_current_enabled_revision_only(tmp_path):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path)})

    def commit(memory_id, bucket_id, revision, body, policy="enabled"):
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        prepared = authority.prepare_memory_commit(
            memory_id=memory_id,
            bucket_id=bucket_id,
            expected_revision=revision - 1,
            body_sha256=digest,
            snapshot_path=f"revisions/{memory_id}/{revision}.md",
            metadata={},
            source_refs=[f"test:{memory_id}:{revision}"],
            decision_source="test",
            idempotency_key=f"view-index:{memory_id}:{revision}",
            actor="owner-test",
            recall_policy=policy,
        )
        authority.record_body_written(
            prepared["operation_id"], observed_body_sha256=digest,
        )
        authority.finalize_memory_commit(prepared["operation_id"])
        return digest

    commit("memory-enabled", "bucket-enabled", 1, "旧的已提交内容。")
    current_sha = commit(
        "memory-enabled", "bucket-enabled", 2, "当前已提交内容。",
    )
    commit(
        "memory-manual", "bucket-manual", 1, "仅手动可读内容。",
        policy="manual_only",
    )
    view = MemoryAuthorityRecallView({
        "state_dir": str(tmp_path),
        "memory_authority": {"enabled": True},
    })

    assert view.memory_index_metadata_map([
        "bucket-enabled", "bucket-manual", "memory-enabled", "missing",
    ]) == {
        "bucket-enabled": {
            "memory_id": "memory-enabled",
            "revision": 2,
            "body_sha256": current_sha,
        },
    }


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
        source_refs=["owner-confirmation-1", "memory:memory-1"],
    )
    body = "主人与晏晏有一段共同经历。"
    import hashlib
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="memory-1", bucket_id="bucket-1", expected_revision=0,
        body_sha256=digest, snapshot_path="revisions/memory-1/1.md",
        metadata={}, source_refs=["evt-1"], decision_source="owner",
        idempotency_key="alias-memory-1", actor="owner",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
    authority.finalize_memory_commit(prepared["operation_id"])
    matches = view.match_aliases("你还记得晏晏吗")

    assert len(matches) == 1
    assert matches[0]["entity_id"] == "person:yanyan"
    assert matches[0]["trust"] == "owner"
    assert matches[0]["revision"] == 2
    assert matches[0]["memory_ids"] == ["memory-1"]
    assert matches[0]["bucket_ids"] == ["bucket-1"]


def test_trusted_alias_without_active_memory_does_not_force_empty_fast_route(tmp_path):
    state_dir = tmp_path / "state"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    authority.upsert_alias(
        entity_id="person:yanyan", alias="晏晏", trust="owner",
        source_refs=["memory:missing-memory"],
    )
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir), "memory_authority": {"enabled": True},
    })
    assert view.trusted_aliases()
    assert view.match_aliases("晏晏是谁") == []


def test_manual_only_memory_alias_cannot_force_automatic_fast_recall(tmp_path):
    state_dir = tmp_path / "state"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    body = "仅手动可读的私人记忆。"
    import hashlib
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    prepared = authority.prepare_memory_commit(
        memory_id="memory-private", bucket_id="bucket-private", expected_revision=0,
        body_sha256=digest, snapshot_path="revisions/memory-private/1.md",
        metadata={}, source_refs=["owner-1"], decision_source="owner",
        idempotency_key="manual-only-memory", actor="owner",
        recall_policy="manual_only",
    )
    authority.record_body_written(prepared["operation_id"], observed_body_sha256=digest)
    authority.finalize_memory_commit(prepared["operation_id"])
    authority.upsert_alias(
        entity_id="person:private", alias="私密称呼", trust="owner",
        source_refs=["memory:memory-private"],
    )
    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir), "memory_authority": {"enabled": True},
    })

    assert view.match_aliases("私密称呼是谁") == []


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


def _configured_rerank_cache_service():
    service = GatewayService.__new__(GatewayService)
    service.config = {"reranker": {
        "enabled": True, "model": "same-synthetic-model",
        "base_url": "https://same.invalid/v1", "api_key": "synthetic-key-old",
    }}
    service.reranker_engine = RerankerEngine(service.config)
    service.rerank_cache_ttl_seconds = 300.0
    service.rerank_cache_max_entries = 8
    service._rerank_cache = {}
    service._rerank_inflight = {}
    return service


def _overlay_rerank_cache_service():
    service = _configured_rerank_cache_service()
    service.config = {"gateway": {}, "embedding": {}, "reranker": {
        "enabled": True, "model": "synthetic-model-old",
        "base_url": "https://same.invalid/v1",
    }}
    service.gateway_cfg = service.config["gateway"]
    service.embedding_engine = SimpleNamespace(
        model="synthetic-embedding", base_url="https://embedding.invalid/v1",
        api_key="synthetic-embedding-key", enabled=True,
    )
    service.reranker_engine = RerankerEngine({"reranker": {
        **service.config["reranker"], "api_key": "synthetic-key-old",
    }})
    env_key = {"value": "synthetic-key-old"}
    service._runtime_env_credentials = lambda: {
        "OMBRE_RERANKER_API_KEY": env_key["value"],
    }
    return service, env_key


@pytest.mark.parametrize("switch", ["model", "env_key"])
def test_rerank_runtime_overlay_switch_isolates_cache_and_inflight_and_reapply_is_idempotent(
    monkeypatch, switch,
):
    service, env_key = _overlay_rerank_cache_service()
    started = {phase: asyncio.Event() for phase in ("old", "new")}
    release = {phase: asyncio.Event() for phase in started}
    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, score):
            self.score = score

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": self.score}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, *, headers, json):
            phase = ("old" if json["model"] == "synthetic-model-old" else "new") if switch == "model" else (
                "old" if headers["Authorization"] == "Bearer synthetic-key-old" else "new"
            )
            calls.append((json["query"], phase))
            if json["query"] == "inflight":
                started[phase].set()
                await release[phase].wait()
            return FakeResponse(0.81 if phase == "old" else 0.92)

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", FakeClient)
    old_cache, old_hit, new_cache, new_hit, old_inflight, new_inflight = ({}, {}, {}, {}, {}, {})

    async def invoke(query, debug):
        return await service._rerank_cached(
            namespace="bucket", query=query, documents=["synthetic document"],
            top_n=1, cache_debug=debug,
        )

    async def run():
        assert [row.score for row in await invoke("cache", old_cache)] == [0.81]
        assert [row.score for row in await invoke("cache", old_hit)] == [0.81]
        old_task = asyncio.create_task(invoke("inflight", old_inflight))
        await asyncio.wait_for(started["old"].wait(), 1)
        if switch == "model":
            overlay, env_changed = {"reranker": {"model": "synthetic-model-new"}}, False
        else:
            env_key["value"] = "synthetic-key-new"
            overlay, env_changed = {}, True
        service._apply_runtime_overlay(overlay, env_changed=env_changed)
        revision = service._rerank_config_revision
        service._apply_runtime_overlay(overlay, env_changed=env_changed)
        assert service._rerank_config_revision == revision == 1
        assert [row.score for row in await invoke("cache", new_cache)] == [0.92]
        new_task = asyncio.create_task(invoke("inflight", new_inflight))
        await asyncio.wait_for(started["new"].wait(), 1)
        release["new"].set()
        assert [row.score for row in await new_task] == [0.92]
        release["old"].set()
        assert [row.score for row in await old_task] == [0.81]
        assert [row.score for row in await invoke("cache", new_hit)] == [0.92]

    asyncio.run(run())
    assert calls == [("cache", "old"), ("inflight", "old"),
                     ("cache", "new"), ("inflight", "new")]
    assert old_cache["key"] != new_cache["key"]
    assert old_inflight["key"] != new_inflight["key"]
    assert [item["status"] for item in (old_cache, old_hit, new_cache, new_hit)] == [
        "miss", "hit", "miss", "hit",
    ]
    assert old_inflight["status"] == new_inflight["status"] == "miss"
    assert len(service._rerank_cache) == 4 and service._rerank_inflight == {}
    assert all("synthetic-key-" not in str(item) for item in (
        old_cache, old_hit, new_cache, new_hit, old_inflight, new_inflight,
    ))


def test_rerank_scheduled_before_overlay_change_never_calls_stale_provider(monkeypatch):
    service, _env_key = _overlay_rerank_cache_service()
    child_scheduled, child_release = asyncio.Event(), asyncio.Event()
    provider_calls = []
    real_create_task = asyncio.create_task

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            provider_calls.append(True)
            raise AssertionError("stale configuration must stop before provider post")

    def gated_create_task(coro):
        async def gated():
            child_scheduled.set()
            await child_release.wait()
            return await coro

        return real_create_task(gated())

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(gateway_module, "asyncio", SimpleNamespace(
        create_task=gated_create_task, shield=asyncio.shield,
        CancelledError=asyncio.CancelledError, current_task=asyncio.current_task,
        gather=asyncio.gather,
    ))
    debug = {}

    async def run():
        owner = real_create_task(service._rerank_cached(
            namespace="bucket", query="scheduled", documents=["synthetic document"],
            top_n=1, cache_debug=debug,
        ))
        await asyncio.wait_for(child_scheduled.wait(), 1)
        service._apply_runtime_overlay({"reranker": {"model": "synthetic-model-new"}}, env_changed=False)
        child_release.set()
        return await owner

    assert asyncio.run(run()) == []
    assert provider_calls == []
    assert debug["provider_status"] == "configuration_changed"
    assert debug["provider_failed"] is True
    assert debug["provider_requests"] == 0
    assert debug["provider_error_type"] == "ConfigurationChanged"
    assert service._rerank_cache == {} and service._rerank_inflight == {}


def test_rerank_formal_credential_update_invalidates_old_success_cache(monkeypatch):
    for name in ("OMBRE_RERANKER_API_KEY", "OMBRE_RERANKER_BASE_URL", "OMBRE_RERANKER_MODEL"):
        monkeypatch.setenv(name, "synthetic-before-test")
    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, score):
            self.score = score

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": self.score}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, *, headers, **_kwargs):
            authorization = headers["Authorization"]
            calls.append(authorization)
            return FakeResponse(0.81 if authorization == "Bearer synthetic-key-old" else 0.92)

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", FakeClient)
    service = _configured_rerank_cache_service()
    debug = [{}, {}, {}, {}]

    async def invoke(index):
        return await service._rerank_cached(
            namespace="bucket", query="same synthetic query", documents=["same synthetic document"],
            top_n=1, cache_debug=debug[index],
        )

    async def run():
        old = await invoke(0)
        assert await invoke(1) == old
        assert service._apply_reranker_config({"api_key": "synthetic-key-new"}) == ["reranker.api_key"]
        new = await invoke(2)
        assert await invoke(3) == new
        return old, new

    old, new = asyncio.run(run())
    assert [row.score for row in old] == [0.81]
    assert [row.score for row in new] == [0.92]
    assert calls == ["Bearer synthetic-key-old", "Bearer synthetic-key-new"]
    assert [item["status"] for item in debug] == ["miss", "hit", "miss", "hit"]
    assert debug[0]["key"] != debug[2]["key"]
    assert debug[2]["provider_requests"] == 1 and debug[3]["provider_requests"] == 0
    assert service._rerank_config_revision == 1
    assert all("synthetic-key-" not in str(item) for item in debug)


def test_rerank_formal_credential_update_does_not_join_old_inflight_request(monkeypatch):
    for name in ("OMBRE_RERANKER_API_KEY", "OMBRE_RERANKER_BASE_URL", "OMBRE_RERANKER_MODEL"):
        monkeypatch.setenv(name, "synthetic-before-test")
    started = {name: asyncio.Event() for name in ("old", "new")}
    release = {name: asyncio.Event() for name in started}
    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, score):
            self.score = score

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": self.score}]}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, *, headers, **_kwargs):
            identity = "old" if headers["Authorization"] == "Bearer synthetic-key-old" else "new"
            calls.append(identity)
            started[identity].set()
            await release[identity].wait()
            return FakeResponse(0.81 if identity == "old" else 0.92)

    monkeypatch.setattr("reranker_engine.httpx.AsyncClient", FakeClient)
    service = _configured_rerank_cache_service()
    old_debug, new_debug, cached_debug = {}, {}, {}

    async def invoke(debug):
        return await service._rerank_cached(
            namespace="bucket", query="same synthetic query", documents=["same synthetic document"],
            top_n=1, cache_debug=debug,
        )

    async def run():
        old_engine = service.reranker_engine
        old_task = asyncio.create_task(invoke(old_debug))
        await asyncio.wait_for(started["old"].wait(), 1)
        service._apply_reranker_config({"api_key": "synthetic-key-new"})
        assert service.reranker_engine is not old_engine
        new_task = asyncio.create_task(invoke(new_debug))
        await asyncio.wait_for(started["new"].wait(), 1)
        release["new"].set()
        new = await new_task
        release["old"].set()
        old = await old_task
        cached = await invoke(cached_debug)
        return old, new, cached

    old, new, cached = asyncio.run(run())
    assert calls == ["old", "new"]
    assert [row.score for row in old] == [0.81]
    assert [row.score for row in new] == [0.92]
    assert [row.score for row in cached] == [0.92]
    assert old_debug["key"] != new_debug["key"]
    assert old_debug["status"] == new_debug["status"] == "miss"
    assert old_debug["provider_requests"] == new_debug["provider_requests"] == 1
    assert cached_debug["status"] == "hit" and cached_debug["provider_requests"] == 0
    assert len(service._rerank_cache) == 2 and service._rerank_inflight == {}


@pytest.mark.parametrize(("initial_status", "initial_rows"), [
    ("error", []),
    ("invalid_response", []),
    ("partial", [SimpleNamespace(index=0, score=0.92)]),
])
def test_rerank_failed_or_partial_outcome_is_not_cached_then_success_is_cached(
    initial_status, initial_rows,
):
    class SequenceReranker:
        model = "synthetic-reranker-v4"
        base_url = "https://reranker.invalid/v1"

        def __init__(self):
            self.calls = 0

        async def rerank_with_diagnostics(self, _query, _documents, *, top_n):
            self.calls += 1
            if self.calls == 1:
                return initial_rows[:top_n], {
                    "last_status": initial_status, "last_http_status": 403 if initial_status == "error" else 200,
                    "last_error_type": "HTTPStatusError" if initial_status == "error" else "InvalidRerankResponse",
                    "last_latency_ms": 3,
                }
            return [SimpleNamespace(index=0, score=0.91)][:top_n], {
                "last_status": "ok", "last_http_status": 200, "last_error_type": "",
                "last_latency_ms": 4,
            }

    reranker = SequenceReranker()
    service = GatewayService.__new__(GatewayService)
    service.reranker_engine = reranker
    service.rerank_cache_ttl_seconds = 300.0
    service.rerank_cache_max_entries = 8
    service._rerank_cache = {}
    service._rerank_inflight = {}
    debug = [{}, {}, {}]

    async def run():
        return [await service._rerank_cached(
            namespace="bucket", query="synthetic query", documents=["synthetic document"],
            top_n=1, cache_debug=entry,
        ) for entry in debug]

    first, second, third = asyncio.run(run())

    assert [(row.index, row.score) for row in first] == [
        (row.index, row.score) for row in initial_rows
    ]
    assert [(row.index, row.score) for row in second] == [(0, 0.91)]
    assert [(row.index, row.score) for row in third] == [(0, 0.91)]
    assert reranker.calls == 2
    assert debug[0]["status"] == "miss" and debug[0]["provider_failed"] is True
    assert debug[0]["provider_status"] == initial_status
    assert debug[0]["provider_requests"] == 1
    assert debug[1]["status"] == "miss" and debug[1]["provider_status"] == "ok"
    assert debug[2]["status"] == "hit" and debug[2]["provider_status"] == "cached_success"
    assert debug[2]["provider_failed"] is False and debug[2]["provider_requests"] == 0
    assert debug[2].get("provider_http_status") is None


def test_rerank_distinct_concurrent_keys_keep_their_request_local_outcomes():
    started = {name: asyncio.Event() for name in ("failed", "success")}
    release = {name: asyncio.Event() for name in started}

    class ConcurrentReranker:
        model = "synthetic-reranker-v4"
        base_url = "https://reranker.invalid/v1"

        async def rerank_with_diagnostics(self, query, _documents, *, top_n):
            started[query].set()
            await release[query].wait()
            if query == "failed":
                return [], {"last_status": "error", "last_http_status": 403,
                            "last_error_type": "HTTPStatusError", "last_latency_ms": 8}
            return [SimpleNamespace(index=0, score=0.93)][:top_n], {
                "last_status": "ok", "last_http_status": 200,
                "last_error_type": "", "last_latency_ms": 2,
            }

    service = GatewayService.__new__(GatewayService)
    service.reranker_engine = ConcurrentReranker()
    service.rerank_cache_ttl_seconds = 300.0
    service.rerank_cache_max_entries = 8
    service._rerank_cache = {}
    service._rerank_inflight = {}
    failed_debug, success_debug = {}, {}

    async def run():
        failed = asyncio.create_task(service._rerank_cached(
            namespace="bucket", query="failed", documents=["one"], top_n=1,
            cache_debug=failed_debug,
        ))
        success = asyncio.create_task(service._rerank_cached(
            namespace="bucket", query="success", documents=["one"], top_n=1,
            cache_debug=success_debug,
        ))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 1)
        release["success"].set()
        success_rows = await success
        release["failed"].set()
        failed_rows = await failed
        return failed_rows, success_rows

    failed_rows, success_rows = asyncio.run(run())
    assert failed_rows == [] and [(row.index, row.score) for row in success_rows] == [(0, 0.93)]
    assert failed_debug["provider_status"] == "error"
    assert failed_debug["provider_http_status"] == 403
    assert failed_debug["provider_failed"] is True
    assert success_debug["provider_status"] == "ok"
    assert success_debug["provider_http_status"] == 200
    assert success_debug["provider_failed"] is False
    assert len(service._rerank_cache) == 1


def test_rerank_singleflight_waiter_shares_failed_outcome_without_second_request():
    started, release = asyncio.Event(), asyncio.Event()

    class BlockedReranker:
        model = "synthetic-reranker-v4"
        base_url = "https://reranker.invalid/v1"

        def __init__(self):
            self.calls = 0

        async def rerank_with_diagnostics(self, _query, _documents, *, top_n):
            self.calls += 1
            started.set()
            await release.wait()
            return [], {"last_status": "error", "last_http_status": 403,
                        "last_error_type": "HTTPStatusError", "last_latency_ms": 5}

    reranker = BlockedReranker()
    service = GatewayService.__new__(GatewayService)
    service.reranker_engine = reranker
    service.rerank_cache_ttl_seconds = 300.0
    service.rerank_cache_max_entries = 8
    service._rerank_cache = {}
    service._rerank_inflight = {}
    owner_debug, waiter_debug, retry_debug = {}, {}, {}

    async def invoke(debug):
        return await service._rerank_cached(
            namespace="bucket", query="same", documents=["one"], top_n=1,
            cache_debug=debug,
        )

    async def run():
        owner = asyncio.create_task(invoke(owner_debug))
        await asyncio.wait_for(started.wait(), 1)
        waiter = asyncio.create_task(invoke(waiter_debug))
        await asyncio.sleep(0)
        release.set()
        assert await owner == await waiter == []
        assert await invoke(retry_debug) == []

    asyncio.run(run())
    assert reranker.calls == 2
    assert service._rerank_cache == {} and service._rerank_inflight == {}
    assert owner_debug["provider_status"] == waiter_debug["provider_status"] == "error"
    assert owner_debug["provider_http_status"] == waiter_debug["provider_http_status"] == 403
    assert owner_debug["provider_requests"] == 1
    assert waiter_debug["status"] == "singleflight"
    assert waiter_debug["shared_provider_request"] is True
    assert waiter_debug["provider_requests"] == 0
    assert retry_debug["status"] == "miss" and retry_debug["provider_requests"] == 1


def test_rerank_bucket_candidate_diagnostics_preserve_request_local_failure():
    service = GatewayService.__new__(GatewayService)
    service.reranker_engine = SimpleNamespace(enabled=True, candidate_limit=20, score_weight=0.65)
    service._bucket_rerank_candidate_priority = lambda _query, item: (-item["score"],)
    service._bucket_rerank_document = lambda bucket: bucket["content"]
    calls = []

    async def failed_cached(*, namespace, query, documents, top_n, cache_debug):
        calls.append((namespace, query, len(documents), top_n))
        cache_debug.update(status="miss", provider_status="error", provider_failed=True,
                           provider_http_status=403, provider_error_type="HTTPStatusError",
                           provider_latency_ms=7, provider_requests=1)
        return []

    service._rerank_cached = failed_cached
    item = {"bucket": {"id": "synthetic-memory", "content": "synthetic document"}, "score": 0.79}
    diagnostics = {}
    results = asyncio.run(service._rerank_scored_bucket_candidates(
        "synthetic query", [item], diagnostics=diagnostics,
    ))

    assert results == [item]
    assert calls == [("bucket", "synthetic query", 1, 1)]
    assert diagnostics["provider_input_count"] == 1
    assert diagnostics["provider_output_count"] == 0
    assert diagnostics["provider_status"] == "error"
    assert diagnostics["provider_failed"] is True
    assert diagnostics["provider_http_status"] == 403
    assert diagnostics["provider_error_type"] == "HTTPStatusError"
    assert diagnostics["provider_requests"] == 1
    assert diagnostics["cache"]["status"] == "miss"


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


def test_gateway_serves_owner_safe_memory_projection_on_existing_bearer_transport():
    class Request:
        method = "GET"
        headers = {"Authorization": "Bearer synthetic"}

        def __init__(self, resource, *, memory_id="", query=None):
            self.path_params = {"resource": resource, "memory_id": memory_id}
            self.query_params = dict(query or {})

    class Buckets:
        async def get(self, bucket_id):
            return {"id": bucket_id, "content": "owner body"}

    view = SimpleNamespace(
        available=lambda: True,
        overview=lambda: {"available": True, "outbox": {"projected": 1}},
        list_memories=lambda **_kwargs: [{"memory_id": "memory-1", "bucket_id": "bucket-1"}],
        memory_detail=lambda _memory_id: {
            "memory": {"memory_id": "memory-1", "bucket_id": "bucket-1"},
            "revisions": [], "rings": [], "projection_status": [],
        },
        list_aliases=lambda **_kwargs: [{"alias": "晏晏", "trust": "owner"}],
    )
    service = GatewayService.__new__(GatewayService)
    service._authorize = lambda _header: None
    service.memory_authority_view = view
    service.bucket_mgr = Buckets()
    service.embedding_engine = SimpleNamespace(enabled=True)
    service.reranker_engine = SimpleNamespace(enabled=True)
    service.query_planner_enabled = True
    service.semantic_rescue_enabled = False
    service.domain_sentinel_remote_in_prepare = False
    service._retrieval_runtime_debug = lambda: {
        "embedding": {"status": "ok"}, "reranker": {"status": "ok"}
    }

    overview = asyncio.run(service.handle_memory_authority_read(Request("overview")))
    metadata = asyncio.run(service.handle_memory_authority_read(Request("memories")))
    with_body = asyncio.run(service.handle_memory_authority_read(Request(
        "memories", query={"include_body": "true"}
    )))
    detail = asyncio.run(service.handle_memory_authority_read(Request(
        "memory_detail", memory_id="memory-1"
    )))
    settings = asyncio.run(service.handle_memory_authority_read(Request("settings")))
    diagnostics = asyncio.run(service.handle_memory_authority_read(Request("diagnostics")))

    bodies = [json.loads(response.body.decode("utf-8")) for response in (
        overview, metadata, with_body, detail, settings, diagnostics,
    )]
    assert all(response.status_code == 200 for response in (
        overview, metadata, with_body, detail, settings, diagnostics,
    ))
    assert "body" not in bodies[1]["items"][0]
    assert bodies[2]["items"][0]["body"] == "owner body"
    assert bodies[3]["body"] == "owner body"
    assert bodies[4]["settings"]["policy_revision"] == "recall-r1-natural-v1"
    assert bodies[5]["diagnostics"]["route"] == "unknown"
    assert "synthetic" not in str(bodies)


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
