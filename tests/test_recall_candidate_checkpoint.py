"""Request-local natural Recall checkpoint regressions.

All records and providers below are synthetic. These tests exercise the
timeout/fallback seam without making a provider or touching a Memory store.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import gateway as gateway_module
from test_recall_natural_r1 import _natural_bucket, _natural_candidate, _natural_finish_service


def test_timeout_fallback_reuses_planner_candidate_pool_after_rerank_cancelled():
    query = "An ordinary paraphrase about an evening experience."
    supplemental_query = "What happened during that evening together?"
    weak_bucket = _natural_bucket("memory-checkpoint-weak", "A weakly related committed fact.")
    strong_bucket = _natural_bucket("memory-checkpoint-strong", "A verified evening experience.")
    weak = _natural_candidate(weak_bucket, 0.62)
    strong = _natural_candidate(strong_bucket, 0.94, unit_id="verified-evening-unit")
    service, _fixture_calls, _metadata = _natural_finish_service(
        None, [weak_bucket, strong_bucket], {},
    )
    service.recall_timeout_seconds = 0.05
    service.query_planner_enabled = True
    service.query_planner_supplemental_semantic = True

    candidate_calls = []
    semantic_scans = []
    planner_calls = []
    rerank_started = asyncio.Event()
    rerank_cancelled = asyncio.Event()
    rerank_calls = []

    async def planner(planner_query):
        planner_calls.append(planner_query)
        return {"queries": [{"query": supplemental_query}]}, None

    async def candidate_builder(candidate_query, _session_id, _buckets, *, allow_semantic, **_kwargs):
        candidate_calls.append((candidate_query, allow_semantic))
        if allow_semantic:
            semantic_scans.append(candidate_query)
        if candidate_query == supplemental_query:
            return ([dict(strong)] if allow_semantic else []), []
        return [dict(weak)], []

    async def blocked_rerank(_query, items, **_kwargs):
        rerank_calls.append([item["bucket"]["id"] for item in items])
        rerank_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            rerank_cancelled.set()

    async def selector(selected_query, session_id, buckets, *, natural_input,
                       allow_query_planner, allow_rerank, **_kwargs):
        natural_input.update(
            allow_query_planner=allow_query_planner,
            allow_rerank=allow_rerank,
        )
        return await service._finish_natural_selection(
            selected_query, session_id, buckets, recall_input=natural_input,
        )

    service._call_query_planner = planner
    service._dynamic_bucket_candidate_items = candidate_builder
    service._rerank_scored_bucket_candidates = blocked_rerank
    natural_input = {
        "route": "ordinary",
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "_tasks": [],
    }

    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector,
        query,
        "synthetic-checkpoint-session",
        [weak_bucket, strong_bucket],
        sentinel_debug={"route": "ordinary"},
        natural_input=natural_input,
        allow_semantic=True,
        allow_query_planner=True,
        allow_rerank=True,
    ))

    assert rerank_started.is_set() and rerank_cancelled.is_set()
    assert planner_calls == [query]
    assert candidate_calls == [(query, True), (supplemental_query, True)]
    assert semantic_scans == [query, supplemental_query]
    assert rerank_calls == [[weak_bucket["id"], strong_bucket["id"]]]
    assert [bucket["id"] for bucket in selected] == [strong_bucket["id"]]
    assert [item["bucket"]["id"] for item in suppressed] == [weak_bucket["id"]]
    assert debug["candidate_checkpoint"]["status"] == "reused"
    assert debug["candidate_checkpoint"]["candidate_count"] == 2
    assert [attempt["status"] for attempt in debug["selection_attempts"]] == [
        "timeout", "completed",
    ]
    assert "recall_total_timeout" in natural_input["incomplete_reasons"]
    assert service.recall_policy.semantic_threshold == 0.8


def _rebuild_fixture():
    query = "A synthetic query about a committed preference."
    bucket = _natural_bucket("memory-checkpoint-revalidation", "A committed preference fact.")
    candidate = _natural_candidate(bucket, 0.91)
    service, _fixture_calls, metadata = _natural_finish_service(None, [bucket], {})
    builder_calls = []
    allowed_ids = {bucket["id"]}

    async def candidate_builder(candidate_query, _session_id, eligible_buckets,
                                *, allow_semantic, **_kwargs):
        eligible_ids = {str(item.get("id") or "") for item in eligible_buckets}
        builder_calls.append((candidate_query, allow_semantic, sorted(eligible_ids)))
        return ([dict(candidate)] if bucket["id"] in eligible_ids else []), []

    service._dynamic_bucket_candidate_items = candidate_builder
    service.query_planner_enabled = False
    service._authority_auto_recall_allowed = lambda source: str(source.get("id") or "") in allowed_ids
    natural_input = {
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "allow_query_planner": False,
        "allow_rerank": False,
    }
    return query, bucket, candidate, service, metadata, builder_calls, allowed_ids, natural_input


def _finish_once(service, query, session_id, buckets, natural_input):
    return asyncio.run(service._finish_natural_selection(
        query, session_id, buckets, recall_input=natural_input,
    ))


@pytest.mark.parametrize("context_change", ["query", "context", "session"])
def test_checkpoint_is_not_reused_when_request_context_changes(context_change):
    query, bucket, _candidate, service, _metadata, builder_calls, _allowed_ids, natural_input = (
        _rebuild_fixture()
    )
    _finish_once(service, query, "same-session", [bucket], natural_input)
    assert len(builder_calls) == 1
    natural_input["local_only"] = True

    next_query = query
    next_session = "same-session"
    if context_change == "query":
        next_query = query + " with a new current query"
    elif context_change == "context":
        natural_input["q_context"] = query + " with changed conversation context"
    else:
        next_session = "different-session"

    _selected, _suppressed, debug = _finish_once(
        service, next_query, next_session, [bucket], natural_input,
    )

    assert len(builder_calls) == 2
    assert builder_calls[-1][1] is False
    assert debug["candidate_checkpoint"]["status"] == "invalidated"


@pytest.mark.parametrize("authority_change", ["revision", "state", "body"])
def test_checkpoint_revalidates_authority_revision_state_and_body(authority_change):
    query, bucket, _candidate, service, metadata, builder_calls, allowed_ids, natural_input = (
        _rebuild_fixture()
    )
    _finish_once(service, query, "same-session", [bucket], natural_input)
    assert len(builder_calls) == 1
    natural_input["local_only"] = True

    if authority_change == "revision":
        metadata[bucket["id"]]["revision"] = 2
    elif authority_change == "state":
        allowed_ids.clear()
    else:
        bucket["content"] = "The committed source body changed after checkpointing."

    selected, _suppressed, debug = _finish_once(
        service, query, "same-session", [bucket], natural_input,
    )

    assert len(builder_calls) == 2
    assert debug["candidate_checkpoint"]["status"] == "invalidated"
    if authority_change == "revision":
        assert [item["id"] for item in selected] == [bucket["id"]]
        assert builder_calls[-1][2] == [bucket["id"]]
        assert metadata[bucket["id"]]["revision"] == 2
    else:
        assert selected == []
        assert builder_calls[-1][2] == []


def test_checkpoint_is_request_local_and_not_shared_across_requests():
    query, bucket, _candidate, service, _metadata, builder_calls, _allowed_ids, first_input = (
        _rebuild_fixture()
    )
    _finish_once(service, query, "same-session", [bucket], first_input)
    assert "_completed_candidate_pool" in first_input

    second_input = {
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "allow_query_planner": False,
        "allow_rerank": False,
        "local_only": True,
    }
    assert "_completed_candidate_pool" not in second_input
    _finish_once(service, query, "same-session", [bucket], second_input)

    assert len(builder_calls) == 2
    assert builder_calls[-1][1] is False
    assert second_input.get("_completed_candidate_pool") is None


def test_successful_non_timeout_pass_does_not_repeat_candidate_or_rerank_work():
    query = "A synthetic ordinary preference question."
    bucket = _natural_bucket("memory-checkpoint-normal", "A verified preference fact.")
    candidate = _natural_candidate(bucket, 0.92)
    service, calls, _metadata = _natural_finish_service(
        None, [bucket], {query: [candidate]}, rerank_score=0.93,
    )
    service.recall_timeout_seconds = 1.0
    natural_input = {
        "route": "ordinary",
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "_tasks": [],
    }

    async def selector(selected_query, session_id, buckets, *, natural_input,
                       allow_query_planner, allow_rerank, **_kwargs):
        natural_input.update(
            allow_query_planner=allow_query_planner,
            allow_rerank=allow_rerank,
        )
        return await service._finish_natural_selection(
            selected_query, session_id, buckets, recall_input=natural_input,
        )

    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector,
        query,
        "synthetic-normal-session",
        [bucket],
        sentinel_debug={"route": "ordinary"},
        natural_input=natural_input,
        allow_semantic=True,
        allow_query_planner=False,
        allow_rerank=True,
    ))

    assert [item["id"] for item in selected] == [bucket["id"]]
    assert suppressed == []
    assert calls["candidate"] == [query]
    assert len(calls["rerank"]) == 1
    assert debug["candidate_checkpoint"]["status"] == "saved"
    assert [(attempt["phase"], attempt["status"]) for attempt in debug["selection_attempts"]] == [
        ("initial", "completed"),
    ]
    assert "_completed_candidate_pool" not in natural_input


def test_graph_checkpoint_resume_preserves_matched_moment_after_rerank_timeout(monkeypatch):
    query = "An ordinary paraphrase about a shared evening."
    supplemental_query = "What happened together that evening?"
    weak_bucket = _natural_bucket("memory-graph-checkpoint-weak", "A weakly related fact.")
    strong_bucket = _natural_bucket("memory-graph-checkpoint-strong", "A verified evening experience.")
    weak = _natural_candidate(weak_bucket, 0.61)
    strong = _natural_candidate(strong_bucket, 0.95, unit_id="verified-evening-moment")
    service, _fixture_calls, _metadata = _natural_finish_service(
        None, [weak_bucket, strong_bucket], {},
    )
    service.recall_timeout_seconds = 0.05
    service.query_planner_enabled = True
    service.query_planner_supplemental_semantic = True

    old_moment = {
        "moment_id": "verified-evening-moment",
        "bucket_id": strong_bucket["id"],
        "source": "content",
        "score": 0.97,
        "content": "The verified shared-evening passage.",
    }
    replacement_moment = {
        **old_moment,
        "moment_id": "replacement-after-timeout",
        "content": "A different passage made visible after timeout.",
    }
    weak_moment = {
        "moment_id": "weak-local-moment",
        "bucket_id": weak_bucket["id"],
        "source": "content",
        "score": 0.4,
        "content": "A weak local passage.",
    }
    moment_state = {
        weak_bucket["id"]: weak_moment,
        strong_bucket["id"]: old_moment,
    }
    parse_calls = []
    search_calls = []
    candidate_calls = []
    planner_calls = []
    rerank_documents = []
    rerank_cancelled = asyncio.Event()

    def parse_moments(bucket, _options):
        bucket_id = str(bucket.get("id") or "")
        parse_calls.append(bucket_id)
        moment = moment_state.get(bucket_id)
        return [dict(moment)] if moment else []

    monkeypatch.setattr(gateway_module, "parse_bucket_moments", parse_moments)

    def search_moments(_query, moments, *, limit):
        search_calls.append((len(moments), limit))
        return []

    service.memory_moment_store = SimpleNamespace(search_moment_items=search_moments)
    service._moment_rerank_document = lambda moment: f"document:{moment['moment_id']}"
    service._direct_moments_for_bucket = lambda bucket, _query: [
        dict(moment_state[str(bucket.get("id") or "")])
    ] if str(bucket.get("id") or "") in moment_state else []
    service._moment_with_bucket_recall_signal = lambda moment, _signal: dict(moment)
    service._bucket_candidate_recall_signal = lambda item: {"bucket_id": item["bucket"]["id"]}

    async def planner(planner_query):
        planner_calls.append(planner_query)
        return {"queries": [{"query": supplemental_query}]}, None

    async def candidate_builder(candidate_query, _session_id, _buckets, *, allow_semantic, **_kwargs):
        candidate_calls.append((candidate_query, allow_semantic))
        if candidate_query == supplemental_query:
            return ([dict(strong)] if allow_semantic else []), []
        return [dict(weak)], []

    async def blocked_rerank(_query, items, *, documents_override=None, **_kwargs):
        rerank_documents.append(dict(documents_override or {}))
        try:
            await asyncio.Event().wait()
        finally:
            moment_state[strong_bucket["id"]] = replacement_moment
            rerank_cancelled.set()

    async def selector(selected_query, session_id, buckets, *, natural_input,
                       allow_query_planner, allow_rerank, **_kwargs):
        natural_input.update(
            allow_query_planner=allow_query_planner,
            allow_rerank=allow_rerank,
        )
        return await service._finish_natural_selection(
            selected_query,
            session_id,
            buckets,
            recall_input=natural_input,
            grouped_moments={},
        )

    service._call_query_planner = planner
    service._dynamic_bucket_candidate_items = candidate_builder
    service._rerank_scored_bucket_candidates = blocked_rerank
    natural_input = {
        "route": "ordinary",
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "_tasks": [],
    }

    moments, candidates, _suppressed_moments, suppressed_buckets, debug = asyncio.run(
        service._select_recall_with_fallback(
            selector,
            query,
            "synthetic-graph-session",
            [weak_bucket, strong_bucket],
            sentinel_debug={"route": "ordinary"},
            natural_input=natural_input,
            allow_semantic=True,
            allow_query_planner=True,
            allow_rerank=True,
        )
    )

    assert rerank_cancelled.is_set()
    assert planner_calls == [query]
    assert candidate_calls == [(query, True), (supplemental_query, True)]
    assert len(parse_calls) == 2
    assert len(search_calls) == 1
    assert rerank_documents == [{
        strong_bucket["id"]: "document:verified-evening-moment",
    }]
    assert [item["moment_id"] for item in moments] == ["verified-evening-moment"]
    assert [item["moment_id"] for item in candidates if item["bucket_id"] == strong_bucket["id"]] == [
        "verified-evening-moment",
    ]
    assert [item["bucket"]["id"] for item in suppressed_buckets] == [weak_bucket["id"]]
    assert debug["candidate_checkpoint"]["status"] == "reused"


def test_revoked_owner_alias_is_reparsed_after_rerank_timeout_before_forced_admission():
    query = "Who is the synthetic owner-alias test person?"
    bucket = _natural_bucket("memory-checkpoint-alias", "An otherwise weak committed fact.")
    candidate = _natural_candidate(bucket, 0.22)
    service, _fixture_calls, metadata = _natural_finish_service(None, [bucket], {})
    service.recall_timeout_seconds = 0.05
    service.inject_max_cards = 3
    service.first_card_min_score = 0.55
    service._dedupe_evidence_labels = lambda labels: list(dict.fromkeys(labels))
    service._bucket_evidence_labels = lambda _query, item: list(
        item.get("evidence_labels") or item.get("_test_labels") or []
    )

    stamp = [(1, 100, 0, 0)]
    alias_active = [True]
    alias_queries = []
    metadata_before = (
        metadata[bucket["id"]]["revision"],
        metadata[bucket["id"]]["body_sha256"],
    )

    def match_aliases(alias_query):
        alias_queries.append(alias_query)
        if not alias_active[0]:
            return []
        return [{"source_refs": [f"memory:{bucket['id']}"]}]

    service.memory_authority_view = SimpleNamespace(
        enabled=True,
        _file_stamp=lambda: stamp[0],
        memory_index_metadata_map=lambda _ids: metadata,
        match_aliases=match_aliases,
    )
    candidate_calls = []
    rerank_started = asyncio.Event()
    rerank_cancelled = asyncio.Event()
    rerank_inputs = []

    async def candidate_builder(candidate_query, _session_id, _buckets, **_kwargs):
        candidate_calls.append(candidate_query)
        if _kwargs.get("allow_semantic", True):
            return [], []
        return [dict(candidate)], []

    async def blocked_rerank(_rerank_query, items, **_kwargs):
        rerank_inputs.append([dict(item) for item in items])
        rerank_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            alias_active[0] = False
            stamp[0] = (2, 200, 0, 0)
            rerank_cancelled.set()

    async def selector(selected_query, session_id, buckets, *, natural_input,
                       forced_memory_ids, allow_query_planner, allow_rerank, **_kwargs):
        natural_input.update(
            allow_query_planner=allow_query_planner,
            allow_rerank=allow_rerank,
        )
        return await service._finish_natural_selection(
            selected_query,
            session_id,
            buckets,
            recall_input=natural_input,
            forced_memory_ids=forced_memory_ids,
        )

    service._dynamic_bucket_candidate_items = candidate_builder
    service._rerank_scored_bucket_candidates = blocked_rerank
    natural_input = {
        "route": "ordinary",
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "_tasks": [],
    }

    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector,
        query,
        "synthetic-alias-session",
        [bucket],
        sentinel_debug={"route": "ordinary"},
        natural_input=natural_input,
        forced_memory_ids=[bucket["id"]],
        allow_semantic=True,
        allow_query_planner=False,
        allow_rerank=True,
    ))

    assert rerank_started.is_set() and rerank_cancelled.is_set()
    assert alias_queries == [query, query]
    assert candidate_calls == [query, query]
    assert rerank_inputs[0][0]["owner_alias_match"] is True
    assert "owner_alias" in rerank_inputs[0][0]["evidence_labels"]
    assert selected == []
    assert [item["bucket"]["id"] for item in suppressed] == [bucket["id"]]
    assert debug["candidate_checkpoint"]["status"] == "invalidated"
    assert (
        metadata[bucket["id"]]["revision"],
        metadata[bucket["id"]]["body_sha256"],
    ) == metadata_before


@pytest.mark.parametrize("change_phase", ["candidate", "planner"])
def test_authority_generation_change_during_candidate_work_prevents_checkpoint_save(change_phase):
    query = "An ordinary synthetic question with a weak candidate."
    supplemental_query = "What is the relevant committed preference?"
    weak_bucket = _natural_bucket("memory-checkpoint-generation-weak", "A weak fact.")
    strong_bucket = _natural_bucket("memory-checkpoint-generation-strong", "A strong fact.")
    weak = _natural_candidate(weak_bucket, 0.48)
    strong = _natural_candidate(strong_bucket, 0.93)
    service, _fixture_calls, metadata = _natural_finish_service(
        None, [weak_bucket, strong_bucket], {},
    )
    generation = [(10, 1000, 0, 0)]
    service.memory_authority_view = SimpleNamespace(
        _file_stamp=lambda: generation[0],
        memory_index_metadata_map=lambda _ids: metadata,
    )
    service.query_planner_enabled = change_phase == "planner"
    service.query_planner_supplemental_semantic = True
    candidate_calls = []
    planner_calls = []

    async def candidate_builder(candidate_query, _session_id, _buckets, **_kwargs):
        candidate_calls.append(candidate_query)
        if change_phase == "candidate":
            generation[0] = (11, 1100, 0, 0)
            return [dict(strong)], []
        if candidate_query == supplemental_query:
            return [dict(strong)], []
        return [dict(weak)], []

    async def planner(_planner_query):
        planner_calls.append(query)
        generation[0] = (11, 1100, 0, 0)
        return {"queries": [{"query": supplemental_query}]}, None

    service._dynamic_bucket_candidate_items = candidate_builder
    service._call_query_planner = planner
    natural_input = {
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "allow_query_planner": True,
        "allow_rerank": False,
    }
    if change_phase == "planner":
        service.query_planner_enabled = True
    selected, _suppressed, debug = _finish_once(
        service, query, "generation-session", [weak_bucket, strong_bucket], natural_input,
    )

    assert candidate_calls == ([query] if change_phase == "candidate" else [query, supplemental_query])
    assert planner_calls == ([] if change_phase == "candidate" else [query])
    assert debug["candidate_checkpoint"]["status"] == "authority_changed"
    assert "_completed_candidate_pool" not in natural_input
    if change_phase == "candidate":
        assert [item["id"] for item in selected] == [strong_bucket["id"]]


def test_reranker_mutating_pool_then_cancelling_does_not_contaminate_resume():
    query = "A verified synthetic fact that should survive timeout."
    bucket = _natural_bucket("memory-checkpoint-rerank-mutation", "A committed verified fact.")
    candidate = _natural_candidate(bucket, 0.92)
    service, calls, _metadata = _natural_finish_service(None, [bucket], {query: [candidate]})
    service.recall_timeout_seconds = 0.05
    rerank_started = asyncio.Event()
    rerank_cancelled = asyncio.Event()
    rerank_observed = []

    async def mutating_blocked_rerank(_rerank_query, items, **_kwargs):
        rerank_observed.append(items[0])
        items[0]["semantic_score"] = 0.1
        items[0]["rerank_score"] = 0.99
        items[0]["_test_labels"] = ["owner_alias"]
        items[0]["mutated_by_reranker"] = True
        rerank_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            rerank_cancelled.set()

    async def selector(selected_query, session_id, buckets, *, natural_input,
                       allow_query_planner, allow_rerank, **_kwargs):
        natural_input.update(
            allow_query_planner=allow_query_planner,
            allow_rerank=allow_rerank,
        )
        return await service._finish_natural_selection(
            selected_query, session_id, buckets, recall_input=natural_input,
        )

    service._rerank_scored_bucket_candidates = mutating_blocked_rerank
    natural_input = {
        "route": "ordinary",
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "_tasks": [],
    }
    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector,
        query,
        "synthetic-rerank-mutation-session",
        [bucket],
        sentinel_debug={"route": "ordinary"},
        natural_input=natural_input,
        allow_semantic=True,
        allow_query_planner=False,
        allow_rerank=True,
    ))

    assert rerank_started.is_set() and rerank_cancelled.is_set()
    assert len(rerank_observed) == 1
    assert calls["candidate"] == [query]
    assert [item["id"] for item in selected] == [bucket["id"]]
    assert suppressed == []
    assert calls["selected"][0]["semantic_score"] == 0.92
    assert calls["selected"][0].get("rerank_score") is None
    assert calls["selected"][0]["admission_reason"] == "strong_semantic"
    assert calls["selected"][0]["evidence_labels"] == ["semantic_only"]
    assert "mutated_by_reranker" not in calls["selected"][0]
    assert debug["candidate_checkpoint"]["status"] == "reused"


def test_authority_stamp_change_during_metadata_validation_blocks_checkpoint_resume():
    query = "A synthetic query for a stable committed source."
    bucket = _natural_bucket("memory-checkpoint-validation-stamp", "A committed source body.")
    candidate = _natural_candidate(bucket, 0.91)
    service, _fixture_calls, metadata = _natural_finish_service(None, [bucket], {})
    generation = [(21, 2100, 0, 0)]
    metadata_calls = []
    builder_calls = []

    def metadata_map(_ids):
        metadata_calls.append(True)
        result = metadata
        if len(metadata_calls) == 2:
            generation[0] = (22, 2200, 0, 0)
        return result

    service.memory_authority_view = SimpleNamespace(
        _file_stamp=lambda: generation[0],
        memory_index_metadata_map=metadata_map,
    )
    service.query_planner_enabled = False

    async def candidate_builder(candidate_query, _session_id, _buckets, *, allow_semantic, **_kwargs):
        builder_calls.append((candidate_query, allow_semantic))
        return [dict(candidate)], []

    service._dynamic_bucket_candidate_items = candidate_builder
    natural_input = {
        "q_current": query,
        "q_context": query,
        "metadata": {},
        "allow_query_planner": False,
        "allow_rerank": False,
    }
    _finish_once(service, query, "stamp-validation-session", [bucket], natural_input)
    assert len(builder_calls) == 1
    natural_input["local_only"] = True

    _selected, _suppressed, debug = _finish_once(
        service, query, "stamp-validation-session", [bucket], natural_input,
    )

    assert len(metadata_calls) == 2
    assert builder_calls == [(query, True), (query, False)]
    assert debug["candidate_checkpoint"]["status"] == "invalidated"
