"""Synthetic regressions for natural candidate starvation; never call a provider."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from gateway import GatewayService
from recall_rerank_window import admission_evidence, select_natural_window
from test_recall_natural_r1 import _natural_finish_service


def candidate(identifier, *, semantic=0.0, keyword=0.0, moment=False, exact=False, kind="event"):
    return {
        "bucket": {"id": identifier, "content": "Private synthetic text " + identifier,
                   "metadata": {"memory_type": kind, "narrator_id": "original-author"}},
        "score": max(semantic, keyword, 0.3), "semantic_score": semantic,
        "keyword_score": keyword, "moment_source_match": moment,
        "exact_anchor_match": exact,
        "natural_semantic": {"index_status": "verified"} if semantic else {},
    }


def priority(_query, item):
    return (not item.get("exact_anchor_match", False), True, True, True,
            -item["semantic_score"], -item["keyword_score"], -item["score"])


def observed_shape_pool():
    # Shape of 11410, not private documents or fabricated provider evidence:
    # four disjoint lexical hits, 24 semantic hits and 16 moment-only hits.
    return ([candidate(f"keyword-{i}", keyword=0.74 - i * 0.01) for i in range(4)]
            + [candidate(f"semantic-{i:02}", semantic=0.58 - i * 0.002) for i in range(24)]
            + [candidate(f"moment-{i:02}", moment=True) for i in range(16)])


def test_44_candidate_window_keeps_current_keyword_leaders_with_same_six_slot_cap():
    pool = observed_shape_pool()
    before = deepcopy(pool)
    selected, notes, debug = select_natural_window(pool, 6, [priority("", x) for x in pool])
    ids = {pool[i]["bucket"]["id"] for i in selected}
    assert {"keyword-0", "keyword-1"} <= ids
    assert len(selected) == 6
    assert debug["channel_selected"] == {"keyword": 2, "semantic": 2, "moment": 2, "structural": 0}
    assert len(notes) == 44 and pool == before


def test_rank_fusion_ignores_cross_channel_scale_and_pool_set_order():
    pool = observed_shape_pool()
    selected, _, _ = select_natural_window(pool, 6, [priority("", x) for x in pool])
    expected = {pool[i]["bucket"]["id"] for i in selected}
    changed = deepcopy(list(reversed(pool)))
    for item in changed:
        item["keyword_score"] *= 100
        item["semantic_score"] *= 0.001
    selected, _, _ = select_natural_window(changed, 6, [priority("", x) for x in changed])
    assert {changed[i]["bucket"]["id"] for i in selected} == expected


@pytest.mark.parametrize("kind", ["preference", "promise", "event", "diary", "reflection", "technical"])
def test_memory_type_and_first_person_do_not_control_semantic_eligibility(kind):
    item = candidate("one", semantic=0.9, kind=kind)
    item["bucket"]["content"] = "I remember this scene. " * 600
    before = deepcopy(item)
    selected, notes, _ = select_natural_window([item], 1, [priority("", item)])
    assert selected == {0} and notes[0]["channel_ranks"] == {"semantic": 1}
    assert item == before  # No narrative conversion or retrieval-window truncation.


def test_special_evidence_priority_and_single_channel_semantic_order_are_preserved():
    pool = [candidate("weak-exact", exact=True), candidate("strong", semantic=0.99)]
    selected, _, _ = select_natural_window(pool, 1, [priority("", x) for x in pool])
    assert selected == {0}
    pool = [candidate("low", semantic=0.2), candidate("high", semantic=0.9)]
    selected, _, _ = select_natural_window(pool, 1, [priority("", x) for x in pool])
    assert selected == {1}


def test_channel_overlap_uses_one_document_slot_and_never_exceeds_pool_or_limit():
    pool = [candidate("both", keyword=0.8, semantic=0.8), candidate("single", semantic=0.9)]
    selected, notes, debug = select_natural_window(pool, 1, [priority("", x) for x in pool])
    assert selected == {0} and debug["selected_count"] == 1
    assert set(notes[0]["channel_ranks"]) == {"keyword", "semantic"}
    assert select_natural_window(pool, 100, [priority("", x) for x in pool])[0] == {0, 1}
    assert select_natural_window([], 6, [])[0] == set()
    assert select_natural_window(pool, 0, [priority("", x) for x in pool])[0] == set()


def service_with_provider(pool, *, fail=False, partial=False, enabled=True):
    service = GatewayService.__new__(GatewayService)
    service.reranker_engine = SimpleNamespace(enabled=enabled, candidate_limit=6, score_weight=0.65)
    service._bucket_rerank_candidate_priority = priority
    service._bucket_rerank_document = lambda b: b["content"]
    service._bucket_reranked_candidate_rank = lambda _q, x: (-x["score"],)
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        kwargs["cache_debug"].update(provider_status="error" if fail else "ok",
                                     provider_failed=fail, provider_requests=1)
        if fail:
            return []
        # Reverse response order to check provider index->original head mapping.
        count = 1 if partial else len(kwargs["documents"])
        return [SimpleNamespace(index=i, score=0.1 + i * 0.01) for i in reversed(range(count))]

    service._rerank_cached = provider
    return service, calls


@pytest.mark.parametrize("natural", [False, True])
def test_only_natural_path_changes_window_and_provider_indices_stay_attached(natural):
    pool = observed_shape_pool()
    original = deepcopy(pool)
    service, calls = service_with_provider(pool)
    debug = {}
    result = asyncio.run(service._rerank_scored_bucket_candidates(
        "unchanged-context-query", pool, diagnostics=debug, natural_window=natural))
    assert len(calls) == 1 and calls[0]["top_n"] == 6
    documents = calls[0]["documents"]
    assert sum("keyword-" in doc for doc in documents) == (2 if natural else 0)
    assert calls[0]["query"] == "unchanged-context-query"
    assert len(result) == 44 and pool == original
    for item in result:
        document = item["bucket"]["content"]
        if document in documents:
            assert item["rerank_score"] == round(0.1 + documents.index(document) * 0.01, 4)
        else:
            assert item.get("rerank_score") is None


@pytest.mark.parametrize("fail,partial", [(True, False), (False, True)])
def test_failed_or_partial_provider_never_turns_unscored_candidates_into_zero(fail, partial):
    pool = observed_shape_pool()
    service, calls = service_with_provider(pool, fail=fail, partial=partial)
    result = asyncio.run(service._rerank_scored_bucket_candidates("query", pool, natural_window=True))
    assert len(result) == 44 and len(calls) == 1
    status_counts = {}
    for item in result:
        status = item["_rerank_window"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "scored":
            assert item.get("rerank_score") is None
    assert status_counts["outside_window"] == 38
    assert status_counts.get("provider_failed", 0) == (6 if fail else 0)
    assert status_counts.get("no_score", 0) == (5 if partial else 0)


def test_disabled_reranker_makes_no_call_and_keeps_all_candidate_evidence():
    pool = observed_shape_pool()
    service, calls = service_with_provider(pool, enabled=False)
    result = asyncio.run(service._rerank_scored_bucket_candidates("q", pool, natural_window=True))
    assert result == pool and not calls


def test_natural_integration_still_rejects_all_low_scores_and_records_every_candidate():
    pool = observed_shape_pool()
    service, _, _ = _natural_finish_service(None, [x["bucket"] for x in pool], {"*": pool})
    real, calls = service_with_provider(pool)
    service._rerank_scored_bucket_candidates = real._rerank_scored_bucket_candidates
    stages = []
    service._record_candidate_stage = lambda _stages, stage, before, after, **kw: stages.append(
        {"stage": stage, "input_count": before, "output_count": after, **kw})
    selected, suppressed, _ = asyncio.run(service._finish_natural_selection(
        "current query", "test-session", [x["bucket"] for x in pool],
        recall_input={"q_current": "current query", "q_context": "full contextual query", "metadata": {}}))
    assert not selected and len(suppressed) == 44 and len(calls) == 1
    rerank = next(s for s in stages if s["stage"] == "natural.final_rerank")
    assert rerank["window"]["policy"] == "natural_channel_rrf_v1"
    evidence = next(s for s in stages if s["stage"] == "natural.admit_candidates")["candidate_evidence"]
    assert evidence["recorded_count"] == 44 and evidence["truncated_count"] == 0
    assert sum(x["rerank_status"] == "scored" for x in evidence["candidates"]) == 6
    assert not any(x["admitted"] or x["selected"] for x in evidence["candidates"])
    assert evidence["semantic_threshold"] == evidence["rerank_threshold"] == 0.8


def test_admission_diagnostics_are_bounded_numeric_and_owner_safe():
    item = candidate("PRIVATE_ID_123", semantic=float("nan"))
    item["admission_reason"] = "PRIVATE BODY with spaces"
    evidence = admission_evidence([item] * 150, [], [], semantic_threshold=0.72, rerank_threshold=0.65)
    assert evidence["recorded_count"] == 128 and evidence["truncated_count"] == 22
    text = json.dumps(evidence, allow_nan=False)
    assert "PRIVATE" not in text and "original-author" not in text
    assert evidence["candidates"][0]["semantic_score"] is None
    assert evidence["candidates"][0]["admission_reason"] == "other"


def test_window_opportunity_can_reach_existing_admission_only_with_strong_provider_evidence():
    pool = observed_shape_pool()
    service, _, _ = _natural_finish_service(None, [x["bucket"] for x in pool], {"*": pool})
    real, _ = service_with_provider(pool)

    async def provider(**kwargs):
        return [SimpleNamespace(index=i, score=0.91 if "keyword-0" in doc else 0.02)
                for i, doc in enumerate(kwargs["documents"])]

    real._rerank_cached = provider
    service._rerank_scored_bucket_candidates = real._rerank_scored_bucket_candidates
    selected, suppressed, _ = asyncio.run(service._finish_natural_selection(
        "current query", "test-session", [x["bucket"] for x in pool],
        recall_input={"q_current": "current query", "q_context": "context", "metadata": {}}))
    assert [b["id"] for b in selected] == ["keyword-0"]
    assert selected[0] == pool[0]["bucket"]
    assert len(suppressed) == 43


def test_cancelled_window_propagates_without_retry_or_candidate_mutation():
    pool = observed_shape_pool()
    original = deepcopy(pool)
    service, _ = service_with_provider(pool)
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        raise asyncio.CancelledError

    service._rerank_cached = provider
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service._rerank_scored_bucket_candidates("q", pool, natural_window=True))
    assert len(calls) == 1 and pool == original
