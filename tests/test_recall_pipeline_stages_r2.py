"""Count-only deadline stages and inactive Planner work; synthetic providers only."""
import asyncio
import json

import pytest

import gateway as gateway_module
from test_recall_natural_r1 import _natural_bucket, _natural_candidate, _natural_finish_service


@pytest.mark.parametrize("enabled, expected_scans", [(False, 41), (True, 82)])
def test_planner_evidence_scan_only_runs_when_expansion_is_enabled(enabled, expected_scans):
    query = "synthetic preference query"
    buckets = [_natural_bucket(f"stage-{i}", "PRIVATE_BODY") for i in range(41)]
    pool = [_natural_candidate(bucket, 0.2) for bucket in buckets]
    service, calls, _ = _natural_finish_service(None, buckets, {query: pool}, rerank_score=0.9)
    service.query_planner_enabled = enabled
    scans, planner_calls = [], []

    def evidence(_query, item):
        scans.append(item["bucket"]["id"])
        return []

    async def planner(_query):
        planner_calls.append(_query)
        return {"queries": [], "_cache": {"status": "hit"}}, None

    service._bucket_evidence_labels = evidence
    service._call_query_planner = planner
    request = {"q_current": query, "q_context": query, "metadata": {}}
    selected, _, debug = asyncio.run(service._finish_natural_selection(
        query, "synthetic-session", buckets, recall_input=request,
    ))
    assert len(selected) == 41
    assert len(scans) == expected_scans
    assert len(calls["rerank"]) == 1
    assert len(planner_calls) == int(enabled)
    eligibility = next(row for row in debug["candidate_stages"] if row["stage"] == "natural.planner_eligibility")
    assert eligibility["skipped"] is not enabled
    if enabled:
        planner_stage = next(row for row in debug["candidate_stages"] if row["stage"] == "natural.query_planner")
        assert planner_stage["cache"] == {"status": "hit"}


def test_source_deadline_records_cancelled_stage_without_retry_or_rerank(monkeypatch):
    query = "synthetic source wait"
    bucket = _natural_bucket("source-stage", "PRIVATE_BODY")
    service, calls, _ = _natural_finish_service(
        None, [bucket], {query: [_natural_candidate(bucket, 0.9)]},
    )
    service.recall_timeout_seconds = 0.03
    source_calls, source_cancelled = [], []

    async def sources(_service, _request, _query, items):
        source_calls.append(len(items))
        try:
            await asyncio.Event().wait()
        finally:
            source_cancelled.append(True)

    monkeypatch.setattr(gateway_module.memory_use, "verify_sources", sources)

    async def selector(selected_query, session_id, buckets, *, natural_input, **kwargs):
        return await service._finish_natural_selection(
            selected_query, session_id, buckets, recall_input=natural_input,
        )

    request = {"q_current": query, "q_context": query, "metadata": {}, "_tasks": []}
    selected, _, debug = asyncio.run(service._select_recall_with_fallback(
        selector, query, "synthetic-session", [bucket], natural_input=request,
        sentinel_debug={"route": "ordinary"},
    ))
    assert selected == []
    assert source_calls == [1] and source_cancelled == [True]
    assert calls["rerank"] == []
    attempt = debug["selection_attempts"][0]
    stage = next(row for row in attempt["candidate_stages"] if row["stage"] == "natural.source_verification")
    assert stage["status"] == "cancelled"
    assert stage["input_count"] == 1 and stage["output_count"] == 0
    assert stage["duration_ms"] == attempt["timing_ms"]["natural.source_verification"]
    assert "PRIVATE_BODY" not in json.dumps(attempt)
    assert request["local_only"] is True


def test_slow_planner_cancellation_has_stage_and_no_second_provider_call():
    query = "synthetic slow planner"
    bucket = _natural_bucket("planner-stage", "PRIVATE_BODY")
    service, calls, _ = _natural_finish_service(
        None, [bucket], {query: [_natural_candidate(bucket, 0.2)]},
    )
    service.query_planner_enabled = True
    planner_calls = []
    request = {"q_current": query, "q_context": query, "metadata": {}}

    async def planner(_query):
        planner_calls.append(_query)
        await asyncio.Event().wait()

    service._call_query_planner = planner

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(service._finish_natural_selection(
                query, "synthetic-session", [bucket], recall_input=request,
            ), timeout=0.03)

    asyncio.run(run())
    assert planner_calls == [query]
    assert calls["rerank"] == []
    debug = request["_selection_debug"]
    stage = next(row for row in debug["candidate_stages"] if row["stage"] == "natural.query_planner")
    assert stage["status"] == "cancelled"
    assert stage["duration_ms"] == debug["timing_ms"]["natural.query_planner"]
    assert "PRIVATE_BODY" not in json.dumps(stage)


def test_source_upstream_diagnostics_keep_only_known_nonnegative_integer_fields(tmp_path, monkeypatch):
    from test_memory_recall_gateway_r2 import (
        _authority_fixture, _base_config, _gateway_service, _natural_candidate as candidate,
        _recall_input,
    )

    _, view, records, buckets = _authority_fixture(tmp_path, [("diagnostic-source", "synthetic body")])
    service, _ = _gateway_service(tmp_path, monkeypatch, view, buckets,
                                  [candidate(buckets[0], score=0.9, keyword_score=0.8)], _base_config())
    request = _recall_input()
    io_calls = []

    async def read_batch(_service, _request, stage, _path, payload):
        io_calls.append(stage)
        return {"items": [{"memory_id": item["memory_id"], "revision": item["revision"],
                           "state": "ok", "sources": {binding: {"state": "available", "eligible_source": True}
                             for binding in item["binding_ids"]}} for item in payload["items"]],
                "diagnostics": {"timing_ms": {"prepare": 3, "source_reads": 4,
                    "final_authority": 5, "body": "PRIVATE_BODY"},
                    "member_batches": 2, "member_cache_hits": True,
                    "unique_binding_reads": -1, "raw_source": "PRIVATE_BODY"}}

    monkeypatch.setattr(gateway_module.memory_use, "read_batch", read_batch)
    gateway_module.memory_use.prepare(service, request, request["q_current"], buckets)
    pool = [candidate(buckets[0], score=0.9, keyword_score=0.8)]
    permitted = asyncio.run(gateway_module.memory_use.verify_sources(
        service, request, request["q_current"], pool))
    assert len(permitted) == 1 and io_calls == ["sources"]
    stats = request["recall_io"]["sources"]
    assert stats["upstream_timing_ms"] == {"prepare": 3, "source_reads": 4, "final_authority": 5}
    assert stats["member_batches"] == 2
    assert "member_cache_hits" not in stats and "unique_binding_reads" not in stats
    assert "PRIVATE_BODY" not in json.dumps(stats)
