"""Offline score adoption evidence; synthetic records, no providers or writes to Memory."""
import copy
import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone

import pytest

from gateway import GatewayService
from memory_l1_schema import document_template, statement_template
from memory_recall_semantics import score_document
from memory_score_evidence import CANDIDATE_LIMIT, EVIDENCE_BYTES, request_score_evidence, score_evidence
from memory_system_config import REQUEST_CONFIGURATION, preview, values_from_config
from model_request_trace import ModelRequestTraceStore


def example(*, direct=True):
    values = values_from_config({})
    values["time_profiles"] = [{"id": "dated_current_state", "selector": {
        "kind": "state", "stability": "phase", "query_purpose": "current"},
        "function": "exponential", "half_life_days": 7, "weight": 1}]
    values["recall"]["scoring_profiles"] = [{"id": "current_state", "selector": {
        "kind": "state", "query_purpose": "current"}, "formula_id": "final"}]
    statement = statement_template()
    statement.update(statement_id="state-1", content="PRIVATE_BODY_NOT_FOR_METADATA")
    statement["classification"].update(kind={"state": "known", "value": "state"},
                                        stability={"state": "known", "value": "phase"})
    statement["time"]["event"]["start"] = {"state": "known", "value": "2026-10-13T00:00:00+00:00"}
    document = document_template("PRIVATE_BODY_NOT_FOR_METADATA")
    document["statements"] = [statement]
    calculation = score_document(document, values, {"now": datetime(2026, 10, 20, tzinfo=timezone.utc),
        "purpose": "current", "query": "", "facts": {}}, {"relevance": .8, "confidence": None,
        "statement_selection": {"direct_ids": ["state-1"] if direct else [], "support_ids": [], "matches": {}}})
    computation = {"query_use": {"purpose": "current"}, "candidate_scores": [{
        "bucket_id": "memory-1", "selected": True, "admitted": True, "admission_reason": "strong_rerank",
        "raw_base_score": .7, "raw_fused_score": .8, "rerank_score": .9,
        "final_score": calculation["score"], "computation": calculation}]}
    return values, computation


def test_time_is_applied_before_relevance_formula_and_recorded_inputs_can_replay():
    values, computation = example()
    evidence = score_evidence({"revision": "mc-first", "values": values}, computation)
    assert evidence["replayable"] is True
    row = evidence["candidates"][0]["computation"]["statements"][0]
    assert row["features"]["base_relevance"] == .8
    assert row["features"]["freshness"] == pytest.approx(.5)
    assert row["features"]["time_factor"] == pytest.approx(.75)
    assert row["features"]["relevance"] == pytest.approx(.6)
    assert row["features"]["confidence"] is None
    assert (row["time_profile"], row["scoring_profile"], row["formula_id"]) == ("dated_current_state", "current_state", "final")
    recorded = evidence["configuration"]["values"]
    replay_values = values_from_config({})
    replay_values.update(recorded)
    inputs = row["calculation"]["inputs"]
    assert preview(replay_values, inputs)["formulas"] == row["calculation"]["formulas"]
    assert "PRIVATE_BODY" not in json.dumps(evidence)


def test_bucket_semantic_without_attributed_statement_records_global_formula():
    values, computation = example(direct=False)
    evidence = score_evidence({"revision": "mc-global", "values": values}, computation)
    row = evidence["candidates"][0]["computation"]
    assert row["state"] == "unassigned_semantic_evidence"
    assert row["aggregation_statement_ids"] == []
    assert row["statements"][0]["participates_in_aggregation"] is False
    assert row["calculation"]["inputs"]["relevance"] == .8
    assert row["score"] == .8
    assert row["reason_codes"] == ["no_attributable_statement_match"]


def test_partial_affinity_is_visible_in_actual_score_evidence_without_source_prose():
    values, computation = example()
    values["catalog"]["topics"] = ["care"]
    values["recall"]["affinity_mappings"] = [
        {"id": "care_known", "selector": {"topic": "care"}, "signal_id": "known",
         "target": "drive_affinity", "weight": .8},
        {"id": "care_unknown", "selector": {"topic": "care"}, "signal_id": "missing",
         "target": "drive_affinity", "weight": .5},
    ]
    document = document_template("PRIVATE_BODY_NOT_FOR_METADATA")
    statement = statement_template()
    statement.update(statement_id="state-1", content="PRIVATE_BODY_NOT_FOR_METADATA")
    statement["classification"]["topics"] = ["care"]
    document["statements"] = [statement]
    calculation = score_document(document, values, {"now": datetime(2026, 10, 20, tzinfo=timezone.utc),
        "purpose": "current", "query": "", "facts": {}, "signals": {
        "known": {"state": "known", "value": .5, "reason": None,
                  "evidence_refs": ["PRIVATE_SOURCE_REFERENCE"]},
        "missing": {"state": "unknown", "value": None, "reason": "signal_unknown"},
    }}, {"relevance": .8, "statement_selection": {
        "direct_ids": ["state-1"], "support_ids": [], "matches": {}}})
    computation["candidate_scores"][0]["computation"] = calculation
    evidence = score_evidence({"revision": "mc-partial", "values": values}, computation)
    row = evidence["candidates"][0]["computation"]["statements"][0]
    target = row["affinity"]["targets"]["drive_affinity"]
    assert target["state"] == "partial" and target["complete"] is False
    assert target["value"] == pytest.approx(.4)
    assert target["coverage"] == {"known": 1, "unknown": 1, "matched": 2, "ratio": .5}
    assert target["contributions"][1]["value"] is None
    assert target["contributions"][1]["reason"] == "signal_unknown"
    assert "PRIVATE" not in json.dumps(evidence)


def test_missing_or_oversized_configuration_never_claims_replayable_inputs():
    values, computation = example()
    values["formulas"] = [{"id": f"formula_{i}", "expression": "relevance + " * 100 + "relevance",
                            "weights": {}} for i in range(100)]
    evidence = score_evidence({"revision": "mc-large", "values": values}, computation)
    assert evidence["configuration"]["state"] == "oversized"
    assert evidence["configuration"]["values"] is None
    assert evidence["replayable"] is False
    assert score_evidence({"revision": "mc-missing"}, computation)["replayable"] is False


def test_bounded_projection_rejects_private_fields_nonfinite_and_prose():
    values, computation = example()
    values["features"][0].update(label="PRIVATE_LABEL", description="PRIVATE_DESCRIPTION", private="PRIVATE_FIELD")
    row = computation["candidate_scores"][0]
    row.update(content="PRIVATE_BODY", api_key="PRIVATE_KEY", raw_base_score=float("nan"))
    statement = row["computation"]["statements"][0]
    statement.update(match_evidence={"query": "PRIVATE_QUERY"}, private="PRIVATE_EXTRA")
    statement["features"].update(content="PRIVATE_BODY", api_key="PRIVATE_KEY", confidence=float("inf"))
    statement["calculation"]["features"]["relevance"] = "PRIVATE INPUT WITH WHITESPACE"
    computation["candidate_scores"] = [dict(copy.deepcopy(row), bucket_id=f"suppressed-{i}", selected=False,
        admitted=False) for i in range(CANDIDATE_LIMIT + 5)] + [row]
    evidence = score_evidence({"revision": "mc-safe", "values": values}, computation)
    text = json.dumps(evidence, allow_nan=False)
    assert "PRIVATE" not in text
    assert evidence["candidates"][0]["selected"] is True
    assert evidence["recorded_count"] <= CANDIDATE_LIMIT
    assert evidence["truncated_count"] == evidence["candidate_count"] - evidence["recorded_count"]
    assert evidence["unknown_fields"] and evidence["replayable"] is False
    assert len(text.encode()) <= EVIDENCE_BYTES
    assert evidence["candidates"][0]["raw_base_score"] is None
    assert score_evidence(None, None)["configuration"]["state"] == "unavailable"
    assert score_evidence({}, {"candidate_scores": ["invalid"]})["state"] == "unavailable"


def test_request_revision_prepare_and_retry_snapshots_do_not_cross(tmp_path):
    service = GatewayService.__new__(GatewayService)
    service.config = {}
    service.model_request_trace = ModelRequestTraceStore(tmp_path / "trace.sqlite")
    # Deliberately unrelated shared mutable slot must never supply this evidence.
    service._last_memory_configuration_use = {"revision": "WRONG", "computation": {}}
    expected = []
    for trace, revision, direct in [("trace-a", "mc-a1", True), ("trace-b", "mc-b", False), ("trace-a", "mc-a2", False)]:
        values, computation = example(direct=direct)
        token = REQUEST_CONFIGURATION.set({"service": id(service), "revision": revision, "values": values})
        try:
            evidence = request_score_evidence(service, {"configured_computation": computation})
            assert request_score_evidence(object(), {"configured_computation": computation})["configuration"]["state"] == "unavailable"
        finally:
            REQUEST_CONFIGURATION.reset(token)
        payload = {"messages": [], "_ombre_trace_context": {"trace_id": trace,
            "request_id": trace, "logical_request_id": trace + ":model:1"}}
        service._record_logical_trace(payload, {"prepare_timing_debug": {"total_ms": 2},
            "memory_score_evidence": evidence, "retrieval_runtime": {"fixture": True},
            "memory_detail_recall_debug": {}}, "fixture")
        attempt = service.model_request_trace.record_attempt(trace_id=trace, ordinal=1, provider="fixture",
            upstream="fixture", model="fixture")
        expected.append((trace, revision, attempt))
        # Evidence must be a durable snapshot, not a mutable alias to future config.
        values["time_profiles"][0]["weight"] = 999
    a = service.model_request_trace.get("trace-a", view="metadata")
    b = service.model_request_trace.get("trace-b", view="resolved")
    assert [p["memory_configuration_revision"] for p in a["preparations"]] == ["mc-a1", "mc-a2"]
    assert [p["attempt_ids"] for p in a["preparations"]] == [[expected[0][2]], [expected[2][2]]]
    assert b["preparations"][0]["memory_configuration_revision"] == "mc-b"
    assert a["metadata"]["memory_configuration_revision"] == "mc-a2"
    assert a["preparations"][0]["memory_score_evidence"]["configuration"]["values"]["time_profiles"][0]["weight"] == 1
    assert "WRONG" not in json.dumps(a)
    assert "PRIVATE_BODY" not in json.dumps(a)


def test_projection_has_no_io_or_additional_trace_write(tmp_path, monkeypatch):
    values, computation = example()
    service = GatewayService.__new__(GatewayService)
    service.config = {}
    store = service.model_request_trace = ModelRequestTraceStore(tmp_path / "trace.sqlite")
    with sqlite3.connect(store.path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        started = time.perf_counter()
        evidence = score_evidence({"revision": "mc-offline", "values": values}, computation)
        assert time.perf_counter() - started < 1
    calls = []
    original = store.begin_logical
    def record(metadata):
        calls.append(metadata)
        return original(metadata)
    monkeypatch.setattr(store, "begin_logical", record)
    monkeypatch.setattr(store, "update_metadata", lambda *a, **k: pytest.fail("new observation write"))
    service._record_logical_trace({"_ombre_trace_context": {"trace_id": "trace"}},
        {"prepare_timing_debug": {"total_ms": 1}, "memory_score_evidence": evidence,
         "retrieval_runtime": {"fixture": True}, "memory_detail_recall_debug": {}}, "fixture")
    assert len(calls) == 1
    assert store.get("trace")["preparations"][0]["memory_score_evidence"]["replayable"] is True


def test_real_natural_selection_records_selected_and_suppressed_inputs(tmp_path, monkeypatch):
    from test_memory_recall_gateway_r2 import (_authority_fixture, _base_config, _gateway_service,
        _install_source_http, _natural_candidate, _recall_input)
    _authority, view, _records, buckets = _authority_fixture(tmp_path, [
        ("bucket-low", "Synthetic low scoring record."), ("bucket-high", "Synthetic high scoring record.")])
    candidates = [_natural_candidate(buckets[0], score=.95, keyword_score=.2),
                  _natural_candidate(buckets[1], score=.2, keyword_score=.8)]
    _install_source_http(monkeypatch)
    config = _base_config(local_threshold=.5)
    service, _calls = _gateway_service(tmp_path, monkeypatch, view, buckets, candidates, config)
    request = _recall_input()
    values = copy.deepcopy(config)
    token = REQUEST_CONFIGURATION.set({"service": id(service), "revision": "mc-selection", "values": values})
    try:
        selected, _suppressed, _debug = asyncio.run(service._finish_natural_selection(
            request["q_context"], "synthetic-session", buckets, recall_input=request))
        evidence = request_score_evidence(service, request)
    finally:
        REQUEST_CONFIGURATION.reset(token)
    assert [item["id"] for item in selected] == ["bucket-high"]
    assert evidence["configuration"]["revision"] == "mc-selection"
    assert evidence["selection_policy"]["local_threshold"] == .5
    assert [row["bucket_id"] for row in evidence["candidates"]] == ["bucket-high", "bucket-low"]
    low = evidence["candidates"][1]
    assert low["admitted"] is False and low["admission_reason"] == "insufficient_contextual_relevance"
    assert low["raw_base_score"] == .95 and low["final_score"] == .2
    assert low["before_rerank"]["score"] == .2
    assert low["before_rerank"]["statements"][0]["features"]["base_relevance"] == .95
    assert low["computation"]["formula_id"] == "keyword_rank"
    assert "Synthetic low scoring record" not in json.dumps(evidence)
