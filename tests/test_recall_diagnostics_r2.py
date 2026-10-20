"""Native picker and trace persistence; fixture IO only, no provider calls."""
import copy
import ast
import hashlib
import json
import random
import subprocess

import pytest

from gateway import GatewayService
from memory_recall_runtime import source_diagnostic_trace
from model_request_trace import ModelRequestTraceStore
from recall_rerank_window import admission_evidence, select_natural_window
from recall_rerank_window import explicit_query_parts


def row(identifier, score, rerank, part=0):
    return {"bucket": {"id": identifier, "content": "PRIVATE_BODY"},
            "score": score, "_raw_base_score": score - .1, "_raw_fused_score": score,
            "rerank_score": rerank, "keyword_score": .8, "semantic_score": .7,
            "_rerank_window": {"query_part": part, "priority_tier": [True] * 4},
            "_configured_score": {"window_score": score},
            "admission_reason": "strong_rerank"}


def service():
    instance = GatewayService.__new__(GatewayService)
    instance.inject_max_cards = 2
    instance.first_card_min_score = .5
    instance.second_card_min_score = .4
    instance.second_card_relative_score = .8
    # No axis or reliable-signal overrides apply to this synthetic score case.
    instance._recall_query_plan = lambda query: None
    return instance


def evidence(pool, chosen, **kw):
    return admission_evidence(pool, pool, chosen, semantic_threshold=.72,
        rerank_threshold=.6, selection_limits={"max_cards": 2}, **kw)


def test_native_picker_does_not_rank_by_raw_rerank_or_force_admitted_target():
    instance = service()
    pool = [row("first", .95, .72), row("second", .90, .70), row("target", .70, .686)]
    before = copy.deepcopy(pool)
    observed = {}
    selected = instance._pick_dynamic_cards(pool, query="One ordinary question?", diagnostics=observed)
    assert selected == pool[:2]
    trace = evidence(pool, selected, selection_diagnostics=observed)
    assert trace["candidates"][2]["selection_state"] == "admitted_not_selected"
    assert trace["candidates"][2]["final_score"] == .7
    assert trace["candidates"][2]["admitted_rank"] == 3
    assert trace["candidates"][2]["selection_reason"] == "card_limit_reached"
    assert pool == before


def test_native_multiquestion_picker_and_window_preserve_coverage():
    instance = service()
    pool = [row("first", .95, .72, 1), row("second", .90, .70, 1),
            row("other-part", .70, .686, 0)]
    observed = {}
    chosen = instance._pick_dynamic_cards(pool, query="Who is Alice? What should search provide?",
                                         diagnostics=observed)
    assert chosen == [pool[0], pool[2]]
    assert observed["decisions"]["second"]["reason"] == "multiquestion_part_lower_rank"
    assert observed["mode"] == "multiquestion_coverage"
    selected, notes, _ = select_natural_window(pool, 2,
        [(True, True, True, True, -item["score"]) for item in pool],
        query_view_scores=[{1: .95}, {1: .9}, {0: .7}])
    assert selected == {0, 2}
    assert notes[2]["query_part"] == 0 and notes[2]["priority_tier"] == [True] * 4


def test_diagnostic_survives_real_sqlite_trace_update_and_owner_read(tmp_path):
    store = ModelRequestTraceStore(tmp_path / "trace.sqlite3")
    store.begin_logical({"trace_id": "diagnostic", "conversation_id": "synthetic"})
    pool = [row("first", .95, .72), row("target", .70, .686)]
    source = source_diagnostic_trace({"state": "ok", "failures": [{
        "binding_id_sha256": hashlib.sha256(b"synthetic-binding").hexdigest(),
        "detail": {"failure_kind": "http_status", "http_status": 503,
                   "body": "PRIVATE_BODY", "url": "PRIVATE_URL"},
    }]})
    metadata = {"candidate_evidence": evidence(pool, [pool[0]]), "source_diagnostic": source}
    store.update_metadata("diagnostic", metadata)
    # Reopen actual persistence rather than asserting the in-memory dict alone.
    reopened = ModelRequestTraceStore(tmp_path / "trace.sqlite3")
    detail = reopened.get("diagnostic")
    serialized = json.dumps(detail, ensure_ascii=False)
    assert "PRIVATE_" not in serialized
    assert detail["metadata"]["source_diagnostic"] == source
    assert detail["metadata"]["candidate_evidence"] == metadata["candidate_evidence"]


@pytest.mark.parametrize("malformed", [[], {}, ["PRIVATE_BODY"], {"PRIVATE_URL": "value"}])
def test_untrusted_diagnostic_enums_do_not_raise_or_leak(malformed):
    diagnostic = source_diagnostic_trace({"state": malformed, "failures": [{
        "binding_id_sha256": "a" * 64, "detail": {"failure_kind": malformed,
        "reason_code": malformed, "field_changes": [{"field": malformed}]},
    }]})
    assert diagnostic["state"] == "unknown"
    assert diagnostic["failures"][0]["detail"]["failure_kind"] == "unknown"
    assert "PRIVATE_" not in json.dumps(diagnostic)


def test_picker_diagnostics_preserve_baseline_choices_across_scores_limits_and_queries():
    # Extract only the two original picker methods from the verified Git parent.
    # All helper methods still use the complete native Gateway implementation.
    baseline = subprocess.check_output(["git", "show",
        "299ca3bdea1f86e238d20c6cc8270496bd097946:gateway.py"]).decode("utf-8")
    klass = next(node for node in ast.parse(baseline).body
                 if isinstance(node, ast.ClassDef) and node.name == "GatewayService")
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_pick_dynamic_cards", "_pick_dynamic_cards_by_score"}]
    namespace = {"explicit_query_parts": explicit_query_parts}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "verified-parent-picker", "exec"), namespace)
    rng = random.Random(20261020)
    for count in (0, 1, 3, 12):
        for limit in (0, 1, 2, 4):
            for query in ("One ordinary question?", "Who is Alice? What should search provide?"):
                pool = [row(str(i), rng.random(), rng.random(), i % 3) for i in range(count)]
                for i, item in enumerate(pool):
                    item["matched_query_terms"] = ["shared", str(i % 4)] if i % 2 else []
                pool.sort(key=lambda item: item["score"], reverse=True)
                old, new = service(), service()
                old.inject_max_cards = new.inject_max_cards = limit
                old._pick_dynamic_cards_by_score = namespace["_pick_dynamic_cards_by_score"].__get__(old)
                expected = namespace["_pick_dynamic_cards"](old, copy.deepcopy(pool), query=query)
                before, observed = copy.deepcopy(pool), {}
                actual = new._pick_dynamic_cards(pool, query=query, diagnostics=observed)
                assert [item["bucket"]["id"] for item in actual] == [item["bucket"]["id"] for item in expected]
                assert pool == before


def test_actual_threshold_branch_is_recorded_without_weakening_it():
    instance = service()
    pool = [row("first", .9, .8), row("relative", .6, .99), row("absolute", .2, .99)]
    observed = {}
    selected = instance._pick_dynamic_cards(pool, query="One ordinary question?", diagnostics=observed)
    assert selected == pool[:1]
    assert observed["decisions"]["relative"]["reason"] == "below_relative_score"
    assert observed["decisions"]["absolute"]["reason"] == "below_second_card_threshold"


def test_hierarchy_filter_is_distinct_from_initial_admission():
    pool = [row("kept", .9, .8), row("suppressed", .8, .7)]
    trace = admission_evidence(pool, pool[:1], pool[:1], semantic_threshold=.72,
        rerank_threshold=.6, hierarchy_suppressed=["suppressed"],
        admitted_before_hierarchy={"kept", "suppressed"})
    suppressed = trace["candidates"][1]
    assert suppressed["admitted_before_hierarchy"] is True
    assert suppressed["admitted"] is False and suppressed["selected"] is False
    assert suppressed["selection_state"] == "hierarchy_suppressed"
