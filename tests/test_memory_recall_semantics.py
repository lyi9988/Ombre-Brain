from datetime import datetime, timedelta, timezone

import memory_recall_semantics as semantics
from memory_system_config import preview, values_from_config


def known(value):
    return {"state": "known", "value": value, "evidence_refs": ["binding:synthetic"]}


def unknown():
    return {"state": "unknown", "value": None, "evidence_refs": []}


def unbounded():
    return {"state": "unbounded", "value": None, "evidence_refs": []}


def predicate(op, field, value):
    return {"op": op, "field": field, "value": value}


def statement(statement_id, *, kind="fact", stability="phase", modality="assertion",
              basis="record", topics=(), condition=None, applicability=None,
              event_start=None, event_end=None, content=None):
    if applicability is None:
        applicability = {
            "start": unbounded(), "end": unbounded(), "temporal_scope": "unknown"
        }
    event = {}
    if event_start is not None:
        event["start"] = known(event_start)
    if event_end is not None:
        event["end"] = known(event_end)
    return {
        "statement_id": statement_id,
        "statement_revision_id": f"revision:{statement_id}",
        "content": content or f"合成陈述 {statement_id}",
        "subject_ref": "entity:user",
        "classification": {
            "kind": known(kind),
            "topics": list(topics),
            "stability": known(stability),
            "modality": known(modality),
            "basis": known(basis),
        },
        "conditions": condition if condition is not None else {"state": "not_applicable"},
        "time": {"event": event, "applicability": applicability},
        "evidence_refs": ["binding:synthetic"],
    }


def profile(profile_id, selector, function="none", **parameters):
    return {"id": profile_id, "selector": selector, "function": function, **parameters}


def test_address_change_keeps_old_address_historical_and_new_address_current():
    now = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
    old_address = statement(
        "old-address", kind="fact", basis="record",
        applicability={"start": known("2019-01-01"), "end": known("2024-01-01")},
        content="旧住址是青石路。",
    )
    current_address = statement(
        "current-address", kind="fact", basis="record",
        applicability={"start": known("2024-01-01"), "end": unbounded()},
        content="现住址是白桦街。",
    )

    old = semantics.resolve_statement(old_address, {"now": now}, source_state="available")
    current = semantics.resolve_statement(current_address, {"now": now}, source_state="available")

    assert old["usage"] == "historical_evidence"
    assert old["scope_match"] is False
    assert current["usage"] == "current_basis"
    assert current["scope_match"] is True


def test_stable_preference_survives_age_while_temporary_request_stays_request_local():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    profiles = [profile(
        "preference-decay", {"kind": "preference", "stability": "stable"},
        "exponential", half_life_days=10, review_after_hours=1, weight=0.8,
    )]
    old_long_term_preference = statement(
        "long-term-preference", kind="preference", stability="stable",
        modality="assertion", basis="direct_expression", event_start="2020-01-01",
    )
    one_time_request = statement(
        "one-time-request", kind="preference", stability="phase",
        modality="request", basis="direct_expression",
    )

    long_term = semantics.resolve_statement(
        old_long_term_preference, {"now": now}, profiles, source_state="available"
    )
    request = semantics.resolve_statement(
        one_time_request, {"now": now}, profiles, source_state="available"
    )

    assert long_term["features"]["event_age_days"] > 2000
    assert long_term["features"]["freshness"] < 0.001
    assert long_term["features"]["review_due"] is True
    assert long_term["usage"] == "conditional_basis"
    assert request["usage"] == "relationship_reference"


def test_old_symptom_is_not_current_and_review_due_does_not_resolve_unknown_state():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    symptom = statement(
        "old-symptom", kind="state", stability="phase", basis="record",
        event_start="2026-10-01",
        condition={"state": "known", "predicate": predicate("eq", "recovered", True)},
    )
    selector = profile(
        "symptom-review", {"kind": "state"}, "exponential",
        half_life_days=30, review_after_hours=1, weight=0.6,
    )

    result = semantics.resolve_statement(
        symptom, {"now": now, "facts": {"recovered": unknown()}},
        [selector], source_state="available",
    )

    assert result["features"]["review_due"] is True
    assert result["condition_match"] is None
    assert result["usage"] == "historical_evidence"
    assert result["reason_codes"] == ["state_needs_reconfirmation"]


def test_plans_requests_and_commitments_are_not_evidence_of_completion():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    for modality in ("request", "plan", "commitment"):
        item = statement(
            f"expression-{modality}", kind="intention", stability="phase",
            modality=modality, basis="direct_expression",
        )
        result = semantics.resolve_statement(item, {"now": now}, source_state="available")
        assert result["usage"] == "relationship_reference"
        assert result["reason_codes"] == ["expression_not_execution_or_completion"]


def test_unknown_conditions_and_exceptions_keep_three_valued_logic():
    facts = {"night": known(True), "deadline_met": unknown()}
    assert semantics.predicate({"op": "all", "args": [
        predicate("eq", "night", True), predicate("eq", "deadline_met", True),
    ]}, facts) is None
    assert semantics.predicate({"op": "any", "args": [
        predicate("eq", "night", True), predicate("eq", "deadline_met", True),
    ]}, facts) is True
    assert semantics.predicate({"op": "not", "args": [
        predicate("eq", "deadline_met", True),
    ]}, facts) is None

    conditional = {
        "state": "known",
        "predicate": predicate("eq", "night", True),
        "exceptions": [{"state": "known", "predicate": predicate("eq", "deadline_met", True)}],
    }
    assert semantics.conditions(conditional, facts) is None
    assert semantics.conditions({"state": "known", "predicate": predicate("eq", "night", True),
                                 "exceptions": [{"state": "not_applicable"}]}, facts) is False


def test_changed_unverified_revoked_and_unavailable_sources_cannot_be_current():
    item = statement("source-qualification", kind="fact", basis="record")
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)

    available = semantics.resolve_statement(item, {"now": now}, source_state="available")
    assert available["usage"] == "current_basis"
    for source_state in ("changed", "unverified"):
        result = semantics.resolve_statement(item, {"now": now}, source_state=source_state)
        assert result["usage"] == "unresolved_evidence"
        assert result["usage"] != "current_basis"
    for source_state in ("revoked", "unavailable"):
        result = semantics.resolve_statement(item, {"now": now}, source_state=source_state)
        assert result["usage"] == "excluded"


def test_partial_date_precision_and_timezone_offsets_are_preserved():
    plus_eight = timezone(timedelta(hours=8))
    year = semantics.interval(known("2025"), plus_eight)
    month = semantics.interval(known("2024-02"), plus_eight)
    day = semantics.interval(known("2026-10-17"), plus_eight)
    instant = semantics.interval(known("2026-10-17T02:00:00+08:00"), timezone.utc)

    assert year[0] == datetime(2025, 1, 1, tzinfo=plus_eight)
    assert year[1] - year[0] == timedelta(days=365)
    assert month[1] - month[0] == timedelta(days=29)
    assert day == (datetime(2026, 10, 17, tzinfo=plus_eight),
                   datetime(2026, 10, 18, tzinfo=plus_eight))
    assert instant[0].utcoffset() == timedelta(hours=8)
    assert instant[1] - instant[0] == timedelta(microseconds=1)


def test_missing_temporal_evidence_and_missing_score_inputs_remain_unknown():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    item = statement("no-date", kind="state")
    temporal = semantics.temporal_features(
        item, {"now": now}, [profile("state-time", {"kind": "state"}, "exponential",
                                    half_life_days=30, weight=0.5)]
    )
    assert temporal["event_age_days"] is None
    assert temporal["freshness"] is None

    values = values_from_config({})
    result = semantics.evaluate_scores(values, {})
    assert result == preview(values, {})
    assert result["features"]["relevance"] is None
    assert result["formulas"]["final"] == {"value": None, "state": "unknown"}


def test_profile_selector_specificity_and_history_purpose_change_profile_and_usage():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    item = statement("purpose-sensitive", kind="state", stability="phase",
                    basis="direct_expression", event_start="2026-10-01", topics=("health",))
    profiles = [
        profile("broad", {"kind": "state"}, "none", weight=0.9),
        profile("current-specific", {"kind": "state", "query_purpose": "current"},
                "exponential", half_life_days=10, weight=0.8),
        profile("history-specific", {"kind": "state", "query_purpose": "history"},
                "none", weight=0.2),
    ]
    current_profile = semantics.profile_for(item, profiles, "current")
    history_profile = semantics.profile_for(item, profiles, "history")
    current = semantics.resolve_statement(
        item, {"now": now, "purpose": "current"}, profiles, source_state="available"
    )
    history = semantics.resolve_statement(
        item, {"now": now, "purpose": "history"}, profiles, source_state="available"
    )

    assert current_profile["id"] == "current-specific"
    assert history_profile["id"] == "history-specific"
    assert current["features"]["time_profile"] == "current-specific"
    assert history["features"]["time_profile"] == "history-specific"
    assert current["features"]["freshness"] < 1
    assert history["features"]["freshness"] == 1.0
    assert current["usage"] == "current_basis"
    assert history["usage"] == "historical_evidence"


def test_statements_in_one_memory_keep_separate_usage_results():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    stable_preference = statement(
        "s-preference", kind="preference", stability="stable",
        modality="assertion", basis="direct_expression",
    )
    episode = statement(
        "s-episode", kind="experience", stability="episode",
        modality="assertion", basis="record", event_start="2026-10-01",
    )
    memory = {"memory_id": "memory:synthetic", "statements": [stable_preference, episode]}

    results = [semantics.resolve_statement(row, {"now": now}, source_state="available")
               for row in memory["statements"]]

    assert [row["statement_id"] for row in results] == ["s-preference", "s-episode"]
    assert [row["usage"] for row in results] == ["conditional_basis", "historical_evidence"]


def test_lexical_features_match_declared_terms_without_general_semantic_inference():
    document = {
        "retrieval": {"keywords": ["灯光"], "tags": ["晚间流程"]},
        "entities": [{"name": "青石路住所", "aliases": ["旧居"]}],
        "statements": [{"classification": {"kind": known("preference"),
                                             "topics": ["routine"]}}],
    }
    catalog = {"definitions": [
        {"axis": "kinds", "id": "preference", "label": "偏好", "aliases": ["喜好"]},
        {"axis": "topics", "id": "routine", "label": "日常作息", "aliases": ["作息规律"]},
    ]}

    alias_match = semantics.lexical_features("我说的旧居", document, catalog)
    tag_match = semantics.lexical_features("请想起晚间流程", document, catalog)
    declared_label_match = semantics.lexical_features("这是我的作息规律", document, catalog)
    semantic_paraphrase = semantics.lexical_features("我觉得熬夜更舒服", document, catalog)

    assert alias_match["entity_match"] == 1.0
    assert tag_match["tag_match"] == 1.0
    assert declared_label_match["topic_match"] == 1.0
    assert all(value == 0.0 for value in semantic_paraphrase.values())


def test_runtime_score_evaluator_matches_shared_owner_preview_for_known_input():
    values = values_from_config({})
    inputs = {"relevance": 0.7}

    assert semantics.evaluate_scores(values, inputs) == preview(values, inputs)
