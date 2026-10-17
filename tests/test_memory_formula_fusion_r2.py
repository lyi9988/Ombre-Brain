"""Focused synthetic checks for the R2 memory formula fusion contracts."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math

import pytest

from memory_l1_schema import (
    conditions_template,
    document_template,
    scene_template,
    statement_template,
    unknown,
    validate_l1_document,
)
from memory_l1_contract import make_text_anchor
from memory_recall_semantics import (
    apply_relations,
    preview_pipeline,
    resolve_document,
    score_document,
    temporal_features,
)
from memory_system_config import (
    ConfigurationError,
    read_overlay,
    revision,
    save_configuration,
    validate_values,
    values_from_config,
)


UTC = timezone.utc
LOCAL_TZ = timezone(timedelta(hours=8))
NOW = "2026-10-17T12:00:00+08:00"


def _time_profile(*, weight=2.0, half_life_days=30.0):
    return {
        "id": "test_decay",
        "selector": {},
        "function": "exponential",
        "half_life_days": half_life_days,
        "weight": weight,
    }


def _values_with_time_profile(*, weight=2.0, half_life_days=30.0):
    values = values_from_config({})
    values["time_profiles"] = [
        _time_profile(weight=weight, half_life_days=half_life_days)
    ]
    return validate_values(values)


def _claim_document(values, scenario=None, *, attributes=()):
    scenario = scenario or {}
    statement = statement_template()
    statement.update(statement_id="synthetic:formula-claim", content="Synthetic preview claim")
    for name, axis in (
        ("kind", "kinds"),
        ("stability", "stabilities"),
        ("modality", "modalities"),
        ("basis", "bases"),
    ):
        value = scenario.get(name)
        if value:
            statement["classification"][name] = unknown("known", value)
    statement["classification"]["topics"] = list(scenario.get("topics", []))

    event_time = scenario.get("event_time")
    if event_time:
        statement["time"]["event"]["start"] = unknown("known", event_time)
    if "conditions" in scenario:
        statement["conditions"] = deepcopy(scenario["conditions"])
    if "applicability" in scenario:
        statement["time"]["applicability"] = deepcopy(scenario["applicability"])
    statement["attributes"] = [deepcopy(item) for item in attributes]

    document = document_template("Synthetic preview only", catalog=values["catalog"])
    document["statements"] = [statement]
    validate_l1_document(document, catalog=values["catalog"])
    return document


def _context(scenario, tz=LOCAL_TZ):
    now = datetime.fromisoformat(scenario.get("now", NOW))
    target_time = scenario.get("target_time")
    return {
        "now": now.astimezone(tz),
        "target_time": datetime.fromisoformat(target_time).astimezone(tz) if target_time else None,
        "purpose": scenario.get("purpose", "current"),
        "facts": {},
    }


def _unknown_condition():
    value = conditions_template()
    value.update(state="unresolved", text="该条件目前尚未确认")
    return value


def test_time_factor_only_retains_or_reduces_base_and_does_not_invent_freshness():
    base = 0.24
    old_event = "2025-01-01"
    scenario = {"now": NOW, "event_time": old_event, "purpose": "current"}

    weighted_values = _values_with_time_profile(weight=4.0, half_life_days=10.0)
    weighted = score_document(
        _claim_document(weighted_values, scenario),
        weighted_values,
        _context(scenario),
        {"relevance": base},
    )["statements"][0]
    weighted_features = weighted["features"]
    assert weighted_features["freshness"] is not None
    assert weighted_features["freshness"] < 0.01
    assert weighted_features["time_factor"] == pytest.approx(
        1 - 4.0 / 5.0 * (1 - weighted_features["freshness"])
    )
    assert weighted_features["relevance"] == pytest.approx(base * weighted_features["time_factor"])
    assert weighted["score"] <= base

    zero_weight_values = _values_with_time_profile(weight=0.0, half_life_days=10.0)
    zero_weight = score_document(
        _claim_document(zero_weight_values, scenario),
        zero_weight_values,
        _context(scenario),
        {"relevance": base},
    )["statements"][0]
    assert zero_weight["features"]["freshness"] is not None
    assert zero_weight["features"]["time_factor"] == 1.0
    assert zero_weight["score"] == pytest.approx(base)

    unknown_scenario = {"now": NOW, "purpose": "current"}
    unknown_time = score_document(
        _claim_document(zero_weight_values, unknown_scenario),
        zero_weight_values,
        _context(unknown_scenario),
        {"relevance": base},
    )["statements"][0]
    assert unknown_time["features"]["time_weight"] == 0.0
    assert unknown_time["features"]["freshness"] is None
    assert unknown_time["features"]["time_factor"] is None
    assert unknown_time["features"]["relevance"] == base
    assert unknown_time["score"] == pytest.approx(base)

    no_profile_values = validate_values(values_from_config({}))
    no_profile = score_document(
        _claim_document(no_profile_values, unknown_scenario),
        no_profile_values,
        _context(unknown_scenario),
        {"relevance": base},
    )["statements"][0]
    assert no_profile["features"]["freshness"] is None
    assert no_profile["features"]["time_factor"] is None
    assert no_profile["score"] == pytest.approx(base)


def test_scoring_profiles_choose_most_specific_then_first_tie_and_reject_unknown_formula():
    values = values_from_config({})
    values["catalog"]["topics"] = ["work"]
    values["recall"]["scoring_profiles"] = [
        {"id": "purpose_only", "selector": {"query_purpose": "current"}, "formula_id": "final"},
        {"id": "kind_topic", "selector": {"kind": "experience", "topic": "work"}, "formula_id": "final"},
        {"id": "first_three", "selector": {"query_purpose": "current", "kind": "experience", "topic": "work"}, "formula_id": "final"},
        {"id": "second_three", "selector": {"query_purpose": "current", "kind": "experience", "stability": "phase"}, "formula_id": "final"},
        {"id": "all_four", "selector": {"query_purpose": "current", "kind": "experience", "topic": "work", "stability": "phase"}, "formula_id": "final"},
    ]
    values = validate_values(values)
    formulas = {item["id"] for item in values["formulas"]}
    assert {item["formula_id"] for item in values["recall"]["scoring_profiles"]} <= formulas

    scenario = {
        "now": NOW,
        "purpose": "current",
        "kind": "experience",
        "stability": "phase",
        "modality": "assertion",
        "basis": "record",
        "topics": ["work"],
    }
    document = _claim_document(values, scenario)
    result = score_document(document, values, _context(scenario), {"relevance": 0.4})["statements"][0]
    assert result["scoring_profile"] == "all_four"
    assert result["formula_id"] == "final"

    tied_values = deepcopy(values)
    tied_values["recall"]["scoring_profiles"] = tied_values["recall"]["scoring_profiles"][:-1]
    tied_values = validate_values(tied_values)
    tied = score_document(
        document,
        tied_values,
        _context(scenario),
        {"relevance": 0.4},
    )["statements"][0]
    assert tied["scoring_profile"] == "first_three"

    invalid_values = deepcopy(values)
    invalid_values["recall"]["scoring_profiles"][0]["formula_id"] = "missing_formula"
    with pytest.raises(ConfigurationError) as exc:
        validate_values(invalid_values)
    assert exc.value.code == "unknown_recall_formula"


@pytest.mark.parametrize("purpose", ["current", "history", "relationship", "autonomous"])
@pytest.mark.parametrize("event_time", ["2026-10-15", None])
def test_preview_pipeline_matches_direct_scoring_for_all_purposes_and_unknown_inputs(purpose, event_time):
    values = _values_with_time_profile(weight=2.5, half_life_days=20.0)
    scenario = {
        "now": NOW,
        "target_time": "2026-10-16T12:00:00+08:00",
        "purpose": purpose,
        "kind": "experience",
        "stability": "phase",
        "modality": "assertion",
        "basis": "record",
        "conditions": _unknown_condition(),
    }
    if event_time:
        scenario["event_time"] = event_time
    inputs = {"relevance": 0.42}

    previewed = preview_pipeline(values, inputs, scenario, tz=LOCAL_TZ)
    direct = score_document(
        _claim_document(values, scenario),
        values,
        _context(scenario),
        {"relevance": inputs["relevance"]},
    )
    preview_row = previewed["pipeline"]["statements"][0]
    direct_row = direct["statements"][0]

    assert preview_row["score"] == direct_row["score"]
    assert preview_row["features"] == direct_row["features"]
    assert preview_row["features"]["condition_match"] is None
    assert (preview_row["features"]["freshness"] is None) is (event_time is None)
    assert previewed["provider_requests"] == 0
    assert previewed["saved"] is False
    assert previewed["pipeline"]["context"]["simulation"] is True
    assert previewed["pipeline"]["context"]["source_state"] == "simulated_only"


def test_runtime_input_key_maps_and_custom_attributes_cannot_replace_reserved_fields():
    values = values_from_config({})
    values["features"].append({
        "id": "context_signal",
        "value_type": "number",
        "source": "runtime",
        "missing": "unknown",
        "input_key": "external_context_signal",
    })
    values["formulas"][0]["expression"] = "relevance + context_signal"
    values["catalog"]["attributes"].extend([
        {"id": "interest_affinity", "value_type": "number", "scope": "statement"},
        {"id": "drive_affinity", "value_type": "number", "scope": "statement"},
    ])
    values = validate_values(values)

    scenario = {"now": NOW, "purpose": "current", "kind": "experience", "basis": "record"}
    attributes = [
        {"definition_id": "interest_affinity", "value": unknown("known", 0.99), "evidence_refs": []},
        {"definition_id": "drive_affinity", "value": unknown("known", 0.88), "evidence_refs": []},
    ]
    document = _claim_document(values, scenario, attributes=attributes)
    direct = score_document(
        document,
        values,
        _context(scenario),
        {"relevance": 0.2, "external_context_signal": 0.35},
    )["statements"][0]
    assert direct["calculation"]["features"]["context_signal"] == pytest.approx(0.35)
    assert direct["score"] == pytest.approx(0.55)
    assert direct["features"]["attribute_interest_affinity"] == pytest.approx(0.99)
    assert direct["features"]["attribute_drive_affinity"] == pytest.approx(0.88)
    assert direct["features"]["interest_affinity"] is None
    assert direct["features"]["drive_affinity"] is None

    previewed = preview_pipeline(
        values,
        {"relevance": 0.2, "context_signal": 0.35},
        scenario,
        tz=LOCAL_TZ,
    )
    assert previewed["formulas"]["final"]["value"] == pytest.approx(0.55)
    assert previewed["pipeline"]["statements"][0]["score"] == pytest.approx(0.55)


def test_unbound_pending_interest_and_drive_inputs_are_unknown_not_zero():
    values = validate_values(values_from_config({}))
    scenario = {"now": NOW, "purpose": "autonomous"}
    row = score_document(
        _claim_document(values, scenario),
        values,
        _context(scenario),
        {"relevance": 0.3},
    )["statements"][0]
    assert row["features"]["interest_affinity"] is None
    assert row["features"]["drive_affinity"] is None


def test_legacy_recall_save_preserves_scoring_profiles_in_cas_runtime_overlay(tmp_path):
    base = {}
    path = tmp_path / "config.runtime.yaml"
    first_values = values_from_config(base)
    profile = {
        "id": "current_experience",
        "selector": {"query_purpose": "current", "kind": "experience"},
        "formula_id": "final",
    }
    first_values["recall"]["scoring_profiles"] = [profile]
    first, _ = save_configuration(str(path), base, {
        "expected_revision": revision(values_from_config(base)),
        "request_id": "formula-fusion-initial-save",
        "values": first_values,
    })
    assert first["validation"]["valid"] is True

    legacy_values = deepcopy(first["values"])
    legacy_values["recall"].pop("scoring_profiles")
    after_legacy, _ = save_configuration(str(path), base, {
        "expected_revision": first["revision"],
        "request_id": "formula-fusion-legacy-client-save",
        "values": legacy_values,
    })

    assert after_legacy["values"]["recall"]["scoring_profiles"] == [profile]
    persisted = read_overlay(path)
    assert persisted["memory_system"]["recall"]["scoring_profiles"] == [profile]


def test_exact_exponential_settlement_composes_across_segments_without_external_runtime():
    initial, target = -0.35, 0.72
    half_life = 17.0
    rate = math.log(2.0) / half_life
    first_days, second_days = 3.25, 11.5

    def settle(value, duration):
        return target + (value - target) * math.exp(-rate * duration)

    whole = settle(initial, first_days + second_days)
    split = settle(settle(initial, first_days), second_days)
    assert split == pytest.approx(whole, rel=1e-13, abs=1e-13)

    values = _values_with_time_profile(weight=1.0, half_life_days=half_life)
    statement = _claim_document(values, {"event_time": "2026-10-01T00:00:00+00:00"})["statements"][0]
    event = datetime.fromisoformat("2026-10-01T00:00:00+00:00")
    first = temporal_features(statement, {"now": event + timedelta(days=first_days), "purpose": "current"}, values["time_profiles"])["freshness"]
    second_statement = deepcopy(statement)
    second_statement["time"]["event"]["start"] = unknown("known", (event + timedelta(days=first_days)).isoformat())
    second = temporal_features(second_statement, {"now": event + timedelta(days=first_days + second_days), "purpose": "current"}, values["time_profiles"])["freshness"]
    combined = temporal_features(statement, {"now": event + timedelta(days=first_days + second_days), "purpose": "current"}, values["time_profiles"])["freshness"]
    assert first * second == pytest.approx(combined, rel=1e-13, abs=1e-13)


def _relation_statement(statement_id, binding_id, *, start="2026-01-01", end="2027-01-01"):
    statement = statement_template()
    statement.update(
        statement_id=statement_id,
        content="Synthetic explicitly related claim",
        evidence_refs=[binding_id],
    )
    statement["classification"].update(
        kind=unknown("known", "fact"),
        stability=unknown("known", "phase"),
        modality=unknown("known", "assertion"),
        basis=unknown("known", "record"),
    )
    statement["time"]["applicability"].update(
        temporal_scope="time_range",
        start=unknown("known", start),
        end=unknown("known", end),
    )
    statement["conditions"] = conditions_template()
    statement["conditions"]["state"] = "not_applicable"
    return statement


def _relation_binding(binding_id, source_id):
    text = "Synthetic explicit relation evidence."
    return {
        "binding_id": binding_id,
        "source_namespace": "synthetic",
        "source_id": source_id,
        "source_revision": "source-r1",
        "author_ref": None,
        "speaker_ref": None,
        "anchors": [make_text_anchor(
            text,
            target_namespace="l0",
            target_id=source_id,
            target_revision="source-r1",
            text_view_id="plain",
            start=0,
            end=len(text),
        )],
    }


def _apply_relation_case(
    *,
    confirmation_state="confirmed",
    origin="owner",
    purpose="current",
    relation_source_state="available",
    left_scope=("2026-01-01", "2027-01-01"),
    right_scope=("2026-01-01", "2027-01-01"),
    missing_counterpart=False,
    reverse_replacement=False,
    scene_target=False,
):
    values = values_from_config({})
    document = document_template("Synthetic explicit relation test", catalog=values["catalog"])
    left_id, right_id = "synthetic:from", "synthetic:to"
    left_binding, right_binding, relation_binding = "source_from", "source_to", "source_relation"
    document["statements"] = [
        _relation_statement(left_id, left_binding, start=left_scope[0], end=left_scope[1]),
        _relation_statement(right_id, right_binding, start=right_scope[0], end=right_scope[1]),
    ]
    document["evidence_bindings"] = [
        _relation_binding(left_binding, "source-from"),
        _relation_binding(right_binding, "source-to"),
        _relation_binding(relation_binding, "source-relation"),
    ]
    right_endpoint = (
        {"memory_id": "missing-memory", "revision": 1, "object_type": "statement", "object_id": "missing:counterpart"}
        if missing_counterpart
        else {"memory_id": None, "revision": "this_revision", "object_type": "statement", "object_id": right_id}
    )
    document["relations"] = [{
        "relation_id": "synthetic:relation",
        "from": {"memory_id": None, "revision": "this_revision", "object_type": "statement", "object_id": left_id},
        "to": right_endpoint,
        "kind": "supersedes",
        "evidence_refs": [relation_binding],
        "origin": origin,
        "confirmation_state": confirmation_state,
    }]
    if reverse_replacement:
        reverse = deepcopy(document["relations"][0])
        reverse["relation_id"] = "synthetic:reverse-relation"
        reverse["from"], reverse["to"] = reverse["to"], reverse["from"]
        document["relations"].append(reverse)
    if scene_target:
        scene = scene_template(values["catalog"])
        scene.update(scene_id="synthetic:scope-scene", title="Targeted scope", statement_refs=[right_id])
        document["scenes"] = [scene]
        document["relations"][0]["to"].update(object_type="scene", object_id=scene["scene_id"])
    validate_l1_document(document, catalog=values["catalog"])

    source_states = {
        left_binding: "available",
        right_binding: "available",
        relation_binding: relation_source_state,
    }
    context = _context({"now": NOW, "purpose": purpose})
    items = resolve_document(document, context, source_states=source_states)
    records = {"bundle-1": {
        "state": "l1",
        "document": document,
        "memory_id": "memory-1",
        "revision": 1,
    }}
    usages = {"bundle-1": {
        "state": "l1",
        "items": items,
        "source_states": source_states,
    }}
    applied = apply_relations(records, usages, context)
    return {item["statement_id"]: item for item in applied["bundle-1"]["items"]}


def test_apply_relations_confirmed_owner_supersedes_only_current_matching_scope():
    matched = _apply_relation_case()
    assert matched["synthetic:from"]["usage"] == "current_basis"
    assert matched["synthetic:to"]["usage"] == "historical_evidence"
    assert "confirmed_revision_replacement" in matched["synthetic:to"]["reason_codes"]

    mismatched = _apply_relation_case(left_scope=("2027-01-01", "2028-01-01"))
    assert mismatched["synthetic:from"]["usage"] == "historical_evidence"
    assert mismatched["synthetic:to"]["usage"] == "current_basis"
    assert "confirmed_revision_replacement" not in mismatched["synthetic:to"]["reason_codes"]


def test_apply_relations_proposed_supersedes_remains_unresolved():
    result = _apply_relation_case(confirmation_state="proposed", origin="model")
    assert result["synthetic:to"]["usage"] == "unresolved_evidence"
    assert "proposed_relation_unconfirmed" in result["synthetic:to"]["reason_codes"]
    assert "confirmed_revision_replacement" not in result["synthetic:to"]["reason_codes"]


def test_apply_relations_unavailable_relation_source_does_not_confirm_replacement():
    result = _apply_relation_case(relation_source_state="unavailable")
    assert result["synthetic:to"]["usage"] == "unresolved_evidence"
    assert "related_scope_requires_confirmation" in result["synthetic:to"]["reason_codes"]
    assert "confirmed_revision_replacement" not in result["synthetic:to"]["reason_codes"]


def test_apply_relations_history_query_preserves_historical_evidence():
    result = _apply_relation_case(purpose="history")
    assert result["synthetic:from"]["usage"] == "historical_evidence"
    assert result["synthetic:to"]["usage"] == "historical_evidence"
    assert "historical_query" in result["synthetic:to"]["reason_codes"]
    assert "confirmed_revision_replacement" not in result["synthetic:to"]["reason_codes"]


def test_apply_relations_missing_counterpart_remains_unresolved():
    result = _apply_relation_case(missing_counterpart=True)
    assert result["synthetic:from"]["usage"] == "unresolved_evidence"
    assert "related_revision_not_source_verified" in result["synthetic:from"]["reason_codes"]
    assert "confirmed_revision_replacement" not in result["synthetic:from"]["reason_codes"]


def test_mutual_replacements_remain_unresolved_instead_of_hiding_both_as_history():
    result = _apply_relation_case(reverse_replacement=True)
    assert {use["usage"] for use in result.values()} == {"unresolved_evidence"}
    assert all("replacement_cycle_unresolved" in use["reason_codes"] for use in result.values())


def test_scene_replacement_applies_only_to_its_declared_statement_scope():
    result = _apply_relation_case(scene_target=True)
    assert result["synthetic:from"]["usage"] == "current_basis"
    assert result["synthetic:to"]["usage"] == "historical_evidence"
    assert "confirmed_revision_replacement" in result["synthetic:to"]["reason_codes"]
