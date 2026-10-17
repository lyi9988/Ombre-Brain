import copy

import pytest

from memory_system_config import (
    ConfigurationError,
    configuration,
    preview,
    revision,
    save_configuration,
    validate_values,
    values_from_config,
)


def signal_binding(**overrides):
    return {
        "id": "health_interest",
        "label": "健康兴趣",
        "producer": "aiz.interests",
        "key": "health",
        "max_age_seconds": 3600,
        **overrides,
    }


def values_with_affinity():
    values = values_from_config({})
    values["catalog"]["topics"] = ["health"]
    values["catalog"]["definitions"].append({
        "axis": "topics", "id": "health", "label": "健康", "description": "",
        "aliases": ["健康生活"], "examples": [], "parent_id": None,
    })
    values["recall"]["signal_bindings"] = [signal_binding()]
    values["recall"]["affinity_mappings"] = [{
        "id": "health_interest", "selector": {"topic": "健康生活", "domain": "life"},
        "signal_id": "health_interest", "target": "interest_affinity", "weight": 0.6,
    }]
    values["recall"]["affinity_aggregation"] = "mean"
    return values


def test_affinity_defaults_are_empty_and_topic_aliases_normalize_to_catalog_ids():
    defaults = values_from_config({})["recall"]
    assert defaults["signal_bindings"] == []
    assert defaults["affinity_mappings"] == []
    assert defaults["affinity_aggregation"] == "max"

    values = values_with_affinity()
    values["recall"]["signal_bindings"].append(signal_binding(
        id="health_interest_secondary", max_age_seconds=None,
    ))
    validated = validate_values(values)
    assert validated["recall"]["affinity_mappings"][0]["selector"] == {
        "topic": "health", "domain": "life",
    }
    assert validated["recall"]["affinity_aggregation"] == "mean"
    assert [row["key"] for row in validated["recall"]["signal_bindings"]] == ["health", "health"]


@pytest.mark.parametrize(("field", "value", "code"), [
    ("producer", "not.a.producer", "unknown_signal_producer"),
    ("max_age_seconds", 0, "invalid_signal_max_age"),
    ("max_age_seconds", float("inf"), "finite_number_required"),
    ("id", "x" * 57, "signal_id_too_long"),
])
def test_invalid_signal_binding_metadata_is_rejected(field, value, code):
    values = values_with_affinity()
    values["recall"]["signal_bindings"][0][field] = value
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == code


def test_signal_bindings_allow_shared_producer_key_but_reject_duplicate_ids():
    values = values_with_affinity()
    values["recall"]["signal_bindings"].append(signal_binding(
        id="another_interest", max_age_seconds=None,
    ))
    assert len(validate_values(values)["recall"]["signal_bindings"]) == 2

    values["recall"]["signal_bindings"][1]["id"] = "health_interest"
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == "duplicate_identifier"


@pytest.mark.parametrize(("field", "value", "code"), [
    ("signal_id", "missing", "unknown_affinity_signal"),
    ("target", "relationship", "invalid_affinity_target"),
    ("weight", 1.01, "affinity_weight_out_of_range"),
])
def test_invalid_affinity_mapping_reference_target_or_weight_is_rejected(field, value, code):
    values = values_with_affinity()
    values["recall"]["affinity_mappings"][0][field] = value
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == code


def test_affinity_mapping_rejects_freeform_topic_and_noncanonical_domain():
    values = values_with_affinity()
    values["recall"]["affinity_mappings"][0]["selector"]["topic"] = "自由标签"
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == "unknown_affinity_topic"

    values = values_with_affinity()
    values["recall"]["affinity_mappings"][0]["selector"]["domain"] = "生活"
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == "unknown_affinity_domain"


def test_affinity_aggregation_and_mapping_ids_are_validated():
    values = values_with_affinity()
    values["recall"]["affinity_aggregation"] = "sum"
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == "invalid_affinity_aggregation"

    values = values_with_affinity()
    values["recall"]["affinity_mappings"].append(copy.deepcopy(values["recall"]["affinity_mappings"][0]))
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == "duplicate_identifier"

    values = values_with_affinity()
    values["recall"]["affinity_mappings"][0]["selector"] = {}
    with pytest.raises(ConfigurationError) as error:
        validate_values(values)
    assert error.value.code == "empty_affinity_selector"


def test_conditional_runtime_bindings_publish_metadata_without_claiming_live_availability():
    result = configuration({"memory_system": {"recall": {
        "signal_bindings": [signal_binding()],
    }}})
    inputs = {row["id"]: row for row in result["bindings"]["runtime_inputs"]}
    assert inputs["interest_affinity"]["availability"] == "conditional"
    assert inputs["drive_affinity"]["availability"] == "conditional"
    assert inputs["goal_alignment"]["availability"] == "conditional"
    assert inputs["feedback_score"]["availability"] == "pending"
    assert inputs["repeat_penalty"]["availability"] == "pending"
    dynamic = inputs["signal_health_interest"]
    assert dynamic["availability"] == "conditional"
    assert dynamic["producer"] == "aiz.interests"
    assert dynamic["key"] == "health"
    assert dynamic["max_age_seconds"] == 3600
    assert "仍需真实producer快照" in dynamic["availability_note"]
    assert result["application"]["features"]["pending_inputs"] == ["feedback_score", "repeat_penalty"]
    assert result["application"]["affinity_config"]["revision"] == result["revision"]


def test_missing_signal_input_stays_unknown_in_owner_preview():
    values = values_from_config({})
    values["recall"]["signal_bindings"] = [signal_binding()]
    values["features"] = [{
        "id": "interest_score", "value_type": "number", "source": "runtime",
        "missing": "unknown", "input_key": "signal_health_interest",
    }]
    values["formulas"] = [{"id": "final", "expression": "interest_score", "weights": {}}]
    result = preview(values, {})
    assert result["features"]["interest_score"] is None
    assert result["formulas"]["final"] == {"value": None, "state": "unknown"}
    assert result["provider_requests"] == 0 and result["saved"] is False


def test_old_client_save_and_readback_preserve_new_affinity_fields(tmp_path):
    base = {}
    values = values_with_affinity()
    first, _ = save_configuration(tmp_path / "runtime.yaml", base, {
        "expected_revision": configuration(base)["revision"],
        "request_id": "affinity-owner-save",
        "values": values,
    })
    assert first["values"]["recall"]["affinity_mappings"][0]["selector"]["topic"] == "health"

    old_client = copy.deepcopy(first["values"])
    for key in ("signal_bindings", "affinity_mappings", "affinity_aggregation"):
        old_client["recall"].pop(key)
    second, readback = save_configuration(tmp_path / "runtime.yaml", base, {
        "expected_revision": first["revision"],
        "request_id": "affinity-old-client-save",
        "values": old_client,
    })
    assert second["values"]["recall"]["signal_bindings"] == first["values"]["recall"]["signal_bindings"]
    assert second["values"]["recall"]["affinity_mappings"] == first["values"]["recall"]["affinity_mappings"]
    assert second["values"]["recall"]["affinity_aggregation"] == "mean"
    assert readback["memory_system"]["recall"] == second["values"]["recall"]
    assert revision(second["values"]) == second["revision"]
