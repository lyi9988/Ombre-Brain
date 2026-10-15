import copy
import json

import pytest

from memory_system_config import (ConfigurationError, configuration, preview, read_overlay,
                                  save_configuration, update_overlay, validate_values)


def values():
    return configuration({})["values"]


def test_configuration_keeps_configured_limits_and_exposes_binding_and_pending_consumers():
    result = configuration({"gateway": {"inject_max_cards": 6, "diffusion_inject_max_items": 4}})
    assert result["values"]["limits"]["inject_max_cards"] == 6
    assert result["bindings"]["limits"]["inject_max_cards"]["maximum"] == 200
    assert result["application"]["limits"]["state"] == "active"
    assert result["application"]["formulas"]["consumer"] == "P5"
    assert result["values"]["time_profiles"] == []


def test_locked_save_preserves_other_settings_and_replays_without_overwriting_newer_configuration(tmp_path):
    path = tmp_path / "config.runtime.yaml"
    update_overlay(path, lambda overlay: {"unrelated": {"keep": "original"}, "gateway": {"cooldown_hours": 9}})
    base = {"gateway": {"inject_max_cards": 2}}
    original = configuration(base)
    draft = copy.deepcopy(original["values"])
    draft["limits"].update(inject_max_cards=7, diffusion_inject_max_items=5)
    request = {"expected_revision": original["revision"], "request_id": "owner-first", "values": draft}
    saved, effective = save_configuration(path, base, request)
    assert effective["gateway"]["inject_max_cards"] == 7
    assert read_overlay(path)["unrelated"] == {"keep": "original"}
    assert read_overlay(path)["gateway"]["cooldown_hours"] == 9
    assert "limits" not in read_overlay(path)["memory_system"]
    next_values = copy.deepcopy(draft)
    next_values["limits"]["inject_max_cards"] = 9
    newer, _ = save_configuration(path, base, {"expected_revision": saved["revision"],
        "request_id": "owner-second", "values": next_values})
    replay, _ = save_configuration(path, base, request)
    assert replay["replayed"] is True and replay["revision"] == saved["revision"]
    assert replay["revision"] != newer["revision"]
    assert read_overlay(path)["gateway"]["inject_max_cards"] == 9
    with pytest.raises(ConfigurationError) as error:
        save_configuration(path, base, {**request, "request_id": "stale-owner"})
    assert error.value.status == 409
    with pytest.raises(ConfigurationError) as error:
        save_configuration(path, base, {**request, "values": next_values})
    assert error.value.code == "request_id_conflict"


def test_zero_limits_are_not_replaced_by_defaults(tmp_path):
    draft = values()
    for key in ["inject_max_cards", "diffusion_inject_max_items", "inject_total_budget", "memory_detail_recall_max_ids"]:
        draft["limits"][key] = 0
    saved, effective = save_configuration(tmp_path / "runtime.yaml", {}, {
        "expected_revision": configuration({})["revision"], "request_id": "zeros", "values": draft})
    assert saved["values"] == draft
    assert effective["gateway"]["inject_max_cards"] == 0


def test_largest_dynamic_count_has_valid_default_and_independent_explicit_channels():
    limits = configuration({"gateway": {"dynamic_top_k": 200}})["values"]["limits"]
    assert limits["moment_search_limit"] == 200
    explicit = configuration({"gateway": {"dynamic_top_k": 200, "moment_search_limit": 5}})
    assert explicit["values"]["limits"]["moment_search_limit"] == 5


def test_formula_unknown_is_not_zero_and_explicit_coalesce_works():
    draft = values()
    assert preview(draft, {})["formulas"]["final"] == {"value": None, "state": "unknown"}
    draft["features"].append({"id": "freshness", "label": "新鲜度", "value_type": "number",
                              "source": "annotation", "missing": "unknown", "expression": ""})
    draft["formulas"][0].update(expression="a*relevance+b*coalesce(freshness, 0)", weights={"a": .8, "b": .2})
    result = preview(draft, {"relevance": .5})
    assert result["formulas"]["final"]["value"] == pytest.approx(.4)
    assert result["provider_requests"] == 0 and result["saved"] is False


def test_computed_features_follow_dependencies_and_cannot_be_supplied_by_inputs():
    draft = values()
    draft["features"].append({"id": "doubled", "value_type": "number", "source": "computed",
                              "missing": "unknown", "expression": "relevance*2"})
    draft["formulas"][0]["expression"] = "clamp(doubled, 0, 1)"
    assert preview(draft, {"relevance": .7})["formulas"]["final"]["value"] == 1
    with pytest.raises(ConfigurationError, match="computed_input_override"):
        preview(draft, {"relevance": .7, "doubled": 0})
    draft["features"][0].update(source="computed", expression="doubled")
    with pytest.raises(ConfigurationError, match="feature_dependency_cycle"):
        validate_values(draft)


@pytest.mark.parametrize("expression", ["__import__('os')", "relevance.real", "[1]*10", "x+1", "relevance[0]"])
def test_expression_rejects_unregistered_names_or_executable_syntax(expression):
    draft = values()
    draft["formulas"][0]["expression"] = expression
    with pytest.raises(ConfigurationError):
        validate_values(draft)


@pytest.mark.parametrize("value", [True, -1, 201, 2.1, float("inf")])
def test_limit_validation_never_silently_clamps(value):
    draft = values()
    draft["limits"]["inject_max_cards"] = value
    with pytest.raises(ConfigurationError, match="limit_out_of_range"):
        validate_values(draft)


def test_division_overflow_and_wrong_input_type_are_explicit_errors():
    draft = values()
    draft["formulas"][0]["expression"] = "1/relevance"
    with pytest.raises(ConfigurationError, match="expression_evaluation_failed"):
        preview(draft, {"relevance": 0})
    with pytest.raises(ConfigurationError, match="finite_number_required"):
        preview(draft, {"relevance": "0.7"})
    draft["formulas"][0]["expression"] = "exp(relevance)"
    draft["features"][0]["maximum"] = None
    with pytest.raises(ConfigurationError, match="expression_evaluation_failed"):
        preview(draft, {"relevance": 1000})


def test_feature_ranges_are_explicit_and_never_silently_normalized():
    draft = values()
    with pytest.raises(ConfigurationError, match="feature_out_of_range"):
        preview(draft, {"relevance": 2})
    draft["features"][0].update(minimum=2, maximum=1)
    with pytest.raises(ConfigurationError, match="invalid_feature_range"):
        validate_values(draft)


def test_profiles_require_real_positive_half_life_without_inventing_valid_until():
    draft = values()
    profile = {"id": "acute", "selector": {"kind": "state"}, "function": "exponential"}
    draft["time_profiles"] = [profile]
    with pytest.raises(ConfigurationError, match="half_life_required"):
        validate_values(draft)
    profile.update(half_life_days=1, review_after_hours=24, weight=.2)
    assert "valid_until" not in validate_values(draft)["time_profiles"][0]


def test_invalid_configuration_never_changes_overlay(tmp_path):
    path = tmp_path / "runtime.yaml"
    update_overlay(path, lambda overlay: {"preserved": 17})
    before = path.read_bytes()
    draft = values()
    draft["formulas"][0]["expression"] = "invalid_attribute.name"
    with pytest.raises(ConfigurationError):
        save_configuration(path, {}, {"expected_revision": configuration({})["revision"],
                                     "request_id": "invalid", "values": draft})
    assert path.read_bytes() == before
