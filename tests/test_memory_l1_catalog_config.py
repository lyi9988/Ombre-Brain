from copy import deepcopy

import pytest

from memory_system_config import (
    ConfigurationError,
    configuration,
    defaults,
    revision,
    save_configuration,
    values_from_config,
)


def _payload(config, *, request_id="request-1"):
    values = values_from_config(config)
    return {"expected_revision": revision(values), "request_id": request_id, "values": values}


def test_catalog_supports_custom_axis_values_and_definitions(tmp_path):
    config = {}
    payload = _payload(config)
    catalog = payload["values"]["catalog"]
    catalog["stabilities"].append("seasonal")
    catalog["definitions"].append({
        "axis": "stabilities", "id": "seasonal", "label": "季节性",
        "description": "随季节变化。", "aliases": ["seasonal"],
        "examples": ["冬季时更明显"], "parent_id": "phase",
    })
    saved, effective = save_configuration(str(tmp_path / "overlay.yaml"), config, payload)
    assert saved["validation"]["valid"] is True
    assert "seasonal" in effective["memory_system"]["catalog"]["stabilities"]


def test_legacy_client_save_preserves_effective_new_catalog_axes(tmp_path):
    config = deepcopy(defaults())
    config["memory_system"] = {"catalog": {
        "subjects": ["user", "assistant"], "kinds": ["fact"], "topics": ["music"],
        "attributes": [], "stabilities": ["episode", "phase", "stable", "custom_stability"],
        "modalities": ["assertion"], "bases": ["record"],
        "entity_types": ["user"], "relation_types": ["same_event"],
        "definitions": [{"axis": "stabilities", "id": "custom_stability", "label": "自定义",
                         "description": "保留值", "aliases": [], "examples": [], "parent_id": None}],
    }}
    payload = _payload(config)
    for field in ("stabilities", "modalities", "bases", "entity_types", "relation_types", "definitions"):
        del payload["values"]["catalog"][field]
    payload["values"]["catalog"]["topics"] = ["music", "custom_topic"]
    saved, effective = save_configuration(str(tmp_path / "overlay.yaml"), config, payload)
    assert saved["validation"]["valid"] is True
    assert effective["memory_system"]["catalog"]["stabilities"][-1] == "custom_stability"
    assert effective["memory_system"]["catalog"]["definitions"][0]["id"] == "custom_stability"


@pytest.mark.parametrize("bad_definition,code", [
    ({"axis": "stabilities", "id": "episode", "label": "x", "description": "", "aliases": [], "examples": [], "parent_id": None}, "duplicate_definition"),
    ({"axis": "stabilities", "id": "missing", "label": "x", "description": "", "aliases": [], "examples": [], "parent_id": None}, "definition_value_missing"),
    ({"axis": "stabilities", "id": "episode", "label": "x", "description": "", "aliases": [], "examples": [], "parent_id": "phase"}, "definition_parent_cycle"),
])
def test_definition_reference_and_hierarchy_errors(tmp_path, bad_definition, code):
    config = {}
    payload = _payload(config)
    catalog = payload["values"]["catalog"]
    if code == "definition_parent_cycle":
        catalog["definitions"] = deepcopy(catalog["definitions"])
        catalog["definitions"][0]["parent_id"] = "phase"
        catalog["definitions"][1]["parent_id"] = "episode"
    else:
        catalog["definitions"].append(bad_definition)
    with pytest.raises(ConfigurationError) as exc:
        save_configuration(str(tmp_path / "overlay.yaml"), config, payload)
    assert exc.value.code == code


def test_deleting_value_still_referenced_by_definition_is_rejected(tmp_path):
    config = {}
    payload = _payload(config)
    catalog = payload["values"]["catalog"]
    catalog["stabilities"].remove("phase")
    with pytest.raises(ConfigurationError) as exc:
        save_configuration(str(tmp_path / "overlay.yaml"), config, payload)
    assert exc.value.code == "definition_value_missing"


def test_partial_new_catalog_submission_is_not_treated_as_legacy(tmp_path):
    config = {}
    payload = _payload(config)
    del payload["values"]["catalog"]["bases"]
    with pytest.raises(ConfigurationError) as exc:
        save_configuration(str(tmp_path / "overlay.yaml"), config, payload)
    assert exc.value.code == "incomplete_extended_catalog"


def test_save_keeps_revision_cas_and_idempotency(tmp_path):
    config = {}
    path = str(tmp_path / "overlay.yaml")
    payload = _payload(config)
    payload["values"]["catalog"]["modalities"].append("custom_mode")
    payload["values"]["catalog"]["definitions"].append({
        "axis": "modalities", "id": "custom_mode", "label": "自定义方式",
        "description": "测试定义", "aliases": [], "examples": [], "parent_id": None,
    })
    first, _ = save_configuration(path, config, payload)
    replay, _ = save_configuration(path, config, payload)
    assert replay["replayed"] is True
    assert replay["saved_revision"] == first["saved_revision"]
    stale = _payload(config, request_id="request-stale")
    with pytest.raises(ConfigurationError) as exc:
        save_configuration(path, config, stale)
    assert exc.value.code == "configuration_revision_conflict"
    assert exc.value.status == 409


def test_default_catalog_contains_all_initial_axes():
    catalog = defaults()["catalog"]
    assert catalog["stabilities"] == ["episode", "phase", "stable"]
    assert catalog["modalities"] == ["assertion", "request", "wish", "plan", "commitment", "hypothesis", "question"]
    assert catalog["bases"] == ["direct_expression", "record", "reported", "interpretation", "hypothesis"]
    assert catalog["entity_types"] == catalog["subjects"]
    assert catalog["relation_types"] == ["same_event", "supplements", "revises", "limits", "supersedes", "possible_conflict", "split_from", "merged_from"]


def test_attribute_types_options_and_scope_round_trip(tmp_path):
    payload = _payload({})
    entries = [
        {"id": "level", "value_type": "integer", "scope": "statement"},
        {"id": "person", "value_type": "entity_ref", "scope": "any"},
        {"id": "mood", "value_type": "enum", "scope": "scene", "options": ["calm", "excited"]},
    ]
    payload["values"]["catalog"]["attributes"] = entries
    _, effective = save_configuration(str(tmp_path / "overlay.yaml"), {}, payload)
    assert effective["memory_system"]["catalog"]["attributes"] == entries


@pytest.mark.parametrize("attribute,code", [
    ({"id": "x", "value_type": []}, "invalid_value_type"),
    ({"id": "x", "value_type": "string", "scope": []}, "invalid_attribute_scope"),
    ({"id": "x", "value_type": "enum", "options": []}, "enum_options_required"),
])
def test_invalid_attributes_are_typed_errors(tmp_path, attribute, code):
    payload = _payload({})
    payload["values"]["catalog"]["attributes"] = [attribute]
    with pytest.raises(ConfigurationError) as exc:
        save_configuration(str(tmp_path / "overlay.yaml"), {}, payload)
    assert exc.value.code == code and exc.value.status == 400
