import copy

import pytest

from memory_l1_contract import L1ContractError
from memory_l1_ingestion import prepare_model_document
from memory_l1_runtime import namespace_local_ids
from memory_l1_schema import unknown
from test_memory_l1_ingestion_p2c import setup as setup_l1


def ordinary_local_document(document):
    draft = copy.deepcopy(document)
    draft["entities"][0]["entity_id"] = "owner"
    draft["statements"][0]["statement_id"] = "symptom"
    draft["statements"][1]["statement_id"] = "comfort"
    draft["scenes"][0]["scene_id"] = "scene"
    draft["scenes"][0]["statement_refs"] = ["symptom", "comfort"]
    draft["statements"][0]["subject_ref"] = unknown("known", "owner", ["source:current"])
    draft["statements"][1]["subject_ref"] = unknown("known", "owner", ["source:current"])
    draft["statements"][0]["role_hint"] = unknown("known", "core", ["source:current"])
    draft["statements"][1]["role_hint"] = unknown("known", "detail", ["source:current"])
    for statement in draft["statements"]:
        statement["time"]["event"]["start"] = unknown("not_stated")
        statement["time"]["event"]["end"] = unknown("not_stated")
        statement["time"]["applicability"]["temporal_scope"] = "unknown"
        statement["time"]["applicability"]["scene_ref"] = "scene"
    endpoint = lambda object_id: {
        "memory_id": None,
        "revision": "this_revision",
        "object_type": "statement",
        "object_id": object_id,
    }
    draft["relations"] = [{
        "relation_id": "relation",
        "from": endpoint("symptom"),
        "to": endpoint("comfort"),
        "kind": "supplements",
        "evidence_refs": ["source:current"],
    }]
    return draft


def test_namespace_local_ids_accepts_plain_keys_and_remaps_references(tmp_path):
    env = setup_l1(tmp_path)
    service, _source, model_document, _generation = env
    catalog, _revision = service.catalog_provider()

    namespaced, mapping = namespace_local_ids(
        ordinary_local_document(model_document), "w0", catalog=catalog
    )

    assert namespaced["entities"][0]["entity_id"].startswith("new:w0:")
    assert namespaced["statements"][0]["statement_id"].startswith("new:w0:")
    assert namespaced["scenes"][0]["scene_id"].startswith("new:w0:")
    assert namespaced["relations"][0]["relation_id"].startswith("new:w0:")
    assert mapping["owner"] == namespaced["entities"][0]["entity_id"]
    assert mapping["symptom"] == namespaced["statements"][0]["statement_id"]
    assert mapping["comfort"] == namespaced["statements"][1]["statement_id"]
    assert mapping["scene"] == namespaced["scenes"][0]["scene_id"]
    assert mapping["relation"] == namespaced["relations"][0]["relation_id"]
    assert all(
        statement["subject_ref"]["value"] == namespaced["entities"][0]["entity_id"]
        for statement in namespaced["statements"]
    )
    assert namespaced["scenes"][0]["statement_refs"] == [
        statement["statement_id"] for statement in namespaced["statements"]
    ]
    assert namespaced["relations"][0]["from"]["object_id"] == namespaced["statements"][0]["statement_id"]
    assert namespaced["relations"][0]["to"]["object_id"] == namespaced["statements"][1]["statement_id"]


def test_prepare_model_document_maps_plain_keys_and_preserves_semantics(tmp_path):
    env = setup_l1(tmp_path)
    service, source, model_document, generation = env

    prepared = prepare_model_document(
        service,
        memory_id="ordinary-local-keys",
        document=ordinary_local_document(model_document),
        sources=[source],
        generation=generation,
        request_id="ordinary-local-keys",
    )

    entity_id = prepared["entities"][0]["entity_id"]
    statement_ids = [item["statement_id"] for item in prepared["statements"]]
    scene_id = prepared["scenes"][0]["scene_id"]
    relation_id = prepared["relations"][0]["relation_id"]
    assert entity_id.startswith("en_") and entity_id != "owner"
    assert all(value.startswith("st_") for value in statement_ids)
    assert scene_id.startswith("sc_") and scene_id != "scene"
    assert relation_id.startswith("rl_") and relation_id != "relation"
    assert [item["subject_ref"]["value"] for item in prepared["statements"]] == [entity_id, entity_id]
    assert [item["role_hint"]["value"] for item in prepared["statements"]] == ["core", "detail"]
    assert prepared["scenes"][0]["statement_refs"] == statement_ids
    assert prepared["relations"][0]["from"]["object_id"] == statement_ids[0]
    assert prepared["relations"][0]["to"]["object_id"] == statement_ids[1]
    assert all(item["time"]["event"]["start"]["state"] == "not_stated"
               for item in prepared["statements"])
    assert all(item["time"]["applicability"]["temporal_scope"] == "unknown"
               for item in prepared["statements"])
    assert all(item["time"]["applicability"]["scene_ref"] == scene_id
               for item in prepared["statements"])
    assert prepared["generation"]["origin"] == "first_pass_model"
    assert env[0].authority.list_memories() == []


def test_prepare_model_document_does_not_guess_user_as_an_entity_reference(tmp_path):
    env = setup_l1(tmp_path)
    service, source, model_document, generation = env
    draft = ordinary_local_document(model_document)
    draft["statements"][0]["subject_ref"] = unknown("known", "user", ["source:current"])

    with pytest.raises(L1ContractError) as raised:
        prepare_model_document(
            service,
            memory_id="undeclared-user-subject",
            document=draft,
            sources=[source],
            generation=generation,
            request_id="undeclared-user-subject",
        )

    assert raised.value.code == "unknown_catalog_value"
    assert raised.value.path == "statements[0].subject_ref.value"


@pytest.mark.parametrize("role_hint", ["user", "assistant"])
def test_prepare_model_document_rejects_speaker_labels_as_role_hint(tmp_path, role_hint):
    env = setup_l1(tmp_path)
    service, source, model_document, generation = env
    draft = ordinary_local_document(model_document)
    draft["statements"][0]["role_hint"] = unknown("known", role_hint, ["source:current"])

    with pytest.raises(L1ContractError) as raised:
        prepare_model_document(
            service,
            memory_id=f"invalid-role-hint-{role_hint}",
            document=draft,
            sources=[source],
            generation=generation,
            request_id=f"invalid-role-hint-{role_hint}",
        )

    assert raised.value.code == "unknown_catalog_value"
    assert raised.value.path == "statements[0].role_hint.value"
