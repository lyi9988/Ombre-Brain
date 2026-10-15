from copy import deepcopy

import pytest

from memory_l1_contract import L1ContractError, make_text_anchor, verify_text_anchor
from memory_l1_schema import (
    document_template,
    generation_template,
    join_document,
    split_document,
    statement_template,
    validate_l1_document,
    verify_l1_metadata,
    bind_l1_metadata,
)


SOURCE_ID = "diary:2026-10-15:entry-7"
SOURCE_REVISION = 4
SOURCE_TEXT = "我和衍衍🧵在窗边说话：备考时晚上别催我。"
CATALOG_REVISION = "catalog-r7"
BINDING_ID = "binding:diary-entry-7-r4"


def custom_catalog():
    return {
        "kinds": [
            "experience", "state", "fact", "preference", "boundary", "intention",
            "interpretation", "procedure", "craft_preference",
        ],
        "topics": ["relationship", "needlework"],
        "attributes": [
            {"id": "is_exam_period", "value_type": "boolean", "scope": "statement"},
            {"id": "pressure", "value_type": "number", "scope": "statement"},
            {"id": "craft_level", "value_type": "integer", "scope": "statement"},
        ],
    }


def anchor(start_text, end_text=None, *, namespace="l0", target_id=SOURCE_ID,
           revision=SOURCE_REVISION, view_id="diary-entry-7-r4"):
    start = SOURCE_TEXT.index(start_text)
    end = start + len(end_text if end_text is not None else start_text)
    return make_text_anchor(
        SOURCE_TEXT,
        target_namespace=namespace,
        target_id=target_id,
        target_revision=revision,
        text_view_id=view_id,
        start=start,
        end=end,
    )


def known(value, evidence_refs=None):
    return {"state": "known", "value": value, "evidence_refs": list(evidence_refs or [])}


def test_custom_attributes_enforce_scope_enum_and_entity_references():
    doc = document_template("完整合成正文", catalog={"attributes": [
        {"id": "mood", "value_type": "enum", "options": ["calm", "excited"], "scope": "memory"},
        {"id": "person", "value_type": "entity_ref", "scope": "any"},
    ]})
    doc["entities"] = [{"entity_id": "new:user", "entity_type": known("user"), "name": "合成人物", "aliases": []}]
    doc["attributes"] = [{"definition_id": "mood", "value": known("calm"), "evidence_refs": []}]
    validate_l1_document(doc)
    bad = deepcopy(doc); bad["attributes"][0]["value"] = known("invalid")
    with pytest.raises(L1ContractError, match="unknown_catalog_value"): validate_l1_document(bad)
    bad = deepcopy(doc); bad["attributes"] = [{"definition_id": "person", "value": known("missing"), "evidence_refs": []}]
    with pytest.raises(L1ContractError, match="unknown_catalog_value"): validate_l1_document(bad)
    bad = deepcopy(doc); item = statement_template(); item["content"] = "合成陈述"
    item["attributes"] = deepcopy(doc["attributes"]); bad["statements"] = [item]
    with pytest.raises(L1ContractError, match="attribute_scope_mismatch"): validate_l1_document(bad)


def valid_diary_scene_document():
    catalog = custom_catalog()
    doc = document_template(
        body="完整保留这次聊天与相处过程。",
        title="备考时的相处边界",
        catalog=catalog,
        catalog_revision=CATALOG_REVISION,
    )
    doc["entry"].update({
        "form": "diary_scene",
        "parent_ref": {"memory_id": "memory:diary-full-2026-10-15", "revision": 3},
        "document_group_id": "diary-group:2026-10-15",
    })

    doc["entities"] = [
        {"entity_id": "entity:user", "entity_type": known("user"), "name": "主人", "aliases": ["我"]},
        {"entity_id": "entity:assistant", "entity_type": known("assistant"), "name": "衍衍", "aliases": []},
    ]

    first_anchor = anchor("窗边说话")
    second_anchor = anchor("备考时晚上别催我")
    doc["evidence_bindings"] = [{
        "binding_id": BINDING_ID,
        "source_namespace": "diary",
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "author_ref": "entity:user",
        "speaker_ref": "entity:user",
        "anchors": [first_anchor, second_anchor],
    }]

    boundary = statement_template()
    boundary.update({
        "statement_id": "new:statement-boundary",
        "content": "主人说，备考时晚上不希望被催。",
        "subject_ref": known("entity:user", [BINDING_ID]),
        "classification": {
            "kind": known("boundary", [BINDING_ID]),
            "topics": ["relationship", "needlework"],
            "stability": known("phase", [BINDING_ID]),
            "modality": known("request", [BINDING_ID]),
            "basis": known("direct_expression", [BINDING_ID]),
        },
        "conditions": {
            "state": "known",
            "text": "备考时，晚上",
            "predicate": {
                "op": "all",
                "args": [
                    {"op": "eq", "field": "is_exam_period", "value": True},
                    {"op": "any", "args": [
                        {"op": "lt", "field": "pressure", "value": 5.0},
                        {"op": "not", "args": [
                            {"op": "eq", "field": "is_exam_period", "value": False}
                        ]},
                    ]},
                ],
            },
            "exceptions": [],
            "evidence_refs": [BINDING_ID],
        },
        "property": {
            "key": "craft_level",
            "value": known(2, [BINDING_ID]),
            "polarity": "qualified",
            "evidence_refs": [BINDING_ID],
        },
        "evidence_refs": [BINDING_ID],
        "attributes": [{
            "definition_id": "is_exam_period",
            "value": known(True, [BINDING_ID]),
            "evidence_refs": [BINDING_ID],
        }],
    })

    reflection = statement_template()
    reflection.update({
        "statement_id": "new:statement-reflection",
        "content": "衍衍觉得这次交流让彼此更亲近。",
        "subject_ref": known("entity:assistant", [BINDING_ID]),
        "classification": {
            "kind": known("craft_preference", [BINDING_ID]),
            "topics": ["relationship", "needlework"],
            "stability": known("episode", [BINDING_ID]),
            "modality": known("assertion", [BINDING_ID]),
            "basis": known("interpretation", [BINDING_ID]),
        },
        "evidence_refs": [BINDING_ID],
        "attributes": [{
            "definition_id": "craft_level",
            "value": known(2, [BINDING_ID]),
            "evidence_refs": [BINDING_ID],
        }],
    })
    doc["statements"] = [boundary, reflection]
    doc["scenes"] = [{
        "scene_id": "new:scene-window-chat",
        "scene_revision_id": None,
        "title": "窗边谈到备考时的相处边界",
        "anchors": [first_anchor, second_anchor],
        "statement_refs": [boundary["statement_id"], reflection["statement_id"]],
        "generation": generation_template(catalog, revision=CATALOG_REVISION),
    }]
    doc["statements"][0]["time"]["applicability"]["scene_ref"] = "new:scene-window-chat"
    doc["statements"][0]["time"]["applicability"]["temporal_scope"] = "episode"
    doc["retrieval"].update({"keywords": ["备考", "相处边界"], "tags": ["关系", "手工"]})
    return doc, catalog


def assert_contract_error(document, code, path, *, catalog=None):
    with pytest.raises(L1ContractError) as exc_info:
        validate_l1_document(document, catalog=catalog)
    assert exc_info.value.code == code
    assert exc_info.value.path == path


def test_full_template_and_multistatement_owner_draft_accept_custom_catalog_and_unicode_anchors():
    document, catalog = valid_diary_scene_document()

    validated = validate_l1_document(document)

    assert validated["entry"]["form"] == "diary_scene"
    assert validated["entry"]["parent_ref"] == {
        "memory_id": "memory:diary-full-2026-10-15", "revision": 3
    }
    assert len(validated["statements"]) == 2
    assert validated["statements"][0]["statement_id"].startswith("new:")
    assert validated["statements"][0]["statement_revision_id"] is None
    assert validated["scenes"][0]["scene_revision_id"] is None
    assert validated["statements"][0]["classification"]["kind"]["value"] == "boundary"
    assert validated["statements"][1]["classification"]["kind"]["value"] == "craft_preference"
    assert validated["statements"][0]["classification"]["topics"] == ["relationship", "needlework"]
    assert validated["statements"][0]["property"]["key"] == "craft_level"
    assert set(validated["generation"]) == {
        "origin", "prompt_revision", "catalog_revision", "model_route_revision",
        "task_id", "actor", "catalog_snapshot",
    }
    assert validated["generation"] == generation_template(catalog, revision=CATALOG_REVISION)
    assert len(validated["scenes"][0]["anchors"]) == 2
    for item in validated["scenes"][0]["anchors"]:
        verify_text_anchor(
            SOURCE_TEXT,
            item,
            target_namespace="l0",
            target_id=SOURCE_ID,
            target_revision=SOURCE_REVISION,
            text_view_id="diary-entry-7-r4",
        )


def test_catalog_drives_custom_kind_topic_and_typed_property_validation():
    document, catalog = valid_diary_scene_document()
    validate_l1_document(document)

    bad_kind = deepcopy(document)
    bad_kind["statements"][1]["classification"]["kind"] = known("not_in_catalog")
    assert_contract_error(
        bad_kind, "unknown_catalog_value", "statements[1].classification.kind.value"
    )

    bad_topic = deepcopy(document)
    bad_topic["statements"][0]["classification"]["topics"] = ["not_in_catalog"]
    assert_contract_error(
        bad_topic, "unknown_catalog_value", "statements[0].classification.topics"
    )

    bad_property_type = deepcopy(document)
    bad_property_type["statements"][0]["property"]["value"] = known("two")
    assert_contract_error(
        bad_property_type, "integer_required", "statements[0].property.value.value"
    )


def test_unknown_is_not_false_or_permanent_and_explicit_unbounded_needs_evidence():
    document, _ = valid_diary_scene_document()
    statement = document["statements"][0]
    assert statement["time"]["event"]["end"] == {
        "state": "not_stated", "value": None, "evidence_refs": []
    }
    validate_l1_document(document)

    explicit_unbounded = deepcopy(document)
    explicit_unbounded["statements"][0]["time"]["applicability"]["end"] = {
        "state": "unbounded", "value": None, "evidence_refs": [BINDING_ID]
    }
    validate_l1_document(explicit_unbounded)

    no_evidence = deepcopy(explicit_unbounded)
    no_evidence["statements"][0]["time"]["applicability"]["end"]["evidence_refs"] = []
    assert_contract_error(
        no_evidence,
        "explicit_unbounded_evidence_required",
        "statements[0].time.applicability.end.evidence_refs",
    )

    false_smuggled_as_unknown = deepcopy(document)
    false_smuggled_as_unknown["statements"][0]["classification"]["kind"] = {
        "state": "not_stated", "value": False, "evidence_refs": []
    }
    assert_contract_error(
        false_smuggled_as_unknown,
        "unknown_value_must_be_null",
        "statements[0].classification.kind.value",
    )

    explicit_false = deepcopy(document)
    explicit_false["statements"][0]["conditions"]["predicate"]["args"][0]["value"] = False
    validate_l1_document(explicit_false)


@pytest.mark.parametrize(
    ("mutate", "code", "path"),
    [
        (
            lambda d: d["statements"][0]["subject_ref"].update(value="entity:missing"),
            "unknown_catalog_value",
            "statements[0].subject_ref.value",
        ),
        (
            lambda d: d["statements"][0]["evidence_refs"].append("binding:missing"),
            "dangling_evidence_ref",
            "statements[0].evidence_refs",
        ),
        (
            lambda d: d["statements"][0]["time"]["applicability"].update(scene_ref="scene:missing"),
            "dangling_scene_ref",
            "statements[0].time.applicability.scene_ref",
        ),
        (
            lambda d: d["scenes"][0]["statement_refs"].append("statement:missing"),
            "dangling_statement_ref",
            "scenes[0].statement_refs",
        ),
    ],
)
def test_invalid_subject_evidence_and_scene_references_report_field_paths(mutate, code, path):
    document, _ = valid_diary_scene_document()
    mutate(document)
    assert_contract_error(document, code, path)


def test_nested_conditions_resolve_only_catalog_property_fields():
    document, _ = valid_diary_scene_document()
    validate_l1_document(document)

    invalid_field = deepcopy(document)
    invalid_field["statements"][0]["conditions"]["predicate"]["args"][1]["args"][0]["field"] = "mood_guess"
    assert_contract_error(
        invalid_field,
        "unknown_condition_field",
        "statements[0].conditions.predicate.args[1].args[0].field",
    )


def test_split_join_keeps_one_body_authority():
    document, _ = valid_diary_scene_document()
    body, semantic = split_document(document)

    assert body == document["narrative"]["body"]
    assert "body" not in semantic["narrative"]
    assert join_document(body, semantic) == validate_l1_document(document)

    duplicate_body = deepcopy(semantic)
    duplicate_body["narrative"]["body"] = "a second writable copy"
    with pytest.raises(L1ContractError) as exc_info:
        join_document(body, duplicate_body)
    assert exc_info.value.code == "duplicate_body_authority"
    assert exc_info.value.path == "narrative.body"


def test_semantic_only_edit_changes_full_digest_and_inconsistent_readback_is_rejected():
    document, _ = valid_diary_scene_document()
    body, semantic = split_document(document)
    original = bind_l1_metadata(body, {"memory_l1": semantic, "legacy_marker": "kept"})
    assert original["legacy_marker"] == "kept"

    edited_semantic = deepcopy(semantic)
    edited_semantic["retrieval"]["tags"].append("owner-added-tag")
    edited = bind_l1_metadata(body, {"memory_l1": edited_semantic, "legacy_marker": "kept"})
    assert edited["memory_l1_content_sha256"] != original["memory_l1_content_sha256"]
    assert edited["memory_l1_semantic_sha256"] != original["memory_l1_semantic_sha256"]

    verify_l1_metadata(body, edited, deepcopy(edited))
    with pytest.raises(L1ContractError) as exc_info:
        verify_l1_metadata(body, edited, original)
    assert exc_info.value.code == "projected_semantic_mismatch"
    assert exc_info.value.path == "memory_l1"


def test_document_template_builds_a_valid_generation_snapshot():
    catalog = custom_catalog()
    document = document_template(
        body="一段完整正文。", title="模板验证", catalog=catalog,
        catalog_revision=CATALOG_REVISION,
    )

    assert document["generation"] == generation_template(catalog, revision=CATALOG_REVISION)
    assert validate_l1_document(document) == document
