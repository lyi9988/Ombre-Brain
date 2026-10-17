"""Cognition annotations stay optional, revisioned, source-bound L1 semantics."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest
from jsonschema import Draft202012Validator, ValidationError

from memory_authority import IdempotencyConflict, RevisionConflict
from memory_l1_catalog_schema import model_document_schema
from memory_l1_contract import L1ContractError, make_text_anchor
from memory_l1_features import read_features
from memory_l1_ingestion import model_template
from memory_l1_model_identity import normalize_model_local_keys
from memory_l1_runtime import namespace_local_ids
from memory_l1_schema import (
    BASE_CATALOG,
    cognition_template,
    document_template,
    scene_template,
    statement_template,
    unknown,
    validate_l1_document,
)
from memory_recall_semantics import (
    lexical_features,
    preview_pipeline,
    profile_for,
    resolve_document,
    resolve_statement,
    score_document,
)
from memory_system_config import validate_values, values_from_config
from test_memory_recall_contract_r2 import (
    CATALOG_REVISION,
    DAY,
    make_committed_diary,
    run,
    setup_diary,
)


def _known(value):
    return unknown("known", value)


def _cognition(*, role="question", status="supported", about_refs=(), open_questions=()):
    return {
        "role": _known(role),
        "epistemic_status": _known(status),
        "about_refs": list(about_refs),
        "open_questions": list(open_questions),
    }


def _model_document(catalog):
    document = model_template(catalog=catalog)
    statement = statement_template()
    statement.pop("statement_revision_id")
    statement.update(statement_id="claim-1", content="一条合成陈述。")
    document["statements"] = [statement]
    return document


def test_model_schema_keeps_cognition_optional_and_binds_custom_and_empty_catalog_axes():
    catalog = copy.deepcopy(BASE_CATALOG)
    catalog["cognition_roles"].append("owner_added_role")
    catalog["cognition_statuses"].append("owner_added_status")
    schema = model_document_schema(catalog=catalog)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    document = _model_document(catalog)
    validator.validate(document)  # Legacy/model drafts may omit cognition.
    document["statements"][0]["cognition"] = None
    validator.validate(document)
    document["statements"][0]["cognition"] = _cognition(
        role="owner_added_role", status="owner_added_status"
    )
    validator.validate(document)

    invalid = copy.deepcopy(document)
    invalid["statements"][0]["cognition"]["role"] = _known("not_in_catalog")
    with pytest.raises(ValidationError):
        validator.validate(invalid)

    empty_catalog = copy.deepcopy(BASE_CATALOG)
    empty_catalog["cognition_roles"] = []
    empty_catalog["cognition_statuses"] = []
    empty_validator = Draft202012Validator(model_document_schema(catalog=empty_catalog))
    empty_document = _model_document(empty_catalog)
    empty_document["statements"][0]["cognition"] = cognition_template()
    empty_validator.validate(empty_document)

    unsupported_known = copy.deepcopy(empty_document)
    unsupported_known["statements"][0]["cognition"]["epistemic_status"] = _known("supported")
    with pytest.raises(ValidationError):
        empty_validator.validate(unsupported_known)


def test_model_and_runtime_identity_normalizers_rewrite_cognition_entity_refs():
    catalog = copy.deepcopy(BASE_CATALOG)
    document = _model_document(catalog)
    document["entities"] = [{
        "entity_id": "local-person", "entity_type": unknown(),
        "name": "合成实体", "aliases": [],
    }]
    document["statements"][0]["cognition"] = _cognition(about_refs=["local-person"])

    normalized, first_mapping = normalize_model_local_keys(document, catalog=catalog)
    normalized_ref = normalized["statements"][0]["cognition"]["about_refs"][0]
    assert normalized_ref == normalized["entities"][0]["entity_id"]
    assert normalized_ref == first_mapping["local-person"]

    namespaced, namespace_mapping = namespace_local_ids(document, "cognition-task", catalog=catalog)
    namespaced_ref = namespaced["statements"][0]["cognition"]["about_refs"][0]
    assert namespaced_ref == namespaced["entities"][0]["entity_id"]
    assert namespaced_ref.startswith("new:cognition-task:")
    assert namespaced_ref == namespace_mapping["local-person"]


def _cognition_diary_payload(env, request_id, *, include_entity=True, about_ref="new:cognition-entity"):
    source = run(env["sources"].capture_diary(DAY, binding_id="cognition-fixture"))["source"]
    scene_id = "new:cognition-scene"
    statement_id = "new:cognition-statement"
    scene = scene_template(env["catalog"], CATALOG_REVISION)
    scene.update(
        scene_id=scene_id,
        title="合成认知场景",
        anchors=[make_text_anchor(
            source["text"], target_namespace="l0", target_id=source["source_id"],
            target_revision=source["source_revision"], text_view_id=source["text_view_id"],
            start=0, end=len(source["text"]),
        )],
        statement_refs=[statement_id],
    )
    statement = statement_template()
    statement.update(
        statement_id=statement_id,
        content="合成陈述保留其证据和认识状态。",
        evidence_refs=["diary-source"],
        cognition=_cognition(
            about_refs=[about_ref],
            open_questions=["这条合成陈述还有什么未确认？"],
        ),
    )
    document = document_template(
        "合成日记全文。", "合成认知日记",
        catalog=env["catalog"], catalog_revision=CATALOG_REVISION,
    )
    document["entry"]["form"] = "diary_full"
    if include_entity:
        document["entities"] = [{
            "entity_id": about_ref,
            "entity_type": _known("user"),
            "name": "合成主人实体",
            "aliases": [],
        }]
    document["scenes"] = [scene]
    document["statements"] = [statement]
    return {
        "day": DAY,
        "document": document,
        "scene_bodies": {scene_id: "合成场景正文。"},
        "generation": {
            "prompt_revision": "synthetic-cognition-prompt-r1",
            "model_route_revision": "synthetic-cognition-route-r1",
            "task_id": request_id,
            "source_revision": source["source_revision"],
        },
        "input_ranges": [{"start": 0, "end": len(source["text"])}],
        "request_id": request_id,
        "confidence": 0.99,
        "requested_mode": "auto",
    }


def _prepare_and_adopt_cognition(env, request_id="cognition-adoption-r1"):
    candidate = run(env["diary"].prepare(**_cognition_diary_payload(env, request_id)))
    assert candidate["status"] == "accepted"
    adopted = run(env["diary"].adopt(candidate["candidate_id"]))
    assert adopted["status"] == "complete"
    memory_id = candidate["proposal"]["metadata"]["diary_l1_bundle"]["memory_id"]
    detail = run(env["revisions"].detail(memory_id, 1))
    return candidate, memory_id, detail


def test_prepare_adopt_authority_sql_features_fixed_detail_page_and_source_binding_keep_cognition(tmp_path):
    env = setup_diary(tmp_path)
    _candidate, memory_id, detail = _prepare_and_adopt_cognition(env)
    assert detail["l1"]["state"] == "current"
    document = detail["l1"]["document"]
    statement = document["statements"][0]
    entity_id = document["entities"][0]["entity_id"]
    cognition = statement["cognition"]
    assert not entity_id.startswith("new:")
    assert cognition["about_refs"] == [entity_id]
    assert statement["evidence_refs"] == ["diary-source"]
    assert document["evidence_bindings"][0]["binding_id"] == "diary-source"

    conn = env["authority"]._connect()
    try:
        member = conn.execute(
            "SELECT content_json FROM memory_l1_members WHERE memory_id=? AND revision=1 "
            "AND module='statements' AND object_id=?",
            (memory_id, statement["statement_id"]),
        ).fetchone()
        assert member is not None
        assert json.loads(member[0])["cognition"] == cognition
        feature_rows = conn.execute(
            "SELECT feature_path, value_json FROM memory_l1_features WHERE memory_id=? "
            "AND revision=1 AND object_type='statement' AND object_id=?",
            (memory_id, statement["statement_id"]),
        ).fetchall()
    finally:
        conn.close()
    projected = {row[0]: json.loads(row[1]) for row in feature_rows}
    assert projected["cognition.role"] == "question"
    assert projected["cognition.epistemic_status"] == "supported"
    assert projected["cognition.about_refs"] == entity_id
    assert projected["cognition.open_questions"] == "这条合成陈述还有什么未确认？"
    assert any(row["feature_path"] == "cognition.about_refs" and row["value"] == entity_id
               for row in read_features(env["authority"], memory_id, 1))

    fixed_detail = run(env["revisions"].detail(memory_id, 1))["l1"]["document"]
    page = env["reads"].page(memory_id, 1, "statements")
    assert fixed_detail["statements"][0]["cognition"] == cognition
    assert page["items"][0]["cognition"] == cognition

    source_status = run(env["reads"].recall_source_status([{
        "memory_id": memory_id, "revision": 1, "binding_ids": ["diary-source"],
    }]))["items"][0]
    assert source_status["state"] == "ok"
    assert source_status["sources"]["diary-source"]["eligible_source"] is True
    usage = resolve_document(document, {
        "now": datetime(2026, 10, 17, tzinfo=timezone.utc),
        "purpose": "current", "facts": {},
    }, source_states={"diary-source": "available"})[0]
    assert usage["evidence_refs"] == ["diary-source"]
    assert usage["source_state"] == "available"
    assert usage["cognition"] == cognition
    assert usage["usage"] == "relationship_reference"


def test_owner_cognition_edit_versions_statement_and_preserves_replay_and_cas(tmp_path):
    env = setup_diary(tmp_path)
    _candidate, memory_id, original = _prepare_and_adopt_cognition(env, "cognition-owner-r1")
    original_document = original["l1"]["document"]
    original_statement = original_document["statements"][0]
    payload = {
        "expected_revision": 1,
        "request_id": "cognition-owner-edit-r1",
        "document": copy.deepcopy(original_document),
        "edit_note": "synthetic cognition edit",
    }
    payload["document"]["statements"][0]["cognition"]["open_questions"] = [
        "修订后的合成问题是什么？"
    ]
    saved = run(env["revisions"].save(memory_id, payload))
    revised_statement = saved["result"]["l1"]["document"]["statements"][0]
    assert saved["revision"] == 2
    assert revised_statement["statement_id"] == original_statement["statement_id"]
    assert revised_statement["statement_revision_id"] != original_statement["statement_revision_id"]
    assert revised_statement["cognition"]["open_questions"] == ["修订后的合成问题是什么？"]
    assert run(env["revisions"].detail(memory_id, 1))["l1"]["document"]["statements"][0]["cognition"] == original_statement["cognition"]

    replay = run(env["revisions"].save(memory_id, payload))
    assert replay["replayed"] is True
    assert replay["revision"] == 2
    changed_replay = copy.deepcopy(payload)
    changed_replay["document"]["statements"][0]["cognition"]["open_questions"] = ["同请求的另一份问题？"]
    with pytest.raises(IdempotencyConflict):
        run(env["revisions"].save(memory_id, changed_replay))

    stale = copy.deepcopy(payload)
    stale["request_id"] = "stale-cognition-owner-edit"
    with pytest.raises(RevisionConflict):
        run(env["revisions"].save(memory_id, stale))


def test_dangling_cognition_about_refs_are_rejected_before_candidate_adoption(tmp_path):
    env = setup_diary(tmp_path)
    payload = _cognition_diary_payload(
        env, "dangling-cognition-r1", include_entity=False, about_ref="new:missing-entity"
    )
    with pytest.raises(L1ContractError) as exc:
        run(env["diary"].prepare(**payload))
    assert exc.value.code == "dangling_cognition_subject"
    assert env["authority"].list_candidates(limit=20) == []


def test_legacy_statement_without_cognition_stays_optional_through_owner_revision(tmp_path):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env, request_id="legacy-cognition-r1")
    before = run(env["revisions"].detail(committed["memory_id"], 1))["l1"]["document"]
    before_statement = before["statements"][0]
    assert "cognition" not in before_statement
    payload = {
        "expected_revision": 1,
        "request_id": "legacy-cognition-owner-edit-r1",
        "document": copy.deepcopy(before),
        "edit_note": "synthetic metadata-only legacy edit",
    }
    payload["document"]["narrative"]["title"] = "合成旧陈述的标题修订"
    saved = run(env["revisions"].save(committed["memory_id"], payload))
    after_statement = saved["result"]["l1"]["document"]["statements"][0]
    assert "cognition" not in after_statement
    assert after_statement["statement_revision_id"] == before_statement["statement_revision_id"]
    assert "cognition" not in run(env["revisions"].detail(committed["memory_id"], 1))["l1"]["document"]["statements"][0]


@pytest.mark.parametrize(
    ("status", "expected_usage"),
    [
        ("tentative", "unresolved_evidence"),
        ("contested", "unresolved_evidence"),
        ("revised", "historical_evidence"),
        ("supported", "relationship_reference"),
    ],
)
def test_cognition_status_controls_use_without_promoting_supported_to_world_fact(status, expected_usage):
    statement = statement_template()
    statement.update(
        statement_id="cognition-use-claim",
        content="合成的认知陈述。",
        evidence_refs=["synthetic-binding"],
        cognition=_cognition(status=status),
    )
    use = resolve_statement(statement, {
        "now": datetime(2026, 10, 17, tzinfo=timezone.utc),
        "purpose": "current", "facts": {},
    }, source_state="available")
    assert use["usage"] == expected_usage
    assert use["cognition"] == statement["cognition"]
    assert use["evidence_refs"] == ["synthetic-binding"]
    if status == "supported":
        assert use["reason_codes"] == ["cognitive_reference_not_world_fact"]


@pytest.mark.parametrize(
    ("purpose", "false_condition", "expected_reason"),
    [
        ("history", False, "historical_query"),
        ("current", True, "condition_not_met"),
    ],
)
def test_supported_cognition_is_historical_for_history_or_false_condition(purpose, false_condition, expected_reason):
    statement = statement_template()
    statement.update(
        statement_id="cognition-historical-claim",
        content="合成的认知陈述。",
        evidence_refs=["synthetic-binding"],
        cognition=_cognition(status="supported"),
    )
    facts = {}
    if false_condition:
        statement["conditions"] = {
            "state": "known", "text": "合成条件", "predicate": {
                "op": "eq", "field": "mood", "value": "calm",
            }, "exceptions": [], "evidence_refs": [],
        }
        facts["mood"] = _known("angry")
    use = resolve_statement(statement, {
        "now": datetime(2026, 10, 17, tzinfo=timezone.utc),
        "purpose": purpose, "facts": facts,
    }, source_state="available")
    assert use["usage"] == "historical_evidence"
    assert use["reason_codes"] == [expected_reason]


def test_cognition_profiles_scoring_features_and_question_aliases_are_consumed():
    values = values_from_config({})
    values["recall"]["scoring_profiles"] = [{
        "id": "open_question_profile",
        "selector": {"cognition_role": "question", "cognition_status": "tentative"},
        "formula_id": "final",
    }]
    values = validate_values(values)
    statement = statement_template()
    statement.update(
        statement_id="cognition-score-claim",
        content="合成的认知评分陈述。",
        cognition=_cognition(
            status="tentative",
            open_questions=["我是不是把工作和休息都安排得太满？"],
        ),
    )
    document = document_template("合成预览文档", catalog=values["catalog"])
    document["statements"] = [statement]
    validate_l1_document(document, catalog=values["catalog"])
    profile = profile_for(statement, values["recall"]["scoring_profiles"], "current")
    assert profile["id"] == "open_question_profile"

    context = {
        "now": datetime(2026, 10, 17, tzinfo=timezone.utc),
        "purpose": "current", "facts": {}, "query": "我是不是把工作和休息都安排得太满？",
    }
    scored = score_document(document, values, context, {"relevance": 0.4})["statements"][0]
    assert scored["scoring_profile"] == "open_question_profile"
    assert scored["features"]["cognition_role"] == "question"
    assert scored["features"]["cognition_status"] == "tentative"
    assert scored["features"]["open_question_count"] == 1

    aliases = lexical_features("我有点纠结", document, values["catalog"])
    explicit_question = lexical_features(
        "我是不是把工作和休息都安排得太满？", document, values["catalog"]
    )
    paraphrase = lexical_features("最近的安排是否过量", document, values["catalog"])
    assert aliases["cognition_role_match"] == 1.0
    assert explicit_question["cognition_question_match"] == 1.0
    assert paraphrase["cognition_question_match"] == 0.0


def test_preview_pipeline_scores_cognition_scenario_without_provider_or_write():
    values = validate_values(values_from_config({}))
    scenario = {
        "now": "2026-10-17T10:00:00+00:00",
        "purpose": "current",
        "cognition": _cognition(
            status="tentative",
            open_questions=["预览留下的合成未解问题是什么？"],
        ),
    }
    inputs = {"relevance": 0.37}
    previewed = preview_pipeline(values, inputs, scenario)

    direct_statement = statement_template()
    direct_statement["content"] = "Synthetic preview claim"
    direct_statement["cognition"] = scenario["cognition"]
    direct_document = document_template("Synthetic preview only", catalog=values["catalog"])
    direct_document["statements"] = [direct_statement]
    validate_l1_document(direct_document, catalog=values["catalog"])
    direct = score_document(direct_document, values, {
        "now": datetime.fromisoformat(scenario["now"]),
        "target_time": None, "purpose": "current", "facts": {},
    }, inputs)["statements"][0]
    preview_row = previewed["pipeline"]["statements"][0]

    assert preview_row["score"] == direct["score"]
    assert preview_row["features"] == direct["features"]
    assert preview_row["calculation"] == direct["calculation"]
    assert previewed["pipeline"]["usage"][0]["cognition"] == scenario["cognition"]
    assert previewed["pipeline"]["usage"][0]["usage"] == "unresolved_evidence"
    assert previewed["provider_requests"] == 0
    assert previewed["saved"] is False
