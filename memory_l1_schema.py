"""Complete L1 content schema. Validation does not confer source authority."""
from __future__ import annotations

import copy
from datetime import date, datetime
import re
from typing import Any

from memory_l1_contract import (ANCHOR_FIELDS, L1ContractError, SCHEMA_VERSION,
    canonical_json, canonical_l1_body, validate_unknown_value)

TOP_FIELDS = {"schema_version", "entry", "narrative", "entities", "scenes", "statements",
              "retrieval", "evidence_bindings", "relations", "attributes", "generation"}
BASE_CATALOG = {
    "subjects": ["user", "assistant", "third_party", "object"],
    "entity_types": ["user", "assistant", "third_party", "object"],
    "kinds": ["experience", "state", "fact", "preference", "boundary", "intention", "interpretation", "procedure"],
    "topics": [], "attributes": [],
    "stabilities": ["episode", "phase", "stable"],
    "modalities": ["assertion", "request", "wish", "plan", "commitment", "hypothesis", "question"],
    "bases": ["direct_expression", "record", "reported", "interpretation", "hypothesis"],
    "cognition_roles": ["observation", "question", "hypothesis", "insight", "view", "practice"],
    "cognition_statuses": ["tentative", "supported", "contested", "revised", "resolved"],
    "relation_types": ["same_event", "supplements", "revises", "limits", "supersedes", "possible_conflict", "split_from", "merged_from"],
    "definitions": [],
}
GENERATION_FIELDS = {"origin", "prompt_revision", "catalog_revision", "model_route_revision", "task_id", "actor", "catalog_snapshot"}
MAX_PREDICATE_DEPTH = 8


def fail(code, path):
    raise L1ContractError(code, path)


def shape(value, fields, path, optional=()):
    if not isinstance(value, dict) or set(value) - set(fields) or set(fields) - set(optional) - set(value):
        fail("invalid_object_fields", path)


def text(value, path, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        fail("string_required", path)
    canonical_json(value)


def strings(value, path):
    if not isinstance(value, list): fail("array_required", path)
    for index, item in enumerate(value): text(item, f"{path}[{index}]")
    if len(set(value)) != len(value): fail("duplicate_reference", path)


def array(value, path):
    if not isinstance(value, list): fail("array_required", path)


def unknown(state="not_stated", value=None, evidence_refs=None):
    return {"state": state, "value": value, "evidence_refs": list(evidence_refs or [])}


def catalog_with_defaults(catalog=None):
    result = copy.deepcopy(BASE_CATALOG)
    if catalog is not None:
        if not isinstance(catalog, dict): fail("object_required", "catalog")
        result.update(copy.deepcopy(catalog))
    for axis in ("subjects", "entity_types", "kinds", "topics", "stabilities", "modalities", "bases", "relation_types", "cognition_roles", "cognition_statuses"):
        strings(result[axis], "catalog." + axis)
    array(result["attributes"], "catalog.attributes")
    seen = set()
    for i, item in enumerate(result["attributes"]):
        p = f"catalog.attributes[{i}]"
        shape(item, {"id", "value_type", "label", "description", "options", "scope"}, p,
              optional={"label", "description", "options", "scope"})
        text(item["id"], p + ".id")
        if item["id"] in seen: fail("duplicate_attribute_definition", p + ".id")
        seen.add(item["id"])
        if item["value_type"] not in ("string", "number", "integer", "boolean", "enum", "entity_ref"): fail("invalid_attribute_type", p + ".value_type")
        if "scope" in item and item["scope"] not in ("memory", "scene", "statement", "any"): fail("invalid_attribute_scope", p + ".scope")
        for key in ("label", "description"):
            if key in item: text(item[key], p + "." + key, empty=True)
        if "options" in item: strings(item["options"], p + ".options")
        if item["value_type"] == "enum" and not item.get("options"): fail("enum_options_required", p + ".options")
    return result


def generation_template(catalog=None, revision=None, origin="legacy_projection"):
    return {"origin": origin, "prompt_revision": None, "catalog_revision": revision,
            "model_route_revision": None, "task_id": None, "actor": None,
            "catalog_snapshot": catalog_with_defaults(catalog)}


def conditions_template():
    return {"state": "not_stated", "text": None, "predicate": None, "exceptions": [], "evidence_refs": []}


def statement_template():
    return {"statement_id": "new:statement", "statement_revision_id": None, "content": "",
            "subject_ref": unknown(),
            "classification": {"kind": unknown(), "topics": [], "stability": unknown(), "modality": unknown(), "basis": unknown()},
            "time": {"event": {"start": unknown(), "end": unknown(), "precision": "unknown",
                                 "interpretation": "unknown", "timezone": unknown()},
                     "applicability": {"temporal_scope": "unknown", "scene_ref": None,
                                       "start": unknown(), "end": unknown(), "recurrence_text": None}},
            "conditions": conditions_template(), "property": None, "evidence_refs": [],
            "role_hint": unknown(), "attributes": []}


def cognition_template():
    """Optional annotation of a claim, never a second body or truth authority."""
    return {"role": unknown(), "epistemic_status": unknown(),
            "about_refs": [], "open_questions": []}


def scene_template(catalog=None, catalog_revision=None):
    return {"scene_id": "new:scene", "scene_revision_id": None, "title": "", "anchors": [],
            "statement_refs": [], "attributes": [], "generation": generation_template(catalog, catalog_revision)}


def document_template(body="", title="", *, catalog=None, catalog_revision=None):
    return {"schema_version": SCHEMA_VERSION,
            "entry": {"form": "episode", "parent_ref": None, "document_group_id": None, "derived_from_revision": None,
                      "scene_ref": None, "shared_statement_refs": []},
            "narrative": {"title": title, "body": body}, "entities": [], "scenes": [], "statements": [],
            "retrieval": {"keywords": [], "tags": []}, "evidence_bindings": [], "relations": [], "attributes": [],
            "generation": generation_template(catalog, catalog_revision)}


def _revision(value, path, *, nullable=False):
    if nullable and value is None: return
    if not ((type(value) is int and value > 0) or (isinstance(value, str) and value.strip())):
        fail("invalid_revision", path)


def _nullable_text(value, path):
    if value is not None: text(value, path)


def _anchor(value, path):
    shape(value, ANCHOR_FIELDS, path)
    if value["target_namespace"] not in ("l0", "l1"): fail("invalid_anchor_namespace", path)
    for key in ("target_id", "text_view_id"): text(value[key], path + "." + key)
    _revision(value["target_revision"], path + ".target_revision")
    if value["coordinate_unit"] != "unicode_codepoint": fail("invalid_coordinate_unit", path)
    if type(value["start"]) is not int or type(value["end"]) is not int or not 0 <= value["start"] < value["end"]:
        fail("invalid_codepoint_range", path)
    for key in ("text_sha256", "quote_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]): fail("invalid_sha256", path + "." + key)


def _date_value(value, path, *, bound=False):
    validate_unknown_value(value, value_type="string", path=path, allow_unbounded=bound)
    if value["state"] != "known": return
    raw = value["value"]
    try:
        if re.fullmatch(r"\d{4}", raw): date(int(raw), 1, 1)
        elif re.fullmatch(r"\d{4}-\d{2}", raw): date(int(raw[:4]), int(raw[5:]), 1)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw): date.fromisoformat(raw)
        else: datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, OverflowError): fail("invalid_iso_time", path + ".value")


def _predicate(node, definitions, path, entity_ids, *, depth=0):
    if not isinstance(node, dict): fail("object_required", path)
    op = node.get("op")
    if op in ("all", "any", "not"):
        if depth >= MAX_PREDICATE_DEPTH: fail("predicate_depth_exceeded", path)
        shape(node, {"op", "args"}, path)
        if not isinstance(node["args"], list) or not node["args"] or (op == "not" and len(node["args"]) != 1): fail("invalid_predicate_arity", path)
        for i, child in enumerate(node["args"]):
            _predicate(child, definitions, f"{path}.args[{i}]", entity_ids, depth=depth + 1)
        return
    if op not in ("eq", "in", "lt", "le", "gt", "ge"): fail("unsupported_predicate", path)
    shape(node, {"op", "field", "value"}, path)
    field = node["field"]
    if not isinstance(field, str) or field not in definitions: fail("unknown_condition_field", path + ".field")
    values = node["value"] if op == "in" else [node["value"]]
    if not isinstance(values, list) or not values: fail("invalid_predicate_values", path)
    kind = definitions[field].get("value_type")
    if op in ("lt", "le", "gt", "ge") and kind not in ("number", "integer"): fail("numeric_comparison_required", path)
    for i, item in enumerate(values): _attribute_value(item, kind, f"{path}.value[{i}]", definitions[field], entity_ids)


def _attribute_value(value, kind, path, definition=None, entity_ids=None):
    scalar = "string" if kind in ("enum", "entity_ref") else kind
    validate_unknown_value(unknown("known", value), value_type=scalar, path=path,
                           allowed_values=(set((definition or {}).get("options", [])) if kind == "enum" else entity_ids if kind == "entity_ref" else None))


def _attributes(values, definitions, path, *, scope, entity_ids):
    array(values, path)
    seen = set()
    for index, item in enumerate(values):
        p = f"{path}[{index}]"
        shape(item, {"definition_id", "value", "evidence_refs"}, p)
        key = item["definition_id"]
        if not isinstance(key, str) or key not in definitions: fail("unknown_attribute", p + ".definition_id")
        if key in seen: fail("duplicate_attribute", p)
        seen.add(key)
        if definitions[key].get("scope", "any") not in (scope, "any"):
            fail("attribute_scope_mismatch", p + ".definition_id")
        kind = definitions[key].get("value_type")
        validate_unknown_value(item["value"], value_type="string" if kind in ("enum", "entity_ref") else kind, path=p + ".value",
                               allowed_values=(set(definitions[key].get("options", [])) if kind == "enum" else entity_ids if kind == "entity_ref" else None))
        strings(item["evidence_refs"], p + ".evidence_refs")


def _conditions(value, definitions, path, entity_ids, *, exception=False):
    fields = {"state", "text", "predicate", "evidence_refs"} | (set() if exception else {"exceptions"})
    shape(value, fields, path)
    if value["state"] not in ("known", "not_stated", "unresolved", "not_applicable"): fail("invalid_knowledge_state", path)
    _nullable_text(value["text"], path + ".text")
    strings(value["evidence_refs"], path + ".evidence_refs")
    if value["predicate"] is not None: _predicate(value["predicate"], definitions, path + ".predicate", entity_ids)
    if value["state"] in ("not_stated", "not_applicable") and (value["text"] is not None or value["predicate"] is not None): fail("unknown_condition_has_value", path)
    if value["state"] == "known" and value["text"] is None and value["predicate"] is None: fail("known_condition_requires_value", path)
    if not exception:
        array(value["exceptions"], path + ".exceptions")
        for i, item in enumerate(value["exceptions"]): _conditions(item, definitions, f"{path}.exceptions[{i}]", entity_ids, exception=True)


def validate_l1_document(document: Any, *, catalog=None) -> dict:
    """Validate every semantic module; source/owner authorization is separate."""
    canonical_json(document)
    shape(document, TOP_FIELDS, "document")
    if document["schema_version"] != SCHEMA_VERSION: fail("unsupported_l1_schema", "document.schema_version")
    shape(document["generation"], GENERATION_FIELDS, "generation")
    c = catalog_with_defaults(catalog if catalog is not None else document.get("generation", {}).get("catalog_snapshot"))
    if catalog is not None and c != catalog_with_defaults(document["generation"]["catalog_snapshot"]):
        fail("catalog_snapshot_mismatch", "generation.catalog_snapshot")
    definitions = {item["id"]: item for item in c["attributes"] if isinstance(item, dict) and "id" in item}
    d = copy.deepcopy(document)
    shape(d["entry"], {"form", "parent_ref", "document_group_id", "derived_from_revision", "scene_ref", "shared_statement_refs"},
          "entry", optional={"scene_ref", "shared_statement_refs"})
    entry = d["entry"]
    if entry["form"] not in ("episode", "diary_full", "diary_scene", "reference"): fail("invalid_entry_form", "entry.form")
    if entry["parent_ref"] is not None:
        shape(entry["parent_ref"], {"memory_id", "revision"}, "entry.parent_ref")
        text(entry["parent_ref"]["memory_id"], "entry.parent_ref.memory_id")
        if type(entry["parent_ref"]["revision"]) is not int or entry["parent_ref"]["revision"] <= 0: fail("invalid_revision", "entry.parent_ref.revision")
    if entry["form"] == "diary_scene" and entry["parent_ref"] is None: fail("scene_parent_required", "entry.parent_ref")
    _nullable_text(entry["document_group_id"], "entry.document_group_id")
    _revision(entry["derived_from_revision"], "entry.derived_from_revision", nullable=True)
    if entry.get("scene_ref") is not None:
        shape(entry["scene_ref"], {"scene_id", "scene_revision_id"}, "entry.scene_ref")
        text(entry["scene_ref"]["scene_id"], "entry.scene_ref.scene_id")
        _revision(entry["scene_ref"]["scene_revision_id"], "entry.scene_ref.scene_revision_id")
        if entry["form"] != "diary_scene" or entry["parent_ref"] is None:
            fail("scene_parent_required", "entry.scene_ref")
        if d["statements"]: fail("shared_statements_must_not_be_duplicated", "statements")
    strings(entry.get("shared_statement_refs", []), "entry.shared_statement_refs")
    if entry.get("shared_statement_refs"):
        if entry.get("scene_ref") is None: fail("shared_statement_parent_required", "entry.shared_statement_refs")
    shape(d["narrative"], {"title", "body"}, "narrative")
    text(d["narrative"]["title"], "narrative.title", empty=True)
    canonical_l1_body(d["narrative"]["body"])
    shape(d["generation"], GENERATION_FIELDS, "generation")
    if d["generation"]["origin"] not in ("first_pass_model", "owner_edit", "legacy_projection"): fail("invalid_generation_origin", "generation.origin")
    for key in GENERATION_FIELDS - {"origin", "catalog_snapshot"}: _nullable_text(d["generation"][key], "generation." + key)
    catalog_with_defaults(d["generation"]["catalog_snapshot"])
    ids = {}
    for module, key in (("entities", "entity_id"), ("scenes", "scene_id"), ("statements", "statement_id"), ("evidence_bindings", "binding_id"), ("relations", "relation_id")):
        array(d[module], module)
        ids[module] = set()
        for i, item in enumerate(d[module]):
            if not isinstance(item, dict): fail("object_required", f"{module}[{i}]")
            text(item.get(key), f"{module}[{i}].{key}")
            if item[key] in ids[module]: fail("duplicate_identity", f"{module}[{i}].{key}")
            ids[module].add(item[key])
    for i, item in enumerate(d["entities"]):
        p = f"entities[{i}]"; shape(item, {"entity_id", "entity_type", "name", "aliases"}, p)
        validate_unknown_value(item["entity_type"], value_type="string", allowed_values=set(c["entity_types"]), path=p + ".entity_type")
        text(item["name"], p + ".name"); strings(item["aliases"], p + ".aliases")
    for i, item in enumerate(d["evidence_bindings"]):
        p = f"evidence_bindings[{i}]"
        shape(item, {"binding_id", "source_namespace", "source_id", "source_revision", "author_ref", "speaker_ref", "anchors"}, p)
        for key in ("source_namespace", "source_id"): text(item[key], p + "." + key)
        _revision(item["source_revision"], p + ".source_revision")
        for key in ("author_ref", "speaker_ref"): _nullable_text(item[key], p + "." + key)
        array(item["anchors"], p + ".anchors")
        if not item["anchors"]: fail("evidence_anchor_required", p + ".anchors")
        for j, a in enumerate(item["anchors"]):
            _anchor(a, f"{p}.anchors[{j}]")
            if a["target_namespace"] != "l0" or a["target_id"] != item["source_id"] or type(a["target_revision"]) is not type(item["source_revision"]) or a["target_revision"] != item["source_revision"]: fail("source_anchor_mismatch", p)
    for i, item in enumerate(d["statements"]):
        p = f"statements[{i}]"
        shape(item, {"statement_id", "statement_revision_id", "content", "subject_ref", "classification", "time", "conditions", "property", "evidence_refs", "role_hint", "attributes", "cognition"}, p,
              optional={"cognition"})
        cognition = item.get("cognition")
        if cognition is not None:
            shape(cognition, {"role", "epistemic_status", "about_refs", "open_questions"}, p + ".cognition")
            for name, axis in (("role", "cognition_roles"), ("epistemic_status", "cognition_statuses")):
                validate_unknown_value(cognition[name], value_type="string", allowed_values=set(c[axis]),
                                       path=p + ".cognition." + name)
            strings(cognition["about_refs"], p + ".cognition.about_refs")
            if set(cognition["about_refs"]) - ids["entities"]:
                fail("dangling_cognition_subject", p + ".cognition.about_refs")
            strings(cognition["open_questions"], p + ".cognition.open_questions")
        _revision(item["statement_revision_id"], p + ".statement_revision_id", nullable=True)
        text(item["content"], p + ".content")
        validate_unknown_value(item["subject_ref"], value_type="string", allowed_values=ids["entities"], path=p + ".subject_ref")
        shape(item["classification"], {"kind", "topics", "stability", "modality", "basis"}, p + ".classification")
        for key, axis in (("kind", "kinds"), ("stability", "stabilities"), ("modality", "modalities"), ("basis", "bases")):
            validate_unknown_value(item["classification"][key], value_type="string", allowed_values=set(c[axis]), path=p + ".classification." + key)
        strings(item["classification"]["topics"], p + ".classification.topics")
        if set(item["classification"]["topics"]) - set(c["topics"]): fail("unknown_catalog_value", p + ".classification.topics")
        shape(item["time"], {"event", "applicability"}, p + ".time")
        event = item["time"]["event"]; app = item["time"]["applicability"]
        shape(event, {"start", "end", "precision", "interpretation", "timezone"}, p + ".time.event")
        if event["precision"] not in ("instant", "day", "month", "year", "approximate", "unknown"): fail("invalid_precision", p + ".time.event.precision")
        if event["interpretation"] not in ("instant", "duration", "uncertainty_window", "unknown"): fail("invalid_time_interpretation", p + ".time.event.interpretation")
        for key in ("start", "end"): _date_value(event[key], f"{p}.time.event.{key}")
        validate_unknown_value(event["timezone"], value_type="string", path=p + ".time.event.timezone")
        shape(app, {"temporal_scope", "scene_ref", "start", "end", "recurrence_text"}, p + ".time.applicability")
        if app["temporal_scope"] not in ("episode", "time_range", "recurring", "unknown"): fail("invalid_temporal_scope", p + ".time.applicability")
        _nullable_text(app["scene_ref"], p + ".time.applicability.scene_ref")
        if app["scene_ref"] is not None and app["scene_ref"] not in ids["scenes"]: fail("dangling_scene_ref", p + ".time.applicability.scene_ref")
        for key in ("start", "end"): _date_value(app[key], f"{p}.time.applicability.{key}", bound=True)
        _nullable_text(app["recurrence_text"], p + ".time.applicability.recurrence_text")
        _conditions(item["conditions"], definitions, p + ".conditions", ids["entities"])
        if item["property"] is not None:
            prop = item["property"]; shape(prop, {"key", "value", "polarity", "evidence_refs"}, p + ".property")
            if not isinstance(prop["key"], str) or prop["key"] not in definitions: fail("unknown_property", p + ".property.key")
            if definitions[prop["key"]].get("scope", "any") not in ("statement", "any"):
                fail("attribute_scope_mismatch", p + ".property.key")
            if prop["polarity"] not in ("affirmed", "negated", "qualified", "unknown"): fail("invalid_polarity", p + ".property")
            kind = definitions[prop["key"]]["value_type"]
            validate_unknown_value(prop["value"], value_type="string" if kind in ("enum", "entity_ref") else kind, path=p + ".property.value",
                                   allowed_values=(set(definitions[prop["key"]].get("options", [])) if kind == "enum" else ids["entities"] if kind == "entity_ref" else None))
            strings(prop["evidence_refs"], p + ".property.evidence_refs")
        strings(item["evidence_refs"], p + ".evidence_refs")
        validate_unknown_value(item["role_hint"], value_type="string", allowed_values={"core", "detail"}, path=p + ".role_hint")
        _attributes(item["attributes"], definitions, p + ".attributes", scope="statement", entity_ids=ids["entities"])
    for i, item in enumerate(d["scenes"]):
        p = f"scenes[{i}]"; shape(item, {"scene_id", "scene_revision_id", "title", "anchors", "statement_refs", "generation", "attributes"}, p, optional={"attributes"})
        _revision(item["scene_revision_id"], p + ".scene_revision_id", nullable=True)
        text(item["title"], p + ".title"); array(item["anchors"], p + ".anchors")
        for j, a in enumerate(item["anchors"]): _anchor(a, f"{p}.anchors[{j}]")
        strings(item["statement_refs"], p + ".statement_refs")
        if set(item["statement_refs"]) - ids["statements"]: fail("dangling_statement_ref", p + ".statement_refs")
        shape(item["generation"], GENERATION_FIELDS, p + ".generation")
        if item["generation"]["origin"] not in ("first_pass_model", "owner_edit", "legacy_projection"):
            fail("invalid_generation_origin", p + ".generation.origin")
        for key in GENERATION_FIELDS - {"origin", "catalog_snapshot"}:
            _nullable_text(item["generation"][key], p + ".generation." + key)
        catalog_with_defaults(item["generation"]["catalog_snapshot"])
        _attributes(item.get("attributes", []), definitions, p + ".attributes", scope="scene", entity_ids=ids["entities"])
    for i, item in enumerate(d["relations"]):
        p = f"relations[{i}]"; shape(item, {"relation_id", "from", "to", "kind", "evidence_refs", "origin", "confirmation_state"}, p)
        if item["kind"] not in c["relation_types"]: fail("unknown_relation_kind", p + ".kind")
        if item["confirmation_state"] not in ("proposed", "confirmed", "rejected", "unresolved"): fail("invalid_confirmation_state", p)
        if item["origin"] not in ("model", "owner", "program"): fail("invalid_relation_origin", p)
        strings(item["evidence_refs"], p + ".evidence_refs")
        for key in ("from", "to"):
            endpoint = item[key]; shape(endpoint, {"memory_id", "revision", "object_type", "object_id"}, p + "." + key)
            _nullable_text(endpoint["memory_id"], p + "." + key + ".memory_id")
            _revision(endpoint["revision"], p + "." + key + ".revision")
            if endpoint["object_type"] not in ("memory", "scene", "statement"): fail("invalid_relation_target", p)
            text(endpoint["object_id"], p + "." + key + ".object_id")
            if endpoint["memory_id"] is None and endpoint["revision"] == "this_revision":
                module = {"scene": "scenes", "statement": "statements"}.get(endpoint["object_type"])
                if module and endpoint["object_id"] not in ids[module]: fail("dangling_relation_ref", p + "." + key)
    shape(d["retrieval"], {"keywords", "tags"}, "retrieval")
    for key in ("keywords", "tags"): strings(d["retrieval"][key], "retrieval." + key)
    _attributes(d["attributes"], definitions, "attributes", scope="memory", entity_ids=ids["entities"])

    def refs_walk(node, path):
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "evidence_refs":
                    strings(val, path + ".evidence_refs")
                    if set(val) - ids["evidence_bindings"]: fail("dangling_evidence_ref", path + ".evidence_refs")
                else: refs_walk(val, path + "." + key)
        elif isinstance(node, list):
            for i, child in enumerate(node): refs_walk(child, f"{path}[{i}]")
    for key in ("entities", "statements", "relations", "attributes", "scenes"):
        refs_walk(d[key], key)
    return d


def split_document(document, *, catalog=None):
    value = validate_l1_document(document, catalog=catalog)
    body = canonical_l1_body(value["narrative"].pop("body"))
    return body, value


def join_document(body, semantic):
    value = copy.deepcopy(semantic)
    if not isinstance(value.get("narrative"), dict): fail("invalid_narrative", "narrative")
    if "body" in value["narrative"]: fail("duplicate_body_authority", "narrative.body")
    value["narrative"]["body"] = body
    return validate_l1_document(value)


def bind_l1_metadata(body, metadata):
    """Validate and stamp a version's semantic identity; legacy metadata stays intact."""
    from memory_l1_contract import l1_content_sha256, semantic_sha256
    result = copy.deepcopy(dict(metadata or {}))
    if "memory_l1" not in result:
        return result
    document = join_document(body, result["memory_l1"])
    _, semantic = split_document(document)
    result["memory_l1"] = semantic
    result["memory_l1_semantic_sha256"] = semantic_sha256(semantic)
    result["memory_l1_content_sha256"] = l1_content_sha256(body, semantic)
    return result


def verify_l1_metadata(body, expected, observed):
    from memory_l1_contract import l1_content_sha256, semantic_sha256
    if "memory_l1" not in expected:
        return
    if observed.get("memory_l1") != expected["memory_l1"]:
        fail("projected_semantic_mismatch", "memory_l1")
    semantic = observed["memory_l1"]
    for key, value in (("memory_l1_semantic_sha256", semantic_sha256(semantic)),
                       ("memory_l1_content_sha256", l1_content_sha256(body, semantic))):
        if expected.get(key) != value or observed.get(key) != value:
            fail("projected_content_identity_mismatch", key)
