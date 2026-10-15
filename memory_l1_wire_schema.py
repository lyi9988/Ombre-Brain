"""Compact wire schema for model-authored L1 drafts.

The model supplies semantic content and source-relative references. The
application binds source identities, revisions, generation metadata, anchors,
and stable member versions before the document reaches Memory Authority.
"""
from __future__ import annotations

from copy import deepcopy


MEMORY_WRITE_MAX_REQUEST_BYTES = 512 * 1024
MEMORY_WRITE_MAX_BODY_BYTES = 512 * 1024
_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_SOURCE_PATTERN = r"^source:(?:current|[1-9][0-9]*)$"
_UNKNOWN_STATES = ["known", "not_stated", "unresolved", "not_applicable", "unbounded"]
_CONDITION_STATES = ["known", "not_stated", "unresolved", "not_applicable"]
_MEMORY_KINDS = ["preference", "boundary", "commitment", "shared_experience",
                 "key_event", "reflection", "project_state", "identity"]
MAX_PREDICATE_DEPTH = 8


def _ref(name: str) -> dict:
    return {"$ref": f"#/$defs/{name}"}


def _obj(required=(), properties=None) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(required), "properties": properties or {}}


def _arr(items) -> dict:
    return {"type": "array", "items": items}


def _definitions() -> dict:
    """Shared local definitions. Every reference is rooted at the containing schema."""
    definitions = {
        "EvidenceRef": {"type": "string", "pattern": _SOURCE_PATTERN},
        "EvidenceRefs": {"type": "array", "items": _ref("EvidenceRef"), "uniqueItems": True},
        "SceneSourceRef": {"type": "string", "pattern": _SOURCE_PATTERN},
        "Text160": {"type": "string", "minLength": 1, "maxLength": 160},
        "MemoryKind": {"type": "string", "enum": _MEMORY_KINDS},
        "UnknownState": {"type": "string", "enum": _UNKNOWN_STATES},
        "ConditionState": {"type": "string", "enum": _CONDITION_STATES},
        "Unknown": _obj(("state", "value", "evidence_refs"), {
            "state": _ref("UnknownState"), "value": {},
            "evidence_refs": _ref("EvidenceRefs"),
        }),
        "Attribute": _obj(("definition_id", "value", "evidence_refs"), {
            "definition_id": _ref("Text160"),
            "value": _ref("Unknown"), "evidence_refs": _ref("EvidenceRefs"),
        }),
        "AttributeList": _arr(_ref("Attribute")),
        "ConditionException": _obj(("state", "text", "predicate", "evidence_refs"), {
            "state": _ref("ConditionState"),
            "text": {"type": ["string", "null"]},
            "predicate": {"oneOf": [_ref("Predicate"), {"type": "null"}]},
            "evidence_refs": _ref("EvidenceRefs"),
        }),
        "Condition": _obj(("state", "text", "predicate", "evidence_refs", "exceptions"), {
            "state": _ref("ConditionState"),
            "text": {"type": ["string", "null"]},
            "predicate": {"oneOf": [_ref("Predicate"), {"type": "null"}]},
            "evidence_refs": _ref("EvidenceRefs"),
            "exceptions": _arr(_ref("ConditionException")),
        }),
        "Entity": _obj(("entity_id", "entity_type", "name", "aliases"), {
            "entity_id": _ref("Text160"),
            "entity_type": _ref("Unknown"),
            "name": {"type": "string", "minLength": 1, "maxLength": 500},
            "aliases": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 500}, "uniqueItems": True},
        }),
        "SceneSourceRange": _obj(("source_ref", "start", "end"), {
            "source_ref": _ref("SceneSourceRef"),
            "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1},
        }),
        "Scene": _obj(("scene_id", "title", "statement_refs", "attributes"), {
            "scene_id": _ref("Text160"),
            "title": {"type": "string", "minLength": 1, "maxLength": 500},
            "statement_refs": {"type": "array", "items": _ref("Text160"), "uniqueItems": True},
            "attributes": _ref("AttributeList"),
            "source_refs": {"type": "array", "items": _ref("SceneSourceRef"), "uniqueItems": True},
            "source_ranges": _arr(_ref("SceneSourceRange")),
        }),
        "Classification": _obj(("kind", "topics", "stability", "modality", "basis"), {
            "kind": _ref("Unknown"),
            "topics": {"type": "array", "items": _ref("Text160"), "uniqueItems": True},
            "stability": _ref("Unknown"), "modality": _ref("Unknown"), "basis": _ref("Unknown"),
        }),
        "EventTime": _obj(("start", "end", "precision", "interpretation", "timezone"), {
            "start": _ref("Unknown"), "end": _ref("Unknown"),
            "precision": {"type": "string", "enum": ["instant", "day", "month", "year", "approximate", "unknown"]},
            "interpretation": {"type": "string", "enum": ["instant", "duration", "uncertainty_window", "unknown"]},
            "timezone": _ref("Unknown"),
        }),
        "Applicability": _obj(("temporal_scope", "scene_ref", "start", "end", "recurrence_text"), {
            "temporal_scope": {"type": "string", "enum": ["episode", "time_range", "recurring", "unknown"]},
            "scene_ref": {"type": ["string", "null"], "maxLength": 160},
            "start": _ref("Unknown"), "end": _ref("Unknown"),
            "recurrence_text": {"type": ["string", "null"], "maxLength": 500},
        }),
        "StatementTime": _obj(("event", "applicability"), {
            "event": _ref("EventTime"), "applicability": _ref("Applicability"),
        }),
        "StatementProperty": {
            "oneOf": [
                _obj(("key", "value", "polarity", "evidence_refs"), {
                    "key": _ref("Text160"),
                    "value": _ref("Unknown"),
                    "polarity": {"type": "string", "enum": ["affirmed", "negated", "qualified", "unknown"]},
                    "evidence_refs": _ref("EvidenceRefs"),
                }),
                {"type": "null"},
            ],
        },
        "Statement": _obj(("statement_id", "content", "subject_ref", "classification", "time", "conditions", "property", "evidence_refs", "role_hint", "attributes"), {
            "statement_id": _ref("Text160"),
            "content": {"type": "string", "minLength": 1, "maxLength": 12000},
            "subject_ref": _ref("Unknown"), "classification": _ref("Classification"),
            "time": _ref("StatementTime"), "conditions": _ref("Condition"),
            "property": _ref("StatementProperty"), "evidence_refs": _ref("EvidenceRefs"),
            "role_hint": _ref("Unknown"), "attributes": _ref("AttributeList"),
        }),
        "RelationEndpoint": _obj(("object_type", "object_id"), {
            "object_type": {"type": "string", "enum": ["memory", "scene", "statement"]},
            "object_id": _ref("Text160"),
        }),
        "Relation": _obj(("relation_id", "from", "to", "kind", "evidence_refs"), {
            "relation_id": {"type": "string", "minLength": 1, "maxLength": 160},
            "from": _ref("RelationEndpoint"), "to": _ref("RelationEndpoint"),
            "kind": _ref("Text160"),
            "evidence_refs": _ref("EvidenceRefs"),
        }),
    }
    # Keep the finite operator grammar local to this definition. The owner
    # normalizer enforces MAX_PREDICATE_DEPTH before it accepts recursive input.
    no_leaf_fields = {"not": {"anyOf": [{"required": ["field"]}, {"required": ["value"]}]}}
    definitions["Predicate"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "op": {},
            "args": {"type": "array", "items": _ref("Predicate"), "minItems": 1},
            "field": _ref("Text160"), "value": {},
        },
        "oneOf": [
            {"required": ["op", "args"],
             "properties": {"op": {"enum": ["all", "any"]}}, **deepcopy(no_leaf_fields)},
            {"required": ["op", "args"],
             "properties": {"op": {"const": "not"}, "args": {"maxItems": 1}},
             **deepcopy(no_leaf_fields)},
            {"required": ["op", "field", "value"],
             "properties": {"op": {"enum": ["eq", "in", "lt", "le", "gt", "ge"]}},
             "not": {"required": ["args"]}},
        ],
    }
    return definitions


def _model_document_core(entry_forms) -> dict:
    return _obj(("entry", "narrative", "entities", "scenes", "statements", "retrieval", "relations", "attributes"), {
        "entry": _obj(("form",), {"form": {"type": "string", "enum": list(entry_forms)}}),
        "narrative": _obj(("title", "body"), {
            "title": {"type": "string", "maxLength": 1000},
            "body": {"type": "string", "maxLength": MEMORY_WRITE_MAX_BODY_BYTES},
        }),
        "entities": _arr(_ref("Entity")), "scenes": _arr(_ref("Scene")),
        "statements": _arr(_ref("Statement")),
        "retrieval": _obj(("keywords", "tags"), {
            "keywords": {"type": "array", "items": _ref("Text160"), "uniqueItems": True},
            "tags": {"type": "array", "items": _ref("Text160"), "uniqueItems": True},
        }),
        "relations": _arr(_ref("Relation")), "attributes": _ref("AttributeList"),
    })


_MODEL_ENTRY_FORMS = ("episode", "reference", "diary_full", "diary_scene")
_CHAT_ENTRY_FORMS = ("episode", "reference")
_BASE_MODEL_CORE = _model_document_core(_MODEL_ENTRY_FORMS)
_DEFINITIONS = _definitions()


def _set_source_ids(schema: dict, source_ids) -> None:
    """Tighten source references to actual program binding IDs plus source:current."""
    ids = []
    for value in source_ids:
        if not isinstance(value, str) or not value or value == "source:current" or len(value) > 200:
            raise ValueError("invalid_memory_l1_source_id")
        if value not in ids:
            ids.append(value)
    allowed = ["source:current", *ids]
    definitions = schema["$defs"]
    definitions["EvidenceRef"] = {"type": "string", "enum": allowed}
    definitions["SceneSourceRef"] = {"type": "string", "enum": allowed}


def model_document_schema(source_ids=None, *, entry_forms=None) -> dict:
    """Return a standalone rooted model schema, optionally bound to real IDs.

    Background callers pass the binding IDs supplied with that task. The
    static native chat tool uses only source:N/current and episode/reference.
    Diary callers may explicitly select diary_full/diary_scene entry forms.
    """
    result = {"$schema": _SCHEMA_URI, "$defs": deepcopy(_DEFINITIONS),
              **deepcopy(_BASE_MODEL_CORE)}
    if entry_forms is not None:
        result["properties"]["entry"]["properties"]["form"]["enum"] = list(entry_forms)
    if source_ids is not None:
        _set_source_ids(result, source_ids)
    return result


MODEL_DOCUMENT_SCHEMA = model_document_schema()


_LEGACY_PROPERTIES = {
    "title": {"type": "string", "minLength": 1, "maxLength": 100},
    "content": {"type": "string", "minLength": 1, "maxLength": 12000},
    "kind": _ref("MemoryKind"),
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "sensitive": {"type": "boolean"},
    "annotations": {"type": "array", "maxItems": 8, "items": _obj(("semantic_kind", "subject", "assertion_basis", "evidence"), {
        "semantic_kind": _ref("MemoryKind"),
        "subject": {"type": "string", "enum": ["user", "assistant"]},
        "assertion_basis": {"type": "string", "enum": ["owner_statement", "assistant_commitment", "shared_experience", "assistant_interpretation"]},
        "evidence": {"type": "string", "minLength": 1, "maxLength": 600},
        "polarity": {"type": "string", "enum": ["positive", "negative", "mixed", "unspecified"]},
        "assertion_state": {"type": "string", "enum": ["stated", "changed", "cancelled", "fulfilled", "uncertain"]},
        **{field: _ref("Text160")
           for field in ("topic", "value", "conditions", "exceptions", "valid_time")},
    })},
    "event_time": _obj(("expression", "evidence"), {
        "expression": {"type": "string", "maxLength": 60},
        "evidence": {"type": "string", "maxLength": 600},
    }),
}


def _tool_parameters() -> dict:
    definitions = deepcopy(_DEFINITIONS)
    definitions["ModelDocument"] = _model_document_core(_CHAT_ENTRY_FORMS)
    return {
        "$schema": _SCHEMA_URI,
        "$defs": definitions,
        "type": "object", "additionalProperties": False,
        "properties": {"document": _ref("ModelDocument"), **deepcopy(_LEGACY_PROPERTIES)},
        "oneOf": [
            {"required": ["document"],
             "not": {"anyOf": [{"required": [field]} for field in ("title", "content", "annotations", "event_time")]}},
            {"required": ["title", "content", "kind"], "not": {"required": ["document"]}},
        ],
    }


MEMORY_WRITE_PARAMETERS = _tool_parameters()


def tool_parameters() -> dict:
    """Return an isolated schema object for the native ToolRegistry definition."""
    return deepcopy(MEMORY_WRITE_PARAMETERS)


__all__ = [
    "EVIDENCE_REF_SCHEMA", "EVIDENCE_REFS_SCHEMA", "SCENE_SOURCE_REF_SCHEMA", "UNKNOWN_SCHEMA",
    "MEMORY_WRITE_MAX_REQUEST_BYTES", "MEMORY_WRITE_MAX_BODY_BYTES", "MODEL_DOCUMENT_SCHEMA",
    "MEMORY_WRITE_PARAMETERS", "MAX_PREDICATE_DEPTH", "model_document_schema", "tool_parameters",
]


# Public compatibility fragments remain self-contained; actual exported roots
# above use local refs so repetition does not consume the discovery budget.
EVIDENCE_REF_SCHEMA = {"type": "string", "pattern": _SOURCE_PATTERN}
EVIDENCE_REFS_SCHEMA = {"type": "array", "items": deepcopy(EVIDENCE_REF_SCHEMA), "uniqueItems": True}
SCENE_SOURCE_REF_SCHEMA = deepcopy(EVIDENCE_REF_SCHEMA)
UNKNOWN_SCHEMA = _obj(("state", "value", "evidence_refs"), {
    "state": {"type": "string", "enum": _UNKNOWN_STATES}, "value": {},
    "evidence_refs": deepcopy(EVIDENCE_REFS_SCHEMA),
})
