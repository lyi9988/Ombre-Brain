"""Pure P2-A content identity, unknown values and versioned text anchors.

These helpers neither grant source access nor adopt a memory. Source adapters
must supply the actual source identity/version and the authorized text view.
The full L1 validator and Authority/API integration are subsequent P2-A work.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "memory-l1-v1"
UNKNOWN_STATES = frozenset({"known", "not_stated", "unresolved", "not_applicable"})
ANCHOR_FIELDS = frozenset({"target_namespace", "target_id", "target_revision", "text_view_id",
                           "coordinate_unit", "start", "end", "text_sha256", "quote_sha256"})


class L1ContractError(ValueError):
    def __init__(self, code: str, path: str):
        super().__init__(f"{code}: {path}")
        self.code = code
        self.path = path


def _fail(code: str, path: str) -> None:
    raise L1ContractError(code, path)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("nonempty_string_required", path)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        _fail("invalid_unicode", path)
    return value


def _json_tree(value: Any, path: str) -> None:
    if value is None or type(value) in (str, bool, int):
        if isinstance(value, str):
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeError:
                _fail("invalid_unicode", path)
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail("finite_number_required", path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                _fail("string_json_key_required", path)
            _json_tree(key, path)
            _json_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_tree(item, f"{path}[{index}]")
        return
    _fail("json_type_required", path)


def canonical_json(value: Any) -> str:
    """Canonical JSON for semantic identity, without coercion or silent drops."""
    try:
        _json_tree(value, "semantic")
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except RecursionError:
        _fail("cyclic_or_overdeep_json", "semantic")


def canonical_l1_body(value: str) -> str:
    """The existing MemoryCommitService body normalization, unchanged."""
    if not isinstance(value, str):
        _fail("string_required", "narrative.body")
    body = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    _text(body, "narrative.body")
    return body


def semantic_sha256(semantic: Mapping[str, Any]) -> str:
    if not isinstance(semantic, dict):
        _fail("object_required", "semantic")
    return hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()


def l1_content_sha256(body: str, semantic: Mapping[str, Any]) -> str:
    """Bind the complete semantic document and canonical body, not just prose.

    Callers pass metadata.memory_l1, excluding telemetry/projection state.
    Fields are never dropped here: same body plus edited semantics differs.
    """
    if not isinstance(semantic, dict):
        _fail("object_required", "semantic")
    envelope = {"identity_format": "memory-l1-content-v1", "body": canonical_l1_body(body),
                "semantic": semantic}
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def validate_unknown_value(payload: Any, *, value_type: str, path: str,
                           allowed_values: set[str] | None = None,
                           allow_unbounded: bool = False) -> None:
    """Validate a typed value without turning uncertainty into false or zero."""
    if value_type not in {"string", "number", "integer", "boolean"}:
        _fail("unsupported_value_type", path)
    if not isinstance(payload, dict) or set(payload) != {"state", "value", "evidence_refs"}:
        _fail("invalid_unknown_value_shape", path)
    state = payload["state"]
    if not isinstance(state, str) or state not in UNKNOWN_STATES | ({"unbounded"} if allow_unbounded else set()):
        _fail("invalid_knowledge_state", path + ".state")
    refs = payload["evidence_refs"]
    if not isinstance(refs, list) or any(not isinstance(item, str) or not item.strip() for item in refs):
        _fail("invalid_evidence_refs", path + ".evidence_refs")
    if len(set(refs)) != len(refs):
        _fail("duplicate_evidence_ref", path + ".evidence_refs")
    value = payload["value"]
    if state != "known":
        if value is not None:
            _fail("unknown_value_must_be_null", path + ".value")
        if state == "unbounded" and not refs:
            _fail("explicit_unbounded_evidence_required", path + ".evidence_refs")
        return
    if value_type == "string":
        _text(value, path + ".value")
        if allowed_values is not None and value not in allowed_values:
            _fail("unknown_catalog_value", path + ".value")
    elif value_type == "boolean":
        if type(value) is not bool:
            _fail("boolean_required", path + ".value")
    elif value_type == "integer":
        if type(value) is not int:
            _fail("integer_required", path + ".value")
    elif type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        _fail("finite_number_required", path + ".value")


def make_text_anchor(text: str, *, target_namespace: str, target_id: str,
                     target_revision: str | int, text_view_id: str, start: int, end: int) -> dict:
    """Anchor an authorized exact text view; offsets are code points, end exclusive."""
    _text(text, "anchor.text")
    _text(target_namespace, "anchor.target_namespace")
    if target_namespace not in {"l0", "l1"}:
        _fail("invalid_anchor_namespace", "anchor.target_namespace")
    _text(target_id, "anchor.target_id")
    _text(text_view_id, "anchor.text_view_id")
    if not ((type(target_revision) is int and target_revision > 0)
            or (isinstance(target_revision, str) and target_revision.strip())):
        _fail("invalid_anchor_revision", "anchor.target_revision")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        _fail("invalid_codepoint_range", "anchor.range")
    return {"target_namespace": target_namespace, "target_id": target_id,
            "target_revision": target_revision, "text_view_id": text_view_id,
            "coordinate_unit": "unicode_codepoint", "start": start, "end": end,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "quote_sha256": hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()}


def verify_text_anchor(text: str, anchor: Any, *, target_namespace: str, target_id: str,
                       target_revision: str | int, text_view_id: str) -> None:
    """Use current adapter-supplied identity; an anchor cannot assert its own version."""
    if not isinstance(anchor, dict) or set(anchor) != ANCHOR_FIELDS:
        _fail("invalid_anchor_shape", "anchor")
    # Strict checks precede equality: Python otherwise considers True == 1.
    if type(anchor["target_revision"]) is not type(target_revision):
        _fail("anchor_version_mismatch", "anchor.target_revision")
    expected = make_text_anchor(text, target_namespace=target_namespace, target_id=target_id,
                                target_revision=target_revision, text_view_id=text_view_id,
                                start=anchor["start"], end=anchor["end"])
    for key in ANCHOR_FIELDS:
        if type(anchor[key]) is not type(expected[key]) or anchor[key] != expected[key]:
            _fail("anchor_mismatch", "anchor." + key)
