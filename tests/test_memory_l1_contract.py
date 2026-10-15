"""P2-A semantic integrity and uncertainty boundaries; synthetic text only."""
import copy
import hashlib

import pytest

from memory_l1_contract import (L1ContractError, canonical_json, l1_content_sha256,
    make_text_anchor, semantic_sha256, validate_unknown_value, verify_text_anchor)


def value(state="known", content="state", refs=None):
    return {"state": state, "value": content, "evidence_refs": refs or []}


def test_metadata_only_edit_changes_complete_identity_even_when_body_unchanged():
    before = {"schema_version": "memory-l1-v1", "statements": [{"kind": "state", "stability": "episode"}]}
    after = copy.deepcopy(before)
    after["statements"][0]["stability"] = "stable"
    assert semantic_sha256(before) != semantic_sha256(after)
    assert l1_content_sha256("same body", before) != l1_content_sha256("same body", after)


def test_digest_is_key_order_independent_and_matches_existing_body_normalization():
    assert l1_content_sha256("  first\r\nsecond\r\n", {"a": 1, "b": False}) == l1_content_sha256(
        "first\nsecond", {"b": False, "a": 1})
    assert semantic_sha256({"kind": "state"}) == hashlib.sha256(b'{"kind":"state"}').hexdigest()


@pytest.mark.parametrize("payload", [{"x": float("nan")}, {"x": float("inf")}, {1: "key"}, {"x": {1}}, {"x": "\ud800"}])
def test_json_rejects_nonfinite_values_and_lossy_non_json_types(payload):
    with pytest.raises(L1ContractError):
        canonical_json(payload)


@pytest.mark.parametrize("state", ["not_stated", "unresolved", "not_applicable"])
def test_unknown_is_null_and_never_silently_zero_or_false(state):
    validate_unknown_value(value(state, None), value_type="number", path="time")
    for bad in (0, False, ""):
        with pytest.raises(L1ContractError, match="unknown_value_must_be_null"):
            validate_unknown_value(value(state, bad), value_type="number", path="time")


def test_explicit_unbounded_requires_field_permission_and_evidence():
    data = value("unbounded", None, ["e1"])
    with pytest.raises(L1ContractError):
        validate_unknown_value(data, value_type="string", path="end")
    validate_unknown_value(data, value_type="string", path="end", allow_unbounded=True)
    with pytest.raises(L1ContractError, match="explicit_unbounded_evidence_required"):
        validate_unknown_value(value("unbounded", None), value_type="string", path="end", allow_unbounded=True)


@pytest.mark.parametrize("type_name,bad", [("number", True), ("number", float("inf")), ("integer", 1.0), ("boolean", 1), ("string", None)])
def test_known_value_checks_type_instead_of_coercion(type_name, bad):
    with pytest.raises(L1ContractError):
        validate_unknown_value(value(content=bad), value_type=type_name, path="field")


def test_custom_catalog_is_used_without_changing_builtin_classification():
    validate_unknown_value(value(content="owner_custom_kind"), value_type="string", path="kind",
                           allowed_values={"owner_custom_kind"})
    with pytest.raises(L1ContractError, match="unknown_catalog_value"):
        validate_unknown_value(value(content="owner_custom_kind"), value_type="string", path="kind", allowed_values={"state"})


def test_unicode_codepoint_anchor_preserves_raw_crlf_and_astral_character():
    text = "甲😀乙\r\n尾"
    params = dict(target_namespace="l0", target_id="diary-1", target_revision="v1", text_view_id="raw-v1")
    anchor = make_text_anchor(text, start=1, end=2, **params)
    assert anchor["quote_sha256"] == hashlib.sha256("😀".encode()).hexdigest()
    verify_text_anchor(text, anchor, **params)
    with pytest.raises(L1ContractError):
        verify_text_anchor(text.replace("\r\n", "\n"), anchor, **params)


def test_anchor_rejects_changed_source_revision_and_text_even_if_quote_survives():
    params = dict(target_namespace="l0", target_id="diary-1", target_revision=1, text_view_id="raw")
    anchor = make_text_anchor("same quote tail", start=0, end=4, **params)
    with pytest.raises(L1ContractError):
        verify_text_anchor("same quote other tail", anchor, **params)
    with pytest.raises(L1ContractError):
        verify_text_anchor("same quote tail", anchor, **{**params, "target_revision": 2})
    forged = {**anchor, "target_revision": True}
    with pytest.raises(L1ContractError):
        verify_text_anchor("same quote tail", forged, **params)


@pytest.mark.parametrize("start,end", [(True, 2), (0, False), (-1, 1), (1, 1), (0, 20)])
def test_anchor_rejects_bad_offsets(start, end):
    with pytest.raises(L1ContractError):
        make_text_anchor("text", target_namespace="l1", target_id="m1", target_revision=1,
                         text_view_id="body-v1", start=start, end=end)


@pytest.mark.parametrize("namespace", [[], None, "l2", ""])
def test_anchor_rejects_bad_namespace_with_a_structured_error(namespace):
    with pytest.raises(L1ContractError):
        make_text_anchor("text", target_namespace=namespace, target_id="m1", target_revision=1,
                         text_view_id="body-v1", start=0, end=1)
