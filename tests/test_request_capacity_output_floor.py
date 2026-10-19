"""Output-only overflow remains provable when input token cost is unknown."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core" / "request_capacity.py"
if not SOURCE.exists():
    SOURCE = ROOT / "request_capacity.py"
SPEC = importlib.util.spec_from_file_location("capacity_output_floor", SOURCE)
CAPACITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPACITY)


def limits(**changes):
    return {"model": "target", "context_window": 5000,
            "reserved_output_tokens": 1000, **changes}


def opaque_payload(**changes):
    return {"model": "target", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://offline.invalid/image"}}]}],
        **changes}


@pytest.mark.parametrize("contract,payload", [
    (limits(reserved_output_tokens=5001), opaque_payload()),
    (limits(), opaque_payload(max_tokens=5001)),
    (limits(), opaque_payload(max_completion_tokens=5001)),
    (limits(reserved_output_tokens=5001), opaque_payload(max_tokens=None)),
    (limits(reserved_output_tokens=5001), {"model": "target", "messages": [],
        "tools": [{"type": "web_search"}]}),
    (limits(reserved_output_tokens=5001), {"model": "target", "messages": [
        {"role": "user", "content": float("nan")}]})])
def test_proven_output_overflow_precedes_input_unknown(contract, payload):
    result = CAPACITY.assess_capacity(payload, contract)
    assert result["state"] == "exceeded"
    assert result["reason"] == "output_reservation_exceeds_context_window"
    assert result["dispatch_allowed"] is False


@pytest.mark.parametrize("attachments", [False, True])
def test_equal_output_and_window_keeps_existing_input_unknown_policy(attachments):
    result = CAPACITY.assess_capacity(opaque_payload(),
        limits(reserved_output_tokens=5000), has_attachments=attachments)
    assert result["state"] == "capacity_unverified"
    assert result["dispatch_allowed"] is (not attachments)


@pytest.mark.parametrize("contract", [
    limits(context_window=None, reserved_output_tokens=5001),
    limits(model="different", reserved_output_tokens=5001),
    limits(reserved_output_tokens=5001, contract_error="invalid")])
def test_missing_or_untrusted_window_does_not_prove_overflow(contract):
    result = CAPACITY.assess_capacity(opaque_payload(), contract)
    assert result["state"] == "capacity_unverified"
    assert result["dispatch_allowed"] is True


def test_known_complete_request_scoring_still_counts_input_plus_reserve():
    payload = {"model": "target", "messages": [{"role": "user", "content": "hello"}]}
    result = CAPACITY.assess_capacity(payload, limits(reserved_output_tokens=5000))
    assert result["state"] == "exceeded"
    assert result["reason"] == "complete_request_and_output_reservation_exceed_capacity"
    assert CAPACITY.assess_capacity(payload, limits())["state"] == "within_capacity"
