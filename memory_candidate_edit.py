"""Owner-only candidate revisions. Saving never accepts or commits Memory."""
from __future__ import annotations

import copy
import math
from datetime import date

from memory_authority import (CandidateNotFound, IdempotencyConflict,
                              InvalidTransition, MemoryProposal, RevisionConflict)

FIELDS = {"title", "content", "kind", "tags", "domain", "importance", "confidence", "event_date"}
KINDS = {"preference", "boundary", "commitment", "shared_experience", "key_event",
         "reflection", "project_state", "identity", "durable_fact", "memory"}


def owner_edit_payload(row, edit):
    if not isinstance(edit, dict) or not edit or set(edit) - FIELDS:
        raise ValueError("invalid_owner_edit")
    payload = copy.deepcopy(row["proposal"])
    if payload.get("source_type") not in {"daily_chat", "chat_tool"}:
        raise ValueError("unsupported_candidate_source")
    metadata = payload.setdefault("metadata", {})
    legacy = metadata.setdefault("legacy_candidate", {})
    narrative = metadata.setdefault("narrative", copy.deepcopy(legacy.get("narrative") or {}))
    for key, limit in (("title", 100), ("content", 12000)):
        if key in edit:
            value = edit[key]
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError("invalid_" + key)
            value = value.strip()
            if key == "title":
                legacy["title"] = value
            else:
                # Only an actual body change invalidates its old time evidence.
                if value != payload["proposed_body"]:
                    narrative.update(edited_by="owner", event_time={
                        "precision": "unknown", "basis": "owner_edit_requires_time_review"})
                    legacy["soft_flags"] = [f for f in legacy.get("soft_flags", [])
                                            if f not in {"excerpt_overlap", "needs_owner_edit"}]
                payload["proposed_body"] = value
    if "kind" in edit:
        if not isinstance(edit["kind"], str) or edit["kind"] not in KINDS:
            raise ValueError("invalid_memory_kind")
        payload["memory_type"] = edit["kind"]
    for key in ("tags", "domain"):
        if key in edit:
            value = edit[key]
            if (not isinstance(value, list) or len(value) > 12
                    or any(not isinstance(v, str) or not v.strip() or len(v) > 100 for v in value)):
                raise ValueError("invalid_" + key)
            legacy[key] = list(dict.fromkeys(v.strip() for v in value))
    if "importance" in edit:
        if type(edit["importance"]) is not int or not 1 <= edit["importance"] <= 10:
            raise ValueError("invalid_importance")
        legacy["importance"] = edit["importance"]
    if "confidence" in edit:
        value = edit["confidence"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid_confidence")
        payload["confidence"] = value
    if "event_date" in edit:
        value = edit["event_date"]
        if not isinstance(value, str):
            raise ValueError("invalid_event_date")
        if value:
            if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
                raise ValueError("invalid_event_date")
            narrative["event_time"] = {"precision": "day", "value": value, "basis": "owner_edit"}
        else:
            narrative["event_time"] = {"precision": "unknown", "basis": "owner_edit"}
        narrative["edited_by"] = "owner"
    legacy.update(content=payload["proposed_body"], proposed_memory=payload["proposed_body"],
                  kind=payload["memory_type"], confidence=payload["confidence"],
                  narrative=copy.deepcopy(narrative))
    return payload


def save_owner_candidate(store, candidate_id, *, edit, expected_revision, request_id):
    if (type(expected_revision) is not int or expected_revision < 1
            or not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 160):
        raise ValueError("revision_and_request_id_required")
    fingerprint = store.fingerprint({"candidate_id": candidate_id, "edit": edit,
                                     "expected_revision": expected_revision})
    decision_key = "owner-candidate-edit:" + request_id
    prior = store.get_candidate_decision(decision_key)
    if prior:
        result = prior["result"]
        if result["proposal"]["metadata"].get("owner_edit_request_hash") != fingerprint:
            raise IdempotencyConflict("edit request payload differs")
        return result
    row = store.get_candidate(candidate_id)
    if not row:
        raise CandidateNotFound(candidate_id)
    if row["revision"] != expected_revision:
        raise RevisionConflict("candidate revision changed")
    if row["status"] not in {"pending", "deferred"}:
        raise InvalidTransition("only uncommitted pending candidates can be edited")
    payload = owner_edit_payload(row, edit)
    payload["metadata"]["owner_edit_request_hash"] = fingerprint
    return store.revise_candidate(candidate_id, expected_revision=expected_revision,
        proposal=MemoryProposal.from_mapping(payload), request_id=decision_key, actor="owner")
