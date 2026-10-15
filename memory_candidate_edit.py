"""Owner-only candidate revisions. Saving never accepts or commits Memory."""
from __future__ import annotations

import copy
import math
from datetime import date

from memory_authority import (CandidateNotFound, IdempotencyConflict,
                              InvalidTransition, MemoryProposal, RevisionConflict)

FIELDS = {"title", "content", "kind", "tags", "domain", "importance", "confidence", "event_date"}
L1_FIELDS = {"document", "scene_bodies"}
KINDS = {"preference", "boundary", "commitment", "shared_experience", "key_event",
         "reflection", "project_state", "identity", "durable_fact", "memory"}


def _owner_l1_edit_payload(row, edit, *, revision_service, request_id):
    """Apply one complete owner L1 document while preserving its managed bindings.

    Authority candidate revision is the outer CAS.  Diary's root revision lives
    separately in ``diary_l1_bundle.expected_revision`` and is used only to stamp
    the prospective L1 revision passed to the shared owner-document validator.
    """
    if revision_service is None:
        raise ValueError("l1_owner_service_required")
    if set(edit) - L1_FIELDS or not isinstance(edit.get("document"), dict):
        raise ValueError("invalid_l1_owner_edit")

    from memory_l1_schema import join_document, split_document

    payload = copy.deepcopy(row["proposal"])
    metadata = payload.setdefault("metadata", {})
    semantic = metadata.get("memory_l1")
    if not isinstance(semantic, dict):
        raise ValueError("l1_document_unavailable")
    try:
        previous = join_document(str(payload.get("proposed_body") or ""), semantic)
    except Exception as exc:
        raise ValueError("l1_candidate_invalid") from exc

    source_type = str(payload.get("source_type") or "")
    bundle = metadata.get("diary_l1_bundle")
    if source_type == "diary":
        if not isinstance(bundle, dict):
            raise ValueError("diary_l1_bundle_unavailable")
        memory_id = str(bundle.get("memory_id") or "").strip()
        base_revision = bundle.get("expected_revision")
        if not memory_id or type(base_revision) is not int or base_revision < 0:
            raise ValueError("diary_l1_root_revision_invalid")
    else:
        memory_id = str(row.get("candidate_id") or payload.get("proposal_id") or "").strip()
        base_revision = 0
        if not memory_id:
            raise ValueError("candidate_memory_id_missing")

    catalog, catalog_revision = revision_service.catalog_provider()
    submitted = copy.deepcopy(edit["document"])
    normalized = revision_service._owner_document(
        submitted,
        previous,
        memory_id=memory_id,
        request_id=request_id,
        catalog=catalog,
        catalog_revision=catalog_revision,
        revision=base_revision + 1,
    )
    body, stored_semantic = split_document(
        normalized,
        catalog=normalized.get("generation", {}).get("catalog_snapshot"),
    )

    submitted_scenes = submitted.get("scenes")
    normalized_scenes = normalized.get("scenes")
    if not isinstance(submitted_scenes, list) or not isinstance(normalized_scenes, list):
        raise ValueError("invalid_l1_scenes")
    scene_id_map = {
        str(before.get("scene_id")): str(after.get("scene_id"))
        for before, after in zip(submitted_scenes, normalized_scenes)
        if isinstance(before, dict) and isinstance(after, dict)
    }

    if source_type == "diary":
        prior_scene_bodies = bundle.get("scene_bodies") or {}
        requested_scene_bodies = edit.get("scene_bodies", prior_scene_bodies)
        if not isinstance(requested_scene_bodies, dict):
            raise ValueError("invalid_scene_bodies")
        allowed_scene_ids = {str(scene.get("scene_id")) for scene in submitted_scenes
                             if isinstance(scene, dict)}
        if any(str(scene_id) not in allowed_scene_ids for scene_id in requested_scene_bodies):
            raise ValueError("unknown_scene_body")
        from memory_l1_contract import canonical_l1_body
        mapped_scene_bodies = {}
        for scene_id, scene_body in requested_scene_bodies.items():
            if not isinstance(scene_body, str):
                raise ValueError("invalid_scene_body")
            mapped_scene_bodies[scene_id_map.get(str(scene_id), str(scene_id))] = canonical_l1_body(scene_body)
        bundle["scene_bodies"] = mapped_scene_bodies
        metadata["diary_l1_bundle"] = bundle
    elif "scene_bodies" in edit:
        raise ValueError("scene_bodies_only_supported_for_diary")

    payload["proposed_body"] = body
    metadata["memory_l1"] = stored_semantic
    metadata["name"] = stored_semantic["narrative"]["title"]
    legacy = metadata.setdefault("legacy_candidate", {})
    legacy.update({
        "title": stored_semantic["narrative"]["title"],
        "content": body,
        "proposed_memory": body,
    })
    # The complete L1 document supersedes the old bounded annotation projection.
    # Removing it prevents legacy readers from presenting stale semantics.
    legacy.pop("semantic_annotations", None)
    metadata.pop("semantic_annotations", None)
    return payload


def owner_edit_payload(row, edit, *, revision_service=None, request_id=""):
    if not isinstance(edit, dict) or not edit or set(edit) - (FIELDS | L1_FIELDS):
        raise ValueError("invalid_owner_edit")
    payload = copy.deepcopy(row["proposal"])
    if payload.get("source_type") not in {"daily_chat", "chat_tool", "diary", "action_receipt"}:
        raise ValueError("unsupported_candidate_source")
    has_l1 = isinstance((payload.get("metadata") or {}).get("memory_l1"), dict)
    if has_l1:
        if "document" not in edit:
            raise ValueError("l1_document_required")
        return _owner_l1_edit_payload(
            row, edit, revision_service=revision_service, request_id=request_id
        )
    if "document" in edit or "scene_bodies" in edit:
        raise ValueError("l1_document_unavailable")

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


def save_owner_candidate(store, candidate_id, *, edit, expected_revision, request_id,
                         revision_service=None):
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
    payload = owner_edit_payload(
        row, edit, revision_service=revision_service, request_id=request_id
    )
    payload["metadata"]["owner_edit_request_hash"] = fingerprint
    return store.revise_candidate(candidate_id, expected_revision=expected_revision,
        proposal=MemoryProposal.from_mapping(payload), request_id=decision_key, actor="owner")
