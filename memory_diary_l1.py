"""Diary whole/scene organization around the existing Proposal/Policy/Commit.

The caller supplies an actual first-pass task result. This component never
copies raw prose and labels it a model result, and never calls a model itself.
"""
from __future__ import annotations

import copy
import hashlib
import math

from memory_authority import IdempotencyConflict, MemoryProposal, PolicyDecision, RevisionConflict
from memory_l1_contract import canonical_json, canonical_l1_body, semantic_sha256, verify_text_anchor
from memory_l1_index import scene_memory_id
from memory_l1_schema import document_template, generation_template, split_document
from memory_l1_service import L1OwnerError


def input_coverage(length, ranges):
    if not isinstance(ranges, list): raise L1OwnerError("input_ranges_required", field_path="input_ranges")
    intervals = []
    for index, value in enumerate(ranges):
        if (not isinstance(value, dict) or set(value) != {"start", "end"}
                or type(value["start"]) is not int or type(value["end"]) is not int
                or not 0 <= value["start"] < value["end"] <= length):
            raise L1OwnerError("invalid_codepoint_range", field_path=f"input_ranges[{index}]")
        intervals.append((value["start"], value["end"]))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], end)
        else: merged.append([start, end])
    missing, cursor = [], 0
    for start, end in merged:
        if cursor < start: missing.append({"start": cursor, "end": start})
        cursor = end
    if cursor < length: missing.append({"start": cursor, "end": length})
    return {"coordinate_unit": "unicode_codepoint", "source_length": length,
            "input_ranges": [{"start": a, "end": b} for a, b in merged], "missing_ranges": missing,
            "state": "complete" if not missing else "partial"}


class DiaryL1Service:
    def __init__(self, revisions, sources, policy):
        self.revisions, self.sources, self.policy = revisions, sources, policy
        self.authority, self.commit = revisions.authority, revisions.commit

    async def prepare(self, *, day, document, scene_bodies, generation, input_ranges, request_id,
                      confidence=0.0, requested_mode="review"):
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
            raise L1OwnerError("invalid_request_id", field_path="request_id")
        if not isinstance(scene_bodies, dict): raise L1OwnerError("object_required", field_path="scene_bodies")
        if not isinstance(generation, dict) or set(generation) != {"prompt_revision", "model_route_revision", "task_id", "source_revision"}:
            raise L1OwnerError("generation_context_required", field_path="generation")
        if any(not isinstance(value, str) or not value for value in generation.values()):
            raise L1OwnerError("generation_context_required", field_path="generation")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise L1OwnerError("invalid_confidence", field_path="confidence")
        if requested_mode not in ("auto", "review"):
            raise L1OwnerError("invalid_mode", field_path="requested_mode")
        captured = await self.sources.capture_diary(day, binding_id="diary-source")
        source, binding = captured["source"], captured["binding"]
        if binding is None: raise L1OwnerError("diary_source_unavailable", status=409, field_path="source")
        if generation["source_revision"] != source["source_revision"]:
            raise L1OwnerError("diary_source_changed_during_generation", status=409, field_path="generation.source_revision")
        memory_id = "diary_full_" + hashlib.sha256(source["source_id"].encode("utf-8")).hexdigest()[:32]
        candidate_id = "diary-l1:" + memory_id + ":" + request_id
        request_hash = semantic_sha256({"day": day, "document": document, "scene_bodies": scene_bodies,
            "generation": generation, "input_ranges": input_ranges, "confidence": confidence, "requested_mode": requested_mode})
        prior_candidate = self.authority.get_candidate(candidate_id)
        if prior_candidate:
            if prior_candidate["proposal"]["metadata"].get("diary_request_sha256") != request_hash:
                raise IdempotencyConflict("diary request id has different task result")
            return prior_candidate
        memory = self.authority.get_memory(memory_id)
        if memory and memory["state"] != "active":
            raise L1OwnerError("diary_root_not_active", status=409, field_path="memory.state")
        expected = memory["active_revision"] if memory else 0
        catalog, catalog_revision = self.revisions.catalog_provider()
        prior = (await self.revisions.detail(memory_id))["l1"]["document"] if memory else document_template(catalog=catalog, catalog_revision=catalog_revision)
        doc = copy.deepcopy(document)
        if not isinstance(doc, dict): raise L1OwnerError("object_required", field_path="document")
        for module, key in (("scenes", "scene_id"), ("statements", "statement_id"), ("relations", "relation_id")):
            if (not isinstance(doc.get(module), list)
                    or any(not isinstance(item, dict) or not isinstance(item.get(key), str) for item in doc[module])):
                raise L1OwnerError("invalid_member_array", field_path="document." + module)
        local_scenes = [s["scene_id"] for s in doc.get("scenes", [])]
        if set(scene_bodies) - set(local_scenes): raise L1OwnerError("unknown_scene_body", field_path="scene_bodies")
        base = copy.deepcopy(prior)
        base["evidence_bindings"] = [binding]
        base["entry"] = {"form": "diary_full", "parent_ref": None, "document_group_id": memory_id,
                         "derived_from_revision": expected or None, "scene_ref": None, "shared_statement_refs": []}
        doc["entry"] = copy.deepcopy(base["entry"])
        doc["evidence_bindings"] = copy.deepcopy(base["evidence_bindings"])
        doc["generation"] = copy.deepcopy(base["generation"])
        for module, key, revision_key in (("scenes", "scene_id", "scene_revision_id"),
                                          ("statements", "statement_id", "statement_revision_id")):
            old = {item[key]: item for item in prior.get(module, [])}
            for item in doc.get(module, []):
                previous_item = old.get(item[key])
                item[revision_key] = previous_item[revision_key] if previous_item else None
                if module == "scenes":
                    item["generation"] = copy.deepcopy(previous_item["generation"] if previous_item else base["generation"])
        prior_relations = {item["relation_id"]: item for item in prior.get("relations", [])}
        for item in doc.get("relations", []):
            if item.get("relation_id") in prior_relations:
                item["origin"] = prior_relations[item["relation_id"]]["origin"]
        # Identity assignment is shared with the owner editor. Provenance is
        # then stamped from the actual task context, never model JSON.
        normalized = self.revisions._owner_document(doc, base, memory_id=memory_id, request_id=request_id,
            catalog=catalog, catalog_revision=catalog_revision, revision=expected + 1)
        gen = generation_template(normalized["generation"]["catalog_snapshot"], catalog_revision, "first_pass_model")
        gen.update({key: generation[key] for key in ("prompt_revision", "model_route_revision", "task_id")}); gen["actor"] = "diary_first_pass"
        normalized["generation"] = gen
        for relation in normalized["relations"]:
            previous_relation = prior_relations.get(relation["relation_id"])
            semantic = {key: value for key, value in relation.items() if key not in ("origin", "confirmation_state")}
            previous_semantic = {key: value for key, value in (previous_relation or {}).items() if key not in ("origin", "confirmation_state")}
            if not previous_relation or semantic != previous_semantic:
                relation["origin"] = "model"
                if relation["confirmation_state"] != "unresolved": relation["confirmation_state"] = "proposed"
            else:
                relation["origin"] = previous_relation["origin"]
                relation["confirmation_state"] = previous_relation["confirmation_state"]
        body_map = {}
        for local, scene in zip(local_scenes, normalized["scenes"]):
            # Every boundary must bind an actual source or this exact L1 view.
            for anchor in scene["anchors"]:
                if anchor["target_namespace"] == "l0":
                    verify_text_anchor(source["text"], anchor, target_namespace="l0", target_id=source["source_id"],
                        target_revision=source["source_revision"], text_view_id=source["text_view_id"])
                else:
                    verify_text_anchor(canonical_l1_body(normalized["narrative"]["body"]), anchor, target_namespace="l1",
                        target_id=memory_id, target_revision=expected + 1, text_view_id="memory.narrative.canonical-v1")
            old_scene = next((s for s in prior.get("scenes", []) if s["scene_id"] == scene["scene_id"]), None)
            if not old_scene or scene["scene_revision_id"] != old_scene["scene_revision_id"]:
                scene["generation"] = copy.deepcopy(gen)
            if local in scene_bodies: body_map[scene["scene_id"]] = canonical_l1_body(scene_bodies[local])
        body, semantic = split_document(normalized)
        coverage = input_coverage(len(source["text"]), input_ranges)
        current_record = self.authority.get_memory_revision(memory_id, expected) if expected else None
        metadata = copy.deepcopy(current_record["metadata"]) if current_record else {}
        metadata.pop("memory_l1_owner_request", None)
        metadata.update({"name": normalized["narrative"]["title"], "memory_l1": semantic,
                    "diary_request_sha256": request_hash,
                    "diary_l1_bundle": {"memory_id": memory_id, "expected_revision": expected,
                        "scene_bodies": body_map, "coverage": coverage, "source_binding": binding}})
        # An incomplete source input or a previously owner-edited root stays
        # on the existing Review path; it cannot silently replace owner work.
        review_reasons = []
        if coverage["state"] != "complete": review_reasons.append("REVIEW_PARTIAL_DIARY_INPUT")
        if current_record and current_record["created_by"] == "owner": review_reasons.append("REVIEW_OWNER_EDITED_ROOT")
        proposal = MemoryProposal(proposal_id=candidate_id, source_type="diary", proposed_body=body,
            original_excerpt="", source_refs=("source_snapshot:" + semantic_sha256(binding),), source_status="verified",
            memory_type="diary_memory", confidence=confidence, requested_mode=requested_mode, metadata=metadata)
        decision = self.policy.evaluate(proposal)
        if review_reasons:
            decision = PolicyDecision("queue_review", tuple(review_reasons), "review", requires_owner_confirmation=True)
        return self.authority.put_candidate(proposal, decision)

    async def adopt(self, candidate_id):
        candidate = self.authority.get_candidate(candidate_id)
        if not candidate: raise L1OwnerError("candidate_not_found", status=404)
        if candidate["status"] not in ("accepted", "committed"):
            return {"status": candidate["status"], "candidate_id": candidate_id, "root": None, "scenes": []}
        proposal = candidate["proposal"]
        metadata = copy.deepcopy(proposal["metadata"])
        bundle = copy.deepcopy(metadata.get("diary_l1_bundle"))
        if not isinstance(bundle, dict): raise L1OwnerError("diary_bundle_required", field_path="candidate")
        memory_id, expected = bundle["memory_id"], bundle["expected_revision"]
        idem = candidate_id + ":root"
        original = self.authority.get_commit_by_idempotency(idem)
        if original and original["status"] == "committed":
            root = original["result"]
        else:
            source = await self.sources.locate(bundle["source_binding"])
            if not source["eligible_source"]: raise L1OwnerError("source_not_currently_readable", status=409, field_path="source")
            current = self.authority.get_memory(memory_id)
            if current and current["state"] != "active":
                raise L1OwnerError("diary_root_not_active", status=409, field_path="memory.state")
            if (current["active_revision"] if current else 0) != expected: raise RevisionConflict("diary root changed after proposal")
            # The candidate is the prepared task record. Keep prose there for
            # per-scene retry, not as a second editable copy in the root.
            scene_bodies = metadata["diary_l1_bundle"].pop("scene_bodies")
            metadata["diary_l1_bundle"]["scene_body_sha256"] = {key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in scene_bodies.items()}
            root = await self.commit.commit_memory(memory_id=memory_id, bucket_id=memory_id,
                expected_revision=expected, body=proposal["proposed_body"], metadata=metadata,
                source_refs=proposal["source_refs"], decision_source="diary_first_pass", idempotency_key=idem,
                actor="diary_first_pass", l1_write_intent=True)
        root_revision = root["revision"]
        current = self.authority.get_memory(memory_id)
        if current["active_revision"] != root_revision or current["state"] != "active":
            return {"status": "superseded", "candidate_id": candidate_id, "root": root, "scenes": []}
        document = (await self.revisions.detail(memory_id, root_revision))["l1"]["document"]
        results = []
        for scene in document["scenes"]:
            child_id = scene_memory_id(memory_id, scene["scene_id"])
            child_idem = candidate_id + ":scene:" + scene["scene_id"]
            prior_commit = self.authority.get_commit_by_idempotency(child_idem)
            if prior_commit and prior_commit["status"] == "committed":
                results.append({"scene_id": scene["scene_id"], "state": "ready", **prior_commit["result"]})
                continue
            body = bundle["scene_bodies"].get(scene["scene_id"])
            if body is None:
                results.append({"scene_id": scene["scene_id"], "state": "not_prepared", "memory_id": child_id})
                continue
            try:
                source = await self.sources.locate(bundle["source_binding"])
                if not source["eligible_source"]: raise L1OwnerError("source_not_currently_readable", status=409)
                current_root = self.authority.get_memory(memory_id)
                if current_root["active_revision"] != root_revision or current_root["state"] != "active":
                    raise L1OwnerError("diary_root_dependency_changed", status=409)
                existing = self.authority.get_memory(child_id)
                child_expected = existing["active_revision"] if existing else 0
                previous = self.authority.get_memory_revision(child_id, child_expected) if existing else None
                if previous and previous["created_by"] == "owner":
                    raise L1OwnerError("owner_scene_requires_review", status=409)
                child = document_template(body, scene["title"], catalog=document["generation"]["catalog_snapshot"],
                                          catalog_revision=document["generation"]["catalog_revision"])
                child["entry"] = {"form": "diary_scene", "parent_ref": {"memory_id": memory_id, "revision": root_revision},
                    "document_group_id": memory_id, "derived_from_revision": root_revision,
                    "scene_ref": {"scene_id": scene["scene_id"], "scene_revision_id": scene["scene_revision_id"]},
                    "shared_statement_refs": list(scene["statement_refs"])}
                child["generation"] = copy.deepcopy(document["generation"])
                body, semantic = split_document(child)
                child_metadata = copy.deepcopy(previous["metadata"]) if previous else {}
                child_metadata.pop("memory_l1_owner_request", None)
                child_metadata.update({"name": scene["title"], "memory_l1": semantic})
                committed = await self.commit.commit_memory(memory_id=child_id, bucket_id=child_id,
                    expected_revision=child_expected, body=body, metadata=child_metadata,
                    source_refs=proposal["source_refs"], decision_source="diary_scene", idempotency_key=child_idem,
                    actor="diary_first_pass", l1_write_intent=True)
                self.authority.set_projection_status(memory_id=memory_id, memory_revision=root_revision,
                    projector="l1_scene:" + scene["scene_id"], status="projected", source_sha256=root["l1_content_sha256"],
                    details={"state": "ready", "memory_id": child_id, "revision": committed["revision"]})
                results.append({"scene_id": scene["scene_id"], "state": "ready", **committed})
            except Exception as exc:
                self.authority.set_projection_status(memory_id=memory_id, memory_revision=root_revision,
                    projector="l1_scene:" + scene["scene_id"], status="degraded", source_sha256=root["l1_content_sha256"],
                    details={"state": "failed", "memory_id": child_id, "error_code": getattr(exc, "code", type(exc).__name__)})
                results.append({"scene_id": scene["scene_id"], "state": "failed", "memory_id": child_id,
                                "issue": {"code": getattr(exc, "code", type(exc).__name__)}})
        current_candidate = self.authority.get_candidate(candidate_id)
        if current_candidate["status"] == "accepted":
            self.authority.decide_candidate(candidate_id, action="commit", expected_revision=current_candidate["revision"],
                request_id=candidate_id + ":committed", actor="commit_service", reason_codes=["DIARY_ROOT_COMMITTED"])
        return {"status": "complete" if all(r["state"] == "ready" for r in results) and bundle["coverage"]["state"] == "complete" else "partial",
                "candidate_id": candidate_id, "root": root, "scenes": results, "coverage": bundle["coverage"]}
