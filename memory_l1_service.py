"""Owner L1 revision service over the existing Authority and commit coordinator."""
from __future__ import annotations

import copy
from pathlib import Path
import uuid

import frontmatter

from memory_authority import CommitStateError, IdempotencyConflict, MemoryNotFound, RevisionConflict
from memory_commit_service import body_sha256
from memory_l1_contract import L1ContractError, canonical_json, semantic_sha256
from memory_l1_schema import (catalog_with_defaults, document_template, generation_template,
    join_document, split_document, statement_template, scene_template, validate_l1_document, verify_l1_metadata)


class L1OwnerError(ValueError):
    def __init__(self, code, *, status=400, field_path=""):
        super().__init__(code)
        self.code, self.status, self.field_path = code, status, field_path


class MemoryL1RevisionService:
    def __init__(self, authority, commit_service, bucket_manager, catalog_provider):
        self.authority = authority
        self.commit = commit_service
        self.buckets = bucket_manager
        self.catalog_provider = catalog_provider

    def schema(self):
        catalog, revision = self.catalog_provider()
        from memory_l1_wire_schema import MODEL_DOCUMENT_SCHEMA, tool_parameters
        return {"status": "ok", "schema_version": "memory-l1-v1", "catalog_revision": revision, "catalog": catalog_with_defaults(catalog),
                "model_document_schema": MODEL_DOCUMENT_SCHEMA, "native_tool_schema": tool_parameters(),
                "template": document_template(catalog=catalog, catalog_revision=revision),
                "templates": {"statement": statement_template(),
                              "scene": scene_template(catalog, revision),
                              "entity": {"entity_id": "new:entity", "entity_type": {"state": "not_stated", "value": None, "evidence_refs": []}, "name": "", "aliases": []}},
                "managed_fields": ["generation", "evidence_bindings", "entry.parent_ref", "entry.document_group_id", "entry.derived_from_revision",
                                   "entry.scene_ref", "entry.shared_statement_refs",
                                   "entities.*.entity_id", "statements.*.statement_id", "statements.*.statement_revision_id",
                                   "scenes.*.scene_id", "scenes.*.scene_revision_id", "scenes.*.generation", "relations.*.relation_id", "relations.*.origin"],
                "limits": {"request_bytes": 524288, "request_id_characters": 128}}

    async def detail(self, memory_id, revision=None):
        memory = self.authority.get_memory(memory_id)
        if not memory: raise MemoryNotFound(memory_id)
        selected = int(memory["active_revision"]) if revision is None else revision
        if type(selected) is not int or selected <= 0: raise L1OwnerError("invalid_revision", field_path="revision")
        record = self.authority.get_memory_revision(memory_id, selected)
        if not record: raise MemoryNotFound(f"{memory_id}@{selected}")
        metadata = record["metadata"]
        snapshot = Path(record["snapshot_path"])
        expected = self.commit.projection.snapshot_path(memory_id, selected)
        if snapshot.resolve() != expected.resolve() or not expected.resolve().is_relative_to(self.commit.projection.revisions_dir):
            raise L1OwnerError("revision_snapshot_path_mismatch", status=503)
        if snapshot.is_file():
            post = frontmatter.loads(snapshot.read_text(encoding="utf-8"))
            body, observed = str(post.content), dict(post.metadata)
        elif selected == int(memory["active_revision"]) and "memory_l1" not in metadata:
            bucket = await self.buckets.get(memory["bucket_id"])
            body, observed = str((bucket or {}).get("content") or ""), dict((bucket or {}).get("metadata") or {})
        else:
            raise L1OwnerError("revision_snapshot_unavailable", status=503)
        if body_sha256(body) != record["body_sha256"]:
            raise L1OwnerError("revision_body_mismatch", status=503)
        catalog, config_revision = self.catalog_provider()
        if "memory_l1" in metadata:
            try:
                verify_l1_metadata(body, metadata, observed)
                document = join_document(body, metadata["memory_l1"])
                state, issue = "current", None
            except L1ContractError as exc:
                # Preserve inspectable original structure; never replace with an empty template.
                document = copy.deepcopy(metadata["memory_l1"])
                if isinstance(document, dict) and isinstance(document.get("narrative"), dict): document["narrative"]["body"] = body
                state, issue = "invalid", {"code": exc.code, "field_path": exc.path}
        else:
            document = document_template(body, str(metadata.get("name") or ""), catalog=catalog, catalog_revision=config_revision)
            state, issue = "legacy", None
        return {"status": "ok", "memory": memory, "body": body,
                "requested_revision": selected, "current_revision": int(memory["active_revision"]),
                "revisions": self.authority.list_memory_revisions(memory_id, limit=200),
                "rings": self.authority.list_memory_rings(memory_id, state="all", limit=200),
                "projection_status": self.authority.list_projection_status(memory_id),
                "l1": {"state": state, "document": document, "issue": issue,
                       "coverage": (metadata.get("diary_l1_bundle") or {}).get("coverage"),
                       "semantic_sha256": metadata.get("memory_l1_semantic_sha256"),
                       "l1_content_sha256": metadata.get("memory_l1_content_sha256"),
                       "preparation": {"l2": {"state": "tracked" if state == "current" else "not_prepared",
                                              "status_url": "/api/memory-authority/memories/" + memory_id + "/presentation"}}}}

    def _owner_document(self, submitted, previous, *, memory_id, request_id, catalog, catalog_revision, revision):
        doc = copy.deepcopy(submitted)
        if not isinstance(doc, dict): raise L1OwnerError("object_required", field_path="document")
        for key in ("generation", "evidence_bindings"):
            if doc.get(key) != previous.get(key): raise L1OwnerError("managed_field_changed", field_path="document." + key)
        if not isinstance(doc.get("entry"), dict):
            raise L1OwnerError("object_required", field_path="document.entry")
        for key in ("parent_ref", "document_group_id", "derived_from_revision", "scene_ref", "shared_statement_refs"):
            if doc.get("entry", {}).get(key) != previous.get("entry", {}).get(key): raise L1OwnerError("managed_field_changed", field_path="document.entry." + key)
        # Keep the recorded vocabulary available for historical labels. This is
        # a per-revision snapshot, not a mutation of the active owner catalog.
        effective = catalog_with_defaults(catalog)
        previous_generation = previous.get("generation")
        historical = previous_generation.get("catalog_snapshot") if isinstance(previous_generation, dict) else {}
        historical = historical if isinstance(historical, dict) else {}
        for axis in ("subjects", "entity_types", "kinds", "topics", "stabilities", "modalities", "bases", "relation_types", "cognition_roles", "cognition_statuses"):
            effective[axis] = list(dict.fromkeys(effective[axis] + list(historical.get(axis, []))))
        existing_defs = {item["id"]: item for item in effective["attributes"]}
        # A catalogue edit must not silently reinterpret an existing field's
        # saved type, scope or enum values. This record retains its vocabulary;
        # newly introduced identifiers use the current catalogue.
        for item in historical.get("attributes", []): existing_defs[item["id"]] = item
        effective["attributes"] = list(existing_defs.values())
        current_descriptions = {(item["axis"], item["id"]): item for item in effective.get("definitions", [])}
        for item in historical.get("definitions", []): current_descriptions[(item["axis"], item["id"])] = item
        effective["definitions"] = list(current_descriptions.values())
        doc["generation"] = generation_template(effective, catalog_revision, "owner_edit")
        validate_l1_document(doc, catalog=effective)
        mapping = {}
        for module, key, prefix in (("entities", "entity_id", "en"), ("statements", "statement_id", "st"), ("scenes", "scene_id", "sc"), ("relations", "relation_id", "rl")):
            old = {item[key]: item for item in previous.get(module, [])}
            for index, item in enumerate(doc[module]):
                ident = item[key]
                prior = old.get(ident)
                version_key = {"statements": "statement_revision_id", "scenes": "scene_revision_id"}.get(module)
                if version_key and item[version_key] != (prior[version_key] if prior else None):
                    raise L1OwnerError("managed_field_changed", field_path=f"document.{module}[{index}].{version_key}")
                if prior:
                    if module == "scenes" and item["generation"] != prior["generation"]:
                        raise L1OwnerError("managed_field_changed", field_path=f"document.scenes[{index}].generation")
                    if module == "relations" and item["origin"] != prior["origin"]:
                        raise L1OwnerError("managed_field_changed", field_path=f"document.relations[{index}].origin")
                    continue
                if not ident.startswith("new:"): raise L1OwnerError("new_identity_requires_local_key", field_path="document." + module)
                if ident in mapping: raise L1OwnerError("duplicate_local_identity", field_path="document." + module)
                mapping[ident] = prefix + "_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{memory_id}:{request_id}:{module}:{ident}").hex
        for module, key in (("entities", "entity_id"), ("statements", "statement_id"), ("scenes", "scene_id"), ("relations", "relation_id")):
            for item in doc[module]: item[key] = mapping.get(item[key], item[key])
        for item in doc["statements"]:
            item["subject_ref"]["value"] = mapping.get(item["subject_ref"]["value"], item["subject_ref"]["value"])
            if item.get("cognition"):
                item["cognition"]["about_refs"] = [mapping.get(ref, ref) for ref in item["cognition"].get("about_refs", [])]
            app = item["time"]["applicability"]; app["scene_ref"] = mapping.get(app["scene_ref"], app["scene_ref"])
        for item in doc["scenes"]:
            item["statement_refs"] = [mapping.get(ref, ref) for ref in item["statement_refs"]]
        definitions = {item["id"]: item for item in effective["attributes"]}
        def remap_entity_values(node):
            if isinstance(node, list):
                for child in node: remap_entity_values(child)
            elif isinstance(node, dict):
                definition_id = node.get("definition_id", node.get("key"))
                if definitions.get(definition_id, {}).get("value_type") == "entity_ref":
                    value = node.get("value")
                    if isinstance(value, dict) and value.get("state") == "known":
                        value["value"] = mapping.get(value["value"], value["value"])
                if definitions.get(node.get("field"), {}).get("value_type") == "entity_ref":
                    value = node.get("value")
                    node["value"] = [mapping.get(v, v) for v in value] if node.get("op") == "in" else mapping.get(value, value)
                for child in node.values(): remap_entity_values(child)
        for key in ("attributes", "statements", "scenes"): remap_entity_values(doc[key])
        previous_relations = {item["relation_id"]: item for item in previous.get("relations", [])}
        for item in doc["relations"]:
            item["origin"] = previous_relations.get(item["relation_id"], {}).get("origin", "owner")
            for key in ("from", "to"):
                endpoint = item[key]
                if endpoint["memory_id"] in (None, memory_id):
                    endpoint["memory_id"] = memory_id
                    endpoint["object_id"] = mapping.get(endpoint["object_id"], endpoint["object_id"])
                    if endpoint["revision"] == "this_revision": endpoint["revision"] = revision
                target_memory = endpoint["memory_id"]
                if target_memory == memory_id and endpoint["revision"] == revision:
                    valid_ids = {"memory": {memory_id}, "statement": {s["statement_id"] for s in doc["statements"]},
                                 "scene": {s["scene_id"] for s in doc["scenes"]}}
                    if endpoint["object_id"] not in valid_ids[endpoint["object_type"]]:
                        raise L1OwnerError("dangling_relation_ref", field_path="document.relations")
                else:
                    if type(endpoint["revision"]) is not int:
                        raise L1OwnerError("invalid_relation_revision", field_path="document.relations")
                    target = self.authority.get_memory_revision(target_memory, endpoint["revision"])
                    if target is None: raise L1OwnerError("relation_revision_not_found", field_path="document.relations")
                    target_doc = target["metadata"].get("memory_l1") or {}
                    if endpoint["object_type"] == "memory": valid = endpoint["object_id"] == target_memory
                    else:
                        target_module, id_key = ("statements", "statement_id") if endpoint["object_type"] == "statement" else ("scenes", "scene_id")
                        valid = any(o[id_key] == endpoint["object_id"] for o in target_doc.get(target_module, []))
                    if not valid: raise L1OwnerError("relation_object_not_found", field_path="document.relations")
        stamped_generation = generation_template(effective, catalog_revision, "owner_edit")
        stamped_generation["actor"] = "owner"
        doc["generation"] = stamped_generation
        for module, key, version_key in (("statements", "statement_id", "statement_revision_id"), ("scenes", "scene_id", "scene_revision_id")):
            old = {item[key]: item for item in previous.get(module, [])}
            for item in doc[module]:
                prior = old.get(item[key])
                semantic = {k: v for k, v in item.items() if k != version_key and k != "generation"}
                prior_semantic = {k: v for k, v in (prior or {}).items() if k != version_key and k != "generation"}
                item[version_key] = prior[version_key] if prior and semantic == prior_semantic else "sr_" + semantic_sha256(semantic)
                if module == "scenes":
                    item["generation"] = copy.deepcopy(prior["generation"] if prior and semantic == prior_semantic else stamped_generation)
        return validate_l1_document(doc, catalog=effective)

    async def save(self, memory_id, payload):
        if not isinstance(payload, dict) or set(payload) - {"expected_revision", "request_id", "document", "edit_note"} or not {"expected_revision", "request_id", "document"} <= set(payload):
            raise L1OwnerError("invalid_revision_request")
        expected = payload["expected_revision"]; request_id = payload["request_id"]
        if type(expected) is not int or expected < 0: raise L1OwnerError("invalid_revision", field_path="expected_revision")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128: raise L1OwnerError("invalid_request_id", field_path="request_id")
        if "edit_note" in payload and not isinstance(payload["edit_note"], str): raise L1OwnerError("string_required", field_path="edit_note")
        fingerprint = semantic_sha256(payload)
        idem = f"owner-l1:{memory_id}:{request_id}"
        prior = self.authority.get_commit_by_idempotency(idem)
        if prior:
            if prior["payload"].get("metadata", {}).get("memory_l1_owner_request", {}).get("request_sha256") != fingerprint:
                raise IdempotencyConflict("same request id has different content")
            if prior["status"] == "committed":
                return await self._receipt(memory_id, request_id, prior["result"], replayed=True)
            if prior["status"] not in ("prepared", "body_written"): raise L1OwnerError("original_operation_not_resumable", status=409)
            prepared = prior["payload"]
            result = await self.commit.commit_memory(**{key: prepared[key] for key in
                ("memory_id", "bucket_id", "expected_revision", "metadata", "source_refs", "decision_source", "actor", "memory_state", "recall_policy")},
                body=payload["document"]["narrative"]["body"], idempotency_key=idem, l1_write_intent=True)
            return await self._receipt(memory_id, request_id, result, replayed=True)
        memory = self.authority.get_memory(memory_id)
        if not memory: raise MemoryNotFound(memory_id)
        if memory["active_revision"] != expected: raise RevisionConflict("owner draft is stale")
        if memory["state"] == "tombstoned": raise L1OwnerError("memory_withdrawn", status=409)
        base = await self.detail(memory_id, expected)
        previous = base["l1"]["document"]
        if not isinstance(previous, dict): raise L1OwnerError("original_structure_unavailable", status=409)
        catalog, catalog_revision = self.catalog_provider()
        document = self._owner_document(payload["document"], previous, memory_id=memory_id,
                                        request_id=request_id, catalog=catalog, catalog_revision=catalog_revision, revision=expected + 1)
        body, semantic = split_document(document)
        record = self.authority.get_memory_revision(memory_id, expected)
        metadata = copy.deepcopy(record["metadata"])
        metadata["memory_l1"] = semantic
        metadata["name"] = document["narrative"]["title"]
        metadata["memory_l1_owner_request"] = {"request_id": request_id, "request_sha256": fingerprint, "edit_note": payload.get("edit_note", "")}
        result = await self.commit.commit_memory(memory_id=memory_id, bucket_id=memory["bucket_id"],
            expected_revision=expected, body=body, metadata=metadata, source_refs=record["source_refs"],
            decision_source="owner_l1_edit", idempotency_key=idem, actor="owner", memory_state=memory["state"], recall_policy=memory["recall_policy"], l1_write_intent=True)
        return await self._receipt(memory_id, request_id, result, replayed=False)

    async def _receipt(self, memory_id, request_id, result, *, replayed):
        return {"status": "ok", "request_id": request_id, "replayed": replayed,
                "memory_id": memory_id, "revision": result["revision"],
                "semantic_sha256": result.get("semantic_sha256"), "l1_content_sha256": result.get("l1_content_sha256"),
                "result": await self.detail(memory_id, result["revision"])}
