"""Chat-authored candidates on the existing Memory authority, without an LLM.

Only the authenticated Aiz runtime submits the source snapshot. Provenance
verification means these messages were in its canonical input, not that the
agent's interpretation is an owner-confirmed fact.
"""
from __future__ import annotations

import copy
import hashlib
import re
from datetime import datetime, timezone

from memory_authority import MemoryProposal, IdempotencyConflict
from memory_narrative import build_chat_narrative, chat_commit_metadata
from memory_commit_service import ProjectionNotApplied
from memory_source_provenance import canonical_source
from memory_semantics import build_annotations, attach_source_links, owner_semantic_summary


KINDS = {"preference", "boundary", "commitment", "shared_experience", "key_event",
         "reflection", "project_state", "identity"}


class ChatMemoryValidationError(ValueError):
    pass


def candidate_id(operation_id):
    if not isinstance(operation_id, str) or not re.fullmatch(r"op-[0-9a-f]{32}", operation_id):
        raise ValueError("invalid_operation_id")
    return "chat_memory_" + operation_id[3:]


def receipt(store, memory_id):
    row = store.get_candidate(memory_id)
    if not row or row["proposal"].get("source_type") != "chat_tool":
        return {"ok": False, "memory_status": "not_found", "candidate_id": memory_id}
    state = row["status"]
    return {"ok": True, "candidate_id": memory_id,
            "memory_id": memory_id if state == "committed" else None,
            "memory_status": state, "revision": row["revision"],
            "body_preserved": not bool(row["proposal"].get("metadata", {}).get("narrative", {}).get("edited_by")),
            "recall_guaranteed": False,
            "semantics": owner_semantic_summary(row["proposal"].get("metadata") or {},
                                                 row["proposal"].get("proposed_body") or "")}


class ChatMemoryTool:
    def __init__(self, engine, bucket_mgr):
        self.engine, self.bucket_mgr = engine, bucket_mgr
        self.store = engine.memory_authority_store

    def _proposal(self, envelope):
        if not isinstance(envelope, dict) or set(envelope) != {"operation_id", "arguments", "context"}:
            raise ValueError("invalid_envelope")
        memory_id = candidate_id(envelope["operation_id"])
        args, context = envelope["arguments"], envelope["context"]
        if not isinstance(args, dict) or set(args) - {"content", "title", "kind", "confidence", "sensitive", "event_time", "annotations"}:
            raise ValueError("invalid_arguments")
        if not isinstance(context, dict) or set(context) != {"conversation_id", "request_id", "turn_id", "sources"}:
            raise ValueError("invalid_source_context")
        if any(not isinstance(context.get(key), str) or not context[key] or len(context[key]) > 200
               for key in ("conversation_id", "request_id", "turn_id")):
            raise ValueError("invalid_source_identity")
        content, title, kind = args.get("content"), args.get("title"), args.get("kind")
        if not isinstance(content, str) or not content.strip() or len(content) > 12000:
            raise ValueError("invalid_memory_body")
        if not isinstance(title, str) or not title.strip() or len(title) > 100 or kind not in KINDS:
            raise ValueError("invalid_memory_kind_or_title")
        confidence = args.get("confidence", 0.7)
        if type(confidence) not in (float, int) or not 0 <= confidence <= 1 or type(args.get("sensitive", False)) is not bool:
            raise ValueError("invalid_policy_fields")
        sources = context["sources"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= 16:
            raise ValueError("source_missing")
        turns, seen = [], set()
        for source in sources:
            if (not isinstance(source, dict) or set(source) != {"event_id", "version_id", "role", "text", "created_at"}
                    or source.get("role") not in {"user", "assistant"}
                    or any(not isinstance(source.get(key), str) or not source[key] for key in source)):
                raise ValueError("invalid_source")
            if (source["event_id"] in seen or len(source["event_id"]) > 200 or len(source["version_id"]) > 200
                    or len(source["text"]) > 12000):
                raise ValueError("source_out_of_bounds")
            stamp = datetime.fromisoformat(source["created_at"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("source_timezone_missing")
            seen.add(source["event_id"])
            turns.append({"id": source["event_id"], "created_at": source["created_at"],
                          f"{source['role']}_text": source["text"]})
        if sum(len(s["text"]) for s in sources) > 60000 or context["turn_id"] not in seen:
            raise ValueError("source_coverage_invalid")
        temporal = args.get("event_time") or {}
        if not isinstance(temporal, dict) or set(temporal) - {"expression", "evidence"}:
            raise ValueError("invalid_event_time")
        if any(not isinstance(v, str) or len(v) > (60 if k == "expression" else 600) for k, v in temporal.items()):
            raise ValueError("invalid_event_time")
        matches = [s for s in sources if temporal.get("evidence") and temporal["evidence"] in s["text"]]
        time_candidate = {}
        if len(matches) == 1:
            time_candidate["event_time"] = {**temporal, "source_turn_id": matches[0]["event_id"], "source_role": matches[0]["role"]}
        narrative = build_chat_narrative(time_candidate, turns, self.engine.identity, self.engine.tz, "")
        narrative.update(generation_kind="chat_authored", source_kind="chat_tool", provenance_only=True)
        versions = {s["event_id"]: s["version_id"] for s in sources}
        for item in narrative["sources"]:
            event_id = item["source_ref"].split(":", 1)[1]
            item.update(source_ref=f"canonical_event:{event_id}", version_id=versions[event_id])
        if narrative["event_time"].get("source_ref"):
            narrative["event_time"]["source_ref"] = narrative["event_time"]["source_ref"].replace("conversation_turn:", "canonical_event:", 1)
        # This excerpt is a display aid; the private immutable snapshot remains
        # in the same candidate metadata for attribution and owner review.
        excerpt = "\n".join(f"{s['role']}: {s['text']}" for s in sources)[-3000:]
        excerpt = self.engine._daily_chat_memory_owner_text(excerpt).replace("<", "‹").replace(">", "›")
        evidence_sources = [canonical_source(s, conversation_id=context["conversation_id"],
                            profile_id=self.engine.memory_semantic_profile_id) for s in sources]
        semantic = build_annotations(body=content, proposed=args.get("annotations"), sources=evidence_sources,
                                     proposal_id=memory_id)
        related = self.store.find_candidates_by_sources([s["source_key"] for s in evidence_sources])
        semantic = attach_source_links(semantic, related, proposal_id=memory_id)
        legacy = {"id": memory_id, "title": title, "content": content.strip(), "proposed_memory": content.strip(),
                  "kind": kind, "date": narrative["mentioned_dates"][-1] if narrative["mentioned_dates"] else "",
                  "mode": self.engine.daily_chat_memory_mode, "source_verification": "verified",
                  "source_canonical_event_ids": [s["event_id"] for s in sources],
                  "original_excerpt": excerpt, "narrative": narrative, "semantic_annotations": semantic,
                  "reason": "聊天模型当场写入；来源核对不代表主人确认全部事实", "tags": ["chat_authored", kind]}
        return MemoryProposal.from_mapping({
            "proposal_id": memory_id, "source_type": "chat_tool", "proposed_body": content,
            "original_excerpt": excerpt, "source_status": "verified",
            "source_refs": [f"canonical_event:{s['event_id']}@{s['version_id']}" for s in sources],
            "memory_type": kind, "confidence": confidence, "sensitive": args.get("sensitive", False),
            "requested_mode": self.engine.daily_chat_memory_mode, "owner_explicit": False,
            "metadata": {"legacy_candidate": legacy, "narrative": narrative, "semantic_annotations": semantic,
                         "runtime_context": context, "tool_operation_id": envelope["operation_id"],
                         "submission_fingerprint": self.store.fingerprint(envelope)},
        })

    async def submit(self, envelope):
        try:
            proposal = self._proposal(envelope)
        except (ValueError, TypeError, KeyError) as exc:
            raise ChatMemoryValidationError("invalid_chat_memory_request") from exc
        row = self.store.get_candidate(proposal.proposal_id)
        if row:
            if row["proposal"].get("metadata", {}).get("submission_fingerprint") != proposal.metadata["submission_fingerprint"]:
                raise IdempotencyConflict("operation payload changed")
            # Replaying submission is read-only, including failed/accepted:
            # only explicit authority recovery may resume an interrupted write.
            return receipt(self.store, proposal.proposal_id)
        policy = self.engine.memory_ingestion_policy.evaluate(proposal)
        row = self.store.put_candidate(proposal, policy)
        if row["status"] == "accepted":
            await self.commit(proposal.proposal_id)
        return receipt(self.store, proposal.proposal_id)

    async def commit(self, memory_id):
        row = self.store.get_candidate(memory_id)
        if not row or row["proposal"].get("source_type") != "chat_tool":
            raise ValueError("not_chat_candidate")
        if row["status"] == "committed":
            return {"id": memory_id, "status": "exists"}
        if row["status"] != "accepted":
            raise ValueError("candidate_not_accepted")
        proposal = row["proposal"]
        candidate = copy.deepcopy(proposal["metadata"]["legacy_candidate"])
        candidate.update(content=proposal["proposed_body"], narrative=proposal["metadata"]["narrative"])
        candidate["semantic_annotations"] = proposal["metadata"].get("semantic_annotations")
        key = f"chat-tool:{memory_id}:candidate-revision:{row['revision']}"
        prior = self.store.get_commit_by_idempotency(key)
        prior_meta = ((prior or {}).get("payload") or {}).get("metadata") or {}
        recorded = prior_meta.get("recorded_at") or datetime.now(self.engine.tz).isoformat(timespec="seconds")
        metadata = {"name": candidate["title"], "tags": candidate.get("tags") or ["chat_authored"],
                    "memory_type": proposal["memory_type"], "source": "chat_tool", "importance": candidate.get("importance", 5),
                    "domain": candidate.get("domain") or [],
                    "created": recorded, "last_active": recorded,
                    **chat_commit_metadata(candidate, recorded)}
        try:
            committed = await self.engine._memory_authority_service(self.bucket_mgr).commit_memory(
                memory_id=memory_id, bucket_id=memory_id, expected_revision=0,
                body=proposal["proposed_body"], metadata=metadata, source_refs=proposal["source_refs"],
                decision_source="owner" if proposal.get("owner_explicit") else "auto",
                idempotency_key=key, actor="chat_tool")
        except ProjectionNotApplied:
            self.store.decide_candidate(memory_id, action="commit_failed", expected_revision=row["revision"],
                                        request_id=key + ":not-applied", actor="commit_service",
                                        reason_codes=["PROJECTION_NOT_APPLIED"])
            return {"id": memory_id, "status": "commit_failed"}
        current = self.store.get_candidate(memory_id)
        if current["status"] == "accepted":
            self.store.decide_candidate(memory_id, action="commit", expected_revision=current["revision"],
                                        request_id=key + ":committed", actor="commit_service",
                                        reason_codes=["MEMORY_REVISION_COMMITTED"])
        return {"id": memory_id, "status": "created", "revision": committed.get("revision")}

    async def confirm(self, row, *, edit, request_id):
        memory_id = row["candidate_id"]
        if row["status"] == "committed":
            return {"id": memory_id, "status": "exists"}
        # Accepted may mean a crash after body commit; never mutate its input.
        if row["status"] == "accepted":
            if edit:
                raise ValueError("pending_commit_cannot_be_edited")
            return await self.commit(memory_id)
        if row["status"] == "commit_failed":
            if edit:
                raise ValueError("retry_requires_original_payload")
            self.store.decide_candidate(memory_id, action="accept", expected_revision=row["revision"],
                                        request_id=request_id + ":retry", actor="owner", reason_codes=["OWNER_RETRY_NOT_APPLIED"])
            return await self.commit(memory_id)
        if row["status"] not in {"pending", "deferred"}:
            return {"id": memory_id, "status": row["status"]}
        payload = copy.deepcopy(row["proposal"])
        payload.update(owner_explicit=True, requested_mode="review")
        if edit:
            if not isinstance(edit, dict) or set(edit) - {"title", "content", "kind", "tags", "domain", "importance", "confidence", "reason"}:
                raise ValueError("invalid_owner_edit")
            if "content" in edit:
                content = edit["content"]
                if not isinstance(content, str) or not content.strip() or len(content) > 12000:
                    raise ValueError("invalid_memory_body")
                payload["proposed_body"] = content.strip()
                payload["metadata"]["narrative"].update(edited_by="owner", event_time={"precision": "unknown", "basis": "owner_edit_requires_time_review"})
            if "title" in edit:
                payload["metadata"]["legacy_candidate"]["title"] = str(edit["title"])[:100]
            if "kind" in edit:
                if edit["kind"] not in KINDS:
                    raise ValueError("invalid_memory_kind")
                payload["memory_type"] = edit["kind"]
            for field in ("tags", "domain"):
                if field in edit:
                    value = edit[field]
                    if not isinstance(value, list) or len(value) > 12 or any(not isinstance(v, str) or len(v) > 100 for v in value):
                        raise ValueError("invalid_owner_edit")
                    payload["metadata"]["legacy_candidate"][field] = value
            if "importance" in edit:
                if type(edit["importance"]) is not int or not 1 <= edit["importance"] <= 10:
                    raise ValueError("invalid_owner_edit")
                payload["metadata"]["legacy_candidate"]["importance"] = edit["importance"]
            if "confidence" in edit:
                value = edit["confidence"]
                if type(value) not in (int, float) or not 0 <= value <= 1:
                    raise ValueError("invalid_owner_edit")
                payload["confidence"] = value
            if "reason" in edit:
                payload["metadata"]["legacy_candidate"]["reason"] = str(edit["reason"])[:160]
        proposal = MemoryProposal.from_mapping(payload)
        revised = self.store.revise_candidate(memory_id, expected_revision=row["revision"], proposal=proposal,
                                              request_id=request_id + ":edit", actor="owner")
        self.store.decide_candidate(memory_id, action="accept", expected_revision=revised["revision"],
                                    request_id=request_id + ":accept", actor="owner", reason_codes=["OWNER_ACCEPTED"])
        return await self.commit(memory_id)
