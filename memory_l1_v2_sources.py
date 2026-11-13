"""Read agent-authored memory drafts from their original durable task records."""
from __future__ import annotations

import hashlib
import json

from memory_l1_contract import canonical_json
from memory_l1_service import L1OwnerError


NAMESPACE = "snapshot.ombre.agent_memory_draft"
PREFIXES = {"chat": "memory_l1_chat_task:", "browse": "memory_l1_browse_task:",
            "action": "memory_l1_source_task:"}


def draft_source(task, config, *, kind="chat"):
    if kind not in PREFIXES:
        raise L1OwnerError("unsupported_draft_type", field_path="draft_type")
    envelope = task.get("envelope") or {}
    context = envelope.get("context") or {}
    if kind == "chat":
        narrative = (envelope.get("arguments") or {}).get("narrative") or {}
        text = narrative.get("body")
    elif kind == "browse":
        text = envelope.get("content")
    else:
        intent = task.get("memory_intent") or {}
        text = intent.get("draft")
        if isinstance(text, dict):
            text = text.get("body")
    if not isinstance(text, str) or not text.strip():
        raise L1OwnerError("agent_draft_body_required", field_path="draft")
    options = (config.get("gateway") or {}).get("canonical_continuation") or {}
    return {"system": "ombre", "namespace": "agent_draft", "draft_type": kind,
            "event_id": task["task_id"], "version_id": str(task.get("draft_revision", 1)),
            "profile_id": str((config.get("persona") or {}).get("profile_id") or "haven_xiaoyu"),
            "conversation_id": context.get("conversation_id") or options.get("conversation_id", "conv-jiajia-main"),
            "role": "assistant", "recorded_at": task.get("created_at"), "text": text}


def snapshot(source, config=None):
    config = config or {}
    if (source.get("system") != "ombre" or source.get("namespace") != "agent_draft"
            or source.get("role") != "assistant"
            or source.get("draft_type", "chat") not in PREFIXES):
        raise L1OwnerError("invalid_agent_draft_source", field_path="source")
    from memory_source_identity import source_participant, source_group
    actor = source_participant({**source, "system": "aizizhu", "namespace": "canonical_event"}, config)
    if actor.get("kind") != "ai" or not actor.get("subject_id"):
        raise L1OwnerError("agent_draft_identity_unbound", field_path="source.participant")
    locator = {"task_type": source.get("draft_type", "chat"), "task_id": source["event_id"],
               "profile_id": source["profile_id"], "conversation_id": source["conversation_id"]}
    identity = {"draft_revision": str(source["version_id"]), "text": source["text"],
                "participant_id": actor["subject_id"]}
    revision = "draft_sha256:" + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return {"state": "available", "source_namespace": NAMESPACE,
            "source_id": canonical_json(locator), "source_revision": revision,
            "text_view_id": "agent_memory_draft.exact-v1", "text": source["text"],
            "author": "assistant", "speaker": "assistant", "participant": actor,
            "recorded_at": source.get("recorded_at"), "upstream_revision": str(source["version_id"]),
            "source_group": source_group(source),
            "limitations": ["顾衍自己提交的回忆草稿，保留主观理解的归属。"]}


class AgentMemoryDraftReader:
    namespace = NAMESPACE

    def __init__(self, config, authority):
        self.config = config
        self.authority = authority

    async def read(self, binding):
        try:
            locator = json.loads(binding["source_id"])
            kind = locator["task_type"]
            prefix = PREFIXES[kind]
            task_id = locator["task_id"]
        except (KeyError, ValueError, TypeError):
            return {"state": "unavailable", "reason": "unsupported_source_locator"}
        task = self.authority.get_meta(prefix + task_id)
        if not isinstance(task, dict):
            return {"state": "unavailable", "reason": "source_not_found"}
        if task.get("phase") in {"withdrawn", "deleted", "cancelled"}:
            return {"state": "withdrawn", "reason": "source_withdrawn"}
        try:
            raw = draft_source(task, self.config, kind=kind)
            current = snapshot(raw, self.config)
        except L1OwnerError as error:
            return {"state": "unavailable", "reason": error.code}
        if current["source_id"] != binding["source_id"]:
            return {"state": "changed", "reason": "source_identity_mismatch"}
        return current
