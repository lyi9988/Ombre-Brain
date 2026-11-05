"""Evidence-bound projection of the existing portrait.user version.

The foreground reader never generates or persists a portrait. Source digests are
limited to the explicitly cited records, and deliberately ignore activation data.
"""
from __future__ import annotations

import hashlib
import json
import os


CLAIM_KINDS = {"explicit_preference", "revisable_impression", "historical"}
SEMANTIC_METADATA = ("id", "source", "speaker", "role", "created", "active",
                     "deprecated", "type", "tags", "profile_kind", "retracted",
                     "withdrawn", "superseded_by", "corrected_by", "event_id",
                     "original_event_id", "source_event_id", "source_refs",
                     "system", "namespace", "profile_id", "conversation_id")


def portrait_state_path(config: dict) -> str:
    configured = str((config.get("portrait") or {}).get("state_path") or "").strip()
    if configured:
        return os.path.abspath(configured)
    state_dir = config.get("state_dir") or os.path.join(
        os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))), "state")
    return os.path.join(state_dir, "portrait_state.json")


def chat_evidence_ids(state: dict) -> list[str]:
    user = (state.get("portrait") or {}).get("user") or {}
    if user.get("chat_claims_revision") != user.get("stable_revision"):
        return []
    return list(dict.fromkeys(
        str(ref.get("bucket_id") or "") for claim in user.get("chat_claims") or []
        if isinstance(claim, dict) and claim.get("kind") in CLAIM_KINDS - {"historical"}
        for ref in claim.get("evidence") or [] if isinstance(ref, dict) and ref.get("bucket_id")))


def source_revision(bucket: dict) -> str:
    meta = bucket.get("metadata") or {}
    body = {"content": str(bucket.get("content") or ""),
            "metadata": {key: meta[key] for key in SEMANTIC_METADATA if key in meta}}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                    default=str).encode("utf-8")).hexdigest()


def bind_source_authority(bucket: dict | None, authority: dict | None,
                          semantic: dict | None = None) -> dict | None:
    """Join a direct source to the existing active/auto-enabled Authority view.

    ``authority`` comes from memory_index_metadata_map; absent/ambiguous/manual
    entries are rejected there. L1's existing parent validation is reused.
    """
    if not source_is_current(bucket) or not authority:
        return None
    body_sha = hashlib.sha256(str(bucket.get("content") or "").encode("utf-8")).hexdigest()
    if (not authority.get("memory_id") or int(authority.get("revision") or 0) < 1
            or authority.get("body_sha256") != body_sha):
        return None
    if not isinstance(semantic, dict):
        return None
    if semantic.get("l1_sync_required") or semantic.get("state") not in {"legacy", "l1"}:
        return None
    token = hashlib.sha256(json.dumps(
        {"authority_dependency": authority.get("dependency_token"),
         "state": semantic.get("state"), "document": semantic.get("document")},
        ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    origins = set()
    meta = bucket.get("metadata") or {}
    for key in ("original_event_id", "source_event_id", "event_id"):
        if meta.get(key):
            origins.add("event:" + str(meta[key]))
    for ref in meta.get("source_refs") or []:
        if isinstance(ref, str) and ref:
            origins.add("origin:" + ref)
    document = semantic.get("document") or {}
    for binding in document.get("evidence_bindings") or []:
        if binding.get("source_id"):
            origins.add("origin:" + str(binding["source_id"]))
    parent = (document.get("entry") or {}).get("parent_ref") or {}
    if parent.get("memory_id"):
        origins.add("memory:" + str(parent["memory_id"]))
    for relation in document.get("relations") or []:
        if (relation.get("kind") in {"same_event", "split_from"}
                and relation.get("confirmation_state") == "confirmed"):
            for endpoint in (relation.get("from") or {}, relation.get("to") or {}):
                if endpoint.get("memory_id"):
                    origins.add("memory:" + str(endpoint["memory_id"]))
    origins.add("memory:" + str(authority["memory_id"]))
    origins.add("body:" + body_sha)
    return {**bucket, "_portrait_occurrence_ids": sorted(origins), "_portrait_authority": {
        "memory_id": str(authority["memory_id"]),
        "memory_revision": int(authority["revision"]), "body_sha256": body_sha,
        "dependency_token": token}}


def source_binding_matches(ref: dict, source: dict | None) -> bool:
    if not source_is_current(source) or source_revision(source) != ref.get("source_revision"):
        return False
    authority = source.get("_portrait_authority") or {}
    return bool(authority and all(authority.get(key) == ref.get(key) for key in
                ("memory_id", "memory_revision", "body_sha256", "dependency_token")))


def source_is_current(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict):
        return False
    meta = bucket.get("metadata") or {}
    return not (meta.get("active") is False or meta.get("deprecated")
                or meta.get("type") == "archived"
                or any(meta.get(key) for key in
                       ("retracted", "withdrawn", "superseded_by", "corrected_by")))


def render_claims(claims: list[dict], *, include_history: bool = False) -> str:
    lines = []
    for claim in claims:
        kind = claim.get("kind")
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        if kind == "explicit_preference":
            lines.append("她明确表达过：" + text)
        elif kind == "revisable_impression":
            lines.append("我目前觉得，" + text)
        elif kind == "owner_authored":
            lines.append(text)
        elif kind == "historical" and include_history:
            lines.append("过去（" + str(claim.get("as_of") or "") + "）：" + text)
    return "\n".join(lines)


def build_user_portrait_projection(state: dict, source_lookup, *, enabled: bool = True) -> dict:
    """Read current claims; invalidate each claim before any attempted revision."""
    user = (state.get("portrait") or {}).get("user") or {}
    revision = int(user.get("stable_revision") or 0)
    result = {"text": "", "revision": revision, "status": "empty", "claim_count": 0}
    if not enabled:
        return {**result, "status": "disabled"}
    if user.get("chat_claims_revision") != revision:
        # Prior edits belong to the owner's archive. A new explicit edit binds
        # its own version; merely enabling the chat block does not publish them.
        return {**result, "status": "unreviewed"}
    active = []
    for claim in user.get("chat_claims") or []:
        if not isinstance(claim, dict) or claim.get("kind") == "historical":
            continue
        if claim.get("kind") == "owner_authored" and user.get("stable_source") in {"manual", "rollback"}:
            active.append(claim)
            continue
        evidence = claim.get("evidence") or []
        if claim.get("kind") not in CLAIM_KINDS or not evidence:
            continue
        valid = True
        for ref in evidence:
            try:
                source = source_lookup(ref)
                if not source_binding_matches(ref, source):
                    valid = False
                    break
            except Exception:
                valid = False
                break
        if valid:
            active.append(claim)
    text = render_claims(active)
    return {**result, "text": text, "status": "ready" if text else "invalidated",
            "claim_count": len(active)}
