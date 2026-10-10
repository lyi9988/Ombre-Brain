"""Body-free, exact source identities shared by chat submission and raw archive.

An archive ID is never a canonical ID. Only a runtime coverage bridge whose
whole source hash still matches can cross namespaces. Legacy rows stay local.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

BRIDGE_VERSION = "canonical-source-bridge-v1"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity_hash(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _identity(value) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 200 and not any(ord(c) < 32 for c in value)


def _stamp(value) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.isoformat() if parsed.tzinfo is not None else ""
    except (ValueError, TypeError, OverflowError):
        return ""


def canonical_source(source: dict, *, conversation_id: str, profile_id: str) -> dict:
    """Input is the authenticated runtime snapshot, never model arguments."""
    text = source["text"]
    result = {
        "system": "aizizhu", "namespace": "canonical_event",
        "profile_id": profile_id, "conversation_id": conversation_id,
        "event_id": str(source["event_id"]), "version_id": str(source["version_id"]),
        "role": source["role"], "recorded_at": _stamp(source.get("created_at")),
        "snapshot_sha256": digest(text), "text": text, "bridge_status": "canonical",
    }
    return _with_key(result)


def _with_key(source: dict) -> dict:
    result = dict(source)
    # A new version or a later occurrence with identical words is not the same
    # source. Recording time is descriptive, not an identity substitute.
    result["source_key"] = identity_hash({key: result.get(key, "") for key in (
        "system", "namespace", "profile_id", "conversation_id", "event_id",
        "version_id", "role", "snapshot_sha256")})
    return result


def bridge_from_coverage(coverage, messages, *, profile_id: str) -> list[dict]:
    """Bind only current, whole canonical user messages at their wire position.

    The output contains no text. Ambiguous repeated text is rejected later at
    archive time. An assistant still being generated has no persisted canonical
    version yet, so this bridge deliberately makes no assertion about it.
    """
    if not isinstance(coverage, dict) or not isinstance(messages, list):
        return []
    conversation = coverage.get("conversation_id")
    items = coverage.get("items")
    if not _identity(conversation) or not isinstance(items, list):
        return []
    result = []
    for item in items[:64]:
        if not isinstance(item, dict) or item.get("turn_member") is not True:
            continue
        position = item.get("message_index")
        if type(position) is not int or not 0 <= position < len(messages):
            continue
        message = messages[position]
        if (not isinstance(message, dict) or message.get("role") != "user"
                or not isinstance(message.get("content"), str)
                or not _identity(item.get("event_id")) or not _identity(item.get("version_id"))):
            continue
        text = message["content"]
        if not text or len(text) > 12000 or item.get("source_text_sha256") != digest(text):
            continue
        stamp = item.get("recorded_at_ms")
        try:
            recorded = datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat() if type(stamp) is int and stamp > 0 else ""
        except (ValueError, OverflowError, OSError):
            recorded = ""
        source = canonical_source({"event_id": item["event_id"], "version_id": item["version_id"],
                                   "role": "user", "text": text, "created_at": recorded},
                                  conversation_id=conversation, profile_id=profile_id)
        result.append({key: value for key, value in source.items() if key != "text"})
        if len(result) >= 16:
            break
    return result


def archive_bridge(sources, *, role: str, text: str) -> dict:
    if not isinstance(sources, list):
        return {}
    matches = [s for s in sources if isinstance(s, dict) and s.get("role") == role
               and s.get("snapshot_sha256") == digest(text)]
    if len(matches) != 1:
        return {}
    return {"bridge_version": BRIDGE_VERSION, **matches[0]}


def raw_event_source(event: dict, *, profile_id: str) -> dict:
    text = str(event.get("text") or "")
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    bridge = metadata.get("canonical_source")
    archive_ref = {"system": "ombre", "namespace": "raw_event", "event_id": str(event.get("id") or "")}
    if (str(event.get("source") or "") == "gateway" and isinstance(bridge, dict)
            and bridge.get("bridge_version") == BRIDGE_VERSION
            and bridge.get("system") == "aizizhu" and bridge.get("namespace") == "canonical_event"
            and bridge.get("profile_id") == profile_id == metadata.get("profile_id")
            and bridge.get("role") == event.get("role")
            and bridge.get("snapshot_sha256") == digest(text)
            and all(_identity(bridge.get(k)) for k in ("conversation_id", "event_id", "version_id"))):
        resolved = canonical_source({"event_id": bridge["event_id"], "version_id": bridge["version_id"],
                                     "role": bridge["role"], "text": text,
                                     "created_at": bridge.get("recorded_at")},
                                    conversation_id=bridge["conversation_id"], profile_id=profile_id)
        if resolved["source_key"] == bridge.get("source_key"):
            return {**resolved, "bridge_status": "exact_runtime_bridge", "archive_ref": archive_ref}
    return _with_key({**archive_ref, "profile_id": profile_id,
                      "conversation_id": str(event.get("conversation_id") or event.get("session_id") or ""),
                      "version_id": "", "role": event.get("role"), "text": text,
                      "snapshot_sha256": digest(text), "recorded_at": _stamp(event.get("created_at")),
                      "bridge_status": "unresolved"})


def daily_sources(turns: list[dict], *, profile_id: str) -> list[dict]:
    found = {}
    for turn in turns:
        archived = turn.get("_semantic_sources")
        if isinstance(archived, list) and archived:
            sources = [raw_event_source(e, profile_id=profile_id) for e in archived if isinstance(e, dict)]
        else:
            sources = [_with_key({"system": "ombre", "namespace": "conversation_turn",
                       "profile_id": profile_id, "conversation_id": str(turn.get("session_id") or ""),
                       "event_id": str(turn.get("id") or ""), "version_id": "", "role": role,
                       "text": str(turn.get(role + "_text") or ""),
                       "snapshot_sha256": digest(str(turn.get(role + "_text") or "")),
                       "recorded_at": _stamp(turn.get("created_at")), "bridge_status": "unresolved"})
                       for role in ("user", "assistant") if turn.get(role + "_text")]
        for source in sources:
            if source.get("text") and source.get("event_id"):
                found[source["source_key"]] = source
    return list(found.values())[:80]
