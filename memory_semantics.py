"""Optional, revision-bound annotations; evidence checks are not fact approval.

No model, embedding, task creation, new Memory store or prompt injection lives
here. Free-text qualifiers remain source expressions, not inferred deadlines.
"""
from __future__ import annotations

import copy
import re

from memory_narrative import narrative_body_hash
from memory_source_provenance import digest, identity_hash
from raw_events import raw_event_text_looks_injected, strip_raw_client_context

VERSION = "memory-semantics-v1"
KINDS = {"preference", "boundary", "commitment", "shared_experience", "key_event", "reflection", "project_state", "identity"}
BASES = {"owner_statement", "assistant_commitment", "shared_experience", "assistant_interpretation"}
QUALIFIERS = ("topic", "value", "conditions", "exceptions", "valid_time")
FIELDS = {"semantic_kind", "subject", "assertion_basis", "evidence", "polarity", "assertion_state", *QUALIFIERS}

# A quoted source is not evidence if its position is hidden or is protocol
# markup. Mask spans, rather than rewriting the source, to retain Unicode
# offsets and the original immutable snapshot hash.
_HIDDEN_RE = re.compile(r"<(silent|think|thinking|analysis|tool_call|tool_result|attachment|workspace_attachment)\b[^>]*>.*?(?:</\1\s*>|$)", re.I | re.S)
_CONTROL_RE = re.compile(r'<[^>]+>|\b(?:as|reason|mood|action|role|source|mode|tags|metadata|id|session|event_hash|client)="[^"]*"|\[(?:语音|图片|文件|贴纸|表情)[^\]]*\]', re.I)
_DUMP_MARKERS = ("近期素材", "[app/bridge]", "今天的聊天:", "今天的聊天：", "[telegram]")


def _visible_evidence(text: str, start: int, end: int) -> bool:
    if raw_event_text_looks_injected(text) or any(marker in text for marker in _DUMP_MARKERS):
        return False
    for pattern in (_HIDDEN_RE, _CONTROL_RE):
        if any(match.start() < end and start < match.end() for match in pattern.finditer(text)):
            return False
    quote = text[start:end]
    # Client-context sections are not utterances, even if attached to a user
    # message. Reject ambiguous formatting changes rather than invent offsets.
    return quote in strip_raw_client_context(text) and not any(ord(c) < 32 and c not in "\n\t\r" for c in quote)


def build_annotations(*, body: str, proposed, sources: list[dict], proposal_id: str) -> dict:
    body_hash = narrative_body_hash(body)
    coverage = [{k: v for k, v in source.items() if k != "text"} for source in sources]
    result = {"version": VERSION, "body_sha256": body_hash, "state": "current",
              "source_coverage": coverage, "items": [], "issues": [], "links": []}
    if proposed is None:
        return result
    if not isinstance(proposed, list) or len(proposed) > 8:
        result["issues"].append({"reason": "invalid_annotation_list"})
        return result
    for number, item in enumerate(proposed):
        reason = ""
        if (not isinstance(item, dict) or set(item) - FIELDS
                or not isinstance(item.get("semantic_kind"), str) or item.get("semantic_kind") not in KINDS
                or item.get("subject") not in ("user", "assistant")
                or not isinstance(item.get("assertion_basis"), str) or item.get("assertion_basis") not in BASES
                or not isinstance(item.get("evidence"), str) or not 1 <= len(item["evidence"]) <= 600):
            reason = "invalid_annotation"
        elif any(k in item and (not isinstance(item[k], str) or not 1 <= len(item[k]) <= 160) for k in QUALIFIERS):
            reason = "invalid_qualifier"
        elif item.get("polarity", "unspecified") not in ("positive", "negative", "mixed", "unspecified"):
            reason = "invalid_polarity"
        elif item.get("assertion_state", "stated") not in ("stated", "changed", "cancelled", "fulfilled", "uncertain"):
            reason = "invalid_assertion_state"
        if reason:
            result["issues"].append({"index": number, "reason": reason})
            continue
        quote = item["evidence"]
        matches = [(s, s["text"].index(quote)) for s in sources if quote in s["text"]]
        if len(matches) != 1 or matches[0][0]["text"].count(quote) != 1:
            result["issues"].append({"index": number, "reason": "evidence_not_unique"})
            continue
        source, start = matches[0]
        if not _visible_evidence(source["text"], start, start + len(quote)):
            result["issues"].append({"index": number, "reason": "evidence_not_owner_visible"})
            continue
        basis = item["assertion_basis"]
        if ((basis == "owner_statement" and (source["role"] != "user" or item["subject"] != "user"))
                or (basis == "assistant_commitment" and (source["role"] != "assistant" or item["subject"] != "assistant"
                                                        or item["semantic_kind"] != "commitment"))
                or (basis == "assistant_interpretation" and source["role"] != "assistant")):
            result["issues"].append({"index": number, "reason": "speaker_basis_mismatch"})
            continue
        # Conditions/exceptions/time must be quoted, not generated defaults.
        if any(item.get(k) and item[k] not in quote for k in ("conditions", "exceptions", "valid_time")):
            result["issues"].append({"index": number, "reason": "qualifier_not_in_evidence"})
            continue
        evidence = {k: v for k, v in source.items() if k != "text"}
        evidence.update(span=[start, start + len(quote)], span_unit="unicode_codepoint",
                        quote_sha256=digest(quote))
        evidence["evidence_id"] = "ev-" + identity_hash({"source_key": source["source_key"], "span": evidence["span"],
                                                       "quote_sha256": evidence["quote_sha256"]})[:32]
        annotation = {"semantic_kind": item["semantic_kind"],
                      "subject_ref": {"profile_id": source["profile_id"], "conversation_id": source["conversation_id"],
                                      "role": item["subject"]},
                      "assertion_basis": basis, "body_sha256": body_hash,
                      "evidence_refs": [evidence], "evidence_status": "quote_verified",
                      "interpretation_status": "proposed", "execution_authority": False,
                      "polarity": item.get("polarity", "unspecified"),
                      "assertion_state": item.get("assertion_state", "stated"),
                      "qualifiers": {k: item[k] for k in QUALIFIERS if k in item},
                      "conditions_status": "stated" if item.get("conditions") else "not_stated"}
        annotation["annotation_id"] = "ann-" + identity_hash({"proposal_id": proposal_id, **annotation})[:32]
        if annotation not in result["items"]:
            result["items"].append(annotation)
    return result


def current_annotations(metadata: dict, body: str) -> dict:
    value = metadata.get("semantic_annotations") if isinstance(metadata, dict) else None
    if not isinstance(value, dict) or value.get("version") != VERSION:
        return {}
    if (value.get("body_sha256") != narrative_body_hash(body) or value.get("state") != "current"
            or any(not isinstance(value.get(k), list) for k in ("items", "source_coverage", "links", "issues"))):
        return {}
    return value


def guard_metadata(metadata: dict, body: str) -> dict:
    result = copy.deepcopy(metadata or {})
    value = result.get("semantic_annotations")
    if isinstance(value, dict) and value.get("version") == VERSION and value.get("body_sha256") != narrative_body_hash(body):
        value.update(state="invalidated", invalidation_reason="body_changed_requires_review")
    legacy = result.get("legacy_candidate")
    if isinstance(legacy, dict) and isinstance(value, dict):
        legacy["semantic_annotations"] = copy.deepcopy(value)
    return result


def source_keys(proposal: dict) -> list[str]:
    semantic = current_annotations(proposal.get("metadata") or {}, str(proposal.get("proposed_body") or ""))
    return list(dict.fromkeys(str(s["source_key"]) for s in semantic.get("source_coverage", [])
                             if isinstance(s, dict) and isinstance(s.get("source_key"), str) and len(s["source_key"]) == 64))[:80]


def attach_source_links(semantic: dict, related: list[dict], *, proposal_id: str) -> dict:
    result = copy.deepcopy(semantic)
    keys = {s["source_key"] for s in semantic.get("source_coverage", [])}
    for row in related[:12]:
        if row.get("candidate_id") == proposal_id:
            continue
        proposal = row.get("proposal") or {}
        other = current_annotations(proposal.get("metadata") or {}, proposal.get("proposed_body") or "")
        overlap = keys.intersection(source_keys(proposal))
        if not overlap:
            continue
        # This is a source relation, not permission to replace or adopt a fact.
        result["links"].append({"kind": "source_overlap", "state": "proposed",
            "target_candidate_id": row["candidate_id"], "target_candidate_revision": row["revision"],
            "target_body_sha256": other["body_sha256"], "target_status_at_link": row["status"],
            "target_source_type": proposal.get("source_type"), "shared_source_keys": sorted(overlap),
            "fact_equivalence": "unproven", "body_replacement_allowed": False})
    return result


def owner_semantic_summary(metadata: dict, body: str) -> dict:
    stored = (metadata or {}).get("semantic_annotations") or {}
    if not isinstance(stored, dict):
        stored = {}
    current = current_annotations(metadata, body)
    return {"state": "current" if current else ("invalidated" if stored else "legacy_unknown"),
            "annotation_count": len(current.get("items", [])),
            "unresolved_count": len(stored["issues"]) if isinstance(stored.get("issues"), list) else 0,
            "source_link_count": len(current.get("links", [])),
            "execution_authority": False}
