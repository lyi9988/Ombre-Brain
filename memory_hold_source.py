"""Closed source descriptors for hold; descriptors never grant source access.

The browse namespace describes an assistant-produced summary of external
material. It does not establish the page author's identity or owner testimony.
"""
from __future__ import annotations

import copy
import hashlib
from urllib.parse import urlsplit

from memory_l1_contract import canonical_json

FIELDS = {"schema", "namespace", "source_id", "source_revision", "content_sha256",
          "author", "coverage", "source_scope"}
BROWSE_NAMESPACE = "snapshot.aiz.browse.summary"


def hold_source_descriptor(content, metadata, *, whisper=False, feel=False):
    if metadata is None:
        return None
    if (not isinstance(metadata, dict) or set(metadata) != FIELDS
            or metadata.get("schema") != "aiz.memory-source-ref-v1"
            or any(not isinstance(metadata.get(k), str) or not metadata[k]
                   for k in ("source_id", "source_revision", "content_sha256"))
            or metadata["content_sha256"] != hashlib.sha256(content.encode("utf-8")).hexdigest()):
        raise ValueError("invalid_source_descriptor")
    namespace = metadata.get("namespace")
    if namespace in {"snapshot.aiz.diary.private", "snapshot.aiz.diary.digest"}:
        if not whisper:
            raise ValueError("diary_requires_whisper")
    elif namespace == BROWSE_NAMESPACE:
        scope = metadata.get("source_scope")
        if (whisper or feel
                or metadata["source_revision"] != metadata["content_sha256"]
                or metadata.get("author") != {"role": "assistant_self", "producer": "skill.web_browse"}
                or metadata.get("coverage") != {"kind": "external_source_summary", "original_page_stored": False}
                or not isinstance(scope, dict)
                or set(scope) != {"schema", "source_kind", "producer", "actor_role", "source_event_id",
                                  "source_ref", "reported_author", "claim_kind"}
                or scope.get("schema") != "browse-experience-v1"
                or scope.get("source_kind") != "web_browse"
                or scope.get("producer") != "skill.web_browse"
                or scope.get("actor_role") != "assistant_self"
                or scope.get("claim_kind") != "external_source_summary"
                or not isinstance(scope.get("source_event_id"), str)
                or (scope.get("reported_author") is not None and not isinstance(scope["reported_author"], str))):
            raise ValueError("invalid_browse_source")
        ref = scope.get("source_ref")
        if (not isinstance(ref, dict)
                or set(ref) != {"content_item_id", "source_domain", "content_hash", "summary_hash", "canonical_url"}
                or ref.get("content_item_id") != metadata["source_id"]
                or any(not isinstance(ref.get(k), str) for k in ref)
                or not ref.get("content_hash") or not ref.get("summary_hash")):
            raise ValueError("invalid_browse_reference")
        url = urlsplit(ref["canonical_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("invalid_browse_url")
    else:
        raise ValueError("unsupported_source_namespace")
    canonical_json(metadata)
    return {"descriptor": copy.deepcopy(metadata), "observation": "not_checked", "eligible_source": None}


def browse_classification(descriptor):
    """The event is reading external material, never an inferred owner profile."""
    if descriptor and descriptor["descriptor"]["namespace"] == BROWSE_NAMESPACE:
        return {"memory_subject": "event", "memory_layer": "process_event",
                "memory_classification_source": "browse_source_descriptor"}
    return None
