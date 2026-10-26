"""Versioned transport of one settled checkpoint; no state writes or arithmetic."""
from __future__ import annotations
import base64
import copy
from datetime import datetime
import hashlib
import json
import math
import zlib

PRODUCER = "guyan.inner_state"
DIMENSION_SCHEMA = "inner-state-dimensions-v1"
COMPRESSED_PREFIX = "recall-signals-z1:"
MAX_DECODED_BYTES = 262144
INNER_NEEDS = ("energy", "rest", "curiosity", "novelty", "expression",
               "creativity", "autonomy", "competence", "solitude")
INNER_AFFECT = ("valence", "arousal", "tenderness", "possessiveness",
                "longing", "security", "protective_drive")
INNER_DIMENSIONS = {
    "energy": ("aiz", "resource"),
    **{key: ("aiz", "need") for key in (*INNER_NEEDS[1:], "boredom", "duty", "reflection")},
    **{key: ("brain", "affect") for key in (*INNER_AFFECT, "libido", "grieve", "anger",
        "favored", "session_defensiveness", "cling")},
    "fatigue": ("brain", "resource"),
    **{key: ("brain", "slow") for key in ("slow_security", "slow_confidence", "slow_mood")},
    **{key: ("brain", "tendency") for key in ("exploration_affinity", "social_recovery_rate",
        "connection_initiative", "creative_satisfaction", "challenge_tolerance", "reflection_preference")},
}
FEELINGS = frozenset(("warm_neutral","warm_playful","neutral","happy","sad","calm","curious",
    "tired","playful","mixed-playful","longing","affectionate","tender","jealous","hurt","angry",
    "clingy","secure","insecure","conflicted","guarded","focused","desire","sensual","possessive",
    "intimate","yearning","attached","vulnerable","disappointed"))
SOURCE_KINDS = frozenset(("conversation", "operation", "memory_reaction", "tendency_review", "clock"))

def dumps(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)

def date(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else None
        return parsed if parsed and parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (ValueError, TypeError):
        return None

def unit(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        return False

def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 1024 and bool(value.strip())

def _safe_ref(ref):
    if (not isinstance(ref, dict) or ref.get("kind") not in SOURCE_KINDS
            or not _identity(ref.get("event_id")) or not date(ref.get("occurred_at"))):
        return None
    result = {key: ref[key] for key in ("kind", "event_id", "occurred_at")}
    observation = ref.get("observation_id")
    if (isinstance(observation, str) and observation.startswith("obs:") and len(observation) == 68
            and all(c in "0123456789abcdef" for c in observation[4:])):
        result["observation_id"] = observation
    return result

def evidence_from(snapshot):
    refs, lookup = [], {}
    def indexes(values):
        result = []
        for raw in values if isinstance(values, list) else []:
            ref = _safe_ref(raw)
            if ref is None:
                continue
            token = dumps(ref)
            if token not in lookup:
                lookup[token] = len(refs); refs.append(ref)
            if lookup[token] not in result:
                result.append(lookup[token])
        return result
    unscoped = indexes(snapshot.get("source_refs"))
    changes, feelings = [], []
    for field, target in (("recent_changes", changes), ("feelings", feelings)):
        for raw in snapshot.get(field, []) if isinstance(snapshot.get(field), list) else []:
            if not isinstance(raw, dict):
                continue
            name = "key" if field == "recent_changes" else "label"
            if raw.get(name) not in (INNER_DIMENSIONS if name == "key" else FEELINGS):
                continue
            at = raw.get("occurred_at") if name == "key" else raw.get("at")
            until = raw.get("until")
            if not date(at) or (until is not None and (not date(until) or date(until) < date(at))):
                continue
            row = {name: raw[name], "at": at, "refs": indexes(raw.get("source_refs"))}
            if until is not None:
                row["until"] = until
            target.append(row)
    return {"state": "available", "refs": refs, "changes": changes,
            "feelings": feelings, "unscoped_refs": unscoped}

def unavailable(reason):
    return {"producer": PRODUCER, "revision": None, "as_of": None, "state": "unavailable",
            "signals": {key: {"state": "unknown", "value": None, "reason": reason,
                "unit": "0..1", "owner": owner, "kind": kind}
                for key, (owner, kind) in INNER_DIMENSIONS.items()}, "reason": reason}

def snapshot_from_inner(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("schema") != "inner-state-v1":
        raise ValueError("unified_snapshot_required")
    if snapshot.get("availability") == "unavailable":
        return unavailable("inner_state_unavailable")
    if snapshot.get("availability") not in ("complete", "stale"):
        raise ValueError("settled_unified_snapshot_required")
    if (any(not _identity(snapshot.get(key)) for key in ("subject_id", "checkpoint_id", "rules_version"))
            or not date(snapshot.get("settled_at"))):
        raise ValueError("unified_checkpoint_identity_required")
    revision = snapshot.get("revision")
    if (not isinstance(revision, dict) or set(revision) != {"aiz", "brain"}
            or any(type(value) is not int or value < 0 for value in revision.values())):
        raise ValueError("both_owner_revisions_required")
    dimensions = snapshot.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(INNER_DIMENSIONS):
        raise ValueError("complete_unified_dimensions_required")
    signals = {}
    for key, row in dimensions.items():
        if (not isinstance(row, dict) or set(row) != {"value", "unit", "owner", "kind", "status", "reason"}
                or row.get("unit") != "0..1" or (row.get("owner"), row.get("kind")) != INNER_DIMENSIONS[key]):
            raise ValueError("normalized_dimension_required")
        if row["status"] == "known":
            if not unit(row["value"]) or row["reason"] is not None:
                raise ValueError("known_dimension_invalid")
        elif (row["status"] != "unknown" or row["value"] is not None or not _identity(row["reason"])):
            raise ValueError("unknown_dimension_invalid")
        signals[key] = {**{k: row[k] for k in ("value", "reason", "unit", "owner", "kind")},
                        "state": row["status"]}
    material = {key: snapshot[key] for key in
        ("schema", "subject_id", "checkpoint_id", "revision", "rules_version", "settled_at")}
    material.update(unit="0..1", values={key: row["value"] for key, row in dimensions.items()})
    reference = "inner-" + hashlib.sha256(dumps(material).encode("ascii")).hexdigest()
    checkpoint = {key: copy.deepcopy(snapshot[key]) for key in
        ("schema", "subject_id", "checkpoint_id", "revision", "rules_version", "availability")}
    checkpoint.update(unit="0..1", dimension_schema=DIMENSION_SCHEMA)
    return {"producer": PRODUCER, "revision": reference, "as_of": snapshot["settled_at"],
            "state": "available", "signals": signals, "checkpoint": checkpoint,
            "evidence": evidence_from(snapshot)}

def validate_unified(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("producer") != PRODUCER:
        raise ValueError("unified_producer")
    if snapshot.get("state") == "unavailable":
        if snapshot != unavailable(snapshot.get("reason")) or not _identity(snapshot.get("reason")):
            raise ValueError("unavailable_snapshot")
        return
    if set(snapshot) != {"producer", "revision", "as_of", "state", "signals", "checkpoint", "evidence"}:
        raise ValueError("unified_snapshot_shape")
    checkpoint = snapshot.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint_shape")
    if checkpoint.get("dimension_schema") != DIMENSION_SCHEMA or checkpoint.get("unit") != "0..1":
        raise ValueError("dimension_schema")
    signals = snapshot.get("signals")
    if not isinstance(signals, dict):
        raise ValueError("signals")
    source = {k: checkpoint.get(k) for k in
        ("schema", "subject_id", "checkpoint_id", "revision", "rules_version", "availability")}
    source["settled_at"] = snapshot.get("as_of")
    source["dimensions"] = {}
    for key, row in signals.items():
        if not isinstance(row, dict) or set(row) != {"value", "reason", "unit", "owner", "kind", "state"}:
            raise ValueError("dimension_shape")
        source["dimensions"][key] = {**{k: row[k] for k in ("value","reason","unit","owner","kind")}, "status": row["state"]}
    rebuilt = snapshot_from_inner(source)
    if (snapshot.get("state") != "available" or checkpoint != rebuilt["checkpoint"]
            or snapshot.get("revision") != rebuilt["revision"]):
        raise ValueError("checkpoint_mismatch")
    evidence = snapshot.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence_required")
    if evidence.get("state") == "unavailable":
        if evidence != {"state":"unavailable","reason":"transport_oversize","refs":[],
                        "changes":[],"feelings":[],"unscoped_refs":[]}:
            raise ValueError("evidence_unavailable")
        return
    if set(evidence) != {"state","refs","changes","feelings","unscoped_refs"} or evidence["state"] != "available":
        raise ValueError("evidence_shape")
    refs = evidence["refs"]
    if not isinstance(refs, list) or any(_safe_ref(ref) != ref for ref in refs):
        raise ValueError("evidence_ref")
    def check_indexes(indexes):
        return isinstance(indexes, list) and len(set(indexes)) == len(indexes) and all(type(i) is int and 0 <= i < len(refs) for i in indexes)
    if not check_indexes(evidence["unscoped_refs"]):
        raise ValueError("unscoped_refs")
    for field, name, choices in (("changes","key",INNER_DIMENSIONS),("feelings","label",FEELINGS)):
        if not isinstance(evidence[field], list):
            raise ValueError("evidence_rows")
        for row in evidence[field]:
            if (not isinstance(row, dict) or set(row) not in ({name,"at","refs"},{name,"at","refs","until"})
                    or row.get(name) not in choices or not date(row.get("at")) or not check_indexes(row.get("refs"))
                    or ("until" in row and (not date(row["until"]) or date(row["until"]) < date(row["at"])))):
                raise ValueError("evidence_row")

def encode_transport(payload, max_bytes=7000):
    raw = dumps(payload)
    if len(raw.encode("ascii")) <= max_bytes:
        return raw
    if len(raw.encode("ascii")) > MAX_DECODED_BYTES:
        return None
    compressed = COMPRESSED_PREFIX + base64.urlsafe_b64encode(zlib.compress(raw.encode("ascii"), 9)).decode("ascii")
    return compressed if len(compressed) <= max_bytes else None

def decode_transport(raw):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 8192:
        raise ValueError("size")
    if raw.startswith(COMPRESSED_PREFIX):
        encoded = raw[len(COMPRESSED_PREFIX):]
        data = base64.b64decode(encoded, altchars=b"-_", validate=True)
        inflater = zlib.decompressobj()
        unpacked = inflater.decompress(data, MAX_DECODED_BYTES + 1)
        if (len(unpacked) > MAX_DECODED_BYTES or inflater.unconsumed_tail
                or not inflater.eof or inflater.unused_data):
            raise ValueError("decoded_size_or_stream")
        raw = unpacked.decode("ascii")
    elif len(raw.encode("utf-8")) > MAX_DECODED_BYTES:
        raise ValueError("decoded_size")
    return json.loads(raw)

def active_evidence(snapshot, key, now):
    evidence = snapshot.get("evidence") or {}
    if evidence.get("state") != "available" or now is None:
        return [], {"state": evidence.get("state", "unavailable"), "reason": evidence.get("reason")}
    refs = evidence["refs"]
    def active(row):
        return date(row["at"]) <= now and ("until" not in row or date(row["until"]) >= now)
    changes = [row for row in evidence["changes"] if active(row)]
    feelings = [row for row in evidence["feelings"] if active(row)]
    linked, seen = [], set()
    for change in changes:
        if change["key"] != key:
            continue
        for index in change["refs"]:
            ref = refs[index]
            if ref["kind"] == "clock" or date(ref["occurred_at"]) > now:
                continue
            token = dumps(ref)
            if token not in seen:
                seen.add(token); linked.append(copy.deepcopy(ref))
    # Only valid event references leave this read view. Global evidence remains
    # global; clock progression and unbound refs are never dimension causes.
    valid = [index for index, ref in enumerate(refs)
             if ref["kind"] != "clock" and date(ref["occurred_at"]) <= now]
    remap = {old: new for new, old in enumerate(valid)}
    def mapped(rows):
        return [{**copy.deepcopy(row), "refs": [remap[i] for i in row["refs"] if i in remap]} for row in rows]
    return linked, {"state":"available", "refs":[copy.deepcopy(refs[i]) for i in valid],
                    "changes":mapped(changes), "feelings":mapped(feelings),
                    "unscoped_refs":[remap[i] for i in evidence["unscoped_refs"] if i in remap]}
