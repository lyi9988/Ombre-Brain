"""Pure, per-turn state-to-memory affinity. No state writes or model calls.

The producer owns the state. This module only validates a sampled value and
matches owner mappings against declared topics/domains, never free text/tags.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import zlib

from recall_signal_protocol import (
    PRODUCER as INNER_PRODUCER, INNER_NEEDS, INNER_AFFECT, INNER_DIMENSIONS,
    snapshot_from_inner, validate_unified, decode_transport, active_evidence,
)

TARGETS = ("interest_affinity", "drive_affinity", "goal_alignment")
AIZ_PRODUCERS = {"aiz.needs", "aiz.goals", "aiz.interests"}
TRANSPORT_PRODUCERS = AIZ_PRODUCERS | {"ombre.persona", INNER_PRODUCER}


def _unit(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        return False


def _date(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        return result if isinstance(result, datetime) and result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def inner_state_snapshots(snapshot):
    """One canonical checkpoint; old bindings are resolved as read views."""
    return [snapshot_from_inner(snapshot)]


def read_transport(raw, *, conversation_id, turn_id, source_event_id):
    """Internal authenticated request metadata, scoped to its canonical turn.

    Rejected metadata does not prevent chat. It leaves these optional scoring
    inputs unknown. The caller must not forward this envelope to a provider.
    """
    if not raw:
        return {"state": "unavailable", "reason": "not_supplied", "snapshots": []}
    try:
        value = decode_transport(raw)
        if not isinstance(value, dict) or value.get("schema_version") not in ("recall-signals-v1", "recall-signals-v2"):
            raise ValueError("schema")
        scope = value.get("scope")
        expected = dict(conversation_id=conversation_id, turn_id=turn_id, source_event_id=source_event_id)
        if not all(isinstance(v, str) and v for v in expected.values()) or scope != expected:
            return {"state": "unavailable", "reason": "scope_mismatch", "snapshots": []}
        snapshots = value.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) > len(TRANSPORT_PRODUCERS):
            raise ValueError("snapshots")
        producers = [item.get("producer") for item in snapshots if isinstance(item, dict)]
        if len(producers) != len(snapshots) or len(set(producers)) != len(producers) or set(producers) - TRANSPORT_PRODUCERS:
            raise ValueError("producer")
        if value["schema_version"] == "recall-signals-v2":
            if (set(producers) - {INNER_PRODUCER, "aiz.goals", "aiz.interests"}
                    or INNER_PRODUCER not in producers):
                raise ValueError("single_unified_producer_required")
            validate_unified(snapshots[producers.index(INNER_PRODUCER)])
        elif INNER_PRODUCER in producers:
            raise ValueError("v2_required")
        if "ombre.persona" in producers:
            persona = snapshots[producers.index("ombre.persona")]
            if persona.get("state") == "available":
                needs = snapshots[producers.index("aiz.needs")] if "aiz.needs" in producers else {}
                checkpoint = persona.get("checkpoint")
                if (not isinstance(checkpoint, dict) or checkpoint.get("schema") != "inner-state-v1"
                        or checkpoint.get("unit") != "0..1" or checkpoint.get("availability") not in ("complete", "stale")
                        or not isinstance(checkpoint.get("revision"), dict)
                        or set(checkpoint["revision"]) != {"aiz", "brain"}
                        or any(type(v) is not int or v < 0 for v in checkpoint["revision"].values())
                        or checkpoint != needs.get("checkpoint") or needs.get("state") != "available"
                        or checkpoint.get("reference") != persona.get("revision")
                        or persona.get("revision") != needs.get("revision")
                        or persona.get("as_of") != needs.get("as_of")
                        or set(persona.get("signals") or {}) != set(INNER_AFFECT)
                        or set(needs.get("signals") or {}) != set(INNER_NEEDS)):
                    raise ValueError("unified_checkpoint_mismatch")
        return {"state": "received", "scope": scope, "snapshots": copy.deepcopy(snapshots)}
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError, KeyError, zlib.error):
        return {"state": "unavailable", "reason": "invalid_transport", "snapshots": []}


def persona_snapshot(state):
    """Use only source metadata supplied by the existing Persona read.

    A legacy/self-model dictionary without provenance cannot be relabelled as
    a fresh Persona sample. No extra state read, decay, or evaluation occurs.
    """
    source = state.get("recall_signal_source") if isinstance(state, dict) else None
    if not isinstance(source, dict):
        return {"producer": "ombre.persona", "state": "unavailable", "revision": None,
                "as_of": None, "signals": {}, "reason": "source_metadata_missing"}
    return copy.deepcopy(source)


def sample_persona(affect, *, profile_id, session_id, updated_at):
    """Producer-side snapshot of actual persisted numerical affect fields."""
    signals = {}
    for name in ("valence", "arousal", "tenderness", "possessiveness", "longing", "security", "protective_drive"):
        value = affect.get(name)
        valid = _unit(value)
        signals[name] = {"state": "known" if valid else "unknown", "value": value if valid else None,
                         "evidence_refs": [f"persona_session_state:{profile_id}:{session_id}:{updated_at}"]}
    material = json.dumps([profile_id, session_id, updated_at, signals], sort_keys=True, separators=(",", ":"))
    return {"producer": "ombre.persona", "revision": hashlib.sha256(material.encode()).hexdigest(),
            "as_of": updated_at, "state": "available" if _date(updated_at) else "unavailable",
            "signals": signals}


def resolve_signals(settings, snapshots, now):
    """Resolve compatibility names from one complete checkpoint, never a copy."""
    by_producer = {}
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and isinstance(snapshot.get("producer"), str):
            by_producer.setdefault(snapshot["producer"], []).append(snapshot)
    result = {}
    now = _date(now)
    unified = by_producer.get(INNER_PRODUCER, [])
    for binding in settings.get("recall", {}).get("signal_bindings", []):
        producer, key = binding["producer"], binding["key"]
        compatible = (producer == INNER_PRODUCER or
            producer == "aiz.needs" and key in INNER_DIMENSIONS and INNER_DIMENSIONS[key][0] == "aiz" or
            producer == "ombre.persona" and key in INNER_DIMENSIONS and INNER_DIMENSIONS[key][0] == "brain")
        canonical = bool(unified) and compatible
        candidates = unified if canonical else by_producer.get(producer, [])
        snapshot = candidates[0] if len(candidates) == 1 else {}
        invalid = False
        if canonical:
            try:
                validate_unified(snapshot)
            except (ValueError, TypeError, OverflowError, KeyError, RecursionError):
                invalid = True
        signal = (snapshot.get("signals") or {}).get(key) if isinstance(snapshot.get("signals"), dict) else None
        signal = signal if isinstance(signal, dict) else {}
        row = {"producer": INNER_PRODUCER if canonical else producer, "binding_producer": producer,
               "key": key, "state": "unknown", "value": None,
               "revision": snapshot.get("revision"),
               "as_of": signal["as_of"] if "as_of" in signal else snapshot.get("as_of"),
               "age_seconds": None, "evidence_refs": [], "source_refs": []}
        stamp = _date(row["as_of"])
        age = (now - stamp).total_seconds() if now and stamp else None
        row["age_seconds"] = age
        number = signal.get("value")
        if len(candidates) > 1:
            reason = "ambiguous_producer"
        elif invalid:
            reason = "invalid_unified_snapshot"
        elif snapshot.get("state") != "available":
            reason = "producer_" + str(snapshot.get("state") or "unavailable")
        elif not isinstance(row["revision"], str) or not row["revision"]:
            reason = "revision_unknown"
        elif age is None:
            reason = "time_unknown"
        elif age < 0:
            reason = "future_snapshot"
        elif canonical and (snapshot.get("checkpoint") or {}).get("availability") == "stale":
            reason = "stale_checkpoint"
        elif binding.get("max_age_seconds") is not None and age > binding["max_age_seconds"]:
            reason = "stale"
        elif signal.get("state") != "known":
            reason = "signal_unknown"
        elif not _unit(number):
            reason = "invalid_signal_value"
        else:
            reason = None
            row.update(state="known", value=number)
        if canonical and not invalid:
            row["checkpoint"] = copy.deepcopy(snapshot.get("checkpoint"))
            row["dimension_owner"] = signal.get("owner")
            row["dimension_kind"] = signal.get("kind")
            row["unknown_reason"] = signal.get("reason")
            if reason is None:
                row["evidence_refs"] = [snapshot["revision"] + ":dimensions." + key]
                row["source_refs"], row["snapshot_evidence"] = active_evidence(snapshot, key, now)
        elif not canonical:
            refs = signal.get("evidence_refs")
            if isinstance(refs, list) and all(isinstance(ref, str) for ref in refs):
                row["evidence_refs"] = list(refs)
        row["reason"] = reason
        row["producer_reason"] = snapshot.get("reason")
        result[binding["id"]] = row
    return result


def _topics(statement, catalog):
    allowed = set(catalog.get("topics") or [])
    aliases = {alias: definition["id"] for definition in catalog.get("definitions", [])
               if definition.get("axis") == "topics" and definition.get("id") in allowed
               for alias in definition.get("aliases", [])}
    return {item if item in allowed else aliases.get(item)
            for item in (statement.get("classification") or {}).get("topics", [])} - {None}


def statement_affinity(statement, settings, signals, *, domain=None):
    """Compute separate affinities; no matching rule means unknown, not zero.

    Multiple topics using one signal contribute once at their largest declared
    weight. Only known matching evidence contributes to this optional reward;
    missing evidence stays unknown and makes the aggregate explicitly partial.
    """
    policy = settings.get("recall") or {}
    topics = _topics(statement, settings.get("catalog") or {})
    groups = {target: {} for target in TARGETS}
    matched = []
    for rule in policy.get("affinity_mappings", []):
        selector = rule["selector"]
        if selector.get("topic") and selector["topic"] not in topics:
            continue
        if selector.get("domain") and selector["domain"] != domain:
            continue
        target, ident, weight = rule["target"], rule["signal_id"], rule["weight"]
        groups[target][ident] = max(weight, groups[target].get(ident, 0))
        matched.append(rule["id"])
    def usable(source):
        return (isinstance(source, dict) and source.get("state") == "known"
                and source.get("reason") is None and _unit(source.get("value")))

    features = {"signal_" + ident: row["value"] if usable(row) else None
                for ident, row in signals.items()}
    diagnostics = {}
    for target, weights in groups.items():
        contributions = []
        for ident, weight in weights.items():
            source = signals.get(ident)
            source = source if isinstance(source, dict) else {}
            is_known = usable(source)
            reason = None if is_known else (source.get("reason") or
                ("invalid_signal_value" if source.get("state") == "known" else
                 "signal_unknown" if source else "binding_unknown"))
            intensity = source["value"] if is_known else None
            # A zero mapping weight cannot manufacture known source evidence.
            value = weight * intensity if is_known else None
            contributions.append({"signal_id": ident, "topic_affinity": weight,
                                  "intensity": intensity, "value": value,
                                  "state": "known" if is_known else "unknown", "reason": reason,
                                  "producer": source.get("producer"), "revision": source.get("revision"),
                                  "as_of": source.get("as_of"), "age_seconds": source.get("age_seconds"),
                                  "evidence_refs": source.get("evidence_refs", [])})
        numbers = [row["value"] for row in contributions if row["state"] == "known"]
        unknown_count = len(contributions) - len(numbers)
        reason = ("not_mapped" if not contributions else "signal_unknown" if not numbers else
                  "partial_signal_unknown" if unknown_count else None)
        value = (sum(numbers) / len(numbers) if policy.get("affinity_aggregation") == "mean"
                 else max(numbers)) if numbers else None
        features[target] = value
        diagnostics[target] = {"value": value, "reason": reason,
            "state": "unknown" if not numbers else "partial" if unknown_count else "computed",
            "complete": bool(contributions) and not unknown_count,
            "coverage": {"known": len(numbers), "unknown": unknown_count, "matched": len(contributions),
                         "ratio": len(numbers) / len(contributions) if contributions else None},
            "contributions": contributions}
    return features, {"matched_rules": matched, "aggregation": policy.get("affinity_aggregation", "max"),
                      "targets": diagnostics}
