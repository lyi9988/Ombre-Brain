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

TARGETS = ("interest_affinity", "drive_affinity", "goal_alignment")
AIZ_PRODUCERS = {"aiz.needs", "aiz.goals", "aiz.interests"}


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


def read_transport(raw, *, conversation_id, turn_id, source_event_id):
    """Internal authenticated request metadata, scoped to its canonical turn.

    Rejected metadata does not prevent chat. It leaves these optional scoring
    inputs unknown. The caller must not forward this envelope to a provider.
    """
    if not raw:
        return {"state": "unavailable", "reason": "not_supplied", "snapshots": []}
    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 8192:
            raise ValueError("size")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("schema_version") != "recall-signals-v1":
            raise ValueError("schema")
        scope = value.get("scope")
        expected = dict(conversation_id=conversation_id, turn_id=turn_id, source_event_id=source_event_id)
        if not all(isinstance(v, str) and v for v in expected.values()) or scope != expected:
            return {"state": "unavailable", "reason": "scope_mismatch", "snapshots": []}
        snapshots = value.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) > len(AIZ_PRODUCERS):
            raise ValueError("snapshots")
        producers = [item.get("producer") for item in snapshots if isinstance(item, dict)]
        if len(producers) != len(snapshots) or len(set(producers)) != len(producers) or set(producers) - AIZ_PRODUCERS:
            raise ValueError("producer")
        return {"state": "received", "scope": scope, "snapshots": copy.deepcopy(snapshots)}
    except (ValueError, TypeError, UnicodeError, RecursionError):
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
    """Freeze named bindings once. Age and unknown status remain inspectable."""
    by_producer = {}
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and isinstance(snapshot.get("producer"), str):
            by_producer.setdefault(snapshot.get("producer"), []).append(snapshot)
    result = {}
    now = _date(now)
    for binding in settings.get("recall", {}).get("signal_bindings", []):
        producer, key = binding["producer"], binding["key"]
        row = {"producer": producer, "key": key, "state": "unknown", "value": None,
               "revision": None, "as_of": None, "age_seconds": None, "evidence_refs": []}
        candidates = by_producer.get(producer, [])
        snapshot = candidates[0] if len(candidates) == 1 else {}
        signal = (snapshot.get("signals") or {}).get(key) if isinstance(snapshot.get("signals"), dict) else None
        signal = signal if isinstance(signal, dict) else {}
        row.update(revision=snapshot.get("revision"),
                   as_of=signal["as_of"] if "as_of" in signal else snapshot.get("as_of"))
        stamp = _date(row["as_of"])
        age = (now - stamp).total_seconds() if now and stamp else None
        row["age_seconds"] = age
        number = signal.get("value")
        if len(candidates) > 1:
            reason = "ambiguous_producer"
        elif snapshot.get("state") != "available":
            reason = "producer_" + str(snapshot.get("state") or "unavailable")
        elif not isinstance(row["revision"], str) or not row["revision"]:
            reason = "revision_unknown"
        elif age is None:
            reason = "time_unknown"
        elif age < 0:
            reason = "future_snapshot"
        elif binding.get("max_age_seconds") is not None and age > binding["max_age_seconds"]:
            reason = "stale"
        elif signal.get("state") != "known":
            reason = "signal_unknown"
        elif not _unit(number):
            reason = "invalid_signal_value"
        else:
            reason = None
            row.update(state="known", value=number)
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
    weight. Missing matching evidence makes the aggregate unknown rather than
    presenting a partial lower bound as the full score.
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
    features = {"signal_" + ident: row.get("value") for ident, row in signals.items()}
    diagnostics = {}
    for target, weights in groups.items():
        contributions = []
        for ident, weight in weights.items():
            source = signals.get(ident) or {"state": "unknown", "value": None, "reason": "binding_unknown"}
            # A configured zero multiplier is deterministically zero, even if
            # that optional producer is unavailable.
            value = weight * source["value"] if source.get("state") == "known" else (0 if weight == 0 else None)
            contributions.append({"signal_id": ident, "topic_affinity": weight,
                                  "intensity": source.get("value"), "value": value,
                                  "state": source.get("state"), "reason": source.get("reason"),
                                  "producer": source.get("producer"), "revision": source.get("revision"),
                                  "as_of": source.get("as_of"), "age_seconds": source.get("age_seconds"),
                                  "evidence_refs": source.get("evidence_refs", [])})
        numbers = [row["value"] for row in contributions]
        reason = "not_mapped" if not numbers else ("signal_unknown" if None in numbers else None)
        value = None if reason else (sum(numbers) / len(numbers) if policy.get("affinity_aggregation") == "mean" else max(numbers))
        features[target] = value
        diagnostics[target] = {"value": value, "reason": reason, "contributions": contributions}
    return features, {"matched_rules": matched, "aggregation": policy.get("affinity_aggregation", "max"),
                      "targets": diagnostics}
