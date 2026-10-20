"""Bounded score adoption evidence, never Memory prose or provider credentials.

Replay means evaluating the recorded computation inputs with the recorded scoring
definitions. It does not mean repeating matching, retrieval, source checks or IO.
Display labels/descriptions and catalog aliases are deliberately not recorded.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re

from memory_system_config import CATALOG_AXES, RECALL_DEFAULTS, REQUEST_CONFIGURATION, RUNTIME_INPUTS, CONDITIONAL_RUNTIME_INPUTS, PENDING_RUNTIME_INPUTS

CANDIDATE_LIMIT = 12
STATEMENT_LIMIT = 16
CONFIG_BYTES = 32768
EVIDENCE_BYTES = 98304
_IDENTIFIER = re.compile(r"[\w.:-]{1,160}\Z", re.UNICODE)
_PRIVATE = re.compile(r"(?:authorization|api_key|access_token|refresh_token|password|secret|credential|^body$|^content$)", re.I)
_STRINGS = {name for name, kind, _ in RUNTIME_INPUTS if kind == "string"}
_RUNTIME = {row[0] for row in [*RUNTIME_INPUTS, *CONDITIONAL_RUNTIME_INPUTS, *PENDING_RUNTIME_INPUTS]}


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class Projection:
    def __init__(self):
        self.unknown = set()
        self.string_features = set(_STRINGS)
        self.categories = {"current", "history", "relationship", "mixed", "autonomous", "conditional", "comparison",
                           "direct", "support", "unmatched", "core", "detail"}

    def identifier(self, value, path):
        if value is None:
            return None
        if isinstance(value, str) and _IDENTIFIER.fullmatch(value) and not _PRIVATE.search(value):
            return value
        self.unknown.add(path)
        return None

    def scalar(self, value, path, *, string=False):
        if value is None or type(value) is bool:
            return value
        if type(value) in (int, float):
            try:
                if math.isfinite(value):
                    return value
            except OverflowError:
                pass
        if string:
            return self.identifier(value, path)
        self.unknown.add(path)
        return None

    def expression(self, value, path):
        # Keep actual executable arithmetic, but never arbitrary string prose.
        if isinstance(value, str) and len(value) <= 2048:
            try:
                nodes = list(ast.walk(ast.parse(value or "0", mode="eval")))
                if all(not isinstance(n, ast.Constant) or not isinstance(n.value, str)
                       or n.value in self.categories for n in nodes) and not any(
                           isinstance(n, ast.Name) and _PRIVATE.search(n.id) for n in nodes):
                    return value
            except (SyntaxError, ValueError, RecursionError):
                pass
        self.unknown.add(path)
        return None

    def fields(self, row, names, path, *, strings=()):
        row = row if isinstance(row, dict) else {}
        return {name: self.scalar(row[name], path + "." + name, string=name in strings)
                for name in names if name in row}

    def definition(self, row, names, path):
        result = self.fields(row, names, path, strings=names)
        if "expression" in row:
            result["expression"] = self.expression(row["expression"], path + ".expression")
        if "weights" in row:
            result["weights"] = self.mapping(row["weights"], path + ".weights")
        if "selector" in row:
            selector = row["selector"] if isinstance(row["selector"], dict) else {}
            result["selector"] = self.fields(selector, ("kind", "stability", "topic", "query_purpose",
                "cognition_role", "cognition_status", "domain"), path + ".selector",
                strings=("kind", "stability", "topic", "query_purpose", "cognition_role", "cognition_status", "domain"))
        return result

    def mapping(self, row, path, *, strings=(), allowed=None):
        result = {}
        if not isinstance(row, dict):
            self.unknown.add(path)
            return result
        for key, value in row.items():
            if allowed is not None and key not in allowed:
                continue
            # Invalid/private keys are not copied even as diagnostic paths.
            if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key) or _PRIVATE.search(key):
                self.unknown.add(path)
                continue
            if key in strings and value is not None and value not in self.categories:
                self.unknown.add(path + "." + key)
                result[key] = None
            else:
                result[key] = self.scalar(value, path + "." + key, string=key in strings)
        return result

    def configuration(self, frozen):
        frozen = frozen if isinstance(frozen, dict) else {}
        values = frozen.get("values")
        revision = self.identifier(frozen.get("revision"), "configuration.revision")
        if not isinstance(values, dict) or revision is None:
            self.unknown.add("configuration")
            return {"revision": revision, "state": "unavailable", "values": None}
        snapshot = {"features": [], "formulas": [], "time_profiles": [], "recall": {}, "limits": {}}
        # IDs support validation/selector explanation only. Original display
        # prose, aliases, examples and annotation definitions are not a log.
        snapshot["catalog"] = {"definitions": [], "attributes": []}
        for axis in CATALOG_AXES:
            rows = (values.get("catalog") or {}).get(axis, [])
            if not isinstance(rows, list) or len(rows) > 512:
                self.unknown.add("configuration.catalog." + axis)
                rows = []
            ids = [self.identifier(item, "configuration.catalog." + axis) for item in rows]
            snapshot["catalog"][axis] = ids
            self.categories.update(item for item in ids if item)
        definitions = {
            "features": ("id", "value_type", "source", "missing", "input_key", "minimum", "maximum"),
            "formulas": ("id",),
            "time_profiles": ("id", "function", "half_life_days", "review_after_hours", "weight"),
        }
        for section, names in definitions.items():
            rows = values.get(section)
            if not isinstance(rows, list) or len(rows) > 256:
                self.unknown.add("configuration." + section)
                rows = []
            snapshot[section] = [self.definition(row, names, "configuration." + section)
                                 for row in rows if isinstance(row, dict)]
        self.string_features.update(row["id"] for row in snapshot["features"]
                                    if row.get("value_type") == "string" and row.get("id"))
        recall = values.get("recall") or {}
        for name in RECALL_DEFAULTS:
            if name not in recall:
                continue
            path = "configuration.recall." + name
            if name in {"scoring_profiles", "signal_bindings", "affinity_mappings"}:
                names = {"scoring_profiles": ("id", "formula_id"),
                         "signal_bindings": ("id", "producer", "key", "max_age_seconds"),
                         "affinity_mappings": ("id", "signal_id", "target", "weight")}[name]
                rows = recall[name]
                if not isinstance(rows, list) or len(rows) > 256:
                    self.unknown.add(path)
                    rows = []
                snapshot["recall"][name] = [self.definition(row, names, path) for row in rows if isinstance(row, dict)]
            elif name == "local_match_weights":
                snapshot["recall"][name] = self.mapping(recall[name], path)
            else:
                snapshot["recall"][name] = self.scalar(recall[name], path,
                    string=name in {"mode", "formula_id", "hierarchy_mode", "recent_context_mode", "affinity_aggregation"})
        snapshot["limits"] = self.mapping(values.get("limits", {}), "configuration.limits")
        if len(_bytes(snapshot)) > CONFIG_BYTES:
            self.unknown.add("configuration.size")
            return {"revision": revision, "state": "oversized", "values": None}
        return {"revision": revision, "state": "recorded", "values": snapshot,
                "sha256": hashlib.sha256(_bytes(snapshot)).hexdigest(),
                "scope": "scoring_definitions_and_recorded_inputs_not_matching"}

    def calculation(self, calculation, path, feature_ids):
        calculation = calculation if isinstance(calculation, dict) else {}
        if not {"features", "formulas"} <= calculation.keys():
            self.unknown.add(path + ".calculation")
        features = self.mapping(calculation.get("features", {}), path + ".inputs", strings=self.string_features, allowed=feature_ids)
        formulas = {}
        for name, row in (calculation.get("formulas") or {}).items():
            name = self.identifier(name, path + ".formula_id")
            if name and isinstance(row, dict):
                formulas[name] = self.fields(row, ("value", "state"), path + ".output", strings=("state",))
        return {"inputs": features, "formulas": formulas}

    def computation(self, row, path, feature_ids):
        if not isinstance(row, dict):
            self.unknown.add(path)
            return {"state": "unavailable"}
        result = self.fields(row, ("state", "score", "formula_id", "window_score", "winning_statement_id"),
                             path, strings=("state", "formula_id", "winning_statement_id"))
        result["reason_codes"] = [self.identifier(reason, path + ".reason_codes") for reason in (row.get("reason_codes") or [])[:16]]
        result["reason_codes_truncated"] = max(0, len(row.get("reason_codes") or []) - 16)
        result["aggregation_statement_ids"] = [self.identifier(sid, path + ".statement_id")
                                               for sid in (row.get("aggregation_statement_ids") or [])[:STATEMENT_LIMIT]]
        result["aggregation_ids_truncated"] = max(0, len(row.get("aggregation_statement_ids") or []) - STATEMENT_LIMIT)
        result["calculation"] = self.calculation(row.get("calculation"), path + ".global", feature_ids) if "calculation" in row else None
        statements = row.get("statements") or []
        # Preserve the winner and participating claims before unselected context.
        statements = sorted(statements, key=lambda s: (s.get("statement_id") != row.get("winning_statement_id"),
                                                       not s.get("participates_in_aggregation")))
        result["statement_count"] = len(statements)
        result["statements_truncated"] = max(0, len(statements) - STATEMENT_LIMIT)
        result["statements"] = []
        for statement in statements[:STATEMENT_LIMIT]:
            projected = self.fields(statement, ("statement_id", "score", "participates_in_aggregation", "match_kind",
                "formula_id", "scoring_profile", "time_profile"), path + ".statement",
                strings=("statement_id", "match_kind", "formula_id", "scoring_profile", "time_profile"))
            projected["features"] = self.mapping(statement.get("features", {}), path + ".features",
                strings=self.string_features, allowed=_RUNTIME | feature_ids)
            projected["topic_ids"] = [self.identifier(topic, path + ".topic_ids")
                                      for topic in (statement.get("topic_ids") or [])[:32]]
            projected["topic_ids_truncated"] = max(0, len(statement.get("topic_ids") or []) - 32)
            projected["calculation"] = self.calculation(statement.get("calculation"), path + ".statement", feature_ids)
            projected["reason_codes"] = [self.identifier(reason, path + ".reason_codes")
                                          for reason in (statement.get("reason_codes") or [])[:16]]
            projected["reason_codes_truncated"] = max(0, len(statement.get("reason_codes") or []) - 16)
            result["statements"].append(projected)
        return result


def _score_evidence(frozen, computation):
    """Project request-local results only; never read mutable last_request state."""
    projection = Projection()
    config = projection.configuration(frozen)
    source = computation if isinstance(computation, dict) else {}
    rows = source.get("candidate_scores") or []
    feature_ids = {row.get("id") for row in (config.get("values") or {}).get("features", []) if row.get("id")}
    result = {"schema_version": "memory-score-evidence-v1", "configuration": config,
              "state": "recorded" if isinstance(computation, dict) else "not_computed",
              "replay_scope": "recorded_computation_inputs_not_retrieval_or_matching",
              "query_purpose": projection.identifier((source.get("query_use") or {}).get("purpose"), "query_purpose"),
              "candidate_count": len(rows), "selected_count": sum(bool(row.get("selected")) for row in rows),
              "recorded_count": 0, "truncated_count": 0, "candidates": []}
    result["selection_policy"] = projection.fields(source.get("selection_policy"),
        ("semantic_threshold", "rerank_threshold", "local_threshold", "max_cards", "first_card_min_score",
         "second_card_min_score", "second_card_relative_score", "rerank_score_weight"), "selection_policy")
    # Selected -> explicitly suppressed -> admitted remainder, retaining source order.
    ordered = sorted(rows, key=lambda row: (not bool(row.get("selected")), bool(row.get("admitted"))))
    for row in ordered[:CANDIDATE_LIMIT]:
        candidate = projection.fields(row, ("bucket_id", "selected", "admitted", "admitted_before_hierarchy", "admission_reason", "selection_reason",
            "raw_base_score", "raw_fused_score", "keyword_score", "semantic_score", "rerank_score", "final_score"),
            "candidate", strings=("bucket_id", "admission_reason", "selection_reason"))
        candidate["computation"] = projection.computation(row.get("computation"), "candidate.computation", feature_ids)
        candidate["before_rerank"] = (projection.computation(row["before_rerank"], "candidate.before_rerank", feature_ids)
                                       if row.get("before_rerank") is not None else None)
        result["candidates"].append(candidate)
        if len(_bytes(result)) > EVIDENCE_BYTES - 4096:
            result["candidates"].pop()
            break
    result["recorded_count"] = len(result["candidates"])
    result["truncated_count"] = len(rows) - result["recorded_count"]
    result["unknown_fields"] = sorted(projection.unknown)[:64]
    result["unknown_field_count"] = len(projection.unknown)
    result["selected_recorded_count"] = sum(bool(row.get("selected")) for row in result["candidates"])
    result["replayable"] = (bool(computation) and config["state"] == "recorded" and not projection.unknown
                            and not result["truncated_count"]
                            and not any(phase.get("statements_truncated") or phase.get("aggregation_ids_truncated")
                                        for row in result["candidates"]
                                        for phase in (row["computation"], row["before_rerank"] or {})))
    while len(_bytes(result)) > EVIDENCE_BYTES and result["candidates"]:
        result["candidates"].pop()
        result["recorded_count"] = len(result["candidates"])
        result["truncated_count"] = len(rows) - result["recorded_count"]
        result["selected_recorded_count"] = sum(bool(row.get("selected")) for row in result["candidates"])
        result["replayable"] = False
    return result


def score_evidence(frozen, computation):
    """Observation errors cannot fail or change a model request."""
    try:
        return _score_evidence(frozen, computation)
    except Exception:
        return {"schema_version": "memory-score-evidence-v1", "state": "unavailable",
                "replayable": False, "unknown_fields": ["projection_failed"]}


def request_score_evidence(service, request):
    frozen = REQUEST_CONFIGURATION.get()
    return score_evidence(frozen if frozen and frozen.get("service") == id(service) else None,
                          request.get("configured_computation"))
