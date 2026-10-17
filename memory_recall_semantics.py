"""Request-local interpretation of committed L1 facts, never a new authority.

Dates describe evidence and applicability. A review interval is a reason to
reconfirm a state, not evidence that it ended. Unknown values stay unknown.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone
import math
import re


USAGES = frozenset({"current_basis", "conditional_basis", "relationship_reference",
                    "historical_evidence", "unresolved_evidence", "excluded"})


def known(value):
    if isinstance(value, dict):
        return value.get("value") if value.get("state") == "known" else None
    return None


def interval(value, tz=timezone.utc):
    """An ISO value's actual precision, as [start,end); no guessed date."""
    raw = known(value)
    if not isinstance(raw, str):
        return None
    try:
        if len(raw) == 4 and raw.isdigit():
            start = datetime(int(raw), 1, 1, tzinfo=tz)
            return start, start.replace(year=start.year + 1)
        if len(raw) == 7:
            start = datetime.strptime(raw, "%Y-%m").replace(tzinfo=tz)
            return start, start + timedelta(days=calendar.monthrange(start.year, start.month)[1])
        if len(raw) == 10:
            start = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=tz)
            return start, start + timedelta(days=1)
        start = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        return start, start + timedelta(microseconds=1)
    except (ValueError, OverflowError):
        return None


def predicate(node, facts):
    """The existing L1 predicate vocabulary, with three-valued logic."""
    if not isinstance(node, dict):
        return None
    op = node.get("op")
    if op in ("all", "any", "not"):
        args = node.get("args")
        if not isinstance(args, list) or not args or (op == "not" and len(args) != 1):
            return None
        values = [predicate(arg, facts) for arg in args]
        if op == "not":
            return None if values[0] is None else not values[0]
        if op == "all":
            return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    actual = known(facts.get(node.get("field")))
    expected = node.get("value")
    if actual is None:
        return None
    if op == "in":
        if not isinstance(expected, list):
            return None
        return any(type(actual) is type(item) and actual == item for item in expected)
    compatible = type(actual) is type(expected)
    numeric = (type(actual) in (int, float) and type(expected) in (int, float)
               and math.isfinite(actual) and math.isfinite(expected))
    if op == "eq":
        return actual == expected if compatible or numeric else None
    if not numeric:
        return None
    return {"lt": lambda: actual < expected, "le": lambda: actual <= expected,
            "gt": lambda: actual > expected, "ge": lambda: actual >= expected}.get(op, lambda: None)()


def conditions(value, facts):
    value = value or {}
    if value.get("state") == "not_applicable":
        result = True
    elif value.get("state") == "known" and value.get("predicate") is not None:
        result = predicate(value["predicate"], facts)
    else:
        # Even a clear-looking natural-language condition is not executable.
        result = None
    exceptions = [conditions(item, facts) for item in value.get("exceptions", [])]
    if result is False or True in exceptions:
        return False
    if result is None or None in exceptions:
        return None
    return True


def profile_for(statement, profiles, purpose):
    classification = statement.get("classification") or {}
    actual = {"kind": known(classification.get("kind")),
              "stability": known(classification.get("stability")),
              "topic": classification.get("topics") or [], "query_purpose": purpose,
              "cognition_role": known((statement.get("cognition") or {}).get("role")),
              "cognition_status": known((statement.get("cognition") or {}).get("epistemic_status"))}
    matches = []
    for position, profile in enumerate(profiles):
        selector = profile.get("selector") or {}
        if all((value in actual[axis] if axis == "topic" else actual.get(axis) == value)
               for axis, value in selector.items()):
            matches.append((-len(selector), position, profile))
    return min(matches, key=lambda row: row[:2])[2] if matches else None


def temporal_features(statement, context, profiles):
    now = context["now"]
    target = context.get("target_time") or now
    event = (statement.get("time") or {}).get("event") or {}
    event_start = interval(event.get("start"), now.tzinfo)
    event_end = interval(event.get("end"), now.tzinfo)
    event_range = ((event_start[0], event_end[1] if event_end else event_start[1])
                   if event_start else None)
    age = max(0.0, (target - event_range[0]).total_seconds() / 86400) if event_range else None
    profile = profile_for(statement, profiles, context.get("purpose", "current"))
    freshness = review_due = weight = None
    if profile:
        weight = profile.get("weight")
        function = profile["function"]
        if function == "none":
            freshness = 1.0
        elif event_range and function == "exponential":
            freshness = math.exp(-math.log(2) * age / profile["half_life_days"])
        elif event_range and function == "target_time":
            distance = max(0.0, (event_range[0] - target).total_seconds(),
                           (target - event_range[1]).total_seconds()) / 86400
            half_life = profile.get("half_life_days")
            freshness = (math.exp(-math.log(2) * distance / half_life)
                         if half_life else float(distance == 0))
        hours = profile.get("review_after_hours")
        if hours is not None and event_range:
            review_due = (now - event_range[1]).total_seconds() > hours * 3600
    return {"event_age_days": age, "freshness": freshness, "time_weight": weight,
            "review_due": review_due, "time_profile": profile.get("id") if profile else None,
            "event_range": [value.isoformat() for value in event_range] if event_range else None}


def applicability(statement, context):
    value = (statement.get("time") or {}).get("applicability") or {}
    point = context.get("target_time") or context["now"]
    start, end = value.get("start") or {}, value.get("end") or {}
    lower, upper = interval(start, point.tzinfo), interval(end, point.tzinfo)
    if lower and point < lower[0] or upper and point >= upper[1]:
        return False
    lower_known = lower is not None or start.get("state") == "unbounded"
    upper_known = upper is not None or end.get("state") == "unbounded"
    if value.get("temporal_scope") == "recurring" and value.get("recurrence_text"):
        # Natural-language recurrence is retained; do not pretend to parse it.
        return None
    return True if lower_known and upper_known else None


def resolve_statement(statement, context, profiles=(), *, source_state="unverified",
                      superseded=False, conflict=False):
    """Resolve permitted use, independently of retrieval score or source trust."""
    classification = statement.get("classification") or {}
    kind, stability, modality, basis = [known(classification.get(name))
                                         for name in ("kind", "stability", "modality", "basis")]
    time = temporal_features(statement, context, profiles)
    scope = applicability(statement, context)
    condition = conditions(statement.get("conditions"), context.get("facts") or {})
    purpose = context.get("purpose", "current")
    cognition = statement.get("cognition")
    cognitive_status = known((cognition or {}).get("epistemic_status"))
    reasons = []
    if source_state in ("revoked", "unavailable"):
        use, reasons = "excluded", ["source_" + source_state]
    elif source_state != "available":
        use, reasons = "unresolved_evidence", ["source_" + source_state]
    elif conflict:
        use, reasons = "unresolved_evidence", ["conflict_unresolved"]
    elif superseded or scope is False:
        use = "historical_evidence"
        reasons = ["superseded_for_current_use" if superseded else "outside_applicability"]
    elif purpose == "history":
        use, reasons = "historical_evidence", ["historical_query"]
    elif condition is False:
        use, reasons = "historical_evidence", ["condition_not_met"]
    elif cognition and time["review_due"] is True:
        use, reasons = "historical_evidence", ["cognition_needs_reconfirmation"]
    elif cognition and cognitive_status in (None, "tentative", "contested"):
        use, reasons = "unresolved_evidence", ["cognition_unconfirmed_or_contested"]
    elif cognition and cognitive_status == "revised":
        use, reasons = "historical_evidence", ["previous_cognition_revised"]
    elif cognition:
        use, reasons = "relationship_reference", ["cognitive_reference_not_world_fact"]
    elif basis in (None, "interpretation", "hypothesis"):
        use, reasons = "unresolved_evidence", ["evidence_basis_unconfirmed"]
    elif kind in ("preference", "boundary", "procedure") and stability == "stable":
        use = "conditional_basis" if condition is True else "relationship_reference"
        reasons = ["stable_explicit_reference", "condition_met" if condition is True else "condition_unknown"]
    elif stability == "episode" or ((statement.get("time") or {}).get("applicability") or {}).get("temporal_scope") == "episode":
        use, reasons = "historical_evidence", ["episode_does_not_establish_present"]
    elif time["review_due"] is True:
        use, reasons = "historical_evidence", ["state_needs_reconfirmation"]
    elif modality in ("request", "wish", "plan", "commitment", "question", "hypothesis"):
        use, reasons = "relationship_reference", ["expression_not_execution_or_completion"]
    elif scope is True and condition is True:
        use, reasons = "current_basis", ["explicit_scope_and_condition_match"]
    else:
        use, reasons = "relationship_reference", ["current_applicability_unconfirmed"]
    return {"statement_id": statement.get("statement_id"),
            "statement_revision_id": statement.get("statement_revision_id"),
            "content": statement.get("content", ""), "subject_ref": statement.get("subject_ref"),
            "classification": classification, "usage": use, "reason_codes": reasons,
            "scope_match": scope, "condition_match": condition, "source_state": source_state,
            "conditions": statement.get("conditions"), "time": statement.get("time"),
            "cognition": cognition,
            "evidence_refs": list(statement.get("evidence_refs") or []), "features": time}


def lexical_features(query, document, catalog):
    """Exact declared tags/aliases/entities; this is not embedding semantics."""
    query = str(query or "").casefold()
    retrieval = document.get("retrieval") or {}
    keywords = set(retrieval.get("keywords") or [])
    tags = set(retrieval.get("tags") or [])
    entities = {str(value) for item in document.get("entities", [])
                for value in [item.get("name"), *(item.get("aliases") or [])] if value}
    topics = {topic for item in document.get("statements", [])
              for topic in (item.get("classification") or {}).get("topics", [])}
    terms = {"keyword_match": keywords, "tag_match": tags, "entity_match": entities,
             "type_match": {known((item.get("classification") or {}).get("kind"))
                            for item in document.get("statements", [])},
             "cognition_role_match": {known((item.get("cognition") or {}).get("role"))
                                      for item in document.get("statements", [])},
             "cognition_question_match": {question for item in document.get("statements", [])
                                          for question in (item.get("cognition") or {}).get("open_questions", [])}}
    definitions = {(item.get("axis"), item.get("id")): item
                   for item in catalog.get("definitions", [])}
    def match(values, axis=None):
        for value in values:
            if not value:
                continue
            definition = definitions.get((axis, value), {})
            aliases = [value, definition.get("label"), *(definition.get("aliases") or [])]
            if any((re.search(r"(?<!\w)" + re.escape(str(alias).casefold()) + r"(?!\w)", query)
                    if str(alias).isascii() else str(alias).casefold() in query)
                   for alias in aliases if alias):
                return 1.0
        return 0.0
    return {**{name: match(values, "kinds" if name == "type_match" else
                          "cognition_roles" if name == "cognition_role_match" else None)
               for name, values in terms.items()}, "topic_match": match(topics, "topics")}


def evaluate_scores(values, inputs):
    """Use exactly the owner preview evaluator, preserving unknown features."""
    from memory_system_config import preview
    bound = {row["id"]: inputs.get(row.get("input_key") or row["id"])
             for row in values["features"] if row["source"] != "computed"}
    return preview(values, bound)


def resolve_document(document, context, profiles=(), *, source_states=None,
                     superseded=(), conflicts=()):
    """Each claim keeps its own use; a mixed story never gets one fact label."""
    sources = source_states or {}
    result = []
    for statement in document.get("statements") or []:
        refs = statement.get("evidence_refs") or []
        states = [sources.get(ref, "unverified") for ref in refs]
        # Alternative evidence can support a claim when one source vanished.
        state = ("available" if "available" in states else "revoked" if states and all(s == "revoked" for s in states)
                 else "unavailable" if states and all(s in {"revoked", "unavailable"} for s in states)
                 else "changed" if "changed" in states else "unverified")
        result.append(resolve_statement(statement, context, profiles, source_state=state,
                      superseded=statement.get("statement_id") in superseded,
                      conflict=statement.get("statement_id") in conflicts))
    return result


def apply_relations(records, usages, context):
    """Use explicit revision-bound relations; never infer contradiction by cosine.

Only a source-qualified counterpart can establish a replacement. A missing
counterpart or a merely proposed relation remains visible as unresolved use.
No permanent truth is written and a relation outside this query's scope does
not erase the historical claim.
"""
    endpoints, relations, replacements = {}, [], []
    for bid, row in records.items():
        if row.get("state") != "l1" or bid not in usages or usages[bid]["state"] != "l1":
            continue
        document = row["document"]
        parent = (document.get("entry") or {}).get("parent_ref")
        host = (parent["memory_id"], parent["revision"]) if parent else (row["memory_id"], row["revision"])
        for use in usages[bid].get("items", []):
            key = (*host, "statement", use["statement_id"])
            endpoints.setdefault(key, []).append(use)
            endpoints.setdefault((*host, "memory", host[0]), []).append(use)
        for scene in document.get("scenes", []):
            refs = set(scene.get("statement_refs") or [])
            endpoints.setdefault((*host, "scene", scene["scene_id"]), []).extend(
                use for use in usages[bid].get("items", []) if use["statement_id"] in refs)
        for relation in document.get("relations", []):
            relations.append((host, relation, usages[bid].get("source_states") or {}))
    def find(host, endpoint):
        memory = endpoint.get("memory_id") or host[0]
        rev = host[1] if endpoint.get("revision") == "this_revision" else endpoint.get("revision")
        obj = memory if endpoint.get("object_type") == "memory" else endpoint.get("object_id")
        return endpoints.get((memory, rev, endpoint.get("object_type"), obj), [])
    def scope(use):
        statement = {"time": use.get("time"), "conditions": use.get("conditions")}
        a, b = applicability(statement, context), conditions(statement["conditions"], context.get("facts") or {})
        return False if a is False or b is False else True if a is True and b is True else None
    def mark(use, usage, reason):
        if use["usage"] == "excluded":
            return
        # Conflict/unknown is not silently replaced by another relation's claim.
        if usage == "historical_evidence" and use["usage"] == "unresolved_evidence":
            return
        use["usage"] = usage
        if reason not in use["reason_codes"]:
            use["reason_codes"].append(reason)
    for host, relation, sources in relations:
        kind, confirmation = relation.get("kind"), relation.get("confirmation_state")
        if kind not in {"supersedes", "revises", "limits", "possible_conflict"} or confirmation == "rejected":
            continue
        left, right = find(host, relation["from"]), find(host, relation["to"])
        affected = [use for use in [*left, *right] if scope(use) is not False]
        if context.get("purpose") == "history":
            continue  # Preserve original evidence and then/now distinction.
        if not left or not right:
            for use in affected:
                mark(use, "unresolved_evidence", "related_revision_not_source_verified")
            continue
        if not affected or any(scope(use) is False for use in [*left, *right]):
            continue
        verified = (all(use.get("source_state") == "available" for use in [*left, *right])
                    and all(sources.get(ref) == "available" for ref in relation.get("evidence_refs") or []))
        trusted = confirmation == "confirmed" and relation.get("origin") in {"owner", "program"}
        if kind == "supersedes" and verified and trusted and all(scope(use) is True for use in [*left, *right]):
            replacements.extend((source, target) for source in left for target in right)
        else:
            reason = ("conflict_unresolved" if kind == "possible_conflict" else
                      "related_scope_requires_confirmation" if trusted else "proposed_relation_unconfirmed")
            for use in affected:
                mark(use, "unresolved_evidence", reason)
    # Mutually replacing claims have no consistent current winner. Keep the
    # conflicting scope unresolved instead of aging every side into history.
    edges = {}
    for source, target in replacements:
        edges.setdefault(id(source), set()).add(id(target))
    def reaches(start, target):
        pending, seen = [start], set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(edges.get(node, ()))
        return False
    for source, target in replacements:
        if reaches(id(target), id(source)):
            mark(source, "unresolved_evidence", "replacement_cycle_unresolved")
            mark(target, "unresolved_evidence", "replacement_cycle_unresolved")
        else:
            mark(target, "historical_evidence", "confirmed_revision_replacement")
    return usages


def score_document(document, values, context, inputs):
    """One scoring projection, before and after rerank, with inspectable inputs.

Time retains a proportion of an existing relevance signal; freshness cannot
create relevance on its own. Owner formulas can use raw base_relevance and
time_factor to choose a different policy. No profile means no implicit time
adjustment. Formula errors retain the base with reasons, never a guessed input.
"""
    from memory_system_config import ConfigurationError, RUNTIME_INPUTS, PENDING_RUNTIME_INPUTS
    base = inputs.get("relevance")
    if type(base) not in (int, float) or not math.isfinite(base):
        return {"score": None, "state": "unknown", "reason_codes": ["base_score_unknown"], "statements": []}
    common = {**inputs, **lexical_features(context.get("query", ""), document, values["catalog"]),
              "base_relevance": base, "query_purpose": context.get("purpose", "current")}
    reserved = {row[0] for row in [*RUNTIME_INPUTS, *PENDING_RUNTIME_INPUTS]}
    def attributes(target, items, prefix="attribute_"):
        for item in items:
            key, value = item["definition_id"], known(item.get("value"))
            target[prefix + key] = value
            if key not in reserved:
                target[key] = value
    attributes(common, document.get("attributes", []))
    for name, _kind, _label in PENDING_RUNTIME_INPUTS:
        # A memory's own arbitrary attribute cannot pretend to be a live
        # interest/drive producer. Simulations may supply those inputs explicitly.
        common[name] = inputs.get(name)
    profiles = values.get("time_profiles") or []
    policy = values.get("recall") or {}
    rows = []
    # An unannotated record still participates in the original retrieval path.
    for statement in document.get("statements") or [{}]:
        features = {**common, **temporal_features(statement, context, profiles)}
        classification = statement.get("classification") or {}
        for name in ("kind", "stability", "modality", "basis"):
            features[name] = known(classification.get(name))
        features["condition_match"] = conditions(statement.get("conditions"), context.get("facts") or {})
        features["scope_match"] = applicability(statement, context)
        features["role_hint"] = known(statement.get("role_hint"))
        cognition = statement.get("cognition")
        features["cognition_role"] = known((cognition or {}).get("role"))
        features["cognition_status"] = known((cognition or {}).get("epistemic_status"))
        features["open_question_count"] = len(cognition["open_questions"]) if cognition else None
        attributes(features, statement.get("attributes", []))
        prop = statement.get("property")
        if prop:
            features["property_" + prop["key"]] = known(prop.get("value"))
            if prop["key"] not in reserved:
                features[prop["key"]] = known(prop.get("value"))
        weight, freshness = features["time_weight"], features["freshness"]
        features["time_factor"] = (1 - weight / (1 + weight) * (1 - freshness)
                                   if weight is not None and freshness is not None else None)
        factor = features["time_factor"]
        features["relevance"] = base * factor if factor is not None and base >= 0 else base
        profile = profile_for(statement, policy.get("scoring_profiles") or [], context.get("purpose", "current"))
        formula = profile["formula_id"] if profile else policy.get("formula_id")
        if formula is None and any(row["id"] == "final" for row in values["formulas"]):
            formula = "final"
        reasons = []
        try:
            calculation = evaluate_scores(values, features)
            value = calculation["formulas"].get(formula, {}).get("value") if formula else features["relevance"]
            if value is None:
                value = base
                reasons.append("formula_unknown_retained_base")
        except ConfigurationError as exc:
            value, calculation = base, {"error": exc.payload()["error"]}
            reasons.append("formula_error_retained_base")
        rows.append({"statement_id": statement.get("statement_id"), "score": value,
                     "formula_id": formula, "scoring_profile": profile.get("id") if profile else None,
                     "time_profile": features["time_profile"], "features": features,
                     "calculation": calculation, "reason_codes": reasons})
    best = max(rows, key=lambda row: row["score"])
    return {"score": best["score"], "state": "computed", "formula_id": best["formula_id"],
            "reason_codes": best["reason_codes"], "statements": rows}


def preview_pipeline(values, inputs, scenario, *, tz=timezone.utc):
    """A synthetic claim through the actual time/use/scoring functions.

No source read, ingestion, provider, or truth write. Source availability here
is explicitly simulated so that this preview can never authorize a real claim.
"""
    from memory_system_config import ConfigurationError, validate_values, preview
    from memory_l1_schema import statement_template
    values = validate_values(values)
    # Preserve the original evaluator's unknown-input/type validations.
    preview(values, inputs)
    allowed = {"now", "event_time", "target_time", "purpose", "kind", "stability",
               "basis", "modality", "topics", "conditions", "applicability", "facts", "cognition"}
    def fail(field):
        raise ConfigurationError("invalid_preview_scenario", field="scenario." + field)
    if not isinstance(scenario, dict) or set(scenario) - allowed or not scenario.get("now"):
        fail("now")
    def date(name, required=False):
        value = scenario.get(name)
        if value in (None, "") and not required:
            return None
        if not isinstance(value, str):
            fail(name)
        bounds = interval({"state": "known", "value": value}, tz)
        if bounds is None:
            fail(name)
        return bounds[0].astimezone(tz)
    now, target = date("now", True), date("target_time")
    date("event_time")
    purpose = scenario.get("purpose", "current")
    if purpose not in {"current", "history", "relationship", "autonomous"}:
        fail("purpose")
    statement = statement_template()
    statement["content"] = "Synthetic preview claim"
    if "cognition" in scenario:
        statement["cognition"] = scenario["cognition"]
    for name, axis in (("kind", "kinds"), ("stability", "stabilities"),
                       ("modality", "modalities"), ("basis", "bases")):
        value = scenario.get(name)
        if value not in (None, ""):
            if value not in values["catalog"][axis]:
                fail(name)
            statement["classification"][name] = {"state": "known", "value": value, "evidence_refs": []}
    topics = scenario.get("topics", [])
    if not isinstance(topics, list) or any(topic not in values["catalog"]["topics"] for topic in topics):
        fail("topics")
    statement["classification"]["topics"] = topics
    event = scenario.get("event_time")
    if event:
        statement["time"]["event"]["start"] = {"state": "known", "value": event, "evidence_refs": []}
    for name in ("conditions", "applicability"):
        if name in scenario:
            # Use the existing semantic schema validator, not a second predicate
            # language. Unknown free-text conditions remain unknown at runtime.
            if name == "conditions": statement[name] = scenario[name]
            else: statement["time"][name] = scenario[name]
    facts = scenario.get("facts", {})
    if not isinstance(facts, dict):
        fail("facts")
    facts = {name: value if isinstance(value, dict) else {"state": "known", "value": value}
             for name, value in facts.items()}
    ctx = {"now": now, "target_time": target, "purpose": purpose, "facts": facts}
    # Validate optional semantic subtrees with the same L1 shape checks.
    from memory_l1_schema import document_template, validate_l1_document
    from memory_l1_contract import L1ContractError
    document = document_template("Synthetic preview only", catalog=values["catalog"])
    document["statements"] = [statement]
    try:
        validate_l1_document(document, catalog=values["catalog"])
    except L1ContractError as exc:
        fail(str(getattr(exc, "field_path", "statement")))
    runtime_inputs = {row.get("input_key") or row["id"]: inputs.get(row["id"])
                      for row in values["features"] if row["source"] != "computed"}
    calculation = score_document(document, values, ctx, runtime_inputs)
    best = max(calculation["statements"], key=lambda row: row["score"]) if calculation["statements"] else {}
    statement["evidence_refs"] = ["simulation"]
    usage = resolve_document(document, ctx, values["time_profiles"], source_states={"simulation": "available"})
    return {**best.get("calculation", {"features": {}, "formulas": {}}),
            "provider_requests": 0, "saved": False,
            "pipeline": {**calculation, "usage": usage, "context": {
                "now": now.isoformat(), "target_time": target.isoformat() if target else None,
                "purpose": purpose, "simulation": True, "source_state": "simulated_only"}}}
