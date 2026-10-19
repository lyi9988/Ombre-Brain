"""The existing Gateway's L1/legacy projection and source-check helpers."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import time

import httpx

from memory_recall_semantics import (apply_relations, known, lexical_features, resolve_document,
                                     score_document, temporal_features)
from memory_system_config import REQUEST_CONFIGURATION, values_from_config
from memory_affinity import resolve_signals
from memory_query_evidence import query_purpose, match_statements, correction_matches, semantic_source_refs


def deadline(service, request):
    """One owner-configured recall budget, including final source/L2 IO."""
    if "_deadline" not in request:
        request["_deadline"] = time.monotonic() + getattr(service, "recall_timeout_seconds", 15.0)
    return request["_deadline"]


def remaining(service, request):
    return max(0.0, deadline(service, request) - time.monotonic())


def authority_stamp(service):
    reader = getattr(getattr(service, "memory_authority_view", None), "_file_stamp", None)
    try:
        return reader() if callable(reader) else None
    except OSError:
        return None


async def snapshot_sources_current(service, entry, request):
    """Cached prose cannot outlive the source qualification that produced it."""
    guard = entry.get("memory_guard")
    view = getattr(service, "memory_authority_view", None)
    if not guard:
        return not getattr(view, "enabled", False)
    if not guard["complete"]:
        return False
    stamp = authority_stamp(service)
    if getattr(view, "enabled", False) and (stamp is None or stamp != guard["authority_stamp"]):
        return False
    ids = list(guard["records"])
    if not ids:
        return True
    metadata = view.memory_index_metadata_map(ids)
    records = view.memory_l1_recall_map(ids)
    if any(not metadata.get(bid) or metadata[bid].get("revision") != old["revision"]
           or metadata[bid].get("memory_id") != old["memory_id"]
           or metadata[bid].get("body_sha256") != old["body_sha256"]
           for bid, old in guard["records"].items()):
        return False
    sub = {"_semantic_records": records, "_deadline": deadline(service, request),
           "recall_io": request.setdefault("recall_io", {}), "_chain_projection": True}
    permitted = await verify_sources(service, sub, guard["query"],
        [{"bucket": {"id": bid}, "score": 0.0} for bid in ids], refresh=True)
    request.setdefault("incomplete_reasons", []).extend(sub.get("incomplete_reasons") or [])
    if {item["bucket"]["id"] for item in permitted} != set(ids):
        return False
    for bid, old in guard["records"].items():
        usage = sub["usage_by_bucket"][bid]
        # Replaying an old whole narrative after losing only one source is
        # unsafe even if another claim keeps the same bucket eligible.
        if usage["state"] == "l1":
            if not usage.get("narrative_allowed"):
                return False
            current = {use["statement_id"]: use["usage"] for use in usage["items"]}
            if any(current.get(sid) != use for sid, use in old["statement_uses"].items()):
                return False
    return stamp == authority_stamp(service)


def snapshot_guard(service, request, *, complete):
    view = getattr(service, "memory_authority_view", None)
    ids = request.get("injected_usage_bucket_ids") or []
    reader = getattr(view, "memory_index_metadata_map", None)
    metadata = reader(ids) if callable(reader) else {}
    records = {}
    for bid in ids:
        row = metadata.get(bid)
        if not row:
            complete = False
            continue
        records[bid] = {key: row.get(key) for key in ("memory_id", "revision", "body_sha256")}
        records[bid]["statement_uses"] = {use["statement_id"]: use["usage"] for use in
            request.get("usage_by_bucket", {}).get(bid, {}).get("items", [])}
    return {"complete": complete, "authority_stamp": authority_stamp(service),
            "query": request.get("q_current", ""), "records": records}


async def read_batch(service, request, stage, path, payload):
    """Coalesce identical in-flight reads inside this request, never across owners.

    Completed source qualifications are cached separately with their exact
    binding scope. A final refresh still makes a new observation of L0.
    """
    budget = remaining(service, request)
    stats = request.setdefault("recall_io", {}).setdefault(stage, {
        "calls": 0, "singleflight_hits": 0, "duration_ms": 0, "timeouts": 0,
    })
    if budget <= 0:
        stats["timeouts"] += 1
        raise asyncio.TimeoutError
    base = os.environ.get("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://ombre-brain:8000").rstrip("/")
    token = os.environ.get("OMBRE_MEMORY_WRITE_TOKEN") or getattr(service, "gateway_token", "")
    key = (base, hashlib.sha256(token.encode()).hexdigest(), path,
           json.dumps(payload, sort_keys=True, ensure_ascii=False))
    pending = request.setdefault("_recall_reads", {})
    task = pending.get(key)
    owner = task is None
    if owner:
        async def fetch():
            stats["calls"] += 1
            started = time.perf_counter()
            try:
                # HTTPX's per-operation cap is not a new stage allowance. The
                # surrounding absolute deadline covers connect/read/cleanup.
                async with httpx.AsyncClient(timeout=min(20.0, budget), trust_env=False) as client:
                    response = await client.post(base + path,
                        headers={"Authorization": "Bearer " + token}, json=payload)
                    response.raise_for_status()
                    return response.json()
            finally:
                stats["duration_ms"] += max(0, int((time.perf_counter() - started) * 1000))
        task = asyncio.create_task(fetch())
        pending[key] = task
    else:
        stats["singleflight_hits"] += 1
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=remaining(service, request))
    except asyncio.TimeoutError:
        stats["timeouts"] += 1
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    except asyncio.CancelledError:
        if owner:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if stage == "sources":
                request["_source_owner_cancelled"] = True
        raise
    finally:
        if pending.get(key) is task and task.done():
            pending.pop(key, None)


def values(service):
    snapshot = REQUEST_CONFIGURATION.get()
    if snapshot and snapshot.get("service") == id(service):
        return snapshot["values"]
    return values_from_config(getattr(service, "config", {}))


def option(service, name, fallback=None):
    value = values(service).get("recall", {}).get(name)
    return fallback if value is None else value


def context(service, request, query):
    if "_use_context" not in request:
        from turn_clock import reference_now
        now = reference_now(getattr(service, "gateway_tz", timezone.utc))
        date_hint = service._query_date_recall_hint(query) if hasattr(service, "gateway_tz") else None
        planner = getattr(service.recall_policy, "plan_query", None)
        plan = planner(query) if callable(planner) else None
        target = (datetime.fromisoformat(date_hint["date"]).replace(tzinfo=now.tzinfo)
                  if date_hint and date_hint.get("date") else None)
        temporal_intent = query_purpose(query, target_time=target, now=now)
        request["_use_context"] = {
            "now": now, "target_time": target, "query": query,
            **temporal_intent,
            "overview": bool(date_hint and getattr(plan, "auto_too_vague", False)),
            "facts": {"hour": {"state": "known", "value": now.hour},
                      "weekday": {"state": "known", "value": now.weekday()}},
            "signals": resolve_signals(values(service), request.get("_signal_snapshots", []),
                                       request.get("_signal_sampled_at") or datetime.now(timezone.utc)),
        }
    return request["_use_context"]


def prepare(service, request, query, buckets):
    settings = values(service)
    mode = settings.get("recall", {}).get("mode", "inherit")
    if mode == "fast_local":
        request.update(local_only=True, allow_semantic=False, allow_rerank=False)
    elif mode == "fast_hybrid" and not request.get("local_only"):
        request["allow_semantic"] = True
    context(service, request, query)
    request.setdefault("metadata", {})["state_signals"] = {
        "transport": request.get("_signal_transport", {"state": "unavailable", "reason": "not_supplied"}),
        "bindings": request["_use_context"].get("signals", {}),
    }
    view = getattr(service, "memory_authority_view", None)
    reader = getattr(view, "memory_l1_recall_map", None)
    request["_semantic_records"] = reader([b["id"] for b in buckets]) if callable(reader) else {}
    if getattr(view, "enabled", False) and callable(reader):
        for bucket in buckets:
            request["_semantic_records"].setdefault(str(bucket["id"]), {"state": "semantic_unavailable"})
    request["_semantic_buckets"] = {str(b["id"]): b for b in buckets}


def add_local_candidates(service, request, query, pool):
    settings = values(service)
    present = {str(item["bucket"]["id"]) for item in pool}
    for bid, record in request.get("_semantic_records", {}).items():
        document = record.get("document")
        if record["state"] != "l1" or not document or bid in present:
            continue
        signals = lexical_features(query, document, settings["catalog"])
        weights = settings["recall"]["local_match_weights"]
        # A generic type/role hit is a search cue, not 100% content relevance.
        # Weights are owner-visible; matching more labels cannot inflate a sum.
        strength = max((signal * weights[name] for name, signal in signals.items()), default=0)
        if strength > 0 and bid in request["_semantic_buckets"]:
            pool.append({"bucket": request["_semantic_buckets"][bid], "score": strength,
                         "keyword_score": 0.0, "semantic_score": 0.0,
                         "l1_local_match": signals})
    return pool


def score_pool(service, request, query, pool, *, after_rerank=False):
    settings = values(service)
    use_context = context(service, request, query)
    rows = request.get("_semantic_records", {})
    for item in pool:
        bid = str(item["bucket"]["id"])
        record = rows.get(bid) or {}
        document = record.get("document") or {"retrieval": {}, "statements": [], "entities": []}
        if "_raw_base_score" not in item:
            item["_raw_base_score"] = item["score"]
        base = item.get("_raw_fused_score", item["_raw_base_score"]) if after_rerank else item["_raw_base_score"]
        metadata = item["bucket"].get("metadata") or {}
        from memory_metadata import normalize_domain_key
        # Explicit canonical/domain metadata only; inferred free tags are not
        # evidence for a state-affinity mapping.
        domain = normalize_domain_key(metadata.get("canonical_domain") or item["bucket"].get("canonical_domain"))
        semantic = item.get("semantic_score")
        if request.get("local_only") or request.get("allow_semantic") is False:
            semantic = None
        calculation = score_document(document, settings, use_context, {
            "relevance": base, "keyword_score": item.get("keyword_score"),
            "semantic_score": semantic, "rerank_score": item.get("rerank_score"),
            "importance": metadata.get("importance", item["bucket"].get("importance")),
            "confidence": metadata.get("confidence", item["bucket"].get("confidence")),
            "canonical_domain": domain or None,
            "statement_selection": item.get("_statement_selection"),
            "eligible_statement_ids": ([use["statement_id"] for use in
                request.get("usage_by_bucket", {}).get(bid, {}).get("items", [])]
                if bid in request.get("usage_by_bucket", {}) else None),
        })
        policy_formula = calculation.get("formula_id") or "final"
        formula = next((f for f in settings["formulas"] if f["id"] == policy_formula), {})
        custom_formula = (formula.get("expression", "relevance").strip() != "relevance"
                          or bool(settings.get("recall", {}).get("formula_id"))
                          or any(row.get("participates_in_aggregation") and row.get("scoring_profile")
                                 for row in calculation["statements"]))
        time_applied = any(row.get("participates_in_aggregation") and row["features"].get("freshness") is not None
                           and row["features"].get("time_weight") is not None
                           for row in calculation["statements"])
        formula_failed = any(reason in {"formula_unknown_retained_base", "formula_error_retained_base"}
                             for reason in calculation["reason_codes"])
        calculation["window_score"] = (calculation["score"] if not formula_failed
            and (custom_formula or time_applied or item.get("l1_local_match")) else None)
        item["_configured_score"] = calculation
        if calculation["score"] is not None:
            item["score"] = calculation["score"]
            if after_rerank:
                item["combined_score"] = item["score"]
    return pool


async def verify_sources(service, request, query, items, *, refresh=False):
    """No provider model calls; exact current L0 readers remain on Brain."""
    records = request.get("_semantic_records", {})
    view = getattr(service, "memory_authority_view", None)
    authority_generation = authority_stamp(service)
    endpoint = os.environ.get("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://ombre-brain:8000").rstrip("/")
    credential = os.environ.get("OMBRE_MEMORY_WRITE_TOKEN") or getattr(service, "gateway_token", "")
    authority_scope = (id(service), endpoint, hashlib.sha256(credential.encode()).hexdigest())
    wanted, bindings_by_bucket, cache_keys = [], {}, {}
    for item in items:
        bid = str(item["bucket"]["id"])
        record = records.get(bid) or {}
        document = record.get("document")
        if record.get("state") == "l1" and document:
            referenced = semantic_source_refs(document)
            bindings = [binding["binding_id"] for binding in document.get("evidence_bindings", [])
                        if not referenced or binding["binding_id"] in referenced]
            bindings_by_bucket[bid] = bindings
            if bindings:
                wanted.append({"memory_id": record["memory_id"], "revision": record["revision"],
                               "binding_ids": bindings,
                               **({"allow_historical": True} if record.get("historical") else {})})
                # Binding identities alone do not prove unchanged provenance.
                cache_keys[(record["memory_id"], record["revision"])] = (
                    authority_scope,
                    record["memory_id"], record["revision"], bool(record.get("historical")),
                    authority_generation,
                    json.dumps([b for b in document.get("evidence_bindings", [])
                                if b["binding_id"] in bindings], sort_keys=True, ensure_ascii=False))
    frozen = REQUEST_CONFIGURATION.get()
    validation = (frozen.setdefault("source_validation", {})
                  if frozen and frozen.get("service") == id(service) else request)
    checked = validation.setdefault("_checked_sources", {})
    if refresh:
        # Selection may have awaited semantic/rerank work since this cache was
        # filled. Revalidate the final batch, including when L2 is not ready.
        for item in wanted:
            checked.pop(cache_keys[(item["memory_id"], item["revision"])], None)
    stats = request.setdefault("recall_io", {}).setdefault("sources", {
        "calls": 0, "singleflight_hits": 0, "duration_ms": 0, "timeouts": 0,
    })
    stats["cache_hits"] = stats.get("cache_hits", 0) + sum(
        cache_keys[(item["memory_id"], item["revision"])] in checked for item in wanted)
    wanted = list({cache_keys[(item["memory_id"], item["revision"])]: item for item in wanted
                   if cache_keys[(item["memory_id"], item["revision"])] not in checked}.values())
    if wanted and not validation.get("_source_check_cancelled"):
        try:
            response = await read_batch(service, request, "sources",
                "/api/memory-authority/recall-sources", {"items": wanted})
            for row in response["items"]:
                identity = (row["memory_id"], row["revision"])
                if identity in cache_keys:
                    checked[cache_keys[identity]] = row
        except asyncio.CancelledError:
            if request.get("_source_owner_cancelled"):
                validation["_source_check_cancelled"] = True
            raise
        except asyncio.TimeoutError:
            request.setdefault("incomplete_reasons", []).append("final_source_deadline" if refresh else "source_deadline")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            request.setdefault("incomplete_reasons", []).append("l0_source_verification_unavailable")
    result = []
    use_context, settings = context(service, request, query), values(service)
    usage_by_bucket = request.setdefault("usage_by_bucket", {})
    if frozen and frozen.get("service") == id(service) and not request.get("_chain_projection"):
        frozen["memory_uses"] = usage_by_bucket
    for item in items:
        bid = str(item["bucket"]["id"])
        usage_by_bucket[bid] = {"state": "excluded", "items": [], "narrative_allowed": False}
        record = records.get(bid)
        if record and record.get("state") not in {"legacy", "l1"}:
            item["admission_reason"] = "l1_" + record["state"]
            request.setdefault("incomplete_reasons", []).append(item["admission_reason"])
            continue
        document = (record or {}).get("document")
        if not document:
            usage_by_bucket[bid] = {"state": "legacy", "items": [], "narrative_allowed": True}
            result.append(item)
            continue
        check = checked.get(cache_keys.get((record["memory_id"], record["revision"])))
        if bindings_by_bucket.get(bid) and (not check or check.get("state") != "ok"):
            item["admission_reason"] = "l0_source_not_current_or_eligible"
            request.setdefault("incomplete_reasons", []).append(item["admission_reason"])
            continue
        statuses = (check or {}).get("sources") or {}
        source_states = {key: "available" if value.get("eligible_source") is True
                         else "revoked" if value.get("state") in {"withdrawn", "revoked"}
                         else "unavailable" for key, value in statuses.items()}
        item_context = {**use_context, "purpose": "history"} if record.get("historical") else use_context
        uses = resolve_document(document, item_context, settings["time_profiles"], source_states=source_states)
        allowed = [use for use in uses if use["usage"] != "excluded"]
        narrative_allowed = all(source_states.get(key) == "available" for key in bindings_by_bucket.get(bid, []))
        if (uses and not allowed) or (not uses and not narrative_allowed):
            item["admission_reason"] = "l0_source_not_eligible"
            continue
        usage_by_bucket[bid] = {"state": "l1", "items": allowed,
                               "narrative_allowed": narrative_allowed, "record": record,
                               "source_states": source_states, "query_purpose": use_context["purpose"]}
        result.append(item)
    apply_relations(records, usage_by_bucket, use_context)
    for item in result:
        bid = str(item["bucket"]["id"])
        usage = usage_by_bucket[bid]
        record = records.get(bid) or {}
        if usage["state"] != "l1":
            continue
        selection = match_statements(record["document"], {**use_context,
            "memory_id": record.get("memory_id"), "revision": record.get("revision")})
        eligible_ids = {use["statement_id"] for use in usage["items"] if use["usage"] != "excluded"}
        # Keep the declared dependency before eligibility filtering: losing a
        # necessary condition is not permission to present the positive claim.
        selection["required_support_ids"] = list(selection["support_ids"])
        selection["missing_support_ids"] = [sid for sid in selection["support_ids"] if sid not in eligible_ids]
        for key in ("direct_ids", "support_ids"):
            selection[key] = [sid for sid in selection[key] if sid in eligible_ids]
        item["_statement_selection"] = selection
        usage["statement_selection"] = selection
        if selection["missing_support_ids"]:
            item["admission_reason"] = "required_statement_support_not_eligible"
            usage.update(state="excluded", narrative_allowed=False)
            continue
        explicit = {use["statement_id"] for use in usage["items"]
                    if use.get("source_state") == "available"
                    and known((use.get("classification") or {}).get("basis")) in {"direct_expression", "record"}}
        corrections = [sid for sid in correction_matches(record["document"], use_context, selection) if sid in explicit]
        usage["correction_statement_ids"] = corrections
        item["correction_statement_ids"] = corrections
        item["matched_correction_background"] = bool(corrections and settings["recall"].get("include_correction_background", True))
    return [item for item in result if usage_by_bucket[str(item["bucket"]["id"])]["state"] != "excluded"]


def usage_lines(usage):
    rows = []
    if usage["state"] == "legacy":
        rows.append("[historical_evidence] legacy_without_annotations; event_time=unknown; present_state=unconfirmed")
    else:
        entities = {item["entity_id"]: item.get("name") for item in usage["record"]["document"].get("entities", [])}
        for item in usage["items"]:
            event = (item.get("time") or {}).get("event") or {}
            when = known(event.get("start")) or "unknown"
            subject = entities.get(known(item.get("subject_ref"))) or "unknown"
            constraint = item.get("conditions") or {}
            condition = constraint.get("text") or constraint.get("predicate")
            basis = known((item.get("classification") or {}).get("basis")) or "unknown"
            row = f'[{item["usage"]}; subject={subject}; basis={basis}; event_time={when}] {item["content"]}'
            cognition = item.get("cognition")
            if cognition:
                role = known(cognition.get("role")) or "unknown"
                status = known(cognition.get("epistemic_status")) or "unknown"
                about = [entities.get(ref) or "unknown" for ref in cognition.get("about_refs", [])]
                row += f' | cognition={role}/{status}'
                if about:
                    row += " | about=" + ", ".join(about)
                if cognition.get("open_questions"):
                    row += " | open_questions=" + "; ".join(cognition["open_questions"])
            if condition:
                row += " | condition=" + str(condition)
            exceptions = [value.get("text") or value.get("predicate") for value in constraint.get("exceptions", [])]
            if any(exceptions):
                row += " | exceptions=" + "; ".join(str(value) for value in exceptions if value)
            scope = (item.get("time") or {}).get("applicability") or {}
            bounds = [known(scope.get(key)) or (scope.get(key) or {}).get("state", "unknown") for key in ("start", "end")]
            if scope.get("temporal_scope") not in (None, "unknown"):
                row += f' | scope={scope["temporal_scope"]}:{bounds[0]}..{bounds[1]}'
            rows.append(row)
    return "\n".join(rows)


def render(service, usage, header, original, budget, *, trace=None):
    """A source-faithful interim L1 view, not the future persisted L2."""
    if usage["state"] == "excluded":
        return ""
    # Factory/body/owner_override/off are resolved through the normal Composer
    # component registry. Turning it off must not resurrect the old raw body.
    from memory_readable_projection import qualified_units, qualified_prose, fit_units
    selection = usage.get("statement_selection") or {}
    if selection.get("direct_ids") and usage.get("scope_to_query", True):
        refs = set(selection["direct_ids"]) | set(selection.get("support_ids") or [])
        usage = {**usage, "items": [item for item in usage.get("items", []) if item["statement_id"] in refs]}
    required = set(usage.get("correction_statement_ids") or [])
    required.update(selection.get("support_ids") or [])
    if usage.get("scope_to_query", True):
        required.update(selection.get("direct_ids") or [])
    if required:
        usage = {**usage, "items": sorted(usage.get("items", []), key=lambda item: item["statement_id"] not in required)}
        by_id = {item["statement_id"]: item for item in usage["items"]}
        missing = [sid for sid in required if sid not in by_id or not qualified_units(
            {**usage, "items": [by_id[sid]], "narrative_allowed": False}, "")]
        if missing:
            if trace is not None:
                trace.update(render_reason="required_context_not_readable", unrenderable_required_statement_ids=sorted(missing))
            return ""
    def wrap(narrative):
        return service._resolve_gateway_fixed_prompt("ombre.memory_usage_card_prompt", runtime_values={
            "header": "", "statements": "", "narrative": narrative,
        })
    units = qualified_units(usage, original)
    if required:
        # A fallback must not retain a positive claim but drop its required
        # correction/condition merely because another paragraph fits first.
        units = [qualified_prose(usage, original)] if units else []
    return fit_units(units, budget, wrap=wrap)


def render_presentation(service, usage, header, presentation, budget):
    """Render Brain's already-qualified persisted L2 through the existing card."""
    if usage["state"] == "excluded":
        return "", "usage_excluded", [], False
    from utils import count_tokens_approx

    # L1 controls qualification and purpose; its prose must not duplicate the
    # separately persisted L2 body or consume the L2 narrative's budget.
    presentation_text = str(presentation.get("text") or "")

    def wrap(narrative):
        return service._resolve_gateway_fixed_prompt("ombre.memory_usage_card_prompt", runtime_values={
            "header": "", "statements": "",
            # Brain has already applied source, dependency, and per-statement
            # use checks to the complete L2 or to each returned unit.
            "narrative": narrative,
        })

    text = wrap(presentation_text)
    if not text:
        return "", "usage_card_prompt_unavailable", [], False

    units = []
    seen_unit_ids = set()
    for unit in presentation.get("units") or []:
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id") or "").strip()
        unit_text = unit.get("text")
        if (not unit_id or unit_id in seen_unit_ids
                or not isinstance(unit_text, str) or not unit_text.strip()):
            continue
        seen_unit_ids.add(unit_id)
        units.append({"unit_id": unit_id, "text": unit_text,
                      "statement_refs": list(unit.get("statement_refs") or [])})
    unit_ids = [unit["unit_id"] for unit in units]
    required = set(usage.get("correction_statement_ids") or [])
    required.update((usage.get("statement_selection") or {}).get("support_ids") or [])
    required.update((usage.get("statement_selection") or {}).get("direct_ids") or [])
    coverage = presentation.get("coverage")
    if coverage is not None and not required <= set(coverage.get("used_statement_ids") or []):
        return "", "required_l2_context_missing", [], False

    # Keep the complete preferred L2 whenever it fits the remaining owner
    # budget. Otherwise choose complete resolver-approved units, never slices.
    if count_tokens_approx(text) <= max(0, budget):
        return text, None, unit_ids, False
    if not units:
        return "", "presentation_exceeds_remaining_budget", [], False

    wrapper_cost = max(0, count_tokens_approx(text) - count_tokens_approx(presentation_text))
    available_narrative_budget = max(0, budget - wrapper_cost)
    selected = []
    matched = set((usage.get("statement_selection") or {}).get("direct_ids") or [])
    ordered = sorted(enumerate(units), key=lambda pair: (
        -len(set(pair[1]["statement_refs"]) & required),
        -len(set(pair[1]["statement_refs"]) & matched), pair[0]))
    for _position, unit in ordered:
        candidate = selected + [unit]
        narrative = "\n\n".join(item["text"] for item in candidate)
        if count_tokens_approx(narrative) <= available_narrative_budget:
            selected = candidate
    if not selected:
        return "", "no_complete_unit_fits_remaining_budget", [], False
    if required and not required <= {ref for unit in selected for ref in unit["statement_refs"]}:
        return "", "no_complete_correction_unit_fits_remaining_budget", [], False
    selected_ids = {unit["unit_id"] for unit in selected}
    selected = [unit for unit in units if unit["unit_id"] in selected_ids]

    # The final result receives one shared usage card wrapper around all units.
    # Recheck its exact estimate and drop whole trailing units if wrapper
    # boundaries made the conservative estimate differ by a token.
    while selected:
        narrative = "\n\n".join(item["text"] for item in selected)
        selected_text = wrap(narrative)
        if required and not required <= {ref for unit in selected for ref in unit["statement_refs"]}:
            return "", "no_complete_correction_unit_fits_remaining_budget", [], False
        if selected_text and count_tokens_approx(selected_text) <= max(0, budget):
            selected_ids = [item["unit_id"] for item in selected]
            return selected_text, None, selected_ids, len(selected_ids) < len(unit_ids)
        selected.pop()
    return "", "no_complete_unit_fits_remaining_budget", [], False


async def resolve_presentations(service, candidates, *, request=None):
    """Resolve only source-qualified, non-targeted L1 memories in one Brain call."""
    if not candidates:
        return {}, {}
    request = request if request is not None else {}
    items = []
    for candidate in candidates:
        usage = candidate["usage"]
        statement_ids = list(dict.fromkeys(
            str(item.get("statement_id") or "")
            for item in usage.get("items", [])
            if item.get("statement_id") and item.get("usage") != "excluded"
        ))
        allowed_uses = list(dict.fromkeys(
            str(item.get("usage") or "")
            for item in usage.get("items", [])
            if item.get("usage") and item.get("usage") != "excluded"
        ))
        statement_uses = {
            str(item["statement_id"]): str(item["usage"])
            for item in usage.get("items", [])
            if item.get("statement_id") and item.get("usage") != "excluded"
        }
        items.append({
            "memory_id": candidate["memory_id"],
            "revision": candidate["revision"],
            "statement_ids": statement_ids,
            "allowed_uses": allowed_uses,
            "statement_uses": statement_uses,
            "narrative_allowed": bool(usage.get("narrative_allowed")),
            **({"matched_statement_ids": usage["statement_selection"]["direct_ids"],
                "support_statement_ids": usage["statement_selection"]["support_ids"]}
               if (usage.get("statement_selection") or {}).get("direct_ids") else {}),
        })

    base = os.environ.get("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://ombre-brain:8000").rstrip("/")
    token = os.environ.get("OMBRE_MEMORY_WRITE_TOKEN") or getattr(service, "gateway_token", "")
    failure = "brain_unconfigured" if not base or not token else None
    rows_by_memory = {}
    try:
        if failure:
            raise RuntimeError(failure)
        payload = await read_batch(service, request, "l2_resolve",
            "/api/memory-authority/presentations/resolve", {"items": items})
        if not isinstance(payload, dict) or payload.get("status") != "ok" \
                or not isinstance(payload.get("items"), list):
            raise ValueError("invalid_presentation_response")
        for row in payload["items"]:
            if isinstance(row, dict) and row.get("memory_id"):
                rows_by_memory[str(row["memory_id"])] = row
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        failure = "resolver_deadline"
        request.setdefault("incomplete_reasons", []).append(failure)
    except RuntimeError:
        pass
    except httpx.HTTPStatusError as exc:
        failure = f"resolver_http_{exc.response.status_code}"
    except httpx.HTTPError:
        failure = "resolver_transport_error"
    except (ValueError, TypeError, KeyError):
        failure = "resolver_invalid_response"

    presentations, trace = {}, {}
    for candidate in candidates:
        bid = candidate["bucket_id"]
        memory_id = candidate["memory_id"]
        usage = candidate["usage"]
        common = {
            "bucket_id": bid,
            "memory_id": memory_id,
            "l1_revision": candidate["revision"],
            "presentation_revision": None,
            "resolved_presentation_revision": None,
            "selected_unit_ids": [],
            "partial_presentation": False,
            "resolver_state": None,
            "source": "l1",
            "fallback_reason": failure or "resolver_missing_item",
            "query_purpose": usage.get("query_purpose"),
            "statement_selection": usage.get("statement_selection"),
        }
        row = rows_by_memory.get(memory_id) if not failure else None
        if row is not None:
            state = str(row.get("state") or "unknown")
            reason = str(row.get("reason") or "").strip()
            common["resolver_state"] = state
            common["fallback_reason"] = None
            if state != "ready":
                common["fallback_reason"] = reason or f"presentation_{state}"
            else:
                text = row.get("text")
                revision = row.get("presentation_revision")
                if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
                    common["fallback_reason"] = "invalid_presentation_revision"
                elif not isinstance(text, str) or not text.strip():
                    common["resolved_presentation_revision"] = revision
                    common["fallback_reason"] = "empty_ready_presentation"
                else:
                    common["resolved_presentation_revision"] = revision
                    common["fallback_reason"] = None
                    presentations[bid] = {
                        "text": text,
                        "units": row.get("units") if isinstance(row.get("units"), list) else [],
                        "presentation_revision": revision,
                        "resolver_state": state,
                        "coverage": row.get("coverage"),
                    }
                    common["l2_coverage"] = row.get("coverage")
        trace[bid] = common
    return presentations, trace


def hierarchy_pool(service, request, pool):
    mode = option(service, "hierarchy_mode", "auto")
    if mode == "mixed":
        return pool
    if mode == "auto":
        mode = "overview" if request.get("_use_context", {}).get("overview") else "scene"
    records = request.get("_semantic_records", {})
    grouped = {}
    for item in pool:
        bid = str(item["bucket"]["id"])
        row = records.get(bid) or {}
        parent = ((row.get("document") or {}).get("entry") or {}).get("parent_ref")
        root = parent["memory_id"] if parent else row.get("memory_id", bid)
        grouped.setdefault(root, []).append((bool(parent), item))
    result = []
    for group in grouped.values():
        parents = [item for child, item in group if not child]
        children = [item for child, item in group if child]
        preferred = parents if mode == "overview" else children
        result.extend(preferred or [item for _child, item in group])
    kept = {str(item["bucket"]["id"]) for item in result}
    request["hierarchy_suppressed"] = [str(item["bucket"]["id"]) for item in pool if str(item["bucket"]["id"]) not in kept]
    return [item for item in pool if str(item["bucket"]["id"]) in kept]


async def cognition_context(service, request, query, permitted, budget):
    """Trace explicit cognitive dependencies; never re-run semantic retrieval."""
    from utils import count_tokens_approx
    view = getattr(service, "memory_authority_view", None)
    reader = getattr(view, "cognition_chain", None)
    records = request.get("_semantic_records") or {}
    seeds = [(row["memory_id"], row["revision"]) for item in permitted
             if (row := records.get(str(item["bucket"]["id"]))) and row.get("state") == "l1"
             and any(s.get("cognition") for s in row["document"].get("statements", []))]
    depth, limit = option(service, "cognition_chain_depth", 2), option(service, "cognition_chain_nodes", 8)
    if not seeds or not callable(reader) or depth <= 0 or limit <= 0 or budget <= 0:
        return ""
    graph = reader(seeds, depth=depth, limit=limit)
    initial = set(seeds)
    rows = [row for row in graph["nodes"] if (row["memory_id"], row["revision"]) not in initial]
    sub = {"_semantic_records": {str(index): {**row, "state": "l1"} for index, row in enumerate(rows)},
           "_use_context": context(service, request, query), "_chain_projection": True,
           "_deadline": deadline(service, request),
           "_recall_reads": request.setdefault("_recall_reads", {}),
           "recall_io": request.setdefault("recall_io", {})}
    candidates = [{"bucket": {"id": str(index)}, "score": 0.0} for index in range(len(rows))]
    allowed = await verify_sources(service, sub, query, candidates)
    request.setdefault("incomplete_reasons", []).extend(sub.get("incomplete_reasons") or [])
    parts, spans, remaining = [], [], budget
    for item in allowed:
        key = item["bucket"]["id"]
        row = rows[int(key)]
        usage = sub["usage_by_bucket"][key]
        # The main selected version is already shown. Repeated unchanged claim
        # versions in old snapshots add no further cognitive evidence.
        initial_versions = {s.get("statement_revision_id") for seed in graph["nodes"]
                            if (seed["memory_id"], seed["revision"]) in initial
                            for s in seed["document"].get("statements", [])}
        uses = [use for use in usage["items"]
                if not row["historical"] or use.get("statement_revision_id") not in initial_versions]
        paths = row.get("via_paths") or ([row["via"]] if row.get("via") else [])
        refs = set()
        whole = any(path.get("kind") == "revision_history" or path.get("object_type") == "memory" for path in paths)
        for path in paths:
            if path.get("object_type") == "statement":
                refs.add(path["object_id"])
            elif path.get("object_type") == "scene":
                scene = next((scene for scene in row["document"].get("scenes", [])
                              if scene["scene_id"] == path["object_id"]), None)
                if scene:
                    refs.update(scene["statement_refs"])
        if paths and not whole:
            uses = [use for use in uses if use["statement_id"] in refs]
        if not uses:
            continue
        from memory_readable_projection import qualified_prose, fit_units
        label = '以前记下的想法：' if row['historical'] else '相关的想法：'
        text = label + "\n" + qualified_prose({**usage, 'items': uses, 'narrative_allowed': False}, '')
        for use in uses:
            cognition = use.get('cognition') or {}
            status = known(cognition.get('epistemic_status'))
            if status in {'tentative', 'contested', 'revised'}:
                text += '\n' + {'tentative': '这段想法当时尚未确定。', 'contested': '这段想法仍有争议。',
                                 'revised': '这段想法后来有过修订。'}[status]
            for question in cognition.get('open_questions') or []:
                if question and question not in text:
                    text += '\n' + str(question)
        if any(path.get("confirmation_state") in {"proposed", "unresolved"} for path in paths):
            text += "\n它与前述想法的关系还未确认。"
        clipped = fit_units([text], max(0, remaining))
        if not clipped:
            break
        parts.append(clipped)
        spans.append({"memory_id": row["memory_id"], "revision": row["revision"], "historical": row["historical"],
                      "label": label, "text": clipped, "partial": clipped != text})
        remaining -= count_tokens_approx(clipped)
        if remaining <= 0:
            break
    content = "\n\n".join(parts)
    text = (service._resolve_gateway_fixed_prompt("ombre.memory_cognition_chain_prompt", runtime_values={"content": content})
            if content else "")
    text = service._trim_text(text, budget) if text else ""
    emitted = [row for row in spans if row["text"] in text]
    request["cognition_chain"] = {"depth": depth, "limit": limit, "budget": budget,
        "eligible_versions": len(allowed), "rendered_versions": emitted,
        "truncated": bool(graph["truncated"] or len(emitted) < len(allowed) or any(s["partial"] for s in emitted)),
        "issues": [*graph["issues"], *({"reason": reason} for reason in sub.get("incomplete_reasons") or [])],
        "provider_requests": 0}
    return text


async def project_sections(service, request, query, buckets, sections):
    started = time.perf_counter()
    try:
        rendered = await _project_sections(service, request, query, buckets, sections)
        request["projection_status"] = {
            "state": "projected" if any(rendered.values()) else "empty",
            "projected_count": len(request.get("injected_usage_bucket_ids") or []),
            "reason": request.pop("_projection_skip_reason", None),
        }
        return rendered
    finally:
        request.setdefault("recall_io", {})["projection_ms"] = max(0, int((time.perf_counter() - started) * 1000))
        request["recall_io"]["remaining_ms"] = max(0, int(remaining(service, request) * 1000))


async def _project_sections(service, request, query, buckets, sections):
    """Apply the same use/source gates to direct, detail and diffused sections."""
    request.update(presentation_trace=[], rendered_memory_spans=[], injected_usage_bucket_ids=[])
    ids_by_section = {key: list(dict.fromkeys(service._extract_bucket_ids_from_context(body) or refs))
                     for key, (body, refs, _budget) in sections.items() if body}
    active = {key for key, (body, _refs, budget) in sections.items() if body and budget > 0}
    reason = ("composer_live_recall_disabled" if request.get("composer_recall_enabled") is False
              else "section_budget_zero" if not active
              else "usage_card_prompt_unavailable" if not service._resolve_gateway_fixed_prompt(
                  "ombre.memory_usage_card_prompt", runtime_values={"header": "", "statements": "", "narrative": "memory"})
              else None)
    if reason:
        request["_projection_skip_reason"] = reason
        request["presentation_trace"] = [{"bucket_id": bid, "source": "none", "render_reason": reason,
            "presentation_revision": None, "fallback_reason": reason}
            for bid in dict.fromkeys(bid for ids in ids_by_section.values() for bid in ids)]
        request.update(rendered_memory_spans=[], injected_usage_bucket_ids=[])
        return {key: "" for key in sections}
    ids_by_section = {key: ids for key, ids in ids_by_section.items() if key in active}
    requested = set(value for ids in ids_by_section.values() for value in ids)
    if not requested:
        return {key: body for key, (body, _refs, _budget) in sections.items()}
    view = getattr(service, "memory_authority_view", None)
    metadata_reader = getattr(view, "memory_index_metadata_map", None)
    metadata = metadata_reader(list(requested)) if callable(metadata_reader) else {}
    valid = []
    for bucket in buckets:
        bid = str(bucket.get("id", ""))
        if bid not in requested:
            continue
        row = metadata.get(bid)
        if getattr(view, "enabled", False) and (not row or hashlib.sha256(str(bucket.get("content") or "").encode()).hexdigest() != row.get("body_sha256")):
            continue
        valid.append(bucket)
    if "_semantic_records" not in request:
        prepare(service, request, query, valid)
    else:
        reader = getattr(view, "memory_l1_recall_map", None)
        if callable(reader):
            request["_semantic_records"].update(reader([b["id"] for b in valid]))
    candidates = [{"bucket": bucket, "score": 0.0} for bucket in valid]
    final_check = request.setdefault("recall_io", {}).setdefault(
        "final_source_recheck", {"passes": 0, "calls": 0, "duration_ms": 0})
    final_check["passes"] += 1
    source_calls_before = request["recall_io"].get("sources", {}).get("calls", 0)
    final_check_started = time.perf_counter()
    try:
        permitted = await verify_sources(service, request, query, candidates, refresh=True)
    finally:
        final_check["duration_ms"] += max(0, int((time.perf_counter() - final_check_started) * 1000))
        final_check["calls"] += max(0, request["recall_io"].get("sources", {}).get("calls", 0) - source_calls_before)
    allowed = {str(item["bucket"]["id"]) for item in permitted}
    uses = request.get("usage_by_bucket", {})
    for bid in requested - allowed:
        uses[bid] = {"state": "excluded", "items": [], "narrative_allowed": False}
    bucket_map = {str(bucket["id"]): bucket for bucket in valid}
    render_ids_by_section, planned_section, seen = {}, {}, set()
    for key in ("targeted", "direct", "diffused", "recent"):
        body, _refs, budget = sections.get(key, ("", [], 0))
        ids = [bid for bid in ids_by_section.get(key, []) if bid in allowed and bid not in seen]
        seen.update(ids)
        render_ids_by_section[key] = ids
        if body and budget > 0:
            for bid in ids:
                planned_section[bid] = key

    records = request.get("_semantic_records", {})
    trace_by_bucket = {}
    l2_candidates = []
    for bid, key in planned_section.items():
        usage = uses[bid]
        record = records.get(bid) or {}
        if key == "targeted":
            trace_by_bucket[bid] = {
                "bucket_id": bid, "memory_id": record.get("memory_id"),
                "l1_revision": record.get("revision"), "presentation_revision": None,
                "source": "l1_targeted_original" if usage["state"] == "l1" else "legacy",
                "resolver_state": None, "fallback_reason": None,
                "reason": "explicit_targeted_original",
            }
        elif usage["state"] == "legacy":
            trace_by_bucket[bid] = {
                "bucket_id": bid, "memory_id": None, "l1_revision": None,
                "presentation_revision": None, "source": "legacy",
                "resolver_state": None, "fallback_reason": "no_l1_authority",
            }
        elif (record.get("state") == "l1" and record.get("memory_id")
              and isinstance(record.get("revision"), int)):
            l2_candidates.append({
                "bucket_id": bid, "memory_id": str(record["memory_id"]),
                "revision": int(record["revision"]), "usage": usage,
            })
        else:
            trace_by_bucket[bid] = {
                "bucket_id": bid, "memory_id": record.get("memory_id"),
                "l1_revision": record.get("revision"), "presentation_revision": None,
                "source": "l1", "resolver_state": None,
                "fallback_reason": "l1_revision_unavailable",
            }

    presentations, resolved_trace = await resolve_presentations(service, l2_candidates, request=request)
    trace_by_bucket.update(resolved_trace)
    # Authority/permission may change while either batch is in flight. This
    # local metadata read is not another L0/provider round trip.
    if getattr(view, "enabled", False) and callable(metadata_reader):
        latest = metadata_reader(list(planned_section))
        for bid in list(allowed):
            if latest.get(bid) != metadata.get(bid):
                allowed.discard(bid)
                uses[bid] = {"state": "excluded", "items": [], "narrative_allowed": False}
        render_ids_by_section = {key: [bid for bid in ids if bid in allowed]
                                 for key, ids in render_ids_by_section.items()}
    candidate_by_bucket = {str(item["bucket"]["id"]): item for item in candidates}
    for bid in requested - allowed:
        item = candidate_by_bucket.get(bid)
        reason = (item or {}).get("admission_reason") or "l1_source_or_body_not_eligible"
        record = records.get(bid) or {}
        trace_by_bucket[bid] = {
            "bucket_id": bid, "memory_id": record.get("memory_id"),
            "l1_revision": record.get("revision"), "presentation_revision": None,
            "source": "none", "resolver_state": None, "fallback_reason": reason,
        }

    rendered = {}
    spans = []
    # Explicit original detail wins over a brief of the same record.
    for key in ("targeted", "direct", "diffused", "recent"):
        body, _refs, budget = sections.get(key, ("", [], 0))
        ids = render_ids_by_section.get(key, [])
        if not body or not ids or budget <= 0:
            rendered[key] = ""
            continue
        if ids:
            parts, remaining = [], budget
            for bid in ids:
                bucket = bucket_map[bid]
                header = ""
                row = trace_by_bucket.setdefault(bid, {"bucket_id": bid, "source": "none"})
                row["section"] = key
                original = str(bucket.get('content') or '')
                comments_reader = getattr(service, '_bucket_text_with_comments', None)
                if uses[bid]['state'] == 'legacy' and callable(comments_reader):
                    original = comments_reader(bucket)
                presentation = presentations.get(bid)
                if presentation:
                    text, render_reason, selected_unit_ids, partial_presentation = render_presentation(
                        service, uses[bid], header, presentation, remaining,
                    )
                    if text:
                        row["source"] = "l2"
                        row["presentation_revision"] = presentation["presentation_revision"]
                        row["selected_unit_ids"] = selected_unit_ids
                        row["partial_presentation"] = partial_presentation
                        row["fallback_reason"] = None
                        coverage = presentation.get("coverage") or {}
                        selected_set = set(selected_unit_ids)
                        actual_refs = (coverage.get("used_statement_ids") if not partial_presentation else
                            list(dict.fromkeys(ref for unit in presentation.get("units") or []
                                if unit.get("unit_id") in selected_set for ref in unit.get("statement_refs") or [])))
                        row["injected_statement_ids"] = actual_refs
                        if actual_refs is not None:
                            row["uncovered_matched_statement_ids"] = sorted(set(
                                (uses[bid].get("statement_selection") or {}).get("direct_ids") or []) - set(actual_refs))
                    else:
                        row["fallback_reason"] = render_reason
                        if render_reason in {
                            "presentation_exceeds_remaining_budget",
                            "no_complete_unit_fits_remaining_budget",
                            "no_complete_correction_unit_fits_remaining_budget",
                            "required_l2_context_missing",
                        }:
                            text = render(
                                service, uses[bid], header,
                                original, remaining, trace=row,
                            )
                            if text:
                                row["source"] = "l1"
                            else:
                                row["source"] = "none"
                                row.setdefault("render_reason", "usage_card_prompt_unavailable")
                        else:
                            row["source"] = "none"
                            row["render_reason"] = render_reason
                elif row.get("fallback_reason") in {"source_dependency_changed", "memory_not_found",
                        "memory_not_eligible", "l1_revision_mismatch", "l1_dependency_changed",
                        "l1_parent_dependency_changed", "scene_dependency_mismatch"}:
                    # Brain has fresher source evidence than the request-local
                    # selection cache. Never revive the rejected source via L1.
                    text = ""
                    row["source"] = "none"
                    row["render_reason"] = row["fallback_reason"]
                else:
                    text = render(service, {**uses[bid], "scope_to_query": key != "targeted"}, header,
                                  original, remaining, trace=row)
                    if text:
                        if row.get("source") not in {"l1_targeted_original", "legacy"}:
                            row["source"] = "legacy" if uses[bid]["state"] == "legacy" else "l1"
                    else:
                        row["source"] = "none"
                        row.setdefault("render_reason", "usage_card_prompt_unavailable")
                if text:
                    parts.append(text)
                    spans.append({"bucket_id": bid, "section": key, "text": text})
                    # The existing Gateway owns token accounting and budgets.
                    from utils import count_tokens_approx
                    remaining -= count_tokens_approx(text)
                elif remaining > 0:
                    row['source'] = 'none'
                    row.setdefault("fallback_reason", "not_rendered")
                if remaining <= 0:
                    break
            if remaining <= 0:
                included_ids = {span["bucket_id"] for span in spans if span["section"] == key}
                for bid in ids:
                    if bid not in included_ids:
                        row = trace_by_bucket.setdefault(bid, {"bucket_id": bid, "source": "none"})
                        row["source"] = "none"
                        row["fallback_reason"] = "section_budget_exhausted"
            rendered[key] = "\n".join(parts)
    presentation_trace = [
        trace_by_bucket[bid]
        for bid in list(planned_section) + sorted(requested - allowed)
        if bid in trace_by_bucket
    ]
    for row in presentation_trace:
        row.setdefault("selected_unit_ids", [])
        row.setdefault("partial_presentation", False)
        if row.get("source") in {"l1", "l1_targeted_original"}:
            selection = (uses.get(row["bucket_id"]) or {}).get("statement_selection") or {}
            row.setdefault("statement_selection", selection)
    request["presentation_trace"] = presentation_trace
    # A dependency supplements an actually shown memory; it cannot reserve
    # tokens first and clip the selected memory's decisive claim away.
    from utils import count_tokens_approx
    target = next((key for key in ("targeted", "direct", "diffused", "recent") if rendered.get(key)), None)
    rendered_ids = {span["bucket_id"] for span in spans}
    chain_seeds = [item for item in permitted if str(item["bucket"]["id"]) in rendered_ids
                   and any(use.get("cognition") for use in uses.get(str(item["bucket"]["id"]), {}).get("items", []))]
    remaining = max(0, sections[target][2] - count_tokens_approx(rendered[target] + "\n\n")) if target else 0
    chain_budget = min(option(service, "cognition_chain_budget", 300), remaining)
    chain = await cognition_context(service, request, query, chain_seeds, chain_budget)
    if chain and target:
        rendered[target] = service._trim_text(rendered[target] + "\n\n" + chain, sections[target][2])
    elif chain_seeds and remaining <= 0:
        request["cognition_chain"] = {"rendered_versions": [], "budget": 0, "provider_requests": 0,
                                      "issues": [{"reason": "no_budget_after_main_memory"}], "truncated": True}
    request["rendered_memory_spans"] = spans
    request["requested_usage_bucket_ids"] = list(dict.fromkeys(
        bid for ids in ids_by_section.values() for bid in ids))
    request["injected_usage_bucket_ids"] = list(dict.fromkeys(
        span["bucket_id"] for span in spans if span["bucket_id"] in allowed))
    return rendered
