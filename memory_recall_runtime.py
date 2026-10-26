"""The existing Gateway's L1/legacy projection and source-check helpers."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import os

import httpx

from memory_recall_semantics import (apply_relations, known, lexical_features, resolve_document,
                                     score_document, temporal_features)
from memory_system_config import REQUEST_CONFIGURATION, values_from_config
from memory_inner_affinity import (build_inner_context, document_inner_affinity, local_inner_candidates)


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
        request["_use_context"] = {
            "now": now, "target_time": target, "query": query,
            "purpose": "history" if target is not None or getattr(plan, "explicit_old_memory", False) else "current",
            "overview": bool(date_hint and getattr(plan, "auto_too_vague", False)),
            "facts": {"hour": {"state": "known", "value": now.hour},
                      "weekday": {"state": "known", "value": now.weekday()}},
        }
    if "inner_state_affinity" not in request["_use_context"]:
        request["_use_context"]["inner_state_affinity"] = build_inner_context(
            values(service), request.get("_signal_snapshots", []),
            request.get("_signal_sampled_at") or datetime.now(timezone.utc), query,
            scope={key: request[key] for key in ("conversation_id", "source_event_id", "profile_id")
                   if request.get(key)})
    return request["_use_context"]


def prepare(service, request, query, buckets):
    settings = values(service)
    mode = settings.get("recall", {}).get("mode", "inherit")
    if mode == "fast_local":
        request.update(local_only=True, allow_semantic=False, allow_rerank=False)
    elif mode == "fast_hybrid" and not request.get("local_only"):
        request["allow_semantic"] = True
    context(service, request, query)
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
    capacity = min(settings.get("limits", {}).get("semantic_candidate_top_k", 50),
                   settings.get("limits", {}).get("rerank_candidate_limit", 20))
    additions = local_inner_candidates(request.get("_semantic_records", {}),
        request.get("_semantic_buckets", {}), context(service, request, query)["inner_state_affinity"],
        settings, {str(item["bucket"]["id"]) for item in pool}, limit=max(0, capacity - len(pool)))
    pool.extend(additions)
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
        semantic = item.get("semantic_score")
        if request.get("local_only") or request.get("allow_semantic") is False:
            semantic = None
        calculation = score_document(document, settings, use_context, {
            "relevance": base, "keyword_score": item.get("keyword_score"),
            "semantic_score": semantic, "rerank_score": item.get("rerank_score"),
            "importance": metadata.get("importance", item["bucket"].get("importance")),
            "confidence": metadata.get("confidence", item["bucket"].get("confidence")),
        })
        policy_formula = calculation.get("formula_id") or "final"
        formula = next((f for f in settings["formulas"] if f["id"] == policy_formula), {})
        custom_formula = (formula.get("expression", "relevance").strip() != "relevance"
                          or bool(settings.get("recall", {}).get("formula_id"))
                          or any(row.get("scoring_profile") for row in calculation["statements"]))
        time_applied = any(row["features"].get("freshness") is not None
                           and row["features"].get("time_weight") is not None
                           for row in calculation["statements"])
        calculation["window_score"] = (calculation["score"] if not calculation["reason_codes"]
            and (custom_formula or time_applied or item.get("l1_local_match")) else None)
        calculation["inner_state_affinity"] = document_inner_affinity(document,
            use_context["inner_state_affinity"], usage=request.get("usage_by_bucket", {}).get(bid),
            selection=item.get("_statement_selection") or calculation.get("statement_selection"),
            calculation=calculation, settings=settings)
        # Both scoring passes retain the original relevance/formula result.
        # This component is adopted once, after ordinary admission/hierarchy.
        item["_configured_score"] = calculation
        if calculation["score"] is not None:
            item["score"] = calculation["score"]
            if after_rerank:
                item["combined_score"] = item["score"]
    return pool


async def verify_sources(service, request, query, items):
    """No provider model calls; exact current L0 readers remain on Brain."""
    records = request.get("_semantic_records", {})
    wanted, bindings_by_bucket = [], {}
    for item in items:
        bid = str(item["bucket"]["id"])
        record = records.get(bid) or {}
        document = record.get("document")
        if record.get("state") == "l1" and document:
            bindings = [binding["binding_id"] for binding in document.get("evidence_bindings", [])]
            bindings_by_bucket[bid] = bindings
            if bindings:
                wanted.append({"memory_id": record["memory_id"], "revision": record["revision"],
                               "binding_ids": bindings,
                               **({"allow_historical": True} if record.get("historical") else {})})
    frozen = REQUEST_CONFIGURATION.get()
    validation = (frozen.setdefault("source_validation", {})
                  if frozen and frozen.get("service") == id(service) else request)
    checked = validation.setdefault("_checked_sources", {})
    wanted = [item for item in wanted if (item["memory_id"], item["revision"]) not in checked]
    if wanted and not validation.get("_source_check_cancelled"):
        base = os.environ.get("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://ombre-brain:8000").rstrip("/")
        token = os.environ.get("OMBRE_MEMORY_WRITE_TOKEN") or getattr(service, "gateway_token", "")
        try:
            async with httpx.AsyncClient(timeout=20.0, trust_env=False) as client:
                response = await client.post(base + "/api/memory-authority/recall-sources",
                    headers={"Authorization": "Bearer " + token}, json={"items": wanted})
                response.raise_for_status()
                response_items = response.json()["items"]
            for row in response_items:
                checked[(row["memory_id"], row["revision"])] = row
        except asyncio.CancelledError:
            validation["_source_check_cancelled"] = True
            raise
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
        check = checked.get((record["memory_id"], record["revision"]))
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
                               "source_states": source_states}
        result.append(item)
    apply_relations(records, usage_by_bucket, use_context)
    return result



def apply_inner_ranking(service, request, query, pool):
    """Adopt the independent component once after source/admission decisions."""
    changed = False
    for item in pool:
        calculation = item.get("_configured_score") or {}
        component = calculation.get("inner_state_affinity") or {}
        base = calculation.get("score", item.get("_inner_ranking_base", item.get("score")))
        item["_inner_ranking_base"] = base
        bonus = component.get("bonus", 0)
        if type(base) not in (int, float) or type(bonus) not in (int, float):
            continue
        item["_admission_score"] = base
        # Card confidence thresholds also retain their original baseline.
        reliable = getattr(service, "_dynamic_bucket_item_has_reliable_recall_signal", None)
        baseline_eligible = (base >= getattr(service, "first_card_min_score", 0)
            or callable(reliable) and reliable(query, item))
        if not baseline_eligible:
            bonus = 0
            component["ranking_reason"] = "original_card_confidence_not_met"
        item["score"] = base + bonus
        component.update(applied_bonus=bonus, base_score=base, final_score=item["score"],
                         applied_at="after_admission")
        if "combined_score" in item:
            item["combined_score"] = item["score"]
        if bonus > 0:
            # Preserve score/window_score as the formula evidence, separately
            # expose final ranking adoption without overwriting owner formulas.
            calculation["ranking_score"] = item["score"]
            changed = True
    if changed:
        pool.sort(key=lambda item: item.get("score", 0), reverse=True)
    return pool

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


def render(service, usage, header, original, budget):
    """A source-faithful interim L1 view, not the future persisted L2."""
    if usage["state"] == "excluded":
        return ""
    # Factory/body/owner_override/off are resolved through the normal Composer
    # component registry. Turning it off must not resurrect the old raw body.
    text = service._resolve_gateway_fixed_prompt("ombre.memory_usage_card_prompt", runtime_values={
        "header": header, "statements": usage_lines(usage),
        "narrative": original if usage.get("narrative_allowed") else "",
    })
    return service._trim_text(text, budget) if text else ""


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
           "_use_context": context(service, request, query), "_chain_projection": True}
    candidates = [{"bucket": {"id": str(index)}, "score": 0.0} for index in range(len(rows))]
    allowed = await verify_sources(service, sub, query, candidates)
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
        label = f'Cognition {len(spans) + 1} ({"earlier saved version" if row["historical"] else "related recorded view/evidence"})'
        text = label + "\n" + usage_lines({**usage, "items": uses})
        if paths:
            text += "\nRecorded connection: " + ", ".join(dict.fromkeys(path["kind"] for path in paths))
        if any(path.get("confirmation_state") in {"proposed", "unresolved"} for path in paths):
            text += "\nRelationship remains unconfirmed."
        clipped = service._trim_text(text, max(0, remaining))
        if not clipped:
            break
        parts.append(clipped)
        spans.append({"memory_id": row["memory_id"], "revision": row["revision"], "historical": row["historical"],
                      "label": label, "partial": clipped != text})
        remaining -= count_tokens_approx(clipped)
        if remaining <= 0:
            break
    content = "\n\n".join(parts)
    text = (service._resolve_gateway_fixed_prompt("ombre.memory_cognition_chain_prompt", runtime_values={"content": content})
            if content else "")
    text = service._trim_text(text, budget) if text else ""
    emitted = [row for row in spans if row["label"] in text]
    request["cognition_chain"] = {"depth": depth, "limit": limit, "budget": budget,
        "eligible_versions": len(allowed), "rendered_versions": emitted,
        "truncated": bool(graph["truncated"] or len(emitted) < len(allowed) or any(s["partial"] for s in emitted)),
        "issues": graph["issues"], "provider_requests": 0}
    return text


async def project_sections(service, request, query, buckets, sections):
    """Apply the same use/source gates to direct, detail and diffused sections."""
    ids_by_section = {key: list(dict.fromkeys(service._extract_bucket_ids_from_context(body) or refs))
                     for key, (body, refs, _budget) in sections.items() if body}
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
    permitted = await verify_sources(service, request, query, candidates)
    allowed = {str(item["bucket"]["id"]) for item in permitted}
    uses = request.get("usage_by_bucket", {})
    for bid in requested - allowed:
        uses[bid] = {"state": "excluded", "items": [], "narrative_allowed": False}
    bucket_map = {str(bucket["id"]): bucket for bucket in valid}
    rendered, seen = {}, set()
    # Explicit original detail wins over a brief of the same record.
    for key in ("targeted", "direct", "diffused"):
        body, _refs, budget = sections.get(key, ("", [], 0))
        ids = [bid for bid in ids_by_section.get(key, []) if bid in allowed and bid not in seen]
        seen.update(ids)
        if not body or not ids or budget <= 0:
            rendered[key] = ""
            continue
        if len(ids) == len(ids_by_section[key]) and all(uses[bid]["narrative_allowed"] for bid in ids):
            notes = "\n".join(f'[bucket_id:{bid}] {usage_lines(uses[bid])}' for bid in ids)
            text = service._resolve_gateway_fixed_prompt("ombre.memory_usage_card_prompt", runtime_values={
                "header": "", "statements": notes, "narrative": body})
            rendered[key] = service._trim_text(text, budget) if text else ""
        else:
            parts, remaining = [], budget
            for bid in ids:
                bucket = bucket_map[bid]
                header = service._direct_bucket_header(bucket, {"bucket_id": bid})
                text = render(service, uses[bid], header, str(bucket.get("content") or ""), remaining)
                if text:
                    parts.append(text)
                    # The existing Gateway owns token accounting and budgets.
                    from utils import count_tokens_approx
                    remaining -= count_tokens_approx(text)
                if remaining <= 0:
                    break
            rendered[key] = "\n".join(parts)
    # A dependency supplements an actually shown memory; it cannot reserve
    # tokens first and clip the selected memory's decisive claim away.
    from utils import count_tokens_approx
    target = next((key for key in ("targeted", "direct", "diffused") if rendered.get(key)), None)
    rendered_ids = {bid for body in rendered.values() for bid in service._extract_bucket_ids_from_context(body)}
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
    request["injected_usage_bucket_ids"] = list(dict.fromkeys(
        bid for body in rendered.values() for bid in service._extract_bucket_ids_from_context(body)
        if bid in allowed))
    return rendered
