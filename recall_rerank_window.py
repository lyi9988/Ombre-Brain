"""Bounded natural-Recall window selection; not an admission or source authority.

Channel scores have different scales. Fuse their ranks, preserving the caller's
existing special-evidence priority, without changing scores used by admission.
No source body, title, query or untrusted identifier enters diagnostic output.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


POLICY = "natural_channel_rrf_v1"
EVIDENCE_LIMIT = 128


def explicit_query_parts(query):
    """Split independent, explicitly punctuated questions, never arbitrary prose.

    A short follow-up or shared conditional context stays on the original path.
    The original query is retained by the caller for scope/source decisions.
    """
    text = str(query or "").strip()
    if len(re.findall(r"[?？]", text)) < 2:
        return [text]
    parts = [part.strip() for part in re.split(r"(?<=[?？])\s*", text) if part.strip()]
    parts = [re.sub(r"^(?:另外|还有|再问(?:一下)?|以及|顺便问(?:一下)?|also\b)[，,：:\s]*", "", part,
                    flags=re.IGNORECASE).strip() for part in parts]
    # Pronoun-only follow-ups and conditions depend on the surrounding question.
    if (any(not part.endswith(("?", "？")) for part in parts)
            or any(re.search(r"如果|假如|除非|只要|\b(?:if|unless)\b", part, re.I) for part in parts)
            or any(re.match(r"^(?:他|她|它|这个|那个|这样|那时候)", part) for part in parts)
            or any(len(re.sub(r"[?？\W]", "", part)) < 4 for part in parts)):
        return [text]
    return list(dict.fromkeys(parts))


def finite_score(value):
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def select_natural_window(items, limit, priorities, *, configured_scores=None,
                          configured_weight=1.0, rrf_constant=60.0,
                          query_view_scores=None):
    """Return pool indices and request-local evidence, without modifying items.

The four special-evidence flags retain the old priority. Within a tier, reciprocal
rank fusion lets a top lexical/moment candidate compete with a top semantic one.
Types, narrator and text length are deliberately not eligibility conditions.
"""
    if len(items) != len(priorities):
        raise ValueError("window_priority_count_mismatch")
    if configured_scores is not None and len(configured_scores) != len(items):
        raise ValueError("window_feature_count_mismatch")
    limit = max(0, min(int(limit), len(items)))
    channels = {name: [] for name in ("keyword", "semantic", "moment", "structural")}
    for index, item in enumerate(items):
        scores = {
            "keyword": finite_score(item.get("keyword_score")),
            "semantic": finite_score(item.get("semantic_score")),
            "moment": finite_score(item.get("score")) if item.get("moment_source_match") else None,
            "structural": max(finite_score(item.get(key)) or 0.0
                              for key in ("word_map_score", "entity_edge_score")),
        }
        for name, score in scores.items():
            if score is not None and score > 0:
                channels[name].append((index, score))
    # Pool order can originate in a set union; stable source identity resolves
    # equal ranks without changing provider result-index mapping.
    def identity(index):
        return str((items[index].get("bucket") or {}).get("id") or ""), index

    notes = [{"channel_ranks": {}, "fusion_score": 0.0} for _ in items]
    for name, candidates in channels.items():
        candidates.sort(key=lambda pair: (-pair[1], identity(pair[0])))
        for rank, (index, _score) in enumerate(candidates, 1):
            notes[index]["channel_ranks"][name] = rank
            notes[index]["fusion_score"] += 1.0 / (rrf_constant + rank)
    configured = [(i, finite_score(score)) for i, score in enumerate(configured_scores or [])]
    configured = [(i, score) for i, score in configured if score is not None]
    if configured and configured_weight:
        configured.sort(key=lambda pair: (-pair[1], identity(pair[0])))
        previous, rank = None, 0
        for position, (index, score) in enumerate(configured, 1):
            if previous is None or score != previous:
                rank = position
            previous = score
            notes[index]["channel_ranks"]["configured"] = rank
            notes[index]["fusion_score"] += configured_weight / (rrf_constant + rank)

    def rank_key(index):
        ranks = notes[index]["channel_ranks"]
        return (tuple(priorities[index][:4]), -notes[index]["fusion_score"],
                min(ranks.values(), default=10**9),
                # Prefer a current-query lexical hit only for exact rank ties.
                "keyword" not in ranks, tuple(priorities[index][4:]), identity(index))

    ordered = sorted(range(len(items)), key=rank_key)
    selected, part_for_index = set(), {}
    if query_view_scores:
        if len(query_view_scores) != len(items):
            raise ValueError("window_query_view_count_mismatch")
        parts = sorted({key for scores in query_view_scores for key in scores})
        queues = {part: sorted(
            (i for i in ordered if (finite_score(query_view_scores[i].get(part)) or 0) > 0),
            key=lambda i: (tuple(priorities[i][:4]),
                           -query_view_scores[i][part], rank_key(i))) for part in parts}
        # Each explicit question gets an opportunity within the existing total
        # window. This supplies evidence to the reranker, never admission.
        while len(selected) < limit:
            added = False
            for part in parts:
                index = next((i for i in queues[part] if i not in selected), None)
                if index is not None:
                    selected.add(index)
                    part_for_index[index] = part
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
    for index in ordered:
        if len(selected) >= limit:
            break
        selected.add(index)
    for rank, index in enumerate(ordered, 1):
        notes[index].update(rank=rank, selected=index in selected,
                            status="pending" if index in selected else "outside_window")
        if index in part_for_index:
            notes[index]["query_part"] = part_for_index[index]
        notes[index]["fusion_score"] = round(notes[index]["fusion_score"], 8)
        # Actual ordering keys, not an inferred explanation of the final decision.
        notes[index]["priority_tier"] = [value if type(value) is bool else None
                                         for value in priorities[index][:4]]
    selected_counts = Counter(name for index in selected for name in notes[index]["channel_ranks"])
    summary = {
        "policy": POLICY, "candidate_count": len(items), "limit": limit,
        "selected_count": len(selected),
        "channel_candidates": {name: len(rows) for name, rows in channels.items()},
        "channel_selected": {name: selected_counts[name] for name in channels},
        "configured_candidates": len(configured), "configured_weight": configured_weight,
        "rrf_constant": rrf_constant,
    }
    if query_view_scores:
        summary["query_part_selected"] = dict(Counter(part_for_index.values()))
    return selected, notes, summary


def admission_evidence(items, accepted, selected, *, semantic_threshold, rerank_threshold,
                       selection_limits=None, query_purpose=None, hierarchy_suppressed=(),
                       selection_diagnostics=None):
    """Bounded, owner-safe score/window evidence, including not-scored candidates."""
    def bucket_id(item):
        return str((item.get("bucket") or {}).get("id") or "")

    admitted_ids = {bucket_id(item) for item in accepted}
    selected_ids = {bucket_id(item) for item in selected}
    admitted_ranks = {bucket_id(item): index + 1 for index, item in enumerate(accepted)}
    selection_ranks = {bucket_id(item): index + 1 for index, item in enumerate(selected)}
    hierarchy_ids = set(hierarchy_suppressed)
    selection_diagnostics = selection_diagnostics or {}
    decisions = selection_diagnostics.get("decisions") or {}
    reasons = {"card_limit_disabled", "card_limit_reached", "axis_diversity_selected",
        "axis_diversity_not_selected", "first_card_selected", "below_first_card_threshold",
        "category_overview_selected", "category_overview_path_returned", "new_term_coverage_selected",
        "second_card_selected", "below_second_card_threshold", "below_relative_score",
        "multiquestion_part_selected", "ordinary_fill_selected", "multiquestion_part_lower_rank",
        "multiquestion_card_limit", "multiquestion_not_eligible_or_no_slot"}
    rows = []
    for item in items[:EVIDENCE_LIMIT]:
        identifier = bucket_id(item)
        window = item.get("_rerank_window") or {}
        reason = str(item.get("admission_reason") or "unclassified")
        status = str(window.get("status") or "not_requested")
        decision = (decisions.get(identifier) or {}).get("reason")
        rows.append({
            "bucket_ref": hashlib.sha256(identifier.encode()).hexdigest()[:16],
            "channel_ranks": {name: rank for name, rank in window.get("channel_ranks", {}).items()
                              if name in {"keyword", "semantic", "moment", "structural"}
                              and type(rank) is int and rank > 0},
            "window_rank": window.get("rank"),
            "window_selected": bool(window.get("selected")),
            "rerank_status": status if status in {
                "pending", "outside_window", "provider_failed", "no_score", "scored", "not_requested"
            } else "unknown",
            "keyword_score": finite_score(item.get("keyword_score")),
            "semantic_score": finite_score(item.get("semantic_score")),
            "rerank_score": finite_score(item.get("rerank_score")),
            "raw_base_score": finite_score(item.get("_raw_base_score")),
            "raw_fused_score": finite_score(item.get("_raw_fused_score")),
            "final_score": finite_score(item.get("score")),
            "configured_window_score": finite_score((item.get("_configured_score") or {}).get("window_score")),
            "query_part": window.get("query_part") if type(window.get("query_part")) is int else None,
            "window_fusion_score": finite_score(window.get("fusion_score")),
            "window_priority_tier": [v if type(v) is bool else None
                                     for v in (window.get("priority_tier") or [])[:4]],
            "admitted_rank": admitted_ranks.get(identifier),
            "selection_rank": selection_ranks.get(identifier),
            "selection_reason": decision if isinstance(decision, str) and decision in reasons else None,
            # Do not fabricate quota/threshold causes the picker did not record.
            "selection_state": ("selected" if identifier in selected_ids else
                "hierarchy_suppressed" if identifier in hierarchy_ids else
                "admitted_not_selected" if identifier in admitted_ids else "not_admitted"),
            "semantic_verified": (item.get("natural_semantic") or {}).get("index_status") == "verified",
            "admission_reason": reason if re.fullmatch(r"[a-z0-9_]{1,80}", reason) else "other",
            "admitted": identifier in admitted_ids, "selected": identifier in selected_ids,
        })
    return {"candidate_count": len(items), "recorded_count": len(rows),
            "truncated_count": max(0, len(items) - len(rows)),
            "semantic_threshold": finite_score(semantic_threshold),
            "rerank_threshold": finite_score(rerank_threshold),
            "selection_limits": {key: finite_score((selection_limits or {}).get(key)) for key in
                ("max_cards", "first_card_min_score", "second_card_min_score", "second_card_relative_score")},
            "query_purpose": query_purpose if isinstance(query_purpose, str) and query_purpose in
                {"current", "history", "conditional", "comparison"} else "unknown",
            "selection_mode": selection_diagnostics.get("mode") if selection_diagnostics.get("mode") in
                ("score", "axis_diversity", "multiquestion_coverage") else None,
            "coverage_state": selection_diagnostics.get("coverage_state") if selection_diagnostics.get("coverage_state") in
                ("fewer_than_two_eligible_parts", "applied") else None,
            "candidates": rows}
