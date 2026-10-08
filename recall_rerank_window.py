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


def finite_score(value):
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def select_natural_window(items, limit, priorities):
    """Return pool indices and request-local evidence, without modifying items.

The four special-evidence flags retain the old priority. Within a tier, reciprocal
rank fusion lets a top lexical/moment candidate compete with a top semantic one.
Types, narrator and text length are deliberately not eligibility conditions.
"""
    if len(items) != len(priorities):
        raise ValueError("window_priority_count_mismatch")
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
            notes[index]["fusion_score"] += 1.0 / (60 + rank)

    def rank_key(index):
        ranks = notes[index]["channel_ranks"]
        return (tuple(priorities[index][:4]), -notes[index]["fusion_score"],
                min(ranks.values(), default=10**9),
                # Prefer a current-query lexical hit only for exact rank ties.
                "keyword" not in ranks, tuple(priorities[index][4:]), identity(index))

    ordered = sorted(range(len(items)), key=rank_key)
    selected = set(ordered[:limit])
    for rank, index in enumerate(ordered, 1):
        notes[index].update(rank=rank, selected=index in selected,
                            status="pending" if index in selected else "outside_window")
        notes[index]["fusion_score"] = round(notes[index]["fusion_score"], 8)
    selected_counts = Counter(name for index in selected for name in notes[index]["channel_ranks"])
    summary = {
        "policy": POLICY, "candidate_count": len(items), "limit": limit,
        "selected_count": len(selected),
        "channel_candidates": {name: len(rows) for name, rows in channels.items()},
        "channel_selected": {name: selected_counts[name] for name in channels},
    }
    return selected, notes, summary


def admission_evidence(items, accepted, selected, *, semantic_threshold, rerank_threshold):
    """Bounded, owner-safe score/window evidence, including not-scored candidates."""
    def bucket_id(item):
        return str((item.get("bucket") or {}).get("id") or "")

    admitted_ids = {bucket_id(item) for item in accepted}
    selected_ids = {bucket_id(item) for item in selected}
    rows = []
    for item in items[:EVIDENCE_LIMIT]:
        identifier = bucket_id(item)
        window = item.get("_rerank_window") or {}
        reason = str(item.get("admission_reason") or "unclassified")
        status = str(window.get("status") or "not_requested")
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
            "semantic_verified": (item.get("natural_semantic") or {}).get("index_status") == "verified",
            "admission_reason": reason if re.fullmatch(r"[a-z0-9_]{1,80}", reason) else "other",
            "admitted": identifier in admitted_ids, "selected": identifier in selected_ids,
        })
    return {"candidate_count": len(items), "recorded_count": len(rows),
            "truncated_count": max(0, len(items) - len(rows)),
            "semantic_threshold": finite_score(semantic_threshold),
            "rerank_threshold": finite_score(rerank_threshold), "candidates": rows}
