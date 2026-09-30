"""Regression matrix for natural semantic Recall (RECALL-R1 §9).

This file is deliberately local-only: tests use synthetic Memory/event records
and provider doubles. A separately authorized, metadata-only provider smoke is
kept out of pytest and never writes its request text or credentials to disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from embedding_engine import EmbeddingEngine
import gateway as gateway_module
from gateway import GatewayService
from memory_authority import MemoryAuthorityStore
from memory_authority_view import MemoryAuthorityRecallView
from prompt_source_registry import source_detail
from recall_input import build_recall_input


@dataclass(frozen=True)
class RecallScenario:
    """Behavioral case description; concrete adapters follow the core API."""

    case_id: str
    requirement: str
    input_kind: str


# Keep the contract matrix visible while the Gateway/EmbeddingEngine API is
# being implemented. Each row will become a focused executable regression
# against the production-facing service seam once the core interface lands.
RECALL_NATURAL_R1_CASES = (
    RecallScenario(
        "same_turn_named_question_then_followup",
        "all staged user events reach both retrieval/admission and rerank",
        "multi_message_turn",
    ),
    RecallScenario(
        "same_turn_bare_followup_without_name_or_pronoun",
        "a bare recall follow-up inherits the named topic from the same user batch",
        "multi_message_turn",
    ),
    RecallScenario(
        "ordinary_experience_paraphrase",
        "meaningful experience queries get semantic opportunity without trigger words",
        "ordinary_query",
    ),
    RecallScenario(
        "ordinary_preference_query",
        "preference queries get semantic opportunity without trigger words",
        "ordinary_query",
    ),
    RecallScenario(
        "ordinary_emotional_experience_query",
        "emotional-experience queries are not gated by fixed lookup phrases",
        "ordinary_query",
    ),
    RecallScenario(
        "current_and_context_query_views",
        "q_current and q_context preserve the current batch and bounded source context",
        "multi_message_turn",
    ),
    RecallScenario(
        "default_recent_context_is_four_completed_turns",
        "the default recent scope is four completed turns while retaining every current user event",
        "recent_context",
    ),
    RecallScenario(
        "authority_before_top_k",
        "identity, privacy, enabled policy, active revision and manual_only are filtered before Top-K",
        "candidate_pool",
    ),
    RecallScenario(
        "no_candidate_reembedding",
        "chat retrieval embeds query views only and reuses stored candidate vectors",
        "candidate_pool",
    ),
    RecallScenario(
        "query_cache_singleflight_and_cancel",
        "query cache/singleflight deduplicates work and cancellation reaches provider work",
        "async_provider",
    ),
    RecallScenario(
        "task_mode_and_planner_off_keep_semantic_path",
        "task mode and a disabled optional Planner do not disable ordinary semantic retrieval",
        "routing_policy",
    ),
    RecallScenario(
        "internal_worker_cannot_reenter_owner_recall",
        "trusted internal scope cannot recursively enter owner-facing Recall",
        "request_scope",
    ),
    RecallScenario(
        "planner_and_final_rerank_are_bounded",
        "at most one supplemental Planner pass and one final rerank round run",
        "ambiguous_candidates",
    ),
    RecallScenario(
        "semantic_similarity_does_not_assert_identity",
        "semantic relevance can surface a fact but cannot establish person identity",
        "ambiguous_identity",
    ),
    RecallScenario(
        "same_projection_reused_by_continuation",
        "tool continuation reuses its parent Recall projection and snapshot",
        "continuation",
    ),
)


def _extract_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(part.get("text") or "")
            for part in value
            if isinstance(part, dict)
        )
    return ""


def _clean_user_text(value):
    return " ".join(str(value or "").split())


def _sample_named_question():
    return "\u664f\u664f\u662f\u8c01\uff5e\u563f\u563f\u3002"


def _sample_bare_followup():
    return "\u4e00\u70b9\u4e5f\u8bb0\u4e0d\u4f4f\u4e86\u5417\uff1f"


def _trace_for_messages(messages, *, current_indices, event_ids=None, covered_indices=None):
    event_ids = event_ids or {}
    covered_indices = set(range(len(messages))) if covered_indices is None else set(covered_indices)
    return {
        "conversation_id": "conversation-natural-r1-test",
        "turn_id": "turn-current",
        "coverage": {
            "context_revision": 12,
            "items": [
                {
                    "message_index": index,
                    "turn_member": index in set(current_indices),
                    "event_id": event_ids.get(index, f"event-{index}"),
                }
                for index, _message in enumerate(messages)
                if index in covered_indices
            ],
        },
    }


def _embedding_engine(tmp_path):
    return EmbeddingEngine({
        "buckets_dir": str(tmp_path / "buckets"),
        "embedding": {
            "enabled": True,
            "api_key": "local-test-key",
            "base_url": "https://embedding.example/v1",
            "model": "test-embedding-model",
            "query_cache_ttl_seconds": 300,
        },
    })


def _provider_input_items(input_value):
    return input_value if isinstance(input_value, list) else [input_value]


def _natural_bucket(bucket_id, content, *, revision=1):
    return {"id": bucket_id, "content": content, "revision": revision, "metadata": {}}


def _natural_candidate(bucket, score, *, unit_id="unit-1", index_status="verified",
                       labels=("semantic_only",), rerank_score=None):
    item = {
        "bucket": bucket,
        "score": score,
        "semantic_score": score,
        "keyword_score": 0.0,
        "natural_semantic": {"unit_id": unit_id, "index_status": index_status},
        "_test_labels": list(labels),
    }
    if rerank_score is not None:
        item["_rerank_for_test"] = rerank_score
    return item


def _natural_finish_service(monkeypatch, buckets, candidates_by_query, *, manual_only_ids=(),
                            metadata_overrides=None, rerank_score=None):
    metadata = {}
    for bucket in buckets:
        bucket_id = str(bucket["id"])
        body_sha = hashlib.sha256(str(bucket.get("content") or "").encode("utf-8")).hexdigest()
        metadata[bucket_id] = {
            "revision": int(bucket.get("revision") or 1),
            "body_sha256": body_sha,
            "memory_id": f"memory:{bucket_id}",
        }
    metadata.update(metadata_overrides or {})

    service = GatewayService.__new__(GatewayService)
    calls = {
        "candidate": [], "candidate_options": [],
        "rerank": [], "selected": [], "eligible": [],
    }
    service.memory_authority_view = SimpleNamespace(
        memory_index_metadata_map=lambda _ids: metadata,
    )
    service.recall_policy = SimpleNamespace(semantic_threshold=0.8, rerank_threshold=0.8)
    service.semantic_candidate_top_k = 24
    service.query_planner_enabled = False
    service.query_planner_max_queries = 2
    service.relevance_options = {}
    service._query_planner_debug_base = lambda query: {"query": query, "triggered": False}
    service._authority_auto_recall_allowed = lambda bucket: str(bucket.get("id") or "") not in manual_only_ids
    service._is_self_anchor_recall_excluded_bucket = lambda bucket: bool(bucket.get("self_anchor"))
    service._safe_float = lambda value, default=0.0: default if value is None else float(value)
    service._clamp = lambda value: max(0.0, min(1.0, float(value or 0.0)))
    service._hard_bucket_evidence_labels = lambda labels: [
        label for label in labels if label not in {"semantic_only", "strong_semantic", "strong_rerank"}
    ]
    service._bucket_evidence_labels = lambda _query, item: list(item.get("_test_labels") or [])
    service._record_candidate_stage = lambda *_args, **_kwargs: None
    service._rejected_bucket_evidence = lambda items: {"count": len(items)}
    service._pick_dynamic_cards = lambda items, **_kwargs: list(items)

    async def candidate_builder(query, _session_id, _eligible_buckets, **_kwargs):
        calls["candidate"].append(query)
        natural_input = _kwargs.get("natural_input") or {}
        metadata_value = natural_input.get("metadata") or {}
        calls["candidate_options"].append({
            "query": query,
            "allow_semantic": _kwargs.get("allow_semantic"),
            "q_current": natural_input.get("q_current"),
            "q_context": natural_input.get("q_context"),
            "query_views": [
                dict(view) for view in metadata_value.get("query_views") or []
                if isinstance(view, dict)
            ],
        })
        calls["eligible"].append([str(bucket.get("id") or "") for bucket in _eligible_buckets])
        candidates = candidates_by_query.get(query, candidates_by_query.get("*", []))
        return [dict(item) for item in candidates], []

    async def rerank(_query, items, *, diagnostics=None, documents_override=None):
        calls["rerank"].append({
            "ids": [str(item["bucket"].get("id") or "") for item in items],
            "documents": dict(documents_override or {}),
        })
        if isinstance(diagnostics, dict):
            diagnostics.update(provider_input_count=len(items), provider_output_count=len(items))
        output = []
        for item in items:
            cloned = dict(item)
            current_score = cloned.pop("_rerank_for_test", rerank_score)
            cloned["rerank_score"] = current_score
            output.append(cloned)
        return output

    def keep_selected(item):
        calls["selected"].append(dict(item))
        return item["bucket"]

    service._dynamic_bucket_candidate_items = candidate_builder
    service._rerank_scored_bucket_candidates = rerank
    service._bucket_with_recall_signal = keep_selected
    return service, calls, metadata


def test_build_recall_input_keeps_named_question_and_bare_same_turn_followup():
    messages = [
        {"role": "user", "content": _sample_named_question()},
        {"role": "user", "content": _sample_bare_followup()},
    ]
    before = [dict(message) for message in messages]

    recall_input = build_recall_input(
        messages,
        current_query=_sample_bare_followup(),
        recent_turns=4,
        text_extractor=_extract_text,
        cleaner=_clean_user_text,
        trace_context=_trace_for_messages(messages, current_indices={0, 1}),
    )

    assert recall_input["q_current"] == f"{_sample_named_question()}\n{_sample_bare_followup()}"
    assert recall_input["current_indices"] == [0, 1]
    assert recall_input["metadata"]["current_message_count"] == 2
    assert recall_input["metadata"]["current_input_complete"] is True
    assert messages == before


def test_build_recall_input_uses_coverage_indices_to_exclude_noncanonical_user_rows():
    messages = [
        {"role": "user", "content": "Injected worldbook text about an unrelated person."},
        {"role": "user", "content": "之前谈到她想去海边。"},
        {"role": "assistant", "content": "我会把这件事记下来。"},
        {"role": "user", "content": _sample_named_question()},
        {"role": "user", "content": _sample_bare_followup()},
    ]

    recall_input = build_recall_input(
        messages,
        current_query=_sample_bare_followup(),
        recent_turns=4,
        text_extractor=_extract_text,
        cleaner=_clean_user_text,
        trace_context=_trace_for_messages(
            messages, current_indices={3, 4}, covered_indices={1, 2, 3, 4}
        ),
    )

    assert recall_input["current_indices"] == [3, 4]
    assert recall_input["q_current"] == f"{_sample_named_question()}\n{_sample_bare_followup()}"
    assert "Injected worldbook text" not in recall_input["q_current"]
    assert "之前谈到她想去海边。" not in recall_input["q_current"]
    assert "之前谈到她想去海边。" in recall_input["q_context"]
    assert "assistant: 我会把这件事记下来。" in recall_input["q_context"]


def test_build_recall_input_marks_missing_final_current_event_incomplete():
    messages = [
        {"role": "user", "content": _sample_named_question()},
        {"role": "user", "content": _sample_bare_followup()},
    ]
    trace = _trace_for_messages(messages, current_indices={0}, covered_indices={0})

    recall_input = build_recall_input(
        messages,
        current_query=_sample_bare_followup(),
        text_extractor=_extract_text,
        cleaner=_clean_user_text,
        trace_context=trace,
    )

    assert recall_input["metadata"]["current_input_complete"] is False


def test_build_recall_input_defaults_to_four_completed_turns_not_four_messages():
    messages = []
    current_indices = set()
    for turn in range(5):
        messages.extend([
            {"role": "user", "content": f"completed-turn-{turn}-user"},
            {"role": "assistant", "content": f"completed-turn-{turn}-assistant"},
        ])
    current_indices.update({len(messages), len(messages) + 1})
    messages.extend([
        {"role": "user", "content": "current-first-staged-message"},
        {"role": "user", "content": "current-final-staged-message"},
    ])

    recall_input = build_recall_input(
        messages,
        current_query="current-final-staged-message",
        text_extractor=_extract_text,
        cleaner=_clean_user_text,
        trace_context=_trace_for_messages(messages, current_indices=current_indices),
    )

    assert recall_input["metadata"]["recent_turns_requested"] == 4
    assert recall_input["metadata"]["recent_turns_selected"] == 4
    assert "completed-turn-0-user" not in recall_input["q_context"]
    for turn in range(1, 5):
        assert f"completed-turn-{turn}-user" in recall_input["q_context"]
        assert f"completed-turn-{turn}-assistant" in recall_input["q_context"]
    assert "current-first-staged-message\ncurrent-final-staged-message" in recall_input["q_current"]
    assert recall_input["current_indices"] == [10, 11]


def test_build_recall_input_deduplicates_equal_current_and_context_query_views():
    messages = [{"role": "user", "content": "今晚吃什么？"}]
    recall_input = build_recall_input(
        messages,
        current_query="今晚吃什么？",
        text_extractor=_extract_text,
        cleaner=_clean_user_text,
        trace_context=_trace_for_messages(messages, current_indices={0}),
    )

    assert recall_input["q_current"] == recall_input["q_context"]
    assert recall_input["query_views"] == ["今晚吃什么？"]
    assert recall_input["metadata"]["query_views"][0]["kind"] == "current"


def test_natural_ordinary_semantics_do_not_change_legacy_local_fast_option():
    legacy_fast = GatewayService._recall_route_execution_options("fast")
    ordinary_fast_hint = GatewayService._recall_route_execution_options(
        "fast", ordinary=True,
    )

    assert legacy_fast["allow_semantic"] is False
    assert ordinary_fast_hint["allow_semantic"] is True


def test_task_context_and_disabled_planner_do_not_turn_off_ordinary_semantics():
    service = GatewayService.__new__(GatewayService)
    service.recall_timeout_seconds = 1.0
    calls = []
    options = GatewayService._recall_route_execution_options("ordinary", ordinary=True)
    options["allow_query_planner"] = False

    async def selector(_query, *, context_mode, natural_input, **kwargs):
        calls.append({"context_mode": context_mode, **kwargs})
        return [], [], {}

    sentinel = {"route": "ordinary"}
    result = asyncio.run(service._select_recall_with_fallback(
        selector,
        "ordinary task-mode query",
        sentinel_debug=sentinel,
        natural_input={"route": "ordinary", "_tasks": []},
        context_mode="task",
        **options,
    ))

    assert len(calls) == 1
    assert calls[0]["context_mode"] == "task"
    assert calls[0]["allow_semantic"] is True
    assert calls[0]["allow_query_planner"] is False
    assert result == ([], [], {})


def test_natural_total_timeout_retries_local_only_with_all_provider_steps_disabled():
    service = GatewayService.__new__(GatewayService)
    service.recall_timeout_seconds = 0.01
    calls = []
    natural_input = {"route": "ordinary", "_tasks": []}

    async def selector(_query, *, natural_input, allow_semantic, allow_query_planner,
                       allow_rerank, **_kwargs):
        calls.append({
            "local_only": bool(natural_input.get("local_only")),
            "allow_semantic": allow_semantic,
            "allow_query_planner": allow_query_planner,
            "allow_rerank": allow_rerank,
        })
        if len(calls) == 1:
            await asyncio.Event().wait()
        return [], [], {}

    result = asyncio.run(service._select_recall_with_fallback(
        selector,
        "ordinary query",
        sentinel_debug={"route": "ordinary"},
        natural_input=natural_input,
        allow_semantic=True,
        allow_query_planner=True,
        allow_rerank=True,
    ))

    assert calls == [
        {
            "local_only": False,
            "allow_semantic": True,
            "allow_query_planner": True,
            "allow_rerank": True,
        },
        {
            "local_only": True,
            "allow_semantic": False,
            "allow_query_planner": False,
            "allow_rerank": False,
        },
    ]
    assert "recall_total_timeout" in natural_input["incomplete_reasons"]
    assert result == ([], [], {})


def test_natural_timeout_retains_completed_verified_semantic_without_second_embedding_or_rerank():
    query = "An ordinary paraphrase about a shared evening experience."
    bucket = _natural_bucket("memory-timeout-semantic", "A committed source-backed evening fact.")
    service, calls, _metadata = _natural_finish_service(None, [bucket], {})
    service.recall_timeout_seconds = 0.03
    provider_calls = []
    rerank_started = asyncio.Event()
    rerank_cancelled = asyncio.Event()

    async def search(queries, *, cache_debug, **_kwargs):
        provider_calls.append(list(queries))
        cache_debug.update(status="miss", provider_requests=1, coverage_status="complete")
        return [{"bucket_id": bucket["id"], "score": 0.93,
                 "unit_id": "unit-timeout", "index_status": "verified"}]

    service.embedding_engine = SimpleNamespace(enabled=True, search_similar_queries=search)

    async def candidate_builder(_query, _session_id, buckets, *, natural_input,
                                allow_semantic, **_kwargs):
        eligible = {item["id"] for item in buckets}
        scores = (await service._natural_semantic_candidates(natural_input, eligible)
                  if allow_semantic else service._reuse_completed_natural_semantic(natural_input, eligible))
        hits = natural_input.get("semantic_hits") or {}
        return ([_natural_candidate(bucket, scores[bucket["id"]],
                                    index_status=hits[bucket["id"]]["index_status"])]
                if bucket["id"] in scores else []), []

    async def blocked_rerank(_query, _items, **_kwargs):
        rerank_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            rerank_cancelled.set()

    async def selector(selected_query, session_id, buckets, *, natural_input,
                       allow_query_planner, allow_rerank, **_kwargs):
        natural_input.update(allow_query_planner=allow_query_planner, allow_rerank=allow_rerank)
        return await service._finish_natural_selection(
            selected_query, session_id, buckets, recall_input=natural_input,
        )

    service._dynamic_bucket_candidate_items = candidate_builder
    service._rerank_scored_bucket_candidates = blocked_rerank
    natural_input = {"route": "ordinary", "q_current": query, "q_context": query,
                     "metadata": {}, "_tasks": []}
    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector, query, "session-timeout", [bucket], sentinel_debug={"route": "ordinary"},
        natural_input=natural_input, allow_semantic=True, allow_query_planner=False,
        allow_rerank=True,
    ))

    assert rerank_started.is_set() and rerank_cancelled.is_set()
    assert provider_calls == [[query]]
    assert [item["id"] for item in selected] == [bucket["id"]]
    assert suppressed == []
    assert debug["final_bucket_ids"] == [bucket["id"]]
    assert calls["selected"][0]["admission_reason"] == "strong_semantic"
    assert calls["selected"][0].get("rerank_score") is None
    assert natural_input["semantic_debug"]["provider_requests"] == 1
    assert "recall_total_timeout" in natural_input["incomplete_reasons"]


@pytest.mark.parametrize("change", [
    "context_query", "current_query", "memory_id", "revision", "body_sha256",
    "disabled", "partial",
])
def test_completed_semantic_timeout_checkpoint_rejects_changed_or_unverified_source(change):
    service = GatewayService.__new__(GatewayService)
    service.semantic_candidate_top_k = 4
    service._clamp = lambda score: max(0.0, min(1.0, float(score)))
    calls = []
    bucket_id = "memory-checkpoint"
    query = "A synthetic ordinary experience paraphrase."
    metadata = {bucket_id: {"memory_id": "authority-memory", "revision": 7,
                            "body_sha256": "a" * 64}}

    async def search(queries, *, cache_debug, **_kwargs):
        calls.append(list(queries))
        cache_debug.update(status="miss", provider_requests=1)
        return [{"bucket_id": bucket_id, "score": 0.92, "unit_id": "unit-1",
                 "index_status": "partial" if change == "partial" else "verified"}]

    service.embedding_engine = SimpleNamespace(enabled=True, search_similar_queries=search)
    recall_input = {"q_context": query, "q_current": query, "metadata": {},
                    "index_metadata": metadata, "local_only": False}
    asyncio.run(service._natural_semantic_candidates(recall_input, {bucket_id}))
    assert len(calls) == 1
    recall_input["local_only"] = True
    if change == "context_query":
        recall_input["q_context"] = query + " changed"
    elif change == "current_query":
        recall_input["q_current"] = query + " changed"
    elif change == "disabled":
        pass
    elif change != "partial":
        recall_input["index_metadata"] = {bucket_id: {**metadata[bucket_id], change: (
            "different-memory" if change == "memory_id" else 8 if change == "revision" else "b" * 64)}}

    eligible = set() if change == "disabled" else {bucket_id}
    assert service._reuse_completed_natural_semantic(recall_input, eligible) == {}
    assert recall_input["semantic_hits"] == {}
    assert len(calls) == 1


def test_external_natural_recall_cancellation_does_not_trigger_deadline_fallback():
    service = GatewayService.__new__(GatewayService)
    service.recall_timeout_seconds = 1.0
    started = asyncio.Event()
    calls = []
    natural_input = {"route": "ordinary", "_tasks": []}

    async def selector(*_args, **_kwargs):
        calls.append(True)
        started.set()
        await asyncio.Event().wait()

    async def scenario():
        task = asyncio.create_task(service._select_recall_with_fallback(
            selector, "synthetic query", sentinel_debug={"route": "ordinary"},
            natural_input=natural_input, allow_semantic=True,
        ))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert calls == [True]
    assert natural_input.get("local_only") is not True
    assert "recall_total_timeout" not in natural_input.get("incomplete_reasons", [])


def test_internal_runtime_scope_exits_before_memory_or_canonical_work():
    service = GatewayService.__new__(GatewayService)
    service._maybe_reload_runtime_overlay = lambda: None
    service.upstream_default_model = "test-model"
    service._get_upstream_for_model = lambda _model: None

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("internal worker must not enter owner Recall or canonical work")

    service._list_gateway_buckets = forbidden
    service._record_raw_event = forbidden
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "synthetic internal query"}],
        "_ombre_trace_context": {"runtime_scope": "embedding_worker"},
    }

    forwarded, recalled_ids, debug = asyncio.run(service.prepare_payload(
        payload, "synthetic-session", include_debug=True,
    ))

    assert forwarded == payload
    assert recalled_ids is None
    assert debug["internal_scope"] is True
    assert debug["recall_status"]["status"] == "disabled"
    assert debug["recall_status"]["injected_count"] == 0


def test_continuation_snapshot_reuses_the_same_memory_recall_projection():
    service = GatewayService.__new__(GatewayService)
    service.prepare_snapshot_ttl_seconds = 300
    service._restore_cached_reasoning_content = lambda *_args, **_kwargs: None
    service._apply_prompt_cache_hints = lambda *_args, **_kwargs: None
    service._inject_context_messages = lambda messages, stable, dynamic: [
        *messages,
        {"role": "system", "content": "\n".join(part for part in (stable, dynamic) if part)},
    ]
    base_messages = [{"role": "user", "content": "synthetic initial user turn"}]
    tool_tail = [
        {"role": "assistant", "tool_calls": [{"id": "tool-1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "tool-1", "content": "synthetic tool result"},
    ]
    dynamic_projection = "[source=ombre.memory_recall revision=4] synthetic selected experience"
    shared_recall_status = {
        "status": "selected",
        "route": "deep",
        "selected_count": 2,
        "injected_count": 1,
        "embedding_request_count": 1,
        "embedding_timing": {"total_ms": 18},
    }
    identity = {
        "key_hash": "snapshot-key-hash",
        "snapshot_parent_key": "snapshot-parent-key",
        "source_phase": "initial",
        "transport_session_id": "synthetic-session",
        "baseline_messages": base_messages,
        "tool_tail": tool_tail,
    }
    service._prepare_snapshot_store(
        identity,
        prepared_payload={"model": "test-model", "messages": base_messages},
        base_forward_messages=base_messages,
        stable_context="stable synthetic context",
        dynamic_context=dynamic_projection,
        injected_ids=["memory-projection-id"],
        debug_payload={
            "recall_status": shared_recall_status,
            "prepare_timing_debug": {"recall_status": shared_recall_status},
            "memory_recall_projection": {
                "recall_status": shared_recall_status,
                "selected_memory_ids": ["memory-id-1", "memory-id-2"],
                "selected_bucket_ids": ["memory-projection-id", "memory-other-id"],
                "injected_bucket_ids": ["memory-projection-id"],
            },
        },
    )
    continuation_payload = {
        "model": "test-model",
        "stream": False,
        "messages": [*base_messages, *tool_tail],
    }

    replayed, injected_ids, debug = service._prepare_snapshot_replay(
        service._prepare_snapshots[identity["key_hash"]],
        identity,
        continuation_payload,
        continuation_phase=True,
        started_at=time.perf_counter(),
        include_debug=True,
    )

    replayed_context = replayed["messages"][-1]["content"]
    assert replayed_context.count("source=ombre.memory_recall revision=4") == 1
    assert replayed_context.endswith("synthetic selected experience")
    assert replayed["messages"][1:3] == tool_tail
    assert injected_ids == ["memory-projection-id"]
    assert debug["prepare_snapshot_cache"]["status"] == "hit"
    status = debug["recall_status"]
    assert debug["prepare_timing_debug"]["recall_status"] is status
    assert debug["memory_recall_projection"]["recall_status"] is status
    assert status["embedding_request_count"] == 0
    assert status["parent_embedding_request_count"] == 1
    assert status["embedding_timing"] is None
    assert status["route"] == "local"
    assert status["parent_route"] == "deep"
    assert status["snapshot_reused_from_parent"] is True
    assert status["status"] == "selected"
    assert status["selected_count"] == 2
    assert status["injected_count"] == 1
    assert debug["memory_recall_projection"]["selected_memory_ids"] == [
        "memory-id-1", "memory-id-2",
    ]
    assert debug["memory_recall_projection"]["selected_bucket_ids"] == [
        "memory-projection-id", "memory-other-id",
    ]
    assert debug["memory_recall_projection"]["injected_bucket_ids"] == [
        "memory-projection-id",
    ]


@pytest.mark.parametrize(
    ("current", "context"),
    [
        (
            "The coast trip felt different from my usual weekends.",
            "Recent dialogue mentions how the quiet coast reset my energy.",
        ),
        (
            "For short trips I tend to choose trains over driving.",
            "Recent dialogue mentions enjoying quiet rail routes.",
        ),
        (
            "I felt a little left behind at the farewell.",
            "Recent dialogue mentions the final evening with friends.",
        ),
    ],
    ids=["experience", "preference", "emotional-experience"],
)
def test_natural_semantic_candidates_use_context_view_without_keyword_gate(current, context):
    service = GatewayService.__new__(GatewayService)
    service.semantic_candidate_top_k = 24
    service._clamp = lambda value: max(0.0, min(1.0, float(value or 0.0)))
    calls = []
    async def search(queries, *, top_k, eligible_ids, index_metadata, cache_debug, deadline):
        calls.append((queries, top_k, eligible_ids, index_metadata, deadline))
        cache_debug.update(provider_requests=1, coverage_status="complete")
        return [{
            "bucket_id": "authorized-memory",
            "score": 0.84,
            "unit_id": "moment-1",
            "index_status": "verified",
        }]

    service.embedding_engine = SimpleNamespace(enabled=True, search_similar_queries=search)
    recall_input = {
        "q_current": current,
        "q_context": context,
        "index_metadata": {"authorized-memory": {"revision": 4}},
        "_deadline": 123.0,
        "metadata": {"query_views": [
            {"kind": "current", "sha256": hashlib.sha256(current.encode()).hexdigest()},
            {"kind": "context", "sha256": hashlib.sha256(context.encode()).hexdigest()},
        ]},
    }

    scores = asyncio.run(service._natural_semantic_candidates(recall_input, {"authorized-memory"}))

    assert calls == [([context], 24, {"authorized-memory"}, recall_input["index_metadata"], 123.0)]
    assert scores == {"authorized-memory": 0.84}
    assert recall_input["semantic_hits"]["authorized-memory"]["unit_id"] == "moment-1"
    assert [view["semantic_used"] for view in recall_input["metadata"]["query_views"]] == [False, True]


def test_verified_semantic_candidate_can_pass_without_literal_evidence():
    query = "An ordinary paraphrase about the evening routine."
    bucket = _natural_bucket("memory-paraphrase", "The owner often rests after a long day.")
    candidate = _natural_candidate(bucket, 0.9)
    service, calls, _metadata = _natural_finish_service(
        None, [bucket], {query: [candidate]},
    )
    recall_input = {"q_current": query, "q_context": query, "metadata": {}}

    selected, suppressed, debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", [bucket], recall_input=recall_input,
    ))

    assert [bucket["id"] for bucket in selected] == ["memory-paraphrase"]
    assert suppressed == []
    assert calls["rerank"] and len(calls["rerank"]) == 1
    assert calls["selected"][0]["admission_reason"] == "strong_semantic"
    assert calls["selected"][0]["hard_evidence_labels"] == []
    assert debug["final_bucket_ids"] == ["memory-paraphrase"]


@pytest.mark.parametrize(
    ("score", "index_status"),
    [(0.79, "verified"), (0.99, "partial")],
)
def test_below_threshold_or_incomplete_semantic_only_candidate_is_not_admitted(score, index_status):
    query = "A general recollection paraphrase."
    bucket = _natural_bucket(f"memory-negative-{index_status}", "A loosely related experience.")
    candidate = _natural_candidate(bucket, score, index_status=index_status)
    service, _calls, _metadata = _natural_finish_service(
        None, [bucket], {query: [candidate]},
    )

    selected, suppressed, _debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", [bucket],
        recall_input={"q_current": query, "q_context": query, "metadata": {}},
    ))

    assert selected == []
    assert len(suppressed) == 1
    assert suppressed[0]["admission_reason"] == "insufficient_contextual_relevance"


def test_semantic_similarity_surfaces_related_experience_without_asserting_person_identity():
    query = "Who is Avery?"
    bucket = _natural_bucket("memory-other-person", "A different friend enjoyed a weekend walk.")
    candidate = _natural_candidate(bucket, 0.91, labels=("semantic_only",))
    service, calls, _metadata = _natural_finish_service(
        None, [bucket], {query: [candidate]},
    )
    recall_input = {"q_current": query, "q_context": query, "metadata": {}}

    selected, suppressed, _debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", [bucket], recall_input=recall_input,
    ))

    assert [item["id"] for item in selected] == ["memory-other-person"]
    assert suppressed == []
    assert calls["selected"][0]["admission_reason"] == "strong_semantic"
    assert calls["selected"][0]["hard_evidence_labels"] == []
    assert not any(key in calls["selected"][0] for key in ("identity_alias", "entity_identity", "identified_person"))


def test_manual_only_and_body_revision_mismatch_are_removed_before_candidate_generation():
    query = "A plain paraphrase about a prior activity."
    active = _natural_bucket("memory-active", "A valid current memory body.", revision=4)
    manual = _natural_bucket("memory-manual", "A manual-only body.", revision=2)
    stale = _natural_bucket("memory-stale", "A body changed after its index.", revision=3)
    candidate = _natural_candidate(active, 0.9)
    service, calls, metadata = _natural_finish_service(
        None, [active, manual, stale], {query: [candidate]},
        manual_only_ids={"memory-manual"},
    )
    metadata["memory-stale"]["body_sha256"] = "stale-index-hash"
    recall_input = {"q_current": query, "q_context": query, "metadata": {}}

    selected, _suppressed, _debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", [active, manual, stale], recall_input=recall_input,
    ))

    assert calls["eligible"] == [["memory-active"]]
    assert [item["id"] for item in selected] == ["memory-active"]
    assert "bucket_revision_mismatch" in recall_input["incomplete_reasons"]


def test_fresh_content_moment_enters_parent_pool_and_supplies_final_rerank_document(monkeypatch):
    query = "An ordinary paraphrase about a past evening."
    bucket = _natural_bucket("memory-moment-parent", "Committed parent body without a direct phrase hit.")
    fresh_moment = {
        "moment_id": "fresh-content-unit",
        "bucket_id": "memory-moment-parent",
        "source": "content",
        "score": 0.92,
        "content": "A fresh content passage describes a calming evening.",
    }
    service, calls, _metadata = _natural_finish_service(
        None, [bucket], {query: []}, rerank_score=0.95,
    )
    parsed = []
    searched = []

    def parse_fresh(_bucket, _options):
        parsed.append(True)
        return [dict(fresh_moment)]

    def search_fresh(_query, moments, *, limit):
        searched.extend(moments)
        return [dict(fresh_moment)]

    monkeypatch.setattr(gateway_module, "parse_bucket_moments", parse_fresh)
    service.memory_moment_store = SimpleNamespace(search_moment_items=search_fresh)
    service._moment_rerank_document = lambda moment: f"document:{moment['moment_id']}"
    service._direct_moments_for_bucket = lambda _bucket, _query: [dict(fresh_moment)]
    service._moment_with_bucket_recall_signal = lambda moment, _signal: dict(moment)
    service._bucket_candidate_recall_signal = lambda item: {"bucket_id": item["bucket"]["id"]}

    moments, candidates, _suppressed_moments, _suppressed_buckets, debug = asyncio.run(
        service._finish_natural_selection(
            query, "session-natural-test", [bucket],
            recall_input={"q_current": query, "q_context": query, "metadata": {}},
            grouped_moments={},
        )
    )

    assert parsed == [True]
    assert [item["moment_id"] for item in searched] == ["fresh-content-unit"]
    assert [item["bucket_id"] for item in candidates] == ["memory-moment-parent"]
    assert [item["moment_id"] for item in moments] == ["fresh-content-unit"]
    assert calls["rerank"][0]["ids"] == ["memory-moment-parent"]
    assert calls["rerank"][0]["documents"] == {
        "memory-moment-parent": "document:fresh-content-unit",
    }
    assert len(calls["rerank"]) == 1
    assert debug["final_bucket_ids"] == ["memory-moment-parent"]


@pytest.mark.parametrize(
    "supplemental_semantic_enabled",
    [False, True],
    ids=["default-local-only", "explicit-supplemental-semantic"],
)
def test_weak_pool_runs_one_planner_and_supplemental_semantic_is_opt_in(
    supplemental_semantic_enabled,
):
    query = "An ordinary question with unresolved event context."
    context_view = f"{query}\nRecent dialogue adds bounded background."
    buckets = [
        _natural_bucket("memory-original", "Original candidate body."),
        _natural_bucket("memory-supplement-1", "First supplemental candidate body."),
        _natural_bucket("memory-supplement-2", "Second supplemental candidate body."),
    ]
    original = _natural_candidate(buckets[0], 0.6, unit_id="unit-original")
    extra_1 = _natural_candidate(
        buckets[1], 0.6, unit_id="unit-supplement-1", labels=("exact_anchor",),
    )
    extra_2 = _natural_candidate(
        buckets[2], 0.6, unit_id="unit-supplement-2", labels=("exact_anchor",),
    )
    service, calls, _metadata = _natural_finish_service(
        None, buckets,
        {
            query: [original],
            "supplemental-one": [extra_1],
            "supplemental-two": [extra_2],
        },
        rerank_score=0.95,
    )
    service.query_planner_enabled = True
    if supplemental_semantic_enabled:
        service.query_planner_supplemental_semantic = True
    planner_calls = []

    async def planner(planner_query):
        planner_calls.append(planner_query)
        return {"queries": [
            {"query": "supplemental-one"},
            {"query": "supplemental-two"},
        ]}, None

    service._call_query_planner = planner
    main_query_views = [
        {"kind": "current", "sha256": hashlib.sha256(query.encode()).hexdigest()},
        {"kind": "context", "sha256": hashlib.sha256(context_view.encode()).hexdigest()},
    ]
    recall_input = {
        "q_current": query,
        "q_context": context_view,
        "metadata": {"query_views": [dict(view) for view in main_query_views]},
    }

    selected, suppressed, debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", buckets, recall_input=recall_input,
    ))

    assert planner_calls == [context_view]
    assert calls["candidate"] == [query, "supplemental-one", "supplemental-two"]
    assert [item["allow_semantic"] for item in calls["candidate_options"]] == [
        True, supplemental_semantic_enabled, supplemental_semantic_enabled,
    ]
    assert calls["candidate_options"][0]["query_views"] == main_query_views
    assert all(
        len(item["query_views"]) == 1
        and item["query_views"][0]["kind"] == "supplemental"
        for item in calls["candidate_options"][1:]
    )
    assert recall_input["q_current"] == query
    assert recall_input["q_context"] == context_view
    assert recall_input["metadata"]["query_views"][:2] == main_query_views
    assert [view["kind"] for view in recall_input["metadata"]["query_views"]] == [
        "current", "context", "supplemental", "supplemental",
    ]
    assert len(calls["rerank"]) == 1
    assert set(calls["rerank"][0]["ids"]) == {
        "memory-original", "memory-supplement-1", "memory-supplement-2",
    }
    assert {item["id"] for item in selected} == {
        "memory-original", "memory-supplement-1", "memory-supplement-2",
    }
    assert suppressed == []
    assert debug["triggered"] is True


@pytest.mark.parametrize(
    ("score", "index_status", "labels"),
    [
        (0.95, "verified", ("semantic_only",)),
        (0.3, "partial", ("exact_anchor",)),
    ],
    ids=["verified-strong-semantic", "hard-local-evidence"],
)
def test_clear_hard_or_verified_semantic_candidate_skips_planner(
    score, index_status, labels,
):
    query = "An ordinary query with already clear evidence."
    bucket = _natural_bucket("memory-clear-candidate", "A current, committed test fact.")
    candidate = _natural_candidate(
        bucket, score, index_status=index_status, labels=labels,
    )
    service, calls, _metadata = _natural_finish_service(
        None, [bucket], {query: [candidate]},
    )
    service.query_planner_enabled = True
    planner_calls = []

    async def unexpected_planner(planner_query):
        planner_calls.append(planner_query)
        return {"queries": [{"query": "unexpected-expansion"}]}, None

    service._call_query_planner = unexpected_planner
    recall_input = {
        "q_current": query,
        "q_context": query,
        "metadata": {"query_views": [{"kind": "current"}]},
    }

    selected, suppressed, debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", [bucket], recall_input=recall_input,
    ))

    assert planner_calls == []
    assert calls["candidate"] == [query]
    assert [item["id"] for item in selected] == ["memory-clear-candidate"]
    assert suppressed == []
    assert debug["triggered"] is False


@pytest.mark.parametrize(
    "intervening",
    [
        {"role": "assistant", "tool_calls": [{"id": "tool-1"}], "content": ""},
        {"role": "tool", "content": "工具结果不应成为对话轮边界。"},
    ],
)
def test_tool_carriers_do_not_become_context_or_complete_turn_boundaries(intervening):
    messages = [
        {"role": "user", "content": "Prior user message with a past plan."},
        intervening,
        {"role": "user", "content": _sample_bare_followup()},
    ]
    recall_input = build_recall_input(
        messages,
        current_query=_sample_bare_followup(),
        text_extractor=_extract_text,
        cleaner=_clean_user_text,
        trace_context=_trace_for_messages(messages, current_indices={2}),
    )

    assert "工具结果不应成为对话轮边界。" not in recall_input["q_context"]
    assert recall_input["current_indices"] == [2]


def test_query_embeddings_batches_views_and_restores_provider_indices(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    calls = []

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        inputs = _provider_input_items(input_value)
        calls.append(inputs)
        rows = [
            {"index": index, "embedding": [float(index + 1), float(index + 10)]}
            for index, _text in enumerate(inputs)
        ]
        return 200, {"data": list(reversed(rows))}

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    queries = ["q-current", "q-context"]
    vectors = asyncio.run(engine.query_embeddings(queries))

    assert len(calls) == 1
    assert len(calls[0]) == 2
    request_text_by_input_order = calls[0]
    expected = {
        text.rsplit("\nQuery: ", 1)[-1]: [float(index + 1), float(index + 10)]
        for index, text in enumerate(request_text_by_input_order)
    }
    assert vectors == [expected[query] for query in queries]
    assert engine.runtime_debug()["last_result_count"] == 2
    assert all(query not in str(engine._query_cache) for query in queries)


def test_query_embeddings_reuses_cached_view_and_batches_only_missing_view(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    request_sizes = []

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        inputs = _provider_input_items(input_value)
        request_sizes.append(len(inputs))
        return 200, {
            "data": [
                {"index": index, "embedding": [float(sum(map(ord, text))), 1.0]}
                for index, text in enumerate(inputs)
            ],
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        first = await engine.query_embeddings(["cache-a", "cache-b"])
        second = await engine.query_embeddings(["cache-a", "cache-c"])
        return first, second

    first, second = asyncio.run(run())

    assert request_sizes == [2, 1]
    assert second[0] == first[0]
    assert second[1] != first[1]


def test_query_embeddings_singleflights_same_batch(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    calls = []
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        inputs = _provider_input_items(input_value)
        calls.append(len(inputs))
        provider_started.set()
        await release_provider.wait()
        return 200, {
            "data": [
                {"index": index, "embedding": [float(index + 1), 2.0]}
                for index, _text in enumerate(inputs)
            ],
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        first = asyncio.create_task(engine.query_embeddings(["flight-a", "flight-b"]))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        second = asyncio.create_task(engine.query_embeddings(["flight-a", "flight-b"]))
        await asyncio.sleep(0)
        release_provider.set()
        return await asyncio.gather(first, second)

    first, second = asyncio.run(run())

    assert calls == [2]
    assert first == second
    assert not engine._query_batch_inflight


def test_query_embeddings_cancellation_cancels_provider_task_and_cleans_inflight(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    async def fake_request(_endpoint, _api_key, _model, _input_value, *, deadline=None):
        provider_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            provider_cancelled.set()

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        task = asyncio.create_task(engine.query_embeddings(["cancel-a", "cancel-b"]))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert provider_cancelled.is_set()
    assert not engine._query_batch_inflight


def test_query_embeddings_cancelled_waiter_does_not_cancel_shared_provider(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    provider_cancelled = asyncio.Event()
    calls = []

    async def fake_request(_endpoint, _api_key, _model, _input_value, *, deadline=None):
        calls.append(True)
        provider_started.set()
        try:
            await release_provider.wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise
        return 200, {
            "data": [{"index": 0, "embedding": [0.5, 0.5]}],
            "_local_http_timing": {"provider_attempts": 1},
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    owner_debug = {}

    async def run():
        owner = asyncio.create_task(engine.query_embeddings(
            ["same-shared-batch"], cache_debug=owner_debug,
        ))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        waiter = asyncio.create_task(engine.query_embeddings(["same-shared-batch"]))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not provider_cancelled.is_set()
        release_provider.set()
        result = await owner
        assert not provider_cancelled.is_set()
        assert not engine._query_batch_inflight
        return result

    result = asyncio.run(run())

    assert len(calls) == 1
    assert result == [[0.5, 0.5]]
    assert owner_debug["provider_requests"] == 1


def test_semantic_search_filters_manual_and_stale_vectors_before_top_k_without_reembedding_candidates(
    tmp_path, monkeypatch
):
    engine = _embedding_engine(tmp_path)
    engine._store_embedding(
        "manual-only", [1.0, 0.0], input_text="stored candidate text",
    )
    engine._store_embedding(
        "stale", [1.0, 0.0], source_revision=1, source_sha256="old-hash",
        source_unit_id="stale-unit", parent_memory_id="memory-stale",
        input_text="stored candidate text",
    )
    engine._store_embedding(
        "eligible", [0.3, 0.953939201], source_revision=3, source_sha256="current-hash",
        source_unit_id="eligible-lower-unit", parent_memory_id="memory-eligible",
        input_text="stored lower-scoring unit",
    )
    engine._store_embedding(
        "eligible", [0.8, 0.6], source_revision=3, source_sha256="current-hash",
        source_unit_id="eligible-unit", parent_memory_id="memory-eligible",
        input_text="stored best-scoring unit",
    )
    query_requests = []

    async def forbidden_candidate_embedding(*_args, **_kwargs):
        raise AssertionError("stored candidate text must not be embedded during chat search")

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        query_requests.append(len(_provider_input_items(input_value)))
        return 200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(engine, "_generate_embedding", forbidden_candidate_embedding)
    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    index_metadata = {
        "stale": {"revision": 2, "body_sha256": "new-hash", "memory_id": "memory-stale"},
        "eligible": {"revision": 3, "body_sha256": "current-hash", "memory_id": "memory-eligible"},
    }
    debug = {}

    results = asyncio.run(engine.search_similar_queries(
        ["ordinary-query"], top_k=1,
        eligible_ids={"stale", "eligible"},
        index_metadata=index_metadata,
        cache_debug=debug,
    ))

    assert [item["bucket_id"] for item in results] == ["eligible"]
    assert results[0]["index_status"] == "verified"
    assert results[0]["unit_id"] == "eligible-unit"
    assert query_requests == [1]
    assert debug["index_rejections"]["stale_memory_revision"] == 1


def test_semantic_search_keeps_current_and_context_views_as_distinct_rank_sources(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    engine._store_embedding("memory-current", [1.0, 0.0], input_text="current fact")
    engine._store_embedding("memory-context", [0.0, 1.0], input_text="context fact")
    request_sizes = []

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        inputs = _provider_input_items(input_value)
        request_sizes.append(len(inputs))
        rows = []
        for index, prepared in enumerate(inputs):
            query = prepared.rsplit("\nQuery: ", 1)[-1]
            vector = [1.0, 0.0] if query == "view-current" else [0.0, 1.0]
            rows.append({"index": index, "embedding": vector})
        return 200, {"data": rows}

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    results = asyncio.run(engine.search_similar_queries(
        ["view-current", "view-context"], top_k=1,
        eligible_ids={"memory-current", "memory-context"},
    ))

    assert request_sizes == [2]
    assert {item["bucket_id"] for item in results} == {"memory-current", "memory-context"}
    assert next(item for item in results if item["bucket_id"] == "memory-current")["view_ranks"] == {"0": 1}
    assert next(item for item in results if item["bucket_id"] == "memory-context")["view_ranks"] == {"1": 1}


def test_semantic_search_marks_missing_active_index_metadata_incomplete_before_top_k(
    tmp_path, monkeypatch
):
    engine = _embedding_engine(tmp_path)
    engine._store_embedding(
        "legacy-no-revision", [1.0, 0.0], input_text="legacy stored vector",
    )
    engine._store_embedding(
        "eligible", [0.8, 0.6], source_revision=3, source_sha256="current-hash",
        source_unit_id="eligible-unit", parent_memory_id="memory-eligible",
        input_text="current stored unit",
    )

    async def fake_request(_endpoint, _api_key, _model, _input_value, *, deadline=None):
        return 200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    debug = {}
    results = asyncio.run(engine.search_similar_queries(
        ["ordinary-query"], top_k=1,
        eligible_ids={"legacy-no-revision", "eligible"},
        index_metadata={
            "eligible": {"revision": 3, "body_sha256": "current-hash", "memory_id": "memory-eligible"},
        },
        cache_debug=debug,
    ))

    assert [item["bucket_id"] for item in results] == ["eligible"]
    assert debug["coverage_status"] == "incomplete"
    assert debug["index_rejections"]["active_index_metadata_missing"] == 1


def test_semantic_search_chunks_long_query_without_losing_input_coverage(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    engine.max_chars = 64
    engine.query_instruction = "x"
    engine._store_embedding("memory-1", [1.0, 0.0], input_text="stored fact")
    provider_inputs = []

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        inputs = _provider_input_items(input_value)
        provider_inputs.extend(inputs)
        return 200, {
            "data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index, _text in enumerate(inputs)
            ],
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    query = "".join(chr(0xE000 + index) for index in range(101))
    debug = {}

    results = asyncio.run(engine.search_similar_queries(
        [query], top_k=1, eligible_ids={"memory-1"}, cache_debug=debug,
    ))

    chunks = [item.rsplit("\nQuery: ", 1)[-1] for item in provider_inputs]
    assert debug["query_chunk_count"] == 3
    assert len(chunks) == 3
    assert "".join(sorted(chunks, key=query.index)) == query
    assert [item["bucket_id"] for item in results] == ["memory-1"]


def test_semantic_search_does_not_embed_queries_without_a_compatible_index(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    engine._store_embedding(
        "memory-stale", [1.0, 0.0], source_revision=1,
        source_sha256="old-body-hash", source_unit_id="old-unit",
        parent_memory_id="memory-stale", input_text="old committed content",
    )
    calls = []

    async def forbidden_request(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("query provider must not run without a compatible index")

    monkeypatch.setattr(engine, "_request_embedding", forbidden_request)
    debug = {}
    results = asyncio.run(engine.search_similar_queries(
        ["ordinary-query"], top_k=1,
        eligible_ids={"memory-stale"},
        index_metadata={"memory-stale": {
            "memory_id": "memory-stale",
            "revision": 2,
            "body_sha256": "current-body-hash",
        }},
        cache_debug=debug,
    ))

    assert results == []
    assert calls == []
    assert debug["status"] == "no_compatible_index"
    assert debug["provider_requests"] == 0
    assert debug["valid_unit_count"] == 0
    assert debug["index_rejections"]["stale_memory_revision"] == 1


def test_semantic_search_reports_query_document_dimension_mismatch(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    engine._store_embedding(
        "memory-dimension", [1.0, 0.0, 0.0], source_revision=4,
        source_sha256="active-body-hash", source_unit_id="unit-dimension",
        parent_memory_id="memory-dimension", input_text="active committed content",
    )
    calls = []

    async def fake_request(_endpoint, _api_key, _model, _input_value, *, deadline=None):
        calls.append(True)
        return 200, {
            "data": [{"index": 0, "embedding": [1.0, 0.0]}],
            "_local_http_timing": {"provider_attempts": 1},
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    debug = {}
    results = asyncio.run(engine.search_similar_queries(
        ["ordinary-query"], top_k=1,
        eligible_ids={"memory-dimension"},
        index_metadata={"memory-dimension": {
            "memory_id": "memory-dimension",
            "revision": 4,
            "body_sha256": "active-body-hash",
        }},
        cache_debug=debug,
    ))

    assert results == []
    assert calls == [True]
    assert debug["index_rejections"]["query_document_dimension_mismatch"] == 1
    assert debug["valid_unit_count"] == 0
    assert debug["coverage_status"] == "incomplete"


def test_concurrent_distinct_query_batches_keep_provider_counts_request_local(tmp_path, monkeypatch):
    engine = _embedding_engine(tmp_path)
    calls = []

    async def fake_request(_endpoint, _api_key, _model, input_value, *, deadline=None):
        request_number = len(calls) + 1
        calls.append(True)
        await asyncio.sleep(0)
        return 200, {
            "data": [{"index": 0, "embedding": [float(request_number), 1.0]}],
            "_local_http_timing": {"provider_attempts": 1},
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    left_debug, right_debug = {}, {}

    async def run():
        return await asyncio.gather(
            engine.query_embeddings(["distinct-left"], cache_debug=left_debug),
            engine.query_embeddings(["distinct-right"], cache_debug=right_debug),
        )

    left, right = asyncio.run(run())

    assert len(calls) == 2
    assert left != right
    assert left_debug["provider_requests"] == 1
    assert right_debug["provider_requests"] == 1
    assert left_debug["shared_provider_request"] is False
    assert right_debug["shared_provider_request"] is False
    assert left_debug["status"] == right_debug["status"] == "miss"


def test_hot_model_change_keeps_inflight_query_vector_under_its_original_cache_key(
    tmp_path, monkeypatch
):
    engine = _embedding_engine(tmp_path)
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    provider_models = []

    async def fake_request(_endpoint, _api_key, model, _input_value, *, deadline=None):
        provider_models.append(model)
        if len(provider_models) == 1:
            provider_started.set()
            await release_provider.wait()
        return 200, {
            "data": [{"index": 0, "embedding": [float(len(provider_models)), 1.0]}],
            "_local_http_timing": {"provider_attempts": 1},
        }

    monkeypatch.setattr(engine, "_request_embedding", fake_request)
    original_snapshot = engine._query_config_snapshot()
    first_debug, next_debug = {}, {}

    async def run():
        first_task = asyncio.create_task(engine.query_embeddings(
            ["same-query-after-hot-change"],
            cache_debug=first_debug,
            config_snapshot=original_snapshot,
        ))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        engine.model = "test-embedding-model-hot"
        release_provider.set()
        first = await first_task
        next_model = await engine.query_embeddings(
            ["same-query-after-hot-change"], cache_debug=next_debug,
        )
        return first, next_model

    first, next_model = asyncio.run(run())

    assert provider_models == ["test-embedding-model", "test-embedding-model-hot"]
    assert first != next_model
    assert first_debug["provider_requests"] == 1
    assert next_debug["provider_requests"] == 1
    assert next_debug["status"] == "miss"


def test_hot_document_config_keeps_original_model_preparation_and_source_provenance(
    tmp_path, monkeypatch
):
    engine = _embedding_engine(tmp_path)
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    provider_requests = []
    source_body = "active source content"
    source_hash = hashlib.sha256(source_body.encode("utf-8")).hexdigest()

    async def fake_request(_endpoint, _api_key, model, input_value, *, deadline=None):
        provider_requests.append((model, input_value))
        provider_started.set()
        await release_provider.wait()
        return 200, {"data": [{"index": 0, "embedding": [0.25, 0.75]}]}

    monkeypatch.setattr(engine, "_request_embedding", fake_request)

    async def run():
        task = asyncio.create_task(engine.generate_and_store(
            "memory-hot-document",
            source_body,
            source_revision=7,
            source_sha256=source_hash,
            source_unit_id="unit-hot-document",
            parent_memory_id="memory-hot-document",
            document_instruction="old document instruction",
        ))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        engine.model = "test-embedding-model-hot"
        engine.document_instruction = "new document instruction"
        release_provider.set()
        return await task

    assert asyncio.run(run()) is True

    assert provider_requests == [(
        "test-embedding-model",
        "Instruct: old document instruction\nDocument: active source content",
    )]
    conn = sqlite3.connect(engine.db_path)
    try:
        row = conn.execute(
            "SELECT model, dimension, parent_bucket_id, parent_memory_id, source_unit_id, "
            "source_revision, source_sha256, preparation_sha256, provider "
            "FROM embeddings WHERE parent_bucket_id = ?",
            ("memory-hot-document",),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[:7] == (
        "test-embedding-model", 2, "memory-hot-document", "memory-hot-document",
        "unit-hot-document", 7, source_hash,
    )
    assert row[7] == engine.preparation_hash(instruction="old document instruction")
    assert row[8] == engine.base_url.rstrip("/")


def _seed_index_map_memory_rows(authority, rows):
    conn = authority._connect()
    try:
        conn.execute("BEGIN")
        for memory_id, state, policy, metadata in rows:
            body = f"synthetic committed body for {memory_id}"
            body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            timestamp = "2026-09-30T00:00:00+00:00"
            conn.execute(
                "INSERT INTO memories(memory_id,bucket_id,active_revision,state,"
                "recall_policy,body_sha256,updated_at) VALUES(?,?,1,?,?,?,?)",
                (memory_id, memory_id, state, policy, body_sha, timestamp),
            )
            conn.execute(
                "INSERT INTO memory_revisions(memory_id,revision,body_sha256,snapshot_path,"
                "metadata_json,source_refs_json,decision_source,operation_id,created_at,created_by) "
                "VALUES(?,1,?,?,?,?,?,?,?,?)",
                (
                    memory_id, body_sha, f"revisions/{memory_id}/1.md",
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    json.dumps([f"test:{memory_id}"]), "test",
                    f"index-map:{memory_id}", timestamp, "test",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_memory_index_metadata_map_keeps_401st_active_memory_without_private_or_manual_rows(
    tmp_path,
):
    state_dir = tmp_path / "authority"
    authority = MemoryAuthorityStore({"state_dir": str(state_dir)})
    enabled_ids = [f"memory-enabled-{index:03d}" for index in range(401)]
    _seed_index_map_memory_rows(authority, [
        *((memory_id, "active", "enabled", {}) for memory_id in enabled_ids),
        ("memory-manual", "active", "manual_only", {}),
        ("memory-private", "active", "disabled", {"visibility": "private"}),
        ("memory-archived", "archived", "enabled", {}),
    ])

    view = MemoryAuthorityRecallView({
        "state_dir": str(state_dir),
        "memory_authority": {"enabled": True},
    })
    index = view.memory_index_metadata_map([
        *enabled_ids, "memory-manual", "memory-private", "memory-archived",
    ])

    assert len(index) == 401
    assert "memory-enabled-400" in index
    assert index["memory-enabled-400"]["memory_id"] == "memory-enabled-400"
    assert index["memory-enabled-400"]["revision"] == 1
    assert "memory-manual" not in index
    assert "memory-private" not in index
    assert "memory-archived" not in index


@pytest.mark.parametrize(
    "setting",
    ["recent_turns", "timeout", "automatic_recall", "query_preparation", "document_preparation"],
)
def test_effective_recall_setting_change_invalidates_prepare_snapshot_key(setting):
    service = GatewayService.__new__(GatewayService)
    service.embedding_engine = SimpleNamespace(
        enabled=True,
        model="synthetic-embedding-model",
        base_url="https://embedding.invalid/v1",
        max_chars=128,
        query_instruction="synthetic query preparation",
        document_instruction="synthetic document preparation",
    )
    service.reranker_engine = SimpleNamespace(
        enabled=False, model="", base_url="", timeout=0,
        candidate_limit=0, score_weight=0,
    )
    service.diffusion_options = None
    service.model_route_mirror = None
    service.recall_recent_turns = 4
    service.recall_timeout_seconds = 15.0
    service.automatic_recall_enabled = True
    service._resolve_gateway_background_prompt = lambda _source, _scope, fallback: fallback
    service._effective_config_sha256 = ""
    service._effective_config_revision_no = 0

    payload = {
        "messages": [{"role": "user", "content": "same synthetic input"}],
        "_ombre_trace_context": {
            "conversation_id": "conversation-snapshot-test",
            "turn_id": "turn-snapshot-test",
            "request_id": "request-snapshot-test",
        },
    }
    snapshot_identity = {
        "h1": "source-user-event",
        "h2": "source-assistant-event",
        "snapshot_parent_key": "parent-snapshot-test",
    }
    before = service._prepare_snapshot_identity(
        payload, "session-snapshot-test", continuation_phase=False,
        snapshot_identity=snapshot_identity,
    )
    assert before is not None

    if setting == "recent_turns":
        service.recall_recent_turns = 5
    elif setting == "timeout":
        service.recall_timeout_seconds = 20.0
    elif setting == "automatic_recall":
        service.automatic_recall_enabled = False
    elif setting == "query_preparation":
        service.embedding_engine.query_instruction += " changed"
    else:
        service.embedding_engine.document_instruction += " changed"

    after = service._prepare_snapshot_identity(
        payload, "session-snapshot-test", continuation_phase=False,
        snapshot_identity=snapshot_identity,
    )

    assert after is not None
    assert after["effective_config_sha256"] != before["effective_config_sha256"]
    assert after["key_hash"] != before["key_hash"]


def _blocking_recall_cache_case(cache_kind):
    started = asyncio.Event()
    release = asyncio.Event()
    child_cancelled = asyncio.Event()
    calls = []

    async def blocked_result(result):
        calls.append(True)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        return result

    service = GatewayService.__new__(GatewayService)
    if cache_kind == "semantic":
        async def search_similar(_query, *, top_k):
            return await blocked_result([("memory-cache-test", 0.91)][:top_k])

        service.embedding_engine = SimpleNamespace(
            model="synthetic-embedding", base_url="https://embedding.invalid/v1",
            db_path="", search_similar=search_similar,
        )
        service._embedding_store_cache_stamp = lambda: "synthetic-store-stamp"
        service.semantic_candidate_top_k = 4
        service.embedding_query_timeout_seconds = 30.0
        service.semantic_query_cache_ttl_seconds = 300.0
        service.semantic_query_cache_max_entries = 8
        service._semantic_query_cache = {}
        service._semantic_query_inflight = {}
        invoke = lambda: service._semantic_search_cached("synthetic cached query")
        inflight = service._semantic_query_inflight
    elif cache_kind == "rerank":
        async def rerank(_query, _documents, *, top_n):
            return await blocked_result([SimpleNamespace(index=0, score=0.91)][:top_n])

        service.reranker_engine = SimpleNamespace(
            model="synthetic-reranker", base_url="https://reranker.invalid/v1",
            rerank=rerank,
        )
        service.rerank_cache_ttl_seconds = 300.0
        service.rerank_cache_max_entries = 8
        service._rerank_cache = {}
        service._rerank_inflight = {}
        invoke = lambda: service._rerank_cached(
            namespace="synthetic", query="synthetic cached query",
            documents=["synthetic candidate"], top_n=1,
        )
        inflight = service._rerank_inflight
    else:
        async def call_planner(*_args, **_kwargs):
            return await blocked_result(({"queries": [{"query": "synthetic expansion"}]}, None))

        service.query_planner_model = "synthetic-planner"
        service.query_planner_uses_dehydrator = False
        service.query_planner_min_chars = 0
        service.query_planner_max_queries = 2
        service.query_planner_max_tokens = 360
        service._internal_model_route = lambda _route: None
        service._resolve_gateway_background_prompt = lambda *_args, **_kwargs: "synthetic planner prompt"
        service._call_query_planner_uncached = call_planner
        service._query_planner_cache = {}
        service._query_planner_inflight = {}
        service.query_planner_cache_ttl_seconds = 300.0
        service.query_planner_cache_max_entries = 8
        invoke = lambda: service._call_query_planner("synthetic cached query")
        inflight = service._query_planner_inflight

    return service, invoke, started, release, child_cancelled, calls, inflight


async def _wait_for_cache_child_start(started, owner):
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        if owner.done():
            await owner
        raise


@pytest.mark.parametrize("cache_kind", ["semantic", "rerank", "planner"])
def test_recall_cache_owner_cancellation_cancels_and_awaits_provider_child(cache_kind):
    _service, invoke, started, _release, child_cancelled, calls, inflight = (
        _blocking_recall_cache_case(cache_kind)
    )

    async def run():
        owner = asyncio.create_task(invoke())
        await _wait_for_cache_child_start(started, owner)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert child_cancelled.is_set()
        assert inflight == {}

    asyncio.run(run())
    assert calls == [True]


@pytest.mark.parametrize("cache_kind", ["semantic", "rerank", "planner"])
def test_recall_cache_waiter_cancellation_does_not_cancel_shared_provider(cache_kind):
    _service, invoke, started, release, child_cancelled, calls, inflight = (
        _blocking_recall_cache_case(cache_kind)
    )

    async def run():
        owner = asyncio.create_task(invoke())
        await _wait_for_cache_child_start(started, owner)
        waiter = asyncio.create_task(invoke())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not child_cancelled.is_set()
        release.set()
        await owner
        assert not child_cancelled.is_set()
        assert inflight == {}

    asyncio.run(run())
    assert calls == [True]


@pytest.mark.parametrize("cache_kind", ["semantic", "rerank", "planner"])
def test_recall_cache_waiter_gets_provider_failure_when_owner_cancels(cache_kind):
    _service, invoke, started, _release, child_cancelled, calls, inflight = (
        _blocking_recall_cache_case(cache_kind)
    )

    async def run():
        owner = asyncio.create_task(invoke())
        await _wait_for_cache_child_start(started, owner)
        waiter = asyncio.create_task(invoke())
        await asyncio.sleep(0)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        with pytest.raises(RuntimeError, match=f"shared_{cache_kind}_request_cancelled"):
            await waiter
        assert not waiter.cancelled()
        assert child_cancelled.is_set()
        assert inflight == {}

    asyncio.run(run())
    assert calls == [True]


def test_configured_memory_recall_off_measures_zero_fact_injection_and_suppresses_legacy_sources():
    service = GatewayService.__new__(GatewayService)
    service.identity = {"ai_name": "Synthetic companion"}
    service.inject_total_budget = 10000
    blocks = []
    for order, source_id in enumerate((
        "ombre.recalled_memory",
        "ombre.targeted_memory_detail",
        "ombre.diffused_memory",
    ), start=1):
        blocks.append({
            "block_id": f"block:{source_id}",
            "source_id": source_id,
            "scope": "talk.initial",
            "stage": "ombre_post_injection",
            "mode": "live_source",
            "role": "user",
            "lane": "message",
            "anchor": "gateway.current_user_prefix",
            "order": order,
            "priority": order,
            "enabled": True,
            "owner_body": "",
            "frozen_body": "",
            "wrapper_text": "",
            "token_budget": None,
        })
    blocks.append({
        "block_id": "block:ombre.memory_recall",
        "source_id": "ombre.memory_recall",
        "scope": "talk.initial",
        "stage": "ombre_post_injection",
        "mode": "off",
        "role": "user",
        "lane": "message",
        "anchor": "gateway.current_user_prefix",
        "order": 10,
        "priority": 10,
        "enabled": True,
        "owner_body": "",
        "frozen_body": "",
        "wrapper_text": "",
        "token_budget": None,
    })
    plan = {
        "scope": "talk.initial",
        "gateway_slice": {"settings": {"scope_inheritance": {}}, "blocks": blocks},
        "binding": {"aiz_binding_revision": 1},
    }
    projection = {
        "body": "Recall status: one candidate selected",
        "fact_body": "[bucket_id:memory-selected] synthetic private fact",
        "selected_bucket_ids": ["memory-selected"],
        "recall_status": {"status": "selected", "selected_count": 1},
    }

    assert service._composer_live_recall_enabled(plan) is False
    stable, dynamic, _compiler_debug = service._build_composed_context_messages(
        plan,
        persona_block="",
        core_memory="",
        portrait_memory="",
        just_now_context="",
        recent_context="",
        recalled_memory="legacy-direct-memory",
        relationship_weather="",
        favorite_memory="",
        related_memory="legacy-diffused-memory",
        targeted_memory_detail="legacy-targeted-memory",
        dream_context="",
        active_reminders="",
        memory_detail_recall_instruction="",
        handoff_tool_hint="",
        context_mode="",
        date_persona_trace="",
        date_recall="",
        memory_recall_projection=projection,
    )
    service._reconcile_memory_recall_injection(
        projection, stable, dynamic,
        enabled=service._composer_live_recall_enabled(plan),
    )

    final_context = f"{stable}\n{dynamic}"
    assert "legacy-direct-memory" not in final_context
    assert "legacy-diffused-memory" not in final_context
    assert "legacy-targeted-memory" not in final_context
    assert "synthetic private fact" not in final_context
    assert projection["injected_bucket_ids"] == []
    assert projection["recall_status"]["status"] == "disabled"
    assert projection["recall_status"]["injected_count"] == 0
    assert projection["recall_status"]["status_only"] is True
    assert projection["recall_status"]["injection_measured_after_compile"] is True


def test_embedding_prompt_source_detail_prefers_configured_query_and_document_instructions():
    config = {
        "embedding": {
            "query_instruction": "synthetic owner query instruction",
            "document_instruction": "synthetic owner document instruction",
        },
    }

    query = source_detail("ombre.memory_embedding_query_prep_prompt", config)
    document = source_detail("ombre.memory_embedding_document_prep_prompt", config)

    assert query["live_body"] == "synthetic owner query instruction"
    assert document["live_body"] == "synthetic owner document instruction"
    assert query["factory_body"] != query["live_body"]
    assert document["factory_body"] != document["live_body"]
    assert query["source_sha256"] == hashlib.sha256(
        query["factory_body"].encode("utf-8")
    ).hexdigest()
    assert document["source_sha256"] == hashlib.sha256(
        document["factory_body"].encode("utf-8")
    ).hexdigest()


def test_missing_committed_memory_source_projection_keeps_negative_recall_incomplete():
    query = "A query with no candidate from one committed source."
    bucket = _natural_bucket("memory-projected", "A current committed test body.")
    service, _calls, _metadata = _natural_finish_service(
        None, [bucket], {query: []},
    )
    service.memory_authority_view.auto_recallable_bucket_ids = lambda: frozenset({
        "memory-projected", "memory-source-not-in-projection",
    })
    service.memory_authority_view.watermark = lambda: {
        "available": True, "memory_count": 2, "memory_revision_watermark": 3,
    }
    service.memory_authority_view.memory_revision_map = lambda _ids: {}
    service._resolve_gateway_fixed_prompt = lambda *_args, **_kwargs: ""
    recall_input = {
        "q_current": query,
        "q_context": query,
        "metadata": {"query_views": [{"kind": "current", "sha256": "a" * 64}]},
    }

    selected, suppressed, _debug = asyncio.run(service._finish_natural_selection(
        query, "session-natural-test", [bucket], recall_input=recall_input,
    ))
    projection = service._build_memory_recall_projection(
        recalled_memory="",
        targeted_memory_detail="",
        related_memory="",
        recalled_moments=[],
        targeted_memory_detail_debug={"accepted_ids": []},
        memory_sentinel_debug={"route": "ordinary", "route_reason_codes": []},
        natural_input=recall_input,
    )

    assert selected == []
    assert suppressed == []
    assert recall_input["authority_missing_source_count"] == 1
    assert "committed_memory_source_projection_unavailable" in recall_input["incomplete_reasons"]
    assert projection["recall_status"]["status"] == "incomplete"
    assert projection["recall_status"]["selected_count"] == 0
    assert projection["recall_status"]["injected_count"] == 0
    assert "committed_memory_source_projection_unavailable" in projection["recall_status"]["incomplete_reasons"]


def test_logical_trace_promotes_recall_metadata_without_message_body():
    service = GatewayService.__new__(GatewayService)
    captured = {}
    service.model_request_trace = SimpleNamespace(
        settings=lambda: {"body_visibility": "metadata_only"},
        begin_logical=lambda record: captured.update(record=record),
    )
    service._trace_from_payload = lambda _payload: {
        "trace_id": "synthetic-trace",
        "conversation_id": "synthetic-conversation",
        "turn_id": "synthetic-turn",
        "request_id": "synthetic-request",
        "logical_request_id": "synthetic-logical-request",
        "request_ordinal": 1,
        "request_type": "initial",
        "coverage": {"context_revision": 7},
        "client_id": "synthetic-client",
        "worldbook": [],
        "prompt_plan_identity": {},
    }
    service._reasoning_trace_metadata = lambda messages, *, body_visibility: {
        "visibility": body_visibility,
        "message_count": len(messages or []),
    }
    service._retrieval_runtime_debug = lambda: {"status": "synthetic"}

    recall_input = {
        "metadata": {"current_input_complete": True, "recent_turns_selected": 4},
    }
    recall_status = {
        "status": "selected", "route": "ordinary", "injected_count": 1,
    }
    query_views = [{
        "kind": "current", "sha256": "b" * 64, "semantic_used": True,
    }]
    prepare_timing = {
        "recall_input": recall_input,
        "recall_status": recall_status,
        "query_views": query_views,
        "selected_memory_ids": ["memory-synthetic"],
        "injected_count": 1,
    }
    payload = {
        "model": "synthetic-model",
        "messages": [{"role": "user"}],
        "stream": False,
    }
    service._record_logical_trace(
        payload,
        {
            "prepare_timing_debug": prepare_timing,
            "post_injection_presence": {"memory_recall": True},
            "memory_detail_recall_debug": {},
        },
        "synthetic-client-label",
    )

    metadata = captured["record"]["metadata"]
    assert metadata["recall_input"] == recall_input
    assert metadata["recall_status"] == recall_status
    assert metadata["query_views"] == query_views
    assert metadata["selected_memory_ids"] == ["memory-synthetic"]
    assert metadata["injected_count"] == 1
    assert metadata["resolved"]["message_count"] == 1
    assert metadata["resolved"]["roles"] == ["user"]
    assert "content" not in metadata["resolved"]
