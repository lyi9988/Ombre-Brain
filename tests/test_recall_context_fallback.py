"""Focused contracts for contextual Recall queries and bounded Fast-to-Deep fallback."""

from __future__ import annotations

import asyncio

import pytest

from gateway import GatewayService


@pytest.mark.parametrize(
    ("prior_user", "current_user"),
    [
        ("我上周和晏晏讨论了夏天去海边。", "她后来去哪了？"),
        ("I talked with Lily about the gallery opening.", "What did she decide?"),
        ("我昨天说了手机坏了，约好今天去修。", "那件事后来呢？"),
    ],
)
def test_contextual_recall_query_uses_recent_user_context_without_editing_messages(
    prior_user, current_user
):
    service = GatewayService.__new__(GatewayService)
    messages = [
        {"role": "user", "content": prior_user},
        {"role": "assistant", "content": "我记得你提过。"},
        {"role": "user", "content": current_user},
    ]
    original_messages = [dict(message) for message in messages]

    contextual, debug = service._contextual_recall_query(current_user, messages)

    assert contextual == f"{prior_user}\n{current_user}"
    assert debug["used"] is True
    assert debug["source_message_indices"] == [0]
    assert messages == original_messages


def test_fast_result_does_not_start_deep_retry():
    service = GatewayService.__new__(GatewayService)
    calls = []
    sentinel = {"route": "fast", "route_reason_codes": ["FAST_LOCAL_TOPIC"]}

    async def selector(query, *_args, **kwargs):
        calls.append((query, kwargs))
        return ([{"id": "bucket-1"}], [], {"candidate_stages": []})

    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector,
        "晏晏的计划",
        "session-1",
        [],
        sentinel_debug=sentinel,
        allow_semantic=False,
        allow_query_planner=False,
        allow_rerank=False,
    ))

    assert [item["id"] for item in selected] == ["bucket-1"]
    assert suppressed == []
    assert len(calls) == 1
    assert calls[0][1]["allow_semantic"] is False
    assert sentinel["deep_fallback"] == {"attempted": False, "reason": "fast_has_result"}


def test_fast_empty_runs_exactly_one_deep_pass_and_enables_graph_bucket_rerank():
    service = GatewayService.__new__(GatewayService)
    service.gateway_cfg = {"recall_deep_fallback_timeout_seconds": 2}
    service._authority_auto_recall_allowed = lambda _bucket: True
    service._query_has_explicit_recall_marker = lambda _query: True
    service._query_looks_emotional_reason_lookup = lambda _query: False
    service._query_has_identity_name_intent = lambda _query: False
    service._rejected_bucket_evidence = lambda items: {"count": len(items)}
    calls = []
    sentinel = {"route": "fast", "route_reason_codes": ["FAST_LOCAL_DEFAULT"]}
    fast_debug = {"candidate_stages": [{"stage": "fast"}], "timing_ms": {}}
    deep_debug = {"candidate_stages": [{"stage": "deep"}], "timing_ms": {}}
    rejected = [{
        "bucket": {"id": "bucket-weak", "metadata": {}},
        "admission_reason": "no_hard_evidence",
    }]

    async def selector(query, *_args, **kwargs):
        calls.append((query, kwargs))
        if len(calls) == 1:
            return ([], rejected, fast_debug)
        return ([{"id": "bucket-1"}], [], deep_debug)

    selected, suppressed, debug = asyncio.run(service._select_recall_with_fallback(
        selector,
        "你还记得我们聊过的计划吗？",
        "session-1",
        [],
        sentinel_debug=sentinel,
        allow_semantic=False,
        allow_query_planner=False,
        allow_rerank=False,
        allow_bucket_rerank=False,
    ))

    assert len(calls) == 2
    assert calls[0][1]["allow_semantic"] is False
    assert calls[1][1]["allow_semantic"] is True
    assert calls[1][1]["allow_query_planner"] is True
    assert calls[1][1]["allow_rerank"] is True
    assert calls[1][1]["allow_bucket_rerank"] is True
    assert [item["id"] for item in selected] == ["bucket-1"]
    assert suppressed == []
    assert sentinel["route"] == "deep"
    assert sentinel["initial_route"] == "fast"
    assert sentinel["route_reason_codes"][-1] == "DEEP_AFTER_FAST_EMPTY"
    assert debug["fast_pass"]["rejections"] == {"count": 1}


def test_dynamic_bucket_selector_cancels_and_awaits_parallel_planner_on_cancel():
    service = GatewayService.__new__(GatewayService)
    planner_started = asyncio.Event()
    planner_cancelled = asyncio.Event()

    async def planner():
        planner_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            planner_cancelled.set()

    async def selection_impl(*_args, _planner_tasks, **_kwargs):
        _planner_tasks.append(asyncio.create_task(planner()))
        await planner_started.wait()
        await asyncio.Event().wait()

    service._select_dynamic_buckets_impl = selection_impl

    async def scenario():
        task = asyncio.create_task(service._select_dynamic_buckets("query", "session", []))
        await planner_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert planner_cancelled.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("prior", [
    [{"role": "assistant", "content": "她叫小满。"}],
    [{"role": "tool", "content": "她叫小满。"}],
    [{"role": "system", "content": "她叫小满。"}],
    [{"role": "user", "content": "小满去了海边"}, {"role": "user", "content": "ok"}],
    [{"role": "user", "content": "很长的前文" * 120}],
])
def test_no_antecedent_from_other_roles_or_past_acknowledgement(prior):
    service = GatewayService.__new__(GatewayService)
    query = "她后来去哪了？"
    actual, debug = service._contextual_recall_query(query, prior + [{"role": "user", "content": query}])
    assert actual == query
    assert debug["used"] is False


def test_standalone_query_does_not_inherit_previous_topic():
    service = GatewayService.__new__(GatewayService)
    query = "查一下今天的天气"
    actual, debug = service._contextual_recall_query(query, [
        {"role": "user", "content": "上次去看海"},
        {"role": "user", "content": query},
    ])
    assert actual == query
    assert debug["used"] is False


def fallback_service():
    service = GatewayService.__new__(GatewayService)
    service._query_has_explicit_recall_marker = lambda _q: False
    service._query_looks_emotional_reason_lookup = lambda _q: False
    service._query_has_identity_name_intent = lambda _q: False
    return service


def test_fast_policy_denial_does_not_trigger_semantic_retry():
    from types import SimpleNamespace
    service = fallback_service()
    service.memory_authority_view = SimpleNamespace(
        enabled=True, auto_recallable_bucket_ids=lambda: frozenset({"public"}),
    )
    calls = []
    sentinel = {"route": "fast"}

    async def selector(*_args, **kwargs):
        calls.append(kwargs)
        return [], [{"bucket": {"id": "private"}, "admission_reason": "no_hard_evidence"}], {}

    asyncio.run(service._select_recall_with_fallback(
        selector, "一般话题", sentinel_debug=sentinel,
    ))
    assert len(calls) == 1
    assert sentinel["deep_fallback"]["reason"] == "no_eligible_memory_signal"


def test_empty_deep_result_does_not_loop():
    service = fallback_service()
    calls = []
    sentinel = {"route": "fast", "query_context": {"used": True}}

    async def selector(*_args, **kwargs):
        calls.append(kwargs)
        return [], [], {}

    result = asyncio.run(service._select_recall_with_fallback(
        selector, "那件事呢", sentinel_debug=sentinel,
    ))
    assert result[0] == []
    assert len(calls) == 2
    assert sentinel["deep_fallback"]["reason"] == "deep_empty"


def test_initial_deep_graph_gets_rerank_evidence_before_admission():
    service = fallback_service()
    calls = []

    async def selector(*_args, **kwargs):
        calls.append(kwargs)
        return [], [], {}

    asyncio.run(service._select_recall_with_fallback(
        selector, "以前那件事", sentinel_debug={"route": "deep"},
        allow_rerank=True, allow_bucket_rerank=False,
    ))
    assert len(calls) == 1
    assert calls[0]["allow_bucket_rerank"] is True


def test_fallback_deadline_cancels_inflight_selection():
    service = fallback_service()
    service.gateway_cfg = {"recall_deep_fallback_timeout_seconds": 1}
    sentinel = {"route": "fast", "query_context": {"used": True}}
    cancelled = []

    async def selector(*_args, **kwargs):
        if not kwargs.get("allow_semantic"):
            return [], [], {}
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    result = asyncio.run(service._select_recall_with_fallback(
        selector, "那件事呢", sentinel_debug=sentinel,
    ))
    assert result[0] == []
    assert cancelled == [True]
    assert sentinel["deep_fallback"]["reason"] == "deep_timeout"
