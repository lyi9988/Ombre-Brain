"""Independent question coverage within the owner's existing recall budgets."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import httpx

from gateway import GatewayService
import memory_recall_runtime as memory_use
from recall_rerank_window import explicit_query_parts, select_natural_window
from test_recall_rerank_window import candidate, priority, service_with_provider
from test_recall_natural_r1 import _natural_finish_service


QUERY = "晏晏是谁？另外我对Reality聊天搜索有什么期待？"
REAL_ASYNC_CLIENT = httpx.AsyncClient


def test_only_explicit_independent_questions_are_split():
    assert explicit_query_parts(QUERY) == ["晏晏是谁？", "我对Reality聊天搜索有什么期待？"]
    for text in ["我对Reality聊天搜索有什么期待？", "晏晏是谁？她现在呢？",
                 "如果我下周出差，要带什么？回来后要做什么？",
                 "晏晏是谁？我对Reality聊天搜索有什么期待？这是引用材料。"]:
        assert explicit_query_parts(text) == [text]


def test_each_question_keeps_window_opportunity_without_more_slots_or_mutating_scores():
    pool = [candidate(f"search-{i}", keyword=.9 - i * .01) for i in range(8)]
    pool += [candidate("person", keyword=.3)]
    views = [{1: 1.0 / (61 + i)} for i in range(8)] + [{0: 1 / 61}]
    before = deepcopy(pool)
    selected, notes, debug = select_natural_window(
        pool, 6, [priority("", item) for item in pool], query_view_scores=views)
    assert len(selected) == debug["selected_count"] == 6
    assert 8 in selected
    assert notes[8]["query_part"] == 0
    assert debug["query_part_selected"] == {0: 1, 1: 5}
    assert pool == before
    assert len(select_natural_window(
        pool, 1, [priority("", item) for item in pool], query_view_scores=views)[0]) == 1


def test_independent_questions_use_one_embedding_batch_and_keep_contextual_query_when_needed():
    service = GatewayService.__new__(GatewayService)
    calls = []

    async def search(queries, **options):
        calls.append((queries, options))
        return [{"bucket_id": "person", "score": .9, "index_status": "verified",
                 "unit_id": "unit-person", "view_ranks": {"0": 1}}]

    service.embedding_engine = SimpleNamespace(enabled=True, search_similar_queries=search)
    service.semantic_candidate_top_k = 24
    request = {"q_current": QUERY, "q_context": QUERY, "metadata": {},
               "_query_parts": explicit_query_parts(QUERY), "index_metadata": {}}
    assert asyncio.run(service._natural_semantic_candidates(request, {"person"})) == {"person": .9}
    assert len(calls) == 1 and calls[0][0] == explicit_query_parts(QUERY)
    assert request["_semantic_query_parts"] is True
    contextual = "Conversation context followed by: " + QUERY
    request["q_context"] = contextual
    asyncio.run(service._natural_semantic_candidates(request, {"person"}))
    assert calls[1][0] == [contextual]
    assert request["_semantic_query_parts"] is False


def test_real_local_keyword_channel_retains_each_explicit_question(tmp_path):
    from test_memory_local_first_settings import _app, _base_config
    service, _application = _app(_base_config(tmp_path))
    service.memory_authority_enabled = False
    service.memory_authority_view.enabled = False
    buckets = [
        {"id": "person", "content": "晏晏是润润介绍的另一个AI伙伴。", "metadata": {
            "name": "晏晏这位伙伴", "type": "dynamic", "tags": ["晏晏", "伙伴"]}},
        {"id": "search", "content": "希望Reality聊天搜索区分原文和渲染文本。", "metadata": {
            "name": "Reality聊天搜索", "type": "dynamic", "tags": ["Reality", "聊天搜索"]}},
    ]
    request = {"q_current": QUERY, "q_context": QUERY, "metadata": {},
               "_query_parts": explicit_query_parts(QUERY)}
    pool, _suppressed = asyncio.run(service._dynamic_bucket_candidate_items(
        QUERY, "question-local", buckets, natural_input=request, search_query=QUERY,
        allow_semantic=False, allow_rerank=False))
    assert {row["bucket"]["id"] for row in pool} == {"person", "search"}
    assert "person" in request["_query_part_keywords"][0]
    assert "search" in request["_query_part_keywords"][1]


@pytest.mark.parametrize("failure", [False, True])
def test_question_rerank_runs_concurrently_keeps_total_document_cap_and_index_mapping(failure):
    pool = [candidate(f"search-{i}", keyword=.9 - i * .01) for i in range(8)]
    pool += [candidate("person", keyword=.3)]
    for i, item in enumerate(pool):
        item["_query_view_scores"] = {1: 1 / (61 + i)} if i < 8 else {0: 1 / 61}
    service, _unused_calls = service_with_provider(pool)
    calls, started = [], asyncio.Event()

    async def provider(**options):
        calls.append(options)
        if len(calls) == 2:
            started.set()
        await asyncio.wait_for(started.wait(), 1)
        is_person = options["query"] == "晏晏是谁？"
        options["cache_debug"].update(provider_status="error" if failure and is_person else "ok",
            provider_failed=failure and is_person, provider_requests=1)
        if failure and is_person:
            return []
        return [SimpleNamespace(index=i, score=.91 if is_person else .74)
                for i in reversed(range(len(options["documents"])))]

    service._rerank_cached = provider
    debug = {}
    output = asyncio.run(service._rerank_scored_bucket_candidates(
        QUERY, pool, natural_window=True, query_parts=explicit_query_parts(QUERY), diagnostics=debug))
    assert len(calls) == 2 and sum(len(call["documents"]) for call in calls) == 6
    assert {call["query"] for call in calls} == set(explicit_query_parts(QUERY))
    person = next(row for row in output if row["bucket"]["id"] == "person")
    assert person.get("rerank_score") == (None if failure else .91)
    assert person["_rerank_window"]["status"] == ("provider_failed" if failure else "scored")
    assert debug["provider_failed"] is failure
    assert len(output) == len(pool)
    assert all(row.get("rerank_score") is None for row in output
               if not row["_rerank_window"]["selected"])


def test_question_rerank_cancellation_awaits_all_provider_children():
    pool = [candidate("person", keyword=.3), candidate("search", keyword=.9)]
    for i, item in enumerate(pool):
        item["_query_view_scores"] = {i: 1}
    service, _ = service_with_provider(pool)
    starts, cancellations = [], []

    async def provider(**options):
        starts.append(options["query"])
        try:
            await asyncio.Event().wait()
        finally:
            cancellations.append(options["query"])

    service._rerank_cached = provider

    async def scenario():
        task = asyncio.create_task(service._rerank_scored_bucket_candidates(
            QUERY, pool, natural_window=True, query_parts=explicit_query_parts(QUERY)))
        while len(starts) < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(cancellations) == 2
        assert not [pending for pending in asyncio.all_tasks()
                    if pending is not asyncio.current_task() and not pending.done()]

    asyncio.run(scenario())


@pytest.mark.parametrize("prefix", [False, True])
def test_question_rerank_keeps_historical_dialogue_conditions(prefix):
    parts = explicit_query_parts(QUERY)
    context = "Recent dialogue: only the 2024 Tokyo trip, not the current trip."
    contextual = QUERY + "\n\n" + context if prefix else context + "\n" + QUERY
    pool = [candidate("person", keyword=.4), candidate("search", keyword=.9)]
    for index, item in enumerate(pool):
        item["_query_view_scores"] = {index: 1}
    service, calls = service_with_provider(pool)
    asyncio.run(service._rerank_scored_bucket_candidates(contextual, pool,
        natural_window=True, query_parts=parts, current_query=QUERY))
    assert len(calls) == 2
    for part, call in zip(parts, calls):
        assert context in call["query"] and part in call["query"]
    assert sum(len(call["documents"]) for call in calls) == 2


@pytest.mark.parametrize("query", ["An ordinary question about the evening?", QUERY])
def test_graph_moment_lists_cannot_overwrite_query_parts(query, monkeypatch):
    import gateway as gateway_module
    bucket = candidate("moment-parent", semantic=.9)["bucket"]
    moments = [{"moment_id": f"moment-{i}", "bucket_id": bucket["id"],
        "source": "content", "score": .9, "content": f"A verified passage {i}."} for i in range(2)]
    service, _calls, _metadata = _natural_finish_service(None, [bucket], {"*": []}, rerank_score=.95)
    monkeypatch.setattr(gateway_module, "parse_bucket_moments", lambda *_: deepcopy(moments))
    service.memory_moment_store = SimpleNamespace(search_moment_items=lambda *_a, **_k: deepcopy(moments))
    service._moment_rerank_document = lambda moment: moment["content"]
    service._direct_moments_for_bucket = lambda *_: [dict(moments[0])]
    service._moment_with_bucket_recall_signal = lambda moment, _signal: dict(moment)
    service._bucket_candidate_recall_signal = lambda item: {"bucket_id": item["bucket"]["id"]}
    original = service._rerank_scored_bucket_candidates
    options_seen = []

    async def rerank(*args, **options):
        options_seen.append(options.get("query_parts"))
        options.pop("query_parts", None)
        options.pop("current_query", None)
        return await original(*args, **options)

    service._rerank_scored_bucket_candidates = rerank
    result = asyncio.run(service._finish_natural_selection(query, "moment-question", [bucket],
        recall_input={"q_current": query, "q_context": query, "metadata": {}}, grouped_moments={}))
    assert result[-1]["final_bucket_ids"] == [bucket["id"]]
    assert options_seen == ([explicit_query_parts(query)] if query == QUERY else [None])


def test_card_selection_preserves_admitted_question_coverage_with_existing_owner_limit():
    service = GatewayService.__new__(GatewayService)
    service.inject_max_cards = 2
    service.first_card_min_score = .5
    service._dynamic_bucket_item_has_reliable_recall_signal = lambda *_: False
    search_one, search_two, person = (candidate("search-one"), candidate("search-two"), candidate("person"))
    for item, score, part in [(search_one, .9, 1), (search_two, .85, 1), (person, .7, 0)]:
        item.update(score=score, _rerank_window={"query_part": part})
    service._pick_dynamic_cards_by_score = lambda *_a, **_k: [search_one, search_two]
    assert service._pick_dynamic_cards([search_one, search_two, person], query=QUERY) == [search_one, person]
    service.inject_max_cards = 1
    service._pick_dynamic_cards_by_score = lambda *_a, **_k: [search_one]
    assert service._pick_dynamic_cards([search_one, person], query=QUERY) == [search_one]


@pytest.mark.parametrize("strong", [False, True])
def test_natural_question_windows_keep_existing_admission_and_trace_missing_parts(strong):
    pool = [candidate(f"search-{i}", keyword=.9 - i * .01) for i in range(8)]
    pool += [candidate("person", keyword=.3)]
    service, _, _ = _natural_finish_service(None, [item["bucket"] for item in pool], {"*": pool})
    service.reranker_engine = SimpleNamespace(enabled=True, candidate_limit=6, score_weight=.65)
    original_builder = service._dynamic_bucket_candidate_items

    async def builder(*args, **options):
        options["natural_input"]["_query_part_keywords"] = [
            {"person": 1}, {f"search-{i}": i + 1 for i in range(8)}]
        return await original_builder(*args, **options)

    service._dynamic_bucket_candidate_items = builder
    real, _ = service_with_provider(pool)
    service._rerank_scored_bucket_candidates = real._rerank_scored_bucket_candidates
    calls = []

    async def provider(**options):
        calls.append(options)
        options["cache_debug"].update(provider_status="ok", provider_failed=False, provider_requests=1)
        return [SimpleNamespace(index=i, score=.95 if strong else .1)
                for i in range(len(options["documents"]))]

    real._rerank_cached = provider
    request = {"q_current": QUERY, "q_context": QUERY, "metadata": {}}
    selected, _suppressed, debug = asyncio.run(service._finish_natural_selection(
        QUERY, "question-session", [item["bucket"] for item in pool], recall_input=request))
    assert sum(len(call["documents"]) for call in calls) == 6
    if strong:
        assert "person" in debug["final_bucket_ids"]
        assert all(part["selected_bucket_ids"] for part in debug["query_part_coverage"])
    else:
        assert not selected
        assert all(not part["selected_bucket_ids"] for part in debug["query_part_coverage"])
    assert all(part["state"] == "selected_candidate_evidence" for part in debug["query_part_coverage"])
    assert service.recall_policy.rerank_threshold == service.recall_policy.semantic_threshold == .8


def test_selected_questions_keep_source_validation_and_corresponding_l2_in_composer(tmp_path, monkeypatch):
    from test_memory_recall_projection_r2 import _fixture
    from test_memory_recall_gateway_r2 import _commit_l1, _install_source_http
    from test_memory_readable_injection_contract import _composer_plan, _compose, _projection

    authority, service, _view, _old_buckets, _old_plan = _fixture(tmp_path, monkeypatch)
    prose = {"person": "主人介绍晏晏是一个AI伙伴。",
             "search": "主人希望Reality聊天搜索区分原文与渲染文本。"}
    buckets, pool = [], []
    for index, (bid, text) in enumerate(prose.items()):
        _commit_l1(authority, bid, text, topic=bid, current=text, old="此前还有其他合成背景。")
        bucket = {"id": bid, "content": text, "metadata": {"name": bid, "type": "dynamic"}}
        buckets.append(bucket)
        pool.append(dict(candidate(bid, keyword=.7), bucket=bucket))
    service.config["memory_system"]["recall"].update(mode="inherit", local_threshold=None,
        semantic_threshold=.8, rerank_threshold=.8)
    service.inject_max_cards = 2

    async def candidates(*_args, **options):
        options["natural_input"]["_query_part_keywords"] = [{"person": 1}, {"search": 1}]
        return deepcopy(pool), []

    real, _ = service_with_provider(pool)

    async def provider(**options):
        options["cache_debug"].update(provider_status="ok", provider_failed=False, provider_requests=1)
        return [SimpleNamespace(index=i, score=.95) for i in range(len(options["documents"]))]

    real._rerank_cached = provider
    service._dynamic_bucket_candidate_items = candidates
    service._rerank_scored_bucket_candidates = real._rerank_scored_bucket_candidates
    monkeypatch.setattr(memory_use.httpx, "AsyncClient", REAL_ASYNC_CLIENT)
    source_calls = _install_source_http(monkeypatch, presentations={
        f"memory:{bid}": {"state": "ready", "text": text, "presentation_revision": 7, "reason": None}
        for bid, text in prose.items()})
    request = {"q_current": QUERY, "q_context": QUERY, "metadata": {}}
    selected, _suppressed, _debug = asyncio.run(service._finish_natural_selection(
        QUERY, "two-question-source", buckets, recall_input=request))
    ids = [bucket["id"] for bucket in selected]
    assert set(ids) == set(prose)
    plan = _composer_plan(service, tmp_path)
    rendered = asyncio.run(memory_use.project_sections(service, request, QUERY, buckets, {
        "direct": ("\n".join(f"[bucket_id:{bid}]" for bid in ids), ids, 4000),
        "recent": ("", [], 0), "targeted": ("", [], 0), "diffused": ("", [], 0)}))
    assert len(source_calls) == 2  # Initial check and mandatory final refresh.
    assert len(source_calls.presentation_requests) == 1
    assert set(request["injected_usage_bucket_ids"]) == set(prose)
    assert all(row["source"] == "l2" and row["presentation_revision"] == 7
               for row in request["presentation_trace"])
    projection = _projection(service, request, rendered)
    messages, *_rest = _compose(service, plan, projection, rendered)
    actual = "\n".join(str(message.get("content") or "") for message in messages)
    assert all(text in actual for text in prose.values())
