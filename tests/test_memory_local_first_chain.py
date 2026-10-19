"""Offline full selectors -> real Authority/sources -> persisted Brain L2.

Only semantic-provider responses are substituted. No selector, lexical scorer,
admission, source validator, L2 resolver or prose renderer is replaced.
"""
import asyncio
import copy
import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest

from memory_l0_sources import McpDiarySourceAdapter, MemoryL0Sources
from memory_l1_reads import MemoryL1Reads
from memory_l1_schema import document_template, generation_template, scene_template, split_document, statement_template, unknown
from test_memory_l2_p4 import setup as setup_l2
from test_memory_l2_gateway_contract_p4 import _BrainReadTransport, _gateway_request, _gateway_runtime, run


QUERY = "苏州出行日期最后是哪天？"
DATE = "苏州出行日期暂定11月3日。"
SUPPORT = "票仍未购买，要等排班确定。"


def fixture(tmp_path, monkeypatch, *, enabled=True, mode="inherit", threshold=0.01, policy="enabled", persist_l2=True):
    if not os.environ.get("GUYAN_GATEWAY_WORKTREE"):
        pytest.skip("Set GUYAN_GATEWAY_WORKTREE to the exact Gateway integration checkout")
    runtime = _gateway_runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "memory_recall_runtime", runtime)
    root = Path(os.environ["GUYAN_GATEWAY_WORKTREE"])
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("_ttft_full_gateway", root / "gateway.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    import jieba
    if not hasattr(jieba, "lcut"):
        monkeypatch.setattr(jieba, "lcut", lambda text, cut_all=False: list(jieba.cut(text, cut_all=cut_all)), raising=False)

    base = tmp_path / "authority"
    l2, authority, model_calls = setup_l2(base)
    records = {day: {"id": "offline:" + day, "content": text}
               for day, text in (("2026-10-14", DATE), ("2026-10-15", SUPPORT))}
    async def read(day):
        return copy.deepcopy(records.get(day))
    sources = MemoryL0Sources(authority, [McpDiarySourceAdapter("ttft-offline", read)])
    bindings = [run(sources.capture_diary(day, binding_id=bid))["binding"]
                for day, bid in (("2026-10-14", "source:date"), ("2026-10-15", "source:limit"))]
    l2.engine._memory_l1_sources = lambda: sources
    catalog, revision = l2.revisions.catalog_provider()
    document = document_template(DATE + SUPPORT, "合成行程", catalog=catalog, catalog_revision=revision)
    document["evidence_bindings"] = bindings
    for sid, text, binding in (("date", DATE, "source:date"), ("limit", SUPPORT, "source:limit")):
        statement = statement_template()
        statement.update(statement_id=sid, statement_revision_id="r:" + sid, content=text, evidence_refs=[binding])
        statement["classification"].update(kind=unknown("known", "fact"), stability=unknown("known", "phase"),
            modality=unknown("known", "assertion"), basis=unknown("known", "record"))
        document["statements"].append(statement)
    scene = scene_template(catalog, revision)
    scene.update(scene_id="trip", title="行程和购票条件", statement_refs=["date", "limit"])
    document["scenes"] = [scene]
    document["generation"] = generation_template(catalog, revision, "first_pass_model")
    body, semantic = split_document(document)
    run(l2.revisions.commit.commit_memory(memory_id="m2", bucket_id="b2", expected_revision=0,
        body=body, metadata={"name": "合成行程", "memory_l1": semantic}, source_refs=[],
        decision_source="offline-ttft-test", idempotency_key="seed-trip", actor="test", l1_write_intent=True,
        recall_policy=policy))
    if persist_l2:
        run(l2.save("m2", {"expected_l1_revision": 1, "expected_presentation_revision": 0,
            "request_id": "persist-offline-l2", "change_kind": "wording", "document": {
                "full_injection_text": DATE + SUPPORT,
                "coherent_sections": [{"section_id": "trip", "text": DATE + SUPPORT, "statement_refs": ["date", "limit"]}],
                "optional_use_variants": []}}))
    assert model_calls == []  # Persist a fixture through owner save, never generate it.
    bucket = run(l2.revisions.buckets.get("b2"))

    config = {"buckets_dir": str(base / "buckets"), "state_dir": str(base / "state"),
        "memory_authority": {"enabled": True}, "embedding": {"enabled": False},
        "gateway": {"query_planner_enabled": False, "domain_sentinel_enabled": False,
                    "model_request_trace_path": str(base / "trace.sqlite3"),
                    "model_route_mirror_path": str(base / "routes.sqlite3"),
                    "prompt_plan_mirror_path": str(base / "prompts.sqlite3")},
        "memory_system": {"recall": {"local_first_enabled": enabled, "mode": mode, "local_threshold": threshold}}}
    gateway = module.GatewayService(config)
    gateway.prompt_plan_mirror = None
    gateway.query_planner_enabled = False
    gateway.word_map_hint_enabled = False
    gateway.reranker_engine.enabled = False
    transport = _BrainReadTransport(l2, MemoryL1Reads(l2.revisions, sources))
    original_client = httpx.AsyncClient
    def client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)
    monkeypatch.setattr(runtime.httpx, "AsyncClient", client)
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.offline.test")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "offline-test")
    calls = []
    async def semantic(request, ids):
        calls.append(set(ids))
        request["semantic_hits"] = {bid: {"index_status": "verified", "score": 0.99} for bid in ids}
        return {bid: 0.99 for bid in ids}
    monkeypatch.setattr(gateway, "_natural_semantic_candidates", semantic)
    return gateway, runtime, l2, bucket, records, transport, calls


async def select(env, *, query=QUERY, hint="fast", request=None):
    gateway, runtime, l2, bucket, records, transport, calls = env
    request = request if request is not None else _gateway_request(query)
    debug = {"route": "ordinary", "hint_route": hint}
    result = await gateway._select_recall_with_fallback(gateway._select_dynamic_buckets,
        query, "offline-ttft", [bucket], natural_input=request, sentinel_debug=debug,
        allow_semantic=True, allow_query_planner=True, allow_rerank=True, include_query_planner_debug=True)
    return result, request


async def project(env, result, request, *, budget=4000):
    gateway, runtime, l2, bucket, records, transport, calls = env
    ids = [row["id"] for row in result[0]]
    raw = "\n".join(f'[bucket_id:{bid}] {bucket["content"]}' for bid in ids)
    return await runtime.project_sections(gateway, request, request["q_context"], [bucket],
        {"direct": (raw, ids, budget), "targeted": ("", [], 0), "diffused": (raw, ids, budget)})


def test_actual_local_selector_keeps_matched_claim_and_support_in_persisted_l2(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch)
    async def scenario():
        result, request = await select(env)
        assert request["local_first"]["selection_sufficient"] is True
        assert [step["phase"] for step in result[-1]["selection_attempts"]] == ["local"]
        assert request["usage_by_bucket"]["b2"]["statement_selection"]["support_ids"] == ["limit"]
        text = await project(env, result, request)
        assert DATE in text["direct"] and SUPPORT in text["direct"]
        assert text["diffused"] == ""  # One bucket, one resolve, no duplicate injection.
        trace = request["presentation_trace"][0]
        assert trace["source"] == "l2" and trace["l1_revision"] == trace["presentation_revision"] == 1
        assert set(trace["injected_statement_ids"]) == {"date", "limit"}
        assert trace["uncovered_matched_statement_ids"] == []
        assert sum(call["path"].endswith("/resolve") for call in env[5].calls) == 1
        assert env[6] == []
    run(scenario())


@pytest.mark.parametrize("options,hint,phases,semantic_calls", [
    ({"enabled": False}, "fast", ["initial"], 1),
    ({}, "ordinary", ["initial"], 1),
    ({}, "deep", ["initial"], 1),
    ({"threshold": None}, "fast", ["initial"], 1),
    ({"mode": "fast_local"}, "fast", ["initial"], 0),
    ({"threshold": 1.0}, "fast", ["local", "initial"], 1),
])
def test_actual_routing_preserves_modes_and_only_upgrades_insufficient_local(tmp_path, monkeypatch, options, hint, phases, semantic_calls):
    env = fixture(tmp_path, monkeypatch, **options)
    result, request = run(select(env, hint=hint))
    assert [step["phase"] for step in result[-1]["selection_attempts"]] == phases
    assert len(env[6]) == semantic_calls
    assert not request.get("local_only") or options.get("mode") == "fast_local"
    assert all(task.done() for task in request.get("_tasks", []))


def test_required_source_change_prevents_local_shortcut_and_unqualified_prose(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch)
    env[4]["2026-10-15"]["content"] += " 来源已修改。"
    result, request = run(select(env))
    assert result[0] == []
    assert request["local_first"]["selection_sufficient"] is False
    assert len(env[6]) == 1
    assert request["usage_by_bucket"]["b2"]["statement_selection"]["missing_support_ids"] == ["limit"]
    assert run(project(env, result, request))["direct"] == ""


@pytest.mark.parametrize("query", ["我推荐过哪款电饭煲？", "合成行程", "你帮我买的机票几点出发？"])
def test_empty_title_only_or_correction_only_local_results_do_not_shortcut(tmp_path, monkeypatch, query):
    env = fixture(tmp_path, monkeypatch)
    result, request = run(select(env, query=query))
    assert request["local_first"]["selection_sufficient"] is False
    assert [step["phase"] for step in result[-1]["selection_attempts"]] == ["local", "initial"]
    assert len(env[6]) == 1


def test_policy_denied_memory_cannot_qualify_in_either_pass(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch, policy="manual_only")
    result, request = run(select(env))
    assert result[0] == [] and request["local_first"]["selection_sufficient"] is False
    assert all("b2" not in ids for ids in env[6])
    assert not any(call["path"].endswith("/resolve") for call in env[5].calls)


def test_missing_support_source_cannot_qualify(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch)
    env[4].pop("2026-10-15")
    result, request = run(select(env))
    assert result[0] == [] and request["local_first"]["selection_sufficient"] is False
    assert request["usage_by_bucket"]["b2"]["statement_selection"]["missing_support_ids"] == ["limit"]


def test_unready_l2_with_current_sources_uses_complete_qualified_l1(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch, persist_l2=False)
    result, request = run(select(env))
    text = run(project(env, result, request))
    assert DATE in text["direct"] and SUPPORT in text["direct"]
    assert request["presentation_trace"][0]["source"] == "l1"


@pytest.mark.parametrize("ready_l2", [True, False])
@pytest.mark.parametrize("frozen", [False, True])
def test_source_changes_after_local_selection_cannot_return_cached_l1(tmp_path, monkeypatch, ready_l2, frozen):
    env = fixture(tmp_path, monkeypatch, persist_l2=ready_l2)
    from memory_system_config import REQUEST_CONFIGURATION
    token = REQUEST_CONFIGURATION.set({"service": id(env[0]), "values": env[1].values(env[0]),
        "revision": "offline-frozen", "source_validation": {}}) if frozen else None
    try:
        result, request = run(select(env))
        assert result[0]
        if not ready_l2:
            assert env[2].detail("m2")["l2"]["state"] != "ready"
        env[4]["2026-10-15"]["content"] += " 来源在选择后变化。"
        text = run(project(env, result, request))
        assert text["direct"] == "" and text["diffused"] == ""
        assert request["presentation_trace"][0]["fallback_reason"] == "required_statement_support_not_eligible"
    finally:
        if token is not None:
            REQUEST_CONFIGURATION.reset(token)


def test_final_projection_empty_is_not_reported_as_actual_memory_use(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch)
    result, request = run(select(env))
    original = env[0]._resolve_gateway_fixed_prompt
    def prompt(name, **kwargs):
        return "" if name == "ombre.memory_usage_card_prompt" else original(name, **kwargs)
    monkeypatch.setattr(env[0], "_resolve_gateway_fixed_prompt", prompt)
    text = run(project(env, result, request))
    assert request["local_first"]["selection_sufficient"] is True  # Selection only, NOT final sufficiency.
    assert text["direct"] == "" and request["injected_usage_bucket_ids"] == []
    assert request["presentation_trace"][0]["source"] == "none"
    assert len(env[6]) == 0  # Post-projection rerouting is explicitly outside this disabled candidate.


def test_l2_units_and_budget_cannot_keep_only_support(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch)
    result, request = run(select(env))
    usage = request["usage_by_bucket"]["b2"]
    presentation = {"text": SUPPORT, "coverage": {"used_statement_ids": ["limit"]},
        "units": [{"unit_id": "support", "text": SUPPORT, "statement_refs": ["limit"]}]}
    text, reason, *_ = env[1].render_presentation(env[0], usage, "", presentation, 4000)
    assert text == "" and reason == "required_l2_context_missing"
    presentation.update(text="长原稿。" * 1000, coverage={"used_statement_ids": ["date", "limit"]},
        units=[{"unit_id": "date", "text": DATE * 1000, "statement_refs": ["date"]},
               {"unit_id": "support", "text": SUPPORT, "statement_refs": ["limit"]}])
    text, reason, *_ = env[1].render_presentation(env[0], usage, "", presentation, 120)
    assert text == "" and reason == "no_complete_correction_unit_fits_remaining_budget"


@pytest.mark.parametrize("cancel_phase", ["local", "semantic"])
def test_actual_pipeline_cancellation_cleans_up_without_a_repeated_pass(tmp_path, monkeypatch, cancel_phase):
    env = fixture(tmp_path, monkeypatch, threshold=1.0)
    async def scenario():
        entered = asyncio.Event()
        never = asyncio.Event()
        if cancel_phase == "local":
            original = env[5].reads.recall_source_status
            async def source(items):
                entered.set()
                await never.wait()
                return await original(items)
            monkeypatch.setattr(env[5].reads, "recall_source_status", source)
        else:
            async def semantic(request, ids):
                env[6].append(set(ids))
                entered.set()
                await never.wait()
            monkeypatch.setattr(env[0], "_natural_semantic_candidates", semantic)
        request = _gateway_request(QUERY)
        task = asyncio.create_task(select(env, request=request))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(child.done() and child.cancelled() for child in request.get("_tasks", []))
        assert "_completed_candidate_pool" not in request and "local_only" not in request
        assert len(env[6]) == (1 if cancel_phase == "semantic" else 0)
    run(scenario())


def test_insufficient_local_and_semantic_timeout_share_one_deadline_and_fallback(tmp_path, monkeypatch):
    env = fixture(tmp_path, monkeypatch, threshold=1.0)
    async def scenario():
        never = asyncio.Event()
        deadlines = []
        request = _gateway_request(QUERY)
        original = env[5].reads.recall_source_status
        async def source(items):
            deadlines.append(request["_deadline"])
            return await original(items)
        monkeypatch.setattr(env[5].reads, "recall_source_status", source)
        async def semantic(request, ids):
            env[6].append(set(ids))
            deadlines.append(request["_deadline"])
            await never.wait()
        monkeypatch.setattr(env[0], "_natural_semantic_candidates", semantic)
        env[0].recall_timeout_seconds = 0.2  # Isolated clock bound, never a production budget edit.
        result, request = await select(env, request=request)
        assert [step["phase"] for step in result[-1]["selection_attempts"]] == ["local", "initial", "timeout_fallback"]
        assert len(env[6]) == 1
        assert len(deadlines) >= 2 and len(set(deadlines)) == 1
        assert "recall_total_timeout" in request["incomplete_reasons"]
        assert all(task.done() for task in request.get("_tasks", []))
    run(scenario())
