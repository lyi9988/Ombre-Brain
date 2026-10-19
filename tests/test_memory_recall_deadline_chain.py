"""Deadline and request-scoped I/O checks for the real local-first recall chain.

The base fixture keeps the real Gateway selector, Brain Authority, L0 source
validation and persisted L2 resolver. Only its HTTP transport is offline.
"""
import asyncio
import copy
import hashlib
import time

import pytest

from memory_l1_schema import (
    document_template,
    generation_template,
    scene_template,
    split_document,
    statement_template,
    unknown,
)
from test_memory_local_first_chain import (
    DATE,
    QUERY,
    SUPPORT,
    fixture as chain_fixture,
    project as chain_project,
    run,
    select as chain_select,
)
from test_memory_l2_gateway_contract_p4 import _gateway_request


SOURCE_PATH = "/api/memory-authority/recall-sources"
RESOLVE_PATH = "/api/memory-authority/presentations/resolve"


async def _select_with_natural_context(env, query=QUERY):
    gateway, runtime, _l2, bucket, _records, _transport, _semantic_calls = env
    request = _gateway_request(query)
    request.pop("_use_context", None)
    runtime.context(gateway, request, query)
    debug = {"route": "ordinary", "hint_route": "fast"}
    result = await gateway._select_recall_with_fallback(
        gateway._select_dynamic_buckets, query, "offline-ttft", [bucket],
        natural_input=request, sentinel_debug=debug,
        allow_semantic=True, allow_query_planner=True, allow_rerank=True,
        include_query_planner_debug=True,
    )
    return result, request


def _path_count(env, suffix):
    return sum(call["path"].endswith(suffix) for call in env[5].calls)


async def _wait_until(predicate, timeout=1.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0)


def test_real_selector_source_and_l2_chain_records_timing_and_http_counts(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        started = time.perf_counter()
        result, request = await chain_select(env)
        selector_ms = max(0, int((time.perf_counter() - started) * 1000))
        deadline = request["_deadline"]

        projection_started = time.perf_counter()
        rendered = await chain_project(env, result, request)
        projection_wall_ms = max(0, int((time.perf_counter() - projection_started) * 1000))

        assert DATE in rendered["direct"] and SUPPORT in rendered["direct"]
        assert request["presentation_trace"][0]["source"] == "l2"
        assert request["_deadline"] == deadline
        stats = request["recall_io"]
        source_http = _path_count(env, SOURCE_PATH)
        resolver_http = _path_count(env, RESOLVE_PATH)
        assert stats["sources"]["calls"] == source_http
        assert stats["l2_resolve"]["calls"] == resolver_http == 1
        assert stats["sources"]["duration_ms"] >= 0
        assert stats["l2_resolve"]["duration_ms"] >= 0
        assert stats["projection_ms"] >= 0
        assert stats["remaining_ms"] >= 0

        print(
            "RECALL_STAGE_EVIDENCE "
            f"selector_ms={selector_ms} projection_wall_ms={projection_wall_ms} "
            f"source_http={source_http} source_ms={stats['sources']['duration_ms']} "
            f"l2_http={resolver_http} l2_ms={stats['l2_resolve']['duration_ms']} "
            f"projection_ms={stats['projection_ms']} remaining_ms={stats['remaining_ms']}"
        )

    run(scenario())


def test_one_question_mark_with_two_conjoined_question_axes_is_not_local_shortcut(
    tmp_path, monkeypatch,
):
    env = chain_fixture(tmp_path, monkeypatch)
    query = "苏州哪天走，咖啡为什么暂停？"
    result, request = run(chain_select(env, query=query))

    assert request["local_first"]["selection_sufficient"] is False
    assert request["local_first"]["query_coverage"] == "multiple_parts_unverified"
    assert [attempt["phase"] for attempt in result[-1]["selection_attempts"]] == ["local", "initial"]
    assert len(env[6]) == 1


def test_same_request_duplicate_final_source_checks_share_one_http_call(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await chain_select(env)
        before = _path_count(env, SOURCE_PATH)
        original = env[5].handle_async_request
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_response(http_request):
            response = await original(http_request)
            if http_request.url.path.endswith("/recall-sources"):
                entered.set()
                await release.wait()
            return response

        monkeypatch.setattr(env[5], "handle_async_request", hold_response)
        candidates = [{"bucket": bucket, "score": 0.0} for bucket in result[0]]
        first = asyncio.create_task(env[1].verify_sources(
            env[0], request, QUERY, candidates, refresh=True,
        ))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(env[1].verify_sources(
            env[0], request, QUERY, candidates, refresh=True,
        ))
        await _wait_until(lambda: request["recall_io"]["sources"]["singleflight_hits"] >= 1)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert len(first_result) == len(second_result) == 1
        assert _path_count(env, SOURCE_PATH) - before == 1
        assert request["recall_io"]["sources"]["singleflight_hits"] >= 1

    run(scenario())


def test_cancelling_singleflight_waiter_keeps_source_owner_and_rpc_alive(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await chain_select(env)
        candidates = [{"bucket": bucket, "score": 0.0} for bucket in result[0]]
        before = _path_count(env, SOURCE_PATH)
        original = env[5].handle_async_request
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_response(http_request):
            response = await original(http_request)
            if http_request.url.path.endswith("/recall-sources"):
                entered.set()
                await release.wait()
            return response

        monkeypatch.setattr(env[5], "handle_async_request", hold_response)
        owner = asyncio.create_task(env[1].verify_sources(
            env[0], request, QUERY, candidates, refresh=True,
        ))
        await asyncio.wait_for(entered.wait(), timeout=1)
        waiter = asyncio.create_task(env[1].verify_sources(
            env[0], request, QUERY, candidates, refresh=True,
        ))
        await _wait_until(lambda: request["recall_io"]["sources"]["singleflight_hits"] >= 1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert owner.done() is False
        assert request.get("_source_check_cancelled") is not True
        assert request.get("_source_owner_cancelled") is not True
        release.set()
        permitted = await owner

        assert len(permitted) == 1
        assert owner.done() and not owner.cancelled()
        assert _path_count(env, SOURCE_PATH) - before == 1
        assert request["recall_io"]["sources"]["calls"] == _path_count(env, SOURCE_PATH)
        assert request.get("_recall_reads") == {}

    run(scenario())


def test_l2_batch_deduplicates_shared_source_bindings_across_memories(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await chain_select(env)
        detail = await env[2].revisions.detail("m2")
        body, semantic = split_document(detail["l1"]["document"])
        await env[2].revisions.commit.commit_memory(
            memory_id="m3", bucket_id="b3", expected_revision=0,
            body=body, metadata={"name": "共享来源行程", "memory_l1": semantic},
            source_refs=[], decision_source="deadline-chain-shared-source",
            idempotency_key="deadline-chain-seed-m3", actor="test",
            l1_write_intent=True, recall_policy="enabled",
        )
        await env[2].save("m3", {
            "expected_l1_revision": 1,
            "expected_presentation_revision": 0,
            "request_id": "deadline-chain-save-m3",
            "change_kind": "wording",
            "document": {
                "full_injection_text": DATE + SUPPORT,
                "coherent_sections": [{
                    "section_id": "trip", "text": DATE + SUPPORT,
                    "statement_refs": ["date", "limit"],
                }],
                "optional_use_variants": [],
            },
        })

        usage = request["usage_by_bucket"]["b2"]
        candidates = [
            {"bucket_id": "b2", "memory_id": "m2", "revision": 1,
             "usage": copy.deepcopy(usage)},
            {"bucket_id": "b3", "memory_id": "m3", "revision": 1,
             "usage": copy.deepcopy(usage)},
        ]
        sources = env[5].reads.sources
        original_locate = sources.locate
        locate_calls = []
        active = 0
        max_active = 0
        both_entered = asyncio.Event()
        release = asyncio.Event()

        async def counted_locate(binding, *, start=None, end=None, verify_only=False):
            nonlocal active, max_active
            locate_calls.append(binding["binding_id"])
            active += 1
            max_active = max(max_active, active)
            if len(locate_calls) >= 2:
                both_entered.set()
            try:
                await release.wait()
                return await original_locate(
                    binding, start=start, end=end, verify_only=verify_only,
                )
            finally:
                active -= 1

        monkeypatch.setattr(sources, "locate", counted_locate)
        before = _path_count(env, RESOLVE_PATH)
        started = time.perf_counter()
        task = asyncio.create_task(env[1].resolve_presentations(
            env[0], candidates, request=request,
        ))
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        release.set()
        presentations, trace = await task
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))

        assert len(locate_calls) == 2
        assert set(locate_calls) == {"source:date", "source:limit"}
        assert max_active == 2
        assert _path_count(env, RESOLVE_PATH) - before == 1
        assert set(presentations) == {"b2", "b3"}
        assert all(presentations[bid]["resolver_state"] == "ready" for bid in ("b2", "b3"))
        assert all(trace[bid]["resolver_state"] == "ready" for bid in ("b2", "b3"))
        assert request["recall_io"]["l2_resolve"]["calls"] == _path_count(env, RESOLVE_PATH)
        print(
            "L2_BATCH_EVIDENCE "
            f"memories=2 unique_source_reads={len(locate_calls)} "
            f"max_parallel_source_reads={max_active} resolver_http_delta=1 elapsed_ms={elapsed_ms}"
        )

    run(scenario())


@pytest.mark.parametrize("change", ["source", "revision", "permission"])
def test_final_source_refresh_invalidates_same_request_qualification(tmp_path, monkeypatch, change):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await chain_select(env)
        assert result[0]
        sources_before = _path_count(env, SOURCE_PATH)

        if change == "source":
            env[4]["2026-10-15"]["content"] += " 来源在最终核验前变化。"
        elif change == "revision":
            detail = await env[2].revisions.detail("m2")
            document = copy.deepcopy(detail["l1"]["document"])
            document["narrative"]["body"] += " 版本修订。"
            body, semantic = split_document(document)
            await env[2].revisions.commit.commit_memory(
                memory_id="m2", bucket_id="b2", expected_revision=1,
                body=body, metadata={"name": "合成行程", "memory_l1": semantic},
                source_refs=[], decision_source="deadline-chain-revision-change",
                idempotency_key="deadline-chain-bump-l1", actor="test",
                l1_write_intent=True, recall_policy="enabled",
            )
            assert env[2].revisions.authority.get_memory("m2")["active_revision"] == 2
        else:
            env[2].revisions.authority.change_memory_state(
                memory_id="m2", state="active", recall_policy="disabled",
                idempotency_key="deadline-chain-disable-recall", actor="test",
                expected_revision=1, expected_state="active",
            )

        rendered = await chain_project(env, result, request)
        if change == "source":
            assert _path_count(env, SOURCE_PATH) == sources_before + 1
        elif change == "revision":
            view = env[0].memory_authority_view
            metadata = view.memory_index_metadata_map(["b2"]).get("b2")
            current_record = view.memory_l1_recall_map(["b2"]).get("b2")
            current_bucket = await env[2].revisions.buckets.get("b2")
            old_body_sha256 = hashlib.sha256(str(env[3].get("content") or "").encode()).hexdigest()
            current_body_sha256 = hashlib.sha256(
                str((current_bucket or {}).get("content") or "").encode()).hexdigest()
            assert env[2].revisions.authority.get_memory("m2")["active_revision"] == 2
            assert request["_semantic_records"]["b2"]["revision"] == 1
            assert current_record and current_record["state"] == "l1"
            assert current_record["memory_id"] == "m2" and current_record["revision"] == 2
            assert current_record.get("document")
            assert metadata and metadata["memory_id"] == "m2" and metadata["revision"] == 2
            assert metadata["body_sha256"] == current_body_sha256 != old_body_sha256
        assert rendered["direct"] == "" and rendered["diffused"] == ""
        assert request["presentation_trace"][0]["source"] == "none"
        assert _path_count(env, RESOLVE_PATH) == 0

    run(scenario())


@pytest.mark.parametrize("terminal", ["timeout", "cancel"])
def test_slow_final_source_is_deadline_bounded_and_cancellable(tmp_path, monkeypatch, terminal):
    env = chain_fixture(tmp_path, monkeypatch)
    env[0].recall_timeout_seconds = 4.0 if terminal == "timeout" else 5.0

    async def scenario():
        result, request = await chain_select(env)
        before = _path_count(env, SOURCE_PATH)
        target_call = before + 1
        source_calls_seen = before
        entered = asyncio.Event()
        release = asyncio.Event()
        original = env[5].handle_async_request

        async def delay_final_source(http_request):
            nonlocal source_calls_seen
            response = await original(http_request)
            if http_request.url.path.endswith("/recall-sources"):
                source_calls_seen += 1
                if source_calls_seen == target_call:
                    entered.set()
                    await release.wait()
            return response

        monkeypatch.setattr(env[5], "handle_async_request", delay_final_source)
        projection = asyncio.create_task(chain_project(env, result, request))
        await asyncio.wait_for(entered.wait(), timeout=2)

        if terminal == "timeout":
            rendered = await asyncio.wait_for(projection, timeout=5)
            assert rendered["direct"] == ""
            assert "final_source_deadline" in request["incomplete_reasons"]
            assert request["recall_io"]["sources"]["timeouts"] >= 1
            assert request["recall_io"]["remaining_ms"] == 0
        else:
            projection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await projection
            assert request.get("_source_check_cancelled") is True
            assert request["_deadline"] > 0

        assert _path_count(env, SOURCE_PATH) == before + 1
        assert _path_count(env, RESOLVE_PATH) == 0
        release.set()

    run(scenario())


def test_lost_local_candidate_gets_one_semantic_recovery_without_memory_write_or_generation(
    tmp_path, monkeypatch,
):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await chain_select(env)
        assert request["local_first"]["selection_sufficient"] is True
        semantic_calls_before = len(env[6])
        env[4]["2026-10-15"]["content"] += " 来源在投影前变化。"

        ids = [row["id"] for row in result[0]]
        raw = "\n".join(f'[bucket_id:{bid}] {env[3]["content"]}' for bid in ids)
        sections = {
            "direct": (raw, ids, 4000),
            "targeted": ("", [], 0),
            "diffused": (raw, ids, 4000),
        }

        def forbidden(*_args, **_kwargs):
            raise AssertionError("projection recovery must not write or generate L2")

        monkeypatch.setattr(env[2], "save", forbidden)
        monkeypatch.setattr(env[2], "run_once", forbidden)
        rendered = await env[0]._project_recall_with_recovery(
            request, QUERY, [env[3]], sections,
        )

        assert rendered["direct"] == "" and rendered["diffused"] == ""
        assert request["projection_recovery"]["attempted"] is True
        assert request["projection_recovery"]["reason"] == "no_qualified_replacement"
        assert request["projection_recovery"]["rejected_bucket_ids"] == ["b2"]
        assert len(env[6]) == semantic_calls_before + 1
        assert request["projection_recovery"]["selection_timing_ms"] >= 0
        assert isinstance(request["projection_recovery"]["candidate_stages"], list)
        assert request["projection_recovery"]["selected_bucket_ids"] == []
        assert _path_count(env, RESOLVE_PATH) == 0
        assert all(call["path"].endswith(SOURCE_PATH) for call in env[5].calls)

    run(scenario())


def test_recovery_selects_and_projects_second_source_backed_l2_only(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        later_date = "苏州出行后来改为12月6日。"
        later_support = "后续车次已改签，等新班次确认。"
        env[4]["2026-10-13"] = {
            "id": "offline:later-trip", "content": later_date + later_support,
        }
        sources = env[5].reads.sources
        captured = await sources.capture_diary("2026-10-13", binding_id="source:later-trip")
        catalog, catalog_revision = env[2].revisions.catalog_provider()
        document = document_template(
            later_date + later_support, "苏州后续行程",
            catalog=catalog, catalog_revision=catalog_revision,
        )
        document["evidence_bindings"] = [captured["binding"]]
        for sid, text in (("later-date", later_date), ("later-limit", later_support)):
            statement = statement_template()
            statement.update(
                statement_id=sid, statement_revision_id="r:" + sid,
                content=text, evidence_refs=["source:later-trip"],
            )
            statement["classification"].update(
                kind=unknown("known", "fact"), stability=unknown("known", "phase"),
                modality=unknown("known", "assertion"), basis=unknown("known", "record"),
            )
            document["statements"].append(statement)
        scene = scene_template(catalog, catalog_revision)
        scene.update(
            scene_id="later-trip", title="苏州后续行程",
            statement_refs=["later-date", "later-limit"],
        )
        document["scenes"] = [scene]
        document["generation"] = generation_template(catalog, catalog_revision, "first_pass_model")
        body, semantic = split_document(document)
        await env[2].revisions.commit.commit_memory(
            memory_id="m3", bucket_id="b3", expected_revision=0,
            body=body, metadata={"name": "苏州后续行程", "memory_l1": semantic},
            source_refs=[], decision_source="deadline-chain-recovery-success",
            idempotency_key="deadline-chain-recovery-seed-m3", actor="test",
            l1_write_intent=True, recall_policy="enabled",
        )
        await env[2].save("m3", {
            "expected_l1_revision": 1,
            "expected_presentation_revision": 0,
            "request_id": "deadline-chain-recovery-save-m3",
            "change_kind": "wording",
            "document": {
                "full_injection_text": later_date + later_support,
                "coherent_sections": [{
                    "section_id": "later-trip", "text": later_date + later_support,
                    "statement_refs": ["later-date", "later-limit"],
                }],
                "optional_use_variants": [],
            },
        })
        bucket3 = await env[2].revisions.buckets.get("b3")
        buckets = [env[3], bucket3]
        env[0].inject_max_cards = 1

        request = _gateway_request(QUERY)
        debug = {"route": "ordinary", "hint_route": "fast"}
        result = await env[0]._select_recall_with_fallback(
            env[0]._select_dynamic_buckets, QUERY, "offline-ttft", buckets,
            natural_input=request, sentinel_debug=debug,
            allow_semantic=True, allow_query_planner=True, allow_rerank=True,
            include_query_planner_debug=True,
        )
        initially_selected = [row["id"] for row in result[0]]
        assert initially_selected == ["b2"]
        assert request["local_first"]["selection_sufficient"] is True
        semantic_calls_before = len(env[6])

        # The first candidate loses one required L0 dependency. Its old L1/L2
        # text must be rejected before one bounded semantic recovery selects m3.
        env[4]["2026-10-15"]["content"] += " 来源已撤回。"
        raw = f'[bucket_id:b2] {env[3]["content"]}'
        sections = {
            "direct": (raw, ["b2"], 4000),
            "targeted": ("", [], 0),
            "diffused": (raw, ["b2"], 4000),
        }

        def forbidden(*_args, **_kwargs):
            raise AssertionError("recovery must not save or generate another L2")

        monkeypatch.setattr(env[2], "save", forbidden)
        monkeypatch.setattr(env[2], "run_once", forbidden)
        rendered = await env[0]._project_recall_with_recovery(
            request, QUERY, buckets, sections,
        )

        recovery = request["projection_recovery"]
        assert recovery["attempted"] is True
        assert recovery["reason"] == "replacement_projected"
        assert recovery["initial_selected_bucket_ids"] == ["b2"]
        assert recovery["rejected_bucket_ids"] == ["b2"]
        assert recovery["selected_bucket_ids"] == ["b3"]
        assert recovery["projected_bucket_ids"] == ["b3"]
        assert recovery["selection_timing_ms"]
        assert any(row.get("bucket_id") == "b3" for row in recovery["candidate_scores"])
        chosen_score = next(row for row in recovery["candidate_scores"] if row.get("bucket_id") == "b3")
        assert chosen_score["selected"] is True and isinstance(chosen_score["score"], (int, float))
        assert recovery["candidate_stages"]
        assert "b2" not in request["selected_buckets"] and "b3" in request["selected_buckets"]
        assert len(env[6]) == semantic_calls_before + 1
        assert later_date in rendered["direct"] and later_support in rendered["direct"]
        assert DATE not in rendered["direct"] and SUPPORT not in rendered["direct"]
        assert set(request["presentation_trace"][0]["injected_statement_ids"]) == {
            "later-date", "later-limit",
        }
        trace = request["presentation_trace"][0]
        assert trace["bucket_id"] == "b3" and trace["source"] == "l2"
        assert trace["l1_revision"] == request["usage_by_bucket"]["b3"]["record"]["revision"] == 1
        assert trace["presentation_revision"] == env[2].detail("m3")["l2"]["current_presentation_revision"] == 1
        assert request["usage_by_bucket"]["b3"]["source_states"]["source:later-trip"] == "available"
        assert _path_count(env, RESOLVE_PATH) == 1
        assert all(call["path"].endswith((SOURCE_PATH, RESOLVE_PATH)) for call in env[5].calls)

    run(scenario())


def test_unready_l2_falls_back_to_complete_source_qualified_l1(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch, persist_l2=False)
    result, request = run(chain_select(env))
    rendered = run(chain_project(env, result, request))

    assert DATE in rendered["direct"] and SUPPORT in rendered["direct"]
    assert request["presentation_trace"][0]["source"] == "l1"
    assert _path_count(env, RESOLVE_PATH) == 1
    assert request["recall_io"]["l2_resolve"]["calls"] == 1


@pytest.mark.parametrize("skip", ["composer", "budget", "usage_prompt"])
def test_disabled_or_empty_projection_adds_no_source_or_l2_io(tmp_path, monkeypatch, skip):
    env = chain_fixture(tmp_path, monkeypatch)
    result, request = run(chain_select(env))
    if skip == "composer":
        request["composer_recall_enabled"] = False
        budget = 4000
    elif skip == "budget":
        budget = 0
    else:
        budget = 4000
        original = env[0]._resolve_gateway_fixed_prompt
        monkeypatch.setattr(env[0], "_resolve_gateway_fixed_prompt", lambda name, **kwargs:
            "" if name == "ombre.memory_usage_card_prompt" else original(name, **kwargs))

    calls_before = len(env[5].calls)
    source_reads_before = request["recall_io"]["sources"]["calls"]
    rendered = run(chain_project(env, result, request, budget=budget))

    assert all(value == "" for value in rendered.values())
    assert len(env[5].calls) == calls_before
    assert request["recall_io"]["sources"]["calls"] == source_reads_before
    assert _path_count(env, RESOLVE_PATH) == 0
    assert request["projection_status"]["state"] == "empty"
    assert request["presentation_trace"][0]["source"] == "none"


def test_snapshot_guard_rechecks_every_source_before_replaying_complete_l2(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await _select_with_natural_context(env)
        rendered = await chain_project(env, result, request)
        assert DATE in rendered["direct"] and SUPPORT in rendered["direct"]
        view = env[0].memory_authority_view
        view.memory_index_metadata_map(["b2"])
        view.memory_l1_recall_map(["b2"])
        guard = env[1].snapshot_guard(env[0], request, complete=True)
        assert guard["complete"] is True and set(guard["records"]) == {"b2"}
        assert guard["records"]["b2"]["statement_uses"]
        assert set(guard["records"]["b2"]["statement_uses"].values()) == {"relationship_reference"}
        before = _path_count(env, SOURCE_PATH)

        current_request = _gateway_request(QUERY)
        current_request.pop("_use_context", None)
        current = await env[1].snapshot_sources_current(
            env[0], {"memory_guard": guard}, current_request,
        )
        stamp_after = env[1].authority_stamp(env[0])
        source_delta = _path_count(env, SOURCE_PATH) - before
        if stamp_after != guard["authority_stamp"]:
            # SQLite may remove its zero-length WAL sidecar during this real
            # source read. The resulting conservative cache miss is expected.
            assert current is False
            assert source_delta == 1
            checked = env[5].responses[-1]["body"]["items"][0]
            assert checked["state"] == "ok"
            assert all(source.get("eligible_source") is True
                       for source in checked["sources"].values())
            return
        assert current is True
        assert source_delta == 1

        # Losing only the support source still blocks replay of the whole L2
        # narrative, even though the date statement remains eligible.
        env[4]["2026-10-15"]["content"] += " 已撤回来源。"
        changed_request = _gateway_request(QUERY)
        changed_request.pop("_use_context", None)
        changed = await env[1].snapshot_sources_current(
            env[0], {"memory_guard": guard}, changed_request,
        )
        assert changed is False
        assert _path_count(env, SOURCE_PATH) == before + 2

    run(scenario())


def test_current_source_cache_hits_and_snapshot_refresh_detects_external_change(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        # Keep only the filesystem epoch stable to isolate source validity
        # from SQLite WAL-sidecar churn. Authority and L0 readers stay real.
        view = env[0].memory_authority_view
        stamp = env[1].authority_stamp(env[0])
        assert stamp is not None
        monkeypatch.setattr(view, "_file_stamp", lambda: stamp)
        result, request = await _select_with_natural_context(env)
        await chain_project(env, result, request)
        before = _path_count(env, SOURCE_PATH)
        permitted = await env[1].verify_sources(env[0], request, QUERY,
            [{"bucket": row, "score": 0.0} for row in result[0]])
        assert permitted and _path_count(env, SOURCE_PATH) == before
        assert request["recall_io"]["sources"]["cache_hits"] > 0
        guard = env[1].snapshot_guard(env[0], request, complete=True)
        current_request = _gateway_request(QUERY)
        current_request.pop("_use_context", None)
        assert await env[1].snapshot_sources_current(env[0], {"memory_guard": guard}, current_request)
        assert _path_count(env, SOURCE_PATH) == before + 1
        env[4]["2026-10-15"]["content"] += " External source changed without an Authority commit."
        changed_request = _gateway_request(QUERY)
        changed_request.pop("_use_context", None)
        assert not await env[1].snapshot_sources_current(env[0], {"memory_guard": guard}, changed_request)
        assert _path_count(env, SOURCE_PATH) == before + 2

    run(scenario())


def test_snapshot_guard_rejects_new_l1_version_before_source_replay(tmp_path, monkeypatch):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        result, request = await chain_select(env)
        rendered = await chain_project(env, result, request)
        assert DATE in rendered["direct"] and SUPPORT in rendered["direct"]
        guard = env[1].snapshot_guard(env[0], request, complete=True)
        before = _path_count(env, SOURCE_PATH)
        monkeypatch.setattr(env[0].memory_authority_view, "_file_stamp",
                            lambda: guard["authority_stamp"])

        detail = await env[2].revisions.detail("m2")
        document = copy.deepcopy(detail["l1"]["document"])
        document["narrative"]["body"] += " 后续修订。"
        body, semantic = split_document(document)
        await env[2].revisions.commit.commit_memory(
            memory_id="m2", bucket_id="b2", expected_revision=1,
            body=body, metadata={"name": "合成行程", "memory_l1": semantic},
            source_refs=[], decision_source="snapshot-version-change",
            idempotency_key="snapshot-version-change", actor="test",
            l1_write_intent=True, recall_policy="enabled",
        )
        assert env[2].revisions.authority.get_memory("m2")["active_revision"] == 2

        current = await env[1].snapshot_sources_current(
            env[0], {"memory_guard": guard}, _gateway_request(QUERY),
        )
        metadata = env[0].memory_authority_view.memory_index_metadata_map(["b2"])["b2"]
        assert current is False
        assert metadata["revision"] == 2
        assert _path_count(env, SOURCE_PATH) == before
        assert _path_count(env, RESOLVE_PATH) == 1

    run(scenario())


@pytest.mark.parametrize("stamp_failure", ["none", "error"])
def test_snapshot_manifest_completeness_and_authority_stamp_fail_closed_without_extra_io(
    tmp_path, monkeypatch, stamp_failure,
):
    env = chain_fixture(tmp_path, monkeypatch)

    async def scenario():
        runtime, gateway = env[1], env[0]
        before = len(env[5].calls)
        partial = runtime.snapshot_guard(gateway, {
            "q_current": QUERY, "injected_usage_bucket_ids": [],
        }, complete=False)
        assert await runtime.snapshot_sources_current(
            gateway, {"memory_guard": partial}, _gateway_request(QUERY),
        ) is False

        empty = runtime.snapshot_guard(gateway, {
            "q_current": QUERY, "injected_usage_bucket_ids": [],
        }, complete=True)
        assert await runtime.snapshot_sources_current(
            gateway, {"memory_guard": empty}, _gateway_request(QUERY),
        ) is True
        assert len(env[5].calls) == before

        result, request = await chain_select(env)
        await chain_project(env, result, request)
        guard = runtime.snapshot_guard(gateway, request, complete=True)
        before_guard_check = len(env[5].calls)
        view = gateway.memory_authority_view
        if stamp_failure == "none":
            monkeypatch.setattr(view, "_file_stamp", lambda: None)
        else:
            def unavailable_stamp():
                raise OSError("offline snapshot stamp")
            monkeypatch.setattr(view, "_file_stamp", unavailable_stamp)
        assert await runtime.snapshot_sources_current(
            gateway, {"memory_guard": guard}, _gateway_request(QUERY),
        ) is False
        assert len(env[5].calls) == before_guard_check

    run(scenario())
