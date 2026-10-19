"""Focused contracts for recent-memory topic selection and owner settings.

These tests use temporary Memory Authority stores and the existing shared L2
projection path. They never call a provider or touch production state.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

import memory_recall_runtime as memory_use
from test_memory_local_first_settings import _app, _base_config
from test_memory_readable_injection_contract import (
    _composer_plan,
    _compose,
    _projection,
    _request_with_records,
    _resolved_l2_stub,
)
from test_memory_recall_gateway_r2 import _commit_l1, _install_source_http
from test_memory_recall_projection_r2 import _fixture, _request
from recall_policy import RecallPolicy

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _restore_real_policy(service) -> None:
    # The shared natural-selection fixture uses a minimal policy namespace.
    # Replace it with the production policy so no selection rule is stubbed.
    service.recall_policy = RecallPolicy()


def _config(service, *, mode: str = "auto", max_items: int = 6) -> None:
    service.config["memory_system"]["recall"].update(
        recent_context_mode=mode,
        recent_context_max_items=max_items,
    )
    service.head_recent_hours = 48


def _bucket(bucket_id: str, content: str, *, age_minutes: int = 0,
            layer: str = "dynamic") -> dict:
    created = datetime.now() - timedelta(minutes=age_minutes)
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "name": bucket_id,
            "type": layer,
            "created": created.isoformat(),
        },
    }


def _select(service, buckets: list[dict], query: str, *, allow_vague: bool = False):
    debug: dict = {}
    carrier = asyncio.run(service._build_recent_context_block(
        buckets, query, allow_vague=allow_vague, selection_debug=debug,
    ))
    return carrier, debug


def test_recall_policy_filters_explicit_topics_but_preserves_open_and_overview_context(
    tmp_path, monkeypatch,
):
    _authority, service, _view, _fixture_buckets, _plan = _fixture(tmp_path, monkeypatch)
    _restore_real_policy(service)

    target = _bucket(
        "reality-search",
        "主人希望聊天搜索能区分原文和渲染后的文本，避免把页面效果当成自己输入的内容。",
        age_minutes=3,
    )
    unrelated = _bucket(
        "browser-browse",
        "记录了浏览器页面截图、标签页整理和按钮操作。",
        age_minutes=1,
    )

    # A direct historical topic question is filtered by the real RecallPolicy.
    query = "之前提过聊天搜索里原文和渲染文本该怎么区分？"
    _config(service, mode="auto")
    carrier, debug = _select(service, [unrelated, target], query)
    assert debug["topic_filter_applied"] is True
    assert debug["selected_bucket_ids"] == ["reality-search"]
    assert debug["unrelated_bucket_ids"] == ["browser-browse"]
    assert "bucket_id:reality-search" in carrier
    assert "bucket_id:browser-browse" not in carrier

    # A topic lookup with no related recent candidate returns no recent content.
    no_match = _bucket("unrelated-only", "厨房里新换了一个木质置物架。")
    carrier, debug = _select(service, [no_match], query)
    assert carrier == ""
    assert debug["selected_bucket_ids"] == []
    assert debug["unrelated_bucket_ids"] == ["unrelated-only"]

    # An open conversational return keeps chronological continuity in auto;
    # relevant mode suppresses the same unrelated recent item.
    open_query = "我回来了，想随便聊聊。"
    carrier, debug = _select(service, [unrelated], open_query)
    assert debug["topic_filter_applied"] is False
    assert debug["selected_bucket_ids"] == ["browser-browse"]
    assert carrier
    _config(service, mode="relevant")
    carrier, debug = _select(service, [unrelated], open_query)
    assert debug["topic_filter_applied"] is True
    assert debug["selected_bucket_ids"] == []
    assert debug["unrelated_bucket_ids"] == ["browser-browse"]

    # A broad, explicit recent overview is exempt from topic filtering in
    # relevant mode and remains newest-first.
    older = _bucket("older-chat", "昨天聊了烘焙。", age_minutes=60)
    newer = _bucket("newer-chat", "刚才整理了书桌。", age_minutes=2)
    carrier, debug = _select(
        service, [older, newer], "最近我们聊了什么？", allow_vague=True,
    )
    assert debug["topic_filter_applied"] is False
    assert debug["selected_bucket_ids"] == ["newer-chat", "older-chat"]
    assert carrier.index("bucket_id:newer-chat") < carrier.index("bucket_id:older-chat")

    # Chronological mode preserves the old behavior for a specific query.
    _config(service, mode="chronological")
    carrier, debug = _select(service, [unrelated], query)
    assert debug["topic_filter_applied"] is False
    assert debug["selected_bucket_ids"] == ["browser-browse"]
    assert carrier


def test_recent_context_count_limits_keep_eligibility_and_expiration(tmp_path, monkeypatch):
    _authority, service, _view, _fixture_buckets, _plan = _fixture(tmp_path, monkeypatch)
    _restore_real_policy(service)
    service.head_recent_hours = 24
    query = "我回来了，想随便聊聊。"
    rows = [
        _bucket(f"recent-{index}", f"近期普通聊天 {index}。", age_minutes=index)
        for index in range(8)
    ]
    rows += [
        _bucket("expired", "旧的聊天记录。", age_minutes=48 * 60),
        _bucket("source-record", "独立来源记录。", layer="source"),
    ]

    _config(service, mode="auto", max_items=0)
    carrier, debug = _select(service, rows, query)
    assert carrier == ""
    assert debug["state"] == "disabled_by_count"
    assert debug["selected_bucket_ids"] == []

    _config(service, mode="auto", max_items=1)
    carrier, debug = _select(service, rows, query)
    assert debug["selected_bucket_ids"] == ["recent-0"]
    assert "bucket_id:recent-0" in carrier
    assert "bucket_id:recent-1" not in carrier

    # More than the historical six-item default is honored, while the real
    # layer and freshness gates still reject source records and expired items.
    _config(service, mode="auto", max_items=8)
    carrier, debug = _select(service, rows, query)
    assert debug["selected_bucket_ids"] == [f"recent-{index}" for index in range(8)]
    assert debug["qualified_count"] == 8
    assert "bucket_id:expired" not in carrier
    assert "bucket_id:source-record" not in carrier


def test_recent_context_settings_save_reload_and_runtime_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "recent-context-test-owner")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "")
    base = _base_config(tmp_path)
    service, app = _app(copy.deepcopy(base))
    overlay_path = Path(base["_runtime_config_path"])
    headers = {"Authorization": "Bearer recent-context-test-owner"}
    endpoint = "/api/memory-authority/configuration"

    assert service.embedding_engine.enabled is False
    assert service.reranker_engine.enabled is False
    assert service.upstream_api_key == ""

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers=headers,
        ) as client:
            initial_response = await client.get(endpoint)
            assert initial_response.status_code == 200
            initial = initial_response.json()
            assert initial["values"]["recall"]["recent_context_mode"] == "auto"
            assert initial["values"]["recall"]["recent_context_max_items"] == 6

            values = copy.deepcopy(initial["values"])
            values["recall"].update(recent_context_mode="chronological", recent_context_max_items=2)
            saved_response = await client.put(endpoint, json={
                "expected_revision": initial["revision"],
                "request_id": "recent-context-settings-save",
                "values": values,
            })
            assert saved_response.status_code == 200
            saved = saved_response.json()
            assert saved["status"] == "ok" and saved["applied"] is True
            assert memory_use.option(service, "recent_context_mode", None) == "chronological"
            assert memory_use.option(service, "recent_context_max_items", None) == 2

            # A fresh service reads the saved overlay and its selector consumes
            # both persisted values in the normal runtime option path.
            restarted, restarted_app = _app(copy.deepcopy(base))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://gateway.test",
                headers=headers,
            ) as restarted_client:
                readback_response = await restarted_client.get(endpoint)
            assert readback_response.status_code == 200
            readback = readback_response.json()
            assert readback["values"]["recall"]["recent_context_mode"] == "chronological"
            assert readback["values"]["recall"]["recent_context_max_items"] == 2
            assert memory_use.option(restarted, "recent_context_mode", None) == "chronological"
            assert memory_use.option(restarted, "recent_context_max_items", None) == 2
            buckets = [
                _bucket(f"reload-{index}", f"open conversation {index}")
                for index in range(3)
            ]
            debug = {}
            carrier = await restarted._build_recent_context_block(
                buckets, "之前提过火星基地门禁密码是多少？", selection_debug=debug,
            )
            assert debug["max_items"] == 2
            assert debug["mode"] == "chronological"
            assert len(debug["selected_bucket_ids"]) == 2
            assert "bucket_id:reload-2" not in carrier

            # A legacy client omitting the new keys preserves their saved values.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://gateway.test",
                headers=headers,
            ) as restarted_client:
                legacy_values = copy.deepcopy(readback["values"])
                legacy_values["recall"].pop("recent_context_mode")
                legacy_values["recall"].pop("recent_context_max_items")
                legacy_values["limits"]["inject_max_cards"] += 1
                legacy_response = await restarted_client.put(endpoint, json={
                    "expected_revision": readback["revision"],
                    "request_id": "legacy-client-omits-recent-fields",
                    "values": legacy_values,
                })
                assert legacy_response.status_code == 200
                assert legacy_response.json()["result"]["values"]["recall"]["recent_context_mode"] == "chronological"
                assert legacy_response.json()["result"]["values"]["recall"]["recent_context_max_items"] == 2
                assert memory_use.option(restarted, "recent_context_mode", None) == "chronological"
                assert memory_use.option(restarted, "recent_context_max_items", None) == 2

                current = (await restarted_client.get(endpoint)).json()
                before_invalid = overlay_path.read_bytes()
                invalid_cases = [
                    ("recent_context_mode", "unknown", "unknown_recent_context_mode"),
                    ("recent_context_max_items", 201, "recent_context_limit_out_of_range"),
                ]
                for field, value, code in invalid_cases:
                    invalid_values = copy.deepcopy(current["values"])
                    invalid_values["recall"][field] = value
                    invalid_response = await restarted_client.put(endpoint, json={
                        "expected_revision": current["revision"],
                        "request_id": f"invalid-{field}",
                        "values": invalid_values,
                    })
                    assert invalid_response.status_code == 400
                    assert invalid_response.json()["error"]["code"] == code
                    assert overlay_path.read_bytes() == before_invalid
                    assert memory_use.option(restarted, "recent_context_mode", None) == "chronological"
                    assert memory_use.option(restarted, "recent_context_max_items", None) == 2

                stale_values = copy.deepcopy(current["values"])
                stale_values["recall"]["recent_context_max_items"] = 1
                conflict = await restarted_client.put(endpoint, json={
                    "expected_revision": initial["revision"],
                    "request_id": "stale-recent-context-update",
                    "values": stale_values,
                })
                assert conflict.status_code == 409
                assert conflict.json()["error"]["code"] == "configuration_revision_conflict"
                assert overlay_path.read_bytes() == before_invalid
                assert memory_use.option(restarted, "recent_context_max_items", None) == 2

    asyncio.run(scenario())


def test_recent_selector_uses_verified_ready_l2_projection_and_composer_adoption(
    tmp_path, monkeypatch,
):
    target_l1 = "关于聊天搜索，主人希望区分原文与渲染后的文本。"
    browse_l1 = "浏览器页面的截图记录和导航栏整理。"
    authority, service, view, _fixture_buckets, _old_plan = _fixture(
        tmp_path, monkeypatch,
    )
    _restore_real_policy(service)
    target_id = "recent-relevant-target"
    browse_id = "recent-unrelated-browse"
    _commit_l1(
        authority, target_id, target_l1, topic="chat_search",
        current=target_l1, old="此前曾记录其他聊天方式。",
    )
    _commit_l1(
        authority, browse_id, browse_l1, topic="browser",
        current=browse_l1, old="更早的浏览器操作记录。",
    )
    target = _bucket(target_id, target_l1, age_minutes=2)
    browse = _bucket(browse_id, browse_l1, age_minutes=1)
    buckets = [target, browse]
    _config(service, mode="auto", max_items=6)

    query = "之前提过聊天搜索里原文和渲染文本该怎么区分？"
    carrier, selection = _select(service, buckets, query)
    assert selection["selected_bucket_ids"] == [target_id]
    assert selection["unrelated_bucket_ids"] == [browse_id]

    # _fixture has already installed its own MockTransport. Restore the real
    # constructor before installing the capturing source-verification transport.
    monkeypatch.setattr(memory_use.httpx, "AsyncClient", _REAL_ASYNC_CLIENT)
    source_calls = _install_source_http(monkeypatch)
    plan = _composer_plan(service, tmp_path)
    request = _request_with_records(view, [target_id, browse_id])
    request["q_current"] = query
    request["q_context"] = query
    request["_use_context"].update(query=query, purpose="history")
    l2_text = "主人希望聊天搜索区分自己输入的原文和页面渲染文本。"
    resolver_calls = _resolved_l2_stub(monkeypatch, text=l2_text)

    rendered = asyncio.run(memory_use.project_sections(
        service, request, query, buckets, {
            "targeted": ("", [], 0),
            "direct": ("", [], 0),
            "diffused": ("", [], 0),
            "recent": (carrier, selection["selected_bucket_ids"], 4000),
        },
    ))

    assert source_calls, {
        "skip": request.get("_projection_skip_reason"),
        "carrier": carrier,
        "trace": request.get("presentation_trace"),
        "usage": request.get("usage_by_bucket"),
        "spans": request.get("rendered_memory_spans"),
    }
    assert len(resolver_calls) == 1
    assert [row["bucket_id"] for row in resolver_calls[0]] == [target_id]
    assert request["usage_by_bucket"][target_id]["state"] == "l1"
    assert request["usage_by_bucket"][target_id]["narrative_allowed"] is True
    trace = next(row for row in request["presentation_trace"] if row["bucket_id"] == target_id)
    assert trace["source"] == "l2"
    assert trace["resolver_state"] == "ready"
    assert trace["presentation_revision"] == 7
    assert l2_text in rendered["recent"]
    assert browse_l1 not in rendered["recent"]
    assert browse_id not in request["injected_usage_bucket_ids"]
    assert request["injected_usage_bucket_ids"] == [target_id]

    projection = _projection(service, request, rendered)
    messages, _stable, _dynamic, _debug = _compose(
        service, plan, projection, rendered, recent=rendered["recent"],
    )
    composed = "\n".join(str(item.get("content") or "") for item in messages)
    assert l2_text in composed
    assert browse_l1 not in composed

    # Composer-off and zero section budget both suppress actual adoption.
    off_root = tmp_path / "composer-off"
    off_root.mkdir()
    off_plan = _composer_plan(service, off_root, memory_mode="off", recent_mode="off")
    assert service._composer_live_recall_enabled(off_plan) is False
    off_request = _request_with_records(view, [target_id, browse_id])
    off_request["q_current"] = query
    off_request["q_context"] = query
    off_request["_use_context"].update(query=query, purpose="history")
    off_request["composer_recall_enabled"] = service._composer_live_recall_enabled(off_plan)
    off_rendered = asyncio.run(memory_use.project_sections(
        service, off_request, query, buckets, {
            "targeted": ("", [], 0),
            "direct": ("", [], 0),
            "diffused": ("", [], 0),
            "recent": (carrier, selection["selected_bucket_ids"], 4000),
        },
    ))
    assert off_rendered["recent"] == ""
    assert off_request["injected_usage_bucket_ids"] == []
    off_projection = _projection(service, off_request, off_rendered)
    off_messages, _stable, _dynamic, _debug = _compose(
        service, off_plan, off_projection, off_rendered, recent="",
    )
    off_text = "\n".join(str(item.get("content") or "") for item in off_messages)
    assert l2_text not in off_text

    budget_request = _request_with_records(view, [target_id, browse_id])
    budget_request["q_current"] = query
    budget_request["q_context"] = query
    budget_request["_use_context"].update(query=query, purpose="history")
    budget_rendered = asyncio.run(memory_use.project_sections(
        service, budget_request, query, buckets, {
            "targeted": ("", [], 0),
            "direct": ("", [], 0),
            "diffused": ("", [], 0),
            "recent": (carrier, selection["selected_bucket_ids"], 0),
        },
    ))
    assert budget_rendered["recent"] == ""
    assert budget_request["injected_usage_bucket_ids"] == []
    budget_projection = _projection(service, budget_request, budget_rendered)
    budget_messages, _stable, _dynamic, _debug = _compose(
        service, plan, budget_projection, budget_rendered,
        recent=budget_rendered["recent"],
    )
    budget_text = "\n".join(str(item.get("content") or "") for item in budget_messages)
    assert l2_text not in budget_text
