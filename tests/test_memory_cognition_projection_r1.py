"""Gateway's natural-recall projection of historical cognition versions."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from gateway import memory_use
from memory_l1_schema import unknown
from memory_system_config import values_from_config
from prompt_plan_mirror import PromptPlanMirrorStore
from test_memory_recall_gateway_r2 import (
    _gateway_service,
    _recall_input,
    _semantic_document,
    _base_config,
)
from utils import count_tokens_approx


BUCKET_ID = "bucket-cognition-chain"
MEMORY_ID = "memory:cognition-chain"


def _cognition(role="question", status="supported", open_questions=()):
    return {
        "role": unknown("known", role),
        "epistemic_status": unknown("known", status),
        "about_refs": [],
        "open_questions": list(open_questions),
    }


def _rows(*, long_history=False, multi_paths=False):
    document, binding_id = _semantic_document(
        BUCKET_ID, "Synthetic source narrative for the current cognitive memory.",
        current="当前版本的合成认知陈述。", old="稳定且未变化的合成陈述。",
    )
    current_statement = document["statements"][1]
    current_statement["cognition"] = _cognition(
        open_questions=["当前版本仍待确认的问题。"],
    )
    current_statement["statement_revision_id"] = "statement-revision-current-r2"
    current = {
        "memory_id": MEMORY_ID, "revision": 2, "bucket_id": BUCKET_ID,
        "historical": False, "document": document,
    }

    history_document = copy.deepcopy(document)
    if multi_paths:
        linked = []
        for name, content in (
            ("a", "第一个关联的合成历史认知。"),
            ("b", "第二个关联的合成历史认知。"),
            ("unrelated", "未关联陈述不得泄漏到认知链。"),
        ):
            statement = copy.deepcopy(current_statement)
            statement["statement_id"] = f"statement-history-{name}"
            statement["statement_revision_id"] = f"statement-revision-history-{name}"
            statement["content"] = content
            statement["cognition"] = _cognition(open_questions=[f"{name} 的早期问题？"])
            linked.append(statement)
        history_document["statements"] = linked
        via_paths = [{
            "kind": "supplements", "relation_id": f"relation-history-{name}",
            "confirmation_state": "confirmed", "from": (MEMORY_ID, 2),
            "object_type": "statement", "object_id": f"statement-history-{name}",
        } for name in ("a", "b")]
    else:
        history_statement = history_document["statements"][1]
        history_statement["content"] = (
            "更早保存的合成认知陈述。" + ("较早版本细节。" * 180 if long_history else "")
        )
        history_statement["statement_revision_id"] = "statement-revision-history-r1"
        history_statement["cognition"] = _cognition(
            open_questions=["早期版本留下的问题。"],
        )
        via_paths = [{"kind": "revision_history", "from": (MEMORY_ID, 2)}]
    history = {
        "memory_id": MEMORY_ID, "revision": 1, "bucket_id": BUCKET_ID,
        "historical": True, "document": history_document,
        "via": via_paths[0], "via_paths": via_paths,
    }

    class View:
        enabled = False

        def __init__(self):
            self.chain_calls = []

        def memory_l1_recall_map(self, bucket_ids):
            return {BUCKET_ID: {**current, "state": "l1"}} if BUCKET_ID in bucket_ids else {}

        def cognition_chain(self, seeds, *, depth, limit):
            self.chain_calls.append((list(seeds), depth, limit))
            return {"nodes": [current, history], "truncated": False, "issues": []}

        def watermark(self):
            return {"available": True, "synthetic": True}

        def memory_revision_map(self, _bucket_ids):
            return {MEMORY_ID: {"revision": 2}}

    return View(), current, history, binding_id


def _install_revision_source_http(monkeypatch, states=None):
    requests = []
    states = dict(states or {})
    original_client = httpx.AsyncClient

    def handler(request):
        payload = json.loads(request.content)
        if request.url.path.endswith('/presentations/resolve'):
            return httpx.Response(200, json={'status': 'ok', 'items': []})
        requests.extend(payload["items"])
        rows = []
        for item in payload["items"]:
            state = states.get(item["revision"], "available")
            sources = {
                binding_id: {
                    "state": state,
                    "eligible_source": state == "available",
                    "reason": None if state == "available" else "synthetic_withdrawal",
                }
                for binding_id in item["binding_ids"]
            }
            rows.append({"memory_id": item["memory_id"], "revision": item["revision"],
                         "state": "ok", "sources": sources})
        return httpx.Response(200, json={"items": rows})

    transport = httpx.MockTransport(handler)

    def mock_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(memory_use.httpx, "AsyncClient", mock_client)
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.synthetic.invalid")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "synthetic-cognition-source-token")
    return requests


def _fixture(tmp_path, monkeypatch, *, states=None, settings=None,
             long_history=False, multi_paths=False):
    view, current, history, binding_id = _rows(
        long_history=long_history, multi_paths=multi_paths,
    )
    buckets = [{"id": BUCKET_ID,
                "content": "Synthetic source narrative for the current cognitive memory.",
                "metadata": {"name": "synthetic cognition"}}]
    config = _base_config()
    config["recall"].update(cognition_chain_depth=2, cognition_chain_nodes=8,
                            cognition_chain_budget=300)
    if settings:
        config["recall"].update(settings)
    service, calls = _gateway_service(tmp_path, monkeypatch, view, buckets, [], config)
    requests = _install_revision_source_http(monkeypatch, states)
    request = _recall_input("请回看这条认知如何变化。")
    request["index_metadata"] = {BUCKET_ID: {"memory_id": MEMORY_ID, "revision": 2}}
    raw = f"[bucket_id:{BUCKET_ID}] Main selected memory body."
    return view, service, request, buckets, raw, requests, calls, current, history, binding_id


def _project(service, request, buckets, raw, *, section_budget=2400):
    sections = asyncio.run(memory_use.project_sections(
        service, request, request["q_context"], buckets,
        {"direct": (raw, [BUCKET_ID], section_budget),
         "targeted": ("", [], 0), "diffused": ("", [], 0)},
    ))
    projection = service._build_memory_recall_projection(
        recalled_memory=sections.get("direct", ""), targeted_memory_detail="",
        related_memory="", recalled_moments=[],
        targeted_memory_detail_debug={"accepted_ids": []},
        memory_sentinel_debug={"route": "ordinary", "route_reason_codes": []},
        natural_input=request,
    )
    return sections, projection


def _disable_chain_prompt(service, tmp_path):
    store = PromptPlanMirrorStore(tmp_path / "prompt-plan.sqlite3")
    plan_sha = "a" * 64
    store.put_plan("preset:cognition-chain-off", 1, plan_sha256=plan_sha,
                   gateway_slice={"schema": "guyan.gateway-prompt-slice.v1", "settings": {},
                                  "blocks": [{
                                      "block_id": "block:memory-cognition-chain",
                                      "source_id": "ombre.memory_cognition_chain_prompt",
                                      "scope": "memory.cognition_chain",
                                      "stage": "ombre_post_injection", "mode": "off",
                                      "role": "system", "lane": "instruction",
                                      "anchor": "instructions.end", "order": 1,
                                      "priority": 1, "enabled": False,
                                      "owner_body": "", "frozen_body": "",
                                      "wrapper_text": "", "token_budget": None,
                                  }]})
    store.put_binding("memory.cognition_chain", preset_id="preset:cognition-chain-off",
                      preset_revision=1, plan_sha256=plan_sha, aiz_binding_revision=1)
    service.prompt_plan_mirror = store


def test_historical_cognition_is_injected_as_history_and_trace_counts_it_separately(
    tmp_path, monkeypatch,
):
    view, service, request, buckets, raw, source_requests, calls, _current, history, binding_id = _fixture(
        tmp_path, monkeypatch,
    )
    sections, projection = _project(service, request, buckets, raw)

    body = sections["direct"]
    assert "当前版本的合成认知陈述。" in body
    assert "更早保存的合成认知陈述。" in body
    assert "[historical_evidence" not in body
    chain_body = body.split("以前记下的想法：", 1)[1]
    assert "稳定且未变化的合成陈述。" not in chain_body
    assert request["cognition_chain"]["provider_requests"] == 0
    assert request["cognition_chain"]["rendered_versions"] == [{
        "memory_id": MEMORY_ID, "revision": 1, "historical": True,
        "label": "以前记下的想法：", "text": "以前记下的想法：\n" + chain_body.strip(), "partial": False,
    }]
    assert view.chain_calls == [([(MEMORY_ID, 2)], 2, 8)]
    assert any(item["revision"] == 1 and item.get("allow_historical") is True
               for item in source_requests)
    assert all(item["binding_ids"] == [binding_id] for item in source_requests)
    assert projection["selected_bucket_ids"] == [BUCKET_ID]
    assert projection["recall_status"]["selected_count"] == 1
    assert projection["recall_status"]["injected_count"] == 1
    assert projection["recall_status"]["cognition_chain_version_count"] == 1
    assert calls["rerank"] == []


def test_withdrawn_historical_source_does_not_render_old_cognition(tmp_path, monkeypatch):
    _view, service, request, buckets, raw, source_requests, _calls, _current, _history, _binding = _fixture(
        tmp_path, monkeypatch, states={1: "withdrawn", 2: "available"},
    )
    sections, projection = _project(service, request, buckets, raw)
    assert "当前版本的合成认知陈述。" in sections["direct"]
    assert "更早保存的合成认知陈述。" not in sections["direct"]
    assert request["cognition_chain"]["rendered_versions"] == []
    assert any(item["revision"] == 1 and item.get("allow_historical") is True
               for item in source_requests)
    assert projection["recall_status"]["cognition_chain_version_count"] == 0
    assert projection["recall_status"]["selected_count"] == 1


def test_multiple_statement_paths_render_their_union_without_unrelated_claims(
    tmp_path, monkeypatch,
):
    _view, service, request, buckets, raw, _source_requests, _calls, *_rest = _fixture(
        tmp_path, monkeypatch, multi_paths=True,
    )
    sections, projection = _project(service, request, buckets, raw)
    body = sections["direct"]
    assert "第一个关联的合成历史认知。" in body
    assert "第二个关联的合成历史认知。" in body
    assert "未关联陈述不得泄漏到认知链。" not in body
    assert "Recorded connection:" not in body
    assert "a 的早期问题？" in body
    assert "b 的早期问题？" in body
    assert projection["recall_status"]["cognition_chain_version_count"] == 1


@pytest.mark.parametrize("zero_option", [
    "cognition_chain_depth", "cognition_chain_nodes", "cognition_chain_budget",
])
def test_each_zero_cognition_limit_disables_expansion_without_hiding_the_main_card(
    tmp_path, monkeypatch, zero_option,
):
    view, service, request, buckets, raw, source_requests, _calls, *_rest = _fixture(
        tmp_path, monkeypatch, settings={zero_option: 0},
    )
    sections, projection = _project(service, request, buckets, raw)
    assert "当前版本的合成认知陈述。" in sections["direct"]
    assert "更早保存的合成认知陈述。" not in sections["direct"]
    assert view.chain_calls == []
    assert request.get("cognition_chain", {}).get("rendered_versions", []) == []
    assert projection["recall_status"]["selected_count"] == 1
    assert projection["recall_status"]["injected_count"] == 1
    assert not any(item["revision"] == 1 for item in source_requests)


def test_owner_off_for_chain_prompt_does_not_restore_history_and_small_budget_keeps_main(
    tmp_path, monkeypatch,
):
    view, service, request, buckets, raw, _source_requests, _calls, *_rest = _fixture(
        tmp_path, monkeypatch, long_history=True,
    )
    _disable_chain_prompt(service, tmp_path)
    sections, projection = _project(service, request, buckets, raw)
    assert "当前版本的合成认知陈述。" in sections["direct"]
    assert "更早保存的合成认知陈述。" not in sections["direct"]
    assert request["cognition_chain"]["rendered_versions"] == []
    assert view.chain_calls
    assert projection["recall_status"]["cognition_chain_version_count"] == 0

    view2, service2, request2, buckets2, raw2, _sources2, _calls2, *_rest2 = _fixture(
        tmp_path / "main-budget-reference", monkeypatch,
        settings={"cognition_chain_depth": 0}, long_history=True,
    )
    main_only, _reference_projection = _project(service2, request2, buckets2, raw2)
    section_budget = count_tokens_approx(main_only["direct"] + "\n\n")
    assert "当前版本的合成认知陈述。" in main_only["direct"]

    view2, service2, request2, buckets2, raw2, _sources2, _calls2, *_rest2 = _fixture(
        tmp_path / "small-budget", monkeypatch, long_history=True,
    )
    sections2, projection2 = _project(
        service2, request2, buckets2, raw2, section_budget=section_budget,
    )
    assert "当前版本的合成认知陈述。" in sections2["direct"]
    assert "更早保存的合成认知陈述。" not in sections2["direct"]
    assert request2["cognition_chain"]["rendered_versions"] == []
    assert {issue["reason"] for issue in request2["cognition_chain"]["issues"]} >= {
        "no_budget_after_main_memory",
    }
    assert projection2["recall_status"]["selected_count"] == 1
    assert projection2["recall_status"]["injected_count"] == 1
