"""Cross-service P4 contract: persisted Brain L2 through Gateway projection."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest

from memory_l0_sources import McpDiarySourceAdapter, MemoryL0Sources
from memory_l1_reads import MemoryL1Reads
from memory_l1_schema import document_template, generation_template, split_document, statement_template
from test_memory_l2_p4 import setup as setup_l2


def run(task):
    return asyncio.run(task)


def _gateway_runtime(monkeypatch):
    brain_root = Path(__file__).resolve().parents[1]
    gateway_root = Path(os.environ.get('GUYAN_GATEWAY_WORKTREE',
        str(brain_root.parent / "ombre-memory-semantic-gateway-r2")))
    runtime_path = gateway_root / "memory_recall_runtime.py"
    if not runtime_path.is_file():
        pytest.skip(f"Gateway worktree not found: {runtime_path}")

    # The runtime is imported under a private alias. Its small pure dependencies
    # are loaded by path from the Gateway checkout so Brain's modules do not
    # shadow the implementation under test.
    for name in ("memory_recall_semantics", "memory_system_config", "utils", "memory_readable_projection"):
        path = gateway_root / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    alias = "_gateway_memory_recall_runtime_p4_contract"
    spec = importlib.util.spec_from_file_location(alias, runtime_path)
    assert spec and spec.loader
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, runtime)
    spec.loader.exec_module(runtime)
    return runtime


def _source_backed_l1(service, authority):
    source_text = "原始来源：那次我听见她说想要抱抱。"
    source_state = {"id": "gateway-contract-diary", "content": source_text}

    async def read_diary(day):
        assert day == "2026-10-14"
        return copy.deepcopy(source_state)

    adapter = McpDiarySourceAdapter("p4-gateway-contract", read_diary)
    sources = MemoryL0Sources(authority, [adapter])
    binding_id = "source:p4-gateway-contract"
    captured = run(sources.capture_diary("2026-10-14", binding_id=binding_id))
    catalog, catalog_revision = service.revisions.catalog_provider()
    document = document_template(
        "L1-R2 原始记忆正文。", "Gateway L2 contract", catalog=catalog,
        catalog_revision=catalog_revision,
    )
    document["evidence_bindings"] = [captured["binding"]]
    statement = statement_template()
    statement.update(
        statement_id="statement:p4-gateway-contract",
        statement_revision_id="revision:p4-gateway-contract",
        content="那次我听见她说想要抱抱。",
        evidence_refs=[binding_id],
    )
    statement["classification"]["basis"] = {
        "state": "known", "value": "record", "evidence_refs": [binding_id],
    }
    document["statements"] = [statement]
    document["generation"] = generation_template(
        catalog, catalog_revision, "first_pass_model",
    )

    commit = service.revisions.commit
    run(commit.commit_memory(
        memory_id="m2", bucket_id="b2", expected_revision=0,
        body="L1-R2 原始记忆正文。", metadata={"name": "Gateway L2 contract"},
        source_refs=[], decision_source="p4-contract-test",
        idempotency_key="gateway-contract-seed", actor="test",
    ))
    body, semantic = split_document(document)
    run(commit.commit_memory(
        memory_id="m2", bucket_id="b2", expected_revision=1,
        body=body, metadata={"name": "Gateway L2 contract", "memory_l1": semantic},
        source_refs=[], decision_source="p4-contract-test",
        idempotency_key="gateway-contract-l1", actor="test", l1_write_intent=True,
    ))
    service.engine._memory_l1_sources = lambda: sources
    return sources


def _record_for_gateway(service, authority):
    detail = run(service.revisions.detail("m2"))
    bucket = run(service.revisions.buckets.get("b2"))
    memory = authority.get_memory("m2")
    record = {
        "state": "l1", "memory_id": "m2", "revision": detail["current_revision"],
        "historical": False, "document": detail["l1"]["document"],
    }
    metadata = {
        "memory_id": "m2", "revision": detail["current_revision"],
        "body_sha256": memory["body_sha256"],
    }
    return bucket, record, metadata


def _gateway_service(view, token):
    def wrapper(source_id, *, runtime_values):
        if source_id != "ombre.memory_usage_card_prompt":
            return ""
        from prompt_source_registry import factory_body, render_runtime_placeholders
        return render_runtime_placeholders(source_id, factory_body(source_id), runtime_values)

    return NS(
        config={"memory_system": {}}, gateway_token=token,
        memory_authority_view=view, memory_authority_enabled=True,
        _extract_bucket_ids_from_context=lambda text: list(dict.fromkeys(
            re.findall(r"\[bucket_id:([^\]\s]+)\]", str(text or "")))),
        _direct_bucket_header=lambda bucket, _moment: f'[bucket_id:{bucket["id"]}] Gateway contract',
        _resolve_gateway_fixed_prompt=wrapper,
        _trim_text=lambda text, budget: text if budget > 0 else "",
    )


def _gateway_request(query):
    now = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
    return {
        "q_current": query, "q_context": query, "metadata": {"query_views": []},
        "_use_context": {
            "now": now, "target_time": None, "query": query,
            "purpose": "history", "overview": False,
            "facts": {"hour": {"state": "known", "value": 12},
                      "weekday": {"state": "known", "value": 5}},
        },
    }


class _BrainReadTransport(httpx.AsyncBaseTransport):
    def __init__(self, l2_service, reads):
        self.l2_service = l2_service
        self.reads = reads
        self.calls = []
        self.responses = []

    async def handle_async_request(self, request):
        payload = json.loads(await request.aread())
        self.calls.append({"path": request.url.path, "payload": payload})
        if request.url.path.endswith("/recall-sources"):
            result = await self.reads.recall_source_status(payload["items"])
        elif request.url.path.endswith("/presentations/resolve"):
            result = await self.l2_service.resolve(payload)
        else:
            return httpx.Response(404, json={"error": "unexpected_brain_route"}, request=request)
        self.responses.append({"path": request.url.path, "body": result})
        return httpx.Response(200, json=result, request=request)


def test_real_brain_resolve_flows_through_gateway_and_stale_l2_falls_back(tmp_path, monkeypatch):
    runtime = _gateway_runtime(monkeypatch)
    l2_service, authority, model_calls = setup_l2(tmp_path)
    sources = _source_backed_l1(l2_service, authority)
    generated = run(l2_service.run_once(10))
    assert any(row.get("task_id") for row in generated["results"])
    assert len(model_calls) >= 1

    # Model output is intentionally pending review in the fixture. Save an
    # owner-authored revision so Brain's real resolver returns ready L2 text.
    initial = l2_service.detail("m2")["l2"]
    owner_payload = {
        "expected_l1_revision": initial["current_dependencies"]["first_pass_revision"],
        "expected_presentation_revision": initial["current_presentation_revision"],
        "request_id": "gateway-contract-owner-l2",
        "change_kind": "wording",
        "document": copy.deepcopy(initial["presentation"]["document"]),
    }
    owner_payload["document"]["full_injection_text"] = "L2-PERSISTED-V2: complete memory for the chat turn."
    saved = run(l2_service.save("m2", owner_payload))
    assert saved["presentation_revision"] == 2
    assert l2_service.detail("m2")["l2"]["state"] == "ready"

    reads = MemoryL1Reads(l2_service.revisions, sources)
    brain_transport = _BrainReadTransport(l2_service, reads)
    original_client = runtime.httpx.AsyncClient

    def mock_client(*args, **kwargs):
        kwargs["transport"] = brain_transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(runtime.httpx, "AsyncClient", mock_client)
    monkeypatch.setenv("OMBRE_MEMORY_INDEX_ADMIN_URL", "http://brain.contract.test")
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "brain-contract-read-token")

    view = NS(enabled=True)
    record_holder = {}
    view.memory_index_metadata_map = lambda ids: (
        {"b2": record_holder["metadata"]} if "b2" in ids else {}
    )
    view.memory_l1_recall_map = lambda ids: (
        {"b2": record_holder["record"]} if "b2" in ids else {}
    )
    gateway = _gateway_service(view, "brain-contract-read-token")

    def project_current(budget=4000):
        bucket, record, metadata = _record_for_gateway(l2_service, authority)
        record_holder.update(bucket=bucket, record=record, metadata=metadata)
        query = "回想那次她说想要抱抱。"
        request = _gateway_request(query)
        raw = f'[bucket_id:b2] {bucket["content"]}'
        sections = run(runtime.project_sections(
            gateway, request, query, [bucket],
            {"direct": (raw, ["b2"], budget),
             "targeted": ("", [], 0), "diffused": ("", [], 0)},
        ))
        return request, sections

    request, sections = project_current()
    assert "L2-PERSISTED-V2: complete memory for the chat turn." in sections["direct"]
    assert "L1-R2 原始记忆正文。" not in sections["direct"]
    trace = request["presentation_trace"][0]
    assert trace["source"] == "l2"
    assert trace["presentation_revision"] == saved["presentation_revision"]
    resolver_item = brain_transport.responses[-1]["body"]["items"][0]
    assert isinstance(resolver_item["units"], list)
    assert all({"unit_id", "text", "statement_refs"} <= unit.keys()
               for unit in resolver_item["units"])
    if resolver_item["units"]:
        assert trace["selected_unit_ids"] == [
            unit["unit_id"] for unit in resolver_item["units"]
        ]
    assert [call["path"].rsplit("/", 1)[-1] for call in brain_transport.calls[-2:]] == [
        "recall-sources", "resolve",
    ]
    resolve_call = next(call for call in reversed(brain_transport.calls)
                        if call["path"].endswith("/presentations/resolve"))
    assert resolve_call["payload"]["items"][0]["statement_uses"]
    assert len(model_calls) >= 1

    # An owner full-text edit is locally indexed into complete new paragraphs.
    # The existing Gateway budget can now choose the short exact paragraph;
    # neither stale model wording nor a slice of the long paragraph is used.
    owner_payload['expected_presentation_revision'] = saved['presentation_revision']
    owner_payload['request_id'] = 'gateway-contract-owner-paragraphs'
    long_paragraph = 'L2-LONG-OWNER-PARAGRAPH: ' + 'complete long owner paragraph. ' * 600
    short_paragraph = 'L2-SHORT-OWNER-PARAGRAPH: she can decline; I will respect her decision.'
    owner_payload['document'] = copy.deepcopy(saved['l2']['presentation']['document'])
    owner_payload['document']['full_injection_text'] = long_paragraph + '\n\n' + short_paragraph
    next_saved = run(l2_service.save('m2', owner_payload))
    request, sections = project_current(budget=160)
    assert short_paragraph in sections['direct']
    assert 'L2-LONG-OWNER-PARAGRAPH' not in sections['direct']
    assert 'L2-PERSISTED-V2' not in sections['direct']
    trace = request['presentation_trace'][0]
    assert trace['source'] == 'l2'
    assert trace['presentation_revision'] == next_saved['presentation_revision']
    assert trace['selected_unit_ids'] == [next_saved['l2']['presentation']['document']['coherent_sections'][1]['section_id']]
    assert trace['partial_presentation'] is True

    # A new L1 dependency makes the persisted L2 stale. The same live transport
    # returns Brain's stale result, and Gateway restores only its existing L1 path.
    current = run(l2_service.revisions.detail("m2"))
    l1_edit = {
        "expected_revision": current["current_revision"],
        "request_id": "gateway-contract-l1-next",
        "document": copy.deepcopy(current["l1"]["document"]),
    }
    l1_edit["document"]["narrative"]["body"] += " L1-R3 narrative after the change."
    run(l2_service.revisions.save("m2", l1_edit))
    request, sections = project_current()
    assert "L1-R3 narrative after the change." in sections["direct"]
    assert "L2-PERSISTED-V2: complete memory for the chat turn." not in sections["direct"]
    trace = request["presentation_trace"][0]
    assert trace["resolver_state"] == "stale"
    assert trace["source"] == "l1"
    assert trace["presentation_revision"] is None
    assert trace["fallback_reason"] == "dependency_changed"
    assert len(model_calls) >= 1
    assert sum(call["path"].endswith("/recall-sources") for call in brain_transport.calls) == 3
    assert sum(call["path"].endswith("/presentations/resolve") for call in brain_transport.calls) == 3
