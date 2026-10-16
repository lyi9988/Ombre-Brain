"""End-to-end chat preparation checks against the configured canonical reader."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from chat_memory_tool import candidate_id
from memory_l0_sources import MemoryL0Sources
from memory_l1_ingestion import bind_sources
from memory_l1_schema import join_document
from memory_l1_source_readers import configured_source_readers
from memory_source_provenance import canonical_source
from test_chat_memory_preparation import setup as setup_chat_preparation


BRIDGE_TOKEN = "synthetic-chat-memory-bridge-token"
BRIDGE_DEVICE = "ombre-chat-memory-source-test"
CONVERSATION_ID = "conv-chat-memory-source-test"
BRIDGE_URL = "https://canonical-bridge.invalid"


def _configured_case(tmp_path, monkeypatch):
    env, engine, envelope, model_calls = setup_chat_preparation(tmp_path)
    envelope["context"]["conversation_id"] = CONVERSATION_ID
    engine.config["gateway"] = dict(engine.config.get("gateway") or {})
    engine.config["gateway"]["canonical_continuation"] = {
        "enabled": True,
        "base_url": BRIDGE_URL,
        "conversation_id": CONVERSATION_ID,
        "device_id": BRIDGE_DEVICE,
    }
    monkeypatch.setenv("OMBRE_CANONICAL_BRIDGE_TOKEN", BRIDGE_TOKEN)

    readers = MemoryL0Sources(
        env["authority"],
        configured_source_readers(engine.config, env["authority"]),
    )
    engine._memory_l1_sources = lambda: readers

    canonical_sources = [
        canonical_source(
            source,
            conversation_id=CONVERSATION_ID,
            profile_id=engine.memory_semantic_profile_id,
        )
        for source in envelope["context"]["sources"]
    ]
    _snapshots, bindings = bind_sources(canonical_sources)
    sources_by_id = {source["event_id"]: source for source in canonical_sources}
    bindings_by_id = {
        json.loads(binding["source_id"])["event_id"]: binding
        for binding in bindings
    }
    assert all(readers.copy(binding) is None for binding in bindings_by_id.values())
    bridge_requests = []
    return env, engine, envelope, model_calls, readers, sources_by_id, bindings_by_id, bridge_requests


def _install_mock_transport(monkeypatch, handler):
    """Keep the real AsyncClient behavior while routing its traffic to MockTransport."""
    real_async_client = httpx.AsyncClient

    def async_client(*args, **kwargs):
        assert "transport" not in kwargs
        return real_async_client(
            *args,
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", async_client)


def _assert_bridge_request(request, bridge_requests, sources_by_id, readers, bindings_by_id):
    assert request.url.path == "/bridge/v1/memory-source"
    assert request.headers.get("Authorization") == f"Bearer {BRIDGE_TOKEN}"
    assert request.headers.get("X-Guyan-Device-ID") == BRIDGE_DEVICE
    params = request.url.params
    assert params.get("kind") == "canonical_event"
    source_id = params.get("source_id")
    pinned = sources_by_id[source_id]
    assert params.get("version") == pinned["version_id"]

    # The task must retain every authenticated snapshot before its first live read.
    if not bridge_requests:
        for event_id, source in sources_by_id.items():
            initial_copy = readers.copy(bindings_by_id[event_id])
            assert initial_copy is not None
            assert initial_copy["upstream_revision"] == source["version_id"]
    retained = readers.copy(bindings_by_id[source_id])
    assert retained is not None
    assert retained["upstream_revision"] == pinned["version_id"]
    bridge_requests.append({"source_id": source_id, "version": params.get("version")})
    return pinned


def _available_bridge_response(request, source, version=None):
    return httpx.Response(
        200,
        json={
            "source": {
                "state": "available",
                "eligible_source": True,
                "source_namespace": "snapshot.aiz.canonical_event",
                "source_id": source["event_id"],
                "source_revision": version or source["version_id"],
                "text_view_id": "snapshot.aiz.canonical_event.content.exact-v1",
                "text": source["text"],
                "permission": {
                    "authorized": True,
                    "scope": "canonical.fixed_owner_conversation",
                },
                "coverage": {"complete": True, "source_length_chars": len(source["text"])},
            }
        },
        request=request,
    )


def test_chat_preparation_reads_retained_canonical_versions_and_leaves_complete_l1_pending(
    tmp_path, monkeypatch
):
    (
        env,
        engine,
        envelope,
        model_calls,
        readers,
        sources_by_id,
        bindings_by_id,
        bridge_requests,
    ) = _configured_case(tmp_path, monkeypatch)

    def bridge(request):
        source = _assert_bridge_request(
            request, bridge_requests, sources_by_id, readers, bindings_by_id
        )
        return _available_bridge_response(request, source)

    _install_mock_transport(monkeypatch, bridge)
    from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks

    tasks = ChatMemoryPreparation(env["authority"])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env["buckets"]))

    assert result["results"][0]["memory_status"] == "pending"
    assert tasks.receipt(candidate_id(envelope["operation_id"]))["memory_status"] == "pending"
    assert len(model_calls) == 1
    assert len(bridge_requests) == 2 * len(sources_by_id)
    assert all(item["version"] == sources_by_id[item["source_id"]]["version_id"]
               for item in bridge_requests)

    row = env["authority"].get_candidate(candidate_id(envelope["operation_id"]))
    assert row["status"] == "pending"
    assert row["proposal"]["proposed_body"] == envelope["arguments"]["narrative"]["body"]
    l1 = row["proposal"]["metadata"]["memory_l1"]
    complete_document = join_document(row["proposal"]["proposed_body"], l1)
    assert complete_document["narrative"]["body"] == envelope["arguments"]["narrative"]["body"]
    assert l1["generation"]["actor"] == "chat_memory_preparation"
    assert len(l1["evidence_bindings"]) == len(sources_by_id)
    assert all(binding["source_namespace"] == "snapshot.aiz.canonical_event"
               for binding in l1["evidence_bindings"])
    assert l1["statements"][0]["evidence_refs"] == ["source:2"]
    assert env["authority"].list_memories() == []


@pytest.mark.parametrize(
    "outcome",
    ["withdrawn", "changed", "owner_excluded", "unauthorized"],
)
def test_retained_chat_source_copy_does_not_make_withdrawn_or_unavailable_source_eligible(
    tmp_path, monkeypatch, outcome
):
    (
        env,
        engine,
        envelope,
        model_calls,
        readers,
        sources_by_id,
        bindings_by_id,
        bridge_requests,
    ) = _configured_case(tmp_path, monkeypatch)
    first_source_id = envelope["context"]["sources"][0]["event_id"]

    def bridge(request):
        source = _assert_bridge_request(
            request, bridge_requests, sources_by_id, readers, bindings_by_id
        )
        if source["event_id"] != first_source_id:
            return _available_bridge_response(request, source)
        if outcome == "unauthorized":
            return httpx.Response(
                401,
                json={"error": {"code": "unauthorized"}},
                request=request,
            )
        state = {
            "withdrawn": ("withdrawn", "owner_withdrew_source"),
            "changed": ("changed", "source_version_changed"),
            "owner_excluded": ("unavailable", "owner_excluded_source"),
        }[outcome]
        return httpx.Response(
            200,
            json={
                "source": {
                    "state": state[0],
                    "eligible_source": False,
                    "source_namespace": "snapshot.aiz.canonical_event",
                    "source_id": source["event_id"],
                    "source_revision": "version-current" if outcome == "changed" else source["version_id"],
                    "reason": state[1],
                    "permission": {
                        "authorized": outcome != "owner_excluded",
                        "scope": "" if outcome == "owner_excluded" else "canonical.fixed_owner_conversation",
                    },
                }
            },
            request=request,
        )

    _install_mock_transport(monkeypatch, bridge)
    from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks

    tasks = ChatMemoryPreparation(env["authority"])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env["buckets"]))

    assert result["results"][0]["issue"]["code"] == "source_not_currently_readable"
    assert tasks.receipt(candidate_id(envelope["operation_id"]))["memory_status"] == "preparation_failed"
    assert len(model_calls) == 0
    assert len(bridge_requests) == 1
    # All copies were retained before locate rejected the first live source.
    for binding in bindings_by_id.values():
        retained = readers.copy(binding)
        assert retained is not None
        assert retained["upstream_revision"] == sources_by_id[
            json.loads(binding["source_id"])["event_id"]
        ]["version_id"]
    assert env["authority"].list_candidates(status="all") == []


def test_canonical_revision_mismatch_does_not_retry_as_current_or_call_model(tmp_path, monkeypatch):
    (
        env,
        engine,
        envelope,
        model_calls,
        readers,
        sources_by_id,
        bindings_by_id,
        bridge_requests,
    ) = _configured_case(tmp_path, monkeypatch)
    first_source_id = envelope["context"]["sources"][0]["event_id"]

    def bridge(request):
        source = _assert_bridge_request(
            request, bridge_requests, sources_by_id, readers, bindings_by_id
        )
        if source["event_id"] != first_source_id:
            return _available_bridge_response(request, source)
        return httpx.Response(
            200,
            json={
                "source": {
                    "state": "changed",
                    "eligible_source": False,
                    "source_namespace": "snapshot.aiz.canonical_event",
                    "source_id": source["event_id"],
                    "source_revision": "version-current",
                    "requested_revision": source["version_id"],
                    "reason": "source_version_changed",
                    "permission": {"authorized": True, "scope": "canonical.fixed_owner_conversation"},
                }
            },
            request=request,
        )

    _install_mock_transport(monkeypatch, bridge)
    from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks

    tasks = ChatMemoryPreparation(env["authority"])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env["buckets"]))

    assert result["results"][0]["issue"]["code"] == "source_not_currently_readable"
    assert len(model_calls) == 0
    assert bridge_requests == [{"source_id": first_source_id,
                                "version": sources_by_id[first_source_id]["version_id"]}]
    assert all(item["version"] != "current" for item in bridge_requests)
    assert env["authority"].list_candidates(status="all") == []
