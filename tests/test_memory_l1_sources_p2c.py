"""Synthetic cross-worktree checks for exact L0 source qualification.

The Aizizhu bridge handler and Brain source consumers run against temporary
SQLite stores and a loopback-only aiohttp server. No model/provider is used.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


_AIZ_WORKTREE = Path(os.environ.get(
    "GUYAN_AIZ_WORKTREE",
    str(Path(__file__).resolve().parents[2] / "aizizhu-memory-chat-submit-r1"),
))
if not _AIZ_WORKTREE.is_dir():
    pytest.skip("Aizizhu P2 worktree is not present for this cross-worktree test",
                allow_module_level=True)
sys.path.insert(0, str(_AIZ_WORKTREE))

from core.canonical_bridge_api import (  # noqa: E402
    CanonicalBridgeHandlers,
    register_canonical_bridge_routes,
)
from core.chat_presentation import ChatPresentationStore  # noqa: E402
from core.conversation_store import ConversationStore  # noqa: E402
from core.diary_revisions import DiaryRevisionStore  # noqa: E402
from core import goals as goals_module  # noqa: E402
from core.goals import GoalManager  # noqa: E402
from core.life_actions import LifeActionStore  # noqa: E402
from core.living_projection import LivingProjection  # noqa: E402
from memory_authority import MemoryAuthorityStore  # noqa: E402
from memory_l0_sources import MemoryL0Sources  # noqa: E402
from memory_l1_contract import canonical_json  # noqa: E402
from memory_l1_ingestion import bind_sources  # noqa: E402
from memory_l1_source_readers import configured_source_readers  # noqa: E402
from memory_source_provenance import canonical_source  # noqa: E402


BRIDGE_TOKEN = "synthetic-memory-source-bridge-token"
DEVICE_ID = "ombre-gateway-source-test"
CONVERSATION_ID = "conv-p2c-source-test"
DAY = "2026-10-14"


async def _start_bridge(
    tmp_path, monkeypatch, *, room=None, life_actions=None, presentation_store=None,
    diary_namespace="snapshot.aiz.diary.private",
):
    monkeypatch.setenv("OMBRE_CANONICAL_BRIDGE_TOKEN", BRIDGE_TOKEN)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    conversation_store = ConversationStore(tmp_path / "conversation.sqlite3")
    diary_store = DiaryRevisionStore(tmp_path / "diary.sqlite3")
    fingerprints = {}
    loop = SimpleNamespace(
        diary_revisions=diary_store,
        now=lambda: datetime(2026, 10, 15, 12, tzinfo=timezone.utc),
        _diary_fingerprint=lambda kind, day: fingerprints[(kind, day)],
        tools={"room": room} if room is not None else {},
        life_actions=life_actions,
        conversation=SimpleNamespace(
            service=SimpleNamespace(presentation_store=presentation_store)
        ),
    )
    handlers = CanonicalBridgeHandlers(
        store=conversation_store,
        identity_id="jiajia-main",
        conversation_id=CONVERSATION_ID,
        device_id=DEVICE_ID,
        agent_loop=loop,
    )

    @web.middleware
    async def test_only_auth(request, handler):
        if (request.headers.get("Authorization") != f"Bearer {BRIDGE_TOKEN}"
                or request.headers.get("X-Guyan-Device-ID") != DEVICE_ID):
            raise web.HTTPUnauthorized()
        return await handler(request)

    app = web.Application(middlewares=[test_only_auth])
    register_canonical_bridge_routes(app, handlers)
    client = TestClient(TestServer(app, host="127.0.0.1"))
    await client.start_server()
    config = {
        "state_dir": str(tmp_path / "state"),
        "buckets_dir": str(tmp_path / "buckets"),
        "gateway": {
            "canonical_continuation": {
                "enabled": True,
                "base_url": str(client.make_url("/")),
                "conversation_id": CONVERSATION_ID,
                "device_id": DEVICE_ID,
            }
        },
    }
    authority = MemoryAuthorityStore(config)
    readers = configured_source_readers(config, authority)
    sources = MemoryL0Sources(
        authority, readers, diary_namespace=diary_namespace
    )
    return client, conversation_store, diary_store, fingerprints, authority, sources, config


async def _post_message(client, source_event_id, role, content, correlation_id=None):
    payload = {"source_event_id": source_event_id, "role": role, "content": content}
    if correlation_id is not None:
        payload["correlation_id"] = correlation_id
    response = await client.post(
        "/bridge/v1/events",
        json=payload,
        headers={
            "Authorization": f"Bearer {BRIDGE_TOKEN}",
            "X-Guyan-Device-ID": DEVICE_ID,
        },
    )
    assert response.status == 201
    return (await response.json())["event"]


def _canonical_binding(store, sources, event):
    state = store.memory_source_event_state(CONVERSATION_ID, event["id"])
    source = canonical_source(
        {
            "event_id": event["id"],
            "version_id": state["version_id"],
            "role": event["role"],
            "text": event["content"],
            "created_at": None,
        },
        conversation_id=CONVERSATION_ID,
        profile_id="synthetic-owner",
    )
    snapshots, bindings = bind_sources([source])
    sources.retain_copy(snapshots[0])
    return state, bindings[0]


def test_canonical_source_reader_checks_versions_withdrawal_and_retained_copy(tmp_path, monkeypatch):
    async def scenario():
        client, store, _diaries, _fps, _authority, sources, _config = await _start_bridge(
            tmp_path, monkeypatch
        )
        try:
            await _post_message(
                client, "p2c:source:user", "user", "合成用户来源。"
            )
            assistant = await _post_message(
                client, "p2c:source:assistant", "assistant", "合成原始答复。",
                correlation_id="p2c:source:user",
            )
            event = store.get_event_by_source("ombre-gateway", "p2c:source:assistant")
            before, binding = _canonical_binding(store, sources, event)

            current = await sources.locate(binding)
            assert current["eligible_source"] is True
            assert current["view"] == "source"
            assert current["text"] == "合成原始答复。"

            edited = store.owner_edit_assistant(
                CONVERSATION_ID,
                target_event_id=event["id"],
                new_content="合成主人修订。",
                source="test",
                source_event_id="p2c:source:edit",
                request_id="p2c-source-edit",
                operator="owner",
            )
            after_edit = store.memory_source_event_state(CONVERSATION_ID, event["id"])
            assert after_edit["context_revision"] > before["context_revision"]
            stale = await sources.locate(binding)
            assert stale["eligible_source"] is False
            assert stale["view"] == "source_copy"
            assert stale["text"] == "合成原始答复。"

            edited_event = edited["owner_edited_event"]
            edited_state, edited_binding = _canonical_binding(store, sources, edited_event)
            withdrawn = store.withdraw_message(
                CONVERSATION_ID,
                target_event_id=edited_event["id"],
                source_event_id="p2c:source:withdraw",
            )
            assert withdrawn["created"] is True
            after_withdrawal = store.memory_source_event_state(
                CONVERSATION_ID, edited_event["id"]
            )
            assert after_withdrawal["context_revision"] > edited_state["context_revision"]
            denied = await sources.locate(edited_binding)
            assert denied["eligible_source"] is False
            assert denied["view"] == "source_copy"
            assert denied["text"] == "合成主人修订。"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_unknown_canonical_binding_without_retained_copy_is_ineligible(tmp_path, monkeypatch):
    async def scenario():
        client, _store, _diaries, _fps, _authority, sources, _config = await _start_bridge(
            tmp_path, monkeypatch
        )
        try:
            binding = {
                "binding_id": "source:unknown",
                "source_namespace": "snapshot.aiz.canonical_event",
                "source_id": canonical_json({
                    "system": "aizizhu",
                    "profile_id": "synthetic-owner",
                    "conversation_id": CONVERSATION_ID,
                    "event_id": "missing-event",
                    "role": "user",
                }),
                "source_revision": "source_sha256:unknown",
                "anchors": [],
            }
            result = await sources.locate(binding)
            assert result["eligible_source"] is False
            assert result["view"] == "source_copy"
            assert result["text"] is None
            assert result["reason"] == "upstream_version_unknown"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_private_diary_reader_pins_revision_and_never_promotes_history(tmp_path, monkeypatch):
    async def scenario():
        client, _store, diary_store, fingerprints, _authority, sources, _config = await _start_bridge(
            tmp_path, monkeypatch
        )
        try:
            fingerprint = {"source_sha256": "synthetic-diary-r1", "source_ids": ["fixture-r1"]}
            fingerprints[("private", DAY)] = fingerprint
            prompt = diary_store.prompt_snapshot("private")
            first = diary_store.record_auto(
                "private", DAY, title="合成私人日记", body="第一版合成正文。",
                fingerprint=fingerprint, prompt=prompt, model_route="synthetic-route",
                request_id="synthetic-private-r1", input_coverage={"complete": True},
            )

            captured = await sources.capture_diary(DAY, binding_id="private-source")
            assert captured["source"]["source_revision"] == first["revision_id"]
            first_current = await sources.locate(captured["binding"])
            assert first_current["eligible_source"] is True
            assert first_current["text"] == "第一版合成正文。"

            second = diary_store.record_auto(
                "private", DAY, title="合成私人日记", body="第二版合成正文。",
                fingerprint=fingerprint, prompt=prompt, model_route="synthetic-route",
                request_id="synthetic-private-r2", operator="owner-device",
                input_coverage={"complete": True},
            )
            stale = await sources.locate(captured["binding"])
            assert stale["eligible_source"] is False
            assert stale["view"] == "source_copy"
            assert stale["text"] == "第一版合成正文。"

            latest = await sources.capture_diary(DAY, binding_id="private-source-r2")
            assert latest["source"]["source_revision"] == second["revision_id"]
            current = await sources.locate(latest["binding"])
            assert current["eligible_source"] is True
            assert current["text"] == "第二版合成正文。"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_shared_diary_source_hash_and_identity_changes_only_return_source_copy(tmp_path, monkeypatch):
    class SyntheticRoom:
        projection_secret = "synthetic-room-secret"
        projection_secret_version = "v1"

        def __init__(self):
            self.item = {
                "id": "synthetic-room-diary",
                "date": DAY,
                "title": "合成共同日记",
                "content": "共同来源初始正文。",
                "authorName": "合成参与者",
                "createdAt": DAY + "T12:00:00Z",
                "updatedAt": DAY + "T12:00:00Z",
                "mood": "calm",
            }
            self.calls = []

        async def read_data(self, item_type, date=""):
            self.calls.append((item_type, date))
            return {"items": [self.item]}

    async def scenario():
        room = SyntheticRoom()
        client, _store, _diaries, _fps, _authority, sources, _config = await _start_bridge(
            tmp_path,
            monkeypatch,
            room=room,
            diary_namespace="snapshot.ourroom.shared_diary",
        )
        try:
            projection_id = LivingProjection._room_entry(
                "shared_diary",
                room.item,
                room.projection_secret,
                LivingProjection._room_projection_secret_version(room),
            )["projection_id"]
            source_id = json.dumps(
                {"date": DAY, "projection_id": projection_id},
                sort_keys=True,
                separators=(",", ":"),
            )
            captured = await sources.capture_diary(
                source_id, binding_id="shared-source"
            )
            assert captured["binding"] is not None
            initial_text = captured["source"]["text"]
            initial_revision = captured["source"]["source_revision"]
            assert initial_revision.startswith("room-view-sha256:")
            current = await sources.locate(captured["binding"])
            assert current["eligible_source"] is True
            assert current["text"] == initial_text

            # Content changes keep the opaque projection identity but change
            # the exact source hash; the old binding remains historical.
            room.item["content"] = "共同来源已修改正文。"
            changed = await sources.locate(captured["binding"])
            assert changed["eligible_source"] is False
            assert changed["view"] == "source_copy"
            assert changed["text"] == initial_text
            assert changed["text"] != room.item["content"]

            # Replacing the Room item identity also cannot substitute the
            # replacement into the old source binding.
            room.item["id"] = "replacement-room-item"
            replaced = await sources.locate(captured["binding"])
            assert replaced["eligible_source"] is False
            assert replaced["view"] == "source_copy"
            assert replaced["text"] == initial_text
            assert all(kind == "diaries" for kind, _ in room.calls)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_action_receipt_requires_executor_artifact_and_expires_to_copy(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            goals_module, "GOALS_FILE", tmp_path / "goals.json"
        )
        goals = GoalManager()
        life_actions = LifeActionStore(goals)
        presentation = ChatPresentationStore(tmp_path / "presentation.sqlite3")
        client, _store, _diaries, _fps, _authority, sources, _config = await _start_bridge(
            tmp_path,
            monkeypatch,
            life_actions=life_actions,
            presentation_store=presentation,
        )
        try:
            action_reader = next(
                reader for reader in sources.adapters.values()
                if reader.namespace == "snapshot.aiz.action_receipt"
            )

            planned = life_actions.create(
                {
                    "title": "合成未完成行动",
                    "assignee": "guyan",
                    "tool_id": "browse.search",
                    "arguments": {"query": "synthetic"},
                },
                operation_id="source-planned-create",
                scope="owner",
                actor="owner",
                source="synthetic-test",
            )
            incomplete = await client.get(
                "/bridge/v1/memory-source",
                params={
                    "kind": "action_receipt",
                    "source_id": planned["id"],
                    "version": "current",
                },
                headers={
                    "Authorization": f"Bearer {BRIDGE_TOKEN}",
                    "X-Guyan-Device-ID": DEVICE_ID,
                },
            )
            incomplete_source = (await incomplete.json())["source"]
            assert incomplete_source["eligible_source"] is False
            assert incomplete_source["reason"] == "executor_completion_receipt_required"

            def create_completed_action(suffix, *, artifact_source_refs):
                action = life_actions.create(
                    {
                        "title": f"合成完成行动-{suffix}",
                        "assignee": "guyan",
                        "tool_id": "browse.search",
                        "arguments": {"query": "synthetic"},
                    },
                    operation_id=f"source-{suffix}-create",
                    scope="owner",
                    actor="owner",
                    source="synthetic-test",
                )
                body = {
                    "text": json.dumps(
                        {"ok": True, "results": [f"synthetic-{suffix}"]},
                        ensure_ascii=False,
                    ),
                    "truncated": False,
                }
                artifact = presentation.upsert_artifact(
                    CONVERSATION_ID,
                    request_id=f"source-{suffix}-request",
                    turn_id=f"source-{suffix}-turn",
                    tool_call_id=f"source-{suffix}-tool-call",
                    provider="synthetic",
                    tool="browse.search",
                    kind="planned_action_result",
                    status="committed",
                    source_refs=artifact_source_refs(action["id"]),
                    body=body,
                )
                claimed = life_actions.claim(
                    action["id"],
                    expected_revision=action["revision"],
                    now=datetime(2026, 10, 15, 12, tzinfo=timezone.utc),
                )
                completed = life_actions.finish(
                    action["id"],
                    claimed["run_id"],
                    outcome="completed",
                    artifact_id=artifact["artifact_id"],
                )
                assert completed["completion_kind"] == "executor"
                return action, artifact, body

            mismatched, bad_artifact, _ = create_completed_action(
                "mismatch",
                artifact_source_refs=lambda _action_id: ["action-ffffffffffffffff"],
            )
            bad = await action_reader.capture(
                mismatched["id"], bad_artifact["artifact_id"]
            )
            assert bad["eligible_source"] is False
            assert bad["reason"] == "authoritative_result_artifact_unavailable"

            action, artifact, body = create_completed_action(
                "valid",
                artifact_source_refs=lambda action_id: [action_id],
            )
            accepted = await action_reader.capture(
                action["id"], artifact["artifact_id"]
            )
            assert accepted["state"] == "available"
            assert accepted["eligible_source"] is True
            assert accepted["source_namespace"] == "snapshot.aiz.action_receipt"

            source = {
                "system": "aizizhu",
                "namespace": "action_receipt",
                "profile_id": "synthetic-owner",
                "conversation_id": CONVERSATION_ID,
                "event_id": action["id"],
                "version_id": artifact["artifact_id"],
                "role": "tool",
                "text": accepted["text"],
                "recorded_at": accepted.get("recorded_at"),
            }
            snapshots, bindings = bind_sources([source])
            sources.retain_copy(snapshots[0])
            current = await sources.locate(bindings[0])
            assert current["eligible_source"] is True
            assert current["text"] == body["text"]

            presentation.upsert_artifact(
                CONVERSATION_ID,
                request_id="source-valid-request",
                turn_id="source-valid-turn",
                tool_call_id="source-valid-tool-call",
                provider="synthetic",
                tool="browse.search",
                kind="planned_action_result",
                status="committed",
                source_refs=[action["id"]],
                body=body,
                ttl_ms=0,
            )
            historical = await sources.locate(bindings[0])
            assert historical["eligible_source"] is False
            assert historical["view"] == "source_copy"
            assert historical["text"] == body["text"]
        finally:
            await client.close()

    asyncio.run(scenario())
