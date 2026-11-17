"""Local Authority/Bucket receipts with exact owner bindings and an isolated model seam."""
import asyncio
from datetime import datetime, timezone
import json
import sqlite3
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore, MemoryProposal, PolicyDecision
from memory_authority_view import MemoryAuthorityRecallView, portrait_l1_v2_block_reason
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_l0_sources import MemoryL0Sources
from memory_l1_ingestion import bind_sources
from memory_l1_schema import split_document
from memory_l1_v2 import prepare_card
from memory_relation_dependencies import cache_versions
from portrait_engine import DailyPortraitMaintainer
from user_portrait_chat import build_user_portrait_projection


class SourceReader:
    namespace = "snapshot.aiz.canonical_event"

    def __init__(self):
        self.rows = {}
        self.fail = set()
        self.reads = []
        self.on_read = None

    async def read(self, binding):
        key = binding["source_id"]
        self.reads.append(key)
        if self.on_read is not None:
            self.on_read()
        if key in self.fail:
            raise TimeoutError
        return self.rows.get(key, {"state": "withdrawn"})


@pytest.fixture
def env(tmp_path):
    config = {
        "buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state"),
        "memory_authority": {"enabled": True}, "persona": {"profile_id": "test"},
        "gateway": {"canonical_continuation": {"enabled": True, "conversation_id": "test-owner"}},
    }
    authority = MemoryAuthorityStore(config)
    buckets = BucketManager(config)
    adapter = SourceReader()
    reader = MemoryL0Sources(authority, [adapter])
    engine = DailyPortraitMaintainer(config, l1_source_reader=reader)
    engine.client = object()  # Never invokes a provider.
    now = datetime(2026, 11, 15, 8, tzinfo=engine.tz)
    local_now = engine._local_now
    engine._local_now = lambda value=None: local_now(value or now)
    state = engine.load_state()
    state.update(runs=[{"date": "2026-11-13"}], last_run_date="2026-11-13")
    engine.save_state(state)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, buckets))
    return SimpleNamespace(engine=engine, adapter=adapter, reader=reader, buckets=buckets, commit=commit,
                           authority=authority, config=config, payloads=[])


def card(env, *, key, revision=1, quote="我希望每次发版前先看预览。", role="user", stability="stable",
         kind="boundary"):
    raw = {"system": "aizizhu", "namespace": "canonical_event", "profile_id": "test",
           "conversation_id": "test-owner", "event_id": f"{key}:{revision}", "version_id": "1",
           "recorded_at": "2026-10-02T09:00:00+08:00", "role": role, "text": quote}
    snapshots, bindings = bind_sources([raw], config=env.config)
    snapshot = snapshots[0]
    # The people list contains the owner even in the assistant-only case.
    owner = {"ref": "owner", "person_id": env.engine._portrait_owner_id(), "name": "测试主人", "kind": "human"}
    document = prepare_card({
        "title": "十月的谈话", "body": "2026年10月2日，我记下了此前谈话的内容。",
        "kind": kind, "stability": stability, "keywords": ["过去"], "people": ["owner"],
        "sources": ["s1"], "primary_topic": "interaction",
        "event_time": {"start": "2026-10-02", "precision": "day", "sources": ["s1"]},
    }, people=[owner], bindings=bindings, generation={"origin": "auto_l1", "task_id": "test"},
        source_ref_map={"s1": bindings[0]["binding_id"]})
    env.adapter.rows[snapshot["source_id"]] = snapshot
    env.reader.retain_copy(snapshot)
    body, semantic = split_document(document)
    return body, {"name": document["title"], "memory_l1": semantic}, snapshot


async def adopt(env, key, *, revision=1, prepared=None, **kwargs):
    body, metadata, snapshot = prepared or card(env, key=key, revision=revision, **kwargs)
    result = await env.commit.commit_memory(
        memory_id=key, bucket_id=key, expected_revision=revision - 1, body=body, metadata=metadata,
        source_refs=[snapshot["source_id"]], decision_source="owner" if revision == 1 else "owner_l1_edit",
        idempotency_key=f"{key}:{revision}", actor="owner", l1_write_intent=True,
    )
    assert result["revision"] == revision
    return snapshot


def mock_model(env, choose=None):
    async def complete(payload, *, max_tokens):
        env.payloads.append(payload)
        review = []
        for item in payload.get("l1_v2_portrait_cards", []):
            excerpt = item["verified_owner_excerpts"][0]
            candidates = choose(item, excerpt) if choose else [
                {"kind": "explicit_fact", "binding_id": excerpt["binding_id"], "quote": excerpt["text"]}
            ]
            review.append({"bucket_id": item["bucket_id"], "candidates": candidates})
        result = {"l1_v2_portrait_review": review}
        if payload.get("maintenance_scope") != "l1_v2_only":
            result["stable_maintenance"] = {scope: {"action": "unchanged"}
                                            for scope in ("user", "persona", "relationship")}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
    env.engine._create_stable_completion = complete
    return complete


def pending(env):
    return [row for row in env.engine.load_state()["profile_fact_candidates"] if row.get("origin") == "l1_v2"]


async def projection(env, bucket_id, *, enabled=True):
    source = await env.buckets.get(bucket_id)
    bound = env.engine._bind_user_sources({bucket_id: source} if source else {})
    return build_user_portrait_projection(env.engine.load_state(), lambda ref: bound.get(ref["bucket_id"]),
                                          enabled=enabled)


def test_old_event_sent_on_adoption_even_with_no_daily_material(env):
    async def scenario():
        mock_model(env)
        prepared = card(env, key="old-memory")
        env.authority.put_candidate(MemoryProposal.from_mapping({
            "proposal_id": "old-memory", "source_type": "daily_chat", "memory_type": "boundary",
            "proposed_body": prepared[0], "metadata": prepared[1], "source_refs": ["s1"],
        }), PolicyDecision("queue_review", (), "fixture"))
        assert (await env.engine.run_due(env.buckets))[0]["status"] == "empty"
        assert not env.payloads and not pending(env)
        snapshot = await adopt(env, "old-memory", prepared=prepared)
        result = (await env.engine.run_due(env.buckets))[0]
        assert result["status"] == "l1_v2_reviewed"
        assert result["l1_v2_candidates"] == 1
        assert env.payloads[0]["maintenance_scope"] == "l1_v2_only"
        assert env.payloads[0]["l1_v2_portrait_cards"][0]["event_time"]["start"] == "2026-10-02"
        candidate, = pending(env)
        assert candidate["memory_revision"] == 1 and snapshot["text"] in candidate["text"]
        assert candidate["evidence"][0]["quote_end"] == len(snapshot["text"])
        assert (await projection(env, "old-memory"))["text"] == ""
        assert (await env.engine.run_due(env.buckets))[0]["status"] == "empty"
        assert len(env.payloads) == len(env.adapter.reads) == 1

    asyncio.run(scenario())


def test_adopt_in_existing_version_preserves_manual_text_and_invalidates_after_withdrawal(env):
    async def scenario():
        proposal = "润润希望发版前能看到预览。"
        approved = "润润希望每次发布前先看预览。"
        mock_model(env, lambda item, excerpt: [{
            "kind": "explicit_preference", "binding_id": excerpt["binding_id"],
            "quote": excerpt["text"], "display_text": proposal,
        }])
        assert env.engine.edit_stable("user", "这是主人手写的资料。", 0)["revision"] == 1
        snapshot = await adopt(env, "old-memory")
        await env.engine.run_due(env.buckets)
        candidate, = pending(env)
        assert candidate["display_text"] == proposal and candidate["evidence"][0]["quote"] == snapshot["text"]
        assert (await env.engine.approve_l1_v2_portrait_candidate(
            candidate["candidate_id"], env.buckets, 0))["reason"] == "revision_mismatch"
        env.engine.set_stable_lock("user", True, 1)
        assert (await env.engine.approve_l1_v2_portrait_candidate(
            candidate["candidate_id"], env.buckets, 1))["reason"] == "stable_locked"
        env.engine.set_stable_lock("user", False, 1)
        for text in ("", f"“{snapshot['text']}”"):
            blocked = await env.engine.approve_l1_v2_portrait_candidate(
                candidate["candidate_id"], env.buckets, 1, display_text=text)
            assert blocked["reason"] == "portrait_wording_required"
        assert pending(env) == [candidate]
        result = await env.engine.approve_l1_v2_portrait_candidate(
            candidate["candidate_id"], env.buckets, 1, display_text=approved)
        assert result["status"] == "updated" and result["revision"] == 2
        assert not pending(env)
        assert (await projection(env, "old-memory"))["text"] == f"这是主人手写的资料。\n{approved}"
        assert (await projection(env, "old-memory", enabled=False))["text"] == ""
        portrait = env.engine.load_state()["portrait"]["user"]
        assert portrait["stable_history"][-1]["text"] == "这是主人手写的资料。"
        claim = portrait["chat_claims"][-1]
        assert claim["text"] == approved and claim["display_reviewed"] is True
        assert claim["evidence"] == candidate["evidence"]
        await env.commit.change_memory_state(memory_id="old-memory", state="tombstoned", recall_policy="disabled",
                                             idempotency_key="withdraw-old", actor="owner")
        assert (await projection(env, "old-memory"))["text"] == "这是主人手写的资料。"

    asyncio.run(scenario())


def test_revision_replaces_stale_candidate_even_after_daily_run(env):
    async def scenario():
        mock_model(env)
        state = env.engine.load_state()
        state.update(runs=[{"date": "2026-11-14"}], last_run_date="2026-11-14")
        env.engine.save_state(state)
        await adopt(env, "revised")
        await env.engine.run_due(env.buckets)
        old, = pending(env)
        revised_snapshot = await adopt(env, "revised", revision=2, quote="我希望每次发版前先看真机预览。")
        assert (await env.engine.approve_l1_v2_portrait_candidate(
            old["candidate_id"], env.buckets, 0,
            display_text="润润希望每次发布前先看预览。"))["reason"] == "source_version_changed"
        env.adapter.fail.add(revised_snapshot["source_id"])
        await env.engine.run_due(env.buckets)
        assert not pending(env)  # Committed replacement is known even while its own review is deferred.
        assert env.engine.load_state()["l1_v2_reviewed"] == {"revised": 1}
        env.adapter.fail.clear()
        await env.engine.run_due(env.buckets)
        new, = pending(env)
        assert new["memory_revision"] == 2 and old["candidate_id"] != new["candidate_id"]
        assert await env.engine.run_due(env.buckets) == []
        assert len(env.payloads) == 2
        assert len(env.engine.load_state()["runs"]) == 1
        await adopt(env, "revised", revision=3, stability="phase")
        await env.engine.run_due(env.buckets)
        assert not pending(env) and len(env.payloads) == 2
        assert env.engine.load_state()["l1_v2_reviewed"] == {"revised": 3}

    asyncio.run(scenario())


def test_current_candidates_survive_capacity_and_delete_a_dedicated_id(env):
    async def scenario():
        env.engine.candidate_max = 1
        mock_model(env)
        for index in range(10):
            await adopt(env, f"current-{index}")
        await env.engine.run_due(env.buckets)
        rows = pending(env)
        assert len(rows) == 10
        after_daily_patch = env.engine._apply_patch(env.engine.load_state(), {}, "2026-11-15")
        assert after_daily_patch["profile_fact_candidates"] == rows
        assert rows[0]["text"] == rows[-1]["text"]
        arguments = {"area": "profile_fact_candidates", "index": 0, "text": rows[0]["text"],
                     "candidate_id": rows[-1]["candidate_id"]}
        assert env.engine.delete_state_item(**arguments)["status"] == "deleted"
        assert env.engine.delete_state_item(**arguments)["reason"] == "candidate_not_found"
        assert pending(env) == rows[:-1]

    asyncio.run(scenario())


def test_pruning_needs_positive_current_authority_evidence(env, monkeypatch):
    async def scenario():
        mock_model(env)
        await adopt(env, "withdrawn")
        await adopt(env, "guarded")
        await adopt(env, "archived")
        await adopt(env, "manually_recallable")
        await env.engine.run_due(env.buckets)
        rows = {row["memory_id"]: row for row in pending(env)}
        assert len(rows) == 4
        await env.commit.change_memory_state(memory_id="withdrawn", state="tombstoned", recall_policy="disabled",
                                             idempotency_key="withdraw-pending", actor="owner")
        await env.commit.change_memory_state(memory_id="archived", state="archived", recall_policy="disabled",
                                             idempotency_key="archive-pending", actor="owner")
        await env.commit.change_memory_state(memory_id="manually_recallable", state="active", recall_policy="manual_only",
                                             idempotency_key="pause-auto", actor="owner")
        change_dependency(env, "guarded", "guard", blocked=True)

        def unavailable(*args):
            raise sqlite3.OperationalError("temporary Authority failure")

        with monkeypatch.context() as patch:
            patch.setattr(MemoryAuthorityRecallView, "portrait_candidate_source_states", unavailable)
            await env.engine.run_due(env.buckets)
        assert {row["memory_id"]: row for row in pending(env)} == rows
        await env.engine.run_due(env.buckets)
        assert {row["memory_id"]: row for row in pending(env)} == {
            key: row for key, row in rows.items() if key != "withdrawn"
        }
        state = env.engine.load_state()
        assert "withdrawn" not in state["l1_v2_reviewed"]
        assert state["last_run_date"] == "2026-11-13"
        assert len(env.payloads) == 1
        change_dependency(env, "guarded", "guard", blocked=False)
        await env.engine.run_due(env.buckets)
        approved = await env.engine.approve_l1_v2_portrait_candidate(rows["guarded"]["candidate_id"],
            env.buckets, 0, display_text="润润希望发版前先看预览。")
        assert approved["status"] == "updated"
        assert {row["memory_id"] for row in pending(env)} == {"archived", "manually_recallable"}

    asyncio.run(scenario())


def test_candidate_expires_only_at_an_authoritative_deadline(env, monkeypatch):
    async def scenario():
        mock_model(env)
        body, metadata, snapshot = card(env, key="time-bound")
        prepared = body, {**metadata, "valid_until": "2099-06-01"}, snapshot
        await adopt(env, "time-bound", prepared=prepared)
        await env.engine.run_due(env.buckets)
        row, = pending(env)

        class Clock(datetime):
            current = datetime(2099, 6, 1, 13, tzinfo=timezone.utc)

            @classmethod
            def now(cls, tz=None):
                return cls.current.astimezone(tz)

        monkeypatch.setattr("portrait_engine.datetime", Clock)
        await env.engine.run_due(env.buckets)
        assert pending(env) == [row]
        Clock.current = datetime(2099, 6, 1, 16, tzinfo=timezone.utc)
        await env.engine.run_due(env.buckets)
        assert not pending(env)
        assert env.engine.load_state()["l1_v2_reviewed"] == {"time-bound": 1}
        assert not env.engine._portrait_source_expired({"expires_at": "unknown"}, Clock.current)

    asyncio.run(scenario())


def test_people_tag_and_ai_summary_do_not_authorize_owner_facts(env):
    async def scenario():
        await adopt(env, "only-gu-yan", role="assistant", quote="我猜她读机械专业。", kind="person_relationship")
        await adopt(env, "passed-phase", stability="phase", quote="我最近在备考。", kind="personal_state")
        await adopt(env, "owner-question", quote="你为什么觉得我是机械专业？", kind="person_relationship")
        mock_model(env, lambda _, excerpt: [{"kind": "explicit_fact", "binding_id": excerpt["binding_id"],
                                              "quote": "我是机械专业"}])
        await env.engine.run_due(env.buckets)
        assert not pending(env)
        assert env.engine.load_state()["l1_v2_reviewed"] == {"only-gu-yan": 1}
        assert {row["bucket_id"] for row in env.payloads[0]["l1_v2_portrait_cards"]} == {"owner-question"}
        mock_model(env, lambda *_: [])
        await env.engine.run_due(env.buckets)
        assert env.engine.load_state()["l1_v2_reviewed"] == {"only-gu-yan": 1, "owner-question": 1}
        assert not pending(env)

    asyncio.run(scenario())


def test_partial_failure_does_not_hide_other_commit_or_checkpoint_unread_card(env):
    async def scenario():
        timed_out = await adopt(env, "timed-out")
        await adopt(env, "healthy")
        env.adapter.fail.add(timed_out["source_id"])

        async def offline(*args, **kwargs):
            raise TimeoutError
        env.engine._create_stable_completion = offline
        assert (await env.engine.run_due(env.buckets))[0]["status"] == "deferred"
        assert not pending(env) and not env.engine.load_state()["l1_v2_reviewed"]
        env.engine.material_limit = 1
        mock_model(env)
        assert (await env.engine.run_due(env.buckets))[0]["status"] == "deferred"
        await env.engine.run_due(env.buckets)
        assert env.engine.load_state()["l1_v2_reviewed"] == {"healthy": 1}
        env.adapter.fail.clear()
        await env.engine.run_due(env.buckets)
        assert env.engine.load_state()["l1_v2_reviewed"] == {"healthy": 1, "timed-out": 1}
        assert {row["memory_id"] for row in pending(env)} == {"healthy", "timed-out"}
        assert [payload["l1_v2_portrait_cards"][0]["bucket_id"] for payload in env.payloads] == ["healthy", "timed-out"]

    asyncio.run(scenario())


def test_revision_committed_during_model_call_is_not_lost_or_staged_stale(env):
    async def scenario():
        await adopt(env, "updated-during-call")
        complete = mock_model(env)

        async def revise_before_model_returns(payload, **kwargs):
            result = await complete(payload, **kwargs)
            await adopt(env, "updated-during-call", revision=2, quote="我喜欢工作日下午先写提纲。")
            return result

        env.engine._create_stable_completion = revise_before_model_returns
        assert (await env.engine.run_due(env.buckets))[0]["status"] == "deferred"
        assert not pending(env) and not env.engine.load_state()["l1_v2_reviewed"]
        mock_model(env)
        await env.engine.run_due(env.buckets)
        candidate, = pending(env)
        assert candidate["memory_revision"] == 2
        assert "我喜欢工作日下午先写提纲。" in candidate["text"]

    asyncio.run(scenario())


def test_two_checks_and_owner_change_during_live_approval(env):
    async def scenario():
        snapshot = await adopt(env, "raced")
        complete = mock_model(env)
        calls = 0
        ready = asyncio.Event()

        async def first_two(payload, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            return await complete(payload, **kwargs)

        env.engine._create_stable_completion = first_two
        await asyncio.gather(env.engine.run_due(env.buckets), env.engine.run_due(env.buckets))
        one, = pending(env)
        assert one["count"] == 1 and env.engine.load_state()["l1_v2_reviewed"] == {"raced": 1}
        env.adapter.on_read = lambda: env.engine.edit_stable("user", "主人在审核期间编辑了正文。", 0)
        result = await env.engine.approve_l1_v2_portrait_candidate(
            one["candidate_id"], env.buckets, 0, display_text="润润希望每次发布前先看预览。")
        assert result["reason"] == "portrait_changed_during_review"
        assert pending(env) == [one]
        env.adapter.on_read = None
        del env.adapter.rows[snapshot["source_id"]]
        result = await env.engine.approve_l1_v2_portrait_candidate(
            one["candidate_id"], env.buckets, 1, display_text="润润希望每次发布前先看预览。")
        assert result["reason"] == "source_quote_changed"
        assert (await projection(env, "raced"))["text"] == "主人在审核期间编辑了正文。"

    asyncio.run(scenario())


def test_gateway_dependency_identity_and_guard_inserted_during_approval(env):
    async def scenario():
        await adopt(env, "guarded")
        view = MemoryAuthorityRecallView(env.config)
        with env.authority._connect() as conn:
            expected = cache_versions(conn, ["guarded"])["guarded"]
        assert view.memory_index_metadata_map(["guarded"], for_portrait=True)["guarded"]["dependency_token"] == expected
        assert "dependency_token" not in view.memory_index_metadata_map(["guarded"])["guarded"]
        mock_model(env)
        await env.engine.run_due(env.buckets)
        candidate, = pending(env)

        def add_guard():
            with env.authority._connect() as conn:
                conn.execute("INSERT INTO memory_relation_guards VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                    "guard", "related-host", 3, "r", "digest", "guarded", 1, "memory",
                    "guarded", "supersedes", "active", "related-owner-change",
                ))
        env.adapter.on_read = add_guard
        result = await env.engine.approve_l1_v2_portrait_candidate(
            candidate["candidate_id"], env.buckets, 0, display_text="润润希望每次发布前先看预览。")
        assert result["reason"] == "source_version_changed"
        assert (await projection(env, "guarded"))["text"] == ""
        assert pending(env) == [candidate]

    asyncio.run(scenario())


def test_mixed_model_candidates_keep_the_valid_item_and_count_specific_rejections(env):
    async def scenario():
        secret = "用于测试的私人片段，不得进入诊断。"
        await adopt(env, "mixed-private-card")
        mock_model(env, lambda _, excerpt: [
            None,
            {"kind": "historical", "binding_id": excerpt["binding_id"], "quote": secret},
            {"kind": "explicit_fact", "binding_id": "unverified", "quote": secret},
            {"kind": "explicit_fact", "binding_id": excerpt["binding_id"], "quote": excerpt["text"]},
        ])
        result = (await env.engine.run_due(env.buckets))[0]
        diagnostic = result["l1_v2_review"]
        assert diagnostic["status"] == "partial"
        assert diagnostic["counts"]["rejected_candidates"] == 3
        assert diagnostic["counts"]["candidates"] == 1
        assert diagnostic["reasons"] == {
            "invalid_candidate_object": 1, "invalid_user_claim_kind": 1,
            "l1_v2_claim_needs_verified_owner_quote": 1,
        }
        assert len(pending(env)) == 1
        assert env.engine.load_state()["l1_v2_reviewed"] == {"mixed-private-card": 1}
        assert env.engine.load_state()["l1_v2_review"] == diagnostic
        assert all(value not in json.dumps(diagnostic, ensure_ascii=False) for value in (secret, "mixed-private-card"))
        assert secret not in json.dumps(result["rejected"], ensure_ascii=False)

    asyncio.run(scenario())


def test_explicit_empty_is_checked_but_all_rejected_items_retry(env):
    async def scenario():
        await adopt(env, "explicit-empty")
        await adopt(env, "all-invalid")
        mock_model(env, lambda row, _: [] if row["bucket_id"] == "explicit-empty" else [
            {"kind": "explicit_fact", "binding_id": None, "quote": "有原句但未提供来源。"},
        ])
        result = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert result["status"] == "partial"
        assert result["counts"]["empty_decisions"] == result["counts"]["deferred"] == 1
        assert result["reasons"] == {"candidate_binding_required": 1, "all_candidates_rejected": 1}
        assert env.engine.load_state()["l1_v2_reviewed"] == {"explicit-empty": 1}
        assert not pending(env)
        mock_model(env)
        await env.engine.run_due(env.buckets)
        submitted, = env.payloads[-1]["l1_v2_portrait_cards"]
        assert submitted["bucket_id"] == "all-invalid"
        assert env.engine.load_state()["l1_v2_reviewed"] == {"explicit-empty": 1, "all-invalid": 1}

    asyncio.run(scenario())


def change_dependency(env, memory_id, dependency, *, blocked):
    with env.authority._connect() as conn:
        if dependency == "guard":
            if blocked:
                conn.execute("INSERT INTO memory_relation_guards VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                    memory_id + "-guard", "relation-author", 4, "r", "digest", memory_id, 1,
                    "statement", "unknown-statement", "supersedes", "active", "related-owner-change",
                ))
            else:
                conn.execute("DELETE FROM memory_relation_guards WHERE target_memory_id=?", (memory_id,))
        elif blocked:
            conn.execute("INSERT INTO memory_l2_sync_proposals VALUES(?,1,1,'pending','{}')", (memory_id,))
        else:
            conn.execute("UPDATE memory_l2_sync_proposals SET state='synchronized' WHERE memory_id=?", (memory_id,))


@pytest.mark.parametrize(("dependency", "reason"), [
    ("guard", "relation_guard_scope_unknown"), ("sync", "l1_sync_required"),
])
def test_dependency_blocks_skip_unchanged_state_without_starving_new_cards(env, dependency, reason):
    async def scenario():
        mock_model(env)
        env.engine.material_limit = 1
        blocked_source = await adopt(env, "blocked")
        await adopt(env, "next-card")
        change_dependency(env, "blocked", dependency, blocked=True)
        first = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert first["status"] == "checked" and first["counts"]["blocked"] == 1
        assert first["reasons"] == {reason: 1}
        assert not env.adapter.reads and not env.payloads
        assert not pending(env)
        assert "blocked" not in env.engine.load_state()["l1_v2_reviewed"]
        second = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert second["counts"]["unchanged_blocked"] == 1 and second["counts"]["reviewed"] == 1
        assert env.payloads[0]["l1_v2_portrait_cards"][0]["bucket_id"] == "next-card"
        third = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert third["status"] == "no_changes" and third["reasons"] == {reason: 1}
        assert blocked_source["source_id"] not in env.adapter.reads
        change_dependency(env, "blocked", dependency, blocked=False)
        fourth = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert fourth["counts"]["reviewed"] == 1
        assert env.engine.load_state()["l1_v2_reviewed"] == {"blocked": 1, "next-card": 1}
        assert not env.engine.load_state()["l1_v2_blocked"]
        assert env.adapter.reads[-1] == blocked_source["source_id"]

    asyncio.run(scenario())


def test_current_guard_blocks_but_a_resolved_historical_scope_remains_eligible():
    assert portrait_l1_v2_block_reason(None, [{"scope": {"historical_only": False, "unknown_scope": False}}]) == (
        "relation_guard_current"
    )
    assert portrait_l1_v2_block_reason(None, [{"scope": {"historical_only": True, "unknown_scope": False}}]) == ""


def test_failed_and_empty_l1_checks_record_time_without_advancing_daily_maintenance(env, caplog):
    async def scenario():
        original = env.engine.load_state()
        no_cards = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert no_cards["status"] == "no_cards"
        await adopt(env, "model-fail")
        private = "模型和异常中的原话"

        async def offline(*args, **kwargs):
            raise TimeoutError(private)

        env.engine._create_stable_completion = offline
        failed = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert failed["status"] == "failed" and failed["reasons"] == {"model_request_failed": 1}

        async def invalid_json(*args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{" + private))])

        env.engine._create_stable_completion = invalid_json
        malformed = (await env.engine.run_due(env.buckets))[0]["l1_v2_review"]
        assert malformed["status"] == "failed" and malformed["reasons"] == {"model_response_invalid_json": 1}
        assert no_cards["checked_at"] < failed["checked_at"] < malformed["checked_at"]
        assert private not in caplog.text
        state = env.engine.load_state()
        assert state["l1_v2_review"] == malformed
        assert state["last_run_date"] == original["last_run_date"] and state["runs"] == original["runs"]
        assert not state["l1_v2_reviewed"] and not pending(env)

    asyncio.run(scenario())


def test_unscored_l1_is_sent_without_confidence_and_legacy_material_keeps_its_score(env):
    async def scenario():
        body, meta, snapshot = card(env, key="unscored")
        await adopt(env, "unscored", prepared=(body, {**meta, "confidence_source": "not_scored", "confidence": 0.0}, snapshot))
        mock_model(env)
        await env.engine.run_due(env.buckets)
        card_payload, = env.payloads[0]["l1_v2_portrait_cards"]
        assert "confidence" not in card_payload
        candidate, = pending(env)
        assert "confidence" not in candidate
        material = env.engine._bucket_payload(await env.buckets.get("unscored"))
        assert "confidence" not in material
        legacy_payload = env.engine._bucket_payload({"metadata": {"confidence": 0.37}, "content": "旧证据"})
        assert legacy_payload["confidence"] == 0.37

        date_key = "2026-11-15"
        material.update(source_date=date_key, created=f"{date_key}T09:00:00+08:00")
        materials = {"buckets": [material], "date": date_key}
        observation = {"scope": "user", "text": "我最近在准备作品集。", "evidence": [{"bucket_id": "unscored"}]}
        raw_patch = {"add_recent": [observation], "add_recent_activity": [observation]}
        patch, rejected = env.engine._normalize_patch(raw_patch, materials)
        assert not rejected
        assert patch["add_recent"] and patch["add_recent_activity"]
        patch["recent_timeline"] = env.engine._build_recent_timeline(materials, patch, date_key)
        assert patch["recent_timeline"]
        state = env.engine._empty_state()
        for _ in range(2):
            state = env.engine._apply_patch(state, patch, date_key)
            assert "confidence" not in state["portrait"]["user"]["recent_buffer"][0]
            assert "confidence" not in state["recent_activities"][0]
            assert all("confidence" not in row for row in state["recent_timeline"])

    asyncio.run(scenario())


def test_daily_stage_keeps_stored_sources_and_blocks_new_writes_from_restricted_l1(env):
    async def scenario():
        for key in ("old-guarded", "current-stage"):
            body, meta, snapshot = card(env, key=key)
            await adopt(env, key, prepared=(body, {**meta, "created": "2026-11-14T08:00:00+08:00"}, snapshot))
        change_dependency(env, "old-guarded", "guard", blocked=True)
        state = env.engine.load_state()
        evidence = [{"bucket_id": "old-guarded"}]
        original_user = dict(state["portrait"]["user"])
        original_user.update(mid_term="之前的真实阶段内容", mid_term_evidence=evidence,
                             mid_term_source_date="2026-10-02", mid_term_source_dates=["2026-10-02"],
                             staging_pool=[{"text": "之前的阶段证据", "evidence": evidence}])
        state["portrait"]["user"] = original_user
        state["handoff_recent_summaries"] = {"2026-11-13": "旧日记录"}
        env.engine.save_state(state)
        mock_model(env)

        async def primary(payload, **kwargs):
            assert [item["bucket_id"] for item in payload["memory_materials"]["buckets"]] == ["current-stage"]
            valid = {"scope": "relationship", "text": "近期协作涉及发布预览", "evidence": [{"bucket_id": "current-stage"}]}
            invalid = {"scope": "user", "text": "这一条用过期的依据", "evidence": evidence}
            data = {"move_to_staging": [valid], "rewrite_mid_term": [valid, invalid]}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

        env.engine._create_patch_completion = primary
        result = (await env.engine.run_due(env.buckets))[0]
        assert result["status"] == "updated"
        current = env.engine.load_state()
        assert current["last_run_date"] == "2026-11-14"
        assert current["portrait"]["user"] == original_user
        assert current["portrait"]["relationship"]["mid_term_evidence"] == [{"bucket_id": "current-stage"}]
        assert current["portrait"]["relationship"]["mid_term_source_dates"] == ["2026-11-14"]
        assert current["handoff_recent_summaries"] == state["handoff_recent_summaries"]

    asyncio.run(scenario())


@pytest.mark.parametrize("blank", [False, True])
def test_daily_model_failure_keeps_previous_portrait_and_l1_review_can_continue(env, blank):
    async def scenario():
        date_key = "2026-11-14"
        body, meta, snapshot = card(env, key="review-on-failure")
        await adopt(env, "review-on-failure", prepared=(body, {**meta, "created": f"{date_key}T09:00:00+08:00"}, snapshot))
        state = env.engine.load_state()
        state["portrait"]["user"].update(
            stable="之前的长期资料", mid_term="之前的阶段内容",
            recent_buffer=[{"text": "之前的近期观察", "source_date": date_key}],
        )
        env.engine.save_state(state)
        before = env.engine.load_state()
        mock_model(env)
        attempts = []

        async def fail_primary(payload, **kwargs):
            attempts.append(payload["date"])
            if blank:
                return SimpleNamespace(choices=[])
            raise TimeoutError("provider-private-details")

        env.engine._create_patch_completion = fail_primary
        result, = await env.engine.run_due(env.buckets)
        assert result["status"] == "failed"
        failure = result["daily_failure"]
        assert failure["date"] == date_key and failure["attempted_at"]
        assert failure["reason"] == ("model_response_invalid_json" if blank else "model_request_failed")
        assert "provider-private-details" not in str(result)
        saved = env.engine.load_state()
        assert saved["daily_failure"] == failure
        for key in ("portrait", "recent_timeline", "recent_activities", "daily_summaries", "runs", "last_run_date"):
            assert saved[key] == before[key]
        assert env.payloads[0]["maintenance_scope"] == "l1_v2_only"
        assert len(pending(env)) == 1

        attempt_count = len(attempts)
        assert await env.engine.run_due(env.buckets) == []
        assert len(attempts) == attempt_count

        async def recover(payload, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

        env.engine._create_patch_completion = recover
        target = datetime(2026, 11, 14, 23, 59, tzinfo=env.engine.tz)
        result = await env.engine.maintain_daily(env.buckets, now=target, force=True)
        assert result["status"] == "updated"
        saved = env.engine.load_state()
        assert saved["last_run_date"] == date_key and saved["daily_failure"] == {}
        assert saved["portrait"]["user"] == before["portrait"]["user"]

    asyncio.run(scenario())


@pytest.mark.parametrize("stable_review_succeeds", [False, True])
def test_daily_stage_can_write_but_only_stable_review_changes_long_term(env, stable_review_succeeds):
    async def scenario():
        card_id = "phase-card"
        evidence = [{"bucket_id": card_id}]
        body, meta, snapshot = card(env, key=card_id)
        await adopt(env, card_id, prepared=(body, {**meta, "created": "2026-11-14T09:00:00+08:00"}, snapshot))
        state = env.engine.load_state()
        state["portrait"]["relationship"].update(stable="之前的长期内容", stable_revision=2)
        env.engine.save_state(state)

        async def primary(payload, **kwargs):
            phase = {"scope": "relationship", "text": "这阶段先协作到月底", "evidence": evidence}
            old_prompt_output = {"scope": "relationship", "text": "第一轮给出的长期改写", "evidence": evidence}
            result = {"move_to_staging": [phase], "rewrite_mid_term": [phase], "rewrite_stable": [old_prompt_output]}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

        async def stable(payload, **kwargs):
            if not stable_review_succeeds:
                raise TimeoutError
            # A separate invalid scope must not discard the accepted decision.
            result = {"stable_maintenance": {
                "user": {"action": "unchanged"},
                "relationship": {"action": "rewrite", "text": "第二轮通过的长期内容", "evidence": evidence},
            }}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

        env.engine._create_patch_completion = primary
        env.engine._create_stable_completion = stable
        result, = await env.engine.run_due(env.buckets)
        assert result["status"] == "updated"
        relationship = env.engine.load_state()["portrait"]["relationship"]
        assert relationship["mid_term"] == "这阶段先协作到月底"
        assert relationship["mid_term_evidence"] == evidence
        assert relationship["stable"] == ("第二轮通过的长期内容" if stable_review_succeeds else "之前的长期内容")
        assert relationship["stable_revision"] == (3 if stable_review_succeeds else 2)

    asyncio.run(scenario())
