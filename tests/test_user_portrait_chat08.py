"""Synthetic evidence only: user portrait's single writer and read-only projection."""
import asyncio
import copy
import hashlib
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memory_authority import MemoryAuthorityStore
from portrait_engine import DailyPortraitMaintainer, PATCH_KEYS
from user_portrait_chat import (bind_source_authority, build_user_portrait_projection,
                                chat_evidence_ids, source_is_current, source_revision, portrait_state_path)


@pytest.fixture
def setup(tmp_path):
    config = {"buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state"),
              "memory_authority": {"enabled": True}, "portrait": {"enabled": False}}
    engine = DailyPortraitMaintainer(config)
    authority = MemoryAuthorityStore(config)
    sources = {}
    for i, date in enumerate(("2026-10-01", "2026-10-03"), 1):
        bid = f"source-{i}"
        source = {"id": bid, "path": str(tmp_path / "buckets" / "permanent" / f"{bid}.md"),
                  "metadata": {"id": bid, "created": date + "T06:00:00Z", "type": "permanent",
                               "tags": ["profile_fact"]}, "content": f"SYNTHETIC OBSERVATION {i}"}
        sources[bid] = source
        authority.import_legacy_memory(memory_id="memory-" + bid, bucket_id=bid,
            body_sha256=hashlib.sha256(source["content"].encode()).hexdigest(),
            snapshot_path=source["path"], metadata=source["metadata"], source_refs=["fixture-event"])
    state = engine.load_state()
    user = state["portrait"]["user"]
    user.update(stable="LEGACY WHOLE PARAGRAPH", stable_revision=3, stable_source="model",
                stable_evidence=[{"bucket_id": bid} for bid in sources])
    state["portrait"]["persona"]["stable"] = "PERSONA EXCLUDED"
    state["portrait"]["relationship"]["stable"] = "RELATIONSHIP EXCLUDED"
    engine.save_state(state)
    mgr = SimpleNamespace(get=AsyncMock(side_effect=lambda bid: copy.deepcopy(sources.get(bid))))
    engine._test_sources = sources
    return engine, sources, mgr, authority


def claim(*ids, kind="revisable_impression", **extra):
    return {"kind": kind, "text": "Synthetic supported understanding",
            "evidence": [{"bucket_id": bid} for bid in ids], **extra}


def install(engine, mgr, claims=None):
    claims = copy.deepcopy(claims or [claim("source-1")])
    # These are the versions the synthetic review actually read.
    sources = engine._test_sources
    bound = engine._bind_user_sources(sources)
    for row in claims:
        for ref in row["evidence"]:
            bid = ref["bucket_id"]
            if bid in bound:
                ref.update(source_revision=source_revision(bound[bid]),
                           memory_revision=bound[bid]["_portrait_authority"]["memory_revision"])
    return asyncio.run(engine.install_reviewed_user_claims(
        claims, mgr, expected_revision=3))


def projection(engine, sources, **kwargs):
    bound = engine._bind_user_sources(sources)
    return build_user_portrait_projection(engine.load_state(),
        lambda ref: bound.get(ref["bucket_id"]), **kwargs)


def test_legacy_never_enters_chat_or_other_scopes(setup):
    engine, sources, _, _ = setup
    value = projection(engine, sources)
    assert value["text"] == "" and value["status"] == "unreviewed"
    assert chat_evidence_ids(engine.load_state()) == []
    state = engine.load_state()
    state["portrait"]["user"]["stable_source"] = "manual"
    assert build_user_portrait_projection(state, lambda _: None)["text"] == ""


@pytest.mark.parametrize("key,value", [
    ("valid_until", "2000-01-01"), ("expires_at", "2000-01-01T00:00:00Z"),
    ("valid_from", "2099-06-01"), ("effective_from", "unresolved-date"),
])
def test_outside_or_unknown_source_window_stays_out_of_chat(setup, key, value):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    sources["source-1"]["metadata"][key] = value
    assert not source_is_current(sources["source-1"])
    assert projection(engine, sources)["text"] == ""


def test_reviewed_install_preserves_same_history_and_is_revisable(setup):
    engine, sources, mgr, _ = setup
    assert install(engine, mgr) == {"status": "updated", "revision": 4, "claim_count": 1}
    user = engine.load_state()["portrait"]["user"]
    assert user["stable_source"] == "model" and user["chat_claims_revision"] == 4
    assert user["stable_history"][-1]["text"] == "LEGACY WHOLE PARAGRAPH"
    assert user["stable_history"][-1]["revision"] == 3
    ref = user["chat_claims"][0]["evidence"][0]
    assert ref["memory_id"] == "memory-source-1" and ref["memory_revision"] == 1
    assert ref["relative_path"] == "permanent/source-1.md"
    result = projection(engine, sources)
    assert result["status"] == "ready" and "我目前觉得" in result["text"]
    assert "LEGACY" not in result["text"] and "EXCLUDED" not in result["text"]


@pytest.mark.parametrize("change", ["deleted", "retracted", "content", "disabled", "revision", "dependency"])
def test_invalidation_precedes_regeneration(setup, change):
    engine, sources, mgr, authority = setup
    install(engine, mgr)
    if change == "deleted":
        sources.pop("source-1")
    elif change == "retracted":
        sources["source-1"]["metadata"]["retracted"] = True
    elif change == "content":
        sources["source-1"]["content"] += " CORRECTION"
    elif change in {"disabled", "revision"}:
        import sqlite3
        with sqlite3.connect(authority.path) as conn:
            conn.execute("UPDATE memories SET " + ("recall_policy='manual_only'" if change == "disabled"
                         else "active_revision=2") + " WHERE bucket_id='source-1'")
    else:
        # A relationship/semantic dependency token changes even with identical body.
        state = engine.load_state()
        state["portrait"]["user"]["chat_claims"][0]["evidence"][0]["dependency_token"] = "OLD_DEPENDENCY"
        engine.save_state(state)
    assert projection(engine, sources)["text"] == ""
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 4


def test_activation_does_not_invalidate_semantic_source(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    sources["source-1"]["metadata"].update(last_active="2099-01-01", activation=0.8, updated_at="new")
    assert projection(engine, sources)["status"] == "ready"


def test_ambiguous_summary_cannot_become_explicit_preference(setup):
    engine, _, mgr, _ = setup
    result = install(engine, mgr, [claim("source-1", kind="explicit_preference")])
    assert result["reason"] == "user_preference_needs_attributed_quote"
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 3


def test_one_owner_quote_suffices_and_transport_user_alone_does_not(setup):
    engine, sources, _, _ = setup
    engine.config["gateway"] = {"canonical_continuation": {"enabled": True}}
    meta = sources["source-1"]["metadata"]
    meta["role"] = "user"
    row = claim("source-1", kind="explicit_preference")
    row["evidence"][0]["quote"] = sources["source-1"]["content"]
    assert engine._normalize_user_claims([row], engine._bind_user_sources(sources))[1] == "user_preference_needs_attributed_quote"
    meta.update(system="aizizhu", namespace="canonical_event", profile_id="haven_xiaoyu",
                conversation_id="conv-jiajia-main")
    normalized, reason = engine._normalize_user_claims([row], engine._bind_user_sources(sources))
    assert not reason
    assert row["evidence"][0]["quote"] in normalized[0]["text"]
    assert row["text"] not in normalized[0]["text"]


def test_multiple_day_rule_and_historical_date(setup):
    engine, sources, _, _ = setup
    bound = engine._bind_user_sources(sources)
    assert engine._normalize_user_claims([claim("source-1")], bound)[1] == "user_impression_needs_distinct_experiences"
    assert engine._normalize_user_claims([claim("source-1", "source-2")], bound)[1] == ""
    assert engine._normalize_user_claims([claim("source-1", kind="historical", as_of="2026-10-01")], bound)[1] == ""
    assert engine._normalize_user_claims([claim("source-1", kind="historical", as_of="2026-11-01")], bound)[1] == "user_history_needs_source_date"


def test_historical_claim_never_poses_as_current_context(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr, [claim("source-1", kind="historical", as_of="2026-10-01")])
    assert projection(engine, sources)["text"] == ""
    assert chat_evidence_ids(engine.load_state()) == []
    assert "2026-10-01" in engine.load_state()["portrait"]["user"]["stable"]


def test_owner_edit_lock_cas_and_rollback_remain_effective(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    assert engine.edit_stable("user", "OWNER EDIT", 3)["status"] == "conflict"
    assert engine.edit_stable("user", "OWNER EDIT", 4, locked=True)["revision"] == 5
    assert projection(engine, sources)["text"] == "OWNER EDIT"
    assert projection(engine, sources, enabled=False)["text"] == ""
    assert asyncio.run(engine.install_reviewed_user_claims([claim("source-1")], mgr, 5))["reason"] == "stable_locked"
    assert engine.rollback_stable("user", 4, 5)["revision"] == 6
    assert projection(engine, sources)["status"] == "ready"
    assert engine.load_state()["portrait"]["user"]["stable_locked"] is True


def test_cas_rechecks_edit_and_lock_during_source_read(setup):
    engine, sources, mgr, _ = setup
    async def get(bid):
        engine.set_stable_lock("user", True, 3)
        return copy.deepcopy(sources[bid])
    mgr.get.side_effect = get
    assert install(engine, mgr)["reason"] == "user_changed_during_review"
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 3


def test_background_adopts_claims_and_can_remove_all(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    state = engine.load_state()
    bound = engine._bind_user_sources(sources)
    claims, reason = engine._normalize_user_claims([claim("source-1", "source-2")], bound)
    assert reason == ""
    patch = {key: [] for key in PATCH_KEYS}
    patch["rewrite_stable"] = [{"scope": "user", "chat_claims": claims}]
    materials = {"buckets": [], "persona_events": []}
    engine._finish_maintenance(state, patch, "2026-11-05", False, [], materials)
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 5
    assert projection(engine, sources)["status"] == "ready"
    state = engine.load_state()
    patch["rewrite_stable"] = [{"scope": "user", "chat_claims": []}]
    engine._finish_maintenance(state, patch, "2026-11-06", False, [], materials)
    assert projection(engine, sources)["text"] == ""
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 6


def test_reviewed_claim_can_be_preserved_when_other_claims_change(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    prior = engine.load_state()["portrait"]["user"]["chat_claims"]
    clean, reason = engine._normalize_user_claims([claim("source-1")], engine._bind_user_sources(sources),
                                                previous_claims=prior)
    assert clean == prior and reason == ""


def test_background_uses_single_writer_and_owner_changes_win(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    state = engine.load_state()
    bound = engine._bind_user_sources(sources)
    materials = {"buckets": [], "persona_events": [], "_user_sources": bound,
                 "previous_portrait": engine._portrait_snapshot(state)}
    raw = {"stable_maintenance": {"user": {"action": "rewrite", "claims": [claim("source-1", "source-2")]},
            "persona": {"action": "unchanged"}, "relationship": {"action": "unchanged"}}}
    rows, rejected, scopes = engine._normalize_stable_maintenance(raw, state, materials)
    user_rows = [row for row in rows if row["scope"] == "user"]
    assert len(user_rows) == 1 and "user" in scopes
    patch = {key: [] for key in PATCH_KEYS}
    patch["rewrite_stable"] = user_rows
    engine.edit_stable("user", "OWNER WHILE MODEL AWAITS", 4)
    engine._finish_maintenance(state, patch, "2026-11-05", False, rejected, materials)
    assert projection(engine, sources)["text"] == "OWNER WHILE MODEL AWAITS"
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 5


def test_legacy_patch_cannot_overwrite_user(setup):
    engine, _, _, _ = setup
    state = engine.load_state()
    patch = {key: [] for key in PATCH_KEYS}
    patch["rewrite_stable"] = [{"scope": "user", "text": "OLD WRITER", "evidence": []}]
    result = engine._apply_patch(state, patch, "2026-11-05")
    assert result["portrait"]["user"]["stable"] == "LEGACY WHOLE PARAGRAPH"


def test_shared_reader_is_read_only_and_no_model(setup):
    engine, sources, mgr, _ = setup
    install(engine, mgr)
    before = open(engine.state_path, "rb").read()
    assert projection(engine, sources)["status"] == "ready"
    assert open(engine.state_path, "rb").read() == before
    assert portrait_state_path(engine.config) == engine.state_path


def test_authority_dependency_token_is_part_of_binding(setup):
    _, sources, _, _ = setup
    source = sources["source-1"]
    row = {"memory_id": "memory-source-1", "revision": 1,
           "body_sha256": hashlib.sha256(source["content"].encode()).hexdigest(), "dependency_token": "v1"}
    first = bind_source_authority(source, row, {"state": "legacy", "document": None})
    second = bind_source_authority(source, {**row, "dependency_token": "v2"}, {"state": "legacy", "document": None})
    assert first["_portrait_authority"]["dependency_token"] != second["_portrait_authority"]["dependency_token"]
    assert bind_source_authority(source, None) is None
    assert bind_source_authority(source, row, None) is None
    assert bind_source_authority(source, row, {"state": "dependency_changed"}) is None


def test_same_day_distinct_experiences_count_and_same_event_derivatives_do_not(setup):
    engine, sources, _, _ = setup
    sources["source-2"]["metadata"]["created"] = sources["source-1"]["metadata"]["created"]
    sources["source-1"]["metadata"]["event_id"] = "event-A"
    sources["source-2"]["metadata"]["event_id"] = "event-B"
    assert engine._normalize_user_claims([claim("source-1", "source-2")], engine._bind_user_sources(sources))[1] == ""
    sources["source-2"]["metadata"]["original_event_id"] = "event-A"
    assert engine._normalize_user_claims([claim("source-1", "source-2")], engine._bind_user_sources(sources))[1] == "user_impression_needs_distinct_experiences"


@pytest.mark.parametrize("tag", ["daily_impression", "weekly_impression"])
def test_retired_relationship_impressions_are_never_user_evidence(setup, tag):
    engine, sources, _, _ = setup
    sources["source-1"]["metadata"]["tags"] = ["relationship_weather", tag]
    clean, reason = engine._normalize_user_claims([claim("source-1")], engine._bind_user_sources(sources), reviewed=True)
    assert clean is None and reason == "user_source_not_current"


def test_reviewed_source_must_match_the_version_actually_read(setup):
    engine, _, mgr, _ = setup
    refs = [claim("source-1")]
    refs[0]["evidence"][0].update(source_revision="AN_OLDER_REVIEW", memory_revision=1)
    result = asyncio.run(engine.install_reviewed_user_claims(refs, mgr, 3))
    assert result["reason"] == "review_source_version_mismatch"


def test_cross_process_owner_cas_uses_the_same_file_lock(setup):
    import subprocess
    import sys
    engine, _, _, _ = setup
    script = """
import json, sys
from portrait_engine import DailyPortraitMaintainer
engine = DailyPortraitMaintainer(json.loads(sys.argv[1]))
print('READY', flush=True)
result = engine.edit_stable('user', 'OTHER PROCESS', 3)
print(result['status'], flush=True)
"""
    with engine._mutation_lock():
        process = subprocess.Popen([sys.executable, "-c", script, json.dumps(engine.config)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert process.stdout.readline().strip() == "READY"
        assert process.poll() is None
        state = engine.load_state()
        engine._replace_stable(state["portrait"]["user"], text="FIRST PROCESS", evidence=[],
                               source_dates=[], source="manual")
        engine.save_state(state)
    out, error = process.communicate(timeout=20)
    assert process.returncode == 0, error
    assert out.strip() == "conflict"
    assert engine.load_state()["portrait"]["user"]["stable"] == "FIRST PROCESS"


def test_unreviewed_version_can_adopt_new_evidence_but_access_is_not_freshness(setup):
    engine, sources, mgr, _ = setup
    engine.config["gateway"] = {"canonical_continuation": {"enabled": True}}
    meta = sources["source-1"]["metadata"]
    meta.update(created="2026-11-05T06:00:00+08:00", system="aizizhu", namespace="canonical_event",
                role="user", profile_id="haven_xiaoyu", conversation_id="conv-jiajia-main")
    sources["source-2"]["metadata"].update(updated_at="2026-11-05T06:00:00+08:00", last_active="2026-11-05")
    mgr.list_all = AsyncMock(return_value=list(sources.values()))
    state = engine.load_state()
    materials = asyncio.run(engine._daily_materials(mgr, None,
        datetime(2026, 11, 5, 8, tzinfo=engine.tz), state, initial=False))
    assert set(materials["_user_sources"]) == {"source-1"}
    row = claim("source-1", kind="explicit_preference")
    row["evidence"][0]["quote"] = sources["source-1"]["content"]
    raw = {"stable_maintenance": {"user": {"action": "rewrite", "claims": [row]},
           "persona": {"action": "unchanged"}, "relationship": {"action": "unchanged"}}}
    patch = {key: [] for key in PATCH_KEYS}
    engine.client = object()
    engine._api_stable_maintenance = AsyncMock(return_value=raw)
    rows, rejected = asyncio.run(engine._maintain_stables("2026-11-05", state, materials, patch))
    assert any(item.get("reason") == "persona_stable_requires_rewrite" for item in rejected)
    assert [item["scope"] for item in rows] == ["user"]
    patch["rewrite_stable"] = rows
    engine._finish_maintenance(state, patch, "2026-11-05", False, [], materials)
    assert engine.load_state()["portrait"]["user"]["stable_revision"] == 4
    assert engine.load_state()["portrait"]["user"]["stable_history"][-1]["text"] == "LEGACY WHOLE PARAGRAPH"
    assert projection(engine, sources)["status"] == "ready"
