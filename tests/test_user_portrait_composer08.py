import asyncio
import hashlib
import json
from types import SimpleNamespace

import frontmatter
import pytest

import memory_recall_runtime as memory_use
from gateway_user_portrait import GatewayUserPortrait
from user_portrait_chat import bind_source_authority, source_revision
from test_prompt_context_placements_r1 import _block, _compose_and_inject, _context_values, _plan, _service


OWNER_QUOTE = "我住在松林镇。"
REVIEWED_FACT = "她住在松林镇。"


def portrait_fixture(tmp_path, *, l1_v2=False):
    folder = tmp_path / "buckets"
    category = folder / "permanent"
    category.mkdir(parents=True)
    path = category / "example.md"
    item = frontmatter.Post("原来的观测。", id="example", type="permanent")
    path.write_text(frontmatter.dumps(item), encoding="utf-8")
    bucket = {"id": "example", "content": item.content, "metadata": item.metadata}
    metadata = {"memory_id": "m1", "revision": 1, "body_sha256": hashlib.sha256(
        item.content.encode()).hexdigest(), "dependency_token": "first"}
    sem = {"state": "legacy"}
    if l1_v2:
        sem = {"state": "l1", "document": {"schema_version": "memory-l1-v2",
               "evidence_bindings": [{"binding_id": "source:1", "source_id": "e1"}]}}
    bound = bind_source_authority(bucket, metadata, sem)
    assert bind_source_authority(bucket, metadata, {**sem, "l1_sync_required": True}) is None
    ref = {"bucket_id": "example", "relative_path": "permanent/example.md",
           "source_revision": source_revision(bucket), **bound["_portrait_authority"]}
    claims = [{"kind": "revisable_impression", "text": "她有时喜欢分享读过的诗。", "evidence": [ref]},
              {"kind": "historical", "text": "旧安排", "evidence": [ref]}]
    if l1_v2:
        sem["state"] = "l1_v2"
        ref.update(binding_id="source:1", quote=OWNER_QUOTE)
        claims[0].update(kind="explicit_fact", text=REVIEWED_FACT, display_reviewed=True)
    user = {"stable_revision": 4, "chat_claims_revision": 4, "stable_source": "model",
            "stable": "旧全文不能作为回退", "chat_claims": claims}
    state_path = tmp_path / "portrait_state.json"
    state_path.write_text(json.dumps({"portrait": {"user": user, "persona": {"stable": "顾衍自己"}}}),
                          encoding="utf-8")
    entries = {"example": metadata}
    view = SimpleNamespace(memory_index_metadata_map=lambda ids: entries.copy(),
                           memory_l1_recall_map=lambda ids: {"example": sem})
    service = _service()
    service.config = {"state_dir": str(tmp_path), "buckets_dir": str(folder)}
    service.user_portrait = GatewayUserPortrait(service.config, view)
    return service, entries, path, state_path


def placed(service, plan):
    text, debug, guard = asyncio.run(service._chat_portrait_input(plan))
    values = _context_values()
    values["portrait_memory"] = text
    original = [{"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "HISTORY"}, {"role": "assistant", "content": "REPLY"},
                {"role": "user", "content": "CURRENT"}]
    messages, *_ = _compose_and_inject(service, plan, original, values=values)
    return "\n".join(row["content"] for row in messages), debug, guard


def test_retired_handoff_cannot_override_rolling_context():
    service = _service()
    handoff = _block("ombre.handoff_hint_prompt", role="user", anchor="history.start", order=10,
                     mode="owner_override", owner_body="RETIRED_OVERRIDE")
    dated = _block("ombre.handoff_date_hint_prompt", role="user", anchor="history.start", order=20,
                   mode="frozen_snapshot")
    dated["frozen_body"] = "RETIRED_SNAPSHOT"
    recent = _block("ombre.just_now_context", role="user", anchor="history.start", order=30)
    values = _context_values()
    values.update(handoff_tool_hint="RETIRED_RUNTIME_HINT", just_now_context="ROLLING_RECENT_CONTEXT")
    messages, *_ = _compose_and_inject(
        service, _plan(handoff, dated, recent),
        [{"role": "user", "content": "PREVIOUS"}, {"role": "assistant", "content": "REPLY"},
         {"role": "user", "content": "CURRENT"}], values=values,
    )
    rendered = "\n".join(row["content"] for row in messages)
    assert "ROLLING_RECENT_CONTEXT" in rendered
    assert "RETIRED_" not in rendered
    legacy = "\n".join(service._build_injected_context_messages(**values))
    assert "ROLLING_RECENT_CONTEXT" in legacy
    assert "RETIRED_" not in legacy


def test_current_user_claim_moves_disappears_and_invalidates_a_prepared_turn(tmp_path):
    service, entries, _, _ = portrait_fixture(tmp_path)
    portrait = _block("ombre.portrait_memory", role="system", anchor="turn.before_current_user", order=20)
    core = _block("ombre.core_memory", role="system", anchor="turn.before_current_user", order=30)
    plan = _plan(portrait, core)
    wire, debug, guard = placed(service, plan)
    expected = "她有时喜欢分享读过的诗。"
    assert debug["generated_portrait_version"] == 4
    assert wire.count(expected) == 1 and wire.index(expected) < wire.index("CORE_BLOCK_R1")
    assert all(body not in wire for body in ("旧安排", "旧全文", "顾衍自己", "bucket_id"))
    entry = {"memory_guard": {"user_portrait": guard}}
    assert asyncio.run(service._portrait_snapshot_current(entry, plan))
    portrait["order"] = 40
    moved, _, _ = placed(service, plan)
    assert moved.index("CORE_BLOCK_R1") < moved.index(expected) < moved.index("CURRENT")
    entries.clear()
    assert not asyncio.run(service._portrait_snapshot_current(entry, plan))
    withdrawn, _, _ = placed(service, plan)
    assert expected not in withdrawn and "CORE_BLOCK_R1" in withdrawn
    portrait.update(mode="frozen_snapshot", frozen_body=expected)
    assert expected not in placed(service, plan)[0]


def test_off_missing_plan_and_foreign_scopes_do_not_read_the_portrait(tmp_path):
    service, _, _, _ = portrait_fixture(tmp_path)
    def forbidden_read():
        raise AssertionError("an off or unverified plan read the live portrait")
    service.user_portrait.read = forbidden_read
    off = _block("ombre.portrait_memory", role="system", anchor="turn.before_current_user", order=20,
                 enabled=False)
    plan = _plan(off)
    assert "分享读过的诗" not in placed(service, plan)[0]
    for other in (None, {**plan, "fallback_legacy": True}, {**plan, "scope": "autonomy.planning"}):
        assert asyncio.run(service._chat_portrait_input(other))[0] == ""


def test_changed_or_unavailable_source_does_not_fall_back_to_old_prose(tmp_path):
    service, entries, path, state_path = portrait_fixture(tmp_path)
    assert service.user_portrait.read()["claim_count"] == 1
    entries["example"]["dependency_token"] = "corrected"
    assert not service.user_portrait.read()["text"]
    entries["example"]["dependency_token"] = "first"
    path.write_text("---\nid: example\ntype: permanent\n---\n不同内容", encoding="utf-8")
    assert not service.user_portrait.read()["text"]
    state_path.unlink()
    assert service.user_portrait.read()["status"] == "unavailable"


def test_brain_v2_reviewed_text_reaches_composer_after_verifying_live_source(tmp_path, monkeypatch):
    service, _, _, _ = portrait_fixture(tmp_path, l1_v2=True)
    portrait = _block("ombre.portrait_memory", role="system", anchor="turn.before_current_user", order=20)
    core = _block("ombre.core_memory", role="system", anchor="turn.before_current_user", order=30)
    plan = _plan(portrait, core)
    expected_items = [{"memory_id": "m1", "revision": 1, "binding_ids": ["source:1"]}]
    pending = service.user_portrait.read()
    assert pending["text"] == "" and pending["v2_source_checks"] == expected_items
    eligible = True
    calls = []

    async def read_batch(_service, _request, stage, path, payload):
        assert stage == "portrait_sources" and path == "/api/memory-authority/recall-sources"
        assert payload == {"items": expected_items}
        calls.append(payload)
        return {"items": [{"memory_id": "m1", "revision": 1, "state": "ok",
                          "sources": {"source:1": {"eligible_source": eligible}}}]}

    monkeypatch.setattr(memory_use, "read_batch", read_batch)
    wire, debug, guard = placed(service, plan)
    assert len(calls) == 1 and debug["claim_count"] == 1
    assert wire.count(REVIEWED_FACT) == 1 and wire.index(REVIEWED_FACT) < wire.index("CORE_BLOCK_R1")
    assert OWNER_QUOTE not in wire
    portrait["order"] = 40
    moved, _, _ = placed(service, plan)
    assert moved.index("CORE_BLOCK_R1") < moved.index(REVIEWED_FACT) < moved.index("CURRENT")
    assert OWNER_QUOTE not in moved
    portrait["order"] = 20
    eligible = False
    assert not asyncio.run(service._portrait_snapshot_current({"memory_guard": {"user_portrait": guard}}, plan))
    withdrawn, _, _ = placed(service, plan)
    assert REVIEWED_FACT not in withdrawn and OWNER_QUOTE not in withdrawn and "CORE_BLOCK_R1" in withdrawn
    previous_calls = len(calls)
    portrait["enabled"] = False
    assert REVIEWED_FACT not in placed(service, plan)[0] and len(calls) == previous_calls


@pytest.mark.parametrize("kind", ["explicit_fact", "explicit_preference"])
def test_quote_only_old_claim_skips_source_reads_and_composer_delivery(tmp_path, monkeypatch, kind):
    service, _, _, state_path = portrait_fixture(tmp_path, l1_v2=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    claim = state["portrait"]["user"]["chat_claims"][0]
    claim.update(kind=kind, text=f"“{OWNER_QUOTE}”")
    claim.pop("display_reviewed")
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("a quote-only claim triggered source verification")

    monkeypatch.setattr(service.user_portrait.authority, "memory_index_metadata_map", unexpected_read)
    monkeypatch.setattr(service.user_portrait.authority, "memory_l1_recall_map", unexpected_read)
    monkeypatch.setattr(service.user_portrait, "_source", unexpected_read)
    monkeypatch.setattr(memory_use, "read_batch", unexpected_read)
    portrait = _block("ombre.portrait_memory", role="system", anchor="turn.before_current_user", order=20)
    core = _block("ombre.core_memory", role="system", anchor="turn.before_current_user", order=30)
    wire, debug, _ = placed(service, _plan(portrait, core))
    assert "CORE_BLOCK_R1" in wire and "CURRENT" in wire
    assert OWNER_QUOTE not in wire and REVIEWED_FACT not in wire
    assert debug["claim_count"] == 0
