import asyncio
import hashlib
import json
from types import SimpleNamespace

import frontmatter

from gateway_user_portrait import GatewayUserPortrait
from user_portrait_chat import bind_source_authority, source_revision
from test_prompt_context_placements_r1 import _block, _compose_and_inject, _context_values, _plan, _service


def portrait_fixture(tmp_path):
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
    bound = bind_source_authority(bucket, metadata, sem)
    assert bind_source_authority(bucket, metadata, {**sem, "l1_sync_required": True}) is None
    ref = {"bucket_id": "example", "relative_path": "permanent/example.md",
           "source_revision": source_revision(bucket), **bound["_portrait_authority"]}
    claims = [{"kind": "revisable_impression", "text": "她有时喜欢分享读过的诗。", "evidence": [ref]},
              {"kind": "historical", "text": "旧安排", "evidence": [ref]}]
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
