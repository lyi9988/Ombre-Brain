"""Request-local member reuse over a real temporary Authority; no external IO."""
import asyncio
import json

import pytest

from test_memory_recall_contract_r2 import setup_diary, make_committed_diary, run


def test_member_batch_keeps_401st_requested_member_and_reads_identity_once(tmp_path, monkeypatch):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env)
    index = env["reads"].index
    ids = [f"synthetic-binding-{i}" for i in range(401)]
    conn = env["authority"]._connect()
    try:
        conn.executemany("INSERT INTO memory_l1_members VALUES(?,?,?,?,?,?,?)", [
            (committed["memory_id"], 1, "evidence_bindings", 1000 + i, bid, None,
             json.dumps({"binding_id": bid})) for i, bid in enumerate(ids)
        ])
        conn.commit()
    finally:
        conn.close()
    checks = []
    original = index.ensure

    def ensure(memory_id, revision):
        checks.append((memory_id, revision))
        return original(memory_id, revision)

    monkeypatch.setattr(index, "ensure", ensure)
    result = index.members(committed["memory_id"], 1, "evidence_bindings", [*ids, ids[0], "missing"])
    assert result == {bid: {"binding_id": bid} for bid in ids}
    assert checks == [(committed["memory_id"], 1)]


def test_shared_root_members_are_reused_only_within_batch_and_source_is_read_again(tmp_path):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env)
    calls = []
    original = env["adapter"].read

    async def read(binding):
        calls.append(binding["binding_id"])
        return await original(binding)

    env["adapter"].read = read
    items = [{"memory_id": committed["child_id"], "revision": 1,
              "binding_ids": [committed["binding_id"]]} for _ in range(41)]
    result = run(env["reads"].recall_source_status(items))
    assert len(result["items"]) == 41 and len(calls) == 1
    assert all(row["state"] == "ok" and row["sources"][committed["binding_id"]]["eligible_source"]
               for row in result["items"])
    assert result["diagnostics"]["member_batches"] == 2  # scene + requested bindings
    assert result["diagnostics"]["member_cache_hits"] == 80
    assert result["diagnostics"]["unique_binding_reads"] == 1

    env["source_state"]["record"]["status"] = "withdrawn"
    second = run(env["reads"].recall_source_status(items))
    assert len(calls) == 2
    assert second["diagnostics"]["member_batches"] == 2
    assert all(row["sources"][committed["binding_id"]]["eligible_source"] is False
               for row in second["items"])
    assert "content" not in json.dumps(second["diagnostics"])


def test_cancelled_batch_cancels_shared_source_task_without_reissuing_it(tmp_path):
    env = setup_diary(tmp_path)
    committed = make_committed_diary(env)
    entered, cancelled = asyncio.Event(), asyncio.Event()
    calls = []

    async def read(_binding):
        calls.append(True)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    env["adapter"].read = read

    async def scenario():
        task = asyncio.create_task(env["reads"].recall_source_status([
            {"memory_id": committed["child_id"], "revision": 1,
             "binding_ids": [committed["binding_id"]]} for _ in range(41)
        ]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    run(scenario())
    assert calls == [True]
