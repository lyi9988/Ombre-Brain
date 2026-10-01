"""Offline maintenance contract tests. No provider or production access."""
import asyncio
import copy
import json
import threading
import time

import pytest

from memory_embedding_jobs import MemoryEmbeddingJobs, EmbeddingJobError
from memory_index_lease import MemoryIndexLease
from scripts import repair_recall_r1_embeddings as repair


def setup_job(tmp_path, monkeypatch):
    cfg = {"state_dir": str(tmp_path), "buckets_dir": str(tmp_path / "buckets"),
           "embedding": {"enabled": True, "model": "test-encoder", "dimensions": 1024,
                         "base_url": "https://example.invalid/v1", "api_key": "never-export-test-secret"},
           "memory_authority": {"enabled": True}}
    manager = MemoryEmbeddingJobs(lambda: cfg)
    inventory = {"path_identity_ok": True, "embedding_schema_ready": True,
                 "prompt_mirror_file_present": True, "memory_authority_enabled": True,
                 "embedding_enabled_config": True, "upgrade_source_units": 5,
                 "repair_candidate_memories": 2, "live_bucket_unavailable": 25}
    async def inv(*args):
        return copy.deepcopy(inventory)
    monkeypatch.setattr(repair, "_inventory", inv)
    monkeypatch.setattr(repair, "_resolve_prompts", lambda *args: ("test-doc", "test-query"))
    return manager, cfg, inventory


def body_for(manager, cap=1200):
    view = asyncio.run(manager.preview())
    return {"expected_space_token": view["space_token"], "max_total_units": cap,
            "confirm_provider_use": True}


def join_job(manager):
    manager.thread.join(timeout=5)
    assert not manager.thread.is_alive()
    return manager.view()["job"]


def test_preview_is_read_only_and_confirmation_binds_full_config(tmp_path, monkeypatch):
    manager, cfg, _ = setup_job(tmp_path, monkeypatch)
    body = body_for(manager)
    assert not manager.path.exists()
    assert "never-export" not in json.dumps(manager.view())
    cfg["embedding"]["base_url"] = "https://example.invalid/another-endpoint"
    with pytest.raises(EmbeddingJobError, match="configuration_changed"):
        manager.start(body)
    assert not manager.path.exists()
    with pytest.raises(EmbeddingJobError, match="confirm_provider"):
        manager.start({})
    with pytest.raises(EmbeddingJobError, match="invalid_unit"):
        manager.start({**body_for(manager), "max_total_units": True})


def test_start_idempotence_safe_stop_and_restart_interruption(tmp_path, monkeypatch):
    manager, _, _ = setup_job(tmp_path, monkeypatch)
    entered = threading.Event()
    async def execution(cfg, job, phase):
        entered.set()
        while not manager.stop_event.is_set():
            await asyncio.sleep(0.01)
        manager._update(job, status="paused", result={"stop_reason": "paused"})
    monkeypatch.setattr(manager, "_execute", execution)
    body = body_for(manager)
    first = manager.start(body)["job"]["job_id"]
    assert entered.wait(2)
    assert manager.start(body)["job"]["job_id"] == first
    with pytest.raises(EmbeddingJobError, match="job_changed"):
        manager.stop({"job_id": "stale"})
    manager.stop({"job_id": first})
    assert join_job(manager)["status"] == "paused"
    stored = manager._read()
    manager._write({**stored, "status": "running"})
    restarted = MemoryEmbeddingJobs(manager.config_getter)
    assert restarted.view()["job"]["status"] == "interrupted"
    assert restarted.view()["space_token"] == ""


def test_job_lease_excludes_other_instances_and_writer_leases(tmp_path, monkeypatch):
    first = MemoryIndexLease(tmp_path)
    second = MemoryIndexLease(tmp_path)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()
    manager, _, _ = setup_job(tmp_path, monkeypatch)
    other = MemoryIndexLease(tmp_path, "embedding-index-job.lock")
    assert other.acquire()
    try:
        with pytest.raises(EmbeddingJobError, match="already_running"):
            manager.start(body_for(manager))
    finally:
        other.release()


def test_pilot_waits_for_review_before_full_and_keeps_whole_job_budget(tmp_path, monkeypatch):
    manager, _, inventory = setup_job(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(manager, "_backup", lambda *args: calls.append("backup"))
    async def apply(cfg, paths, inv, *, mode, total_unit_cap, progress_callback, stop_reason_callback):
        assert stop_reason_callback() == ""
        calls.append((mode, total_unit_cap))
        result = {"embedding_units_attempted": 2, "embedding_units_completed": 2,
                  "memories_checked": 1, "provider_call_invocations": 3 if mode == "pilot" else 2,
                  "stop_reason": "pilot_complete" if mode == "pilot" else "complete",
                  "new_query_verified": 1}
        inventory["upgrade_source_units"] = 3 if mode == "pilot" else 0
        progress_callback(result)
        return result
    monkeypatch.setattr(repair, "_apply_controller", apply)
    start_body = body_for(manager, 5)
    manager.start(start_body)
    job = join_job(manager)
    assert job["status"] == "awaiting_review"
    assert calls == ["backup", ("pilot", 5)]
    with pytest.raises(EmbeddingJobError, match="pilot_review_required"):
        manager.start(start_body)
    with pytest.raises(EmbeddingJobError, match="job_changed"):
        manager.continue_full({"job_id": "wrong", "expected_space_token": start_body["expected_space_token"],
                               "confirm_provider_use": True})
    manager.continue_full({"job_id": job["job_id"], "expected_space_token": start_body["expected_space_token"],
                           "confirm_provider_use": True})
    job = join_job(manager)
    assert job["status"] == "completed"
    assert calls == ["backup", ("pilot", 5), ("full", 3)]
    assert job["progress"]["embedding_units_completed"] == 4
    assert job["progress"]["provider_call_invocations"] == 5
    assert "never-export" not in manager.path.read_text()


def test_repeated_preview_same_config_keeps_pilot_continue_token(tmp_path, monkeypatch):
    manager, _, inventory = setup_job(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(MemoryEmbeddingJobs, "_backup", lambda *args: None)

    async def apply(cfg, paths, inv, *, mode, total_unit_cap, progress_callback, stop_reason_callback):
        assert stop_reason_callback() == ""
        calls.append((mode, total_unit_cap))
        result = {
            "embedding_units_attempted": 2,
            "embedding_units_completed": 2,
            "memories_checked": 1,
            "provider_call_invocations": 3 if mode == "pilot" else 2,
            "stop_reason": "pilot_complete" if mode == "pilot" else "complete",
            "new_query_verified": 1,
        }
        inventory["upgrade_source_units"] = 3 if mode == "pilot" else 0
        progress_callback(result)
        return result

    monkeypatch.setattr(repair, "_apply_controller", apply)
    first_preview = asyncio.run(manager.preview())
    start_body = {
        "expected_space_token": first_preview["space_token"],
        "max_total_units": 5,
        "confirm_provider_use": True,
    }
    manager.start(start_body)
    pilot = join_job(manager)
    assert pilot["status"] == "awaiting_review"

    refreshed_preview = asyncio.run(manager.preview())
    assert refreshed_preview["space_token"] == first_preview["space_token"]
    assert refreshed_preview["job"]["status"] == "awaiting_review"
    manager.continue_full({
        "job_id": pilot["job_id"],
        "expected_space_token": refreshed_preview["space_token"],
        "confirm_provider_use": True,
    })
    completed = join_job(manager)
    assert completed["status"] == "completed"
    assert calls == [("pilot", 5), ("full", 3)]


@pytest.mark.parametrize("invalidation", ["restart", "config_change", "prompt_change"])
def test_stale_review_must_be_cancelled_before_new_pilot(
    tmp_path, monkeypatch, invalidation
):
    manager, cfg, inventory = setup_job(tmp_path, monkeypatch)
    prompt = {"query": "query-v1"}
    monkeypatch.setattr(
        repair, "_resolve_prompts", lambda *args: ("test-doc", prompt["query"])
    )
    monkeypatch.setattr(MemoryEmbeddingJobs, "_backup", lambda *args: None)
    modes = []

    async def apply(cfg_arg, paths, inv, *, mode, total_unit_cap,
                    progress_callback, stop_reason_callback):
        assert stop_reason_callback() == ""
        modes.append(mode)
        pilot_index = modes.count("pilot")
        result = {
            "embedding_units_attempted": 2,
            "embedding_units_completed": 2,
            "memories_checked": 1,
            "provider_call_invocations": 3,
            "stop_reason": "pilot_complete" if mode == "pilot" else "complete",
            "new_query_verified": 1,
            "test_pilot_index": pilot_index,
        }
        inventory["upgrade_source_units"] = 3 if mode == "pilot" else 0
        progress_callback(result)
        return result

    monkeypatch.setattr(repair, "_apply_controller", apply)
    first_preview = asyncio.run(manager.preview())
    first_start = {
        "expected_space_token": first_preview["space_token"],
        "max_total_units": 5,
        "confirm_provider_use": True,
    }
    manager.start(first_start)
    first_job = join_job(manager)
    assert first_job["status"] == "awaiting_review"
    pilot_evidence = first_job["pilot_result"]

    if invalidation == "restart":
        resumed = MemoryEmbeddingJobs(manager.config_getter)
    elif invalidation == "config_change":
        cfg["embedding"]["base_url"] = "https://example.invalid/changed"
        resumed = manager
    else:
        prompt["query"] = "query-v2"
        resumed = manager

    stale_view = resumed.view()
    assert stale_view["job"]["status"] == "awaiting_review"
    assert stale_view["review_restart_required"] is True
    with pytest.raises(EmbeddingJobError):
        resumed.continue_full({
            "job_id": first_job["job_id"],
            "expected_space_token": first_start["expected_space_token"],
            "confirm_provider_use": True,
        })
    assert modes == ["pilot"]

    cancelled = resumed.stop({"job_id": first_job["job_id"]})
    assert cancelled["job"]["status"] == "cancelled"
    assert cancelled["job"]["pilot_result"] == pilot_evidence
    assert modes == ["pilot"]

    new_preview = asyncio.run(resumed.preview())
    new_start = {
        "expected_space_token": new_preview["space_token"],
        "max_total_units": 5,
        "confirm_provider_use": True,
    }
    resumed.start(new_start)
    new_job = join_job(resumed)
    assert new_job["job_id"] != first_job["job_id"]
    assert new_job["status"] == "awaiting_review"
    assert modes == ["pilot", "pilot"]

    history_files = list(
        (resumed.state_dir / "embedding-index-history").glob("*.json")
    )
    assert len(history_files) == 1
    archived_job = json.loads(history_files[0].read_text(encoding="utf-8"))
    assert archived_job["job_id"] == first_job["job_id"]
    assert archived_job["status"] == "cancelled"
    assert archived_job["pilot_result"] == pilot_evidence
    assert resumed._read()["job_id"] == new_job["job_id"]


def test_failed_pilot_never_runs_full_and_no_exception_text_leaks(tmp_path, monkeypatch):
    manager, _, _ = setup_job(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_backup", lambda *args: None)
    modes = []
    async def apply(*args, mode, **kwargs):
        modes.append(mode)
        return {"stop_reason": "query_smoke_failed", "new_query_verified": 0}
    monkeypatch.setattr(repair, "_apply_controller", apply)
    manager.start(body_for(manager))
    assert join_job(manager)["status"] == "failed"
    assert modes == ["pilot"]
    async def broken(*args):
        raise RuntimeError("never-export-provider-body")
    monkeypatch.setattr(manager, "_execute", broken)
    manager.start(body_for(manager))
    assert join_job(manager)["result"] == {"stop_reason": "error", "error_type": "RuntimeError"}
    assert "never-export" not in manager.path.read_text()


def test_cap_and_empty_preview_do_not_call_provider_or_backup(tmp_path, monkeypatch):
    manager, _, inventory = setup_job(tmp_path, monkeypatch)
    def forbidden(*args):
        raise AssertionError("must not reach backup")
    monkeypatch.setattr(manager, "_backup", forbidden)
    manager.start(body_for(manager, 4))
    assert join_job(manager)["result"]["stop_reason"] == "unit_cap_exceeded"
    inventory["upgrade_source_units"] = 0
    manager.start(body_for(manager))
    assert join_job(manager)["result"]["stop_reason"] == "no_candidates"
