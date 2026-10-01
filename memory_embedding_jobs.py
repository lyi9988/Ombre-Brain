"""Owner-started, resumable maintenance of existing Memory vector projections.

Only operational state is stored here. Facts/revisions remain in MemoryAuthority;
all index writes use the same bounded repair controller as the operator CLI.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import uuid

from memory_index_lease import MemoryIndexLease
from scripts import repair_recall_r1_embeddings as repair


ACTIVE = {"running", "stopping"}


class EmbeddingJobError(ValueError):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


class MemoryEmbeddingJobs:
    def __init__(self, config_getter):
        self.config_getter = config_getter
        cfg = config_getter()
        self.state_dir = repair._paths(cfg)["state"]
        self.path = self.state_dir / "embedding-index-job.json"
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.inventory = None
        self.preview_config = None
        self.preview_prompts = None
        self.preview_token = ""

    def _read(self):
        try:
            result = json.loads(self.path.read_text(encoding="utf-8"))
            return result if isinstance(result, dict) else None
        except (FileNotFoundError, ValueError):
            return None

    def _write(self, job, *, path=None):
        # Same-directory replace is safe: this file is not a bind-mount target.
        path = self.path if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(job, stream, ensure_ascii=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _space(config):
        return repair._expected_embedding_space(config)

    @staticmethod
    def _prompts(config):
        return repair._resolve_prompts(config, repair.ReadOnlyPromptPlanMirror(
            repair._paths(config)["prompt_mirror"]))

    def view(self):
        with self.lock:
            job = self._read()
            if job and job.get("status") in ACTIVE and not (self.thread and self.thread.is_alive()):
                probe = MemoryIndexLease(self.state_dir, "embedding-index-job.lock")
                if probe.acquire():
                    probe.release()
                    job = {**job, "status": "interrupted", "stop_requested": False}
                    self._write(job)
            cfg = self.config_getter()
            confirmation_current = self.preview_config == cfg and self.preview_prompts == self._prompts(cfg)
            token = self.preview_token if confirmation_current else ""
            inventory = copy.deepcopy(self.inventory) if confirmation_current else None
            return {"status": "ok", "space": self._space(cfg), "space_token": token,
                    "job": job, "inventory": inventory,
                    "review_restart_required": bool(job and job.get("status") == "awaiting_review"
                        and (not token or job.get("space_token") != token))}

    async def preview(self):
        cfg = copy.deepcopy(self.config_getter())
        prompts = self._prompts(cfg)
        inventory = await asyncio.to_thread(lambda: asyncio.run(repair._inventory(cfg, repair._paths(cfg))))
        with self.lock:
            if cfg != self.config_getter() or prompts != self._prompts(cfg):
                raise EmbeddingJobError("embedding_configuration_changed")
            self.inventory = inventory
            # Bind confirmation to the complete in-memory snapshot, without
            # serializing or hashing any credential. A restart needs a new preview.
            # Merely refreshing the preview must not invalidate a completed
            # pilot. A changed configuration or process restart still requires
            # cancelling the old review and explicitly starting a new pilot.
            if self.preview_config != cfg or self.preview_prompts != prompts or not self.preview_token:
                self.preview_token = uuid.uuid4().hex
            self.preview_config = cfg
            self.preview_prompts = prompts
        return self.view()

    def start(self, body):
        if not isinstance(body, dict) or body.get("confirm_provider_use") is not True:
            raise EmbeddingJobError("confirm_provider_use_required", 400)
        cap = body.get("max_total_units", repair.MAX_TOTAL_UNITS)
        if type(cap) is not int or not 1 <= cap <= repair.MAX_TOTAL_UNITS:
            raise EmbeddingJobError("invalid_unit_budget", 400)
        with self.lock:
            current = self.view()
            if current["job"] and current["job"].get("status") in ACTIVE:
                return current
            if current["job"] and current["job"].get("status") == "awaiting_review":
                raise EmbeddingJobError("pilot_review_required")
            cfg = copy.deepcopy(self.config_getter())
            if (not self.preview_token or self.preview_config != cfg or self.preview_prompts != self._prompts(cfg)
                    or body.get("expected_space_token") != self.preview_token):
                raise EmbeddingJobError("embedding_configuration_changed")
            job_lease = MemoryIndexLease(self.state_dir, "embedding-index-job.lock")
            if not job_lease.acquire():
                raise EmbeddingJobError("embedding_job_already_running")
            job = {"job_id": uuid.uuid4().hex, "status": "running", "phase": "backup",
                   "started_at_ms": int(time.time() * 1000), "stop_requested": False,
                   "space": self._space(cfg), "space_token": self.preview_token,
                   "max_total_units": cap, "progress": {}, "result": {}}
            try:
                if current["job"]:
                    # Preserve maintenance evidence when a new pilot replaces
                    # the current job; no Memory facts or vectors are removed.
                    self._write(current["job"], path=self.state_dir / "embedding-index-history" /
                                (uuid.uuid4().hex + ".json"))
                self._write(job)
                self.stop_event.clear()
                self.thread = threading.Thread(target=self._run, args=(cfg, job, job_lease, "pilot"), daemon=True)
                self.thread.start()
            except BaseException:
                job_lease.release()
                raise
            return self.view()

    def continue_full(self, body):
        if not isinstance(body, dict) or body.get("confirm_provider_use") is not True:
            raise EmbeddingJobError("confirm_provider_use_required", 400)
        with self.lock:
            job = self._read()
            if (not job or body.get("job_id") != job.get("job_id")
                    or job.get("status") != "awaiting_review"):
                raise EmbeddingJobError("embedding_job_changed")
            cfg = copy.deepcopy(self.config_getter())
            if (not self.preview_token or self.preview_config != cfg or self.preview_prompts != self._prompts(cfg)
                    or body.get("expected_space_token") != self.preview_token
                    or job.get("space_token") != self.preview_token):
                raise EmbeddingJobError("embedding_configuration_changed")
            pilot = job.get("pilot_result") or {}
            if pilot.get("stop_reason") != "pilot_complete" or pilot.get("new_query_verified") != 1:
                raise EmbeddingJobError("pilot_not_verified")
            job_lease = MemoryIndexLease(self.state_dir, "embedding-index-job.lock")
            if not job_lease.acquire():
                raise EmbeddingJobError("embedding_job_already_running")
            try:
                job.update(status="running", phase="full", stop_requested=False)
                self._write(job)
                self.stop_event.clear()
                self.thread = threading.Thread(target=self._run, args=(cfg, job, job_lease, "full"), daemon=True)
                self.thread.start()
            except BaseException:
                job_lease.release()
                raise
            return self.view()

    def stop(self, body):
        with self.lock:
            job = self._read()
            if not isinstance(body, dict) or not job or body.get("job_id") != job.get("job_id"):
                raise EmbeddingJobError("embedding_job_changed")
            if job.get("status") in ACTIVE:
                self.stop_event.set()
                job.update(status="stopping", stop_requested=True)
                self._write(job)
            elif job.get("status") == "awaiting_review":
                job_lease = MemoryIndexLease(self.state_dir, "embedding-index-job.lock")
                if not job_lease.acquire():
                    raise EmbeddingJobError("embedding_job_already_running")
                try:
                    job.update(status="cancelled", stop_requested=False,
                               finished_at_ms=int(time.time() * 1000))
                    job["result"] = {**(job.get("result") or {}), "stop_reason": "owner_cancelled_review"}
                    self._write(job)
                finally:
                    job_lease.release()
        return self.view()

    def _update(self, job, **values):
        with self.lock:
            job.update(values)
            if self.stop_event.is_set() and job.get("status") == "running":
                job.update(status="stopping", stop_requested=True)
            self._write(job)

    def _backup(self, cfg, paths, job):
        folder = self.state_dir / "embedding-index-backups" / job["job_id"]
        folder.mkdir(parents=True, mode=0o700, exist_ok=False)
        # Backup only derived vectors and projection bookkeeping, never restore
        # a complete Authority DB over newer owner facts.
        with closing(repair._readonly(paths["embeddings"])) as source:
            target_path = folder / "embeddings.db"
            fd = os.open(target_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            with closing(sqlite3.connect(target_path)) as target:
                source.backup(target)
        with closing(repair._readonly(paths["authority"])) as authority:
            rows = [dict(row) for row in authority.execute(
                "SELECT * FROM memory_projection_status WHERE projector='embedding'")]
        fd = os.open(folder / "projection-status.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
        runtime = Path(cfg.get("_runtime_config_path") or self.state_dir / "config.runtime.yaml")
        if runtime.is_file():
            target = folder / "config.runtime.yaml"
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as output, runtime.open("rb") as source:
                shutil.copyfileobj(source, output)

    def _run(self, cfg, job, job_lease, phase):
        writer = MemoryIndexLease(self.state_dir)
        try:
            # Never race the normal outbox/repair worker or a CLI rebuild.
            deadline = time.monotonic() + 30
            while not writer.acquire():
                if self.stop_event.wait(0.25) or time.monotonic() >= deadline:
                    reason = "paused" if self.stop_event.is_set() else "maintenance_busy"
                    self._update(job, status="paused", result={"stop_reason": reason})
                    return
            asyncio.run(self._execute(cfg, job, phase))
        except Exception as exc:
            # Exception messages may contain provider bodies/URLs; never store them.
            self._update(job, status="failed", result={"stop_reason": "error", "error_type": type(exc).__name__})
        finally:
            writer.release()
            job_lease.release()

    async def _execute(self, cfg, job, phase):
        started = time.monotonic()
        paths = repair._paths(cfg)
        inventory = await repair._inventory(cfg, paths)
        self.inventory = inventory
        cap = job["max_total_units"]
        if (not inventory.get("path_identity_ok") or not inventory.get("embedding_schema_ready")
                or not inventory.get("prompt_mirror_file_present")
                or not inventory.get("memory_authority_enabled") or not inventory.get("embedding_enabled_config")
                or inventory.get("live_body_hash_mismatches") or inventory.get("active_revision_sha_mismatch")):
            self._update(job, status="failed", result={"stop_reason": "preflight_failed"})
            return
        if int(inventory.get("upgrade_source_units") or 0) > cap:
            self._update(job, status="paused", result={"stop_reason": "unit_cap_exceeded"})
            return
        if not int(inventory.get("upgrade_source_units") or 0):
            self._update(job, status="completed", result={"stop_reason": "no_candidates", "embedding_units_remaining": 0})
            return
        mirror = repair.ReadOnlyPromptPlanMirror(paths["prompt_mirror"])
        original_prompts = repair._resolve_prompts(cfg, mirror)

        def should_stop():
            if self.stop_event.is_set():
                return "paused"
            current = self.config_getter()
            if (current != cfg
                    or job.get("space_token") != self.preview_token
                    or original_prompts != self.preview_prompts
                    or repair._resolve_prompts(current, mirror) != original_prompts):
                return "config_changed"
            return ""

        if should_stop():
            self._update(job, status="paused", result={"stop_reason": should_stop()})
            return
        if phase == "pilot":
            self._backup(cfg, paths, job)
        pilot = job.get("pilot_result") or {}
        totals = {key: int(pilot.get(key) or 0) if phase == "full" else 0
                  for key in ("embedding_units_completed", "memories_checked", "provider_call_invocations")}

        def progress(values):
            merged = {key: totals[key] + int(values.get(key) or 0) for key in totals}
            merged["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            self._update(job, progress=merged)

        if phase == "pilot":
            self._update(job, phase="pilot", inventory=inventory)
            pilot = await repair._apply_controller(
                cfg, paths, inventory, mode="pilot", total_unit_cap=min(cap, repair.MAX_UNITS_PER_MEMORY),
                progress_callback=progress, stop_reason_callback=should_stop)
            progress(pilot)
            self._update(job, pilot_result=pilot)
            if pilot.get("stop_reason") != "pilot_complete" or pilot.get("new_query_verified") != 1:
                status = "paused" if pilot.get("stop_reason") in {"paused", "config_changed", "remaining_deferred"} else "failed"
                self._update(job, status=status, result=pilot)
                return
            inventory = await repair._inventory(cfg, paths)
            self.inventory = inventory
            self._update(job, status="awaiting_review", phase="pilot", result={**pilot,
                "embedding_units_remaining": inventory.get("upgrade_source_units")})
            return
        remaining = cap - int(pilot.get("embedding_units_attempted") or 0)
        inventory = await repair._inventory(cfg, paths)
        self.inventory = inventory
        if int(inventory.get("upgrade_source_units") or 0) == 0:
            self._update(job, status="completed", result={**pilot, "stop_reason": "complete", "embedding_units_remaining": 0})
            return
        if remaining <= 0:
            self._update(job, status="paused", result={"stop_reason": "unit_cap_reached", "embedding_units_remaining": inventory.get("upgrade_source_units")})
            return
        self._update(job, phase="full")
        result = await repair._apply_controller(
            cfg, paths, inventory, mode="full", total_unit_cap=remaining,
            progress_callback=progress, stop_reason_callback=should_stop)
        progress(result)
        reason = result.get("stop_reason")
        status = "completed" if reason in {"complete", "no_candidates"} else (
            "paused" if reason in {"paused", "config_changed", "unit_cap_reached", "unit_cap_exceeded", "remaining_deferred"} else "failed")
        self.inventory = await repair._inventory(cfg, paths)
        self._update(job, status=status, result=result, finished_at_ms=int(time.time() * 1000))
