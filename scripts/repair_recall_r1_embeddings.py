#!/usr/bin/env python3
"""Bounded, re-entrant RECALL-R1 embedding migration controller.

Inventory is read-only and is the default.  Provider calls and projection-status
writes require an explicit apply mode.  This script deliberately imports the
Brain stores directly and never imports ``server`` or any Memory commit path.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit


logging.disable(logging.CRITICAL)

_SCRIPT_PATH = Path(__file__)
ROOT = (
    _SCRIPT_PATH.resolve().parents[1]
    if _SCRIPT_PATH.exists()
    else Path.cwd().resolve()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from memory_authority import MemoryAuthorityStore
from memory_moments import _annotation_options_from_config
from memory_projection_worker import MemoryProjectionWorker, embedding_projection_needs_upgrade
from memory_relevance import memory_relevance_options_from_config
from prompt_plan_mirror import PromptPlanMirrorStore
from prompt_source_registry import resolve_fixed_prompt
from utils import load_config


MAX_TOTAL_UNITS = 1200
MAX_UNITS_PER_MEMORY = 7
MAX_PILOT_MEMORY_CHECKS = 25
EXPECTED_BUCKETS_DIR = Path("/data")
EXPECTED_STATE_DIR = Path("/state")
EXPECTED_AUTHORITY_DB = Path("/state/memory_authority.sqlite3")
R1_VECTOR_FIELDS = (
    "parent_memory_id", "parent_bucket_id", "source_unit_id",
    "source_revision", "source_sha256", "input_sha256",
    "preparation_sha256", "provider",
)
R1_VERIFICATION_FIELDS = set(R1_VECTOR_FIELDS) | {"input_truncated_chars"}
STOP_REASONS = {
    "inventory_only", "execute_flag_required", "pilot_approval_required",
    "unsafe_path_identity", "missing_store", "authority_disabled",
    "embedding_disabled", "embedding_schema_not_ready", "hash_mismatch",
    "unit_cap_exceeded", "no_candidates", "degraded_only", "budget_blocked",
    "provider_failure", "metadata_failure", "query_input_too_large",
    "query_smoke_failed", "concurrent_change", "no_progress", "remaining_deferred",
    "pilot_complete",
    "unit_cap_reached", "complete", "error",
}


class ReadOnlyPromptPlanMirror(PromptPlanMirrorStore):
    """Read an existing mirror without schema initialization or write access."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _open(self) -> sqlite3.Connection:
        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


class MeasuredEmbeddingEngine(EmbeddingEngine):
    """Controller-local counters; never retain request headers, bodies or keys."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.call_invocations = 0
        self.call_elapsed_ms: list[int] = []

    async def _request_embedding(self, *args, **kwargs):
        started = time.monotonic()
        self.call_invocations += 1
        try:
            return await super()._request_embedding(*args, **kwargs)
        finally:
            self.call_elapsed_ms.append(round((time.monotonic() - started) * 1000))


def _provider_diagnostics(embedding: Any) -> dict[str, Any]:
    """Last-call classification only: never serialize arbitrary runtime values."""
    try:
        runtime = embedding.runtime_debug()
    except Exception:
        runtime = {}
    if not isinstance(runtime, dict):
        runtime = {}

    def enum_value(key: str, allowed: set[str], missing: str) -> str:
        value = runtime.get(key, missing)
        return value if type(value) is str and value in allowed else "other"

    def bounded_int(key: str, lower: int, upper: int) -> int | None:
        value = runtime.get(key)
        return value if type(value) is int and lower <= value <= upper else None

    return {
        "last_status": enum_value("last_status", {
            "not_requested", "started", "disabled", "ok", "empty",
            "error", "cancelled",
        }, "not_requested"),
        "last_error_category": enum_value("last_error_category", {
            "", "connect", "pool", "read", "write", "remote_protocol",
            "timeout", "request_error", "http_status", "response_decode",
            "cancelled", "unknown",
        }, ""),
        "last_error_type": enum_value("last_error_type", {
            "", "HTTPError", "HTTPStatusError", "ConnectTimeout",
            "ConnectError", "PoolTimeout", "ReadTimeout", "ReadError",
            "WriteTimeout", "WriteError", "RemoteProtocolError",
            "LocalProtocolError", "TimeoutException", "TimeoutError",
            "RequestError", "JSONDecodeError", "ValueError", "TypeError",
            "RuntimeError", "CancelledError", "URLError", "OSError",
        }, ""),
        "last_http_status": bounded_int("last_http_status", 100, 599),
        "last_latency_ms": bounded_int("last_latency_ms", 0, 86400000),
    }


def _progress(*, mode: str, started: float, checked: int, completed: int,
              verified: int, degraded: int, embedding: MeasuredEmbeddingEngine) -> None:
    # Explicit field allowlist: do not emit arbitrary runtime/config dictionaries.
    print(json.dumps({
        "event": "progress", "mode": mode,
        "memories_checked": checked, "embedding_units_completed": completed,
        "verified_vector_rows": verified, "degraded_memories": degraded,
        "provider_call_invocations": embedding.call_invocations,
        "connection_fallback_count": int(embedding.runtime_debug().get("connection_fallback_count") or 0),
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }, sort_keys=True, separators=(",", ":")), flush=True)


def _emit(values: dict[str, Any], *, exit_code: int = 0) -> int:
    safe = {key: value for key, value in values.items() if value is not None}
    reason = str(safe.get("stop_reason") or "error")
    if reason not in STOP_REASONS:
        safe["stop_reason"] = "error"
    print(json.dumps(safe, ensure_ascii=True, sort_keys=True, separators=(",", ":")), flush=True)
    return exit_code


def _identity_id(config: dict) -> str:
    persona = config.get("persona") if isinstance(config.get("persona"), dict) else {}
    return str(persona.get("canonical_session_id") or "jiajia-main")


def _config_section(config: dict, name: str) -> dict:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _paths(config: dict) -> dict[str, Path]:
    buckets = Path(str(config.get("buckets_dir") or "")).resolve()
    state_dir = str(config.get("state_dir") or os.path.join(
        os.path.dirname(os.path.abspath(str(buckets))), "state",
    ))
    state = Path(state_dir).resolve()
    authority_cfg = config.get("memory_authority")
    authority_cfg = authority_cfg if isinstance(authority_cfg, dict) else {}
    authority = Path(str(
        authority_cfg.get("db_path") or (state / "memory_authority.sqlite3")
    )).resolve()
    gateway_cfg = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
    mirror = Path(str(
        gateway_cfg.get("prompt_plan_mirror_path")
        or (buckets / "prompt_plan_mirror.sqlite3")
    )).resolve()
    return {
        "buckets": buckets,
        "state": state,
        "authority": authority,
        "embeddings": (buckets / "embeddings.db").resolve(),
        "prompt_mirror": mirror,
    }


def _readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _bucket_ids_from_rows(rows: list[sqlite3.Row]) -> set[str]:
    return {
        str(row["bucket_id"] or row["memory_id"])
        for row in rows
        if str(row["bucket_id"] or row["memory_id"])
    }


def _active_enabled_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT m.memory_id,COALESCE(NULLIF(m.bucket_id,''),m.memory_id) AS bucket_id,"
        "m.active_revision,COALESCE(r.body_sha256,'') AS body_sha256,"
        "COALESCE(p.status,'missing') AS embedding_status,"
        "COALESCE(p.details_json,'{}') AS details_json "
        "FROM memories AS m "
        "LEFT JOIN memory_revisions AS r "
        "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
        "LEFT JOIN memory_projection_status AS p "
        "ON p.memory_id=m.memory_id AND p.memory_revision=m.active_revision "
        "AND p.projector='embedding' "
        "WHERE m.state='active' AND m.recall_policy='enabled' "
        "ORDER BY m.updated_at DESC,m.memory_id"
    ).fetchall()


def _expected_embedding_space(config: dict) -> dict:
    embedding = _config_section(config, "embedding")
    if not embedding.get("model"):
        return {}  # Preserve legacy inventory behavior when no target is configured.
    base_url = embedding.get("base_url") or _config_section(config, "dehydration").get("base_url")
    return {
        "model": str(embedding["model"]),
        "provider": str(urlsplit(str(base_url or "")).hostname or ""),
        "dimension": EmbeddingEngine._requested_dimension(embedding.get("dimensions")),
    }


def _is_upgrade_candidate(status: str, details_json: str, expected_space: dict | None = None) -> bool:
    try:
        details = json.loads(details_json or "{}")
    except (TypeError, ValueError):
        details = {}
    return embedding_projection_needs_upgrade(status, details, expected_space)


async def _inventory(config: dict, paths: dict[str, Path]) -> dict[str, Any]:
    started = time.monotonic()
    expected_space = _expected_embedding_space(config)
    output: dict[str, Any] = {
        "mode": "inventory",
        "stop_reason": "inventory_only",
        "path_identity_ok": bool(
            paths["buckets"] == EXPECTED_BUCKETS_DIR
            and paths["state"] == EXPECTED_STATE_DIR
            and paths["authority"] == EXPECTED_AUTHORITY_DB
        ),
        "authority_file_present": paths["authority"].is_file(),
        "embedding_file_present": paths["embeddings"].is_file(),
        "prompt_mirror_file_present": paths["prompt_mirror"].is_file(),
        "memory_authority_enabled": bool(
            _config_section(config, "memory_authority").get("enabled", False)
        ),
        "embedding_enabled_config": bool(
            _config_section(config, "embedding").get("enabled", True)
        ),
    }
    if not output["authority_file_present"] or not output["embedding_file_present"]:
        output["stop_reason"] = "missing_store"
        output["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return output

    authority = _readonly(paths["authority"])
    embeddings = _readonly(paths["embeddings"])
    try:
        authority_columns = {
            row["name"] for row in authority.execute("PRAGMA table_info(memories)")
        }
        if not {"memory_id", "bucket_id", "active_revision", "state", "recall_policy"} <= authority_columns:
            output["stop_reason"] = "missing_store"
            return output
        active_count = authority.execute(
            "SELECT COUNT(*) FROM memories WHERE state='active'"
        ).fetchone()[0]
        rows = _active_enabled_rows(authority)
        output["active_memories"] = int(active_count)
        output["active_enabled_memories"] = len(rows)
        output["embedding_projected"] = sum(
            1 for row in rows if row["embedding_status"] == "projected"
        )
        output["embedding_pending"] = sum(
            1 for row in rows if row["embedding_status"] == "pending_rebuild"
        )
        output["legacy_upgrade_candidates"] = sum(
            1 for row in rows if _is_upgrade_candidate(
                str(row["embedding_status"]), str(row["details_json"]), expected_space,
            )
        )
        output["active_revision_sha_mismatch"] = int(authority.execute(
            "SELECT COUNT(*) FROM memories m JOIN memory_revisions r "
            "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
            "WHERE m.state='active' AND m.recall_policy='enabled' "
            "AND (COALESCE(m.body_sha256,'')='' OR COALESCE(r.body_sha256,'')='' "
            "OR m.body_sha256<>r.body_sha256)"
        ).fetchone()[0])

        embedding_columns = {
            row["name"] for row in embeddings.execute("PRAGMA table_info(embeddings)")
        }
        required_present = R1_VERIFICATION_FIELDS <= embedding_columns
        output["embedding_schema_ready"] = required_present
        vector_fields = ["bucket_id"]
        if "parent_bucket_id" in embedding_columns:
            vector_fields.append("parent_bucket_id")
        vector_fields.extend(field for field in R1_VECTOR_FIELDS if field in embedding_columns)
        select_fields = ",".join(dict.fromkeys(vector_fields))
        vector_rows = embeddings.execute(
            f"SELECT {select_fields} FROM embeddings"
        ).fetchall()
        output["vector_rows"] = len(vector_rows)
        active_bucket_ids = _bucket_ids_from_rows(rows)
        active_vector_rows = [
            row for row in vector_rows
            if str(row["bucket_id"] or "") in active_bucket_ids
            or ("parent_bucket_id" in embedding_columns
                and str(row["parent_bucket_id"] or "") in active_bucket_ids)
        ]
        output["active_enabled_vector_rows"] = len(active_vector_rows)
        if not required_present:
            output["vector_rows_missing_r1_metadata"] = len(vector_rows)
            output["active_enabled_vector_rows_missing_r1_metadata"] = len(active_vector_rows)
        else:
            output["vector_rows_missing_r1_metadata"] = sum(
                1 for row in vector_rows
                if any(row[field] is None or str(row[field]).strip() == ""
                       for field in R1_VECTOR_FIELDS)
            )
            output["active_enabled_vector_rows_missing_r1_metadata"] = sum(
                1 for row in active_vector_rows
                if any(row[field] is None or str(row[field]).strip() == ""
                       for field in R1_VECTOR_FIELDS)
            )
    finally:
        authority.close()
        embeddings.close()

    bucket_manager = BucketManager(config)
    source_options = SimpleNamespace(
        relevance_options=memory_relevance_options_from_config(config),
        annotation_options=_annotation_options_from_config(config),
    )
    unit_counter = MemoryProjectionWorker(
        config=config,
        authority=None,
        bucket_manager=bucket_manager,
        moment_store=source_options,
    )
    live_roots = [
        Path(bucket_manager.permanent_dir).resolve(),
        Path(bucket_manager.dynamic_dir).resolve(),
    ]
    live_matches = body_mismatches = unavailable = 0
    source_units = max_units_per_memory = max_unit_chars = 0
    upgrade_source_units = repair_candidate_memories = 0
    for row in rows:
        bucket_id = str(row["bucket_id"] or row["memory_id"])
        bucket = await bucket_manager.get(bucket_id)
        bucket_path = Path(str((bucket or {}).get("path") or "")).resolve()
        if not bucket or not any(bucket_path.is_relative_to(root) for root in live_roots):
            unavailable += 1
            continue
        body_sha = str(row["body_sha256"] or "")
        observed_sha = hashlib.sha256(
            str(bucket.get("content") or "").encode("utf-8")
        ).hexdigest()
        if not body_sha or observed_sha != body_sha:
            body_mismatches += 1
            continue
        live_matches += 1
        units = unit_counter._embedding_source_units(bucket, str(row["memory_id"]))
        source_units += len(units)
        if _is_upgrade_candidate(
            str(row["embedding_status"]), str(row["details_json"]), expected_space,
        ):
            repair_candidate_memories += 1
            upgrade_source_units += len(units)
        max_units_per_memory = max(max_units_per_memory, len(units))
        for unit in units:
            max_unit_chars = max(max_unit_chars, len(unit["text"]))
    embedding_cfg = config.get("embedding") if isinstance(config.get("embedding"), dict) else {}
    try:
        max_chars = max(500, min(32000, int(embedding_cfg.get("max_chars", 6000))))
    except (TypeError, ValueError):
        max_chars = 6000
    output.update({
        "live_body_hash_matches": live_matches,
        "live_body_hash_mismatches": body_mismatches,
        "live_bucket_unavailable": unavailable,
        "source_units": source_units,
        "repair_candidate_memories": repair_candidate_memories,
        "upgrade_source_units": upgrade_source_units,
        "max_units_per_memory": max_units_per_memory,
        "max_source_unit_chars": max_unit_chars,
        "max_segment_chars": max_chars,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    })
    return output


def _next_candidate(authority_path: Path, *, expected_space: dict | None = None) -> dict[str, Any] | None:
    connection = _readonly(authority_path)
    try:
        rows = _active_enabled_rows(connection)
        now_ms = int(time.time() * 1000)
        for row in rows:
            status = str(row["embedding_status"])
            if not _is_upgrade_candidate(status, str(row["details_json"]), expected_space):
                continue
            try:
                details = json.loads(row["details_json"] or "{}")
            except (TypeError, ValueError):
                details = {}
            retry_after = int((details or {}).get("retry_after_ms") or 0)
            if retry_after > now_ms:
                continue
            return {
                "memory_id": str(row["memory_id"]),
                "bucket_id": str(row["bucket_id"]),
                "revision": int(row["active_revision"] or 0),
                "body_sha256": str(row["body_sha256"] or ""),
            }
        return None
    finally:
        connection.close()


def _upgrade_candidate_count(authority_path: Path, *, expected_space: dict | None = None) -> int:
    connection = _readonly(authority_path)
    try:
        rows = _active_enabled_rows(connection)
        return sum(
            1 for row in rows if _is_upgrade_candidate(
                str(row["embedding_status"]), str(row["details_json"]), expected_space,
            )
        )
    finally:
        connection.close()


def _projection_status(
    authority_path: Path, memory_id: str, revision: int,
) -> dict[str, Any] | None:
    connection = _readonly(authority_path)
    try:
        row = connection.execute(
            "SELECT status,source_sha256,details_json,updated_at "
            "FROM memory_projection_status WHERE memory_id=? AND memory_revision=? "
            "AND projector='embedding'",
            (memory_id, int(revision)),
        ).fetchone()
        if not row:
            return None
        try:
            details = json.loads(row["details_json"] or "{}")
        except (TypeError, ValueError):
            details = {}
        return {
            "status": str(row["status"]),
            "source_sha256": str(row["source_sha256"] or ""),
            "details": details if isinstance(details, dict) else {},
            "updated_at": str(row["updated_at"] or ""),
        }
    finally:
        connection.close()


def _active_revision(
    authority_path: Path, memory_id: str,
) -> dict[str, Any] | None:
    connection = _readonly(authority_path)
    try:
        row = connection.execute(
            "SELECT m.state,m.recall_policy,m.active_revision,"
            "COALESCE(NULLIF(m.bucket_id,''),m.memory_id) AS bucket_id,"
            "COALESCE(r.body_sha256,'') AS body_sha256 "
            "FROM memories m LEFT JOIN memory_revisions r "
            "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
            "WHERE m.memory_id=?",
            (memory_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def _resolve_prompts(config: dict, mirror: ReadOnlyPromptPlanMirror) -> tuple[str, str]:
    identity_id = _identity_id(config)
    document_instruction, _ = resolve_fixed_prompt(
        mirror, "ombre.memory_embedding_document_prep_prompt",
        config=config, identity_id=identity_id, conversation_id="",
    )
    query_instruction, _ = resolve_fixed_prompt(
        mirror, "ombre.memory_embedding_query_prep_prompt",
        config=config, identity_id=identity_id, conversation_id="",
    )
    return str(document_instruction or ""), str(query_instruction or "")


def _prepare_apply_components(
    config: dict, paths: dict[str, Path],
) -> tuple[MemoryProjectionWorker, EmbeddingEngine, ReadOnlyPromptPlanMirror]:
    for key in ("authority", "embeddings", "prompt_mirror"):
        if not paths[key].is_file():
            raise FileNotFoundError(key)
    state_dir = config.get("state_dir") or os.path.join(
        os.path.dirname(os.path.abspath(str(paths["buckets"]))), "state",
    )
    authority = MemoryAuthorityStore({
        **config,
        "state_dir": state_dir,
        "memory_authority_db_path": str(paths["authority"]),
    })
    bucket_manager = BucketManager(config)
    embedding = MeasuredEmbeddingEngine(config)
    mirror = ReadOnlyPromptPlanMirror(paths["prompt_mirror"])
    moment_options = SimpleNamespace(
        relevance_options=memory_relevance_options_from_config(config),
        annotation_options=_annotation_options_from_config(config),
    )
    worker = MemoryProjectionWorker(
        config=config,
        authority=authority,
        bucket_manager=bucket_manager,
        embedding_engine=embedding,
        moment_store=moment_options,
        prompt_plan_mirror=mirror,
    )
    return worker, embedding, mirror


async def _verify_projection_rows(
    *, config: dict, paths: dict[str, Path], worker: MemoryProjectionWorker,
    embedding: EmbeddingEngine, mirror: ReadOnlyPromptPlanMirror,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    memory_id = str(candidate["memory_id"])
    bucket_id = str(candidate["bucket_id"])
    revision = int(candidate["revision"])
    body_sha = str(candidate["body_sha256"])
    authority = _active_revision(paths["authority"], memory_id)
    status = _projection_status(paths["authority"], memory_id, revision)
    details = status["details"] if status else {}
    if (not authority or authority["state"] != "active"
            or authority["recall_policy"] != "enabled"
            or int(authority["active_revision"] or 0) != revision
            or str(authority["body_sha256"] or "") != body_sha
            or not status or status["status"] != "projected"
            or status["source_sha256"] != body_sha
            or details.get("metadata_complete") is not True
            or int(details.get("source_revision") or 0) != revision
            or str(details.get("source_sha256") or "") != body_sha
            or str(details.get("source_memory_id") or "") != memory_id
            or str(details.get("parent_bucket_id") or "") != bucket_id):
        return {"ok": False, "rows_checked": 0, "units_expected": 0}
    bucket_manager = worker.bucket_manager
    bucket = await bucket_manager.get(bucket_id)
    if not bucket:
        return {"ok": False, "rows_checked": 0, "units_expected": 0}
    live_roots = [
        Path(bucket_manager.permanent_dir).resolve(),
        Path(bucket_manager.dynamic_dir).resolve(),
    ]
    bucket_path = Path(str(bucket.get("path") or "")).resolve()
    if not any(bucket_path.is_relative_to(root) for root in live_roots):
        return {"ok": False, "rows_checked": 0, "units_expected": 0}
    observed_sha = hashlib.sha256(
        str(bucket.get("content") or "").encode("utf-8")
    ).hexdigest()
    if observed_sha != body_sha:
        return {"ok": False, "rows_checked": 0, "units_expected": 0}

    units = worker._embedding_source_units(bucket, memory_id)
    document_instruction, _query_instruction = _resolve_prompts(config, mirror)
    snapshot = embedding._query_config_snapshot(
        document_instruction=document_instruction,
    )
    preparation_sha = str(snapshot.get("document_preparation") or "")
    expected_input_sha = {}
    units_by_id = {}
    for unit in units:
        units_by_id[unit["source_unit_id"]] = unit
        prepared = embedding._prepare_document_snapshot(unit["text"], snapshot)
        expected_input_sha[unit["source_unit_id"]] = hashlib.sha256(
            prepared[:int(snapshot["max_chars"])].encode("utf-8")
        ).hexdigest()
    connection = _readonly(paths["embeddings"])
    try:
        rows = connection.execute(
            "SELECT embedding,model,dimension,parent_bucket_id,parent_memory_id,"
            "source_unit_id,source_revision,source_sha256,input_sha256,"
            "preparation_sha256,provider,input_truncated_chars "
            "FROM embeddings WHERE parent_memory_id=? AND parent_bucket_id=? "
            "AND source_revision=? AND source_sha256=?",
            (memory_id, bucket_id, revision, body_sha),
        ).fetchall()
    finally:
        connection.close()
    seen_units = set()
    valid_rows = 0
    expected_provider = str(snapshot.get("base_url") or "").rstrip("/")
    expected_model = str(snapshot.get("model") or "")
    try:
        expected_dimension = int(details.get("dimension") or 0)
    except (TypeError, ValueError):
        expected_dimension = 0
    provider_host = str(urlsplit(str(snapshot.get("base_url") or "")).hostname or "")
    details_metadata_match = bool(
        details.get("model") == expected_model
        and details.get("provider") == provider_host
        and details.get("preparation_sha256") == preparation_sha
        and expected_dimension > 0
        and int(details.get("source_unit_count") or 0) == len(units)
    )
    for row in rows:
        unit_id = str(row["source_unit_id"] or "")
        if unit_id not in expected_input_sha or unit_id in seen_units:
            continue
        try:
            vector = json.loads(row["embedding"])
            vector_dimension = len(vector) if isinstance(vector, list) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in vector
            ) else 0
        except (TypeError, ValueError):
            vector_dimension = 0
        prepared = embedding._prepare_document_snapshot(units_by_id[unit_id]["text"], snapshot)
        expected_truncated = max(0, len(prepared) - int(snapshot["max_chars"]))
        matches = bool(
            row["parent_bucket_id"] == bucket_id
            and row["parent_memory_id"] == memory_id
            and int(row["source_revision"] or 0) == revision
            and row["source_sha256"] == body_sha
            and row["model"] == expected_model
            and int(row["dimension"] or 0) == vector_dimension
            and vector_dimension > 0
            and vector_dimension == expected_dimension
            and row["input_sha256"] == expected_input_sha[unit_id]
            and row["preparation_sha256"] == preparation_sha
            and row["provider"] == expected_provider
            and int(row["input_truncated_chars"] or 0) == expected_truncated
        )
        if matches:
            seen_units.add(unit_id)
            valid_rows += 1
    return {
        "ok": (
            len(units) > 0 and details_metadata_match
            and len(rows) == len(units)
            and valid_rows == len(units)
        ),
        "rows_checked": valid_rows,
        "units_expected": len(units),
        "source_unit_ids": [unit["source_unit_id"] for unit in units],
        "document_instruction": document_instruction,
        "query_instruction": _query_instruction,
    }


async def _verify_new_query(
    *, embedding: EmbeddingEngine, candidate: dict[str, Any],
    projection: dict[str, Any],
) -> dict[str, Any]:
    if not projection.get("ok") or not projection.get("source_unit_ids"):
        return {"ok": False, "requests": 0, "input_chars": 0, "input_sha256": ""}
    # Reuse an earlier real owner utterance only as a transport smoke. This
    # single-eligible-candidate query cannot establish relevance or admission;
    # those require the subsequent formal chat + Full owner body acceptance.
    query_text = "继续完成。"
    query_instruction = str(projection["query_instruction"] or "")
    document_instruction = str(projection["document_instruction"] or "")
    snapshot = embedding._query_config_snapshot(
        query_instruction=query_instruction,
        document_instruction=document_instruction,
    )
    prepared = embedding._prepare_query_snapshot(query_text, snapshot)
    max_chars = int(snapshot["max_chars"])
    if max_chars <= 0 or len(prepared) > max_chars:
        return {"ok": False, "requests": 0, "input_chars": 0, "input_sha256": ""}
    api_input = prepared[:max_chars]
    cache_debug: dict[str, Any] = {}
    try:
        results = await embedding.search_similar_queries(
            [query_text],
            top_k=1,
            eligible_ids={str(candidate["bucket_id"])},
            index_metadata={
                str(candidate["bucket_id"]): {
                    "memory_id": str(candidate["memory_id"]),
                    "revision": int(candidate["revision"]),
                    "body_sha256": str(candidate["body_sha256"]),
                }
            },
            cache_debug=cache_debug,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
        )
    except Exception:
        return {
            "ok": False,
            "requests": int(cache_debug.get("provider_requests") or 0),
            "input_chars": len(api_input),
            "input_sha256": hashlib.sha256(api_input.encode("utf-8")).hexdigest(),
            "cache_status": "error",
        }
    requests = int(cache_debug.get("provider_requests") or 0)
    verified = any(
        str(item.get("bucket_id") or "") == str(candidate["bucket_id"])
        and str(item.get("index_status") or "") == "verified"
        for item in results
    )
    runtime = embedding.runtime_debug()
    last_status = str(runtime.get("last_status") or "")
    http_status = runtime.get("last_http_status")
    cache_status = str(cache_debug.get("status") or "")
    try:
        http_ok = 200 <= int(http_status) < 300
    except (TypeError, ValueError):
        http_ok = False
    transport_ok = bool(
        (requests == 1 and last_status == "ok" and http_ok)
        or (requests == 0 and cache_status == "hit"
            and runtime.get("last_query_cache_status") == "hit")
    )
    return {
        "ok": bool(verified and transport_ok),
        "requests": requests,
        "input_chars": len(api_input),
        "input_sha256": hashlib.sha256(api_input.encode("utf-8")).hexdigest(),
        "cache_status": cache_status,
    }


def _build_worker_config(config: dict, paths: dict[str, Path]) -> dict[str, Any]:
    worker_config = dict(config)
    worker_config["state_dir"] = str(paths["state"])
    worker_config["memory_authority_db_path"] = str(paths["authority"])
    return worker_config


async def _apply_controller(
    config: dict, paths: dict[str, Path], inventory: dict[str, Any],
    *, mode: str, total_unit_cap: int,
) -> dict[str, Any]:
    started = time.monotonic()
    if not inventory.get("path_identity_ok"):
        return {"mode": mode, "stop_reason": "unsafe_path_identity"}
    if not inventory.get("authority_file_present") or not inventory.get("embedding_file_present") or not inventory.get("prompt_mirror_file_present"):
        return {"mode": mode, "stop_reason": "missing_store"}
    authority_cfg = config.get("memory_authority") if isinstance(config.get("memory_authority"), dict) else {}
    if not authority_cfg.get("enabled"):
        return {"mode": mode, "stop_reason": "authority_disabled"}
    if not _config_section(config, "embedding").get("enabled", True):
        return {"mode": mode, "stop_reason": "embedding_disabled"}
    if not inventory.get("embedding_schema_ready"):
        return {"mode": mode, "stop_reason": "embedding_schema_not_ready"}
    if int(inventory.get("active_revision_sha_mismatch") or 0) > 0 or int(inventory.get("live_body_hash_mismatches") or 0) > 0:
        return {"mode": mode, "stop_reason": "hash_mismatch"}
    if mode == "full" and int(inventory.get("upgrade_source_units") or 0) > total_unit_cap:
        return {"mode": mode, "stop_reason": "unit_cap_exceeded"}
    embedding_cfg = _config_section(config, "embedding")
    dehydration_cfg = _config_section(config, "dehydration")
    if not str(
        embedding_cfg.get("api_key") or dehydration_cfg.get("api_key") or ""
    ).strip():
        return {"mode": mode, "stop_reason": "embedding_disabled"}

    worker, embedding, mirror = _prepare_apply_components(config, paths)
    document_instruction, _query_instruction = _resolve_prompts(config, mirror)
    expected_space = _expected_embedding_space(config)
    space_kwargs = {"expected_space": expected_space} if expected_space else {}
    max_memory_checks = MAX_PILOT_MEMORY_CHECKS if mode == "pilot" else 10000
    unit_attempts = completed_units = degraded_memories = checked_memories = 0
    verified_rows = query_requests = 0
    stop_reason = "no_candidates"
    pilot_verified = False
    query_result: dict[str, Any] = {
        "ok": False, "requests": 0, "input_chars": 0,
        "input_sha256": "", "cache_status": "",
    }
    try:
        _progress(mode=mode, started=started, checked=0, completed=0,
                  verified=0, degraded=0, embedding=embedding)
        last_progress = time.monotonic()
        while checked_memories < max_memory_checks:
            if checked_memories and time.monotonic() - last_progress >= 20:
                _progress(mode=mode, started=started, checked=checked_memories,
                          completed=completed_units, verified=verified_rows,
                          degraded=degraded_memories, embedding=embedding)
                last_progress = time.monotonic()
            if mode == "full" and unit_attempts >= total_unit_cap:
                stop_reason = "unit_cap_reached"
                break
            candidate = _next_candidate(paths["authority"], **space_kwargs)
            if candidate is None:
                if _upgrade_candidate_count(paths["authority"], **space_kwargs):
                    stop_reason = "remaining_deferred"
                else:
                    stop_reason = "complete" if mode == "full" else "no_candidates"
                break
            remaining_cap = total_unit_cap - unit_attempts
            if remaining_cap <= 0:
                stop_reason = "unit_cap_reached"
                break
            per_memory_cap = min(MAX_UNITS_PER_MEMORY, remaining_cap)
            before = _projection_status(
                paths["authority"], candidate["memory_id"], candidate["revision"],
            )
            active_before = _active_revision(
                paths["authority"], candidate["memory_id"],
            )
            if (not before or not _is_upgrade_candidate(
                    before["status"], json.dumps(before["details"]), expected_space,
                    ) or not active_before or active_before["state"] != "active"
                    or active_before["recall_policy"] != "enabled"
                    or int(active_before["active_revision"] or 0) != int(candidate["revision"])
                    or str(active_before["body_sha256"] or "") != candidate["body_sha256"]):
                stop_reason = "concurrent_change"
                break
            result = await worker.repair_pending_once(
                limit=1,
                upgrade_legacy_indexes=True,
                max_embedding_units=per_memory_cap,
                projectors=("embedding",),
            )
            checked_memories += 1
            budget_left = result.get("embedding_units_budget_remaining")
            used = per_memory_cap - int(budget_left or 0)
            if used < 0 or used > per_memory_cap:
                stop_reason = "no_progress"
                break
            unit_attempts += used
            completed_units += int(result.get("embedding_units_completed") or 0)
            degraded_memories += int(result.get("degraded") or 0)
            if int(result.get("degraded") or 0) > 0 and used == 0:
                # Missing/archived/changed source Buckets are marked degraded
                # by the worker and never sent to the provider.  Continue so
                # each other eligible Memory receives the same fair treatment.
                continue
            if int(result.get("projected") or 0) != 1:
                if int(result.get("deferred") or 0) > 0:
                    stop_reason = "budget_blocked" if used == 0 else "provider_failure"
                else:
                    stop_reason = "no_progress"
                break
            if int(result.get("embedding_units_completed") or 0) <= 0:
                stop_reason = "no_progress"
                break
            post = _projection_status(
                paths["authority"], candidate["memory_id"], candidate["revision"],
            )
            if (not post or post["updated_at"] == (before or {}).get("updated_at")
                    or post["status"] != "projected"
                    or post["details"].get("metadata_complete") is not True):
                stop_reason = "metadata_failure"
                break
            postimage = await _verify_projection_rows(
                config=config,
                paths=paths,
                worker=worker,
                embedding=embedding,
                mirror=mirror,
                candidate=candidate,
            )
            if not postimage.get("ok"):
                stop_reason = "metadata_failure"
                break
            verified_rows += int(postimage.get("rows_checked") or 0)
            if mode == "pilot":
                query_result = await _verify_new_query(
                    embedding=embedding,
                    candidate=candidate,
                    projection=postimage,
                )
                query_requests = int(query_result.get("requests") or 0)
                if not query_result.get("ok"):
                    stop_reason = "query_smoke_failed"
                    break
                pilot_verified = True
                stop_reason = "pilot_complete"
                break
        else:
            stop_reason = "degraded_only" if mode == "pilot" else "unit_cap_reached"
        final_inventory = await _inventory(config, paths)
        return {
            "mode": mode,
            "stop_reason": stop_reason,
            "memories_checked": checked_memories,
            "degraded_memories": degraded_memories,
            "embedding_units_attempted": unit_attempts,
            "embedding_units_completed": completed_units,
            "embedding_units_remaining": int(
                final_inventory.get("upgrade_source_units") or 0
            ),
            "verified_vector_rows": verified_rows,
            "provider_call_invocations": embedding.call_invocations,
            "connection_fallback_count": int(embedding.runtime_debug().get("connection_fallback_count") or 0),
            "provider_call_total_ms": sum(embedding.call_elapsed_ms),
            "provider_call_max_ms": max(embedding.call_elapsed_ms, default=0),
            "provider_diagnostics": _provider_diagnostics(embedding),
            "new_query_verified": int(pilot_verified),
            "query_provider_requests": query_requests,
            "query_input_chars": int(query_result.get("input_chars") or 0)
                if mode == "pilot" else 0,
            "query_input_sha256": str(query_result.get("input_sha256") or "")
                if mode == "pilot" else "",
            "query_cache_status": (
                str(query_result.get("cache_status") or "")
                if query_result.get("cache_status") in {
                    "hit", "miss", "singleflight", "disabled_or_empty",
                } else "other"
            ) if mode == "pilot" else "",
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
    finally:
        await embedding.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only RECALL-R1 embedding inventory; apply requires explicit mode and flag."
    )
    parser.add_argument("--mode", choices=("inventory", "pilot", "full"), default="inventory")
    parser.add_argument("--execute", action="store_true", help="Enable the explicitly selected provider/write mode.")
    parser.add_argument("--pilot-verified", action="store_true", help="Required to select full mode after a reviewed pilot.")
    parser.add_argument("--max-total-units", type=int, default=MAX_TOTAL_UNITS)
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    config = load_config()
    paths = _paths(config)
    inventory = await _inventory(config, paths)
    if args.mode == "inventory":
        return _emit(inventory)
    if not args.execute:
        inventory.update(mode=args.mode, stop_reason="execute_flag_required")
        return _emit(inventory, exit_code=2)
    if args.mode == "full" and not args.pilot_verified:
        inventory.update(mode=args.mode, stop_reason="pilot_approval_required")
        return _emit(inventory, exit_code=2)
    requested_cap = int(args.max_total_units)
    if requested_cap <= 0 or requested_cap > MAX_TOTAL_UNITS:
        inventory.update(mode=args.mode, stop_reason="unit_cap_exceeded")
        return _emit(inventory, exit_code=2)
    if args.mode == "pilot":
        if int(inventory.get("upgrade_source_units") or 0) <= 0:
            inventory.update(mode=args.mode, stop_reason="no_candidates")
            return _emit(inventory, exit_code=2)
        total_cap = min(MAX_UNITS_PER_MEMORY, requested_cap)
    else:
        total_cap = requested_cap
    applied = await _apply_controller(
        config, paths, inventory, mode=args.mode, total_unit_cap=total_cap,
    )
    return _emit(applied, exit_code=0 if applied.get("stop_reason") in {
        "pilot_complete", "complete", "no_candidates",
    } else 2)


def main() -> int:
    args = _parser().parse_args()
    try:
        return asyncio.run(_async_main(args))
    except Exception as exc:
        return _emit({
            "mode": args.mode,
            "stop_reason": "error",
            "error_type": type(exc).__name__,
        }, exit_code=2)


if __name__ == "__main__":
    raise SystemExit(main())
