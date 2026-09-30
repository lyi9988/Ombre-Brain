"""Outbox-driven derived-index projection for RECALL-R1."""
from __future__ import annotations

import logging
import sqlite3
import hashlib
import inspect
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from entity_edges import extract_entity_edges_from_bucket
from identity import identity_names
from memory_authority import MemoryAuthorityStore
from memory_moments import parse_bucket_moments
from prompt_source_registry import fixed_prompt_source, render_factory_body, resolve_fixed_prompt
from utils import bucket_text_for_embedding


logger = logging.getLogger("ombre_brain.memory_projection")


class MemoryProjectionWorker:
    def __init__(
        self,
        *,
        config: dict,
        authority: MemoryAuthorityStore,
        bucket_manager,
        embedding_engine=None,
        moment_store=None,
        node_store=None,
        entity_edge_store=None,
        word_map_store=None,
        identity_semantic_store=None,
        prompt_plan_mirror=None,
    ):
        self.config = dict(config or {})
        self.authority = authority
        self.bucket_manager = bucket_manager
        self.embedding_engine = embedding_engine
        self.moment_store = moment_store
        self.node_store = node_store
        self.entity_edge_store = entity_edge_store
        self.word_map_store = word_map_store
        self.identity_semantic_store = identity_semantic_store
        self.prompt_plan_mirror = prompt_plan_mirror
        self.identity = identity_names(self.config)

    async def run_once(self, *, limit: int = 10) -> dict[str, Any]:
        recovered = self.authority.recover_stale_outbox()
        events = self.authority.claim_outbox(limit=limit)
        projected = degraded = 0
        results = []
        for event in events:
            result = await self.project_event(event)
            results.append(result)
            if result["status"] == "projected":
                projected += 1
            else:
                degraded += 1
        return {
            "claimed": len(events),
            "projected": projected,
            "degraded": degraded,
            "recovered_stale": recovered,
            "results": results,
        }

    async def repair_pending_once(
        self, *, limit: int = 1, upgrade_legacy_indexes: bool = False,
        max_embedding_units: int | None = None,
        projectors: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Bounded background repair of missing auto-recall indexes.

        Runs in the same scheduler as outbox projection, after normal events.
        The source Memory revision and Bucket body must agree before an index
        is written. ``upgrade_legacy_indexes`` opts into embedding-only repair
        of projected legacy rows that do not prove complete revision metadata.
        A unit budget is all-or-nothing per Memory so a projection never writes
        only part of that Memory's source units due to budget exhaustion.
        """
        budget = max(1, min(20, int(limit)))
        allowed_projectors = ("embedding",) if upgrade_legacy_indexes else (
            "embedding", "entity_edges",
        )
        if projectors is not None:
            requested = {str(item) for item in projectors}
            allowed_projectors = tuple(
                item for item in allowed_projectors if item in requested
            )
        unit_budget = (
            None if max_embedding_units is None
            else max(0, int(max_embedding_units))
        )
        result: dict[str, Any] = {
            "attempted": 0, "projected": 0, "deferred": 0, "degraded": 0,
            "embedding_units_completed": 0,
            "embedding_units_remaining": 0,
        }
        offset = 0
        while result["attempted"] < budget:
            page = self.authority.list_memories(state="active", limit=500, offset=offset)
            if not page:
                break
            offset += len(page)
            for memory in page:
                if result["attempted"] >= budget:
                    break
                if memory.get("recall_policy") != "enabled":
                    continue
                memory_id = str(memory.get("memory_id") or "")
                revision = int(memory.get("active_revision") or 0)
                source_sha = str(memory.get("body_sha256") or "")
                statuses = {
                    str(item.get("projector") or ""): item
                    for item in self.authority.list_projection_status(memory_id, revision=revision)
                }
                pending = []
                for name in allowed_projectors:
                    status = statuses.get(name) or {}
                    is_pending = status.get("status") == "pending_rebuild"
                    details = status.get("details") or {}
                    if (upgrade_legacy_indexes and name == "embedding"
                            and status.get("status") == "projected"
                            and details.get("metadata_complete") is not True):
                        is_pending = True
                    if is_pending:
                        pending.append(name)
                if not pending:
                    continue
                now_ms = int(time.time() * 1000)
                pending = [
                    name for name in pending
                    if int(((statuses[name].get("details") or {}).get("retry_after_ms") or 0)) <= now_ms
                ]
                if not pending:
                    result["deferred"] += 1
                    continue
                result["attempted"] += 1
                bucket_id = str(memory.get("bucket_id") or memory_id)
                bucket = await self.bucket_manager.get(bucket_id)
                live_dirs = [
                    Path(self.bucket_manager.permanent_dir).resolve(),
                    Path(self.bucket_manager.dynamic_dir).resolve(),
                ]
                bucket_path = Path(str((bucket or {}).get("path") or "")).resolve()
                source_valid = bool(
                    bucket and source_sha
                    and any(bucket_path.is_relative_to(root) for root in live_dirs)
                    and hashlib.sha256(str(bucket.get("content") or "").encode("utf-8")).hexdigest()
                    == source_sha
                )
                if not source_valid:
                    for name in pending:
                        self._record_status(
                            memory_id, revision, name, "degraded", source_sha,
                            {"reason": "source_bucket_missing_archived_or_changed"},
                        )
                        result["degraded"] += 1
                    continue
                for name in pending:
                    embedding_units: list[dict[str, str]] | None = None
                    unit_count = 0
                    progress = {"attempted": 0, "completed": 0}
                    try:
                        if name == "embedding":
                            if not self.embedding_engine or not self.embedding_engine.enabled:
                                result["deferred"] += 1
                                continue
                            embedding_units = self._embedding_source_units(bucket, memory_id)
                            engine_method = getattr(
                                self.embedding_engine, "generate_and_store", None
                            )
                            if engine_method is None:
                                raise RuntimeError("embedding_generate_method_missing")
                            if not self._accepts_revision_metadata(engine_method):
                                embedding_units = embedding_units[:1]
                            unit_count = len(embedding_units)
                            if unit_count <= 0:
                                raise RuntimeError("embedding_source_unit_missing")
                            if unit_budget is not None and unit_count > unit_budget:
                                result["embedding_units_remaining"] += unit_count
                                self._record_status(
                                    memory_id, revision, name, "pending_rebuild", source_sha,
                                    {
                                        "reason": "embedding_unit_budget_insufficient",
                                        "source_unit_count": unit_count,
                                        "available_unit_budget": unit_budget,
                                        "metadata_complete": False,
                                        "coverage_status": "incomplete",
                                        "outdated": True,
                                        "index_unchanged": True,
                                    },
                                )
                                result["deferred"] += 1
                                continue
                            embedding_result = await self._project_embedding_units(
                                bucket, memory_id, revision, source_sha,
                                source_units=embedding_units,
                                progress=progress,
                            )
                            if not await self.embedding_engine.get_embedding(bucket_id):
                                raise RuntimeError("embedding_not_stored")
                            count = int(embedding_result.get("source_unit_count") or 0)
                        else:
                            if self.entity_edge_store is None:
                                result["deferred"] += 1
                                continue
                            edges = extract_entity_edges_from_bucket(bucket, self.identity)
                            self.entity_edge_store.replace_bucket_edges(bucket_id, edges)
                            count = len(edges)
                        latest = self.authority.get_memory(memory_id)
                        if (not latest or latest.get("state") != "active"
                                or latest.get("recall_policy") != "enabled"
                                or int(latest.get("active_revision") or 0) != revision
                                or str(latest.get("body_sha256") or "") != source_sha):
                            if name == "embedding":
                                result["embedding_units_completed"] += int(
                                    progress.get("completed") or 0
                                )
                                result["embedding_units_remaining"] += unit_count
                                if unit_budget is not None:
                                    unit_budget = max(
                                        0, unit_budget - int(progress.get("attempted") or 0)
                                    )
                            result["deferred"] += 1
                            continue
                        details = {"reason": "bounded_background_repair", "result_count": count}
                        if name == "embedding":
                            details.update(embedding_result)
                        self._record_status(
                            memory_id, revision, name, "projected", source_sha, details,
                        )
                        if name == "embedding":
                            result["embedding_units_completed"] += int(
                                progress.get("completed") or 0
                            )
                            if unit_budget is not None:
                                unit_budget = max(
                                    0, unit_budget - int(progress.get("attempted") or 0)
                                )
                            result["embedding_units_remaining"] = max(
                                0,
                                result["embedding_units_remaining"] - unit_count,
                            )
                        result["projected"] += 1
                    except Exception as exc:
                        if name == "embedding":
                            result["embedding_units_completed"] += int(
                                progress.get("completed") or 0
                            )
                            result["embedding_units_remaining"] += unit_count
                            if unit_budget is not None:
                                unit_budget = max(
                                    0, unit_budget - int(progress.get("attempted") or 0)
                                )
                        retry_ms = int(time.time() * 1000) + 3600 * 1000
                        self._record_status(
                            memory_id, revision, name, "pending_rebuild", source_sha,
                            {"reason": "repair_retry", "error_type": type(exc).__name__,
                             "retry_after_ms": retry_ms},
                        )
                        result["deferred"] += 1
        result["embedding_units_budget_remaining"] = unit_budget
        return result

    async def project_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_id = str(event.get("event_id") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        memory_id = str(payload.get("memory_id") or event.get("aggregate_id") or "")
        memory = self.authority.get_memory(memory_id)
        if not memory:
            self.authority.set_outbox_status(event_id, status="degraded", error="memory_missing")
            return {"event_id": event_id, "memory_id": memory_id, "status": "degraded", "error": "memory_missing"}
        revision = int(memory.get("active_revision") or event.get("aggregate_revision") or 0)
        source_sha = str(memory.get("body_sha256") or "")
        statuses: list[dict[str, Any]] = []

        if str(event.get("event_type") or "") == "LegacyMemoryImported":
            if revision != int(event.get("aggregate_revision") or 0):
                self.authority.set_outbox_status(event_id, status="projected")
                return {
                    "event_id": event_id, "memory_id": memory_id,
                    "revision": revision, "status": "projected",
                    "reason": "superseded_by_new_revision", "projections": [],
                }
            statuses = await self._inspect_legacy_indexes(memory, revision, source_sha)
            # Migration does not request 820 fresh embeddings or rewrite old
            # indexes. Missing derived rows remain visible as pending_rebuild
            # for an explicit, separately budgeted repair.
            self.authority.set_outbox_status(event_id, status="projected")
            return {
                "event_id": event_id, "memory_id": memory_id,
                "revision": revision, "status": "projected",
                "reason": "legacy_indexes_inspected_without_provider_calls",
                "projections": statuses,
            }

        if memory.get("state") == "tombstoned" or memory.get("recall_policy") == "disabled":
            statuses.extend(await self._delete_indexes(
                memory_id, str(memory.get("bucket_id") or memory_id), revision, source_sha,
            ))
        else:
            bucket = await self.bucket_manager.get(str(memory.get("bucket_id") or memory_id))
            if not bucket:
                self.authority.set_outbox_status(event_id, status="degraded", error="bucket_missing")
                return {"event_id": event_id, "memory_id": memory_id, "status": "degraded", "error": "bucket_missing"}
            if hashlib.sha256(
                str(bucket.get("content") or "").encode("utf-8")
            ).hexdigest() != source_sha:
                statuses = [self._record_status(
                    memory_id, revision, projector, "pending_rebuild", source_sha,
                    {"reason": "source_body_revision_mismatch", "index_unchanged": True},
                ) for projector in (
                    "embedding", "moments", "memory_node", "entity_edges",
                    "word_map", "identity_semantics", "memory_edges",
                )]
                self.authority.set_outbox_status(
                    event_id, status="degraded", error="source_body_revision_mismatch",
                )
                return {
                    "event_id": event_id, "memory_id": memory_id,
                    "revision": revision, "status": "degraded",
                    "error": "source_body_revision_mismatch",
                    "projections": statuses,
                }
            statuses.extend(await self._upsert_indexes(bucket, memory_id, revision, source_sha))

        failed = [item for item in statuses if item["status"] == "degraded"]
        if failed:
            error = ",".join(f"{item['projector']}:{item.get('error','failed')}" for item in failed)
            self.authority.set_outbox_status(event_id, status="degraded", error=error)
            final_status = "degraded"
        else:
            self.authority.set_outbox_status(event_id, status="projected")
            final_status = "projected"
        return {
            "event_id": event_id,
            "memory_id": memory_id,
            "revision": revision,
            "status": final_status,
            "projections": statuses,
        }

    async def _inspect_legacy_indexes(
        self, memory: dict[str, Any], revision: int, source_sha: str
    ) -> list[dict[str, Any]]:
        memory_id = str(memory["memory_id"])
        bucket_id = str(memory.get("bucket_id") or memory_id)
        if memory.get("state") != "active" or memory.get("recall_policy") != "enabled":
            return [
                self._record_status(
                    memory_id, revision, projector, "disabled", source_sha,
                    {"reason": "legacy_memory_not_auto_recallable", "index_unchanged": True},
                )
                for projector in (
                    "embedding", "moments", "memory_node", "entity_edges",
                    "word_map", "identity_semantics", "memory_edges",
                )
            ]

        bucket = await self.bucket_manager.get(bucket_id)
        if not bucket:
            return [
                self._record_status(
                    memory_id, revision, projector, "pending_rebuild", source_sha,
                    {"reason": "legacy_bucket_missing", "index_unchanged": True},
                )
                for projector in (
                    "embedding", "moments", "memory_node", "entity_edges",
                    "word_map", "identity_semantics", "memory_edges",
                )
            ]

        results = []

        async def inspect_async(projector: str, enabled: bool, read) -> None:
            if not enabled:
                results.append(self._record_status(
                    memory_id, revision, projector, "disabled", source_sha,
                    {"reason": "provider_disabled", "index_unchanged": True},
                ))
                return
            try:
                present = bool(await read())
                if projector == "embedding" and present:
                    # Legacy rows do not prove which committed body revision
                    # or preparation space produced the vector.
                    status = "pending_rebuild"
                    details = {
                        "reason": "legacy_index_metadata_unavailable",
                        "coverage_status": "incomplete",
                        "outdated": True,
                        "metadata_complete": False,
                        "index_unchanged": True,
                    }
                else:
                    status = "projected" if present else "pending_rebuild"
                    details = {
                        "reason": "legacy_index_present" if present else "legacy_index_missing",
                        "index_unchanged": True,
                    }
            except Exception as exc:
                status = "degraded"
                details = {"error": type(exc).__name__, "index_unchanged": True}
            results.append(self._record_status(
                memory_id, revision, projector, status, source_sha, details,
            ))

        await inspect_async(
            "embedding",
            bool(self.embedding_engine and getattr(self.embedding_engine, "enabled", False)),
            lambda: self.embedding_engine.get_embedding(bucket_id),
        )

        def inspect_sync(projector: str, store, read, *, required: bool = True) -> None:
            if store is None:
                status, details = "disabled", {"reason": "index_not_configured"}
            else:
                try:
                    present = bool(read())
                    status = "projected" if present or not required else "pending_rebuild"
                    details = {
                        "reason": (
                            "legacy_index_present" if present else
                            "no_index_rows_expected" if not required else "legacy_index_missing"
                        )
                    }
                except Exception as exc:
                    status, details = "degraded", {"error": type(exc).__name__}
            results.append(self._record_status(
                memory_id, revision, projector, status, source_sha,
                {**details, "index_unchanged": True},
            ))

        inspect_sync(
            "moments", self.moment_store,
            lambda: self.moment_store.list_for_bucket(bucket_id, limit=1),
        )
        inspect_sync("memory_node", self.node_store, lambda: self.node_store.get(bucket_id))

        try:
            expected_edges = extract_entity_edges_from_bucket(bucket, self.identity)
            if self.entity_edge_store is not None:
                if not hasattr(self, "_legacy_entity_edge_ids"):
                    self._legacy_entity_edge_ids = {
                        str(row.get("bucket_id") or "")
                        for row in self.entity_edge_store.list_edges()
                    }
                edge_ids = self._legacy_entity_edge_ids
            else:
                edge_ids = set()
            inspect_sync(
                "entity_edges", self.entity_edge_store,
                lambda: bucket_id in edge_ids,
                required=bool(expected_edges),
            )
        except Exception as exc:
            results.append(self._record_status(
                memory_id, revision, "entity_edges", "degraded", source_sha,
                {"error": type(exc).__name__, "index_unchanged": True},
            ))

        word_enabled = bool(self.word_map_store and getattr(self.word_map_store, "enabled", False))
        if word_enabled:
            def word_present() -> bool:
                path = Path(str(self.word_map_store.db_path))
                conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
                try:
                    return bool(conn.execute(
                        "SELECT 1 FROM word_card_nodes WHERE bucket_id=? LIMIT 1",
                        (bucket_id,),
                    ).fetchone())
                finally:
                    conn.close()

            try:
                expected_terms = self.word_map_store.extract_bucket_terms(bucket)
                inspect_sync(
                    "word_map", self.word_map_store, word_present,
                    required=bool(expected_terms),
                )
            except Exception as exc:
                results.append(self._record_status(
                    memory_id, revision, "word_map", "degraded", source_sha,
                    {"error": type(exc).__name__, "index_unchanged": True},
                ))
        else:
            inspect_sync("word_map", None, lambda: False)

        results.append(self._record_status(
            memory_id, revision, "identity_semantics", "projected", source_sha,
            {"authority": "memory_authority", "index_unchanged": True},
        ))
        results.append(self._record_status(
            memory_id, revision, "memory_edges", "pending_rebuild", source_sha,
            {"reason": "legacy_edges_not_recomputed", "index_unchanged": True},
        ))
        return results

    async def _upsert_indexes(
        self, bucket: dict, memory_id: str, revision: int, source_sha: str
    ) -> list[dict[str, Any]]:
        results = []
        bucket_id = str(bucket.get("id") or memory_id)
        results.append(await self._project_async(
            "embedding",
            memory_id,
            revision,
            source_sha,
            enabled=bool(self.embedding_engine and getattr(self.embedding_engine, "enabled", False)),
            action=(lambda: self._project_embedding_units(
                bucket, memory_id, revision, source_sha,
            )) if self.embedding_engine else None,
        ))
        results.append(self._project_sync(
            "moments", memory_id, revision, source_sha,
            enabled=self.moment_store is not None,
            action=(lambda: self.moment_store.upsert_bucket(bucket)) if self.moment_store else None,
        ))
        results.append(self._project_sync(
            "memory_node", memory_id, revision, source_sha,
            enabled=self.node_store is not None,
            action=(lambda: self.node_store.upsert_bucket(bucket)) if self.node_store else None,
        ))
        results.append(self._project_sync(
            "entity_edges", memory_id, revision, source_sha,
            enabled=self.entity_edge_store is not None,
            action=(
                lambda: self.entity_edge_store.replace_bucket_edges(
                    bucket_id,
                    extract_entity_edges_from_bucket(bucket, self.identity),
                )
            ) if self.entity_edge_store else None,
        ))
        word_enabled = bool(self.word_map_store and getattr(self.word_map_store, "enabled", False))
        results.append(self._project_sync(
            "word_map", memory_id, revision, source_sha,
            enabled=word_enabled,
            action=(lambda: self.word_map_store.upsert_bucket(bucket)) if word_enabled else None,
        ))
        if self.authority is not None:
            results.append(self._record_status(
                memory_id, revision, "identity_semantics", "projected", source_sha,
                {"authority": "memory_authority", "legacy_rebuild": False},
            ))
        else:
            identity_enabled = bool(
                self.identity_semantic_store
                and getattr(self.identity_semantic_store, "enabled", False)
            )
            results.append(self._record_status(
                memory_id,
                revision,
                "identity_semantics",
                "pending_rebuild" if identity_enabled else "disabled",
                source_sha,
                {"reason": "legacy_store_requires_full_rebuild"} if identity_enabled else {},
            ))
        results.append(self._record_status(
            memory_id, revision, "memory_edges", "pending_rebuild", source_sha,
            {"reason": "relationship_edge_extraction_is_separate"},
        ))
        return results

    def _embedding_source_units(
        self, bucket: dict, memory_id: str,
    ) -> list[dict[str, str]]:
        """Build the provider-bound Bucket seed and committed content units."""
        bucket_id = str(bucket.get("id") or memory_id)
        units: list[dict[str, str]] = []
        seed_text = bucket_text_for_embedding(bucket)
        if seed_text:
            units.append({
                "source_unit_id": bucket_id,
                "source_kind": "bucket_seed",
                "text": seed_text,
            })

        relevance_options = getattr(self.moment_store, "relevance_options", None)
        annotation_options = getattr(self.moment_store, "annotation_options", None)
        moments = parse_bucket_moments(
            bucket, relevance_options, annotation_options,
        )
        for moment in moments:
            if str(moment.get("source") or "") != "content":
                continue
            unit_id = str(moment.get("moment_id") or "").strip()
            text = str(moment.get("text") or "").strip()
            if not unit_id or not text or unit_id == bucket_id:
                continue
            units.append({
                "source_unit_id": unit_id,
                "source_kind": "content_moment",
                "text": text,
            })
        return units

    async def _project_embedding_units(
        self, bucket: dict, memory_id: str, revision: int, source_sha: str,
        *, source_units: list[dict[str, str]] | None = None,
        progress: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Project a legacy seed plus deterministic committed-body moments.

        The instruction and engine config are frozen once for this Memory's
        entire unit batch. Snapshots stay in local memory and are never placed
        in projection status details because they contain credentials.
        """
        document_instruction = self._resolve_document_instruction()
        config_snapshot = self._embedding_config_snapshot(document_instruction)
        bucket_id = str(bucket.get("id") or memory_id)
        units = list(
            source_units if source_units is not None
            else self._embedding_source_units(bucket, memory_id)
        )

        if not units:
            raise RuntimeError("embedding_source_unit_missing")

        engine_method = getattr(self.embedding_engine, "generate_and_store", None)
        if engine_method is None:
            raise RuntimeError("embedding_generate_method_missing")
        supports_metadata = self._accepts_revision_metadata(engine_method)
        if not supports_metadata:
            # Legacy doubles/engines only understand the original bucket key.
            # Keep that call contract; revision-aware Engines receive every
            # content moment as a separately keyed derived unit.
            units = units[:1]

        projected_units: list[dict[str, Any]] = []
        for unit in units:
            if progress is not None:
                progress["attempted"] = int(progress.get("attempted") or 0) + 1
            result = await self._call_embedding_engine(
                engine_method,
                bucket_id,
                unit["text"],
                source_revision=revision,
                source_sha256=source_sha,
                source_unit_id=unit["source_unit_id"],
                parent_memory_id=memory_id,
                document_instruction=document_instruction,
                config_snapshot=config_snapshot,
            )
            if result is False:
                raise RuntimeError("projection_action_returned_false")
            if progress is not None:
                progress["completed"] = int(progress.get("completed") or 0) + 1
            projected_units.append({
                "source_unit_id": unit["source_unit_id"],
                "source_kind": unit["source_kind"],
                "source_text_sha256": hashlib.sha256(
                    unit["text"].encode("utf-8")
                ).hexdigest(),
            })

        space = self._embedding_space_metadata(document_instruction, config_snapshot)
        metadata_complete = bool(
            supports_metadata and space.get("provider") and space.get("model")
            and space.get("dimension") and space.get("preparation_sha256")
        )
        return {
            "source_memory_id": memory_id,
            "source_revision": int(revision),
            "source_sha256": source_sha,
            "parent_bucket_id": bucket_id,
            "revision_metadata_supported": supports_metadata,
            "metadata_complete": metadata_complete,
            "coverage_status": "current" if metadata_complete else "incomplete",
            "outdated": not metadata_complete,
            "source_unit_count": len(projected_units),
            "source_units": projected_units,
            **space,
        }

    @staticmethod
    def _accepts_revision_metadata(method) -> bool:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return False
        if any(item.kind == inspect.Parameter.VAR_KEYWORD
               for item in parameters.values()):
            return True
        return all(name in parameters for name in (
            "source_revision", "source_sha256", "source_unit_id",
            "parent_memory_id",
        ))

    @staticmethod
    async def _call_embedding_engine(
        method, bucket_id: str, text: str, *, source_revision: int,
        source_sha256: str, source_unit_id: str, parent_memory_id: str,
        document_instruction: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ):
        metadata = {
            "source_revision": source_revision,
            "source_sha256": source_sha256,
            "source_unit_id": source_unit_id,
            "parent_memory_id": parent_memory_id,
            "document_instruction": document_instruction,
            "config_snapshot": config_snapshot,
        }
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if any(item.kind == inspect.Parameter.VAR_KEYWORD
               for item in parameters.values()):
            accepted = metadata
        else:
            accepted = {name: value for name, value in metadata.items()
                        if name in parameters}
        return await method(bucket_id, text, **accepted)

    def _embedding_config_snapshot(
        self, document_instruction: str | None,
    ) -> dict[str, Any] | None:
        snapshot_factory = getattr(self.embedding_engine, "_query_config_snapshot", None)
        if not callable(snapshot_factory):
            return None
        try:
            snapshot = snapshot_factory(document_instruction=document_instruction)
        except Exception:
            # Fail closed before any source unit reaches the provider.  In
            # particular, do not log the snapshot: it contains the API key.
            raise RuntimeError("embedding_config_snapshot_unavailable") from None
        if not isinstance(snapshot, dict):
            raise RuntimeError("embedding_config_snapshot_unavailable")
        return dict(snapshot)

    def _resolve_document_instruction(self) -> str | None:
        source_id = "ombre.memory_embedding_document_prep_prompt"
        if self.prompt_plan_mirror is None:
            return None
        try:
            resolved, _metadata = resolve_fixed_prompt(
                self.prompt_plan_mirror,
                source_id,
                config=self.config,
                identity_id=str(
                    self.config.get("persona", {}).get("canonical_session_id")
                    or "jiajia-main"
                ),
                conversation_id="",
            )
            return str(resolved or "")
        except Exception:
            logger.exception(
                "Prompt Composer embedding preparation projection failed | source=%s",
                source_id,
            )
            spec = fixed_prompt_source(source_id)
            if spec is None:
                return ""
            return render_factory_body(spec, spec.factory_body(), self.config)

    def _embedding_space_metadata(
        self, document_instruction: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        engine = self.embedding_engine
        snapshot = config_snapshot if isinstance(config_snapshot, dict) else None
        base_url = str(
            snapshot.get("base_url") or ""
            if snapshot is not None
            else getattr(engine, "base_url", "") or ""
        )
        try:
            raw_provider = str(
                ((snapshot or {}).get("provider") or (
                    getattr(engine, "provider", "") if snapshot is None else ""
                )) or ""
            ).strip()
            if raw_provider and all(
                character.isalnum() or character in "._:-" for character in raw_provider
            ):
                provider = raw_provider
            else:
                provider = str(urlsplit(raw_provider or base_url).hostname or "").strip()
        except ValueError:
            provider = ""
        runtime = getattr(engine, "_runtime", {})
        runtime = runtime if isinstance(runtime, dict) else {}
        dimension = (
            (snapshot or {}).get("dimension")
            or getattr(engine, "dimension", None)
            or runtime.get("last_vector_dimension")
        )
        try:
            dimension = int(dimension) if dimension else None
        except (TypeError, ValueError):
            dimension = None
        preparation_sha = str((snapshot or {}).get("document_preparation") or "").strip()
        preparation_hash = getattr(engine, "preparation_hash", None)
        if not snapshot and callable(preparation_hash):
            try:
                try:
                    parameters = inspect.signature(preparation_hash).parameters
                except (TypeError, ValueError):
                    parameters = {}
                accepts_instruction = (
                    "instruction" in parameters
                    or any(item.kind == inspect.Parameter.VAR_KEYWORD
                           for item in parameters.values())
                )
                kwargs = {"kind": "document"}
                if accepts_instruction:
                    kwargs["instruction"] = document_instruction
                preparation_sha = str(preparation_hash(**kwargs) or "").strip()
            except Exception:
                preparation_sha = ""
        model = (
            str(snapshot.get("model") or "") if snapshot is not None
            else str(getattr(engine, "model", "") or "")
        )
        return {
            "provider": provider[:160],
            "model": model[:160],
            "dimension": dimension,
            "preparation_sha256": preparation_sha[:128],
        }

    async def _delete_indexes(
        self, memory_id: str, bucket_id: str, revision: int, source_sha: str
    ) -> list[dict[str, Any]]:
        results = []
        actions = [
            ("embedding", self.embedding_engine.delete_embedding if self.embedding_engine else None),
            ("moments", self.moment_store.delete_bucket if self.moment_store else None),
            ("memory_node", self.node_store.delete if self.node_store else None),
            ("entity_edges", self.entity_edge_store.delete_for_bucket if self.entity_edge_store else None),
        ]
        for projector, action in actions:
            if action is None:
                results.append(self._record_status(memory_id, revision, projector, "disabled", source_sha, {}))
                continue
            try:
                action(bucket_id)
                results.append(self._record_status(memory_id, revision, projector, "deleted", source_sha, {}))
            except Exception as exc:
                results.append(self._record_status(
                    memory_id, revision, projector, "degraded", source_sha,
                    {"error": type(exc).__name__},
                ))
        results.append(self._record_status(
            memory_id, revision, "word_map", "pending_rebuild", source_sha,
            {"reason": "delete_requires_word_map_rebuild"},
        ))
        results.append(self._record_status(
            memory_id,
            revision,
            "identity_semantics",
            "projected" if self.authority is not None else "pending_rebuild",
            source_sha,
            ({"authority": "memory_authority", "legacy_rebuild": False}
             if self.authority is not None
             else {"reason": "delete_requires_identity_rebuild"}),
        ))
        return results

    async def _project_async(
        self, projector: str, memory_id: str, revision: int, source_sha: str,
        *, enabled: bool, action,
    ) -> dict[str, Any]:
        if not enabled or action is None:
            return self._record_status(memory_id, revision, projector, "disabled", source_sha, {})
        try:
            result = await action()
            if result is False:
                raise RuntimeError("projection_action_returned_false")
            details = {"result": bool(result)}
            if isinstance(result, dict):
                details.update(result)
            return self._record_status(
                memory_id, revision, projector, "projected", source_sha,
                details,
            )
        except Exception as exc:
            logger.warning("Memory projection failed: %s %s", projector, exc)
            return self._record_status(
                memory_id, revision, projector, "degraded", source_sha,
                {"error": type(exc).__name__},
            )

    def _project_sync(
        self, projector: str, memory_id: str, revision: int, source_sha: str,
        *, enabled: bool, action,
    ) -> dict[str, Any]:
        if not enabled or action is None:
            return self._record_status(memory_id, revision, projector, "disabled", source_sha, {})
        try:
            result = action()
            count = len(result) if isinstance(result, (list, tuple, set, dict)) else int(bool(result))
            return self._record_status(
                memory_id, revision, projector, "projected", source_sha,
                {"result_count": count},
            )
        except Exception as exc:
            logger.warning("Memory projection failed: %s %s", projector, exc)
            return self._record_status(
                memory_id, revision, projector, "degraded", source_sha,
                {"error": type(exc).__name__},
            )

    def _record_status(
        self, memory_id: str, revision: int, projector: str, status: str,
        source_sha: str, details: dict[str, Any],
    ) -> dict[str, Any]:
        return self.authority.set_projection_status(
            memory_id=memory_id,
            memory_revision=revision,
            projector=projector,
            status=status,
            source_sha256=source_sha,
            details=details,
        )
