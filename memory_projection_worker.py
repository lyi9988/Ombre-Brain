"""Outbox-driven derived-index projection for RECALL-R1."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from entity_edges import extract_entity_edges_from_bucket
from identity import identity_names
from memory_authority import MemoryAuthorityStore
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
            statuses.extend(await self._delete_indexes(memory_id, revision, source_sha))
        else:
            bucket = await self.bucket_manager.get(str(memory.get("bucket_id") or memory_id))
            if not bucket:
                self.authority.set_outbox_status(event_id, status="degraded", error="bucket_missing")
                return {"event_id": event_id, "memory_id": memory_id, "status": "degraded", "error": "bucket_missing"}
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
        results.append(await self._project_async(
            "embedding",
            memory_id,
            revision,
            source_sha,
            enabled=bool(self.embedding_engine and getattr(self.embedding_engine, "enabled", False)),
            action=(
                lambda: self.embedding_engine.generate_and_store(
                    memory_id, bucket_text_for_embedding(bucket)
                )
                if self.embedding_engine else None
            ),
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
                    memory_id,
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

    async def _delete_indexes(self, memory_id: str, revision: int, source_sha: str) -> list[dict[str, Any]]:
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
                action(memory_id)
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
            return self._record_status(
                memory_id, revision, projector, "projected", source_sha,
                {"result": bool(result)},
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
