"""Outbox-driven derived-index projection for RECALL-R1."""
from __future__ import annotations

import logging
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
