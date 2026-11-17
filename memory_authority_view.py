"""Read-only Gateway view of committed Memory authority facts."""
from __future__ import annotations

import os
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from memory_authority import normalize_alias


def portrait_l1_v2_block_reason(sync_scope, guards) -> str:
    if sync_scope is not None:
        return "l1_sync_required"
    scopes = [guard.get("scope") or {} for guard in guards]
    if any(scope.get("unknown_scope") is not False for scope in scopes):
        return "relation_guard_scope_unknown"
    if any(scope.get("historical_only") is not True for scope in scopes):
        return "relation_guard_current"
    return ""


class MemoryAuthorityRecallView:
    def __init__(self, config: dict[str, Any]):
        authority_cfg = config.get("memory_authority", {}) if isinstance(
            config.get("memory_authority", {}), dict) else {}
        state_dir = str(config.get("state_dir") or config.get("buckets_dir") or "state")
        self.enabled = bool(authority_cfg.get("enabled", False))
        self.path = Path(
            str(authority_cfg.get("db_path") or os.path.join(state_dir, "memory_authority.sqlite3"))
        ).resolve()
        self._stamp: tuple[int, int, int, int] | None = None
        self._trusted_aliases: list[dict[str, Any]] = []
        self._auto_bucket_stamp: tuple[int, int, int, int] | None = None
        self._auto_bucket_ids: frozenset[str] = frozenset()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=2.0)

    def _file_stamp(self) -> tuple[int, int, int, int] | None:
        try:
            stat = self.path.stat()
            wal_path = Path(str(self.path) + "-wal")
            try:
                wal = wal_path.stat()
                wal_stamp = (int(wal.st_mtime_ns), int(wal.st_size))
            except FileNotFoundError:
                wal_stamp = (0, 0)
            return int(stat.st_mtime_ns), int(stat.st_size), *wal_stamp
        except FileNotFoundError:
            return None

    def available(self) -> bool:
        return bool(self.enabled and self.path.exists())

    def auto_recallable_bucket_ids(self) -> frozenset[str]:
        """Only committed, active, auto-enabled Memory may enter chat recall.

        The Bucket files remain a readable projection, not a second policy
        authority. A missing/unreadable Authority fails closed when enabled.
        """
        stamp = self._file_stamp()
        if not self.available():
            self._auto_bucket_stamp = stamp
            self._auto_bucket_ids = frozenset()
            return self._auto_bucket_ids
        if stamp == self._auto_bucket_stamp:
            return self._auto_bucket_ids
        conn = None
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT bucket_id FROM memories "
                "WHERE state='active' AND recall_policy='enabled' "
                "AND bucket_id IS NOT NULL AND bucket_id!=''"
            ).fetchall()
        except sqlite3.Error:
            self._auto_bucket_ids = frozenset()
            self._auto_bucket_stamp = None
        else:
            self._auto_bucket_ids = frozenset(str(row[0]) for row in rows)
            self._auto_bucket_stamp = stamp
        finally:
            if conn is not None:
                conn.close()
        return self._auto_bucket_ids

    def trusted_aliases(self) -> list[dict[str, Any]]:
        stamp = self._file_stamp()
        if not self.available():
            self._stamp = stamp
            self._trusted_aliases = []
            return []
        if stamp == self._stamp:
            return [dict(item) for item in self._trusted_aliases]
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT alias_id,entity_id,alias,normalized_alias,trust,revision,"
                "source_refs_json,updated_at "
                "FROM entity_aliases WHERE state='active' AND trust IN ('owner','trusted_source') "
                "ORDER BY LENGTH(normalized_alias) DESC,alias_id"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            conn.close()
        self._stamp = stamp
        aliases = []
        for row in rows:
            item = dict(row)
            try:
                item["source_refs"] = list(json.loads(item.pop("source_refs_json") or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                item.pop("source_refs_json", None)
                item["source_refs"] = []
            aliases.append(item)
        self._trusted_aliases = aliases
        return [dict(item) for item in self._trusted_aliases]

    def match_aliases(self, query: str) -> list[dict[str, Any]]:
        normalized_query = normalize_alias(query).replace(" ", "")
        if not normalized_query:
            return []
        matches = []
        for row in self.trusted_aliases():
            key = normalize_alias(str(row.get("normalized_alias") or row.get("alias") or "")).replace(" ", "")
            if key and key in normalized_query:
                item = dict(row)
                memory_ids = [
                    str(ref)[len("memory:"):]
                    for ref in item.get("source_refs", []) or []
                    if str(ref or "").startswith("memory:")
                    and len(str(ref or "")) > len("memory:")
                ]
                resolved = self.resolve_memory_refs(memory_ids)
                item["memory_ids"] = list(resolved)
                item["bucket_ids"] = list(dict.fromkeys(
                    str(value.get("bucket_id") or "")
                    for value in resolved.values()
                    if str(value.get("bucket_id") or "")
                ))
                # A trusted name alone is not recall evidence.  If its
                # linked Memory was archived, deleted, or never committed,
                # it must not force a Fast route that cannot return a card.
                if not item["bucket_ids"]:
                    continue
                matches.append(item)
        return matches[:8]

    def watermark(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "memory_count": 0, "alias_revision": 0}
        conn = self._connect()
        try:
            memory_count = int(conn.execute(
                "SELECT COUNT(*) FROM memories WHERE state='active'"
            ).fetchone()[0])
            alias_revision = int(conn.execute(
                "SELECT COALESCE(MAX(revision),0) FROM entity_aliases WHERE state='active'"
            ).fetchone()[0])
            memory_revision = int(conn.execute(
                "SELECT COALESCE(SUM(active_revision),0) FROM memories WHERE state='active'"
            ).fetchone()[0])
        except sqlite3.Error:
            return {"available": False, "memory_count": 0, "alias_revision": 0}
        finally:
            conn.close()
        return {
            "available": True,
            "memory_count": memory_count,
            "alias_revision": alias_revision,
            "memory_revision_watermark": memory_revision,
            "db_stamp": self._file_stamp(),
        }

    @staticmethod
    def _decoded_row(row: sqlite3.Row, *json_fields: str) -> dict[str, Any]:
        value = dict(row)
        for field in json_fields:
            raw = value.pop(f"{field}_json", None)
            try:
                value[field] = json.loads(raw) if raw else ([] if field.endswith("refs") else {})
            except (TypeError, ValueError, json.JSONDecodeError):
                value[field] = [] if field.endswith("refs") else {}
        return value

    def overview(self) -> dict[str, Any]:
        if not self.available():
            return {"available": False, "memories": {}, "candidates": {}, "aliases": {}, "outbox": {}}
        conn = self._connect()
        try:
            def grouped(table: str, field: str, where: str = "") -> dict[str, int]:
                rows = conn.execute(
                    f"SELECT {field},COUNT(*) FROM {table} {where} GROUP BY {field}"
                ).fetchall()
                return {str(row[0]): int(row[1]) for row in rows}
            return {
                "available": True,
                "memories": grouped("memories", "state"),
                "candidates": grouped("candidates", "status"),
                "aliases": grouped("entity_aliases", "trust", "WHERE state='active'"),
                "outbox": grouped("outbox", "status"),
                "watermark": self.watermark(),
            }
        except sqlite3.Error:
            return {"available": False, "memories": {}, "candidates": {}, "aliases": {}, "outbox": {}}
        finally:
            conn.close()

    def list_memories(self, *, state: str = "active", limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if not self.available():
            return []
        clauses = []
        params: list[Any] = []
        if state and state != "all":
            clauses.append("m.state=?")
            params.append(str(state))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT m.*,r.body_sha256,r.snapshot_path,r.metadata_json,r.source_refs_json,"
                "r.decision_source,r.created_at AS revision_created_at "
                "FROM memories m LEFT JOIN memory_revisions r "
                "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
                f"{where} ORDER BY m.updated_at DESC,m.memory_id LIMIT ? OFFSET ?",
                [*params, max(1, min(200, int(limit))), max(0, int(offset))],
            ).fetchall()
            return [self._decoded_row(row, "metadata", "source_refs") for row in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def memory_detail(self, memory_id: str) -> dict[str, Any] | None:
        if not self.available():
            return None
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            memory = conn.execute(
                "SELECT * FROM memories WHERE memory_id=?", (str(memory_id),)
            ).fetchone()
            if not memory:
                return None
            revisions = conn.execute(
                "SELECT * FROM memory_revisions WHERE memory_id=? ORDER BY revision DESC LIMIT 200",
                (str(memory_id),),
            ).fetchall()
            rings = conn.execute(
                "SELECT * FROM memory_rings WHERE memory_id=? ORDER BY created_at DESC,ring_id LIMIT 200",
                (str(memory_id),),
            ).fetchall()
            projections = conn.execute(
                "SELECT * FROM memory_projection_status WHERE memory_id=? "
                "ORDER BY memory_revision DESC,projector",
                (str(memory_id),),
            ).fetchall()
            return {
                "memory": dict(memory),
                "revisions": [
                    self._decoded_row(row, "metadata", "source_refs") for row in revisions
                ],
                "rings": [
                    self._decoded_row(row, "metadata", "source_refs") for row in rings
                ],
                "projection_status": [
                    self._decoded_row(row, "details") for row in projections
                ],
            }
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    def list_aliases(self, *, state: str = "active", trust: str = "all", limit: int = 200) -> list[dict[str, Any]]:
        if not self.available():
            return []
        clauses = []
        params: list[Any] = []
        if state and state != "all":
            clauses.append("state=?")
            params.append(str(state))
        if trust and trust != "all":
            clauses.append("trust=?")
            params.append(str(trust))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT * FROM entity_aliases {where} ORDER BY updated_at DESC,alias_id LIMIT ?",
                [*params, max(1, min(1000, int(limit)))],
            ).fetchall()
            output = []
            for row in rows:
                item = self._decoded_row(row, "source_refs")
                memory_ids = [
                    str(ref)[len("memory:"):]
                    for ref in item.get("source_refs", []) or []
                    if str(ref or "").startswith("memory:")
                ]
                item["memory_ids"] = memory_ids
                output.append(item)
            return output
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def memory_revision_map(self, memory_ids: list[str]) -> dict[str, int]:
        ids = list(dict.fromkeys(
            str(memory_id or "").strip()
            for memory_id in memory_ids
            if str(memory_id or "").strip()
        ))[:64]
        if not ids or not self.available():
            return {}
        resolved = self.resolve_memory_refs(ids)
        return {
            memory_id: int(item.get("active_revision") or 0)
            for memory_id, item in resolved.items()
        }

    def memory_index_metadata_map(
        self, bucket_ids: list[str], *, for_portrait: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Resolve eligible bucket projections to their committed revision.

        This is a read-only lookup for derived Recall indexes.  It deliberately
        includes only active, auto-enabled Memory so manual-only/private rows
        cannot become searchable through an index metadata join.
        """
        ids = list(dict.fromkeys(
            str(bucket_id or "").strip()
            for bucket_id in bucket_ids
            if str(bucket_id or "").strip()
        ))
        if not ids or not self.available():
            return {}
        conn = self._connect()
        dependency_versions = {}
        try:
            # This is the index eligibility pool, not a 64-result owner tool.
            # Batch bind variables without silently losing the tail of Memory.
            conn.execute("BEGIN")
            rows = []
            schema_column = ",json_extract(r.metadata_json,'$.memory_l1.schema_version') " if for_portrait else " "
            for offset in range(0, len(ids), 400):
                chunk = ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(conn.execute(
                    "SELECT m.memory_id,COALESCE(NULLIF(m.bucket_id,''),m.memory_id),"
                    "m.active_revision,COALESCE(r.body_sha256,'')" + schema_column +
                    "FROM memories AS m LEFT JOIN memory_revisions AS r "
                    "ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
                    "WHERE m.state='active' AND m.recall_policy='enabled' "
                    f"AND (m.bucket_id IN ({placeholders}) "
                    f"OR m.memory_id IN ({placeholders}))",
                    [*chunk, *chunk],
                ).fetchall())
            if for_portrait:
                from memory_l1_v2 import SCHEMA_VERSION
                from memory_relation_dependencies import cache_versions
                dependency_versions = cache_versions(conn, [str(row[0]) for row in rows if row[4] == SCHEMA_VERSION])
        except sqlite3.Error:
            return {}
        finally:
            conn.close()

        result: dict[str, dict[str, Any]] = {}
        ambiguous: set[str] = set()
        seen_rows: set[tuple] = set()
        for row in rows:
            signature = tuple(row)
            if signature in seen_rows:
                continue
            seen_rows.add(signature)
            bucket_id = str(row[1] or row[0])
            if not bucket_id or bucket_id in ambiguous:
                continue
            if bucket_id in result:
                # A bucket projected from multiple active authorities is not
                # safe to associate with either revision.
                result.pop(bucket_id, None)
                ambiguous.add(bucket_id)
                continue
            result[bucket_id] = {
                "memory_id": str(row[0]),
                "revision": int(row[2] or 0),
                "body_sha256": str(row[3] or ""),
            }
            if str(row[0]) in dependency_versions:
                result[bucket_id]["dependency_token"] = dependency_versions[str(row[0])]
        return result

    def portrait_candidate_source_states(self, memory_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Return positive lifecycle facts for committed portrait sources, including tombstones."""
        ids = list(dict.fromkeys(memory_ids))
        if not ids or not self.available():
            return {}
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            result = {}
            for offset in range(0, len(ids), 400):
                chunk = ids[offset:offset + 400]
                marks = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT m.memory_id,COALESCE(NULLIF(m.bucket_id,''),m.memory_id) AS bucket_id,"
                    "m.active_revision AS revision,m.state,"
                    "json_extract(r.metadata_json,'$.valid_until') AS valid_until,"
                    "json_extract(r.metadata_json,'$.expires_at') AS expires_at,"
                    "(json_type(r.metadata_json,'$.withdrawn')='true' OR "
                    "json_type(r.metadata_json,'$.retracted')='true') AS withdrawn "
                    "FROM memories m JOIN memory_revisions r ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
                    "JOIN commit_log c ON c.operation_id=r.operation_id "
                    f"WHERE c.status='committed' AND m.memory_id IN ({marks})", chunk,
                )
                result.update((row["memory_id"], dict(row)) for row in rows)
            return result
        finally:
            conn.close()

    def portrait_l1_v2_pending(self, reviewed: dict[str, int], *, limit: int,
                               retry_order: dict[str, str], blocked: dict[str, dict]
                               ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Uninspected current L1 v2 commits, independent of bucket activity dates.

        Checkpoint exact memory/revision pairs rather than a wall-clock cutoff.
        Concurrent commits, restored eligibility and retries can then leave a
        gap without losing any other current revision. Unchanged dependency
        blocks do not consume slots. Never-tried cards precede failed attempts.
        """
        if not self.enabled or limit <= 0:
            return [], {}
        from memory_l1_v2 import SCHEMA_VERSION
        from memory_relation_dependencies import dependency_snapshot

        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            rows = conn.execute(
                "WITH inspected AS (SELECT key,value FROM json_each(?)), "
                "blocked AS (SELECT key FROM json_each(?)) "
                "SELECT m.memory_id,COALESCE(NULLIF(m.bucket_id,''),m.memory_id) AS bucket_id,r.revision FROM memories m "
                "JOIN memory_revisions r ON m.memory_id=r.memory_id AND m.active_revision=r.revision "
                "JOIN commit_log c ON c.operation_id=r.operation_id "
                "LEFT JOIN json_each(?) retry ON retry.key=m.memory_id || ':' || r.revision "
                "WHERE c.status='committed' AND c.operation_kind='memory_revision' "
                "AND m.state='active' AND m.recall_policy='enabled' "
                "AND json_extract(r.metadata_json,'$.memory_l1.schema_version')=? "
                "AND (json_extract(r.metadata_json,'$.memory_l1.stability')='stable' "
                "OR m.memory_id IN (SELECT key FROM inspected) OR m.memory_id IN (SELECT key FROM blocked)) "
                "AND m.memory_id || ':' || r.revision NOT IN "
                "(SELECT key || ':' || value FROM inspected) "
                "ORDER BY COALESCE(retry.value,''), r.rowid",
                (json.dumps(reviewed), json.dumps(blocked), json.dumps(retry_order), SCHEMA_VERSION),
            )
            pending, unchanged = [], Counter()
            for row in rows:
                memory_id, revision = row["memory_id"], row["revision"]
                try:
                    dependency = dependency_snapshot(conn, memory_id)
                except (sqlite3.Error, ValueError, TypeError, KeyError):
                    pending.append({**dict(row), "dependency_token": "", "block_reason": ""})
                else:
                    reason = portrait_l1_v2_block_reason(dependency["sync_scope"], dependency["guards"])
                    checkpoint = {"revision": revision, "dependency_token": dependency["token"]}
                    if reason and blocked.get(memory_id) == checkpoint:
                        unchanged[reason] += 1
                        continue
                    pending.append({**dict(row), "dependency_token": dependency["token"], "block_reason": reason})
                if len(pending) >= limit:
                    break
            return pending, dict(unchanged)
        finally:
            conn.close()

    def memory_l1_recall_map(
        self, bucket_ids: Iterable[str], *, for_portrait: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Batch current semantic data; stale/missing indexes do not hide L1.

Read the committed revision metadata as the fallback, never rebuild the
database during chat. Source readability is verified separately, per binding.
"""
        ids = list(dict.fromkeys(str(value) for value in bucket_ids if value))
        if not ids or not self.available():
            return {}
        conn = self._connect()
        if for_portrait:
            conn.row_factory = sqlite3.Row
            from memory_l1_v2 import SCHEMA_VERSION
            from memory_relation_dependencies import dependency_snapshot
        result = {}
        try:
            conn.execute("BEGIN")
            for offset in range(0, len(ids), 400):
                chunk = ids[offset:offset + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT m.memory_id,COALESCE(NULLIF(m.bucket_id,''),m.memory_id),m.active_revision,"
                    "r.body_sha256,json_extract(r.metadata_json,'$.memory_l1'),"
                    "json_extract(r.metadata_json,'$.memory_l1_semantic_sha256') "
                    "FROM memories m JOIN memory_revisions r ON r.memory_id=m.memory_id AND r.revision=m.active_revision "
                    "WHERE m.state='active' AND m.recall_policy='enabled' "
                    f"AND (m.bucket_id IN ({placeholders}) OR m.memory_id IN ({placeholders}))",
                    [*chunk, *chunk],
                ).fetchall()
                for memory_id, bid, revision, body_sha, raw, semantic_sha in rows:
                    row = {"memory_id": memory_id, "revision": revision, "body_sha256": body_sha,
                           "document": None, "state": "legacy"}
                    if raw:
                        from memory_l1_contract import semantic_sha256
                        semantic = json.loads(raw)
                        if semantic_sha256(semantic) != semantic_sha:
                            row["state"] = "semantic_mismatch"
                        else:
                            row.update(document=semantic, state="l1")
                            if for_portrait and semantic.get("schema_version") == SCHEMA_VERSION:
                                try:
                                    dependency = dependency_snapshot(conn, memory_id)
                                except (sqlite3.Error, ValueError, TypeError, KeyError):
                                    row["state"] = "dependency_unavailable"
                                else:
                                    row.update(l1_sync_required=dependency["l1_sync_required"],
                                               sync_scope=dependency["sync_scope"], relation_guards=dependency["guards"],
                                               relation_dependency_token=dependency["token"])
                            entry = semantic.get("entry") or {}
                            parent = entry.get("parent_ref")
                            if parent:
                                root = conn.execute(
                                    "SELECT m.active_revision,r.metadata_json FROM memories m JOIN memory_revisions r "
                                    "ON r.memory_id=m.memory_id AND r.revision=? WHERE m.memory_id=? "
                                    "AND m.state='active' AND m.recall_policy='enabled'",
                                    (parent["revision"], parent["memory_id"])).fetchone()
                                if not root or root[0] != parent["revision"]:
                                    row["state"] = "dependency_changed"
                                else:
                                    root_metadata = json.loads(root[1])
                                    root_doc = root_metadata.get("memory_l1") or {}
                                    if semantic_sha256(root_doc) != root_metadata.get("memory_l1_semantic_sha256"):
                                        row["state"] = "dependency_changed"
                                    refs = set(entry.get("shared_statement_refs") or [])
                                    semantic = {**semantic,
                                        "statements": [s for s in root_doc.get("statements", []) if s.get("statement_id") in refs],
                                        "evidence_bindings": root_doc.get("evidence_bindings", []),
                                        "entities": root_doc.get("entities", []),
                                        "relations": root_doc.get("relations", [])}
                                    row["document"] = semantic
                    if bid in result:
                        # Multiple authorities cannot silently select a winner.
                        result[bid] = {"state": "ambiguous", "document": None}
                    else:
                        result[bid] = row
        except (sqlite3.Error, ValueError, TypeError):
            return {}
        finally:
            conn.close()
        return result

    def cognition_chain(self, seeds, *, depth, limit):
        """Bounded explicit revision/relationship traversal, no model or writes.

Historical versions remain history. Every returned binding still needs the
same current-source check on Brain before any claim is shown to chat.
"""
        from collections import deque
        from memory_l1_contract import semantic_sha256
        if not self.available() or depth <= 0 or limit <= 0:
            return {"nodes": [], "truncated": False, "issues": []}
        conn = self._connect()
        cache, nodes, seen, issues, depth_limited = {}, [], set(), [], False
        reached = {}
        def load(mid, rev):
            key = (mid, rev)
            if key in cache:
                return cache[key]
            cache[key] = None
            row = conn.execute(
                "SELECT m.active_revision,COALESCE(NULLIF(m.bucket_id,''),m.memory_id),r.metadata_json "
                "FROM memories m JOIN memory_revisions r ON r.memory_id=m.memory_id AND r.revision=? "
                "WHERE m.memory_id=? AND m.state='active' AND m.recall_policy='enabled'", (rev, mid)).fetchone()
            if not row:
                return None
            metadata = json.loads(row[2])
            document = metadata.get("memory_l1")
            if not document or semantic_sha256(document) != metadata.get("memory_l1_semantic_sha256"):
                return None
            parent = (document.get("entry") or {}).get("parent_ref")
            if parent:
                root = load(parent["memory_id"], parent["revision"])
                if root is None:
                    return None
                root_doc = root["document"]
                refs = set(document["entry"].get("shared_statement_refs") or [])
                document = {**document,
                    "statements": [s for s in root_doc.get("statements", []) if s["statement_id"] in refs],
                    "entities": root_doc.get("entities", []), "evidence_bindings": root_doc.get("evidence_bindings", []),
                    "relations": root_doc.get("relations", [])}
            result = {"memory_id": mid, "revision": rev, "bucket_id": row[1],
                      "historical": row[0] != rev or bool(parent and root["historical"]), "document": document}
            cache[key] = result
            return result
        queue = deque((mid, rev, 0, None) for mid, rev in seeds)
        try:
            conn.execute("BEGIN")
            while queue and len(nodes) < limit:
                mid, rev, level, via = queue.popleft()
                key = (mid, rev)
                if key in seen:
                    if via and key in reached and via not in reached[key]["via_paths"]:
                        reached[key]["via_paths"].append(via)
                    continue
                seen.add(key)
                record = load(mid, rev)
                if record is None:
                    issues.append({"memory_id": mid, "revision": rev, "reason": "related_revision_unavailable"})
                    continue
                document = record["document"]
                cognitive = any(s.get("cognition") for s in document.get("statements", []))
                if level == 0 and not cognitive:
                    continue
                node = {**record, "depth": level, "via": via, "via_paths": [via] if via else []}
                nodes.append(node)
                reached[key] = node
                if level >= depth:
                    depth_limited = depth_limited or (cognitive and rev > 1) or bool(document.get("relations"))
                    continue
                if cognitive and rev > 1:
                    queue.append((mid, rev - 1, level + 1, {"kind": "revision_history", "from": key}))
                host = (document.get("entry") or {}).get("parent_ref") or {"memory_id": mid, "revision": rev}
                for relation in document.get("relations", []):
                    if relation.get("confirmation_state") == "rejected":
                        continue
                    for endpoint in (relation["from"], relation["to"]):
                        target_id = endpoint.get("memory_id") or host["memory_id"]
                        target_rev = host["revision"] if endpoint.get("revision") == "this_revision" else endpoint.get("revision")
                        if type(target_rev) is int and (target_id, target_rev) != key:
                            queue.append((target_id, target_rev, level + 1,
                                {"kind": relation["kind"], "relation_id": relation["relation_id"],
                                 "confirmation_state": relation["confirmation_state"], "from": key,
                                 "object_type": endpoint["object_type"], "object_id": endpoint["object_id"]}))
            return {"nodes": nodes, "truncated": depth_limited or any((mid, rev) not in seen for mid, rev, _level, _via in queue),
                    "depth_limited": depth_limited,
                    "issues": issues}
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            return {"nodes": [], "truncated": False, "issues": [{"reason": "cognition_chain_unavailable"}]}
        finally:
            conn.close()

    def resolve_memory_refs(self, memory_or_bucket_ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(
            str(memory_id or "").strip()
            for memory_id in memory_or_bucket_ids
            if str(memory_id or "").strip()
        ))[:64]
        if not ids or not self.available():
            return {}
        placeholders = ",".join("?" for _ in ids)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT memory_id,bucket_id,active_revision FROM memories "
                f"WHERE state='active' AND recall_policy='enabled' "
                f"AND (memory_id IN ({placeholders}) "
                f"OR bucket_id IN ({placeholders}))",
                [*ids, *ids],
            ).fetchall()
        except sqlite3.Error:
            return {}
        finally:
            conn.close()
        return {
            str(row[0]): {
                "memory_id": str(row[0]),
                "bucket_id": str(row[1]),
                "active_revision": int(row[2]),
            }
            for row in rows
        }
