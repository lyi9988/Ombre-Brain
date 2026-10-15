"""Rebuildable, revision-bound L1 members in the existing Authority database.

These tables are projections. They cannot adopt, edit, or authorize a memory.
"""
from __future__ import annotations

import hashlib
import json

from memory_l1_contract import L1ContractError, canonical_json, semantic_sha256

INDEX_VERSION = "memory-l1-members-v2"


def scene_memory_id(memory_id, scene_id):
    return "diary_scene_" + hashlib.sha256(canonical_json([memory_id, scene_id]).encode("utf-8")).hexdigest()[:32]


def initialize(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_l1_index_revisions (
            memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
            content_digest TEXT NOT NULL, semantic_digest TEXT NOT NULL,
            index_version TEXT NOT NULL, entry_json TEXT NOT NULL,
            PRIMARY KEY(memory_id,revision),
            FOREIGN KEY(memory_id,revision) REFERENCES memory_revisions(memory_id,revision)
        );
        CREATE TABLE IF NOT EXISTS memory_l1_members (
            memory_id TEXT NOT NULL, revision INTEGER NOT NULL, module TEXT NOT NULL,
            ordinal INTEGER NOT NULL, object_id TEXT NOT NULL, object_revision TEXT,
            content_json TEXT NOT NULL,
            PRIMARY KEY(memory_id,revision,module,object_id),
            UNIQUE(memory_id,revision,module,ordinal),
            FOREIGN KEY(memory_id,revision) REFERENCES memory_l1_index_revisions(memory_id,revision) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_l1_member_identity ON memory_l1_members(module,object_id,object_revision);
        CREATE TABLE IF NOT EXISTS memory_l1_scene_statements (
            memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
            scene_id TEXT NOT NULL, statement_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            PRIMARY KEY(memory_id,revision,scene_id,statement_id),
            FOREIGN KEY(memory_id,revision) REFERENCES memory_l1_index_revisions(memory_id,revision) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_l1_statement_scenes ON memory_l1_scene_statements(memory_id,revision,statement_id);
        CREATE TABLE IF NOT EXISTS memory_l1_source_members (
            memory_id TEXT NOT NULL, revision INTEGER NOT NULL, binding_id TEXT NOT NULL,
            namespace TEXT NOT NULL, source_id TEXT NOT NULL, source_revision_json TEXT NOT NULL,
            PRIMARY KEY(memory_id,revision,binding_id),
            FOREIGN KEY(memory_id,revision) REFERENCES memory_l1_index_revisions(memory_id,revision) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_l1_source_identity ON memory_l1_source_members(namespace,source_id,source_revision_json);
    """)
    from memory_l1_features import initialize as initialize_features
    initialize_features(conn)


def project(conn, memory_id, revision, metadata):
    """Called in the original finalize transaction, or a bounded rebuild."""
    if "memory_l1" not in metadata:
        return
    semantic = metadata["memory_l1"]
    if semantic_sha256(semantic) != metadata.get("memory_l1_semantic_sha256"):
        raise L1ContractError("indexed_semantic_identity_mismatch", "memory_l1")
    digest = metadata.get("memory_l1_content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise L1ContractError("indexed_content_identity_missing", "memory_l1_content_sha256")
    conn.execute("DELETE FROM memory_l1_index_revisions WHERE memory_id=? AND revision=?", (memory_id, revision))
    conn.execute("INSERT INTO memory_l1_index_revisions VALUES(?,?,?,?,?,?)",
                 (memory_id, revision, digest, metadata["memory_l1_semantic_sha256"], INDEX_VERSION,
                  canonical_json(semantic["entry"])))
    for module, key, version_key in (("statements", "statement_id", "statement_revision_id"),
                                    ("scenes", "scene_id", "scene_revision_id"),
                                    ("evidence_bindings", "binding_id", None),
                                    ("entities", "entity_id", None)):
        for ordinal, item in enumerate(semantic[module]):
            conn.execute("INSERT INTO memory_l1_members VALUES(?,?,?,?,?,?,?)",
                         (memory_id, revision, module, ordinal, item[key],
                          str(item[version_key]) if version_key and item[version_key] is not None else None,
                          canonical_json(item)))
    for scene in semantic["scenes"]:
        for ordinal, statement_id in enumerate(scene["statement_refs"]):
            conn.execute("INSERT INTO memory_l1_scene_statements VALUES(?,?,?,?,?)",
                         (memory_id, revision, scene["scene_id"], statement_id, ordinal))
    for item in semantic["evidence_bindings"]:
        conn.execute("INSERT INTO memory_l1_source_members VALUES(?,?,?,?,?,?)", (memory_id, revision,
                     item["binding_id"], item["source_namespace"], item["source_id"], canonical_json(item["source_revision"])))
    from memory_l1_features import project as project_features
    project_features(conn, memory_id, revision, semantic)


class MemoryL1Index:
    def __init__(self, authority):
        self.authority = authority

    def ensure(self, memory_id, revision):
        """No old-library scan. Rebuild only this explicitly requested revision."""
        conn = self.authority._connect()
        try:
            row = conn.execute("""SELECT json_extract(r.metadata_json,'$.memory_l1_content_sha256') AS digest,
                json_extract(r.metadata_json,'$.memory_l1_semantic_sha256') AS semantic_digest,
                i.content_digest,i.semantic_digest AS indexed_semantic,i.index_version
                FROM memory_revisions r LEFT JOIN memory_l1_index_revisions i
                ON i.memory_id=r.memory_id AND i.revision=r.revision
                WHERE r.memory_id=? AND r.revision=?""", (memory_id, revision)).fetchone()
            if row is None:
                raise L1ContractError("revision_not_found", "revision")
            if row["digest"] is None:
                return False
            if (row["digest"] == row["content_digest"] and row["semantic_digest"] == row["indexed_semantic"]
                    and row["index_version"] == INDEX_VERSION):
                return True
            conn.execute("BEGIN IMMEDIATE")
            record = conn.execute("SELECT metadata_json FROM memory_revisions WHERE memory_id=? AND revision=?",
                                  (memory_id, revision)).fetchone()
            project(conn, memory_id, revision, json.loads(record[0]))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def identity(self, memory_id, revision):
        if not self.ensure(memory_id, revision):
            return None
        conn = self.authority._connect()
        try:
            row = conn.execute("SELECT * FROM memory_l1_index_revisions WHERE memory_id=? AND revision=?",
                               (memory_id, revision)).fetchone()
            result = dict(row)
            result["entry"] = json.loads(result.pop("entry_json"))
            return result
        finally:
            conn.close()

    def member(self, memory_id, revision, module, object_id):
        if not self.ensure(memory_id, revision):
            return None
        conn = self.authority._connect()
        try:
            row = conn.execute("SELECT content_json FROM memory_l1_members WHERE memory_id=? AND revision=? AND module=? AND object_id=?",
                               (memory_id, revision, module, object_id)).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    def page(self, memory_id, revision, module, *, offset=0, limit=20, scene_id=None):
        if not self.ensure(memory_id, revision):
            return [], 0
        conn = self.authority._connect()
        try:
            base = "FROM memory_l1_members m "
            where = "WHERE m.memory_id=? AND m.revision=? AND m.module=? "
            params = [memory_id, revision, module]
            order = "m.ordinal"
            if scene_id is not None:
                base += "JOIN memory_l1_scene_statements s ON s.memory_id=m.memory_id AND s.revision=m.revision AND s.statement_id=m.object_id "
                where += "AND s.scene_id=? "
                params.append(scene_id)
                order = "s.ordinal"
            count = conn.execute("SELECT COUNT(*) " + base + where, params).fetchone()[0]
            rows = conn.execute("SELECT m.content_json " + base + where + "ORDER BY " + order + " LIMIT ? OFFSET ?",
                                [*params, limit, offset]).fetchall()
            return [json.loads(row[0]) for row in rows], count
        finally:
            conn.close()
