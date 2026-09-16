"""Authoritative control plane for RECALL-R1 memory ingestion and revisions.

This module deliberately does not import BucketManager, model clients, recall
indexes, tools, or HTTP code.  It owns deterministic ingestion policy and the
SQLite transaction boundaries that surround file-backed Bucket commits.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


class MemoryAuthorityError(Exception):
    code = "memory_authority_error"


class CandidateNotFound(MemoryAuthorityError):
    code = "candidate_not_found"


class MemoryNotFound(MemoryAuthorityError):
    code = "memory_not_found"


class RevisionConflict(MemoryAuthorityError):
    code = "revision_conflict"


class IdempotencyConflict(MemoryAuthorityError):
    code = "idempotency_conflict"


class InvalidTransition(MemoryAuthorityError):
    code = "invalid_transition"


class CommitStateError(MemoryAuthorityError):
    code = "commit_state_error"


class BodyHashMismatch(MemoryAuthorityError):
    code = "body_hash_mismatch"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def normalize_alias(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().split())


@dataclass(frozen=True)
class MemoryProposal:
    proposal_id: str
    source_type: str
    proposed_body: str
    original_excerpt: str = ""
    source_refs: tuple[str, ...] = ()
    source_status: str = "verified"
    memory_type: str = "durable_fact"
    confidence: float = 0.0
    requested_mode: str = "review"
    owner_explicit: bool = False
    sensitive: bool = False
    transient: bool = False
    generic: bool = False
    duplicate_of: str = ""
    proposed_entities: tuple[Mapping[str, Any], ...] = ()
    proposed_aliases: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryProposal":
        source_refs = payload.get("source_refs") or payload.get("source_event_ids") or []
        if isinstance(source_refs, str):
            source_refs = [source_refs]
        entities = payload.get("proposed_entities") or []
        aliases = payload.get("proposed_aliases") or []
        return cls(
            proposal_id=str(payload.get("proposal_id") or payload.get("candidate_id") or payload.get("id") or "").strip(),
            source_type=str(payload.get("source_type") or "unknown").strip(),
            proposed_body=str(payload.get("proposed_body") or payload.get("proposed_memory") or payload.get("content") or "").strip(),
            original_excerpt=str(payload.get("original_excerpt") or "").strip(),
            source_refs=tuple(str(item) for item in source_refs if str(item).strip()),
            source_status=str(payload.get("source_status") or payload.get("source_verification") or "unverified").strip(),
            memory_type=str(payload.get("memory_type") or payload.get("candidate_type") or payload.get("kind") or "durable_fact").strip(),
            confidence=max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
            requested_mode=str(payload.get("requested_mode") or payload.get("mode") or "review").strip().lower(),
            owner_explicit=bool(payload.get("owner_explicit", False)),
            sensitive=bool(payload.get("sensitive", False)),
            transient=bool(payload.get("transient", False)),
            generic=bool(payload.get("generic", False)),
            duplicate_of=str(payload.get("duplicate_of") or "").strip(),
            proposed_entities=tuple(item for item in entities if isinstance(item, Mapping)),
            proposed_aliases=tuple(item for item in aliases if isinstance(item, Mapping)),
            metadata=dict(payload.get("metadata") or {}),
        )

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = list(self.source_refs)
        data["proposed_entities"] = [dict(item) for item in self.proposed_entities]
        data["proposed_aliases"] = [dict(item) for item in self.proposed_aliases]
        data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason_codes: tuple[str, ...]
    decision_policy: str
    alias_trust: str = "weak"
    requires_owner_confirmation: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "decision_policy": self.decision_policy,
            "alias_trust": self.alias_trust,
            "requires_owner_confirmation": self.requires_owner_confirmation,
        }


class MemoryIngestionPolicy:
    """Pure policy: proposal/config in, deterministic decision out."""

    DEFAULT_AUTO_TYPES = frozenset({
        "durable_fact", "preference", "stable_preference", "relationship_event",
        "shared_experience", "key_event", "boundary", "signal", "commitment",
        "project_state", "relationship_anchor",
    })
    OWNER_REVIEW_TYPES = frozenset({
        "identity", "alias",
    })

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        self.default_mode = str(cfg.get("default_mode") or "review").strip().lower()
        self.source_modes = {
            str(key): str(value).strip().lower()
            for key, value in dict(cfg.get("source_modes") or {}).items()
        }
        self.auto_types = frozenset(str(item) for item in (
            cfg.get("auto_types") or self.DEFAULT_AUTO_TYPES
        ))
        self.owner_review_types = frozenset(str(item) for item in (
            cfg.get("owner_review_types") or self.OWNER_REVIEW_TYPES
        ))
        self.auto_confidence = max(0.0, min(1.0, float(cfg.get("auto_confidence", 0.68))))

    def evaluate(self, proposal: MemoryProposal) -> PolicyDecision:
        mode = self.source_modes.get(proposal.source_type, proposal.requested_mode or self.default_mode)
        if mode not in {"auto", "review", "off"}:
            mode = self.default_mode if self.default_mode in {"auto", "review", "off"} else "review"

        if not proposal.proposal_id or not proposal.proposed_body:
            return PolicyDecision("reject", ("REJECT_EMPTY_PROPOSAL",), mode)
        if proposal.duplicate_of:
            return PolicyDecision("reject", ("REJECT_EXACT_DUPLICATE",), mode)
        if proposal.source_status != "verified" and not proposal.owner_explicit:
            return PolicyDecision("reject", ("REJECT_SOURCE_UNVERIFIED",), mode)
        if not proposal.source_refs and not proposal.owner_explicit:
            return PolicyDecision("reject", ("REJECT_SOURCE_MISSING",), mode)
        if mode == "off":
            return PolicyDecision("reject", ("REJECT_SOURCE_DISABLED",), mode)
        if proposal.owner_explicit:
            return PolicyDecision(
                "auto_accept", ("ACCEPT_EXPLICIT_OWNER",), "explicit_owner",
                alias_trust="owner", requires_owner_confirmation=False,
            )
        if proposal.sensitive or proposal.memory_type in self.owner_review_types:
            reason = "REVIEW_SENSITIVE" if proposal.sensitive else "REVIEW_OWNER_AUTHORITY_REQUIRED"
            return PolicyDecision(
                "queue_review", (reason,), mode,
                alias_trust="weak", requires_owner_confirmation=True,
            )
        if proposal.transient or proposal.generic:
            return PolicyDecision(
                "queue_review", ("REVIEW_TRANSIENT_OR_GENERIC",), mode,
                alias_trust="weak", requires_owner_confirmation=True,
            )
        if (
            mode == "auto"
            and proposal.memory_type in self.auto_types
            and proposal.confidence >= self.auto_confidence
        ):
            return PolicyDecision(
                "auto_accept", ("ACCEPT_AUTO_POLICY",), mode,
                alias_trust="weak", requires_owner_confirmation=False,
            )
        return PolicyDecision(
            "queue_review", ("REVIEW_POLICY_DEFAULT",), mode,
            alias_trust="weak", requires_owner_confirmation=True,
        )


class MemoryAuthorityStore:
    """SQLite control authority around file-backed Bucket revisions."""

    CANDIDATE_TRANSITIONS = {
        "pending": {"accepted", "rejected", "deferred"},
        "deferred": {"pending", "accepted", "rejected"},
        "accepted": {"committed", "commit_failed"},
        "commit_failed": {"accepted", "pending"},
        "committed": set(),
        "rejected": {"pending"},
    }

    def __init__(self, config: Mapping[str, Any] | str):
        if isinstance(config, str):
            path = config
        else:
            cfg = dict(config or {})
            state_dir = str(cfg.get("state_dir") or "state")
            path = str(cfg.get("memory_authority_db_path") or os.path.join(state_dir, "memory_authority.sqlite3"))
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                proposal_sha256 TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_decisions (
                decision_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT NOT NULL UNIQUE,
                candidate_id TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                candidate_revision INTEGER NOT NULL,
                request_id TEXT NOT NULL DEFAULT '',
                fingerprint TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason_codes_json TEXT NOT NULL DEFAULT '[]',
                result_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_decision_request
                ON candidate_decisions(request_id) WHERE request_id != '';
            CREATE TABLE IF NOT EXISTS candidate_revisions (
                candidate_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                status TEXT NOT NULL,
                proposal_sha256 TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(candidate_id, revision),
                FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
            );
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                bucket_id TEXT NOT NULL UNIQUE,
                active_revision INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'active',
                recall_policy TEXT NOT NULL DEFAULT 'enabled',
                body_sha256 TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_revisions (
                memory_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                body_sha256 TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                source_refs_json TEXT NOT NULL DEFAULT '[]',
                decision_source TEXT NOT NULL,
                operation_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                PRIMARY KEY(memory_id, revision),
                FOREIGN KEY(memory_id) REFERENCES memories(memory_id)
            );
            CREATE TABLE IF NOT EXISTS memory_rings (
                ring_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'comment',
                state TEXT NOT NULL DEFAULT 'active',
                source_refs_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                operation_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(memory_id)
            );
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                trust TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                revision INTEGER NOT NULL DEFAULT 1,
                source_refs_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                UNIQUE(entity_id, normalized_alias)
            );
            CREATE TABLE IF NOT EXISTS entity_alias_history (
                alias_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                trust TEXT NOT NULL,
                state TEXT NOT NULL,
                source_refs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                PRIMARY KEY(alias_id, revision),
                FOREIGN KEY(alias_id) REFERENCES entity_aliases(alias_id)
            );
            CREATE TABLE IF NOT EXISTS commit_log (
                operation_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL,
                operation_kind TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                error_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_open_memory_revision
                ON commit_log(aggregate_id)
                WHERE operation_kind='memory_revision' AND status IN ('prepared','body_written');
            CREATE TABLE IF NOT EXISTS outbox (
                event_id TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                aggregate_revision INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(operation_id) REFERENCES commit_log(operation_id)
            );
            CREATE TABLE IF NOT EXISTS authority_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        decision_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(candidate_decisions)").fetchall()
        }
        if "result_json" not in decision_columns:
            conn.execute(
                "ALTER TABLE candidate_decisions ADD COLUMN result_json TEXT NOT NULL DEFAULT '{}'"
            )
        ring_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(memory_rings)").fetchall()
        }
        if "metadata_json" not in ring_columns:
            conn.execute(
                "ALTER TABLE memory_rings ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        conn.execute(
            "INSERT INTO authority_meta(key,value) VALUES('schema_version','memory-authority-v1') "
            "ON CONFLICT(key) DO NOTHING"
        )
        conn.commit()
        conn.close()

    @staticmethod
    def fingerprint(payload: Mapping[str, Any]) -> str:
        return _sha(dict(payload or {}))

    @staticmethod
    def _candidate_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        value = dict(row)
        value["proposal"] = _loads(value.pop("proposal_json"), {})
        value["policy"] = _loads(value.pop("policy_json"), {})
        return value

    def put_candidate(self, proposal: MemoryProposal, policy: PolicyDecision) -> dict[str, Any]:
        if not proposal.proposal_id:
            raise ValueError("proposal_id is required")
        proposal_payload = proposal.payload()
        proposal_sha = _sha(proposal_payload)
        status = {"reject": "rejected", "queue_review": "pending", "auto_accept": "accepted"}[policy.action]
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (proposal.proposal_id,)
            ).fetchone()
            if row:
                current = self._candidate_row(row)
                if current and current["proposal_sha256"] != proposal_sha:
                    raise IdempotencyConflict("candidate_id already exists with different proposal")
                conn.commit()
                return current or {}
            conn.execute(
                "INSERT INTO candidates(candidate_id,revision,status,proposal_sha256,proposal_json,policy_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (proposal.proposal_id, 1, status, proposal_sha, _json(proposal_payload), _json(policy.payload()), now, now),
            )
            created_result = {
                "candidate_id": proposal.proposal_id,
                "revision": 1,
                "status": status,
                "proposal_sha256": proposal_sha,
                "proposal": proposal_payload,
                "policy": policy.payload(),
                "created_at": now,
                "updated_at": now,
            }
            conn.execute(
                "INSERT INTO candidate_revisions(candidate_id,revision,status,proposal_sha256,proposal_json,"
                "policy_json,actor,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (proposal.proposal_id, 1, status, proposal_sha, _json(proposal_payload),
                 _json(policy.payload()), "policy", now),
            )
            conn.execute(
                "INSERT INTO candidate_decisions(decision_id,candidate_id,from_status,to_status,candidate_revision,"
                "request_id,fingerprint,actor,reason_codes_json,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), proposal.proposal_id, "generated", status, 1, "", proposal_sha,
                 "policy", _json(list(policy.reason_codes)), _json(created_result), now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (proposal.proposal_id,)).fetchone()
            return self._candidate_row(row) or {}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (str(candidate_id),)).fetchone()
        conn.close()
        return self._candidate_row(row)

    def list_candidates(self, *, status: str = "pending", limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            clauses.append("status=?")
            params.append(str(status))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        rows = conn.execute(
            f"SELECT * FROM candidates{where} ORDER BY created_at DESC LIMIT ?",
            [*params, max(1, min(1000, int(limit)))],
        ).fetchall()
        conn.close()
        return [self._candidate_row(row) or {} for row in rows]

    def revise_candidate(
        self,
        candidate_id: str,
        *,
        expected_revision: int,
        proposal: MemoryProposal,
        request_id: str,
        actor: str,
    ) -> dict[str, Any]:
        payload = {
            "candidate_id": candidate_id,
            "expected_revision": int(expected_revision),
            "proposal": proposal.payload(),
        }
        fingerprint = self.fingerprint(payload)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if request_id:
                prior = conn.execute(
                    "SELECT * FROM candidate_decisions WHERE request_id=?", (request_id,)
                ).fetchone()
                if prior:
                    if str(prior["fingerprint"]) != fingerprint:
                        raise IdempotencyConflict("request_id payload differs")
                    conn.commit()
                    return _loads(prior["result_json"], {})
            row = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if not row:
                raise CandidateNotFound(candidate_id)
            current = self._candidate_row(row) or {}
            if int(current["revision"]) != int(expected_revision):
                raise RevisionConflict(f"expected {expected_revision}, current {current['revision']}")
            if current["status"] not in {"pending", "deferred", "commit_failed"}:
                raise InvalidTransition(f"cannot revise candidate in {current['status']}")
            proposal_payload = proposal.payload()
            proposal_sha = _sha(proposal_payload)
            next_revision = int(current["revision"]) + 1
            now = _now()
            result = {
                **current,
                "revision": next_revision,
                "proposal_sha256": proposal_sha,
                "proposal": proposal_payload,
                "updated_at": now,
            }
            conn.execute(
                "UPDATE candidates SET revision=?,proposal_sha256=?,proposal_json=?,updated_at=? WHERE candidate_id=?",
                (next_revision, proposal_sha, _json(proposal_payload), now, candidate_id),
            )
            conn.execute(
                "INSERT INTO candidate_revisions(candidate_id,revision,status,proposal_sha256,proposal_json,"
                "policy_json,actor,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (candidate_id, next_revision, current["status"], proposal_sha,
                 _json(proposal_payload), _json(current["policy"]), actor, now),
            )
            conn.execute(
                "INSERT INTO candidate_decisions(decision_id,candidate_id,from_status,to_status,candidate_revision,"
                "request_id,fingerprint,actor,reason_codes_json,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), candidate_id, current["status"], current["status"], next_revision,
                 request_id, fingerprint, actor, _json(["OWNER_EDITED_PROPOSAL"]), _json(result), now),
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def decide_candidate(
        self,
        candidate_id: str,
        *,
        action: str,
        expected_revision: int,
        request_id: str,
        actor: str,
        reason_codes: Sequence[str] = (),
    ) -> dict[str, Any]:
        target = {
            "accept": "accepted", "reject": "rejected", "defer": "deferred", "reopen": "pending",
            "commit": "committed", "commit_failed": "commit_failed",
        }.get(
            str(action or "").strip().lower()
        )
        if not target:
            raise InvalidTransition(action)
        payload = {
            "candidate_id": str(candidate_id), "action": action,
            "expected_revision": int(expected_revision), "reason_codes": list(reason_codes),
        }
        fingerprint = self.fingerprint(payload)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if request_id:
                prior = conn.execute(
                    "SELECT * FROM candidate_decisions WHERE request_id=?", (request_id,)
                ).fetchone()
                if prior:
                    if str(prior["fingerprint"]) != fingerprint:
                        raise IdempotencyConflict("request_id payload differs")
                    conn.commit()
                    return _loads(prior["result_json"], {})
            row = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if not row:
                raise CandidateNotFound(candidate_id)
            current = self._candidate_row(row) or {}
            if int(current["revision"]) != int(expected_revision):
                raise RevisionConflict(f"expected {expected_revision}, current {current['revision']}")
            allowed = self.CANDIDATE_TRANSITIONS.get(str(current["status"]), set())
            if target not in allowed:
                if target == current["status"]:
                    conn.commit()
                    return current
                raise InvalidTransition(f"{current['status']}->{target}")
            next_revision = int(current["revision"]) + 1
            now = _now()
            conn.execute(
                "UPDATE candidates SET status=?,revision=?,updated_at=? WHERE candidate_id=?",
                (target, next_revision, now, candidate_id),
            )
            result = {
                **current,
                "status": target,
                "revision": next_revision,
                "updated_at": now,
            }
            conn.execute(
                "INSERT INTO candidate_revisions(candidate_id,revision,status,proposal_sha256,proposal_json,"
                "policy_json,actor,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (candidate_id, next_revision, target, current["proposal_sha256"],
                 _json(current["proposal"]), _json(current["policy"]), actor, now),
            )
            conn.execute(
                "INSERT INTO candidate_decisions(decision_id,candidate_id,from_status,to_status,candidate_revision,"
                "request_id,fingerprint,actor,reason_codes_json,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), candidate_id, current["status"], target, next_revision,
                 request_id, fingerprint, actor, _json(list(reason_codes)), _json(result), now),
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def prepare_memory_commit(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        expected_revision: int,
        body_sha256: str,
        snapshot_path: str,
        metadata: Mapping[str, Any],
        source_refs: Sequence[str],
        decision_source: str,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        payload = {
            "memory_id": str(memory_id), "bucket_id": str(bucket_id),
            "expected_revision": int(expected_revision), "body_sha256": str(body_sha256),
            "snapshot_path": str(snapshot_path), "metadata": dict(metadata or {}),
            "source_refs": [str(item) for item in source_refs if str(item).strip()],
            "decision_source": str(decision_source), "actor": str(actor),
        }
        fingerprint = self.fingerprint(payload)
        operation_id = f"memory-revision:{uuid.uuid4().hex}"
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT * FROM commit_log WHERE idempotency_key=?", (str(idempotency_key),)
            ).fetchone()
            if prior:
                if str(prior["fingerprint"]) != fingerprint:
                    raise IdempotencyConflict("idempotency_key payload differs")
                conn.commit()
                value = dict(prior)
                value["payload"] = _loads(value.pop("payload_json"), {})
                value["result"] = _loads(value.pop("result_json"), {})
                return value
            memory = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            current_revision = int(memory["active_revision"]) if memory else 0
            if current_revision != int(expected_revision):
                raise RevisionConflict(f"expected {expected_revision}, current {current_revision}")
            payload["revision"] = current_revision + 1
            try:
                conn.execute(
                    "INSERT INTO commit_log(operation_id,idempotency_key,fingerprint,operation_kind,aggregate_id,status,"
                    "payload_json,result_json,error_code,created_at,updated_at) VALUES(?,?,?,?,?,'prepared',?,'{}','',?,?)",
                    (operation_id, str(idempotency_key), fingerprint, "memory_revision", memory_id,
                     _json(payload), now, now),
                )
            except sqlite3.IntegrityError as exc:
                open_commit = conn.execute(
                    "SELECT operation_id FROM commit_log WHERE aggregate_id=? AND operation_kind='memory_revision' "
                    "AND status IN ('prepared','body_written')",
                    (memory_id,),
                ).fetchone()
                if open_commit:
                    raise CommitStateError(
                        f"memory {memory_id} already has open commit {open_commit['operation_id']}"
                    ) from exc
                raise
            conn.commit()
            return {"operation_id": operation_id, "status": "prepared", "payload": payload, "result": {}}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_body_written(self, operation_id: str, *, observed_body_sha256: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM commit_log WHERE operation_id=?", (operation_id,)).fetchone()
            if not row:
                raise CommitStateError("operation not found")
            payload = _loads(row["payload_json"], {})
            if str(payload.get("body_sha256") or "") != str(observed_body_sha256 or ""):
                raise BodyHashMismatch("written Bucket hash differs from prepared body hash")
            if row["status"] == "committed":
                conn.commit()
                return dict(row)
            if row["status"] not in {"prepared", "body_written"}:
                raise CommitStateError(f"cannot record body from {row['status']}")
            conn.execute(
                "UPDATE commit_log SET status='body_written',updated_at=? WHERE operation_id=?",
                (_now(), operation_id),
            )
            conn.commit()
            return {"operation_id": operation_id, "status": "body_written", "payload": payload}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def finalize_memory_commit(self, operation_id: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM commit_log WHERE operation_id=?", (operation_id,)).fetchone()
            if not row:
                raise CommitStateError("operation not found")
            if row["status"] == "committed":
                conn.commit()
                return _loads(row["result_json"], {})
            if row["status"] != "body_written":
                raise CommitStateError(f"cannot finalize from {row['status']}")
            payload = _loads(row["payload_json"], {})
            memory_id = str(payload["memory_id"])
            revision = int(payload["revision"])
            now = _now()
            current = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            current_revision = int(current["active_revision"]) if current else 0
            if current_revision != int(payload["expected_revision"]):
                raise RevisionConflict(f"expected {payload['expected_revision']}, current {current_revision}")
            if current:
                conn.execute(
                    "UPDATE memories SET bucket_id=?,active_revision=?,body_sha256=?,updated_at=? WHERE memory_id=?",
                    (payload["bucket_id"], revision, payload["body_sha256"], now, memory_id),
                )
            else:
                conn.execute(
                    "INSERT INTO memories(memory_id,bucket_id,active_revision,state,recall_policy,body_sha256,updated_at) "
                    "VALUES(?,?,?,'active','enabled',?,?)",
                    (memory_id, payload["bucket_id"], revision, payload["body_sha256"], now),
                )
            conn.execute(
                "INSERT INTO memory_revisions(memory_id,revision,body_sha256,snapshot_path,metadata_json,"
                "source_refs_json,decision_source,operation_id,created_at,created_by) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (memory_id, revision, payload["body_sha256"], payload["snapshot_path"],
                 _json(payload["metadata"]), _json(payload["source_refs"]), payload["decision_source"],
                 operation_id, now, payload["actor"]),
            )
            event_id = f"memory:{memory_id}:revision:{revision}"
            outbox_payload = {
                "memory_id": memory_id, "bucket_id": payload["bucket_id"],
                "revision": revision, "body_sha256": payload["body_sha256"],
                "source_refs": payload["source_refs"],
            }
            conn.execute(
                "INSERT INTO outbox(event_id,operation_id,event_type,aggregate_id,aggregate_revision,payload_json,"
                "status,attempts,last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,'pending',0,'',?,?)",
                (event_id, operation_id, "MemoryRevisionCommitted", memory_id, revision,
                 _json(outbox_payload), now, now),
            )
            result = {"memory_id": memory_id, "bucket_id": payload["bucket_id"], "revision": revision,
                      "body_sha256": payload["body_sha256"], "outbox_event_id": event_id}
            conn.execute(
                "UPDATE commit_log SET status='committed',result_json=?,updated_at=? WHERE operation_id=?",
                (_json(result), now, operation_id),
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_commit(self, operation_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM commit_log WHERE operation_id=?", (operation_id,)).fetchone()
        conn.close()
        if not row:
            return None
        value = dict(row)
        value["payload"] = _loads(value.pop("payload_json"), {})
        value["result"] = _loads(value.pop("result_json"), {})
        return value

    def list_open_commits(self, *, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM commit_log WHERE operation_kind='memory_revision' "
            "AND status IN ('prepared','body_written') ORDER BY created_at LIMIT ?",
            (max(1, min(1000, int(limit))),),
        ).fetchall()
        conn.close()
        result = []
        for row in rows:
            value = dict(row)
            value["payload"] = _loads(value.pop("payload_json"), {})
            value["result"] = _loads(value.pop("result_json"), {})
            result.append(value)
        return result

    def abort_memory_commit(self, operation_id: str, *, error_code: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM commit_log WHERE operation_id=?", (operation_id,)).fetchone()
            if not row:
                raise CommitStateError("operation not found")
            if row["status"] == "aborted":
                conn.commit()
                return self.get_commit(operation_id) or {}
            if row["status"] != "prepared":
                raise CommitStateError(f"cannot abort from {row['status']}")
            conn.execute(
                "UPDATE commit_log SET status='aborted',error_code=?,updated_at=? WHERE operation_id=?",
                (str(error_code or "projection_not_applied")[:120], _now(), operation_id),
            )
            conn.commit()
            return self.get_commit(operation_id) or {}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def set_outbox_status(self, event_id: str, *, status: str, error: str = "") -> dict[str, Any]:
        safe_status = str(status or "").strip().lower()
        if safe_status not in {"pending", "projected", "degraded"}:
            raise ValueError("invalid outbox status")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM outbox WHERE event_id=?", (event_id,)).fetchone()
            if not row:
                raise CommitStateError("outbox event not found")
            attempts = int(row["attempts"] or 0) + (1 if safe_status in {"projected", "degraded"} else 0)
            conn.execute(
                "UPDATE outbox SET status=?,attempts=?,last_error=?,updated_at=? WHERE event_id=?",
                (safe_status, attempts, str(error or "")[:500], _now(), event_id),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM outbox WHERE event_id=?", (event_id,)).fetchone()
            value = dict(updated)
            value["payload"] = _loads(value.pop("payload_json"), {})
            return value
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_ring(
        self,
        *,
        memory_id: str,
        content: str,
        kind: str,
        source_refs: Sequence[str],
        idempotency_key: str,
        actor: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = str(content or "").strip()
        if not body:
            raise ValueError("ring content is required")
        payload = {
            "memory_id": str(memory_id), "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "kind": str(kind or "comment"), "source_refs": [str(item) for item in source_refs if str(item).strip()],
            "actor": str(actor), "metadata": dict(metadata or {}),
        }
        fingerprint = self.fingerprint(payload)
        operation_id = f"memory-ring:{uuid.uuid4().hex}"
        ring_id = f"ring-{uuid.uuid4().hex}"
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT * FROM commit_log WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if prior:
                if str(prior["fingerprint"]) != fingerprint:
                    raise IdempotencyConflict("idempotency_key payload differs")
                conn.commit()
                return _loads(prior["result_json"], {})
            memory = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            if not memory:
                raise MemoryNotFound(memory_id)
            conn.execute(
                "INSERT INTO commit_log(operation_id,idempotency_key,fingerprint,operation_kind,aggregate_id,status,"
                "payload_json,result_json,error_code,created_at,updated_at) VALUES(?,?,?,?,?,'committed',?,'{}','',?,?)",
                (operation_id, idempotency_key, fingerprint, "memory_ring", memory_id, _json(payload), now, now),
            )
            conn.execute(
                "INSERT INTO memory_rings(ring_id,memory_id,content_sha256,content,kind,state,source_refs_json,"
                "metadata_json,operation_id,created_at,created_by) VALUES(?,?,?,?,?,'active',?,?,?,?,?)",
                (ring_id, memory_id, payload["content_sha256"], body, payload["kind"],
                 _json(payload["source_refs"]), _json(payload["metadata"]), operation_id, now, actor),
            )
            event_id = f"memory:{memory_id}:ring:{ring_id}"
            event_payload = {"memory_id": memory_id, "ring_id": ring_id, "kind": payload["kind"],
                             "content_sha256": payload["content_sha256"]}
            conn.execute(
                "INSERT INTO outbox(event_id,operation_id,event_type,aggregate_id,aggregate_revision,payload_json,"
                "status,attempts,last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,'pending',0,'',?,?)",
                (event_id, operation_id, "MemoryRingAppended", memory_id, int(memory["active_revision"]),
                 _json(event_payload), now, now),
            )
            result = {"memory_id": memory_id, "ring_id": ring_id, "outbox_event_id": event_id}
            conn.execute(
                "UPDATE commit_log SET result_json=? WHERE operation_id=?", (_json(result), operation_id)
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def retract_ring(
        self,
        *,
        memory_id: str,
        ring_id: str,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        payload = {"memory_id": memory_id, "ring_id": ring_id, "actor": actor}
        fingerprint = self.fingerprint(payload)
        operation_id = f"memory-ring-retract:{uuid.uuid4().hex}"
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT * FROM commit_log WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if prior:
                if str(prior["fingerprint"]) != fingerprint:
                    raise IdempotencyConflict("idempotency_key payload differs")
                conn.commit()
                return _loads(prior["result_json"], {})
            ring = conn.execute(
                "SELECT * FROM memory_rings WHERE memory_id=? AND ring_id=?", (memory_id, ring_id)
            ).fetchone()
            if not ring:
                raise MemoryNotFound(f"ring {memory_id}/{ring_id}")
            if ring["state"] == "retracted":
                return {"memory_id": memory_id, "ring_id": ring_id, "status": "retracted"}
            conn.execute(
                "INSERT INTO commit_log(operation_id,idempotency_key,fingerprint,operation_kind,aggregate_id,status,"
                "payload_json,result_json,error_code,created_at,updated_at) VALUES(?,?,?,?,?,'committed',?,'{}','',?,?)",
                (operation_id, idempotency_key, fingerprint, "memory_ring_retract", memory_id,
                 _json(payload), now, now),
            )
            conn.execute(
                "UPDATE memory_rings SET state='retracted' WHERE memory_id=? AND ring_id=?",
                (memory_id, ring_id),
            )
            event_id = f"memory:{memory_id}:ring:{ring_id}:retracted"
            event_payload = {"memory_id": memory_id, "ring_id": ring_id}
            memory = conn.execute("SELECT active_revision FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            conn.execute(
                "INSERT INTO outbox(event_id,operation_id,event_type,aggregate_id,aggregate_revision,payload_json,"
                "status,attempts,last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,'pending',0,'',?,?)",
                (event_id, operation_id, "MemoryRingRetracted", memory_id,
                 int(memory["active_revision"] if memory else 0), _json(event_payload), now, now),
            )
            result = {"memory_id": memory_id, "ring_id": ring_id, "status": "retracted",
                      "outbox_event_id": event_id}
            conn.execute(
                "UPDATE commit_log SET result_json=? WHERE operation_id=?", (_json(result), operation_id)
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_alias(
        self,
        *,
        entity_id: str,
        alias: str,
        trust: str,
        source_refs: Sequence[str],
    ) -> dict[str, Any]:
        trust = str(trust or "auto").strip().lower()
        if trust not in {"auto", "trusted_source", "owner"}:
            raise ValueError("invalid alias trust")
        normalized = normalize_alias(alias)
        if not entity_id or not normalized:
            raise ValueError("entity_id and alias are required")
        rank = {"auto": 0, "trusted_source": 1, "owner": 2}
        conn = self._connect()
        now = _now()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM entity_aliases WHERE entity_id=? AND normalized_alias=?",
                (entity_id, normalized),
            ).fetchone()
            if row:
                current = dict(row)
                effective_trust = current["trust"] if rank[current["trust"]] >= rank[trust] else trust
                merged_refs = sorted(set(_loads(current["source_refs_json"], [])) | {
                    str(item) for item in source_refs if str(item).strip()
                })
                changed = (
                    effective_trust != current["trust"]
                    or str(alias).strip() != current["alias"]
                    or merged_refs != _loads(current["source_refs_json"], [])
                )
                revision = int(current["revision"]) + (1 if changed else 0)
                conn.execute(
                    "UPDATE entity_aliases SET alias=?,trust=?,revision=?,source_refs_json=?,updated_at=? "
                    "WHERE alias_id=?",
                    (str(alias).strip(), effective_trust, revision, _json(merged_refs), now, current["alias_id"]),
                )
                alias_id = current["alias_id"]
            else:
                alias_id = f"alias-{uuid.uuid4().hex}"
                conn.execute(
                    "INSERT INTO entity_aliases(alias_id,entity_id,alias,normalized_alias,trust,state,revision,"
                    "source_refs_json,updated_at) VALUES(?,?,?,?,?,'active',1,?,?)",
                    (alias_id, entity_id, str(alias).strip(), normalized, trust,
                     _json([str(item) for item in source_refs if str(item).strip()]), now),
                )
                revision = 1
                merged_refs = [str(item) for item in source_refs if str(item).strip()]
                effective_trust = trust
            history = conn.execute(
                "SELECT 1 FROM entity_alias_history WHERE alias_id=? AND revision=?",
                (alias_id, revision),
            ).fetchone()
            if not history:
                conn.execute(
                    "INSERT INTO entity_alias_history(alias_id,revision,entity_id,alias,normalized_alias,trust,"
                    "state,source_refs_json,created_at) VALUES(?,?,?,?,?,?,'active',?,?)",
                    (alias_id, revision, entity_id, str(alias).strip(), normalized, effective_trust,
                     _json(merged_refs), now),
                )
            conn.commit()
            row = conn.execute("SELECT * FROM entity_aliases WHERE alias_id=?", (alias_id,)).fetchone()
            value = dict(row)
            value["source_refs"] = _loads(value.pop("source_refs_json"), [])
            return value
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def import_legacy_memory(
        self,
        *,
        memory_id: str,
        bucket_id: str,
        body_sha256: str,
        snapshot_path: str,
        metadata: Mapping[str, Any],
        source_refs: Sequence[str],
        state: str = "active",
        recall_policy: str = "enabled",
    ) -> dict[str, Any]:
        """Register an existing Bucket without changing or regenerating it."""
        memory_id = str(memory_id or "").strip()
        bucket_id = str(bucket_id or "").strip()
        if not memory_id or not bucket_id or not body_sha256 or not snapshot_path:
            raise ValueError("legacy memory identity, hash, and snapshot are required")
        state = str(state or "active").strip().lower()
        recall_policy = str(recall_policy or "enabled").strip().lower()
        if state not in {"active", "archived", "tombstoned"}:
            raise ValueError("invalid memory state")
        if recall_policy not in {"enabled", "manual_only", "disabled"}:
            raise ValueError("invalid recall policy")
        operation_id = f"legacy-memory-import:{memory_id}:1"
        idempotency_key = operation_id
        payload = {
            "memory_id": memory_id,
            "bucket_id": bucket_id,
            "revision": 1,
            "body_sha256": str(body_sha256),
            "snapshot_path": str(snapshot_path),
            "metadata": dict(metadata or {}),
            "source_refs": [str(item) for item in source_refs if str(item).strip()],
            "state": state,
            "recall_policy": recall_policy,
        }
        fingerprint = self.fingerprint(payload)
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT * FROM commit_log WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if prior:
                if str(prior["fingerprint"]) != fingerprint:
                    raise IdempotencyConflict("legacy memory changed after import")
                conn.commit()
                return _loads(prior["result_json"], {})
            existing = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            if existing:
                if (
                    str(existing["bucket_id"]) != bucket_id
                    or str(existing["body_sha256"]) != str(body_sha256)
                    or int(existing["active_revision"]) != 1
                ):
                    raise IdempotencyConflict("memory authority already contains a different legacy record")
                result = dict(existing)
                conn.commit()
                return result
            conn.execute(
                "INSERT INTO commit_log(operation_id,idempotency_key,fingerprint,operation_kind,aggregate_id,status,"
                "payload_json,result_json,error_code,created_at,updated_at) VALUES(?,?,?,?,?,'committed',?,'{}','',?,?)",
                (operation_id, idempotency_key, fingerprint, "legacy_memory_import", memory_id,
                 _json(payload), now, now),
            )
            conn.execute(
                "INSERT INTO memories(memory_id,bucket_id,active_revision,state,recall_policy,body_sha256,updated_at) "
                "VALUES(?,?,1,?,?,?,?)",
                (memory_id, bucket_id, state, recall_policy, str(body_sha256), now),
            )
            conn.execute(
                "INSERT INTO memory_revisions(memory_id,revision,body_sha256,snapshot_path,metadata_json,"
                "source_refs_json,decision_source,operation_id,created_at,created_by) VALUES(?,1,?,?,?,?,?,?,?,?)",
                (memory_id, str(body_sha256), str(snapshot_path), _json(dict(metadata or {})),
                 _json(payload["source_refs"]), "migration", operation_id, now, "migration"),
            )
            event_id = f"memory:{memory_id}:legacy-import:1"
            event_payload = {
                "memory_id": memory_id,
                "bucket_id": bucket_id,
                "revision": 1,
                "body_sha256": str(body_sha256),
                "legacy_index_state": "unverified",
            }
            conn.execute(
                "INSERT INTO outbox(event_id,operation_id,event_type,aggregate_id,aggregate_revision,payload_json,"
                "status,attempts,last_error,created_at,updated_at) VALUES(?,?,?,?,1,?,'degraded',0,?,?,?)",
                (event_id, operation_id, "LegacyMemoryImported", memory_id, _json(event_payload),
                 "legacy_index_state_unverified", now, now),
            )
            result = {
                "memory_id": memory_id,
                "bucket_id": bucket_id,
                "revision": 1,
                "body_sha256": str(body_sha256),
                "outbox_event_id": event_id,
            }
            conn.execute(
                "UPDATE commit_log SET result_json=? WHERE operation_id=?",
                (_json(result), operation_id),
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def import_legacy_ring(
        self,
        *,
        memory_id: str,
        ring_id: str,
        content: str,
        kind: str,
        source_refs: Sequence[str],
        actor: str,
        created_at: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = str(content or "").strip()
        ring_id = str(ring_id or "").strip()
        if not ring_id or not body:
            raise ValueError("legacy ring id and content are required")
        operation_id = f"legacy-ring-import:{memory_id}:{ring_id}"
        payload = {
            "memory_id": memory_id,
            "ring_id": ring_id,
            "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "kind": str(kind or "comment"),
            "source_refs": [str(item) for item in source_refs if str(item).strip()],
            "actor": str(actor or "legacy"),
            "created_at": str(created_at or _now()),
            "metadata": dict(metadata or {}),
        }
        fingerprint = self.fingerprint(payload)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            memory = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
            if not memory:
                raise MemoryNotFound(memory_id)
            prior = conn.execute("SELECT * FROM commit_log WHERE operation_id=?", (operation_id,)).fetchone()
            if prior:
                if str(prior["fingerprint"]) != fingerprint:
                    raise IdempotencyConflict("legacy ring changed after import")
                conn.commit()
                return _loads(prior["result_json"], {})
            conn.execute(
                "INSERT INTO commit_log(operation_id,idempotency_key,fingerprint,operation_kind,aggregate_id,status,"
                "payload_json,result_json,error_code,created_at,updated_at) VALUES(?,?,?,?,?,'committed',?,'{}','',?,?)",
                (operation_id, operation_id, fingerprint, "legacy_ring_import", memory_id,
                 _json(payload), payload["created_at"], payload["created_at"]),
            )
            conn.execute(
                "INSERT INTO memory_rings(ring_id,memory_id,content_sha256,content,kind,state,source_refs_json,"
                "metadata_json,operation_id,created_at,created_by) VALUES(?,?,?,?,?,'active',?,?,?,?,?)",
                (ring_id, memory_id, payload["content_sha256"], body, payload["kind"],
                 _json(payload["source_refs"]), _json(payload["metadata"]), operation_id,
                 payload["created_at"], payload["actor"]),
            )
            result = {"memory_id": memory_id, "ring_id": ring_id}
            conn.execute(
                "UPDATE commit_log SET result_json=? WHERE operation_id=?", (_json(result), operation_id)
            )
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def pending_outbox(self, *, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM outbox WHERE status IN ('pending','degraded') ORDER BY created_at LIMIT ?",
            (max(1, min(1000, int(limit))),),
        ).fetchall()
        conn.close()
        result = []
        for row in rows:
            value = dict(row)
            value["payload"] = _loads(value.pop("payload_json"), {})
            result.append(value)
        return result

    def get_meta(self, key: str, default: Any = None) -> Any:
        conn = self._connect()
        row = conn.execute("SELECT value FROM authority_meta WHERE key=?", (str(key),)).fetchone()
        conn.close()
        return _loads(row["value"], default) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO authority_meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(key), _json(value)),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
