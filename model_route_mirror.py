"""Secretless Aizizhu Model Registry route mirror for Ombre internals."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


ALLOWED_ROUTE_IDS = frozenset({
    "memory_query_planner",
    "memory_domain_sentinel",
    "memory_semantic_rescue",
    "autonomy_reasoner",
})
FORBIDDEN_KEYS = frozenset({
    "api_key", "key", "token", "authorization", "secret", "base_url",
    "api_key_env", "extra_headers",
})


class ModelRouteMirrorError(RuntimeError):
    code = "model_route_mirror_failed"


class ModelRouteMirrorValidationError(ModelRouteMirrorError):
    code = "model_route_mirror_invalid"


class ModelRouteMirrorConflict(ModelRouteMirrorError):
    code = "model_route_mirror_conflict"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key or "").strip().lower()
            if normalized in FORBIDDEN_KEYS or normalized.endswith("_token") or normalized.endswith("_secret"):
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


class ModelRouteMirrorStore:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).resolve()
        self._active_stamp: tuple[int, int, int, int] | None = None
        self._active_cache: dict[str, Any] | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS model_route_mirrors (
                    revision INTEGER PRIMARY KEY,
                    route_sha256 TEXT NOT NULL,
                    routes_json TEXT NOT NULL,
                    source_authority TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_route_actions (
                    request_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_route_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _file_stamp(self) -> tuple[int, int, int, int] | None:
        try:
            main = self.path.stat()
            wal_path = Path(str(self.path) + "-wal")
            try:
                wal = wal_path.stat()
                wal_values = (int(wal.st_mtime_ns), int(wal.st_size))
            except FileNotFoundError:
                wal_values = (0, 0)
            return int(main.st_mtime_ns), int(main.st_size), *wal_values
        except FileNotFoundError:
            return None

    @staticmethod
    def normalize_routes(routes: Any) -> list[dict[str, Any]]:
        raw_routes = routes if isinstance(routes, list) else []
        normalized = []
        seen = set()
        for raw in raw_routes:
            if not isinstance(raw, dict):
                raise ModelRouteMirrorValidationError("route must be an object")
            route_id = str(raw.get("route_id") or "").strip()
            if route_id not in ALLOWED_ROUTE_IDS or route_id in seen:
                raise ModelRouteMirrorValidationError(f"unsupported or duplicate route_id: {route_id}")
            seen.add(route_id)
            enabled = bool(raw.get("enabled", True))
            if route_id == "autonomy_reasoner" and enabled:
                raise ModelRouteMirrorValidationError("autonomy_reasoner is reserved and must remain disabled")
            candidates = []
            for position, candidate in enumerate(raw.get("candidates") or []):
                if not isinstance(candidate, dict):
                    raise ModelRouteMirrorValidationError("candidate must be an object")
                if _contains_forbidden_key(candidate):
                    raise ModelRouteMirrorValidationError("route mirror must not contain secrets or endpoints")
                model_id = str(candidate.get("model_id") or candidate.get("model") or "").strip()
                provider_id = str(candidate.get("provider_id") or "").strip()
                if not model_id:
                    raise ModelRouteMirrorValidationError("candidate model_id is required")
                overrides = candidate.get("overrides") if isinstance(candidate.get("overrides"), dict) else {}
                if _contains_forbidden_key(overrides):
                    raise ModelRouteMirrorValidationError("candidate overrides contain forbidden keys")
                candidates.append({
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "enabled": bool(candidate.get("enabled", True)),
                    "position": int(candidate.get("position", position)),
                    "overrides": dict(overrides),
                })
            candidates.sort(key=lambda item: (item["position"], item["model_id"]))
            normalized.append({
                "route_id": route_id,
                "enabled": enabled,
                "candidates": candidates,
            })
        normalized.sort(key=lambda item: item["route_id"])
        return normalized

    def put(
        self,
        *,
        revision: int,
        routes: Any,
        route_sha256: str,
        source_authority: str,
        request_id: str,
    ) -> dict[str, Any]:
        revision = int(revision)
        if revision <= 0:
            raise ModelRouteMirrorValidationError("revision must be positive")
        normalized = self.normalize_routes(routes)
        observed_sha = _sha(normalized)
        if str(route_sha256 or "") != observed_sha:
            raise ModelRouteMirrorValidationError("route_sha256 mismatch")
        payload = {
            "revision": revision,
            "route_sha256": observed_sha,
            "routes": normalized,
            "source_authority": str(source_authority or "aizizhu.model_registry"),
        }
        fingerprint = _sha(payload)
        now = int(time.time() * 1000)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if request_id:
                prior = conn.execute(
                    "SELECT * FROM model_route_actions WHERE request_id=?", (request_id,)
                ).fetchone()
                if prior:
                    if str(prior["fingerprint"]) != fingerprint:
                        raise ModelRouteMirrorConflict("request_id payload differs")
                    conn.commit()
                    return json.loads(prior["result_json"])
            current_row = conn.execute(
                "SELECT value FROM model_route_meta WHERE key='active_revision'"
            ).fetchone()
            current_revision = int(current_row["value"]) if current_row else 0
            if revision < current_revision:
                raise ModelRouteMirrorConflict("stale route revision")
            existing = conn.execute(
                "SELECT * FROM model_route_mirrors WHERE revision=?", (revision,)
            ).fetchone()
            if existing and str(existing["route_sha256"]) != observed_sha:
                raise ModelRouteMirrorConflict("revision already exists with different hash")
            if not existing:
                conn.execute(
                    "INSERT INTO model_route_mirrors(revision,route_sha256,routes_json,"
                    "source_authority,created_at_ms) VALUES(?,?,?,?,?)",
                    (revision, observed_sha, _canonical_json(normalized), payload["source_authority"], now),
                )
            conn.execute(
                "INSERT INTO model_route_meta(key,value) VALUES('active_revision',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(revision),),
            )
            result = {**payload, "status": "mirrored", "secretless": True}
            if request_id:
                conn.execute(
                    "INSERT INTO model_route_actions(request_id,fingerprint,revision,result_json,created_at_ms) "
                    "VALUES(?,?,?,?,?)",
                    (request_id, fingerprint, revision, _canonical_json(result), now),
                )
            conn.commit()
            self._active_stamp = None
            self._active_cache = None
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def active(self) -> dict[str, Any] | None:
        stamp = self._file_stamp()
        if stamp == self._active_stamp:
            return json.loads(_canonical_json(self._active_cache)) if self._active_cache else None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT m.* FROM model_route_mirrors m JOIN model_route_meta meta "
                "ON meta.key='active_revision' AND m.revision=CAST(meta.value AS INTEGER)"
            ).fetchone()
        finally:
            conn.close()
        if not row:
            self._active_stamp = stamp
            self._active_cache = None
            return None
        value = {
            "revision": int(row["revision"]),
            "route_sha256": str(row["route_sha256"]),
            "routes": json.loads(row["routes_json"]),
            "source_authority": str(row["source_authority"]),
            "secretless": True,
        }
        self._active_stamp = stamp
        self._active_cache = value
        return json.loads(_canonical_json(value))

    def resolve(self, route_id: str) -> dict[str, Any] | None:
        active = self.active()
        if not active:
            return None
        route = next(
            (item for item in active["routes"] if item.get("route_id") == str(route_id)),
            None,
        )
        if not route or not route.get("enabled"):
            return None
        candidates = [item for item in route.get("candidates", []) if item.get("enabled")]
        return {
            "route_id": str(route_id),
            "revision": active["revision"],
            "route_sha256": active["route_sha256"],
            "candidates": candidates,
            "source_authority": active["source_authority"],
        }


def route_slice_sha256(routes: Any) -> str:
    return _sha(ModelRouteMirrorStore.normalize_routes(routes))
