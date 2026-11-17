"""Read the current user section and its cited sources from the shared mounts."""
import json
import sqlite3
from pathlib import Path
from threading import Lock

import frontmatter
import yaml

from user_portrait_chat import (
    bind_source_authority, build_user_portrait_projection, chat_evidence_ids,
    portrait_state_path, source_binding_matches,
)


class GatewayUserPortrait:
    def __init__(self, config, authority):
        self.state_path = Path(portrait_state_path(config))
        self.buckets_dir = Path(config.get("buckets_dir") or "buckets").resolve()
        self.authority = authority
        self._lock = Lock()
        self._stamp = None
        self._state = {}

    def _read_user(self):
        # Brain atomically replaces this file. Only retain the user section,
        # and only reuse its parsed version while the file identity is current.
        with self._lock:
            stat = self.state_path.stat()
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if stamp != self._stamp:
                with self.state_path.open(encoding="utf-8") as src:
                    user = json.load(src)["portrait"]["user"]
                if not isinstance(user, dict):
                    raise ValueError("invalid user portrait")
                self._state = {"portrait": {"user": user}}
                self._stamp = stamp
            return self._state

    def _source(self, ref, authorities, semantics):
        bucket_id = ref.get("bucket_id")
        authority = authorities.get(bucket_id)
        semantic = semantics.get(bucket_id)
        rel = Path(str(ref.get("relative_path") or ""))
        if not authority or not semantic or rel.is_absolute() or ".." in rel.parts:
            return None
        path = (self.buckets_dir / rel).resolve()
        if not path.is_relative_to(self.buckets_dir):
            return None
        parts = path.relative_to(self.buckets_dir).parts
        if not parts or parts[0] not in {"permanent", "dynamic", "feel"} or path.suffix != ".md":
            return None
        item = frontmatter.loads(path.read_text(encoding="utf-8"))
        if str(item.get("id", path.stem)) != bucket_id:
            return None
        bucket = {"id": bucket_id, "metadata": item.metadata, "content": item.content}
        return bind_source_authority(bucket, authority, semantic)

    def read(self, *, verified_v2=None):
        try:
            state = self._read_user()
            ids = chat_evidence_ids(state)
            authorities = self.authority.memory_index_metadata_map(ids) if ids else {}
            semantics = self.authority.memory_l1_recall_map(ids) if ids else {}
            sources = {}
            checks = {}

            def source_lookup(ref):
                bucket_id = ref.get("bucket_id")
                cache_key = (bucket_id, ref.get("relative_path"))
                if cache_key not in sources:
                    sources[cache_key] = self._source(ref, authorities, semantics)
                source = sources[cache_key]
                semantic = semantics.get(bucket_id) or {}
                if not source or semantic.get("state") != "l1_v2":
                    return source
                if not source_binding_matches(ref, source):
                    return None
                authority = source["_portrait_authority"]
                key = (authority["memory_id"], authority["memory_revision"])
                bindings = {binding["binding_id"] for binding in semantic["document"]["evidence_bindings"]}
                wanted = {ref["binding_id"]} if ref.get("binding_id") else bindings
                if not wanted or not wanted <= bindings:
                    return None
                if verified_v2 is None:
                    checks.setdefault(key, set()).update(wanted)
                    return source
                return source if wanted <= verified_v2.get(key, set()) else None

            projection = build_user_portrait_projection(state, source_lookup)
            if checks:
                # Only the verified pass can expose v2 claims to the caller.
                verified_v2 = {}
                projection = build_user_portrait_projection(state, source_lookup)
                projection["v2_source_checks"] = [
                    {"memory_id": memory_id, "revision": revision, "binding_ids": sorted(bindings)}
                    for (memory_id, revision), bindings in checks.items()
                ]
            return projection
        except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error, yaml.YAMLError):
            # Chat can continue. An earlier text must never stand in for a
            # missing state file, withdrawn source or unavailable authority.
            return {"text": "", "revision": None, "claim_count": 0, "status": "unavailable"}
