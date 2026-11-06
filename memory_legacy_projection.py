"""Strict proof for migrated revision-1 buckets predating Authority headers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import frontmatter

from memory_commit_service import ProjectionNotApplied, body_sha256

MARKERS = {'memory_id', 'memory_revision', 'memory_body_sha256',
           'memory_operation_id', 'memory_authority_schema'}
ACTIVITY_FIELDS = {'activation_count', 'last_active', 'updated_at'}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)


def _business_hash(metadata):
    return hashlib.sha256(_json({k: v for k, v in metadata.items() if k not in ACTIVITY_FIELDS}).encode()).hexdigest()


def _read(path):
    raw = path.read_bytes()
    return raw, frontmatter.loads(raw.decode('utf-8'))


def verify_legacy_files(projection, proof, existing, *, memory_id, bucket_id, expected_revision):
    """Recheck files immediately before projection; the coordinator mints proof."""
    def fail():
        raise ProjectionNotApplied('legacy projection proof no longer matches')
    if (not isinstance(proof, dict) or proof.get('schema') != 'legacy-unversioned-projection-v1'
            or proof.get('memory_id') != memory_id or proof.get('bucket_id') != bucket_id
            or proof.get('base_revision') != expected_revision or expected_revision != 1):
        fail()
    snapshot = projection.snapshot_path(memory_id, expected_revision).resolve()
    if (str(snapshot) != proof.get('snapshot_path') or not snapshot.is_file()
            or not snapshot.is_relative_to(projection.revisions_dir.resolve())):
        fail()
    try:
        snap_raw, snap = _read(snapshot)
        path = Path(existing['path']).resolve()
        base = Path(projection.config['buckets_dir']).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            fail()
        matches = [p.resolve() for folder in ('permanent', 'dynamic', 'archive', 'feel')
                   for p in (base / folder).rglob('*.md')
                   if p.stem == bucket_id or p.stem.endswith('_' + bucket_id)]
        if matches != [path]:
            fail()
        _, bucket = _read(path)
    except (OSError, ValueError, KeyError, TypeError):
        fail()
    if (hashlib.sha256(snap_raw).hexdigest() != proof.get('snapshot_file_sha256')
            or MARKERS.intersection(snap.metadata) or MARKERS.intersection(bucket.metadata)
            or snap.get('id') != bucket_id or bucket.get('id') != bucket_id
            or 'memory_l1' in snap.metadata or 'memory_l1' in bucket.metadata
            or body_sha256(snap.content) != proof.get('body_sha256')
            or body_sha256(bucket.content) != proof.get('body_sha256')
            or body_sha256(existing.get('content') or '') != proof.get('body_sha256')
            or _business_hash(snap.metadata) != proof.get('business_metadata_sha256')
            or _business_hash(bucket.metadata) != proof.get('business_metadata_sha256')):
        fail()
    return {k: deepcopy(bucket.metadata[k]) for k in ACTIVITY_FIELDS if k in bucket.metadata}


def verify_unversioned_legacy(authority, projection, existing, *, memory_id, bucket_id, expected_revision):
    """Missing headers alone never authorize a write or import a new memory."""
    def fail():
        raise ProjectionNotApplied('unversioned bucket is not the proven imported base')
    memory, generation = authority.get_memory_state_snapshot(memory_id)
    record = authority.get_memory_revision(memory_id, expected_revision)
    operation_id = 'legacy-memory-import:' + memory_id + ':1'
    imported = authority.get_commit(operation_id)
    snapshot = projection.snapshot_path(memory_id, expected_revision).resolve()
    if (expected_revision != 1 or not memory or not record or memory['state'] == 'tombstoned'
            or memory['active_revision'] != expected_revision or memory['bucket_id'] != bucket_id
            or memory['body_sha256'] != record['body_sha256']
            or record['created_by'] != 'migration' or record['decision_source'] != 'migration'
            or record['operation_id'] != operation_id or 'memory_l1' in record['metadata']
            or not imported or imported['status'] != 'committed' or imported['operation_kind'] != 'legacy_memory_import'
            or Path(record['snapshot_path']).resolve() != snapshot or not snapshot.is_file()
            or not snapshot.is_relative_to(projection.revisions_dir.resolve())):
        fail()
    payload = imported['payload']
    if (payload.get('memory_id') != memory_id or payload.get('bucket_id') != bucket_id
            or payload.get('revision') != 1 or payload.get('body_sha256') != record['body_sha256']
            or Path(str(payload.get('snapshot_path') or '')).resolve() != snapshot
            or _json(payload.get('metadata')) != _json(record['metadata'])
            or payload.get('source_refs') != record['source_refs']):
        fail()
    try:
        raw, post = _read(snapshot)
    except (OSError, ValueError, UnicodeError):
        fail()
    if _json(post.metadata) != _json(record['metadata']):
        fail()
    proof = {'schema': 'legacy-unversioned-projection-v1', 'memory_id': memory_id, 'bucket_id': bucket_id,
        'base_revision': expected_revision, 'body_sha256': record['body_sha256'],
        'import_operation_id': operation_id, 'snapshot_path': str(snapshot),
        'snapshot_file_sha256': hashlib.sha256(raw).hexdigest(),
        'business_metadata_sha256': _business_hash(post.metadata), 'state_generation': generation}
    activity = verify_legacy_files(projection, proof, existing, memory_id=memory_id,
        bucket_id=bucket_id, expected_revision=expected_revision)
    return proof, activity
