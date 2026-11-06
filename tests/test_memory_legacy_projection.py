"""Only proven migrated legacy files may omit the Authority version header."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sqlite3

import frontmatter
import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore
from memory_commit_service import BucketMemoryProjection, MemoryCommitService, ProjectionNotApplied, body_sha256
from memory_l0_sources import MemoryL0Sources
from memory_l1_schema import catalog_with_defaults
from memory_l1_service import L1OwnerError, MemoryL1RevisionService
from memory_l1_source_readers import configured_source_readers
from memory_legacy_backfill import LegacyMemoryBackfill


def run(task):
    return asyncio.run(task)


def setup(tmp_path):
    config = {'buckets_dir': str(tmp_path / 'buckets'), 'state_dir': str(tmp_path / 'state')}
    authority, buckets = MemoryAuthorityStore(config), BucketManager(config)
    projection = BucketMemoryProjection(config, buckets)
    commit = MemoryCommitService(authority, projection)
    revisions = MemoryL1RevisionService(authority, commit, buckets, lambda: (catalog_with_defaults(), 'catalog-test'))
    run(buckets.create('完整的旧回忆。', bucket_id='old-memory', name='旧记忆', extra_metadata={
        'owner_unrelated': {'keep': True}, 'activation_count': 3}))
    bucket = run(buckets.get('old-memory'))
    snapshot = projection.snapshot_path('old-memory', 1)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(Path(bucket['path']).read_bytes())
    authority.import_legacy_memory(memory_id='old-memory', bucket_id='old-memory',
        body_sha256=body_sha256(bucket['content']), snapshot_path=str(snapshot),
        metadata=bucket['metadata'], source_refs=['conversation_turn:1'], recall_policy='manual_only')
    run(buckets.update('old-memory', extra_metadata={'activation_count': 7}, updated_at=bucket['metadata']['updated_at']))
    with sqlite3.connect(tmp_path / 'buckets' / 'gateway_state.db') as db:
        db.execute('CREATE TABLE conversation_turns(id INTEGER PRIMARY KEY,profile_id TEXT,session_id TEXT,user_text TEXT,assistant_text TEXT,created_at TEXT)')
        db.execute("INSERT INTO conversation_turns VALUES(1,'p1','c1','那次想抱抱。','','2026-09-01T00:00:00Z')")
    source = {'namespace': 'conversation_turn', 'system': 'ombre', 'event_id': '1', 'profile_id': 'p1',
        'conversation_id': 'c1', 'version_id': '', 'role': 'user', 'text': '那次想抱抱。', 'recorded_at': '2026-09-01T00:00:00Z'}
    payload = {'expected_revision': 1, 'request_id': 'legacy-header-one', 'original_body_sha256': body_sha256(bucket['content']),
        'document': {'format': 'semantic-v1', 'narrative': {'title': '那次陪伴', 'body': '那次她说想要抱抱。'}, 'entities': [],
            'statements': [{'content': '那次说想要抱抱。', 'subject_ref': None,
                'classification': {'kind': 'preference', 'stability': 'episode', 'modality': None, 'basis': None},
                'time': {'event': None, 'applicability': None}, 'conditions': None, 'evidence_refs': ['source:1']}],
            'retrieval': {'keywords': ['那次陪伴'], 'tags': []}},
        'sources': [source], 'source_refs': ['source:1'], 'confidence': .9,
        'generation': {'actor': 'legacy_backfill', 'task_id': 'existing-model-result', 'prompt_revision': None, 'model_route_revision': None}}
    service = LegacyMemoryBackfill(revisions,
        lambda: MemoryL0Sources(authority, configured_source_readers(config, authority)), config=config)
    return service, payload, snapshot


def test_proven_imported_headerless_file_stages_and_adopts_without_resetting_activity(tmp_path):
    service, payload, snapshot = setup(tmp_path)
    old = service.authority.get_memory_revision('old-memory', 1)
    old_bytes = snapshot.read_bytes()
    before = run(service.revisions.buckets.get('old-memory'))
    staged = run(service.stage('old-memory', payload))
    proof = staged['candidate']['proposal']['metadata']['legacy_backfill']['legacy_projection']
    assert proof['schema'] == 'legacy-unversioned-projection-v1'
    assert run(service.revisions.buckets.get('old-memory')) == before
    adopted = run(service.adopt(staged['candidate'], request_id='owner-approve-import'))
    assert adopted['revision'] == 2
    current = run(service.revisions.buckets.get('old-memory'))
    assert current['metadata']['memory_revision'] == 2 and current['metadata']['activation_count'] == 7
    assert current['metadata']['owner_unrelated'] == {'keep': True}
    assert service.authority.get_memory('old-memory')['recall_policy'] == 'manual_only'
    assert service.authority.get_memory_revision('old-memory', 1) == old and snapshot.read_bytes() == old_bytes
    assert service.authority.get_memory_revision('old-memory', 2)['metadata']['activation_count'] == 7


@pytest.mark.parametrize('change', ['body', 'business_metadata', 'missing_snapshot', 'snapshot_metadata',
    'explicit_wrong_version', 'another_managed_marker', 'duplicate_bucket', 'not_migration'])
def test_missing_version_never_bypasses_a_mismatched_original(tmp_path, change):
    service, payload, snapshot = setup(tmp_path)
    bucket = run(service.revisions.buckets.get('old-memory'))
    if change == 'body':
        run(service.revisions.buckets.update('old-memory', content='未记录的新正文'))
    elif change == 'business_metadata':
        run(service.revisions.buckets.update('old-memory', name='主人另改过标题'))
    elif change == 'missing_snapshot':
        snapshot.unlink()
    elif change == 'snapshot_metadata':
        post = frontmatter.loads(snapshot.read_text(encoding='utf-8'))
        post['owner_unrelated'] = {'tampered': True}
        snapshot.write_text(frontmatter.dumps(post), encoding='utf-8')
    elif change == 'explicit_wrong_version':
        run(service.revisions.buckets.update('old-memory', extra_metadata={'memory_revision': '1'}))
    elif change == 'another_managed_marker':
        run(service.revisions.buckets.update('old-memory', extra_metadata={'memory_id': 'someone-else'}))
    elif change == 'duplicate_bucket':
        duplicate = Path(service.revisions.buckets.permanent_dir) / 'duplicate_old-memory.md'
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_bytes(Path(bucket['path']).read_bytes())
    else:
        with service.authority._connect() as db:
            db.execute("UPDATE memory_revisions SET decision_source='synthetic' WHERE memory_id='old-memory'")
    with pytest.raises(L1OwnerError, match='backfill_projection_changed'):
        run(service.stage('old-memory', payload))
    assert service.authority.get_memory('old-memory')['active_revision'] == 1
    assert not service.authority.list_candidates(status='all')


def test_commit_rechecks_the_old_body_after_prepare_and_preserves_a_concurrent_edit(tmp_path, monkeypatch):
    service, payload, snapshot = setup(tmp_path)
    row = run(service.stage('old-memory', payload))['candidate']
    projection = service.revisions.commit.projection
    original = projection.write_revision
    async def changed_before_write(**kwargs):
        await service.revisions.buckets.update('old-memory', content='主人刚编辑的未版本化正文')
        return await original(**kwargs)
    monkeypatch.setattr(projection, 'write_revision', changed_before_write)
    with pytest.raises(ProjectionNotApplied):
        run(service.adopt(row, request_id='race-with-owner'))
    assert service.authority.get_memory('old-memory')['active_revision'] == 1
    assert run(service.revisions.buckets.get('old-memory'))['content'] == '主人刚编辑的未版本化正文'


def test_commit_uses_latest_raw_activity_without_promoting_a_read_projection(tmp_path, monkeypatch):
    service, payload, _ = setup(tmp_path)
    row = run(service.stage('old-memory', payload))['candidate']
    projection = service.revisions.commit.projection
    original = projection.write_revision
    async def activity_before_write(**kwargs):
        await service.revisions.buckets.update('old-memory', extra_metadata={'activation_count': 8})
        return await original(**kwargs)
    monkeypatch.setattr(projection, 'write_revision', activity_before_write)
    run(service.adopt(row, request_id='concurrent-normal-use'))
    current = run(service.revisions.buckets.get('old-memory'))
    assert current['metadata']['activation_count'] == 8
    assert '_memory_activation_projection' not in frontmatter.load(current['path']).metadata


def test_legacy_adoption_preserves_activation_receipts_without_double_counting(tmp_path):
    service, payload, _ = setup(tmp_path)
    buckets = service.revisions.buckets
    buckets.activation_ledger.record('old-memory', 'existing-business-use', '2026-11-06T12:00:00Z')
    before = run(buckets.get('old-memory'))['metadata']
    row = run(service.stage('old-memory', payload))['candidate']
    run(service.adopt(row, request_id='preserve-existing-activation'))
    after = run(buckets.get('old-memory'))
    assert after['metadata']['activation_count'] == before['activation_count'] == 8
    assert after['metadata']['last_active'] == before['last_active']
    assert frontmatter.load(after['path'])['activation_count'] == 7
    assert '_memory_activation_projection' not in service.authority.get_memory_revision('old-memory', 2)['metadata']
