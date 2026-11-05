"""Full bucket listings keep live activation semantics without per-file DB reads."""
import asyncio
from pathlib import Path

from bucket_manager import BucketManager
from memory_activation import PROJECTION, base_metadata


def write_bucket(manager, bucket_id, *, category='dynamic', body='Original body'):
    path = Path(manager.base_dir, category, 'domain', bucket_id + '.md')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('---\nid: ' + bucket_id + '\nactivation_count: 0.5\n'
                    'last_active: "2026-11-01T00:00:00+00:00"\n---\n\n' + body,
                    encoding='utf-8')
    return path


def test_listing_matches_single_reads_beyond_one_sql_batch(tmp_path, monkeypatch):
    manager = BucketManager({'buckets_dir': str(tmp_path)})
    paths = [write_bucket(manager, 'bucket-' + str(index)) for index in range(405)]
    archived = write_bucket(manager, 'archived', category='archive')
    for bucket_id in ('bucket-0', 'bucket-404', 'archived'):
        manager.activation_ledger.record(bucket_id, 'activation-' + bucket_id,
                                         '2026-11-05T12:00:00+08:00')
    expected = {str(path): manager._load_bucket(str(path)) for path in [*paths, archived]}
    original_connect = manager.activation_ledger._connect
    connections, selects = [], []

    def connect():
        connections.append(True)
        db = original_connect()
        db.set_trace_callback(lambda sql: selects.append(sql) if sql.startswith('SELECT') else None)
        return db

    monkeypatch.setattr(manager.activation_ledger, '_connect', connect)
    listing = asyncio.run(manager.list_all(include_archive=True))
    assert len(listing) == 406
    assert {row['path']: row for row in listing} == expected
    assert len(connections) == 1 and len(selects) == 2
    assert next(row for row in listing if row['id'] == 'bucket-404')['metadata']['activation_count'] == 1.5
    assert all(row['content_start_line'] == 7 for row in listing)


def test_next_listing_observes_other_process_activation_and_owner_file_changes(tmp_path):
    manager = BucketManager({'buckets_dir': str(tmp_path)})
    path = write_bucket(manager, 'live')
    removed = write_bucket(manager, 'removed')
    first = asyncio.run(manager.list_all())
    assert len(first) == 2

    other_process = BucketManager({'buckets_dir': str(tmp_path)})
    other_process.activation_ledger.record('live', 'external-activation', '2026-11-06T00:00:00Z')
    path.write_text(path.read_text(encoding='utf-8').replace('Original body', '主人最新正文'), encoding='utf-8')
    removed.unlink()
    second = asyncio.run(manager.list_all())
    assert len(second) == 1 and second[0]['content'] == '主人最新正文'
    assert second[0]['metadata']['activation_count'] == 1.5
    assert second[0]['metadata']['last_active'] == '2026-11-06T00:00:00Z'
    assert second[0]['metadata'][PROJECTION]['receipt_count'] == 1
    assert base_metadata(second[0]['metadata'])['activation_count'] == 0.5
    assert first[0]['content'] == 'Original body'


def test_reprojecting_metadata_never_counts_receipt_twice_and_empty_list_reads_nothing(tmp_path, monkeypatch):
    manager = BucketManager({'buckets_dir': str(tmp_path)})
    ledger = manager.activation_ledger
    ledger.record('live', 'once', '2026-11-06T00:00:00Z')
    metadata = {'activation_count': 0.5, 'last_active': '2026-11-01T00:00:00Z'}
    once = ledger.effective('live', metadata)
    assert ledger.effective_many([('live', once), ('live', metadata)]) == [once, once]
    assert metadata['activation_count'] == 0.5
    monkeypatch.setattr(ledger, '_connect', lambda: (_ for _ in ()).throw(AssertionError('unexpected read')))
    assert asyncio.run(manager.list_all()) == []


def test_numeric_yaml_id_keeps_the_single_read_text_affinity(tmp_path):
    manager = BucketManager({'buckets_dir': str(tmp_path)})
    path = write_bucket(manager, '123')
    manager.activation_ledger.record('123', 'numeric-activation', '2026-11-06T00:00:00Z')
    single = manager._load_bucket(str(path))
    listing = asyncio.run(manager.list_all())
    assert single['id'] == 123 and listing == [single]
    assert listing[0]['metadata']['activation_count'] == 1.5


def test_one_non_scalar_yaml_id_does_not_break_other_memories(tmp_path):
    manager = BucketManager({'buckets_dir': str(tmp_path)})
    good = write_bucket(manager, 'keep')
    bad = write_bucket(manager, 'invalid')
    bad.write_text(bad.read_text(encoding='utf-8').replace('id: invalid', 'id: [one, two]'), encoding='utf-8')
    assert manager._load_bucket(str(bad)) is None
    assert asyncio.run(manager.list_all()) == [manager._load_bucket(str(good))]
