"""Shared bucket activation receipts; never writes a memory body or new score rule."""
from __future__ import annotations

import copy
import os
import sqlite3
from datetime import datetime, timezone

PROJECTION = '_memory_activation_projection'


def base_metadata(metadata):
    """Remove a read projection before any metadata round trip to Markdown."""
    result = copy.deepcopy(dict(metadata))
    marker = result.pop(PROJECTION, None)
    if isinstance(marker, dict):
        count = result.get('activation_count')
        contribution = marker.get('receipt_count')
        if type(count) in (int, float) and type(contribution) is int:
            result['activation_count'] = count - contribution
        if result.get('last_active') == marker.get('effective_last_active'):
            result['last_active'] = marker.get('base_last_active')
    return result


class BucketActivationLedger:
    def __init__(self, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, '.memory_activation.sqlite3')
        with self._connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS bucket_activation_receipts (
                activation_id TEXT PRIMARY KEY, bucket_id TEXT NOT NULL, activated_at TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS bucket_activation_bucket ON bucket_activation_receipts(bucket_id)')

    def _connect(self):
        class Connection(sqlite3.Connection):
            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    self.close()
        db = sqlite3.connect(self.path, timeout=10, factory=Connection)
        db.row_factory = sqlite3.Row
        return db

    def record(self, bucket_id, activation_id, activated_at):
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT bucket_id FROM bucket_activation_receipts WHERE activation_id=?',
                (activation_id,)).fetchone()
            if prior and prior['bucket_id'] != bucket_id:
                raise ValueError('activation_identity_conflict')
            db.execute('INSERT OR IGNORE INTO bucket_activation_receipts VALUES(?,?,?)',
                (activation_id, bucket_id, activated_at))

    def effective(self, bucket_id, metadata):
        with self._connect() as db:
            row = db.execute('SELECT COUNT(*) AS count,MAX(activated_at) AS latest '
                'FROM bucket_activation_receipts WHERE bucket_id=?', (bucket_id,)).fetchone()
        return self._effective_metadata(metadata, row['count'], row['latest'])

    def effective_many(self, items):
        """Project one live receipt snapshot over all requested base metadata.

        Only the reads are shared. Nothing survives this call, so another
        process's activation is observed by the next list or single read.
        """
        items = list(items)
        if not items:
            return []
        def key(bucket_id):
            # A YAML ID can be numeric. Match SQLite's TEXT affinity, as the
            # existing single-bucket query does, without changing the ID.
            if isinstance(bucket_id, bool):
                return str(int(bucket_id))
            return str(bucket_id) if isinstance(bucket_id, (int, float)) else bucket_id
        ids = list(dict.fromkeys(key(bucket_id) for bucket_id, _metadata in items))
        receipts = {}
        with self._connect() as db:
            db.execute('BEGIN')
            # Keep each IN clause under SQLite's older parameter limit while
            # retaining every bucket and using the same read transaction.
            for offset in range(0, len(ids), 400):
                batch = ids[offset:offset + 400]
                rows = db.execute('SELECT bucket_id,COUNT(*) AS count,MAX(activated_at) AS latest '
                    'FROM bucket_activation_receipts WHERE bucket_id IN ('
                    + ','.join('?' for _ in batch) + ') GROUP BY bucket_id', batch)
                receipts.update({row['bucket_id']: (row['count'], row['latest']) for row in rows})
        return [self._effective_metadata(metadata, *receipts.get(key(bucket_id), (0, None)))
                for bucket_id, metadata in items]

    @staticmethod
    def _effective_metadata(metadata, count, latest):
        result = base_metadata(metadata)
        base_count, base_active = result.get('activation_count', 0), result.get('last_active')
        if not count:
            return result
        result['activation_count'] = base_count + count
        def timestamp(value):
            try:
                parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
            except (TypeError, ValueError, OverflowError):
                return float('-inf')
        result['last_active'] = latest if timestamp(latest) > timestamp(base_active) else base_active
        result[PROJECTION] = {'receipt_count': count, 'base_last_active': base_active,
            'effective_last_active': result['last_active']}
        return result
