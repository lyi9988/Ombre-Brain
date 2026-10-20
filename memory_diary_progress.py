"""Validated first-pass diary checkpoints, not adopted memory or owner proposals.

Writes use the existing Authority database and optimistic CAS. No model calls,
expiry-based retries, candidate transitions or new scheduling are performed.
"""
import json
from memory_authority import RevisionConflict


class DiaryProgress:
    def __init__(self, authority):
        self.authority = authority
        db = authority._connect()
        try:
            db.execute('CREATE TABLE IF NOT EXISTS memory_diary_task_progress ('
                       'task_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, progress_json TEXT NOT NULL)')
            db.commit()
        finally:
            db.close()

    def get(self, task_id):
        db = self.authority._connect()
        try:
            row = db.execute('SELECT * FROM memory_diary_task_progress WHERE task_id=?', (task_id,)).fetchone()
            return {'revision': row['revision'], 'progress': json.loads(row['progress_json'])} if row else None
        finally:
            db.close()

    def save(self, task_id, progress, *, expected_revision):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM memory_diary_task_progress WHERE task_id=?', (task_id,)).fetchone()
            current = row['revision'] if row else 0
            if current != expected_revision:
                raise RevisionConflict('diary progress changed during generation')
            next_revision = current + 1
            payload = json.dumps(progress, ensure_ascii=False, sort_keys=True)
            db.execute('INSERT INTO memory_diary_task_progress VALUES(?,?,?) '
                       'ON CONFLICT(task_id) DO UPDATE SET revision=excluded.revision, progress_json=excluded.progress_json',
                       (task_id, next_revision, payload))
            db.commit()
            return {'revision': next_revision, 'progress': progress}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
