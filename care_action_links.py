"""Care memo/action links in the original reminder DB; no second content store."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing


class CareLinkError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def intent_token(item):
    # Injection counters and automatic due/expiry changes are observations,
    # while editing these fields invalidates an earlier close instruction.
    return digest({key: item.get(key) for key in ('title', 'content', 'source',
        'channel', 'session_id', 'start_at', 'end_at', 'repeat_rule',
        'interval_rounds', 'cooldown_minutes', 'daily_limit', 'max_injections',
        'next_due_at', 'care_revision')})


class CareActionLinks:
    def __init__(self, reminders):
        self.reminders = reminders
        with closing(self.reminders._connect()) as db:
            self.reminders._ensure_columns(db, 'reminders', {'care_revision': 'INTEGER NOT NULL DEFAULT 0'})
            fields = ('title','content','status','source','channel','session_id','start_at','end_at',
                      'next_due_at','repeat_rule','interval_rounds','cooldown_minutes','daily_limit','max_injections','resolved_at')
            watched = ','.join(fields)
            changed = ' OR '.join(f'OLD.{key} IS NOT NEW.{key}' for key in fields)
            db.execute(f'''CREATE TRIGGER IF NOT EXISTS care_revision_on_intent
                AFTER UPDATE OF {watched} ON reminders WHEN {changed}
                BEGIN UPDATE reminders SET care_revision=OLD.care_revision+1 WHERE id=NEW.id; END''')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS care_action_links (
                    reminder_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                    action_id TEXT NOT NULL, action_revision INTEGER NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS care_link_operations (
                    operation_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL,
                    result TEXT NOT NULL);
            ''')
            db.commit()

    @staticmethod
    def visible(item, session_id):
        return bool(session_id and item.get('session_id', '') in ('', session_id)
                    and item.get('channel', 'global') in ('', 'global', '*', 'gateway'))

    def _project(self, db, item, scope):
        item = dict(item)
        link = db.execute('SELECT * FROM care_action_links WHERE reminder_id=?', (item['id'],)).fetchone()
        item['snapshot_token'] = digest(dict(item))
        item['intent_token'] = intent_token(item)
        item['action_link'] = json.loads(link['payload']) if link and link['scope'] == scope else None
        item['linked_elsewhere'] = bool(link and link['scope'] != scope)
        item['delivery_status'] = 'chat_injection_only'
        return item

    def read(self, reminder_id, *, scope, session_id):
        with closing(self.reminders._connect()) as db:
            row = db.execute('SELECT * FROM reminders WHERE id=?', (reminder_id,)).fetchone()
            if row is None or not self.visible(dict(row), session_id):
                raise CareLinkError('care_memo_not_found')
            return self._project(db, row, scope)

    def list(self, *, scope, session_id, offset=0, limit=20):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise CareLinkError('invalid_care_page')
        # Deliberately never calls list()/due()/archive_expired().
        with closing(self.reminders._connect()) as db:
            rows = db.execute('''SELECT * FROM reminders
                WHERE session_id IN ('', ?) AND channel IN ('', 'global', '*', 'gateway')
                ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, created_at DESC, id
                LIMIT ? OFFSET ?''', (session_id, limit + 1, offset)).fetchall() if session_id else []
            return {'items': [self._project(db, x, scope) for x in rows[:limit]],
                    'has_more': len(rows) > limit, 'next_offset': offset + min(limit, len(rows))}

    def sync(self, fields):
        keys = {'operation_id', 'scope', 'session_id', 'reminder_id', 'snapshot_token',
            'intent_token', 'action_id', 'action_revision', 'action_status', 'completion_policy',
            'completion_kind', 'actor'}
        if not isinstance(fields, dict) or set(fields) != keys:
            raise CareLinkError('invalid_care_link')
        for key in keys - {'action_revision'}:
            if not isinstance(fields[key], str) or len(fields[key]) > 500:
                raise CareLinkError('invalid_care_link')
        if (not all(fields[k] for k in ('operation_id', 'scope', 'session_id', 'reminder_id', 'action_id'))
                or type(fields['action_revision']) is not int or fields['action_revision'] < 1
                or not re.fullmatch('[0-9a-f]{64}', fields['snapshot_token'])
                or not re.fullmatch('[0-9a-f]{64}', fields['intent_token'])
                or fields['completion_policy'] not in ('keep', 'close_once')
                or fields['actor'] not in ('owner', 'agent', 'executor')
                or fields['action_status'] not in ('planned', 'paused', 'running', 'unknown', 'completed', 'failed', 'cancelled')):
            raise CareLinkError('invalid_care_link')
        fingerprint = digest(fields)
        db = self.reminders._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            receipt = db.execute('SELECT * FROM care_link_operations WHERE operation_id=?', (fields['operation_id'],)).fetchone()
            if receipt:
                if receipt['payload_hash'] != fingerprint:
                    raise CareLinkError('care_operation_payload_conflict')
                db.rollback()
                return json.loads(receipt['result'])
            raw = db.execute('SELECT * FROM reminders WHERE id=?', (fields['reminder_id'],)).fetchone()
            if raw is None or not self.visible(dict(raw), fields['session_id']):
                raise CareLinkError('care_memo_not_found')
            item = dict(raw)
            existing = db.execute('SELECT * FROM care_action_links WHERE reminder_id=?', (item['id'],)).fetchone()
            if existing and (existing['scope'] != fields['scope'] or existing['action_id'] != fields['action_id']):
                raise CareLinkError('care_memo_already_linked')
            prior = json.loads(existing['payload']) if existing else None
            if existing and fields['action_revision'] <= existing['action_revision']:
                if fields['action_revision'] == existing['action_revision'] and any(
                        prior.get(k) != fields[k] for k in keys - {'operation_id'}):
                    raise CareLinkError('care_link_revision_conflict')
                result = {'ok': True, 'state': 'superseded', 'action_link': prior}
            else:
                # A changed snapshot/policy is a fresh relink, not permission
                # to close newly edited content using an old instruction.
                fresh = not prior or any(prior.get(k) != fields[k] for k in ('snapshot_token', 'intent_token', 'completion_policy'))
                if fresh and digest(item) != fields['snapshot_token']:
                    raise CareLinkError('care_memo_revision_conflict')
                if fresh and intent_token(item) != fields['intent_token']:
                    raise CareLinkError('care_memo_revision_conflict')
                if fields['completion_policy'] == 'close_once':
                    if item['repeat_rule'] not in ('once', 'none'):
                        raise CareLinkError('recurring_care_memo_must_remain')
                    if fresh and fields['actor'] != 'owner':
                        raise CareLinkError('care_close_requires_owner')
                    if fields['action_status'] == 'completed' and fields['completion_kind'] == 'owner_mark':
                        if item['status'] == 'active':
                            if intent_token(item) != fields['intent_token']:
                                raise CareLinkError('care_memo_source_changed')
                            from utils import now_iso
                            stamp = now_iso()
                            db.execute("UPDATE reminders SET status='done', resolved_at=?, updated_at=? WHERE id=?",
                                       (stamp, stamp, item['id']))
                link = {k: v for k, v in fields.items() if k != 'operation_id'}
                db.execute('INSERT OR REPLACE INTO care_action_links VALUES (?,?,?,?,?)',
                           (item['id'], fields['scope'], fields['action_id'], fields['action_revision'], json.dumps(link)))
                result = {'ok': True, 'state': 'synced', 'action_link': link}
            db.execute('INSERT INTO care_link_operations VALUES (?,?,?)',
                       (fields['operation_id'], fingerprint, json.dumps(result)))
            db.commit()
            return result
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
