"""Durable preparation receipts referencing the existing Authority candidates.

Only identifiers and counters live in authority_meta. All generated text,
semantic documents, revisions and owner decisions stay in candidates.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json

from memory_authority import CandidateNotFound, IdempotencyConflict

PREFIX = 'memory_candidate_preparation:'


class MemoryCandidatePreparation:
    def __init__(self, authority): self.authority = authority

    def get(self, key): return self.authority.get_meta(PREFIX + key)

    def prepare(self, key, proposals, policy, *, scope, source_fingerprint, generated_count=None,
                issue=None, generation_fingerprint=None, replace_failure=False):
        """Save every validated draft and its ID receipt in one transaction."""
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + key,)).fetchone()
            if prior:
                receipt = json.loads(prior['value'])
                if receipt['source_fingerprint'] != source_fingerprint:
                    raise IdempotencyConflict('preparation source changed')
                if not replace_failure or not receipt.get('issue'):
                    db.commit()
                    return receipt
            else:
                receipt = None
            ids = list(receipt['candidate_ids']) if receipt else []
            for proposal in proposals:
                proposal = replace(proposal, metadata={**proposal.metadata, 'preparation_key': key})
                # An existing owner revision/decision always wins. A completed
                # receipt normally prevents reaching this recovery path.
                row = db.execute('SELECT * FROM candidates WHERE candidate_id=?', (proposal.proposal_id,)).fetchone()
                if not row:
                    self.authority._put_candidate(db, proposal, policy.evaluate(proposal))
                ids.append(proposal.proposal_id)
            unique_ids = list(dict.fromkeys(ids))
            receipt = {'key': key, 'scope': scope, 'source_fingerprint': source_fingerprint,
                'candidate_ids': unique_ids, 'processed_ids': receipt['processed_ids'] if receipt else [], 'stale': False,
                'generated_count': (receipt['generated_count'] if receipt else 0) + (len(proposals) if generated_count is None else generated_count),
                'validated_count': len(unique_ids), 'issue': issue, 'generation_fingerprint': generation_fingerprint,
                'requires_manual_retry': bool(issue),
                'created_at': receipt['created_at'] if receipt else datetime.now(timezone.utc).isoformat(timespec='seconds')}
            db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (PREFIX + key, json.dumps(receipt, ensure_ascii=False, sort_keys=True)))
            db.commit()
            return receipt
        except Exception:
            db.rollback()
            raise
        finally: db.close()

    def rows(self, receipt):
        db = self.authority._connect()
        try:
            rows = db.execute('SELECT c.* FROM json_each(?) i JOIN candidates c ON c.candidate_id=i.value ORDER BY i.key',
                (json.dumps(receipt['candidate_ids']),)).fetchall()
            if len(rows) != len(receipt['candidate_ids']): raise CandidateNotFound(receipt['key'])
            return [self.authority._candidate_row(row) for row in rows]
        finally: db.close()

    def remaining(self, receipt):
        if receipt.get('stale'): return []
        done = set(receipt['processed_ids'])
        # Explicit owner decisions are terminal for scheduling; do not revive
        # rejection/defer or overwrite an owner edit while continuing a batch.
        return [row for row in self.rows(receipt) if row['candidate_id'] not in done
            and row['status'] not in ('rejected', 'deferred', 'committed')]

    def acknowledge(self, key, ids):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + key,)).fetchone()
            if not row: return
            receipt = json.loads(row['value'])
            receipt['processed_ids'] = sorted(set(receipt['processed_ids']) | (set(ids) & set(receipt['candidate_ids'])))
            db.execute('UPDATE authority_meta SET value=? WHERE key=?',
                (json.dumps(receipt, ensure_ascii=False, sort_keys=True), PREFIX + key))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally: db.close()

    def invalidate_scope(self, scope, current_key):
        """Keep old drafts visible while marking changed-source receipts stale."""
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT key,value FROM authority_meta WHERE key LIKE ? AND json_extract(value,'$.scope')=?",
                (PREFIX + '%', scope)).fetchall()
            for row in rows:
                receipt = json.loads(row['value'])
                if receipt['key'] != current_key and not receipt.get('stale'):
                    receipt['stale'] = True
                    db.execute('UPDATE authority_meta SET value=? WHERE key=?',
                        (json.dumps(receipt, ensure_ascii=False, sort_keys=True), row['key']))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally: db.close()

    def counts(self, receipt):
        db = self.authority._connect()
        try:
            rows = db.execute('SELECT c.candidate_id,c.status FROM json_each(?) i JOIN candidates c ON c.candidate_id=i.value',
                (json.dumps(receipt['candidate_ids']),)).fetchall()
        finally: db.close()
        done = set(receipt['processed_ids'])
        return {'generated': receipt['generated_count'], 'validated': receipt['validated_count'],
            'queued': sum(row['candidate_id'] not in done and row['status'] not in ('rejected', 'deferred', 'committed') for row in rows),
            'processed': len(receipt['processed_ids']),
            'adopted': sum(row['status'] == 'committed' for row in rows),
            'rejected': sum(row['status'] == 'rejected' for row in rows),
            'deferred': sum(row['status'] == 'deferred' for row in rows)}
