"""Durable preparation receipts referencing the existing Authority candidates.

Only identifiers and counters live in authority_meta. All generated text,
semantic documents, revisions and owner decisions stay in candidates.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import uuid

from memory_authority import CandidateNotFound, IdempotencyConflict

PREFIX = 'memory_candidate_preparation:'


class MemoryCandidatePreparation:
    def __init__(self, authority): self.authority = authority

    def get(self, key): return self.authority.get_meta(PREFIX + key)

    def _complete_unedited_l1_draft(self, db, row, proposal, policy):
        """A corrected first pass can complete its untouched preparation draft.

        Keep every old revision. Never upgrade owner-edited/decided drafts,
        formal memories or operations already in progress by regeneration.
        The surrounding preparation transaction owns the candidate revision.
        """
        current = self.authority._candidate_row(row)
        previous = current['proposal'].get('metadata') or {}
        if (current['status'] not in {'pending', 'accepted', 'commit_failed'}
                or (previous.get('legacy_candidate') or {}).get('l1_state') != 'preparation_required'
                or previous.get('memory_l1') or not isinstance(proposal.metadata.get('memory_l1'), dict)):
            return False
        cid = proposal.proposal_id
        if db.execute("SELECT 1 FROM candidate_revisions WHERE candidate_id=? "
                "AND actor NOT IN ('policy','commit_service','l1_preparation_recovery') LIMIT 1", (cid,)).fetchone():
            return False
        if (db.execute('SELECT 1 FROM memories WHERE memory_id=?', (cid,)).fetchone()
                or db.execute('SELECT 1 FROM commit_log WHERE aggregate_id=? LIMIT 1', (cid,)).fetchone()):
            return False
        # The preparation receipt already checked the full source fingerprint.
        # Preserve actor/source type and mode instead of turning regeneration
        # into a different authority or a different approval policy.
        if any(current['proposal'].get(k) != proposal.payload().get(k)
               for k in ('source_type', 'requested_mode', 'owner_explicit')):
            return False
        decision = policy.evaluate(proposal)
        status = {'auto_accept': 'accepted', 'queue_review': 'pending', 'reject': 'rejected'}.get(decision.action)
        if status is None:
            return False
        payload, stamp = proposal.payload(), datetime.now(timezone.utc).isoformat(timespec='seconds')
        sha, revision = self.authority.fingerprint(payload), current['revision'] + 1
        encode = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        result = {**current, 'revision': revision, 'status': status, 'proposal': payload,
                  'proposal_sha256': sha, 'policy': decision.payload(), 'updated_at': stamp}
        db.execute('UPDATE candidates SET revision=?,status=?,proposal_sha256=?,proposal_json=?,policy_json=?,updated_at=? WHERE candidate_id=?',
            (revision, status, sha, encode(payload), encode(decision.payload()), stamp, cid))
        self.authority._index_candidate_sources(db, payload)
        db.execute('INSERT INTO candidate_revisions(candidate_id,revision,status,proposal_sha256,proposal_json,policy_json,actor,created_at) VALUES(?,?,?,?,?,?,?,?)',
            (cid, revision, status, sha, encode(payload), encode(decision.payload()), 'l1_preparation_recovery', stamp))
        db.execute('INSERT INTO candidate_decisions(decision_id,candidate_id,from_status,to_status,candidate_revision,request_id,fingerprint,actor,reason_codes_json,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (str(uuid.uuid4()), cid, current['status'], status, revision, '', sha,
             'l1_preparation_recovery', encode(['FIRST_PASS_PREPARATION_COMPLETED']), encode(result), stamp))
        return True

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
            completed_drafts = set()
            for proposal in proposals:
                proposal = replace(proposal, metadata={**proposal.metadata, 'preparation_key': key})
                # An existing owner revision/decision always wins. A completed
                # receipt normally prevents reaching this recovery path.
                row = db.execute('SELECT * FROM candidates WHERE candidate_id=?', (proposal.proposal_id,)).fetchone()
                if not row:
                    self.authority._put_candidate(db, proposal, policy.evaluate(proposal))
                elif replace_failure and self._complete_unedited_l1_draft(db, row, proposal, policy):
                    completed_drafts.add(proposal.proposal_id)
                ids.append(proposal.proposal_id)
            unique_ids = list(dict.fromkeys(ids))
            # A protected owner revision may still lack L1. Do not turn a
            # regeneration that could not replace it into a successful batch.
            incomplete = []
            for cid in unique_ids:
                current = self.authority._candidate_row(db.execute(
                    'SELECT * FROM candidates WHERE candidate_id=?', (cid,)).fetchone())
                metadata = (current or {}).get('proposal', {}).get('metadata', {})
                if (current and current['status'] not in {'committed', 'rejected', 'deferred'}
                        and (metadata.get('legacy_candidate') or {}).get('l1_state') == 'preparation_required'
                        and not isinstance(metadata.get('memory_l1'), dict)):
                    incomplete.append(cid)
            if incomplete and issue is None:
                issue = {'code': 'first_pass_document_required', 'pending_candidate_ids': incomplete}
            receipt = {'key': key, 'scope': scope, 'source_fingerprint': source_fingerprint,
                'candidate_ids': unique_ids, 'processed_ids': [cid for cid in receipt['processed_ids']
                    if cid not in completed_drafts] if receipt else [], 'stale': False,
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
