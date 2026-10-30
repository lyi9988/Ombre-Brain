"""Grounded later understanding, using the original queue, policy and L1 writer.

An emotional reaction is not a memory revision. This queue contains explicit
durable proposals and their real sources, never a second state calculation.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json

from b1_protocol import digest
from memory_authority import MemoryProposal, IdempotencyConflict, RevisionConflict
from memory_chat_preparation import ChatMemoryPreparation, issue
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_service import L1OwnerError

KINDS = {'correction', 'new_experience', 'understanding', 'question'}
VALUES = {'explicit_owner_correction', 'explicit_remember', 'specific_future_use', 'none'}


def validate_submission(payload):
    if isinstance(payload, dict) and payload.get('schema') == 'self-review-followup-v1':
        from memory_self_review_source import validate_native_submission
        return validate_native_submission(payload)
    from persona_memory_reaction_context import validate_context, memory_ref
    from persona_inner_state import evidence_spans
    if not isinstance(payload, dict) or set(payload) != {'context', 'turn', 'updates', 'sources'}:
        raise L1OwnerError('invalid_followup_submission')
    context, turn = validate_context(payload['context']), payload['turn']
    if not isinstance(turn, dict) or set(turn) != {'conversation_id', 'user_event', 'assistant_event'}:
        raise L1OwnerError('canonical_turn_required')
    u, a = turn['user_event'], turn['assistant_event']
    source = {'conversation_key': digest(turn['conversation_id']), 'user_id': u['id'],
              'assistant_id': a['id'], 'user': u['content'], 'assistant': a['content']}
    if context['scope'] != {'conversation_id': turn['conversation_id'],
            'user_source_event_id': u['source_event_id'], 'assistant_source_event_id': a['source_event_id']}:
        raise L1OwnerError('followup_turn_mismatch')
    sources = payload['sources']
    if not isinstance(sources, list) or len(sources) != 2:
        raise L1OwnerError('canonical_sources_required')
    for item, event, role in zip(sources, (u, a), ('user', 'assistant')):
        if (not isinstance(item, dict) or set(item) != {'event_id', 'version_id', 'role', 'text', 'created_at'}
                or item['event_id'] != event['id'] or item['role'] != role or item['text'] != event['content']
                or any(not isinstance(item[k], str) or not item[k] for k in item)):
            raise L1OwnerError('canonical_source_identity_mismatch')
        if datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')).tzinfo is None:
            raise L1OwnerError('canonical_source_time_required')
    updates = payload['updates']
    if not isinstance(updates, list) or len(updates) > 32:
        raise L1OwnerError('bounded_followups_required')
    fragments = {f['fragment_id']: f for f in context['fragments']}
    result, seen = [], set()
    for row in updates:
        if (not isinstance(row, dict) or set(row) != {'update_id', 'memory_ref', 'kind', 'long_term_value', 'content', 'evidence_spans'}
                or row['kind'] not in KINDS or row['long_term_value'] not in VALUES
                or not isinstance(row['content'], str) or not row['content'].strip()):
            raise L1OwnerError('invalid_followup_observation')
        fragment = fragments.get(row['memory_ref'].get('fragment_id'))
        if not fragment or row['memory_ref'] != memory_ref(fragment):
            raise L1OwnerError('followup_memory_not_delivered')
        spans = evidence_spans(row['evidence_spans'])
        for span in spans:
            text = source[span['role']]
            if span['end'] > len(text) or digest(text[span['start']:span['end']]) != span['evidence_hash']:
                raise L1OwnerError('followup_current_evidence_mismatch')
        semantic = {k: v for k, v in row.items() if k != 'update_id'}
        if row['update_id'] != 'ml3:' + digest({'source_hash': digest(source), **semantic}):
            raise L1OwnerError('followup_identity_mismatch')
        if row['long_term_value'] in {'explicit_owner_correction', 'explicit_remember'} and not any(s['role'] == 'user' for s in spans):
            raise L1OwnerError('followup_owner_evidence_missing')
        if row['long_term_value'] != 'none' and row['update_id'] not in seen:
            result.append(deepcopy(row)); seen.add(row['update_id'])
    return result


class FollowupTasks(ChatMemoryPreparation):
    prefix = 'memory_l3_followup:'

    def owns_claim(self, claimed):
        current = self.get(claimed['task_id'])
        return bool(current and current['revision'] == claimed['revision']
            and current.get('claim_token') == claimed.get('claim_token') and current['phase'] == 'generating')

    def publish(self, claimed, proposal, policy, *, generation, model_io_id):
        """Publish one exact draft and its task checkpoint in one Authority CAS."""
        with closing(self.authority._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            stored = db.execute('SELECT value FROM authority_meta WHERE key=?', (self.prefix + claimed['task_id'],)).fetchone()
            current = json.loads(stored[0]) if stored else None
            if (not current or current['revision'] != claimed['revision'] or current['phase'] != 'generating'
                    or current.get('claim_token') != claimed.get('claim_token')):
                raise L1OwnerError('task_claim_changed', status=409, field_path='claim_token')
            prior = self.authority._candidate_row(db.execute('SELECT * FROM candidates WHERE candidate_id=?',
                (claimed['task_id'],)).fetchone())
            if prior and current.get('draft_content') is not None:
                if prior['revision'] != current.get('edited_candidate_revision'):
                    raise L1OwnerError('followup_draft_candidate_changed', status=409)
                candidate = self.authority._revise_candidate(db, prior['candidate_id'], expected_revision=prior['revision'],
                    proposal=proposal, request_id='followup-draft:' + claimed['task_id'] + ':' + str(current['draft_revision']), actor='owner')
            else:
                candidate = prior or self.authority._put_candidate(db, proposal, policy)
            current.update(revision=current['revision'] + 1, candidate_ids=[candidate['candidate_id']], draft_content=None,
                publication={'draft_revision': current['draft_revision'], 'candidate_revision': candidate['revision']},
                generation=generation, model_io_id=model_io_id,
                updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(current), self.prefix + claimed['task_id']))
            db.commit()
            return current, candidate

    @staticmethod
    def public(row):
        return {k: deepcopy(row.get(k)) for k in ('task_id', 'memory_id', 'phase', 'revision', 'created_at',
            'updated_at', 'candidate_ids', 'issue', 'attempts', 'receipt', 'generation', 'related_candidates')}

    def enqueue_update(self, payload, update):
        task_id = 'followup_' + update['update_id'].split(':', 1)[1]
        # Different unrelated updates in a replayed batch do not change this task.
        envelope = {**deepcopy(payload), 'updates': [deepcopy(update)]}
        fingerprint = semantic_sha256(envelope)
        with closing(self.authority._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT value FROM authority_meta WHERE key=?', (self.prefix + task_id,)).fetchone()
            if prior:
                if json.loads(prior[0])['submission_hash'] != fingerprint:
                    raise IdempotencyConflict('followup submission changed')
            else:
                stamp = datetime.now(timezone.utc).isoformat()
                row = {'task_id': task_id, 'operation_id': task_id, 'memory_id': update['memory_ref']['memory_id'],
                    'phase': 'queued', 'revision': 1, 'draft_revision': 1, 'submission_hash': fingerprint,
                    'envelope': envelope, 'created_at': stamp, 'updated_at': stamp, 'candidate_ids': [],
                    'attempts': 0, 'issue': None, 'lease_until': 0, 'history': [], 'edit_requests': {}}
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (self.prefix + task_id, canonical_json(row)))
            db.commit()
        return self.public(self.get(task_id))

    def retry(self, memory_id, task_id, payload):
        if (not isinstance(payload, dict) or set(payload) != {'expected_revision', 'request_id'}
                or type(payload['expected_revision']) is not int or payload['expected_revision'] < 1
                or not isinstance(payload['request_id'], str) or not 1 <= len(payload['request_id']) <= 128):
            raise L1OwnerError('invalid_followup_retry')
        row = self.get(task_id)
        if row is None or row['memory_id'] != memory_id:
            raise L1OwnerError('followup_not_found', status=404)
        fingerprint = semantic_sha256(payload)
        if row['edit_requests'].get(payload['request_id']):
            if row['edit_requests'][payload['request_id']] != fingerprint:
                raise IdempotencyConflict('followup retry changed')
            return {'status': 'ok', 'memory_id': memory_id, 'task': self.public(row), 'replayed': True}
        def change(current):
            if current['phase'] != 'failed':
                raise L1OwnerError('followup_not_failed', status=409)
            current.update(phase='queued', lease_until=0, claim_token=None, issue=None)
            current['edit_requests'][payload['request_id']] = fingerprint
            return current
        updated = self.update(task_id, expected_revision=payload['expected_revision'], transform=change)
        return {'status': 'ok', 'memory_id': memory_id, 'task': self.public(updated), 'replayed': False}


class MemoryFollowups:
    def __init__(self, engine, bucket_mgr, resolver=None, *, review_source_reader=None):
        self.engine, self.bucket_mgr = engine, bucket_mgr
        self.resolver = resolver or getattr(engine, 'memory_followup_resolver', None)
        self.review_source_reader = review_source_reader or getattr(engine, 'memory_review_source_reader', None)
        self.authority = engine.memory_authority_store
        self.tasks = FollowupTasks(self.authority)
        self.revisions = engine._memory_l1_revisions(bucket_mgr)

    async def submit(self, payload):
        updates = validate_submission(payload)
        # Durable recovery is keyed before a later source change. Already
        # recorded submissions return their receipt; they are not new readings.
        missing = [u for u in updates if self.tasks.get('followup_' + u['update_id'].split(':', 1)[1]) is None]
        qualified = set()
        native = payload.get('schema') == 'self-review-followup-v1'
        native_qualified = set()
        if missing:
            if self.resolver is None:
                raise L1OwnerError('memory_delivery_lookup_unconfigured', status=503)
            validation = await self.resolver(payload['context'], {'scope': payload['context']['scope']} if native else payload['turn'])
            if validation.get('state') != 'available':
                raise L1OwnerError(validation.get('reason', 'memory_delivery_unavailable'), status=503)
            qualified = set(validation.get('eligible_fragment_ids', []))
            if native:
                from memory_self_review_source import check_native_sources
                native_qualified = await check_native_sources(payload, self.engine.config, self.review_source_reader)
            for update in missing:
                fid = update['memory_ref']['fragment_id']
                reason = validation.get('reasons', {}).get(fid, '')
                if fid not in qualified and reason not in {'memory_fragment_withdrawn', 'memory_fragment_changed',
                        'memory_not_current_or_eligible', 'legacy_memory_body_changed_or_unknown',
                        'memory_source_withdrawn_or_changed', 'memory_presentation_changed'}:
                    raise L1OwnerError(reason or 'memory_delivery_source_check_failed', status=503)
        receipts, unavailable = [], []
        for update in updates:
            task_id = 'followup_' + update['update_id'].split(':', 1)[1]
            if self.tasks.get(task_id) is None and (update['memory_ref']['fragment_id'] not in qualified
                    or native and update['update_id'] not in native_qualified):
                unavailable.append(update['update_id'])
                stored = self.tasks.enqueue_update(payload, update)
                reason = ('review_or_feedback_source_changed' if native and update['update_id'] not in native_qualified else
                          validation.get('reasons', {}).get(update['memory_ref']['fragment_id']))
                def unavailable_task(current):
                    current.update(phase='source_changed', issue={'code': reason})
                    return current
                self.tasks.update(task_id, expected_revision=stored['revision'], transform=unavailable_task)
                receipts.append(self.tasks.public(self.tasks.get(task_id)))
                continue
            receipts.append(self.tasks.enqueue_update(payload, update))
        return {'status': 'recorded', 'tasks': receipts, 'unavailable_updates': unavailable}

    async def sources(self, task):
        if task['envelope'].get('schema') == 'self-review-followup-v1':
            from memory_self_review_source import native_sources
            return await native_sources(self, task)
        from memory_source_provenance import canonical_source
        from memory_l1_ingestion import bind_sources
        envelope, update = task['envelope'], task['envelope']['updates'][0]
        roles = {span['role'] for span in update['evidence_spans']}
        sources = [canonical_source(source, conversation_id=envelope['turn']['conversation_id'],
            profile_id=self.engine.memory_semantic_profile_id) for source in envelope['sources'] if source['role'] in roles]
        snapshots, bindings = bind_sources(sources, config=self.engine.config)
        reader = self.engine._memory_l1_sources()
        evidence = []
        for source, binding in zip(snapshots, bindings):
            binding['binding_id'] = 'source:followup:' + semantic_sha256({k: binding[k] for k in
                ('source_namespace', 'source_id', 'source_revision')})[:24]
            # Retaining is not adoption: the live reader still independently
            # checks withdrawal/current version using this upstream locator.
            reader.retain_copy(source)
            observed = await reader.locate(binding)
            if observed.get('eligible_source') is not True or observed.get('text') != source['text']:
                raise L1OwnerError('followup_source_not_currently_readable', status=409, field_path='sources')
            evidence.append({'source': source, 'binding': binding})
        return sources, evidence, reader

    async def _check_native_delivery(self, task):
        payload = task['envelope']
        if payload.get('schema') != 'self-review-followup-v1':
            return
        if self.resolver is None:
            raise L1OwnerError('memory_delivery_lookup_unconfigured', status=503)
        validation = await self.resolver(payload['context'], {'scope': payload['context']['scope']})
        fragment = payload['updates'][0]['memory_ref']['fragment_id']
        if validation.get('state') != 'available' or fragment not in validation.get('eligible_fragment_ids', []):
            raise L1OwnerError(validation.get('reasons', {}).get(fragment) or validation.get('reason')
                               or 'followup_memory_not_currently_delivered', status=409)

    async def prepare(self, claimed):
        from memory_l3_sync import prepare_plan, compile_plan
        from memory_semantics import build_annotations, attach_source_links, binding_source_keys
        update = deepcopy(claimed['envelope']['updates'][0])
        if claimed.get('draft_content') is not None:
            update['content'] = claimed['draft_content']
        if self.engine.memory_ingestion_policy.source_modes.get('daily_chat_memory', self.engine.daily_chat_memory_mode) == 'off':
            return {'phase': 'disabled', 'issue': {'code': 'REJECT_SOURCE_DISABLED'}}
        await self._check_native_delivery(claimed)
        mid = claimed['memory_id']
        detail = await self.revisions.detail(mid)
        expected = update['memory_ref']['l1_revision']
        if expected is None:
            expected = next(f['coverage'].get('legacy_authority_revision') for f in claimed['envelope']['context']['fragments']
                            if f['fragment_id'] == update['memory_ref']['fragment_id'])
        if detail['current_revision'] != expected or detail['memory']['state'] != 'active' or detail['l1']['state'] != 'current':
            raise L1OwnerError('followup_target_changed', status=409)
        sources, evidence, reader = await self.sources(claimed)
        # Original current sources remain prerequisites of the complete L1.
        for binding in detail['l1']['document']['evidence_bindings']:
            if not (await reader.locate(binding)).get('eligible_source'):
                raise L1OwnerError('followup_original_source_unavailable', status=409)
        related = self.authority.find_candidates_by_sources([s['source_key'] for s in sources]
            + binding_source_keys([row['binding'] for row in evidence]))
        # Exact content + exact source set is reusable; merely sharing a turn
        # is only a link and cannot discard another distinct memory.
        from memory_semantics import source_keys
        keys = {s['source_key'] for s in sources}
        duplicate = next((r for r in related if r['candidate_id'] != claimed['task_id']
            and r['proposal']['proposed_body'].strip() == update['content'].strip()
            and set(source_keys(r['proposal'])) == keys and r['status'] not in {'rejected', 'commit_failed'}
            and r['proposal'].get('metadata', {}).get('l3_followup', {}).get('memory_id') == mid), None)
        if duplicate:
            return {'phase': 'linked_existing', 'candidate_ids': [duplicate['candidate_id']],
                'related_candidates': [duplicate['candidate_id']], 'receipt': {'state': 'linked_existing',
                'candidate_id': duplicate['candidate_id'], 'target_memory_unchanged': True}}
        material = {'l1_document': detail['l1']['document'], 'followup_kind': update['kind'],
            'durable_reason': update['long_term_value'], 'related_candidates': [
                {'candidate_id': r['candidate_id'], 'content': r['proposal']['proposed_body'], 'status': r['status']} for r in related]}
        if claimed.get('draft_fields'):
            material['owner_draft_fields'] = claimed['draft_fields']
            material['draft_is_source'] = False
        generated = await prepare_plan(self.engine, material, update['content'], 'followup', evidence,
            memory_id=mid, task_id=claimed['task_id'], attempt=claimed['attempts'])
        plan = generated['plan']
        if plan.get('confidence') is not None and (type(plan['confidence']) not in (int, float) or not 0 <= plan['confidence'] <= 1):
            raise L1OwnerError('invalid_followup_confidence')
        if plan['decision'] != 'revise_l1':
            return {'phase': 'already_recorded' if plan['decision'] == 'l1_already_correct' else 'needs_clarification',
                'plan': plan, 'generation': generated['generation'], 'model_io_id': generated['model_io_id']}
        compiled = compile_plan(self.revisions, mid, detail, plan, evidence, request_id=claimed['task_id'])
        new_refs = {row['binding']['binding_id'] for row in evidence}
        if any(not new_refs.intersection(patch['evidence_refs']) for patch in plan['patch']):
            raise L1OwnerError('followup_new_evidence_required', field_path='plan.patch')
        self._check_cognition(update, detail['l1']['document'], compiled['document'], evidence)
        annotations = build_annotations(body=update['content'], proposed=None, sources=sources, proposal_id=claimed['task_id'])
        annotations = attach_source_links(annotations, related, proposal_id=claimed['task_id'])
        draft_fields = claimed.get('draft_fields') or {}
        memory_type = draft_fields.get('kind') or {
            'correction': 'durable_fact', 'new_experience': 'shared_experience',
            'understanding': 'reflection', 'question': 'reflection'}[update['kind']]
        legacy = {'id': claimed['task_id'], 'title': '后续理解与更正', 'content': update['content'],
            'kind': memory_type, 'mode': self.engine.daily_chat_memory_mode, 'source_verification': 'verified',
            'reason': plan['reason'], 'source_event_ids': [s['event_id'] for s in sources]}
        legacy.update({k: deepcopy(v) for k, v in draft_fields.items() if k != 'content'})
        proposal = MemoryProposal.from_mapping({'proposal_id': claimed['task_id'], 'source_type': 'daily_chat_memory',
            'proposed_body': update['content'], 'original_excerpt': '\n'.join(s['text'] for s in sources).replace('<', '‹').replace('>', '›'),
            'source_refs': [s['namespace'] + ':' + s['event_id'] + '@' + s['version_id'] for s in sources],
            'source_status': 'verified', 'memory_type': memory_type,
            'confidence': draft_fields.get('confidence', plan.get('confidence')), 'requested_mode': self.engine.daily_chat_memory_mode,
            'owner_explicit': False, 'metadata': {'semantic_annotations': annotations,
                'l3_followup': {'task_id': claimed['task_id'], 'memory_id': mid, 'expected_revision': expected,
                    'kind': update['kind'], 'plan': plan, 'compiled': compiled, 'generation': generated['generation']},
                'legacy_candidate': legacy}})
        published, candidate = self.tasks.publish(claimed, proposal, self.engine.memory_ingestion_policy.evaluate(proposal),
            generation=generated['generation'], model_io_id=generated['model_io_id'])
        claimed.clear()
        claimed.update(published)
        # Never replace a prepared or owner-edited candidate on queue replay.
        receipt = await self.adopt(candidate) if candidate['status'] == 'accepted' else None
        return {'phase': 'prepared', 'candidate_ids': [candidate['candidate_id']], 'receipt': receipt,
            'draft_content': None,
            'generation': generated['generation'], 'model_io_id': generated['model_io_id'],
            'related_candidates': [r['candidate_id'] for r in related]}

    @staticmethod
    def _check_cognition(update, previous, document, evidence):
        if update['kind'] not in {'understanding', 'question'}:
            return
        refs = {e['binding']['binding_id'] for e in evidence}
        touched = [s for s in document['statements'] if refs.intersection(s['evidence_refs'])]
        if not touched:
            raise L1OwnerError('followup_understanding_statement_required')
        for statement in touched:
            cognition = statement.get('cognition') or {}
            role = (cognition.get('role') or {}).get('value')
            if role not in ({'question'} if update['kind'] == 'question' else {'insight', 'view', 'hypothesis', 'question'}):
                raise L1OwnerError('followup_cognition_required', field_path='cognition')
            if role in {'hypothesis', 'question'} and (cognition.get('epistemic_status') or {}).get('value') not in {'tentative', 'contested'}:
                raise L1OwnerError('followup_uncertainty_required', field_path='cognition.epistemic_status')

    async def adopt(self, row, *, owner=False, edit=None, request_id=None):
        from memory_l3_sync import commit_plan
        meta = row['proposal'].get('metadata', {}).get('l3_followup')
        if not meta:
            raise L1OwnerError('not_followup_candidate')
        cid = row['candidate_id']
        if row['status'] == 'committed':
            return {'id': cid, 'status': 'exists', 'memory_id': meta['memory_id']}
        if edit:
            return self.edit_draft(row, edit, request_id=request_id)
        queued = self.tasks.get(meta['task_id'])
        if meta.get('preparation_state') == 'queued' or (queued and queued.get('draft_content') is not None):
            return {'id': cid, 'status': 'preparing', 'memory_id': meta['memory_id'], 'task_id': queued['task_id']}
        if queued and queued['envelope'].get('schema') == 'self-review-followup-v1':
            await self._check_native_delivery(queued)
            from memory_self_review_source import check_native_sources
            allowed = await check_native_sources(queued['envelope'], self.engine.config, self.review_source_reader)
            if queued['envelope']['updates'][0]['update_id'] not in allowed:
                raise L1OwnerError('followup_source_not_currently_readable', status=409, field_path='sources')
        if owner and row['status'] in {'pending', 'deferred', 'commit_failed'}:
            row = self.authority.decide_candidate(cid, action='accept', expected_revision=row['revision'],
                request_id=request_id + ':accept', actor='owner', reason_codes=['OWNER_ACCEPTED'])
        if row['status'] != 'accepted':
            return {'id': cid, 'status': row['status'], 'memory_id': meta['memory_id']}
        task = self.tasks.get(meta['task_id'])
        if not task:
            raise L1OwnerError('followup_source_task_missing')
        source = meta['compiled']['source_bindings'][0]
        context = {'origin': 'followup', 'proposal_id': cid, 'plan_sha256': semantic_sha256(meta['plan']),
            'source_ref': {k: source[k] for k in ('source_namespace', 'source_id', 'source_revision')},
            'candidate_revision': row['revision'], 'policy': row['policy']}
        receipt = await commit_plan(self.revisions, meta['memory_id'], meta['compiled'],
            expected_revision=meta['expected_revision'], request_id=meta['task_id'], actor='memory_followup',
            decision_source='followup_policy', program_context=context, source_reader=self.engine._memory_l1_sources(),
            edit_note=meta['plan']['reason'])
        current = self.authority.get_candidate(cid)
        if current['status'] == 'accepted':
            self.authority.decide_candidate(cid, action='commit', expected_revision=current['revision'],
                request_id=cid + ':committed', actor='commit_service', reason_codes=['TARGET_L1_REVISION_COMMITTED'])
        return {'id': cid, 'status': 'created', 'memory_id': meta['memory_id'], 'receipt': receipt}

    def edit_draft(self, row, edit, *, request_id, expected_revision=None):
        allowed = {'content', 'title', 'reason', 'kind', 'domain', 'tags', 'event_date', 'importance', 'confidence'}
        if (not isinstance(edit, dict) or set(edit) - allowed or 'content' not in edit
                or not isinstance(edit['content'], str) or not edit['content'].strip()):
            raise L1OwnerError('invalid_followup_draft_edit', field_path='edit')
        for key, value in edit.items():
            if key in {'tags', 'domain'}:
                valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
            elif key in {'importance', 'confidence'}:
                valid = type(value) in (int, float) and (0 <= value <= 1 if key == 'confidence' else 1 <= value <= 10)
            else:
                valid = isinstance(value, str)
            if not valid:
                raise L1OwnerError('invalid_followup_draft_field', field_path='edit.' + key)
        meta = row['proposal']['metadata']['l3_followup']
        expected = row['revision'] if expected_revision is None else expected_revision
        if type(expected) is not int or expected < 1 or not isinstance(request_id, str) or not request_id:
            raise L1OwnerError('invalid_followup_draft_request')
        fingerprint = semantic_sha256({'edit': edit, 'request_id': request_id, 'candidate_revision': expected})
        with closing(self.authority._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            stored = db.execute('SELECT value FROM authority_meta WHERE key=?', (self.tasks.prefix + meta['task_id'],)).fetchone()
            if not stored:
                raise L1OwnerError('followup_source_task_missing')
            task = json.loads(stored[0])
            candidate = self.authority._candidate_row(db.execute('SELECT * FROM candidates WHERE candidate_id=?', (row['candidate_id'],)).fetchone())
            prior = task['edit_requests'].get(request_id)
            if prior and prior != fingerprint:
                raise IdempotencyConflict('followup draft edit changed')
            if not prior:
                if not candidate or candidate['revision'] != expected:
                    raise RevisionConflict('followup owner draft candidate changed')
                if candidate['status'] not in {'pending', 'deferred', 'commit_failed'}:
                    raise L1OwnerError('followup_adoption_already_started', status=409)
                if db.execute('SELECT 1 FROM commit_log WHERE idempotency_key=?',
                        ('program-l1:followup:' + meta['memory_id'] + ':' + task['task_id'],)).fetchone():
                    raise L1OwnerError('followup_original_commit_requires_recovery', status=409)
                fields = {**deepcopy(task.get('draft_fields') or {}), **deepcopy(edit)}
                draft = deepcopy(candidate['proposal'])
                draft['proposed_body'] = edit['content']
                if 'kind' in fields: draft['memory_type'] = fields['kind']
                if 'confidence' in fields: draft['confidence'] = fields['confidence']
                draft['metadata']['legacy_candidate'].update(fields)
                draft['metadata']['l3_followup'].update(preparation_state='queued', plan=None, compiled=None)
                saved = self.authority._revise_candidate(db, candidate['candidate_id'], expected_revision=expected,
                    proposal=MemoryProposal.from_mapping(draft),
                    request_id='followup-owner-draft:' + candidate['candidate_id'] + ':' + request_id, actor='owner')
                task['history'].append({'content': task.get('draft_content') or candidate['proposal']['proposed_body'],
                    'draft_fields': deepcopy(task.get('draft_fields')), 'draft_revision': task['draft_revision'], 'changed_by': 'owner'})
                task.update(draft_content=edit['content'], edited_candidate_revision=saved['revision'], draft_fields=fields,
                    draft_revision=task['draft_revision'] + 1, revision=task['revision'] + 1,
                    phase='queued', lease_until=0, claim_token=None, issue=None,
                    updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
                task['edit_requests'][request_id] = fingerprint
                db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(task), self.tasks.prefix + task['task_id']))
                candidate = saved
            db.commit()
        return {'id': row['candidate_id'], 'status': 'preparing', 'memory_id': task['memory_id'],
            'task_id': task['task_id'], 'draft_preserved': True, 'candidate': candidate}


async def run_followup_tasks(engine, bucket_mgr, *, max_jobs):
    service = MemoryFollowups(engine, bucket_mgr)
    results = []
    for pending in service.tasks.list(limit=max_jobs, phase='queued', private=True):
        try:
            claimed = service.tasks.claim(pending, lease_seconds=engine.daily_chat_memory_timeout_seconds + 60)
        except (L1OwnerError, RevisionConflict):
            continue  # Another worker claimed the same durable task.
        try:
            existing = service.authority.get_candidate(claimed['task_id'])
            if existing and claimed.get('draft_content') is None:
                receipt = await service.adopt(existing) if existing['status'] == 'accepted' else None
                result = {'phase': 'prepared', 'candidate_ids': [existing['candidate_id']], 'receipt': receipt}
            else:
                result = await service.prepare(claimed)
            row = service.tasks.finish(claimed, **result)
            results.append(service.tasks.public(row))
        except Exception as exc:
            if not service.tasks.owns_claim(claimed):
                continue
            row = service.tasks.finish(claimed, phase='failed', issue=issue(exc))
            results.append(service.tasks.public(row))
    return {'status': 'completed', 'results': results}
