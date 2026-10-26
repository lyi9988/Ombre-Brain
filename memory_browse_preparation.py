"""Browse drafts on the shared durable preparation and L1 adoption path."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import time

from memory_authority import IdempotencyConflict, MemoryProposal, PolicyDecision
from memory_chat_preparation import ChatMemoryPreparation, issue
from memory_l1_contract import canonical_json, canonical_l1_body, semantic_sha256
from memory_l1_service import L1OwnerError

PREFIX = 'memory_l1_browse_task:'


class BrowseMemoryPreparation(ChatMemoryPreparation):
    prefix = PREFIX

    @staticmethod
    def public(row):
        return {**ChatMemoryPreparation.public(row), 'source_kind': 'web_browse'}

    def enqueue(self, *, operation_key, content, descriptor, metadata):
        if not operation_key or not operation_key.strip():
            raise L1OwnerError('browse_operation_identity_required', field_path='idempotency_key')
        from memory_hold_source import hold_source_descriptor
        checked = hold_source_descriptor(content, descriptor)['descriptor']
        memory_id = 'memory_' + hashlib.sha256(operation_key.encode()).hexdigest()[:16]
        envelope = {'content': content, 'descriptor': checked, 'metadata': copy.deepcopy(metadata)}
        digest = semantic_sha256(envelope)
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT value FROM authority_meta WHERE key=?', (self.prefix + memory_id,)).fetchone()
            # Old successful/uncertain candidates keep their identity. A new
            # preparation must never overwrite or retry an old body-only write.
            candidate = db.execute('SELECT candidate_id FROM candidates WHERE candidate_id=?', (memory_id,)).fetchone()
            memory = db.execute('SELECT memory_id FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
            old_commit = db.execute('SELECT operation_id FROM commit_log WHERE idempotency_key=?',
                (operation_key + ':revision',)).fetchone()
            if prior:
                if json.loads(prior['value'])['submission_hash'] != digest:
                    raise IdempotencyConflict('browse operation payload changed')
            elif not candidate and not memory and not old_commit:
                stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
                row = {'task_id': memory_id, 'operation_id': operation_key, 'phase': 'queued',
                    'draft_revision': 1, 'revision': 1, 'submission_hash': digest, 'envelope': envelope,
                    'created_at': stamp, 'updated_at': stamp, 'candidate_ids': [], 'attempts': 0,
                    'issue': None, 'lease_until': 0, 'history': [], 'edit_requests': {}}
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (self.prefix + memory_id, canonical_json(row)))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return self.receipt(memory_id)

    def receipt(self, memory_id):
        memory = self.authority.get_memory(memory_id)
        row = self.get(memory_id)
        candidate = self.authority.get_candidate(memory_id)
        active = bool(row and memory and memory.get('state') == 'active' and memory.get('active_revision', 0) > 0)
        return {'ok': True, 'task_id': memory_id, 'memory_id': memory_id if active else None,
            'operation_id': row['operation_id'] if row else None, 'adopted': active,
            'memory_status': 'committed' if active else candidate['status'] if candidate else 'preparing',
            'preparation_status': row['phase'] if row else 'legacy_receipt_requires_reconciliation',
            'issue': row.get('issue') if row else None, 'source_kind': 'web_browse'}

    def detail(self, memory_id):
        row = self.get(memory_id)
        if not row:
            raise L1OwnerError('task_not_found', status=404, field_path='task_id')
        return {'task': self.public(row), 'draft': copy.deepcopy(row['envelope']),
            'history': copy.deepcopy(row['history']), 'owner_edits': copy.deepcopy(row.get('owner_edits', [])),
            'receipt': self.receipt(memory_id), 'preparation_output': row.get('preparation_output')}

    def put_prepared(self, claimed, proposal, policy):
        """Publish the candidate and fence the owner's draft in one CAS."""
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            record = db.execute('SELECT value FROM authority_meta WHERE key=?', (self.prefix + claimed['task_id'],)).fetchone()
            current = json.loads(record['value']) if record else None
            if (current is None or current['revision'] != claimed['revision']
                    or current.get('claim_token') != claimed.get('claim_token') or current['phase'] != 'generating'):
                raise L1OwnerError('submission_revision_conflict', status=409, field_path='revision')
            candidate = self.authority._put_candidate(db, proposal, policy)
            current.update(revision=current['revision'] + 1, candidate_ids=[proposal.proposal_id],
                updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            db.execute('UPDATE authority_meta SET value=? WHERE key=?',
                (canonical_json(current), self.prefix + claimed['task_id']))
            db.commit()
            return current, candidate
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def edit(self, memory_id, payload):
        from jsonschema import Draft202012Validator
        from memory_l1_wire_schema import chat_submission_parameters
        if (not isinstance(payload, dict) or set(payload) != {'expected_revision', 'request_id', 'draft'}
                or type(payload['expected_revision']) is not int or not isinstance(payload['request_id'], str)
                or not 1 <= len(payload['request_id']) <= 128
                or list(Draft202012Validator(chat_submission_parameters()).iter_errors(payload['draft']))):
            raise L1OwnerError('invalid_submission_edit', field_path='edit')
        fingerprint = semantic_sha256(payload)
        original = self.get(memory_id)
        if original is None:
            raise L1OwnerError('task_not_found', status=404, field_path='task_id')
        prior = original['edit_requests'].get(payload['request_id'])
        if prior:
            if prior != fingerprint:
                raise IdempotencyConflict('edit request changed')
            return self.detail(memory_id)
        if self.authority.get_candidate(memory_id):
            raise L1OwnerError('edit_prepared_candidate_in_review', status=409, field_path='draft')
        def change(row):
            draft = payload['draft']['narrative']
            previous = copy.deepcopy(row['envelope'])
            row.setdefault('original_envelope', copy.deepcopy(previous))
            from_hash = hashlib.sha256(canonical_l1_body(previous['content']).encode()).hexdigest()
            to_hash = hashlib.sha256(canonical_l1_body(draft['body']).encode()).hexdigest()
            row['history'].append({'draft_revision': row['draft_revision'], 'draft': previous,
                'updated_at': row['updated_at'], 'issue': row.get('issue')})
            row.setdefault('owner_edits', []).append({'actor': 'owner', 'request_id': payload['request_id'],
                'from_revision': row['draft_revision'], 'to_revision': row['draft_revision'] + 1,
                'from_body_sha256': from_hash, 'to_body_sha256': to_hash})
            row['envelope']['content'] = draft['body']
            row['envelope']['metadata']['name'] = draft.get('title', '')
            row['envelope']['draft_arguments'] = copy.deepcopy(payload['draft'])
            row.update(draft_revision=row['draft_revision'] + 1, phase='queued', lease_until=0,
                issue=None, claim_token=None)
            row['edit_requests'][payload['request_id']] = fingerprint
            return row
        # Fence candidate existence and the draft update under the same write
        # lock as put_prepared, including when an edit started before publish.
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            record = db.execute('SELECT value FROM authority_meta WHERE key=?', (self.prefix + memory_id,)).fetchone()
            current = json.loads(record['value']) if record else None
            if current is None:
                raise L1OwnerError('task_not_found', status=404, field_path='task_id')
            prior = current['edit_requests'].get(payload['request_id'])
            if prior:
                if prior != fingerprint:
                    raise IdempotencyConflict('edit request changed')
            else:
                if current['revision'] != payload['expected_revision']:
                    raise L1OwnerError('submission_revision_conflict', status=409, field_path='revision')
                if db.execute('SELECT 1 FROM candidates WHERE candidate_id=?', (memory_id,)).fetchone():
                    raise L1OwnerError('edit_prepared_candidate_in_review', status=409, field_path='draft')
                current = change(current)
                current.update(revision=payload['expected_revision'] + 1,
                    updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
                db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(current), self.prefix + memory_id))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return self.detail(memory_id)


async def prepare_browse_tasks(engine, bucket_mgr, *, max_jobs=None):
    from memory_browse_source import model_source, semantic_views
    from memory_l1_ingestion import bind_sources, prepare_model_document
    from memory_l1_semantic_wire import assemble_document, semantic_contract
    from memory_l1_schema import split_document
    from memory_l1_runtime import task_options
    from memory_l1_source_readers import AizBrowseSourceReader
    from memory_model_io import MemoryModelIO, completion_parameters
    from memory_source_identity import source_model_identity
    options = task_options(engine.config)
    allowance = options['chat_memory_jobs_per_run'] if max_jobs is None else min(options['chat_memory_jobs_per_run'], max_jobs)
    if not engine.memory_authority_enabled or not engine.memory_authority_store or allowance <= 0:
        return {'status': 'disabled'}
    if getattr(engine, '_browse_preparation_running', False):
        return {'status': 'locked'}
    prompt, prompt_context = engine._memory_l1_prompt('ombre.chat_memory_prepare_prompt')
    if not prompt.strip():
        return {'status': 'disabled', 'reason': 'prompt_disabled'}
    route = engine._memory_task_route('chat_l1')
    if not route['configured']:
        return {'status': 'not_prepared', 'reason': route['unavailable_reason']}
    tasks, io = BrowseMemoryPreparation(engine.memory_authority_store), MemoryModelIO(engine.memory_authority_store)
    engine._browse_preparation_running = True
    results = []
    try:
        for pending in tasks.list(limit=allowance, phase='queued', private=True):
            try:
                claimed = tasks.claim(pending, lease_seconds=engine.daily_chat_memory_timeout_seconds + 60)
            except L1OwnerError:
                continue
            io_id = None
            try:
                started = time.monotonic()
                memory_id, envelope = claimed['task_id'], claimed['envelope']
                existing = engine.memory_authority_store.get_candidate(memory_id)
                if existing:
                    if existing['status'] == 'accepted':
                        await engine._memory_l1_adopt_candidate(existing, bucket_mgr, request_id='browse-task:' + memory_id)
                    tasks.finish(claimed, phase='prepared', candidate_ids=[memory_id])
                    results.append(tasks.receipt(memory_id))
                    continue
                descriptor = envelope['descriptor']
                values = [await AizBrowseSourceReader(engine.config, view).capture(descriptor['source_id'])
                    for view in ('article', 'reading')]
                if any(v.get('state') != 'available' or v.get('eligible_source') is not True
                        or v.get('content_revision') != descriptor['source_scope']['source_ref']['content_hash']
                        or v.get('source_event_id') != descriptor['source_scope']['source_event_id'] for v in values):
                    raise L1OwnerError('browse_source_not_currently_readable', status=409, field_path='sources')
                if values[0]['source_revision'] != values[1]['source_revision']:
                    raise L1OwnerError('browse_reading_revision_changed', status=409, field_path='sources')
                sources = [model_source(value, engine.config) for value in values]
                snapshots, bindings = bind_sources(sources, config=engine.config)
                readers = engine._memory_l1_sources()
                for value in snapshots:
                    readers.retain_copy(value)
                for binding in bindings:
                    if not (await readers.locate(binding))['eligible_source']:
                        raise L1OwnerError('source_not_currently_readable', status=409, field_path='sources')
                service = engine._memory_l1_revisions(bucket_mgr)
                catalog, catalog_revision = service.catalog_provider()
                identity = source_model_identity(engine.config, sources)
                # Preserve the authenticated author's full narrative; only the
                # genuine acquired article/read understanding qualify as L0.
                draft = {'title': envelope['metadata'].get('name') or '', 'body': envelope['content']}
                payload = {'input_kind': 'assistant_browse_reading',
                    'draft': {**envelope.get('draft_arguments', {}), 'narrative': draft},
                    'narrative_from_draft': True, 'identity': identity, 'catalog_revision': catalog_revision,
                    'sources': semantic_views(snapshots, bindings),
                    'semantic_contract': semantic_contract([b['binding_id'] for b in bindings], catalog=catalog,
                        initial_chat=True, require_keywords=True, identity=identity)}
                generation = {'prompt_revision': prompt_context['prompt_revision'],
                    'model_route_revision': 'route_sha256:' + semantic_sha256({k: v for k, v in route.items() if k != 'client'}),
                    'task_id': memory_id + ':' + str(claimed['draft_revision']), 'actor': 'browse_memory_preparation'}
                messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
                source_fingerprint = semantic_sha256({'bindings': bindings})
                output = claimed.get('preparation_output') or {}
                if output.get('draft_revision') == claimed['draft_revision']:
                    if output.get('source_fingerprint') != source_fingerprint:
                        raise L1OwnerError('browse_prepared_source_changed', status=409, field_path='sources')
                    raw, generation = output['text'], output['generation']
                    catalog, catalog_revision = output['catalog'], output['catalog_revision']
                else:
                    io_id = io.start(memory_id, stage='chat_l1', task_id=memory_id, attempt=claimed['attempts'],
                        attempt_id=claimed['claim_token'], messages=messages, parameters=completion_parameters(route, engine.temperature),
                        draft_revision=claimed['draft_revision'], generation=generation, source_revisions=bindings)
                    response = await engine._daily_chat_memory_create_completion(route['client'], model=route['model'],
                        messages=messages, max_tokens=engine.daily_chat_memory_candidate_max_tokens, temperature=engine.temperature,
                        use_daily_client=route['connection'] == 'daily', task_stage='chat_l1', task_route=route, model_io_id=io_id)
                    raw = engine._completion_content(response) or ''
                    io.returned(io_id, response=response, raw_output=raw, finish_reason=engine._completion_finish_reason(response))
                    if engine._completion_finish_reason(response) == 'length':
                        raise L1OwnerError('first_pass_output_incomplete', field_path='generation')
                    from memory_l1_api import validate_response
                    validate_response(response)
                    output = {'text': raw, 'draft_revision': claimed['draft_revision'],
                        'source_fingerprint': source_fingerprint, 'generation': generation,
                        'catalog': copy.deepcopy(catalog), 'catalog_revision': catalog_revision, 'model_io_id': io_id}
                    def preserve(current):
                        if current.get('claim_token') != claimed.get('claim_token'):
                            raise L1OwnerError('task_claim_changed', status=409, field_path='claim_token')
                        current['preparation_output'] = output
                        return current
                    claimed = tasks.update(memory_id, expected_revision=claimed['revision'], transform=preserve)
                parsed = engine._parse_json_object(raw)
                from memory_l1_ingestion import MODEL_FIELDS
                if isinstance(parsed, dict) and (set(parsed) == MODEL_FIELDS or parsed.get('format') == 'semantic-v1'):
                    parsed = {'document': parsed}
                if (not isinstance(parsed, dict) or set(parsed) - {'document', 'kind', 'confidence', 'sensitive'}
                        or not isinstance(parsed.get('document'), dict)):
                    raise L1OwnerError('model_document_fields_required', field_path='document')
                document = assemble_document(parsed['document'], narrative=draft, require_keywords=True)
                document['narrative']['body'] = draft['body']
                full = prepare_model_document(service, memory_id=memory_id, document=document, sources=sources,
                    generation=generation, request_id=claimed['operation_id'], catalog_context=(catalog, catalog_revision), config=engine.config)
                for binding in bindings:
                    if not (await readers.locate(binding))['eligible_source']:
                        raise L1OwnerError('source_not_currently_readable', status=409, field_path='sources')
                body, semantic = split_document(full)
                original = claimed.get('original_envelope') or envelope
                provenance = {'contract': 'browse-l1-submission-v1', 'task_id': memory_id,
                    'operation_id': claimed['operation_id'], 'original_content_sha256': descriptor['content_sha256'],
                    'original_canonical_body_sha256': hashlib.sha256(canonical_l1_body(original['content']).encode()).hexdigest(),
                    'draft_revision': claimed['draft_revision'], 'draft_body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                    'owner_edits': copy.deepcopy(claimed.get('owner_edits', []))}
                proposal = MemoryProposal.from_mapping({'proposal_id': memory_id, 'source_type': 'hold',
                    'proposed_body': body, 'original_excerpt': '', 'source_refs': ['tool_operation:' + claimed['operation_id'],
                        *['browse_material:' + b['source_id'] + '@' + b['source_revision'] for b in bindings]],
                    'source_status': 'verified', 'memory_type': 'durable_fact', 'confidence': parsed.get('confidence', 1.0),
                    'requested_mode': 'auto', 'owner_explicit': True,
                    'metadata': {**envelope['metadata'], 'memory_l1': semantic,
                        'aiz_source_descriptor': {'descriptor': descriptor, 'observation': 'verified', 'eligible_source': True},
                        'browse_preparation': provenance,
                        'legacy_candidate': {'tags': envelope['metadata'].get('tags', [])}}})
                # Keep the existing explicit hold policy; chat policy/mode are
                # unrelated to this producer even though model routing is reused.
                policy = PolicyDecision(action='auto_accept', reason_codes=('ACCEPT_EXPLICIT_MEMORY_TOOL',),
                    decision_policy='explicit_tool', alias_trust='weak', requires_owner_confirmation=False)
                claimed, row = tasks.put_prepared(claimed, proposal, policy)
                if row['status'] == 'accepted':
                    await engine._memory_l1_adopt_candidate(row, bucket_mgr, request_id='browse-task:' + memory_id)
                tasks.finish(claimed, phase='prepared', candidate_ids=[memory_id], generation=generation,
                    duration_ms=int((time.monotonic() - started) * 1000))
                if io_id:
                    io.finish(io_id)
                results.append(tasks.receipt(memory_id))
            except Exception as exc:
                if io_id:
                    io.finish(io_id, error=exc)
                try:
                    tasks.finish(claimed, phase='failed', issue=issue(exc))
                except L1OwnerError:
                    pass  # A newer owner edit keeps its own queued revision.
                results.append({'task_id': claimed['task_id'], 'status': 'failed', 'issue': issue(exc)})
        return {'status': 'completed', 'results': results}
    finally:
        engine._browse_preparation_running = False


async def prepare_memory_tasks(engine, bucket_mgr):
    """Share the existing per-run allowance fairly across both task queues."""
    from memory_chat_preparation import prepare_chat_tasks
    from memory_l1_runtime import task_options
    from memory_l3_sync import run_sync_tasks
    from memory_l3_followup import run_followup_tasks
    allowance = task_options(engine.config)['chat_memory_jobs_per_run']
    choices = (prepare_chat_tasks, prepare_browse_tasks, run_sync_tasks, run_followup_tasks)
    cursor = getattr(engine, '_memory_preparation_cursor', 0) % len(choices)
    engine._memory_preparation_cursor = (cursor + 1) % len(choices)
    workers = choices[cursor:] + choices[:cursor]
    results = []
    for worker in workers:
        if allowance <= 0:
            break
        result = await worker(engine, bucket_mgr, max_jobs=allowance)
        allowance -= len(result.get('results', []))
        results.append(result)
    return {'status': 'completed', 'queues': results}
