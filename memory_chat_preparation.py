"""Durable chat submissions; facts still go through Proposal/Policy/Commit."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import time
import uuid

from memory_authority import IdempotencyConflict
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_service import L1OwnerError
from memory_l1_wire_schema import chat_submission_parameters
from memory_model_io import MemoryModelIO, completion_parameters

PREFIX = 'memory_l1_chat_task:'


def issue(exc):
    """Only program-defined codes/paths, never exception messages with values."""
    return {'code': getattr(exc, 'code', type(exc).__name__),
            'field_path': getattr(exc, 'field_path', getattr(exc, 'path', ''))}


def validate_submission(envelope):
    from chat_memory_tool import candidate_id
    from jsonschema import Draft202012Validator
    if not isinstance(envelope, dict) or set(envelope) != {'operation_id', 'arguments', 'context'}:
        raise L1OwnerError('invalid_chat_envelope', field_path='submission')
    candidate_id(envelope['operation_id'])
    errors = list(Draft202012Validator(chat_submission_parameters()).iter_errors(envelope['arguments']))
    if errors:
        raise L1OwnerError('invalid_chat_submission', field_path='arguments')
    context = envelope['context']
    if not isinstance(context, dict) or set(context) != {'conversation_id', 'request_id', 'turn_id', 'sources'}:
        raise L1OwnerError('invalid_source_context', field_path='context')
    if any(not isinstance(context.get(k), str) or not context[k] or len(context[k]) > 200
           for k in ('conversation_id', 'request_id', 'turn_id')):
        raise L1OwnerError('invalid_source_identity', field_path='context')
    sources = context['sources']
    if not isinstance(sources, list) or not 1 <= len(sources) <= 16:
        raise L1OwnerError('source_missing', field_path='context.sources')
    seen = set()
    for source in sources:
        if (not isinstance(source, dict) or set(source) != {'event_id', 'version_id', 'role', 'text', 'created_at'}
                or any(not isinstance(v, str) or not v for v in source.values())
                or source['role'] not in ('user', 'assistant')):
            raise L1OwnerError('invalid_source', field_path='context.sources')
        if source['event_id'] in seen or len(source['text']) > 12000:
            raise L1OwnerError('source_out_of_bounds', field_path='context.sources')
        try:
            stamp = datetime.fromisoformat(source['created_at'].replace('Z', '+00:00'))
        except ValueError:
            raise L1OwnerError('invalid_source_time', field_path='context.sources') from None
        if stamp.tzinfo is None:
            raise L1OwnerError('source_timezone_missing', field_path='context.sources')
        seen.add(source['event_id'])
    if (sources[-1]['event_id'] != context['turn_id'] or sources[-1]['role'] != 'user'
            or sum(len(s['text']) for s in sources) > 60000):
        raise L1OwnerError('source_coverage_invalid', field_path='context.sources')
    from memory_l1_wire_schema import MEMORY_WRITE_MAX_REQUEST_BYTES
    if len(canonical_json(envelope).encode('utf-8')) > MEMORY_WRITE_MAX_REQUEST_BYTES:
        raise L1OwnerError('memory_request_too_large', status=413, field_path='submission')


class ChatMemoryPreparation:
    def __init__(self, authority):
        self.authority = authority

    def get(self, memory_id):
        db = self.authority._connect()
        try:
            row = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + memory_id,)).fetchone()
            return json.loads(row['value']) if row else None
        finally:
            db.close()

    @staticmethod
    def public(row):
        return {k: copy.deepcopy(row.get(k)) for k in (
            'task_id', 'operation_id', 'phase', 'draft_revision', 'revision', 'created_at', 'updated_at',
            'candidate_ids', 'issue', 'attempts', 'generation', 'usage', 'duration_ms',
            'output_shape', 'input_metrics')} | {'source_kind': 'chat_submission'}

    def enqueue(self, envelope):
        from chat_memory_tool import candidate_id
        validate_submission(envelope)
        memory_id = candidate_id(envelope['operation_id'])
        digest = semantic_sha256(envelope)
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + memory_id,)).fetchone()
            if prior:
                if json.loads(prior['value'])['submission_hash'] != digest:
                    raise IdempotencyConflict('operation payload changed')
            else:
                stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
                row = {'task_id': memory_id, 'operation_id': envelope['operation_id'], 'phase': 'queued',
                    'draft_revision': 1, 'revision': 1, 'submission_hash': digest,
                    'envelope': copy.deepcopy(envelope), 'created_at': stamp, 'updated_at': stamp,
                    'candidate_ids': [], 'attempts': 0, 'issue': None, 'lease_until': 0,
                    'history': [], 'edit_requests': {}}
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (PREFIX + memory_id, canonical_json(row)))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return self.receipt(memory_id)

    def list(self, *, limit=100, phase=None, private=False):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise L1OwnerError('invalid_task_page_limit', field_path='limit')
        db = self.authority._connect()
        try:
            where, params = '', [PREFIX + '%']
            if phase is not None:
                where = " AND (json_extract(value,'$.phase')=?"
                params.append(phase)
                if phase == 'queued':
                    where += " OR (json_extract(value,'$.phase')='generating' AND json_extract(value,'$.lease_until')<?)"
                    params.append(time.time())
                where += ')'
            projection = 'value' if private else "json_remove(value,'$.envelope','$.history','$.edit_requests','$.preparation_output') AS value"
            ordering = 'ASC' if phase is not None else 'DESC'
            rows = db.execute('SELECT ' + projection + ' FROM authority_meta WHERE key LIKE ?' + where
                + " ORDER BY json_extract(value,'$.created_at') " + ordering + ',key LIMIT ?', (*params, limit)).fetchall()
            values = [json.loads(r['value']) for r in rows]
            return values if private else [self.public(r) for r in values]
        finally:
            db.close()

    def update(self, memory_id, *, expected_revision, transform):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            record = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + memory_id,)).fetchone()
            if not record:
                raise L1OwnerError('task_not_found', status=404, field_path='task_id')
            row = json.loads(record['value'])
            if row['revision'] != expected_revision:
                raise L1OwnerError('submission_revision_conflict', status=409, field_path='revision')
            row = transform(row)
            row.update(revision=expected_revision + 1, updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(row), PREFIX + memory_id))
            db.commit()
            return row
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def claim(self, row, *, lease_seconds):
        def change(current):
            if current['phase'] != 'queued' and not (current['phase'] == 'generating' and current['lease_until'] < time.time()):
                raise L1OwnerError('task_not_queued', status=409, field_path='phase')
            current.update(phase='generating', lease_until=time.time() + lease_seconds,
                           claim_token=uuid.uuid4().hex, attempts=current['attempts'] + 1, issue=None)
            return current
        return self.update(row['task_id'], expected_revision=row['revision'], transform=change)

    def finish(self, claimed, *, phase, **result):
        def change(current):
            if current.get('claim_token') != claimed.get('claim_token'):
                raise L1OwnerError('task_claim_changed', status=409, field_path='claim_token')
            current.update(phase=phase, lease_until=0, **result)
            return current
        return self.update(claimed['task_id'], expected_revision=claimed['revision'], transform=change)

    def detail(self, memory_id):
        row = self.get(memory_id)
        if not row:
            raise L1OwnerError('task_not_found', status=404, field_path='task_id')
        return {'task': self.public(row), 'draft': copy.deepcopy(row['envelope']['arguments']),
                'history': copy.deepcopy(row['history']), 'receipt': self.receipt(memory_id),
                'preparation_output': copy.deepcopy(row.get('preparation_output'))}

    def edit(self, memory_id, payload):
        if (not isinstance(payload, dict) or set(payload) != {'expected_revision', 'request_id', 'draft'}
                or type(payload['expected_revision']) is not int or not isinstance(payload['request_id'], str)
                or not 1 <= len(payload['request_id']) <= 128):
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
            envelope = copy.deepcopy(row['envelope'])
            envelope['arguments'] = payload['draft']
            validate_submission(envelope)
            row['history'].append({'draft_revision': row['draft_revision'], 'draft': row['envelope']['arguments'],
                                   'updated_at': row['updated_at'], 'issue': row.get('issue')})
            row.update(envelope=envelope, draft_revision=row['draft_revision'] + 1,
                       phase='queued', lease_until=0, issue=None, claim_token=None)
            row['edit_requests'][payload['request_id']] = fingerprint
            return row
        self.update(memory_id, expected_revision=payload['expected_revision'], transform=change)
        return self.detail(memory_id)

    def receipt(self, memory_id):
        from chat_memory_tool import receipt
        row = self.get(memory_id)
        if not row:
            return receipt(self.authority, memory_id)
        adopted = receipt(self.authority, memory_id)
        if adopted.get('memory_status') != 'not_found':
            return {**adopted, 'submission_status': 'accepted', 'preparation_status': 'prepared', 'task_id': row['task_id']}
        return {'ok': True, 'submission_status': 'accepted', 'preparation_status': row['phase'],
                'memory_status': 'preparation_failed' if row['phase'] == 'failed' else 'preparing',
                'task_id': row['task_id'], 'operation_id': row['operation_id'], 'candidate_id': memory_id,
                'memory_id': None, 'body_preserved': True, 'l1_state': 'not_prepared', 'l1_complete': False,
                'issue': row.get('issue')}


async def prepare_chat_tasks(engine, bucket_mgr):
    """Runs outside the chat request; never executes the original tool again."""
    from chat_memory_tool import ChatMemoryTool
    from memory_l1_ingestion import bind_sources
    from memory_l1_semantic_wire import assemble_document, semantic_contract, semantic_sources
    from memory_source_provenance import canonical_source
    from memory_l1_runtime import task_options
    if getattr(engine, '_chat_preparation_running', False):
        return {'status': 'locked'}
    options = task_options(engine.config)
    if not engine.memory_authority_enabled or not engine.memory_authority_store or options['chat_memory_jobs_per_run'] == 0:
        return {'status': 'disabled'}
    if engine.daily_chat_memory_mode == 'off':
        return {'status': 'disabled', 'reason': 'memory_policy_off'}
    prompt, prompt_context = engine._memory_l1_prompt('ombre.chat_memory_prepare_prompt')
    if not prompt.strip():
        return {'status': 'disabled', 'reason': 'prompt_disabled'}
    route = engine._memory_task_route('chat_l1')
    client, model, use_daily = route['client'], route['model'], route['connection'] == 'daily'
    if not route['configured']:
        return {'status': 'not_prepared', 'reason': route['unavailable_reason']}
    engine._chat_preparation_running = True
    tasks = ChatMemoryPreparation(engine.memory_authority_store)
    model_io = MemoryModelIO(engine.memory_authority_store)
    results = []
    try:
        for pending in tasks.list(limit=options['chat_memory_jobs_per_run'], phase='queued', private=True):
            completion_meta = {}
            io_id = None
            try:
                claimed = tasks.claim(pending, lease_seconds=engine.daily_chat_memory_timeout_seconds + 60)
            except L1OwnerError:
                continue
            try:
                started = time.monotonic()
                memory_id = claimed['task_id']
                existing = engine.memory_authority_store.get_candidate(memory_id)
                if existing:
                    if existing['status'] == 'accepted':
                        await engine._memory_l1_adopt_candidate(existing, bucket_mgr, request_id='chat-task:' + memory_id)
                    tasks.finish(claimed, phase='prepared', candidate_ids=[memory_id])
                    results.append(tasks.receipt(memory_id))
                    continue
                envelope = claimed['envelope']
                sources = [canonical_source(s, conversation_id=envelope['context']['conversation_id'],
                    profile_id=engine.memory_semantic_profile_id) for s in envelope['context']['sources']]
                snapshots, bindings = bind_sources(sources, config=engine.config)
                for source, snapshot in zip(sources, snapshots):
                    source['_participant_revision'] = snapshot['source_revision']
                readers = engine._memory_l1_sources()
                # Preserve the authenticated program snapshots first: the
                # canonical reader resolves their exact upstream revisions
                # from these copies. A retained copy never grants eligibility;
                # locate still checks the live source before model use.
                for snapshot in snapshots:
                    readers.retain_copy(snapshot)
                for binding in bindings:
                    if not (await readers.locate(binding))['eligible_source']:
                        raise L1OwnerError('source_not_currently_readable', status=409, field_path='sources')
                service = engine._memory_l1_revisions(bucket_mgr)
                catalog, catalog_revision = service.catalog_provider()
                from memory_source_identity import source_model_identity
                identity = source_model_identity(engine.config, sources)
                payload = {'input_kind': 'chat_authored_draft', 'draft': envelope['arguments'], 'identity': identity,
                    'narrative_from_draft': True, 'catalog_revision': catalog_revision,
                    'sources': semantic_sources(snapshots, bindings, chat_view=True,
                        source_timezone=getattr(engine, 'tz', None)),
                    'semantic_contract': semantic_contract([b['binding_id'] for b in bindings], catalog=catalog,
                        initial_chat=True, require_keywords=True, identity=identity)}
                serialized_payload = json.dumps(payload, ensure_ascii=False)
                input_metrics = {'contract': 'semantic-v1', 'catalog_revision': catalog_revision,
                    'system_characters': len(prompt), 'payload_characters': len(serialized_payload),
                    'payload_utf8_bytes': len(serialized_payload.encode('utf-8')),
                    'components': {key: len(json.dumps(value, ensure_ascii=False)) for key, value in payload.items()}}
                from memory_task_routes import completion_options
                input_metrics['completion_parameters'] = {'model': model,
                    **completion_options(route, temperature=engine.temperature)}
                completion_meta['input_metrics'] = input_metrics
                generation = {'prompt_revision': prompt_context['prompt_revision'],
                    'model_route_revision': 'route_sha256:' + semantic_sha256({k: v for k, v in route.items() if k != 'client'}),
                    'task_id': memory_id + ':' + str(claimed['draft_revision']), 'actor': 'chat_memory_preparation'}
                messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': serialized_payload}]
                io_id = model_io.start(memory_id, stage='chat_l1', task_id=memory_id,
                    attempt=claimed['attempts'], attempt_id=claimed['claim_token'], messages=messages,
                    parameters=completion_parameters(route, engine.temperature),
                    draft_revision=claimed['draft_revision'], generation={**generation, 'catalog_revision':catalog_revision},
                    source_revisions=bindings)
                response = await engine._daily_chat_memory_create_completion(client, model=model,
                    messages=messages,
                    max_tokens=engine.daily_chat_memory_candidate_max_tokens, temperature=engine.temperature,
                    use_daily_client=use_daily, task_stage='chat_l1', task_route=route, model_io_id=io_id)
                raw_output = engine._completion_content(response) or ''
                model_io.returned(io_id, response=response, raw_output=raw_output,
                                  finish_reason=engine._completion_finish_reason(response))
                reported_usage = getattr(response, 'usage', None)
                usage = {name: getattr(reported_usage, name, None) for name in
                         ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                token_details = getattr(reported_usage, 'completion_tokens_details', None)
                reasoning_tokens = (token_details.get('reasoning_tokens') if isinstance(token_details, dict)
                                    else getattr(token_details, 'reasoning_tokens', None))
                if type(reasoning_tokens) is int:
                    usage['reasoning_tokens'] = reasoning_tokens
                for name in ('prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'):
                    count = getattr(reported_usage, name, None)
                    if type(count) is int:
                        usage[name] = count
                completion_meta = {'generation': generation, 'usage': usage, 'input_metrics': input_metrics,
                    'duration_ms': int((time.monotonic() - started) * 1000),
                    'preparation_output': {'draft_revision': claimed['draft_revision'],
                        'attempt': claimed['attempts'],
                        'model': model, 'model_task': {k: v for k, v in route.items() if k != 'client'},
                        'finish_reason': engine._completion_finish_reason(response),
                        'text': raw_output}, 'output_shape': None}
                if engine._completion_finish_reason(response) == 'length':
                    raise L1OwnerError('first_pass_output_incomplete', field_path='generation')
                from memory_l1_api import validate_response
                validate_response(response)
                parsed = engine._parse_json_object(raw_output)
                from memory_l1_ingestion import MODEL_FIELDS
                output_shape = {'draft_revision': claimed['draft_revision'],
                    'attempt': claimed['attempts'],
                    'finish_reason': engine._completion_finish_reason(response),
                    'output_characters': len(raw_output),
                    'top_level_keys': sorted(parsed) if isinstance(parsed, dict) else [],
                    'normalization': 'document_wrapper' if isinstance(parsed, dict)
                        and isinstance(parsed.get('document'), dict) else 'unrecognized'}
                completion_meta['output_shape'] = output_shape
                # A complete model document at the root has the same semantic
                # contract. Wrapping it is a program-formatting responsibility;
                # no fields are discarded and inner L1 validation still runs.
                if isinstance(parsed, dict) and (set(parsed) == MODEL_FIELDS or parsed.get('format') == 'semantic-v1'):
                    parsed = {'document': parsed}
                    output_shape['normalization'] = ('semantic_document_root' if parsed['document'].get('format') == 'semantic-v1'
                        else 'complete_document_root')
                if not isinstance(parsed, dict) or set(parsed) - {'document', 'kind', 'confidence', 'sensitive'} or not isinstance(parsed.get('document'), dict):
                    raise L1OwnerError('model_document_fields_required', field_path='document')
                document = assemble_document(parsed['document'], narrative=envelope['arguments']['narrative'],
                    require_keywords=True)
                # The model proposes semantics. The first-pass author keeps its
                # complete prose; silently replacing it would lose that product.
                draft = envelope['arguments']['narrative']
                if not isinstance(document.get('narrative'), dict):
                    raise L1OwnerError('invalid_l1_narrative', field_path='document.narrative')
                document['narrative']['body'] = draft['body']
                if draft.get('title'):
                    document['narrative']['title'] = draft['title']
                # Check the draft again after awaiting the model: owner edits
                # invalidate this task before it can create a proposal.
                if tasks.get(memory_id)['revision'] != claimed['revision']:
                    raise L1OwnerError('submission_revision_conflict', status=409, field_path='revision')
                # Rebinding after generation must not assign the old model
                # interpretation to a newly configured participant.
                bind_sources(sources, config=engine.config)
                prepared_envelope = copy.deepcopy(envelope)
                prepared_envelope['arguments'] = {**parsed, 'document': document}
                proposal = ChatMemoryTool(engine, bucket_mgr)._proposal(prepared_envelope, generation=generation,
                    catalog_context=(copy.deepcopy(catalog), catalog_revision))
                # Recheck source access after generation, before putting facts
                # into Review or Auto. Source copies alone do not grant access.
                for binding in proposal.metadata['memory_l1']['evidence_bindings']:
                    if not (await readers.locate(binding))['eligible_source']:
                        raise L1OwnerError('source_not_currently_readable', status=409, field_path='sources')
                if tasks.get(memory_id)['revision'] != claimed['revision']:
                    raise L1OwnerError('submission_revision_conflict', status=409, field_path='revision')
                proposal.metadata['chat_submission'] = {'operation_id': claimed['operation_id'],
                    'task_id': memory_id, 'draft_revision': claimed['draft_revision'], 'draft': copy.deepcopy(envelope['arguments'])}
                policy = engine.memory_ingestion_policy.evaluate(proposal)
                row = engine.memory_authority_store.put_candidate(proposal, policy)
                if row['status'] == 'accepted':
                    await engine._memory_l1_adopt_candidate(row, bucket_mgr, request_id='chat-task:' + memory_id)
                tasks.finish(claimed, phase='prepared', candidate_ids=[memory_id],
                    generation=generation, catalog_revision=catalog_revision, usage=usage, input_metrics=input_metrics,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    output_shape=output_shape, preparation_output=completion_meta['preparation_output'])
                model_io.finish(io_id)
                results.append(tasks.receipt(memory_id))
            except Exception as exc:
                if io_id: model_io.finish(io_id, error=exc)
                try:
                    tasks.finish(claimed, phase='failed', issue=issue(exc), **completion_meta)
                except L1OwnerError:
                    pass  # A newer owner draft owns its own queued task.
                results.append({'task_id': claimed['task_id'], 'status': 'failed', 'issue': issue(exc)})
        return {'status': 'completed', 'results': results}
    finally:
        engine._chat_preparation_running = False
