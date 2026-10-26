"""Scoped owner-L2 correction plans on the existing Authority and L1 writer."""
from __future__ import annotations

import copy
import json
import math
import time
import uuid

from memory_authority import IdempotencyConflict, MemoryNotFound, RevisionConflict
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_service import L1OwnerError

CONTRACT = 'memory-l3-sync-v1'
from memory_relation_dependencies import pending_sync_scope, SYNC_CLOSED as CLOSED


def initial_processing():
    return {'revision': 1, 'attempts': 0, 'scope': {'statement_ids': [], 'unknown_scope': True},
        'plan': None, 'receipt': None, 'issue': None, 'history': [], 'requests': {}, 'lease_until': 0}


def validate_followup_adoption(authority, memory_id, payload, program_context, *, actor,
                               decision_source, source_bindings, source_snapshots):
    """Only an accepted exact candidate authorizes a model-origin L1 update."""
    row = authority.get_candidate(program_context.get('proposal_id', ''))
    meta = ((row or {}).get('proposal') or {}).get('metadata', {}).get('l3_followup') or {}
    compiled = meta.get('compiled') or {}
    if (not row or row['status'] not in {'accepted', 'committed'} or actor != 'memory_followup'
            or decision_source != 'followup_policy' or row['proposal'].get('owner_explicit') is not False
            or meta.get('memory_id') != memory_id or meta.get('expected_revision') != payload['expected_revision']
            or meta.get('task_id') != payload['request_id'] or not isinstance(meta.get('plan'), dict)
            or semantic_sha256(meta['plan']) != program_context.get('plan_sha256')
            or program_context.get('candidate_revision') != row['revision'] or program_context.get('policy') != row['policy']
            or compiled.get('document') != payload['document'] or compiled.get('source_bindings') != source_bindings
            or compiled.get('source_snapshots') != source_snapshots):
        raise L1OwnerError('accepted_followup_plan_required', status=409, field_path='program_context')
    if program_context.get('source_ref') not in [{key: binding[key] for key in ('source_namespace', 'source_id', 'source_revision')}
            for binding in source_bindings]:
        raise L1OwnerError('accepted_followup_source_required', status=409, field_path='sources')
    return row


class SyncProposals:
    def __init__(self, authority):
        self.authority = authority

    @staticmethod
    def public(row):
        original = copy.deepcopy(row['proposal'])
        process = original.pop('processing', {})
        state = row['state']
        actions = [] if state in CLOSED else ['retry'] if state in {'applying', 'l1_applied_l2_pending'} else (
            ['prepare'] if state in {'conflict', 'ambiguous'} else ['prepare', 'apply'] if process.get('plan') else ['prepare'])
        if state == 'failed' and process.get('plan'):
            actions = ['retry', 'prepare']
        if state == 'preparing' and process.get('lease_until', 0) > time.time():
            actions = []
        return {**{key: row[key] for key in ('memory_id', 'presentation_revision', 'first_pass_revision', 'state')},
            'revision': process.get('revision', 1), 'proposal': original, 'plan': process.get('plan'),
            'scope': process.get('scope', {'statement_ids': [], 'unknown_scope': True}),
            'receipt': process.get('receipt'), 'issue': process.get('issue'), 'allowed_actions': actions,
            'history': process.get('history', []), 'generation': process.get('generation'),
            'model_io_id': process.get('model_io_id')}

    def get(self, memory_id, revision, *, conn=None):
        db = conn or self.authority._connect()
        try:
            row = db.execute('SELECT * FROM memory_l2_sync_proposals WHERE memory_id=? AND presentation_revision=?', (memory_id, revision)).fetchone()
            if not row:
                raise L1OwnerError('sync_proposal_not_found', status=404, field_path='presentation_revision')
            return {**dict(row), 'proposal': json.loads(row['proposal_json'])}
        finally:
            if conn is None:
                db.close()

    def list(self, memory_id):
        db = self.authority._connect()
        try:
            rows = db.execute('SELECT * FROM memory_l2_sync_proposals WHERE memory_id=? ORDER BY presentation_revision DESC', (memory_id,)).fetchall()
            return [self.public({**dict(row), 'proposal': json.loads(row['proposal_json'])}) for row in rows]
        finally:
            db.close()

    def update(self, memory_id, revision, expected_revision, transform, *, action=None, request_id=None):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = self.get(memory_id, revision, conn=db)
            process = row['proposal'].setdefault('processing', initial_processing())
            request = {'action': action, 'expected_revision': expected_revision} if request_id else None
            prior = process['requests'].get(request_id) if request_id else None
            if prior:
                if prior != semantic_sha256(request):
                    raise IdempotencyConflict('sync request changed')
                db.commit()
                return row, True
            if process['revision'] != expected_revision:
                raise L1OwnerError('sync_revision_conflict', status=409, field_path='expected_revision')
            row = transform(row)
            process = row['proposal']['processing']
            process['revision'] = expected_revision + 1
            if request_id:
                process['requests'][request_id] = semantic_sha256(request)
            db.execute('UPDATE memory_l2_sync_proposals SET state=?,proposal_json=? WHERE memory_id=? AND presentation_revision=?',
                (row['state'], canonical_json(row['proposal']), memory_id, revision))
            db.commit()
            return row, False
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def pending(self, limit):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            stale = db.execute("SELECT * FROM memory_l2_sync_proposals WHERE state='preparing' AND json_extract(proposal_json,'$.contract')=? AND json_extract(proposal_json,'$.processing.lease_until')<?", (CONTRACT, time.time())).fetchall()
            for value in stale:
                data = json.loads(value['proposal_json'])
                process = data['processing']
                process.update(revision=process['revision'] + 1, lease_until=0,
                    issue={'code': 'worker_interrupted', 'field_path': 'state', 'recovery_action': 'retry' if process.get('plan') else 'prepare'})
                db.execute("UPDATE memory_l2_sync_proposals SET state='failed',proposal_json=? WHERE memory_id=? AND presentation_revision=? AND state='preparing'",
                    (canonical_json(data), value['memory_id'], value['presentation_revision']))
            rows = db.execute("SELECT * FROM memory_l2_sync_proposals WHERE state='pending' AND json_extract(proposal_json,'$.contract')=? ORDER BY presentation_revision DESC LIMIT ?", (CONTRACT, limit)).fetchall()
            db.commit()
            return [{**dict(row), 'proposal': json.loads(row['proposal_json'])} for row in rows]
        finally:
            db.close()


_FIELDS = {'statements': {'content', 'subject_ref', 'classification', 'time', 'conditions', 'property', 'attributes', 'cognition', 'evidence_refs'},
    'entities': {'name', 'aliases', 'entity_type'}, 'scenes': {'title', 'statement_refs', 'attributes'},
    'retrieval': {'keywords', 'tags'}, 'narrative': {'title', 'body'}}


def apply_local_patch(document, plan, *, evidence_ids=()):
    """Only explicit stable-member edits; all untouched content stays exact."""
    result = copy.deepcopy(document)
    allowed_refs = set(evidence_ids)
    member_evidence = {}
    for index, patch in enumerate(plan['patch']):
        if (not isinstance(patch, dict) or set(patch) != {'op', 'path', 'old', 'value', 'evidence_refs'}
                or patch['op'] not in {'replace', 'replace_text', 'add'} or not isinstance(patch['path'], str)
                or not isinstance(patch['evidence_refs'], list) or not patch['evidence_refs']
                or any(not isinstance(ref, str) or ref not in allowed_refs for ref in patch['evidence_refs'])):
            raise L1OwnerError('invalid_l3_patch', field_path='plan.patch[' + str(index) + ']')
        path = [part.replace('~1', '/').replace('~0', '~') for part in patch['path'].split('/')[1:]]
        if not patch['path'].startswith('/') or not path or path[0] not in _FIELDS:
            raise L1OwnerError('l3_patch_outside_local_scope', field_path=patch['path'])
        if patch['op'] == 'add':
            if path != ['statements'] or patch['old'] is not None or not isinstance(patch['value'], dict):
                raise L1OwnerError('l3_statement_addition_required', field_path=patch['path'])
            from memory_l1_semantic_wire import assemble_document
            value = copy.deepcopy(patch['value'])
            if 'statement_revision_id' in value:
                raise L1OwnerError('l3_managed_patch_field', field_path=patch['path'])
            value['statement_id'] = value.get('statement_id') or 'new:l3:' + str(index)
            assembled = assemble_document({'format': 'semantic-v1', 'entities': [], 'statements': [value]},
                narrative=result['narrative'])['statements'][0]
            assembled['statement_revision_id'] = None
            assembled['evidence_refs'] = list(patch['evidence_refs'])
            result['statements'].append(assembled)
            continue
        node = result[path[0]]
        member = None
        if isinstance(node, list):
            key = {'statements': 'statement_id', 'entities': 'entity_id', 'scenes': 'scene_id'}[path[0]]
            matches = [item for item in node if len(path) > 1 and item[key] == path[1]]
            if len(matches) != 1 or len(path) < 3 or path[2] not in _FIELDS[path[0]]:
                raise L1OwnerError('l3_patch_member_not_found', field_path=patch['path'])
            node, parts, member = matches[0], path[2:], matches[0] if path[0] == 'statements' else None
        else:
            if len(path) != 2 or path[1] not in _FIELDS[path[0]]:
                raise L1OwnerError('l3_patch_field_not_allowed', field_path=patch['path'])
            node, parts = node, path[1:]
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise L1OwnerError('l3_patch_path_not_found', field_path=patch['path'])
            node = node[part]
        key = parts[-1]
        if (member is not None and parts == ['cognition'] and key not in node
                and patch['op'] == 'replace' and patch['old'] is None):
            node[key] = None  # This is the sole optional Statement field.
        if not isinstance(node, dict) or key not in node:
            raise L1OwnerError('l3_patch_path_not_found', field_path=patch['path'])
        if patch['op'] == 'replace_text':
            if path != ['narrative', 'body'] or not isinstance(patch['old'], str) or not patch['old'] or not isinstance(patch['value'], str) or node[key].count(patch['old']) != 1:
                raise L1OwnerError('l3_text_patch_not_unambiguous', field_path=patch['path'])
            node[key] = node[key].replace(patch['old'], patch['value'], 1)
        else:
            if path == ['narrative', 'body']:
                raise L1OwnerError('l3_whole_narrative_rewrite_forbidden', field_path=patch['path'])
            if node[key] != patch['old']:
                raise L1OwnerError('l3_patch_preimage_changed', status=409, field_path=patch['path'])
            node[key] = copy.deepcopy(patch['value'])
        if member is not None:
            member_evidence.setdefault(member['statement_id'], []).extend(patch['evidence_refs'])
    for member in result['statements']:
        if member['statement_id'] in member_evidence:
            member['evidence_refs'] = list(dict.fromkeys(member['evidence_refs'] + member_evidence[member['statement_id']]))
    return result


def validate_plan(material, plan, *, evidence_ids=()):
    if (not isinstance(plan, dict) or set(plan) - {'confidence'} != {'decision', 'reason', 'affected_statement_ids', 'patch'}
            or plan['decision'] not in {'l1_already_correct', 'revise_l1', 'ambiguous'}
            or not isinstance(plan['reason'], str) or not plan['reason'].strip()
            or not isinstance(plan['affected_statement_ids'], list) or not isinstance(plan['patch'], list)):
        raise L1OwnerError('invalid_l3_sync_plan', field_path='plan')
    known = {row['statement_id'] for row in material['l1_document']['statements']}
    if plan.get('confidence') is not None and (type(plan['confidence']) not in (float, int)
            or not math.isfinite(plan['confidence']) or not 0 <= plan['confidence'] <= 1):
        raise L1OwnerError('invalid_l3_plan_confidence', field_path='plan.confidence')
    if any(not isinstance(ref, str) or ref not in known for ref in plan['affected_statement_ids']):
        raise L1OwnerError('l3_affected_member_unknown', field_path='plan.affected_statement_ids')
    if plan['decision'] != 'revise_l1' and plan['patch']:
        raise L1OwnerError('l3_unexpected_patch', field_path='plan.patch')
    if plan['decision'] == 'revise_l1' and not plan['patch']:
        raise L1OwnerError('l3_patch_required', field_path='plan.patch')
    return apply_local_patch(material['l1_document'], plan, evidence_ids=evidence_ids)


def compile_plan(revisions, memory_id, base_detail, plan, source_evidence, *, request_id):
    base = base_detail['l1']['document']
    evidence_ids = [row['binding_id'] for row in base['evidence_bindings']] + [row['binding']['binding_id'] for row in source_evidence]
    document = validate_plan({'l1_document': base}, plan, evidence_ids=evidence_ids)
    snapshots = [copy.deepcopy(row['source']) for row in source_evidence]
    bindings = [copy.deepcopy(row['binding']) for row in source_evidence]
    extended = copy.deepcopy(base)
    by_id = {row['binding_id']: row for row in extended['evidence_bindings']}
    for binding in bindings:
        if binding['binding_id'] in by_id and by_id[binding['binding_id']] != binding:
            raise L1OwnerError('l3_source_binding_conflict', field_path='evidence_bindings')
        if binding['binding_id'] not in by_id:
            extended['evidence_bindings'].append(binding)
    document['evidence_bindings'] = copy.deepcopy(extended['evidence_bindings'])
    catalog, revision = revisions.catalog_provider()
    # Preview the complete existing validator; normalization allocates only
    # program identities. No file, candidate or adopted revision is written.
    revisions._owner_document(document, extended, memory_id=memory_id, request_id=request_id,
        catalog=catalog, catalog_revision=revision, revision=base_detail['current_revision'] + 1)
    touched = set(plan['affected_statement_ids'])
    unknown = False
    for patch in plan['patch']:
        parts = patch['path'].split('/')
        if len(parts) > 3 and parts[1] == 'statements':
            touched.add(parts[2])
        elif parts[1] in {'narrative', 'entities', 'scenes'} and not touched:
            unknown = True
    return {'document': document, 'source_snapshots': snapshots, 'source_bindings': bindings,
        'scope': {'statement_ids': sorted(touched), 'unknown_scope': unknown}}


async def commit_plan(revisions, memory_id, compiled, *, expected_revision, request_id,
                      actor, decision_source, program_context, source_reader, edit_note=''):
    return await revisions.save_program_edit(memory_id,
        {'expected_revision': expected_revision, 'request_id': request_id, 'document': compiled['document'], 'edit_note': edit_note},
        source_snapshots=compiled['source_snapshots'], source_bindings=compiled['source_bindings'],
        source_reader=source_reader, actor=actor, decision_source=decision_source, program_context=program_context)


async def prepare_plan(engine, material, proposal_text, kind, source_evidence, *, memory_id, task_id, attempt=1):
    from memory_model_io import MemoryModelIO, completion_parameters
    if kind not in {'owner_l2', 'followup'}:
        raise L1OwnerError('invalid_l3_origin', field_path='kind')
    prompt, prompt_context = engine._memory_l1_prompt('ombre.memory_l3_sync_prompt')
    if not prompt.strip():
        raise L1OwnerError('l3_sync_prompt_disabled', field_path='prompt')
    route = engine._memory_task_route('chat_l1')
    if not route['configured']:
        raise L1OwnerError('l3_sync_model_unconfigured', field_path='model')
    payload = {'task': 'scoped_l1_revision', 'origin': kind, 'material': material, 'proposal_text': proposal_text,
        'source_evidence': source_evidence, 'output_contract': {
            'decision': 'l1_already_correct|revise_l1|ambiguous', 'reason': 'source-grounded explanation',
            'affected_statement_ids': 'existing statement IDs', 'confidence': 'optional extraction confidence, unknown may be null',
            'patch': [{'op': 'replace|replace_text|add', 'path': '/statements/{id}/content or another allowed field',
                'old': 'exact preimage', 'value': 'typed replacement', 'evidence_refs': ['actual binding IDs']}],
            'allowed_fields': {key: sorted(value) for key, value in _FIELDS.items()},
            'rules': ['narrative.body permits only exact, uniquely located replace_text; no whole narrative replacement',
                'add permits a complete semantic-v1 Statement with all classification/time/conditions judgments',
                'never modify identity, generation, source bindings, entry or cross-memory relations',
                'interpretation/question remains cognition; no promotion to objective fact']}}
    payload = copy.deepcopy(payload)
    # Custody tokens stay in the program material and its pre/post checks.
    # They carry no semantic meaning for the model and change on task claims.
    related = payload['material'].get('related_memory_updates', {})
    related.pop('token', None)
    related.pop('dependency_token', None)
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    generation = {'prompt_revision': prompt_context['prompt_revision'], 'actor': 'memory_l3_sync',
        'model_route_revision': semantic_sha256({key: value for key, value in route.items() if key != 'client'})}
    recorder = MemoryModelIO(engine.memory_authority_store)
    generation['input_sha256'] = semantic_sha256({'messages': messages})
    # The raw provider result is persisted before task checkpointing. A local
    # checkpoint/finish interruption must not turn a completed generation into
    # another provider request on an explicit retry of the same actual inputs.
    db = engine.memory_authority_store._connect()
    try:
        prefix = recorder._prefix(memory_id)
        prior = db.execute("SELECT value FROM authority_meta WHERE key>=? AND key<? "
            "AND json_extract(value,'$.stage')='memory_l3_sync' AND json_extract(value,'$.task_id')=? "
            "AND json_extract(value,'$.generation.input_sha256')=? AND json_extract(value,'$.raw_output') IS NOT NULL "
            'ORDER BY key DESC LIMIT 1', (prefix, prefix + '\uffff', task_id, generation['input_sha256'])).fetchone()
    finally:
        db.close()
    if prior:
        saved = json.loads(prior[0])
        if saved.get('finish_reason') != 'length':
            try:
                plan = engine._parse_json_object(saved['raw_output'])
                validate_plan(material, plan, evidence_ids=[row['binding_id'] for row in material['l1_document']['evidence_bindings']]
                    + [row['binding']['binding_id'] for row in source_evidence])
            except (L1OwnerError, ValueError, TypeError):
                pass
            else:
                recorder.finish(saved['io_id'])
                return {'plan': plan, 'generation': saved['generation'], 'model_io_id': saved['io_id'],
                        'raw_output': saved['raw_output']}
    io_id = recorder.start(memory_id, stage='memory_l3_sync', task_id=task_id, attempt=attempt,
        messages=messages, parameters=completion_parameters(route, engine.temperature), generation=generation,
        source_revisions=[row['binding'] for row in source_evidence])
    try:
        response = await engine._daily_chat_memory_create_completion(route['client'], model=route['model'], messages=messages,
            max_tokens=route['max_tokens'], temperature=engine.temperature, use_daily_client=route['connection'] == 'daily',
            task_stage='memory_l3_sync', task_route=route, model_io_id=io_id)
        raw = engine._completion_content(response) or ''
        finish = engine._completion_finish_reason(response)
        recorder.returned(io_id, response=response, raw_output=raw, finish_reason=finish)
        if finish == 'length':
            raise L1OwnerError('l3_sync_output_incomplete', field_path='plan')
        plan = engine._parse_json_object(raw)
        validate_plan(material, plan, evidence_ids=[row['binding_id'] for row in material['l1_document']['evidence_bindings']]
            + [row['binding']['binding_id'] for row in source_evidence])
        recorder.finish(io_id)
        return {'plan': plan, 'generation': generation, 'model_io_id': io_id, 'raw_output': raw}
    except Exception as exc:
        recorder.finish(io_id, error=exc)
        raise


class MemoryL3SyncService:
    def __init__(self, l2):
        self.l2, self.engine, self.revisions = l2, l2.engine, l2.revisions
        self.tasks = SyncProposals(self.revisions.authority)

    def detail(self, memory_id):
        return {'status': 'ok', 'memory_id': memory_id, 'proposals': self.tasks.list(memory_id), 'adoption_is_separate': True}

    @staticmethod
    def _request_id(memory_id, revision, process):
        return 'l3-sync:' + semantic_sha256({'memory_id': memory_id, 'revision': revision,
            'plan': process['plan_sha256'], 'prepared_against': process['material_identity']})

    async def _material(self, memory_id, source_revision):
        from memory_owner_l2_source import OwnerL2EditSourceReader, bind
        base = await self.revisions.detail(memory_id)
        if base['l1']['state'] != 'current' or base['memory']['state'] != 'active':
            raise L1OwnerError('l1_dependency_changed', status=409, field_path='l1')
        # Original sources are checked through the original source reader, not
        # a retained copy and not the edited L2 narrative itself.
        material = await self.l2._material(memory_id, base['current_revision'])
        db = self.revisions.authority._connect()
        try:
            head = db.execute('SELECT presentation_revision FROM memory_l2_heads WHERE memory_id=?', (memory_id,)).fetchone()
            owner = db.execute('SELECT * FROM memory_l2_revisions WHERE memory_id=? AND presentation_revision=?',
                (memory_id, head[0] if head else -1)).fetchone()
            if not owner or owner['actor'] != 'owner':
                raise L1OwnerError('owner_l2_changed', status=409, field_path='presentation_revision')
            rows = db.execute('SELECT presentation_revision,state,proposal_json FROM memory_l2_sync_proposals '
                'WHERE memory_id=? AND presentation_revision<=? ORDER BY presentation_revision', (memory_id, head[0])).fetchall()
            covered = [row['presentation_revision'] for row in rows if row['state'] not in CLOSED]
        finally:
            db.close()
        if source_revision not in covered or any(rev > source_revision for rev in covered):
            raise L1OwnerError('owner_l2_changed', status=409, field_path='presentation_revision')
        reader, evidence = OwnerL2EditSourceReader(self.revisions.authority), []
        for revision in covered:
            snapshot = await reader.capture(memory_id, revision)
            evidence.append({'source': snapshot, 'binding': bind(snapshot, 'owner-correction:' + str(revision))})
        owner_document = json.loads(owner['document_json'])
        record = self.revisions.authority.get_memory_revision(memory_id, base['current_revision'])
        identity = {'l1_revision': base['current_revision'], 'l1_content_sha256': record['metadata']['memory_l1_content_sha256'],
            'owner_head_revision': owner['presentation_revision'], 'owner_head_sha256': semantic_sha256(owner_document),
            'covered_proposals': covered, 'source_fingerprint': semantic_sha256({'sources': [row['binding'] for row in evidence]})}
        proposal = self.tasks.get(memory_id, source_revision)['proposal']
        material.update(before_l2_text=proposal.get('before_text', ''), owner_l2_text=owner_document['full_injection_text'],
            owner_edit_note=proposal.get('edit_note', ''), identity=identity)
        return material, base, evidence, identity

    async def action(self, memory_id, revision, action, payload):
        if (not isinstance(payload, dict) or set(payload) != {'expected_revision', 'request_id'}
                or type(payload['expected_revision']) is not int or not isinstance(payload['request_id'], str)
                or not 1 <= len(payload['request_id']) <= 128 or action not in {'prepare', 'apply', 'retry'}):
            raise L1OwnerError('invalid_l3_sync_action')
        def change(row):
            process = row['proposal']['processing']
            if row['state'] in CLOSED:
                return row
            if row['state'] == 'preparing' and process.get('lease_until', 0) > time.time():
                raise L1OwnerError('sync_preparation_in_progress', status=409, field_path='state')
            if action == 'prepare':
                if row['state'] in {'applying', 'l1_applied_l2_pending'}:
                    raise L1OwnerError('l3_original_commit_requires_recovery', status=409, field_path='receipt')
                if process.get('receipt', {}).get('l1_operation_id') if process.get('receipt') else False:
                    raise L1OwnerError('l3_original_commit_requires_recovery', status=409, field_path='receipt')
                if process.get('plan'):
                    process['history'].append({key: copy.deepcopy(process.get(key)) for key in
                        ('plan', 'receipt', 'issue', 'material_identity', 'generation', 'l1_request_id', 'model_io_id', 'raw_output')})
                # An explicit owner action may prepare an older recorded
                # correction; merely reading historical rows never migrates it.
                row['proposal'].setdefault('contract', CONTRACT)
                row['proposal'].setdefault('origin', 'owner_l2')
                process.update(plan=None, receipt=None, issue=None, lease_until=0, scope={'statement_ids': [], 'unknown_scope': True})
                row['state'] = 'pending'
            elif action == 'apply':
                if not process.get('plan') or process['plan']['decision'] == 'ambiguous':
                    raise L1OwnerError('l3_prepared_plan_required', status=409, field_path='plan')
                row['state'] = 'applying'
            else:
                if (process.get('plan') or {}).get('decision') == 'ambiguous':
                    raise L1OwnerError('l3_ambiguous_requires_clarification', status=409, field_path='plan')
                row['state'] = 'applying' if process.get('plan') else 'pending'
                process['issue'] = None
            return row
        row, replayed = self.tasks.update(memory_id, revision, payload['expected_revision'], change,
            action=action, request_id=payload['request_id'])
        if row['state'] == 'applying':
            row = await self._apply(row, allow_wording_rebase=action == 'retry')
        return {'status': 'ok', 'proposal': SyncProposals.public(row), 'replayed': replayed}

    def _committed_receipt(self, memory_id, source_revision, process):
        request_id = process.get('l1_request_id') or self._request_id(memory_id, source_revision, process)
        prior = self.revisions.authority.get_commit_by_idempotency('program-l1:owner_l2:' + memory_id + ':' + request_id)
        if not prior or prior['status'] != 'committed':
            return None
        context = prior['payload'].get('metadata', {}).get('memory_l1_program_context') or {}
        if context.get('plan_sha256') != process['plan_sha256'] or context.get('owner_source_revision') != source_revision:
            raise L1OwnerError('l3_actual_commit_receipt_required', status=409)
        actual = self.revisions.authority.get_memory_revision(memory_id, prior['result']['revision'])
        return {'phase': 'l1_committed_l2_pending', 'l1_revision': actual['revision'],
            'l1_content_sha256': actual['metadata']['memory_l1_content_sha256'], 'l1_operation_id': actual['operation_id']}

    async def _apply(self, row, *, allow_wording_rebase=False):
        mid, revision = row['memory_id'], row['presentation_revision']
        process = row['proposal']['processing']
        try:
            receipt = process.get('receipt')
            identity = process['material_identity']
            plan = process['plan']
            request_id = process.get('l1_request_id') or self._request_id(mid, revision, process)
            if not receipt and plan['decision'] == 'revise_l1':
                receipt = self._committed_receipt(mid, revision, process)
            if receipt and allow_wording_rebase:
                # A later owner wording save is already authorized prose, but
                # is not evidence for another L1 change. An explicit recovery
                # may preserve that exact head after rechecking the original
                # committed L1 and the unchanged set of factual corrections.
                _material, _base, _evidence, latest_identity = await self._material(mid, revision)
                if (latest_identity['l1_revision'] != receipt['l1_revision']
                        or latest_identity['l1_content_sha256'] != receipt['l1_content_sha256']
                        or latest_identity['covered_proposals'] != identity['covered_proposals']
                        or latest_identity['source_fingerprint'] != identity['source_fingerprint']):
                    raise L1OwnerError('l1_dependency_changed', status=409)
                if latest_identity['owner_head_revision'] != identity['owner_head_revision']:
                    rebased = {**identity, 'owner_head_revision': latest_identity['owner_head_revision'],
                               'owner_head_sha256': latest_identity['owner_head_sha256']}
                    def keep_wording(current_row):
                        p = current_row['proposal']['processing']
                        p['history'].append({'recovery': 'preserve_later_owner_wording',
                            'material_identity': copy.deepcopy(p['material_identity']), 'receipt': receipt})
                        p['material_identity'] = rebased
                        return current_row
                    row, _ = self.tasks.update(mid, revision, process['revision'], keep_wording)
                    process, identity = row['proposal']['processing'], rebased
            if not receipt:
                material, base, evidence, current = await self._material(mid, revision)
                if current != identity:
                    raise L1OwnerError('l1_dependency_changed' if current['l1_revision'] != identity['l1_revision'] else 'owner_l2_changed', status=409)
                if plan['decision'] == 'revise_l1':
                    from memory_owner_l2_source import OwnerL2EditSourceReader
                    compiled = compile_plan(self.revisions, mid, base, plan, evidence, request_id=request_id)
                    main = next(item['binding'] for item in evidence if item['binding']['source_revision'] == revision)
                    ref = {key: main[key] for key in ('source_namespace', 'source_id', 'source_revision')}
                    context = {'origin': 'owner_l2', 'proposal_id': mid + ':' + str(revision), 'plan_sha256': process['plan_sha256'],
                        'owner_source_revision': revision, 'source_ref': ref,
                        'expected_owner_head_revision': identity['owner_head_revision'],
                        'expected_owner_head_sha256': identity['owner_head_sha256'],
                        'owner_source_refs': [{key: value['binding'][key] for key in ('source_namespace', 'source_id', 'source_revision')} for value in evidence]}
                    actual = await commit_plan(self.revisions, mid, compiled, expected_revision=identity['l1_revision'],
                        request_id=request_id, actor='owner', decision_source='owner_l2_fact_sync', program_context=context,
                        source_reader=OwnerL2EditSourceReader(self.revisions.authority), edit_note=plan['reason'])
                    receipt = {'phase': 'l1_committed_l2_pending', 'l1_revision': actual['revision'],
                        'l1_content_sha256': actual['l1_content_sha256'], 'l1_operation_id': actual['operation_id']}
                else:
                    receipt = {'phase': 'l1_already_correct', 'l1_revision': identity['l1_revision'],
                        'l1_content_sha256': identity['l1_content_sha256'], 'l1_operation_id': None}
                def preserve(current_row):
                    current_row['proposal']['processing']['receipt'] = receipt
                    current_row['state'] = 'l1_applied_l2_pending'
                    return current_row
                row, _ = self.tasks.update(mid, revision, process['revision'], preserve)
                process = row['proposal']['processing']
            _prompt, _route, generation = self.l2.generation_context()
            self.l2.store.complete_sync(mid, revision, expected_processing_revision=process['revision'], material_identity=identity,
                plan_sha256=process['plan_sha256'], receipt=receipt, generation=generation)
            return self.tasks.get(mid, revision)
        except Exception as exc:
            def failed(current_row):
                current_process = current_row['proposal']['processing']
                committed = bool((current_process.get('receipt') or {}).get('l1_operation_id'))
                current_row['state'] = 'l1_applied_l2_pending' if committed else 'conflict' if isinstance(exc, (RevisionConflict, L1OwnerError)) and getattr(exc, 'status', 409) == 409 else 'failed'
                current_process['issue'] = {'code': getattr(exc, 'code', type(exc).__name__),
                    'field_path': getattr(exc, 'field_path', ''), 'recovery_action': 'retry' if committed else 'prepare'}
                return current_row
            latest = self.tasks.get(mid, revision)
            if latest['state'] in CLOSED:
                return latest
            original_process = latest['proposal']['processing']
            if not original_process.get('receipt') and original_process.get('plan_sha256'):
                actual = self._committed_receipt(mid, revision, original_process)
                if actual:
                    def recovered(current_row):
                        current_row['proposal']['processing']['receipt'] = actual
                        current_row['state'] = 'l1_applied_l2_pending'
                        return current_row
                    latest, _ = self.tasks.update(mid, revision, original_process['revision'], recovered)
            try:
                row, _ = self.tasks.update(mid, revision, latest['proposal']['processing']['revision'], failed)
            except L1OwnerError:
                return latest
            return row

    async def run_once(self, max_jobs):
        if max_jobs <= 0:
            return {'status': 'disabled', 'results': []}
        results = []
        for pending in self.tasks.pending(max_jobs):
            mid, revision = pending['memory_id'], pending['presentation_revision']
            version = pending['proposal']['processing']['revision']
            def claim(row):
                if row['state'] != 'pending':
                    raise L1OwnerError('sync_phase_changed', status=409)
                row['state'] = 'preparing'
                row['proposal']['processing'].update(lease_until=time.time() + self.engine.daily_chat_memory_timeout_seconds + 60,
                    attempts=row['proposal']['processing']['attempts'] + 1, claim_token=uuid.uuid4().hex)
                return row
            try:
                row, _ = self.tasks.update(mid, revision, version, claim)
            except L1OwnerError:
                continue
            try:
                material, base, evidence, identity = await self._material(mid, revision)
                output = await prepare_plan(self.engine, material, material['owner_l2_text'], 'owner_l2', evidence,
                    memory_id=mid, task_id='l3-owner:' + mid + ':' + str(revision), attempt=row['proposal']['processing']['attempts'])
                plan = output['plan']
                plan_sha = semantic_sha256(plan)
                def checkpoint(current_row):
                    current_row['proposal']['processing'].update(plan=plan, plan_sha256=plan_sha, material_identity=identity,
                        generation=output['generation'], model_io_id=output['model_io_id'], raw_output=output['raw_output'])
                    current_row['proposal']['processing']['l1_request_id'] = self._request_id(mid, revision, current_row['proposal']['processing'])
                    return current_row
                row, _ = self.tasks.update(mid, revision, row['proposal']['processing']['revision'], checkpoint)
                request_id = row['proposal']['processing']['l1_request_id']
                compiled = compile_plan(self.revisions, mid, base, plan, evidence, request_id=request_id)
                # Owner edits or source changes during the model must not make
                # the old draft current. No second model call repairs this.
                _again, _base, _evidence, current = await self._material(mid, revision)
                if current != identity:
                    raise L1OwnerError('l1_dependency_changed', status=409)
                def prepared(current_row):
                    p = current_row['proposal']['processing']
                    p.update(plan=plan, plan_sha256=plan_sha, material_identity=identity, generation=output['generation'],
                        model_io_id=output['model_io_id'], raw_output=output['raw_output'], scope=compiled['scope'], lease_until=0, issue=None)
                    current_row['state'] = 'ambiguous' if plan['decision'] == 'ambiguous' else 'applying'
                    return current_row
                row, _ = self.tasks.update(mid, revision, row['proposal']['processing']['revision'], prepared)
                if row['state'] == 'applying':
                    row = await self._apply(row)
            except Exception as exc:
                def failed(current_row):
                    current_row['state'] = 'conflict' if getattr(exc, 'status', None) == 409 else 'failed'
                    current_row['proposal']['processing'].update(lease_until=0, issue={'code': getattr(exc, 'code', type(exc).__name__),
                        'field_path': getattr(exc, 'field_path', ''), 'recovery_action': 'prepare'})
                    return current_row
                latest = self.tasks.get(mid, revision)
                row, _ = self.tasks.update(mid, revision, latest['proposal']['processing']['revision'], failed)
            results.append(SyncProposals.public(row))
        return {'status': 'completed', 'results': results}


async def run_sync_tasks(engine, bucket_manager, *, max_jobs):
    from memory_l2_service import MemoryL2Service
    return await MemoryL3SyncService(MemoryL2Service(engine, engine._memory_l1_revisions(bucket_manager))).run_once(max_jobs)
