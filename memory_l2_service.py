"""Independent second-pass generation and owner access; never runs in chat."""
from __future__ import annotations

import asyncio
import copy
import json

from memory_authority import MemoryNotFound, RevisionConflict
from memory_l1_contract import L1ContractError, semantic_sha256
from memory_l1_reads import MemoryL1Reads
from memory_l1_service import L1OwnerError
from memory_l2_store import MemoryL2Store, validate_document
from memory_model_io import MemoryModelIO, completion_parameters

USES = {'current_basis', 'conditional_basis', 'relationship_reference', 'historical_evidence', 'unresolved_evidence'}


def semantic_input_projection(material):
    """Remove custody metadata from model input, retaining all semantic content.

    The full material is still loaded and verified before/after generation.
    Local evidence/entity/scene IDs and formal statement IDs remain unchanged.
    """
    def omit(value, fields):
        return {key: copy.deepcopy(item) for key, item in value.items() if key not in fields}
    def anchors(values):
        return [omit(anchor, {'target_id', 'target_namespace', 'target_revision',
                              'text_view_id', 'text_sha256', 'quote_sha256'})
                for anchor in values]
    def member(value, revision_field):
        result = omit(value, {'generation', revision_field})
        if 'anchors' in result:
            result['anchors'] = anchors(result['anchors'])
        return result
    doc = omit(material['l1_document'], {'generation', 'schema_version'})
    doc['entry'] = omit(doc['entry'], {'parent_ref', 'document_group_id', 'derived_from_revision'})
    for module, revision_field in (('statements', 'statement_revision_id'), ('scenes', 'scene_revision_id'),
                                   ('entities', 'entity_revision_id'), ('relations', 'relation_revision_id')):
        doc[module] = [member(item, revision_field) for item in doc.get(module, [])]
    # Source identity, revision and hashes are program custody. Evidence refs
    # retain the exact binding_id and offsets; full source prose stays below.
    doc['evidence_bindings'] = [{key: (anchors(binding[key]) if key == 'anchors' else copy.deepcopy(binding[key])) for key in
        ('binding_id', 'author_ref', 'speaker_ref', 'anchors') if key in binding}
        for binding in material['l1_document'].get('evidence_bindings', [])]
    sources = []
    for source in material['sources']:
        sources.append({'binding_id': source['binding']['binding_id'],
            **{key: copy.deepcopy(source.get(key)) for key in
               ('text', 'title', 'author', 'speaker', 'eligible_source', 'state', 'view', 'limitations')},
            'source_recorded_at': source.get('recorded_at')})
    result = {'l1_document': doc,
              'shared_statements': [member(item, 'statement_revision_id') for item in material['shared_statements']],
              'sources': sources, 'rings': [omit(item, {'memory_id', 'operation_id', 'content_sha256', 'ring_id'})
                                           for item in material['rings']]}
    # Keep custom meanings used by this document without resending the whole
    # generation catalog snapshot and its unused vocabulary.
    from memory_l1_semantic_wire import semantic_catalog
    used = set()
    def strings(value):
        if isinstance(value, str): used.add(value)
        elif isinstance(value, list):
            for item in value: strings(item)
        elif isinstance(value, dict):
            for item in value.values(): strings(item)
    strings(result)
    catalog = material['l1_document'].get('generation', {}).get('catalog_snapshot')
    definitions = {axis: {key: meaning for key, meaning in values.items() if key in used and meaning}
                   for axis, values in semantic_catalog(catalog).items()}
    result['semantic_definitions'] = {axis: values for axis, values in definitions.items() if values}
    return result


class MemoryL2Service:
    def __init__(self, engine, revisions):
        self.engine, self.revisions = engine, revisions
        self.store = MemoryL2Store(revisions.authority)

    def generation_context(self):
        prompt, info = self.engine._memory_l1_prompt('ombre.memory_l2_generation_prompt')
        route = self.engine._memory_task_route('presentation_l2')
        catalog, revision = self.revisions.catalog_provider()
        from memory_l2_schema import output_schema, schema_revision
        generation = {'prompt_revision': info['prompt_revision'], 'catalog_revision': 'catalog:' + semantic_sha256(catalog),
                      'catalog_digest': semantic_sha256(catalog),
                      'model_route_revision': semantic_sha256({k:v for k,v in route.items() if k != 'client'}),
                      'output_schema_revision':schema_revision(output_schema(getattr(self.engine, 'config', {})))}
        return prompt, route, generation

    def detail(self, memory_id, revision=None):
        _prompt, _route, generation = self.generation_context()
        dependency = self.store.current_dependency(memory_id, generation)
        status = self.store.detail(memory_id, dependency, revision)
        product = status['presentation']
        if product:
            record = self.revisions.authority.get_memory_revision(memory_id, product['first_pass_revision'])
            semantic = (record or {}).get('metadata', {}).get('memory_l1') or {}
            statements = list(semantic.get('statements', []))
            bindings = list(semantic.get('evidence_bindings', []))
            parent = semantic.get('entry', {}).get('parent_ref')
            if parent:
                root = self.revisions.authority.get_memory_revision(parent['memory_id'], parent['revision'])
                shared = (root or {}).get('metadata', {}).get('memory_l1') or {}
                refs = semantic['entry'].get('shared_statement_refs', [])
                statements += [s for s in shared.get('statements', []) if s['statement_id'] in refs]
                bindings += shared.get('evidence_bindings', [])
            product['annotation_refs'] = [{'statement_id':s['statement_id'], 'statement_revision_id':s['statement_revision_id']} for s in statements]
            product['source_spans'] = bindings
            product['required_time_and_conditions'] = [{'statement_id':s['statement_id'], 'subject':s['subject_ref'],
                'time':s['time'], 'conditions':s['conditions']} for s in statements]
            product['owner_edit_ref'] = product['request_id'] if product['actor'] == 'owner' else None
        # Head/history/task metadata belongs to the owner response envelope.
        # Keep the same read-only values inside l2 for revision-service callers.
        return {'status': 'ok', 'memory_id': memory_id, 'l2': status,
                **{key:status[key] for key in ('current_presentation_revision','owner_protected',
                    'revisions','jobs','sync_proposals','current_dependencies')}}

    @staticmethod
    def request(payload, required, optional=()):
        if not isinstance(payload, dict) or set(payload) - set(required) - set(optional) or not set(required) <= set(payload):
            raise L1OwnerError('invalid_l2_request')
        if 'request_id' in required and (not isinstance(payload['request_id'], str) or not payload['request_id'].strip() or len(payload['request_id']) > 128):
            raise L1OwnerError('invalid_request_id', field_path='request_id')
        for key in ('expected_l1_revision', 'expected_presentation_revision'):
            if key in payload and (type(payload[key]) is not int or payload[key] < (1 if key == 'expected_l1_revision' else 0)):
                raise L1OwnerError('invalid_revision', field_path=key)

    def prepare(self, memory_id, payload):
        self.request(payload, {'expected_l1_revision', 'request_id'})
        _prompt, _route, generation = self.generation_context()
        dependency = self.store.current_dependency(memory_id, generation)
        if dependency['first_pass_revision'] != payload['expected_l1_revision']: raise RevisionConflict('L1 changed')
        result = self.store.enqueue(memory_id, dependency, payload['request_id'])
        return {'status': 'ok', 'memory_id': memory_id, **result}

    async def save(self, memory_id, payload):
        self.request(payload, {'expected_l1_revision', 'expected_presentation_revision', 'request_id', 'document', 'change_kind'}, {'edit_note','confirm_sync_proposals'})
        if payload['change_kind'] not in ('wording', 'facts'): raise L1OwnerError('invalid_l2_change_kind')
        if 'edit_note' in payload and not isinstance(payload['edit_note'], str): raise L1OwnerError('string_required', field_path='edit_note')
        confirmations = payload.get('confirm_sync_proposals', [])
        if not isinstance(confirmations, list) or any(type(ref) is not int or ref <= 0 for ref in confirmations):
            raise L1OwnerError('invalid_sync_proposal_refs', field_path='confirm_sync_proposals')
        fingerprint = semantic_sha256(payload)
        prior = self.store.request_receipt(memory_id, payload['request_id'], fingerprint)
        if prior:
            return {'status':'ok', 'memory_id':memory_id, **prior,
                    'l2':self.detail(memory_id, prior['presentation_revision'])['l2']}
        base = await self.revisions.detail(memory_id)
        if base['l1']['state'] != 'current': raise L1OwnerError('l1_required', status=409)
        if base['current_revision'] != payload['expected_l1_revision']: raise RevisionConflict('L1 changed')
        document = base['l1']['document']
        # Scene children share their root's structured statements.
        ids = set(s['statement_id'] for s in document['statements']) | set(document['entry'].get('shared_statement_refs', []))
        validate_document(payload['document'], ids)
        _prompt, _route, generation = self.generation_context()
        dependency = self.store.current_dependency(memory_id, generation)
        if dependency['first_pass_revision'] != payload['expected_l1_revision']: raise RevisionConflict('L1 changed')
        receipt = self.store.save(memory_id, payload['document'], dependency, request_id=payload['request_id'],
            expected_revision=payload['expected_presentation_revision'], actor='owner', change_kind=payload['change_kind'],
            request_fingerprint=fingerprint, edit_note=payload.get('edit_note',''), confirm_sync_proposals=confirmations)
        return {'status': 'ok', 'memory_id': memory_id, **receipt,
                'l2': self.detail(memory_id, receipt['presentation_revision'])['l2']}

    async def _material(self, memory_id, expected):
        base = await self.revisions.detail(memory_id)
        if base['current_revision'] != expected or base['l1']['state'] != 'current':
            raise L1OwnerError('l1_dependency_changed', status=409)
        document = base['l1']['document']
        reads = MemoryL1Reads(self.revisions, self.engine._memory_l1_sources())
        bindings = reads.source_list(memory_id, expected)['sources']
        sources = []
        for binding in bindings:
            source = await reads.sources.locate(binding)
            if not source.get('eligible_source'):
                raise L1OwnerError('l2_source_not_available', status=409)
            sources.append(source)
        parent = document['entry'].get('parent_ref')
        shared = []
        if parent:
            root = self.revisions.authority.get_memory(parent['memory_id'])
            if not root or root['active_revision'] != parent['revision'] or root['state'] != 'active':
                raise L1OwnerError('l1_parent_dependency_changed', status=409)
            refs = document['entry'].get('shared_statement_refs', [])
            shared = [reads.index.member(parent['memory_id'], parent['revision'], 'statements', ref) for ref in refs]
            if any(s is None for s in shared): raise L1OwnerError('scene_dependency_mismatch', status=409)
        return {'l1_document': document, 'shared_statements': shared, 'sources': sources,
                'rings': [r for r in base['rings'] if r['state'] == 'active']}

    async def resolve(self, payload):
        self.request(payload, {'items'})
        if not isinstance(payload['items'], list): raise L1OwnerError('array_required', field_path='items')
        results = []
        for item in payload['items']:
            self.request(item, {'memory_id', 'revision', 'statement_ids', 'allowed_uses', 'narrative_allowed'}, {'statement_uses'})
            if (not isinstance(item['memory_id'], str) or type(item['revision']) is not int or item['revision'] < 1
                    or type(item['narrative_allowed']) is not bool or not isinstance(item['statement_ids'], list)
                    or any(not isinstance(s, str) for s in item['statement_ids'])
                    or not isinstance(item['allowed_uses'], list) or any(not isinstance(u, str) or u not in USES for u in item['allowed_uses'])
                    or not isinstance(item.get('statement_uses', {}), dict)
                    or any(not isinstance(k, str) or not isinstance(v, str) or v not in USES for k,v in item.get('statement_uses', {}).items())):
                raise L1OwnerError('invalid_l2_resolve_item', field_path='items')
            result = {'memory_id': item['memory_id'], 'state': 'not_prepared', 'text': '', 'units': [], 'presentation_revision': None, 'reason': 'l2_missing'}
            try:
                status = self.detail(item['memory_id'])['l2']
                result.update(state=status['state'], reason=status['reason'])
                product = status['presentation']
                if status['state'] == 'ready' and product['first_pass_revision'] == item['revision']:
                    # The Gateway has already applied this query's source/use
                    # eligibility. Check the source again only before use, not
                    # by calling a generation model or loading all L0 prose.
                    reads = MemoryL1Reads(self.revisions, self.engine._memory_l1_sources())
                    bindings = reads.source_list(item['memory_id'], item['revision'])['sources']
                    source_status = await reads.recall_source_status([{'memory_id': item['memory_id'], 'revision': item['revision'],
                        'binding_ids': [b['binding_id'] for b in bindings]}])
                    checked = source_status['items'][0]
                    if checked['state'] != 'ok' or any(not s.get('eligible_source') for s in checked['sources'].values()):
                        result.update(state='stale', reason='source_dependency_changed')
                    else:
                        doc = product['document']
                        identity = reads.index.identity(item['memory_id'], item['revision'])
                        refs = set(identity['entry'].get('shared_statement_refs', [])) if identity else set()
                        conn = self.revisions.authority._connect()
                        try:
                            refs.update(r[0] for r in conn.execute("SELECT object_id FROM memory_l1_members WHERE memory_id=? AND revision=? AND module='statements'", (item['memory_id'], item['revision'])))
                        finally: conn.close()
                        statement_uses = item.get('statement_uses', {})
                        allowed = {ref for ref in item['statement_ids'] if statement_uses.get(ref) in item['allowed_uses']}
                        variants = [v for v in doc['optional_use_variants']
                                    if v['intended_use'] in item['allowed_uses'] and set(v['statement_refs']) <= allowed
                                    and all(item.get('statement_uses', {}).get(ref) == v['intended_use'] for ref in v['statement_refs'])]
                        units = variants or [v for v in doc['coherent_sections'] if set(v['statement_refs']) <= allowed]
                        if product['units_state'] != 'current': units = []
                        result['units'] = [{'unit_id':v.get('variant_id', v.get('section_id')), 'text':v['text'],
                                            'statement_refs':v['statement_refs']} for v in units]
                        if item['narrative_allowed'] and refs <= allowed:
                            text = doc['full_injection_text']
                        else:
                            text = '\n\n'.join(dict.fromkeys(v['text'] for v in units))
                        result.update(text=text, presentation_revision=product['presentation_revision'], reason='' if text else 'no_compatible_l2_unit')
                        # Dependency may change while source IO is in flight.
                        latest = self.detail(item['memory_id'])['l2']
                        if latest['state'] != 'ready' or latest['current_presentation_revision'] != product['presentation_revision']:
                            result.update(state='stale', text='', units=[], reason='dependency_changed')
                elif status['state'] == 'ready':
                    result.update(state='stale', reason='l1_revision_mismatch')
            except MemoryNotFound:
                result.update(state='not_prepared', reason='memory_not_found')
            except (L1OwnerError, L1ContractError) as exc:
                result.update(state='stale', text='', units=[], reason=exc.code)
            results.append(result)
        return {'status': 'ok', 'items': results}

    async def run_once(self, limit):
        prompt, route, generation = self.generation_context()
        # Compare existing tracked L1 dependencies only. Never migrate a legacy
        # library or enqueue a second writer for every bucket on startup.
        for memory_id in self.store.tracked_ids():
            try:
                dep = self.store.current_dependency(memory_id, generation)
                status = self.store.detail(memory_id, dep)
                if status['state'] == 'stale' and dep.get('l1_content_sha256') and dep['memory_state'] == 'active' and dep['recall_policy'] != 'disabled':
                    self.store.enqueue(memory_id, dep, 'dependency:' + semantic_sha256(dep))
            except MemoryNotFound: continue
        if limit <= 0: return {'processed': 0, 'reason': 'disabled'}
        from memory_l1_runtime import task_options
        semaphore = asyncio.Semaphore(task_options(getattr(self.engine, 'config', {}))['memory_l2_concurrency'])
        results = []
        async def process(pending):
            async with semaphore:
                await process_claim(pending)
        async def process_claim(pending):
            try: dep = self.store.current_dependency(pending['memory_id'], generation)
            except MemoryNotFound:
                self.store.retire_missing(pending['task_id'])
                return
            job = self.store.claim(pending['task_id'], dep)
            if not job: return
            model_io, io_id = MemoryModelIO(self.revisions.authority), None
            try:
                if not prompt.strip(): raise L1OwnerError('l2_prompt_disabled')
                if not route['configured']: raise L1OwnerError('l2_model_unconfigured')
                full_material = await self._material(job['memory_id'], dep['first_pass_revision'])
                material = semantic_input_projection(full_material)
                from memory_l2_schema import output_schema, validate_output
                schema = output_schema(getattr(self.engine, 'config', {}))
                material['output_schema'] = schema
                messages = [{'role':'system','content':prompt},
                            {'role':'user','content':json.dumps(material, ensure_ascii=False)}]
                io_id = model_io.start(job['memory_id'], stage='presentation_l2', task_id=job['task_id'],
                    attempt=job['lease_token'], attempt_id=job['lease_token'], messages=messages,
                    parameters=completion_parameters(route, self.engine.temperature),
                    generation={**generation, 'dependencies': dep},
                    source_revisions=[source['binding'] for source in full_material['sources']])
                response = await self.engine._daily_chat_memory_create_completion(route['client'], model=route['model'],
                    messages=messages,
                    max_tokens=route['max_tokens'], temperature=self.engine.temperature, use_daily_client=route['connection']=='daily',
                    task_stage='presentation_l2', task_route=route)
                raw = self.engine._completion_content(response) or ''
                model_io.returned(io_id, response=response, raw_output=raw,
                                  finish_reason=self.engine._completion_finish_reason(response))
                if self.engine._completion_finish_reason(response) == 'length': raise L1OwnerError('l2_output_incomplete')
                if raw.strip().startswith('```'): raw = raw.strip().split('\n', 1)[-1].rsplit('```',1)[0]
                try: document = json.loads(raw)
                except (ValueError, TypeError): raise L1OwnerError('invalid_l2_json') from None
                if isinstance(document, dict) and 'review' not in document:
                    document['review'] = {'required':True, 'reasons':['review_assessment_missing']}
                ids = {s['statement_id'] for s in [*material['l1_document']['statements'], *material['shared_statements']]}
                validate_document(document, ids)
                validate_output(document, schema)
                # Recheck the full source material after a potentially long
                # provider call; a revoked or changed source cannot be revived.
                await self._material(job['memory_id'], dep['first_pass_revision'])
                if self.generation_context()[2] != generation: raise L1OwnerError('l2_generation_configuration_changed')
                result = self.store.save(job['memory_id'], document, dep, request_id=job['task_id'],
                    expected_revision=job['base_presentation_revision'], actor='model', job=job)
                model_io.finish(io_id)
                results.append({'task_id': job['task_id'], 'state': 'ready', **result})
            except asyncio.CancelledError:
                if io_id: model_io.finish(io_id, error=L1OwnerError('worker_interrupted'))
                self.store.finish(job, 'failed', 'worker_interrupted')
                raise
            except Exception as exc:
                if io_id: model_io.finish(io_id, error=exc)
                reason = getattr(exc, 'code', type(exc).__name__)
                state = 'stale' if isinstance(exc, RevisionConflict) else 'failed'
                self.store.finish(job, state, reason)
                results.append({'task_id': job['task_id'], 'state': state, 'reason': reason})
        await asyncio.gather(*(process(pending) for pending in self.store.pending(limit)))
        return {'processed': len(results), 'results': results}
