"""First-pass task adapters on ReflectionEngine's existing clients/policy.

No per-chat retrieval model call is introduced. Daily/diary jobs keep their
existing scheduling and Auto/Review control; source and task clocks are distinct.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from memory_l1_contract import canonical_json, make_text_anchor, semantic_sha256
from memory_l1_ingestion import bind_sources, expand_model_document, model_template, prepare_model_document
from memory_l1_schema import document_template, join_document, split_document
from memory_l1_service import L1OwnerError, MemoryL1RevisionService
from memory_source_provenance import daily_sources


GENERATION_OPTIONS = {'diary_l1_input_window_chars': (64000, 256, 2000000),
    'diary_l1_max_windows': (32, 1, 10000), 'diary_l1_scene_limit': (0, 0, 10000),
    'diary_l1_max_tokens': (16000, 512, 200000), 'diary_l1_scene_min_value': (0.0, 0.0, 1.0),
    'daily_chat_memory_candidate_max_tokens': (16000, 300, 200000),
    'daily_chat_memory_max_windows_per_run': (10, 1, 10000),
    'daily_chat_memory_window_max_input_chars': (12000, 256, 2000000),
    'daily_chat_memory_max_per_day': (10, 0, 10000),
    'daily_chat_memory_review_max_per_day': (10, 0, 10000)}
GENERATION_OPTIONS['action_l1_jobs_per_run'] = (1, 0, 200)


def task_options(config):
    cfg = config.get('reflection') or {}
    result = {}
    for key, (default, low, high) in GENERATION_OPTIONS.items():
        value = cfg.get(key, default)
        if type(value) not in ((float, int) if type(default) is float else (int,)) or not low <= value <= high:
            raise L1OwnerError('invalid_generation_configuration', field_path='reflection.' + key)
        result[key] = value
    return result


def namespace_local_ids(document, prefix, known_ids=(), catalog=None):
    doc = copy.deepcopy(document)
    ids = {}
    for module, key in (('entities', 'entity_id'), ('scenes', 'scene_id'), ('statements', 'statement_id'), ('relations', 'relation_id')):
        for item in doc[module]:
            if not isinstance(item, dict) or not isinstance(item.get(key), str):
                raise L1OwnerError('new_identity_requires_local_key', field_path='document.' + module)
            if item[key].startswith('new:'):
                ids[item[key]] = 'new:' + prefix + ':' + item[key][4:]
            elif item[key] in known_ids: ids[item[key]] = item[key]
            else: raise L1OwnerError('unknown_existing_identity', field_path='document.' + module)
    # Rewrite typed references only. A literal name/title/body equal to a local
    # key is still text and must not be accidentally rewritten as an identity.
    for module, key in (('entities', 'entity_id'), ('scenes', 'scene_id'), ('statements', 'statement_id'), ('relations', 'relation_id')):
        for item in doc[module]: item[key] = ids.get(item[key], item[key])
    for item in doc['statements']:
        value = item['subject_ref']['value']
        item['subject_ref']['value'] = ids.get(value, value)
        value = item['time']['applicability']['scene_ref']
        item['time']['applicability']['scene_ref'] = ids.get(value, value)
    for scene in doc['scenes']: scene['statement_refs'] = [ids.get(ref, ref) for ref in scene['statement_refs']]
    for relation in doc['relations']:
        for endpoint in ('from', 'to'):
            value = relation[endpoint]['object_id']
            relation[endpoint]['object_id'] = ids.get(value, value)
    definitions = {item['id']: item for item in (catalog or {}).get('attributes', [])}
    def typed_attributes(node):
        if isinstance(node, list):
            for item in node: typed_attributes(item)
        elif isinstance(node, dict):
            definition = definitions.get(node.get('definition_id', node.get('key'))) or {}
            value = node.get('value')
            if definition.get('value_type') == 'entity_ref' and isinstance(value, dict) and value.get('state') == 'known':
                value['value'] = ids.get(value['value'], value['value'])
            definition = definitions.get(node.get('field')) or {}
            if definition.get('value_type') == 'entity_ref':
                value = node.get('value')
                node['value'] = [ids.get(v, v) for v in value] if node.get('op') == 'in' else ids.get(value, value)
            for value in node.values(): typed_attributes(value)
    for module in ('attributes', 'statements', 'scenes'): typed_attributes(doc[module])
    return doc, ids


class MemoryL1Runtime:
    async def _memory_l1_action_tasks(self, bucket_mgr, *, mode=None):
        """Consume source IDs in an existing daily job, never execute actions."""
        from memory_authority import MemoryProposal
        from memory_l1_source_readers import AizActionSourceReader
        from memory_l1_source_tasks import MemoryL1SourceTasks
        from memory_l1_wire_schema import model_document_schema
        tasks = MemoryL1SourceTasks(self.memory_authority_store)
        options = task_options(self.config)
        maximum = options['action_l1_jobs_per_run']
        mode = mode or self.daily_chat_memory_mode
        if maximum == 0 or mode == 'off' or self._daily_chat_memory_candidate_limit(mode) == 0:
            return {'status': 'disabled', 'results': []}
        prompt, context = self._memory_l1_prompt('ombre.daily_chat_memory_prompt',
            runtime_values={'max_candidates': self._daily_chat_memory_candidate_limit(mode)})
        if not prompt.strip(): return {'status': 'disabled', 'reason': 'prompt_disabled', 'results': []}
        client, model, use_daily = self._daily_chat_memory_model_client(candidate=True)
        if client is None: return {'status': 'not_prepared', 'reason': 'model_unconfigured', 'results': []}
        reader = AizActionSourceReader(self.config)
        service = self._memory_l1_revisions(bucket_mgr)
        catalog, catalog_revision = service.catalog_provider()
        results = []
        for task in tasks.list(limit=maximum, phase='queued'):
            claimed = tasks.transition(task['task_id'], 'queued', 'generating')
            if claimed is None: continue
            try:
                source = await reader.capture(task['action_id'], task['artifact_id'])
                if source.get('state') != 'available' or source.get('eligible_source') is not True:
                    raise L1OwnerError('action_source_not_available', status=409, field_path='source')
                raw_source = {'namespace': 'action_receipt', 'system': 'aizizhu',
                    'conversation_id': reader.options.get('conversation_id', 'conv-jiajia-main'),
                    'profile_id': self.memory_semantic_profile_id, 'event_id': task['action_id'],
                    'version_id': task['artifact_id'], 'role': 'tool', 'text': source['text'],
                    'recorded_at': source.get('recorded_at')}
                snapshots, bindings = bind_sources([raw_source])
                payload = {'input_kind': 'completed_action_receipt', 'conversation_turns': [],
                    'l1_contract': {'document_schema': model_document_schema([b['binding_id'] for b in bindings]), 'catalog': catalog,
                        'catalog_revision': catalog_revision, 'document_template': model_template(catalog=catalog),
                        'source_bindings': bindings, 'source_texts': [{'binding_id': 'source:1', 'text': source['text'],
                            'author': source.get('author'), 'coverage': source.get('coverage')}]} }
                response = await self._daily_chat_memory_create_completion(client, model=model,
                    messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                    max_tokens=self.daily_chat_memory_candidate_max_tokens, temperature=self.temperature, use_daily_client=use_daily)
                if self._completion_finish_reason(response) == 'length':
                    raise L1OwnerError('first_pass_output_incomplete', field_path='generation')
                parsed = self._parse_json_object(self._completion_content(response) or '')
                candidates = parsed.get('candidates') if isinstance(parsed, dict) else None
                if not isinstance(candidates, list) or len(candidates) > self._daily_chat_memory_candidate_limit(mode):
                    raise L1OwnerError('action_task_result_invalid', field_path='candidates')
                prepared = []
                for index, item in enumerate(candidates):
                    if not isinstance(item, dict) or not isinstance(item.get('document'), dict):
                        raise L1OwnerError('model_document_fields_required', field_path='candidates')
                    confidence = item.get('confidence', 0)
                    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise L1OwnerError('invalid_confidence', field_path='candidates.confidence')
                    if type(item.get('sensitive', False)) is not bool:
                        raise L1OwnerError('boolean_required', field_path='candidates.sensitive')
                    memory_id = 'action_l1_' + hashlib.sha256((task['task_id'] + ':' + str(index)).encode()).hexdigest()[:40]
                    generation = {'prompt_revision': context['prompt_revision'], 'model_route_revision': 'route_sha256:'
                        + semantic_sha256({'model': model, 'daily_client': use_daily}),
                        'task_id': task['task_id'], 'actor': 'action_first_pass'}
                    doc = prepare_model_document(service, memory_id=memory_id, document=item['document'], sources=[raw_source],
                        generation=generation, request_id=task['task_id'] + ':' + str(index), catalog_context=(catalog, catalog_revision))
                    body, semantic = split_document(doc)
                    proposal = MemoryProposal.from_mapping({'proposal_id': memory_id, 'source_type': 'action_receipt',
                        'proposed_body': body, 'source_status': 'verified',
                        'source_refs': ['action_receipt:' + task['action_id'] + '@' + task['artifact_id']],
                        'memory_type': item.get('kind', 'key_event'), 'confidence': item.get('confidence', 0),
                        'sensitive': item.get('sensitive', False), 'requested_mode': mode,
                        'metadata': {'memory_l1': semantic, 'source_task_id': task['task_id'],
                            'legacy_candidate': {'title': doc['narrative']['title'], 'content': body, 'proposed_memory': body}}})
                    prepared.append(proposal)
                ids = []
                for proposal in prepared:
                    row = self.memory_authority_store.put_candidate(proposal, self.memory_ingestion_policy.evaluate(proposal))
                    ids.append(row['candidate_id'])
                tasks.transition(task['task_id'], 'generating', 'prepared', candidate_ids=ids)
                adoption_issues = []
                for ident in ids:
                    row = self.memory_authority_store.get_candidate(ident)
                    if row['status'] == 'accepted':
                        try:
                            await self._memory_l1_adopt_candidate(row, bucket_mgr, request_id=task['task_id'], owner=False)
                        except Exception as exc:
                            adoption_issues.append({'candidate_id': ident, 'code': getattr(exc, 'code', type(exc).__name__)})
                states = {ident: self.memory_authority_store.get_candidate(ident)['status'] for ident in ids}
                results.append({'task_id': task['task_id'], 'status': 'prepared', 'candidate_ids': ids,
                    'candidate_states': states, 'adoption_issues': adoption_issues,
                    'adopted': bool(ids) and all(state == 'committed' for state in states.values())})
            except Exception as exc:
                tasks.transition(task['task_id'], 'generating', 'failed',
                    issue={'code': getattr(exc, 'code', type(exc).__name__)})
                results.append({'task_id': task['task_id'], 'status': 'failed',
                    'issue': {'code': getattr(exc, 'code', type(exc).__name__)}})
        return {'status': 'completed', 'results': results}

    def _memory_l1_revisions(self, bucket_mgr=None):
        from memory_system_config import configuration, merge_config, read_overlay, runtime_overlay_path
        def catalog_provider():
            snap = configuration(merge_config(self.config, read_overlay(runtime_overlay_path(self.config))))
            return snap['values']['catalog'], snap['revision']
        return MemoryL1RevisionService(self.memory_authority_store,
            self._memory_authority_service(bucket_mgr) if bucket_mgr is not None else None, bucket_mgr, catalog_provider)

    def _memory_l1_prompt(self, source_id, *, runtime_values=None):
        from prompt_source_registry import resolve_fixed_prompt
        body, meta = resolve_fixed_prompt(self.prompt_plan_mirror, source_id, config=self.config,
            identity_id=str(self.config.get('persona', {}).get('canonical_session_id') or 'jiajia-main'),
            conversation_id='', runtime_values=runtime_values)
        return body, {'prompt_revision': 'prompt_sha256:' + hashlib.sha256(body.encode()).hexdigest(),
                      'binding': meta.get('binding'), 'selected_blocks': meta.get('selected_blocks')}

    def _memory_l1_daily_context(self, window):
        from memory_l1_wire_schema import model_document_schema
        service = self._memory_l1_revisions()
        catalog, catalog_revision = service.catalog_provider()
        sources = daily_sources(window, profile_id=self.memory_semantic_profile_id)
        snapshots, bindings = bind_sources(sources)
        return {'document_schema': model_document_schema([b['binding_id'] for b in bindings]), 'document_template': model_template(catalog=catalog),
                'catalog': catalog, 'catalog_revision': catalog_revision,
                'source_bindings': bindings,
                'source_texts': [{'binding_id': b['binding_id'], 'text': s['text'], 'author': s['author'],
                    'recorded_at': s['recorded_at']} for s, b in zip(snapshots, bindings)]}, sources

    def _memory_l1_daily_document(self, candidate, memory_id):
        sources = candidate.get('_l1_sources')
        generation = candidate.get('_l1_generation')
        if not sources or not generation:
            raise L1OwnerError('generated_task_context_missing', field_path='document')
        return prepare_model_document(self._memory_l1_revisions(), memory_id=memory_id,
            document=candidate['document'], sources=sources, generation=generation, request_id=generation['task_id'],
            catalog_context=candidate.get('_l1_catalog_context'))

    def _memory_l1_sources(self):
        from memory_l0_sources import MemoryL0Sources, McpDiarySourceAdapter
        adapters = []
        if self.diary_mcp_url:
            adapters.append(McpDiarySourceAdapter(self.diary_mcp_url, self._read_diary_for_date))
        # Other program source readers are added by the same service factory.
        from memory_l1_source_readers import configured_source_readers
        adapters.extend(configured_source_readers(self.config, self.memory_authority_store))
        kind = (self.config.get('reflection') or {}).get('diary_l1_source_kind', 'mcp')
        namespace = 'snapshot.ombre.diary_mcp' if kind == 'mcp' else 'snapshot.aiz.diary.' + kind
        return MemoryL0Sources(self.memory_authority_store, adapters, diary_namespace=namespace)

    async def _memory_l1_adopt_candidate(self, row, bucket_mgr, *, edit=None, request_id, owner=False):
        from memory_candidate_edit import save_owner_candidate
        from memory_diary_l1 import DiaryL1Service
        service = self._memory_l1_revisions(bucket_mgr)
        candidate_id = row['candidate_id']
        if row['status'] == 'committed': return {'id': candidate_id, 'status': 'exists'}
        if edit:
            row = save_owner_candidate(self.memory_authority_store, candidate_id, edit=edit,
                expected_revision=row['revision'], request_id=request_id + ':edit', revision_service=service)
        wants_accept = owner and row['status'] in ('pending', 'deferred', 'commit_failed')
        if row['status'] != 'accepted' and not wants_accept: return {'id': candidate_id, 'status': row['status']}
        proposal = row['proposal']
        if 'diary_l1_bundle' in proposal.get('metadata', {}):
            if wants_accept:
                binding = proposal['metadata']['diary_l1_bundle']['source_binding']
                if not (await self._memory_l1_sources().locate(binding))['eligible_source']:
                    raise L1OwnerError('source_not_currently_readable', status=409, field_path='evidence_bindings')
                row = self.memory_authority_store.decide_candidate(candidate_id, action='accept', expected_revision=row['revision'],
                    request_id=request_id + ':accept', actor='owner', reason_codes=['OWNER_ACCEPTED'])
            result = await DiaryL1Service(service, self._memory_l1_sources(), self.memory_ingestion_policy).adopt(candidate_id)
            return {'id': candidate_id, 'status': 'created' if result.get('root') else result['status'], 'bundle': result}
        semantic = proposal['metadata']['memory_l1']
        commit_revision = row['revision'] + 1 if wants_accept else row['revision']
        key = 'l1-candidate:' + candidate_id + ':revision:' + str(commit_revision)
        prior_commit = self.memory_authority_store.get_commit_by_idempotency(key)
        # Check the actual source on a later Review adoption as well as at the
        # authenticated initial capture. A historical copy never grants access.
        source_observations = None
        if not (prior_commit and prior_commit['status'] == 'committed'):
            readers = self._memory_l1_sources()
            source_observations = {}
            for binding in semantic['evidence_bindings']:
                observed = await readers.locate(binding)
                source_observations[binding['binding_id']] = {'state': observed['state'],
                    'eligible_source': observed['eligible_source'], 'reason': observed.get('reason')}
            eligible = {ident for ident, observed in source_observations.items() if observed['eligible_source']}
            if not eligible: raise L1OwnerError('source_not_currently_readable', status=409, field_path='evidence_bindings')
            def check_evidence(node, path):
                if isinstance(node, list):
                    for index, item in enumerate(node): check_evidence(item, path + '[' + str(index) + ']')
                elif isinstance(node, dict):
                    refs = node.get('evidence_refs')
                    if refs and not eligible.intersection(refs):
                        raise L1OwnerError('statement_source_not_currently_readable', status=409, field_path=path + '.evidence_refs')
                    for key, item in node.items():
                        if key != 'evidence_refs': check_evidence(item, path + '.' + key)
            # Alternatives supporting the same statement need one readable
            # source, not all. Unused window context cannot block adoption.
            check_evidence(semantic['statements'], 'statements')
            check_evidence(semantic['attributes'], 'attributes')
        if wants_accept:
            row = self.memory_authority_store.decide_candidate(candidate_id, action='accept', expected_revision=row['revision'],
                request_id=request_id + ':accept', actor='owner', reason_codes=['OWNER_ACCEPTED'])
        metadata = copy.deepcopy(proposal['metadata'])
        if source_observations is not None: metadata['source_observations_at_adoption'] = source_observations
        metadata.pop('legacy_candidate', None)
        legacy = proposal['metadata'].get('legacy_candidate') or {}
        metadata.update({'name': semantic['narrative']['title'], 'tags': legacy.get('tags', []),
            'memory_type': proposal['memory_type'], 'source': proposal['source_type']})
        existing = self.memory_authority_store.get_memory(candidate_id)
        if existing is not None:
            if prior_commit is None:
                raise L1OwnerError('candidate_memory_already_changed', status=409, field_path='memory_id')
        if prior_commit:
            original = prior_commit['payload']
            metadata = original['metadata']
            body = original['body'] if 'body' in original else proposal['proposed_body']
            actor = original.get('actor') or ('owner' if owner else proposal['source_type'])
            decision_source = original['decision_source']
        else:
            body = proposal['proposed_body']
            actor = 'owner' if owner else proposal['source_type']
            decision_source = 'owner' if owner else 'auto'
        committed = await service.commit.commit_memory(memory_id=candidate_id, bucket_id=candidate_id,
            expected_revision=0, body=body, metadata=metadata,
            source_refs=proposal['source_refs'], decision_source=decision_source,
            idempotency_key=key, actor=actor, l1_write_intent=True)
        latest = self.memory_authority_store.get_candidate(candidate_id)
        if latest['status'] == 'accepted':
            self.memory_authority_store.decide_candidate(candidate_id, action='commit', expected_revision=latest['revision'],
                request_id=key + ':committed', actor='commit_service', reason_codes=['MEMORY_REVISION_COMMITTED'])
        return {'id': candidate_id, 'status': 'created', 'revision': committed['revision']}

    async def _memory_l1_diary_task(self, key, bucket_mgr):
        from memory_diary_l1 import DiaryL1Service
        from memory_l1_wire_schema import model_document_schema
        options = task_options(self.config)
        prompt, prompt_context = self._memory_l1_prompt('ombre.diary_memory_prompt')
        if not prompt.strip(): return {'status': 'disabled', 'reason': 'prompt_disabled'}
        if not self.client: return {'status': 'not_prepared', 'reason': 'first_pass_model_unconfigured'}
        sources = self._memory_l1_sources()
        captured = await sources.capture_diary(key, binding_id='diary-source')
        source, binding = captured['source'], captured['binding']
        if binding is None: return {'status': 'not_prepared', 'reason': 'source_unavailable'}
        service = self._memory_l1_revisions(bucket_mgr)
        catalog, catalog_revision = service.catalog_provider()
        # Freeze the vocabulary actually shown to this job; an owner hot edit
        # during generation applies to the next job, not half of this one.
        service = MemoryL1RevisionService(service.authority, service.commit, service.buckets,
            lambda: (copy.deepcopy(catalog), catalog_revision))
        width, max_windows = options['diary_l1_input_window_chars'], options['diary_l1_max_windows']
        text = source['text']
        combined = model_template(catalog=catalog)
        prose, scene_bodies, input_ranges, confidence = [], {}, [], []
        batch_identity = {'source': source['source_revision'], 'prompt': prompt_context, 'options': options,
                          'catalog_revision': catalog_revision, 'model': self.model}
        task_id = 'diary-task-' + semantic_sha256(batch_identity)[:40]
        model_route = 'route_sha256:' + semantic_sha256({'model': self.model, 'thinking_mode': self.thinking_mode})
        memory_id = 'diary_full_' + hashlib.sha256(source['source_id'].encode()).hexdigest()[:32]
        candidate_id = 'diary-l1:' + memory_id + ':' + task_id
        original_candidate = self.memory_authority_store.get_candidate(candidate_id)
        if original_candidate:
            organizer = DiaryL1Service(service, sources, self.memory_ingestion_policy)
            replay = await organizer.adopt(candidate_id) if original_candidate['status'] in ('accepted', 'committed') else {'status': original_candidate['status']}
            return {'status': replay['status'], 'candidate_id': candidate_id, 'task_id': task_id, 'replayed': True, 'result': replay}
        previous_memory = self.memory_authority_store.get_memory(memory_id)
        previous = (await service.detail(memory_id))['l1']['document'] if previous_memory else None
        known_ids = {item[key] for module, key in (('entities', 'entity_id'), ('scenes', 'scene_id'),
            ('statements', 'statement_id'), ('relations', 'relation_id')) for item in (previous or {}).get(module, [])}
        failures = []
        for index, start in enumerate(range(0, len(text), width)):
            if index >= max_windows: break
            end = min(start + width, len(text))
            anchor = make_text_anchor(text, target_namespace='l0', target_id=source['source_id'],
                target_revision=source['source_revision'], text_view_id=source['text_view_id'], start=start, end=end)
            payload = {'date': key, 'source': {k: v for k, v in source.items() if k != 'text'},
                'source_text': text[start:end], 'input_range': {'start': start, 'end': end}, 'source_anchor': anchor,
                'binding_id': binding['binding_id'], 'model_document_schema': model_document_schema([binding['binding_id']], entry_forms=('episode',)),
                'catalog': catalog, 'catalog_revision': catalog_revision, 'options': options,
                'existing_members': {module: (previous or {}).get(module, []) for module in ('entities', 'scenes', 'statements', 'relations')},
                'window_index': index, 'source_length': len(text)}
            try:
                response = await self.client.chat.completions.create(model=self.model,
                    messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                    **self._completion_options(max_tokens=options['diary_l1_max_tokens'], temperature=self.temperature))
                if self._completion_finish_reason(response) == 'length': raise ValueError('first_pass_output_incomplete')
                parsed = self._parse_json_object(self._completion_content(response) or '')
                if not isinstance(parsed, dict) or set(parsed) != {'document', 'scene_bodies', 'confidence'}:
                    raise ValueError('diary_task_result_required')
                draft, local_map = namespace_local_ids(parsed['document'], 'w' + str(index), known_ids, catalog)
                candidate_base = copy.deepcopy(previous) if previous else document_template(catalog=catalog, catalog_revision=catalog_revision)
                window_binding = copy.deepcopy(binding)
                window_binding['anchors'] = [anchor]
                candidate_base['evidence_bindings'] = [window_binding]
                expanded = expand_model_document(draft, candidate_base, {binding['binding_id']: text})
                from memory_l1_schema import validate_l1_document
                validate_l1_document(expanded, catalog=catalog)
                # Store only validated program anchors; the model draft's
                # ranges are kept until the final expansion below.
                for scene, validated_scene in zip(draft['scenes'], expanded['scenes']):
                    scene['source_refs'] = []
                    scene['source_ranges'] = [{'source_ref': binding['binding_id'], 'start': a['start'], 'end': a['end']}
                        for a in validated_scene['anchors']]
                if not isinstance(parsed['scene_bodies'], dict): raise ValueError('scene_bodies_required')
                expected_scene_keys = {s['scene_id'] for s in parsed['document']['scenes']}
                if set(parsed['scene_bodies']) != expected_scene_keys: raise ValueError('scene_body_membership_mismatch')
                staged_scene_bodies = {}
                for key_local, body in parsed['scene_bodies'].items():
                    if not isinstance(body, str) or not body.strip(): raise ValueError('scene_body_required')
                    staged_scene_bodies[local_map[key_local]] = body
                limit = options['diary_l1_scene_limit']
                if limit and len(combined['scenes']) + len(draft['scenes']) > limit:
                    raise ValueError('configured_scene_limit_exceeded')
                value = parsed['confidence']
                if type(value) not in (int, float) or not 0 <= value <= 1: raise ValueError('invalid_confidence')
                duplicate_ids = {s['scene_id'] for s in combined['scenes']} & {s['scene_id'] for s in draft['scenes']}
                if duplicate_ids: raise ValueError('scene_repeated_across_windows')
                scene_bodies.update(staged_scene_bodies)
                confidence.append(value)
                prose.append(draft['narrative']['body'])
                for module in ('entities', 'scenes', 'statements', 'relations', 'attributes'):
                    combined[module].extend(draft[module])
                for field in ('keywords', 'tags'):
                    combined['retrieval'][field] = list(dict.fromkeys(combined['retrieval'].get(field, []) + draft['retrieval'].get(field, [])))
                if not combined['narrative']['title']: combined['narrative']['title'] = draft['narrative']['title']
                input_ranges.append({'start': start, 'end': end})
            except Exception as exc:
                failures.append({'window': index, 'reason': type(exc).__name__, 'code': getattr(exc, 'code', None)})
                break
        if not prose:
            self.memory_authority_store.set_projection_status(memory_id='diary_task_' + key, memory_revision=0,
                projector='memory_l1_first_pass', status='degraded', details={'state': 'failed', 'task_id': task_id,
                    'failures': failures, 'prompt_revision': prompt_context['prompt_revision']})
            return {'status': 'failed', 'reason': 'no_complete_first_pass_window', 'failures': failures, 'task_id': task_id}
        combined['narrative']['body'] = '\n\n'.join(prose)
        base = copy.deepcopy(previous) if previous else document_template(catalog=catalog, catalog_revision=catalog_revision)
        base['evidence_bindings'] = [binding]
        full = expand_model_document(combined, base, {binding['binding_id']: text})
        organizer = DiaryL1Service(service, sources, self.memory_ingestion_policy)
        row = await organizer.prepare(day=key, document=full, scene_bodies=scene_bodies,
            generation={'prompt_revision': prompt_context['prompt_revision'], 'model_route_revision': model_route,
                'task_id': task_id, 'source_revision': source['source_revision']}, input_ranges=input_ranges,
            request_id=task_id, confidence=min(confidence), requested_mode=self.daily_chat_memory_mode)
        result = await organizer.adopt(row['candidate_id']) if row['status'] == 'accepted' else {'status': row['status']}
        return {'status': result['status'], 'candidate_id': row['candidate_id'], 'task_id': task_id,
            'coverage': row['proposal']['metadata']['diary_l1_bundle']['coverage'], 'failures': failures, 'result': result}
