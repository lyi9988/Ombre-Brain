"""Owner-selected summary drafts on the original durable L1/Review/L2 chain.

The summary is a selection aid, never evidence or a fabricated tool turn.
Exact events retain their canonical identities; every input range is queued.
"""
from __future__ import annotations

import copy
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
import re

import httpx

from memory_authority import IdempotencyConflict, MemoryProposal, PolicyDecision
from memory_chat_preparation import ChatMemoryPreparation, issue
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_ingestion import bind_sources, prepare_model_document, source_snapshot
from memory_l1_service import L1OwnerError

PREFIX = 'memory_l1_owner_summary:'
SOURCE_PREFIX = 'memory_owner_summary_source:'


def task_id(operation_id):
    if not isinstance(operation_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', operation_id):
        raise L1OwnerError('invalid_owner_summary_operation', field_path='operation_id')
    return 'owner_summary_' + hashlib.sha256(operation_id.encode()).hexdigest()


def validate(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {'operation_id', 'summary', 'context'}:
        raise L1OwnerError('invalid_owner_summary_envelope', field_path='submission')
    task_id(envelope['operation_id'])
    summary, context = envelope['summary'], envelope['context']
    if (not isinstance(summary, dict) or set(summary) - {'id', 'revision', 'body', 'title', 'body_sha256'}
            or not {'id', 'revision', 'body', 'body_sha256'} <= set(summary)
            or not isinstance(summary['id'], str) or not summary['id']
            or type(summary['revision']) is not int or summary['revision'] < 1
            or not isinstance(summary['body'], str) or not summary['body'].strip()
            or summary['body_sha256'] != hashlib.sha256(summary['body'].encode()).hexdigest()
            or ('title' in summary and not isinstance(summary['title'], str))):
        raise L1OwnerError('invalid_owner_summary_draft', field_path='summary')
    if (not isinstance(context, dict) or set(context) - {'conversation_id', 'branch_id', 'profile_id', 'sources', 'attachment_sources'}
            or any(not isinstance(context.get(k), str) or not context[k] for k in ('conversation_id', 'branch_id', 'profile_id'))
            or not isinstance(context.get('sources'), list) or not context['sources']):
        raise L1OwnerError('invalid_owner_summary_context', field_path='context')
    seen = set()
    for source in context['sources']:
        if (not isinstance(source, dict) or set(source) - {'event_id', 'version_id', 'role', 'text', 'created_at', 'seq', 'text_sha256', 'attachment_refs'}
                or any(not isinstance(source.get(k), str) or not source[k] for k in ('event_id', 'version_id', 'role', 'text', 'created_at'))
                or source['role'] not in ('user', 'assistant', 'tool') or source['event_id'] in seen
                or source.get('text_sha256', hashlib.sha256(source['text'].encode()).hexdigest()) != hashlib.sha256(source['text'].encode()).hexdigest()):
            raise L1OwnerError('invalid_owner_summary_source', field_path='context.sources')
        try:
            stamp = datetime.fromisoformat(source['created_at'].replace('Z', '+00:00'))
            if stamp.tzinfo is None: raise ValueError()
        except ValueError:
            raise L1OwnerError('invalid_source_time', field_path='context.sources') from None
        seen.add(source['event_id'])
    if not isinstance(context.get('attachment_sources', []), list):
        raise L1OwnerError('invalid_attachment_sources', field_path='context.attachment_sources')


class OwnerSummarySourceReader:
    """Operation-scoped access proof, not a new identity for original L0."""
    def __init__(self, config, authority=None):
        self.config, self.authority = config, authority
        self.options = (config.get('gateway') or {}).get('canonical_continuation') or {}

    async def capture(self, operation_id, locator, version):
        token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
        if not token or not self.options.get('enabled') or not self.options.get('base_url'):
            raise L1OwnerError('owner_summary_source_reader_unconfigured', field_path='sources')
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(str(self.options['base_url']).rstrip('/') + '/bridge/v1/memory-source',
                params={'kind': 'owner_summary_source', 'source_id': canonical_json({'operation_id': operation_id, **locator}), 'version': version},
                headers={'Authorization': 'Bearer ' + token,
                    'X-Guyan-Device-ID': str(self.options.get('device_id') or 'ombre-gateway-operit')})
            response.raise_for_status()
            value = response.json().get('source', {})
        return value

    async def sources(self, envelope):
        from memory_source_provenance import canonical_source
        context, summary = envelope['context'], envelope['summary']
        sources = []
        for expected in context['sources']:
            value = await self.capture(envelope['operation_id'], {'event_id': expected['event_id']}, expected['version_id'])
            if (value.get('state') != 'available' or value.get('eligible_source') is not True
                    or value.get('text') != expected['text'] or value.get('role') != expected['role']
                    or value.get('source_revision') != expected['version_id']
                    or value.get('conversation_id') != context['conversation_id']
                    or value.get('branch_id') != context['branch_id']
                    or value.get('profile_id') != context['profile_id']
                    or any((value.get('summary') or {}).get(k) != summary[k] for k in ('id', 'revision', 'body_sha256'))):
                raise L1OwnerError('owner_summary_source_changed', status=409, field_path='sources')
            actual = {**expected, 'created_at': value.get('recorded_at') or expected['created_at']}
            source = canonical_source(actual, conversation_id=context['conversation_id'], profile_id=context['profile_id'])
            source['attachment_refs'] = copy.deepcopy(expected.get('attachment_refs', []))
            sources.append(source)
        for attachment in context.get('attachment_sources', []):
            from memory_attachment_source import normalize
            value = await self.capture(envelope['operation_id'], {'attachment_source_id': attachment['source_id']}, attachment['source_revision'])
            # The bridge returns the genuine attachment reader envelope here.
            raw = value.get('attachment_source', value)
            if (value.get('state') != 'available' or value.get('eligible_source') is not True
                    or raw.get('source_id') != attachment['source_id'] or raw.get('source_revision') != attachment['source_revision']):
                raise L1OwnerError('owner_summary_source_changed', status=409, field_path='attachment_sources')
            sources.append(normalize(raw, profile_id=context['profile_id'], conversation_id=context['conversation_id']))
        return sources

    def register(self, envelope, sources):
        """Keep the original reader identities; retain only an access locator."""
        if self.authority is None: return
        snapshots, bindings = bind_sources(sources, config=self.config)
        from memory_l0_sources import MemoryL0Sources
        for source, snapshot, binding in zip(sources, snapshots, bindings):
            MemoryL0Sources(self.authority).retain_copy(snapshot)
            locator = ({'attachment_source_id': snapshot['source_id']} if source['namespace'] == 'attachment_text'
                       else {'event_id': source['event_id']})
            key = SOURCE_PREFIX + semantic_sha256({k: binding[k] for k in ('source_namespace', 'source_id', 'source_revision')})
            value = {'operation_id': envelope['operation_id'], 'locator': locator, 'version': source['version_id'],
                'profile_id': source['profile_id'], 'conversation_id': source['conversation_id'], 'role': source['role']}
            with closing(self.authority._connect()) as db, db:
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                    (key, canonical_json(value)))

    async def read(self, binding):
        key = SOURCE_PREFIX + semantic_sha256({k: binding[k] for k in ('source_namespace', 'source_id', 'source_revision')})
        with closing(self.authority._connect()) as db, db:
            row = db.execute('SELECT value FROM authority_meta WHERE key=?', (key,)).fetchone()
        if not row: return {'state': 'unavailable', 'reason': 'owner_summary_access_missing'}
        access = json.loads(row['value'])
        value = await self.capture(access['operation_id'], access['locator'], access['version'])
        if value.get('state') != 'available' or value.get('eligible_source') is not True: return value
        if (value.get('source_revision') != access['version']
                or value.get('conversation_id') != access['conversation_id']
                or value.get('profile_id') != access['profile_id']):
            return {'state':'changed', 'reason':'owner_summary_source_identity_changed'}
        if 'event_id' in access['locator']:
            from memory_source_provenance import canonical_source
            source = canonical_source({'event_id': access['locator']['event_id'], 'version_id': access['version'],
                'text': value['text'], 'role': value['role'], 'created_at': value.get('recorded_at')},
                conversation_id=access['conversation_id'], profile_id=access['profile_id'])
        else:
            from memory_attachment_source import normalize
            source = normalize(value.get('attachment_source', value), profile_id=access['profile_id'], conversation_id=access['conversation_id'])
        return source_snapshot(source, config=self.config)


def windows(sources, budget, summary_body=''):
    """Disjoint original-text ranges, including every suffix of a long event."""
    result, current, used = [], [], 0
    for index, source in enumerate(sources):
        text = source_snapshot(source)['text']
        start = 0
        while start < len(text):
            if used == budget:
                result.append(current); current, used = [], 0
            end = min(len(text), start + budget - used)
            current.append({'source_index': index, 'start': start, 'end': end})
            used += end - start
            start = end
    if current: result.append(current)
    draft_ranges = [{'start': i, 'end': min(len(summary_body), i + budget)} for i in range(0, len(summary_body), budget)] or [{'start':0, 'end':0}]
    return [{'ranges': result[i % len(result)], 'draft_range': draft_ranges[i % len(draft_ranges)],
             'state': 'queued', 'candidate_ids': [], 'linked_memories': []}
            for i in range(max(len(result), len(draft_ranges)))]


class OwnerSummaryPreparation(ChatMemoryPreparation):
    prefix = PREFIX

    def __init__(self, authority, engine=None):
        super().__init__(authority)
        self.engine = engine

    @staticmethod
    def public(row):
        return {**ChatMemoryPreparation.public(row), 'source_kind': 'owner_summary'}

    def detail(self, mid):
        row = self.get(mid)
        if not row: raise L1OwnerError('task_not_found', status=404, field_path='task_id')
        return {'task': self.public(row), 'draft': copy.deepcopy(row['envelope']['summary']),
            'receipt': self.receipt(mid), 'history': [], 'editable': False,
            'preparation_output': {'windows': [{'ranges':copy.deepcopy(w['ranges']),
                'draft_range':copy.deepcopy(w['draft_range']), 'state':w['state'],
                'output':copy.deepcopy(w.get('output')), 'candidate_ids':list(w['candidate_ids'])}
                for w in row.get('windows') or []]}}

    def edit(self, mid, payload):
        raise L1OwnerError('owner_summary_edit_at_original_draft', status=409, field_path='draft')

    def enqueue(self, envelope, *, budget):
        validate(envelope)
        mid, digest = task_id(envelope['operation_id']), semantic_sha256(envelope)
        with closing(self.authority._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + mid,)).fetchone()
            if prior:
                if json.loads(prior['value'])['submission_hash'] != digest:
                    raise IdempotencyConflict('owner summary operation payload changed')
            else:
                # Window planning follows a verified capture, not source limits
                # belonging to chat-tool's unrelated turn contract.
                stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
                row = {'task_id': mid, 'operation_id': envelope['operation_id'], 'phase': 'queued',
                    'draft_revision': envelope['summary']['revision'], 'revision': 1, 'submission_hash': digest,
                    'envelope': copy.deepcopy(envelope), 'windows': None, 'window_budget': budget,
                    'created_at': stamp, 'updated_at': stamp, 'candidate_ids': [], 'attempts': 0,
                    'issue': None, 'lease_until': 0, 'history': [], 'edit_requests': {}}
                db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?)', (PREFIX + mid, canonical_json(row)))
        return self.receipt(mid)

    def receipt(self, mid):
        row = self.get(mid)
        if not row: raise L1OwnerError('task_not_found', status=404, field_path='operation_id')
        items, memory_ids, candidate_links = [], [], []
        for cid in row['candidate_ids']:
            candidate = self.authority.get_candidate(cid)
            memory = self.authority.get_memory(cid)
            state = candidate['status'] if candidate else 'unknown'
            detail = {'candidate_id': cid, 'state': state}
            if state == 'linked':
                from memory_summary_coverage import summary_link_receipt
                candidate_links.extend(summary_link_receipt(self.authority, cid))
            if memory and memory['state'] == 'active' and memory['active_revision'] > 0:
                revision = int(memory['active_revision'])
                memory_ids.append(cid)
                with closing(self.authority._connect()) as db, db:
                    product = db.execute('SELECT p.presentation_revision,p.first_pass_revision,p.publication FROM memory_l2_heads h '
                        'JOIN memory_l2_revisions p USING(memory_id,presentation_revision) WHERE h.memory_id=?', (cid,)).fetchone()
                    job = db.execute('SELECT state,reason FROM memory_l2_jobs WHERE memory_id=? AND first_pass_revision=? ORDER BY updated_at DESC LIMIT 1', (cid, revision)).fetchone()
                l2_state = ('requires_dependency_check' if product and product['first_pass_revision'] == revision and product['publication'] == 'active'
                            else job['state'] if job else 'pending')
                if self.engine is not None:
                    from memory_l2_service import MemoryL2Service
                    l2_state = MemoryL2Service(self.engine, self.engine._memory_l1_revisions()).detail(cid)['l2']['state']
                detail.update(memory_id=cid, l1_revision=revision, l2_state=l2_state,
                    presentation_revision=product['presentation_revision'] if product else None)
            items.append(detail)
        completed = [w for w in row.get('windows') or [] if w['state'] == 'prepared']
        linked = [m for w in completed for m in w.get('linked_memories', [])] + candidate_links
        if row['phase'] == 'source_changed': status = 'source_changed'
        elif row['phase'] == 'failed': status = 'partial' if completed else 'failed'
        elif row['phase'] != 'prepared': status = 'partial' if completed else 'queued'
        elif any(i['state'] in ('pending', 'deferred') for i in items): status = 'review'
        elif any(i['state'] in ('rejected', 'commit_failed', 'unknown') or i.get('l2_state') == 'failed' for i in items): status = 'partial' if memory_ids or linked else 'failed'
        elif any(i['state'] == 'accepted' and not i.get('memory_id') for i in items): status = 'queued'
        elif any(i.get('memory_id') for i in items) and all(i.get('l2_state') == 'ready' or i['state'] == 'linked' for i in items): status = 'ready'
        elif memory_ids: status = 'l1_accepted_l2_pending'
        elif linked: status = 'linked'
        else: status = 'no_new_memory'
        effective_issue = row.get('issue')
        if row['phase'] == 'queued' and self.engine is not None:
            if self.engine.daily_chat_memory_mode == 'off':
                status, effective_issue = 'disabled', {'code':'memory_source_disabled','field_path':'mode'}
            elif (self.engine.config.get('reflection') or {}).get('chat_memory_jobs_per_run', 1) == 0:
                status, effective_issue = 'blocked', {'code':'memory_worker_disabled','field_path':'chat_memory_jobs_per_run'}
            elif not self.engine._memory_l1_prompt('ombre.daily_chat_memory_prompt', runtime_values={'max_candidates':max(1,int(self.engine.daily_chat_memory_max_per_day or 1))})[0].strip():
                status, effective_issue = 'blocked', {'code':'memory_prompt_disabled','field_path':'prompt'}
            elif not self.engine._memory_task_route('auto_l1')['configured']:
                status, effective_issue = 'blocked', {'code':'memory_model_unconfigured','field_path':'auto_l1'}
        code = (effective_issue or {}).get('code', '')
        retryable = (row['phase'] == 'failed' and code not in ('execution_outcome_unknown', 'memory_source_disabled',
            'owner_summary_source_changed', 'source_not_currently_readable', 'statement_source_not_currently_readable')
            and bool(row.get('windows')) and any(w['state'] != 'prepared' for w in row['windows']))
        return {'ok': True, 'operation_id': row['operation_id'], 'task_id': mid, 'status': status,
            'task_revision':row['revision'], 'retryable':retryable,
            'source_kind': 'owner_summary', 'preparation_status': row['phase'], 'candidate_ids': row['candidate_ids'],
            'memory_ids': memory_ids, 'memories': items, 'linked_memories': linked, 'issue': effective_issue,
            'windows': {'completed': len(completed), 'total': len(row.get('windows') or [])},
            'recall_guaranteed': False}

    def retry(self, operation_id, expected_revision):
        mid = task_id(operation_id)
        row = self.get(mid)
        if not row: raise L1OwnerError('task_not_found', status=404, field_path='operation_id')
        if type(expected_revision) is not int:
            raise L1OwnerError('invalid_retry_revision', field_path='expected_task_revision')
        if row['phase'] in ('queued','generating'):
            return self.receipt(mid)
        if row['revision'] != expected_revision:
            raise L1OwnerError('submission_revision_conflict', status=409, field_path='expected_task_revision')
        if not self.receipt(mid)['retryable']:
            raise L1OwnerError('owner_summary_not_retryable', status=409, field_path='status')
        def reset(current):
            for window in current['windows']:
                if window['state'] == 'prepared': continue
                if window.get('output'):
                    window.setdefault('failed_outputs', []).append(window.pop('output'))
            current.update(phase='queued', issue=None, lease_until=0, claim_token=None)
            return current
        self.update(mid, expected_revision=expected_revision, transform=reset)
        return self.receipt(mid)

    def put_window(self, claimed, index, proposals, links):
        """One transaction publishes all candidates and the completed range."""
        with closing(self.authority._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            raw = db.execute('SELECT value FROM authority_meta WHERE key=?', (PREFIX + claimed['task_id'],)).fetchone()
            current = json.loads(raw['value'])
            if current['revision'] != claimed['revision'] or current.get('claim_token') != claimed.get('claim_token'):
                raise L1OwnerError('submission_revision_conflict', status=409, field_path='revision')
            for proposal, policy in proposals:
                self.authority._put_candidate(db, proposal, policy)
            ids = [p.proposal_id for p, _ in proposals]
            current['windows'][index].update(state='prepared', candidate_ids=ids, linked_memories=links)
            current['candidate_ids'] = list(dict.fromkeys([*current['candidate_ids'], *ids]))
            current.update(revision=current['revision'] + 1, phase='prepared' if all(w['state'] == 'prepared' for w in current['windows']) else 'queued',
                lease_until=0, updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            db.execute('UPDATE authority_meta SET value=? WHERE key=?', (canonical_json(current), PREFIX + current['task_id']))
            return current


async def prepare_owner_summary_tasks(engine, bucket_mgr, *, max_jobs=None):
    from memory_l1_runtime import task_options
    from memory_l1_semantic_wire import assemble_document, semantic_contract, semantic_sources
    from memory_l1_schema import split_document
    from memory_model_io import MemoryModelIO, completion_parameters
    from memory_source_identity import source_model_identity
    from memory_l1_api import validate_response
    from memory_summary_coverage import check_summary_coverage, related_summary_statements
    options = task_options(engine.config)
    allowance = options['chat_memory_jobs_per_run'] if max_jobs is None else min(options['chat_memory_jobs_per_run'], max_jobs)
    if not engine.memory_authority_enabled or not engine.memory_authority_store or allowance <= 0:
        return {'status': 'disabled', 'results': []}
    if getattr(engine, '_owner_summary_preparation_running', False): return {'status': 'locked', 'results': []}
    limit = max(1, int(engine.daily_chat_memory_max_per_day or 1))
    prompt, context = engine._memory_l1_prompt('ombre.daily_chat_memory_prompt', runtime_values={'max_candidates': limit})
    route = engine._memory_task_route('auto_l1')
    if engine.daily_chat_memory_mode == 'off' or not prompt.strip() or not route['configured']:
        return {'status': 'disabled', 'results': []}
    tasks, io = OwnerSummaryPreparation(engine.memory_authority_store, engine), MemoryModelIO(engine.memory_authority_store)
    reader = OwnerSummarySourceReader(engine.config, engine.memory_authority_store)
    results = []
    engine._owner_summary_preparation_running = True
    try:
        for pending in tasks.list(limit=allowance, phase='queued', private=True):
            try: claimed = tasks.claim(pending, lease_seconds=engine.daily_chat_memory_timeout_seconds + 60)
            except L1OwnerError: continue
            io_id = None
            try:
                # A process lost during a model call has an unknown outcome.
                # Do not silently buy a second call when its lease expires.
                if (pending['phase'] == 'generating' and any(w['state'] != 'prepared' for w in pending.get('windows') or [])
                        and not any(w.get('output') for w in pending.get('windows') or [] if w['state'] != 'prepared')):
                    raise L1OwnerError('execution_outcome_unknown', field_path='generation')
                sources = await reader.sources(claimed['envelope'])
                reader.register(claimed['envelope'], sources)
                for cid in claimed['candidate_ids']:
                    existing = engine.memory_authority_store.get_candidate(cid)
                    if existing and existing['status'] == 'accepted':
                        await engine._memory_l1_adopt_candidate(existing, bucket_mgr, request_id='owner-summary:' + cid)
                if claimed['windows'] and all(w['state'] == 'prepared' for w in claimed['windows']):
                    tasks.finish(claimed, phase='prepared', issue=None)
                    results.append(tasks.receipt(claimed['task_id']))
                    continue
                if claimed['windows'] is None:
                    claimed = tasks.update(claimed['task_id'], expected_revision=claimed['revision'], transform=lambda row: {**row,
                        'windows': windows(sources, row['window_budget'], row['envelope']['summary']['body'])})
                index = next(i for i, w in enumerate(claimed['windows']) if w['state'] != 'prepared')
                window = claimed['windows'][index]
                selected = []
                for extent in window['ranges']:
                    source = copy.deepcopy(sources[extent['source_index']])
                    source['_model_range'] = {k: extent[k] for k in ('start', 'end')}
                    selected.append(source)
                snapshots, bindings = bind_sources(selected, config=engine.config)
                for source, snapshot in zip(selected, snapshots): source['_participant_revision'] = snapshot['source_revision']
                service = engine._memory_l1_revisions(bucket_mgr)
                saved = window.get('output')
                if saved:
                    catalog, catalog_revision, generation, raw = saved['catalog'], saved['catalog_revision'], saved['generation'], saved['text']
                    io_id = saved.get('model_io_id')
                else:
                    catalog, catalog_revision = service.catalog_provider()
                    identity = source_model_identity(engine.config, selected)
                    views = semantic_sources(snapshots, bindings)
                    for view, extent, source in zip(views, window['ranges'], selected):
                        view.update(text=view['text'][extent['start']:extent['end']],
                            observed_range={k: extent[k] for k in ('start', 'end')}, range_coordinates='original_source_text',
                            attachment_refs=copy.deepcopy(source.get('attachment_refs', [])))
                    summary = claimed['envelope']['summary']
                    # The selected draft remains complete in the durable task.
                    # Each draft segment is delivered alongside a source range;
                    # it never acquires an evidence binding.
                    contract = semantic_contract([b['binding_id'] for b in bindings], catalog=catalog,
                        initial_chat=True, require_keywords=True, identity=identity)
                    contract['document']['required'].append('narrative')
                    contract['document']['narrative'] = contract['document']['optional'].pop('narrative')
                    payload = {'input_kind': 'owner_selected_summary', 'identity': identity,
                        'owner_selected_summary': {'id': summary['id'], 'revision': summary['revision'], 'title': summary.get('title', ''),
                            'body': summary['body'][window['draft_range']['start']:window['draft_range']['end']],
                            'observed_range':window['draft_range'], 'total_characters':len(summary['body']), 'evidence': False},
                        'window': {'index': index, 'total': len(claimed['windows']), 'max_candidates': limit},
                        'existing_source_memories': {'comparison_only': True,
                            'memories': related_summary_statements(engine.memory_authority_store, bindings)},
                        'l1_contract': {'source_texts': views, 'catalog_revision': catalog_revision, 'identity': identity,
                            'semantic_contract': contract}}
                    generation = {'prompt_revision': context['prompt_revision'], 'model_route_revision': 'route_sha256:' + semantic_sha256({k:v for k,v in route.items() if k != 'client'}),
                        'task_id': claimed['task_id'] + ':window:' + str(index), 'actor': 'owner_summary_preparation'}
                    messages = [{'role':'system','content':prompt}, {'role':'user','content':json.dumps(payload, ensure_ascii=False)}]
                    io_id = io.start(claimed['task_id'], stage='auto_l1', task_id=generation['task_id'], attempt=claimed['attempts'], attempt_id=claimed['claim_token'],
                        messages=messages, parameters=completion_parameters(route, engine.temperature), generation=generation, source_revisions=bindings)
                    response = await engine._daily_chat_memory_create_completion(route['client'], model=route['model'], messages=messages,
                        max_tokens=engine.daily_chat_memory_candidate_max_tokens, temperature=engine.temperature,
                        use_daily_client=route['connection'] == 'daily', task_stage='auto_l1', task_route=route, model_io_id=io_id)
                    raw = engine._completion_content(response) or ''
                    io.returned(io_id, response=response, raw_output=raw, finish_reason=engine._completion_finish_reason(response))
                    output = {'text':raw, 'catalog':catalog, 'catalog_revision':catalog_revision, 'generation':generation, 'model_io_id':io_id,
                        'finish_reason':engine._completion_finish_reason(response)}
                    def preserve(row):
                        row['windows'][index]['output'] = output
                        return row
                    claimed = tasks.update(claimed['task_id'], expected_revision=claimed['revision'], transform=preserve)
                    if output['finish_reason'] != 'stop': raise L1OwnerError('first_pass_output_incomplete', field_path='generation')
                    validate_response(response)
                parsed = engine._parse_json_object(raw)
                if saved and saved.get('finish_reason') != 'stop':
                    raise L1OwnerError('first_pass_output_incomplete', field_path='generation')
                if not isinstance(parsed, dict) or set(parsed) != {'candidates'} or not isinstance(parsed['candidates'], list):
                    raise L1OwnerError('model_candidate_fields_required', field_path='candidates')
                proposals, links = [], []
                for number, candidate in enumerate(parsed['candidates']):
                    if not isinstance(candidate, dict) or not isinstance(candidate.get('document'), dict):
                        raise L1OwnerError('first_pass_document_required', field_path='candidates')
                    cid = claimed['task_id'] + '_w' + str(index) + '_c' + str(number)
                    document = assemble_document(candidate['document'], require_keywords=True)
                    full = prepare_model_document(service, memory_id=cid, document=document, sources=selected,
                        generation=generation, request_id=claimed['operation_id'] + ':' + str(index) + ':' + str(number),
                        catalog_context=(catalog, catalog_revision), config=engine.config, candidate_refs=candidate.get('source_refs', []))
                    body, semantic = split_document(full)
                    if not semantic['statements']: continue
                    proposal = MemoryProposal.from_mapping({'proposal_id':cid, 'source_type':'owner_summary', 'proposed_body':body,
                        'source_refs':['owner_summary:' + claimed['envelope']['summary']['id'] + '@' + str(claimed['draft_revision']),
                            *['l0:' + b['source_namespace'] + ':' + b['source_id'] + '@' + b['source_revision'] for b in semantic['evidence_bindings']]],
                        'source_status':'verified', 'requested_mode':engine.daily_chat_memory_mode, 'owner_explicit':False,
                        'memory_type':candidate.get('kind', 'durable_fact'), 'confidence':candidate.get('confidence', 0.7),
                        'metadata':{'memory_l1':semantic, 'owner_summary':{'operation_id':claimed['operation_id'],
                            'summary_id':claimed['envelope']['summary']['id'], 'draft_revision':claimed['draft_revision'], 'window':index},
                            'legacy_candidate':{'title':full['narrative']['title'], 'tags':['owner_summary']}}})
                    coverage = check_summary_coverage(engine.memory_authority_store, proposal)
                    if coverage['status'] == 'linked':
                        links.extend(coverage['matches']); continue
                    proposal.metadata['owner_summary_coverage'] = coverage
                    policy = engine.memory_ingestion_policy.evaluate(proposal)
                    if coverage['status'] == 'overlap_review' and policy.action != 'reject':
                        policy = PolicyDecision('queue_review', ('REVIEW_EXISTING_SOURCE_COVERAGE',), 'review', requires_owner_confirmation=True)
                    proposals.append((proposal, policy))
                # Revalidate the frozen operation and every original source
                # after the await, then publish this range and all its drafts.
                after = await reader.sources(claimed['envelope'])
                if semantic_sha256({'sources':after}) != semantic_sha256({'sources':sources}):
                    raise L1OwnerError('owner_summary_source_changed', status=409, field_path='sources')
                current = tasks.put_window(claimed, index, proposals, links)
                for proposal, _ in proposals:
                    row = engine.memory_authority_store.get_candidate(proposal.proposal_id)
                    if row['status'] == 'accepted':
                        await engine._memory_l1_adopt_candidate(row, bucket_mgr, request_id='owner-summary:' + proposal.proposal_id)
                if io_id: io.finish(io_id)
                results.append(tasks.receipt(current['task_id']))
            except Exception as exc:
                current = tasks.get(claimed['task_id'])
                if io_id and current and not any(w.get('output') for w in current.get('windows') or [] if w['state'] != 'prepared'):
                    exc = L1OwnerError('execution_outcome_unknown', field_path='generation')
                if io_id: io.finish(io_id, error=exc)
                if current and current.get('claim_token') == claimed.get('claim_token'):
                    tasks.finish(current, phase='source_changed' if getattr(exc, 'code', '') == 'owner_summary_source_changed' else 'failed', issue=issue(exc))
                results.append(tasks.receipt(claimed['task_id']))
        return {'status':'completed', 'results':results}
    finally:
        engine._owner_summary_preparation_running = False


async def owner_summary_response(request, engine, authority):
    """Private owner-service ingress; ordinary memory writer is insufficient."""
    import hmac
    from starlette.responses import JSONResponse
    from memory_l1_runtime import task_options
    headers = {'Cache-Control': 'no-store'}
    token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
    authorization = request.headers.get('authorization', '')
    if not token or not authorization.startswith('Bearer ') or not hmac.compare_digest(authorization[7:], token):
        return JSONResponse({'error':'unauthorized'}, status_code=401, headers=headers)
    if authority is None or engine is None or not engine.memory_authority_enabled:
        return JSONResponse({'error':'memory_authority_unavailable'}, status_code=503, headers=headers)
    try:
        tasks = OwnerSummaryPreparation(authority, engine)
        if request.method == 'POST':
            payload = await request.json()
            # No narrative/source clipping or chat-tool request-size cap: a
            # long explicit selection is durably processed over bounded jobs.
            if isinstance(payload, dict) and payload.get('retry') is True:
                if set(payload) != {'operation_id','retry','expected_task_revision'}:
                    raise L1OwnerError('invalid_owner_summary_retry', field_path='retry')
                row = tasks.get(task_id(payload['operation_id']))
                if not row: raise L1OwnerError('task_not_found', status=404, field_path='operation_id')
                await OwnerSummarySourceReader(engine.config, authority).sources(row['envelope'])
                receipt = tasks.retry(payload['operation_id'],payload['expected_task_revision'])
            else:
                receipt = tasks.enqueue(payload, budget=task_options(engine.config)['daily_chat_memory_window_max_input_chars'])
        else:
            mid = task_id(request.path_params['operation_id'])
            row = tasks.get(mid)
            if not row: raise L1OwnerError('task_not_found', status=404, field_path='operation_id')
            try:
                await OwnerSummarySourceReader(engine.config, authority).sources(row['envelope'])
            except L1OwnerError as exc:
                if exc.code != 'owner_summary_source_changed': raise
                tasks.update(mid, expected_revision=row['revision'], transform=lambda value: {**value,
                    'phase':'source_changed', 'issue':issue(exc), 'lease_until':0, 'claim_token':None})
            receipt = tasks.receipt(mid)
        return JSONResponse(receipt, status_code=202 if request.method == 'POST' else 200, headers=headers)
    except IdempotencyConflict:
        return JSONResponse({'error':'idempotency_conflict'}, status_code=409, headers=headers)
    except L1OwnerError as exc:
        return JSONResponse({'error':exc.code, 'field_path':exc.field_path}, status_code=exc.status, headers=headers)
    except (ValueError, TypeError, KeyError, UnicodeError):
        return JSONResponse({'error':'invalid_owner_summary_request'}, status_code=400, headers=headers)
    except Exception:
        return JSONResponse({'error':'execution_outcome_unknown'}, status_code=503, headers=headers)
