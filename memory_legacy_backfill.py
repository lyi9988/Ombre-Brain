"""Review-only, same-ID L1 preparation over the existing Memory Authority.

This is a targeted proposal adapter, not a batch runner or another writer.
Models provide semantics; configured readers verify every original source.
The existing commit coordinator retains history and queues the normal L2 job.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

import frontmatter

from memory_authority import (IdempotencyConflict, MemoryNotFound, MemoryProposal,
                              PolicyDecision, RevisionConflict)
from memory_commit_service import ProjectionNotApplied, body_sha256
from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_ingestion import bind_sources, prepare_model_document
from memory_l1_schema import split_document
from memory_l1_semantic_wire import assemble_document
from memory_l1_service import L1OwnerError


def _candidate_id(memory_id, expected_revision):
    return 'legacy_backfill_' + semantic_sha256({'memory_id': memory_id, 'revision': expected_revision})[:40]


def _check_base(authority, db, memory_id, base):
    memory = db.execute('SELECT * FROM memories WHERE memory_id=?', (memory_id,)).fetchone()
    if not memory:
        raise MemoryNotFound(memory_id)
    if memory['state'] != 'active':
        raise L1OwnerError('backfill_memory_not_active', status=409)
    if (memory['active_revision'] != base['expected_revision']
            or memory['body_sha256'] != base['original_body_sha256']
            or authority._memory_state_generation(db, memory_id) != base['state_generation']):
        raise RevisionConflict('legacy backfill target changed')
    record = db.execute('SELECT metadata_json FROM memory_revisions WHERE memory_id=? AND revision=?',
                        (memory_id, base['expected_revision'])).fetchone()
    if not record or 'memory_l1' in json.loads(record['metadata_json']):
        raise L1OwnerError('backfill_requires_legacy_l1', status=409)
    return dict(memory)


def validate_commit_context(authority, db, payload):
    """Run inside original prepare/finalize transactions, including recovery."""
    context = payload['metadata']['memory_l1_program_context']
    candidate = authority._candidate_row(db.execute('SELECT * FROM candidates WHERE candidate_id=?',
                                                   (context['proposal_id'],)).fetchone())
    if (not candidate or candidate['status'] != 'accepted'
            or candidate['revision'] != context.get('candidate_revision')
            or candidate['proposal_sha256'] != context.get('proposal_sha256')):
        raise RevisionConflict('legacy backfill review changed before adoption')
    proposal = candidate['proposal']
    base = proposal['metadata'].get('legacy_backfill') or {}
    if (base.get('memory_id') != payload['memory_id']
            or base.get('expected_revision') != payload['expected_revision']
            or proposal.get('source_type') != 'legacy_backfill'
            or proposal.get('requested_mode') != 'review'
            or payload['metadata'].get('memory_l1') != proposal['metadata'].get('memory_l1')
            or payload['body_sha256'] != body_sha256(proposal['proposed_body'])):
        raise RevisionConflict('legacy backfill accepted document changed')
    _check_base(authority, db, payload['memory_id'], base)


class LegacyMemoryBackfill:
    def __init__(self, revisions, source_reader_factory, *, config=None):
        self.revisions, self.authority = revisions, revisions.authority
        self.source_reader_factory = source_reader_factory
        self.config = config

    async def _projection_base(self, memory_id, memory, base):
        projected = await self.revisions.buckets.get(memory['bucket_id'])
        if not projected or body_sha256(projected.get('content') or '') != base['original_body_sha256']:
            raise L1OwnerError('backfill_projection_changed', status=409)
        markers = projected.get('metadata', {})
        if type(markers.get('memory_revision')) is int and markers['memory_revision'] == base['expected_revision']:
            return projected, None
        if 'memory_revision' not in markers:
            from memory_legacy_projection import verify_unversioned_legacy
            try:
                proof, _ = verify_unversioned_legacy(self.authority, self.revisions.commit.projection, projected,
                    memory_id=memory_id, bucket_id=memory['bucket_id'], expected_revision=base['expected_revision'])
                return projected, proof
            except ProjectionNotApplied:
                pass
        raise L1OwnerError('backfill_projection_changed', status=409)

    def _readers(self):
        readers = self.source_reader_factory()
        # New backfill must prove its speaker. Ordinary historical reads keep
        # their existing compatibility when an older reply lacks author data.
        adapter = getattr(readers, 'adapters', {}).get('snapshot.aiz.canonical_event')
        if adapter is not None:
            adapter.require_author_identity = True
        return readers

    def _source_links(self, memory_id, revision):
        """Only original revisions and exactly linked original candidate history."""
        links, candidates = [], {memory_id}
        def collect(metadata, refs, origin):
            for ref in refs or []:
                if isinstance(ref, str):
                    links.append((ref, origin))
                    kind, _, ident = ref.partition(':')
                    if kind in {'candidate', 'proposal', 'daily_chat_candidate'} and ident:
                        candidates.add(ident)
            if not isinstance(metadata, dict):
                return
            for key in ('daily_chat_memory_candidate_id', 'candidate_id', 'proposal_id', 'source_candidate_id'):
                if isinstance(metadata.get(key), str) and metadata[key]:
                    candidates.add(metadata[key])
            for key, kind in (('source_raw_event_ids', 'raw_event'), ('source_event_ids', 'raw_event'),
                              ('source_conversation_turn_ids', 'conversation_turn'), ('source_turn_ids', 'conversation_turn')):
                values = metadata.get(key)
                if isinstance(values, list):
                    links.extend((kind + ':' + str(value), origin) for value in values if type(value) in (str, int))
            annotations = metadata.get('semantic_annotations')
            for item in (annotations.get('source_coverage', []) if isinstance(annotations, dict) else []):
                if isinstance(item, dict) and item.get('namespace') in {'canonical_event', 'raw_event', 'conversation_turn'} and item.get('event_id'):
                    ref = item['namespace'] + ':' + str(item['event_id'])
                    if item.get('version_id'):
                        ref += '@' + str(item['version_id'])
                    links.append((ref, {**origin, 'source_coverage': {k: item.get(k) for k in
                        ('system', 'namespace', 'profile_id', 'conversation_id', 'event_id', 'version_id', 'role', 'snapshot_sha256')}}))
            legacy = metadata.get('legacy_candidate')
            if isinstance(legacy, dict):
                collect(legacy, legacy.get('source_refs', []), origin)
        with closing(self.authority._connect()) as db:
            rows = db.execute('SELECT revision,source_refs_json,metadata_json FROM memory_revisions WHERE memory_id=? AND revision<=? ORDER BY revision',
                              (memory_id, revision)).fetchall()
            for row in rows:
                collect(json.loads(row['metadata_json']), json.loads(row['source_refs_json']), {'memory_revision': row['revision']})
            # Do not follow candidate-to-candidate guesses or similar text.
            for cid in sorted(candidates):
                rows = db.execute("SELECT revision,proposal_json FROM candidate_revisions WHERE candidate_id=? AND status='committed' ORDER BY revision", (cid,)).fetchall()
                for row in rows:
                    proposal = json.loads(row['proposal_json'])
                    collect(proposal.get('metadata', {}), proposal.get('source_refs', []),
                            {'candidate_id': cid, 'candidate_revision': row['revision']})
        return links

    def _witnesses(self, memory_id, base, document, *, sources=None, previous=None):
        from memory_source_provenance import digest, raw_event_source
        from raw_events import RawEventStore
        links = self._source_links(memory_id, base['expected_revision'])
        previous = {row['binding_id']: row for row in previous or []}
        witnesses = []
        for binding in document['evidence_bindings']:
            locator = json.loads(binding['source_id'])
            namespace = {'snapshot.ombre.raw_event': 'raw_event', 'snapshot.ombre.conversation_turn': 'conversation_turn',
                         'snapshot.aiz.canonical_event': 'canonical_event'}.get(binding['source_namespace'])
            snapshot = self.source_reader_factory().copy(binding)
            if not namespace or not snapshot:
                raise L1OwnerError('backfill_source_not_in_original_memory', status=409, field_path='sources')
            ref = namespace + ':' + str(locator['event_id'])
            if namespace == 'canonical_event':
                ref += '@' + str(snapshot.get('upstream_revision') or '')
            match = next(((value, origin) for value, origin in links if value == ref), None)
            if match is None and namespace == 'canonical_event':
                # Some old refs stored only an event id; their bound source
                # coverage, when present, must still agree with this version.
                match = next(((value, origin) for value, origin in links if value == 'canonical_event:' + str(locator['event_id'])), None)
            archive = ((sources[int(binding['binding_id'].split(':')[1]) - 1].get('archive_ref') or {}) if sources
                       else previous.get(binding['binding_id'], {}).get('archive_ref') or {})
            if match is None and namespace == 'canonical_event' and archive.get('namespace') == 'raw_event':
                raw_ref = 'raw_event:' + str(archive.get('event_id') or '')
                original = next(((value, origin) for value, origin in links if value == raw_ref), None)
                config = self.config or {}
                state = config.get('state_dir') or str(Path(config.get('buckets_dir', 'buckets')).resolve().parent / 'state')
                path = Path((config.get('raw_events') or {}).get('db_path') or str(Path(state) / 'raw_events.sqlite'))
                if original and path.is_file():
                    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                        db.row_factory = sqlite3.Row
                        row = db.execute('SELECT * FROM raw_events WHERE id=?', (str(archive.get('event_id')),)).fetchone()
                    resolved = raw_event_source(RawEventStore._row_to_event(row), profile_id=locator['profile_id']) if row else {}
                    if (resolved.get('bridge_status') == 'exact_runtime_bridge' and resolved.get('text') == snapshot['text']
                            and all(resolved.get(key) == locator.get(key) for key in ('system', 'profile_id', 'conversation_id', 'event_id', 'role'))
                            and resolved.get('version_id') == snapshot.get('upstream_revision')):
                        match = original
            if match is None:
                raise L1OwnerError('backfill_source_not_in_original_memory', status=409, field_path='sources')
            coverage = match[1].get('source_coverage')
            if coverage and (any(coverage.get(k) not in (None, '', locator.get(k)) for k in
                    ('system', 'profile_id', 'conversation_id', 'event_id', 'role'))
                    or coverage.get('version_id') not in (None, '', snapshot.get('upstream_revision'))
                    or coverage.get('snapshot_sha256') not in (None, '', digest(snapshot['text']))):
                raise L1OwnerError('backfill_original_source_identity_changed', status=409, field_path='sources')
            witnesses.append({'binding_id': binding['binding_id'], 'source_ref': match[0],
                              'authority_path': match[1], **({'archive_ref': archive} if archive else {})})
        return witnesses

    @staticmethod
    def _validate(payload):
        required = {'expected_revision', 'request_id', 'original_body_sha256',
                    'document', 'sources', 'generation', 'confidence'}
        if (not isinstance(payload, dict) or not required <= set(payload)
                or set(payload) - required - {'sensitive', 'reason', 'source_refs',
                    'expected_candidate_revision', 'expected_proposal_sha256', 'kind', 'kind_generation'}
                or type(payload['expected_revision']) is not int or payload['expected_revision'] < 1
                or not isinstance(payload['request_id'], str) or not 1 <= len(payload['request_id'].strip()) <= 128
                or not isinstance(payload['original_body_sha256'], str)
                or len(payload['original_body_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in payload['original_body_sha256'])
                or type(payload['confidence']) not in (int, float)
                or not math.isfinite(payload['confidence']) or not 0 <= payload['confidence'] <= 1
                or ('sensitive' in payload and type(payload['sensitive']) is not bool)
                or ('reason' in payload and not isinstance(payload['reason'], str))):
            raise L1OwnerError('invalid_legacy_backfill_request')
        if (not isinstance(payload['sources'], list) or not payload['sources']
                or any(not isinstance(source, dict) for source in payload['sources'])
                or not isinstance(payload['document'], dict)):
            raise L1OwnerError('invalid_legacy_backfill_request')
        generation = payload['generation']
        if not isinstance(generation, dict) or generation.get('actor') != 'legacy_backfill':
            raise L1OwnerError('backfill_model_generation_required', field_path='generation')
        revision_fields = {'expected_candidate_revision', 'expected_proposal_sha256'}
        if revision_fields.intersection(payload):
            if (not revision_fields <= set(payload) or type(payload['expected_candidate_revision']) is not int
                    or payload['expected_candidate_revision'] < 1
                    or not isinstance(payload['expected_proposal_sha256'], str)
                    or len(payload['expected_proposal_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in payload['expected_proposal_sha256'])):
                raise L1OwnerError('backfill_model_revision_preconditions_required')

    async def _sources(self, sources):
        snapshots, bindings = bind_sources(sources, config=self.config)
        if len(snapshots) != len(sources):
            raise L1OwnerError('backfill_source_context_incomplete', field_path='sources')
        readers = self._readers()
        # Exact upstream versions are locators for an authenticated GET. They
        # never make the supplied text or a retained copy eligible by itself.
        hints = {canonical_json({k: b[k] for k in ('source_namespace', 'source_id', 'source_revision')}): s.get('upstream_revision')
                 for s, b in zip(snapshots, bindings)}
        for adapter in getattr(readers, 'adapters', {}).values():
            original = getattr(adapter, 'upstream_version', None)
            if original is not None:
                adapter.upstream_version = lambda b, prior=original: hints.get(canonical_json(
                    {k: b[k] for k in ('source_namespace', 'source_id', 'source_revision')})) or prior(b)
        trusted = deepcopy(sources)
        for index, (snapshot, binding) in enumerate(zip(snapshots, bindings)):
            observed = await readers.locate(binding)
            if observed.get('eligible_source') is not True or observed.get('text') != snapshot['text']:
                raise L1OwnerError('backfill_source_not_currently_readable', status=409, field_path='sources')
            actual = readers.copy(binding)
            if not actual or actual.get('text') != snapshot['text']:
                raise L1OwnerError('backfill_authenticated_source_missing', status=409, field_path='sources')
            trusted[index]['recorded_at'] = actual.get('recorded_at')
        return trusted

    def detail(self, memory_id):
        memory = self.authority.get_memory(memory_id)
        if not memory:
            raise MemoryNotFound(memory_id)
        cid = _candidate_id(memory_id, memory['active_revision'])
        return {'status': 'ok', 'memory_id': memory_id, 'current_revision': memory['active_revision'],
                'candidate': self.authority.get_candidate(cid), 'adoption_is_separate': True}

    @staticmethod
    def _revision_prior(db, key, fingerprint):
        prior = db.execute('SELECT result_json FROM candidate_decisions WHERE request_id=?', (key,)).fetchone()
        if not prior:
            return None
        row = json.loads(prior['result_json'])
        request = row['proposal']['metadata']['legacy_backfill'].get('model_revision_request') or {}
        if request.get('request_sha256') != fingerprint:
            raise IdempotencyConflict('model revision request id content changed')
        return row

    def _check_model_revision(self, db, row, payload):
        if row is None:
            raise L1OwnerError('backfill_candidate_not_found', status=404)
        cid, proposal = row['candidate_id'], row['proposal']
        if row['status'] != 'pending':
            code = ('backfill_candidate_adoption_started' if row['status'] in {'accepted', 'committed', 'commit_failed'}
                    else 'backfill_candidate_review_decided')
            raise L1OwnerError(code, status=409)
        if db.execute('SELECT 1 FROM candidate_decisions WHERE candidate_id=? AND from_status!=to_status '
                      "AND actor!='policy' LIMIT 1", (cid,)).fetchone():
            raise L1OwnerError('backfill_candidate_review_decided', status=409)
        if db.execute("SELECT 1 FROM candidate_revisions WHERE candidate_id=? AND actor='owner' LIMIT 1", (cid,)).fetchone():
            raise L1OwnerError('backfill_candidate_owner_modified', status=409)
        history = db.execute('SELECT actor,status,proposal_json FROM candidate_revisions WHERE candidate_id=?', (cid,)).fetchall()
        for version in history:
            if version['status'] != 'pending':
                raise L1OwnerError('backfill_candidate_review_decided', status=409)
            saved = json.loads(version['proposal_json'])
            metadata = saved.get('metadata') or {}
            generation = (metadata.get('memory_l1') or {}).get('generation') or {}
            if (version['actor'] not in {'policy', 'legacy_backfill_revision'}
                    or saved.get('source_type') != 'legacy_backfill' or saved.get('requested_mode') != 'review'
                    or saved.get('owner_explicit') is not False or metadata.get('owner_edit_request_hash')
                    or generation.get('origin') != 'first_pass_model' or generation.get('actor') != 'legacy_backfill'):
                raise L1OwnerError('backfill_candidate_not_model_only', status=409)
        base = proposal.get('metadata', {}).get('legacy_backfill') or {}
        if (not isinstance(base.get('memory_id'), str)
                or _candidate_id(base['memory_id'], payload['expected_revision']) != cid
                or base.get('expected_revision') != payload['expected_revision']
                or base.get('original_body_sha256') != payload['original_body_sha256']):
            raise L1OwnerError('backfill_revision_base_changed', status=409)
        if row['revision'] != payload['expected_candidate_revision']:
            raise L1OwnerError('backfill_candidate_revision_conflict', status=409, field_path='expected_candidate_revision')
        if row['proposal_sha256'] != payload['expected_proposal_sha256']:
            raise L1OwnerError('backfill_candidate_proposal_conflict', status=409, field_path='expected_proposal_sha256')
        if payload['request_id'] == base.get('request_id'):
            raise IdempotencyConflict('model revision must not reuse the original stage request id')
        if db.execute("SELECT 1 FROM commit_log WHERE aggregate_id=? AND operation_kind='memory_revision' "
                      "AND status IN ('prepared','body_written') LIMIT 1", (base['memory_id'],)).fetchone():
            raise L1OwnerError('backfill_target_commit_in_progress', status=409)
        if row.get('policy', {}).get('requires_owner_confirmation') is not True:
            raise L1OwnerError('backfill_candidate_not_model_only', status=409)

    def _check_revision_scope(self, previous, document):
        old = previous['proposal']['metadata']['memory_l1']['evidence_bindings']
        transitions = []
        for binding in document['evidence_bindings']:
            supports = [item for item in old if all(item[key] == binding[key]
                for key in ('source_namespace', 'source_id', 'source_revision', 'author_ref', 'speaker_ref'))]
            if not supports:
                from memory_legacy_participants import participant_only_rebinding, compatible_scope
                for item in old:
                    proof = participant_only_rebinding(item, binding, readers=self.source_reader_factory(), config=self.config)
                    if proof:
                        supports.append(compatible_scope(item, binding))
                        transitions.append({**proof, 'previous_candidate_revision': previous['revision']})
            for anchor in binding['anchors']:
                if not any(all(prior[key] == anchor[key] for key in
                    ('target_namespace', 'target_id', 'target_revision', 'text_view_id', 'coordinate_unit'))
                    and prior['start'] <= anchor['start'] < anchor['end'] <= prior['end']
                    for item in supports for prior in item['anchors']):
                    raise L1OwnerError('backfill_revision_source_scope_changed', status=409, field_path='sources')
        return transitions

    @staticmethod
    def _revision_result(row, memory_id, base_revision, *, replayed):
        return {'status': 'revised', 'candidate': row, 'candidate_id': row['candidate_id'],
                'memory_id': memory_id, 'base_revision': base_revision, 'replayed': replayed,
                'adopted': False, 'adoption_is_separate': True}

    async def stage(self, memory_id, payload):
        self._validate(payload)
        cid = _candidate_id(memory_id, payload['expected_revision'])
        fingerprint = semantic_sha256(payload)
        existing = self.authority.get_candidate(cid)
        revising = 'expected_candidate_revision' in payload
        revision_key = 'legacy-backfill-model-revision:' + cid + ':' + payload['request_id']
        if revising:
            with closing(self.authority._connect()) as db:
                prior = self._revision_prior(db, revision_key, fingerprint)
                if prior:
                    return self._revision_result(prior, memory_id, payload['expected_revision'], replayed=True)
                self._check_model_revision(db, existing, payload)
        elif existing:
            if existing['proposal']['metadata']['legacy_backfill']['request_sha256'] != fingerprint:
                raise L1OwnerError('backfill_draft_already_exists', status=409, field_path='request_id')
            return {'status': 'exists', 'candidate': existing, 'replayed': True,
                    'candidate_id': cid, 'memory_id': memory_id, 'base_revision': payload['expected_revision'],
                    'adopted': existing['status'] == 'committed', 'adoption_is_separate': True}
        memory, generation = self.authority.get_memory_state_snapshot(memory_id)
        if not memory:
            raise MemoryNotFound(memory_id)
        base = deepcopy(existing['proposal']['metadata']['legacy_backfill']) if revising else {
                'memory_id': memory_id, 'expected_revision': payload['expected_revision'],
                'original_body_sha256': payload['original_body_sha256'], 'state_generation': generation,
                'request_id': payload['request_id'], 'request_sha256': fingerprint,
                'reason': payload.get('reason', '为同一条历史记忆补齐有来源的第一遍语义结构。')}
        with closing(self.authority._connect()) as db:
            _check_base(self.authority, db, memory_id, base)
        from memory_legacy_classification import candidate_kind
        base['candidate_kind'] = candidate_kind(payload, existing['proposal'] if revising else None)
        original = await self.revisions.detail(memory_id, payload['expected_revision'])
        projected, legacy_proof = await self._projection_base(memory_id, memory, base)
        if revising and legacy_proof != base.get('legacy_projection'):
            raise RevisionConflict('legacy projection base changed after preparation')
        if legacy_proof:
            base['legacy_projection'] = legacy_proof
        # A migrated legacy record can still lack its immutable file. Retain
        # its verified original before a later Review can replace the bucket.
        snapshot = self.revisions.commit.projection.snapshot_path(memory_id, payload['expected_revision'])
        if not snapshot.is_file():
            raw = Path(projected['path']).read_text(encoding='utf-8')
            post = frontmatter.loads(raw)
            if body_sha256(post.content) != payload['original_body_sha256'] or post.get('memory_revision') != payload['expected_revision']:
                raise L1OwnerError('backfill_projection_changed', status=409)
            from bucket_manager import BucketManager
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            BucketManager._atomic_write_text(str(snapshot), raw)
        sources = await self._sources(payload['sources'])
        document = prepare_model_document(self.revisions, memory_id=memory_id,
            document=assemble_document(payload['document']), sources=sources,
            generation=payload['generation'], request_id=cid,
            previous=original['l1']['document'], config=self.config,
            candidate_refs=payload.get('source_refs', []))
        body, semantic = split_document(document)
        if revising:
            transitions = self._check_revision_scope(existing, document)
            if transitions:
                base['source_participant_rebindings'] = [*base.get('source_participant_rebindings', []), *transitions]
        base['source_witnesses'] = self._witnesses(memory_id, base, semantic, sources=sources)
        if revising:
            base['model_revision_request'] = {'request_id': payload['request_id'], 'request_sha256': fingerprint,
                'expected_candidate_revision': payload['expected_candidate_revision'],
                'expected_proposal_sha256': payload['expected_proposal_sha256'], 'reason': payload.get('reason', '')}
        proposal = MemoryProposal.from_mapping({'proposal_id': cid, 'source_type': 'legacy_backfill',
            'proposed_body': body, 'source_refs': original['revisions'][0]['source_refs'],
            'source_status': 'verified', 'memory_type': base['candidate_kind']['value'] or 'memory', 'confidence': payload['confidence'],
            'requested_mode': 'review', 'owner_explicit': False,
            'sensitive': payload.get('sensitive', False) or (existing['proposal'].get('sensitive', False) if revising else False),
            'metadata': {'memory_l1': semantic, 'legacy_backfill': base,
                         'legacy_candidate': {'title': document['narrative']['title'], 'kind': base['candidate_kind']['value'] or 'memory',
                             'mode': 'review', 'source_verification': 'verified'}}})
        policy = PolicyDecision('queue_review', ('REVIEW_LEGACY_SAME_ID_BACKFILL',), 'review',
                                requires_owner_confirmation=True)
        with closing(self.authority._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if revising:
                prior = self._revision_prior(db, revision_key, fingerprint)
                if prior:
                    return self._revision_result(prior, memory_id, payload['expected_revision'], replayed=True)
            _check_base(self.authority, db, memory_id, base)
            current = self.authority._candidate_row(db.execute('SELECT * FROM candidates WHERE candidate_id=?', (cid,)).fetchone())
            if revising:
                self._check_model_revision(db, current, payload)
                revised = self.authority._revise_candidate(db, cid, expected_revision=current['revision'],
                    proposal=proposal, request_id=revision_key, actor='legacy_backfill_revision')
                return self._revision_result(revised, memory_id, payload['expected_revision'], replayed=False)
            elif current:
                if current['proposal']['metadata']['legacy_backfill']['request_sha256'] != fingerprint:
                    raise IdempotencyConflict('another backfill draft was saved')
                row, replayed = current, True
            else:
                row, replayed = self.authority._put_candidate(db, proposal, policy), False
        return {'status': 'exists' if replayed else 'saved', 'candidate': row,
                'candidate_id': cid, 'memory_id': memory_id, 'base_revision': payload['expected_revision'],
                'replayed': replayed, 'adopted': False, 'adoption_is_separate': True}

    async def adopt(self, row, *, request_id, edit=None):
        cid = row['candidate_id']
        if row['status'] == 'committed':
            return {'id': cid, 'status': 'exists', 'memory_id': row['proposal']['metadata']['legacy_backfill']['memory_id']}
        if edit:
            from memory_candidate_edit import save_owner_candidate
            row = save_owner_candidate(self.authority, cid, edit=edit,
                expected_revision=row['revision'], request_id=request_id + ':edit', revision_service=self.revisions)
        if row['status'] not in {'pending', 'deferred', 'accepted'}:
            return {'id': cid, 'status': row['status']}
        proposal = row['proposal']
        base = proposal['metadata']['legacy_backfill']
        memory_id = base['memory_id']
        accepted_revision = row['revision'] if row['status'] == 'accepted' else row['revision'] + 1
        idem = 'legacy-backfill:' + cid + ':' + str(accepted_revision)
        prior = self.authority.get_commit_by_idempotency(idem)
        if not (prior and prior['status'] == 'committed'):
            with closing(self.authority._connect()) as db:
                memory = _check_base(self.authority, db, memory_id, base)
            if not prior:
                _, legacy_proof = await self._projection_base(memory_id, memory, base)
                if legacy_proof != base.get('legacy_projection'):
                    raise RevisionConflict('legacy projection base changed after preparation')
            readers = self._readers()
            for binding in proposal['metadata']['memory_l1']['evidence_bindings']:
                if not (await readers.locate(binding)).get('eligible_source'):
                    raise L1OwnerError('backfill_source_not_currently_readable', status=409, field_path='sources')
            witnesses = self._witnesses(memory_id, base, proposal['metadata']['memory_l1'], previous=base['source_witnesses'])
            if witnesses != base['source_witnesses']:
                raise RevisionConflict('legacy backfill original source chain changed')
            if row['status'] != 'accepted':
                row = self.authority.decide_candidate(cid, action='accept', expected_revision=row['revision'],
                    request_id=request_id + ':accept', actor='owner', reason_codes=['OWNER_ACCEPTED_LEGACY_BACKFILL'])
            if prior:
                metadata, source_refs = prior['payload']['metadata'], prior['payload']['source_refs']
            else:
                old = self.authority.get_memory_revision(memory_id, base['expected_revision'])
                from memory_legacy_classification import active_metadata
                metadata, classification_projection = active_metadata(self.authority, old, proposal)
                metadata.update(memory_l1=proposal['metadata']['memory_l1'],
                    name=proposal['metadata']['memory_l1']['narrative']['title'],
                    memory_l1_legacy_backfill={**base, 'candidate_id': cid,
                        'classification_projection': classification_projection},
                    memory_l1_program_context={'origin': 'legacy_backfill', 'proposal_id': cid,
                        'candidate_revision': row['revision'], 'proposal_sha256': row['proposal_sha256']})
                source_refs = old['source_refs']
            result = await self.revisions.commit.commit_memory(memory_id=memory_id,
                bucket_id=memory['bucket_id'], expected_revision=base['expected_revision'],
                body=proposal['proposed_body'], metadata=metadata, source_refs=source_refs,
                decision_source='owner_backfill_review', idempotency_key=idem, actor='legacy_backfill',
                memory_state=memory['state'], recall_policy=memory['recall_policy'], l1_write_intent=True)
        else:
            result = prior['result']
        current = self.authority.get_candidate(cid)
        if current['status'] == 'accepted':
            self.authority.decide_candidate(cid, action='commit', expected_revision=current['revision'],
                request_id=idem + ':committed', actor='commit_service', reason_codes=['MEMORY_REVISION_COMMITTED'])
        return {'id': cid, 'status': 'created', 'memory_id': memory_id, 'revision': result['revision'],
                'receipt': result, 'replayed': bool(prior)}
