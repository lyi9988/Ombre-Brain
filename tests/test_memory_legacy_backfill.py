"""Targeted Review backfill against real temporary Authority, buckets and L0."""
import asyncio
from contextlib import closing
from copy import deepcopy
import json
import sqlite3

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore, MemoryIngestionPolicy, MemoryProposal, RevisionConflict
from memory_candidate_edit import save_owner_candidate
from memory_commit_service import BucketMemoryProjection, MemoryCommitService, body_sha256
from memory_l0_sources import MemoryL0Sources
from memory_l1_schema import catalog_with_defaults, join_document
from memory_l1_service import L1OwnerError, MemoryL1RevisionService
from memory_l1_source_readers import configured_source_readers
from memory_legacy_backfill import LegacyMemoryBackfill
from memory_l3_records import MemoryL3Records


def run(task):
    return asyncio.run(task)


def setup(tmp_path, *, source_refs=None, metadata=None):
    config = {'buckets_dir': str(tmp_path / 'buckets'), 'state_dir': str(tmp_path / 'state')}
    authority, manager = MemoryAuthorityStore(config), BucketManager(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, manager))
    revisions = MemoryL1RevisionService(authority, commit, manager, lambda: (catalog_with_defaults(), 'mc-test'))
    run(commit.commit_memory(memory_id='m1', bucket_id='b1', expected_revision=0,
        body='完整的合成回忆。', metadata={'name': '标题', 'owner_unrelated': {'keep': 7}, **(metadata or {})},
        source_refs=['conversation_turn:1'] if source_refs is None else source_refs,
        decision_source='synthetic', idempotency_key='seed', actor='test'))
    text = '今天闲聊。两个月前肚子痛，那次想要抱抱。🙂另外还有一件事。'
    with sqlite3.connect(tmp_path / 'buckets' / 'gateway_state.db') as db:
        db.execute('CREATE TABLE conversation_turns (id INTEGER PRIMARY KEY,profile_id TEXT,session_id TEXT,user_text TEXT,assistant_text TEXT,created_at TEXT)')
        db.execute('INSERT INTO conversation_turns VALUES(1,?,?,?,?,?)',
            ('p1', 'c1', text, '那次陪着你。', '2026-10-15T00:00:00Z'))
    source = {'namespace': 'conversation_turn', 'system': 'ombre', 'event_id': '1',
        'profile_id': 'p1', 'conversation_id': 'c1', 'version_id': '', 'role': 'user',
        'text': text, 'recorded_at': '2026-10-15T00:00:00Z', '_model_range': {'start': 5, 'end': 22}}
    document = {'format': 'semantic-v1', 'narrative': {'title': '那次陪伴', 'body': '那一次她说肚子痛，希望我抱抱。'},
        'entities': [{'entity_id': 'e1', 'name': '主人', 'entity_type': 'user', 'evidence_refs': ['source:1']}],
        'statements': [{'statement_id': 's1', 'content': '那次她希望抱抱。', 'subject_ref': 'e1',
            'classification': {'kind': 'preference', 'stability': 'episode', 'modality': None, 'basis': None},
            'time': {'event': None, 'applicability': None}, 'conditions': None, 'evidence_refs': ['source:1']}],
        'retrieval': {'keywords': ['那次陪伴'], 'tags': []},
        'scenes': [{'scene_id': 'sc1', 'title': '那次', 'statement_refs': ['s1'],
            'attributes': [], 'source_ranges': [{'source_ref': 'source:1', 'start': 5, 'end': 22}]}]}
    payload = {'expected_revision': 1, 'request_id': 'backfill-one',
        'original_body_sha256': body_sha256('完整的合成回忆。'), 'document': document, 'sources': [source],
        'generation': {'actor': 'legacy_backfill', 'task_id': 'model-task-1',
            'prompt_revision': 'prompt-1', 'model_route_revision': 'route-1'}, 'confidence': 0.9}
    factory = lambda: MemoryL0Sources(authority, configured_source_readers(config, authority))
    return LegacyMemoryBackfill(revisions, factory, config=config), payload, config


def test_stage_only_persists_review_and_exact_fragment_then_replays_owner_draft(tmp_path):
    service, payload, _ = setup(tmp_path)
    before = service.authority.get_memory_revision('m1', 1)
    bucket_before = run(service.revisions.buckets.get('b1'))
    staged = run(service.stage('m1', payload))
    row = staged['candidate']
    assert staged['candidate_id'] == row['candidate_id'] and staged['memory_id'] == 'm1' and staged['base_revision'] == 1
    assert row['status'] == 'pending' and row['policy']['decision_policy'] == 'review'
    assert row['proposal']['owner_explicit'] is False
    assert service.authority.get_memory_revision('m1', 1) == before
    assert service.authority.get_memory('m1')['active_revision'] == 1
    assert run(service.revisions.buckets.get('b1')) == bucket_before
    semantic = row['proposal']['metadata']['memory_l1']
    binding = semantic['evidence_bindings'][0]
    assert (binding['anchors'][0]['start'], binding['anchors'][0]['end']) == (5, 22)
    assert semantic['scenes'][0]['anchors'][0] == binding['anchors'][0]
    document = join_document(row['proposal']['proposed_body'], semantic)
    document['narrative']['body'] = '主人保留的修订稿。'
    edited = save_owner_candidate(service.authority, row['candidate_id'], edit={'document': document},
        expected_revision=row['revision'], request_id='owner-edit', revision_service=service.revisions)
    replay = run(service.stage('m1', payload))
    assert replay['replayed'] and replay['candidate']['revision'] == edited['revision']
    assert replay['candidate']['proposal']['proposed_body'] == '主人保留的修订稿。'
    changed = deepcopy(payload)
    changed['request_id'] = 'regenerate-over-owner'
    with pytest.raises(L1OwnerError, match='backfill_draft_already_exists'):
        run(service.stage('m1', changed))


def test_review_adopts_same_id_preserves_history_metadata_and_queues_original_l2(tmp_path):
    service, payload, _ = setup(tmp_path)
    before = service.authority.get_memory_revision('m1', 1)
    staged = run(service.stage('m1', payload))
    result = run(service.adopt(staged['candidate'], request_id='approve'))
    assert result['memory_id'] == 'm1' and result['revision'] == 2
    assert service.authority.get_memory_revision('m1', 1) == before
    actual = service.authority.get_memory_revision('m1', 2)
    assert actual['metadata']['owner_unrelated'] == {'keep': 7}
    assert actual['source_refs'] == before['source_refs']
    assert actual['created_by'] == 'legacy_backfill'
    generation = actual['metadata']['memory_l1']['generation']
    assert generation['origin'] == 'first_pass_model' and generation['actor'] == 'legacy_backfill'
    assert service.authority.get_memory(staged['candidate_id']) is None
    outbox = service.authority.pending_outbox()[-1]['payload']
    assert outbox['l2_preparation']['state'] == 'pending'
    timeline = MemoryL3Records(service.authority, service.revisions.commit).timeline('m1')
    assert any(item['kind'] == 'l1_revision' and item['revision'] == 2 for item in timeline['items'])
    assert next(item for item in timeline['items'] if item['event_id'] == 'l1:2')['details']['edit_note'] == row_reason(staged['candidate'])
    assert run(service.adopt(service.authority.get_candidate(staged['candidate_id']), request_id='again'))['status'] == 'exists'
    assert service.authority.get_memory('m1')['active_revision'] == 2


@pytest.mark.parametrize('change', ['source', 'archive', 'withdraw', 'restore', 'owner_l1', 'bucket'])
def test_source_withdrawal_and_concurrent_target_changes_cannot_adopt(tmp_path, change):
    service, payload, config = setup(tmp_path)
    row = run(service.stage('m1', payload))['candidate']
    if change == 'source':
        with sqlite3.connect(tmp_path / 'buckets' / 'gateway_state.db') as db:
            db.execute("UPDATE conversation_turns SET user_text='已改变的原文' WHERE id=1")
    elif change in {'archive', 'withdraw', 'restore'}:
        target = 'tombstoned' if change == 'withdraw' else 'archived'
        service.authority.change_memory_state(memory_id='m1', state=target, recall_policy='disabled',
            idempotency_key='withdraw', actor='owner')
        if change == 'restore':
            service.authority.change_memory_state(memory_id='m1', state='active', recall_policy='enabled',
                idempotency_key='restore', actor='owner')
    elif change == 'owner_l1':
        detail = run(service.revisions.detail('m1'))
        run(service.revisions.save('m1', {'expected_revision': 1, 'request_id': 'owner-current', 'document': detail['l1']['document']}))
    else:
        run(service.revisions.buckets.update('b1', content='尚未纳入Authority的改动'))
    with pytest.raises((L1OwnerError, RevisionConflict)):
        run(service.adopt(row, request_id='stale-approval'))
    assert service.authority.get_candidate(row['candidate_id'])['status'] == 'pending'
    assert service.authority.get_memory('m1')['active_revision'] == (2 if change == 'owner_l1' else 1)


def test_stage_rejects_wrong_body_existing_formal_l1_and_bad_fragment(tmp_path):
    service, payload, _ = setup(tmp_path)
    wrong = deepcopy(payload)
    wrong['original_body_sha256'] = '0' * 64
    with pytest.raises(RevisionConflict):
        run(service.stage('m1', wrong))
    bad_range = deepcopy(payload)
    bad_range['document']['scenes'][0]['source_ranges'][0]['start'] = 0
    with pytest.raises(L1OwnerError, match='scene_range_outside_observed_input'):
        run(service.stage('m1', bad_range))
    detail = run(service.revisions.detail('m1'))
    run(service.revisions.save('m1', {'expected_revision': 1, 'request_id': 'owner', 'document': detail['l1']['document']}))
    payload.update(expected_revision=2)
    with pytest.raises(L1OwnerError, match='backfill_requires_legacy_l1'):
        run(service.stage('m1', payload))


def test_authority_rechecks_lifecycle_inside_prepare_transaction(tmp_path, monkeypatch):
    service, payload, _ = setup(tmp_path)
    row = run(service.stage('m1', payload))['candidate']
    prepare = service.authority.prepare_memory_commit
    def revoked_before_prepare(**kwargs):
        service.authority.change_memory_state(memory_id='m1', state='tombstoned', recall_policy='disabled',
            idempotency_key='between-check-and-prepare', actor='owner')
        return prepare(**kwargs)
    monkeypatch.setattr(service.authority, 'prepare_memory_commit', revoked_before_prepare)
    with pytest.raises(L1OwnerError, match='backfill_memory_not_active'):
        run(service.adopt(row, request_id='racing-review'))
    assert service.authority.get_memory('m1')['active_revision'] == 1
    assert run(service.revisions.buckets.get('b1'))['content'] == '完整的合成回忆。'


def test_two_stage_requests_keep_one_candidate(tmp_path):
    service, payload, _ = setup(tmp_path)
    async def race():
        return await asyncio.gather(service.stage('m1', payload), service.stage('m1', payload))
    first, second = run(race())
    assert first['candidate_id'] == second['candidate_id']
    with closing(service.authority._connect()) as db:
        assert db.execute('SELECT count(*) FROM candidates').fetchone()[0] == 1


def test_selected_sources_exclude_context_and_original_manual_policy_is_preserved(tmp_path):
    service, payload, _ = setup(tmp_path)
    service.authority.change_memory_state(memory_id='m1', state='active', recall_policy='manual_only',
        idempotency_key='owner-scope', actor='owner')
    context = deepcopy(payload['sources'][0])
    context.update(role='assistant', text='那次陪着你。')
    context.pop('_model_range')
    payload['sources'].append(context)
    payload['source_refs'] = ['source:1']
    row = run(service.stage('m1', payload))['candidate']
    bindings = row['proposal']['metadata']['memory_l1']['evidence_bindings']
    assert [binding['binding_id'] for binding in bindings] == ['source:1']
    run(service.adopt(row, request_id='manual-scope-approve'))
    assert service.authority.get_memory('m1')['recall_policy'] == 'manual_only'
    # The retained backfill provenance must not block later ordinary edits.
    detail = run(service.revisions.detail('m1'))
    detail['l1']['document']['narrative']['title'] = '主人后来改过标题'
    saved = run(service.revisions.save('m1', {'expected_revision': 2, 'request_id': 'later-owner',
        'document': detail['l1']['document']}))
    assert saved['revision'] == 3 and service.authority.get_memory('m1')['recall_policy'] == 'manual_only'


def test_owner_l2_remains_the_head_after_original_worker_prepares_new_product(tmp_path):
    from types import SimpleNamespace as NS
    from memory_l2_store import MemoryL2Store
    from memory_l2_service import MemoryL2Service
    service, payload, _ = setup(tmp_path)
    store = MemoryL2Store(service.authority)
    generation = {'prompt_revision': 'p1', 'model_route_revision': 'r1'}
    old_dependency = store.current_dependency('m1', generation)
    owner_document = {'full_injection_text': '主人手编的完整旧稿。', 'coherent_sections': [], 'optional_use_variants': []}
    store.save('m1', owner_document, old_dependency, request_id='owner-old-l2', expected_revision=0, actor='owner')
    with closing(service.authority._connect()) as db:
        old_head = dict(db.execute("SELECT * FROM memory_l2_heads WHERE memory_id='m1'").fetchone())
        old_l2 = dict(db.execute("SELECT * FROM memory_l2_revisions WHERE memory_id='m1'").fetchone())
    row = run(service.stage('m1', payload))['candidate']
    run(service.adopt(row, request_id='preserve-l2'))
    engine = NS(temperature=0.3)
    engine._memory_l1_prompt = lambda source_id: ('second-pass prompt', {'prompt_revision': 'p1'})
    engine._memory_task_route = lambda stage: {'stage': stage, 'client': object(), 'configured': True,
        'model': 'synthetic-l2', 'connection': 'reflection', 'max_tokens': 16000}
    engine._memory_l1_sources = service.source_reader_factory
    engine._completion_content = lambda response: response.text
    engine._completion_finish_reason = lambda response: response.finish
    async def complete(client, **kwargs):
        material = json.loads(kwargs['messages'][1]['content'])
        sid = material['l1_document']['statements'][0]['statement_id']
        return NS(text=json.dumps({'full_injection_text': '新的独立第二遍完整稿。',
            'coherent_sections': [{'section_id': 'one', 'text': '新的独立第二遍完整稿。', 'statement_refs': [sid]}],
            'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}, ensure_ascii=False), finish='stop')
    engine._daily_chat_memory_create_completion = complete
    result = run(MemoryL2Service(engine, service.revisions).run_once(1))
    assert result['results'][0]['state'] == 'pending_review'
    with closing(service.authority._connect()) as db:
        assert dict(db.execute("SELECT * FROM memory_l2_heads WHERE memory_id='m1'").fetchone()) == old_head
        assert dict(db.execute("SELECT * FROM memory_l2_revisions WHERE memory_id='m1' AND presentation_revision=1").fetchone()) == old_l2
        assert db.execute("SELECT publication FROM memory_l2_revisions WHERE memory_id='m1' AND presentation_revision=2").fetchone()[0] == 'candidate'


def test_committed_memory_with_lost_candidate_receipt_recovers_once(tmp_path, monkeypatch):
    service, payload, _ = setup(tmp_path)
    row = run(service.stage('m1', payload))['candidate']
    decide = service.authority.decide_candidate
    failed_once = [False]
    def lost_commit_receipt(candidate_id, **kwargs):
        if kwargs['action'] == 'commit' and not failed_once[0]:
            failed_once[0] = True
            raise OSError('synthetic receipt loss')
        return decide(candidate_id, **kwargs)
    monkeypatch.setattr(service.authority, 'decide_candidate', lost_commit_receipt)
    with pytest.raises(OSError):
        run(service.adopt(row, request_id='lost-response'))
    assert service.authority.get_memory('m1')['active_revision'] == 2
    accepted = service.authority.get_candidate(row['candidate_id'])
    assert accepted['status'] == 'accepted'
    recovered = run(service.adopt(accepted, request_id='retry-after-readback'))
    assert recovered['revision'] == 2 and recovered['replayed']
    with closing(service.authority._connect()) as db:
        assert db.execute("SELECT count(*) FROM memory_revisions WHERE memory_id='m1'").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM memory_l2_jobs WHERE memory_id='m1'").fetchone()[0] == 1


@pytest.mark.parametrize('reported_role', ['user', 'assistant', None])
def test_canonical_hint_only_locates_authenticated_current_source(tmp_path, monkeypatch, reported_role):
    import httpx
    import memory_l1_source_readers as readers_module
    service, payload, config = setup(tmp_path, source_refs=['canonical_event:event-1@version-1'])
    config['gateway'] = {'canonical_continuation': {'enabled': True, 'base_url': 'http://canonical-test', 'conversation_id': 'c1'}}
    source = payload['sources'][0]
    source.update(namespace='canonical_event', system='aizizhu', event_id='event-1', version_id='version-1')
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'synthetic-private-bridge')
    client_class = httpx.AsyncClient
    calls = []
    def transport(request):
        calls.append(request)
        assert request.headers['authorization'] == 'Bearer synthetic-private-bridge'
        assert request.url.params['version'] == 'version-1'
        return httpx.Response(200, json={'source': {'state': 'available', 'eligible_source': True,
            'source_id': 'event-1', 'source_revision': 'version-1', 'text': source['text'],
            'recorded_at': source['recorded_at'], 'author_ref': {'role': reported_role}}})
    monkeypatch.setattr(readers_module.httpx, 'AsyncClient', lambda **kwargs: client_class(
        transport=httpx.MockTransport(transport), **kwargs))
    if reported_role != source['role']:
        if reported_role is None:
            from memory_l1_ingestion import bind_sources
            from memory_l1_source_readers import CanonicalMemorySourceReader
            legacy_reader = CanonicalMemorySourceReader(config)
            legacy_reader.upstream_version = lambda binding: 'version-1'
            _, bindings = bind_sources(payload['sources'], config=config)
            assert run(legacy_reader.read(bindings[0]))['state'] == 'available'
        with pytest.raises(L1OwnerError, match='backfill_source_not_currently_readable'):
            run(service.stage('m1', payload))
        assert service.authority.get_memory('m1')['active_revision'] == 1
        return
    row = run(service.stage('m1', payload))['candidate']
    assert len(calls) == 1
    run(service.adopt(row, request_id='canonical-approve'))
    assert len(calls) == 2


def test_missing_legacy_snapshot_is_retained_before_new_body_can_be_adopted(tmp_path):
    service, payload, _ = setup(tmp_path)
    snapshot = service.revisions.commit.projection.snapshot_path('m1', 1)
    original = snapshot.read_bytes()
    snapshot.unlink()
    row = run(service.stage('m1', payload))['candidate']
    assert snapshot.read_bytes() == original
    run(service.adopt(row, request_id='approve-restored-legacy-snapshot'))
    assert run(service.revisions.detail('m1', 1))['body'] == '完整的合成回忆。'


def row_reason(row):
    return row['proposal']['metadata']['legacy_backfill']['reason']


def test_other_existing_source_with_identical_text_cannot_be_attached_to_target(tmp_path):
    service, payload, _ = setup(tmp_path)
    with sqlite3.connect(tmp_path / 'buckets' / 'gateway_state.db') as db:
        db.execute('INSERT INTO conversation_turns SELECT 2,profile_id,session_id,user_text,assistant_text,created_at FROM conversation_turns WHERE id=1')
    payload['sources'][0]['event_id'] = '2'
    with pytest.raises(L1OwnerError, match='backfill_source_not_in_original_memory'):
        run(service.stage('m1', payload))
    assert not service.authority.list_candidates(status='all')
    assert service.authority.get_memory('m1')['active_revision'] == 1


def test_exact_original_candidate_link_recovers_its_sources(tmp_path):
    service, payload, _ = setup(tmp_path, source_refs=['memory_revision:1'], metadata={'daily_chat_memory_candidate_id': 'original-candidate'})
    proposal = MemoryProposal.from_mapping({'proposal_id': 'original-candidate', 'source_type': 'daily_chat',
        'proposed_body': '历史提炼稿', 'source_refs': ['conversation_turn:1'], 'source_status': 'verified', 'requested_mode': 'review'})
    service.authority.put_candidate(proposal, MemoryIngestionPolicy().evaluate(proposal))
    service.authority.decide_candidate('original-candidate', action='accept', expected_revision=1,
        request_id='original-accepted', actor='owner')
    service.authority.decide_candidate('original-candidate', action='commit', expected_revision=2,
        request_id='original-committed', actor='commit_service')
    row = run(service.stage('m1', payload))['candidate']
    witness = row['proposal']['metadata']['legacy_backfill']['source_witnesses'][0]
    assert witness['source_ref'] == 'conversation_turn:1'
    assert witness['authority_path'] == {'candidate_id': 'original-candidate', 'candidate_revision': 3}
    run(service.adopt(row, request_id='original-candidate-linked'))
    assert service.authority.get_memory('m1')['active_revision'] == 2


def test_uncommitted_original_candidate_is_not_an_adopted_source_link(tmp_path):
    service, payload, _ = setup(tmp_path, source_refs=['candidate:unreviewed-original'])
    proposal = MemoryProposal.from_mapping({'proposal_id': 'unreviewed-original', 'source_type': 'daily_chat',
        'proposed_body': '尚未采用的提案', 'source_refs': ['conversation_turn:1'], 'source_status': 'verified', 'requested_mode': 'review'})
    service.authority.put_candidate(proposal, MemoryIngestionPolicy().evaluate(proposal))
    with pytest.raises(L1OwnerError, match='backfill_source_not_in_original_memory'):
        run(service.stage('m1', payload))


@pytest.mark.parametrize('archive_changed', [False, True])
def test_exact_raw_archive_bridge_is_checked_again_before_adoption(tmp_path, monkeypatch, archive_changed):
    import httpx
    import memory_l1_source_readers as readers_module
    from raw_events import RawEventStore
    from memory_source_provenance import BRIDGE_VERSION, canonical_source
    service, payload, config = setup(tmp_path, source_refs=['raw_event:91'])
    config['gateway'] = {'canonical_continuation': {'enabled': True, 'base_url': 'http://canonical-test', 'conversation_id': 'c1'}}
    source = payload['sources'][0]
    source.update(namespace='canonical_event', system='aizizhu', event_id='event-1', version_id='version-1',
        archive_ref={'system': 'ombre', 'namespace': 'raw_event', 'event_id': '91'})
    bridged = canonical_source({'event_id': 'event-1', 'version_id': 'version-1', 'role': 'user',
        'text': source['text'], 'created_at': source['recorded_at']}, conversation_id='c1', profile_id='p1')
    bridge = {'bridge_version': BRIDGE_VERSION, **{key: value for key, value in bridged.items() if key != 'text'}}
    raw_store = RawEventStore(config)
    with sqlite3.connect(raw_store.db_path) as db:
        db.execute('INSERT INTO raw_events(id,source,event_hash,role,text,created_at,ingested_at,metadata_json) VALUES(91,?,?,?,?,?,?,?)',
            ('gateway', 'fixture-hash', 'user', source['text'], source['recorded_at'], source['recorded_at'],
             json.dumps({'profile_id': 'p1', 'canonical_source': bridge})))
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'synthetic-private-bridge')
    client_class = httpx.AsyncClient
    monkeypatch.setattr(readers_module.httpx, 'AsyncClient', lambda **kwargs: client_class(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'source': {'state': 'available', 'eligible_source': True,
            'source_id': 'event-1', 'source_revision': 'version-1', 'text': source['text'],
            'recorded_at': source['recorded_at'], 'author_ref': {'role': 'user'}}})), **kwargs))
    row = run(service.stage('m1', payload))['candidate']
    witness = row['proposal']['metadata']['legacy_backfill']['source_witnesses'][0]
    assert witness['source_ref'] == 'raw_event:91' and witness['authority_path'] == {'memory_revision': 1}
    if not archive_changed:
        assert run(service.adopt(row, request_id='exact-raw-bridge-approve'))['revision'] == 2
        return
    with sqlite3.connect(raw_store.db_path) as db:
        db.execute("UPDATE raw_events SET text='不再与原精确桥相符' WHERE id=91")
    with pytest.raises(L1OwnerError, match='backfill_source_not_in_original_memory'):
        run(service.adopt(row, request_id='invalidated-archive-bridge'))
    assert service.authority.get_memory('m1')['active_revision'] == 1
