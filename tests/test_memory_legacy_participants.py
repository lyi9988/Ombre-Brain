"""Exact old-main attribution can be versioned without broad source rebinding."""
from copy import deepcopy
import json
import sqlite3

import pytest

from memory_candidate_edit import save_owner_candidate
from memory_l1_ingestion import source_snapshot
from memory_l1_schema import join_document
from memory_l1_service import L1OwnerError
from memory_source_identity import source_participant
from test_memory_legacy_model_revision import revision_payload
from test_memory_legacy_projection import setup, run


def enable_exact_bindings(config, profile='jiajia-main', conversation='jiajia-main'):
    config['persona'] = {'profile_id': 'jiajia-main'}
    config['gateway'] = {'canonical_continuation': {'enabled': True, 'conversation_id': 'conv-jiajia-main'}}
    actors = {role: source_participant({'system': 'aizizhu', 'namespace': 'canonical_event',
        'profile_id': 'jiajia-main', 'conversation_id': 'conv-jiajia-main', 'role': role}, config)
        for role in ('user', 'assistant')}
    config.setdefault('identity', {})['source_participants'] = [
        {'system': 'ombre', 'namespace': namespace, 'profile_id': profile,
         'conversation_id': conversation, 'role': role,
         'participant': {key: actors[role][key] for key in ('subject_id', 'kind', 'display_name')}}
        for namespace in ('raw_event', 'conversation_turn') for role in ('user', 'assistant')]
    return actors


def initial(tmp_path, *, main=True, conversation='jiajia-main'):
    service, payload, snapshot = setup(tmp_path)
    if main:
        with sqlite3.connect(tmp_path / 'buckets/gateway_state.db') as db:
            db.execute("UPDATE conversation_turns SET profile_id='jiajia-main',session_id=?", (conversation,))
        payload['sources'][0].update(profile_id='jiajia-main', conversation_id=conversation)
    for source in payload['sources']:
        source['_participant_revision'] = source_snapshot(source, config=service.config)['source_revision']
    row = run(service.stage('old-memory', payload))['candidate']
    return service, payload, snapshot, row


def refresh(payload, row, config):
    payload = deepcopy(payload)
    payload.update(request_id='identity-refresh', expected_candidate_revision=row['revision'],
        expected_proposal_sha256=row['proposal_sha256'])
    for source in payload['sources']:
        source['_participant_revision'] = source_snapshot(source, config=config)['source_revision']
    return payload


@pytest.mark.parametrize('conversation', ['jiajia-main', 'aizizhu-app'])
def test_exact_config_rebinds_only_actor_digest_and_keeps_old_candidate_and_memory(tmp_path, conversation):
    service, payload, snapshot, first = initial(tmp_path, conversation=conversation)
    memory = service.authority.get_memory_revision('old-memory', 1)
    old_file = snapshot.read_bytes()
    old = first['proposal']['metadata']['memory_l1']['evidence_bindings'][0]
    old_copy = service.source_reader_factory().copy(old)
    actors = enable_exact_bindings(service.config, conversation=conversation)
    stale = revision_payload(payload, first)
    with pytest.raises(L1OwnerError, match='source_participant_changed'):
        run(service.stage('old-memory', stale))
    request = refresh(payload, first, service.config)
    result = run(service.stage('old-memory', request))
    current = result['candidate']
    new = current['proposal']['metadata']['memory_l1']['evidence_bindings'][0]
    assert current['candidate_id'] == first['candidate_id'] and current['revision'] == 2 and current['status'] == 'pending'
    assert new['source_id'] == old['source_id'] and new['source_revision'] != old['source_revision']
    assert new['author_ref'] == new['speaker_ref'] == actors['user']['subject_id']
    assert [(a['start'], a['end']) for a in new['anchors']] == [(a['start'], a['end']) for a in old['anchors']]
    assert current['proposal']['proposed_body'] == first['proposal']['proposed_body']
    proof = current['proposal']['metadata']['legacy_backfill']['source_participant_rebindings']
    assert len(proof) == 1 and proof[0]['previous_candidate_revision'] == 1
    assert service.source_reader_factory().copy(old) == old_copy
    assert service.authority.get_memory_revision('old-memory', 1) == memory and snapshot.read_bytes() == old_file
    assert service.authority.get_memory('old-memory')['recall_policy'] == 'manual_only'
    assert run(service.stage('old-memory', request))['replayed']
    with service.authority._connect() as db:
        rows = db.execute('SELECT actor,proposal_json FROM candidate_revisions WHERE candidate_id=? ORDER BY revision', (first['candidate_id'],)).fetchall()
        assert json.loads(rows[0]['proposal_json']) == first['proposal']
        assert [r['actor'] for r in rows] == ['policy', 'legacy_backfill_revision']
        assert db.execute('SELECT count(*) FROM memory_l2_jobs').fetchone()[0] == 0


@pytest.mark.parametrize('change', ['outside', 'expand', 'known_actor_changed', 'old_copy_missing', 'source_text_changed'])
def test_participant_rebinding_does_not_relax_source_or_actor_scope(tmp_path, change):
    service, payload, _, first = initial(tmp_path, main=change != 'outside')
    if change == 'outside':
        enable_exact_bindings(service.config, 'p1', 'c1')
    else:
        enable_exact_bindings(service.config)
    if change == 'expand':
        # First card observed a strict subrange; a new attribution grants no
        # access to the rest of the already archived message.
        # Recreate a clean independent base with a shortened observation.
        service, payload, _, first = initial(tmp_path / 'range')
        payload['sources'][0]['_model_range'] = {'start': 0, 'end': len(payload['sources'][0]['text']) - 1}
        # The existing first row was full range, so use a new model revision
        # to narrow it before attempting an identity change plus expansion.
        first = run(service.stage('old-memory', refresh(payload, first, service.config)))['candidate']
        enable_exact_bindings(service.config)
    request = refresh(payload, first, service.config)
    if change == 'expand':
        request['request_id'] += '-expand'; request['sources'][0].pop('_model_range')
    elif change == 'known_actor_changed':
        first = run(service.stage('old-memory', request))['candidate']
        for item in service.config['identity']['source_participants']:
            if item['role'] == 'user': item['participant']['subject_id'] = 'peer:different'
        request = refresh(payload, first, service.config); request['request_id'] += '-again'
    elif change == 'old_copy_missing':
        old = first['proposal']['metadata']['memory_l1']['evidence_bindings'][0]
        with service.authority._connect() as db, db:
            db.execute('DELETE FROM memory_l0_source_copies WHERE revision_json=?', (json.dumps(old['source_revision'], ensure_ascii=False),))
    elif change == 'source_text_changed':
        with sqlite3.connect(tmp_path / 'buckets/gateway_state.db') as db:
            db.execute("UPDATE conversation_turns SET user_text='来源正文变了'")
    with pytest.raises(L1OwnerError, match='backfill_revision_source_scope_changed|backfill_source_not_currently_readable'):
        run(service.stage('old-memory', request))
    assert service.authority.get_candidate(first['candidate_id']) == first


def test_owner_edit_is_protected_even_after_the_owner_authorizes_attribution(tmp_path):
    service, payload, _, first = initial(tmp_path)
    doc = join_document(first['proposal']['proposed_body'], first['proposal']['metadata']['memory_l1'])
    doc['narrative']['body'] = '主人保存的稿。'
    owner = save_owner_candidate(service.authority, first['candidate_id'], edit={'document': doc},
        expected_revision=first['revision'], request_id='owner-saved', revision_service=service.revisions)
    enable_exact_bindings(service.config)
    with pytest.raises(L1OwnerError, match='backfill_candidate_owner_modified'):
        run(service.stage('old-memory', refresh(payload, owner, service.config)))
    assert service.authority.get_candidate(owner['candidate_id']) == owner


def test_external_document_and_its_quoted_assistant_role_never_gain_this_rebind_path():
    from memory_legacy_participants import participant_only_rebinding
    config = {}
    enable_exact_bindings(config)
    external = {'system': 'aizizhu', 'namespace': 'attachment_text', 'profile_id': 'jiajia-main',
        'conversation_id': 'jiajia-main', 'role': 'assistant', 'text': '引文：我是顾衍。'}
    config['identity']['source_participants'].append({**external, 'participant': {
        'subject_id': 'participant:9bebdca0f18ac25be0a898b9:self', 'kind': 'ai', 'display_name': '顾衍'}})
    assert source_participant(external, config)['resolution'] == 'external_document'
    assert source_participant(external, config)['subject_id'] is None
    binding = {'binding_id': 'source:1', 'source_namespace': 'snapshot.aiz.attachment_text', 'source_id': '{}'}
    assert participant_only_rebinding(binding, binding, readers=None, config=config) is None
