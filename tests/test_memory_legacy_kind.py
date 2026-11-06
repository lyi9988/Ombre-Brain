"""P7 native kind survives model repair and replaces only proven machine terms."""
from copy import deepcopy

import pytest

from memory_authority import MemoryIngestionPolicy, MemoryProposal
from memory_candidate_edit import save_owner_candidate
from memory_l1_schema import join_document
from memory_l1_service import L1OwnerError
from memory_metadata import normalize_memory_metadata
from test_memory_legacy_backfill import run, setup
from test_memory_legacy_model_revision import revision_payload


def test_explicit_native_kind_survives_granularity_revision_and_adoption(tmp_path):
    service, payload, _ = setup(tmp_path)
    payload.update(kind=' Relationship_Anchor ', kind_generation=deepcopy(payload['generation']))
    first = run(service.stage('m1', payload))['candidate']
    assert first['proposal']['memory_type'] == first['proposal']['metadata']['legacy_candidate']['kind'] == 'relationship_anchor'
    repaired = revision_payload(payload, first)
    repaired.pop('kind'); repaired.pop('kind_generation')
    repaired['generation']['task_id'] = 'new-granularity-only-task'
    second = run(service.stage('m1', repaired))['candidate']
    kind = second['proposal']['metadata']['legacy_backfill']['candidate_kind']
    assert kind['value'] == 'relationship_anchor' and kind['generation'] == payload['kind_generation']
    assert second['proposal']['metadata']['memory_l1']['generation']['task_id'] == 'new-granularity-only-task'
    run(service.adopt(second, request_id='private-owner-accept-kind'))
    record = service.authority.get_memory_revision('m1', 2)
    assert record['metadata']['memory_type'] == 'relationship_anchor'
    assert normalize_memory_metadata(run(service.revisions.buckets.get('b1')))['kind'] == 'event'
    assert record['metadata']['memory_l1']['statements'][0]['classification']['topics'] == []


def test_old_request_stays_unknown_and_explicit_original_generation_repairs_it(tmp_path):
    service, payload, _ = setup(tmp_path)
    # Words in a document must never invent an admission category.
    payload['document']['retrieval']['keywords'] += ['稳定偏好', '承诺', 'intimacy']
    first = run(service.stage('m1', payload))['candidate']
    assert first['proposal']['memory_type'] == 'memory'
    assert first['proposal']['metadata']['legacy_backfill']['candidate_kind']['state'] == 'unknown'
    update = revision_payload(payload, first)
    update['kind'] = 'relationship_anchor'
    with pytest.raises(L1OwnerError, match='backfill_kind_generation_required'):
        run(service.stage('m1', update))
    update['kind_generation'] = deepcopy(payload['generation'])
    fixed = run(service.stage('m1', update))['candidate']
    assert fixed['candidate_id'] == first['candidate_id'] and fixed['revision'] == 2
    assert fixed['proposal']['memory_type'] == 'relationship_anchor'
    assert fixed['proposal']['metadata']['legacy_backfill']['candidate_kind']['generation'] == payload['generation']
    with service.authority._connect() as db:
        assert db.execute('SELECT count(*) FROM candidate_revisions WHERE candidate_id=?', (first['candidate_id'],)).fetchone()[0] == 2


@pytest.mark.parametrize('kind', ['intimacy', 'experience', 'profile_fact', 7])
def test_domain_statement_kind_and_unknown_text_are_not_native_candidate_kinds(tmp_path, kind):
    service, payload, _ = setup(tmp_path)
    payload['kind'] = kind
    with pytest.raises(L1OwnerError, match='invalid_backfill_kind'):
        run(service.stage('m1', payload))
    assert not service.authority.list_candidates(status='all')


def test_kind_correction_cannot_overwrite_an_owner_l1_edit(tmp_path):
    service, payload, _ = setup(tmp_path)
    row = run(service.stage('m1', payload))['candidate']
    doc = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    doc['narrative']['title'] = '主人自己的分类说明'
    owner = save_owner_candidate(service.authority, row['candidate_id'], edit={'document': doc},
        expected_revision=row['revision'], request_id='owner-before-kind-repair', revision_service=service.revisions)
    repair = revision_payload(payload, owner)
    repair.update(kind='relationship_anchor', kind_generation=deepcopy(payload['generation']))
    with pytest.raises(L1OwnerError, match='backfill_candidate_owner_modified'):
        run(service.stage('m1', repair))
    assert service.authority.get_candidate(row['candidate_id']) == owner


def old_generator(service, record):
    tags = ['from_daily_chat', 'daily_chat_extract', 'stable_preference', 'communication_preference', 'old-machine-game']
    keywords = ['old-machine-preference-phrase']
    proposal = MemoryProposal.from_mapping({'proposal_id': 'old-generator', 'source_type': 'daily_chat',
        'proposed_body': '完整的合成回忆。', 'source_refs': ['conversation_turn:1'], 'source_status': 'verified',
        'memory_type': 'stable_preference', 'requested_mode': 'review',
        'metadata': {'legacy_candidate': {'tags': tags, 'keywords': keywords}}})
    service.authority.put_candidate(proposal, MemoryIngestionPolicy().evaluate(proposal))
    service.authority.decide_candidate('old-generator', action='accept', expected_revision=1, request_id='old-accepted', actor='owner')
    service.authority.decide_candidate('old-generator', action='commit', expected_revision=2, request_id='old-committed', actor='commit_service')
    result = deepcopy(record)
    result['metadata'].update(source='daily_chat_memory', from_daily_chat=True, daily_chat_memory_candidate_id='old-generator',
        domain=['主人自定义领域'], canonical_domain='inner', tags=[*tags, 'owner-custom-label'],
        keywords=[*keywords, 'owner-search-word'], importance=8, protected=True)
    return result


def test_adoption_projection_retires_exact_old_generator_but_preserves_owner_fields(tmp_path):
    from memory_legacy_classification import active_metadata
    service, payload, _ = setup(tmp_path)
    payload.update(kind='relationship_anchor', kind_generation=deepcopy(payload['generation']))
    candidate = run(service.stage('m1', payload))['candidate']
    old = old_generator(service, service.authority.get_memory_revision('m1', 1))
    projected, report = active_metadata(service.authority, old, candidate['proposal'])
    assert projected['domain'] == ['主人自定义领域'] and projected['canonical_domain'] == 'inner'
    assert projected['importance'] == 8 and projected['protected'] is True
    assert 'stable_preference' not in projected['tags'] and 'communication_preference' not in projected['tags']
    assert 'old-machine-preference-phrase' not in projected['keywords']
    assert 'owner-custom-label' in projected['tags'] and 'owner-search-word' in projected['keywords']
    assert set(candidate['proposal']['metadata']['memory_l1']['retrieval']['keywords']) <= set(projected['keywords'])
    assert report['original_generator']['candidate_id'] == 'old-generator'
    assert normalize_memory_metadata({'metadata': projected})['kind'] == 'event'
    assert old['metadata']['tags'][2] == 'stable_preference'


def test_no_generator_proof_does_not_erase_custom_labels_or_domains(tmp_path):
    from memory_legacy_classification import active_metadata
    service, payload, _ = setup(tmp_path)
    payload.update(kind='relationship_anchor', kind_generation=deepcopy(payload['generation']))
    candidate = run(service.stage('m1', payload))['candidate']
    old = deepcopy(service.authority.get_memory_revision('m1', 1))
    old['metadata'].update(domain=['custom-owner-domain'], tags=['owner-preference-label'], keywords=['owner-custom-word'], kind='reflection')
    projected, report = active_metadata(service.authority, old, candidate['proposal'])
    assert report['original_generator'] is None
    assert projected['domain'] == ['custom-owner-domain'] and projected['kind'] == 'reflection'
    assert 'owner-preference-label' in projected['tags'] and 'owner-custom-word' in projected['keywords']


def test_recorded_owner_metadata_edit_disables_machine_term_retirement(tmp_path):
    from memory_legacy_classification import active_metadata
    service, payload, _ = setup(tmp_path)
    payload.update(kind='relationship_anchor', kind_generation=deepcopy(payload['generation']))
    candidate = run(service.stage('m1', payload))['candidate']
    old = old_generator(service, service.authority.get_memory_revision('m1', 1))
    old['decision_source'] = 'owner_metadata_edit'
    projected, report = active_metadata(service.authority, old, candidate['proposal'])
    assert report['original_generator'] is None
    assert set(old['metadata']['tags']) <= set(projected['tags'])
    assert set(old['metadata']['keywords']) <= set(projected['keywords'])


def test_missing_kind_keeps_original_memory_type_without_inventing_one(tmp_path):
    from memory_legacy_classification import active_metadata
    service, payload, _ = setup(tmp_path)
    candidate = run(service.stage('m1', payload))['candidate']
    old = service.authority.get_memory_revision('m1', 1)
    assert 'memory_type' not in old['metadata']
    assert 'memory_type' not in active_metadata(service.authority, old, candidate['proposal'])[0]
    old['metadata']['memory_type'] = 'owner_custom_classification'
    assert active_metadata(service.authority, old, candidate['proposal'])[0]['memory_type'] == 'owner_custom_classification'
