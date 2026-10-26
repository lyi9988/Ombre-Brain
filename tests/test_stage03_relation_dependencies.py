"""Actual L1 commits, owner adoption and scoped relation invalidation."""
import asyncio
from contextlib import closing
from copy import deepcopy

from memory_l1_schema import document_template
from memory_relation_dependencies import dependency_snapshot, relation_context
import test_memory_l1_members_p2b as fixtures
import test_memory_diary_l1_p2b as diary_fixtures
from memory_authority import MemoryProposal
from memory_candidate_edit import save_owner_candidate
from memory_l1_schema import join_document


def setup_relations(tmp_path, *, confirmation='proposed'):
    authority, manager, commit, service, reads, catalog = fixtures.setup_memory(tmp_path)
    docs = {}
    for mid in ('A', 'B', 'C'):
        document = document_template(f'{mid} 的完整原始记忆。', f'记忆 {mid}', catalog=catalog, catalog_revision='catalog-p2b')
        document['statements'] = [fixtures.statement(f'statement-{mid}', f'{mid} 的具体事实。', f'statement-{mid}-r1')]
        if mid == 'B':
            document['relations'] = [{'relation_id': 'relation-BA', 'kind': 'limits',
                'from': {'memory_id': None, 'revision': 'this_revision', 'object_type': 'statement', 'object_id': 'statement-B'},
                'to': {'memory_id': 'A', 'revision': 1, 'object_type': 'statement', 'object_id': 'statement-A'},
                'origin': 'model', 'confirmation_state': confirmation, 'evidence_refs': []}]
        fixtures.commit_l1_fixture(commit, mid, f'bucket-{mid}', 0, document, f'seed-{mid}')
        docs[mid] = document
    return authority, service, docs


def draft(service, mid, request_id):
    detail = asyncio.run(service.detail(mid))
    return {'expected_revision': detail['current_revision'], 'request_id': request_id,
            'document': deepcopy(detail['l1']['document'])}


def dependency(authority, mid):
    with closing(authority._connect()) as conn:
        return dependency_snapshot(conn, mid)


def context(authority, mid):
    with closing(authority._connect()) as conn:
        return relation_context(conn, [mid])


def jobs(authority, mid):
    with closing(authority._connect()) as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM memory_l2_jobs WHERE memory_id=? ORDER BY task_id', (mid,))]


def confirm(service):
    payload = draft(service, 'B', 'owner-confirm-specific-relation')
    payload['document']['relations'][0]['confirmation_state'] = 'confirmed'
    return asyncio.run(service.save('B', payload)), payload


def test_owner_confirmation_creates_actual_receipt_and_bidirectional_dependency_with_targeted_jobs(tmp_path):
    authority, service, _docs = setup_relations(tmp_path)
    before = {mid: dependency(authority, mid) for mid in ('A', 'B', 'C')}
    before_jobs = {mid: jobs(authority, mid) for mid in ('A', 'B', 'C')}
    assert context(authority, 'A')['relations'] == []
    assert context(authority, 'B')['relations'] == []
    saved, payload = confirm(service)
    incoming, outgoing = context(authority, 'A'), context(authority, 'B')
    assert len(incoming['relations']) == len(outgoing['relations']) == 1
    for relation in (incoming['relations'][0], outgoing['relations'][0]):
        assert relation['host_memory_id'] == 'B' and relation['host_revision'] == saved['revision']
        assert relation['from_memory_id'] == 'B' and relation['to_memory_id'] == 'A'
        assert relation['from_id'] == 'statement-B' and relation['to_id'] == 'statement-A'
        assert relation['adoption']['state'] == 'adopted'
    with closing(authority._connect()) as conn:
        receipt = dict(conn.execute('SELECT * FROM memory_relation_adoptions').fetchone())
        operation = dict(conn.execute('SELECT * FROM commit_log WHERE operation_id=?', (receipt['operation_id'],)).fetchone())
    assert receipt['actor'] == 'owner' and receipt['request_id'] == payload['request_id']
    assert operation['status'] == 'committed'
    assert dependency(authority, 'A')['generation'] > before['A']['generation']
    assert dependency(authority, 'A')['token'] != before['A']['token']
    assert len(jobs(authority, 'A')) > len(before_jobs['A'])
    assert dependency(authority, 'C') == before['C'] and jobs(authority, 'C') == before_jobs['C']
    # Original owner request recovery must not create another dependency pulse.
    settled = dependency(authority, 'A')
    queued = jobs(authority, 'A')
    assert asyncio.run(service.save('B', payload))['replayed']
    assert dependency(authority, 'A') == settled and jobs(authority, 'A') == queued
    assert authority.get_memory_revision('B', 1)['metadata']['memory_l1']['relations'][0]['confirmation_state'] == 'proposed'


def test_title_edit_of_model_confirmed_relation_does_not_adopt_until_specific_owner_intent(tmp_path):
    authority, service, _docs = setup_relations(tmp_path, confirmation='confirmed')
    original = dependency(authority, 'A')
    payload = draft(service, 'B', 'owner-title-only')
    payload['document']['narrative']['title'] = '仅改标题，不采纳已有模型关系'
    asyncio.run(service.save('B', payload))
    assert context(authority, 'A')['relations'] == []
    assert dependency(authority, 'A') == original
    explicit = draft(service, 'B', 'owner-adopt-existing-relation')
    explicit['owner_relation_adoptions'] = ['relation-BA']
    asyncio.run(service.save('B', explicit))
    assert context(authority, 'A')['relations'][0]['adoption']['state'] == 'adopted'
    assert dependency(authority, 'A')['generation'] > original['generation']


def test_withdrawal_preserves_scope_guard_and_history_without_reviving_the_previous_fact(tmp_path):
    authority, service, _docs = setup_relations(tmp_path)
    saved, _payload = confirm(service)
    old_a = authority.get_memory_revision('A', 1)
    before = dependency(authority, 'A')
    authority.change_memory_state(memory_id='B', state='archived', recall_policy='disabled',
        expected_revision=saved['revision'], idempotency_key='withdraw-correction-B', actor='owner')
    restricted = context(authority, 'A')['guards']
    assert len(restricted) == 1 and restricted[0]['state'] == 'dependency_withdrawn'
    assert restricted[0]['scope'] == {'statement_ids': ['statement-A'], 'unknown_scope': False, 'historical_only': False}
    assert dependency(authority, 'A')['generation'] > before['generation']
    assert authority.get_memory_revision('A', 1) == old_a
    assert authority.get_memory_revision('B', 1) and authority.get_memory_revision('B', 2)
    # A new title/version alone cannot turn the same stale statement current.
    title = draft(service, 'A', 'A-title-after-withdrawal')
    title['document']['narrative']['title'] = '标题更新，旧陈述未改'
    asyncio.run(service.save('A', title))
    assert context(authority, 'A')['guards'][0]['scope']['statement_ids'] == ['statement-A']
    # An actual independent correction of that statement resolves the old guard
    # into historical context rather than permanently excluding the memory.
    corrected = draft(service, 'A', 'A-actual-correction')
    corrected['document']['statements'][0]['content'] = 'A 已有新的真实修订事实。'
    asyncio.run(service.save('A', corrected))
    scope = context(authority, 'A')['guards'][0]['scope']
    assert scope['statement_ids'] == [] and scope['historical_only'] is True


def diary_relation_candidate(tmp_path):
    text = '早上先约定了一个安排。晚上补充了该安排的适用限制。'
    env = diary_fixtures.setup_diary(tmp_path, text)
    source = diary_fixtures.capture(env)
    statements = [diary_fixtures.make_statement('new:statement-a', '早上的原安排。'),
                  diary_fixtures.make_statement('new:statement-b', '晚上明确了该安排的限制。')]
    scenes = [diary_fixtures.make_scene(env, source, 'new:scene-a', '早上', statement_refs=['new:statement-a']),
              diary_fixtures.make_scene(env, source, 'new:scene-b', '晚上', statement_refs=['new:statement-b'])]
    relation = {'relation_id': 'new:scene-relation', 'kind': 'limits', 'origin': 'model', 'confirmation_state': 'proposed',
        'from': {'memory_id': None, 'revision': 'this_revision', 'object_type': 'scene', 'object_id': 'new:scene-b'},
        'to': {'memory_id': None, 'revision': 'this_revision', 'object_type': 'scene', 'object_id': 'new:scene-a'}, 'evidence_refs': []}
    document = diary_fixtures.make_document(env, source, '整篇第一遍整理，保留早上的安排与晚上的限制。', scenes,
        statements=statements, relations=[relation])
    candidate = diary_fixtures.prepare(env, source, document,
        {'new:scene-a': '早上提出安排的完整场景。', 'new:scene-b': '晚上说明限制的完整场景。'}, request_id='relation-diary-review')
    assert candidate['status'] == 'pending'
    return env, candidate


def test_diary_owner_confirmation_then_review_adopt_records_owner_proof_without_forging_worker_identity(tmp_path):
    env, candidate = diary_relation_candidate(tmp_path)
    proposal = candidate['proposal']
    document = join_document(proposal['proposed_body'], proposal['metadata']['memory_l1'])
    relation_id = document['relations'][0]['relation_id']
    assert document['relations'][0]['confirmation_state'] == 'proposed'
    document['relations'][0]['confirmation_state'] = 'confirmed'
    edited = save_owner_candidate(env['authority'], candidate['candidate_id'], edit={'document': document},
        expected_revision=candidate['revision'], request_id='owner-diary-confirm-relation', revision_service=env['revisions'])
    accepted = diary_fixtures.accept(env, edited)
    result = asyncio.run(env['diary'].adopt(candidate['candidate_id']))
    assert result['status'] == 'complete'
    root_id = proposal['metadata']['diary_l1_bundle']['memory_id']
    root = env['authority'].get_memory_revision(root_id, result['root']['revision'])
    assert root['created_by'] == 'diary_first_pass'
    assert root['metadata']['diary_l1_review'] == {'candidate_id': candidate['candidate_id'], 'candidate_revision': accepted['revision']}
    with closing(env['authority']._connect()) as conn:
        receipt = dict(conn.execute('SELECT * FROM memory_relation_adoptions WHERE host_memory_id=? AND relation_id=?',
            (root_id, relation_id)).fetchone())
        committed = dict(conn.execute('SELECT * FROM commit_log WHERE operation_id=?', (receipt['operation_id'],)).fetchone())
    assert receipt['actor'] == 'owner'
    assert receipt['request_id'] == candidate['candidate_id'] + ':owner-accept'
    assert committed['status'] == 'committed'
    assert context(env['authority'], root_id)['relations'][0]['adoption']['state'] == 'adopted'


def test_diary_review_title_edit_of_legacy_model_confirmed_relation_does_not_adopt_it(tmp_path):
    env, candidate = diary_relation_candidate(tmp_path)
    legacy = deepcopy(candidate['proposal'])
    legacy['metadata']['memory_l1']['relations'][0]['confirmation_state'] = 'confirmed'
    candidate = env['authority'].revise_candidate(candidate['candidate_id'], expected_revision=candidate['revision'],
        proposal=MemoryProposal.from_mapping(legacy), request_id='seed-existing-model-confirmed', actor='legacy_import')
    document = join_document(candidate['proposal']['proposed_body'], candidate['proposal']['metadata']['memory_l1'])
    document['narrative']['title'] = '主人只改日记标题，未明确采用关系'
    edited = save_owner_candidate(env['authority'], candidate['candidate_id'], edit={'document': document},
        expected_revision=candidate['revision'], request_id='owner-diary-title-only', revision_service=env['revisions'])
    accepted = diary_fixtures.accept(env, edited)
    result = asyncio.run(env['diary'].adopt(candidate['candidate_id']))
    assert result['status'] == 'complete'
    root_id = candidate['proposal']['metadata']['diary_l1_bundle']['memory_id']
    root = env['authority'].get_memory_revision(root_id, result['root']['revision'])
    assert root['created_by'] == 'diary_first_pass'
    assert root['metadata']['diary_l1_review']['candidate_revision'] == accepted['revision']
    with closing(env['authority']._connect()) as conn:
        assert conn.execute('SELECT count(*) FROM memory_relation_adoptions WHERE host_memory_id=?', (root_id,)).fetchone()[0] == 0
    assert context(env['authority'], root_id)['relations'] == []
