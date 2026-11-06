"""Model-only repairs keep the original Review candidate and owner boundary."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from memory_candidate_edit import save_owner_candidate
from memory_l1_schema import join_document
from memory_l1_service import L1OwnerError
from test_memory_legacy_backfill_api import app_for
from test_memory_legacy_projection import setup


def run(task):
    return asyncio.run(task)


def revision_payload(original, row, request_id='model-revision-one'):
    payload = deepcopy(original)
    payload.update(request_id=request_id, expected_candidate_revision=row['revision'],
        expected_proposal_sha256=row['proposal_sha256'])
    payload['document']['narrative']['body'] = '合成返修稿：那次她说想要抱抱。'
    payload['generation']['task_id'] = 'synthetic-existing-model-attempt-two'
    return payload


def test_existing_http_post_revises_same_candidate_and_preserves_owner_edits(tmp_path, monkeypatch):
    service, original, snapshot = setup(tmp_path)
    monkeypatch.setenv('OMBRE_MEMORY_WRITE_TOKEN', 'private-model-revision')
    app = app_for(service, service.config)
    before_memory = service.authority.get_memory('old-memory')
    before_revision = service.authority.get_memory_revision('old-memory', 1)
    before_bucket = run(service.revisions.buckets.get('old-memory'))
    before_snapshot = snapshot.read_bytes()
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
            headers={'Authorization': 'Bearer private-model-revision'}) as client:
            url = '/api/memory-authority/memories/old-memory/legacy-backfill'
            created = (await client.post(url, json=original)).json()
            first = created['candidate']
            payload = revision_payload(original, first)
            response = await client.post(url, json=payload)
            assert response.status_code == 200
            repaired = response.json()
            row = repaired['candidate']
            assert repaired['status'] == 'revised' and repaired['adopted'] is False
            assert repaired['candidate_id'] == created['candidate_id'] and row['revision'] == 2 and row['status'] == 'pending'
            assert row['proposal']['metadata']['legacy_backfill']['request_sha256'] == first['proposal']['metadata']['legacy_backfill']['request_sha256']
            assert row['proposal']['metadata']['memory_l1']['generation']['origin'] == 'first_pass_model'
            assert row['proposal']['metadata']['memory_l1']['generation']['actor'] == 'legacy_backfill'
            assert row['proposal']['proposed_body'] == payload['document']['narrative']['body']
            replay = await client.post(url, json=payload)
            assert replay.status_code == 200 and replay.json()['replayed'] and replay.json()['candidate']['revision'] == 2
            initial_replay = await client.post(url, json=original)
            assert initial_replay.status_code == 200 and initial_replay.json()['candidate']['revision'] == 2
            changed = deepcopy(payload)
            changed['document']['narrative']['title'] = '同请求不同内容'
            assert (await client.post(url, json=changed)).json()['error']['code'] == 'idempotency_conflict'
            stale = deepcopy(payload)
            stale['request_id'] = 'another-stale-request'
            rejected = await client.post(url, json=stale)
            assert rejected.status_code == 409 and rejected.json()['error']['code'] == 'backfill_candidate_revision_conflict'
            again_payload = revision_payload(original, row, 'model-revision-two')
            again_payload['document']['narrative']['title'] = '合成第二次程序返修'
            second = await client.post(url, json=again_payload)
            assert second.status_code == 200 and second.json()['candidate']['revision'] == 3
            latest = second.json()['candidate']
            document = join_document(latest['proposal']['proposed_body'], latest['proposal']['metadata']['memory_l1'])
            document['narrative']['body'] = '主人手动保存的正文，必须保留。'
            edited = save_owner_candidate(service.authority, latest['candidate_id'], edit={'document': document},
                expected_revision=latest['revision'], request_id='owner-saved-between-model-runs', revision_service=service.revisions)
            protected = await client.post(url, json=revision_payload(original, edited, 'model-after-owner-edit'))
            assert protected.status_code == 409 and protected.json()['error']['code'] == 'backfill_candidate_owner_modified'
            replay_after_owner = await client.post(url, json=payload)
            assert replay_after_owner.status_code == 200 and replay_after_owner.json()['replayed']
            assert service.authority.get_candidate(latest['candidate_id'])['proposal']['proposed_body'] == document['narrative']['body']
            with service.authority._connect() as db:
                history = db.execute('SELECT revision,proposal_json,actor FROM candidate_revisions WHERE candidate_id=? ORDER BY revision',
                                     (latest['candidate_id'],)).fetchall()
                assert len(history) == 4 and json.loads(history[0]['proposal_json']) == first['proposal']
                assert [entry['actor'] for entry in history] == ['policy', 'legacy_backfill_revision', 'legacy_backfill_revision', 'owner']
                reasons = [json.loads(entry[0]) for entry in db.execute("SELECT reason_codes_json FROM candidate_decisions WHERE candidate_id=? AND actor='legacy_backfill_revision'",
                    (latest['candidate_id'],))]
                assert reasons == [['MODEL_DRAFT_REVISED'], ['MODEL_DRAFT_REVISED']]
                assert db.execute('SELECT count(*) FROM memory_l2_jobs').fetchone()[0] == 0
    run(scenario())
    assert service.authority.get_memory('old-memory') == before_memory
    assert service.authority.get_memory_revision('old-memory', 1) == before_revision
    assert run(service.revisions.buckets.get('old-memory')) == before_bucket
    assert snapshot.read_bytes() == before_snapshot


@pytest.mark.parametrize('change,code', [
    ('wrong_hash', 'backfill_candidate_proposal_conflict'),
    ('accepted', 'backfill_candidate_adoption_started'),
    ('rejected', 'backfill_candidate_review_decided'),
    ('deferred', 'backfill_candidate_review_decided'),
    ('defer_reopen', 'backfill_candidate_review_decided'),
    ('archived', 'backfill_memory_not_active'),
    ('inflight', 'backfill_target_commit_in_progress'),
    ('source_changed', 'backfill_source_not_currently_readable'),
])
def test_program_revision_rejects_changed_or_reviewed_candidates(tmp_path, change, code):
    service, original, _ = setup(tmp_path)
    row = run(service.stage('old-memory', original))['candidate']
    if change in {'accepted', 'rejected', 'deferred', 'defer_reopen'}:
        action = {'accepted': 'accept', 'rejected': 'reject', 'deferred': 'defer', 'defer_reopen': 'defer'}[change]
        row = service.authority.decide_candidate(row['candidate_id'], action=action, expected_revision=row['revision'],
            request_id='owner-decision', actor='owner')
        if change == 'defer_reopen':
            row = service.authority.decide_candidate(row['candidate_id'], action='reopen', expected_revision=row['revision'],
                request_id='owner-reopened', actor='owner')
    elif change == 'archived':
        service.authority.change_memory_state(memory_id='old-memory', state='archived', recall_policy='disabled',
            idempotency_key='owner-archived', actor='owner')
    elif change == 'inflight':
        record = service.authority.get_memory_revision('old-memory', 1)
        service.authority.prepare_memory_commit(memory_id='old-memory', bucket_id='old-memory', expected_revision=1,
            body_sha256=record['body_sha256'], snapshot_path=str(service.revisions.commit.projection.snapshot_path('old-memory', 2)),
            metadata=record['metadata'], source_refs=record['source_refs'], decision_source='owner', idempotency_key='other-inflight', actor='owner')
    elif change == 'source_changed':
        import sqlite3
        with sqlite3.connect(tmp_path / 'buckets' / 'gateway_state.db') as db:
            db.execute("UPDATE conversation_turns SET user_text='原话已经变化' WHERE id=1")
    payload = revision_payload(original, row)
    if change == 'wrong_hash':
        payload['expected_proposal_sha256'] = '0' * 64
    before = deepcopy(service.authority.get_candidate(row['candidate_id']))
    with pytest.raises(L1OwnerError, match=code):
        run(service.stage('old-memory', payload))
    assert service.authority.get_candidate(row['candidate_id']) == before


def test_model_revision_cannot_expand_previously_observed_source_range(tmp_path):
    service, original, _ = setup(tmp_path)
    original['sources'][0]['_model_range'] = {'start': 0, 'end': len(original['sources'][0]['text']) - 1}
    row = run(service.stage('old-memory', original))['candidate']
    payload = revision_payload(original, row)
    payload['sources'][0].pop('_model_range')
    with pytest.raises(L1OwnerError, match='backfill_revision_source_scope_changed'):
        run(service.stage('old-memory', payload))
    assert service.authority.get_candidate(row['candidate_id']) == row


def test_revision_preconditions_are_a_pair(tmp_path):
    service, original, _ = setup(tmp_path)
    original['expected_candidate_revision'] = 1
    with pytest.raises(L1OwnerError, match='backfill_model_revision_preconditions_required'):
        run(service.stage('old-memory', original))


def test_owner_edit_while_sources_are_read_is_not_replaced(tmp_path, monkeypatch):
    service, original, _ = setup(tmp_path)
    row = run(service.stage('old-memory', original))['candidate']
    read_sources = service._sources
    async def edit_during_source_read(sources):
        result = await read_sources(sources)
        document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
        document['narrative']['body'] = '主人并发保存的正文。'
        save_owner_candidate(service.authority, row['candidate_id'], edit={'document': document},
            expected_revision=row['revision'], request_id='concurrent-owner', revision_service=service.revisions)
        return result
    monkeypatch.setattr(service, '_sources', edit_during_source_read)
    with pytest.raises(L1OwnerError, match='backfill_candidate_owner_modified'):
        run(service.stage('old-memory', revision_payload(original, row)))
    assert service.authority.get_candidate(row['candidate_id'])['proposal']['proposed_body'] == '主人并发保存的正文。'


def test_concurrent_same_request_returns_one_persisted_model_revision(tmp_path):
    service, original, _ = setup(tmp_path)
    row = run(service.stage('old-memory', original))['candidate']
    payload = revision_payload(original, row)
    async def race():
        return await asyncio.gather(service.stage('old-memory', payload), service.stage('old-memory', payload))
    results = run(race())
    assert sorted(result['replayed'] for result in results) == [False, True]
    assert all(result['candidate']['revision'] == 2 for result in results)
    with service.authority._connect() as db:
        assert db.execute('SELECT count(*) FROM candidate_revisions WHERE candidate_id=?', (row['candidate_id'],)).fetchone()[0] == 2
