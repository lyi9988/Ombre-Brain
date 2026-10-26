"""New daily writes cannot silently bypass L1; old receipts stay readable."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from reflection_engine import ReflectionEngine
import reflection_engine
from memory_candidate_preparation import MemoryCandidatePreparation
from test_memory_authority_r1 import reflection_config, daily_candidate, attach_fake_commit_service


def run(value):
    return asyncio.run(value)


def test_target_module_is_the_isolated_tree():
    assert Path(reflection_engine.__file__).resolve().parent == Path(__file__).resolve().parents[1]


def test_new_auto_body_only_stays_visible_failed_without_commit_or_retry_churn(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='auto'))
    projection = attach_fake_commit_service(engine)
    draft = daily_candidate()
    first = run(engine._write_daily_chat_memory_candidates([draft], object()))
    row = engine.memory_authority_store.get_candidate(draft['id'])
    assert first['created'] == 0 and first['failed'] == 1
    assert first['results'][0]['reason'] == 'first_pass_document_required'
    assert row['status'] == 'commit_failed'
    assert row['proposal']['proposed_body'] == draft['content']
    assert engine.memory_authority_store.get_memory(draft['id']) is None
    assert projection.revisions == []
    again = run(engine._write_daily_chat_memory_candidates([draft], object()))
    assert again['results'] == first['results']
    assert engine.memory_authority_store.get_candidate(draft['id'])['revision'] == row['revision']


def test_review_is_still_review_but_accepting_missing_l1_is_not_formal_commit(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    projection = attach_fake_commit_service(engine)
    draft = daily_candidate()
    engine._store_daily_chat_memory_pending([draft])
    row = engine.memory_authority_store.get_candidate(draft['id'])
    assert row['status'] == 'pending'
    pending = run(engine._write_daily_chat_memory_candidates_authority([{**draft, 'mode': 'review'}], object()))
    assert pending['results'][0]['status'] == 'pending'
    assert pending['results'][0]['preparation_status'] == 'required'
    result = run(engine.confirm_daily_chat_memory([draft['id']], object(), action='confirm', request_id='review-once'))
    assert result['created'] == 0
    assert engine.memory_authority_store.get_memory(draft['id']) is None
    assert projection.revisions == []
    assert engine.memory_authority_store.get_candidate(draft['id'])['proposal']['proposed_body'] == draft['content']


def test_original_committed_body_is_not_rewritten_or_forced_to_regenerate(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='auto'))
    projection = attach_fake_commit_service(engine)
    draft = daily_candidate()
    proposal = engine._daily_chat_memory_proposal(draft, mode='auto')
    row = engine.memory_authority_store.put_candidate(proposal, engine.memory_ingestion_policy.evaluate(proposal))
    service = engine._memory_authority_service(object())
    run(service.commit_memory(memory_id=draft['id'], bucket_id=draft['id'], expected_revision=0,
        body=draft['content'], metadata={'source': 'daily_chat_memory'}, source_refs=proposal.source_refs,
        decision_source='auto', idempotency_key='historical-original', actor='daily_chat_memory'))
    engine.memory_authority_store.decide_candidate(draft['id'], action='commit', expected_revision=row['revision'],
        request_id='historical-receipt', actor='commit_service', reason_codes=['MEMORY_REVISION_COMMITTED'])
    result = run(engine._write_daily_chat_memory_candidates([draft], object()))
    assert result['exists'] == 1 and result['created'] == 0
    assert len(projection.revisions) == 1
    assert engine.memory_authority_store.get_memory(draft['id'])['active_revision'] == 1


def test_body_only_response_remains_a_draft_not_a_fabricated_document(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    body = '保留完整旧提示词成稿，不把它假装成L1。'
    async def create(*args, **kwargs):
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates': [
            {'kind': 'key_event', 'content': body, 'confidence': .9}]})), finish_reason='stop')])
    engine._daily_chat_memory_create_completion = create
    result, _ = run(engine._extract_window_candidates('2026-10-25', [
        {'id': 11, 'session_id': 's', 'user_text': '请保留我说的这个事实。', 'assistant_text': '',
         'created_at': '2026-10-25T00:00:00Z'}]))
    assert result[0]['l1_state'] == 'preparation_required'
    assert result[0]['content'] == body
    assert 'document' not in result[0] and '_l1_generation' not in result[0]


def test_normalized_body_only_is_not_truncated_and_preparation_receipt_exposes_failure(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    body = '我明确说过不喜欢苦味明显的黑咖啡，喜欢加奶。' * 80
    raw = {'kind': 'stable_preference', 'title': '咖啡偏好', 'content': body, 'confidence': .9,
           'source_turn_ids': ['11'], 'source_event_ids': [], 'reason': '明确表达', 'tags': ['偏好']}
    turns = [{'id': 11, 'session_id': 's', 'user_text': body, 'assistant_text': '',
              'created_at': '2026-10-25T00:00:00Z'}]
    audit = {}
    rows = engine._normalize_daily_chat_memory_candidates('2026-10-25', [raw], turns,
        max_candidates=10, min_confidence=.5, mode='review', audit=audit)
    assert len(rows) == 1, audit
    row = rows[0]
    assert row['content'] == body
    assert row['l1_state'] == 'preparation_required'
    assert row['l1_issue']['code'] == 'first_pass_document_required'
    assert 'memory_l1' not in row
    queue = MemoryCandidatePreparation(engine.memory_authority_store)
    receipt = queue.prepare('partial', [engine._daily_chat_memory_proposal(row, mode='review')],
        engine.memory_ingestion_policy, scope='day', source_fingerprint='fixture',
        issue={'code': 'first_pass_document_required', 'pending_count': 1})
    assert receipt['requires_manual_retry'] is True
    stored = queue.rows(receipt)[0]
    assert stored['proposal']['proposed_body'] == body
    assert stored['status'] == 'pending'
    assert queue.get('partial')['issue']['code'] == 'first_pass_document_required'


@pytest.mark.parametrize('commit_state', ['prepared', 'body_written', 'committed'])
@pytest.mark.parametrize('candidate_failed', [False, True])
def test_original_prepared_operation_survives_candidate_revision_change(tmp_path, commit_state, candidate_failed):
    from memory_commit_service import body_sha256
    engine = ReflectionEngine(reflection_config(tmp_path, mode='auto'))
    projection = attach_fake_commit_service(engine)
    draft = daily_candidate()
    proposal = engine._daily_chat_memory_proposal(draft, mode='auto')
    authority = engine.memory_authority_store
    row = authority.put_candidate(proposal, engine.memory_ingestion_policy.evaluate(proposal))
    key = f"daily-chat:{draft['id']}:candidate-revision:{row['revision']}"
    prepared = authority.prepare_memory_commit(memory_id=draft['id'], bucket_id=draft['id'], expected_revision=0,
        body_sha256=body_sha256(draft['content']),
        snapshot_path=f"memory-revisions/{draft['id']}/revision-00000001.md",
        metadata={'source': 'daily_chat_memory', 'created': '2026-10-20T00:00:00Z'},
        source_refs=proposal.source_refs, decision_source='auto', idempotency_key=key, actor='daily_chat_memory')
    if commit_state != 'prepared':
        authority.record_body_written(prepared['operation_id'], observed_body_sha256=body_sha256(draft['content']))
    if commit_state == 'committed':
        authority.finalize_memory_commit(prepared['operation_id'])
    if candidate_failed:
        authority.decide_candidate(draft['id'], action='commit_failed', expected_revision=row['revision'],
            request_id='old-failure', actor='commit_service', reason_codes=['OLD_INTERRUPTED'])
    original = authority.get_commit(prepared['operation_id'])
    result = run(engine._write_daily_chat_memory_candidates([draft], object()))
    assert result['failed'] == 0, result
    after = authority.get_commit_by_idempotency(key)
    assert after['operation_id'] == original['operation_id']
    assert after['payload'] == original['payload']
    assert after['status'] == 'committed'
    assert authority.get_memory(draft['id'])['active_revision'] == 1
    assert authority.get_candidate(draft['id'])['status'] == 'committed'
    assert len(projection.revisions) == (1 if commit_state == 'prepared' else 0)
    with authority._connect() as db:
        assert db.execute("SELECT count(*) FROM commit_log WHERE aggregate_id=? AND operation_kind='memory_revision'", (draft['id'],)).fetchone()[0] == 1
    again = run(engine._write_daily_chat_memory_candidates([draft], object()))
    assert again['exists'] == 1


@pytest.mark.parametrize('mode', ['auto', 'review'])
@pytest.mark.parametrize('owner_edited', [False, True])
def test_corrected_l1_completes_only_untouched_draft_and_preserves_history(tmp_path, mode, owner_edited):
    from dataclasses import replace
    from memory_l1_schema import split_document
    from test_memory_l1_ingestion_p2c import setup, prepare
    service, source, draft, generation = setup(tmp_path / 'source')
    document = prepare((service, source, draft, generation))
    body, semantic = split_document(document)
    engine = ReflectionEngine(reflection_config(tmp_path / 'engine', mode=mode))
    authority = engine.memory_authority_store
    candidate = {**daily_candidate(), 'content': body, 'proposed_memory': body,
                 'l1_state': 'preparation_required', 'mode': mode}
    initial = engine._daily_chat_memory_proposal(candidate, mode=mode)
    queue = MemoryCandidatePreparation(authority)
    first = queue.prepare('original-window', [initial], engine.memory_ingestion_policy,
        scope='day', source_fingerprint='same-window', issue={'code': 'first_pass_document_required'})
    row = authority.get_candidate(initial.proposal_id)
    if row['status'] == 'accepted':
        row = authority.decide_candidate(initial.proposal_id, action='commit_failed', expected_revision=row['revision'],
            request_id='missing-l1', actor='commit_service', reason_codes=['L1_PREPARATION_REQUIRED'])
    if owner_edited:
        authority.revise_candidate(initial.proposal_id, expected_revision=row['revision'],
            proposal=replace(initial, proposed_body='主人保留的修订稿'), request_id='owner-edit', actor='owner')
    queue.acknowledge('original-window', [initial.proposal_id])
    completed = replace(initial, metadata={**initial.metadata, 'memory_l1': semantic,
        'legacy_candidate': {**candidate, 'l1_state': 'ready'}})
    recovered = queue.prepare('original-window', [completed], engine.memory_ingestion_policy,
        scope='day', source_fingerprint='same-window', replace_failure=True)
    saved = authority.get_candidate(initial.proposal_id)
    if owner_edited:
        assert saved['proposal']['proposed_body'] == '主人保留的修订稿'
        assert 'memory_l1' not in saved['proposal']['metadata']
        assert recovered['requires_manual_retry'] is True
        assert recovered['issue']['code'] == 'first_pass_document_required'
    else:
        assert saved['proposal']['metadata']['memory_l1'] == semantic
        assert saved['status'] == ('accepted' if mode == 'auto' else 'pending')
        assert recovered['processed_ids'] == []
        assert len(queue.remaining(recovered)) == 1
        with authority._connect() as db:
            old = db.execute('SELECT proposal_json FROM candidate_revisions WHERE candidate_id=? AND revision=1', (initial.proposal_id,)).fetchone()[0]
        assert json.loads(old)['proposed_body'] == body
    assert authority.get_memory(initial.proposal_id) is None
