import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from b1_protocol import digest
from memory_authority import MemoryIngestionPolicy
from memory_l0_sources import MemoryL0Sources
from memory_l1_service import L1OwnerError
from memory_l3_followup import MemoryFollowups, FollowupTasks, run_followup_tasks, validate_submission
from memory_l3_records import MemoryL3Records
from persona_memory_reaction_context import memory_ref
from test_memory_l2_p4 import setup, run, owner_payload


def payload(kind='correction', value='explicit_owner_correction'):
    user, assistant = '那次不是我想要抱抱，是你先说想抱我。', '我明白了，是我先提出的。'
    turn = {'conversation_id': 'c1', 'user_event': {'id': 'u1', 'source_event_id': 'su1', 'role': 'user', 'content': user},
        'assistant_event': {'id': 'a1', 'source_event_id': 'sa1', 'role': 'assistant', 'content': assistant}}
    fragment = {'memory_id': 'm1', 'bucket_id': 'b1', 'l1_revision': 2, 'l2_revision': None,
        'source_group_id': 'source-group-m1', 'text': '那次她想要抱抱。',
        'source_refs': [], 'coverage': {}}
    import hashlib
    fragment['text_sha256'] = hashlib.sha256(fragment['text'].encode()).hexdigest()
    fragment['fragment_id'] = 'fragment_' + digest(fragment)[:32]
    context = {'schema': 'memory-context-v1', 'scope': {'conversation_id': 'c1', 'user_source_event_id': 'su1',
        'assistant_source_event_id': 'sa1'}, 'fragments': [fragment]}
    context['delivery_sha256'] = digest(context)
    context['delivery_receipt_id'] = 'delivery-1'
    source = {'conversation_key': digest('c1'), 'user_id': 'u1', 'assistant_id': 'a1', 'user': user, 'assistant': assistant}
    update = {'memory_ref': memory_ref(fragment), 'kind': kind, 'long_term_value': value,
        'content': '那次是我先说想抱她。', 'evidence_spans': [
            {'role': 'user', 'start': 0, 'end': len(user), 'evidence_hash': digest(user)}]}
    update['update_id'] = 'ml3:' + digest({'source_hash': digest(source), **update})
    sources = [{'event_id': turn[role + '_event']['id'], 'version_id': role + '-version-1',
        'role': role, 'text': turn[role + '_event']['content'], 'created_at': '2026-10-26T02:00:00+00:00'}
        for role in ('user', 'assistant')]
    return {'context': context, 'turn': turn, 'updates': [update], 'sources': sources}


def fixture(tmp_path, *, mode='auto'):
    l2, authority, _ = setup(tmp_path)
    engine = l2.engine
    engine.config = {}; engine.memory_authority_store = authority
    engine.memory_authority_enabled = True
    engine.memory_semantic_profile_id = 'test-profile'; engine.daily_chat_memory_mode = mode
    engine.daily_chat_memory_timeout_seconds = 180.0
    engine.memory_ingestion_policy = MemoryIngestionPolicy()
    engine._memory_l1_revisions = lambda mgr: l2.revisions
    engine._parse_json_object = json.loads
    reader = MemoryL0Sources(authority)
    class Adapter:
        namespace = 'snapshot.aiz.canonical_event'
        available = True
        async def read(self, binding):
            return reader.copy(binding) if self.available else {'state': 'withdrawn'}
    adapter = Adapter(); reader.adapters[adapter.namespace] = adapter
    engine._memory_l1_sources = lambda: reader
    calls = []
    async def complete(client, **kwargs):
        calls.append(kwargs)
        material = json.loads(kwargs['messages'][1]['content'])
        doc = material['material']['l1_document']
        sid = doc['statements'][0]['statement_id']
        binding = material['source_evidence'][0]['binding']['binding_id']
        plan = {'decision': 'revise_l1', 'reason': '她明确更正了当时的发起人。', 'confidence': .9,
            'affected_statement_ids': [sid], 'patch': [{'op': 'replace', 'path': '/statements/' + sid + '/content',
                'old': doc['statements'][0]['content'], 'value': '那次是我先说想抱她。', 'evidence_refs': [binding]}]}
        return NS(text=json.dumps(plan, ensure_ascii=False), finish='stop')
    engine._daily_chat_memory_create_completion = complete
    async def resolve(context, turn):
        return {'state': 'available', 'eligible_fragment_ids': [f['fragment_id'] for f in context['fragments']]}
    service = MemoryFollowups(engine, None, resolve)
    return service, l2, calls, adapter


def test_only_durable_sourced_updates_queue_and_replay(tmp_path):
    service, _, calls, _ = fixture(tmp_path)
    assert run(service.submit(payload(value='none')))['tasks'] == []
    initial = run(service.submit(payload()))
    assert len(initial['tasks']) == 1 and calls == []
    assert run(service.submit(payload())) == initial
    changed = payload(); changed['updates'][0]['content'] = '伪造新内容'
    with pytest.raises(L1OwnerError): run(service.submit(changed))
    assert len(service.tasks.list()) == 1


def test_auto_updates_existing_l1_once_and_keeps_owner_l2(tmp_path):
    service, l2, calls, _ = fixture(tmp_path)
    # Create a real owner L2 before the followup. New L1 must not overwrite it.
    original_complete = l2.engine._daily_chat_memory_create_completion
    async def second_pass(client, **kwargs):
        return NS(text=json.dumps({'full_injection_text': '主人自己的写法。', 'coherent_sections': [],
            'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}), finish='stop')
    l2.engine._daily_chat_memory_create_completion = second_pass
    run(l2.run_once(1))
    run(l2.save('m1', owner_payload(l2)))
    l2.engine._daily_chat_memory_create_completion = original_complete
    run(service.submit(payload()))
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert result['phase'] == 'prepared', result
    assert len(calls) == 1
    assert service.authority.get_memory('m1')['active_revision'] == 3
    cid = result['candidate_ids'][0]
    assert service.authority.get_candidate(cid)['status'] == 'committed'
    assert run(service.adopt(service.authority.get_candidate(cid)))['status'] == 'exists'
    assert service.authority.get_memory('m1')['active_revision'] == 3
    detail = l2.detail('m1')['l2']
    assert detail['presentation']['document']['full_injection_text'] == '主人自己的写法。'
    assert detail['state'] != 'ready'
    events = MemoryL3Records(service.authority, l2.revisions.commit).timeline('m1')['items']
    assert any(e['kind'] == 'followup_proposal' and e['state'] == 'committed' for e in events)


def test_review_waits_then_commits_target_with_actual_receipt(tmp_path):
    service, l2, calls, _ = fixture(tmp_path, mode='review')
    run(service.submit(payload()))
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert result['phase'] == 'prepared', result
    row = service.authority.get_candidate(result['candidate_ids'][0])
    assert row['status'] == 'pending' and not row['proposal']['owner_explicit']
    assert service.authority.get_memory('m1')['active_revision'] == 2
    adopted = run(service.adopt(row, owner=True, request_id='owner-accept'))
    assert adopted['receipt']['revision'] == 3
    event = next(r for r in MemoryL3Records(service.authority, l2.revisions.commit).timeline('m1')['items']
        if r['kind'] == 'followup_proposal')
    assert event['details']['receipt']['operation_id']
    assert event['source_refs'] == ['canonical_event:u1@user-version-1']
    assert len(calls) == 1


def test_withdrawn_new_source_fails_without_model_then_retry_uses_same_task(tmp_path):
    service, _, calls, adapter = fixture(tmp_path)
    task = run(service.submit(payload()))['tasks'][0]
    adapter.available = False
    failed = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert failed['phase'] == 'failed' and calls == []
    adapter.available = True
    retry = {'expected_revision': failed['revision'], 'request_id': 'retry-1'}
    service.tasks.retry('m1', task['task_id'], retry)
    assert service.tasks.retry('m1', task['task_id'], retry)['replayed']
    assert run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]['phase'] == 'prepared'
    assert len(calls) == 1 and len(service.tasks.list()) == 1


def test_interpretation_cannot_be_written_as_plain_fact(tmp_path):
    service, _, calls, _ = fixture(tmp_path)
    run(service.submit(payload(kind='understanding', value='specific_future_use')))
    failed = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert failed['phase'] == 'failed' and failed['issue']['code'] == 'followup_cognition_required'
    assert service.authority.get_memory('m1')['active_revision'] == 2
    assert len(calls) == 1


def test_off_mode_does_not_generate_or_adopt(tmp_path):
    service, _, calls, _ = fixture(tmp_path, mode='off')
    run(service.submit(payload()))
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert result['phase'] == 'disabled' and calls == []
    assert service.authority.get_memory('m1')['active_revision'] == 2


def test_review_text_edit_reprepares_preserves_draft_and_never_uses_old_plan(tmp_path):
    service, _, calls, _ = fixture(tmp_path, mode='review')
    run(service.submit(payload()))
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    cid = result['candidate_ids'][0]
    row = service.authority.get_candidate(cid)
    edited = run(service.adopt(row, owner=True, request_id='draft-1', edit={
        'content': '补充：那次是你先说想抱我的。', 'title': '当时的发起人', 'kind': 'reflection',
        'tags': [], 'domain': [], 'event_date': '', 'confidence': .85}))
    assert edited['status'] == 'preparing' and edited['draft_preserved']
    assert run(service.adopt(row, owner=True, request_id='while-preparing'))['status'] == 'preparing'
    assert service.authority.get_memory('m1')['active_revision'] == 2
    refreshed = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert refreshed['phase'] == 'prepared', refreshed
    current = service.authority.get_candidate(cid)
    assert current['status'] == 'pending' and current['proposal']['proposed_body'].startswith('补充：')
    assert current['proposal']['confidence'] == .85 and current['proposal']['memory_type'] == 'reflection'
    assert current['proposal']['metadata']['legacy_candidate']['title'] == '当时的发起人'
    assert len(calls) == 2
    assert run(service.adopt(current, owner=True, request_id='apply-new'))['receipt']['revision'] == 3


def test_unknown_after_actual_commit_recovers_original_operation_without_second_model(tmp_path):
    service, _, calls, _ = fixture(tmp_path)
    run(service.submit(payload()))
    original = service.authority.decide_candidate
    lose_once = [True]
    def interrupted(*args, **kwargs):
        if kwargs.get('action') == 'commit' and lose_once[0]:
            lose_once[0] = False
            raise RuntimeError('synthetic post-commit interruption')
        return original(*args, **kwargs)
    service.authority.decide_candidate = interrupted
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert result['phase'] == 'failed' and service.authority.get_memory('m1')['active_revision'] == 3
    service.tasks.retry('m1', result['task_id'], {'expected_revision': result['revision'], 'request_id': 'recover'})
    recovered = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert recovered['phase'] == 'prepared', recovered
    assert service.authority.get_memory('m1')['active_revision'] == 3 and len(calls) == 1


def test_transient_delivery_lookup_does_not_discard_durable_proposal(tmp_path):
    service, _, calls, _ = fixture(tmp_path)
    async def transient(context, turn):
        return {'state': 'available', 'eligible_fragment_ids': [], 'reasons': {
            context['fragments'][0]['fragment_id']: 'memory_fragment_source_check_failed'}}
    service.resolver = transient
    with pytest.raises(L1OwnerError) as failure:
        run(service.submit(payload()))
    assert failure.value.status == 503 and service.tasks.list() == [] and calls == []


def test_permanent_source_change_has_visible_receipt_not_an_adoption(tmp_path):
    service, l2, calls, _ = fixture(tmp_path)
    async def withdrawn(context, turn):
        return {'state': 'available', 'eligible_fragment_ids': [], 'reasons': {
            context['fragments'][0]['fragment_id']: 'memory_fragment_withdrawn'}}
    service.resolver = withdrawn
    result = run(service.submit(payload()))
    assert result['tasks'][0]['phase'] == 'source_changed'
    assert run(run_followup_tasks(service.engine, None, max_jobs=1))['results'] == [] and calls == []
    assert MemoryL3Records(service.authority, l2.revisions.commit).timeline('m1')['items'][0]['state'] == 'source_changed'


@pytest.mark.parametrize('kind,role,status', [('understanding', 'insight', 'supported'), ('question', 'question', 'tentative')])
def test_grounded_understanding_and_question_keep_cognition_through_real_commit(tmp_path, kind, role, status):
    from memory_l1_schema import unknown
    service, l2, calls, _ = fixture(tmp_path)
    original = service.engine._daily_chat_memory_create_completion
    async def cognition_model(client, **kwargs):
        result = await original(client, **kwargs)
        plan = json.loads(result.text)
        first = plan['patch'][0]
        plan['patch'].append({'op': 'replace', 'path': first['path'].rsplit('/', 1)[0] + '/cognition',
            'old': None, 'value': {'role': unknown('known', role, first['evidence_refs']),
                'epistemic_status': unknown('known', status, first['evidence_refs']),
                'about_refs': [], 'open_questions': ['那次发起人的理解是否还需别的细节？'] if kind == 'question' else []},
            'evidence_refs': first['evidence_refs']})
        return NS(text=json.dumps(plan, ensure_ascii=False), finish='stop')
    service.engine._daily_chat_memory_create_completion = cognition_model
    run(service.submit(payload(kind=kind, value='specific_future_use')))
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert result['phase'] == 'prepared', result
    document = run(l2.revisions.detail('m1'))['l1']['document']
    assert document['statements'][0]['cognition']['role']['value'] == role
    assert document['statements'][0]['cognition']['epistemic_status']['value'] == status
    assert len(calls) == 1 and service.authority.get_memory('m1')['active_revision'] == 3


def test_formal_l1_candidate_is_found_by_source_without_treating_overlap_as_duplicate(tmp_path):
    from memory_authority import MemoryProposal
    from memory_semantics import binding_source_keys
    service, _, calls, _ = fixture(tmp_path, mode='review')
    task = run(service.submit(payload()))['tasks'][0]
    row = service.tasks.get(task['task_id'])
    _, evidence, _ = run(service.sources(row))
    prior = MemoryProposal(proposal_id='existing-formal', source_type='daily_chat_memory', proposed_body='同一轮另一个事实。',
        source_refs=('canonical_event:u1@user-version-1',), metadata={
            'memory_l1': {'evidence_bindings': [e['binding'] for e in evidence]}})
    service.authority.put_candidate(prior, service.engine.memory_ingestion_policy.evaluate(prior))
    found = service.authority.find_candidates_by_sources(binding_source_keys([e['binding'] for e in evidence]))
    assert [item['candidate_id'] for item in found] == ['existing-formal']
    result = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert result['phase'] == 'prepared', result
    assert result['related_candidates'] == ['existing-formal']
    assert result['candidate_ids'] != ['existing-formal']
    assert service.authority.get_candidate('existing-formal')['status'] == 'pending'
    assert len(calls) == 1


def review_draft(service):
    run(service.submit(payload()))
    first = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    row = service.authority.get_candidate(first['candidate_ids'][0])
    edit = {'content': '第一稿：那次是你先说想抱我的。', 'title': '主人保存的标题', 'kind': 'reflection', 'confidence': .85}
    saved = service.edit_draft(row, edit, request_id='owner-first-draft', expected_revision=row['revision'])
    return row, edit, saved


def test_owner_second_edit_during_preparation_fences_old_publication_and_latest_draft_remains_recoverable(tmp_path):
    from memory_authority import RevisionConflict
    service, _, calls, _ = fixture(tmp_path, mode='review')
    original_row, first_edit, saved = review_draft(service)
    candidate = saved['candidate']
    assert candidate['revision'] == original_row['revision'] + 1
    assert candidate['proposal']['proposed_body'] == first_edit['content']
    assert candidate['proposal']['metadata']['l3_followup']['compiled'] is None
    replayed = service.edit_draft(candidate, first_edit, request_id='owner-first-draft', expected_revision=original_row['revision'])
    assert replayed['candidate']['revision'] == candidate['revision']
    with pytest.raises(RevisionConflict):
        service.edit_draft(candidate, {'content': '多设备旧稿'}, request_id='stale-device', expected_revision=original_row['revision'])
    original_model = service.engine._daily_chat_memory_create_completion
    second_edit = {'content': '最新稿：当时是你先提出想抱我的。'}
    async def owner_changes_while_model_runs(client, **kwargs):
        output = await original_model(client, **kwargs)
        current = service.authority.get_candidate(candidate['candidate_id'])
        service.edit_draft(current, second_edit, request_id='owner-second-draft', expected_revision=current['revision'])
        return output
    service.engine._daily_chat_memory_create_completion = owner_changes_while_model_runs
    interrupted = run(run_followup_tasks(service.engine, None, max_jobs=1))
    assert interrupted['results'] == []
    latest = service.authority.get_candidate(candidate['candidate_id'])
    assert latest['revision'] == candidate['revision'] + 1
    assert latest['proposal']['proposed_body'] == second_edit['content']
    assert latest['proposal']['metadata']['l3_followup']['compiled'] is None
    task = service.tasks.get(candidate['candidate_id'])
    assert task['phase'] == 'queued' and task['draft_content'] == second_edit['content']
    service.engine._daily_chat_memory_create_completion = original_model
    recovered = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    current = service.authority.get_candidate(candidate['candidate_id'])
    assert recovered['phase'] == 'prepared' and current['status'] == 'pending'
    assert current['proposal']['proposed_body'] == second_edit['content']
    assert current['proposal']['metadata']['legacy_candidate']['title'] == first_edit['title']
    assert current['proposal']['metadata']['l3_followup']['compiled']
    assert service.authority.get_memory('m1')['active_revision'] == 2 and len(calls) == 3


def test_candidate_revision_and_task_publication_rollback_together_then_reuse_successful_generation(tmp_path, monkeypatch):
    service, _, calls, _ = fixture(tmp_path, mode='review')
    _, _, saved = review_draft(service)
    candidate = saved['candidate']
    original = service.authority._revise_candidate
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('after candidate SQL, before shared task checkpoint')
    monkeypatch.setattr(service.authority, '_revise_candidate', interrupted)
    failed = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert failed['phase'] == 'failed'
    actual = service.authority.get_candidate(candidate['candidate_id'])
    assert actual['revision'] == candidate['revision']
    assert actual['proposal']['metadata']['l3_followup']['compiled'] is None
    task = service.tasks.get(candidate['candidate_id'])
    assert task['draft_content'] and task['edited_candidate_revision'] == candidate['revision']
    monkeypatch.setattr(service.authority, '_revise_candidate', original)
    service.tasks.retry('m1', task['task_id'], {'expected_revision': task['revision'], 'request_id': 'recover-transaction'})
    restored = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert restored['phase'] == 'prepared'
    assert service.authority.get_candidate(candidate['candidate_id'])['revision'] == candidate['revision'] + 1
    assert len(calls) == 2


def test_crash_after_publication_before_finish_resumes_without_revising_candidate_or_calling_model_again(tmp_path, monkeypatch):
    service, _, calls, _ = fixture(tmp_path, mode='review')
    _, _, saved = review_draft(service)
    original_finish = FollowupTasks.finish
    stop_once = [True]
    def interrupted(self, *args, **kwargs):
        if stop_once[0]:
            stop_once[0] = False
            raise RuntimeError('publication committed, finish interrupted')
        return original_finish(self, *args, **kwargs)
    monkeypatch.setattr(FollowupTasks, 'finish', interrupted)
    failed = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert failed['phase'] == 'failed'
    candidate = service.authority.get_candidate(saved['candidate']['candidate_id'])
    assert candidate['proposal']['metadata']['l3_followup']['compiled']
    task = service.tasks.get(candidate['candidate_id'])
    assert task['draft_content'] is None and task['publication']['candidate_revision'] == candidate['revision']
    service.tasks.retry('m1', task['task_id'], {'expected_revision': task['revision'], 'request_id': 'recover-published'})
    restored = run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]
    assert restored['phase'] == 'prepared'
    assert service.authority.get_candidate(candidate['candidate_id'])['revision'] == candidate['revision']
    assert len(calls) == 2


def test_followup_claim_uses_original_configured_chat_timeout_plus_margin(tmp_path, monkeypatch):
    service, _, _, _ = fixture(tmp_path, mode='review')
    service.engine.daily_chat_memory_timeout_seconds = 285.0
    original_claim = FollowupTasks.claim
    leases = []
    def capture(self, *args, **kwargs):
        leases.append(kwargs['lease_seconds'])
        return original_claim(self, *args, **kwargs)
    monkeypatch.setattr(FollowupTasks, 'claim', capture)
    run(service.submit(payload()))
    assert run(run_followup_tasks(service.engine, None, max_jobs=1))['results'][0]['phase'] == 'prepared'
    assert leases == [345.0]
