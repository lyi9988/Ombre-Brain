import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

from memory_authority import RevisionConflict
from memory_l0_sources import MemoryL0Sources
from memory_l1_service import L1OwnerError
from memory_l1_schema import statement_template
from memory_l2_service import MemoryL2Service
from test_memory_l1_revisions import setup_service, request_for


def run(task): return asyncio.run(task)


def setup(tmp_path):
    revisions, authority, manager = setup_service(tmp_path)
    request = request_for(revisions)
    stmt = statement_template()
    stmt['content'] = '那次我听见她说想要抱抱。'
    request['document']['statements'] = [stmt]
    run(revisions.save('m1', request))
    calls = []
    engine = NS(temperature=0.3)
    engine._memory_l1_prompt = lambda source_id: ('second-pass prompt', {'prompt_revision': 'p1'})
    engine._memory_task_route = lambda stage: {'stage':stage,'client':object(),'configured':True,
        'model':'synthetic-l2','connection':'reflection','max_tokens':16000}
    engine._memory_l1_sources = lambda: MemoryL0Sources(authority)
    engine._completion_content = lambda response: response.text
    engine._completion_finish_reason = lambda response: response.finish
    async def complete(client, **kwargs):
        calls.append(kwargs)
        material = json.loads(kwargs['messages'][1]['content'])
        sid = material['l1_document']['statements'][0]['statement_id']
        return NS(text=json.dumps({'full_injection_text':'那次她想要抱抱，我记得那次相处。',
            'coherent_sections':[{'section_id':'that-time','text':'那次她想要抱抱。','statement_refs':[sid]}],
            'optional_use_variants':[], 'review':{'required':False,'reasons':[]}}, ensure_ascii=False), finish='stop')
    engine._daily_chat_memory_create_completion = complete
    return MemoryL2Service(engine, revisions), authority, calls


def test_commit_enqueues_only_adopted_l1_and_second_pass_is_persisted(tmp_path):
    service, authority, calls = setup(tmp_path)
    assert len(service.store.pending(10)) == 1  # legacy seed did not migrate
    result = run(service.run_once(1))
    assert result['results'][0]['state'] == 'ready'
    assert len(calls) == 1 and calls[0]['task_stage'] == 'presentation_l2'
    material = json.loads(calls[0]['messages'][1]['content'])
    assert material['l1_document']['narrative']['body'] == '完整的合成回忆。'
    detail = service.detail('m1')['l2']
    assert detail['state'] == 'ready'
    assert detail['presentation']['presentation_revision'] == 1
    assert detail['presentation']['first_pass_revision'] == 2
    # A fresh service instance reads the same persisted product, not a cache.
    assert MemoryL2Service(service.engine, service.revisions).detail('m1')['l2']['presentation'] == detail['presentation']
    assert authority.get_memory('m1')['active_revision'] == 2


def test_l1_change_makes_l2_stale_and_auto_queue_does_not_double_generate(tmp_path):
    service, _, calls = setup(tmp_path)
    run(service.run_once(1))
    payload = request_for(service.revisions, 'next-l1')
    payload['document']['narrative']['body'] += ' 后来有了新的理解。'
    run(service.revisions.save('m1', payload))
    assert service.detail('m1')['l2']['state'] == 'stale'
    run(service.run_once(10))
    assert len(calls) == 2  # commit job and dependency scan collapse
    assert len(service.detail('m1')['l2']['revisions']) == 2


def owner_payload(service, request='owner', kind='wording'):
    detail = service.detail('m1')['l2']
    return {'expected_l1_revision': detail['current_dependencies']['first_pass_revision'],
        'expected_presentation_revision':detail['current_presentation_revision'],
        'request_id':request, 'change_kind':kind,
        'document':copy.deepcopy(detail['presentation']['document'])}


def test_owner_edit_is_versioned_idempotent_and_never_overwritten(tmp_path):
    service, _, calls = setup(tmp_path)
    run(service.run_once(1))
    payload = owner_payload(service)
    payload['document']['full_injection_text'] = '我亲自改好的完整文字。'
    receipt = run(service.save('m1', payload))
    assert receipt['presentation_revision'] == 2
    assert run(service.save('m1', payload))['replayed']
    service.prepare('m1', {'expected_l1_revision':2,'request_id':'retry'})
    run(service.run_once(1))
    assert len(calls) == 2
    assert service.detail('m1')['l2']['presentation']['document']['full_injection_text'] == payload['document']['full_injection_text']
    assert service.detail('m1', 3)['l2']['presentation']['publication'] == 'candidate'
    with pytest.raises(RevisionConflict):
        run(service.save('m1', {**payload, 'request_id':'stale-other'}))


def test_fact_edit_saved_as_l3_pending_sync_then_confirmed_after_l1_revision(tmp_path):
    service, _, _ = setup(tmp_path)
    run(service.run_once(1))
    payload = owner_payload(service, kind='facts')
    payload['document']['full_injection_text'] = '主人补充的事实稿。'
    saved = run(service.save('m1', payload))
    assert saved['sync_state'] == 'pending_l1'
    assert saved['l2']['state'] == 'pending_sync'
    assert saved['l2']['sync_proposals'][0]['state'] == 'pending'
    edit = request_for(service.revisions, 'sync-l1')
    edit['document']['narrative']['body'] += ' 主人补充的事实。'
    run(service.revisions.save('m1', edit))
    confirmed = owner_payload(service, 'owner-confirmed')
    confirmed['confirm_sync_proposals'] = [saved['presentation_revision']]
    final = run(service.save('m1', confirmed))
    assert final['l2']['state'] == 'ready'
    assert final['l2']['sync_proposals'][0]['state'] == 'owner_confirmed_after_l1_revision'


def test_resolve_uses_l2_but_not_after_dependency_change(tmp_path):
    service, _, calls = setup(tmp_path)
    run(service.run_once(1))
    sid = run(service.revisions.detail('m1'))['l1']['document']['statements'][0]['statement_id']
    item = {'memory_id':'m1','revision':2,'statement_ids':[sid],
        'allowed_uses':['historical_evidence'],'statement_uses':{sid:'historical_evidence'},'narrative_allowed':True}
    result = run(service.resolve({'items':[item]}))['items'][0]
    assert result['text'] == '那次她想要抱抱，我记得那次相处。'
    assert len(calls) == 1  # resolve never calls generation
    service.engine._memory_l1_prompt = lambda sid: ('changed', {'prompt_revision':'p2'})
    result = run(service.resolve({'items':[item]}))['items'][0]
    assert result['state'] == 'stale' and not result['text']


def test_mixed_units_do_not_promote_historical_claim_using_another_claims_use(tmp_path):
    service, _, _ = setup(tmp_path)
    run(service.run_once(1))
    payload = owner_payload(service)
    sid = payload['document']['coherent_sections'][0]['statement_refs'][0]
    payload['document']['coherent_sections'] = []
    payload['document']['optional_use_variants'] = [{'variant_id':'current','text':'不能被错误选为当前的稿。',
        'statement_refs':[sid],'intended_use':'current_basis'}]
    run(service.save('m1', payload))
    result = run(service.resolve({'items':[{'memory_id':'m1','revision':2,'statement_ids':[sid],
        'allowed_uses':['historical_evidence','current_basis'], 'statement_uses':{sid:'historical_evidence'},
        'narrative_allowed':False}]}))['items'][0]
    assert not result['text'] and result['reason'] == 'no_compatible_l2_unit'


def test_revision_changes_during_provider_call_cannot_publish_stale_product(tmp_path):
    service, _, calls = setup(tmp_path)
    original = service.engine._daily_chat_memory_create_completion
    async def changing(*args, **kwargs):
        output = await original(*args, **kwargs)
        detail = await service.revisions.detail('m1')
        request = {'expected_revision':detail['current_revision'], 'request_id':'during-call',
                   'document':copy.deepcopy(detail['l1']['document'])}
        request['document']['narrative']['body'] += ' 并发修改。'
        await service.revisions.save('m1', request)
        return output
    service.engine._daily_chat_memory_create_completion = changing
    result = run(service.run_once(1))
    assert result['results'][0]['state'] != 'ready'
    assert service.detail('m1')['l2']['presentation'] is None
    assert len(calls) == 1


def test_length_finish_and_foreign_statement_refs_are_not_saved(tmp_path):
    service, _, _ = setup(tmp_path)
    async def incomplete(*args, **kwargs): return NS(text='{}', finish='length')
    service.engine._daily_chat_memory_create_completion = incomplete
    result = run(service.run_once(1))
    assert result['results'][0]['reason'] == 'l2_output_incomplete'
    assert service.detail('m1')['l2']['presentation'] is None
    bad = {'full_injection_text':'完整稿','coherent_sections':[{'section_id':'x','text':'x','statement_refs':['foreign']}], 'optional_use_variants':[]}
    with pytest.raises(L1OwnerError, match='invalid_l2_unit'):
        run(service.save('m1', {'expected_l1_revision':2,'expected_presentation_revision':0,
            'request_id':'bad','document':bad,'change_kind':'wording'}))


def test_prompt_off_and_manual_retry_are_visible_without_blind_provider_retry(tmp_path):
    service, _, calls = setup(tmp_path)
    service.engine._memory_l1_prompt = lambda sid: ('', {'prompt_revision':'off'})
    assert run(service.run_once(1))['results'][0]['reason'] == 'l2_prompt_disabled'
    assert not calls
    service.engine._memory_l1_prompt = lambda sid: ('owner prompt', {'prompt_revision':'owner-r2'})
    service.prepare('m1', {'expected_l1_revision':2,'request_id':'explicit-retry'})
    run(service.run_once(1))
    assert calls[0]['messages'][0]['content'] == 'owner prompt'


def test_unknown_save_response_can_be_recovered_after_l1_and_prompt_change(tmp_path):
    service, _, _ = setup(tmp_path)
    run(service.run_once(1))
    payload = owner_payload(service)
    payload['edit_note'] = '保存这次修改原因'
    saved = run(service.save('m1', payload))
    request = request_for(service.revisions, 'later')
    request['document']['narrative']['body'] += ' 另一次更新。'
    run(service.revisions.save('m1', request))
    service.engine._memory_l1_prompt = lambda sid: ('later', {'prompt_revision':'later'})
    replay = run(service.save('m1', payload))
    assert replay['replayed'] and replay['presentation_revision'] == saved['presentation_revision']
    assert replay['l2']['presentation']['edit_note'] == payload['edit_note']


def test_source_withdrawal_before_or_during_second_pass_blocks_publication(tmp_path):
    from test_memory_diary_l1_p2b import setup_diary, capture
    from memory_l1_schema import document_template, generation_template, split_document
    service, _, calls = setup(tmp_path / 'model')
    env = setup_diary(tmp_path / 'source', '这是一段完整的合成来源。')
    source = capture(env)
    document = document_template('我记下这段合成经历。', '完整合成回忆', catalog=env['catalog'], catalog_revision='test')
    # Use the normal qualified diary capture for an actual version/anchor.
    captured = run(env['sources'].capture_diary('2026-10-14', binding_id='source:1'))
    document['evidence_bindings'] = [captured['binding']]
    statement = statement_template()
    statement['content'] = '合成经历。'
    statement['evidence_refs'] = ['source:1']
    document['statements'] = [statement]
    # Adopt via the same owner revision service; runtime binds new identities.
    run(env['commit'].commit_memory(memory_id='m1',bucket_id='b1',expected_revision=0,body='我记下这段合成经历。',
        metadata={'name':'合成'},source_refs=[],decision_source='test',idempotency_key='seed',actor='test'))
    # Source bindings are managed; an adopted model document supplies them at
    # commit, while the owner path may not fabricate them.
    # Keep immutable diary provenance by constructing a valid internal L1 doc.
    document['generation'] = generation_template(env['catalog'], 'test', 'first_pass_model')
    document['statements'][0]['statement_id'] = 'st-synthetic'
    document['statements'][0]['statement_revision_id'] = 'sr-synthetic'
    body, semantic = split_document(document)
    run(env['commit'].commit_memory(memory_id='m1',bucket_id='b1',expected_revision=1,body=body,
        metadata={'name':'合成','memory_l1':semantic},source_refs=[],decision_source='test',idempotency_key='l1',actor='test',l1_write_intent=True))
    service = MemoryL2Service(service.engine, env['revisions'])
    service.engine._memory_l1_sources = lambda: env['sources']
    env['source_state']['record']['revoked'] = True
    assert run(service.run_once(1))['results'][0]['reason'] == 'l2_source_not_available'
    assert not calls and service.detail('m1')['l2']['presentation'] is None
    env['source_state']['record']['revoked'] = False
    service.prepare('m1', {'expected_l1_revision':2,'request_id':'retry-readable'})
    original = service.engine._daily_chat_memory_create_completion
    async def revoke(*args, **kwargs):
        response = await original(*args, **kwargs)
        env['source_state']['record']['revoked'] = True
        return response
    service.engine._daily_chat_memory_create_completion = revoke
    assert run(service.run_once(1))['results'][0]['reason'] == 'l2_source_not_available'
    assert len(calls) == 1 and service.detail('m1')['l2']['presentation'] is None


def test_missing_model_review_is_saved_for_review_and_owner_can_confirm(tmp_path):
    service, _, _ = setup(tmp_path)
    original = service.engine._daily_chat_memory_create_completion
    async def missing_review(*args, **kwargs):
        result = await original(*args, **kwargs)
        document = json.loads(result.text)
        document.pop('review')
        result.text = json.dumps(document)
        return result
    service.engine._daily_chat_memory_create_completion = missing_review
    assert run(service.run_once(1))['results'][0]['state'] == 'pending_review'
    assert service.detail('m1')['l2']['state'] == 'pending_review'
    request = owner_payload(service)
    assert run(service.save('m1', request))['l2']['state'] == 'ready'


def test_ordinary_wording_does_not_confirm_pending_fact_proposal(tmp_path):
    service, _, _ = setup(tmp_path)
    run(service.run_once(1))
    facts = owner_payload(service, 'fact-change', 'facts')
    facts['document']['full_injection_text'] += ' 主人修正了事实。'
    first = run(service.save('m1', facts))
    request = request_for(service.revisions, 'unrelated-l1')
    request['document']['narrative']['title'] = '仅修改标题'
    run(service.revisions.save('m1', request))
    ordinary = owner_payload(service, 'ordinary-wording')
    ordinary['document']['full_injection_text'] += ' 润色文字。'
    result = run(service.save('m1', ordinary))
    assert result['l2']['sync_proposals'][0]['state'] == 'pending'
    assert result['sync_state'] == 'pending_l1'
    explicit = owner_payload(service, 'explicit-confirm')
    explicit['confirm_sync_proposals'] = [first['presentation_revision']]
    result = run(service.save('m1', explicit))
    assert result['l2']['state'] == 'ready'
    proposal = result['l2']['sync_proposals'][0]
    assert proposal['proposal']['synchronized_l1_revision'] == 3
    assert proposal['proposal']['confirmed_presentation_revision'] == result['presentation_revision']


def test_full_text_edit_rebuilds_units_from_exact_owner_paragraphs(tmp_path):
    service, _, _ = setup(tmp_path)
    run(service.run_once(1))
    request = owner_payload(service)
    original = copy.deepcopy(request['document'])
    request['document']['full_injection_text'] = '主人改好的第一段。\n这一行仍属于同一段。\r\n\r\n主人改好的第二段，保留条件和例外。'
    saved = run(service.save('m1', request))
    assert saved['l2']['presentation']['units_state'] == 'current'
    rebuilt = saved['l2']['presentation']['document']
    assert rebuilt['full_injection_text'] == request['document']['full_injection_text']
    assert [unit['text'] for unit in rebuilt['coherent_sections']] == [
        '主人改好的第一段。\n这一行仍属于同一段。', '主人改好的第二段，保留条件和例外。']
    assert not rebuilt['optional_use_variants']
    assert service.detail('m1', 1)['l2']['presentation']['document'] == original
    assert request['document']['coherent_sections'] == original['coherent_sections']  # input draft remains intact
    assert run(service.save('m1', request))['replayed']
    sid = request['document']['coherent_sections'][0]['statement_refs'][0]
    query = {'memory_id':'m1','revision':2,'statement_ids':[sid], 'allowed_uses':['historical_evidence'],
             'statement_uses':{sid:'historical_evidence'}, 'narrative_allowed':False}
    result = run(service.resolve({'items':[query]}))['items'][0]
    assert result['text'] == '\n\n'.join(unit['text'] for unit in rebuilt['coherent_sections'])
    assert [unit['text'] for unit in result['units']] == [unit['text'] for unit in rebuilt['coherent_sections']]
    result = run(service.resolve({'items':[{**query, 'narrative_allowed':True}]}))['items'][0]
    assert result['text'] == request['document']['full_injection_text']


def test_owner_paragraph_units_keep_large_single_paragraph_and_all_l1_refs():
    from memory_l2_store import owner_paragraph_document
    text = '只有她愿意才继续；如果她拒绝，我就停下。' * 600
    document = {'full_injection_text': text, 'coherent_sections': [], 'optional_use_variants': []}
    rebuilt = owner_paragraph_document(document, ['current', 'historical', 'current', 'shared'])
    assert rebuilt['full_injection_text'] == text
    assert rebuilt['coherent_sections'] == [{
        'section_id': rebuilt['coherent_sections'][0]['section_id'], 'text': text,
        'statement_refs': ['current', 'historical', 'shared']}]
    assert owner_paragraph_document(document, [])['coherent_sections'] == []


def test_resaving_legacy_stale_owner_units_creates_new_revision_without_rewriting_history(tmp_path):
    service, authority, calls = setup(tmp_path)
    run(service.run_once(1))
    edit = owner_payload(service)
    edit['document']['full_injection_text'] = '主人保留的第一段。\n\n主人保留的第二段。'
    saved = run(service.save('m1', edit))
    # Reproduce a persisted revision from the old owner-edit implementation.
    legacy = copy.deepcopy(edit['document'])
    conn = authority._connect()
    try:
        conn.execute("UPDATE memory_l2_revisions SET document_json=?,units_state='stale_after_owner_text_edit' WHERE memory_id='m1' AND presentation_revision=?",
            (json.dumps(legacy, ensure_ascii=False), saved['presentation_revision']))
        conn.commit()
    finally:
        conn.close()
    resave = owner_payload(service, 'owner-reindex')
    recovered = run(service.save('m1', resave))
    assert recovered['presentation_revision'] == saved['presentation_revision'] + 1
    assert recovered['l2']['presentation']['units_state'] == 'current'
    assert [unit['text'] for unit in recovered['l2']['presentation']['document']['coherent_sections']] == [
        '主人保留的第一段。', '主人保留的第二段。']
    assert service.detail('m1', saved['presentation_revision'])['l2']['presentation']['document'] == legacy
    assert len(calls) == 1


def test_owner_paragraphs_are_not_qualified_by_only_one_of_multiple_l1_claims(tmp_path):
    service, _, _ = setup(tmp_path)
    payload = request_for(service.revisions, 'two-statements')
    second = statement_template()
    second['content'] = '另一条仍需独立核对的事实。'
    payload['document']['statements'].append(second)
    run(service.revisions.save('m1', payload))
    run(service.run_once(10))
    edit = owner_payload(service)
    edit['document']['full_injection_text'] = '主人第一段。\n\n主人第二段。'
    saved = run(service.save('m1', edit))
    refs = saved['l2']['presentation']['document']['coherent_sections'][0]['statement_refs']
    assert len(refs) == 2
    result = run(service.resolve({'items':[{'memory_id':'m1','revision':3,
        'statement_ids':refs[:1], 'allowed_uses':['historical_evidence'],
        'statement_uses':{refs[0]:'historical_evidence'}, 'narrative_allowed':False}]}))['items'][0]
    assert not result['text'] and not result['units']


def test_pending_requests_for_same_dependency_share_one_model_run(tmp_path):
    service, _, calls = setup(tmp_path)
    service.engine.config = {'reflection':{'memory_l2_concurrency':4}}
    service.prepare('m1', {'expected_l1_revision':2,'request_id':'click-1'})
    service.prepare('m1', {'expected_l1_revision':2,'request_id':'click-2'})
    run(service.run_once(10))
    run(service.run_once(10))
    assert len(calls) == 1
    assert service.detail('m1')['l2']['current_presentation_revision'] == 1


def test_owner_schema_is_passed_to_model_and_changes_invalidate_l2(tmp_path):
    from memory_l2_schema import default_schema
    service, _, calls = setup(tmp_path)
    run(service.run_once(1))
    assert calls[0]['messages'][1]['content']
    assert json.loads(calls[0]['messages'][1]['content'])['output_schema'] == default_schema()
    schema = default_schema()
    schema['properties']['full_injection_text']['minLength'] = 1000
    service.engine.config = {'reflection':{'memory_l2_output_schema':schema}}
    assert service.detail('m1')['l2']['state'] == 'stale'
    result = run(service.run_once(1))
    assert result['results'][0]['reason'] == 'l2_output_schema_mismatch'
    assert service.detail('m1')['l2']['presentation']['presentation_revision'] == 1


def test_runtime_affinity_settings_do_not_regenerate_persistent_l2(tmp_path):
    service, _, calls = setup(tmp_path)
    run(service.run_once(1))
    before = service.generation_context()[2]
    service.engine.config = {'memory_system': {'recall': {
        'signal_bindings': [{'id': 'care', 'producer': 'ombre.persona', 'key': 'protective_drive',
                             'max_age_seconds': 60}],
        'affinity_mappings': [{'id': 'care_life', 'selector': {'domain': 'life'},
                               'signal_id': 'care', 'target': 'drive_affinity', 'weight': .5}],
        'affinity_aggregation': 'max'}}}
    assert service.generation_context()[2] == before
    assert service.detail('m1')['l2']['state'] == 'ready'
    run(service.run_once(1))
    assert len(calls) == 1
