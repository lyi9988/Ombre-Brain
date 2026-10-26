"""Real temporary Authority/L1/L2 and owner corrections; only model is fake."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from memory_l1_schema import statement_template
from memory_l2_service import MemoryL2Service
from memory_l3_sync import MemoryL3SyncService, pending_sync_scope, apply_local_patch
from memory_l0_sources import MemoryL0Sources
from memory_l1_source_readers import configured_source_readers
from memory_owner_l2_source import OwnerL2EditSourceReader
from memory_l1_service import L1OwnerError
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_l1_revisions import setup_service, request_for


def run(coro):
    return asyncio.run(coro)


def setup(tmp_path, *, decision='revise_l1', before='3号', during=None, two_statements=False):
    revisions, authority, buckets = setup_service(tmp_path / 'memory')
    request = request_for(revisions)
    request['document']['narrative'] = {'title': '行程', 'body': '我记得行程是3号。'}
    statement = statement_template()
    statement['content'] = '行程是3号。'
    request['document']['statements'] = [statement]
    if two_statements:
        second = statement_template()
        second['statement_id'] = 'new:second-statement'
        second['content'] = '地点在北京。'
        request['document']['statements'].append(second)
    run(revisions.save('m1', request))
    engine = ReflectionEngine(reflection_config(tmp_path / 'engine', mode='review'))
    engine.memory_authority_store = authority
    engine._memory_l1_revisions = lambda manager=None: revisions
    engine._memory_l1_sources = lambda: MemoryL0Sources(authority, configured_source_readers(engine.config, authority))
    engine._memory_task_route = lambda stage: {'stage': stage, 'client': object(), 'configured': True,
        'model': 'fake-only', 'connection': 'reflection', 'thinking_mode': 'enabled', 'reasoning_effort': 'low',
        'max_tokens': 32000, 'unavailable_reason': ''}
    calls = []
    async def completion(client, **kwargs):
        stage = kwargs['task_stage']
        material = json.loads(kwargs['messages'][1]['content'])
        calls.append((stage, material, kwargs))
        if stage == 'presentation_l2':
            sid = material['l1_document']['statements'][0]['statement_id']
            output = {'full_injection_text': '我记得行程是' + before + '。',
                'coherent_sections': [{'section_id': 'trip', 'text': '行程是' + before + '。', 'statement_refs': [sid]}],
                'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
        else:
            if during:
                during(authority, material)
            document = material['material']['l1_document']
            sid = document['statements'][0]['statement_id']
            ref = material['source_evidence'][-1]['binding']['binding_id']
            output = {'decision': decision, 'reason': '仅处理主人这次对日期的明确纠正。', 'affected_statement_ids': [sid], 'patch': []}
            if decision == 'revise_l1':
                output['patch'] = [
                    {'op': 'replace_text', 'path': '/narrative/body', 'old': '3号', 'value': '5号', 'evidence_refs': [ref]},
                    {'op': 'replace', 'path': '/statements/' + sid + '/content', 'old': document['statements'][0]['content'],
                        'value': '行程是5号。', 'evidence_refs': [ref]},
                    {'op': 'replace', 'path': '/statements/' + sid + '/evidence_refs', 'old': document['statements'][0]['evidence_refs'],
                        'value': [ref], 'evidence_refs': [ref]}]
        return NS(choices=[NS(message=NS(content=json.dumps(output, ensure_ascii=False)), finish_reason='stop')])
    engine._daily_chat_memory_create_completion = completion
    l2 = MemoryL2Service(engine, revisions)
    initial = run(l2.run_once(1))
    assert initial['results'][0]['state'] == 'ready', initial
    product = l2.detail('m1')['l2']['presentation']
    owner_document = deepcopy(product['document'])
    owner_document['full_injection_text'] = '主人原稿：行程是' + ('3号' if decision == 'l1_already_correct' else '5号') + '。'
    saved = run(l2.save('m1', {'expected_l1_revision': 2, 'expected_presentation_revision': product['presentation_revision'],
        'request_id': 'owner-facts', 'document': owner_document, 'change_kind': 'facts', 'edit_note': '纠正日期。'}))
    assert saved['sync_state'] == 'pending_l1'
    service = MemoryL3SyncService(l2)
    return NS(authority=authority, engine=engine, buckets=buckets, revisions=revisions, l2=l2, sync=service,
        owner_revision=saved['presentation_revision'], owner_text=owner_document['full_injection_text'], calls=calls)


def test_owner_l2_facts_actually_updates_local_l1_and_rebinds_original_owner_text(tmp_path):
    f = setup(tmp_path)
    before = run(f.revisions.detail('m1'))['l1']['document']
    with f.authority._connect() as db:
        assert pending_sync_scope(db, 'm1')['unknown_scope'] is True
    result = run(f.sync.run_once(1))
    proposal = result['results'][0]
    assert proposal['state'] == 'synchronized', proposal
    receipt = proposal['receipt']
    assert receipt['l1_operation_id'] and receipt['l1_revision'] == 3
    after = run(f.revisions.detail('m1'))['l1']['document']
    assert after['narrative']['body'] == '我记得行程是5号。'
    assert after['statements'][0]['content'] == '行程是5号。'
    assert after['statements'][0]['time'] == before['statements'][0]['time']
    assert after['statements'][0]['conditions'] == before['statements'][0]['conditions']
    assert after['entities'] == before['entities'] and after['relations'] == before['relations']
    assert after['evidence_bindings'][-1]['source_namespace'] == 'snapshot.ombre.owner_l2_edit'
    assert f.l2.detail('m1')['l2']['state'] == 'ready'
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == f.owner_text
    assert f.l2.detail('m1')['l2']['owner_protected']
    assert len([call for call in f.calls if call[0] == 'memory_l3_sync']) == 1
    assert run(f.sync.run_once(1))['results'] == []
    assert f.authority.get_memory('m1')['active_revision'] == 3


def test_l1_already_correct_repairs_only_l2_without_a_fabricated_l1_revision(tmp_path):
    f = setup(tmp_path, decision='l1_already_correct', before='5号')
    result = run(f.sync.run_once(1))['results'][0]
    assert result['state'] == 'synchronized', result
    assert result['receipt']['l1_operation_id'] is None
    assert f.authority.get_memory('m1')['active_revision'] == 2
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == f.owner_text


def test_ambiguous_plan_preserves_owner_draft_and_does_not_claim_adoption(tmp_path):
    f = setup(tmp_path, decision='ambiguous')
    result = run(f.sync.run_once(1))['results'][0]
    assert result['state'] == 'ambiguous' and result['receipt'] is None
    assert f.authority.get_memory('m1')['active_revision'] == 2
    assert f.l2.detail('m1')['l2']['state'] == 'pending_sync'
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == f.owner_text


def test_checkbox_cannot_mark_a_fact_sync_complete(tmp_path):
    f = setup(tmp_path)
    product = f.l2.detail('m1')['l2']['presentation']
    with pytest.raises(L1OwnerError, match='l3_sync_action_required'):
        run(f.l2.save('m1', {'expected_l1_revision': 2, 'expected_presentation_revision': f.owner_revision,
            'request_id': 'checkbox-only', 'document': product['document'], 'change_kind': 'wording',
            'confirm_sync_proposals': [f.owner_revision]}))
    assert f.l2.detail('m1')['l2']['state'] == 'pending_sync'


def test_owner_source_contains_only_actual_changes_not_unchanged_model_prose(tmp_path):
    f = setup(tmp_path)
    source = run(OwnerL2EditSourceReader(f.authority).capture('m1', f.owner_revision))
    text = json.loads(source['text'])
    assert source['eligible_source'] and text['changed_passages']
    assert all('我记得行程是' not in row['after'] for row in text['changed_passages'])
    assert text['owner_edit_note'] == '纠正日期。'
    assert f.owner_text not in source['text']


def test_patch_cannot_free_rewrite_full_narrative_or_managed_bindings(tmp_path):
    f = setup(tmp_path)
    document = run(f.revisions.detail('m1'))['l1']['document']
    for path in ('/narrative/body', '/evidence_bindings'):
        plan = {'patch': [{'op': 'replace', 'path': path, 'old': document['narrative']['body'],
            'value': '整桶重写', 'evidence_refs': ['owner-correction']}]}
        with pytest.raises(L1OwnerError):
            apply_local_patch(document, plan, evidence_ids=['owner-correction'])


def test_source_withdrawal_after_generation_preserves_prepared_result_without_commit(tmp_path):
    def withdraw(authority, material):
        binding = material['source_evidence'][-1]['binding']
        authority.set_meta('owner_l2_source_state:' + binding['source_id'], {'state': 'withdrawn'})
    f = setup(tmp_path, during=withdraw)
    result = run(f.sync.run_once(1))['results'][0]
    assert result['state'] in {'failed', 'conflict'} and result['receipt'] is None
    assert result['issue']['code'] == 'owner_l2_correction_source_unavailable'
    process = f.sync.tasks.get('m1', f.owner_revision)['proposal']['processing']
    assert process['plan'] and process['raw_output'] and process['model_io_id']
    assert f.authority.get_memory('m1')['active_revision'] == 2
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == f.owner_text
    assert run(f.sync.run_once(1))['results'] == []
    assert len([c for c in f.calls if c[0] == 'memory_l3_sync']) == 1


def test_owner_change_during_model_conflicts_and_preserves_both_drafts(tmp_path):
    f = setup(tmp_path)
    original = f.engine._daily_chat_memory_create_completion
    async def changed(client, **kwargs):
        output = await original(client, **kwargs)
        if kwargs['task_stage'] == 'memory_l3_sync':
            product = f.l2.detail('m1')['l2']['presentation']
            document = deepcopy(product['document'])
            document['full_injection_text'] += '这是新保存的主人原稿。'
            await f.l2.save('m1', {'expected_l1_revision': 2, 'expected_presentation_revision': product['presentation_revision'],
                'request_id': 'during-model', 'document': document, 'change_kind': 'wording'})
        return output
    f.engine._daily_chat_memory_create_completion = changed
    result = run(f.sync.run_once(1))['results'][0]
    assert result['state'] == 'conflict' and result['plan']
    assert f.authority.get_memory('m1')['active_revision'] == 2
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'].endswith('这是新保存的主人原稿。')
    assert result['allowed_actions'] == ['prepare']


@pytest.mark.parametrize('after_commit', [False, True])
def test_retry_preserves_successful_model_and_recovers_original_commit(tmp_path, monkeypatch, after_commit):
    import memory_l3_sync
    f = setup(tmp_path)
    original = memory_l3_sync.commit_plan
    calls = []
    async def interrupted(*args, **kwargs):
        calls.append(kwargs['request_id'])
        if after_commit:
            await original(*args, **kwargs)
        raise RuntimeError('simulated_transport_interruption')
    monkeypatch.setattr(memory_l3_sync, 'commit_plan', interrupted)
    first = run(f.sync.run_once(1))['results'][0]
    assert first['state'] == ('l1_applied_l2_pending' if after_commit else 'failed'), first
    assert first['plan']
    if after_commit:
        assert first['receipt']['l1_operation_id']
        assert f.authority.get_memory('m1')['active_revision'] == 3
    monkeypatch.setattr(memory_l3_sync, 'commit_plan', original)
    payload = {'expected_revision': first['revision'], 'request_id': 'retry-original'}
    second = run(f.sync.action('m1', f.owner_revision, 'retry', payload))
    assert second['proposal']['state'] == 'synchronized', second
    assert f.authority.get_memory('m1')['active_revision'] == 3
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == f.owner_text
    replay = run(f.sync.action('m1', f.owner_revision, 'retry', payload))
    assert replay['replayed'] and replay['proposal']['receipt'] == second['proposal']['receipt']
    assert len([c for c in f.calls if c[0] == 'memory_l3_sync']) == 1
    assert len(calls) == 1


def test_notes_do_not_reenter_l0_or_invalidate_l2_material(tmp_path):
    from memory_l3_records import MemoryL3Records
    f = setup(tmp_path)
    assert run(f.sync.run_once(1))['results'][0]['state'] == 'synchronized'
    before = f.l2.detail('m1')['l2']['current_dependencies']
    notes = MemoryL3Records(f.authority, f.revisions.commit)
    saved = run(notes.save_note('m1', {'expected_revision': 3, 'request_id': 'owner-note', 'content': '只是补充感受，不是日期事实。'}))
    assert f.l2.detail('m1')['l2']['current_dependencies'] == before
    material = run(f.l2._material('m1', 3))
    assert material['rings'] == []
    assert saved['ring_id'] in str(notes.timeline('m1'))
    run(notes.retract_note('m1', saved['ring_id'], {'expected_revision': 3, 'request_id': 'retract-note',
        'expected_content_sha256': saved['note']['details']['content_sha256']}))
    assert f.l2.detail('m1')['l2']['current_dependencies'] == before
    assert f.l2.detail('m1')['l2']['state'] == 'ready'


def test_actual_related_correction_reaches_original_l2_call_and_withdrawal_cannot_render_old_product(tmp_path):
    from test_stage03_relation_dependencies import setup_relations, confirm
    authority, revisions, _ = setup_relations(tmp_path / 'relations')
    confirm(revisions)
    f = setup(tmp_path / 'engine-fixture')
    f.engine.memory_authority_store = authority
    f.engine._memory_l1_sources = lambda: MemoryL0Sources(authority, configured_source_readers(f.engine.config, authority))
    calls = []
    async def complete(client, **kwargs):
        material = json.loads(kwargs['messages'][1]['content'])
        calls.append(material)
        sid = material['l1_document']['statements'][0]['statement_id']
        output = {'full_injection_text': '核对相关修订后的实际第二遍。',
            'coherent_sections': [{'section_id': 'checked', 'text': '核对相关修订后的实际第二遍。', 'statement_refs': [sid]}],
            'optional_use_variants': [], 'review': {'required': False, 'reasons': []}}
        return NS(choices=[NS(message=NS(content=json.dumps(output, ensure_ascii=False)), finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion = complete
    l2 = MemoryL2Service(f.engine, revisions)
    generated = run(l2.run_once(3))
    assert any(row.get('state') == 'ready' for row in generated['results']), generated
    a_material = next(row for row in calls if row['l1_document']['statements'][0]['statement_id'] == 'statement-A')
    related = a_material['related_memory_updates']
    assert len(related['updates']) == 1
    assert related['updates'][0]['memory_id'] == 'B'
    assert related['updates'][0]['statements'][0]['content'] == 'B 的具体事实。'
    assert 'narrative' not in related['updates'][0]
    assert len([row for row in calls if row['l1_document']['statements'][0]['statement_id'] == 'statement-A']) == 1
    item = {'memory_id': 'A', 'revision': 1, 'statement_ids': ['statement-A'],
        'allowed_uses': ['current_basis'], 'statement_uses': {'statement-A': 'current_basis'}, 'narrative_allowed': True}
    assert run(l2.resolve({'items': [item]}))['items'][0]['text']
    authority.change_memory_state(memory_id='B', state='archived', recall_policy='disabled', expected_revision=2,
        idempotency_key='withdraw-B-after-l2', actor='owner')
    invalid = run(l2.resolve({'items': [item]}))['items'][0]
    assert invalid['text'] == '' and invalid['units'] == [] and invalid['state'] == 'stale'


def test_owner_sync_http_auth_fast_queue_cas_and_original_request_recovery(tmp_path):
    import ast
    from pathlib import Path
    from starlette.responses import JSONResponse
    f = setup(tmp_path)
    path = Path(__file__).resolve().parents[1] / 'server.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'api_memory_l3_sync')
    node.decorator_list = []
    namespace = {'_json_lib': json, '_memory_l2_service': lambda: f.l2,
        'reflection_engine': f.engine, 'memory_authority_store': f.authority, 'memory_commit_service': f.revisions.commit,
        '_require_owner_or_internal_index_maintenance': lambda req: None if req.owner else JSONResponse({'error': 'unauthorized'}, status_code=401),
        'logger': NS(warning=lambda *args: None)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    def request(owner=True, payload=None):
        async def body():
            assert owner, 'credentials must be checked before reading payload'
            return json.dumps(payload).encode()
        return NS(method='POST', owner=owner, query_params={}, body=body,
            path_params={'memory_id': 'm1', 'presentation_revision': str(f.owner_revision), 'action': 'prepare'})
    handler = namespace['api_memory_l3_sync']
    denied = run(handler(request(False)))
    assert denied.status_code == 401
    payload = {'expected_revision': 1, 'request_id': 'owner-http-prepare'}
    first = run(handler(request(payload=payload)))
    result = json.loads(first.body)
    assert first.status_code == 200 and result['proposal']['state'] == 'pending'
    assert not [c for c in f.calls if c[0] == 'memory_l3_sync']
    replayed = json.loads(run(handler(request(payload=payload))).body)
    assert replayed['replayed'] is True
    conflict = run(handler(request(payload={'expected_revision': 1, 'request_id': 'stale-other-request'})))
    assert conflict.status_code == 409
    assert json.loads(conflict.body)['proposal']['revision'] == result['proposal']['revision']


def test_l1_commit_then_new_owner_wording_is_preserved_and_explicit_retry_only_rebinds(tmp_path, monkeypatch):
    import memory_l3_sync
    f = setup(tmp_path)
    original = memory_l3_sync.commit_plan
    changed = []
    async def edit_after_commit(*args, **kwargs):
        receipt = await original(*args, **kwargs)
        product = f.l2.detail('m1')['l2']['presentation']
        document = deepcopy(product['document'])
        document['full_injection_text'] += '这是提交后新保存的措辞。'
        await f.l2.save('m1', {'expected_l1_revision': 3, 'expected_presentation_revision': product['presentation_revision'],
            'request_id': 'wording-after-l1', 'document': document, 'change_kind': 'wording'})
        changed.append(document['full_injection_text'])
        return receipt
    monkeypatch.setattr(memory_l3_sync, 'commit_plan', edit_after_commit)
    first = run(f.sync.run_once(1))['results'][0]
    assert first['state'] == 'l1_applied_l2_pending' and first['issue']['code'] == 'owner_l2_changed'
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == changed[0]
    actual_operation = first['receipt']['l1_operation_id']
    second = run(f.sync.action('m1', f.owner_revision, 'retry', {'expected_revision': first['revision'], 'request_id': 'recover-later-wording'}))
    assert second['proposal']['state'] == 'synchronized', second
    assert second['proposal']['receipt']['l1_operation_id'] == actual_operation
    assert second['proposal']['receipt']['l2_preserved_revision'] > f.owner_revision
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == changed[0]
    assert f.authority.get_memory('m1')['active_revision'] == 3
    assert len([c for c in f.calls if c[0] == 'memory_l3_sync']) == 1 and len(changed) == 1


def test_owner_edit_between_last_check_and_commit_is_rejected_by_real_authority_cas(tmp_path, monkeypatch):
    import memory_l3_sync
    f = setup(tmp_path)
    original = memory_l3_sync.commit_plan
    changed = []
    async def edit_before_commit(*args, **kwargs):
        product = f.l2.detail('m1')['l2']['presentation']
        document = deepcopy(product['document'])
        document['full_injection_text'] += '最后检查后保存的新措辞。'
        await f.l2.save('m1', {'expected_l1_revision': 2, 'expected_presentation_revision': product['presentation_revision'],
            'request_id': 'last-check-race', 'document': document, 'change_kind': 'wording'})
        changed.append(document['full_injection_text'])
        return await original(*args, **kwargs)
    monkeypatch.setattr(memory_l3_sync, 'commit_plan', edit_before_commit)
    first = run(f.sync.run_once(1))['results'][0]
    assert first['state'] == 'conflict' and first['receipt'] is None, first
    assert f.authority.get_memory('m1')['active_revision'] == 2
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == changed[0]
    monkeypatch.setattr(memory_l3_sync, 'commit_plan', original)
    queued = run(f.sync.action('m1', f.owner_revision, 'prepare', {'expected_revision': first['revision'], 'request_id': 'recompare-current-owner'}))
    assert queued['proposal']['state'] == 'pending'
    second = run(f.sync.run_once(1))['results'][0]
    assert second['state'] == 'synchronized', second
    assert f.authority.get_memory('m1')['active_revision'] == 3
    assert f.l2.detail('m1')['l2']['presentation']['document']['full_injection_text'] == changed[0]


def test_pending_known_scope_only_returns_complete_unaffected_units(tmp_path):
    f = setup(tmp_path, two_statements=True)
    statements = run(f.revisions.detail('m1'))['l1']['document']['statements']
    trip, place = [s['statement_id'] for s in statements]
    product = f.l2.detail('m1')['l2']['presentation']
    document = deepcopy(product['document'])
    document.update(full_injection_text='行程是5号。\n\n地点在北京。', coherent_sections=[
        {'section_id': 'owner-trip', 'text': '行程是5号。', 'statement_refs': [trip]},
        {'section_id': 'owner-place', 'text': '地点在北京。', 'statement_refs': [place]}])
    run(f.l2.save('m1', {'expected_l1_revision': 2, 'expected_presentation_revision': product['presentation_revision'],
        'request_id': 'owner-separate-units', 'document': document, 'change_kind': 'wording'}))
    def known(row):
        row['state'] = 'ambiguous'
        row['proposal']['processing']['scope'] = {'statement_ids': [trip], 'unknown_scope': False}
        return row
    f.sync.tasks.update('m1', f.owner_revision, 1, known)
    item = {'memory_id': 'm1', 'revision': 2, 'statement_ids': [trip, place], 'allowed_uses': ['current_basis'],
        'statement_uses': {trip: 'current_basis', place: 'current_basis'}, 'narrative_allowed': True}
    resolved = run(f.l2.resolve({'items': [item]}))['items'][0]
    assert resolved['state'] == 'pending_sync' and resolved['reason'] == 'l1_sync_required'
    assert resolved['text'] == '地点在北京。'
    assert resolved['coverage']['used_statement_ids'] == [place]
    assert not resolved['coverage']['full_narrative']
    row = f.sync.tasks.get('m1', f.owner_revision)
    def unknown(row):
        row['proposal']['processing']['scope']['unknown_scope'] = True
        return row
    f.sync.tasks.update('m1', f.owner_revision, row['proposal']['processing']['revision'], unknown)
    assert run(f.l2.resolve({'items': [item]}))['items'][0]['text'] == ''


def test_model_result_before_task_checkpoint_is_reused_after_local_failure(tmp_path, monkeypatch):
    f = setup(tmp_path)
    original = f.sync.tasks.update
    fail = [True]
    def interrupted(mid, revision, expected_revision, transform, **kwargs):
        if transform.__name__ == 'checkpoint' and fail[0]:
            fail[0] = False
            raise RuntimeError('checkpoint interrupted after model result persisted')
        return original(mid, revision, expected_revision, transform, **kwargs)
    monkeypatch.setattr(f.sync.tasks, 'update', interrupted)
    first = run(f.sync.run_once(1))['results'][0]
    assert first['state'] == 'failed' and first['plan'] is None
    assert f.authority.get_memory('m1')['active_revision'] == 2
    run(f.sync.action('m1', f.owner_revision, 'retry', {'expected_revision': first['revision'], 'request_id': 'recover-raw-generation'}))
    second = run(f.sync.run_once(1))['results'][0]
    assert second['state'] == 'synchronized', second
    assert len([c for c in f.calls if c[0] == 'memory_l3_sync']) == 1
    assert f.authority.get_memory('m1')['active_revision'] == 3
