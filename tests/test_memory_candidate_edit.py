import asyncio
import copy
import json

import pytest

from memory_authority import IdempotencyConflict, InvalidTransition, MemoryProposal, RevisionConflict
from memory_candidate_edit import save_owner_candidate
from memory_l1_schema import catalog_with_defaults, document_template, join_document, scene_template, split_document
from memory_l1_service import L1OwnerError, MemoryL1RevisionService
from memory_l1_contract import make_text_anchor
from memory_l1_schema import statement_template
from reflection_engine import ReflectionEngine
from test_chat_memory_tool import envelope, service
from test_memory_authority_r1 import reflection_config, attach_fake_commit_service, daily_candidate


def daily(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    projection = attach_fake_commit_service(engine)
    engine._store_daily_chat_memory_pending([{**daily_candidate(), 'mode': 'review', 'status': 'pending'}])
    return engine, projection, 'daily-1'


def save(engine, ident, edit, revision=1, request='save-1'):
    return save_owner_candidate(engine.memory_authority_store, ident, edit=edit,
                                expected_revision=revision, request_id=request)


def owner_service(engine):
    catalog = catalog_with_defaults()
    commit = engine._memory_authority_service(object())
    return MemoryL1RevisionService(engine.memory_authority_store, commit, object(),
                                   lambda: (catalog, 'owner-catalog-r1'))


def seed_l1(engine, ident='daily-1', *, source_type='daily_chat', diary=False, statement=False):
    row = engine.memory_authority_store.get_candidate(ident)
    proposal = dict(row['proposal'])
    proposal['source_type'] = source_type
    catalog = catalog_with_defaults()
    document = document_template('原始L1完整正文。', '原始L1标题', catalog=catalog,
                                 catalog_revision='model-catalog-r1')
    if diary:
        document['entry']['form'] = 'diary_full'
        document['entry']['document_group_id'] = 'diary-root-1'
        document['entry']['derived_from_revision'] = 4
        scene = scene_template(catalog, 'model-catalog-r1')
        scene.update({'scene_id': 'scene-1', 'scene_revision_id': 'scene-rev-1', 'title': '第一场景'})
        if statement:
            statement_value = statement_template()
            statement_value.update({
                'statement_id': 'statement-1', 'statement_revision_id': 'statement-rev-1',
                'content': '原始场景陈述。',
            })
            document['statements'] = [statement_value]
            scene['statement_refs'] = ['statement-1']
        document['scenes'] = [scene]
    body, semantic = split_document(document, catalog=catalog)
    metadata = proposal.setdefault('metadata', {})
    metadata['memory_l1'] = semantic
    metadata['semantic_annotations'] = {'version': 'memory-semantics-v1', 'items': [{'stale': True}]}
    if diary:
        metadata['diary_l1_bundle'] = {
            'memory_id': 'diary-root-1', 'expected_revision': 4,
            'scene_bodies': {'scene-1': '原始场景完整正文。'},
            'coverage': {'state': 'complete', 'source_length': 100},
        }
    proposal['proposed_body'] = body
    saved_proposal = MemoryProposal.from_mapping(proposal)
    return engine.memory_authority_store.revise_candidate(
        ident, expected_revision=row['revision'], proposal=saved_proposal,
        request_id='seed-l1-' + ident, actor='model')


def save_l1(engine, ident, edit, revision_service, revision=2, request='l1-save-1'):
    return save_owner_candidate(engine.memory_authority_store, ident, edit=edit,
                                expected_revision=revision, request_id=request,
                                revision_service=revision_service)


def test_save_keeps_pending_source_body_and_policy_then_explicitly_adopts(tmp_path):
    engine, projection, ident = daily(tmp_path)
    before = engine.memory_authority_store.get_candidate(ident)
    content = '主人自己的完整修订。' * 180
    row = save(engine, ident, {'content': content, 'tags': ['标签'], 'event_date': '2026-10-09'})
    assert row['status'] == 'pending' and row['revision'] == 2 and not projection.revisions
    assert row['proposal']['proposed_body'] == content
    for key in ('source_refs', 'original_excerpt', 'source_status', 'requested_mode', 'owner_explicit'):
        assert row['proposal'][key] == before['proposal'][key]
    result = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='adopt', expected_revisions={ident: 2}))
    assert result['created'] == 1
    assert engine.memory_authority_store.get_candidate(ident)['proposal']['proposed_body'] == content
    assert len(projection.revisions) == 1


def test_chat_save_uses_same_adapter_without_losing_body_kind_or_arrays(tmp_path):
    tool, projection = service(tmp_path, 'review')
    original = asyncio.run(tool.submit(envelope()))
    ident = original['candidate_id']
    row = save(tool.engine, ident, {'content': '主人保留的原文', 'kind': 'boundary', 'tags': ['不催'], 'domain': []})
    assert row['proposal']['memory_type'] == 'boundary'
    assert row['proposal']['metadata']['legacy_candidate']['kind'] == 'boundary'
    assert row['proposal']['metadata']['legacy_candidate']['tags'] == ['不催']
    assert not projection.revisions
    result = asyncio.run(tool.engine.confirm_daily_chat_memory([ident], object(), request_id='adopt', expected_revisions={ident: 2}))
    assert result['created'] == 1 and tool.store.get_candidate(ident)['status'] == 'committed'


def test_title_only_and_unchanged_body_preserve_time(tmp_path):
    tool, _ = service(tmp_path, 'review')
    original = envelope()
    ident = asyncio.run(tool.submit(original))['candidate_id']
    before = tool.store.get_candidate(ident)['proposal']['metadata']['narrative']['event_time']
    row = save(tool.engine, ident, {'title': '只改标题', 'content': original['arguments']['content']})
    assert row['proposal']['metadata']['narrative']['event_time'] == before
    row = save(tool.engine, ident, {'content': '正文发生变化'}, revision=2, request='save-2')
    assert row['proposal']['metadata']['narrative']['event_time']['precision'] == 'unknown'


def test_same_edit_receipt_replays_after_other_edits_and_rejects_changed_payload(tmp_path):
    engine, _, ident = daily(tmp_path)
    first = save(engine, ident, {'title': '第一版'})
    save(engine, ident, {'title': '第二版'}, revision=2, request='save-2')
    assert save(engine, ident, {'title': '第一版'}) == first
    assert engine.memory_authority_store.get_candidate(ident)['revision'] == 3
    with pytest.raises(IdempotencyConflict): save(engine, ident, {'title': '偷换请求'})
    with pytest.raises(RevisionConflict): save(engine, ident, {'title': '旧页面'}, request='old')


def test_stale_confirm_does_not_adopt_new_owner_revision(tmp_path):
    engine, projection, ident = daily(tmp_path)
    save(engine, ident, {'content': '另一个页面保存的修改'})
    result = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='stale', expected_revisions={ident: 1}))
    assert result['results'][0]['status'] == 'revision_conflict'
    assert engine.memory_authority_store.get_candidate(ident)['status'] == 'pending'
    assert not projection.revisions
    replay = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='stale', expected_revisions={ident: 2}))
    assert replay['reason'] == 'idempotency_conflict' and not projection.revisions


def test_concurrent_edit_during_confirm_returns_item_conflict(tmp_path, monkeypatch):
    engine, projection, ident = daily(tmp_path)
    def concurrent(*args, **kwargs):
        raise RevisionConflict('simulated CAS race')
    monkeypatch.setattr(engine.memory_authority_store, 'revise_candidate', concurrent)
    result = asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='race',
        expected_revisions={ident: 1}, edits={ident: {'content': '主人本次修改'}}))
    assert result['results'] == [{'id': ident, 'status': 'revision_conflict'}]
    assert not projection.revisions


@pytest.mark.parametrize('edit', [
    {}, {'source_refs': ['fake']}, {'narrative': {}}, {'owner_explicit': True},
    {'title': ''}, {'title': 't' * 101}, {'content': 'c' * 12001}, {'content': '  '},
    {'kind': 'fake'}, {'tags': 'not-an-array'}, {'domain': ['']}, {'confidence': float('nan')},
    {'confidence': 1.1}, {'importance': True}, {'importance': 11}, {'event_date': '2026-02-30'},
])
def test_invalid_edits_do_not_change_candidate(tmp_path, edit):
    engine, projection, ident = daily(tmp_path)
    before = engine.memory_authority_store.get_candidate(ident)
    with pytest.raises((ValueError, TypeError)): save(engine, ident, edit)
    assert engine.memory_authority_store.get_candidate(ident) == before
    assert not projection.revisions


def test_committed_candidate_is_not_editable_as_pending(tmp_path):
    engine, _, ident = daily(tmp_path)
    asyncio.run(engine.confirm_daily_chat_memory([ident], object(), request_id='adopt'))
    row = engine.memory_authority_store.get_candidate(ident)
    with pytest.raises(InvalidTransition): save(engine, ident, {'content': 'wrong path'}, revision=row['revision'])


def test_revisioned_save_does_not_clear_needs_edit_when_body_unchanged(tmp_path):
    engine, _, ident = daily(tmp_path)
    row = engine.memory_authority_store.get_candidate(ident)
    from memory_candidate_edit import owner_edit_payload
    row['proposal']['metadata']['legacy_candidate']['soft_flags'] = ['needs_owner_edit', 'weak_source_support']
    result = owner_edit_payload(row, {'content': row['proposal']['proposed_body']})
    assert 'needs_owner_edit' in result['metadata']['legacy_candidate']['soft_flags']


def test_l1_json_only_save_keeps_full_body_pending_and_replays_same_cas(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident)
    row = engine.memory_authority_store.get_candidate(ident)
    service = owner_service(engine)
    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    document['retrieval']['keywords'] = ['主人补充的检索词']
    edit = {'document': document}
    saved = save_l1(engine, ident, edit, service)
    assert saved['status'] == 'pending' and saved['revision'] == 3
    assert saved['proposal']['proposed_body'] == document['narrative']['body']
    assert 'body' not in saved['proposal']['metadata']['memory_l1']['narrative']
    assert saved['proposal']['metadata']['memory_l1']['retrieval']['keywords'] == ['主人补充的检索词']
    assert saved['proposal']['metadata']['memory_l1']['generation']['origin'] == 'owner_edit'
    assert 'semantic_annotations' not in saved['proposal']['metadata']
    assert not projection.revisions
    replay = save_l1(engine, ident, edit, service, request='l1-save-1')
    assert replay == saved
    assert engine.memory_authority_store.get_candidate(ident)['revision'] == 3


def test_l1_body_and_semantics_stay_in_one_revision_and_legacy_content_patch_is_rejected(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident)
    row = engine.memory_authority_store.get_candidate(ident)
    service = owner_service(engine)
    with pytest.raises(ValueError, match='l1_document_required'):
        save_owner_candidate(engine.memory_authority_store, ident, edit={'content': '只改正文'},
            expected_revision=2, request_id='legacy-patch', revision_service=service)

    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    document['narrative']['body'] = '主人改过的完整L1正文。'
    document['retrieval']['tags'] = ['owner-reviewed']
    saved = save_l1(engine, ident, {'document': document}, service)
    metadata = saved['proposal']['metadata']
    assert saved['proposal']['proposed_body'] == '主人改过的完整L1正文。'
    assert metadata['memory_l1']['retrieval']['tags'] == ['owner-reviewed']
    assert metadata['legacy_candidate']['content'] == saved['proposal']['proposed_body']
    assert metadata['legacy_candidate']['proposed_memory'] == saved['proposal']['proposed_body']
    assert 'semantic_annotations' not in metadata.get('legacy_candidate', {})
    assert saved['status'] == 'pending' and not projection.revisions


def test_l1_edit_requires_owner_service_and_rejects_managed_source_changes(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident)
    row = engine.memory_authority_store.get_candidate(ident)
    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    with pytest.raises(ValueError, match='l1_owner_service_required'):
        save_owner_candidate(engine.memory_authority_store, ident, edit={'document': document},
            expected_revision=2, request_id='missing-service')

    document['evidence_bindings'].append({'binding_id': 'forged-source-binding'})
    with pytest.raises(L1OwnerError) as exc:
        save_l1(engine, ident, {'document': document}, owner_service(engine))
    assert exc.value.code == 'managed_field_changed'
    assert engine.memory_authority_store.get_candidate(ident)['revision'] == 2
    assert not projection.revisions


def test_diary_l1_edit_keeps_root_cas_separate_and_updates_scene_bodies(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident, source_type='diary', diary=True)
    row = engine.memory_authority_store.get_candidate(ident)
    service = owner_service(engine)
    original_owner_document = service._owner_document
    observed = []

    def record_owner_document(*args, **kwargs):
        observed.append((kwargs['memory_id'], kwargs['revision']))
        return original_owner_document(*args, **kwargs)

    service._owner_document = record_owner_document
    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    document['narrative']['body'] = '主人编辑后的整篇日记记忆。'
    saved = save_l1(engine, ident, {
        'document': document,
        'scene_bodies': {'scene-1': '主人编辑后的完整场景正文。'},
    }, service)
    bundle = saved['proposal']['metadata']['diary_l1_bundle']
    assert observed == [('diary-root-1', 5)]
    assert bundle['expected_revision'] == 4
    assert bundle['scene_bodies'] == {'scene-1': '主人编辑后的完整场景正文。'}
    assert saved['proposal']['proposed_body'] == '主人编辑后的整篇日记记忆。'
    assert saved['status'] == 'pending' and not projection.revisions


def test_l1_candidate_outer_revision_conflict_keeps_saved_version(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident)
    row = engine.memory_authority_store.get_candidate(ident)
    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    document['retrieval']['keywords'] = ['first']
    service = owner_service(engine)
    save_l1(engine, ident, {'document': document}, service, request='l1-first')
    with pytest.raises(RevisionConflict):
        save_l1(engine, ident, {'document': document}, service, request='stale-l1', revision=2)
    assert engine.memory_authority_store.get_candidate(ident)['revision'] == 3
    assert not projection.revisions


def test_action_receipt_l1_candidate_uses_candidate_identity_for_first_revision(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident, source_type='action_receipt')
    row = engine.memory_authority_store.get_candidate(ident)
    service = owner_service(engine)
    observed = []
    original = service._owner_document

    def record_owner_document(*args, **kwargs):
        observed.append((kwargs['memory_id'], kwargs['revision']))
        return original(*args, **kwargs)

    service._owner_document = record_owner_document
    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    document['narrative']['body'] = '行动回执提炼后的完整记忆。'
    saved = save_l1(engine, ident, {'document': document}, service)
    assert observed == [(ident, 1)]
    assert saved['proposal']['proposed_body'] == '行动回执提炼后的完整记忆。'
    assert saved['status'] == 'pending' and not projection.revisions


def test_owner_semantic_statement_and_scene_anchor_edits_advance_program_subrevisions(tmp_path):
    engine, projection, ident = daily(tmp_path)
    seed_l1(engine, ident, source_type='diary', diary=True, statement=True)
    row = engine.memory_authority_store.get_candidate(ident)
    service = owner_service(engine)
    document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
    old_statement_revision = document['statements'][0]['statement_revision_id']
    old_scene_revision = document['scenes'][0]['scene_revision_id']
    document['statements'][0]['content'] = '主人修订后的场景陈述。'
    body = document['narrative']['body']
    document['scenes'][0]['anchors'] = [make_text_anchor(
        body, target_namespace='l1', target_id='diary-root-1', target_revision=5,
        text_view_id='memory.narrative.canonical-v1', start=0, end=4)]
    saved = save_l1(engine, ident, {'document': document}, service)
    updated = join_document(saved['proposal']['proposed_body'], saved['proposal']['metadata']['memory_l1'])
    assert updated['statements'][0]['content'] == '主人修订后的场景陈述。'
    assert updated['statements'][0]['statement_revision_id'] != old_statement_revision
    assert updated['statements'][0]['statement_revision_id'].startswith('sr_')
    assert updated['scenes'][0]['anchors'][0]['target_id'] == 'diary-root-1'
    assert updated['scenes'][0]['scene_revision_id'] != old_scene_revision
    assert updated['scenes'][0]['scene_revision_id'].startswith('sr_')
    assert saved['proposal']['metadata']['diary_l1_bundle']['scene_bodies']['scene-1'] == '原始场景完整正文。'
    assert saved['status'] == 'pending' and not projection.revisions


def test_edit_api_auth_validation_and_saved_receipt(tmp_path, monkeypatch):
    server = pytest.importorskip('server')
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    engine, projection, ident = daily(tmp_path)
    monkeypatch.setattr(server, 'reflection_engine', engine)
    monkeypatch.setattr(server, '_require_dashboard_auth', lambda request:
        None if request.headers.get('x-test-owner') == 'yes' else JSONResponse({'error': 'unauthorized'}, status_code=401))
    def request(body, owner=True):
        async def receive():
            return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': 'POST', 'path': '/',
                        'headers': [(b'x-test-owner', b'yes')] if owner else []}, receive)
    body = {'candidate_id': ident, 'edit': {'title': '独立保存'}, 'expected_revision': 1, 'request_id': 'api-save'}
    async def run():
        assert (await server.api_daily_chat_memory_edit(request(body, False))).status_code == 401
        assert (await server.api_daily_chat_memory_edit(request({**body, 'owner': True}))).status_code == 400
        response = await server.api_daily_chat_memory_edit(request(body))
        assert response.status_code == 200 and response.headers['Cache-Control'] == 'no-store'
        data = json.loads(response.body)
        assert data['status'] == 'saved' and data['adopted'] is False
        assert data['item']['authority_revision'] == 2 and data['item']['status'] == 'pending'
        assert (await server.api_daily_chat_memory_edit(request({**body, 'request_id': 'different'}))).status_code == 409
    asyncio.run(run())
    assert not projection.revisions
