"""Exact annotation targets and adopted revisions on isolated real Authority DBs."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from b1_protocol import digest
from memory_l1_contract import canonical_json
from memory_l1_service import L1OwnerError
from memory_l3_followup import FollowupTasks, run_followup_tasks
from memory_authority import MemoryAuthorityStore, RevisionConflict
from memory_l3_records import MemoryL3Records
from memory_self_review_source import check_native_sources, matches_l2_diagnosis_draft, snapshot, target_key, TARGET_FIELDS
from persona_memory_reaction_context import memory_ref
from test_memory_l2_p4 import run, owner_payload
from test_stage03_l3_followup import fixture


def native_fixture(tmp_path, *, layer='l1', mode='auto', owner_l2=False,
                   kind='correction', value='explicit_owner_correction',
                   diagnosis_only=False, diagnosis_contract='l2-owner-feedback-v1'):
    service, l2, calls, _ = fixture(tmp_path, mode=mode)
    engine = service.engine
    engine.config = {'persona': {'profile_id':'test-profile'},
        'gateway': {'canonical_continuation': {'enabled':True, 'conversation_id':'c1'}}}
    l1_model = engine._daily_chat_memory_create_completion
    async def model(client, **kwargs):
        if kwargs['task_stage'] != 'presentation_l2':
            return await l1_model(client, **kwargs)
        calls.append(kwargs)
        material = json.loads(kwargs['messages'][1]['content'])
        refs = [s['statement_id'] for s in material['l1_document']['statements']]
        correction = material.get('owner_feedback_correction')
        text = '基于原来源与批注修订的完整L2，保留其余经过。' if correction else '原L2完整内容。'
        return NS(text=json.dumps({'full_injection_text':text,
            'coherent_sections':[{'section_id':'whole','text':text,'statement_refs':refs}],
            'optional_use_variants':[], 'review':{'required':False,'reasons':[]}}, ensure_ascii=False), finish='stop')
    engine._daily_chat_memory_create_completion = model
    run(l2.run_once(1))
    if owner_l2:
        edit = owner_payload(l2)
        edit['document']['full_injection_text'] = '主人原手稿，不可被自动覆盖。'
        run(l2.save('m1', edit))
    current = l2.detail('m1')['l2']['presentation']
    fragment = {'memory_id':'m1', 'bucket_id':'b1','l1_revision':2,
        'l2_revision':current['presentation_revision'],'source_group_id':'source-group-m1',
        'text':current['document']['full_injection_text'],'source_refs':[], 'coverage':{}}
    fragment['text_sha256'] = sha256(fragment['text'].encode()).hexdigest()
    fragment['fragment_id'] = 'fragment_' + digest(fragment)[:32]
    scope = {'consumer_kind':'autonomy','identity_id':'owner','conversation_id':'c1',
        'activity_id':None,'activity_revision':None,'source_scope':'private','cycle_id':'cycle-1','operation_id':'operation-1'}
    context = {'schema':'memory-context-v1','scope':scope,'fragments':[fragment]}
    context['delivery_sha256'] = digest(context)
    context['delivery_receipt_id'] = 'native-delivery-1'
    key, feedback_key = target_key(fragment), 'reply-feedback:fb-1:r1'
    semantic = {'kind':kind,'long_term_value':value,
        'content':'只更正原材料中这一个具体差异。','evidence_refs':[feedback_key]}
    original = {'memory_fragment_ref':key, **semantic}
    diagnosis = {'feedback_id':'fb-1','layer':layer,'reason':'原文与使用文字在这里不一致。','memory_fragment_ref':key}
    parsed = {'feedback_ids':['fb-1'],'feedback_diagnoses':[diagnosis]}
    if diagnosis_only:
        semantic['content'] = diagnosis['reason']
    else:
        parsed['memory_updates'] = [original]
    text = canonical_json(parsed)
    review = {'identity_id':'owner','conversation_id':'c1','run_id':'run-1','cycle_id':'cycle-1',
        'operation_id':'operation-1','output_sha256':sha256(text.encode()).hexdigest()}
    def source(kind, key, text, locator, revision):
        return {'state':'available','eligible_source':True,'kind':kind,
            'source_namespace':'snapshot.aiz.' + ('self_review' if kind=='daily_review' else 'reply_feedback'),
            'source_id':canonical_json(locator),'source_revision':revision,'source_key':key,
            'text':text,'text_view_id':'native.exact-v1','content_sha256':sha256(text.encode()).hexdigest(),
            'recorded_at':'2026-11-06T02:00:00+00:00'}
    daily = source('daily_review','self_review',text, {'identity_id':'owner','conversation_id':'c1','run_id':'run-1'},review['output_sha256'])
    daily.update(review=review, memory_updates=parsed.get('memory_updates') or [],
        memory_targets=[{'target_key':key,'fragment':{field:fragment.get(field) for field in TARGET_FIELDS}}],
        feedback_origins={feedback_key:{'status':'available','targets':[{'target_key':key,'source_state':'available',
            'memory_id':'m1','l1_revision':2,'l2_revision':fragment['l2_revision']}]}},
        feedback_ids=['fb-1'],feedback_diagnoses=[diagnosis])
    if diagnosis_only and diagnosis_contract is not None:
        daily['diagnosis_draft_contract'] = diagnosis_contract
    feedback = source('reply_feedback',feedback_key,'主人具体指出原L2中的这一处差异。',
        {'identity_id':'owner','conversation_id':'c1','feedback_id':'fb-1'}, 'feedback:1:receipt')
    feedback.update(feedback_id='fb-1',feedback_revision=1)
    actual = {'self_review':daily, feedback_key:feedback}
    update = {'target_key':key,'memory_ref':memory_ref(fragment),**semantic}
    update['update_id'] = 'ml3:' + digest({'review':review, **{'target_key':key, **semantic}})
    payload = {'schema':'self-review-followup-v1','context':context,'review':review,
        'sources':deepcopy(list(actual.values())),'updates':[update]}
    async def reader(kind, source_id, revision):
        return deepcopy(next(s for s in actual.values() if s['kind']==kind and s['source_id']==source_id))
    service.review_source_reader = reader
    engine.memory_review_source_reader = reader
    engine.memory_followup_resolver = service.resolver
    original_sources = engine._memory_l1_sources()
    for kind in ('daily_review','reply_feedback'):
        class Adapter:
            async def read(self, binding):
                found = next(s for s in actual.values() if s['source_id']==binding['source_id'])
                return snapshot(found, engine.config) if found['eligible_source'] else {'state':'withdrawn'}
        original_sources.adapters['snapshot.aiz.' + ('self_review' if kind=='daily_review' else 'reply_feedback')] = Adapter()
    calls.clear()
    return service, l2, calls, payload, actual


@pytest.mark.parametrize('change', ['wrong_target','not_chosen','wrong_layer','wrong_diagnosis_target',
    'inactive_origin','withdrawn_target','wrong_l1','wrong_l2','wrong_feedback_revision','not_originally_sent'])
def test_new_native_correction_requires_exact_chosen_original_target(tmp_path, change):
    service, _, calls, payload, actual = native_fixture(tmp_path)
    review, feedback = actual['self_review'], actual['reply-feedback:fb-1:r1']
    origin = review['feedback_origins']['reply-feedback:fb-1:r1']
    if change=='wrong_target': origin['targets'][0]['target_key']='other'
    elif change=='not_chosen': review['feedback_ids']=[]
    elif change=='wrong_layer': review['feedback_diagnoses'][0]['layer']='reply'
    elif change=='wrong_diagnosis_target': review['feedback_diagnoses'][0]['memory_fragment_ref']='other'
    elif change=='inactive_origin': origin['status']='unavailable'
    elif change=='withdrawn_target': origin['targets'][0]['source_state']='withdrawn'
    elif change=='wrong_l1': origin['targets'][0]['l1_revision']=1
    elif change=='wrong_l2': origin['targets'][0]['l2_revision']=999
    elif change=='wrong_feedback_revision': feedback['feedback_revision']=2
    else: review['memory_targets'][0]['fragment']['text_sha256']='0'*64
    with pytest.raises(L1OwnerError): run(service.submit(payload))
    assert service.tasks.list()==[] and calls==[]


def test_old_native_review_without_new_fields_remains_compatible(tmp_path):
    service, _, _, payload, actual = native_fixture(tmp_path)
    actual['self_review'].pop('feedback_origins')
    actual['self_review'].pop('feedback_diagnoses')
    actual['self_review'].pop('feedback_ids')
    assert run(service.submit(payload))['tasks'][0]['phase']=='queued'


def test_diagnosed_feedback_cannot_relabel_l2_correction_as_ordinary_understanding(tmp_path):
    service,_,calls,payload,_=native_fixture(tmp_path,layer='l2',
        kind='understanding',value='specific_future_use')
    with pytest.raises(L1OwnerError) as caught:
        run(service.submit(payload))
    assert caught.value.code=='followup_diagnosed_feedback_requires_correction'
    assert service.tasks.list()==[] and calls==[]
    assert service.authority.get_memory('m1')['active_revision']==2


def test_native_l1_adoption_has_exact_receipt_and_no_fake_rebuilt_l2(tmp_path):
    service, l2, calls, payload, _ = native_fixture(tmp_path)
    task = run(service.submit(payload))['tasks'][0]
    result = run(run_followup_tasks(service.engine,None,max_jobs=1))['results'][0]
    assert result['phase']=='prepared', result
    status = FollowupTasks(service.authority).current_receipt(task['task_id'])
    assert status['phase']=='adopted' and status['l1_revision']==3 and status['l2_revision'] is None
    event = next(e for e in MemoryL3Records(service.authority,l2.revisions.commit).timeline('m1')['items']
                 if e['kind']=='followup_proposal')
    assert event['source_refs'][0].startswith('snapshot.aiz.reply_feedback:')
    assert len(calls)==1
    regenerated=run(l2.run_once(1))['results'][0]
    after=service.tasks.current_receipt(task['task_id'])
    assert regenerated['state']=='ready' and after['l2_revision']==regenerated['presentation_revision']
    assert after['l1_revision']==3


@pytest.mark.parametrize('owner_l2',[False,True])
def test_l2_only_full_draft_waits_then_exact_owner_save_adopts_without_l1_write(tmp_path, owner_l2):
    service, l2, calls, payload, _ = native_fixture(tmp_path,layer='l2',owner_l2=owner_l2)
    old = deepcopy(l2.detail('m1')['l2']['presentation'])
    task = run(service.submit(payload))['tasks'][0]
    prepared = run(run_followup_tasks(service.engine,None,max_jobs=1))['results'][0]
    assert prepared['phase']=='waiting_review' and calls==[]
    row = service.authority.get_candidate(task['task_id'])
    assert row['status']=='pending' and row['policy']['requires_owner_confirmation']
    handoff = run(service.adopt(row,owner=True,request_id='review-l2'))
    assert handoff['status']=='waiting_review'
    queued = run(l2.prepare_followup('m1',handoff['review_actions']['prepare_payload']))
    assert queued['review_required'] and l2.detail('m1')['l2']['presentation']==old
    assert service.tasks.current_receipt(task['task_id'])['phase']=='queued'
    generated = run(l2.run_once(1))['results'][0]
    assert generated['state']=='pending_review', generated
    draft = l2.detail('m1',generated['presentation_revision'])['l2']['presentation']
    assert draft['publication']=='candidate' and l2.detail('m1')['l2']['presentation']==old
    material = json.loads(calls[0]['messages'][1]['content'])['owner_feedback_correction']
    assert material['original_l2']==old['document']
    assert material['owner_annotations'][0]['text'].startswith('主人具体指出')
    assert service.authority.get_memory('m1')['active_revision']==2
    edit = owner_payload(l2,request='owner-applies-l2')
    edit.update(followup_task_id=task['task_id'],document=deepcopy(draft['document']))
    edit['document']['full_injection_text']+=' 主人审核后的补充。'
    saved = run(l2.save('m1',edit))
    assert saved['l2']['state']=='ready' and run(l2.save('m1',edit))['replayed']
    assert saved['l2']['presentation']['document']['coherent_sections'][0]['text'].endswith('主人审核后的补充。')
    status = FollowupTasks(service.authority).current_receipt(task['task_id'])
    assert status=={'task_id':task['task_id'],'memory_id':'m1','phase':'adopted',
        'l1_revision':2,'l2_revision':saved['presentation_revision']}
    assert service.authority.get_candidate(task['task_id'])['status']=='committed'
    assert service.authority.get_memory('m1')['active_revision']==2
    assert l2.detail('m1',old['presentation_revision'])['l2']['presentation']['document']==old['document']
    assert FollowupTasks(service.authority).current_receipt(task['task_id'])==status


def test_l2_source_revocation_during_generation_keeps_head_and_no_adoption(tmp_path):
    service,l2,calls,payload,actual = native_fixture(tmp_path,layer='l2',owner_l2=True)
    task = run(service.submit(payload))['tasks'][0]
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    row=service.authority.get_candidate(task['task_id'])
    args=run(service.adopt(row,owner=True,request_id='review'))['review_actions']['prepare_payload']
    run(l2.prepare_followup('m1',args))
    old=deepcopy(l2.detail('m1')['l2']['presentation'])
    original=l2.engine._daily_chat_memory_create_completion
    async def revoke(client,**kwargs):
        result=await original(client,**kwargs)
        actual['reply-feedback:fb-1:r1']['eligible_source']=False
        return result
    l2.engine._daily_chat_memory_create_completion=revoke
    result=run(l2.run_once(1))['results'][0]
    assert result['state'] in {'failed','stale'}
    assert l2.detail('m1')['l2']['presentation']==old
    assert service.tasks.current_receipt(task['task_id'])['phase']!='adopted'


def test_l2_owner_save_revalidates_sources_and_original_head(tmp_path):
    service,l2,_,payload,actual = native_fixture(tmp_path,layer='l2',owner_l2=True)
    task=run(service.submit(payload))['tasks'][0]
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    edit=owner_payload(l2,request='exact-save')
    edit.update(followup_task_id=task['task_id'])
    original=deepcopy(l2.detail('m1')['l2']['presentation'])
    with pytest.raises(L1OwnerError): run(l2.save('m1',{**edit,'change_kind':'facts'}))
    actual['reply-feedback:fb-1:r1']['eligible_source']=False
    with pytest.raises(L1OwnerError): run(l2.save('m1',edit))
    assert l2.detail('m1')['l2']['presentation']==original
    actual['reply-feedback:fb-1:r1']['eligible_source']=True
    other=owner_payload(l2,request='unrelated-owner-edit')
    other['document']['full_injection_text']='主人另存的新手稿。'
    run(l2.save('m1',other))
    with pytest.raises(L1OwnerError): run(l2.save('m1',edit))
    assert l2.detail('m1')['l2']['presentation']['document']['full_injection_text']=='主人另存的新手稿。'
    assert service.tasks.current_receipt(task['task_id'])['phase']!='adopted'


def receipt_app(service):
    import hmac
    tree=ast.parse((Path(__file__).resolve().parents[1]/'server.py').read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
           and node.name in {'_authorized_memory_write','api_memory_followup_receipt'}]
    route_node=next(node for node in nodes if node.name=='api_memory_followup_receipt')
    path=ast.literal_eval(route_node.decorator_list[0].args[0])
    route_node.decorator_list=[]
    scope={'hmac':hmac,'_memory_write_token':lambda:'internal-test',
        'reflection_engine':service.engine,'memory_authority_store':service.authority}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-followup-route','exec'),scope)
    return Starlette(routes=[Route(path,scope['api_memory_followup_receipt'],methods=['GET'])])


def test_exact_internal_get_is_authenticated_read_only_and_survives_reopen(tmp_path):
    service,_,calls,payload,_=native_fixture(tmp_path,mode='review')
    task=run(service.submit(payload))['tasks'][0]
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    service.authority=MemoryAuthorityStore(service.authority.path)
    statements=[]
    original_connect=service.authority._connect
    def connect():
        connection=original_connect();connection.set_trace_callback(statements.append);return connection
    service.authority._connect=connect
    app=receipt_app(service)
    async def get(task_id,token=None):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            return await client.get('/internal/memory-followups/'+task_id,
                headers={'Authorization':'Bearer '+token} if token else {})
    assert run(get(task['task_id'])).status_code==401
    assert run(get(task['task_id'],'wrong')).status_code==401
    response=run(get(task['task_id'],'internal-test'))
    assert response.status_code==200 and response.headers['cache-control']=='no-store'
    assert response.json()['task']['phase']=='waiting_review'
    assert set(response.json()['task'])=={'task_id','memory_id','phase','l1_revision','l2_revision'}
    assert run(get('followup_missing','internal-test')).status_code==404
    assert all(not s.lstrip().upper().startswith(('INSERT','UPDATE','DELETE','BEGIN IMMEDIATE')) for s in statements)
    assert not any(' LIKE ' in s.upper() for s in statements)
    assert len(calls)==1


def test_existing_l2_http_prepare_and_owner_save_link_the_original_task(tmp_path):
    from test_memory_l2_api_p4 import _api_app, _send
    service,l2,_,payload,_=native_fixture(tmp_path,layer='l2',owner_l2=True)
    task=run(service.submit(payload))['tasks'][0]
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    app,_=_api_app(l2,service.authority)
    prepare={'expected_l1_revision':2,'request_id':'http-prepare','followup_task_id':task['task_id']}
    url='/api/memory-authority/memories/m1/presentation'
    denied=run(_send(app,'POST',url+'/prepare',token=None,json_body=prepare))
    assert denied.status_code==401
    queued=run(_send(app,'POST',url+'/prepare',json_body=prepare))
    assert queued.status_code==200 and queued.json()['review_required']
    generated=run(l2.run_once(1))['results'][0]
    draft=l2.detail('m1',generated['presentation_revision'])['l2']['presentation']['document']
    save=owner_payload(l2,request='http-save')
    save.update(document=draft,followup_task_id=task['task_id'])
    applied=run(_send(app,'POST',url+'/revisions',json_body=save))
    assert applied.status_code==200,applied.text
    status=FollowupTasks(MemoryAuthorityStore(service.authority.path)).current_receipt(task['task_id'])
    assert status['phase']=='adopted' and status['l1_revision']==2
    assert status['l2_revision']==applied.json()['presentation_revision']


def test_candidate_committed_or_task_label_without_authority_adoption_is_not_adopted(tmp_path):
    service,_,_,payload,_=native_fixture(tmp_path,mode='review')
    task=run(service.submit(payload))['tasks'][0]
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    candidate=service.authority.get_candidate(task['task_id'])
    accepted=service.authority.decide_candidate(task['task_id'],action='accept',expected_revision=candidate['revision'],
        request_id='synthetic-accept',actor='owner')
    service.authority.decide_candidate(task['task_id'],action='commit',expected_revision=accepted['revision'],
        request_id='synthetic-status-only',actor='commit_service')
    assert service.tasks.current_receipt(task['task_id'])['phase']=='awaiting_receipt'
    stored=service.tasks.get(task['task_id'])
    service.tasks.update(task['task_id'],expected_revision=stored['revision'],
        transform=lambda value:{**value,'phase':'adopted'})
    assert service.tasks.current_receipt(task['task_id'])['phase']=='awaiting_receipt'


def test_owner_changed_followup_draft_invalidates_queued_l2_generation(tmp_path):
    service,l2,calls,payload,_=native_fixture(tmp_path,layer='l2',owner_l2=True)
    task=run(service.submit(payload))['tasks'][0]
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    row=service.authority.get_candidate(task['task_id'])
    original_actions=run(service.adopt(row,owner=True,request_id='open'))['review_actions']
    run(l2.prepare_followup('m1',original_actions['prepare_payload']))
    run(service.adopt(row,owner=True,request_id='owner-edit',edit={'content':'主人补充的具体呈现修订意见。','title':'保留这个标题'}))
    run(run_followup_tasks(service.engine,None,max_jobs=1))
    changed=service.authority.get_candidate(task['task_id'])
    result=run(l2.run_once(1))['results'][0]
    assert result['state']=='stale' and calls==[]
    actions=run(service.adopt(changed,owner=True,request_id='reopen'))['review_actions']
    assert actions['prepare_payload']['request_id']!=original_actions['prepare_payload']['request_id']
    run(l2.prepare_followup('m1',actions['prepare_payload']))
    assert run(l2.run_once(1))['results'][0]['state']=='pending_review'
    assert changed['proposal']['proposed_body']=='主人补充的具体呈现修订意见。'
    assert changed['proposal']['metadata']['legacy_candidate']['title']=='保留这个标题'
    assert service.authority.get_memory('m1')['active_revision']==2


def resign_native_update(payload):
    update=payload['updates'][0]
    semantic={key:update[key] for key in ('target_key','kind','long_term_value','content','evidence_refs')}
    update['update_id']='ml3:'+digest({'review':payload['review'],**semantic})


def test_exact_l2_diagnosis_without_raw_update_enters_original_waiting_review_task(tmp_path):
    service,l2,calls,payload,actual=native_fixture(tmp_path,layer='l2',diagnosis_only=True,owner_l2=True)
    raw=deepcopy(actual['self_review']['text'])
    assert 'memory_updates' not in json.loads(raw)
    assert actual['self_review']['memory_updates']==[]
    recorded=run(service.submit(payload))
    task_id=recorded['tasks'][0]['task_id']
    prepared=run(run_followup_tasks(service.engine,None,max_jobs=1))['results'][0]
    assert prepared['task_id']==task_id and prepared['phase']=='waiting_review',prepared
    assert calls==[] and service.authority.get_memory('m1')['active_revision']==2
    candidate=service.authority.get_candidate(task_id)
    assert candidate['status']=='pending' and candidate['proposal']['metadata']['l3_followup']['layer']=='l2'
    assert candidate['proposal']['proposed_body']==actual['self_review']['feedback_diagnoses'][0]['reason']
    assert actual['self_review']['text']==raw and actual['self_review']['memory_updates']==[]
    assert run(service.submit(payload))['tasks'][0]['task_id']==task_id
    handoff=run(service.adopt(candidate,owner=True,request_id='open-original-l2-review'))
    assert handoff['status']=='waiting_review' and handoff['review_actions']['save_parameters']['followup_task_id']==task_id
    assert service.tasks.current_receipt(task_id)['phase']=='waiting_review'


@pytest.mark.parametrize('bad',['no_contract','wrong_contract','no_origins','l1','reply','unclear','not_chosen',
    'wrong_feedback','wrong_target','not_delivered','changed_origin','changed_reason','extra_evidence',
    'understanding_label','explicit_same_feedback','explicit_same_target'])
def test_diagnosis_draft_cannot_expand_original_owner_l2_authority(tmp_path,bad):
    service,_,calls,payload,actual=native_fixture(tmp_path,layer='l2',diagnosis_only=True)
    review=actual['self_review'];update=payload['updates'][0]
    if bad=='no_contract':review.pop('diagnosis_draft_contract')
    elif bad=='wrong_contract':review['diagnosis_draft_contract']='unknown-v1'
    elif bad=='no_origins':review.pop('feedback_origins')
    elif bad in {'l1','reply','unclear'}:review['feedback_diagnoses'][0]['layer']=bad
    elif bad=='not_chosen':review['feedback_ids']=[]
    elif bad=='wrong_feedback':review['feedback_diagnoses'][0]['feedback_id']='other-feedback'
    elif bad=='wrong_target':review['feedback_diagnoses'][0]['memory_fragment_ref']='memory-target:other'
    elif bad=='not_delivered':review['memory_targets']=[]
    elif bad=='changed_origin':review['feedback_origins']['reply-feedback:fb-1:r1']['targets'][0]['source_state']='changed'
    elif bad=='changed_reason':update['content']+='添加的推断。';resign_native_update(payload)
    elif bad=='extra_evidence':update['evidence_refs'].append('self_review');resign_native_update(payload)
    elif bad=='understanding_label':
        update.update(kind='understanding',long_term_value='specific_future_use');resign_native_update(payload)
    elif bad=='explicit_same_feedback':
        review['memory_updates']=[{'memory_fragment_ref':'different-target','kind':'understanding',
            'long_term_value':'none','content':'明确选择不产生另一项长期更正。','evidence_refs':['reply-feedback:fb-1:r1']}]
    elif bad=='explicit_same_target':
        review['memory_updates']=[{'memory_fragment_ref':update['target_key'],'kind':'question',
            'long_term_value':'none','content':'该目标已被显式处理。','evidence_refs':['self_review']}]
    with pytest.raises(L1OwnerError):run(service.submit(payload))
    assert service.tasks.list()==[] and calls==[]
    assert service.authority.get_memory('m1')['active_revision']==2


@pytest.mark.parametrize('revision',[None,0,-1,True])
def test_diagnosis_draft_requires_an_original_positive_l2_revision(tmp_path,revision):
    _,_,_,payload,actual=native_fixture(tmp_path,layer='l2',diagnosis_only=True)
    update=deepcopy(payload['updates'][0]);update['memory_ref']['l2_revision']=revision
    assert not matches_l2_diagnosis_draft(actual['self_review'],update,actual)


def test_diagnosis_draft_loses_source_after_read_without_adopting_or_calling_model(tmp_path):
    service,_,calls,payload,actual=native_fixture(tmp_path,layer='l2',diagnosis_only=True)
    recorded=run(service.submit(payload));task_id=recorded['tasks'][0]['task_id']
    actual['reply-feedback:fb-1:r1']['eligible_source']=False
    failed=run(run_followup_tasks(service.engine,None,max_jobs=1))['results'][0]
    assert failed['task_id']==task_id and failed['phase']=='failed'
    assert calls==[] and service.authority.get_candidate(task_id) is None
    assert service.authority.get_memory('m1')['active_revision']==2


def test_unrelated_malformed_raw_evidence_does_not_block_a_legitimate_diagnosis_draft(tmp_path):
    service,_,calls,payload,actual=native_fixture(tmp_path,layer='l2',diagnosis_only=True)
    actual['self_review']['memory_updates']=[{'memory_fragment_ref':'unrelated-target','evidence_refs':123}]
    assert matches_l2_diagnosis_draft(actual['self_review'],payload['updates'][0],actual)
    recorded=run(service.submit(payload))
    assert recorded['tasks'][0]['phase']=='queued' and calls==[]
