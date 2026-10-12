import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from care_action_links import CareActionLinks, CareLinkError, digest, intent_token
from reminder_store import ReminderStore


@pytest.fixture
def service(tmp_path):
    store = ReminderStore({'reminder_db_path': str(tmp_path/'care.sqlite')})
    store.create(title='合成备忘', content='第一行\n<img src=x>', reminder_id='fixture',
                 repeat_rule='once', end_at='2000-01-01')
    return store, CareActionLinks(store)


def fields(links, **changes):
    row = links.read('fixture', scope='owner-scope', session_id='owner')
    return {'operation_id':'link-one','scope':'owner-scope','session_id':'owner','reminder_id':'fixture',
        'snapshot_token':row['snapshot_token'],'intent_token':row['intent_token'],'action_id':'action-one',
        'action_revision':1,'action_status':'planned','completion_policy':'keep','completion_kind':'','actor':'owner',**changes}


def test_read_has_no_archive_or_count_side_effects(service):
    store, links = service
    before = store.get('fixture')
    first = links.list(scope='owner-scope', session_id='owner')
    again = links.read('fixture', scope='owner-scope', session_id='owner')
    assert again['snapshot_token'] == first['items'][0]['snapshot_token']
    assert before == store.get('fixture') and again['status'] == 'active'
    assert again['delivery_status'] == 'chat_injection_only'


def test_session_channel_scope_and_pagination(service):
    store, links = service
    store.create(title='other',content='other',session_id='other',reminder_id='other')
    store.create(title='telegram',content='other',channel='telegram',reminder_id='telegram')
    page = links.list(scope='owner-scope',session_id='owner',limit=1)
    assert len(page['items']) == 1 and not page['has_more']
    assert links.list(scope='owner-scope',session_id='')['items'] == []
    for key in ('other','telegram'):
        with pytest.raises(CareLinkError,match='not_found'):links.read(key,scope='owner-scope',session_id='owner')
    links.sync(fields(links))
    hidden = links.read('fixture',scope='different-scope',session_id='owner')
    assert hidden['linked_elsewhere'] and hidden['action_link'] is None


def test_receipt_restart_payload_conflict_and_uniqueness(service):
    store, links = service
    args = fields(links)
    first = links.sync(args)
    store.update('fixture', content='后来更新')
    assert CareActionLinks(store).sync(args) == first
    with pytest.raises(CareLinkError,match='payload_conflict'):links.sync({**args,'action_id':'other'})
    with pytest.raises(CareLinkError,match='already_linked'):links.sync(fields(links,operation_id='other',action_id='other'))


def test_snapshot_conflict_does_not_write_link(service):
    store, links = service
    args = fields(links)
    store.update('fixture', title='新的内容')
    with pytest.raises(CareLinkError,match='revision_conflict'):links.sync(args)
    assert links.read('fixture',scope='owner-scope',session_id='owner')['action_link'] is None


@pytest.mark.parametrize('policy,kind,expected',[('keep','owner_mark','active'),
    ('close_once','owner_mark','done'),('close_once','executor','active'),('close_once','self_assessment','active')])
def test_explicit_completion_semantics_and_counter_preservation(service,policy,kind,expected):
    store,links=service;args=fields(links,completion_policy=policy)
    links.sync(args)
    before=store.get('fixture')
    result=links.sync({**args,'operation_id':'finished','action_revision':2,
        'action_status':'completed','completion_kind':kind,'actor':'owner' if kind=='owner_mark' else 'executor'})
    after=store.get('fixture')
    assert after['status']==expected and result['state']=='synced'
    for key in ('title','content','reminder_count','daily_reminder_count'):assert before[key]==after[key]


def test_cancel_or_expiry_does_not_complete_action(service):
    store,links=service;args=fields(links,completion_policy='close_once');links.sync(args)
    links.sync({**args,'operation_id':'cancel','action_revision':2,'action_status':'cancelled'})
    assert store.get('fixture')['status']=='active'
    store.set_status('fixture','archived')
    assert links.read('fixture',scope='owner-scope',session_id='owner')['action_link']['action_status']=='cancelled'


def test_later_memo_edit_cannot_be_closed_by_old_intent(service):
    store,links=service;args=fields(links,completion_policy='close_once');links.sync(args)
    store.update('fixture',content='改成了另一项关怀')
    with pytest.raises(CareLinkError,match='source_changed'):
        links.sync({**args,'operation_id':'finished','action_revision':2,'action_status':'completed','completion_kind':'owner_mark'})
    assert store.get('fixture')['status']=='active'


@pytest.mark.parametrize('change',['postpone','reactivate'])
def test_manual_lifecycle_change_invalidates_old_close_instruction(service,change):
    store,links=service;args=fields(links,completion_policy='close_once');links.sync(args)
    if change=='postpone':store.snooze('fixture',minutes=60)
    else:store.set_status('fixture','archived');store.set_status('fixture','active')
    with pytest.raises(CareLinkError,match='source_changed'):
        links.sync({**args,'operation_id':'done','action_revision':2,'action_status':'completed','completion_kind':'owner_mark'})
    assert store.get('fixture')['status']=='active'


@pytest.mark.parametrize('rule',['daily','every_n_rounds','morning_evening'])
def test_recurring_memo_stays_independent(service,rule):
    store,links=service;store.update('fixture',repeat_rule=rule)
    with pytest.raises(CareLinkError,match='must_remain'):links.sync(fields(links,completion_policy='close_once'))


def test_agent_cannot_set_close_policy(service):
    _,links=service
    with pytest.raises(CareLinkError,match='requires_owner'):links.sync(fields(links,completion_policy='close_once',actor='agent'))


def test_out_of_order_and_concurrent_receipts(service):
    store,links=service;args=fields(links);links.sync(args)
    newest={**args,'operation_id':'newest','action_revision':4,'action_status':'cancelled'}
    with ThreadPoolExecutor(max_workers=2) as workers:
        results=list(workers.map(lambda _:CareActionLinks(store).sync(newest),range(2)))
    assert results[0]==results[1]
    late=links.sync({**args,'operation_id':'late','action_revision':2,'action_status':'paused'})
    assert late['state']=='superseded'
    assert links.read('fixture',scope='owner-scope',session_id='owner')['action_link']['action_revision']==4


@pytest.mark.parametrize('changes',[{'action_revision':True},{'snapshot_token':'fake'}, {'actor':'spoof'},
    {'completion_policy':'delete'}, {'action_status':'delivered'}, {'scope':''}])
def test_invalid_protocol_fails_before_mutation(service,changes):
    store,links=service;before=store.get('fixture')
    with pytest.raises(CareLinkError):links.sync(fields(links,**changes))
    assert before==store.get('fixture')


@pytest.mark.parametrize('header,cookie,configured,expected',[
    ('Bearer fixture',False,True,True),('bearer fixture',False,True,True),
    ('Bearer wrong',False,True,False),('',False,True,False),
    ('Bearer fixture',False,False,False),('',True,False,True),
    ('Basic fixture',False,True,False),('Bearer 合成',False,True,False)])
def test_actual_service_auth_helper_without_initializing_private_brain(monkeypatch,header,cookie,configured,expected):
    import ast,hmac,os
    from pathlib import Path
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'server.py').read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_require_owner_or_internal_care_access')
    namespace={'os':os,'hmac':hmac,'config':{},'_dashboard_authenticated':lambda request:cookie,
        '_require_dashboard_auth':lambda request:'unauthorized'}
    if configured:monkeypatch.setenv('OMBRE_GATEWAY_TOKEN','fixture')
    else:monkeypatch.delenv('OMBRE_GATEWAY_TOKEN',raising=False)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-care-auth>','exec'),namespace)
    result=namespace[node.name](SimpleNamespace(headers={'authorization':header}))
    assert (result is None)==expected
    # This helper must be used by all three new routes, while the old write
    # routes remain on their existing Dashboard authentication.
    names={'api_care_memos','api_care_memo_read','api_care_memo_action'}
    for route in tree.body:
        if isinstance(route,ast.AsyncFunctionDef) and route.name in names:
            assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==node.name for n in ast.walk(route))
        if isinstance(route,ast.AsyncFunctionDef) and route.name in {'api_reminder_create','api_reminder_update'}:
            assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_require_dashboard_auth' for n in ast.walk(route))
