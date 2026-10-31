"""Safe task metadata using the actual existing route/fallback resolver."""
from copy import deepcopy
import ast
import hmac
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from starlette.responses import JSONResponse

from memory_task_routes import public_task_models
from model_task_view07 import owner_response, public_view


def engine():
    def client(url,key):return SimpleNamespace(base_url=url,api_key=key)
    value=SimpleNamespace(config={'reflection':{'memory_task_routes':{
        'auto_l1':{'connection':'reflection','model':'same-model'},
        'chat_l1':{'connection':'daily','model':'same-model'},
        'diary_l1':{'connection':'dehydration','model':'same-model'}}}},
        client=client('https://u:private-password@reflection.invalid/v1?token=private-query','private-reflection-key'),
        dehydration_client=client('https://dehydration.invalid/v1','private-dehydration-key'),
        daily_chat_memory_client=client('https://daily.invalid/v1','private-daily-key'),
        reflection_connection_explicit=True,model='same-model',dehydration_model='same-model',
        daily_chat_memory_candidate_model='same-model',daily_chat_memory_summary_model='same-model')
    value.memory_task_models_public=lambda:public_task_models(value)
    return value


def test_existing_resolver_keeps_same_model_on_independent_connections_and_no_keys():
    source=engine();before=deepcopy(source.config)
    view=public_view(source)
    assert source.config==before
    assert view['effective']['auto_l1']['connection']=='reflection'
    assert view['effective']['chat_l1']['connection']=='daily'
    assert view['effective']['diary_l1']['connection']=='dehydration'
    assert len({row['connection_id'] for row in view['connections'].values()})==3
    assert view['connections']['reflection']['base_url']=='https://reflection.invalid/v1'
    body=json.dumps(view)
    for secret in ('private-password','private-query','private-reflection-key','private-daily-key','private-dehydration-key'):
        assert secret not in body
    assert all(value is None for value in view['last_observed'].values())
    assert all(row['credential_configured'] is True for row in view['connections'].values())


def test_unconfigured_explicit_connection_is_not_replaced_by_an_available_other_client():
    source=engine();source.daily_chat_memory_client=None
    view=public_view(source)
    assert view['effective']['chat_l1']['connection']=='daily'
    assert view['effective']['chat_l1']['configured'] is False
    assert view['connections']['daily']['credential_configured'] is None
    assert view['last_observed']['chat_l1'] is None


def test_owner_guard_runs_before_getter_and_no_query_or_body_surfaces():
    source=engine();calls=[]
    original=source.memory_task_models_public
    source.memory_task_models_public=lambda:(calls.append('read') or original())
    request=SimpleNamespace(query_params={})
    denied=owner_response(request,source,lambda req:JSONResponse({'error':'unauthorized'},status_code=401))
    assert denied.status_code==401 and calls==[]
    bad=owner_response(SimpleNamespace(query_params={'owner':'peer'}),source,lambda req:None)
    assert bad.status_code==400 and calls==[]
    good=owner_response(request,source,lambda req:None)
    assert good.status_code==200 and calls==['read']
    assert 'client' not in json.loads(good.body)['effective']['chat_l1']


def test_actual_internal_memory_guard_token_precedence_without_reading_real_credentials():
    # Execute the actual three original authentication helpers in an isolated
    # namespace; importing server would initialize unrelated runtime owners.
    tree=ast.parse((Path(__file__).parents[1]/'server.py').read_text(encoding='utf-8-sig'))
    names={'_memory_write_token','_authorized_memory_write','_require_owner_or_internal_memory_read'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    space={'os':os,'hmac':hmac,'config':{'gateway':{'token':'synthetic-config-token'}},
        '_dashboard_authenticated':lambda request:False,
        '_require_dashboard_auth':lambda request:JSONResponse({'error':'unauthorized'},status_code=401)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_memory_read_auth','exec'),space)
    guard=space['_require_owner_or_internal_memory_read']
    request=lambda token:SimpleNamespace(headers={'authorization':'Bearer '+token})
    with patch.dict(os.environ,{'OMBRE_MEMORY_WRITE_TOKEN':'synthetic-memory-token','OMBRE_GATEWAY_TOKEN':'synthetic-gateway-token'},clear=True):
        assert guard(request('synthetic-memory-token')) is None
        assert guard(request('synthetic-gateway-token')).status_code==401
    with patch.dict(os.environ,{'OMBRE_GATEWAY_TOKEN':'synthetic-gateway-token'},clear=True):
        assert guard(request('synthetic-gateway-token')) is None
    with patch.dict(os.environ,{},clear=True):
        assert guard(request('synthetic-config-token')) is None
