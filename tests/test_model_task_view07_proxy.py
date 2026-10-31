"""Actual owner read helper/auth guard plus mock HTTP; synthetic tokens only."""
import ast
import asyncio
import __future__
import hmac
import json
import os
from pathlib import Path
import secrets
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from starlette.responses import JSONResponse

import model_task_view07_proxy as proxy


def gateway_owner():
    root=Path(__file__).parents[1]
    tree=ast.parse((root/'gateway.py').read_text(encoding='utf-8-sig'))
    klass=next(node for node in tree.body if isinstance(node,ast.ClassDef)
        and any(isinstance(method,ast.AsyncFunctionDef) and method.name=='handle_memory_authority_read' for method in node.body))
    methods=[method for method in klass.body if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef))
        and method.name in {'_authorize','handle_memory_authority_read'}]
    owner=ast.ClassDef(name='OwnerRead',bases=[],keywords=[],body=methods,decorator_list=[])
    space={'JSONResponse':JSONResponse,'secrets':secrets}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[owner],type_ignores=[])),'actual_gateway_read','exec',
        flags=__future__.annotations.compiler_flag),space)
    service=space['OwnerRead']();service.gateway_token='synthetic-gateway-owner'
    return service


def brain_guard():
    root=Path(__file__).parents[2]/'stage04-autonomy-brain'
    tree=ast.parse((root/'server.py').read_text(encoding='utf-8-sig'))
    names={'_memory_write_token','_authorized_memory_write'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    space={'os':os,'hmac':hmac,'config':{'gateway':{'token':'synthetic-gateway-owner'}}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_brain_memory_auth','exec'),space)
    return space['_authorized_memory_write']


@pytest.mark.parametrize('dedicated',[True,False])
def test_existing_owner_guard_fixed_url_and_original_memory_bearer_precedence(dedicated):
    async def scenario():
        service=gateway_owner();guard=brain_guard();calls=[]
        expected='synthetic-memory-write' if dedicated else service.gateway_token
        def transport(request):
            calls.append(request)
            assert str(request.url)=='https://brain-admin.invalid/internal/memory-task-models'
            assert request.headers['Authorization']=='Bearer '+expected
            assert guard(SimpleNamespace(headers=request.headers))
            return httpx.Response(200,json={'schema':'brain-model-task-view07-v1','routes':{},'effective':{},
                'connections':{},'last_observed':{},'unexpected_private':'never-forward'})
        original_client=httpx.AsyncClient
        def client(**kwargs):return original_client(transport=httpx.MockTransport(transport),**kwargs)
        env={'OMBRE_MEMORY_INDEX_ADMIN_URL':'https://brain-admin.invalid/', 'OMBRE_GATEWAY_TOKEN':service.gateway_token,
            'UNRELATED_B1_TOKEN':'synthetic-b1-not-memory'}
        if dedicated:env['OMBRE_MEMORY_WRITE_TOKEN']=expected
        request=SimpleNamespace(method='GET',query_params={},path_params={'resource':'model-task-models'},
            headers={'Authorization':'Bearer '+service.gateway_token})
        with patch.dict(os.environ,env,clear=True),patch.object(proxy.httpx,'AsyncClient',client):
            response=await service.handle_memory_authority_read(request)
            assert response.status_code==200 and len(calls)==1
            data=json.loads(response.body)
            assert 'unexpected_private' not in data
            assert expected not in response.body.decode() and service.gateway_token not in response.body.decode()
            request.headers={'Authorization':'Bearer wrong-owner'}
            denied=await service.handle_memory_authority_read(request)
            assert denied.status_code==401 and len(calls)==1
    asyncio.run(scenario())


@pytest.mark.parametrize('status,code',[(401,'model_task_view_auth_failed'),(404,'model_task_view_not_deployed')])
def test_upstream_auth_and_missing_endpoint_remain_visible_without_error_body(status,code):
    async def scenario():
        service=gateway_owner();original_client=httpx.AsyncClient
        def transport(request):return httpx.Response(status,text='private-upstream-body-secret')
        def client(**kwargs):return original_client(transport=httpx.MockTransport(transport),**kwargs)
        request=SimpleNamespace(method='GET',query_params={},path_params={'resource':'model-task-models'},
            headers={'Authorization':'Bearer '+service.gateway_token})
        with patch.dict(os.environ,{'OMBRE_MEMORY_INDEX_ADMIN_URL':'https://brain-admin.invalid'},clear=True),patch.object(proxy.httpx,'AsyncClient',client):
            response=await service.handle_memory_authority_read(request)
        assert response.status_code==status and json.loads(response.body)['error']['code']==code
        assert 'private-upstream' not in response.body.decode()
    asyncio.run(scenario())
