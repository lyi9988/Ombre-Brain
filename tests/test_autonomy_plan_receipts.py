import asyncio
import hashlib
import json

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from gateway import GatewayService
from gateway_state import GatewayStateStore


def test_authenticated_plan_replay_pickup_and_acknowledgement(tmp_path):
    async def scenario():
        class LocalService(GatewayService):
            @property
            def recall_policy(self):
                return None
        service = object.__new__(LocalService)
        service.state_store = GatewayStateStore(str(tmp_path / 'gateway.db'))
        service._authorize = lambda token: None if token == 'Bearer private' else JSONResponse({}, status_code=401)
        service._maybe_reload_runtime_overlay = lambda: None
        service._memory_configuration_snapshot = lambda **kwargs: {'revision': 1, 'values': {}}
        scheduled, released = asyncio.Event(), asyncio.Event()
        calls = []
        async def model(request):
            calls.append(1)
            scheduled.set()
            await released.wait()
            return JSONResponse({'choices': [{'message': {'role': 'assistant', 'content': '{"decision":"rest"}',
                                                           'reasoning_content': 'hidden internal reasoning'}}]})
        service._handle_chat_unfrozen = model
        app = Starlette(routes=[Route('/v1/chat/completions', service.handle_chat, methods=['POST']),
                                Route('/internal/autonomy/planning-result', service.handle_autonomy_plan_result, methods=['POST'])])
        messages = [{'role': 'system', 'content': '计划'}]
        body = {'model': 'fixture', 'messages': messages}
        h = lambda text: hashlib.sha256(text.encode()).hexdigest()
        headers = {'Authorization': 'Bearer private', 'X-Guyan-Request-Id': 'example-cycle',
                   'X-Guyan-Trace-Id': 'trace-autonomy-' + h('example-cycle')[:32],
                   'X-Guyan-Plan-Receipt': 'ab' * 32, 'X-Guyan-Runtime-Scope': 'internal:autonomy_chain'}
        key = {'cycle_id': 'example-cycle', 'ticket': 'ab' * 32,
               'input_sha256': h(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')))}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testing') as client:
            async def read(arguments):
                return await client.post('/internal/autonomy/planning-result', json=arguments,
                                         headers={'Authorization': 'Bearer private'})
            async def send(payload):
                return await client.post('/v1/chat/completions', json=payload, headers=headers)
            assert (await client.post('/internal/autonomy/planning-result',json=key)).status_code == 401
            original = asyncio.create_task(send(body))
            await asyncio.wait_for(scheduled.wait(), 2)
            assert (await read(key)).json()['state'] == 'pending'
            assert (await send(body)).status_code == 409
            released.set()
            assert (await original).status_code == 200
            service.state_store = GatewayStateStore(str(tmp_path / 'gateway.db'))
            original_text = (await read(key)).json()
            assert original_text['state'] == 'available' and original_text['output'] == '{"decision":"rest"}'
            assert 'hidden' not in str(original_text)
            assert (await read({**key, 'ticket': 'cc' * 32})).json()['state'] == 'not_found'
            assert (await read({**key, 'input_sha256':'0'*64})).status_code == 400
            assert (await read({**key, 'acknowledge':'0'*64})).status_code == 400
            assert (await send(body)).json()['choices'][0]['message']['content'] == original_text['output']
            assert (await read({**key, 'acknowledge':h(original_text['output'])})).json()['state'] == 'acknowledged'
            assert (await read(key)).json() == {'state': 'acknowledged'}
            assert (await send(body)).status_code == 409
            assert calls == [1]
    asyncio.run(scenario())
