"""Existing owner API, original overlay, reload, and the actual policy consumer."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone

import httpx

from gateway import create_gateway_app
from memory_inner_affinity import build_inner_context
from test_memory_configuration_api import service_at


def test_existing_owner_api_saves_reads_reloads_and_consumer_receives_explicit_zero(tmp_path):
    path = tmp_path / 'runtime.yaml'
    service = service_at(path)
    app = create_gateway_app(config=service.config, service=service)
    app.state.gateway_service = service
    async def check():
        route = '/api/memory-authority/configuration'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://isolated') as client:
            assert (await client.get(route)).status_code == 401
            headers = {'Authorization': 'Bearer owner'}
            first = (await client.get(route, headers=headers)).json()
            values = deepcopy(first['values'])
            policy = {'enabled': False, 'max_bonus': 0, 'weights': {'curiosity': 0, 'libido': 1}}
            values['recall']['inner_state_affinity'] = policy
            body = {'expected_revision': first['revision'], 'request_id': 'actual-owner-inner-settings', 'values': values}
            saved = await client.put(route, headers=headers, json=body)
            assert saved.status_code == 200, saved.text
            assert saved.json()['applied'] is True
            readback = (await client.get(route, headers=headers)).json()
            assert readback['values']['recall']['inner_state_affinity'] == policy
            assert readback['revision'] == saved.json()['saved_revision']
            assert service.config['unrelated'] == {'kept': True}
            assert service.config['memory_system']['recall']['inner_state_affinity'] == policy
            consumed = build_inner_context(readback['values'], [], datetime.now(timezone.utc), '合成配置检查')
            assert consumed['policy']['enabled'] is False and consumed['policy']['max_bonus'] == 0
            assert consumed['policy']['weights']['curiosity'] == 0
            assert (await client.put(route, headers=headers, json=body)).json()['replayed'] is True
            before = path.read_bytes()
            bad = deepcopy(values)
            bad['recall']['inner_state_affinity']['weights']['not_registered'] = 1
            rejected = await client.put(route, headers=headers, json={
                'expected_revision': readback['revision'], 'request_id': 'unknown-dimension', 'values': bad})
            assert rejected.status_code == 400 and path.read_bytes() == before
            reloaded = service_at(path)
            snapshot = reloaded._memory_configuration_snapshot(apply=True)
            assert snapshot['values']['recall']['inner_state_affinity'] == policy
            assert reloaded.config['memory_system']['recall']['inner_state_affinity'] == policy
    asyncio.run(check())
