"""Original-image wire evidence and owner-approved unknown-capacity policy."""
import asyncio
import base64
from copy import deepcopy
import hashlib
import json

import httpx
import pytest

from attachment_adoption import canonical, sha, verified_refs
from attachment_adoption_support import (
    attachment_payload, gateway_service, openai_response, r2_response_lookup_query,
)
from gateway import GatewayService
from gateway_request_capacity import has_full_attachment


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=')


def image_payload(protocol='openai'):
    payload, _, _, _ = attachment_payload()
    locator = dict(owner_id='owner-1', conversation_id='conversation-1',
        event_id='event-1', event_version_id='event-version-1', resource_id='res-image-1',
        raw_sha256=hashlib.sha256(PNG).hexdigest(), media_view_id='attachment.image.original-v1')
    sid = canonical(locator)
    source = dict(locator=locator, source_id=sid, source_revision='attachment_sha256:' + sha(sid),
        source_namespace='snapshot.aiz.attachment_image', source_kind='image',
        event_id=locator['event_id'], event_version_id=locator['event_version_id'],
        resource_id=locator['resource_id'], sha256=locator['raw_sha256'], mime='image/png')
    wrapper = dict(attachment_source=source, content=None, content_available=False,
        media_available=True, adoption='native_image', adopted_bytes=len(PNG), total_bytes=len(PNG))
    descriptor = json.dumps(wrapper, ensure_ascii=False)
    content = [{'type': 'text', 'text': descriptor}, {'type': 'image_url',
        'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}}]
    if protocol == 'anthropic':
        content = GatewayService._openai_content_to_anthropic_blocks(
            GatewayService.__new__(GatewayService), content)
    payload['messages'] = [{'role': 'user', 'content': content}]
    payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = [sha(descriptor)]
    expected = dict(source_id=sid, source_revision=source['source_revision'],
                    body_sha256=sha(descriptor), source_kind='image')
    return payload, expected


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_exact_original_inline_image_is_verified_on_both_provider_formats(protocol):
    payload, expected = image_payload(protocol)
    refs = verified_refs(payload, payload['_ombre_trace_context']['attachment_adoption'], 'conversation-1')
    assert refs == [expected]
    assert not has_full_attachment(payload, payload['_ombre_trace_context'])
    assert 'data:image' not in json.dumps(refs)


@pytest.mark.parametrize('mutation', ['different_bytes', 'separate_message', 'missing_image',
    'wrong_mime', 'wrong_scope', 'wrong_version_descriptor', 'wrong_size', 'unwanted_descriptor'])
def test_image_receipt_needs_complete_descriptor_and_matching_same_message_bytes(mutation):
    payload, _ = image_payload()
    message = payload['messages'][0]
    if mutation == 'different_bytes':
        message['content'][1]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(PNG + b'x').decode()
    elif mutation == 'separate_message':
        payload['messages'].append({'role': 'user', 'content': [message['content'].pop()]})
    elif mutation == 'missing_image':
        message['content'].pop()
    elif mutation == 'unwanted_descriptor':
        payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = ['a' * 64]
    else:
        wrapper = json.loads(message['content'][0]['text'])
        if mutation == 'wrong_mime': wrapper['attachment_source']['mime'] = 'image/jpeg'
        if mutation == 'wrong_scope': wrapper['attachment_source']['locator']['conversation_id'] = 'other'
        if mutation == 'wrong_version_descriptor': wrapper['attachment_source']['event_version_id'] = 'different'
        if mutation == 'wrong_size': wrapper['adopted_bytes'] += 1
        text = json.dumps(wrapper, ensure_ascii=False)
        message['content'][0]['text'] = text
        payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = [sha(text)]
    assert verified_refs(payload, payload['_ombre_trace_context']['attachment_adoption'], 'conversation-1') == []


@pytest.mark.parametrize('path', ['openai', 'anthropic', 'openai_stream', 'anthropic_stream'])
def test_real_forwarding_preserves_native_image_and_only_completed_attempt_is_adopted(path):
    protocol = 'anthropic' if path.startswith('anthropic') else 'openai'
    payload, expected = image_payload()
    original = deepcopy(payload['messages'])
    if protocol == 'anthropic':
        converter = GatewayService.__new__(GatewayService)
        original[0]['content'] = converter._openai_content_to_anthropic_blocks(original[0]['content'])
    sent = []
    streaming = path.endswith('_stream')
    sse = (b'data: {"choices":[{"index":0,"delta":{"content":"seen"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        if protocol == 'openai' else
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n')
    def provider(request):
        sent.append(json.loads(request.content))
        if streaming: return httpx.Response(200, content=sse, headers={'Content-Type': 'text/event-stream'})
        if protocol == 'anthropic':
            return httpx.Response(200, json={'content': [{'type': 'text', 'text': 'seen'}], 'stop_reason': 'end_turn'})
        return httpx.Response(200, json=openai_response(finish_reason='stop'))
    service, route = gateway_service(provider)
    service.gateway_cfg = {}  # actual capacity deliberately unknown
    async def run():
        if path == 'openai': response = await service._forward_upstream(payload)
        elif path == 'anthropic': response = await service._forward_anthropic_upstream(payload, route)
        elif path == 'openai_stream': response = await service._open_upstream_stream(route, payload)
        else: response = await service._open_anthropic_upstream_stream(route, payload)
        assert response.status_code == 200
        if streaming:
            assert service.attachment_adoption.lookup(r2_response_lookup_query())['state'] == 'unavailable'
            output = b''
            async for chunk in service._iter_upstream_bytes(response, {}, stream_started_at=0, body_started_at=0):
                output += chunk
            assert output == sse
        report = payload['_ombre_trace_context']['context_capacity']
        assert report['state'] == 'capacity_unverified'
        assert report['dispatch_allowed'] is True
        assert report['input_tokens_upper_bound'] is None
        assert report['required_tokens_upper_bound'] is None
        assert report['opaque_input_present'] is True
        assert sent[0]['messages'] == original
        receipt = service.attachment_adoption.lookup(r2_response_lookup_query())
        assert receipt['state'] == 'available' and receipt['attachments'] == [expected]
        await response.aclose()
        await service.http_client.aclose()
    asyncio.run(run())


def test_mixed_text_attachment_keeps_unknown_capacity_gate():
    payload, _ = image_payload()
    text_payload, _, _, _ = attachment_payload()
    payload['messages'].extend(text_payload['messages'])
    payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'].extend(
        text_payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'])
    assert has_full_attachment(payload, payload['_ombre_trace_context'])
    sent = []
    def provider(request):
        sent.append(request)
        return httpx.Response(200, json=openai_response(finish_reason='stop'))
    service, _ = gateway_service(provider)
    service.gateway_cfg = {}
    async def run():
        response = await service._forward_upstream(payload)
        assert response.status_code == 400
        assert response.json()['error']['code'] == 'context_capacity_unknown'
        assert sent == []
        await service.http_client.aclose()
    asyncio.run(run())
