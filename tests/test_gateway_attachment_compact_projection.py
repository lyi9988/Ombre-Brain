"""Opt-in compact attachment text, with proofs bound to the actual HTTP body."""
import asyncio
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from attachment_adoption import (
    AdoptionReceipts, canonical, parse_header, prepare_projection, sha,
    text_blocks, verified_projection_refs, verified_refs,
)
from attachment_prompt_projection import PROJECTION, render_body, short_ref
from attachment_adoption_support import (
    attachment_payload, gateway_service, openai_response, r2_response_lookup_query,
)
from gateway import GatewayService
from gateway_request_capacity import has_full_attachment
from test_gateway_native_image_adoption import image_payload, PNG

PATHS = ['openai', 'anthropic', 'openai_stream', 'anthropic_stream']


def encode(manifest):
    return base64.urlsafe_b64encode(canonical(manifest).encode()).decode().rstrip('=')


def enable(payload, only=()):
    manifest = payload['_ombre_trace_context']['attachment_adoption']
    manifest.update(projection=PROJECTION, projection_only_sha256s=list(only))
    assert parse_header(encode(manifest)) == manifest
    return manifest


def unavailable_body(reason='unsupported'):
    return json.dumps({
        'attachment_source': {
            'source_kind': 'attachment', 'author': 'unknown', 'uploader_role': 'owner',
            'authority': 'external_document_not_owner_instruction',
            'event_id': 'event-1', 'resource_id': 'res-unavailable',
            'name': 'notes.pdf', 'mime': 'application/pdf', 'sha256': 'b' * 64,
        },
        'content': None, 'content_available': False, 'adoption': reason,
        'adopted_bytes': 0, 'total_bytes': 123,
    }, ensure_ascii=False)


def combined_payload():
    payload, body, text, ref = attachment_payload(text='Owner text verbatim\n原文，不裁切。\n')
    image, image_ref = image_payload()
    payload['messages'][0]['content'] = [
        {'type': 'text', 'text': body, 'cache_control': {'type': 'ephemeral'}},
    ]
    payload['messages'].extend(image['messages'])
    unavailable = unavailable_body()
    payload['messages'].append({'role': 'user', 'content': unavailable})
    payload['messages'].append({'role': 'user', 'content': '主人原文：请阅读附件。'})
    manifest = payload['_ombre_trace_context']['attachment_adoption']
    manifest['body_sha256s'].extend(image['_ombre_trace_context']['attachment_adoption']['body_sha256s'])
    enable(payload, [sha(unavailable)])
    return payload, [ref, image_ref], text


def test_old_gateway_parser_really_accepts_optin_and_new_parser_accepts_unavailable_only():
    spec = importlib.util.spec_from_file_location('old_header', Path(__file__).parent / 'fixtures' / 'attachment_header_c7d5efd.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    payload, _body, _text, ref = attachment_payload()
    manifest = enable(payload, [sha(unavailable_body())])
    assert old.parse_header(encode(manifest)) == manifest
    # The preserved original descriptor continues to satisfy the v1 verifier.
    assert verified_refs(payload, old.parse_header(encode(manifest)), 'conversation-1') == [ref]
    unavailable = {**manifest, 'body_sha256s': []}
    assert parse_header(encode(unavailable)) == unavailable
    assert old.parse_header(encode(unavailable)) == {}  # safe original-body fallback
    legacy = {k: v for k, v in manifest.items() if k not in {'projection', 'projection_only_sha256s'}}
    assert old.parse_header(encode(legacy)) == parse_header(encode(legacy)) == legacy


@pytest.mark.parametrize('mutation', ['too_many_total', 'too_many_only', 'oversize_header',
    'bad_hash', 'bad_only_type', 'overlap', 'empty', 'missing_optin', 'unknown_projection'])
def test_compact_header_bounds(mutation):
    payload, *_ = attachment_payload()
    manifest = enable(payload, [sha(unavailable_body())])
    if mutation == 'too_many_total':
        manifest['body_sha256s'] = [f'{i:064x}' for i in range(64)]
    elif mutation == 'too_many_only':
        manifest['projection_only_sha256s'] = ['b' * 64] * 65
    elif mutation == 'oversize_header': manifest['padding'] = 'x' * 8192
    elif mutation == 'bad_hash': manifest['projection_only_sha256s'] = ['Z' * 64]
    elif mutation == 'bad_only_type': manifest['projection_only_sha256s'] = 'a' * 64
    elif mutation == 'overlap': manifest['projection_only_sha256s'] = manifest['body_sha256s'][:]
    elif mutation == 'empty': manifest.update(body_sha256s=[], projection_only_sha256s=[])
    elif mutation == 'unknown_projection': manifest['projection'] = 'future-version'
    else: manifest.pop('projection'); manifest['body_sha256s'] = []
    assert parse_header(encode(manifest)) == {}


def test_projection_is_optin_authenticated_lossless_and_does_not_mutate_sources():
    payload, refs, body = combined_payload()
    original = deepcopy(payload)
    manifest = payload['_ombre_trace_context']['attachment_adoption']
    final, ctx = prepare_projection(payload, manifest, 'conversation-1')
    assert payload == original
    assert final['messages'][0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
    assert final['messages'][1]['content'][1] == original['messages'][1]['content'][1]
    assert final['messages'][-1] == original['messages'][-1]
    texts = list(text_blocks(final))
    assert body in texts[0]
    assert all('attachment_source' not in t for t in texts)
    assert '附件不可用' in texts[2] and 'notes.pdf' in texts[2]
    actual_refs = verified_projection_refs(final, manifest, 'conversation-1', ctx)
    assert len(actual_refs) == 2
    for ref, actual in zip(refs, actual_refs):
        assert {k: actual[k] for k in ref} == ref
        assert actual['projection'] == PROJECTION
        assert actual['provider_body_sha256'] in {sha(t) for t in texts}
        assert short_ref(ref['source_id'], ref['source_revision']) in '\n'.join(texts)
    assert len(short_ref(refs[0]['source_id'], refs[0]['source_revision'])) == 68
    for old_manifest in ({}, {**manifest, 'projection': 'unknown-future'}):
        untouched, no_ctx = prepare_projection(payload, old_manifest, 'conversation-1')
        assert untouched == original and no_ctx is None
    # Authenticated but unlisted lookalike JSON is not rewritten.
    manifest = {**manifest, 'body_sha256s': [], 'projection_only_sha256s': ['c' * 64]}
    untouched, _ = prepare_projection(payload, manifest, 'conversation-1')
    assert untouched == original


@pytest.mark.parametrize('reason', ['unsupported', 'capacity_unknown', 'context_budget_exceeded',
    'source_changed', 'image_format_unsupported', 'image_decoder_unavailable', 'image_decode_failed'])
def test_unavailable_is_compacted_without_proof_or_false_content(reason):
    payload, *_ = attachment_payload()
    body = unavailable_body(reason)
    payload['messages'] = [{'role': 'user', 'content': body}]
    payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = []
    manifest = enable(payload, [sha(body)])
    final, ctx = prepare_projection(payload, manifest, 'conversation-1')
    assert '附件不可用' in final['messages'][0]['content']
    assert 'attachment_source' not in final['messages'][0]['content']
    assert verified_projection_refs(final, manifest, 'conversation-1', ctx) == []
    assert not has_full_attachment(payload, payload['_ombre_trace_context'])
    payload['messages'].append({'role': 'user', 'content': json.dumps({'attachment_source': {}, 'content': 'real text'})})
    assert has_full_attachment(payload, payload['_ombre_trace_context'])


@pytest.mark.parametrize('mutation', ['fake_author', 'available_text', 'unknown_reason', 'native_image', 'negative_size', 'extra_content'])
def test_invalid_unavailable_shapes_stay_raw_and_never_prove_adoption(mutation):
    payload, *_ = attachment_payload()
    view = json.loads(unavailable_body())
    if mutation == 'fake_author': view['attachment_source']['author'] = 'owner'
    elif mutation == 'available_text': view.update(content='must remain', content_available=True)
    elif mutation == 'unknown_reason': view['adoption'] = 'invented'
    elif mutation == 'native_image': view.update(media_available=True, adoption='native_image')
    elif mutation == 'extra_content': view['full_text'] = 'must not disappear'
    else: view['total_bytes'] = -1
    body = json.dumps(view)
    payload['messages'] = [{'role': 'user', 'content': body}]
    payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = []
    manifest = enable(payload, [sha(body)])
    final, ctx = prepare_projection(payload, manifest, 'conversation-1')
    assert final == payload
    assert verified_projection_refs(final, manifest, 'conversation-1', ctx) == []


@pytest.mark.parametrize('mutation', ['changed_text', 'changed_caption', 'moved_text',
    'changed_image', 'missing_image', 'image_in_other_message', 'wrong_conversation', 'wrong_manifest'])
def test_begin_reverifies_final_body_positions_image_and_scope(mutation):
    payload, _ref = image_payload() if 'image' in mutation or mutation == 'changed_caption' else (attachment_payload()[0], None)
    manifest = enable(payload)
    final, ctx = prepare_projection(payload, manifest, 'conversation-1')
    trace = deepcopy(payload['_ombre_trace_context'])
    if mutation == 'changed_text': final['messages'][0]['content'] += 'tampered'
    elif mutation == 'changed_caption': final['messages'][0]['content'][0]['text'] += 'tampered'
    elif mutation == 'moved_text': final['messages'].insert(0, {'role': 'user', 'content': 'inserted'})
    elif mutation == 'changed_image':
        final['messages'][0]['content'][1]['image_url']['url'] = 'data:image/png;base64,' + base64.b64encode(PNG + b'changed').decode()
    elif mutation == 'missing_image': final['messages'][0]['content'].pop()
    elif mutation == 'image_in_other_message':
        image = final['messages'][0]['content'].pop()
        final['messages'].append({'role': 'user', 'content': [image]})
    elif mutation == 'wrong_conversation': trace['conversation_id'] = 'other'
    else: trace['attachment_adoption']['profile_id'] = 'other-profile'
    attempt = AdoptionReceipts().begin(trace, final, projection_bindings=ctx)
    assert attempt['attachments'] == []


def test_duplicate_image_descriptor_without_same_message_image_stays_original():
    payload, _ref = image_payload()
    descriptor = payload['messages'][0]['content'][0]['text']
    payload['messages'].append({'role': 'user', 'content': descriptor})
    manifest = enable(payload)
    final, ctx = prepare_projection(payload, manifest, 'conversation-1')
    assert final['messages'][1]['content'] == descriptor
    assert len(verified_projection_refs(final, manifest, 'conversation-1', ctx)) == 1


def test_alias_collision_does_not_compact(monkeypatch):
    payload, *_ = combined_payload()
    monkeypatch.setattr('attachment_adoption.short_ref', lambda *_: 'collision')
    final, ctx = prepare_projection(payload, payload['_ombre_trace_context']['attachment_adoption'], 'conversation-1')
    assert final['messages'][:2] == payload['messages'][:2]
    assert ctx['bindings'] == []


def sse_for(protocol, mode):
    if mode == 'stream_error':
        return b'data: {"type":"error","error":{"message":"failed"}}\n\n'
    if protocol == 'openai':
        reason = 'length' if mode == 'length' else 'stop'
        data = ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'content': 'seen'}, 'finish_reason': reason}]}) + '\n\n').encode()
        return data if mode == 'truncated' else data + b'data: [DONE]\n\n'
    reason = 'max_tokens' if mode == 'length' else 'end_turn'
    data = ('event: message_delta\ndata: ' + json.dumps({'type': 'message_delta', 'delta': {'stop_reason': reason}}) + '\n\n').encode()
    return data if mode == 'truncated' else data + b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def exercise_forward(path, payload, *, mode='success', unknown_capacity=False, tamper=False):
    protocol = 'anthropic' if path.startswith('anthropic') else 'openai'
    streaming = path.endswith('_stream')
    sent, recorded = [], []
    sse = sse_for(protocol, mode)
    class CancelledStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n' if protocol == 'openai' else b'event: message_start\ndata: {"type":"message_start"}\n\n'
            raise asyncio.CancelledError()
    def provider(request):
        sent.append(json.loads(request.content))
        if mode == 'http_failure': return httpx.Response(500, json={'error': 'failed'})
        if mode == 'retry' and len(sent) == 1: return httpx.Response(503, json={'error': 'retryable'})
        if mode == 'cancelled': return httpx.Response(200, stream=CancelledStream(), headers={'Content-Type': 'text/event-stream'})
        if streaming: return httpx.Response(200, content=sse, headers={'Content-Type': 'text/event-stream'})
        if protocol == 'anthropic':
            return httpx.Response(200, json={'content': [{'type': 'text', 'text': 'seen'}],
                'stop_reason': 'max_tokens' if mode == 'length' else 'end_turn'})
        return httpx.Response(200, json=openai_response(finish_reason='length' if mode == 'length' else 'stop'))
    service, route = gateway_service(provider)
    if mode == 'retry':
        service._available_upstream_api_keys = lambda _upstream: [
            {'label': 'first', 'value': 'provider-secret'},
            {'label': 'second', 'value': 'provider-secret-2'},
        ]
    if unknown_capacity: service.gateway_cfg = {}
    service.model_request_trace.record_attempt = lambda **kwargs: recorded.append(deepcopy(kwargs['payload']))
    if tamper:
        original_build = service.http_client.build_request
        def altered_request(*args, **kwargs):
            request = original_build(*args, **kwargs)
            wire = json.loads(request.content)
            content = wire['messages'][0]['content']
            if tamper == 'image':
                image = content[1]
                data = base64.b64encode(PNG + b'changed on serialization').decode()
                if protocol == 'anthropic': image['source']['data'] = data
                else: image['image_url']['url'] = 'data:image/png;base64,' + data
            elif isinstance(content, str): wire['messages'][0]['content'] += ' altered on serialization'
            else: content[0]['text'] += ' altered on serialization'
            headers = dict(request.headers)
            headers.pop('content-length', None)
            return httpx.Request(request.method, request.url, headers=headers, json=wire, extensions=request.extensions)
        service.http_client.build_request = altered_request
    async def run():
        if path == 'openai': response = await service._forward_upstream(payload)
        elif path == 'anthropic': response = await service._forward_anthropic_upstream(payload, route)
        elif path == 'openai_stream': response = await service._open_upstream_stream(route, payload)
        else: response = await service._open_anthropic_upstream_stream(route, payload)
        if streaming and response.status_code == 200:
            assert service.attachment_adoption.lookup(r2_response_lookup_query())['state'] == 'unavailable'
            output = b''
            if mode == 'cancelled':
                with pytest.raises(asyncio.CancelledError):
                    async for chunk in service._iter_upstream_bytes(response, {}, stream_started_at=0, body_started_at=0):
                        output += chunk
                assert output
            else:
                async for chunk in service._iter_upstream_bytes(response, {}, stream_started_at=0, body_started_at=0):
                    output += chunk
                assert output == sse
        receipt = service.attachment_adoption.lookup(r2_response_lookup_query())
        status = response.status_code
        await response.aclose()
        await service.http_client.aclose()
        return status, receipt
    status, receipt = asyncio.run(run())
    return status, receipt, sent, recorded


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('kind', ['text', 'image'])
def test_all_four_real_http_paths_send_compact_view_and_prove_actual_wire(path, kind):
    payload, refs, text = combined_payload()
    selected = 0 if kind == 'text' else 1
    payload['messages'] = [payload['messages'][selected]] + payload['messages'][2:]
    refs = [refs[selected]]
    payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = [refs[0]['body_sha256']]
    original = deepcopy(payload)
    status, receipt, sent, recorded = exercise_forward(path, payload)
    assert status == 200 and receipt['state'] == 'available'
    assert payload['messages'] == original['messages']
    assert len(sent) == 1 and recorded == sent
    assert len(sent[0]['messages']) == len(original['messages'])
    actual_texts = list(text_blocks(sent[0]))
    if kind == 'text': assert text in actual_texts[0]
    assert all('attachment_source' not in t for t in actual_texts)
    assert '主人原文：请阅读附件。' in actual_texts[-1]
    # The existing Anthropic converter drops input OpenAI cache fields. The
    # projector preserves metadata present AFTER that unchanged conversion.
    if kind == 'text' and not path.startswith('anthropic'):
        assert sent[0]['messages'][0]['content'][0]['cache_control'] == {'type': 'ephemeral'}
    assert '_ombre_trace_context' not in sent[0]
    assert len(receipt['attachments']) == 1
    for expected, actual in zip(refs, receipt['attachments']):
        assert {key: actual[key] for key in expected} == expected
        assert actual['projection'] == PROJECTION
        assert actual['provider_body_sha256'] in {sha(t) for t in actual_texts}
    if kind == 'image':
        image = sent[0]['messages'][0]['content'][1]
        encoded = image['source']['data'] if path.startswith('anthropic') else image['image_url']['url'].split(',', 1)[1]
        assert base64.b64decode(encoded) == PNG


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('kind', ['text', 'image'])
def test_proofs_follow_serialized_request_not_pre_serialization_variable(path, kind):
    payload = attachment_payload()[0] if kind == 'text' else image_payload()[0]
    enable(payload)
    status, receipt, sent, recorded = exercise_forward(path, payload, tamper=kind)
    assert status == 200 and receipt['state'] == 'available'
    assert receipt['attachments'] == []
    assert recorded == sent
    if kind == 'text': assert 'altered on serialization' in next(text_blocks(sent[0]))


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('mode', ['http_failure', 'length'])
def test_failures_and_output_limits_do_not_publish_compact_proofs(path, mode):
    payload, *_ = attachment_payload()
    enable(payload)
    _status, receipt, sent, _ = exercise_forward(path, payload, mode=mode)
    assert sent
    assert receipt['state'] == 'unavailable'


@pytest.mark.parametrize('path', ['openai_stream', 'anthropic_stream'])
@pytest.mark.parametrize('mode', ['truncated', 'stream_error', 'cancelled'])
def test_truncated_or_error_stream_does_not_publish_compact_proofs(path, mode):
    payload, *_ = attachment_payload()
    enable(payload)
    _status, receipt, sent, _ = exercise_forward(path, payload, mode=mode)
    assert sent and receipt['state'] == 'unavailable'


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('kind', ['image', 'unavailable', 'text'])
def test_unknown_capacity_policy_is_not_weakened_or_regressed(path, kind):
    if kind == 'image':
        payload, _ref = image_payload()
        enable(payload)
    else:
        payload, *_ = attachment_payload()
        if kind == 'unavailable':
            body = unavailable_body()
            payload['messages'] = [{'role': 'user', 'content': body}]
            payload['_ombre_trace_context']['attachment_adoption']['body_sha256s'] = []
            enable(payload, [sha(body)])
        else: enable(payload)
    status, receipt, sent, _ = exercise_forward(path, payload, unknown_capacity=True)
    if kind == 'text':
        assert status == 400 and sent == [] and receipt['state'] == 'unavailable'
    else:
        assert status == 200 and len(sent) == 1
        assert receipt['state'] == 'available'
        assert len(receipt['attachments']) == (1 if kind == 'image' else 0)


@pytest.mark.parametrize('mutation', ['missing_hash', 'wrong_hash', 'missing_request', 'empty_turn',
    'invalid_conversation', 'invalid_profile', 'wrong_locator_scope', 'unknown_shape',
    'available_text', 'hidden_text', 'mixed_fulltext', 'wrong_role', 'unparsed_version', 'extra_content'])
def test_unavailable_capacity_exception_requires_all_original_hashes_shapes_and_scope(mutation):
    payload, *_ = attachment_payload()
    view = json.loads(unavailable_body())
    trace = payload['_ombre_trace_context']
    if mutation == 'wrong_locator_scope':
        view['attachment_source']['locator'] = {'conversation_id': 'other'}
    elif mutation == 'unknown_shape': view['adoption'] = 'invented'
    elif mutation == 'available_text': view.update(content='real attachment text', content_available=True)
    elif mutation == 'hidden_text': view['content'] = 'real text despite false availability flag'
    elif mutation == 'extra_content': view['full_text'] = 'must not disappear'
    body = json.dumps(view)
    payload['messages'] = [{'role': 'user', 'content': body}]
    trace['attachment_adoption']['body_sha256s'] = []
    manifest = enable(payload, [sha(body)])
    if mutation == 'missing_hash': manifest['projection_only_sha256s'].append('c' * 64)
    elif mutation == 'wrong_hash': manifest['projection_only_sha256s'] = ['c' * 64]
    elif mutation == 'missing_request': trace.pop('request_id')
    elif mutation == 'empty_turn': trace['turn_id'] = ''
    elif mutation == 'invalid_conversation': trace['conversation_id'] = None
    elif mutation == 'invalid_profile': manifest['profile_id'] = ''
    elif mutation == 'mixed_fulltext': payload['messages'].extend(attachment_payload()[0]['messages'])
    elif mutation == 'wrong_role': payload['messages'][0]['role'] = 'assistant'
    elif mutation == 'unparsed_version': manifest['version'] = 99
    assert has_full_attachment(payload, trace)


@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_native_projection_preserves_post_conversion_cache_and_image_metadata(protocol):
    payload, _ref = image_payload(protocol)
    block = payload['messages'][0]['content'][0]
    block['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}
    block['existing_metadata'] = {'keep': True}
    original_image = deepcopy(payload['messages'][0]['content'][1])
    manifest = enable(payload)
    final, _ctx = prepare_projection(payload, manifest, 'conversation-1')
    assert final['messages'][0]['content'][0]['cache_control'] == block['cache_control']
    assert final['messages'][0]['content'][0]['existing_metadata'] == block['existing_metadata']
    assert final['messages'][0]['content'][1] == original_image


@pytest.mark.parametrize('path', PATHS)
def test_existing_mixed_text_plus_image_capacity_gate_is_not_weakened(path):
    payload, *_ = combined_payload()
    status, receipt, sent, _ = exercise_forward(path, payload)
    assert status == 400 and sent == [] and receipt['state'] == 'unavailable'


@pytest.mark.parametrize('path', PATHS)
def test_failed_attempt_retry_reprojects_original_without_duplicate_wrapping(path):
    payload, _body, _text, ref = attachment_payload()
    enable(payload)
    original = deepcopy(payload)
    status, receipt, sent, recorded = exercise_forward(path, payload, mode='retry')
    assert status == 200 and receipt['state'] == 'available'
    assert len(sent) == 2 and sent[0] == sent[1] and recorded == sent
    assert payload['messages'] == original['messages']
    assert len(receipt['attachments']) == 1
    assert receipt['attachments'][0]['body_sha256'] == ref['body_sha256']
    assert receipt['attachments'][0]['provider_body_sha256'] == sha(next(text_blocks(sent[1])))
