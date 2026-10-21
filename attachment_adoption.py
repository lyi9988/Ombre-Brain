"""Ephemeral, body-free evidence of exact attachment blocks in a completed attempt.

Not a memory writer, document authority, or proof of semantic understanding.
SDK callers still revalidate canonical attachment/source revisions before writing.
"""
import codecs
import base64
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from attachment_prompt_projection import PROJECTION, render_body, short_ref

HEADER = 'X-Guyan-Attachment-Adoption'
SCOPE = ('request_id', 'turn_id', 'conversation_id', 'profile_id')
LOCATOR = {'owner_id','conversation_id','event_id','event_version_id','resource_id','raw_sha256','text_sha256','text_view_id'}
IMAGE_LOCATOR = {'owner_id','conversation_id','event_id','event_version_id','resource_id','raw_sha256','media_view_id'}
IMAGE_NAMESPACE = 'snapshot.aiz.attachment_image'
IMAGE_VIEW = 'attachment.image.original-v1'

def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def parse_header(raw):
    # Bounds limit ephemeral evidence bookkeeping, never the underlying chat.
    try:
        if not isinstance(raw, str) or len(raw.encode('utf-8')) > 65536:
            return {}
        decoded = base64.b64decode(raw + '=' * (-len(raw) % 4), altchars=b'-_', validate=True)
        value = json.loads(decoded.decode('utf-8'))
        if not isinstance(value, dict) or value.get('version') != 1:
            return {}
        if 'projection' in value and value['projection'] != PROJECTION:
            return {}
        compact = value.get('projection') == PROJECTION
        wanted = value.get('body_sha256s')
        only = value.get('projection_only_sha256s', []) if compact else []
        if not isinstance(wanted, list) or len(wanted) > 64:
            return {}
        if compact and (len(raw.encode('utf-8')) > 8192 or not isinstance(only, list) or len(only) > 64):
            return {}
        if not wanted and not (compact and only):
            return {}
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in wanted + only):
            return {}
        if compact and (len(set(wanted + only)) > 64 or set(wanted).intersection(only)):
            return {}
        if not isinstance(value.get('profile_id'), str) or not 0 < len(value['profile_id']) <= 200:
            return {}
        if compact:
            canonical(value)  # bindings require finite, deterministic JSON
        return value
    except (ValueError, TypeError, UnicodeError):
        return {}

def text_blocks(payload):
    for message in payload.get('messages', []):
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str):
                    yield block['text']

def _inline_image_hashes(message):
    """Original inline bytes in the final provider message; never fetch URLs."""
    result = set()
    content = message.get('content')
    if not isinstance(content, list):
        return result
    for block in content:
        if not isinstance(block, dict):
            continue
        mime, encoded = None, None
        if block.get('type') == 'image_url':
            image = block.get('image_url')
            url = image.get('url') if isinstance(image, dict) else image
            if isinstance(url, str) and url.startswith('data:') and ';base64,' in url:
                mime, encoded = url[5:].split(';base64,', 1)
        elif block.get('type') == 'image':
            source = block.get('source')
            if isinstance(source, dict) and source.get('type') == 'base64':
                mime, encoded = source.get('media_type'), source.get('data')
        if mime not in {'image/png', 'image/jpeg'} or not isinstance(encoded, str):
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeError):
            continue
        signature_matches = (raw.startswith(b'\x89PNG\r\n\x1a\n') if mime == 'image/png'
                             else raw.startswith(b'\xff\xd8\xff'))
        if raw and signature_matches:
            result.add((mime, hashlib.sha256(raw).hexdigest(), len(raw)))
    return result

def _verified_image_refs(payload, wanted, conversation_id):
    found = []
    for message in payload.get('messages') or []:
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        images = _inline_image_hashes(message)
        if not images:
            continue
        for text in text_blocks({'messages': [message]}):
            block_hash = sha(text)
            if block_hash not in wanted:
                continue
            try:
                wrapper = json.loads(text)
                desc = wrapper['attachment_source']
                locator = json.loads(desc['source_id'])
                if not isinstance(locator, dict) or set(locator) != IMAGE_LOCATOR:
                    continue
                if any(not isinstance(v, str) or not v or len(v) > 200 for v in locator.values()):
                    continue
                if (canonical(locator) != desc['source_id'] or desc['locator'] != locator
                        or locator['conversation_id'] != conversation_id
                        or locator['media_view_id'] != IMAGE_VIEW
                        or desc.get('source_namespace') != IMAGE_NAMESPACE
                        or desc.get('source_kind') != 'image'):
                    continue
                raw_hash = locator['raw_sha256']
                if (len(raw_hash) != 64 or any(c not in '0123456789abcdef' for c in raw_hash)
                        or desc.get('sha256') != raw_hash
                        or desc['source_revision'] != 'attachment_sha256:' + sha(desc['source_id'])):
                    continue
                if (wrapper.get('adoption') != 'native_image' or wrapper.get('media_available') is not True
                        or wrapper.get('content_available') is not False or wrapper.get('content') is not None):
                    continue
                size = wrapper.get('adopted_bytes')
                if (type(size) is not int or wrapper.get('total_bytes') != size
                        or (desc.get('mime'), raw_hash, size) not in images):
                    continue
                if any(desc.get(k) != locator[k] for k in ('event_id', 'event_version_id', 'resource_id')):
                    continue
                ref = {'source_id': desc['source_id'], 'source_revision': desc['source_revision'],
                       'body_sha256': block_hash, 'source_kind': 'image'}
                if ref not in found:
                    found.append(ref)
            except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
                continue
    return found

def verified_refs(payload, manifest, conversation_id):
    """Match complete text or same-message original image bytes on final wire."""
    wanted = set(manifest.get('body_sha256s') or [])
    found = []
    for text in text_blocks(payload):
        block_hash = sha(text)
        if block_hash not in wanted:
            continue
        try:
            wrapper = json.loads(text)
            desc = wrapper['attachment_source']
            locator = json.loads(desc['source_id'])
            if not isinstance(locator, dict) or set(locator) != LOCATOR:
                continue
            if any(not isinstance(v, str) or not v or len(v) > 200 for v in locator.values()):
                continue
            if canonical(locator) != desc['source_id'] or locator['conversation_id'] != conversation_id:
                continue
            if locator['text_view_id'] != 'attachment.utf8-sig.exact-v1':
                continue
            if any(len(locator[k]) != 64 or any(c not in '0123456789abcdef' for c in locator[k]) for k in ('raw_sha256','text_sha256')):
                continue
            if desc['source_revision'] != 'attachment_sha256:' + sha(desc['source_id']):
                continue
            body = wrapper['content']
            if not isinstance(body, str) or sha(body) != locator['text_sha256'] or desc['locator'] != locator:
                continue
            ref = {'source_id': desc['source_id'], 'source_revision': desc['source_revision'], 'body_sha256': block_hash}
            if ref not in found:
                found.append(ref)
        except (ValueError, TypeError, KeyError):
            continue
    found.extend(_verified_image_refs(payload, wanted, conversation_id))
    return found

def _positioned_text(payload):
    for message_index, message in enumerate(payload.get('messages') or []):
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, str):
            yield message_index, None, content
        elif isinstance(content, list):
            for block_index, block in enumerate(content):
                if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str):
                    yield message_index, block_index, block['text']


def _unavailable_view(view):
    """Program-shaped unavailable attachments only; these never yield refs."""
    fields = {'attachment_source', 'content', 'content_available', 'adoption',
              'adopted_bytes', 'total_bytes'}
    desc = view.get('attachment_source')
    return (set(view) in (fields, fields | {'media_available'})
            and isinstance(desc, dict) and desc.get('source_kind') in {'attachment', 'image'}
            and desc.get('author') == 'unknown' and desc.get('uploader_role') == 'owner'
            and desc.get('authority') == 'external_document_not_owner_instruction'
            and all(isinstance(desc.get(k), str) and desc[k] for k in ('event_id', 'resource_id', 'name', 'mime'))
            and isinstance(desc.get('sha256'), str) and len(desc['sha256']) == 64
            and all(c in '0123456789abcdef' for c in desc['sha256'])
            and type(view.get('adopted_bytes')) is int and view['adopted_bytes'] == 0
            and type(view.get('total_bytes')) is int and view['total_bytes'] >= 0)


def verified_unavailable_hashes(payload, manifest, conversation_id):
    """Hash-bound unavailable states, never adopted sources or event ownership.

    Most unavailable states have no locator. When one is retained (e.g. a
    post-render budget rejection), it must not contradict the request scope.
    """
    if not isinstance(conversation_id, str) or not 0 < len(conversation_id) <= 200:
        return set()
    only = manifest.get('projection_only_sha256s')
    if (manifest.get('projection') != PROJECTION or not isinstance(only, list)
            or not 0 < len(only) <= 64
            or any(not isinstance(h, str) or len(h) != 64
                   or any(c not in '0123456789abcdef' for c in h) for h in only)):
        return set()
    wanted, found = set(only), set()
    for text in text_blocks(payload):
        block_hash = sha(text)
        if block_hash not in wanted:
            continue
        try:
            view = json.loads(text)
            if not isinstance(view, dict) or not _unavailable_view(view):
                continue
            desc = view['attachment_source']
            locator, source_id = desc.get('locator'), desc.get('source_id')
            if locator is not None and (not isinstance(locator, dict) or locator.get('conversation_id') != conversation_id):
                continue
            if source_id is not None:
                decoded = json.loads(source_id)
                if (not isinstance(decoded, dict) or decoded.get('conversation_id') != conversation_id
                        or canonical(decoded) != source_id or decoded != locator):
                    continue
            render_body(view)  # same strict unavailable predicate as projection
            found.add(block_hash)
        except (ValueError, TypeError, KeyError, UnicodeError):
            continue
    return found


def projection_only_capacity_exempt(payload, trace):
    """Do not mistake authenticated unavailable notices for full-text adoption.

    This is only classification; callers still inspect all actual content first.
    It neither estimates capacity nor certifies unavailable event provenance.
    """
    manifest = trace.get('attachment_adoption') or {}
    scope = [trace.get(k) for k in SCOPE[:3]] + [manifest.get('profile_id')]
    if (manifest.get('version') != 1 or manifest.get('body_sha256s') != []
            or not all(isinstance(v, str) and 0 < len(v) <= 200 for v in scope)):
        return False
    matched = verified_unavailable_hashes(payload, manifest, trace['conversation_id'])
    return bool(matched) and matched == set(manifest.get('projection_only_sha256s') or [])


def prepare_projection(payload, manifest, conversation_id):
    """Project only authenticated originals after the conservative capacity gate.

    The original payload/manifest are never mutated. Bindings are ephemeral server
    data, NOT headers or model input. body_sha256 retains source-view identity;
    provider_body_sha256 binds the exact final visible text (not the HTTP body).
    """
    if manifest.get('projection') != PROJECTION:
        return payload, None
    original_refs = verified_refs(payload, manifest, conversation_id)
    by_hash = {ref['body_sha256']: ref for ref in original_refs}
    aliases, ambiguous = {}, set()
    for ref in original_refs:
        alias = short_ref(ref['source_id'], ref['source_revision'])
        identity = (ref['source_id'], ref['source_revision'])
        if alias in aliases and aliases[alias] != identity:
            ambiguous.add(alias)
        aliases[alias] = identity
    only = verified_unavailable_hashes(payload, manifest, conversation_id)
    result = deepcopy(payload)
    bindings = []
    for message_index, block_index, text in _positioned_text(payload):
        block_hash = sha(text)
        ref = by_hash.get(block_hash)
        if ref is None and block_hash not in only:
            continue
        try:
            view = json.loads(text)
            if not isinstance(view, dict):
                continue
            if ref is None:
                if not _unavailable_view(view):
                    continue
            elif short_ref(ref['source_id'], ref['source_revision']) in ambiguous:
                continue
            if ref is not None and ref.get('source_kind') == 'image':
                desc = view['attachment_source']
                if (desc['mime'], desc['sha256'], view['adopted_bytes']) not in _inline_image_hashes(payload['messages'][message_index]):
                    continue
            rendered = render_body(view, ref)
        except (ValueError, TypeError, KeyError, UnicodeError):
            continue
        target = result['messages'][message_index]
        if block_index is None:
            target['content'] = rendered
        else:
            target['content'][block_index]['text'] = rendered
        if ref is None:
            continue
        binding = {'message_index': message_index, 'block_index': block_index,
                   'ref': {**ref, 'projection': PROJECTION, 'provider_body_sha256': sha(rendered)}}
        if ref.get('source_kind') == 'image':
            desc = view['attachment_source']
            binding['image'] = (desc['mime'], desc['sha256'], view['adopted_bytes'])
        bindings.append(binding)
    return result, {'conversation_id': conversation_id,
                    'manifest_sha256': sha(canonical(manifest)), 'bindings': bindings}


def verified_projection_refs(payload, manifest, conversation_id, context):
    """Recheck actual final provider text, positions, scope and same-message image.

    A prepared binding alone is not evidence. Changed, moved, missing or malformed
    final blocks yield no compact ref. Unconverted v1 refs are verified separately.
    """
    if (manifest.get('projection') != PROJECTION or not isinstance(context, dict)
            or context.get('conversation_id') != conversation_id
            or context.get('manifest_sha256') != sha(canonical(manifest))):
        return []
    actual = {(mi, bi): text for mi, bi, text in _positioned_text(payload)}
    wanted = set(manifest.get('body_sha256s') or [])
    found = []
    for binding in context.get('bindings') or []:
        try:
            ref = binding['ref']
            locator = json.loads(ref['source_id'])
            if (ref.get('projection') != PROJECTION or ref.get('body_sha256') not in wanted
                    or locator['conversation_id'] != conversation_id
                    or ref['source_revision'] != 'attachment_sha256:' + sha(ref['source_id'])):
                continue
            mi, bi = binding['message_index'], binding['block_index']
            text = actual.get((mi, bi))
            if not isinstance(text, str) or sha(text) != ref['provider_body_sha256']:
                continue
            if ref.get('source_kind') == 'image':
                image = binding.get('image')
                if (not isinstance(image, tuple) or len(image) != 3
                        or image[1] != locator['raw_sha256']
                        or image not in _inline_image_hashes(payload['messages'][mi])):
                    continue
            if ref not in found:
                found.append(deepcopy(ref))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, UnicodeError):
            continue
    return found


class AdoptionReceipts:
    def __init__(self, *, ttl=900, capacity=1024, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self.rows = OrderedDict()

    def _prune(self):
        now = self.clock()
        for key in list(self.rows):
            if self.rows[key][0] <= now:
                del self.rows[key]
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def begin(self, trace, payload, *, projection_bindings=None):
        manifest = trace.get('attachment_adoption') or {}
        scope = {k: trace.get(k) for k in SCOPE[:3]}
        scope['profile_id'] = manifest.get('profile_id')
        if not all(isinstance(v, str) and 0 < len(v) <= 200 for v in scope.values()):
            return None
        refs = verified_refs(payload, manifest, scope['conversation_id'])
        refs.extend(verified_projection_refs(payload, manifest, scope['conversation_id'], projection_bindings))
        attempt = {'attempt_id': 'adopt-' + uuid.uuid4().hex, **scope, 'attachments': refs}
        logical = trace.get('logical_request_id')
        if isinstance(logical, str) and 0 < len(logical) <= 200:
            attempt['logical_request_id'] = logical
        return attempt

    def complete(self, attempt, ids):
        if not attempt:
            return
        self._prune()
        for call_id in set(ids):
            if not isinstance(call_id, str) or not 0 < len(call_id) <= 200:
                continue
            key = tuple(attempt[k] for k in SCOPE) + (call_id,)
            row = {**attempt, 'tool_call_id': call_id, 'state': 'available'}
            prior = self.rows.get(key)
            if prior and prior[1].get('attempt_id') != attempt['attempt_id']:
                row = {'state': 'unknown', 'reason': 'ambiguous_physical_attempt'}
            self.rows[key] = (self.clock() + self.ttl, row)
        self._prune()

    def complete_response(self, attempt):
        """Only a unique successfully terminated physical attempt is eligible."""
        if not attempt or not attempt.get('logical_request_id'):
            return
        self._prune()
        key = ('response',) + tuple(attempt[k] for k in SCOPE) + (attempt['logical_request_id'],)
        row = {**attempt, 'lookup_type': 'response', 'state': 'available'}
        prior = self.rows.get(key)
        if prior and prior[1].get('attempt_id') != attempt['attempt_id']:
            row = {k: attempt[k] for k in (*SCOPE, 'logical_request_id')}
            row.update(lookup_type='response', state='unknown', reason='ambiguous_physical_attempt')
        self.rows[key] = (self.clock() + self.ttl, row)
        self._prune()

    def lookup(self, query):
        self._prune()
        if isinstance(query, dict) and query.get('lookup_type') == 'response':
            if not all(isinstance(query.get(k), str) and 0 < len(query[k]) <= 200 for k in (*SCOPE, 'logical_request_id')):
                return {'state': 'unavailable', 'reason': 'invalid_scope'}
            key = ('response',) + tuple(query[k] for k in SCOPE) + (query['logical_request_id'],)
            row = self.rows.get(key)
            absent = {k: query[k] for k in (*SCOPE, 'logical_request_id')}
            return deepcopy(row[1]) if row else {**absent, 'lookup_type': 'response', 'state': 'unavailable', 'reason': 'no_completed_attempt_receipt'}
        if not isinstance(query, dict) or not all(isinstance(query.get(k), str) and query[k] for k in (*SCOPE, 'tool_call_id')):
            return {'state': 'unavailable', 'reason': 'invalid_scope'}
        row = self.rows.get(tuple(query[k] for k in SCOPE) + (query['tool_call_id'],))
        return deepcopy(row[1]) if row else {'state': 'unavailable', 'reason': 'no_completed_attempt_receipt'}

    def block_dispatch(self, trace, reason):
        """Keep prior physical evidence, revoke adoption for this failed logical.

        Never broaden a missing/malformed profile, turn or logical scope into
        revocation of another request. This only affects ephemeral receipts.
        """
        manifest = trace.get('attachment_adoption') or {}
        scope = {k: trace.get(k) for k in SCOPE[:3]}
        scope['profile_id'] = manifest.get('profile_id')
        logical = trace.get('logical_request_id')
        if (not all(isinstance(v, str) and 0 < len(v) <= 200 for v in scope.values())
                or not isinstance(logical, str) or not 0 < len(logical) <= 200):
            return
        self._prune()
        for key, (expires, row) in list(self.rows.items()):
            if row.get('logical_request_id') != logical or any(row.get(k) != v for k, v in scope.items()):
                continue
            self.rows[key] = (expires, {
                **{k: row[k] for k in (*SCOPE, 'logical_request_id', 'lookup_type', 'tool_call_id') if k in row},
                'state': 'unavailable', 'reason': 'logical_dispatch_blocked',
                'capacity_error_code': reason,
                'prior_physical_attempt_id': row.get('attempt_id') or row.get('prior_physical_attempt_id'),
                'prior_physical_state': row.get('prior_physical_state') or row.get('state'),
            })

    def response(self, attempt, response, protocol):
        if not attempt or not 200 <= response.status_code < 300:
            return
        try:
            obj = response.json()
            if protocol == 'anthropic':
                ids = [b['id'] for b in obj.get('content', []) if b.get('type') == 'tool_use'] if obj.get('stop_reason') == 'tool_use' else []
            else:
                choices = obj.get('choices') or []
                first = next((c for c in choices if c.get('index', 0) == 0), {})
                ids = [b['id'] for b in first.get('message', {}).get('tool_calls', [])] if first.get('finish_reason') == 'tool_calls' else []
            terminal = obj.get('stop_reason') in {'end_turn', 'tool_use', 'stop_sequence'} if protocol == 'anthropic' else first.get('finish_reason') in {'stop', 'tool_calls'}
            if terminal:
                self.complete_response(attempt)
            self.complete(attempt, ids)
        except (ValueError, TypeError, KeyError, AttributeError):
            return

    def wrap(self, attempt, response, protocol):
        if attempt and 200 <= response.status_code < 300:
            response.extensions['attachment_adoption_observer'] = ReceiptObserver(self, attempt, protocol)
        return response

class ReceiptObserver:
    """Observe raw SSE without buffering/replacing user-visible output."""
    def __init__(self, store, attempt, protocol):
        self.store, self.attempt, self.protocol = store, attempt, protocol
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.buffer, self.ids, self.finished, self.done, self.invalid = '', {}, False, False, False
        self.response_finished = False

    def consume(self, chunk):
        if self.invalid or self.done:
            return
        try:
            self.buffer += self.decoder.decode(chunk)
            self.buffer = self.buffer.replace('\r\n', '\n')
            while '\n\n' in self.buffer:
                event, self.buffer = self.buffer.split('\n\n', 1)
                data = '\n'.join(line[5:].lstrip() for line in event.split('\n') if line.startswith('data:'))
                if not data:
                    continue
                if self.protocol == 'openai' and data == '[DONE]':
                    self.done = True
                    if self.response_finished:
                        self.store.complete_response(self.attempt)
                    if self.finished:
                        self.store.complete(self.attempt, self.ids.values())
                    return
                obj = json.loads(data)
                if 'error' in obj or obj.get('type') == 'error':
                    self.invalid = True
                    return
                if self.protocol == 'anthropic':
                    kind = obj.get('type')
                    if kind == 'content_block_start' and obj.get('content_block', {}).get('type') == 'tool_use':
                        self.ids[obj['index']] = obj['content_block']['id']
                    elif kind == 'message_delta':
                        reason = obj.get('delta', {}).get('stop_reason')
                        self.finished = reason == 'tool_use'
                        self.response_finished = reason in {'end_turn', 'tool_use', 'stop_sequence'}
                    elif kind == 'message_stop':
                        self.done = True
                        if self.response_finished:
                            self.store.complete_response(self.attempt)
                        if self.finished:
                            self.store.complete(self.attempt, self.ids.values())
                else:
                    for choice in obj.get('choices', []):
                        if choice.get('index', 0) != 0:
                            continue
                        for call in choice.get('delta', {}).get('tool_calls', []):
                            if 'id' in call:
                                index = call.get('index', 0)
                                self.ids[index] = self.ids.get(index, '') + call['id']
                        if choice.get('finish_reason') is not None:
                            self.finished = choice['finish_reason'] == 'tool_calls'
                            self.response_finished = choice['finish_reason'] in {'stop', 'tool_calls'}
                if len(self.ids) > 128:
                    self.invalid = True
                    return
            if len(self.buffer) > 1024 * 1024:
                self.invalid = True
                self.buffer = ''
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
            self.invalid = True
            self.buffer = ''
