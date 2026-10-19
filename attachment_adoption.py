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

HEADER = 'X-Guyan-Attachment-Adoption'
SCOPE = ('request_id', 'turn_id', 'conversation_id', 'profile_id')
LOCATOR = {'owner_id','conversation_id','event_id','event_version_id','resource_id','raw_sha256','text_sha256','text_view_id'}

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
        if not isinstance(value.get('body_sha256s'), list) or not 0 < len(value['body_sha256s']) <= 64:
            return {}
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in value['body_sha256s']):
            return {}
        if not isinstance(value.get('profile_id'), str) or not 0 < len(value['profile_id']) <= 200:
            return {}
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

def verified_refs(payload, manifest, conversation_id):
    """Match a complete final wire text block, not a substring or claimed flag."""
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

    def begin(self, trace, payload):
        manifest = trace.get('attachment_adoption') or {}
        scope = {k: trace.get(k) for k in SCOPE[:3]}
        scope['profile_id'] = manifest.get('profile_id')
        if not all(isinstance(v, str) and 0 < len(v) <= 200 for v in scope.values()):
            return None
        refs = verified_refs(payload, manifest, scope['conversation_id'])
        return {'attempt_id': 'adopt-' + uuid.uuid4().hex, **scope, 'attachments': refs}

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

    def lookup(self, query):
        self._prune()
        if not isinstance(query, dict) or not all(isinstance(query.get(k), str) and query[k] for k in (*SCOPE, 'tool_call_id')):
            return {'state': 'unavailable', 'reason': 'invalid_scope'}
        row = self.rows.get(tuple(query[k] for k in SCOPE) + (query['tool_call_id'],))
        return deepcopy(row[1]) if row else {'state': 'unavailable', 'reason': 'no_completed_attempt_receipt'}

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
                        self.finished = obj.get('delta', {}).get('stop_reason') == 'tool_use'
                    elif kind == 'message_stop':
                        self.done = True
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
                if len(self.ids) > 128:
                    self.invalid = True
                    return
            if len(self.buffer) > 1024 * 1024:
                self.invalid = True
                self.buffer = ''
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
            self.invalid = True
            self.buffer = ''
