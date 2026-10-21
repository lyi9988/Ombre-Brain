"""Verbatim parse_header from Ombre-Brain c7d5efdbe9ab4f4c9ea55194a60caec77d35ab33.

Compatibility fixture, NOT production code. Keep original implementation intact.
"""
import base64
import json

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
