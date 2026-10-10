"""Bounded decoding reuse, never an index, eligibility or relevance cache.

Only immutable numeric vectors are cached by their exact serialized value.
Callers must still read current rows and validate source/configuration metadata
on every search. No source IDs, query text or admission decisions live here.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import sys
import threading


class VectorDecodeCache:
    def __init__(self, *, max_entries: int = 2048, max_bytes: int = 64 * 1024 * 1024):
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[bytes, tuple[tuple, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _count(diagnostics, name):
        if diagnostics is not None:
            diagnostics[name] = diagnostics.get(name, 0) + 1

    def decode(self, payload, *, diagnostics=None):
        """Return (decoded value, nonempty finite numeric vector).

        Malformed JSON keeps json.loads' error contract. Invalid vector values
        are returned uncached so callers retain their existing rejection reason.
        """
        if isinstance(payload, str):
            encoded = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray)):
            encoded = bytes(payload)
        else:
            raise TypeError("stored vector must be serialized JSON")
        key = hashlib.sha256(encoded).digest()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._count(diagnostics, "hits")
                return cached[0], True
        self._count(diagnostics, "misses")
        value = json.loads(payload)
        valid = (isinstance(value, list) and bool(value)
                 and all(type(item) in (int, float) and math.isfinite(item)
                         for item in value))
        if not valid:
            self._count(diagnostics, "invalid")
            return value, False
        vector = tuple(value)
        # Include values, key and a conservative per-entry mapping overhead.
        size = (sys.getsizeof(vector) + sum(sys.getsizeof(item) for item in vector)
                + sys.getsizeof(key) + 256)
        if not self.max_entries or size > self.max_bytes:
            self._count(diagnostics, "bypassed")
            return vector, True
        with self._lock:
            # Another caller may have decoded the same payload meanwhile.
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached[0], True
            while self._entries and (len(self._entries) >= self.max_entries
                                     or self._bytes + size > self.max_bytes):
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._bytes -= evicted_size
                self._count(diagnostics, "evictions")
            self._entries[key] = (vector, size)
            self._bytes += size
        return vector, True

    def snapshot(self):
        with self._lock:
            return {"entries": len(self._entries), "accounted_bytes": self._bytes,
                    "max_entries": self.max_entries, "max_bytes": self.max_bytes}

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
