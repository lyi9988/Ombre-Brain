"""Authority-recorded time crosses the canonical reader without event guesses."""
import asyncio
import json

import pytest

from memory_l1_source_readers import CanonicalMemorySourceReader, canonical_recorded_at


@pytest.mark.parametrize(('source', 'expected'), [
    ({'created_at_ms': 0}, '1970-01-01T00:00:00+00:00'),
    ({'created_at_ms': 1001}, '1970-01-01T00:00:01.001000+00:00'),
    ({'recorded_at':'2026-10-18T08:00:00+08:00', 'created_at_ms':0}, '2026-10-18T08:00:00+08:00'),
    ({}, None), ({'created_at_ms':True}, None), ({'created_at_ms':'1001'}, None),
    ({'created_at_ms':float('nan')}, None), ({'created_at_ms':float('inf')}, None),
])
def test_canonical_record_time_uses_authority_values_only(source, expected):
    assert canonical_recorded_at(source) == expected


def test_canonical_reader_maps_bridge_created_at_ms_into_verified_source(monkeypatch):
    locator = {'system':'aiz', 'profile_id':'owner', 'conversation_id':'conversation',
               'event_id':'event', 'role':'user'}
    binding = {'source_id':json.dumps(locator)}
    reader = CanonicalMemorySourceReader({'gateway':{'canonical_continuation':{
        'enabled':True, 'base_url':'https://synthetic.invalid', 'conversation_id':'conversation'}}})
    reader.upstream_version = lambda value: 'version-exact'
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'synthetic-test-token')

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'source':{'state':'available', 'eligible_source':True,
                'source_id':'event', 'source_revision':'version-exact',
                'text':'这只是表达愿望的记录，不代表实现时间。', 'created_at_ms':1001}}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, **kwargs): return Response()

    monkeypatch.setattr('memory_l1_source_readers.httpx.AsyncClient', Client)
    result = asyncio.run(reader.read(binding))
    assert result['recorded_at'] == '1970-01-01T00:00:01.001000+00:00'
    assert result['upstream_revision'] == 'version-exact'
    assert result['text'] == '这只是表达愿望的记录，不代表实现时间。'
