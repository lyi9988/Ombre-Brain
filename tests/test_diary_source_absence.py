"""Only an authenticated current-source 404 is a normal empty diary day."""
import asyncio
from types import SimpleNamespace as NS

import httpx
import pytest

from memory_l0_sources import MemoryL0Sources
from memory_l1_runtime import MemoryL1Runtime
from memory_l1_source_readers import AizDiarySourceReader


DAY = '2026-11-05'
SOURCE_ID = 'diary:private:' + DAY


def reader_with_response(monkeypatch, status, payload=None, *, text=None):
    original_client = httpx.AsyncClient
    monkeypatch.setenv('OMBRE_CANONICAL_BRIDGE_TOKEN', 'synthetic-bridge-token')

    def send(request):
        assert request.url.host == 'authority.invalid'
        assert request.url.path == '/bridge/v1/memory-source'
        assert request.headers['authorization'] == 'Bearer synthetic-bridge-token'
        assert request.headers['x-guyan-device-id'] == 'synthetic-device'
        return (httpx.Response(status, text=text) if text is not None
                else httpx.Response(status, json=payload))

    def client(**kwargs):
        assert kwargs['follow_redirects'] is False
        assert kwargs['trust_env'] is False
        return original_client(transport=httpx.MockTransport(send), **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', client)
    return AizDiarySourceReader({'gateway': {'canonical_continuation': {
        'enabled': True, 'base_url': 'https://authority.invalid', 'device_id': 'synthetic-device',
    }}}, 'private')


def run_task(reader):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('empty/unreadable days must not write or start extraction')

    sources = MemoryL0Sources(NS(_connect=forbidden), [reader], diary_namespace=reader.namespace)
    engine = NS(config={}, _memory_l1_prompt=lambda _: ('synthetic prompt', {}),
                _memory_task_route=lambda *_: {'configured': True},
                _memory_l1_sources=lambda: sources, _memory_l1_revisions=forbidden)
    return asyncio.run(MemoryL1Runtime._memory_l1_diary_task_locked(engine, DAY, None))


def test_explicit_current_source_404_skips_without_extraction_or_persistence(monkeypatch):
    reader = reader_with_response(monkeypatch, 404, {'error': {'code': 'source_not_found'}})
    assert run_task(reader) == {'status': 'skipped', 'reason': 'no_diary'}


@pytest.mark.parametrize('status,payload,text', [
    (401, {'error': {'code': 'source_not_found'}}, None),
    (503, {'error': {'code': 'diary_unavailable'}}, None),
    (404, {'error': {'code': 'route_not_found'}}, None),
    (404, {'detail': 'Not Found'}, None),
    (404, None, '404: Not Found'),
    (404, None, None),
    (307, None, None),
])
def test_auth_service_route_and_invalid_responses_stay_failures(monkeypatch, status, payload, text):
    reader = reader_with_response(monkeypatch, status, payload, text=text)
    with pytest.raises(httpx.HTTPStatusError) as failure:
        run_task(reader)
    assert failure.value.response.status_code == status


def test_missing_pinned_revision_is_not_an_empty_day(monkeypatch):
    reader = reader_with_response(monkeypatch, 404, {'error': {'code': 'source_not_found'}})
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(reader.read({'source_id': SOURCE_ID, 'source_revision': 'retained-old-revision'}))


@pytest.mark.parametrize('state,reason', [
    ('unavailable', 'diary_legacy_revision_unavailable'),
    ('unavailable', 'diary_active_revision_unavailable'),
    ('unknown', 'legacy_unknown'),
    ('not_found', 'source_not_found'),
])
def test_unreadable_original_or_unconfirmed_absence_does_not_skip(monkeypatch, state, reason):
    reader = reader_with_response(monkeypatch, 200, {'source': {
        'state': state, 'reason': reason, 'eligible_source': False,
        'source_namespace': 'snapshot.aiz.diary.private', 'source_id': SOURCE_ID,
        'source_revision': '',
    }})
    assert run_task(reader) == {'status': 'not_prepared', 'reason': 'source_unavailable'}


def test_existing_current_diary_remains_available(monkeypatch):
    source = {'state': 'available', 'eligible_source': True,
              'source_namespace': 'snapshot.aiz.diary.private', 'source_id': SOURCE_ID,
              'source_revision': 'formal-revision', 'text': '合成正式日记。'}
    reader = reader_with_response(monkeypatch, 200, {'source': source})
    captured = asyncio.run(reader.capture(DAY))
    assert captured['state'] == 'available'
    assert captured['source_revision'] == 'formal-revision'
    assert captured['text'] == source['text']
