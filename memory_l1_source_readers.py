"""Read actual configured authorities; retained source copies never self-authorize."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import httpx

from memory_l1_ingestion import source_snapshot
from memory_l1_service import L1OwnerError


def attributed_snapshot(actual, binding, config):
    snapshot = source_snapshot(actual, config=config)
    # Old bindings did not carry participant identity. Keep their exact old
    # source revision readable, but do not retroactively promote unknown
    # authorship to today's configured actor. New bindings hash actor id/kind.
    if snapshot['source_revision'] != binding.get('source_revision') and not binding.get('author_ref') and not binding.get('speaker_ref'):
        legacy = source_snapshot(actual)
        if legacy['source_revision'] == binding.get('source_revision'):
            return legacy
    return snapshot


def canonical_recorded_at(source):
    """Keep the authority's record time; never substitute event or local time."""
    recorded = source.get('recorded_at')
    if isinstance(recorded, str) and recorded.strip():
        return recorded
    milliseconds = source.get('created_at_ms')
    if type(milliseconds) not in (int, float):
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


class LocalMemorySourceReader:
    def __init__(self, config, kind):
        self.config, self.kind = config, kind
        self.namespace = 'snapshot.ombre.' + kind

    async def read(self, binding):
        locator = json.loads(binding['source_id'])
        if self.kind == 'raw_event':
            from raw_events import RawEventStore
            # Use only an already initialized, configured archive. A source
            # GET must not create an empty authority and claim source absence.
            state = self.config.get('state_dir') or str(Path(self.config.get('buckets_dir', 'buckets')).resolve().parent / 'state')
            path = (self.config.get('raw_events') or {}).get('db_path') or str(Path(state) / 'raw_events.sqlite')
            if not Path(path).is_file(): return {'state': 'unavailable', 'reason': 'archive_unavailable'}
            with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
                db.row_factory = sqlite3.Row
                row = db.execute('SELECT * FROM raw_events WHERE id=?', (locator['event_id'],)).fetchone()
                if row is None: return {'state': 'unavailable', 'reason': 'source_not_found'}
                event = RawEventStore._row_to_event(row)
            from memory_source_provenance import raw_event_source
            actual = raw_event_source(event, profile_id=locator['profile_id'])
            # An exact canonical bridge is a different namespace. Do not use
            # an archive-only record to claim current canonical availability.
            if actual['namespace'] != 'raw_event': return {'state': 'changed', 'reason': 'source_namespace_changed'}
        else:
            path = str(Path(self.config.get('buckets_dir', 'buckets')) / 'gateway_state.db')
            if not Path(path).is_file(): return {'state': 'unavailable', 'reason': 'turn_archive_unavailable'}
            with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
                db.row_factory = sqlite3.Row
                row = db.execute('SELECT * FROM conversation_turns WHERE id=? AND profile_id=? AND session_id=?',
                    (locator['event_id'], locator['profile_id'], locator['conversation_id'])).fetchone()
                if row is None: return {'state': 'unavailable', 'reason': 'source_not_found'}
                actual = {'namespace': 'conversation_turn', 'system': 'ombre', 'event_id': str(row['id']),
                    'profile_id': row['profile_id'], 'conversation_id': row['session_id'], 'version_id': '',
                    'role': locator['role'], 'text': row[locator['role'] + '_text'], 'recorded_at': row['created_at']}
        snapshot = attributed_snapshot(actual, binding, self.config)
        if snapshot['source_id'] != binding['source_id']:
            return {'state': 'unavailable', 'reason': 'source_identity_mismatch'}
        return snapshot


class CanonicalMemorySourceReader:
    namespace = 'snapshot.aiz.canonical_event'

    def __init__(self, config):
        self.config = config
        self.options = (config.get('gateway') or {}).get('canonical_continuation') or {}

    async def read(self, binding):
        locator = json.loads(binding['source_id'])
        options = self.options
        token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
        if not options.get('enabled') or not token or not options.get('base_url'):
            return {'state': 'unavailable', 'reason': 'canonical_source_reader_unconfigured'}
        if locator['conversation_id'] != options.get('conversation_id', 'conv-jiajia-main'):
            return {'state': 'unavailable', 'reason': 'source_outside_bound_conversation'}
        # The binding stores a digest identity. Upstream exact version is read
        # from its retained authenticated snapshot, supplied by the factory.
        version = self.upstream_version(binding) if hasattr(self, 'upstream_version') else None
        if not version: return {'state': 'unavailable', 'reason': 'upstream_version_unknown'}
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(str(options['base_url']).rstrip('/') + '/bridge/v1/memory-source',
                params={'kind': 'canonical_event', 'source_id': locator['event_id'], 'version': version},
                headers={'Authorization': 'Bearer ' + token,
                         'X-Guyan-Device-ID': str(options.get('device_id') or 'ombre-gateway-operit')})
            response.raise_for_status()
            envelope = response.json()
            result = envelope.get('source', {})
        if result.get('state') != 'available' or result.get('eligible_source') is not True:
            return {'state': result.get('state', 'unavailable'), 'reason': result.get('reason', 'source_not_effective')}
        if result.get('source_id') != locator['event_id'] or result.get('source_revision') != version:
            return {'state': 'unavailable', 'reason': 'source_identity_mismatch'}
        actual = {**locator, 'namespace': 'canonical_event', 'text': result['text'], 'version_id': version,
                  'recorded_at': canonical_recorded_at(result)}
        return attributed_snapshot(actual, binding, self.config)


class AizDiarySourceReader(CanonicalMemorySourceReader):
    def __init__(self, config, kind):
        super().__init__(config)
        self.kind = kind
        self.namespace = 'snapshot.aiz.diary.' + kind

    async def _fetch(self, source_id, version):
        options = self.options
        token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
        if not options.get('enabled') or not token or not options.get('base_url'):
            return {'state': 'unavailable', 'reason': 'diary_source_reader_unconfigured'}
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(str(options['base_url']).rstrip('/') + '/bridge/v1/memory-source',
                params={'kind': 'diary_' + self.kind, 'source_id': source_id, 'version': version},
                headers={'Authorization': 'Bearer ' + token,
                         'X-Guyan-Device-ID': str(options.get('device_id') or 'ombre-gateway-operit')})
            response.raise_for_status()
            result = response.json().get('source', {})
        if result.get('source_namespace') != self.namespace or result.get('source_id') != source_id:
            return {'state': 'unavailable', 'reason': 'diary_source_identity_mismatch'}
        result['upstream_revision'] = result.get('source_revision')
        if result.get('eligible_source') is not True and result.get('state') == 'available':
            result['state'] = 'unavailable'
        return result

    async def capture(self, day):
        from datetime import date
        try:
            valid = isinstance(day, str) and date.fromisoformat(day).isoformat() == day
        except ValueError:
            valid = False
        if not valid:
            raise L1OwnerError('invalid_diary_date', field_path='date')
        return await self._fetch('diary:' + self.kind + ':' + day, 'current')

    async def read(self, binding):
        return await self._fetch(binding['source_id'], binding['source_revision'])


class AizSharedDiarySourceReader(AizDiarySourceReader):
    def __init__(self, config):
        super().__init__(config, 'shared')
        self.namespace = 'snapshot.ourroom.shared_diary'

    async def capture(self, source_id):
        # A shared diary is selected by its formal date + HMAC projection ID,
        # never by taking an arbitrary first item from that day's diaries.
        locator = json.loads(source_id)
        if not isinstance(locator, dict) or set(locator) != {'date', 'projection_id'}:
            raise L1OwnerError('shared_diary_locator_required', field_path='source_id')
        return await self._fetch(source_id, 'current')


class AizActionSourceReader(CanonicalMemorySourceReader):
    namespace = 'snapshot.aiz.action_receipt'

    async def capture(self, action_id, artifact_id):
        options = self.options
        token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
        if not options.get('enabled') or not token or not options.get('base_url'):
            return {'state': 'unavailable', 'reason': 'action_source_reader_unconfigured'}
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(str(options['base_url']).rstrip('/') + '/bridge/v1/memory-source',
                params={'kind': 'action_receipt', 'source_id': action_id, 'version': artifact_id},
                headers={'Authorization': 'Bearer ' + token,
                    'X-Guyan-Device-ID': str(options.get('device_id') or 'ombre-gateway-operit')})
            response.raise_for_status()
            source = response.json().get('source', {})
        if (source.get('source_namespace') != self.namespace or source.get('source_id') != action_id
                or source.get('source_revision') != artifact_id):
            return {'state': 'unavailable', 'reason': 'action_receipt_identity_mismatch'}
        return source

    async def read(self, binding):
        locator = json.loads(binding['source_id'])
        if locator.get('conversation_id') != self.options.get('conversation_id', 'conv-jiajia-main'):
            return {'state': 'unavailable', 'reason': 'source_outside_bound_conversation'}
        version = self.upstream_version(binding) if hasattr(self, 'upstream_version') else None
        if not version: return {'state': 'unavailable', 'reason': 'action_receipt_version_unknown'}
        source = await self.capture(locator['event_id'], version)
        if source.get('state') != 'available' or source.get('eligible_source') is not True:
            return {'state': source.get('state', 'unavailable'), 'reason': source.get('reason')}
        return source_snapshot({**locator, 'namespace': 'action_receipt', 'text': source['text'],
            'version_id': version, 'recorded_at': source.get('recorded_at')})


class AizAttachmentTextSourceReader(CanonicalMemorySourceReader):
    namespace = 'snapshot.aiz.attachment_text'

    async def read(self, binding):
        from memory_attachment_source import normalize, snapshot, canonical, digest
        locator = json.loads(binding['source_id'])
        raw_revision = 'attachment_sha256:' + digest(canonical(locator))
        conversation = self.options.get('conversation_id', 'conv-jiajia-main')
        if locator.get('conversation_id') != conversation:
            return {'state': 'unavailable', 'reason': 'source_outside_bound_conversation'}
        token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
        if not self.options.get('enabled') or not token or not self.options.get('base_url'):
            return {'state': 'unavailable', 'reason': 'attachment_source_reader_unconfigured'}
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(str(self.options['base_url']).rstrip('/') + '/bridge/v1/memory-source',
                params={'kind': 'attachment_text', 'source_id': binding['source_id'], 'version': raw_revision},
                headers={'Authorization': 'Bearer ' + token,
                         'X-Guyan-Device-ID': str(self.options.get('device_id') or 'ombre-gateway-operit')})
            response.raise_for_status()
            result = response.json().get('source', {})
        if result.get('state') != 'available' or result.get('eligible_source') is not True:
            return {'state': result.get('state', 'unavailable'), 'reason': result.get('reason', 'attachment_not_eligible')}
        if result.get('source_id') != binding['source_id'] or result.get('source_revision') != raw_revision:
            return {'state': 'changed', 'reason': 'attachment_source_changed'}
        source = normalize(result, profile_id=str((self.config.get('persona') or {}).get('profile_id') or 'haven_xiaoyu'),
                           conversation_id=conversation)
        return snapshot(source)


def configured_source_readers(config, authority=None):
    readers = [LocalMemorySourceReader(config, 'raw_event'), LocalMemorySourceReader(config, 'conversation_turn')]
    canonical = CanonicalMemorySourceReader(config)
    action = AizActionSourceReader(config)
    if authority:
        from memory_l0_sources import MemoryL0Sources
        def version(binding):
            retained = MemoryL0Sources(authority).copy(binding)
            return retained.get('upstream_revision') if retained else None
        canonical.upstream_version = version
        action.upstream_version = version
    readers.append(canonical)
    readers.append(AizAttachmentTextSourceReader(config))
    readers.extend(AizDiarySourceReader(config, kind) for kind in ('private', 'digest'))
    readers.extend([AizSharedDiarySourceReader(config), action])
    return readers
