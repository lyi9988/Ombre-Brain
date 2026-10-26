"""Normalize authenticated browse views, preserving article/read distinctions."""
from __future__ import annotations

import copy
import hashlib
import json

from memory_l1_contract import make_text_anchor
from memory_l1_service import L1OwnerError


def snapshot(source):
    value = copy.deepcopy(source['browse_snapshot'])
    if (value.get('state') != 'available' or value.get('eligible_source') is not True
            or value.get('source_namespace') not in {'snapshot.aiz.browse.article', 'snapshot.aiz.browse.reading'}
            or not isinstance(value.get('text'), str) or not value['text']):
        raise L1OwnerError('browse_source_not_available', field_path='sources')
    return value


def model_source(value, config):
    from memory_source_identity import source_participant, source_group
    from memory_l1_contract import canonical_json
    locator = json.loads(value['source_id'])
    coordinates = {'system': 'aizizhu', 'namespace': 'browse_material',
        'profile_id': str((config.get('persona') or {}).get('profile_id') or 'haven_xiaoyu'),
        'conversation_id': ((config.get('gateway') or {}).get('canonical_continuation') or {}).get('conversation_id', 'conv-jiajia-main'),
        'event_id': value['source_id'], 'role': 'assistant' if locator['view'] == 'reading' else 'tool'}
    if locator['view'] == 'reading':
        participant = source_participant({**coordinates, 'namespace': 'canonical_event'}, config)
    else:
        participant = {'subject_id': None, 'kind': 'unknown', 'display_name': value.get('author') or '外部作者未知',
            'resolution': 'unverified_external_byline'}
    revision = 'source_sha256:' + hashlib.sha256(canonical_json({'upstream_revision': value['source_revision'],
        'participant': {key: participant[key] for key in ('subject_id', 'kind')}}).encode()).hexdigest()
    value = {**value, 'participant': participant, 'speaker': value.get('author'),
        'source_group': source_group(coordinates), 'upstream_revision': value['source_revision'], 'source_revision': revision}
    return {**coordinates, 'text': value['text'], 'recorded_at': value.get('recorded_at'),
        'browse_snapshot': value, '_model_ranges': [{'start': left, 'end': right} for left, right in value['read_ranges']]}


def read_bindings(source, snapshot, binding):
    """Only observed article ranges are evidence, even if full L0 is retained."""
    binding['anchors'] = [make_text_anchor(snapshot['text'], target_namespace='l0', target_id=snapshot['source_id'],
        target_revision=snapshot['source_revision'], text_view_id=snapshot['text_view_id'],
        start=extent['start'], end=extent['end']) for extent in source['_model_ranges']]
    return binding


def semantic_views(snapshots, bindings):
    # Exact absolute offsets let scenes bind to real read ranges. Unread text is
    # retained as L0 for inspection but never sent to the preparation model.
    return [{'binding_id': binding['binding_id'], 'author': value.get('author'),
        'participant': value.get('participant'), 'source_recorded_at': value.get('recorded_at'),
        'read_ranges': [{'start': a['start'], 'end': a['end'], 'text': value['text'][a['start']:a['end']]}
            for a in binding['anchors']], 'coverage': value.get('coverage'),
        'limitations': value.get('limitations', [])} for value, binding in zip(snapshots, bindings)]
