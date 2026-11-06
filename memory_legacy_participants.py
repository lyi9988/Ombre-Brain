"""Prove an attribution-only P7 revision after an exact old-main config binding.

Identity still comes exclusively from source_participants. This compatibility
proof grants no source access and does not rewrite semantic subjects or prose.
"""
from copy import deepcopy
import hashlib
import json

from memory_l1_contract import canonical_json, verify_text_anchor
from memory_source_identity import source_participant


def _revisions(snapshot, participant):
    # Chat readers have no upstream version. Historical descriptors used both
    # absent and empty version_id; neither changes the immutable source text.
    version = snapshot.get('upstream_revision')
    versions = (version,) if version is not None else ('', None)
    return {'source_sha256:' + hashlib.sha256(canonical_json({
        'version_id': value, 'text_sha256': hashlib.sha256(snapshot['text'].encode()).hexdigest(),
        'participant': {key: participant[key] for key in ('subject_id', 'kind')}}).encode()).hexdigest()
        for value in versions}


def participant_only_rebinding(old, new, *, readers, config):
    """Unknown -> exact configured canonical pair, same raw source and text.

    A project may retain several old client conversation coordinates. Each
    requires its own explicit five-coordinate binding; role alone grants none.
    """
    namespace = {'snapshot.ombre.raw_event': 'raw_event',
                 'snapshot.ombre.conversation_turn': 'conversation_turn'}.get(new.get('source_namespace'))
    if (not namespace or old.get('binding_id') != new.get('binding_id')
            or any(old.get(key) != new.get(key) for key in ('source_namespace', 'source_id'))
            or old.get('author_ref') is not None or old.get('speaker_ref') is not None):
        return None
    try:
        locator = json.loads(new['source_id'])
        config = config or {}
        profile = (config.get('persona') or {}).get('profile_id') or 'haven_xiaoyu'
        if (locator.get('system') != 'ombre' or locator.get('profile_id') != profile
                or not isinstance(locator.get('conversation_id'), str) or not locator['conversation_id'].strip()
                or locator.get('role') not in ('user', 'assistant')):
            return None
        configured = source_participant({**locator, 'namespace': namespace}, config)
        options = (config.get('gateway') or {}).get('canonical_continuation') or {}
        canonical = source_participant({'system': 'aizizhu', 'namespace': 'canonical_event',
            'profile_id': profile,
            'conversation_id': options.get('conversation_id', 'conv-jiajia-main'), 'role': locator['role']}, config)
        if (configured.get('resolution') != 'explicit_source_binding'
                or not canonical.get('subject_id')
                or configured.get('kind') != ('human' if locator['role'] == 'user' else 'ai')
                or any(configured.get(key) != canonical.get(key) for key in ('subject_id', 'kind'))
                or new.get('author_ref') != configured['subject_id'] or new.get('speaker_ref') != configured['subject_id']):
            return None
        before, after = readers.copy(old), readers.copy(new)
        if not isinstance(before, dict) or not isinstance(after, dict):
            return None
        if any(before.get(key) != after.get(key)
               for key in (before.keys() | after.keys()) - {'participant', 'source_revision'}):
            return None
        unknown = before.get('participant') or {}
        if (unknown.get('resolution') != 'unbound_source' or unknown.get('subject_id') is not None
                or unknown.get('kind') != 'unknown' or after.get('participant') != configured
                or before.get('state') != 'available' or after.get('state') != 'available'
                or not isinstance(before.get('text'), str) or not before['text']
                or any(before.get(key) != after.get(key) for key in
                       ('source_namespace', 'source_id', 'text_view_id', 'text', 'upstream_revision'))
                or before.get('source_revision') != old['source_revision']
                or after.get('source_revision') != new['source_revision']
                or old['source_revision'] not in _revisions(before, unknown)
                or new['source_revision'] not in _revisions(after, configured)):
            return None
        for binding, snapshot in ((old, before), (new, after)):
            for anchor in binding['anchors']:
                verify_text_anchor(snapshot['text'], anchor, target_namespace='l0', target_id=snapshot['source_id'],
                    target_revision=snapshot['source_revision'], text_view_id=snapshot['text_view_id'])
    except (ValueError, KeyError, TypeError):
        return None
    return {'binding_id': new['binding_id'], 'source_namespace': new['source_namespace'], 'source_id': new['source_id'],
        'previous_source_revision': old['source_revision'], 'source_revision': new['source_revision'],
        'previous_participant': {key: unknown[key] for key in ('subject_id', 'kind')},
        'participant': {key: configured[key] for key in ('subject_id', 'kind')},
        'text_sha256': hashlib.sha256(before['text'].encode()).hexdigest(),
        'upstream_revision': before.get('upstream_revision'),
        'basis': 'exact_configured_ombre_old_main_to_current_canonical_participant'}


def compatible_scope(old, new):
    """Only after the proof above: preserve every old observed coordinate."""
    result = deepcopy(old)
    for key in ('source_revision', 'author_ref', 'speaker_ref'):
        result[key] = new[key]
    for anchor in result['anchors']:
        anchor['target_revision'] = new['source_revision']
    return result
