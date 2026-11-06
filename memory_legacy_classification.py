"""Explicit P7 admission kind and provenance-qualified legacy metadata refresh.

Admission kind, statement kind, catalog topics, domains and bucket type remain
different fields. No text/keyword inference or ranking rule lives here.
"""
from copy import deepcopy
from contextlib import closing
import json

from memory_commit_service import body_sha256
from memory_l1_api import AUTO_KINDS
from memory_l1_service import L1OwnerError


def _generation(value):
    keys = {'actor', 'task_id', 'prompt_revision', 'model_route_revision'}
    if (not isinstance(value, dict) or set(value) != keys or value.get('actor') != 'legacy_backfill'
            or not isinstance(value.get('task_id'), str) or not value['task_id'].strip()
            or any(value[k] is not None and (not isinstance(value[k], str) or not value[k])
                   for k in ('prompt_revision', 'model_route_revision'))):
        raise L1OwnerError('backfill_kind_generation_required', field_path='kind_generation')
    return deepcopy(value)


def candidate_kind(payload, previous=None):
    """Use the native auto-L1 vocabulary; never invoke its domain/text fallback."""
    base = ((previous or {}).get('metadata') or {}).get('legacy_backfill') or {}
    prior = deepcopy(base.get('candidate_kind') or {})
    if prior.get('state') != 'known' or prior.get('value') not in AUTO_KINDS:
        prior = {'state': 'unknown', 'value': None, 'generation': None}
    value = payload.get('kind')
    if value is None or (isinstance(value, str) and not value.strip()):
        if 'kind_generation' in payload:
            raise L1OwnerError('backfill_kind_value_required', field_path='kind')
        return prior
    if not isinstance(value, str) or value.strip().lower() not in AUTO_KINDS:
        raise L1OwnerError('invalid_backfill_kind', field_path='kind')
    value = value.strip().lower()
    if 'kind_generation' in payload:
        generation = _generation(payload['kind_generation'])
    elif prior['state'] == 'known' and prior['value'] == value:
        return prior
    elif previous is not None:
        # A generic P7 card has no retained root kind to inherit. Its original
        # real assembly (or a new explicit model classification) must be named.
        raise L1OwnerError('backfill_kind_generation_required', field_path='kind_generation')
    else:
        generation = _generation(payload['generation'])
    return {'state': 'known', 'value': value, 'generation': generation}


def _generated_terms(authority, record):
    """Only a linked, committed, unedited generator can retire its own terms."""
    metadata = record['metadata']
    if (metadata.get('source') != 'daily_chat_memory' or metadata.get('from_daily_chat') is not True
            or str(metadata.get('memory_classification_source') or '') in {'owner', 'manual', 'owner_edit'}
            or record.get('created_by') in {'owner', 'owner_edit'}
            or record.get('decision_source') in {'owner_edit', 'owner_metadata_edit'}):
        return None
    cid = metadata.get('daily_chat_memory_candidate_id')
    if not isinstance(cid, str) or not cid:
        return None
    row = authority.get_candidate(cid)
    if not row or row['status'] != 'committed':
        return None
    proposal = row['proposal']; info = proposal.get('metadata') or {}
    if (proposal.get('source_type') != 'daily_chat' or info.get('owner_edit_request_hash')
            or body_sha256(proposal.get('proposed_body') or '') != record['body_sha256']):
        return None
    with closing(authority._connect()) as db:
        if db.execute("SELECT 1 FROM candidate_decisions WHERE candidate_id=? AND actor='owner' "
                      'AND from_status=to_status LIMIT 1', (cid,)).fetchone():
            return None
        for value in db.execute('SELECT proposal_json FROM candidate_revisions WHERE candidate_id=?', (cid,)):
            if (json.loads(value[0]).get('metadata') or {}).get('owner_edit_request_hash'):
                return None
    legacy = info.get('legacy_candidate') or {}
    return {'candidate_id': cid, 'candidate_revision': row['revision'], 'proposal_sha256': row['proposal_sha256'],
        **{key: deepcopy(legacy[key]) for key in ('tags', 'keywords')
           if isinstance(legacy.get(key), list) and all(isinstance(v, str) for v in legacy[key])}}


def active_metadata(authority, record, proposal):
    """Refresh proven old machine terms; preserve domains and owner additions."""
    from reflection_engine import ReflectionEngine
    from memory_metadata import normalize_memory_metadata
    metadata = deepcopy(record['metadata'])
    semantic = proposal['metadata']['memory_l1']
    kind = (proposal['metadata']['legacy_backfill'].get('candidate_kind') or {})
    known = kind.get('state') == 'known' and kind.get('value') in AUTO_KINDS
    value = kind['value'] if known else 'memory'
    proof = _generated_terms(authority, record)
    generated = [value, ReflectionEngine._kind_tag(value)] if known else []
    if metadata.get('from_daily_chat') is True:
        generated = ['from_daily_chat', 'daily_chat_extract', *generated]
    report = {'candidate_kind': deepcopy(kind), 'original_generator': proof,
              'domain_preserved': True, 'bucket_type_preserved': True}
    for key in ('tags', 'keywords'):
        old = metadata.get(key, [])
        if not isinstance(old, list) or any(not isinstance(v, str) for v in old):
            report[key] = {'state': 'unproven_legacy_value_preserved'}
            continue
        retired = set((proof or {}).get(key, []))
        preserved = [v for v in old if v not in retired]
        new = [*(generated if key == 'tags' else []), *semantic['retrieval'][key]]
        metadata[key] = list(dict.fromkeys([*preserved, *new]))
        report[key] = {'retired_generated': [v for v in old if v in retired], 'preserved': preserved,
                       'active': deepcopy(metadata[key])}
    if known:
        metadata['memory_type'] = value
    if known and not metadata.get('kind'):
        # `kind` here is the pre-existing normalized Bucket-view vocabulary
        # (event/preference/...), not the candidate admission kind above.
        # Derive it from the ordinary writer's declared structural kind tags,
        # excluding the old pathname and all free-form tags/keywords.
        metadata['kind'] = normalize_memory_metadata({'metadata': {'tags': generated}})['kind']
        report['normalized_bucket_kind'] = metadata['kind']
    elif metadata.get('kind'):
        report['explicit_existing_bucket_kind_preserved'] = metadata['kind']
    return metadata, report
