"""Native daily-review and owner-annotation sources for the existing L3 chain.

These namespaces are not canonical user turns. Live GETs remain prerequisites
after retaining a source copy and at the original L1 adoption boundary.
"""
from copy import deepcopy
import hashlib
import json
import os

import httpx

from b1_protocol import digest
from memory_l1_contract import canonical_json, make_text_anchor
from memory_l1_service import L1OwnerError
from persona_memory_reaction_context import validate_context, memory_ref

SCHEMA = 'self-review-followup-v1'
NAMESPACES = {'daily_review': 'snapshot.aiz.self_review', 'reply_feedback': 'snapshot.aiz.reply_feedback'}
TARGET_FIELDS = ('memory_id', 'bucket_id', 'l1_revision', 'l2_revision', 'source_group_id',
                 'text_sha256', 'source_refs', 'coverage', 'legacy_body_sha256')


def target_key(fragment):
    return 'memory-target:' + digest({key: fragment.get(key) for key in TARGET_FIELDS})


def validate_native_submission(payload):
    if not isinstance(payload, dict) or set(payload) != {'schema', 'context', 'review', 'sources', 'updates'} or payload['schema'] != SCHEMA:
        raise L1OwnerError('invalid_review_followup_submission')
    context = validate_context(payload['context'])
    review = payload['review']
    if (context['scope'].get('consumer_kind') != 'autonomy' or not isinstance(review, dict)
            or set(review) != {'identity_id', 'conversation_id', 'run_id', 'cycle_id', 'operation_id', 'output_sha256'}
            or any(not isinstance(value, str) or not value for value in review.values())
            or any(review[key] != context['scope'][key] for key in ('identity_id', 'conversation_id', 'cycle_id', 'operation_id'))):
        raise L1OwnerError('native_review_scope_mismatch')
    sources = payload['sources']
    if not isinstance(sources, list) or not 1 <= len(sources) <= 65:
        raise L1OwnerError('native_review_sources_required')
    by_key = {}
    for source in sources:
        if (not isinstance(source, dict) or source.get('kind') not in NAMESPACES
                or source.get('source_namespace') != NAMESPACES[source['kind']]
                or not isinstance(source.get('text'), str) or not source['text']
                or hashlib.sha256(source['text'].encode()).hexdigest() != source.get('content_sha256')
                or not all(isinstance(source.get(key), str) and source[key] for key in
                           ('source_id', 'source_revision', 'source_key', 'recorded_at'))):
            raise L1OwnerError('invalid_native_review_source')
        locator = json.loads(source['source_id'])
        tail = 'run_id' if source['kind'] == 'daily_review' else 'feedback_id'
        if (set(locator) != {'identity_id', 'conversation_id', tail} or canonical_json(locator) != source['source_id']
                or any(locator[key] != review[key] for key in ('identity_id', 'conversation_id'))
                or source['source_key'] in by_key):
            raise L1OwnerError('native_review_source_identity_mismatch')
        by_key[source['source_key']] = source
    if (by_key.get('self_review', {}).get('kind') != 'daily_review'
            or by_key['self_review']['source_revision'] != review['output_sha256']
            or json.loads(by_key['self_review']['source_id'])['run_id'] != review['run_id']):
        raise L1OwnerError('exact_native_review_output_required')
    if not isinstance(payload['updates'], list) or len(payload['updates']) > 32:
        raise L1OwnerError('bounded_followups_required')
    fragments = {fragment['fragment_id']: fragment for fragment in context['fragments']}
    result, seen = [], set()
    for row in payload['updates']:
        if (not isinstance(row, dict) or set(row) != {'update_id', 'target_key', 'memory_ref', 'kind', 'long_term_value', 'content', 'evidence_refs'}
                or row['kind'] not in {'correction', 'understanding', 'question'}
                or row['long_term_value'] not in {'explicit_owner_correction', 'explicit_remember', 'specific_future_use', 'none'}
                or not isinstance(row['content'], str) or not row['content'].strip()
                or not isinstance(row['memory_ref'], dict)
                or not isinstance(row['evidence_refs'], list) or not row['evidence_refs']
                or any(not isinstance(key, str) or key not in by_key for key in row['evidence_refs'])):
            raise L1OwnerError('invalid_native_review_observation')
        fragment = fragments.get(row['memory_ref'].get('fragment_id'))
        if not fragment or row['memory_ref'] != memory_ref(fragment) or row['target_key'] != target_key(fragment):
            raise L1OwnerError('followup_memory_not_delivered')
        owner = any(by_key[key]['kind'] == 'reply_feedback' for key in row['evidence_refs'])
        if (row['kind'] == 'correction' or row['long_term_value'] in
                {'explicit_owner_correction', 'explicit_remember'}) and not owner:
            raise L1OwnerError('explicit_owner_feedback_required')
        semantic = {key: deepcopy(row[key]) for key in ('target_key', 'kind', 'long_term_value', 'content', 'evidence_refs')}
        if row['update_id'] != 'ml3:' + digest({'review': review, **semantic}):
            raise L1OwnerError('followup_identity_mismatch')
        if row['long_term_value'] != 'none' and row['update_id'] not in seen:
            result.append(deepcopy(row)); seen.add(row['update_id'])
    return result


class AizReviewSourceReader:
    def __init__(self, config, kind):
        self.config, self.kind, self.namespace = config, kind, NAMESPACES[kind]

    async def capture(self, source_id, version, *, client=None):
        options = (self.config.get('gateway') or {}).get('canonical_continuation') or {}
        token = os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN', '')
        locator = json.loads(source_id)
        if (not options.get('enabled') or not token or not options.get('base_url')
                or locator.get('conversation_id') != options.get('conversation_id', 'conv-jiajia-main')):
            return {'state': 'unavailable', 'reason': 'review_source_reader_unconfigured_or_scope_mismatch'}
        url = str(options['base_url']).rstrip('/') + '/bridge/v1/self-review-source'
        kwargs = {'params': {'kind': self.kind, 'source_id': source_id, 'version': version},
            'headers': {'Authorization': 'Bearer ' + token,
                        'X-Guyan-Device-ID': str(options.get('device_id') or 'ombre-gateway-operit')}}
        if client is not None:
            response = await client.get(url, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as reader:
                response = await reader.get(url, **kwargs)
        response.raise_for_status()
        source = response.json().get('source') or {}
        if source.get('state') != 'available' or source.get('eligible_source') is not True:
            return source
        if (source.get('kind') != self.kind or source.get('source_namespace') != self.namespace
                or source.get('source_id') != source_id or source.get('source_revision') != version
                or hashlib.sha256(source['text'].encode()).hexdigest() != source.get('content_sha256')):
            raise L1OwnerError('native_review_source_identity_mismatch', status=409)
        return source

    async def read(self, binding):
        version = self.upstream_version(binding) if hasattr(self, 'upstream_version') else None
        if not version:
            return {'state': 'unavailable', 'reason': 'review_upstream_revision_unknown'}
        source = await self.capture(binding['source_id'], version)
        return snapshot(source, self.config) if source.get('state') == 'available' and source.get('eligible_source') else source


def snapshot(source, config):
    """Bind the configured existing actor, retaining the native source namespace."""
    from memory_source_identity import source_participant, source_group
    locator = json.loads(source['source_id'])
    role = 'user' if source['kind'] == 'reply_feedback' else 'assistant'
    coordinates = {'system': 'aizizhu', 'namespace': 'canonical_event',
        'profile_id': str((config.get('persona') or {}).get('profile_id') or 'haven_xiaoyu'),
        'conversation_id': locator['conversation_id'], 'role': role}
    # Resolve only the already configured owner/self actor pair. This does
    # not turn the annotation or review into a canonical message or turn.
    participant = source_participant(coordinates, config)
    native = {**coordinates, 'namespace': 'reply_feedback' if role == 'user' else 'self_review',
              'role': 'owner_feedback' if role == 'user' else 'ai_review'}
    revision = 'source_sha256:' + digest({'upstream_revision': source['source_revision'],
        'participant': {key: participant[key] for key in ('subject_id', 'kind')}})
    return {'state': 'available', 'source_namespace': source['source_namespace'], 'source_id': source['source_id'],
        'source_revision': revision, 'upstream_revision': source['source_revision'], 'text_view_id': source['text_view_id'],
        'text': source['text'], 'author': native['role'], 'speaker': native['role'], 'participant': participant,
        'recorded_at': source['recorded_at'], 'source_group': source_group(native),
        'source_key': digest([source['source_namespace'], source['source_id'], revision]),
        'limitations': ['主人主动批注；不是canonical用户回合。' if role == 'user' else '顾衍后台整理形成的理解；不是主人发言或事实确认。']}


async def read_payload_source(config, source, reader=None):
    actual = await reader(source['kind'], source['source_id'], source['source_revision']) if reader else await AizReviewSourceReader(
        config, source['kind']).capture(source['source_id'], source['source_revision'])
    if actual.get('state') != 'available' or actual.get('eligible_source') is not True:
        return None
    if any(actual.get(key) != source.get(key) for key in
           ('kind', 'source_namespace', 'source_id', 'source_revision', 'source_key', 'text', 'content_sha256', 'recorded_at')):
        raise L1OwnerError('native_review_source_changed', status=409)
    return actual


async def check_native_sources(payload, config, reader=None, *, with_sources=False):
    actual = {}
    required = {'self_review'} | {key for update in payload['updates'] for key in update['evidence_refs']}
    for source in payload['sources']:
        if source['source_key'] not in required:
            continue
        value = await read_payload_source(config, source, reader)
        if value:
            actual[source['source_key']] = value
    review = actual.get('self_review')
    if not review:
        return (set(), actual) if with_sources else set()
    if review.get('review') != payload['review']:
        raise L1OwnerError('native_review_output_identity_mismatch')
    originals = review.get('memory_updates') or []
    catalogue = {row['target_key'] for row in review.get('memory_targets') or []}
    accepted = set()
    def matches_original(row, update, semantic):
        if (not isinstance(row, dict) or row.get('memory_fragment_ref', row.get('target_key')) != update['target_key']
                or not isinstance(row.get('evidence_refs'), list)
                or any(not isinstance(key, str) for key in row['evidence_refs'])):
            return False
        return all((list(dict.fromkeys(row[key])) if key == 'evidence_refs' else row.get(key)) == value
                   for key, value in semantic.items())
    for update in payload['updates']:
        semantic = {key: update[key] for key in ('kind', 'long_term_value', 'content', 'evidence_refs')}
        matches = any(matches_original(row, update, semantic) for row in originals)
        if not matches or update['target_key'] not in catalogue:
            raise L1OwnerError('followup_not_in_original_review')
        if all(key in actual for key in update['evidence_refs']):
            accepted.add(update['update_id'])
    return (accepted, actual) if with_sources else accepted


async def native_sources(service, task):
    reader = service.engine._memory_l1_sources()
    update = task['envelope']['updates'][0]
    allowed, actual_sources = await check_native_sources(task['envelope'], service.engine.config,
        service.review_source_reader, with_sources=True)
    if update['update_id'] not in allowed:
        raise L1OwnerError('followup_source_not_currently_readable', status=409, field_path='sources')
    models, evidence = [], []
    for source in task['envelope']['sources']:
        if source['source_key'] not in update['evidence_refs']:
            continue
        actual = actual_sources.get(source['source_key'])
        if actual is None:
            raise L1OwnerError('followup_source_not_currently_readable', status=409, field_path='sources')
        value = snapshot(actual, service.engine.config)
        binding = {'binding_id': 'source:followup:' + digest([value['source_namespace'], value['source_id'], value['source_revision']])[:24],
            'source_namespace': value['source_namespace'], 'source_id': value['source_id'], 'source_revision': value['source_revision'],
            'author_ref': value['participant']['subject_id'], 'speaker_ref': value['participant']['subject_id'],
            'anchors': [make_text_anchor(value['text'], target_namespace='l0', target_id=value['source_id'],
                target_revision=value['source_revision'], text_view_id=value['text_view_id'], start=0, end=len(value['text']))]}
        reader.retain_copy(value)
        locator = json.loads(value['source_id'])
        models.append({'system': 'aizizhu', 'namespace': 'reply_feedback' if actual['kind'] == 'reply_feedback' else 'self_review',
            'profile_id': service.engine.memory_semantic_profile_id, 'conversation_id': locator['conversation_id'],
            'event_id': locator.get('feedback_id', locator.get('run_id')), 'version_id': actual['source_revision'],
            'role': value['author'], 'text': value['text'], 'recorded_at': value['recorded_at'], 'source_key': value['source_key']})
        evidence.append({'source': value, 'binding': binding})
    return models, evidence, reader
