"""Exact delivered-memory evidence; source access is rechecked outside settlement."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from urllib.parse import urlsplit

import httpx

from b1_protocol import digest, identity
from persona_inner_state import evidence_spans, hash_value

ROW_FIELDS = {'reaction_id', 'reaction', 'dimension', 'effect', 'direction', 'strength',
    'memory_ref', 'evidence_spans', 'evidence_hash', 'interaction_observation_ids'}
MEMORY_FIELDS = {'fragment_id', 'memory_id', 'l1_revision', 'l2_revision', 'source_group_id', 'fragment_sha256'}
BRAIN_DIMENSIONS = {'valence', 'arousal', 'tenderness', 'possessiveness', 'longing', 'security',
    'protective_drive', 'session_defensiveness', 'libido', 'grieve', 'anger', 'favored', 'cling'}
AIZ_DIMENSIONS = {'curiosity', 'expression', 'boredom', 'duty', 'reflection', 'novelty',
    'creativity', 'autonomy', 'competence', 'solitude'}


def memory_ref(fragment):
    return {key: fragment[key] for key in MEMORY_FIELDS - {'fragment_sha256'}} | {
        'fragment_sha256': fragment['text_sha256']}


def validate_context(context):
    if (not isinstance(context, dict) or set(context) != {'schema', 'scope', 'delivery_receipt_id', 'delivery_sha256', 'fragments'}
            or context['schema'] != 'memory-context-v1' or not isinstance(context['scope'], dict)
            or (context['scope'].get('consumer_kind') != 'autonomy' and
                set(context['scope']) != {'conversation_id', 'user_source_event_id', 'assistant_source_event_id'})
            or not isinstance(context['fragments'], list)):
        raise ValueError('exact delivered memory context required')
    native = context['scope'].get('consumer_kind') == 'autonomy'
    if native:
        from autonomy_memory import validate_scope
        validate_scope(context['scope'])
    else:
        for key, value in context['scope'].items(): identity(value, key)
    identity(context['delivery_receipt_id'], 'delivery_receipt_id')
    hash_value(context['delivery_sha256'])
    if digest({key: context[key] for key in ('schema', 'scope', 'fragments')}) != context['delivery_sha256']:
        raise ValueError('memory delivery digest mismatch')
    seen = set()
    for fragment in context['fragments']:
        required = {'fragment_id', 'memory_id', 'bucket_id', 'l1_revision', 'l2_revision', 'source_group_id',
            'text', 'text_sha256', 'source_refs', 'coverage'}
        if (not isinstance(fragment, dict) or not required <= set(fragment)
                or set(fragment) - required - {'legacy_body_sha256'} or not isinstance(fragment['text'], str)
                or not fragment['text'] or not isinstance(fragment['coverage'], dict)
                or not isinstance(fragment['source_refs'], list)):
            raise ValueError('complete delivered memory fragment required')
        for key in ('fragment_id', 'memory_id', 'bucket_id', 'source_group_id'):
            identity(fragment[key], key)
        for key in ('l1_revision', 'l2_revision'):
            if fragment[key] is not None and (type(fragment[key]) is not int or fragment[key] < 1):
                raise ValueError('exact positive memory revision or explicit unknown required')
        if hashlib.sha256(fragment['text'].encode()).hexdigest() != hash_value(fragment['text_sha256']):
            raise ValueError('delivered text hash mismatch')
        if fragment['fragment_id'] != 'fragment_' + digest({k: v for k, v in fragment.items() if k != 'fragment_id'})[:32]:
            raise ValueError('delivered fragment identity mismatch')
        if fragment['fragment_id'] in seen:
            raise ValueError('duplicate delivered memory fragment')
        seen.add(fragment['fragment_id'])
    return deepcopy(context)


def reaction_id(source_hash, row):
    return 'mr:' + digest({'source_hash': source_hash, **{
        key: value for key, value in row.items() if key not in ('reaction_id', 'evidence_hash')}})


def validate_reactions(payload, context, source, accepted):
    if not isinstance(payload, dict) or set(payload) != {'schema', 'observations'} or payload['schema'] != 'memory-reactions-v1':
        raise ValueError('versioned memory reaction observations required')
    if not isinstance(payload['observations'], list):
        raise ValueError('memory reaction array required')
    context = validate_context(context)
    by_id = {fragment['fragment_id']: fragment for fragment in context['fragments']}
    source_hash = source if isinstance(source, str) else digest(source)
    hash_value(source_hash)
    accepted_ids = {row['observation_id'] for row in accepted}
    result, seen = [], set()
    for row in payload['observations']:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS or not isinstance(row['memory_ref'], dict) or set(row['memory_ref']) != MEMORY_FIELDS:
            raise ValueError('complete normalized memory reaction required')
        fragment = by_id.get(row['memory_ref']['fragment_id'])
        if fragment is None or row['memory_ref'] != memory_ref(fragment):
            raise ValueError('memory reaction must bind an actually delivered fragment')
        if (row['dimension'] not in BRAIN_DIMENSIONS | AIZ_DIMENSIONS
                or row['reaction'] not in {'new_reaction', 'new_understanding', 'continuing', 'uncertain', 'none'}
                or row['effect'] not in {'evoke', 'ease'} or row['direction'] not in {'increase', 'decrease', 'unknown'}
                or row['strength'] not in {'light', 'clear', 'strong', 'unknown'}):
            raise ValueError('approved memory reaction semantics required')
        native = context['scope'].get('consumer_kind') == 'autonomy'
        spans = evidence_spans(row['evidence_spans'], roles=('planning_output', 'step_outcome') if native else ('user', 'assistant'))
        if isinstance(source, dict):
            for span in spans:
                text = source[span['role']]
                if span['end'] > len(text) or digest(text[span['start']:span['end']]) != span['evidence_hash']:
                    raise ValueError('memory reaction current trigger source mismatch')
        if row['evidence_hash'] != digest({'source_hash': source_hash, 'evidence_spans': spans, 'memory_ref': row['memory_ref']}):
            raise ValueError('memory reaction dual-source evidence mismatch')
        linked = row['interaction_observation_ids']
        if (not isinstance(linked, list) or any(not isinstance(value, str) or value not in accepted_ids for value in linked)
                or linked != sorted(set(linked))):
            raise ValueError('memory reaction interaction must reference current accepted observations')
        if row['reaction_id'] != reaction_id(source_hash, row):
            raise ValueError('memory reaction identity mismatch')
        if row['reaction_id'] not in seen:
            result.append(deepcopy(row))
            seen.add(row['reaction_id'])
    return result


class MemoryReactionResolver:
    """Lookup the trusted Gateway receipt, then use the existing Brain readers."""
    def __init__(self, credentials, reads, l2, *, client=None):
        self.credentials, self.reads, self.l2, self.client = credentials, reads, l2, client

    async def __call__(self, context, turn):
        context = validate_context(context)
        native = context['scope'].get('consumer_kind') == 'autonomy'
        expected = turn['scope'] if native else {'conversation_id': turn['conversation_id'],
            'user_source_event_id': turn['user_event'].get('source_event_id'),
            'assistant_source_event_id': turn['assistant_event'].get('source_event_id')}
        if context['scope'] != expected or (not native and any(not isinstance(value, str) or not value for value in expected.values())):
            return {'state': 'unavailable', 'eligible_fragment_ids': [], 'reason': 'canonical_delivery_pair_mismatch'}
        admin_url, token = self.credentials()
        if not admin_url or not token:
            return {'state': 'unavailable', 'eligible_fragment_ids': [], 'reason': 'memory_delivery_lookup_unconfigured'}
        parsed = urlsplit(admin_url)
        path = parsed.path.rstrip('/')
        if path.endswith('/api/config'):
            path = path[:-len('/api/config')]
        url = parsed._replace(path=path + '/internal/memory-delivery/lookup', query='', fragment='').geturl()
        body = {**expected, 'delivery_receipt_id': context['delivery_receipt_id'], 'delivery_sha256': context['delivery_sha256']}
        try:
            if self.client is not None:
                response = await self.client.post(url, json=body, headers={'Authorization': 'Bearer ' + token})
            else:
                async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
                    response = await client.post(url, json=body, headers={'Authorization': 'Bearer ' + token})
            response.raise_for_status()
            envelope = response.json()
            if envelope.get('state') != 'available' or envelope.get('context') != context:
                return {'state': 'unavailable', 'eligible_fragment_ids': [], 'reason': 'memory_delivery_not_current_or_exact'}
            states = envelope.get('fragment_states')
            if not isinstance(states, list) or any(not isinstance(row, dict) or row.get('state') not in
                    {'available', 'changed', 'withdrawn', 'unavailable'} or not isinstance(row.get('fragment_id'), str) for row in states):
                return {'state': 'unavailable', 'eligible_fragment_ids': [], 'reason': 'memory_fragment_qualification_missing'}
            by_id = {row['fragment_id']: row for row in states}
            if len(by_id) != len(states):
                return {'state': 'unavailable', 'eligible_fragment_ids': [], 'reason': 'memory_fragment_qualification_conflict'}
            eligible, reasons, qualifications = [], {}, {}
            reads, l2 = self.reads(), self.l2()
            for fragment in context['fragments']:
                fid = fragment['fragment_id']
                current = by_id.get(fid) or {'state': 'unavailable'}
                if current['state'] != 'available':
                    reasons[fid] = 'memory_fragment_' + current['state']
                    continue
                try:
                    reason, qualification = await self._fragment(fragment, reads, l2)
                except Exception:
                    reason, qualification = 'memory_fragment_source_check_failed', None
                if reason:
                    reasons[fid] = reason
                else:
                    eligible.append(fid)
                    qualifications[fid] = qualification
            return {'state': 'available', 'eligible_fragment_ids': eligible, 'reasons': reasons,
                'qualifications': qualifications, 'delivery_receipt_id': context['delivery_receipt_id'],
                'delivery_sha256': context['delivery_sha256']}
        except Exception:
            return {'state': 'unavailable', 'eligible_fragment_ids': [], 'reason': 'memory_delivery_source_check_failed'}


    async def _fragment(self, fragment, reads, l2):
        mid = fragment['memory_id']
        legacy = fragment['l1_revision'] is None
        revision = fragment['coverage'].get('legacy_authority_revision') if legacy else fragment['l1_revision']
        memory = reads.authority.get_memory(mid)
        if (type(revision) is not int or revision < 1 or not memory or memory['state'] != 'active'
                or memory['recall_policy'] != 'enabled' or memory['active_revision'] != revision):
            return 'memory_not_current_or_eligible', None
        record = reads.authority.get_memory_revision(mid, revision)
        if record is None:
            return 'memory_revision_unavailable', None
        if legacy:
            body_hash = fragment.get('legacy_body_sha256')
            if not isinstance(body_hash, str) or body_hash != record['body_sha256'] or body_hash != memory['body_sha256']:
                return 'legacy_memory_body_changed_or_unknown', None
        l1 = record.get('metadata', {}).get('memory_l1') or {}
        bindings = {binding['binding_id']: binding for binding in l1.get('evidence_bindings', [])}
        refs = fragment['source_refs']
        if bindings:
            if legacy:
                # A legacy card can still reference a revision which already
                # acquired formal bindings. Recheck those without rewriting it.
                refs = [{'binding_id': key, **{name: binding[name] for name in
                    ('source_namespace', 'source_id', 'source_revision')}} for key, binding in bindings.items()]
            if not refs or any(not isinstance(ref, dict) or ref.get('binding_id') not in bindings
                    or any(ref.get(key) != bindings[ref['binding_id']].get(key)
                        for key in ('source_namespace', 'source_id', 'source_revision')) for ref in refs):
                return 'memory_source_bindings_unavailable', None
            checked = await reads.recall_source_status([{'memory_id': mid, 'revision': revision,
                'binding_ids': [ref['binding_id'] for ref in refs]}])
            status = checked['items'][0]
            if status['state'] != 'ok' or any(not value.get('eligible_source') for value in status['sources'].values()):
                return 'memory_source_withdrawn_or_changed', None
        elif not legacy:
            return 'memory_source_bindings_unavailable', None
        if fragment['l2_revision'] is not None:
            presentation = l2.detail(mid)['l2']
            product = presentation.get('presentation') or {}
            if (presentation['state'] != 'ready' or product.get('presentation_revision') != fragment['l2_revision']
                    or product.get('first_pass_revision') != revision):
                return 'memory_presentation_changed', None
        return None, 'legacy_authority_current' if legacy and not bindings else 'bound_source_current'


def settle_reactions(state, policy, observations, context, validation, occurred_at, interaction_directions):
    """Mutate only the existing short axes inside their owner's transaction."""
    from persona_memory_reactions import RULES_VERSION, repeat_factor, apply_dimension
    history = state.setdefault('memory_reaction_history', {})
    valid = (isinstance(validation, dict) and validation.get('state') == 'available'
        and validation.get('delivery_receipt_id') == context['delivery_receipt_id']
        and validation.get('delivery_sha256') == context['delivery_sha256'])
    eligible = set(validation.get('eligible_fragment_ids', [])) if valid else set()
    results, used, triggers = {}, set(), {}
    strength_order = {'strong': 0, 'clear': 1, 'light': 2, 'unknown': 3}
    ordered = sorted(observations, key=lambda row: (strength_order[row['strength']],
        row['memory_ref']['source_group_id'], row['dimension'], row['direction'], row['reaction_id']))
    for row in ordered:
        key = row['dimension']
        result = {**deepcopy(row), 'rules_version': RULES_VERSION}
        if key not in BRAIN_DIMENSIONS:
            results[row['reaction_id']] = {**result, 'status': 'owned_by_aiz'}
            continue
        before = state['values'][key]
        result.update(before=before, after=before)
        if row['reaction'] in {'continuing', 'uncertain', 'none'}:
            results[row['reaction_id']] = {**result, 'status': 'no_new_pulse', 'reason': row['reaction']}
            continue
        if row['memory_ref']['fragment_id'] not in eligible:
            reason = ((validation or {}).get('reasons') or {}).get(row['memory_ref']['fragment_id'])
            results[row['reaction_id']] = {**result, 'status': 'deferred',
                'reason': reason or (validation or {}).get('reason', 'memory_delivery_not_verified')}
            continue
        result['source_qualification'] = (validation.get('qualifications') or {}).get(row['memory_ref']['fragment_id'], 'verified_delivery_source')
        if any(row['direction'] in interaction_directions.get((observation, key), set())
                for observation in row['interaction_observation_ids']):
            results[row['reaction_id']] = {**result, 'status': 'same_cause_already_settled'}
            continue
        repeat_key = digest([row['memory_ref']['source_group_id'], key, row['direction']])
        trigger_key = digest([key, row['direction'], row['evidence_spans']])
        shared = triggers.get(trigger_key)
        if shared:
            results[row['reaction_id']] = {**result, 'status': 'same_trigger_coalesced',
                'shared_effect_reaction_id': shared, 'contribution_delta': 0}
            # Every source attached to this one applied pulse shares its repeat
            # history; changing which card is named next cannot restore full gain.
            history[repeat_key] = {'occurred_at': occurred_at, 'reaction_id': row['reaction_id'],
                'shared_effect_reaction_id': shared, 'reaction': row['reaction'],
                'memory_ref': deepcopy(row['memory_ref']), 'evidence_hash': row['evidence_hash']}
            used.add(repeat_key)
            continue
        if repeat_key in used:
            results[row['reaction_id']] = {**result, 'status': 'same_turn_coalesced'}
            continue
        factor = repeat_factor(occurred_at, (history.get(repeat_key) or {}).get('occurred_at'))
        # The original legal range of each short axis already has lower bound
        # zero. Keep a higher explicit ordinary-relief floor when present;
        # time-law targets and intimacy/tool satisfaction amounts are unrelated.
        floor = (policy.get('events', {}).get('soothing', {}).get(key) or {}).get('floor', 0.0)
        calculation = apply_dimension(before, effect=row['effect'], direction=row['direction'], strength=row['strength'],
            floor=floor, factor=factor, emotion_inertia=key in {'valence', 'arousal'}, reaction=row['reaction'])
        if key == 'grieve' and calculation.get('requested_delta', 0) > 0 and policy.get('security_grief_policy'):
            modulation, security = policy['security_grief_policy'], state['values'].get('slow_security')
            if security is None:
                calculation.update(after=before, status='unknown_not_updated', reason='unknown_security_modulation', delta=0)
            elif security < modulation['security_below']:
                calculation['unmodulated_delta'] = calculation['delta']
                calculation['unmodulated_requested_delta'] = calculation['requested_delta']
                calculation['requested_delta'] *= modulation['grief_event_multiplier']
                calculation['after'] = min(1.0, max(0.0, before + calculation['requested_delta']))
                calculation['delta'] = calculation['after'] - before
                calculation['status'] = 'applied' if calculation['delta'] else 'no_change'
        state['values'][key] = calculation['after']
        result.update(calculation)
        results[row['reaction_id']] = result
        if calculation['status'] in {'applied', 'no_change'}:
            used.add(repeat_key)
            triggers[trigger_key] = row['reaction_id']
            history[repeat_key] = {'occurred_at': occurred_at, 'reaction_id': row['reaction_id'],
                'reaction': row['reaction'], 'memory_ref': deepcopy(row['memory_ref']), 'evidence_hash': row['evidence_hash']}
    return [results[row['reaction_id']] for row in observations]
