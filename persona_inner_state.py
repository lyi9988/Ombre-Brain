"""Versioned B1 inner-state migration and observation bindings. No I/O.

The original checkpoint is retained in migration evidence. Unknown historical
tendency scales never become fabricated midpoint or clipped known values.
"""
from copy import deepcopy
from datetime import datetime
import math

from b1_protocol import digest

STATE_SCHEMA = 'persona-inner-state-v2'
DECISION_SCHEMA = 'b1-decision-v2'
TENDENCIES = ('exploration_affinity', 'social_recovery_rate',
              'connection_initiative', 'creative_satisfaction',
              'challenge_tolerance', 'reflection_preference')
SOOTHING_POLICY = {'repeat_hours': 1, 'repeat_factor': .5}


def tendency_factor(name, value):
    """Only the two explicitly approved local candidate consumers are active."""
    if name not in ('social_recovery_rate', 'creative_satisfaction'):
        raise ValueError('tendency consumer has no approved calibration')
    value = unit_value(value)
    return 1.0 if value is None else 1 + .4 * (value - .5)


def source_ref(kind, event_id, occurred_at, observation=None):
    ref = {'kind': kind, 'event_id': event_id, 'occurred_at': occurred_at}
    if observation is not None:
        ref['observation_id'] = observation
    return ref


def safe_source_refs(refs):
    """Only reference identities cross the snapshot, never source prose."""
    if not isinstance(refs, list):
        return []
    result, seen = [], set()
    for ref in refs:
        if (not isinstance(ref, dict)
                or set(ref) - {'observation_id'} != {'kind', 'event_id', 'occurred_at'}
                or ref['kind'] not in ('conversation', 'operation')
                or not isinstance(ref['event_id'], str) or not 0 < len(ref['event_id']) <= 256):
            continue
        try:
            when = datetime.fromisoformat(ref['occurred_at'])
            if when.tzinfo is None or when.utcoffset() is None:
                continue
            if 'observation_id' in ref:
                obs = ref['observation_id']
                if not isinstance(obs, str) or not obs.startswith('obs:'):
                    continue
                hash_value(obs[4:])
        except (ValueError, TypeError):
            continue
        key = (ref['kind'], ref['event_id'], ref['occurred_at'], ref.get('observation_id'))
        if key not in seen:
            result.append(deepcopy(ref)); seen.add(key)
    return result


def unit_value(value):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('finite 0..1 value or explicit unknown required')
    return float(value)


def hash_value(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError('canonical SHA256 required')
    return value


def evidence_spans(spans):
    if not isinstance(spans, list) or not spans:
        raise ValueError('nonempty evidence spans required')
    keys = []
    for span in spans:
        if not isinstance(span, dict) or set(span) != {'role', 'start', 'end', 'evidence_hash'}:
            raise ValueError('complete evidence span required')
        if span['role'] not in ('user', 'assistant'):
            raise ValueError('source role required')
        if (any(isinstance(span[k], bool) or not isinstance(span[k], int) for k in ('start', 'end'))
                or not 0 <= span['start'] < span['end']):
            raise ValueError('bounded evidence span required')
        hash_value(span['evidence_hash'])
        keys.append((0 if span['role'] == 'user' else 1, span['start'], span['end']))
    if keys != sorted(set(keys)):
        raise ValueError('canonical sorted unique evidence spans required')
    return deepcopy(spans)


def observation_id(source_hash, row):
    return 'obs:' + digest({'source_hash': source_hash, **{
        key: row[key] for key in ('type', 'actor', 'target', 'context',
                                  'assertion', 'certainty', 'evidence_spans', 'subject_binding')}})


def bound_observation(detail, accepted, kind=None):
    if not isinstance(detail, dict) or not isinstance(detail.get('observation_id'), str):
        raise ValueError('observation-bound detail required')
    matches = [row for row in accepted if row['observation_id'] == detail['observation_id']]
    if (len(matches) != 1 or matches[0]['type'] != detail.get('type')
            or matches[0]['evidence_hash'] != detail.get('evidence_hash')
            or (kind is not None and matches[0]['type'] != kind)):
        raise ValueError('detail observation mismatch')
    return matches[0]


def verify_original_spans(row, source, source_hash):
    """Verify actual original strings, never trust evaluator offsets alone."""
    spans = evidence_spans(row['evidence_spans'])
    for span in spans:
        text = source[span['role']]
        if (span['end'] > len(text)
                or digest(text[span['start']:span['end']]) != span['evidence_hash']):
            raise ValueError('source span binding mismatch')
    if row['evidence_hash'] != digest({'source_hash': source_hash, 'evidence_spans': spans}):
        raise ValueError('source evidence bundle mismatch')


def subject_binding(detail, observation, source_hash, source=None):
    binding = detail.get('subject_binding')
    if not isinstance(binding, dict) or set(binding) != {'source_hash', 'role', 'start', 'end', 'mention_hash'}:
        raise ValueError('complete empathy subject binding required')
    if binding != observation.get('subject_binding'):
        raise ValueError('subject mention must match its observation identity')
    if binding['source_hash'] != source_hash or binding['role'] not in ('user', 'assistant'):
        raise ValueError('empathy subject source mismatch')
    if (any(isinstance(binding[k], bool) or not isinstance(binding[k], int) for k in ('start', 'end'))
            or not 0 <= binding['start'] < binding['end']):
        raise ValueError('bounded empathy subject span required')
    hash_value(binding['mention_hash'])
    if not any(span['role'] == binding['role'] and span['start'] <= binding['start']
               and binding['end'] <= span['end'] for span in observation['evidence_spans']):
        raise ValueError('subject mention must belong to its observation')
    if source is not None:
        text = source[binding['role']]
        if binding['end'] > len(text) or digest(text[binding['start']:binding['end']]) != binding['mention_hash']:
            raise ValueError('empathy subject mention mismatch')
    return deepcopy(binding)


def upgrade_policy_v2(config, revision=None):
    """Explicit local candidate upgrade using the approved ordinary relief.

    Existing ordinary conflict pulses and all intimacy/light parameters stay in
    their original policies. Reconciliation and the special episode retire.
    """
    result = deepcopy(config)
    result['events'].pop('reconciliation', None)
    result['slow_axis_events'].pop('reconciliation', None)
    result['conflict_policy'] = {}
    result['events'].setdefault('soothing', {
        key: {'delta': 0, 'satisfaction': relief, 'floor': 0, 'plateau_hours': 0}
        for key, relief in (('anger', .15), ('grieve', .10))})
    result.setdefault('soothing_policy', deepcopy(SOOTHING_POLICY))
    if revision is not None:
        result['revision'] = revision
    elif not result['revision'].endswith('-inner-state-v2'):
        result['revision'] += '-inner-state-v2'
    return result


def normalize_tendencies(data=None, sources=None):
    data = data if isinstance(data, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    values, resolved, evidence, issues = {}, {}, {}, []
    for name in TENDENCIES:
        raw = data.get(name)
        source = sources.get(name)
        explicit_source = (isinstance(source, dict) and source.get('unit') == '0..1'
                        and isinstance(source.get('source_ref'), str)
                        and bool(source['source_ref'].strip()))
        known_source = explicit_source and source.get('status', 'known') == 'known'
        value = None
        issue = None
        if raw is not None:
            if not known_source:
                issue = 'tendency_scale_unknown'
            else:
                try:
                    value = unit_value(raw)
                except ValueError:
                    issue = 'tendency_value_invalid'
            evidence[name] = {'value': deepcopy(raw), 'source': deepcopy(source)}
        if issue:
            issues.append({'field': 'tendencies.' + name, 'code': issue})
        values[name] = value
        resolved[name] = ({'unit': '0..1', 'source_ref': source['source_ref'], 'status': 'known'}
                          if known_source and value is not None else
                          {'unit': '0..1', 'source_ref': source['source_ref'] if explicit_source else None, 'status': 'unknown'})
    return values, resolved, evidence, issues


def new_state_fields():
    tendencies, sources, _, _ = normalize_tendencies()
    return {'state_schema': STATE_SCHEMA, 'tendencies': tendencies,
            'tendency_sources': sources, 'tendency_evidence': {},
            'migration_issues': [], 'state_migrations': [], 'label_sources': {}, 'memory_reaction_history': {}}


def migrate_state(raw, *, tendencies=None, tendency_sources=None):
    """Return a standard-scale copy; persisting it is an explicit store action."""
    state = deepcopy(raw)
    old_schema = state.get('state_schema')
    if old_schema not in (None, 'persona-inner-state-v1', STATE_SCHEMA):
        raise ValueError('unsupported inner-state schema')
    mood = deepcopy(state.get('slow_mood', {}))
    if old_schema != STATE_SCHEMA:
        original_mood = deepcopy(mood)
        value = mood.get('value')
        unit = mood.get('unit', '-1..1')
        if unit == '-1..1':
            if value is not None:
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not -1 <= value <= 1):
                    raise ValueError('legacy mood scale invalid')
                mood['value'] = (float(value) + 1) / 2
        elif unit != '0..1':
            raise ValueError('explicit mood scale required')
        mood['unit'] = '0..1'
        state.setdefault('state_migrations', []).append({
            'from_schema': old_schema or 'b1-state-v1', 'to_schema': STATE_SCHEMA,
            'at': state.get('at'), 'prior_slow_mood': original_mood,
            'retired_conflict': {'grudge': deepcopy(state.get('grudge')),
                                 'episode': deepcopy(state.get('conflict_episode'))}})
    if mood.get('unit') != '0..1':
        raise ValueError('v2 mood must use 0..1')
    mood['value'] = unit_value(mood.get('value'))
    state['slow_mood'] = mood
    data = state.get('tendencies') if tendencies is None else tendencies
    sources = state.get('tendency_sources') if tendency_sources is None else tendency_sources
    vals, resolved, evidence, issues = normalize_tendencies(data, sources)
    state['tendencies'] = vals
    state['tendency_sources'] = resolved
    retained = state.setdefault('tendency_evidence', {})
    for name, original in evidence.items():
        retained.setdefault(name, original)
    prior = state.setdefault('migration_issues', [])
    for issue in issues:
        if issue not in prior:
            prior.append(issue)
    state['grudge'] = None
    state['conflict_episode'] = None
    state['state_schema'] = STATE_SCHEMA
    state.setdefault('state_migrations', [])
    state.setdefault('label_sources', {})
    state.setdefault('memory_reaction_history', {})
    return state
