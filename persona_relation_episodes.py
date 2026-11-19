"""Plan B: retain feelings as source-bound experiences, never baseline drives."""
from copy import deepcopy
from datetime import datetime, timedelta
from math import exp2

from b1_protocol import digest
from persona_inner_state import evidence_spans, hash_value, safe_source_refs, source_ref, upgrade_policy_v2
import persona_relation_dynamics as relation

PROFILE = 'plan-b15-v1'
LOCAL_REVISION = 'inner-brain-r7-plan-b15-local'
CORE = ('valence', 'arousal', 'security', 'slow_security', 'slow_confidence')
SHORT = ('anger', 'grieve', 'cling', 'favored')
KEYS = relation.NEEDS + SHORT
PHASES = {**relation.PHASES, **{key: ('rise', 'ease', 'satisfy') for key in SHORT},
          **{key: (*relation.PHASES[key], 'unresolved') for key in relation.NEEDS}}
STATUSES = ('active', 'unresolved', 'fading', 'sated')
LEGACY_LABELS = {'longing': 'longing', 'possessive': 'possessiveness',
    'desire': 'libido', 'sensual': 'libido', 'angry': 'anger', 'sad': 'grieve',
    'hurt': 'grieve', 'clingy': 'cling', 'jealous': 'favored'}
RETIRED_LABELS = {'yearning', 'guarded', 'tender', 'attached'} | set(LEGACY_LABELS)
# Each number belongs to an actual sourced experience. No idle process seeds
# any of these keys. Defaults are local comparison values, not calibration.
# (initial intensity, half-life hours, relief fraction, buffer hours)
DEFAULTS = {
    'longing': (.4, 6, .50, 2), 'protective_drive': (.45, 10, .55, 2),
    'libido': (.45, 3, .55, 2), 'possessiveness': (.4, 4, .55, 1.5),
    'anger': (.35, 4, .45, 1), 'grieve': (.35, 7, .45, 1.5),
    'cling': (.35, .75, .55, .75), 'favored': (.35, 5, .55, 1.5),
}
DISABLED = ('couplings', 'mixed_policy', 'subtype_policy', 'cling_relief',
            'conflict_policy', 'presence_policy', 'security_grief_policy',
            'growth_policy', 'rhythm_policy', 'fatigue_policy', 'pressure_policy')


def enabled(config):
    return config.get('relation_dynamics', {}).get('profile') == PROFILE


def local_policy(base, revision=LOCAL_REVISION, *, frame_anchor=None):
    """Convert an existing r5/r6 policy; r7 has no Brain-side frame inputs."""
    config = upgrade_policy_v2(base, revision)
    old = config.get('relation_dynamics', {})
    config['relation_dynamics'] = {
        'profile': PROFILE, 'episodes': {},
        'security': deepcopy(old.get('security', {'rise': .08, 'ease': .08})),
        'yearning_hours': old.get('yearning_hours', 2),
        'stance': deepcopy(old.get('stance', {'step': .25, 'half_life_hours': 3})),
        'unresolved_half_life_hours': 72, 'sated_half_life_hours': 1,
        'visible_min': .015, 'fading_below': .12,
    }
    for key, (initial, half, relief, buffer) in DEFAULTS.items():
        config['relation_dynamics']['episodes'][key] = {
            'initial': initial, 'half_life_hours': half, 'ease': relief, 'buffer_hours': buffer}
    for key in DISABLED:
        config[key] = {}
    for name in ('laws', 'sleep_multipliers'):
        config[name] = {key: config[name][key] for key in CORE}
    for rules in config['events'].values():
        for key in set(rules) - (set(CORE) - {'security'}):
            del rules[key]
    if presentation := config.get('presentation'):
        for label, rule in list(presentation['label_rules'].items()):
            if label in RETIRED_LABELS or rule['key'] not in CORE:
                del presentation['label_rules'][label]
        for label in RETIRED_LABELS:
            presentation['cause_priorities'].pop(label, None)
    if not config.get('display_policy'):
        config['display_policy'] = {'mode': 'coexist'}
    return config


def validate_profile(config, number):
    d = config['relation_dynamics']
    if (set(d) != {'profile', 'episodes', 'security', 'stance', 'yearning_hours',
            'unresolved_half_life_hours', 'sated_half_life_hours', 'visible_min', 'fading_below'}
            or set(d['episodes']) != set(KEYS) or set(d['security']) != {'rise', 'ease'}
            or set(d['stance']) != {'step', 'half_life_hours'}
            or any(config[key] for key in DISABLED)
            or any('security' in row for row in config['events'].values())):
        raise ValueError('explicit Plan B episode policy required')
    for row in d['episodes'].values():
        if set(row) != {'initial', 'half_life_hours', 'ease', 'buffer_hours'}:
            raise ValueError('explicit episode dynamics required')
        number(row['initial'], .001); number(row['ease'], .001)
        for key in ('half_life_hours', 'buffer_hours'):
            number(row[key], .001, 168)
    for key in ('unresolved_half_life_hours', 'sated_half_life_hours', 'yearning_hours'):
        number(d[key], .001, 8760)
    number(d['visible_min'], .001)
    number(d['fading_below'], d['visible_min'])
    for value in d['security'].values():
        number(value, .001)
    number(d['stance']['step'], .001, .5)
    number(d['stance']['half_life_hours'], .001, 168)


def is_episode_id(value):
    if not isinstance(value, str) or not value.startswith('ep:'):
        return False
    try:
        hash_value(value[3:])
    except ValueError:
        return False
    return True


def origin(source, spans):
    if source is not None:
        return {**{key: source[key] for key in ('conversation_key', 'user_id', 'assistant_id')},
                'source_hash': digest(source), 'evidence_spans': deepcopy(spans)}


def validate_origin(raw):
    if (not isinstance(raw, dict)
            or set(raw) != {'conversation_key', 'user_id', 'assistant_id', 'source_hash', 'evidence_spans'}):
        return None
    try:
        hash_value(raw['conversation_key'])
        hash_value(raw['source_hash'])
        for key in ('user_id', 'assistant_id'):
            if not isinstance(raw[key], str) or not 0 < len(raw[key]) <= 256:
                return None
        return {**raw, 'evidence_spans': evidence_spans(raw['evidence_spans'])}
    except ValueError:
        return None


def public_episodes(rows):
    """Only clocks, source IDs and hashes are public, never quotes/subject text."""
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    result, seen = [], set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not is_episode_id(row.get('id')) or row['id'] in seen:
            continue
        key, status, intensity = row.get('key'), row.get('status'), row.get('intensity')
        if (key not in KEYS or status not in STATUSES or type(intensity) not in (float, int)
                or not 0 <= intensity <= 1):
            continue
        item = {key: row[key] for key in ('id', 'key', 'status', 'intensity')}
        for date_field in ('started_at', 'updated_at', 'buffer_until'):
            try:
                stamp = datetime.fromisoformat(row.get(date_field))
                if stamp.utcoffset() is not None:
                    item[date_field] = stamp.isoformat()
            except (ValueError, TypeError):
                pass
        refs = safe_source_refs(row.get('source_refs'))
        if not refs or 'started_at' not in item or 'updated_at' not in item:
            continue
        item['source_refs'] = refs
        if original := validate_origin(row.get('origin')):
            item['origin'] = original
        seen.add(row['id'])
        result.append(item)
    return sorted(result, key=lambda item: (item['updated_at'], item['id']), reverse=True)


def advance(state, settings, start, end):
    result = relation.advance_non_numeric(state, settings, start, end)
    hours = (end - start).total_seconds() / 3600
    active = state['relation_episodes']
    for episode_id, item in list(active.items()):
        status = item['status']
        half = (settings['unresolved_half_life_hours'] if status == 'unresolved'
                else settings['sated_half_life_hours'] if status == 'sated'
                else settings['episodes'][item['key']]['half_life_hours'])
        end_buffer = False
        if status == 'sated':
            until = datetime.fromisoformat(item['buffer_until'])
            end_buffer = until <= end
            if end_buffer:
                # Split at the actual end of the buffer, independent of the
                # number of clock events or snapshot reads in the interval.
                buffered = max(0, (until - start).total_seconds() / 3600)
                item['intensity'] *= exp2(-buffered / half)
                half = settings['episodes'][item['key']]['half_life_hours']
                elapsed = hours - buffered
            else:
                elapsed = hours
        else:
            elapsed = hours
        if elapsed:
            item['intensity'] *= exp2(-elapsed / half)
        if end_buffer:
            status = item['status'] = 'fading'
            item.pop('buffer_until')
        if status == 'active' and item['intensity'] < settings['fading_below']:
            status = item['status'] = 'fading'
        if status == 'fading' and item['intensity'] < settings['visible_min']:
            del active[episode_id]
            result.append({'status': 'expired', 'episode_id': episode_id, 'dimension': item['key']})
    return result


def change(state, settings, key, phase, now, ref, *, episode_id=None, original=None, binding=None,
           amount=None, fractional_ease=False):
    episodes = state['relation_episodes']
    row = episodes.get(episode_id) if episode_id else None
    if episode_id and (not row or row['key'] != key):
        return {'status': 'episode_target_unavailable'}
    if row and any(old == ref for old in row['source_refs']):
        return {'status': 'same_cause_already_settled', 'episode_id': row['id']}
    if phase in ('rise', 'unresolved'):
        if row is None:
            episode_id = 'ep:' + digest([key, ref])
            if episode_id in episodes:
                return {'status': 'same_cause_already_settled', 'episode_id': episode_id}
            row = episodes[episode_id] = {
                'id': episode_id, 'key': key, 'intensity': 0, 'status': 'active',
                'started_at': now.isoformat(timespec='microseconds'), 'source_refs': []}
            if original is not None:
                row['origin'] = original
            if binding is not None:
                row['binding'] = binding
        before = row['intensity']
        add = settings['episodes'][key]['initial'] if amount is None else amount
        row['intensity'] = min(1., before + add)
        row['status'] = 'unresolved' if phase == 'unresolved' or row['status'] == 'unresolved' else 'active'
        row.pop('buffer_until', None)
        row.pop('resolved_at', None)
    elif row is None:
        return {'status': 'episode_target_required'}
    elif 'resolved_at' in row:
        return {'status': 'already_sated', 'episode_id': row['id'],
            **({'buffer_until': row['buffer_until']} if 'buffer_until' in row else {})}
    else:
        before = row['intensity']
        if phase == 'satisfy':
            row['intensity'] *= .15
            row['status'] = 'sated'
            row['resolved_at'] = now.isoformat(timespec='microseconds')
            row['buffer_until'] = (now + timedelta(hours=settings['episodes'][key]['buffer_hours'])).isoformat(timespec='microseconds')
        else:
            ratio = settings['episodes'][key]['ease'] if amount is None else amount
            row['intensity'] = max(0., before * (1 - ratio) if fractional_ease or amount is None else before - ratio)
            if row['status'] != 'unresolved':
                row['status'] = 'fading'
    row['source_refs'].append(ref)
    row['updated_at'] = now.isoformat(timespec='microseconds')
    result = {'status': 'applied', 'episode_id': row['id'], 'before': before, 'after': row['intensity']}
    if 'buffer_until' in row:
        result['buffer_until'] = row['buffer_until']
    return result


def legacy_rows(source_hash, originals, bindings, modes, current, nuances):
    """Translate explicit old feelings; generic actions do not invent wants."""
    by_id = {row['observation_id']: row for row in originals}
    result = []
    for binding in bindings + nuances:
        evidence = by_id[binding['observation_id']]
        key = binding.get('key') or LEGACY_LABELS.get(binding['label'])
        if (key not in KEYS or modes.get(binding['observation_id'], 'apply') != 'apply'
                or evidence['actor'] != 'assistant' or evidence['context'] != 'direct'):
            continue
        spans = evidence['evidence_spans']
        if any(row['dimension'] == key and relation.overlaps(spans, row['evidence_spans']) for row in current):
            continue
        row = {'dimension': key, 'phase': 'rise', 'evidence_spans': spans}
        result.append({**row, 'observation_id': 'rel:' + digest({'source_hash': source_hash, **row}),
            'evidence_hash': evidence['evidence_hash'], 'legacy_observation_id': evidence['observation_id']})
    return result


def apply_rows(state, settings, rows, event, now, trace, source):
    """Independently reject invalid targets and group the same source pulse."""
    positive = {'rise', 'unresolved'}
    details, validated, result, good, represented = {}, [], [], [], []
    for row in rows:
        key, phase = row['dimension'], row['phase']
        if key not in KEYS:
            continue
        ref = source_ref('conversation', event['event_id'], event['occurred_at'], row['observation_id'])
        detail = {'dimension': key, 'phase': phase, 'evidence_hash': row['evidence_hash'],
                  'source_ref': ref, 'observation_id': row['observation_id']}
        if 'legacy_observation_id' in row:
            detail['legacy_observation_id'] = row['legacy_observation_id']
        found = row.get('episode_id')
        if found:
            target = state['relation_episodes'].get(found)
            if not target or target['key'] != key or target.get('origin', {}).get('conversation_key') != source['conversation_key']:
                result.append({**detail, 'status': 'rejected', 'reason': 'episode_target_not_bound'})
                continue
        details[row['observation_id']] = detail
        validated.append(row)
    def same_cause(left, right):
        return left['dimension']==right['dimension'] and relation.overlaps(left['evidence_spans'], right['evidence_spans'])

    # Unsupported linked rows cannot block a separate good observation.
    validated.sort(key=lambda row: (row['phase'] != 'unresolved',
        min(span['start'] for span in row['evidence_spans'] if span['role'] == 'assistant')))
    for row in validated:
        key, phase = row['dimension'], row['phase']
        found = row.get('episode_id')
        detail = dict(details[row['observation_id']])
        if any(same_cause(row, other) and (
                ((phase in positive) != (other['phase'] in positive))
                or found and other.get('episode_id') and found != other['episode_id']) for other in validated):
            detail['status'] = 'conflicting_source_phases'
        elif any(same_cause(row, prior) for prior in represented):
            detail['status'] = 'same_cause_already_settled'
        else:
            # Explicit empathy already posted this pulse from the same turn
            # and incident. Preserve its exact help linkage and one intensity.
            empathy = [item for item in state['relation_episodes'].values()
                if item['key'] == key and item.get('binding', {}).get('kind') == 'empathy'
                and (item.get('origin') or {}).get('source_hash') == event['decision']['source_hash']
                and relation.overlaps(row['evidence_spans'], item['origin']['evidence_spans'])]
            if phase in positive and not found and len(empathy) == 1:
                detail.update(status='same_cause_already_settled', episode_id=empathy[0]['id'])
            else:
                detail.update(change(state, settings, key, phase, now, detail['source_ref'], episode_id=found,
                    original=origin(source, row['evidence_spans'])))
            if detail['status'] in ('applied', 'same_cause_already_settled'):
                represented.append(row)
                good.append(row)
        result.append(detail)
    non_episode = [row for row in rows if row['dimension'] not in KEYS]
    for row in list(non_episode):
        if row['dimension'] == 'yearning' and any(fine['dimension'] in relation.YEARNING_REFINERS
                and relation.overlaps(row['evidence_spans'], fine['evidence_spans']) for fine in good):
            non_episode.remove(row)
            result.append({'dimension': 'yearning', 'phase': row['phase'], 'evidence_hash': row['evidence_hash'],
                'observation_id': row['observation_id'], 'status': 'refined_by_supported_dimension',
                'source_ref': source_ref('conversation', event['event_id'], event['occurred_at'], row['observation_id'])})
    prior_stance=state.get('relation_stance')
    rest, diagnostics = relation.apply_rows(state, settings, non_episode, event, now, trace)
    if prior_stance and state.get('relation_stance') != prior_stance:
        stance=state['relation_stance']
        trace('relational_stance', prior_stance['value'], stance['value'], 'relation_stance', stance['ref'])
    return good + rest, result + diagnostics
