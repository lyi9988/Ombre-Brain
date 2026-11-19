"""Explicit local candidate for short relationship feelings in the Brain ledger."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import exp2

from b1_protocol import digest
from persona_inner_state import evidence_spans, source_ref, verify_original_spans


PROFILE = 'relation-dynamics-v1'
LOCAL_REVISION = 'inner-brain-r6-local-20261120'
NEEDS = ('longing', 'possessiveness', 'protective_drive', 'libido')
YEARNING_REFINERS = ('longing', 'possessiveness', 'libido')
RETIRED = ('tenderness', 'session_defensiveness')
PHASES = {**{name: ('rise', 'ease', 'satisfy') for name in NEEDS},
    'security': ('rise', 'ease'), 'yearning': ('felt',),
    'relational_stance': ('soften', 'lead')}
ROWS = {'dimension', 'phase', 'evidence_spans', 'evidence_hash', 'observation_id'}

# These are local comparison values, not an owner calibration. A frozen old
# coordinate's historical value remains readable from its old policy/receipt.
# (usual, half-life hours, rise, ease, satisfaction drop, buffer hours)
LOCAL_NEEDS = {
    'longing': (.60, 6, .02, .12, .24, 2),
    'possessiveness': (.50, 4, .055, .12, .23, 1.5),
    'protective_drive': (.52, 5, .05, .14, .26, 1.5),
    'libido': (.40, 3, .07, .15, .28, 2),
}


def local_policy(base, revision=LOCAL_REVISION, *, frame_anchor=None):
    """Select only on an isolated ledger; preserve unrelated r5 rules."""
    from persona_inner_state import upgrade_policy_v2

    if frame_anchor is not None:
        frame_anchor = datetime.fromisoformat(frame_anchor)
        if frame_anchor.utcoffset() is None:
            raise ValueError('explicit timezone required for frame anchor')
        frame_anchor = frame_anchor.astimezone(timezone.utc).isoformat()
    config = upgrade_policy_v2(base, revision)
    config['relation_dynamics'] = {
        'profile': PROFILE, 'needs': {}, 'security': {'rise': .08, 'ease': .08},
        'yearning_hours': 2, 'stance': {'step': .25, 'half_life_hours': 3},
    }
    for name, (usual, half, rise, ease, satisfy, buffer) in LOCAL_NEEDS.items():
        config['laws'][name] = {'target': usual, 'half_life_hours': half, 'growth_per_hour': 0}
        config['sleep_multipliers'][name] = 1
        config['relation_dynamics']['needs'][name] = {
            'rise': rise, 'ease': ease, 'satisfy': satisfy, 'buffer_hours': buffer}
    retired = set(RETIRED)
    for name in retired:
        config['laws'][name] = {'target': 0, 'half_life_hours': None, 'growth_per_hour': 0}
        config['sleep_multipliers'][name] = 1
    for rules in config['events'].values():
        for name in set(NEEDS) | retired | {'security'}:
            rules.pop(name, None)
    config['cling_relief'] = {}
    if rhythm := config.get('rhythm_policy'):
        rhythm['night_multipliers'] = {k: v for k, v in rhythm['night_multipliers'].items() if k not in NEEDS}
        rhythm['dawn_freeze'] = [key for key in rhythm['dawn_freeze'] if key not in NEEDS]
    if growth := config.get('growth_policy'):
        growth['targets'] = {key: value for key, value in growth['targets'].items() if key not in NEEDS}
        if not growth['targets']:
            config['growth_policy'] = {}
        else:
            growth.update(revision=growth['revision'] + '-relation-v1', source_policy_revision=revision)
            if frame_anchor is not None:
                growth['anchor_utc'] = frame_anchor
    if pressure := config.get('pressure_policy'):
        pressure['revision'] += '-relation-v1'
        pressure['hosts']['brain']['policy_revision'] = revision
        if frame_anchor is not None:
            pressure['anchor_utc'] = frame_anchor
    if presentation := config.get('presentation'):
        inactive = {'yearning', 'guarded', 'tender'}
        inactive.update(label for label, rule in presentation['label_rules'].items() if rule['key'] in retired)
        for label in inactive:
            presentation['label_rules'].pop(label, None)
            presentation['cause_priorities'].pop(label, None)
    return config


def validate_profile(config, number):
    dynamics = config['relation_dynamics']
    if (not isinstance(dynamics, dict)
            or set(dynamics) != {'profile', 'needs', 'security', 'yearning_hours', 'stance'}
            or dynamics['profile'] != PROFILE or not isinstance(dynamics['needs'], dict)
            or set(dynamics['needs']) != set(NEEDS)):
        raise ValueError('explicit local relationship dynamics required')
    for name, row in dynamics['needs'].items():
        if set(row) != {'rise', 'ease', 'satisfy', 'buffer_hours'}:
            raise ValueError('explicit independent relationship effect required')
        for key in ('rise', 'ease', 'satisfy'):
            number(row[key], .001)
        number(row['buffer_hours'], .001, 168)
        law = config['laws'][name]
        if law['half_life_hours'] is None or law['growth_per_hour'] or 'rest_ceiling' in law or config['sleep_multipliers'][name] != 1:
            raise ValueError('relationship time must return independently to usual')
    if (set(dynamics['security']) != {'rise', 'ease'} or set(dynamics['stance']) != {'step', 'half_life_hours'}
            or config['cling_relief']
            or any(set(rules) & (set(NEEDS) | set(RETIRED) | {'security'}) for rules in config['events'].values())
            or set(config['growth_policy'].get('targets', {})) & set(NEEDS)
            or set(config['rhythm_policy'].get('dawn_freeze', [])) & set(NEEDS)):
        raise ValueError('relationship dynamics must replace old implicit effects')
    for key in RETIRED:
        law = config['laws'][key]
        if law['half_life_hours'] is not None or law['growth_per_hour'] or 'rest_ceiling' in law or config['sleep_multipliers'][key] != 1:
            raise ValueError('retired coordinates must remain frozen')
    for value in dynamics['security'].values():
        number(value, .001)
    number(dynamics['yearning_hours'], .001, 168)
    number(dynamics['stance']['step'], .001, .5)
    number(dynamics['stance']['half_life_hours'], .001, 168)


def verify_rows(raw, source_hash, source):
    """Return independently verified rows and source-safe rejection diagnostics."""
    if not isinstance(raw, list) or len(raw) > 32:
        raise ValueError('bounded relationship observations required')
    good, rejected = [], []
    for index, row in enumerate(raw):
        try:
            if not isinstance(row, dict) or set(row) != ROWS:
                raise ValueError('incomplete_observation')
            dimension, phase = row['dimension'], row['phase']
            if not isinstance(dimension, str) or dimension not in PHASES or phase not in PHASES[dimension]:
                raise ValueError('unsupported_phase_or_dimension')
            spans = evidence_spans(row['evidence_spans'])
            if not any(span['role'] == 'assistant' for span in spans):
                raise ValueError('assistant_evidence_required')
            verify_original_spans(row, source, source_hash)
            if row['observation_id'] != 'rel:' + digest({
                    'source_hash': source_hash, 'dimension': dimension, 'phase': phase, 'evidence_spans': spans}):
                raise ValueError('observation_identity_mismatch')
        except ValueError as exc:
            rejected.append({'index': index, 'status': 'rejected', 'reason': str(exc),
                **({'dimension': row['dimension']} if isinstance(row, dict)
                    and isinstance(row.get('dimension'), str) and row['dimension'] in PHASES else {})})
        else:
            good.append(row)
    return good, rejected


def overlaps(left, right):
    return any(a['role'] == b['role'] == 'assistant' and max(a['start'], b['start']) < min(a['end'], b['end'])
        for a in left for b in right)


def legacy_yearnings(source_hash, originals, bindings, modes, current):
    """Preserve independently supported old labels without keeping old expiry."""
    by_id = {row['observation_id']: row for row in originals}
    result = []
    for binding in bindings:
        evidence = by_id[binding['observation_id']]
        if (binding['label'] != 'yearning' or modes.get(binding['observation_id'], 'apply') != 'apply'
                or evidence['actor'] != 'assistant' or evidence['target'] != 'owner'
                or evidence['context'] != 'direct'):
            continue
        spans = evidence['evidence_spans']
        if any(row['dimension'] == 'yearning' and overlaps(spans, row['evidence_spans']) for row in current):
            continue
        row = {'dimension': 'yearning', 'phase': 'felt', 'evidence_spans': spans}
        result.append({**row, 'observation_id': 'rel:' + digest({'source_hash': source_hash, **row}),
            'evidence_hash': evidence['evidence_hash'], 'legacy_observation_id': evidence['observation_id']})
    return result


def memory_direction(row):
    return 'increase' if row['phase'] == 'rise' else 'decrease'


def advance_non_numeric(state, settings, start, end):
    """Persist changes from the real clock; expiry never deletes old receipts."""
    expired = []
    active = state.get('relation_yearnings', {})
    for key, row in list(active.items()):
        if datetime.fromisoformat(row['until']) <= end:
            expired.append({'status': 'expired', 'dimension': 'yearning', 'observation_id': key})
            active.pop(key)
    stance = state.get('relation_stance')
    if stance is not None and end > start:
        hours = (end - start).total_seconds() / 3600
        stance['value'] = .5 + (stance['value'] - .5) * exp2(-hours / settings['stance']['half_life_hours'])
    return expired


def apply_rows(state, settings, rows, event, now, trace):
    """Interpret already validated rows; one same-root directional pulse."""
    applied, results, represented = [], [], []
    key = lambda row: min(span['start'] for span in row['evidence_spans'] if span['role'] == 'assistant')
    rows = sorted(rows, key=key)
    conflicted = {i for i, row in enumerate(rows) for j, other in enumerate(rows)
        if i != j and row['dimension'] == other['dimension'] and row['phase'] != other['phase']
        and overlaps(row['evidence_spans'], other['evidence_spans'])}
    fine = [row for i, row in enumerate(rows) if i not in conflicted and row['dimension'] in YEARNING_REFINERS]
    active = state.setdefault('relation_yearnings', {})
    for i, row in enumerate(rows):
        dimension, phase = row['dimension'], row['phase']
        ref = source_ref('conversation', event['event_id'], event['occurred_at'], row['observation_id'])
        result = {'dimension': dimension, 'phase': phase, 'evidence_hash': row['evidence_hash'], 'source_ref': ref,
            'observation_id': row['observation_id']}
        if 'legacy_observation_id' in row:
            result['legacy_observation_id'] = row['legacy_observation_id']
        if i in conflicted:
            status = 'conflicting_source_phases'
        elif any(dimension == prior['dimension'] and phase == prior['phase']
                and overlaps(row['evidence_spans'], prior['evidence_spans']) for prior in represented):
            status = 'same_cause_already_settled'
        elif dimension == 'yearning' and any(overlaps(row['evidence_spans'], named['evidence_spans']) for named in fine):
            status = 'refined_by_supported_dimension'
        elif dimension == 'yearning':
            active[row['observation_id']] = {'ref': ref,
                'until': (now + timedelta(hours=settings['yearning_hours'])).isoformat(timespec='microseconds')}
            status = 'applied'
        elif dimension == 'relational_stance':
            stance = state.get('relation_stance')
            old = stance['value'] if stance else .5
            step = settings['stance']['step'] * (1 if phase == 'lead' else -1)
            state['relation_stance'] = {'value': min(1., max(0., old + step)), 'ref': ref}
            status = 'applied'
        else:
            value = state['values'][dimension]
            if value is None:
                status = 'unknown_not_updated'
            else:
                rules = settings['needs'][dimension] if dimension in NEEDS else settings['security']
                delta = rules[phase] * (1 if phase == 'rise' else -1)
                after = min(1., max(0., value + delta))
                state['values'][dimension] = after
                if phase == 'satisfy':
                    until = (now + timedelta(hours=rules['buffer_hours'])).isoformat(timespec='microseconds')
                    previous = state['plateaus'].get(dimension)
                    if previous is None or datetime.fromisoformat(previous) < datetime.fromisoformat(until):
                        state['plateaus'][dimension] = until
                    result['buffer_until'] = state['plateaus'][dimension]
                result.update(before=value, after=after)
                trace(dimension, value, after, 'relation_' + phase, ref)
                status = 'applied' if after != value else 'no_change'
        if status in ('applied', 'no_change'):
            applied.append(row)
        if status in ('applied', 'no_change', 'same_cause_already_settled'):
            represented.append(row)
        results.append({**result, 'status': status})
    return applied, results
