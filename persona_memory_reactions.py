"""Approved first memory-reaction rule; no source reads or independent state."""
from __future__ import annotations

from datetime import datetime
import math


RULES_VERSION = 'memory-reaction-r1'
EVOKE = {'light': .02, 'clear': .05, 'strong': .08}
EASE = {'light': .05, 'clear': .10, 'strong': .15}
REPEAT_HOURS = 1
REPEAT_FACTOR = .5


def unit(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('finite unit value or explicit unknown required')
    return float(value)


def instant(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('aware reaction time required')
    return stamp


def repeat_factor(occurred_at, previous_at=None):
    now = instant(occurred_at)
    if previous_at is None:
        return 1.0
    elapsed = (now - instant(previous_at)).total_seconds()
    if elapsed < 0:
        raise ValueError('late memory reaction requires chronological reconciliation')
    return REPEAT_FACTOR if elapsed < REPEAT_HOURS * 3600 else 1.0


def apply_dimension(value, *, effect, direction, strength, factor=1.0, floor=None,
                    emotion_inertia=False, reaction='new_reaction'):
    """Return one calculation; the common ledger owns persistence/idempotency.

    ``floor`` is the existing dimension rule's effective floor, supplied by its
    owner. Its absence never creates a synthetic neutral value.
    """
    value = unit(value)
    if reaction not in {'new_reaction', 'new_understanding', 'continuing', 'uncertain', 'none'}:
        raise ValueError('known memory reaction category required')
    if effect not in {'evoke', 'ease'} or direction not in {'increase', 'decrease', 'unknown'}:
        raise ValueError('explicit effect and direction required')
    if strength not in {*EVOKE, 'unknown'} or type(factor) not in (float, int) or not math.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError('approved strength and repeat factor required')
    result = {'rules_version': RULES_VERSION, 'before': value, 'after': value,
        'repeat_factor': float(factor), 'effect': effect, 'direction': direction, 'strength': strength}
    if reaction in {'continuing', 'uncertain', 'none'}:
        return {**result, 'status': 'no_new_pulse', 'reason': reaction}
    if value is None:
        return {**result, 'status': 'unknown_not_updated', 'reason': 'current_value_unknown'}
    if direction == 'unknown' or strength == 'unknown':
        return {**result, 'status': 'unknown_not_updated', 'reason': 'reaction_unknown'}
    if effect == 'ease':
        if direction != 'decrease':
            raise ValueError('ease is proportional relief above an existing floor')
        floor = unit(floor)
        if floor is None:
            return {**result, 'status': 'unknown_not_updated', 'reason': 'effective_floor_unknown'}
        delta = -max(0.0, value - floor) * EASE[strength] * factor
    else:
        delta = EVOKE[strength] * factor * (1 if direction == 'increase' else -1)
        if emotion_inertia:
            delta *= 1 - value if delta > 0 else value
    after = min(1.0, max(0.0, value + delta))
    actual_delta = after - value
    return {**result, 'after': after, 'requested_delta': delta, 'delta': actual_delta,
        'status': 'applied' if actual_delta else 'no_change'}
