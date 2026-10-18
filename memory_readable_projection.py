"""Readable memory payloads; custody and retrieval fields stay out of prose."""
from __future__ import annotations

import json
import re
from datetime import datetime


def _known(value):
    return value.get('value') if isinstance(value, dict) and value.get('state') == 'known' else None


def _when(value):
    raw = _known(value)
    if not raw:
        return ''
    try:
        parsed = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    except ValueError:
        return str(raw)
    # Preserve recorded precision and timezone; these are event information,
    # rather than an index timestamp to abbreviate for display.
    return str(raw).replace('T', ' ')


def prose(value):
    """Read existing prose, never serialize an index/summary object to chat."""
    text = str(value or '').strip()
    if not text:
        return ''
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # A cut-off JSON summary is not a readable paragraph either.
        return '' if re.match(r'^\{\s*"|^\[\s*(?:\{|\")', text) else text
    if isinstance(parsed, str):
        return parsed.strip()
    if not isinstance(parsed, dict):
        return ''
    narrative = parsed.get('narrative')
    if isinstance(narrative, dict) and isinstance(narrative.get('body'), str):
        return narrative['body'].strip()
    for key in ('full_injection_text', 'body', 'content', 'text'):
        if isinstance(parsed.get(key), str) and parsed[key].strip():
            return parsed[key].strip()
    return ''


def _condition_text(condition, *, exception=False):
    """None means the predicate has no supplied, faithful natural wording."""
    condition = condition or {}
    text = prose(condition.get('text'))
    if condition.get('predicate') is not None and not text:
        return None
    parts = []
    if text:
        parts.append(('以下情况除外：' if exception else '适用条件是：') + text)
    elif condition.get('state') == 'unresolved':
        parts.append('例外情况尚未确定。' if exception else '适用条件尚未确定。')
    for value in condition.get('exceptions') or []:
        lines = _condition_text(value, exception=True)
        if lines is None:
            return None
        parts.extend(lines)
    return parts


def _unknown_bound(value, label):
    state = (value or {}).get('state')
    return {
        'unbounded': f'未限定{label}',
        'unresolved': f'{label}尚未确定',
        'not_applicable': f'{label}不适用',
    }.get(state, f'{label}未说明')


def _time_text(item, document):
    time = item.get('time') or {}
    event, scope = time.get('event') or {}, time.get('applicability') or {}
    start, end = _when(event.get('start')), _when(event.get('end'))
    interpretation = event.get('interpretation')
    parts = []
    if interpretation == 'uncertainty_window' and (start or end):
        bounds = f'{start}至{end}' if start and end else (f'{start}之后' if start else f'{end}之前')
        parts.append(f'事件可能发生在{bounds}，具体时间尚未确定。')
        if not start or not end:
            parts.append(_unknown_bound(event.get('start') if not start else event.get('end'),
                '时间范围的起点' if not start else '时间范围的终点') + '。')
    elif interpretation == 'duration' and (start or end):
        if start and end:
            parts.append(f'这段经历的时间为{start}至{end}。')
        else:
            parts.append((f'这段经历从{start}开始，' if start else f'这段经历在{end}结束，')
                + _unknown_bound(event.get('end') if start else event.get('start'),
                    '结束时间' if start else '开始时间') + '。')
    elif interpretation == 'instant' and start:
        parts.append(f'这发生在{start}。')
        if end and end != start:
            parts.append(f'另有时间记为{end}，二者关系未明确。')
    elif start or end:
        recorded = '；'.join(value for value in
            (f'时间起点记为{start}' if start else '', f'时间终点记为{end}' if end else '') if value)
        parts.append(recorded + '；这些时间的含义未明确。')
    elif any((event.get(key) or {}).get('state') == 'unresolved' for key in ('start', 'end')):
        parts.append('事件发生时间尚未确定。')
    timezone = _known(event.get('timezone'))
    if timezone and (start or end):
        parts.append(f'上述事件时间的时区为{timezone}。')
    if event.get('precision') == 'approximate' and (start or end):
        parts.append('上述事件时间是约略时间。')

    first, last = _when(scope.get('start')), _when(scope.get('end'))
    if first and last:
        parts.append(f'适用时间为{first}至{last}。')
    elif first or last:
        parts.append((f'从{first}开始适用，' if first else f'适用到{last}，')
            + _unknown_bound(scope.get('end') if first else scope.get('start'),
                '结束时间' if first else '开始时间') + '。')
    elif scope.get('temporal_scope') == 'time_range':
        parts.append(_unknown_bound(scope.get('start'), '适用开始时间') + '，'
            + _unknown_bound(scope.get('end'), '适用结束时间') + '。')
    elif any((scope.get(key) or {}).get('state') == 'unresolved' for key in ('start', 'end')):
        parts.append('适用时间范围尚未确定。')
    recurrence = prose(scope.get('recurrence_text'))
    if recurrence:
        parts.append('重复适用的规律是：' + recurrence)
    elif scope.get('temporal_scope') == 'recurring':
        parts.append('具体重复规律未说明。')
    if scope.get('temporal_scope') == 'episode':
        scene = next((row for row in document.get('scenes') or [] if row.get('scene_id') == scope.get('scene_ref')), {})
        title = prose(scene.get('title'))
        parts.append(f'这只涉及“{title}”这段经历。' if title else '这只涉及那次经历。')
    return parts


def _statement_unit(item, document):
    text = prose(item.get('content'))
    if not text or item.get('usage') == 'excluded':
        return None
    condition = _condition_text(item.get('conditions'))
    if condition is None:
        return None
    subject = item.get('subject_ref') or {}
    ident = _known(subject)
    entity = next((row for row in document.get('entities') or [] if row.get('entity_id') == ident), {})
    name = prose(entity.get('name'))
    if ident and not name:
        return None  # A known reference is not an unknown or anonymous subject.
    # A subject reference identifies whom a claim concerns, not who said it.
    # Quoting the original wording does not invent a third party's speech act
    # or reinterpret their first person as the current assistant.
    text = f'关于{name}：“{text}”' if name else ('主体尚未确定：' if subject.get('state') == 'unresolved'
        else '这条记录未明确主体：') + f'“{text}”'
    classification = item.get('classification') or {}
    modality, basis = _known(classification.get('modality')), _known(classification.get('basis'))
    prefix = {'request': '当时提出的请求是：', 'wish': '当时表达的愿望是：', 'plan': '当时的计划是：',
        'commitment': '当时作出的约定是：', 'question': '当时提出的问题是：', 'hypothesis': '当时的假设是：'}.get(modality)
    if not prefix and basis in {'interpretation', 'hypothesis'}:
        prefix = '当时的理解是：' if basis == 'interpretation' else '当时的假设是：'
    if prefix:
        text = prefix + text
    elif item.get('usage') in {'historical_evidence', 'relationship_reference'}:
        text = '当时，' + text
    if item.get('usage') == 'unresolved_evidence':
        text = '尚未确定的是：' + text
    return '\n'.join([text, *_time_text(item, document), *condition])


def qualified_units(usage, original):
    """Return prose units whose subjects and claim limits cannot be split off."""
    if usage.get('state') == 'excluded':
        return []
    if usage.get('state') == 'legacy':
        text = prose(original)
        return [text] if text else []
    document = (usage.get('record') or {}).get('document') or {}
    uses = [item for item in usage.get('items') or [] if item.get('usage') != 'excluded']
    rendered = [(item, _statement_unit(item, document)) for item in uses]
    units = list(dict.fromkeys(unit for _, unit in rendered if unit))
    allowed = {item.get('statement_id') for item, unit in rendered if unit and item.get('statement_id')}
    refs = {item.get('statement_id') for item in document.get('statements') or []}
    narrative = prose(original)
    entities = {row.get('entity_id'): row for row in document.get('entities') or []}
    # Subject references do not identify a narrator. When only other people
    # are named, retain their first person only inside the attributed statement
    # unit, rather than projecting it as the assistant's own narrative voice.
    subject_types = {_known((entities.get(_known(item.get('subject_ref'))) or {}).get('entity_type'))
        for item, unit in rendered if unit}
    other_person_first_person = bool(re.match(r'^[\s“"「『]*(?:我|我们|本人|咱们)', narrative)
        and 'assistant' not in subject_types and subject_types.intersection({'user', 'third_party'}))
    if (usage.get('narrative_allowed') and narrative and not other_person_first_person
            and refs and None not in refs and refs <= allowed):
        # Preserve the complete narrative and all per-claim qualifications as a
        # single unit; packing must never select the narrative on its own.
        # Complete statement units remain alternatives when that full unit is
        # too large. Packing skips them when the full unit already covers them.
        return list(dict.fromkeys(['\n\n'.join(dict.fromkeys([narrative, *units])), *units]))
    return units


def qualified_prose(usage, original):
    """Unbudgeted compatibility entry point, including cognition projection."""
    selected = []
    for unit in qualified_units(usage, original):
        if unit not in '\n\n'.join(selected):
            selected.append(unit)
    return '\n\n'.join(selected)


def fit_units(units, budget, wrap=None):
    """Pack whole semantic units and account for one shared owner template."""
    from utils import count_tokens_approx
    selected = []
    for unit in dict.fromkeys(str(value).strip() for value in units if value and str(value).strip()):
        if unit in '\n\n'.join(selected):
            continue
        candidate = '\n\n'.join([*selected, unit])
        rendered = wrap(candidate) if wrap is not None else candidate
        if rendered and count_tokens_approx(rendered) <= max(0, budget):
            selected.append(unit)
    if not selected:
        return ''
    body = '\n\n'.join(selected)
    return wrap(body) if wrap is not None else body


def fit_paragraphs(text, budget):
    """Keep complete existing paragraphs; never emit a clipped JSON/string."""
    from utils import count_tokens_approx
    text = str(text or '').strip()
    if count_tokens_approx(text) <= max(0, budget):
        return text
    selected = []
    for paragraph in re.split(r'\n\s*\n', text):
        candidate = '\n\n'.join([*selected, paragraph])
        if count_tokens_approx(candidate) > max(0, budget):
            continue
        selected.append(paragraph)
    return '\n\n'.join(selected)
