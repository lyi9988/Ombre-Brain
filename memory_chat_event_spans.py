"""Exact model-selected event spans over unchanged, program-bound L0 sources."""
from __future__ import annotations

from copy import deepcopy
import re

from memory_l1_service import L1OwnerError


def _fail(code, path):
    raise L1OwnerError(code, field_path=path)


def _source_index(value, path):
    match = re.fullmatch(r'source:([1-9][0-9]*)', value) if isinstance(value, str) else None
    if match is None:
        _fail('event_source_reference_invalid', path)
    return int(match[1]) - 1


def _bounds(start, end, path, *, length=None):
    if (type(start) is not int or type(end) is not int or not 0 <= start < end
            or (length is not None and end > length)):
        _fail('event_source_range_invalid', path)
    return start, end


def _event(value, path):
    if (not isinstance(value, dict) or set(value) != {'point', 'anchors'}
            or not isinstance(value['point'], str) or not value['point'].strip()
            or not isinstance(value['anchors'], list)):
        _fail('event_selection_invalid', path)
    if not value['anchors']:
        _fail('event_evidence_required', path + '.anchors')


def parse_event_selection(raw_dict, projected_sources):
    """Resolve quotes uniquely within visible views, returning absolute codepoints.

    ``context`` is validated as commentary and discarded. A projected
    ``input_range`` must cover its text exactly; its start is the outer offset.
    Original source:N identifiers remain unchanged, even for sparse views.
    """
    if (not isinstance(raw_dict, dict) or set(raw_dict) != {'context', 'memories'}
            or not isinstance(raw_dict['context'], str) or not isinstance(raw_dict['memories'], list)):
        _fail('event_selection_invalid', 'selection')
    if not raw_dict['memories']:
        return []
    if not isinstance(projected_sources, list):
        _fail('event_evidence_required', 'source_texts')
    visible = {}
    for index, row in enumerate(projected_sources):
        path = f'source_texts.{index}'
        if not isinstance(row, dict) or not isinstance(row.get('text'), str) or not row['text']:
            _fail('event_source_view_invalid', path)
        source = row.get('binding_id')
        _source_index(source, path + '.binding_id')
        if source in visible:
            _fail('event_source_reference_ambiguous', path + '.binding_id')
        extent = row.get('input_range', {'start': 0, 'end': len(row['text'])})
        if not isinstance(extent, dict) or set(extent) != {'start', 'end'}:
            _fail('event_source_range_invalid', path + '.input_range')
        start, end = _bounds(extent['start'], extent['end'], path + '.input_range')
        if end - start != len(row['text']):
            _fail('event_source_range_invalid', path + '.input_range')
        if 'source_length' in row and (type(row['source_length']) is not int or end > row['source_length']):
            _fail('event_source_range_invalid', path + '.source_length')
        visible[source] = row['text'], start
    events = []
    for index, memory in enumerate(raw_dict['memories']):
        path = f'memories.{index}'
        _event(memory, path)
        anchors, seen = [], set()
        for number, anchor in enumerate(memory['anchors']):
            field = f'{path}.anchors.{number}'
            if not isinstance(anchor, dict) or set(anchor) != {'source', 'quote'}:
                _fail('event_selection_invalid', field)
            source = anchor['source']
            _source_index(source, field + '.source')
            if source not in visible:
                _fail('event_source_not_visible', field + '.source')
            quote = anchor['quote']
            if not isinstance(quote, str) or not quote.strip():
                _fail('event_evidence_required', field + '.quote')
            text, offset = visible[source]
            start = text.find(quote)
            if start < 0 or text.find(quote, start + 1) >= 0:
                _fail('event_quote_not_unique', field + '.quote')
            key = source, offset + start, offset + start + len(quote)
            if key in seen:
                _fail('event_anchor_duplicate', field)
            seen.add(key)
            anchors.append({'source': source, 'start': key[1], 'end': key[2]})
        anchors.sort(key=lambda row: (_source_index(row['source'], path), row['start'], row['end']))
        events.append({'point': memory['point'], 'anchors': anchors})
    return events


def event_source_views(event, original_sources):
    """Return one independent range view per anchor; never join intervening text.

    Fragments map each NEW ``source`` binding to its original zero-based source
    index and absolute start/end. Source text, identity, role and version remain
    original. Existing outer ``_model_range``/exclusion bounds are enforced.
    """
    _event(event, 'event')
    if not isinstance(original_sources, list):
        _fail('event_evidence_required', 'sources')
    selected, seen = [], set()
    for index, anchor in enumerate(event['anchors']):
        path = f'event.anchors.{index}'
        if not isinstance(anchor, dict) or set(anchor) != {'source', 'start', 'end'}:
            _fail('event_selection_invalid', path)
        original_index = _source_index(anchor['source'], path + '.source')
        if original_index >= len(original_sources):
            _fail('event_source_not_visible', path + '.source')
        source = original_sources[original_index]
        if (not isinstance(source, dict) or not isinstance(source.get('text'), str)
                or not source['text'] or source.get('_model_excluded')):
            _fail('event_source_not_visible', path + '.source')
        start, end = _bounds(anchor['start'], anchor['end'], path, length=len(source['text']))
        outer = source.get('_model_range', {'start': 0, 'end': len(source['text'])})
        if not isinstance(outer, dict) or set(outer) != {'start', 'end'}:
            _fail('event_source_range_invalid', path)
        low, high = _bounds(outer['start'], outer['end'], path, length=len(source['text']))
        if start < low or end > high:
            _fail('event_source_range_invalid', path)
        key = original_index, start, end
        if key in seen:
            _fail('event_anchor_duplicate', path)
        seen.add(key)
        selected.append(key)
    views, fragments = [], []
    for original_index, start, end in sorted(selected):
        source = deepcopy(original_sources[original_index])
        source['_model_range'] = {'start': start, 'end': end}
        views.append(source)
        fragments.append({'source': f'source:{len(views)}', 'original_index': original_index,
                          'start': start, 'end': end})
    return views, fragments


def event_envelope(engine, source_views, fragments, *, core_source=None, core_range=None):
    """Use the formal physical-slicing path and return its unchanged tuple.

    Default core is the first selected span of this independent event. To retain
    an outer segment's core/context distinction, pass its ORIGINAL source:N and
    absolute core_range; these are mapped to an intersecting selected view.
    A context-only event then raises ``event_core_not_observed``.
    """
    if not source_views or len(source_views) != len(fragments):
        _fail('event_evidence_required', 'sources')
    for index, (source, fragment) in enumerate(zip(source_views, fragments)):
        if (not isinstance(fragment, dict)
                or set(fragment) != {'source', 'original_index', 'start', 'end'}
                or fragment['source'] != f'source:{index + 1}'
                or type(fragment['original_index']) is not int or fragment['original_index'] < 0):
            _fail('event_source_view_invalid', 'fragments')
        _bounds(fragment['start'], fragment['end'], 'fragments', length=len(source['text']))
        if source.get('_model_range') != {'start': fragment['start'], 'end': fragment['end']}:
            _fail('event_source_range_invalid', 'sources')
    core = fragments[0]
    extent = {'start': core['start'], 'end': core['end']}
    if (core_source is None) != (core_range is None):
        _fail('event_source_range_invalid', 'core')
    if core_source is not None:
        original_index = _source_index(core_source, 'core.source')
        if not isinstance(core_range, dict) or set(core_range) != {'start', 'end'}:
            _fail('event_source_range_invalid', 'core.range')
        start, end = _bounds(core_range['start'], core_range['end'], 'core.range')
        matches = [row for row in fragments if row['original_index'] == original_index
                   and row['start'] < end and start < row['end']]
        if not matches:
            _fail('event_core_not_observed', 'core')
        core = matches[0]
        extent = {'start': max(start, core['start']), 'end': min(end, core['end'])}
    segment = {'core_source': core['source'], 'core_range': extent,
               'views': {str(index): {'start': row['start'], 'end': row['end']}
                         for index, row in enumerate(fragments)}}
    from memory_l1_runtime import MemoryL1Runtime
    result = MemoryL1Runtime._memory_l1_daily_context(engine, [],
        attachment_sources=deepcopy(source_views), segment_context=segment)
    projected = result[0]['source_texts']
    if len(projected) != len(fragments):
        _fail('event_source_view_invalid', 'source_texts')
    for row, source, fragment in zip(projected, source_views, fragments):
        if (row['text'] != source['text'][fragment['start']:fragment['end']]
                or row['input_range'] != {'start': fragment['start'], 'end': fragment['end']}):
            _fail('event_source_range_invalid', 'source_texts')
    return result
