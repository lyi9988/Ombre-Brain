"""Source-faithful L1 text for derived retrieval, never a new Memory authority."""
from __future__ import annotations
import re

TEXT_PROJECTION_VERSION = 'memory-l1-retrieval-text-v1'


def value_text(value, names=None):
    if isinstance(value, dict):
        state = value.get('state')
        if state != 'known':
            return str(state or 'unknown')
        value = value.get('value')
    if value is None:
        return 'unknown'
    if names and isinstance(value, str) and value in names:
        return names[value]
    return str(value)


def predicate_text(value, names):
    if not isinstance(value, dict):
        return ''
    op = str(value.get('op') or '')
    if op in {'all', 'any', 'and', 'or', 'not'}:
        arguments = [predicate_text(item, names) for item in value.get('args', [])]
        return op + ' (' + '; '.join(text for text in arguments if text) + ')'
    item = value.get('value')
    rendered = ', '.join(value_text(member, names) for member in item) if isinstance(item, list) else value_text(item, names)
    return ' '.join([str(value.get('field') or ''), op, rendered])


def condition_text(value, names):
    value = value or {}
    parts = [str(value.get('state') or 'unknown')]
    if value.get('text'):
        parts.append(value['text'])
    if value.get('predicate'):
        parts.append(predicate_text(value['predicate'], names))
    for exception in value.get('exceptions', []):
        parts.append('exception: ' + condition_text(exception, names))
    return '; '.join(parts)


def l1_retrieval_blocks(document, *, body=None, title=None):
    """Explicit semantic whitelist; no identity, anchors, hashes, or catalog dump.

    The caller supplies a verified current L1 (including resolved parent
    statements). ``body`` can select a committed content moment without losing
    its parent's factual qualifiers. Units remain whole; consumers own budgets.
    """
    narrative = document.get('narrative') or {}
    title = narrative.get('title') if title is None else title
    body = narrative.get('body') if body is None else body
    rows = []
    if title:
        rows.append('Title: ' + str(title))
    if body:
        rows.append('Content: ' + str(body))
    entities = document.get('entities') or []
    names = {item['entity_id']: str(item.get('name') or 'unknown') for item in entities}
    scenes = {item['scene_id']: str(item.get('title') or 'unknown') for item in document.get('scenes', [])}
    for entity in entities:
        rows.append('Entity: ' + '; '.join([str(entity.get('name') or 'unknown'),
            value_text(entity.get('entity_type')), *map(str, entity.get('aliases') or [])]))
    retrieval = document.get('retrieval') or {}
    for key in ('keywords', 'tags'):
        if retrieval.get(key):
            rows.append(key + ': ' + ', '.join(map(str, retrieval[key])))
    def attributes(values):
        return [str(item.get('definition_id') or '') + '=' + value_text(item.get('value'), names)
                for item in values or []]
    if document.get('attributes'):
        rows.append('Attributes: ' + '; '.join(attributes(document['attributes'])))
    for statement in document.get('statements') or []:
        parts = ['Statement: ' + str(statement.get('content') or ''),
                 'subject: ' + value_text(statement.get('subject_ref'), names)]
        classification = statement.get('classification') or {}
        parts.extend(key + ': ' + value_text(classification[key])
                     for key in ('kind', 'stability', 'modality', 'basis')
                     if classification.get(key) and classification[key].get('state') != 'not_stated')
        if classification.get('topics'):
            parts.append('topics: ' + ', '.join(map(str, classification['topics'])))
        time = statement.get('time') or {}
        for key, fields in (('event', ('start', 'end', 'precision', 'interpretation', 'timezone')),
                            ('applicability', ('start', 'end', 'temporal_scope', 'recurrence_text'))):
            temporal = time.get(key) or {}
            meaningful = [field + '=' + value_text(temporal[field]) for field in fields
                if temporal.get(field) is not None
                and temporal[field] != 'unknown'
                and not (isinstance(temporal[field], dict) and temporal[field].get('state') == 'not_stated')]
            if meaningful:
                parts.append(key + ': ' + ', '.join(meaningful))
            if temporal.get('scene_ref'):
                parts.append('scene: ' + scenes.get(temporal['scene_ref'], 'unknown'))
        parts.append('conditions: ' + condition_text(statement.get('conditions'), names))
        prop = statement.get('property')
        if prop:
            parts.append('property: ' + '; '.join([str(prop.get('key') or ''),
                str(prop.get('polarity') or 'unknown'), value_text(prop.get('value'), names)]))
        parts.extend(attributes(statement.get('attributes')))
        cognition = statement.get('cognition')
        if cognition:
            parts.append('cognition: ' + value_text(cognition.get('role')) + '; ' + value_text(cognition.get('epistemic_status')))
            if cognition.get('about_refs'):
                parts.append('about: ' + ', '.join(names.get(ref, 'unknown') for ref in cognition['about_refs']))
            parts.extend('open question: ' + str(question) for question in cognition.get('open_questions', []))
        rows.append('\n'.join(parts))
    return rows


def l1_retrieval_text(document, *, body=None, title=None):
    return '\n\n'.join(l1_retrieval_blocks(document, body=body, title=title))


class SemanticUnitTooLarge(ValueError):
    def __init__(self, required_chars, max_chars):
        super().__init__('embedding_semantic_unit_exceeds_character_budget')
        self.details = {'reason':str(self), 'required_chars':required_chars, 'max_chars':max_chars,
                        'coverage_status':'incomplete', 'provider_requests':0}


def pack_semantic_blocks(blocks, max_chars):
    """Pack complete paragraphs/qualified statements without slicing content."""
    complete = []
    for block in blocks:
        if block.startswith('Content: '):
            # Blank lines delimit paragraphs; single line breaks may be wraps.
            complete.extend('Content: ' + part.strip() for part in re.split(r'\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*',
                block[len('Content: '):]) if part.strip())
        else:
            complete.append(block)
    oversized = max((len(block) for block in complete), default=0)
    if oversized > max_chars:
        raise SemanticUnitTooLarge(oversized, max_chars)
    packed, current = [], ''
    for block in complete:
        candidate = current + '\n\n' + block if current else block
        if len(candidate) <= max_chars:
            current = candidate
        else:
            packed.append(current)
            current = block
    if current:
        packed.append(current)
    return packed


def pack_source_units(units, max_chars):
    """Keep original parent/moment identity beside packed vector storage IDs."""
    packed = []
    for unit in units:
        if not unit.get('semantic_blocks'):
            packed.append(unit)
            continue
        for index, text in enumerate(pack_semantic_blocks(unit['semantic_blocks'], max_chars)):
            packed.append({**unit, 'text':text, 'source_origin_unit_id':unit['source_unit_id'],
                'source_unit_id':unit['source_unit_id'] if index == 0 else unit['source_unit_id'] + ':l1-text:' + str(index + 1)})
    return packed
