"""Lossless semantic shorthand; the canonical L1 schema remains authoritative.

Only explicit semantic-v1 documents are expanded. Missing required judgments
are errors, not evidence that a source omitted a fact. No model calls or writes.
"""
from copy import deepcopy
from datetime import datetime

from memory_l1_schema import catalog_with_defaults, statement_template
from memory_l1_service import L1OwnerError


def _shape(value, required, optional=(), path='document'):
    if (not isinstance(value, dict) or set(required) - set(value)
            or set(value) - set(required) - set(optional)):
        raise L1OwnerError('semantic_fields_required', field_path=path)
    return deepcopy(value)


def _array(value, path):
    if not isinstance(value, list):
        raise L1OwnerError('array_required', field_path=path)
    return value


def _value(value, evidence):
    if isinstance(value, dict):
        result = _shape(value, ('state',), ('value', 'evidence_refs'))
        if result['state'] == 'known' and 'value' not in result:
            raise L1OwnerError('semantic_fields_required', field_path='value')
        result.setdefault('value', None)
        result.setdefault('evidence_refs', deepcopy(evidence))
        return result
    return {'state': 'not_stated' if value is None else 'known',
            'value': value, 'evidence_refs': deepcopy(evidence)}


def _attributes(items, evidence):
    result = []
    for item in _array(items, 'attributes'):
        row = _shape(item, ('definition_id', 'value'), ('evidence_refs',))
        row.setdefault('evidence_refs', deepcopy(evidence))
        row['value'] = _value(row['value'], row['evidence_refs'])
        result.append(row)
    return result


def _condition(value, evidence, *, exception=False):
    if value is None:
        value = {'state': 'not_stated'}
    elif isinstance(value, str):
        value = {'state': 'known', 'text': value}
    result = _shape(value, ('state',), ('text', 'predicate', 'evidence_refs') +
                    (() if exception else ('exceptions',)), 'conditions')
    result.setdefault('text', None)
    result.setdefault('predicate', None)
    result.setdefault('evidence_refs', deepcopy(evidence))
    if not exception:
        result['exceptions'] = [_condition(row, result['evidence_refs'], exception=True)
                                for row in _array(result.get('exceptions', []), 'exceptions')]
    return result


def _time(value, evidence):
    defaults = statement_template()['time']
    if value is None:
        return defaults
    result = _shape(value, ('event', 'applicability'), path='time')
    if result['event'] is None:
        result['event'] = defaults['event']
    else:
        event = _shape(result['event'], ('precision', 'interpretation'),
                       ('start', 'end', 'timezone'), path='time.event')
        for key in ('start', 'end', 'timezone'):
            event[key] = _value(event[key], evidence) if key in event else deepcopy(defaults['event'][key])
        result['event'] = event
    if result['applicability'] is None:
        result['applicability'] = defaults['applicability']
    else:
        scope = _shape(result['applicability'], ('temporal_scope',),
                       ('start', 'end', 'scene_ref', 'recurrence_text'), 'time.applicability')
        scope.setdefault('scene_ref', None)
        scope.setdefault('recurrence_text', None)
        for key in ('start', 'end'):
            scope[key] = _value(scope[key], evidence) if key in scope else deepcopy(defaults['applicability'][key])
        result['applicability'] = scope
    return result


def semantic_sources(snapshots, bindings, *, chat_view=False, source_timezone=None):
    """Project source evidence for the model; keep identity/anchors in program custody."""
    if len(snapshots) != len(bindings):
        raise L1OwnerError('source_binding_count_mismatch', field_path='sources')
    result = [{'binding_id': binding['binding_id'], 'text': source['text'],
             'author': source.get('author'), 'participant': deepcopy(source.get('participant')),
             'uploader': deepcopy(source.get('uploader')),
             'source_group_id': (source.get('source_group') or {}).get('source_group_id'),
             'author_ref': binding.get('author_ref'), 'speaker_ref': binding.get('speaker_ref'),
             'source_recorded_at': source.get('recorded_at')}
            for source, binding in zip(snapshots, bindings)]
    if chat_view:
        for source in result:
            if source['source_recorded_at']:
                stamp = datetime.fromisoformat(source['source_recorded_at'].replace('Z', '+00:00'))
                if source_timezone is not None:
                    stamp = stamp.astimezone(source_timezone)
                source['source_recorded_at'] = stamp.isoformat(timespec='minutes')
    return result


def assemble_document(document, *, narrative=None, require_keywords=False):
    """Expand a model draft before existing identity binding and full validation.

    Scalar/null are explicit known/not-stated judgments. Evidence inheritance
    is document-local shorthand, never new source authority. A full legacy
    document passes unchanged to its original validator.
    """
    if require_keywords:
        retrieval = document.get('retrieval') if isinstance(document, dict) else None
        if not isinstance(retrieval, dict) or 'keywords' not in retrieval:
            raise L1OwnerError('semantic_fields_required', field_path='retrieval.keywords')
        keywords = _array(retrieval['keywords'], 'retrieval.keywords')
        if not keywords or any(not isinstance(word, str) or not word.strip() for word in keywords):
            raise L1OwnerError('nonempty_keywords_required', field_path='retrieval.keywords')
    if not isinstance(document, dict) or 'format' not in document:
        return deepcopy(document)
    doc = _shape(document, ('format', 'entities', 'statements'),
                 ('entry', 'narrative', 'scenes', 'retrieval', 'relations', 'attributes'))
    if doc.pop('format') != 'semantic-v1':
        raise L1OwnerError('unsupported_semantic_format', field_path='format')
    if narrative is not None:
        proposed = _shape(doc.get('narrative', {}), (), ('title', 'body'), 'narrative')
        doc['narrative'] = deepcopy(narrative)
        if not doc['narrative'].get('title'):
            doc['narrative']['title'] = proposed.get('title', '')
    elif 'narrative' not in doc:
        raise L1OwnerError('semantic_fields_required', field_path='narrative')
    doc.setdefault('entry', {'form': 'episode'})
    doc.setdefault('retrieval', {'keywords': [], 'tags': []})
    doc['retrieval'] = _shape(doc['retrieval'], (), ('keywords', 'tags'), 'retrieval')
    doc['retrieval'].setdefault('keywords', [])
    doc['retrieval'].setdefault('tags', [])
    for index, item in enumerate(_array(doc['entities'], 'entities')):
        row = _shape(item, ('name', 'entity_type'), ('entity_id', 'aliases', 'evidence_refs'), 'entities')
        row.setdefault('entity_id', f'e{index + 1}')
        row.setdefault('aliases', [])
        row['entity_type'] = _value(row['entity_type'], row.pop('evidence_refs', []))
        doc['entities'][index] = row
    for index, item in enumerate(_array(doc['statements'], 'statements')):
        row = _shape(item, ('content', 'subject_ref', 'classification', 'time', 'conditions', 'evidence_refs'),
                     ('statement_id', 'property', 'role_hint', 'attributes', 'cognition'), 'statements')
        row.setdefault('statement_id', f's{index + 1}')
        evidence = _array(row['evidence_refs'], 'statements.evidence_refs')
        row['subject_ref'] = _value(row['subject_ref'], evidence)
        classification = _shape(row['classification'], ('kind', 'stability', 'modality', 'basis'), ('topics',), 'classification')
        for key in ('kind', 'stability', 'modality', 'basis'):
            classification[key] = _value(classification[key], evidence)
        classification.setdefault('topics', [])
        row['classification'] = classification
        row['time'] = _time(row['time'], evidence)
        row['conditions'] = _condition(row['conditions'], evidence)
        row['role_hint'] = _value(row.get('role_hint'), [])
        row['attributes'] = _attributes(row.get('attributes', []), evidence)
        row.setdefault('property', None)
        if row['property'] is not None:
            prop = _shape(row['property'], ('key', 'value', 'polarity'), ('evidence_refs',), 'property')
            prop.setdefault('evidence_refs', deepcopy(evidence))
            prop['value'] = _value(prop['value'], prop['evidence_refs'])
            row['property'] = prop
        if row.get('cognition') is not None:
            cog = _shape(row['cognition'], ('role', 'epistemic_status'), ('about_refs', 'open_questions'), 'cognition')
            for key in ('role', 'epistemic_status'):
                cog[key] = _value(cog[key], evidence)
            cog.setdefault('about_refs', [])
            cog.setdefault('open_questions', [])
            row['cognition'] = cog
        doc['statements'][index] = row
    doc.setdefault('scenes', [])
    for index, item in enumerate(_array(doc['scenes'], 'scenes')):
        row = _shape(item, ('title', 'statement_refs'), ('scene_id', 'attributes', 'source_refs', 'source_ranges'), 'scenes')
        row.setdefault('scene_id', f'c{index + 1}')
        row['attributes'] = _attributes(row.get('attributes', []), [])
        doc['scenes'][index] = row
    doc.setdefault('relations', [])
    for index, item in enumerate(_array(doc['relations'], 'relations')):
        row = _shape(item, ('from', 'to', 'kind', 'evidence_refs'), ('relation_id',), 'relations')
        row.setdefault('relation_id', f'r{index + 1}')
        doc['relations'][index] = row
    doc['attributes'] = _attributes(doc.get('attributes', []), [])
    return doc


def semantic_catalog(catalog):
    """One derived vocabulary; retain owner semantic definitions and options."""
    catalog = catalog_with_defaults(catalog)
    definitions = {(row['axis'], row['id']): row for row in catalog['definitions']}
    result = {}
    for axis, values in catalog.items():
        if axis in ('definitions', 'attributes'):
            continue
        result[axis] = {key: {name: deepcopy(value) for name, value in
                            definitions.get((axis, key), {}).items()
                            if name not in ('axis', 'id') and value not in ('', None, [], {})}
                        for key in values}
    result['attributes'] = {row['id']: {key: deepcopy(value) for key, value in row.items()
                                      if key != 'id' and value not in ('', None, [], {})}
                            for row in catalog['attributes']}
    return result


def semantic_contract(source_ids, *, catalog, initial_chat=False, require_keywords=False, identity=None):
    """Compact structural notation, not an alternate persistence schema."""
    result = {
        'format': 'semantic-v1', 'source_ids': list(source_ids),
        'catalog': semantic_catalog(catalog),
        'document': {
            'required': ['format', 'entities', 'statements'],
            'entities': 'Entity[]', 'statements': 'Statement[]',
            'optional': {'entry': {'form': 'episode|reference'}, 'narrative': {'title': 'string', 'body': 'string'},
                         'scenes': 'Scene[]', 'relations': 'Relation[]', 'attributes': 'Attribute[]',
                         'retrieval': {'keywords': 'string[]', 'tags': 'string[]'}},
        },
        'types': {
            'Value': 'scalar|null|{state:known|not_stated|unresolved|not_applicable|unbounded, value?:scalar|null, evidence_refs?:source_id[]}',
            'Entity': {'name': 'string', 'entity_type': 'Value:catalog.entity_types',
                       'optional': {'entity_id': 'string', 'aliases': 'string[]', 'evidence_refs': 'source_id[]'}},
            'Statement': {'content': 'string', 'subject_ref': 'Value:entity_id',
                          'classification': {'kind': 'Value:catalog.kinds', 'stability': 'Value:catalog.stabilities',
                                             'modality': 'Value:catalog.modalities', 'basis': 'Value:catalog.bases',
                                             'topics?': 'catalog.topics[]'},
                          'time': 'Time|null', 'conditions': 'Condition|string|null', 'evidence_refs': 'source_id[]',
                          'optional': {'statement_id': 'string', 'property': 'Property|null', 'attributes': 'Attribute[]',
                                       'role_hint': 'Value:core|detail', 'cognition': 'Cognition|null'}},
            'Time': {'event': 'EventTime|null', 'applicability': 'Applicability|null'},
            'EventTime': {'precision': 'instant|day|month|year|approximate|unknown',
                          'interpretation': 'instant|duration|uncertainty_window|unknown',
                          'start?': 'Value:ISO8601', 'end?': 'Value:ISO8601', 'timezone?': 'Value:string'},
            'Applicability': {'temporal_scope': 'episode|time_range|recurring|unknown', 'start?': 'Value:ISO8601',
                              'end?': 'Value:ISO8601', 'scene_ref?': 'scene_id|null', 'recurrence_text?': 'string|null'},
            'Condition': {'state': 'known|not_stated|unresolved|not_applicable', 'text?': 'string|null',
                          'predicate?': 'Predicate|null', 'evidence_refs?': 'source_id[]', 'exceptions?': 'ConditionException[]'},
            'ConditionException': {'state': 'known|not_stated|unresolved|not_applicable', 'text?': 'string|null',
                                   'predicate?': 'Predicate|null', 'evidence_refs?': 'source_id[]'},
            'Predicate': 'all|any:{op,args:Predicate[]}; not:{op,args:[Predicate]}; eq|in|lt|le|gt|ge:{op,field:catalog.attributes,value}',
            'Property': {'key': 'catalog.attributes', 'value': 'Value', 'polarity': 'affirmed|negated|qualified|unknown',
                         'evidence_refs?': 'source_id[]'},
            'Attribute': {'definition_id': 'catalog.attributes', 'value': 'Value', 'evidence_refs?': 'source_id[]'},
            'Cognition': {'role': 'Value:catalog.cognition_roles', 'epistemic_status': 'Value:catalog.cognition_statuses',
                          'about_refs?': 'entity_id[]', 'open_questions?': 'string[]'},
            'Scene': {'title': 'string', 'statement_refs': 'statement_id[]', 'scene_id?': 'string',
                      'attributes?': 'Attribute[]', 'source_refs?': 'source_id[]',
                      'source_ranges?': '{source_ref,start:codepoint,end:codepoint}[]'},
            'Relation': {'from': '{object_type:memory|scene|statement,object_id}',
                         'to': '{object_type:memory|scene|statement,object_id}', 'kind': 'catalog.relation_types',
                         'evidence_refs': 'source_id[]', 'relation_id?': 'string'},
        },
    }
    if require_keywords:
        result['document']['required'].append('retrieval')
        result['document']['retrieval'] = result['document']['optional'].pop('retrieval')
        result['document']['retrieval']['keywords'] = 'non-empty string[];由模型根据材料提供'
    if initial_chat:
        # Only the model-facing vocabulary is abbreviated. Canonical catalog,
        # validation, persisted Schema and owner-edited definitions stay intact.
        from memory_system_config import defaults
        baseline = semantic_catalog(defaults()['catalog'])
        view = result['catalog']
        for kind in ('revises', 'supersedes', 'split_from', 'merged_from'):
            view.get('relation_types', {}).pop(kind, None)
        relation_labels = {'same_event': '同一事件', 'supplements': '补充',
                           'limits': '限定', 'possible_conflict': '可能冲突'}
        for axis, rows in view.items():
            if axis == 'attributes':
                continue
            for key, row in list(rows.items()):
                standard = baseline.get(axis, {}).get(key, {})
                label = row.get('label') or standard.get('label') or relation_labels.get(key, key)
                if identity and axis in ('subjects', 'entity_types') and key in ('user', 'assistant'):
                    label = identity[key]['name']
                extra = {name: deepcopy(value) for name, value in row.items()
                         if name != 'label' and value != standard.get(name)}
                rows[key] = {'label': label, **extra} if extra else label
    return result
