"""Model-facing constraints derived from the same revisioned L1 catalogue."""
from __future__ import annotations

from copy import deepcopy

from memory_l1_schema import catalog_with_defaults
from memory_l1_wire_schema import model_document_schema as base_model_schema


def model_document_schema(source_ids=None, *, catalog, entry_forms=None):
    schema = base_model_schema(source_ids, entry_forms=entry_forms)
    vocabulary = catalog_with_defaults(catalog)
    definitions = schema['$defs']

    def classified_value(axis):
        value = deepcopy(definitions['Unknown'])
        choices = list(vocabulary[axis])
        value['properties']['value'] = {'enum': [None, *choices]}
        if not choices:
            value['properties']['state'] = {'type': 'string', 'enum': [
                state for state in definitions['UnknownState']['enum'] if state != 'known']}
        return value

    classification = definitions['Classification']['properties']
    for name, axis in (('kind', 'kinds'), ('stability', 'stabilities'),
                       ('modality', 'modalities'), ('basis', 'bases')):
        classification[name] = classified_value(axis)
    topics = classification['topics']
    if vocabulary['topics']:
        topics['items']['enum'] = list(vocabulary['topics'])
    else:
        topics['maxItems'] = 0
    definitions['Entity']['properties']['entity_type'] = classified_value('entity_types')
    for name, axis in (('role', 'cognition_roles'), ('epistemic_status', 'cognition_statuses')):
        definitions['Cognition']['properties'][name] = classified_value(axis)
    if vocabulary['relation_types']:
        definitions['Relation']['properties']['kind']['enum'] = list(vocabulary['relation_types'])
    else:
        schema['properties']['relations']['maxItems'] = 0

    attributes = [item['id'] for item in vocabulary['attributes']]
    if attributes:
        definitions['Attribute']['properties']['definition_id']['enum'] = attributes
        definitions['Predicate']['properties']['field']['enum'] = attributes
    else:
        definitions['AttributeList']['maxItems'] = 0
        for name in ('Condition', 'ConditionException'):
            definitions[name]['properties']['predicate'] = {'type': 'null'}
    if attributes:
        definitions['StatementProperty']['oneOf'][0]['properties']['key']['enum'] = attributes
    else:
        definitions['StatementProperty'] = {'type': 'null'}
    return schema
