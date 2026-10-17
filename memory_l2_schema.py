"""Owner-configurable output constraints for the independent second pass."""
from __future__ import annotations

import copy
from jsonschema import Draft202012Validator, SchemaError

from memory_l1_contract import canonical_json, semantic_sha256
from memory_l1_service import L1OwnerError

USES = ['current_basis', 'conditional_basis', 'relationship_reference', 'historical_evidence', 'unresolved_evidence']


def default_schema():
    def unit(key, variant=False):
        properties = {key:{'type':'string','minLength':1}, 'text':{'type':'string','minLength':1},
                      'statement_refs':{'type':'array','minItems':1,'items':{'type':'string'}}}
        if variant: properties['intended_use'] = {'type':'string','enum':list(USES)}
        return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema', 'type':'object',
        'required':['full_injection_text','coherent_sections','optional_use_variants'], 'additionalProperties':False,
        'properties':{'full_injection_text':{'type':'string','minLength':1},
                      'coherent_sections':{'type':'array','items':unit('section_id')},
                      'optional_use_variants':{'type':'array','items':unit('variant_id', True)},
                      'review':{'type':'object', 'required':['required','reasons'], 'additionalProperties':False,
                                'properties':{'required':{'type':'boolean'},'reasons':{'type':'array','items':{'type':'string'}}}}}}


def normalize_schema(schema):
    if not isinstance(schema, dict): raise L1OwnerError('invalid_l2_output_schema', field_path='reflection.memory_l2_output_schema')
    canonical_json(schema)
    try: Draft202012Validator.check_schema(schema)
    except SchemaError: raise L1OwnerError('invalid_l2_output_schema', field_path='reflection.memory_l2_output_schema') from None
    def local_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('$ref', '$dynamicRef') and (not isinstance(value, str) or not value.startswith('#')):
                    raise L1OwnerError('l2_schema_local_refs_only', field_path='reflection.memory_l2_output_schema')
                local_refs(value)
        elif isinstance(node, list):
            for value in node: local_refs(value)
    local_refs(schema)
    return copy.deepcopy(schema)


def output_schema(config):
    value = (config.get('reflection') or {}).get('memory_l2_output_schema')
    return default_schema() if value is None else normalize_schema(value)


def schema_revision(schema): return 'l2-schema:' + semantic_sha256(schema)


def validate_output(document, schema):
    try:
        error = next(Draft202012Validator(schema).iter_errors(document), None)
    except Exception:
        raise L1OwnerError('l2_output_schema_unresolvable', field_path='reflection.memory_l2_output_schema') from None
    if error is not None:
        raise L1OwnerError('l2_output_schema_mismatch', field_path='document')
