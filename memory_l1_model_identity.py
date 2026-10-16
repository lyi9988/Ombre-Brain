"""Turn model-local labels into program-managed local identity keys."""
from __future__ import annotations

import copy
import hashlib

from memory_l1_service import L1OwnerError


def normalize_model_local_keys(document, *, previous=None, catalog=None, known_ids=()):
    """No role guessing: references must point to a declared local member."""
    if not isinstance(document, dict):
        raise L1OwnerError('model_document_fields_required', field_path='document')
    draft = copy.deepcopy(document)
    mapping = {}
    for module, key in (('entities', 'entity_id'), ('scenes', 'scene_id'),
                        ('statements', 'statement_id'), ('relations', 'relation_id')):
        known = {x[key] for x in (previous or {}).get(module, [])} | set(known_ids)
        for item in draft.get(module, []):
            local = item.get(key) if isinstance(item, dict) else None
            if not isinstance(local, str) or not local:
                raise L1OwnerError('local_member_key_required', field_path='document.' + module)
            if local in known or local.startswith('new:'):
                continue
            if local in mapping:
                raise L1OwnerError('duplicate_local_identity', field_path='document.' + module)
            mapping[local] = 'new:' + module + ':' + hashlib.sha256(local.encode()).hexdigest()[:32]
    for module, key in (('entities', 'entity_id'), ('scenes', 'scene_id'),
                        ('statements', 'statement_id'), ('relations', 'relation_id')):
        for item in draft.get(module, []):
            item[key] = mapping.get(item[key], item[key])
    for item in draft.get('statements', []):
        subject = item.get('subject_ref')
        if isinstance(subject, dict): subject['value'] = mapping.get(subject.get('value'), subject.get('value'))
        scope = (item.get('time') or {}).get('applicability') or {}
        scope['scene_ref'] = mapping.get(scope.get('scene_ref'), scope.get('scene_ref'))
    for item in draft.get('scenes', []):
        item['statement_refs'] = [mapping.get(ref, ref) for ref in item.get('statement_refs', [])]
    for item in draft.get('relations', []):
        for key in ('from', 'to'):
            end = item.get(key) or {}
            end['object_id'] = mapping.get(end.get('object_id'), end.get('object_id'))
    definitions = {x['id']: x for x in (catalog or {}).get('attributes', [])}
    def remap(node):
        if isinstance(node, list):
            for child in node: remap(child)
        elif isinstance(node, dict):
            definition = definitions.get(node.get('definition_id', node.get('key')), {})
            value = node.get('value')
            if definition.get('value_type') == 'entity_ref' and isinstance(value, dict) and value.get('state') == 'known':
                value['value'] = mapping.get(value.get('value'), value.get('value'))
            if definitions.get(node.get('field'), {}).get('value_type') == 'entity_ref':
                node['value'] = [mapping.get(v, v) for v in value] if node.get('op') == 'in' else mapping.get(value, value)
            for child in node.values(): remap(child)
    for module in ('attributes', 'statements', 'scenes'): remap(draft.get(module, []))
    return draft, mapping
