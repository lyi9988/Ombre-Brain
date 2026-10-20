"""Program-bound first-pass L1, shared by existing ingestion entrances.

Models supply semantic drafts, never source status, canonical identities,
revisions or authority decisions. This module performs no model calls.
"""
from __future__ import annotations

import copy
import hashlib

from memory_l1_contract import canonical_json, make_text_anchor, verify_text_anchor
from memory_l1_schema import document_template, generation_template, validate_l1_document
from memory_l1_service import L1OwnerError

_UNSCOPED_CANDIDATE = object()

MODEL_FIELDS = {'entry', 'narrative', 'entities', 'scenes', 'statements', 'retrieval', 'relations', 'attributes'}


def model_template(*, catalog=None, catalog_revision=None):
    doc = document_template(catalog=catalog, catalog_revision=catalog_revision)
    return {key: ({'form': 'episode'} if key == 'entry' else copy.deepcopy(doc[key])) for key in MODEL_FIELDS}


def source_snapshot(source, *, config=None):
    """Only callers holding actual authenticated reader results call this."""
    if not isinstance(source, dict) or not isinstance(source.get('text'), str) or not source['text']:
        raise L1OwnerError('source_text_required', field_path='sources')
    if source.get('namespace') == 'attachment_text':
        from memory_attachment_source import snapshot
        return snapshot(source)
    namespace = {'canonical_event': 'snapshot.aiz.canonical_event', 'raw_event': 'snapshot.ombre.raw_event',
                 'conversation_turn': 'snapshot.ombre.conversation_turn',
                 'action_receipt': 'snapshot.aiz.action_receipt'}.get(source.get('namespace'))
    if namespace is None: raise L1OwnerError('unsupported_source_namespace', field_path='sources')
    locator = {key: source.get(key) for key in ('system', 'profile_id', 'conversation_id', 'event_id', 'role')}
    if not locator['event_id'] or locator['role'] not in ('user', 'assistant', 'tool'):
        raise L1OwnerError('source_identity_required', field_path='sources')
    from memory_source_identity import source_participant, source_group
    participant = source_participant(source, config)
    revision = {'version_id': source.get('version_id'), 'text_sha256': hashlib.sha256(source['text'].encode()).hexdigest()}
    if config is not None:
        revision['participant'] = {key: participant[key] for key in ('subject_id', 'kind')}
    return {'state': 'available', 'source_namespace': namespace, 'source_id': canonical_json(locator),
        'source_revision': 'source_sha256:' + hashlib.sha256(canonical_json(revision).encode()).hexdigest(),
        'text_view_id': 'memory.source.exact-v1', 'text': source['text'], 'author': source.get('role'),
        'speaker': source.get('role'), 'participant': participant, 'source_group': source_group(source), 'upstream_revision': source.get('version_id') or None,
        'recorded_at': source.get('recorded_at'), 'source_key': source.get('source_key'),
        'archive_ref': source.get('archive_ref'),
        'limitations': ['来源记录时间不代表事件发生时间；源快照不是主人确认事实。']}


def bind_sources(sources, *, config=None):
    if not isinstance(sources, list) or not sources:
        raise L1OwnerError('sources_required', field_path='sources')
    snapshots, bindings = [], []
    for index, source in enumerate(sources):
        if source.get('_model_excluded'):
            continue
        snapshot = source_snapshot(source, config=config)
        expected = source.get('_participant_revision')
        if expected is not None and expected != snapshot['source_revision']:
            raise L1OwnerError('source_participant_changed', status=409, field_path='sources')
        extent = source.get('_model_range', {'start': 0, 'end': len(snapshot['text'])})
        if (set(extent) != {'start', 'end'} or type(extent['start']) is not int
                or type(extent['end']) is not int
                or not 0 <= extent['start'] < extent['end'] <= len(snapshot['text'])):
            raise L1OwnerError('invalid_model_source_range', field_path='sources')
        snapshots.append(snapshot)
        bindings.append({'binding_id': f'source:{index + 1}',
            'source_namespace': snapshot['source_namespace'], 'source_id': snapshot['source_id'],
            'source_revision': snapshot['source_revision'], 'author_ref': snapshot['participant']['subject_id'], 'speaker_ref': snapshot['participant']['subject_id'],
            'anchors': [make_text_anchor(snapshot['text'], target_namespace='l0', target_id=snapshot['source_id'],
                target_revision=snapshot['source_revision'], text_view_id=snapshot['text_view_id'],
                start=extent['start'], end=extent['end'])]})
    return snapshots, bindings



def candidate_source_refs(document, source_refs, source_count):
    """Resolve one candidate's explicit support, not its broad scan window.

    Source labels keep their original window positions. Entity evidence and
    scene ranges are support too; never reinterpret them as local entity IDs.
    This validates references, not the truth/relevance of a model's citations.
    """
    refs = set()
    def add(values):
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise L1OwnerError('candidate_source_refs_required', field_path='candidate.source_refs')
        refs.update(values)
    add(source_refs)
    def visit(node):
        if isinstance(node, list):
            for child in node: visit(child)
        elif isinstance(node, dict):
            for key in ('evidence_refs', 'source_refs'):
                if key in node: add(node[key])
            if 'source_ranges' in node:
                ranges = node['source_ranges']
                if not isinstance(ranges, list):
                    raise L1OwnerError('invalid_scene_source_range', field_path='document.scenes.source_ranges')
                for extent in ranges:
                    if not isinstance(extent, dict) or not isinstance(extent.get('source_ref'), str):
                        raise L1OwnerError('invalid_scene_source_range', field_path='document.scenes.source_ranges')
                    refs.add(extent['source_ref'])
            for child in node.values(): visit(child)
    visit(document)
    available = {f'source:{i + 1}' for i in range(source_count)}
    if refs - available:
        raise L1OwnerError('unknown_source_binding', field_path='candidate.source_refs')
    if not refs:
        raise L1OwnerError('candidate_sources_required', field_path='candidate.source_refs')
    return refs


def expand_model_document(document, base, source_texts=None):
    from memory_l1_model_identity import normalize_model_local_keys
    document, _ = normalize_model_local_keys(document, previous=base,
        catalog=base.get('generation', {}).get('catalog_snapshot'))
    if not isinstance(document, dict) or set(document) != MODEL_FIELDS:
        raise L1OwnerError("model_document_fields_required", field_path="document")
    bindings = base["evidence_bindings"]
    doc = copy.deepcopy(base)
    for key, value in document.items():
        if key == 'entry':
            if not isinstance(value, dict) or set(value) != {'form'} or value['form'] not in ('episode', 'reference'):
                raise L1OwnerError('model_entry_form_required', field_path='document.entry')
            doc['entry']['form'] = value['form']
        else:
            doc[key] = copy.deepcopy(value)
    old_members = {module: {item[id_key]: item for item in base[module]} for module, id_key in
        (('statements', 'statement_id'), ('scenes', 'scene_id'), ('relations', 'relation_id'))}
    for module, id_key, revision_key in (('statements', 'statement_id', 'statement_revision_id'), ('scenes', 'scene_id', 'scene_revision_id')):
        if not isinstance(doc[module], list): raise L1OwnerError('array_required', field_path='document.' + module)
        for index, item in enumerate(doc[module]):
            if not isinstance(item, dict) or revision_key in item or (module == 'scenes' and 'generation' in item):
                raise L1OwnerError('model_managed_field', field_path=f'document.{module}[{index}]')
            prior = old_members[module].get(item.get(id_key))
            item[revision_key] = prior[revision_key] if prior else None
            if module == 'scenes':
                if 'anchors' in item: raise L1OwnerError('model_managed_field', field_path=f'document.scenes[{index}].anchors')
                refs = item.pop('source_refs', [])
                ranges = item.pop('source_ranges', [])
                if not isinstance(refs, list) or not isinstance(ranges, list):
                    raise L1OwnerError('scene_source_refs_required', field_path=f'document.scenes[{index}]')
                by_id = {binding['binding_id']: binding for binding in bindings}
                item['anchors'] = []
                for ref in refs:
                    ref = bindings[-1]['binding_id'] if ref == 'source:current' else ref
                    if ref not in by_id: raise L1OwnerError('unknown_scene_source', field_path=f'document.scenes[{index}].source_refs')
                    item['anchors'].extend(copy.deepcopy(by_id[ref]['anchors']))
                for extent in ranges:
                    if not isinstance(extent, dict) or set(extent) != {'source_ref', 'start', 'end'}:
                        raise L1OwnerError('invalid_scene_source_range', field_path=f'document.scenes[{index}].source_ranges')
                    ref = bindings[-1]['binding_id'] if extent['source_ref'] == 'source:current' else extent['source_ref']
                    binding = by_id.get(ref)
                    source_text = (source_texts or {}).get(ref)
                    if binding is None or source_text is None:
                        raise L1OwnerError('scene_source_text_required', field_path=f'document.scenes[{index}].source_ranges')
                    view = binding['anchors'][0]
                    if type(extent['start']) is not int or type(extent['end']) is not int:
                        raise L1OwnerError('invalid_scene_source_range', field_path=f'document.scenes[{index}].source_ranges')
                    if not any(a['start'] <= extent['start'] < extent['end'] <= a['end'] for a in binding['anchors']):
                        raise L1OwnerError('scene_range_outside_observed_input', field_path=f'document.scenes[{index}].source_ranges')
                    item['anchors'].append(make_text_anchor(source_text, target_namespace='l0', target_id=binding['source_id'],
                        target_revision=binding['source_revision'], text_view_id=view['text_view_id'], start=extent['start'], end=extent['end']))
                item['generation'] = copy.deepcopy(prior['generation'] if prior else base['generation'])
    for index, item in enumerate(doc['relations']):
        if not isinstance(item, dict) or 'origin' in item or 'confirmation_state' in item:
            raise L1OwnerError('model_managed_field', field_path=f'document.relations[{index}]')
        prior = old_members['relations'].get(item.get('relation_id'))
        item['origin'] = prior['origin'] if prior else 'model'
        item['confirmation_state'] = 'proposed'
        for side in ('from', 'to'):
            endpoint = item.get(side)
            if isinstance(endpoint, dict) and set(endpoint) == {'object_type', 'object_id'}:
                # Model-local references do not invent a persistent identity.
                # The owner normalizer binds this pair to the committed revision.
                endpoint.update(memory_id=None, revision='this_revision')

    def remap_evidence(node):
        if isinstance(node, list):
            for item in node: remap_evidence(item)
        elif isinstance(node, dict):
            if 'evidence_refs' in node and isinstance(node['evidence_refs'], list):
                node['evidence_refs'] = [bindings[-1]['binding_id'] if ref == 'source:current' else ref for ref in node['evidence_refs']]
            for value in node.values(): remap_evidence(value)
    remap_evidence(doc)
    return doc


def prepare_model_document(revisions, *, memory_id, document, sources, generation, request_id, previous=None, catalog_context=None, config=None, candidate_refs=_UNSCOPED_CANDIDATE):
    if not isinstance(document, dict) or set(document) != MODEL_FIELDS:
        raise L1OwnerError('model_document_fields_required', field_path='document')
    if (not isinstance(generation, dict) or set(generation) != {'prompt_revision', 'model_route_revision', 'task_id', 'actor'}
            or any(not isinstance(generation.get(k), str) or not generation[k] for k in ('task_id', 'actor'))
            or any(generation[k] is not None and (not isinstance(generation[k], str) or not generation[k])
                   for k in ('prompt_revision', 'model_route_revision'))):
        raise L1OwnerError('generation_context_required', field_path='generation')
    catalog, catalog_revision = catalog_context if catalog_context is not None else revisions.catalog_provider()
    snapshots, bindings = bind_sources(sources, config=config)
    if candidate_refs is not _UNSCOPED_CANDIDATE:
        selected = candidate_source_refs(document, candidate_refs, len(sources))
        if selected - {binding['binding_id'] for binding in bindings}:
            raise L1OwnerError('source_outside_observed_segment', field_path='candidate.source_refs')
        pairs = [(snapshot, binding) for snapshot, binding in zip(snapshots, bindings)
                 if binding['binding_id'] in selected]
        snapshots, bindings = map(list, zip(*pairs))
    base = copy.deepcopy(previous) if previous else document_template(catalog=catalog, catalog_revision=catalog_revision)
    base['evidence_bindings'] = bindings
    doc = expand_model_document(document, base, {b['binding_id']: s['text'] for b, s in zip(bindings, snapshots)})
    revision = 1 if previous is None else int(revisions.authority.get_memory(memory_id)['active_revision']) + 1
    normalized = revisions._owner_document(doc, base, memory_id=memory_id, request_id=request_id,
        catalog=catalog, catalog_revision=catalog_revision, revision=revision)
    gen = generation_template(normalized['generation']['catalog_snapshot'], catalog_revision, 'first_pass_model')
    gen.update(generation)
    normalized['generation'] = gen
    for scene in normalized['scenes']:
        scene['generation'] = copy.deepcopy(gen)
        for anchor in scene['anchors']:
            snapshot = next((s for s in snapshots if anchor['target_namespace'] == 'l0'
                and s['source_id'] == anchor['target_id'] and s['source_revision'] == anchor['target_revision']), None)
            if snapshot is None:
                raise L1OwnerError('model_scene_source_anchor_required', field_path='document.scenes.anchors')
            verify_text_anchor(snapshot['text'], anchor, target_namespace='l0', target_id=snapshot['source_id'],
                target_revision=snapshot['source_revision'], text_view_id=snapshot['text_view_id'])
    old_relations = base['relations']
    for relation in normalized['relations']:
        prior = next((old for old in old_relations if old['relation_id'] == relation['relation_id']), None)
        same = prior and all(relation[key] == prior[key] for key in ('from', 'to', 'kind', 'evidence_refs'))
        relation['origin'] = prior['origin'] if same else 'model'
        relation['confirmation_state'] = prior['confirmation_state'] if same else 'proposed'
    validate_l1_document(normalized, catalog=normalized['generation']['catalog_snapshot'])
    from memory_l0_sources import MemoryL0Sources
    for snapshot in snapshots: MemoryL0Sources(revisions.authority).retain_copy(snapshot)
    return normalized
