"""Native API output contracts derived from the complete, editable L1 schema."""
from copy import deepcopy

from memory_l1_catalog_schema import model_document_schema


AUTO_KINDS = ('key_event', 'stable_preference', 'boundary', 'signal',
              'commitment', 'project_state', 'relationship_anchor', 'self_insight', 'love_letter')


def ref(name):
    return {'$ref': '#/$defs/' + name}


def obj(properties, required=()):
    return {'type': 'object', 'properties': properties,
            'required': list(required), 'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


def nullable(value):
    return {'anyOf': [value, {'type': 'null'}]}


def value_schema(scalar):
    """Keep both explicit scalars and all existing epistemic containers."""
    return {'anyOf': [deepcopy(scalar), {'type': 'null'},
        obj({'state': {'const': 'known'}, 'value': deepcopy(scalar),
             'evidence_refs': ref('EvidenceRefs')}, ('state', 'value')),
        obj({'state': {'enum': ['not_stated', 'unresolved', 'not_applicable', 'unbounded']},
             'value': {'type': 'null'}, 'evidence_refs': ref('EvidenceRefs')}, ('state',))]}


def build_schema(source_ids, catalog, *, stage="auto_l1", narrative_from_draft=False):
    # Start with the original complete model schema, retaining its semantic
    # modules and vocabulary. Only model wire shorthands change below.
    schema = model_document_schema(source_ids, catalog=catalog, entry_forms=['episode', 'reference'])
    defs = schema['$defs']
    evidence = {'type': 'string', 'enum': list(source_ids)}
    defs['EvidenceRef'] = evidence
    defs['SceneSourceRef'] = deepcopy(evidence)
    defs['Unknown'] = value_schema({'type': ['string', 'number', 'boolean']})
    text_value = value_schema({'type': 'string', 'minLength': 1})
    defs['SubjectRef'] = deepcopy(text_value)
    defs['SubjectRef']['description'] = 'Reference a declared entity_id; user/assistant are entity types, not entity IDs.'

    classification = defs['Classification']
    classification['required'] = ['kind', 'stability', 'modality', 'basis']
    for field, axis in [('kind', 'kinds'), ('stability', 'stabilities'),
                        ('modality', 'modalities'), ('basis', 'bases')]:
        classification['properties'][field] = value_schema({'type': 'string', 'enum': list(catalog[axis])})
    classification['properties']['kind']['description'] = 'Content type only. commitment is NOT allowed here; it belongs to modality.'
    defs['Entity']['required'] = ['name', 'entity_type']
    defs['Entity']['properties']['entity_type'] = value_schema({'type': 'string', 'enum': list(catalog['entity_types'])})
    defs['Entity']['properties']['evidence_refs'] = ref('EvidenceRefs')

    for name in ('Attribute', 'StatementProperty'):
        if name == 'Attribute':
            defs[name]['required'] = ['definition_id', 'value']
        elif 'oneOf' in defs[name]:
            defs[name]['oneOf'][0]['required'] = ['key', 'value', 'polarity']

    for name in ('Condition', 'ConditionException'):
        defs[name]['required'] = ['state']
    defs['EventTime']['required'] = ['precision', 'interpretation']
    for field in ('start', 'end', 'timezone'):
        defs['EventTime']['properties'][field] = deepcopy(text_value)
    defs['Applicability']['required'] = ['temporal_scope']
    for field in ('start', 'end'):
        defs['Applicability']['properties'][field] = deepcopy(text_value)
    defs['StatementTime']['properties']['event'] = nullable(ref('EventTime'))
    defs['StatementTime']['properties']['applicability'] = nullable(ref('Applicability'))

    cognition = defs['Cognition']
    cognition['required'] = ['role', 'epistemic_status']
    for field, axis in [('role', 'cognition_roles'), ('epistemic_status', 'cognition_statuses')]:
        cognition['properties'][field] = value_schema({'type': 'string', 'enum': list(catalog[axis])})

    statement = defs['Statement']
    statement['required'] = ['content', 'subject_ref', 'classification', 'time', 'conditions', 'evidence_refs']
    statement['properties']['subject_ref'] = ref('SubjectRef')
    statement['properties']['time'] = nullable(ref('StatementTime'))
    statement['properties']['conditions'] = {'anyOf': [ref('Condition'), {'type': 'string'}, {'type': 'null'}]}
    statement['properties']['role_hint'] = value_schema({'type': 'string', 'enum': ['core', 'detail']})
    defs['Scene']['required'] = ['title', 'statement_refs']
    defs['Relation']['required'] = ['from', 'to', 'kind', 'evidence_refs']
    # The current initial-task projection permits only source-local relations.
    relation_values = [key for key in catalog['relation_types']
                       if key not in {'revises', 'supersedes', 'split_from', 'merged_from'}]
    defs['Relation']['properties']['kind'] = {'type': 'string', 'enum': relation_values}

    document = {key: deepcopy(value) for key, value in schema.items() if key not in ('$schema', '$defs')}
    document['properties']['format'] = {'const': 'semantic-v1'}
    document['required'] = ['format', 'entities', 'statements']
    if not narrative_from_draft:
        document['required'].append('narrative')
    if stage != 'diary_l1':
        document['required'].append('retrieval')
    document['properties']['retrieval']['required'] = ['keywords']
    if stage != 'diary_l1':
        document['properties']['retrieval']['properties']['keywords']['minItems'] = 1
    defs['Document'] = document
    defs['Candidate'] = obj({
        'kind': {'type': 'string', 'enum': list(AUTO_KINDS),
                 'description': 'Automatic ingestion policy only, separate from statement classification.'},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'source_refs': ref('EvidenceRefs'), 'reason': {'type': 'string'},
        'document': ref('Document'),
    }, ('kind', 'confidence', 'source_refs', 'reason', 'document'))
    if stage == 'chat_l1':
        root = obj({'document': ref('Document'), 'kind': ref('MemoryKind'),
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'sensitive': {'type': 'boolean'}}, ('document',))
    elif stage == 'diary_l1':
        # Entire diary plus independently complete scene bodies; no scene limit
        # or optional semantic module is removed by the API projection.
        root = obj({'document': ref('Document'),
                    'scene_bodies': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
                   ('document', 'scene_bodies', 'confidence'))
        defs['Relation']['properties']['kind'] = {'type': 'string', 'enum': list(catalog['relation_types'])}
    else:
        root = obj({'candidates': array(ref('Candidate'))}, ('candidates',))
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', '$defs': defs, **root}


def prepare_request(messages, *, stage):
    """One API schema, one source copy; keep all source/context values intact."""
    import json
    from memory_l1_service import L1OwnerError
    result = deepcopy(messages)
    payload = json.loads(result[-1]['content'])
    envelope = payload.get('l1_contract', payload)
    context = envelope.get('semantic_contract')
    if not isinstance(context, dict):
        raise L1OwnerError('native_l1_contract_required', field_path='semantic_contract')
    catalog = context['catalog']
    # Reconstruct enum/attribute shapes from the exact revisioned model view,
    # rather than re-reading a potentially changed live catalogue mid-call.
    schema_catalog = {axis: list(values) if isinstance(values, dict) else deepcopy(values)
                      for axis, values in catalog.items() if axis not in ('attributes', 'definitions')}
    attrs = catalog.get('attributes', {})
    schema_catalog['attributes'] = ([{'id': key, **deepcopy(row)} for key, row in attrs.items()]
                                    if isinstance(attrs, dict) else deepcopy(attrs))
    schema = build_schema(context['source_ids'], schema_catalog, stage=stage,
                          narrative_from_draft=bool(payload.get('narrative_from_draft')))
    # The compact model catalogue is a projection (including owner definitions),
    # not another classification authority. The same values constrain the API.
    envelope['semantic_contract'] = {
        'source_ids': context['source_ids'], 'catalog': catalog,
        'output_rules': '按API提供的JSON Schema输出；保留全部主体、时间、条件、否定、未知及认知语义。只填写有来源的内容。',
    }
    result[-1]['content'] = json.dumps(payload, ensure_ascii=False)
    return result, schema


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


async def create_completion(client, *, model, messages, route, temperature, capture=None):
    """Native Responses for DeepSeek; standard Chat JSON Schema elsewhere.

    No protocol fallback, semantic repair, or implicit retry is performed.
    Returned adapters preserve native usage and the exact credential-free wire.
    """
    from types import SimpleNamespace as NS
    from urllib.parse import urlsplit
    from memory_task_routes import completion_options
    messages, schema = prepare_request(messages, stage=route['stage'])
    timeout = route.get('timeout_seconds', 180)
    client = client.with_options(timeout=timeout, max_retries=0)
    base = urlsplit(str(client.base_url))
    if base.hostname == 'api.deepseek.com':
        # DeepSeek Responses lives at /responses, not /v1/responses.
        if base.path.rstrip('/') == '/v1':
            client = client.with_options(base_url=base._replace(path='/', query='', fragment='').geturl())
        request = {'model': model, 'input': messages,
                   'max_output_tokens': route['max_tokens'], 'temperature': temperature,
                   'text': {'format': {'type': 'json_schema', 'name': 'memory_l1', 'schema': schema}}}
        effort = route.get('reasoning_effort')
        if route.get('thinking_mode') == 'disabled':
            effort = 'none'
        if effort:
            request['reasoning'] = {'effort': effort}
        if capture is not None:
            capture(request)
        response = await client.responses.create(**request)
        text = ''.join(_field(part, 'text', '') for item in (_field(response, 'output', []) or [])
                       if _field(item, 'type') == 'message'
                       for part in (_field(item, 'content', []) or [])
                       if _field(part, 'type') == 'output_text')
        native_usage = _field(response, 'usage')
        usage = native_usage.model_dump() if hasattr(native_usage, 'model_dump') else (native_usage or {})
        usage = dict(usage)
        usage.update(prompt_tokens=usage.get('input_tokens'), completion_tokens=usage.get('output_tokens'),
                     completion_tokens_details=usage.get('output_tokens_details', {}))
        status = _field(response, 'status')
        finish = 'stop' if status == 'completed' else 'length' if status == 'incomplete' else 'error'
        adapted = NS(choices=[NS(message=NS(content=text), finish_reason=finish)], usage=NS(**usage),
                     model=_field(response, 'model', model), native_status=status)
    else:
        request = {'model': model, 'messages': messages,
                   **completion_options(route, temperature=temperature),
                   'response_format': {'type': 'json_schema', 'json_schema': {'name': 'memory_l1', 'schema': schema}}}
        if capture is not None:
            capture(request)
        response = await client.chat.completions.create(**request)
        adapted = response
    # Only request fields built above are retained: never client/base URL/key/headers.
    adapted.native_request = deepcopy(request)
    adapted.native_schema = schema
    return adapted


def validate_response(response):
    """Validate the original output; never fix a malformed or incomplete JSON."""
    schema = getattr(response, 'native_schema', None)
    if schema is None:
        return  # Existing non-native fixtures/legacy callers keep their validator.
    import json
    from jsonschema import Draft202012Validator
    from memory_l1_service import L1OwnerError
    if (getattr(response, 'native_status', 'completed') != 'completed'
            or response.choices[0].finish_reason != 'stop'):
        raise L1OwnerError('native_l1_incomplete_completion', field_path='generation')
    try:
        parsed = json.loads(response.choices[0].message.content)
    except (ValueError, TypeError, AttributeError, IndexError):
        raise L1OwnerError('native_l1_invalid_json', field_path='generation') from None
    error = next(Draft202012Validator(schema).iter_errors(parsed), None)
    if error is not None:
        raise L1OwnerError('native_l1_schema_mismatch',
                           field_path='.'.join(map(str, error.absolute_path)) or 'document')


def validate_candidate_items(response):
    """Retain individually valid drafts without claiming the batch was valid.

    The root and every retained Candidate use the unchanged native schema.
    Invalid items become error markers, never repaired or adopted documents.
    """
    import json
    from memory_l1_service import L1OwnerError
    schema = getattr(response, 'native_schema', None)
    try:
        validate_response(response)
        schema_issue = None
    except L1OwnerError as exc:
        if exc.code != 'native_l1_schema_mismatch': raise
        schema_issue = {'code': exc.code, 'field_path': exc.field_path}
    parsed = json.loads(response.choices[0].message.content)
    candidates = parsed.get('candidates') if isinstance(parsed, dict) else None
    if not isinstance(candidates, list):
        raise L1OwnerError('first_pass_candidates_required', field_path='candidates')
    if schema is None or schema_issue is None: return candidates, schema_issue
    from jsonschema import Draft202012Validator
    root = deepcopy(schema)
    root['properties']['candidates']['items'] = {}
    error = next(Draft202012Validator(root).iter_errors(parsed), None)
    if error is not None:
        raise L1OwnerError('native_l1_schema_mismatch', field_path='.'.join(map(str, error.absolute_path)))
    validator = Draft202012Validator({'$defs': schema['$defs'], '$ref': '#/$defs/Candidate'})
    result = []
    for item in candidates:
        error = next(validator.iter_errors(item), None)
        result.append(item if error is None else {'_candidate_validation_issue': {
            'code': 'native_l1_schema_mismatch', 'field_path': '.'.join(map(str, error.absolute_path))}})
    return result, schema_issue
