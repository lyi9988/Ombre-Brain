"""Native API output contracts derived from the complete, editable L1 schema."""
from copy import deepcopy
import re

from memory_l1_catalog_schema import model_document_schema


AUTO_KINDS = ('key_event', 'stable_preference', 'boundary', 'signal',
              'commitment', 'project_state', 'relationship_anchor', 'self_insight', 'love_letter')

POLICY_KIND_GUIDE = (
    '整份回忆的 kind 回答“为什么留下它”。key_event=具体经历，stable_preference=持续偏好，'
    'boundary=边界，signal=约定信号，commitment=承诺，project_state=事项进展，'
    'relationship_anchor=关系中的重要经历，self_insight=对自己的理解，love_letter=有原文依据的情书锚点。'
    '每条 statement.classification.kind 描述那一句的性质。'
    '例如“以后都喜欢这个装订方式”可用 stable_preference，里面的陈述可用 preference；'
    '一次出游可用 key_event，里面可有 experience 或 fact。亲密表达照实保留，写过、收到或送达分别以来源为准。'
)

CHAT_CANDIDATE_KIND_GUIDE = (
    '先按本次自动记忆提示选材，再给选中的回忆标 kind。kind 说明这份记忆为什么值得留下。'
    'key_event=值得回看的重要经历，stable_preference=有据的稳定偏好，boundary=边界或授权范围，'
    'signal=有约定含义的信号，commitment=承诺或待履行的约定，'
    'project_state=持续项目里有来源说明会影响后续执行的稳定决定、实质进展或状态，'
    'relationship_anchor=有具体意义、值得回味的共同相处，self_insight=当事人表达的自我认识，'
    'love_letter=有原文依据的情书锚点。statement.classification.kind 描述每条陈述的性质。'
)


def ref(name):
    return {'$ref': '#/$defs/' + name}


def event_selection_schema(sources):
    """The selector may cite only a visible message; exact spans are checked against its text."""
    from memory_l1_service import L1OwnerError
    if not isinstance(sources, list) or not sources:
        raise L1OwnerError('event_sources_required', field_path='sources')
    ids = [row.get('source') if isinstance(row, dict) else None for row in sources]
    if any(not isinstance(key, str) or not re.fullmatch(r'source:[1-9][0-9]*', key) for key in ids):
        raise L1OwnerError('event_sources_required', field_path='sources')
    if len(set(ids)) != len(ids):
        raise L1OwnerError('event_sources_required', field_path='sources')
    text = {'type': 'string', 'minLength': 1}
    anchor = obj({'source': {'type': 'string', 'enum': ids}, 'quote': text}, ('source', 'quote'))
    event = obj({'point': text, 'anchors': {**array(anchor), 'minItems': 1}}, ('point', 'anchors'))
    return obj({'context': {'type': 'string'}, 'memories': array(event)}, ('context', 'memories'))


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


def build_schema(source_ids, catalog, *, stage="auto_l1", narrative_from_draft=False,
                 candidate_kind_guide=POLICY_KIND_GUIDE):
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
        defs[name]['description'] = (
            'A requirement or exception governing a claim. state=known means the condition is specified, '
            'not that it has been satisfied. Preserve all jointly required clauses and unknown fulfilment.')
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
    statement['properties']['conditions']['description'] = (
        'Attach requirements to the plan/claim they govern. A and B are both required when joined by AND. '
        'Do not turn a requirement into a separate asserted fact, especially when the source says its fulfilment is unknown.')
    statement['properties']['role_hint'] = value_schema({'type': 'string', 'enum': ['core', 'detail']})
    defs['Scene']['required'] = ['title', 'statement_refs']
    defs['Scene']['properties']['source_refs']['description'] = (
        'Source identifiers only: an array of strings chosen from the supplied source_ids. '
        'Range objects belong in source_ranges, never in source_refs.')
    defs['Scene']['properties']['source_ranges']['description'] = (
        'Optional exact original-text spans: objects with source_ref, start and end. '
        'Use source_refs for source identifiers without a verified span.')
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
                 'description': candidate_kind_guide},
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
    if stage == 'auto_l1' and payload.get('input_kind') == 'daily_chat_event_selection':
        return result, event_selection_schema(payload.get('sources') or [])
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
    event_assembly = stage == 'auto_l1' and payload.get('input_kind') == 'daily_chat_event_assembly'
    ordinary_chat = stage == 'auto_l1' and 'window' in payload and (not payload.get('input_kind') or event_assembly)
    kind_guide = CHAT_CANDIDATE_KIND_GUIDE if ordinary_chat else POLICY_KIND_GUIDE
    if event_assembly:
        kind_guide = kind_guide.replace('先按本次自动记忆提示选材，再给选中的回忆标 kind。', '本次选材已完成，为已选事件标 kind。', 1)
    schema = build_schema(context['source_ids'], schema_catalog, stage=stage,
                          narrative_from_draft=bool(payload.get('narrative_from_draft')),
                          candidate_kind_guide=kind_guide)
    if event_assembly:
        schema['properties']['candidates']['maxItems'] = 1
    if ordinary_chat and not payload.get('candidate_repairs'):
        # Some providers repeat precisely the declared JSON Schema dialect in the
        # response object. Allow that literal annotation, never another root key or URI.
        schema['properties']['$schema'] = {'const': schema['$schema']}
    # The compact model catalogue is a projection (including owner definitions),
    # not another classification authority. The same values constrain the API.
    # Keep the existing compact contract in the model-visible input as well.
    # Sending response_format through a compatible provider does not prove that
    # its backend enforces the schema. Do not strip the only text explanation
    # of fields such as scene references versus source-relative ranges.
    envelope['semantic_contract'] = deepcopy(context)
    coverage = (
        '完整性按每份已选记忆计算。source_ids 是可引用的原始材料，source_refs 可取其中一部分。'
        'statements 中的每项会被单独检索，其主体、时间、条件、否定、未知和认知须完整。'
        '只为这份记忆实际需要保存的内容建立陈述，无须逐条覆盖输入消息或逐位记录参与人。'
        'subject_ref 标明该项主要讲谁或什么，谁说了这句话由来源另行记录。'
        if ordinary_chat and not payload.get('candidate_repairs') else
        '保留全部主体、时间、条件、否定、未知及认知语义。'
    )
    envelope['semantic_contract']['output_rules'] = (
        f'按API提供的JSON Schema及本结构说明输出JSON；{coverage}'
        '只填写有来源的内容。source_refs与evidence_refs只放来源标记字符串；'
        '带source_ref、start、end的对象只能放在source_ranges，坐标必须对应实际原文。'
        '没有可靠精确范围时省略source_ranges，保留有据的source_refs；不得猜测范围。'
        'conditions.state=known仅表示条件内容明确，不表示已满足。联合条件挂在它约束的计划或承诺上，'
        '不可拆成已经成立的事实；结合原文前后段的未知、否定和更正，逐条核对statements，不只核对正文。')
    if stage == 'auto_l1':
        envelope['semantic_contract']['output_rules'] += kind_guide
        repairs = payload.get('candidate_repairs')
        needs_document = not repairs or any(not item['kind_only'] for item in repairs.values())
        if ordinary_chat and needs_document:
            if not repairs:
                envelope['semantic_contract']['output_rules'] += (
                    '\n根对象只有 candidates 数组。批次计数和处理状态由程序记录。')
            envelope['semantic_contract']['output_rules'] += (
                '\n每份 document 中，narrative 只有 title 和 body。retrieval 与 narrative 同级；'
                '如填写 entry，也只在 document 顶层写一次。字段形状示意（略去其余字段）：'
                '{"document":{"entry":{"form":"episode"},"narrative":{"title":"…","body":"…"},'
                '"retrieval":{"keywords":["…"]}}}。填实际形式和检索词，勿照抄示意符号。')
        if 'candidate_repairs' in payload:
            # Each key belongs to one rejected item, including when two items
            # use the same original message. No slot exists for a valid sibling.
            slots = {key: ({'type': 'string', 'enum': list(AUTO_KINDS)}
                           if item['kind_only'] else ref('Candidate')) for key, item in repairs.items()}
            schema = {'$schema': schema['$schema'], '$defs': schema['$defs'], **obj(slots, slots)}
            envelope['semantic_contract']['response_format'] = {
                'root': {key: '入库类别字符串' if item['kind_only'] else '一份修正后的完整Candidate'
                         for key, item in repairs.items()},
                'rules': '这是对上次失败草稿的定点修正。按上面每个键各返回一个值；参照原消息，不新增回忆。'}
            if any('cognition' in item.get('issue', {}).get('field_path', '').split('.') for item in repairs.values()):
                roles = '、'.join(schema_catalog['cognition_roles'])
                statuses = '、'.join(schema_catalog['cognition_statuses'])
                envelope['semantic_contract']['response_format']['rules'] += (
                    f' 若陈述有认知标注，cognition.role 只能从 {roles} 中选，epistemic_status 只能从 {statuses} 中选。'
                    '陈述的 classification.kind 和认知角色是两个目录。按原话判断该值，'
                    '同时核对这一条经历里各方实际说过什么；证据须指向能支持这句话的原消息。')
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
    if route['stage'] == 'diary_l1' and route.get('read_timeout_seconds') is not None:
        from httpx import Timeout
        timeout = Timeout(timeout, read=route['read_timeout_seconds'])
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


def validate_candidate_items(response, *, rejected_drafts=None, repair_slots=(),
                             review_layout_repairs=None):
    """Retain individually valid drafts without claiming the batch was valid.

    The root and every retained Candidate use the unchanged native schema.
    Invalid items become error markers. An explicit Review adapter may move an
    existing retrieval object to its absent sibling slot, only if the same full
    Candidate schema then validates. The original response remains invalid.
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
    for index, item in enumerate(candidates):
        error = next(validator.iter_errors(item), None)
        if error is None:
            result.append(item)
            continue
        field = '.'.join(map(str, error.absolute_path))
        slot = repair_slots[index] if repair_slots else f'item_{index}'
        source_index = int(slot.removeprefix('item_'))
        issue = {'code': 'native_l1_schema_mismatch', 'field_path': field}
        document = item.get('document') if isinstance(item, dict) else None
        narrative = document.get('narrative') if isinstance(document, dict) else None
        if (review_layout_repairs is not None and isinstance(narrative, dict)
                and 'retrieval' not in document and isinstance(narrative.get('retrieval'), dict)):
            relocated = deepcopy(item)
            relocated['document']['retrieval'] = relocated['document']['narrative'].pop('retrieval')
            if validator.is_valid(relocated):
                review_layout_repairs[slot] = {
                    'issue': deepcopy(issue), 'from': 'document.narrative.retrieval',
                    'to': 'document.retrieval'}
                result.append(relocated)
                continue
        if field == 'kind':
            value = error.instance
            diagnostic = {'candidate_index': source_index, 'validator': error.validator,
                          'actual_type': type(value).__name__}
            kind_schema = schema['$defs']['Candidate'].get('properties', {}).get('kind')
            if isinstance(kind_schema, dict) and isinstance(kind_schema.get('enum'), list):
                diagnostic['allowed_kinds'] = list(kind_schema['enum'])
            # Record only vocabulary-shaped identifiers, never source prose or a raw response.
            if isinstance(value, str) and re.fullmatch(r'[a-z][a-z_]{0,39}', value):
                diagnostic['actual_kind'] = value
            elif isinstance(value, str):
                diagnostic['actual_length'] = len(value)
            issue.update(diagnostic)
            if schema_issue['field_path'] == f'candidates.{index}.kind':
                schema_issue.update(diagnostic)
        result.append({'_candidate_validation_issue': issue})
        if rejected_drafts is not None:
            # A category-only retry must not regenerate or rewrite an otherwise
            # valid document. Other errors need the full item validated again.
            kind_only = False
            if isinstance(item, dict) and ('kind' not in item or field == 'kind'):
                kind_only = validator.is_valid({**item, 'kind': AUTO_KINDS[0]})
            rejected_drafts[slot] = {'draft': item, 'issue': issue, 'kind_only': kind_only}
    return result, schema_issue


def repaired_candidate_response(response, repairs):
    """Project a validated per-item correction back through the full original validator."""
    import json
    from types import SimpleNamespace
    from jsonschema import Draft202012Validator
    from memory_l1_service import L1OwnerError
    schema = response.native_schema
    try:
        validate_response(response)
    except L1OwnerError as exc:
        if exc.code != 'native_l1_schema_mismatch':
            raise
    values = json.loads(response.choices[0].message.content)
    if not isinstance(values, dict) or values.keys() != repairs.keys():
        raise L1OwnerError('candidate_repair_slots_mismatch', field_path='candidate_repairs')
    items = []
    for key, row in repairs.items():
        result = row['draft']
        validator = Draft202012Validator({'$defs': schema['$defs'], **schema['properties'][key]})
        if validator.is_valid(values[key]):
            result = {**result, 'kind': values[key]} if row['kind_only'] else values[key]
        items.append(result)
    candidate_schema = {'$defs': schema['$defs'], **obj({'candidates': array(ref('Candidate'))}, ('candidates',))}
    message = SimpleNamespace(content=json.dumps({'candidates': items}, ensure_ascii=False))
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')],
                           native_schema=candidate_schema)
