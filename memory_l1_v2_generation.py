"""Prepare L1 v2 cards from full, authenticated source views."""
from __future__ import annotations

import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import monotonic

import frontmatter

from memory_candidate_preparation import MemoryCandidatePreparation
from memory_chat_segments import ChatSegmentProgress
from memory_l1_api import array, create_completion, obj, validate_response
from memory_commit_service import body_sha256
from memory_l1_contract import canonical_json, make_text_anchor, semantic_sha256
from memory_l1_ingestion import bind_sources
from memory_l1_service import L1OwnerError
from memory_l1_v2_prompts import L1_V2_CONTINUATION_RULE, L1_V2_MATERIAL_PLACEHOLDER, L1_V2_SOURCE_RULES
from memory_model_io import MemoryModelIO, completion_parameters, model_error_details, response_usage
from memory_l1_v2_chat import CHAT_PROMPT_ID, build_chat_response_format

PROTOCOL = 'whole_window_v2'
PROMPT_ID = 'ombre.memory_l1_v2_generation_prompt'
ACTIVE_PROMPT_ID = 'ombre.memory_l1_v2_active_prompt'
INPUT_TOKENS = 32000
TOTAL_TOKENS = 64000
COMPRESSION_OUTPUT_TOKENS = 8000
CONTEXT_TOKENS = 2000
CONTINUATION_TOKENS = 4000
CROSS_DAY_MINUTES = 30
AUTO_L1_BAD_REQUEST_RETRIES = 2

_WINDOW_PREFIX = 'daily_chat_l1_v2_window:'
_EVENT_TIME_ERRORS = {'event_date_missing_from_body', 'invalid_event_time',
                      'reversed_event_range', 'inconsistent_event_timezone'}
_PROMPTS = Path(__file__).with_name('resources') / 'l1_v2'


def enabled(config):
    return (config.get('reflection') or {}).get('daily_chat_memory_extraction_mode') == PROTOCOL


def has_daily_window(authority, profile_id, day, raw_cursor, turn_cursor):
    return authority.get_meta(_WINDOW_PREFIX + _daily_run_id(profile_id, day, raw_cursor, turn_cursor)) is not None


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding('o200k_base')


def _tokens(value):
    return len(_encoding().encode_ordinary(value))


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _daily_run_id(profile_id, day, raw_cursor, turn_cursor):
    return 'l1-v2-daily-' + _digest(canonical_json([profile_id, day, raw_cursor, turn_cursor]))[:32]


def _local_time(value, tz):
    try:
        parsed = datetime.fromisoformat(str(value or '').replace('Z', '+00:00'))
        return parsed.astimezone(tz).isoformat() if parsed.tzinfo else None
    except ValueError:
        return None


def _diagnostic(engine, task_id, stage, data):
    from memory_l1_v2_diagnostics import MemoryV2Diagnostics

    MemoryV2Diagnostics(engine.memory_authority_store).record(task_id, stage, data)


def _route(engine, origin):
    stage = 'chat_l1' if origin == 'browse_l1' else origin
    route = engine._memory_task_route(stage)
    if not route['configured']:
        raise L1OwnerError(route['unavailable_reason'], field_path='generation')
    # The v2 trial has its own documented context budget and uses the owner's
    # current provider, model, effort and timeout settings.
    return {**route, 'max_tokens': TOTAL_TOKENS - INPUT_TOKENS,
            'thinking_mode': 'enabled', 'reasoning_effort': route.get('reasoning_effort') or 'low'}


def _generation_prompt(engine, origin):
    if origin == 'chat_l1':
        return engine._memory_l1_prompt(CHAT_PROMPT_ID)
    if origin != 'auto_l1':
        return engine._memory_l1_prompt(PROMPT_ID)
    # Supply the Auto L1 factory body without changing the shared Composer
    # source. The same resolver applies overrides, snapshots and wrappers.
    body, _ = engine.prompt_plan_mirror.resolve_text(
        scope='memory.l1_v2_generation', source_id=PROMPT_ID,
        live_body=(_PROMPTS / 'J-owner-1114g.txt').read_text(encoding='utf-8'),
        identity_id=str((engine.config.get('persona') or {}).get('canonical_session_id') or 'jiajia-main'),
        conversation_id='')
    return body, {'prompt_revision': 'prompt_sha256:' + _digest(body)}


def _schema_rejected(exc):
    if getattr(exc, 'status_code', None) not in (400, 422):
        return False
    body = getattr(exc, 'body', None)
    body = body.get('error', body) if isinstance(body, dict) else {}
    if not isinstance(body, dict):
        return False
    parameter = str(body.get('param') or '').casefold()
    message = str(body.get('message') or '').casefold()
    return ((parameter.startswith(('response_format', 'text.format')) or 'json_schema' in message
             or 'response_format' in message) and any(word in message for word in (
                 'not supported', 'unsupported', 'invalid schema', 'not permitted', 'not allowed',
                 'required to be supplied', 'required property')))


def _people(engine, snapshots, sources):
    from identity import model_identity

    narrator_name = model_identity(engine.config)['assistant']['name']
    participants, refs, narrators = {}, [], set()
    for snapshot, source in zip(snapshots, sources, strict=True):
        person = snapshot['participant']
        person_id = person.get('subject_id')
        if not person_id:
            refs.append(None)
            continue
        entry = participants.setdefault(person_id, {
            'ref': f'p{len(participants) + 1}', 'person_id': person_id,
            'name': person['display_name'], 'kind': person['kind'],
        })
        refs.append(entry['ref'])
        if source.get('role') == 'assistant' and entry['kind'] == 'ai' and entry['name'] == narrator_name:
            narrators.add(entry['ref'])
    narrator = next(iter(narrators)) if len(narrators) == 1 else None
    direct = None
    chat = [(source, ref) for source, ref in zip(sources, refs, strict=True)
            if source.get('namespace') in ('canonical_event', 'raw_event', 'conversation_turn')]
    if (narrator and len(participants) == 2 and chat and all(ref for _, ref in chat)
            and len({source.get('conversation_id') for source, _ in chat}) == 1
            and {source.get('role') for source, _ in chat} == {'user', 'assistant'}):
        other = next(entry for entry in participants.values() if entry['ref'] != narrator)
        if other['kind'] == 'human':
            direct = {'narrator': narrator, 'other': other['ref']}
    return list(participants.values()), refs, narrator or {'name': narrator_name}, direct


class Generation:
    def __init__(self, engine, *, origin, task_id, sources, context=None, include_review=False,
                 active_reading=False, candidate_id=None):
        from memory_l1_v2 import catalog

        self.engine, self.origin, self.task_id = engine, origin, task_id
        self.model_calls = 0
        if origin not in L1_V2_SOURCE_RULES:
            raise L1OwnerError('l1_v2_unknown_source_kind', field_path='input_kind')
        self.model_io = MemoryModelIO(engine.memory_authority_store)
        self.model_io_id = candidate_id or task_id
        self.source_rule = L1_V2_SOURCE_RULES[origin]
        self.route = _route(engine, origin)
        self.active_reading = active_reading
        self.material_position = 1
        prompt_source_id = ACTIVE_PROMPT_ID if active_reading else PROMPT_ID
        if active_reading:
            from prompt_source_registry import fixed_prompt_source

            source = fixed_prompt_source(prompt_source_id)
            identity_id = str((engine.config.get('persona') or {}).get('canonical_session_id') or 'jiajia-main')
            self.prompt_before, self.prompt_after, source_enabled = engine.prompt_plan_mirror.compile_background_messages(
                scope=source.scope, source_id=prompt_source_id, live_body=source.factory_body(), identity_id=identity_id)
            self.material_position = len(self.prompt_before)
            prompt_revision = 'prompt_sha256:' + _digest(self._wire_prompt([*self.prompt_before, {}, *self.prompt_after]))
        else:
            self.prompt, prompt_context = _generation_prompt(engine, origin)
            source_enabled = bool(self.prompt.strip())
            prompt_revision = prompt_context['prompt_revision']
        if not source_enabled:
            raise L1OwnerError('prompt_disabled', field_path='generation')
        self.compress_window = origin == 'auto_l1'
        # Leave space for the first pass output in the original-source request.
        self.planning_input_limit = INPUT_TOKENS - (COMPRESSION_OUTPUT_TOKENS if self.compress_window else 0)
        self.sources = sources
        self.snapshots, self.bindings = bind_sources(sources, config=engine.config)
        self.people, self.speakers, narrator, direct = _people(engine, self.snapshots, sources)
        self.catalog = catalog()
        self.include_review = include_review
        context = dict(context or {})
        if active_reading:
            event = context['reading_event']
            context['reading_event'] = {**event, 'occurred_at': _local_time(event['occurred_at'], engine.tz)}
        self.cross_day_sources = set(context.pop('cross_day_sources', ()))
        previous = context.get('previous_day_drafts', [])
        # Frozen windows from the earlier protocol carry source-bound draft objects.
        self.previous_day_drafts = previous if previous and isinstance(previous[0], dict) else []
        if self.previous_day_drafts:
            context.pop('previous_day_drafts')
        self.common = {'input_kind': origin, 'narrator': narrator, 'catalogs': self.catalog,
                       'people': [{key: item[key] for key in ('ref', 'name', 'kind')} for item in self.people],
                       **context}
        if direct:
            self.common['direct_conversation'] = direct
        self.generation = {'origin': origin, 'task_id': task_id, 'model': self.route['model'],
                           'prompt_source_id': prompt_source_id, 'prompt_revision': prompt_revision,
                           'route_revision': 'route_sha256:' + semantic_sha256({
                               k: v for k, v in self.route.items() if k != 'client'}),
                           'catalog_revision': self.catalog['version']}
        if origin == 'chat_l1':
            submitted_at = self.snapshots[0].get('recorded_at')
            local_submission_time = _local_time(submitted_at, engine.tz)
            if local_submission_time is None:
                raise L1OwnerError('chat_submission_time_missing', field_path='generation.submitted_at')
            self.generation.update(prompt_source_id=CHAT_PROMPT_ID, submitted_at=submitted_at)
            self.common['agent_draft'] = {**self.common['agent_draft'], 'submitted_at': local_submission_time}
        self.views = [(index, anchor['start'], anchor['end'])
                      for index, binding in enumerate(self.bindings) for anchor in binding['anchors']]

    def _attach_chat_draft(self, card):
        card['body'] = self.snapshots[0]['text']
        title = self.common['agent_draft']['title']
        if title.strip():
            card['title'] = title
        card['sources'] = list(dict.fromkeys(['m1', *card['sources']]))

    def _system_prompt(self, *, continuing):
        rules = self.source_rule
        if continuing:
            rules += '\n' + L1_V2_CONTINUATION_RULE
        slot = L1_V2_MATERIAL_PLACEHOLDER
        template = self.prompt.replace('{' + slot + '}', slot)
        before, found, after = template.partition(slot)
        if found:
            return before + rules + after.replace(slot, '')
        # Preserve owner-authored blocks created before the slot existed.
        return self.prompt + '\n\n' + rules

    def _wire_prompt(self, wire):
        if not self.active_reading:
            return wire[0]['content']
        return canonical_json([*wire[:self.material_position], None, *wire[self.material_position + 1:]])

    def _source_material(self, views):
        messages = []
        observed = {}
        ranges = {}
        for index, left, right in views:
            snapshot = self.snapshots[index]
            ref = f'm{index + 1}'
            messages.append([ref, self.speakers[index], _local_time(snapshot.get('recorded_at'), self.engine.tz),
                             snapshot['text'][left:right]])
            observed.setdefault(ref, []).append([left, right])
            if (left, right) != (0, len(snapshot['text'])):
                ranges.setdefault(ref, []).append([left, right])
        return messages, observed, ranges

    def request(self, views, *, core=None, continuation=None):
        from memory_l1_v2 import build_model_response_format

        messages, observed, ranges = self._source_material(views)
        source_refs = list(observed)
        envelope = {**self.common, 'messages': messages}
        if self.compress_window:
            envelope['window_summaries'] = []
        if ranges:
            envelope['source_ranges'] = ranges
        if self.origin == 'chat_l1':
            native = build_chat_response_format(source_refs, [person['ref'] for person in self.people],
                self.catalog, has_title=bool(self.common['agent_draft']['title'].strip()),
                include_review=self.include_review)
            wire = [{'role': 'system', 'content': self.prompt},
                    {'role': 'user', 'content': canonical_json(envelope)}]
            return self._request_with_budget(wire, native, source_refs, observed)
        if core is not None:
            envelope['core_sources'] = list(dict.fromkeys(f'm{index + 1}' for index, _, _ in core))
        if continuation:
            envelope['continuation'] = continuation
        cross_day = [ref for ref in source_refs if ref in self.cross_day_sources]
        if cross_day:
            envelope['cross_day_sources'] = cross_day
        material = {'role': 'user', 'content': canonical_json(envelope)}
        if self.active_reading:
            wire = [*self.prompt_before, material, *self.prompt_after]
        else:
            continuing = bool(continuation and (continuation.get('part_after_first') or continuation.get('draft_cards')))
            system = self._system_prompt(continuing=continuing)
            if cross_day:
                system += '\n\ncross_day_sources：这是跨天的聊天，可能能帮助理解，按需取用。'
            wire = [{'role': 'system', 'content': system}, material]
        native = build_model_response_format(source_refs, [person['ref'] for person in self.people],
            self.catalog, include_review=self.include_review, one_card=self.active_reading)
        if self.active_reading:
            cards = native['json_schema']['schema']['properties']['cards']
            cards['items']['required'].remove('body')
        return self._request_with_budget(wire, native, source_refs, observed)

    def _request_with_budget(self, wire, native, refs, observed):
        serialized = canonical_json({'messages': wire, 'response_format': native})
        # Keep room for the exact same schema in the user envelope if the
        # provider explicitly rejects native JSON Schema before processing.
        estimate = max(_tokens(serialized), _tokens(canonical_json({
            'messages': self._json_object_wire(wire, native), 'response_format': {'type': 'json_object'}})))
        return wire, native, refs, estimate, observed

    def add_window_summaries(self, request, events):
        wire, native, refs, _, observed = request
        payload = json.loads(wire[1]['content'])
        payload['window_summaries'] = events
        wire = [wire[0], {'role': 'user', 'content': canonical_json(payload)}]
        return self._request_with_budget(wire, native, refs, observed)

    def compression_request(self, request, current):
        payload = json.loads(request[0][1]['content'])
        cross_day = payload.get('cross_day_sources', [])
        original, observed, _ = self._source_material(current)
        messages = [message for message in payload['messages'] if message[0] in cross_day] + original
        material = {'messages': messages}
        if cross_day:
            material['cross_day_sources'] = cross_day
        wire = [{'role': 'system', 'content': (_PROMPTS / 'compress3-1114.txt').read_text(encoding='utf-8')},
                {'role': 'user', 'content': canonical_json(material)}]
        refs = {'type': 'string', 'enum': list(dict.fromkeys(message[0] for message in messages))}
        event = obj({'summary': {'type': 'string', 'minLength': 1, 'pattern': r'\S'},
                     'source_refs': {**array(refs), 'minItems': 1}}, ('summary', 'source_refs'))
        schema = obj({'events': array(event)}, ('events',))
        estimated = _tokens(canonical_json({'messages': wire, 'response_format': {'type': 'json_object'}}))
        if estimated > INPUT_TOKENS:
            raise L1OwnerError('l1_v2_compression_input_exceeds_budget', field_path='generation')
        return wire, schema, set(observed), estimated

    async def compress(self, request, *, part):
        wire, schema, core_refs, estimated = request
        route = {**self.route, 'max_tokens': COMPRESSION_OUTPUT_TOKENS,
                 'thinking_mode': 'disabled', 'reasoning_effort': ''}
        started = monotonic()
        meta = {'origin': self.origin, 'part': part, 'model': route['model'], 'source_count': len(core_refs),
                'prompt_sha256': _digest(wire[0]['content']), 'user_sha256': _digest(wire[1]['content']),
                'input_tokens_estimate': estimated, 'max_output_tokens': route['max_tokens']}
        io_id = None
        try:
            response, io_id = await self._complete(wire, route=route, response_format={'type': 'json_object'},
                validation_schema=schema, stage='window_compression', meta=meta)
            _diagnostic(self.engine, self.task_id, 'window_compression_return', {
                **meta, 'usage': response_usage(response), 'finish_reason': self.engine._completion_finish_reason(response),
                'duration_ms': int((monotonic() - started) * 1000)})
            validate_response(response)
            events = json.loads(self.engine._completion_content(response))['events']
            events = [event for event in events if core_refs.intersection(event['source_refs'])]
        except Exception as exc:
            if io_id:
                self.model_io.finish(io_id, error=exc)
            _diagnostic(self.engine, self.task_id, 'window_compression_issue', {
                **meta, 'issue': model_error_details(exc), 'duration_ms': int((monotonic() - started) * 1000)})
            raise
        self.model_io.finish(io_id)
        return events

    def _json_object_wire(self, wire, native):
        wire = list(wire)
        payload = json.loads(wire[self.material_position]['content'])
        payload['output_schema'] = native['json_schema']['schema']
        wire[self.material_position] = {'role': 'user', 'content': canonical_json(payload)}
        return wire

    async def _complete(self, wire, *, route, response_format, stage, meta, validation_schema=None):
        auto = self.origin == 'auto_l1'
        retries = AUTO_L1_BAD_REQUEST_RETRIES if auto else int(response_format['type'] == 'json_schema')
        io_stage = self.origin if stage == 'model' else stage
        draft_revision = (self.common.get('agent_draft') or {}).get('draft_revision')
        material_index = 1 if stage == 'window_compression' else self.material_position
        refs = dict.fromkeys(row[0] for row in json.loads(wire[material_index]['content'])['messages'])
        source_revisions = []
        for ref in refs:
            binding = self.bindings[int(ref[1:]) - 1]
            source_revisions.append({key: binding[key] for key in
                                     ('binding_id', 'source_namespace', 'source_id', 'source_revision')})
        parameters = completion_parameters(route, self.engine.temperature)
        generation = {**self.generation, 'prompt_revision': 'prompt_sha256:' + meta['prompt_sha256']}
        for attempt in range(1, retries + 2):
            if auto and getattr(self.engine, '_daily_chat_memory_cancel_requested', False):
                raise L1OwnerError('cancelled_or_disabled', field_path='generation')
            meta['attempt'] = attempt
            _diagnostic(self.engine, self.task_id, stage + '_dispatch', meta)
            io_id = self.model_io.start(self.model_io_id, stage=io_stage, task_id=self.task_id, attempt=attempt,
                messages=[], parameters=parameters, generation=generation, draft_revision=draft_revision,
                source_revisions=source_revisions, part=meta['part'])

            def capture(request):
                self.model_io.requested(io_id, request, timeout_seconds=route.get('timeout_seconds', 180))

            self.model_calls += 1
            try:
                response = await create_completion(route['client'], model=route['model'], messages=wire, route=route,
                    temperature=self.engine.temperature, response_format=response_format, capture=capture,
                    validation_schema=validation_schema)
            except asyncio.CancelledError:
                self.model_io.finish(io_id, error=L1OwnerError('worker_interrupted'))
                raise
            except Exception as exc:
                self.model_io.finish(io_id, error=exc)
                if attempt > retries:
                    raise
                issue = model_error_details(exc)
                if response_format['type'] == 'json_schema' and _schema_rejected(exc):
                    _diagnostic(self.engine, self.task_id, 'native_schema_rejected', {**meta, 'issue': issue})
                    validation_schema = response_format['json_schema']['schema']
                    wire = self._json_object_wire(wire, response_format)
                    response_format = {'type': 'json_object'}
                    material = wire[self.material_position]['content']
                    meta.update(protocol='json_object', user_sha256=_digest(material),
                        user_chars=len(material), input_tokens_estimate=_tokens(canonical_json({
                            'messages': wire, 'response_format': response_format})))
                elif auto and issue.get('http_status') == 400:
                    _diagnostic(self.engine, self.task_id, stage + '_retry', {**meta, 'issue': issue})
                    await asyncio.sleep(0.5 * attempt)
                else:
                    raise
            else:
                self.model_io.returned(io_id, response=response,
                    raw_output=self.engine._completion_content(response),
                    finish_reason=self.engine._completion_finish_reason(response))
                return response, io_id

    async def call(self, request, *, part):
        wire, native, refs, estimated, _ = request
        if estimated > INPUT_TOKENS:
            raise L1OwnerError('l1_v2_input_exceeds_budget', field_path='generation')
        started = monotonic()
        schema = canonical_json(native)
        prompt = self._wire_prompt(wire)
        material = wire[self.material_position]['content']
        payload = json.loads(material)
        meta = {'origin': self.origin, 'part': part, 'model': self.route['model'], 'protocol': 'json_schema',
                'prompt_sha256': _digest(prompt), 'user_sha256': _digest(material),
                'schema_sha256': _digest(schema), 'system_chars': len(prompt),
                'user_chars': len(material), 'schema_chars': len(schema),
                'input_tokens_estimate': _tokens(canonical_json({'messages': wire, 'response_format': native})),
                'input_tokenizer': 'o200k_base',
                'max_output_tokens': self.route['max_tokens'], 'total_token_budget': TOTAL_TOKENS,
                'thinking_mode': self.route['thinking_mode'], 'source_count': len(refs),
                'visible_message_count': len(payload['messages']), 'partial_source_ranges': payload.get('source_ranges', {}),
                'first_source_ref': refs[0] if refs else None, 'last_source_ref': refs[-1] if refs else None}
        io_id = None
        try:
            response, io_id = await self._complete(wire, route=self.route, response_format=native, stage='model', meta=meta)
            usage = response_usage(response)
            finish = self.engine._completion_finish_reason(response)
            _diagnostic(self.engine, self.task_id, 'model_return', {
                **meta, 'usage': usage, 'finish_reason': finish, 'duration_ms': int((monotonic() - started) * 1000)})
            validate_response(response)
            cards = json.loads(self.engine._completion_content(response))['cards']
            if self.origin == 'chat_l1':
                self._attach_chat_draft(cards[0])
                cards[0]['_prompt_source_id'] = CHAT_PROMPT_ID
        except Exception as exc:
            if io_id:
                self.model_io.finish(io_id, error=exc)
            _diagnostic(self.engine, self.task_id, 'model_issue', {
                **meta, 'issue': model_error_details(exc), 'duration_ms': int((monotonic() - started) * 1000)})
            raise
        self.model_io.finish(io_id)
        return cards, usage

    def prepare_reading_response(self, card):
        from memory_l1_v2 import prepare_active_reading_card

        event = self.common['reading_event']
        return prepare_active_reading_card(card, original_body=self.snapshots[0]['text'],
            receipt_ref=event['source_ref'], occurred_at=event['occurred_at'],
            article_time=_local_time(self.snapshots[1].get('recorded_at'), self.engine.tz))

    def prepare(self, draft):
        from memory_l1_v2 import prepare_card

        card = {key: value for key, value in draft.items()
                if key not in {'_observed', '_prompt_revision', '_prompt_source_id', 'review'}}
        generation = {**self.generation, 'prompt_revision': draft.get('_prompt_revision',
            self.generation['prompt_revision'])}
        if self.origin == 'chat_l1':
            self._attach_chat_draft(card)
            generation['prompt_source_id'] = draft.get('_prompt_source_id', PROMPT_ID)
        source_ref_map, bindings = {}, []
        for ref, ranges in draft['_observed'].items():
            index = int(ref[1:]) - 1
            binding, snapshot = self.bindings[index], self.snapshots[index]
            source_ref_map[ref] = binding['binding_id']
            bindings.append({**binding, 'anchors': [make_text_anchor(snapshot['text'], target_namespace='l0',
                target_id=binding['source_id'], target_revision=binding['source_revision'],
                text_view_id=snapshot['text_view_id'], start=left, end=right) for left, right in _ranges(ranges)]})
        return prepare_card(card, people=self.people, bindings=bindings, generation=generation,
                             catalog=self.catalog, source_ref_map=source_ref_map, allow_classification_gap=True)

    async def verify(self, document):
        reader = self.engine._memory_l1_sources()
        wanted = {binding['binding_id']: binding for binding in document['evidence_bindings']}
        for snapshot, binding in zip(self.snapshots, self.bindings, strict=True):
            chosen = wanted.get(binding['binding_id'])
            if chosen:
                reader.retain_copy(snapshot)
                if not (await reader.locate(chosen, verify_only=True))['eligible_source']:
                    raise L1OwnerError('source_not_currently_readable', status=409, field_path='sources')


def _continuation(drafts, refs):
    selected = []
    for index in range(len(drafts) - 1, -1, -1):
        card = drafts[index]
        # Cards already adopted on a previous day carry no refs in this request.
        # They stay in the continuation so the model can see what has been
        # recorded and skip the routine that repeats every day.
        if card.get('sources') and not refs.intersection(card['sources']):
            continue
        # Cross-segment context is an in-flight draft, not a historical L1.
        item = {key: card[key] for key in ('title', 'body', 'kind', 'sources')}
        if _tokens(canonical_json([item, *selected])) <= CONTINUATION_TOKENS:
            selected.insert(0, item)
    return {'draft_cards': selected} if selected else None


def _ranges(values):
    output = []
    for start, end in sorted(values):
        if output and start <= output[-1][1]:
            output[-1][1] = max(end, output[-1][1])
        else:
            output.append([start, end])
    return output


def _eligible_daily_cards(job, cards, core_refs, source_refs, *, part, prior_drafts):
    from memory_l1_contract import L1ContractError
    from memory_l1_v2 import validate_model_card

    accepted, errors = [], {}
    field_paths = {}
    people_refs = [person['ref'] for person in job.people]
    for card in cards:
        if not core_refs.intersection(card['sources']):
            code = 'l1_v2_card_without_core_source'
        else:
            try:
                validate_model_card({key: value for key, value in card.items() if key != 'review'},
                    source_refs=source_refs, people_refs=people_refs, catalog=job.catalog)
            except L1ContractError as exc:
                if exc.code not in _EVENT_TIME_ERRORS:
                    raise
                code = exc.code
                field_paths[code] = exc.path
            else:
                accepted.append(card)
                continue
        errors[code] = errors.get(code, 0) + 1
    for code, count in errors.items():
        _diagnostic(job.engine, job.task_id, 'card_source_rejected', {
            'origin': job.origin, 'part': part, 'code': code, 'rejected_count': count})
    if errors and not accepted and not prior_drafts:
        code = next(iter(errors))
        raise L1OwnerError(code, field_path=field_paths.get(code, 'cards.sources'))
    return accepted


def _fold(drafts, cards, core_refs, context_refs, observed, *, prompt_revision):
    from memory_l1_v2 import referenced_sources

    for card in cards:
        if not core_refs.intersection(card['sources']):
            raise L1OwnerError('l1_v2_card_without_core_source', field_path='cards.sources')
        # A continuing card cites the same observed boundary evidence. Keep
        # source coverage from its earlier draft while taking the fuller prose.
        peers = [index for index, prior in enumerate(drafts) if prior['kind'] == card['kind']
                 and context_refs.intersection(prior['sources']).intersection(card['sources'])
                 and (prior['title'] == card['title'] or set(prior['keywords']).intersection(card['keywords']))]
        refs = referenced_sources(card)
        value = {**card, '_observed': {ref: observed[ref] for ref in refs}, '_prompt_revision': prompt_revision}
        if len(peers) == 1:
            prior = drafts[peers[0]]
            if prior.get('review') and value.get('review'):
                left, right = prior['review'], value['review']
                value['review'] = {flag: left[flag] or right[flag] for flag in ('sensitive', 'transient', 'generic')}
                value['review']['confidence'] = min(left['confidence'], right['confidence'])
            else:
                value.pop('review', None)
            value['sources'] = list(dict.fromkeys([*prior['sources'], *card['sources']]))
            value['people'] = list(dict.fromkeys([*prior['people'], *card['people']]))
            value['_observed'] = {ref: _ranges([*prior['_observed'].get(ref, []), *observed.get(ref, [])])
                                  for ref in referenced_sources(value)}
            drafts[peers[0]] = value
        else:
            drafts.append(value)


def _tail(job, prior_views):
    remaining = CONTEXT_TOKENS
    result = []
    for index, start, end in reversed(prior_views):
        text = job.snapshots[index]['text']
        overhead = 24
        if remaining <= overhead:
            break
        if _tokens(text[start:end]) + overhead > remaining:
            left, right = start, end
            while left < right:
                middle = (left + right) // 2
                if _tokens(text[middle:end]) + overhead <= remaining:
                    right = middle
                else:
                    left = middle + 1
            start = left
        if start < end:
            result.append((index, start, end))
            remaining -= _tokens(text[start:end]) + overhead
    return list(reversed(result))


def _next_part(job, core, offset, earlier, drafts):
    context = _tail(job, earlier)
    context_refs = {f'm{index + 1}' for index, _, _ in context}
    # In-flight drafts need visible source references. Adopted cards from the
    # previous day are context and have no references in this request.
    continuation = _continuation([*job.previous_day_drafts, *drafts], context_refs)

    def request(views):
        details = {'part_after_first': bool(earlier), **(continuation or {})}
        return job.request([*context, *views], core=views, continuation=details)

    # Search for the largest continuous prefix that fits, including schema.
    start, end = offset + 1, len(core)
    best = None
    while start <= end:
        middle = (start + end) // 2
        value = request(core[offset:middle])
        if value[3] <= job.planning_input_limit:
            best = (middle, 0, value)
            start = middle + 1
        else:
            end = middle - 1
    if best:
        stop, split, wire = best
        return stop, split, wire, core[offset:stop], context_refs

    # A single message can exceed the model budget. Split the exact source
    # view by codepoint, with the next request resuming at this offset.
    index, left, right = core[offset]
    start, end = left + 1, right
    best = None
    while start <= end:
        middle = (start + end) // 2
        views = [(index, left, middle)]
        value = request(views)
        if value[3] <= job.planning_input_limit:
            best = (middle, value)
            start = middle + 1
        else:
            end = middle - 1
    if best is None:
        raise L1OwnerError('l1_v2_input_overhead_exceeds_budget', field_path='generation')
    end, wire = best
    return (offset + 1, 0, wire, [(index, left, end)], context_refs) if end == right else (
        offset, end, wire, [(index, left, end)], context_refs)


def _chat_request(job, views):
    request = job.request(views)
    if request[3] <= job.planning_input_limit:
        return request, views
    draft = [view for view in views if view[0] == 0]
    nearby = views[len(draft):]
    low, high = 1, len(nearby)
    while low < high:
        middle = (low + high) // 2
        if job.request([*draft, *nearby[middle:]])[3] <= job.planning_input_limit:
            high = middle
        else:
            low = middle + 1
    views = [*draft, *nearby[low:]]
    request = job.request(views)
    if request[3] > job.planning_input_limit:
        raise L1OwnerError('l1_v2_input_exceeds_budget', field_path='generation')
    return request, views


def _proposal(document, *, candidate_id, source_type, mode, metadata=None, owner_explicit=False,
              review=None):
    from memory_authority import MemoryProposal
    from memory_l1_schema import split_document

    body, semantic = split_document(document, allow_classification_gap=True)
    kind = document['kind']
    source_refs = [binding['source_namespace'] + ':' + binding['source_id'] + '@' + str(binding['source_revision'])
                   for binding in document['evidence_bindings']]
    operation = ((metadata or {}).get('browse_preparation') or {}).get('operation_id')
    if operation:
        source_refs.insert(0, 'tool_operation:' + operation)
    return MemoryProposal.from_mapping({'proposal_id': candidate_id, 'source_type': source_type,
        'proposed_body': body, 'source_status': 'verified', 'requested_mode': mode,
        'owner_explicit': owner_explicit, 'memory_type': kind,
        'confidence': (review or {}).get('confidence', 0.0),
        'sensitive': (review or {}).get('sensitive', False),
        'transient': (review or {}).get('transient', False),
        'generic': (review or {}).get('generic', False),
        'source_refs': source_refs,
        'metadata': {**(metadata or {}), 'memory_l1': semantic,
                     'confidence_source': 'model_review' if review else 'not_scored',
                     'legacy_candidate': {'title': document['title'], 'kind': kind,
                                          'source_verification': 'verified'}}})


async def generate(engine, bucket_mgr, *, origin, task_id, source_type, mode, sources,
                   context=None, core_start=0, force=False, allow_empty=True,
                   candidate_id=None, policy=None, owner_explicit=False, metadata=None,
                   required_source_ref=None, required_any_source_refs=None, before_prepare=None,
                   prepare_policy=None, expected_source_fingerprint=None, before_progress_save=None,
                   active_reading=False):
    """Durably prepare once, resume per range, and drain through the normal policy."""
    authority = engine.memory_authority_store
    queue = MemoryCandidatePreparation(authority)
    key = 'l1-v2:' + task_id
    receipt = queue.get(key)
    effective_mode = engine.memory_ingestion_policy.mode_for(source_type, mode)
    calls = 0
    if receipt is not None and expected_source_fingerprint is not None:
        if receipt.get('stale') or receipt['source_fingerprint'] != expected_source_fingerprint:
            raise L1OwnerError('l1_v2_source_changed', status=409, field_path='sources')
    if receipt is None:
        if effective_mode == 'off':
            raise L1OwnerError('memory_source_disabled', status=409, field_path='mode')
        job = Generation(engine, origin=origin, task_id=task_id, sources=sources, context=context,
                         include_review=effective_mode == 'auto' and not owner_explicit,
                         active_reading=active_reading, candidate_id=candidate_id)
        fingerprint = semantic_sha256({'sources': job.bindings, 'context': context, 'core_start': core_start})
        if expected_source_fingerprint is not None and fingerprint != expected_source_fingerprint:
            raise L1OwnerError('l1_v2_source_changed', status=409, field_path='sources')
        store = ChatSegmentProgress(authority)
        prior = store.get(key)
        state = prior['progress'] if prior else {'phase': 'ready', 'offset': 0, 'split': 0, 'parts': 0, 'drafts': []}
        revision = prior['revision'] if prior else 0
        if state.get('fingerprint', fingerprint) != fingerprint:
            raise L1OwnerError('l1_v2_source_changed', status=409, field_path='sources')
        if state['phase'] in ('dispatched', 'failed') and not force:
            code = ('model_outcome_unknown_requires_retry_decision' if state['phase'] == 'dispatched'
                    else state['issue']['code'])
            raise L1OwnerError(code, status=409, field_path='generation')

        def save(value, *, dispatch=False):
            nonlocal revision, state
            updated = store.save(key, key, {**value, 'fingerprint': fingerprint}, revision,
                                 dispatch=dispatch, guard=before_progress_save)
            revision, state = updated['revision'], updated['progress']

        def record_model_failure(exc):
            error = model_error_details(exc)
            if error.get('http_status') in (400, 401, 403, 404, 413, 422, 429) or isinstance(
                    exc, (L1OwnerError, ValueError)):
                save({**state, 'phase': 'failed', 'issue': error})

        base = [view for view in job.views if view[0] < core_start]
        original_core = [view for view in job.views if view[0] >= core_start]
        if state['phase'] != 'generated':
            while state['offset'] < len(original_core):
                if getattr(engine, '_daily_chat_memory_cancel_requested', False):
                    raise L1OwnerError('cancelled_or_disabled', field_path='generation')
                offset = state['offset']
                core = original_core.copy()
                earlier = [*base, *core[:offset]]
                if state['split']:
                    index, start, end = core[offset]
                    earlier.append((index, start, state['split']))
                    core[offset] = (index, state['split'], end)
                if offset == 0 and not state['split'] and not base and not job.previous_day_drafts:
                    whole = job.request(core)
                else:
                    whole = None
                if origin == 'chat_l1':
                    wire, sent = (whole, core) if whole and whole[3] <= job.planning_input_limit else _chat_request(
                        job, original_core)
                    stop, split, context_refs = len(original_core), 0, set()
                elif whole and whole[3] <= job.planning_input_limit:
                    stop, split, wire, sent, context_refs = len(core), 0, whole, core, set()
                else:
                    if active_reading:
                        # A one-card response must see the complete original draft and its sources together.
                        raise L1OwnerError('l1_v2_input_exceeds_budget', field_path='generation')
                    stop, split, wire, sent, context_refs = _next_part(job, core, offset, earlier, state['drafts'])
                part = state['parts'] + 1
                # Clear only the original uncertain dispatch that the owner's
                # explicit retry authorizes. Never re-dispatch from the scheduler.
                if state['phase'] == 'dispatched':
                    save({**state, 'phase': 'ready'})
                if job.compress_window:
                    compression = job.compression_request(wire, sent)
                    compressed_input = _digest(canonical_json(compression[0]))
                    cached = state.get('compression') or {}
                    if cached.get('input_sha256') != compressed_input:
                        save({**state, 'phase': 'dispatched', 'dispatch_step': 'window_compression'}, dispatch=True)
                        try:
                            events = await job.compress(compression, part=part)
                        except Exception as exc:
                            record_model_failure(exc)
                            raise
                        cached = {'input_sha256': compressed_input, 'events': events}
                        save({**state, 'phase': 'ready', 'dispatch_step': None, 'compression': cached})
                    if getattr(engine, '_daily_chat_memory_cancel_requested', False):
                        raise L1OwnerError('cancelled_or_disabled', field_path='generation')
                    wire = job.add_window_summaries(wire, cached['events'])
                    if wire[3] > INPUT_TOKENS:
                        raise L1OwnerError('l1_v2_input_exceeds_budget', field_path='generation')
                save({**state, 'phase': 'dispatched', 'dispatch_step': 'cards'}, dispatch=True)
                try:
                    cards, _usage = await job.call(wire, part=part)
                    drafts = [] if origin == 'chat_l1' else list(state['drafts'])
                    core_refs = {f'm{index + 1}' for index, _, _ in sent}
                    if origin == 'auto_l1':
                        cards = _eligible_daily_cards(job, cards, core_refs, wire[2], part=part, prior_drafts=drafts)
                    elif active_reading:
                        cards = [job.prepare_reading_response(card) for card in cards]
                    prompt_revision = 'prompt_sha256:' + _digest(job._wire_prompt(wire[0]))
                    _fold(drafts, cards, core_refs, context_refs, wire[4], prompt_revision=prompt_revision)
                except Exception as exc:
                    # A known provider rejection/validation failure can be
                    # surfaced precisely. A timeout may already have run.
                    record_model_failure(exc)
                    raise
                phase = 'generated' if stop >= len(core) else 'ready'
                save({'phase': phase, 'offset': stop, 'split': split, 'parts': part, 'drafts': drafts})
        calls = job.model_calls
        cards = state['drafts']
        if (not allow_empty and not cards) or ((candidate_id is not None or active_reading) and len(cards) != 1):
            save({**state, 'phase': 'failed', 'offset': 0, 'split': 0, 'parts': 0, 'drafts': [],
                  'issue': {'code': 'l1_v2_expected_one_card'}})
            raise L1OwnerError('l1_v2_expected_one_card', field_path='cards')
        if required_source_ref and any(required_source_ref not in card['sources'] for card in cards):
            save({**state, 'phase': 'failed', 'offset': 0, 'split': 0, 'parts': 0, 'drafts': [],
                  'issue': {'code': 'l1_v2_primary_source_required'}})
            raise L1OwnerError('l1_v2_primary_source_required', field_path='cards.sources')
        if required_any_source_refs and any(not required_any_source_refs.intersection(card['sources']) for card in cards):
            save({**state, 'phase': 'failed', 'offset': 0, 'split': 0, 'parts': 0, 'drafts': [],
                  'issue': {'code': 'l1_v2_result_source_required'}})
            raise L1OwnerError('l1_v2_result_source_required', field_path='cards.sources')
        proposals = []
        decisions = {}
        for index, card in enumerate(cards):
            document = job.prepare(card)
            await job.verify(document)
            identifier = candidate_id or 'l1_v2_' + _digest(task_id + ':' + str(index))[:40]
            proposal = _proposal(document, candidate_id=identifier, source_type=source_type, mode=mode,
                                 metadata=metadata, owner_explicit=owner_explicit, review=card.get('review'))
            proposals.append(proposal)
            if prepare_policy is not None:
                decisions[identifier] = prepare_policy(proposal)
        active_policy = policy or engine.memory_ingestion_policy
        if prepare_policy is not None:
            from types import SimpleNamespace

            active_policy = SimpleNamespace(evaluate=lambda proposal: decisions[proposal.proposal_id])
        receipt = queue.prepare(key, proposals, active_policy, scope=key,
                                source_fingerprint=fingerprint, generated_count=len(proposals), guard=before_prepare)
        # Only the immutable source fingerprint and cursor are needed after the
        # canonical proposals are committed. Drop the transient model drafts.
        save({**state, 'phase': 'prepared', 'drafts': []})
        _diagnostic(engine, task_id, 'prepared', {'origin': origin, 'model_calls': calls,
                    'source_count': len(sources), 'candidate_ids': receipt['candidate_ids'],
                    'candidate_count': len(proposals)})

    adoption_issues, handled = [], []
    for row in queue.remaining(receipt):
        identifier = row['candidate_id']
        if row['status'] == 'accepted':
            try:
                outcome = await engine._memory_l1_adopt_candidate(row, bucket_mgr, request_id=task_id + ':' + identifier)
            except Exception as exc:
                adoption_issues.append({'candidate_id': identifier, **model_error_details(exc)})
                continue
            if outcome['status'] not in ('created', 'exists', 'linked', 'pending'):
                adoption_issues.append({'candidate_id': identifier, 'code': outcome['status']})
                continue
        handled.append(identifier)
    queue.acknowledge(key, handled)
    receipt = queue.get(key)
    rows = queue.rows(receipt)
    if rows:
        _diagnostic(engine, task_id, 'adoption', {'origin': origin, 'candidate_states': {
            row['candidate_id']: row['status'] for row in rows}, 'issues': adoption_issues})
    return {'task_id': task_id, 'candidate_ids': receipt['candidate_ids'], 'rows': rows,
            'model_calls': calls, 'adoption_issues': adoption_issues}


def paused_legacy_checkpoint(audit, profile_id):
    if (audit.get('prompt_version') == PROTOCOL
            or audit.get('status') not in ('partial', 'needs_review', 'failed', 'degraded_empty_outputs')):
        return None
    return {'status': 'needs_review', 'reason': 'legacy_daily_chat_memory_requires_legacy_mode',
            'date': audit['date'], 'profile_id': profile_id, 'cursor_updated': False}


def _previous_day_cards(engine, day, profile_id):
    yesterday = (datetime.fromisoformat(day) - timedelta(days=1)).date().isoformat()
    authority = engine.memory_authority_store
    queue = MemoryCandidatePreparation(authority)
    results = {}
    for audit in reversed(engine._load_daily_chat_memory_run_audit()):
        if (audit.get('date') != yesterday or audit.get('profile_id') != profile_id
                or audit.get('prompt_version') != PROTOCOL or not audit.get('run_id')):
            continue
        receipt = queue.get('l1-v2:' + (audit.get('generation_task_id') or audit['run_id']))
        if not receipt:
            continue
        for row in queue.rows(receipt):
            candidate_id = row['candidate_id']
            if candidate_id in results or row['status'] != 'committed':
                continue
            if row['proposal']['source_type'] != 'daily_chat':
                continue
            active = authority.get_memory(candidate_id)
            if active and active['state'] == 'active' and active['recall_policy'] == 'enabled':
                results[candidate_id] = _with_prior_memory(engine, row, active['active_revision'])
    return list(reversed(results.values()))


def _with_prior_memory(engine, row, revision):
    memory_id = row['candidate_id']
    record = engine.memory_authority_store.get_memory_revision(memory_id, revision)
    if not record:
        raise L1OwnerError('l1_v2_continuation_source_unavailable', status=409, field_path='sources')
    body = row['proposal']['proposed_body']
    if body_sha256(body) != record['body_sha256']:
        projection = engine._memory_authority_service(None).projection
        snapshot = Path(record['snapshot_path']).resolve()
        if (snapshot != projection.snapshot_path(memory_id, revision).resolve()
                or not snapshot.is_relative_to(projection.revisions_dir) or not snapshot.is_file()):
            raise L1OwnerError('l1_v2_continuation_source_unavailable', status=409, field_path='sources')
        body = frontmatter.loads(snapshot.read_text(encoding='utf-8')).content
        if body_sha256(body) != record['body_sha256']:
            raise L1OwnerError('revision_body_mismatch', status=503, field_path='sources')
    return {**row, 'memory_revision': revision, 'memory_body': body,
            'memory_semantic': record['metadata']['memory_l1']}


def _prior_card_version(row):
    version = [row['candidate_id'], row['revision']]
    if 'memory_revision' in row:
        version.append(row['memory_revision'])
    return version


def _binding_origin(binding):
    return binding['source_namespace'], binding['source_id'], binding['source_revision']


def _previous_day_drafts(rows, boundary_bindings):
    if rows and 'memory_revision' in rows[0]:
        from memory_l1_v2 import current_kind
        drafts = []
        for row in rows:
            semantic = row.get('memory_semantic') or {}
            drafts.append({'title': semantic.get('title') or '',
                           'body': row['memory_body'],
                           'kind': current_kind(semantic.get('kind')) or '',
                           # Yesterday's adopted cards are reference material, not
                           # sources of today's request, so they carry no refs.
                           'sources': []})
        return drafts, [_prior_card_version(row) for row in rows]

    # Keep the input of an already frozen, source-bound window reproducible.
    available = {_binding_origin(binding): f'm{index}' for index, binding in enumerate(boundary_bindings, 1)}
    drafts, versions = [], []
    for row in reversed(rows):
        prior = row['proposal']['metadata'].get('memory_l1') or {}
        refs = list(dict.fromkeys(available[key] for binding in prior.get('evidence_bindings', [])
            if (key := _binding_origin(binding)) in available))
        if not refs:
            continue
        draft = {'title': prior['title'], 'body': row['proposal']['proposed_body'],
                 'kind': prior['kind'], 'sources': refs}
        if _tokens(canonical_json([draft, *drafts])) <= CONTINUATION_TOKENS:
            drafts.insert(0, draft)
            versions.insert(0, [row['candidate_id'], row['revision']])
    return drafts, versions


def _record_day_continuation(proposal, previous_cards):
    document = proposal.metadata['memory_l1']

    present = {_binding_origin(binding): binding['binding_id'] for binding in document['evidence_bindings']}
    links = []
    for row in previous_cards:
        prior = row.get('memory_semantic') or row['proposal']['metadata'].get('memory_l1') or {}
        if (prior.get('kind') != document['kind'] or (prior.get('title') != document['title']
                and not set(prior.get('keywords', [])).intersection(document['keywords']))):
            continue
        shared = [present[key] for binding in prior['evidence_bindings'] if (key := _binding_origin(binding)) in present]
        if shared:
            links.append({'previous_candidate_id': row['candidate_id'], 'previous_candidate_revision': row['revision'],
                          'previous_state': row['status'], 'previous_title': prior['title'],
                          'shared_source_refs': list(dict.fromkeys(shared))})
    if links:
        proposal.metadata['l1_v2_continuation'] = links


def _usable_prior_draft(authority, row):
    latest = authority.get_candidate(row['candidate_id'])
    if (not latest or latest['status'] in ('deferred', 'rejected')
            or latest['proposal_sha256'] != row['proposal_sha256']):
        return False
    if 'memory_revision' in row:
        active = authority.get_memory(row['candidate_id'])
        return bool(latest['status'] == 'committed' and active and active['state'] == 'active'
                    and active['recall_policy'] == 'enabled' and active['active_revision'] == row['memory_revision']
                    and active['body_sha256'] == body_sha256(row['memory_body']))
    if latest['status'] == 'committed':
        active = authority.get_memory(row['candidate_id'])
        return bool(active and active['active_revision'] == 1 and active['state'] == 'active' and active['recall_policy'] != 'disabled'
                    and active['body_sha256'] == body_sha256(row['proposal']['proposed_body']))
    return True


def _daily_sources(engine, *, start, end, profile_id, raw_event_store, conversation_turn_store,
                   raw_cursor, turn_cursor, source_kind=None, through=None, cross_day_minutes=CROSS_DAY_MINUTES):
    from memory_source_provenance import daily_sources

    session = str((engine.config.get('persona') or {}).get('canonical_session_id') or 'jiajia-main')
    # Older frozen windows must replay with the range that produced their fingerprint.
    lookback = timedelta(days=1) if cross_day_minutes is None else timedelta(minutes=cross_day_minutes)
    context_start = start - lookback
    if ((source_kind == 'raw_events' and raw_event_store is None)
            or (source_kind == 'conversation_turns' and conversation_turn_store is None)):
        raise L1OwnerError('l1_v2_daily_source_store_unavailable', status=503, field_path='sources')
    if source_kind != 'conversation_turns' and raw_event_store is not None:
        events = raw_event_store.list_events_between(start_at=context_start, end_at=end,
            limit=0, chronological=True, profile_id=profile_id, source='gateway', session_id=session,
            conversation_id=session)
        events = [event for event in events if event.get('role') in ('user', 'assistant')
                  and str(event.get('text') or '').strip() and (through is None or int(event['id']) <= through)]
        if events:
            before, current = [], []
            for event in events:
                stamp = _local_time(event.get('created_at'), engine.tz)
                if not stamp:
                    continue
                if int(event['id']) > raw_cursor and datetime.fromisoformat(stamp) >= start:
                    current.append(event)
                else:
                    before.append(event)
            ordered = lambda event: (_local_time(event.get('created_at'), engine.tz), int(event['id']))
            if current:
                current.sort(key=ordered)
                # Preserve true chronology, including raw events that share a round.
                first = ordered(current[0])
                before = sorted((event for event in before if ordered(event) < first), key=ordered)
                wrap = lambda values: [{'_semantic_sources': values}]
                core = daily_sources(wrap(current), profile_id=profile_id, limit=None)
                context = daily_sources(wrap(before), profile_id=profile_id, limit=None)
                core_keys = {source['source_key'] for source in core}
                context = [source for source in context if source['source_key'] not in core_keys]
                return context, core, 'raw_events', max(int(event['id']) for event in current), len(
                    engine._raw_event_turn_payloads(current, limit=0))
            if raw_cursor:
                return [], [], 'raw_events', raw_cursor, 0
    if source_kind == 'raw_events':
        return [], [], 'raw_events', raw_cursor, 0
    if conversation_turn_store is None:
        return [], [], 'conversation_turns', turn_cursor, 0
    rows = conversation_turn_store.list_conversation_turns_between(profile_id=profile_id,
        start_at=context_start, end_at=end, limit=0, chronological=True)
    rows = [row for row in rows if row.get('session_id') == session
            and (through is None or int(row['id']) <= through)]
    before, current = [], []
    for row in rows:
        stamp = _local_time(row.get('created_at'), engine.tz)
        if stamp and int(row['id']) > turn_cursor and datetime.fromisoformat(stamp) >= start:
            current.append(row)
        else:
            before.append(row)
    core = daily_sources(current, profile_id=profile_id, limit=None)
    context = daily_sources(before, profile_id=profile_id, limit=None)
    return context, core, 'conversation_turns', max((int(row['id']) for row in current), default=turn_cursor), len(current)


def _daily_window(engine, *, start, end, read_sources, source_kind=None, through=None, previous_cards=(),
                  cross_day_minutes=CROSS_DAY_MINUTES):
    context, core, turn_source, watermark, turns = read_sources(end=end, source_kind=source_kind, through=through,
                                                                cross_day_minutes=cross_day_minutes)
    tail = []
    count_tokens = 0
    for source in reversed(context):
        tail.append(source)
        count_tokens += _tokens(source['text']) + 24
        if count_tokens >= CONTEXT_TOKENS:
            break
    tail.reverse()
    envelope = {'window': {'from': start.isoformat(), 'to_exclusive': end.isoformat(), 'turns': turns}}
    if cross_day_minutes is not None:
        cross_day = [f'm{index}' for index, source in enumerate(tail, 1)
                     if (stamp := _local_time(source.get('recorded_at'), engine.tz)) and datetime.fromisoformat(stamp) < start]
        if cross_day:
            envelope['cross_day_sources'] = cross_day
    sources = [*tail, *core]
    if sources:
        _, bindings = bind_sources(sources, config=engine.config)
    else:
        bindings = []
    if previous_cards:
        drafts, versions = _previous_day_drafts(previous_cards, bindings[:len(tail)])
        if drafts:
            envelope['previous_day_drafts'] = drafts
    fingerprint = semantic_sha256({'sources': bindings, 'context': envelope, 'core_start': len(tail)})
    # Recorded time and archive identity affect the actual input independently
    # of the underlying text/author anchor fingerprint.
    source_manifest = [{key: source.get(key) for key in (
        'source_key', 'event_id', 'version_id', 'recorded_at', 'archive_ref', 'snapshot_sha256')} for source in sources]
    window = {'turn_source': turn_source, 'observed_watermark': watermark, 'end': end.isoformat(), 'turns': turns,
              'source_fingerprint': fingerprint, 'manifest_sha256': semantic_sha256({'sources': source_manifest}),
              'core_count': len(core), 'boundary_count': len(tail)}
    if cross_day_minutes is not None:
        window['cross_day_minutes'] = cross_day_minutes
    if 'previous_day_drafts' in envelope:
        window['previous_card_versions'] = versions
    return window, sources, envelope


def _save_daily_window(authority, key, window, *, previous=None):
    """Pin the window once per start cursor, using CAS for explicit replacement."""
    with closing(authority._connect()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT value FROM authority_meta WHERE key=?', (key,)).fetchone()
        current = json.loads(row['value']) if row else None
        if current == previous:
            db.execute('INSERT INTO authority_meta(key,value) VALUES(?,?) '
                       'ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, canonical_json(window)))
            return window
        if previous is not None:
            raise L1OwnerError('l1_v2_daily_window_changed', status=409, field_path='sources')
        return current


async def run_daily(engine, bucket_mgr, *, key, start, end, profile_id, mode, force,
                    raw_event_store, conversation_turn_store, raw_cursor, turn_cursor):
    for row in reversed(engine._load_daily_chat_memory_run_audit()):
        if row.get('date') == key and row.get('profile_id', profile_id) == profile_id:
            if paused := paused_legacy_checkpoint(row, profile_id):
                return {**paused, 'mode': mode}
            break
    run_id = _daily_run_id(profile_id, key, raw_cursor, turn_cursor)
    window_key = _WINDOW_PREFIX + run_id
    authority = engine.memory_authority_store
    queue = MemoryCandidatePreparation(authority)
    audit = {'run_id': run_id, 'date': key, 'mode': mode, 'profile_id': profile_id, 'prompt_version': PROTOCOL,
             'source_start_seq': raw_cursor, 'source_start_turn_id': turn_cursor,
             'processed_until_seq': raw_cursor, 'processed_until_turn_id': turn_cursor,
             'created_at': datetime.now(timezone.utc).isoformat(), 'window_count': 0, 'model_call_count': 0}

    def read_sources(*, end, source_kind, through, cross_day_minutes):
        return _daily_sources(engine, start=start, end=end, profile_id=profile_id,
            raw_event_store=raw_event_store, conversation_turn_store=conversation_turn_store,
            raw_cursor=raw_cursor, turn_cursor=turn_cursor, source_kind=source_kind, through=through,
            cross_day_minutes=cross_day_minutes)

    try:
        window = authority.get_meta(window_key)
        previous_cards = []
        if window is None:
            generation_key = 'l1-v2:' + run_id
            if queue.get(generation_key) or ChatSegmentProgress(authority).get(generation_key):
                window = authority.get_meta(window_key)
                if window is None:
                    raise L1OwnerError('l1_v2_daily_window_missing', status=409, field_path='sources')
            else:
                previous_cards = _previous_day_cards(engine, key, profile_id)
                observed, _, _ = _daily_window(engine, start=start, end=end, read_sources=read_sources,
                                                previous_cards=previous_cards)
                if not observed['core_count']:
                    return {'status': 'skipped', 'reason': 'no_new_chat_sources', 'date': key, 'mode': mode}
                window = _save_daily_window(authority, window_key, {**observed,
                    'watermark': observed['observed_watermark'], 'task_id': run_id})
        frozen_end = datetime.fromisoformat(window['end'])
        previous_cards = []
        for candidate_id, candidate_revision, *memory_revision in window.get('previous_card_versions', []):
            row = authority.get_candidate_revision(candidate_id, candidate_revision)
            if row is None:
                raise L1OwnerError('l1_v2_continuation_source_unavailable', status=409, field_path='sources')
            previous_cards.append(_with_prior_memory(engine, row, memory_revision[0]) if memory_revision else row)

        def reread_window():
            return _daily_window(engine, start=start, end=frozen_end, read_sources=read_sources,
                                 source_kind=window['turn_source'], through=window['watermark'],
                                 previous_cards=previous_cards, cross_day_minutes=window.get('cross_day_minutes'))

        def matches_window(observed):
            return observed == {name: window.get(name) for name in observed}

        observed, sources, envelope = reread_window()
        prior_changed = any(not _usable_prior_draft(authority, row) for row in previous_cards)
        if not matches_window(observed) or prior_changed:
            if not force:
                code = 'l1_v2_continuation_changed_requires_rescan' if prior_changed else 'l1_v2_daily_source_changed_requires_rescan'
                raise L1OwnerError(code, status=409, field_path='sources')
            if observed['observed_watermark'] > window['watermark']:
                raise L1OwnerError('l1_v2_daily_window_changed', status=409, field_path='sources')
            # An explicit rescan of a changed prefix gets a new source identity.
            # Preserve the old draft/revision as audit and stop replaying it.
            old_task = window['task_id']
            if previous_cards and 'memory_revision' in previous_cards[0]:
                previous_cards = _previous_day_cards(engine, key, profile_id)
            else:
                previous_cards = [row for row in previous_cards if _usable_prior_draft(authority, row)]
            observed, sources, envelope = reread_window()
            replacement = {**observed, 'watermark': window['watermark'],
                'task_id': run_id + ':' + semantic_sha256({'previous': old_task, 'window': observed})[:24],
                'supersedes_task_id': old_task}
            window = _save_daily_window(authority, window_key, replacement, previous=window)
            selected_versions = {tuple(pair) for pair in window.get('previous_card_versions', [])}
            previous_cards = [row for row in previous_cards
                              if tuple(_prior_card_version(row)) in selected_versions]
        turn_source, watermark, count = window['turn_source'], window['watermark'], window['turns']
        task_id = window['task_id']
        audit['generation_task_id'] = task_id

        def guard_active_window(db):
            active = db.execute("SELECT 1 FROM authority_meta WHERE key=? AND json_extract(value,'$.task_id')=?",
                                (window_key, task_id)).fetchone()
            if not active:
                raise L1OwnerError('l1_v2_daily_window_superseded', status=409, field_path='sources')

        def guard_preceding_cards():
            if any(not _usable_prior_draft(authority, row) for row in previous_cards):
                raise L1OwnerError('l1_v2_continuation_changed_requires_rescan', status=409, field_path='sources')

        def guard_preparation(db):
            guard_active_window(db)
            guard_preceding_cards()
            if not matches_window(reread_window()[0]):
                raise L1OwnerError('l1_v2_daily_source_changed_requires_rescan', status=409, field_path='sources')

        if window.get('supersedes_task_id'):
            superseded = window['supersedes_task_id']
            audit['supersedes_generation_task_id'] = superseded
            queue.invalidate_scope('l1-v2:' + superseded, 'l1-v2:' + task_id)
        audit.update(turn_source=turn_source, window_count=1, eligible_turn_count=count,
                     source_end_seq=watermark if turn_source == 'raw_events' else raw_cursor,
                     source_end_turn_id=watermark if turn_source == 'conversation_turns' else turn_cursor)
        _diagnostic(engine, run_id, 'sources', {'origin': 'auto_l1', 'source_count': window['core_count'],
                    'boundary_source_count': window['boundary_count'], 'source_kind': turn_source, 'turn_count': count,
                    'source_from': start.isoformat(), 'source_to_exclusive': frozen_end.isoformat(),
                    'source_watermark': watermark, 'source_manifest_sha256': window['manifest_sha256']})
        def evaluate(proposal):
            from memory_authority import PolicyDecision

            _record_day_continuation(proposal, previous_cards)
            decision = engine.memory_ingestion_policy.evaluate(proposal)
            if decision.action == 'auto_accept' and proposal.metadata.get('l1_v2_continuation'):
                # Neither equal topics nor an overlapping reply authorizes
                # overwriting an existing experience or adopting a duplicate.
                return PolicyDecision('queue_review', ('REVIEW_CONTINUING_EXPERIENCE',),
                    decision.decision_policy, requires_owner_confirmation=True)
            return decision

        if window['core_count']:
            guard_preceding_cards()
            result = await generate(engine, bucket_mgr, origin='auto_l1', task_id=task_id, source_type='daily_chat',
                mode=mode, sources=sources, context=envelope, core_start=window['boundary_count'], force=force,
                expected_source_fingerprint=window['source_fingerprint'],
                before_prepare=guard_preparation, before_progress_save=guard_active_window,
                metadata={'date': key}, prepare_policy=evaluate if previous_cards else None)
        else:
            # Explicitly revalidated withdrawal of every source in the frozen
            # prefix needs no model request. Persist the empty preparation.
            queue.prepare('l1-v2:' + task_id, [], engine.memory_ingestion_policy, scope='l1-v2:' + task_id,
                          source_fingerprint=window['source_fingerprint'], generated_count=0, guard=guard_preparation)
            result = {'rows': [], 'model_calls': 0, 'adoption_issues': []}
        rows = result['rows']
        pending = sum(row['status'] == 'pending' for row in rows)
        adopted = sum(row['status'] == 'committed' for row in rows)
        candidates = [engine._authority_candidate_legacy_item(row)['candidate'] for row in rows]
        audit.update(model_call_count=result['model_calls'], model_candidate_count=len(rows),
                     pending_count=pending, auto_applied_count=adopted)
        cursor_updated = False
        reason = ''
        if result['adoption_issues']:
            status = 'partial'
            reason = 'candidate_persistence_failed'
        else:
            with closing(authority._connect()) as db:
                guard_preparation(db)
            status = 'pending' if pending else 'created' if adopted else 'zero_candidates'
            raw = turn_source == 'raw_events'
            continuation = _daily_sources(engine, start=start, end=end, profile_id=profile_id,
                raw_event_store=raw_event_store, conversation_turn_store=conversation_turn_store,
                raw_cursor=watermark if raw else raw_cursor, turn_cursor=turn_cursor if raw else watermark,
                source_kind=turn_source)
            update_cursor = (engine._update_daily_chat_memory_raw_cursor if raw
                             else engine._update_daily_chat_memory_conversation_cursor)
            current_cursor = (engine._daily_chat_memory_last_raw_event_id if raw
                              else engine._daily_chat_memory_last_conversation_turn_id)
            try:
                cursor_updated = update_cursor(profile_id, watermark, key)
                if not cursor_updated and current_cursor(profile_id) < watermark:
                    raise RuntimeError('cursor write did not persist')
            except Exception as exc:
                raise L1OwnerError('l1_v2_daily_cursor_not_persisted', status=503, field_path='cursor') from exc
            audit['processed_until_seq' if raw else 'processed_until_turn_id'] = watermark
            if continuation[1]:
                audit['source_batch_has_more'] = True
                status, reason = 'partial', 'source_batch_exhausted'
        audit.update(status=status, error_category=reason, processed_window_count=int(not result['adoption_issues']),
                     cursor_updated=cursor_updated)
        return {**audit, 'reason': reason, 'candidates': candidates, 'turns': count,
                'adoption_issues': result['adoption_issues']}
    except Exception as exc:
        issue = model_error_details(exc)
        if audit.get('generation_task_id') and issue['code'] != 'l1_v2_daily_window_superseded':
            current = authority.get_meta(window_key)
            if current and current['task_id'] != audit['generation_task_id']:
                issue = model_error_details(L1OwnerError('l1_v2_daily_window_superseded', status=409, field_path='sources'))
        audit.update(status='partial', error_category=issue['code'], cursor_updated=False)
        _diagnostic(engine, run_id, 'generation_issue', {'origin': 'auto_l1', 'issue': issue})
        return {**audit, 'reason': issue['code'], 'issue': issue}
    finally:
        if audit.get('error_category') != 'l1_v2_daily_window_superseded' and (audit['window_count'] or audit.get('error_category')):
            engine._store_daily_chat_memory_run_audit({**audit, 'completed_at': datetime.now(timezone.utc).isoformat()})
