"""Prepare L1 v2 cards from full, authenticated source views."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from time import monotonic

from memory_candidate_preparation import MemoryCandidatePreparation
from memory_chat_segments import ChatSegmentProgress
from memory_l1_api import create_completion, validate_response
from memory_commit_service import body_sha256
from memory_l1_contract import canonical_json, make_text_anchor, semantic_sha256
from memory_l1_ingestion import bind_sources
from memory_l1_service import L1OwnerError
from memory_model_io import model_error_details, response_usage

PROTOCOL = 'whole_window_v2'
PROMPT_ID = 'ombre.memory_l1_v2_generation_prompt'
INPUT_TOKENS = 32000
TOTAL_TOKENS = 64000
CONTEXT_TOKENS = 2000
CONTINUATION_TOKENS = 4000

_WINDOW_PREFIX = 'daily_chat_l1_v2_window:'


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
    stage = {'browse_l1': 'chat_l1', 'attachment_l1': 'auto_l1', 'owner_summary': 'auto_l1'}.get(origin, origin)
    route = engine._memory_task_route(stage)
    if not route['configured']:
        raise L1OwnerError(route['unavailable_reason'], field_path='generation')
    # The v2 trial has its own documented context budget and uses the owner's
    # current provider, model, effort and timeout settings.
    return {**route, 'max_tokens': TOTAL_TOKENS - INPUT_TOKENS,
            'thinking_mode': 'enabled', 'reasoning_effort': route.get('reasoning_effort') or 'low'}


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
    def __init__(self, engine, *, origin, task_id, sources, context=None, include_review=False):
        from memory_l1_v2 import catalog

        self.engine, self.origin, self.task_id = engine, origin, task_id
        self.route = _route(engine, origin)
        self.prompt, prompt_context = engine._memory_l1_prompt(PROMPT_ID)
        if not self.prompt.strip():
            raise L1OwnerError('prompt_disabled', field_path='generation')
        self.sources = sources
        self.snapshots, self.bindings = bind_sources(sources, config=engine.config)
        self.people, self.speakers, narrator, direct = _people(engine, self.snapshots, sources)
        self.catalog = catalog()
        self.include_review = include_review
        self.previous_day_drafts = (context or {}).get('previous_day_drafts') or []
        context = {key: value for key, value in (context or {}).items() if key != 'previous_day_drafts'}
        self.summary_body = ((context or {}).get('owner_selected_summary') or {}).get('body')
        if self.summary_body is not None:
            context = {**context, 'owner_selected_summary': {key: value for key, value in
                       context['owner_selected_summary'].items() if key != 'body'}}
        self.common = {'input_kind': origin, 'narrator': narrator, 'catalogs': self.catalog,
                       'people': [{key: item[key] for key in ('ref', 'name', 'kind')} for item in self.people],
                       **(context or {})}
        if direct:
            self.common['direct_conversation'] = direct
        self.generation = {'origin': origin, 'task_id': task_id, 'model': self.route['model'],
                           'prompt_source_id': PROMPT_ID, 'prompt_revision': prompt_context['prompt_revision'],
                           'route_revision': 'route_sha256:' + semantic_sha256({
                               k: v for k, v in self.route.items() if k != 'client'}),
                           'catalog_revision': self.catalog['version']}
        self.views = [(index, anchor['start'], anchor['end'])
                      for index, binding in enumerate(self.bindings) for anchor in binding['anchors']]

    def request(self, views, *, core=None, continuation=None):
        from memory_l1_v2 import build_model_response_format

        source_refs = list(dict.fromkeys(f'm{index + 1}' for index, _, _ in views))
        messages = []
        ranges = {}
        for index, left, right in views:
            snapshot = self.snapshots[index]
            ref = f'm{index + 1}'
            messages.append([ref, self.speakers[index], _local_time(snapshot.get('recorded_at'), self.engine.tz),
                             snapshot['text'][left:right]])
            if (left, right) != (0, len(snapshot['text'])):
                ranges.setdefault(ref, []).append([left, right])
        envelope = {**self.common, 'messages': messages}
        if self.summary_body is not None:
            summary = envelope['owner_selected_summary']
            envelope['owner_selected_summary'] = {**summary, **self._summary_view(views)}
        if ranges:
            envelope['source_ranges'] = ranges
        if core is not None:
            envelope['core_sources'] = list(dict.fromkeys(f'm{index + 1}' for index, _, _ in core))
        if continuation:
            envelope['continuation'] = continuation
        wire = [{'role': 'system', 'content': self.prompt}, {'role': 'user', 'content': canonical_json(envelope)}]
        native = build_model_response_format(source_refs, [person['ref'] for person in self.people],
            self.catalog, include_review=self.include_review)
        serialized = canonical_json({'messages': wire, 'response_format': native})
        # Keep room for the exact same schema in the user envelope if the
        # provider explicitly rejects native JSON Schema before processing.
        estimate = max(_tokens(serialized), _tokens(canonical_json({
            'messages': self._json_object_wire(wire, native), 'response_format': {'type': 'json_object'}})))
        observed = {}
        for index, left, right in views:
            observed.setdefault(f'm{index + 1}', []).append([left, right])
        return wire, native, source_refs, estimate, observed

    def _summary_view(self, views):
        if views == self.views or not views:
            return {'body': self.summary_body}
        # The summary provides focus; only the actual source messages can
        # support the memory. Long summaries follow their original order.
        length = len(self.summary_body)
        first, left, _ = views[0]
        last, _, right = views[-1]
        start = int(length * (first + left / len(self.snapshots[first]['text'])) / len(self.snapshots))
        end = int(length * (last + right / len(self.snapshots[last]['text'])) / len(self.snapshots))
        return {'body': self.summary_body[start:end], 'observed_range': {'start': start, 'end': end},
                'total_characters': length}

    @staticmethod
    def _json_object_wire(wire, native):
        payload = json.loads(wire[1]['content'])
        payload['output_schema'] = native['json_schema']['schema']
        return [wire[0], {'role': 'user', 'content': canonical_json(payload)}]

    async def call(self, request, *, part):
        wire, native, refs, estimated, _ = request
        if estimated > INPUT_TOKENS:
            raise L1OwnerError('l1_v2_input_exceeds_budget', field_path='generation')
        started = monotonic()
        schema = canonical_json(native)
        payload = json.loads(wire[1]['content'])
        meta = {'origin': self.origin, 'part': part, 'model': self.route['model'], 'protocol': 'json_schema',
                'prompt_sha256': _digest(wire[0]['content']), 'user_sha256': _digest(wire[1]['content']),
                'schema_sha256': _digest(schema), 'system_chars': len(wire[0]['content']),
                'user_chars': len(wire[1]['content']), 'schema_chars': len(schema),
                'input_tokens_estimate': _tokens(canonical_json({'messages': wire, 'response_format': native})),
                'input_tokenizer': 'o200k_base',
                'max_output_tokens': self.route['max_tokens'], 'total_token_budget': TOTAL_TOKENS,
                'thinking_mode': self.route['thinking_mode'], 'source_count': len(refs),
                'visible_message_count': len(payload['messages']), 'partial_source_ranges': payload.get('source_ranges', {}),
                'first_source_ref': refs[0] if refs else None, 'last_source_ref': refs[-1] if refs else None}
        _diagnostic(self.engine, self.task_id, 'model_dispatch', meta)
        try:
            try:
                response = await create_completion(self.route['client'], model=self.route['model'],
                    messages=wire, route=self.route, temperature=self.engine.temperature, response_format=native)
            except Exception as exc:
                if not _schema_rejected(exc):
                    raise
                _diagnostic(self.engine, self.task_id, 'native_schema_rejected', {
                    **meta, 'issue': model_error_details(exc)})
                fallback = self._json_object_wire(wire, native)
                response_format = {'type': 'json_object'}
                meta = {**meta, 'protocol': 'json_object', 'user_sha256': _digest(fallback[1]['content']),
                        'user_chars': len(fallback[1]['content']), 'input_tokens_estimate': _tokens(canonical_json({
                            'messages': fallback, 'response_format': response_format}))}
                _diagnostic(self.engine, self.task_id, 'model_dispatch', meta)
                response = await create_completion(self.route['client'], model=self.route['model'],
                    messages=fallback, route=self.route, temperature=self.engine.temperature,
                    response_format=response_format, validation_schema=native['json_schema']['schema'])
            usage = response_usage(response)
            finish = self.engine._completion_finish_reason(response)
            _diagnostic(self.engine, self.task_id, 'model_return', {
                **meta, 'usage': usage, 'finish_reason': finish, 'duration_ms': int((monotonic() - started) * 1000)})
            validate_response(response)
            return json.loads(self.engine._completion_content(response))['cards'], usage
        except Exception as exc:
            _diagnostic(self.engine, self.task_id, 'model_issue', {
                **meta, 'issue': model_error_details(exc), 'duration_ms': int((monotonic() - started) * 1000)})
            raise

    def prepare(self, draft):
        from memory_l1_v2 import prepare_card

        card = {key: value for key, value in draft.items() if key not in {'_observed', 'review'}}
        source_ref_map, bindings = {}, []
        for ref, ranges in draft['_observed'].items():
            index = int(ref[1:]) - 1
            binding, snapshot = self.bindings[index], self.snapshots[index]
            source_ref_map[ref] = binding['binding_id']
            bindings.append({**binding, 'anchors': [make_text_anchor(snapshot['text'], target_namespace='l0',
                target_id=binding['source_id'], target_revision=binding['source_revision'],
                text_view_id=snapshot['text_view_id'], start=left, end=right) for left, right in _ranges(ranges)]})
        return prepare_card(card, people=self.people, bindings=bindings, generation=self.generation,
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
        if not refs.intersection(card['sources']):
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


def _fold(drafts, cards, core_refs, context_refs, observed):
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
        value = {**card, '_observed': {ref: observed[ref] for ref in refs}}
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
    # Earlier-day drafts use the current request's source numbers, and only
    # travel when the corresponding boundary evidence actually fits.
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
        if value[3] <= INPUT_TOKENS:
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
        if value[3] <= INPUT_TOKENS:
            best = (middle, value)
            start = middle + 1
        else:
            end = middle - 1
    if best is None:
        raise L1OwnerError('l1_v2_input_overhead_exceeds_budget', field_path='generation')
    end, wire = best
    return (offset + 1, 0, wire, [(index, left, end)], context_refs) if end == right else (
        offset, end, wire, [(index, left, end)], context_refs)


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
        'owner_explicit': owner_explicit, 'memory_type': kind or 'classification_gap',
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
                   prepare_policy=None, expected_source_fingerprint=None, before_progress_save=None):
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
                         include_review=effective_mode == 'auto' and not owner_explicit)
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
                if offset == 0 and not state['split'] and not base:
                    whole = job.request(core)
                else:
                    whole = None
                if whole and whole[3] <= INPUT_TOKENS:
                    stop, split, wire, sent, context_refs = len(core), 0, whole, core, set()
                else:
                    stop, split, wire, sent, context_refs = _next_part(job, core, offset, earlier, state['drafts'])
                part = state['parts'] + 1
                # Clear only the original uncertain dispatch that the owner's
                # explicit retry authorizes. Never re-dispatch from the scheduler.
                if state['phase'] == 'dispatched':
                    save({**state, 'phase': 'ready'})
                save({**state, 'phase': 'dispatched'}, dispatch=True)
                calls += 1
                try:
                    cards, _usage = await job.call(wire, part=part)
                    drafts = list(state['drafts'])
                    _fold(drafts, cards, {f'm{index + 1}' for index, _, _ in sent}, context_refs, wire[4])
                except Exception as exc:
                    # A known provider rejection/validation failure can be
                    # surfaced precisely. A timeout may already have run.
                    error = model_error_details(exc)
                    if error.get('http_status') in (400, 401, 403, 404, 413, 422, 429) or isinstance(
                            exc, (L1OwnerError, ValueError)):
                        save({**state, 'phase': 'failed', 'issue': error})
                    raise
                phase = 'generated' if stop >= len(core) else 'ready'
                save({'phase': phase, 'offset': stop, 'split': split, 'parts': part, 'drafts': drafts})
        cards = state['drafts']
        if (not allow_empty and not cards) or (candidate_id is not None and len(cards) != 1):
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


def _active_legacy(engine, day, profile_id):
    for row in reversed(engine._load_daily_chat_memory_run_audit()):
        if row.get('date') == day and row.get('profile_id', profile_id) == profile_id:
            return (row.get('prompt_version') != PROTOCOL and row.get('status') in ('partial', 'needs_review'))
    return False


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
        if not receipt or receipt.get('stale'):
            continue
        for row in queue.rows(receipt):
            candidate_id = row['candidate_id']
            if candidate_id in results or row['status'] not in ('pending', 'accepted', 'commit_failed', 'committed'):
                continue
            if row['status'] == 'committed':
                active = authority.get_memory(candidate_id)
                if (not active or active['active_revision'] != 1 or active['state'] != 'active' or active['recall_policy'] == 'disabled'
                        or active['body_sha256'] != body_sha256(row['proposal']['proposed_body'])):
                    continue
            results[candidate_id] = row
    return list(reversed(results.values()))


def _binding_origin(binding):
    return binding['source_namespace'], binding['source_id'], binding['source_revision']


def _previous_day_drafts(rows, boundary_bindings):
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
        prior = row['proposal']['metadata'].get('memory_l1') or {}
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
    if latest['status'] == 'committed':
        active = authority.get_memory(row['candidate_id'])
        return bool(active and active['active_revision'] == 1 and active['state'] == 'active' and active['recall_policy'] != 'disabled'
                    and active['body_sha256'] == body_sha256(row['proposal']['proposed_body']))
    return True


def _daily_sources(engine, *, start, end, profile_id, raw_event_store, conversation_turn_store,
                   raw_cursor, turn_cursor, source_kind=None, through=None):
    from memory_source_provenance import daily_sources

    session = str((engine.config.get('persona') or {}).get('canonical_session_id') or 'jiajia-main')
    if ((source_kind == 'raw_events' and raw_event_store is None)
            or (source_kind == 'conversation_turns' and conversation_turn_store is None)):
        raise L1OwnerError('l1_v2_daily_source_store_unavailable', status=503, field_path='sources')
    if source_kind != 'conversation_turns' and raw_event_store is not None:
        # Use the same exact main-chat archive scope as the whole-window trial.
        # Include the preceding date for bounded boundary context and read the
        # full current interval before applying any model segmentation.
        events = raw_event_store.list_events_between(start_at=start - timedelta(days=1), end_at=end,
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
        start_at=start - timedelta(days=1), end_at=end, limit=0, chronological=True)
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


def _daily_window(engine, *, start, end, read_sources, source_kind=None, through=None, previous_cards=()):
    context, core, turn_source, watermark, turns = read_sources(end=end, source_kind=source_kind, through=through)
    tail = []
    count_tokens = 0
    for source in reversed(context):
        tail.append(source)
        count_tokens += _tokens(source['text']) + 24
        if count_tokens >= CONTEXT_TOKENS:
            break
    tail.reverse()
    envelope = {'window': {'from': start.isoformat(), 'to_exclusive': end.isoformat(), 'turns': turns}}
    sources = [*tail, *core]
    if sources:
        _, bindings = bind_sources(sources, config=engine.config)
    else:
        bindings = []
    if previous_cards and tail:
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
    if _active_legacy(engine, key, profile_id):
        return None
    run_id = _daily_run_id(profile_id, key, raw_cursor, turn_cursor)
    window_key = _WINDOW_PREFIX + run_id
    authority = engine.memory_authority_store
    queue = MemoryCandidatePreparation(authority)
    audit = {'run_id': run_id, 'date': key, 'mode': mode, 'profile_id': profile_id, 'prompt_version': PROTOCOL,
             'source_start_seq': raw_cursor, 'source_start_turn_id': turn_cursor,
             'processed_until_seq': raw_cursor, 'processed_until_turn_id': turn_cursor,
             'created_at': datetime.now(timezone.utc).isoformat(), 'window_count': 0, 'model_call_count': 0}

    def read_sources(*, end, source_kind=None, through=None):
        return _daily_sources(engine, start=start, end=end, profile_id=profile_id,
            raw_event_store=raw_event_store, conversation_turn_store=conversation_turn_store,
            raw_cursor=raw_cursor, turn_cursor=turn_cursor, source_kind=source_kind, through=through)

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
        previous_cards = [authority.get_candidate_revision(candidate_id, revision)
                          for candidate_id, revision in window.get('previous_card_versions', [])]
        if any(row is None for row in previous_cards):
            raise L1OwnerError('l1_v2_continuation_source_unavailable', status=409, field_path='sources')

        def reread_window():
            return _daily_window(engine, start=start, end=frozen_end, read_sources=read_sources,
                                 source_kind=window['turn_source'], through=window['watermark'],
                                 previous_cards=previous_cards)

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
            previous_cards = [row for row in previous_cards if _usable_prior_draft(authority, row)]
            observed, sources, envelope = reread_window()
            replacement = {**observed, 'watermark': window['watermark'],
                'task_id': run_id + ':' + semantic_sha256({'previous': old_task, 'window': observed})[:24],
                'supersedes_task_id': old_task}
            window = _save_daily_window(authority, window_key, replacement, previous=window)
            selected_versions = {tuple(pair) for pair in window.get('previous_card_versions', [])}
            previous_cards = [row for row in previous_cards
                              if (row['candidate_id'], row['revision']) in selected_versions]
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
