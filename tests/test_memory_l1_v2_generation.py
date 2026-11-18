"""Normal Reflection ingestion and native v2 requests with a scripted provider."""
import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from chat_memory_tool import ChatMemoryTool
from memory_candidate_preparation import MemoryCandidatePreparation
from memory_chat_segments import ChatSegmentProgress
from memory_chat_preparation import prepare_chat_tasks
from memory_l1_service import L1OwnerError
from memory_l1_v2_diagnostics import MemoryV2Diagnostics
from memory_l1_v2_generation import Generation, INPUT_TOKENS, PROMPT_ID
from memory_model_io import MemoryModelIO
from raw_events import RawEventStore
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from memory_l1_v2_chat import CHAT_L1_PROMPT, CHAT_PROMPT_ID


DAY = '2026-11-12'


def _engine(path, reply, *, mode='review', compress=lambda _payload: []):
    config = reflection_config(path, mode=mode)
    config['reflection'].update(daily_chat_memory_extraction_mode='whole_window_v2',
                                daily_chat_memory_turn_limit=1, daily_chat_memory_window_turns=2)
    config['persona'] = {'canonical_session_id': 'main', 'profile_id': 'profile'}
    config['gateway']['canonical_continuation'] = {'enabled': True, 'conversation_id': 'canonical-main'}
    engine = ReflectionEngine(config)
    calls = []

    async def create(**request):
        payload = json.loads(request['messages'][-1]['content'])
        if request['response_format']['type'] == 'json_object' and 'input_kind' not in payload:
            result = {'events': compress(payload)}
        else:
            calls.append((request, payload))
            result = {'cards': reply(payload)}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(result, ensure_ascii=False)), finish_reason='stop')],
            usage={'prompt_tokens': 100, 'completion_tokens': 200, 'total_tokens': 300})

    client = SimpleNamespace(base_url='https://fixture.invalid/v1',
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def options(**request_options):
        assert request_options['max_retries'] == 0
        return client

    client.with_options = options
    engine.client, engine.model = client, 'fixture-v2'
    return engine, calls


def _events(engine, texts, *, start=None):
    store = RawEventStore(engine.config)
    start = start or datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
    records = [{'source_event_id': f'event-{index}', 'role': role, 'text': text,
        'created_at': (start + timedelta(seconds=index)).isoformat(),
        'session_id': 'main', 'conversation_id': 'main', 'metadata': {'profile_id': 'profile', 'round_id': index // 2}}
        for index, (role, text) in enumerate(texts)]
    result = store.ingest(records, source='gateway')
    assert result['inserted'] == len(records), result
    return store


def _run(engine, store, *, force=False):
    return asyncio.run(engine.run_daily_chat_memory(None, raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=DAY, chat_windows_only=True, force=force))


def _card(sources):
    return {'title': '雨夜的约定', 'body': '2026年11月12日，我记得那晚我们一起等雨停，约好第二天再走。',
            'kind': 'commitment', 'stability': 'episode', 'keywords': ['雨夜', '等雨停'],
            'people': [], 'sources': sources, 'primary_topic': 'daily_life',
            'event_time': {'start': DAY, 'precision': 'day', 'sources': [sources[-1]]}}


def test_attachment_is_not_an_l1_v2_generation_origin():
    with pytest.raises(L1OwnerError, match='l1_v2_unknown_source_kind'):
        Generation(None, origin='attachment_l1', task_id='retired', sources=[])


def test_daily_window_delivers_every_original_then_queues_cards(tmp_path):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['messages'][0][0], payload['messages'][-1][0]])])
    texts = [('user' if index % 2 == 0 else 'assistant', f'第{index}条对话，在旧小窗限制之外仍需要看到原话。')
             for index in range(92)]
    store = _events(engine, texts)
    result = _run(engine, store)
    assert result['status'] == 'pending', result
    assert result['cursor_updated'] and result['model_call_count'] == 2
    assert result['candidates'][0]['date'] == DAY
    assert len(calls) == 1 and [row[3] for row in calls[0][1]['messages']] == [text for _, text in texts]
    request, payload = calls[0]
    assert payload['window']['turns'] == 46
    assert request['max_tokens'] == 32000 and request['extra_body']['thinking']['type'] == 'enabled'
    assert 'statements' not in json.dumps(request['response_format'])
    system = request['messages'][0]['content']
    assert '一事一卡，不写整天日报' in system and 'primary_topic' in system
    assert system.count('自动记忆：') == 1
    assert not any(label in system for label in (
        '主动记忆：', '浏览记忆：', '行动记忆：', '附件记忆：', '手动摘要：', '续接：', '{l1_material_rules}'))
    generation = result['candidates'][0]['l1_document']['generation']
    assert generation['prompt_source_id'] == PROMPT_ID
    assert generation['prompt_revision'] == 'prompt_sha256:' + sha256(system.encode()).hexdigest()
    recorder = MemoryModelIO(engine.memory_authority_store)
    saved = recorder.list(result['candidates'][0]['id'])['items']
    assert saved == recorder.list(generation['task_id'])['items']
    assert [row['stage'] for row in saved] == ['auto_l1', 'window_compression']
    assert saved[0]['native_request'] == request
    assert saved[0]['messages'] == request['messages']
    assert json.loads(saved[1]['raw_output']) == {'events': []}
    assert engine.memory_authority_store.get_memory(result['candidates'][0]['id']) is None
    assert _run(engine, store)['status'] == 'skipped' and len(calls) == 1
    diagnostic = MemoryV2Diagnostics(engine.memory_authority_store).list(result['run_id'])
    returned = next(item['data'] for item in diagnostic if item['stage'] == 'model_return')
    assert returned['usage']['total_tokens'] == 300 and returned['input_tokens_estimate'] <= INPUT_TOKENS
    assert '在旧小窗限制之外' not in json.dumps(diagnostic, ensure_ascii=False)


def test_compression_refs_are_checked_and_persisted_before_writing_cards(tmp_path):
    event = {'summary': '我们原本约好在雨中等车，后来等来了迟到的车。', 'source_refs': ['m1']}
    received = []

    def compress(payload):
        received.append(payload)
        if len(received) == 1:
            raise TimeoutError('fixture compression outcome unknown')
        if len(received) == 2:
            return [{**event, 'source_refs': ['m99']}]
        return [event]

    def reply(payload):
        assert payload['window_summaries'] == [event]
        assert payload['messages'] == received[-1]['messages']
        if len(calls) == 1:
            raise TimeoutError('fixture card outcome unknown')
        return [_card(['m1'])]

    engine, calls = _engine(tmp_path, reply, compress=compress)
    store = _events(engine, [('user', '那次真的等到了迟到的车。')])
    first = _run(engine, store)
    assert first['status'] == 'partial' and not first['cursor_updated'] and not calls
    assert _run(engine, store)['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert len(received) == 1
    invalid = _run(engine, store, force=True)
    assert invalid['reason'] == 'native_l1_schema_mismatch' and not calls
    assert len(received) == 2
    retry = _run(engine, store, force=True)
    assert retry['status'] == 'partial' and not retry['cursor_updated']
    assert len(received) == 3 and len(calls) == 1
    assert set(received[-1]) == {'messages'}
    assert _run(engine, store)['reason'] == 'model_outcome_unknown_requires_retry_decision'
    completed = _run(engine, store, force=True)
    assert completed['status'] == 'pending' and completed['cursor_updated']
    assert completed['model_call_count'] == 1 and len(received) == 3 and len(calls) == 2


def _bad_request(message='Invalid request', *, param=None):
    response = httpx.Response(400, request=httpx.Request('POST', 'https://fixture.invalid/v1/chat/completions'))
    return BadRequestError(message, response=response, body={'error': {'message': message, 'param': param}})


@pytest.mark.parametrize('step', ['compression', 'cards'])
@pytest.mark.parametrize('recovers', [True, False])
def test_auto_l1_retries_only_failed_400_step_at_most_twice(tmp_path, step, recovers):
    compression_inputs = []
    event = {'summary': '那晚一起等雨停。', 'source_refs': ['m1']}

    def should_reject(attempt):
        return attempt < 3 or not recovers

    def compress(payload):
        compression_inputs.append(payload)
        if step == 'compression' and should_reject(len(compression_inputs)):
            raise _bad_request()
        return [event]

    def reply(payload):
        assert payload['window_summaries'] == [event]
        if step == 'cards' and should_reject(len(calls)):
            raise _bad_request()
        return [_card(['m1'])]

    engine, calls = _engine(tmp_path, reply, compress=compress)
    store = _events(engine, [('user', '我们那晚是在一起等雨停。')])

    result = _run(engine, store)

    assert len(compression_inputs) == (3 if step == 'compression' else 1)
    assert all(value == compression_inputs[0] for value in compression_inputs)
    assert len(calls) == (3 if step == 'cards' else int(recovers))
    if calls:
        assert all(wire == calls[0][0] for wire, _ in calls)
    if recovers:
        assert result['status'] == 'pending' and result['model_call_count'] == 4
        assert result['cursor_updated'] and len(result['candidates']) == 1
    else:
        assert result['issue']['http_status'] == 400 and not result['cursor_updated']
        _run(engine, store)
        assert len(compression_inputs) + len(calls) == (3 if step == 'compression' else 4)
    diagnostics = MemoryV2Diagnostics(engine.memory_authority_store).list(result['generation_task_id'])
    retry_stage = 'window_compression_retry' if step == 'compression' else 'model_retry'
    assert sorted(row['data']['attempt'] for row in diagnostics if row['stage'] == retry_stage) == [1, 2]


def test_auto_l1_schema_fallback_uses_the_same_retry_budget(tmp_path):
    def reply(payload):
        if len(calls) == 1:
            raise _bad_request('json_schema is not supported', param='response_format')
        if len(calls) == 2:
            raise _bad_request()
        return [_card(['m1'])]

    engine, calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '我们那晚是在一起等雨停。')])

    result = _run(engine, store)

    assert result['status'] == 'pending' and result['model_call_count'] == 4
    assert [wire['response_format']['type'] for wire, _ in calls] == ['json_schema', 'json_object', 'json_object']
    assert calls[1] == calls[2]
    assert calls[1][1]['output_schema'] == calls[0][0]['response_format']['json_schema']['schema']
    records = MemoryModelIO(engine.memory_authority_store).list(result['candidates'][0]['id'])['items']
    assert [row['status'] for row in records] == ['succeeded', 'failed', 'failed', 'succeeded']
    assert [row['native_request'] for row in reversed(records[:3])] == [wire for wire, _ in calls]
    assert all(row['error']['http_status'] == 400 and row['raw_output'] is None for row in records[1:3])


@pytest.mark.parametrize('owner_text', ['起点\n{l1_material_rules}\n落点', '早先保存的主人原稿', ''])
def test_owner_l1_block_inserts_source_rules_in_place_and_respects_off(tmp_path, owner_text):
    engine, calls = _engine(tmp_path, lambda _payload: [])
    scope = 'memory.l1_v2_generation'
    store = _events(engine, [('user', '我想明天再看雨。')])
    engine.prompt_plan_mirror.put_plan('preset:l1-v2', 1, plan_sha256='b' * 64,
        gateway_slice={'blocks': [{
            'block_id': 'owner:l1', 'source_id': PROMPT_ID, 'scope': scope,
            'stage': 'ombre_post_injection', 'mode': 'owner_override' if owner_text else 'off',
            'role': 'system', 'lane': 'instruction', 'anchor': 'instructions.end',
            'order': 10, 'priority': 10, 'enabled': True, 'owner_body': owner_text,
            'wrapper_text': '外层开始\n{{content}}\n外层结束',
        }]})
    engine.prompt_plan_mirror.put_binding(scope, preset_id='preset:l1-v2', preset_revision=1,
        plan_sha256='b' * 64, aiz_binding_revision=1, identity_id='main')
    result = _run(engine, store)
    if not owner_text:
        assert result['reason'] == 'prompt_disabled' and not calls
        return
    assert result['status'] == 'zero_candidates'
    system = calls[0][0]['messages'][0]['content']
    assert system.count('自动记忆：') == 1
    assert '{l1_material_rules}' not in system and '主动记忆：' not in system
    if '起点' in owner_text:
        assert system.index('外层开始') < system.index('起点') < system.index('自动记忆：')
        assert system.index('自动记忆：') < system.index('落点') < system.index('外层结束')
    else:
        assert system.startswith('外层开始\n' + owner_text + '\n外层结束\n\n自动记忆：')


@pytest.mark.parametrize('invalid_field', ['kind', 'primary_topic', 'event_time'])
def test_model_requires_a_classification_primary_topic_and_event_time(tmp_path, invalid_field):
    def reply(payload):
        card = _card([payload['messages'][0][0]])
        if invalid_field == 'kind':
            card['kind'] = None
        else:
            del card[invalid_field]
        return [card]

    engine, calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '雨很大，我们留在屋里直到天亮。')])
    result = _run(engine, store)
    assert result['status'] == 'partial' and result['reason'] == 'native_l1_schema_mismatch'
    assert len(calls) == 1
    schema = calls[0][0]['response_format']['json_schema']['schema']['properties']['cards']['items']
    assert 'primary_topic' in schema['required']
    assert 'event_time' in schema['required']
    assert 'secondary_topic' not in schema['required']
    assert set(schema['properties']['kind']['enum']) >= {'person_relationship', 'expressed_need'}
    assert None not in schema['properties']['kind']['enum']
    record = MemoryModelIO(engine.memory_authority_store).list(result['generation_task_id'])['items'][0]
    assert record['status'] == 'failed' and record['finish_reason'] == 'stop'
    assert record['error']['failure_phase'] == 'response_processing'
    assert record['error']['code'] == 'native_l1_schema_mismatch'
    assert record['raw_output'] and record['native_request'] == calls[0][0]


def test_v2_uses_saved_auto_policy_and_retains_sensitive_drafts(tmp_path):
    from memory_authority import MemoryIngestionPolicy

    def reply(payload):
        ref = payload['messages'][0][0]
        ready = {**_card([ref]), 'review': {'confidence': .9, 'sensitive': False,
                                           'transient': False, 'generic': False}}
        return [ready,
                {**ready, 'title': '需要主人决定的记录', 'review': {**ready['review'], 'sensitive': True}},
                {**ready, 'title': '未授权的类别', 'kind': 'personal_state'},
                {**ready, 'title': '不足以自动采用的记录', 'review': {**ready['review'], 'confidence': .3}}]

    engine, calls = _engine(tmp_path, reply, mode='auto')
    engine.memory_ingestion_policy = MemoryIngestionPolicy({'auto_types': ['commitment'], 'auto_confidence': .8})
    store = _events(engine, [('user', '雨很大，我们留在屋里直到天亮。')])
    from bucket_manager import BucketManager
    result = asyncio.run(engine.run_daily_chat_memory(BucketManager(engine.config), raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=DAY, chat_windows_only=True))
    assert len(calls) == 1 and result['auto_applied_count'] == 1 and result['pending_count'] == 3
    assert 'review' in calls[0][0]['response_format']['json_schema']['schema']['properties']['cards']['items']['properties']
    candidates = result['candidates']
    assert candidates[0]['status'] == 'applied'
    assert all(candidate['status'] == 'pending' for candidate in candidates[1:])
    assert engine.memory_authority_store.get_candidate(candidates[1]['id'])['policy']['reason_codes'] == ['REVIEW_SENSITIVE']
    assert 'review' not in candidates[0]['l1_document']


def test_overlong_message_is_read_in_continuous_ranges_and_empty_cards_advance(tmp_path):
    engine, calls = _engine(tmp_path, lambda _payload: [])
    # Insert an already-authenticated archive row so this tests v2's source
    # coverage independently of the public raw ingest size guard.
    text = '相遇而后长谈，雨落得很轻。' * 6000
    store = _events(engine, [('user', '前一句')])
    with store._connect() as db:
        db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', (text, 'event-0'))
    result = _run(engine, store)
    assert result['status'] == 'zero_candidates', result
    assert result['cursor_updated'] and len(calls) > 1
    covered = []
    for _, payload in calls:
        ranges = payload['source_ranges']['m1']
        for left, right in ranges:
            if not covered or left == covered[-1][1]:
                covered.append((left, right))
    assert covered[0][0] == 0 and covered[-1][1] == len(text)
    assert ''.join(text[left:right] for left, right in covered) == text
    assert '续接：' not in calls[0][0]['messages'][0]['content']
    assert all('续接：' in request['messages'][0]['content'] for request, _ in calls[1:])


def test_card_from_one_segment_binds_only_that_observed_range(tmp_path):
    engine, calls = _engine(tmp_path, lambda payload: [_card(['m1'])] if not calls[:-1] else [])
    text = '同走夜路那晚，灯照进雨里。' * 6000
    store = _events(engine, [('user', '前一句')])
    with store._connect() as db:
        db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', (text, 'event-0'))
    result = _run(engine, store)
    assert result['status'] == 'pending', result
    assert len(calls) > 1
    _, first = calls[0]
    anchor = result['candidates'][0]['source_bindings'][0]['anchors'][0]
    assert [anchor['start'], anchor['end']] == first['source_ranges']['m1'][0]
    assert anchor['end'] < len(text)


def test_adjacent_day_context_records_matching_prior_card_revision(tmp_path):
    from bucket_manager import BucketManager

    engine, calls = _engine(tmp_path, lambda payload: [_card(list(dict.fromkeys(
        row[0] for row in payload['messages'])))])
    store = _events(engine, [('assistant', '23:30 前的聊天不补读。'),
                            ('user', '那晚我提议等雨停。'), ('assistant', '好，我陪你等。')],
                    start=datetime.fromisoformat(DAY + 'T15:29:59+00:00'))
    first = _run(engine, store)
    assert first['status'] == 'pending'
    prior_id = first['candidates'][0]['id']
    row = engine.memory_authority_store.get_candidate(prior_id)
    adopted = asyncio.run(engine._memory_l1_adopt_candidate(row, BucketManager(engine.config), owner=True,
        request_id='adopt-prior-day'))
    assert adopted['status'] == 'created'
    tomorrow = '2026-11-13'
    store.ingest([{'source_event_id': 'next-day', 'role': 'user', 'text': '后来雨停了，我们一起走了。',
        'created_at': DAY + 'T16:00:00Z', 'session_id': 'main', 'conversation_id': 'main',
        'metadata': {'profile_id': 'profile', 'round_id': 1}}], source='gateway')
    second = asyncio.run(engine.run_daily_chat_memory(None, raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=tomorrow, chat_windows_only=True))
    assert second['status'] == 'pending' and len(calls) == 2, second
    request, payload = calls[1]
    assert [row[3] for row in payload['messages']] == [
        '那晚我提议等雨停。', '好，我陪你等。', '后来雨停了，我们一起走了。']
    assert [row[2] for row in payload['messages']] == [
        DAY + 'T23:30:00+08:00', DAY + 'T23:30:01+08:00', tomorrow + 'T00:00:00+08:00']
    assert payload['cross_day_sources'] == ['m1', 'm2']
    assert payload['core_sources'] == ['m3']
    assert '这是跨天的聊天，可能能帮助理解，按需取用' in request['messages'][0]['content']
    assert payload['continuation']['draft_cards'] == [{
        'body': first['candidates'][0]['l1_document']['body'], 'title': '雨夜的约定',
        'kind': 'commitment', 'sources': []}]
    row = engine.memory_authority_store.get_candidate(second['candidates'][0]['id'])
    link = row['proposal']['metadata']['l1_v2_continuation'][0]
    assert link['previous_candidate_id'] == first['candidates'][0]['id']
    assert link['previous_candidate_revision'] == engine.memory_authority_store.get_candidate(prior_id)['revision']


@pytest.mark.parametrize('use_raw_events', [True, False])
def test_cross_day_material_alone_cannot_create_a_card(tmp_path, use_raw_events):
    from gateway_state import GatewayStateStore

    engine, calls = _engine(tmp_path, lambda payload: [_card(payload['cross_day_sources'])])
    midnight = datetime.fromisoformat(DAY + 'T00:00:00+08:00')
    if use_raw_events:
        store = _events(engine, [('user', '昨天的原话。'), ('assistant', '今天的新话。')],
                        start=midnight - timedelta(seconds=1))
        source = {'raw_event_store': store}
    else:
        store = GatewayStateStore(str(tmp_path / 'gateway.sqlite3'))
        for index, (stamp, body) in enumerate((
                (midnight - timedelta(minutes=31), '不会补读的原话。'),
                (midnight - timedelta(minutes=30), '昨天的原话。'),
                (midnight, '今天的新话。')), 1):
            store.record_conversation_turn(profile_id='profile', session_id='main', round_id=index,
                                           created_at=stamp, user_text=body)
        source = {'conversation_turn_store': store}
    result = asyncio.run(engine.run_daily_chat_memory(None, **source,
        persona_engine=SimpleNamespace(profile_id='profile'), key=DAY, chat_windows_only=True))
    assert result['reason'] == 'l1_v2_card_without_core_source' and not result['cursor_updated'], result
    _, payload = calls[0]
    assert [row[3] for row in payload['messages']] == ['昨天的原话。', '今天的新话。']
    assert payload['cross_day_sources'] == ['m1'] and payload['core_sources'] == ['m2']


def test_auto_review_keeps_supported_card_beside_context_only_and_invalid_dates(tmp_path):
    def reply(payload):
        dated = _card(payload['core_sources'])
        missing_date_in_body = {**dated, 'body': '那晚我记得我们一起等雨停。'}
        invalid_date = {**dated, 'event_time': {**dated['event_time'], 'start': '2026-11-99'}}
        return [_card(payload['cross_day_sources']), missing_date_in_body, invalid_date, dated]

    engine, calls = _engine(tmp_path, reply)
    midnight = datetime.fromisoformat(DAY + 'T00:00:00+08:00')
    store = _events(engine, [('user', '昨天的原话。'), ('assistant', '今天的新话。')],
                    start=midnight - timedelta(seconds=1))

    result = _run(engine, store)

    assert result['status'] == 'pending' and result['cursor_updated'], result
    assert len(calls) == 1 and len(result['candidates']) == 1
    diagnostic = MemoryV2Diagnostics(engine.memory_authority_store).list(result['generation_task_id'])
    rejected = {row['data']['code']: row['data']['rejected_count'] for row in diagnostic
                if row['stage'] == 'card_source_rejected'}
    assert rejected == {'l1_v2_card_without_core_source': 1, 'event_date_missing_from_body': 1,
                        'invalid_event_time': 1}


def test_previous_day_context_contains_all_adopted_bodies_without_shared_sources(tmp_path):
    from bucket_manager import BucketManager

    def reply(payload):
        if len(calls) > 1:
            return []
        original = _card(['m1'])
        return [original, {**original, 'title': '经朋友认识林工', 'body': '那天经朋友介绍认识了林工。', 'event_time': None,
                           'kind': 'person_relationship', 'primary_topic': 'people_relationships'},
                {**original, 'title': '待审记忆', 'body': '待审正文不应出现在下一日。',
                 'event_time': None, 'kind': 'expressed_need'}]

    engine, calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '边界原话之外的早先一句。'),
                             ('assistant', '天气转凉，我把窗户关上了。')])
    first = _run(engine, store)
    assert first['status'] == 'pending'
    buckets = BucketManager(engine.config)
    authority = engine.memory_authority_store
    for candidate in first['candidates'][:2]:
        row = authority.get_candidate(candidate['id'])
        result = asyncio.run(engine._memory_l1_adopt_candidate(row, buckets, owner=True,
            request_id='adopt-' + candidate['id']))
        assert result['status'] == 'created'

    memory_id = first['candidates'][0]['id']
    service = engine._memory_l1_revisions(buckets)
    document = asyncio.run(service.detail(memory_id))['l1']['document']
    document['body'] = '主人修订后的当前正文。'
    document['event_time'] = None
    asyncio.run(service.save(memory_id, {'expected_revision': 1, 'request_id': 'edit-prior', 'document': document}))

    # Yesterday's originals are outside the half-hour window.
    tomorrow = '2026-11-13'
    store.ingest([{'source_event_id': 'today', 'role': 'user', 'text': '又聊起同样的日常。',
        'created_at': tomorrow + 'T01:00:00Z', 'session_id': 'main', 'conversation_id': 'main',
        'metadata': {'profile_id': 'profile'}}], source='gateway')
    second = asyncio.run(engine.run_daily_chat_memory(buckets, raw_event_store=store,
        persona_engine=SimpleNamespace(profile_id='profile'), key=tomorrow, chat_windows_only=True))
    assert second['status'] == 'zero_candidates' and len(calls) == 2, second
    request, payload = calls[-1]
    drafts = payload['continuation']['draft_cards']
    assert sorted(card['body'] for card in drafts) == sorted([document['body'], '那天经朋友介绍认识了林工。'])
    assert all(card['sources'] == [] for card in drafts)
    assert [entry[3] for entry in payload['messages']] == ['又聊起同样的日常。']
    assert 'cross_day_sources' not in payload
    assert '这是跨天的聊天，可能能帮助理解，按需取用' not in request['messages'][0]['content']
    assert '待审正文不应出现在下一日。' not in request['messages'][-1]['content']
    assert '昨天已入库卡片的正文' in request['messages'][0]['content']


def test_unknown_dispatch_cannot_retry_from_scheduler(tmp_path):
    def unknown(_payload):
        raise TimeoutError('fixture provider outcome unknown')

    engine, calls = _engine(tmp_path, unknown)
    store = _events(engine, [('user', '那次我们等到了迟到的车。')])
    first = _run(engine, store)
    assert first['status'] == 'partial' and not first['cursor_updated']
    second = _run(engine, store)
    assert second['reason'] == 'model_outcome_unknown_requires_retry_decision'
    assert len(calls) == 1
    record = MemoryModelIO(engine.memory_authority_store).list(first['generation_task_id'])['items'][0]
    assert record['native_request'] == calls[0][0]
    assert record['status'] == 'failed' and record['raw_output'] is None
    assert record['error']['failure_phase'] == 'request'


def test_v2_will_not_use_legacy_generator_without_authority(tmp_path):
    engine, calls = _engine(tmp_path, lambda _: pytest.fail('model should not be called'))
    engine.memory_authority_store = None

    class UnreadableTurns:
        def list_conversation_turns_between(self, **_kwargs):
            pytest.fail('legacy source reader should not be called')

    result = asyncio.run(engine.run_daily_chat_memory(None, conversation_turn_store=UnreadableTurns(),
                                                    key=DAY, chat_windows_only=True))
    assert result['status'] == 'failed' and result['reason'] == 'memory_authority_unavailable'
    assert not result['cursor_updated'] and not calls


@pytest.mark.parametrize('late_return', [False, True])
def test_daily_force_rescans_changed_source_while_original_dispatch_is_unknown(tmp_path, late_return):
    def reply(payload):
        return [_card([payload['messages'][-1][0]])]

    engine, original_calls = _engine(tmp_path, reply)
    recovery, recovery_calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    owner = SimpleNamespace(profile_id='profile')

    async def scenario():
        sent = asyncio.Event()
        release = asyncio.Event()
        original_create = engine.client.chat.completions.create

        async def create(**request):
            if request['response_format']['type'] == 'json_object':
                return await original_create(**request)
            if not late_return:
                original_calls.append(request)
                raise TimeoutError('fixture provider outcome unknown')
            sent.set()
            await release.wait()
            return await original_create(**request)

        engine.client.chat.completions.create = create

        async def run(worker, *, force=False):
            return await worker.run_daily_chat_memory(None, raw_event_store=store, persona_engine=owner,
                key=DAY, force=force, chat_windows_only=True)

        original = asyncio.create_task(run(engine))
        if late_return:
            await asyncio.wait_for(sent.wait(), timeout=5)
        else:
            assert (await original)['status'] == 'partial'
        try:
            blocked = await run(recovery)
            assert blocked['reason'] == 'model_outcome_unknown_requires_retry_decision'
            original_key = 'l1-v2:' + blocked['generation_task_id']
            progress = ChatSegmentProgress(recovery.memory_authority_store)
            dispatched = progress.get(original_key)
            assert dispatched['progress']['phase'] == 'dispatched'

            with store._connect() as db:
                db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', ('更正，我们没有约好。', 'event-0'))
            changed = await run(recovery)
            assert changed['reason'] == 'l1_v2_daily_source_changed_requires_rescan'
            assert recovery_calls == [] and recovery._daily_chat_memory_last_raw_event_id('profile') == 0

            resumed = await run(recovery, force=True)
            assert resumed['status'] == 'pending', resumed
            assert resumed['supersedes_generation_task_id'] == blocked['generation_task_id']
            assert resumed['generation_task_id'] != blocked['generation_task_id']
            assert [row[3] for row in recovery_calls[0][1]['messages']] == ['更正，我们没有约好。']
            assert resumed['processed_until_seq'] == 1 and len(recovery_calls) == 1
        finally:
            release.set()
            original_result = await original

        if late_return:
            assert original_result['reason'] == 'l1_v2_daily_window_superseded'
        assert len(original_calls) == 1 and progress.get(original_key) == dispatched
        assert MemoryCandidatePreparation(recovery.memory_authority_store).get(original_key) is None
        assert len(recovery.memory_authority_store.list_candidates(status='pending')) == 1
        assert recovery._load_daily_chat_memory_run_audit()[-1]['status'] == 'pending'
        assert recovery._daily_chat_memory_last_raw_event_id('profile') == 1

    asyncio.run(scenario())


def _append_event(store, text):
    result = store.ingest([{'source_event_id': 'late-event', 'role': 'assistant', 'text': text,
        'created_at': DAY + 'T01:00:00Z', 'session_id': 'main', 'conversation_id': 'main',
        'metadata': {'profile_id': 'profile', 'round_id': 1}}], source='gateway')
    assert result['inserted'] == 1


@pytest.mark.parametrize('first_empty', [False, True])
def test_daily_resume_uses_prepared_watermark_then_reads_new_messages(tmp_path, monkeypatch, first_empty):
    def reply(payload):
        if len(payload['messages']) == 1 and first_empty:
            return []
        return [_card([payload['messages'][-1][0]])]

    engine, calls = _engine(tmp_path, reply)
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    watermarks = []
    update = engine._update_daily_chat_memory_raw_cursor

    def fail_first_write(profile, watermark, day):
        watermarks.append(watermark)
        if len(watermarks) == 1:
            raise OSError('cursor write interrupted after preparation')
        return update(profile, watermark, day)

    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', fail_first_write)
    first = _run(engine, store)
    assert first['reason'] == 'l1_v2_daily_cursor_not_persisted'
    assert first['processed_until_seq'] == 0
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 0
    before = engine.memory_authority_store.list_candidates(status='pending')
    assert len(before) == (0 if first_empty else 1)
    _append_event(store, '雨停后，我又说可以一起走。')

    # DAY is older than yesterday, so selection of its genuine V2 partial
    # checkpoint must come from the scheduler rather than the default date.
    engine.enabled = engine.auto_enabled = True
    engine.diary_memory_extract_enabled = False
    scheduled_at = datetime.fromisoformat(DAY).replace(tzinfo=engine.tz) + timedelta(days=2, hours=10)
    monkeypatch.setattr(engine, '_local_now', lambda now=None: now or scheduled_at)
    legacy_date = (scheduled_at.date() - timedelta(days=3)).isoformat()
    legacy = {'date': legacy_date, 'profile_id': 'profile', 'status': 'partial',
              'source_start_seq': 0, 'error_category': 'source_batch_exhausted'}
    engine._store_daily_chat_memory_run_audit(legacy)
    resumed, = asyncio.run(engine.run_due(None, persona_engine=SimpleNamespace(profile_id='profile'),
                                        raw_event_store=store))
    assert resumed['date'] == DAY
    assert resumed['status'] == 'pending', resumed
    assert [row['date'] for row in resumed['unresolved_dates']] == [legacy_date]
    assert [row for row in engine._load_daily_chat_memory_run_audit() if row['date'] == legacy_date] == [legacy]
    assert watermarks == [1, 1, 2] and len(calls) == 2
    second = calls[1][1]
    assert [row[3] for row in second['messages']] == ['原来我们说好等雨停。', '雨停后，我又说可以一起走。']
    assert second['core_sources'] == [second['messages'][-1][0]]
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 2
    rows = engine.memory_authority_store.list_candidates(status='pending')
    assert len(rows) == (1 if first_empty else 2)
    if not first_empty:
        assert any(row['candidate_id'] == before[0]['candidate_id'] and row['revision'] == before[0]['revision'] for row in rows)
    assert _run(engine, store)['status'] == 'skipped' and len(calls) == 2


@pytest.mark.parametrize('withdraw', [False, True])
def test_daily_resume_refuses_changed_source_until_explicit_rescan(tmp_path, monkeypatch, withdraw):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['messages'][-1][0]])])
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    update = engine._update_daily_chat_memory_raw_cursor
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', lambda *_: False)
    first = _run(engine, store)
    assert first['reason'] == 'l1_v2_daily_cursor_not_persisted'
    old = engine.memory_authority_store.list_candidates(status='pending')[0]
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', update)
    with store._connect() as db:
        if withdraw:
            db.execute('DELETE FROM raw_events WHERE source_event_id=?', ('event-0',))
        else:
            db.execute('UPDATE raw_events SET text=? WHERE source_event_id=?', ('更正，我们没有约好。', 'event-0'))
    _append_event(store, '这是之后新增加的对话。')
    blocked = _run(engine, store)
    assert blocked['reason'] == 'l1_v2_daily_source_changed_requires_rescan'
    assert len(calls) == 1 and engine._daily_chat_memory_last_raw_event_id('profile') == 0

    resumed = _run(engine, store, force=True)
    assert resumed['status'] == 'pending', resumed
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 2
    sent = [[row[3] for row in payload['messages']] for _, payload in calls[1:]]
    if not withdraw:
        assert sent[0] == ['更正，我们没有约好。']
    assert sent[-1][-1] == '这是之后新增加的对话。'
    assert all('原来我们说好等雨停。' not in texts for texts in sent)
    assert engine.memory_authority_store.get_candidate(old['candidate_id'])['revision'] == old['revision']


def test_daily_resume_retries_only_incomplete_adoption_before_new_window(tmp_path, monkeypatch):
    def reply(payload):
        ref = payload['messages'][-1][0]
        card = {**_card([ref]), 'review': {'confidence': .9, 'sensitive': False,
                                          'transient': False, 'generic': False}}
        return [card, {**card, 'title': '雨停后的散步'}] if len(payload['messages']) == 1 else [card]

    engine, calls = _engine(tmp_path, reply, mode='auto')
    store = _events(engine, [('user', '原来我们说好等雨停。')])
    adoption_attempts = []
    failed = set()

    async def adopt(row, _buckets, *, request_id):
        candidate = row['candidate_id']
        adoption_attempts.append(candidate)
        if len(adoption_attempts) == 2:
            failed.add(candidate)
            raise OSError('synthetic write failure for one candidate')
        # Simulate the original durable state transition while keeping this
        # test focused on the generation/receipt/cursor boundary.
        engine.memory_authority_store.decide_candidate(candidate, action='commit',
            expected_revision=row['revision'], request_id=request_id, actor='fixture')
        return {'status': 'created'}

    monkeypatch.setattr(engine, '_memory_l1_adopt_candidate', adopt)
    first = _run(engine, store)
    assert first['reason'] == 'candidate_persistence_failed', first
    assert first['processed_until_seq'] == 0 and len(calls) == 1
    _append_event(store, '雨停后，我又说可以一起走。')
    resumed = _run(engine, store)
    assert resumed['status'] == 'created', resumed
    assert len(calls) == 2 and len(adoption_attempts) == 4
    assert [candidate for candidate in adoption_attempts if candidate in failed] == [*failed, *failed]
    assert adoption_attempts.count(adoption_attempts[0]) == 1
    assert engine._daily_chat_memory_last_raw_event_id('profile') == 2


def test_explicit_retry_keeps_unfinished_nonzero_window(tmp_path, monkeypatch):
    engine, calls = _engine(tmp_path, lambda payload: [_card([payload['messages'][-1][0]])])
    store = _events(engine, [('user', '原来的对话。')])
    assert _run(engine, store)['status'] == 'pending'
    _append_event(store, '新追加的对话。')
    update = engine._update_daily_chat_memory_raw_cursor
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', lambda *_: False)
    assert _run(engine, store)['reason'] == 'l1_v2_daily_cursor_not_persisted'
    monkeypatch.setattr(engine, '_update_daily_chat_memory_raw_cursor', update)

    resumed = _run(engine, store, force=True)
    assert resumed['source_start_seq'] == 1 and resumed['processed_until_seq'] == 2
    assert len(calls) == 2 and len(engine.memory_authority_store.list_candidates(status='pending')) == 2


@pytest.mark.parametrize('has_title', [True, False])
def test_chat_draft_can_prepare_without_nearby_chat_sources(tmp_path, has_title):
    from memory_chat_preparation import ChatMemoryPreparation

    body = '二〇二六年十一月十二日，我在雨夜等车。\n\n我记得我那时有点舍不得走。'
    event_time = {'start': DAY, 'precision': 'day', 'sources': ['m1']} if has_title else None
    metadata = {'kind': 'life_experience', 'stability': 'episode', 'keywords': ['雨夜等车'],
                'people': [], 'sources': [], 'primary_topic': 'daily_life', 'event_time': event_time}
    if not has_title:
        metadata['title'] = '等车的雨夜'
    engine, calls = _engine(tmp_path, lambda _payload: [metadata])
    narrative = {'title': '我的原题', 'body': body} if has_title else {'body': body}
    envelope = {'operation_id': 'op-' + '1' * 32, 'arguments': {'narrative': narrative},
        'context': {'conversation_id': 'canonical-main', 'request_id': 'r1', 'turn_id': 'e1', 'sources': []}}
    queued = asyncio.run(ChatMemoryTool(engine, None).submit(envelope))
    result = asyncio.run(prepare_chat_tasks(engine, None))
    assert result['results'][0]['memory_status'] == 'pending', result
    [(request, payload)] = calls
    assert payload['messages'][0][3] == body
    assert request['messages'][0]['content'] == CHAT_L1_PROMPT
    schema = request['response_format']['json_schema']['schema']['properties']['cards']
    assert schema['minItems'] == schema['maxItems'] == 1
    assert 'body' not in schema['items']['properties']
    assert ('title' in schema['items']['required']) is not has_title
    row = engine.memory_authority_store.get_candidate(queued['task_id'])
    proposal = row['proposal']
    document = proposal['metadata']['memory_l1']
    assert proposal['proposed_body'] == body
    assert document['title'] == (narrative.get('title') or metadata['title'])
    assert document['generation']['prompt_source_id'] == CHAT_PROMPT_ID
    task = ChatMemoryPreparation(engine.memory_authority_store).get(queued['task_id'])
    assert document['generation']['submitted_at'] == task['created_at']
    assert datetime.fromisoformat(payload['agent_draft']['submitted_at']) == datetime.fromisoformat(
        document['generation']['submitted_at'])
    assert document['evidence_bindings'][0]['source_namespace'] == 'snapshot.ombre.agent_memory_draft'
    assert (document.get('event_time') or {}).get('start') == (DAY if has_title else None)
    record = MemoryModelIO(engine.memory_authority_store).list(queued['task_id'])['items'][0]
    assert record['stage'] == 'chat_l1' and record['task_id'] == queued['task_id']
    assert record['generation']['task_id'] == queued['task_id'] + ':1'
    assert record['task_attempt'] == task['attempts'] == 1
    assert record['draft_revision'] == 1 and record['part'] == 1
    assert record['native_request'] == request and record['validation'] == {'status': 'passed'}
    assert json.loads(record['raw_output']) == {'cards': [metadata]}


def test_chat_v2_model_io_groups_fallback_calls_with_the_original_preparation(tmp_path):
    from memory_chat_preparation import ChatMemoryPreparation

    metadata = {'kind': 'life_experience', 'stability': 'episode', 'keywords': ['雨夜等车'],
                'people': [], 'sources': [], 'primary_topic': 'daily_life', 'event_time': None}
    engine, _ = _engine(tmp_path, lambda _payload: [metadata])
    queued = asyncio.run(ChatMemoryTool(engine, None).submit({
        'operation_id': 'op-' + '1' * 32,
        'arguments': {'narrative': {'title': '雨夜', 'body': '我在雨夜等车。'}},
        'context': {'conversation_id': 'canonical-main', 'request_id': 'r1', 'turn_id': 'e1', 'sources': []},
    }))
    task_id = queued['task_id']
    recorder = MemoryModelIO(engine.memory_authority_store)
    provider = engine.client.chat.completions.create
    requests = []
    invalid_output = '  {"cards": "invalid"}\n'

    async def respond(**request):
        requests.append(request)
        record = recorder.list(task_id)['items'][0]
        assert record['status'] == 'running' and record['native_request'] == request
        if len(requests) == 1:
            raise _bad_request('response_format is not supported', param='response_format')
        if len(requests) == 2:
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=invalid_output), finish_reason='stop')])
        return await provider(**request)

    engine.client.chat.completions.create = respond
    first = asyncio.run(prepare_chat_tasks(engine, None))
    assert first['results'][0]['status'] == 'failed'
    tasks = ChatMemoryPreparation(engine.memory_authority_store)
    failed_task = tasks.get(task_id)
    tasks.update(task_id, expected_revision=failed_task['revision'],
                 transform=lambda row: {**row, 'phase': 'queued', 'lease_until': 0,
                                        'claim_token': None, 'issue': None, 'manual_generation_retry': True})
    second = asyncio.run(prepare_chat_tasks(engine, None))
    assert second['results'][0]['memory_status'] == 'pending'
    last, invalid, schema = recorder.list(task_id)['items']
    assert [row['native_request'] for row in (schema, invalid, last)] == requests
    assert (schema['task_attempt'], schema['attempt']) == (1, 1)
    assert (invalid['task_attempt'], invalid['attempt']) == (1, 2)
    assert (last['task_attempt'], last['attempt']) == (2, 1)
    assert schema['attempt_id'] == invalid['attempt_id'] != last['attempt_id']
    assert schema['error']['http_status'] == 400 and schema['raw_output'] is None
    assert invalid['error']['failure_phase'] == 'response_processing'
    assert invalid['raw_output'] == invalid_output and invalid['finish_reason'] == 'stop'
    native_fields = requests[0]['response_format']['json_schema']['schema']['properties']['cards']['items']['properties']
    fallback_fields = json.loads(requests[1]['messages'][-1]['content'])['output_schema']['properties']['cards']['items']['properties']
    assert '±HH:MM' in native_fields['event_time']['description']
    assert native_fields['event_time']['description'] == fallback_fields['event_time']['description']


@pytest.mark.parametrize('bounds,refs,stated,expected', [
    pytest.param(('2026-11-12T01:19', '2026-11-12T01:23'), ['m2', 'm3'], '',
                 ('2026-11-12T01:19+08:00', '2026-11-12T01:23+08:00'), id='matched-range'),
    pytest.param(('2026-11-12T01:19+08:00', '2026-11-12T01:23'), ['m3'], '',
                 ('2026-11-12T01:19+08:00', '2026-11-12T01:23+08:00'), id='only-end-missing'),
    pytest.param(('2026-11-12T01:19',), ['m1'], '2026-11-12T01:19-04:00',
                 ('2026-11-12T01:19-04:00',), id='explicit-time-in-draft'),
    pytest.param(('2026-11-12T01:19-04:00',), ['m1'], '2026-11-12T01:19-04:00',
                 ('2026-11-12T01:19-04:00',), id='already-zoned'),
    pytest.param(('2026-11-12T01:19',), ['m1', 'm2'], '2026-11-12T01:19-04:00', None, id='conflicting-offsets'),
    pytest.param(('2026-11-12T01:19',), ['m1'], '', None, id='submission-and-uncited-message-do-not-count'),
    pytest.param(('2026-11-12T01:19', '2026-11-12T01:23'), ['m2'], '', None, id='unproved-end'),
    pytest.param(('2026-02-30T01:19',), ['m2'], '', 'invalid_event_time', id='invalid-calendar-date'),
])
def test_chat_event_time_uses_only_evidenced_offsets(tmp_path, monkeypatch, bounds, refs, stated, expected):
    from memory_chat_preparation import ChatMemoryPreparation

    body = '我记得那次我们一起谈起雨声。' + stated
    event = {**dict(zip(('start', 'end'), bounds)), 'precision': 'minute', 'sources': refs}
    metadata = {'kind': 'life_experience', 'stability': 'episode', 'keywords': ['雨声'],
                'people': [], 'sources': [], 'primary_topic': 'daily_life', 'event_time': event}
    engine, calls = _engine(tmp_path, lambda _payload: [metadata])

    async def locate(_binding, *, verify_only=False):
        return {'eligible_source': True}

    monkeypatch.setattr(engine, '_memory_l1_sources',
                        lambda: SimpleNamespace(retain_copy=lambda _snapshot: None, locate=locate))
    tasks = ChatMemoryPreparation(engine.memory_authority_store)
    submission = tasks.enqueue({
        'operation_id': 'op-' + '9' * 32, 'arguments': {'narrative': {'title': '谈起雨声', 'body': body}},
        'context': {'conversation_id': 'canonical-main', 'request_id': 'r1', 'turn_id': 'e2', 'sources': [
            {'event_id': 'e1', 'version_id': 'v1', 'role': 'user', 'text': '我们聊到了雨声。',
             'created_at': '2026-11-11T17:19:10.142000+00:00'},
            {'event_id': 'e2', 'version_id': 'v2', 'role': 'user', 'text': '我也记得。',
             'created_at': '2026-11-11T17:23:02.058000+00:00'},
        ]}}, generation_protocol='whole_window_v2')
    task_id = submission['task_id']
    submitted = '2026-11-11T17:19:40+00:00'
    task = {**tasks.get(task_id), 'created_at': submitted}
    engine.memory_authority_store.set_meta(tasks.prefix + task_id, task)
    result = asyncio.run(prepare_chat_tasks(engine, None))['results'][0]
    record, = MemoryModelIO(engine.memory_authority_store).list(task_id)['items']
    assert len(calls) == 1 and json.loads(record['raw_output']) == {'cards': [metadata]}
    assert tasks.get(task_id)['envelope'] == task['envelope']
    candidate = engine.memory_authority_store.get_candidate(task_id)
    if expected == 'invalid_event_time':
        assert result['issue']['code'] == expected and candidate is None
        return
    assert candidate['status'] == 'pending' and candidate['proposal']['proposed_body'] == body
    assert engine.memory_authority_store.get_memory(task_id) is None
    doc = candidate['proposal']['metadata']['memory_l1']
    assert doc['generation']['submitted_at'] == submitted
    if expected is None:
        assert doc.get('event_time') is None
    else:
        assert doc['event_time'] == {**event, **dict(zip(('start', 'end'), expected)),
                                     'sources': ['source:' + ref[1:] for ref in refs]}


def test_chat_v2_accepts_a_staged_window_over_the_legacy_message_cap(tmp_path, monkeypatch):
    from memory_chat_preparation import ChatMemoryPreparation
    from memory_l1_service import L1OwnerError

    def respond(payload):
        return [{'title': '这次聊天', 'kind': 'life_experience', 'stability': 'episode',
                 'keywords': ['分批聊天'], 'people': [], 'sources': [payload['messages'][-1][0]],
                 'primary_topic': 'daily_life', 'event_time': None}]

    engine, calls = _engine(tmp_path, respond)

    async def locate(_binding, *, verify_only=False):
        return {'eligible_source': True}

    monkeypatch.setattr(engine, '_memory_l1_sources',
        lambda: SimpleNamespace(retain_copy=lambda _snapshot: None, locate=locate))
    tasks = ChatMemoryPreparation(engine.memory_authority_store)
    sources = [{'event_id': f'e{index}', 'version_id': f'v{index}', 'role': 'user',
                'text': '同一轮里的原话', 'created_at': '2026-11-12T01:00:00+08:00'}
               for index in range(62)]
    envelope = {'operation_id': 'op-' + '2' * 32, 'arguments': {'narrative': {'body': '我的回忆'}},
                'context': {'conversation_id': 'canonical-main', 'request_id': 'r2',
                            'turn_id': sources[-1]['event_id'], 'sources': sources}}
    with pytest.raises(L1OwnerError, match='source_missing'):
        tasks.enqueue(envelope)
    tasks.enqueue(envelope, generation_protocol='whole_window_v2')
    assert len(tasks.list(private=True)[0]['envelope']['context']['sources']) == len(sources)
    result = asyncio.run(prepare_chat_tasks(engine, None))
    assert result['results'][0]['memory_status'] == 'pending', result
    [(_, payload)] = calls
    assert len(payload['messages']) == len(sources) + 1
    assert payload['messages'][-1][3] == sources[-1]['text']
