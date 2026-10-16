import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from memory_l1_ingestion import model_template
from memory_l1_service import L1OwnerError
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config


def test_auto_model_selects_source_labels_and_program_fills_event_turn_ids(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    calls = []
    engine._daily_chat_memory_model_client = lambda **kw: (object(), 'synthetic-model', False)
    async def create(_client, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        calls.append(payload)
        document = model_template(catalog=payload['l1_contract']['catalog'])
        document['narrative'] = {'title': '当次请求', 'body': '我记下这次明确的请求。'}
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates': [
            {'kind': 'shared_experience', 'confidence': 0.9, 'reason': '合成测试',
             'source_refs': ['source:1'], 'document': document}]})), finish_reason='stop')])
    engine._daily_chat_memory_create_completion = create
    window = [{'id': 11, 'session_id': 's', 'user_text': '我希望你记住这一次。',
               'assistant_text': '', 'created_at': '2026-10-16T10:00:00Z'}]
    result, stats = asyncio.run(engine._extract_window_candidates('2026-10-16', window))
    assert result[0]['source_turn_ids'] == ['11']
    assert result[0]['source_event_ids'] == []
    assert result[0]['_l1_sources'][0]['namespace'] == 'conversation_turn'
    assert calls[0]['l1_contract']['source_texts'][0]['binding_id'] == 'source:1'
    assert not stats['parse_failure']


def test_auto_length_finish_is_failure_not_empty_day(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    engine._daily_chat_memory_model_client = lambda **kw: (object(), 'synthetic-model', False)
    async def create(*args, **kwargs):
        return NS(choices=[NS(message=NS(content='{"candidates":[]}'), finish_reason='length')])
    engine._daily_chat_memory_create_completion = create
    with pytest.raises(L1OwnerError, match='first_pass_output_incomplete'):
        asyncio.run(engine._extract_window_candidates('2026-10-16', [
            {'id': 11, 'session_id': 's', 'user_text': '合成材料', 'assistant_text': '',
             'created_at': '2026-10-16T10:00:00Z'}]))
