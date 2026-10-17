import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from memory_l1_ingestion import model_template
from memory_l1_service import L1OwnerError
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config


@pytest.mark.parametrize('compact', [False, True])
def test_auto_model_selects_source_labels_and_program_fills_event_turn_ids(tmp_path, compact):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    calls = []
    engine._daily_chat_memory_model_client = lambda **kw: (object(), 'synthetic-model', False)
    async def create(_client, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        calls.append(payload)
        document = model_template()
        document['narrative'] = {'title': '当次请求', 'body': '我记下这次明确的请求。'}
        if compact:
            document = {'format': 'semantic-v1', 'narrative': document['narrative'], 'entities': [],
                'statements': [{'content': '这一次明确希望记住。', 'subject_ref': None,
                    'classification': {'kind': None, 'stability': None, 'modality': None, 'basis': None},
                    'time': None, 'conditions': None, 'evidence_refs': ['source:1']}]}
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
    assert 'semantic_contract' in calls[0]['l1_contract']
    assert not {'document_schema', 'document_template', 'catalog'}.intersection(calls[0]['l1_contract'])
    assert result[0]['document']['narrative']['body'] == '我记下这次明确的请求。'
    assert result[0]['_l1_catalog_context'][0] == engine._memory_l1_revisions().catalog_provider()[0]
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


def test_auto_legacy_body_remains_identifiable_without_fabricated_l1(tmp_path):
    engine = ReflectionEngine(reflection_config(tmp_path, mode='review'))
    async def create(*args, **kwargs):
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates': [{
            'kind': 'key_event', 'content': '主人旧提示词返回的完整正文。', 'confidence': 0.9}]})), finish_reason='stop')])
    engine._daily_chat_memory_create_completion = create
    result, _ = asyncio.run(engine._extract_window_candidates('2026-10-16', [
        {'id': 11, 'session_id': 's', 'user_text': '合成材料', 'assistant_text': '',
         'created_at': '2026-10-16T10:00:00Z'}]))
    assert result[0]['l1_state'] == 'legacy_body_only'
    assert result[0]['content'] == '主人旧提示词返回的完整正文。'
    assert 'document' not in result[0] and '_l1_generation' not in result[0]
