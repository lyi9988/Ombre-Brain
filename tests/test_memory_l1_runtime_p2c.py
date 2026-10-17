import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

from memory_l1_ingestion import model_template
from memory_l1_schema import statement_template
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config
from test_memory_diary_l1_p2b import setup_diary, DAY


def runtime(tmp_path, text, *, fail_window=None, limit=32, compact=False):
    env = setup_diary(tmp_path, text)
    config = reflection_config(tmp_path / 'engine', mode='review')
    config['reflection'].update(diary_l1_input_window_chars=256, diary_l1_max_windows=limit)
    engine = ReflectionEngine(config)
    engine.memory_authority_store = env['authority']
    engine.memory_ingestion_policy = env['policy']
    engine._memory_l1_sources = lambda: env['sources']
    engine._memory_l1_revisions = lambda bucket_mgr=None: env['revisions']
    calls = []
    async def create(**kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        calls.append(payload)
        if payload['window_index'] == fail_window: raise RuntimeError('synthetic_model_failure')
        draft = model_template(catalog=env['catalog'])
        draft['narrative'] = {'title': '整篇第一遍', 'body': '我记下这个完整片段：' + payload['source_text']}
        statement = statement_template()
        statement.pop('statement_revision_id')
        statement.update(content='这一次的合成经历。', evidence_refs=['diary-source'])
        draft['statements'] = [statement]
        draft['scenes'] = [{'scene_id': 'new:scene', 'title': '一次连贯经历',
            'source_refs': ['diary-source'], 'statement_refs': ['new:statement'], 'attributes': []}]
        assert 'semantic_contract' in payload
        assert not {'model_document_schema', 'catalog', 'document_template'}.intersection(payload)
        if compact:
            draft = {'format': 'semantic-v1', 'narrative': draft['narrative'], 'entities': [],
                'statements': [{'content': statement['content'], 'subject_ref': None,
                    'classification': {'kind': None, 'stability': None, 'modality': None, 'basis': None},
                    'time': None, 'conditions': None, 'evidence_refs': ['diary-source']}],
                'scenes': [{'title': '一次连贯经历',
                    'source_refs': ['diary-source'], 'statement_refs': ['s1']}]}
        result = {'document': draft, 'scene_bodies': {'c1' if compact else 'new:scene':
            '场景独立完整正文：' + payload['source_text']}, 'confidence': 0.9}
        return NS(choices=[NS(message=NS(content=json.dumps(result, ensure_ascii=False)), finish_reason='stop')])
    engine.model = 'synthetic-model'
    engine.client = NS(chat=NS(completions=NS(create=create)))
    return env, engine, calls


def run(task): return asyncio.run(task)


def test_memory_l2_generation_options_have_bounded_defaults():
    from memory_l1_runtime import GENERATION_OPTIONS, task_options
    from memory_l1_service import L1OwnerError

    assert GENERATION_OPTIONS["memory_l2_jobs_per_run"] == (1, 0, 200)
    assert GENERATION_OPTIONS["memory_l2_prepare_interval_seconds"] == (5, 1, 3600)
    assert task_options({"reflection": {}})["memory_l2_jobs_per_run"] == 1
    assert task_options({"reflection": {}})["memory_l2_prepare_interval_seconds"] == 5

    for key, allowed in (
        ("memory_l2_jobs_per_run", (0, 200)),
        ("memory_l2_prepare_interval_seconds", (1, 3600)),
    ):
        for value in allowed:
            assert task_options({"reflection": {key: value}})[key] == value
        for value in (allowed[0] - 1, allowed[1] + 1, True, 1.5):
            with pytest.raises(L1OwnerError, match="invalid_generation_configuration"):
                task_options({"reflection": {key: value}})


@pytest.mark.parametrize('compact', [False, True])
def test_diary_actual_task_covers_all_segments_and_prepares_full_root_and_scene_bodies(tmp_path, compact):
    text = ('合成日记正文🙂，这是完整资料。' * 90)
    env, engine, calls = runtime(tmp_path, text, compact=compact)
    result = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert result['status'] == 'pending'
    assert ''.join(call['source_text'] for call in calls) == text
    assert result['coverage']['state'] == 'complete'
    row = env['authority'].get_candidate(result['candidate_id'])
    semantic = row['proposal']['metadata']['memory_l1']
    assert len(semantic['scenes']) == len(calls) > 1
    assert len(row['proposal']['metadata']['diary_l1_bundle']['scene_bodies']) == len(calls)
    for scene, call in zip(semantic['scenes'], calls):
        assert scene['anchors'][0]['start'] == call['input_range']['start']
        assert scene['anchors'][0]['end'] == call['input_range']['end']
    assert row['proposal']['proposed_body'].replace('我记下这个完整片段：', '').replace('\n\n', '') == text
    assert semantic['generation']['origin'] == 'first_pass_model'
    assert env['authority'].list_memories() == []


def test_diary_window_failure_keeps_completed_input_and_queues_partial_review(tmp_path):
    env, engine, calls = runtime(tmp_path, '合成材料。' * 160, fail_window=1)
    result = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert result['status'] == 'pending'
    assert result['coverage']['state'] == 'partial'
    assert result['coverage']['input_ranges'] == [{'start': 0, 'end': 256}]
    assert len(result['failures']) == 1
    row = env['authority'].get_candidate(result['candidate_id'])
    assert len(row['proposal']['metadata']['memory_l1']['scenes']) == 1
    assert len(row['proposal']['metadata']['diary_l1_bundle']['scene_bodies']) == 1


def test_configured_window_budget_is_visible_partial_and_replay_does_not_call_model_again(tmp_path):
    env, engine, calls = runtime(tmp_path, '合成材料。' * 160, limit=1)
    result = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert result['coverage']['state'] == 'partial'
    replay = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert replay['candidate_id'] == result['candidate_id'] and replay['replayed']
    assert len(calls) == 1


def test_first_window_failure_is_failed_task_not_fake_empty_memory(tmp_path):
    env, engine, calls = runtime(tmp_path, '完整合成日记。', fail_window=0)
    result = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert result['status'] == 'failed'
    assert env['authority'].list_candidates(status='all') == []
    status = env['authority'].list_projection_status('diary_task_' + DAY)
    assert status[0]['status'] == 'degraded'
    assert status[0]['details']['state'] == 'failed'


def test_off_background_prompt_prevents_call_and_owner_wrapper_reaches_actual_task(tmp_path):
    env, engine, calls = runtime(tmp_path, '一篇合成日记。')
    store = engine.prompt_plan_mirror
    def bind(mode, revision, body=''):
        block = {'block_id': 'diary-block', 'source_id': 'ombre.diary_memory_prompt', 'scope': 'memory.reflection',
            'stage': 'ombre_post_injection', 'mode': mode, 'role': 'system', 'lane': 'instruction',
            'anchor': 'instructions.end', 'order': 0, 'priority': 1, 'enabled': True,
            'owner_body': body, 'wrapper_text': 'A\n{{content}}\nC'}
        digest = str(revision) * 64
        store.put_plan('p2c', revision, plan_sha256=digest, gateway_slice={'blocks': [block]})
        store.put_binding('memory.reflection', preset_id='p2c', preset_revision=revision,
            plan_sha256=digest, aiz_binding_revision=revision)
    bind('off', 1)
    result = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert result['status'] == 'disabled' and calls == []
    bind('owner_override', 2, 'B主人日记规范')
    actual, context = engine._memory_l1_prompt('ombre.diary_memory_prompt')
    assert actual == 'A\nB主人日记规范\nC'
    assert context['prompt_revision'].startswith('prompt_sha256:')
    result = run(engine._memory_l1_diary_task(DAY, env['buckets']))
    assert result['status'] == 'pending' and len(calls) == 1
