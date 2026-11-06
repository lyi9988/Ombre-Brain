"""Event-first uses independent owner-editable Composer prompt sources."""
import hashlib
import json

import pytest

from prompt_plan_mirror import PromptPlanMirrorStore
from prompt_source_registry import FIXED_PROMPT_SOURCES, resolve_fixed_prompt, source_detail


SOURCES = {
    'ombre.daily_chat_event_selection_prompt': 'memory.daily_chat_event_selection',
    'ombre.daily_chat_event_assembly_prompt': 'memory.daily_chat_event_assembly',
}


def bind(store, *, modes, revision):
    blocks = [{'block_id': f'event:{index}', 'source_id': source_id, 'scope': scope,
        'stage': 'ombre_post_injection', 'mode': modes.get(source_id, 'live_source'),
        'owner_body': f'{source_id} 主人改稿给{{ai_name}}', 'wrapper_text': '前缀{{content}}后缀',
        'role': 'system', 'lane': 'instruction', 'anchor': 'instructions.end',
        'order': index, 'enabled': True} for index, (source_id, scope) in enumerate(SOURCES.items())]
    digest = hashlib.sha256(json.dumps(blocks, sort_keys=True).encode()).hexdigest()
    store.put_plan('events:preset', revision, plan_sha256=digest, gateway_slice={'blocks': blocks})
    for scope in SOURCES.values():
        store.put_binding(scope, preset_id='events:preset', preset_revision=revision,
            plan_sha256=digest, aiz_binding_revision=revision)


@pytest.mark.parametrize('source_id,scope', SOURCES.items())
def test_event_factories_are_independently_scoped_and_use_configured_identity(tmp_path, source_id, scope):
    from memory_l1_prompts import DAILY_L1_PROMPT
    store = PromptPlanMirrorStore(tmp_path / 'mirror.sqlite3')
    detail = source_detail(source_id, {'identity': {'ai_name': '配置助手'}})
    assert FIXED_PROMPT_SOURCES[source_id].scope == scope
    assert detail['body_kind'] == 'fixed_prompt' and 'memory_chat_event_prompts' in detail['authority']
    resolved, metadata = resolve_fixed_prompt(store, source_id, config={'identity': {'ai_name': '配置助手'}})
    assert '配置助手' in resolved and '{ai_name}' not in resolved
    assert metadata['status'] == 'legacy_default'
    assert FIXED_PROMPT_SOURCES['ombre.daily_chat_memory_prompt'].factory_body() == DAILY_L1_PROMPT


def test_owner_overrides_and_off_preserve_the_other_event_stage(tmp_path):
    store = PromptPlanMirrorStore(tmp_path / 'mirror.sqlite3')
    selection, assembly = SOURCES
    bind(store, modes={selection: 'owner_override', assembly: 'off'}, revision=1)
    config = {'identity': {'ai_name': '配置助手'}}
    selected, _ = resolve_fixed_prompt(store, selection, config=config)
    assembled, _ = resolve_fixed_prompt(store, assembly, config=config)
    assert selected == f'前缀{selection} 主人改稿给配置助手后缀' and assembled == ''
    bind(store, modes={selection: 'off', assembly: 'owner_override'}, revision=2)
    selected, _ = resolve_fixed_prompt(store, selection, config=config)
    assembled, _ = resolve_fixed_prompt(store, assembly, config=config)
    assert selected == '' and assembled == f'前缀{assembly} 主人改稿给配置助手后缀'


def test_prompt_contract_preserves_confirmed_result_and_shared_event_statements():
    selection = FIXED_PROMPT_SOURCES['ombre.daily_chat_event_selection_prompt'].factory_body()
    assembly = FIXED_PROMPT_SOURCES['ombre.daily_chat_event_assembly_prompt'].factory_body()
    assert '先前尚未确认、需要重做或等待' in selection
    assert '证据无需双方各占一句' in selection
    assert 'quote 逐字摘录' in selection
    assert '简短事实一条写清' in assembly and '共引相应来源' in assembly
    assert 'document.retrieval.keywords' in assembly and '根对象只写 candidates' in assembly
