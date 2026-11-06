"""Repeated planning fields must not repeat pure query preparation work."""
from collections import Counter
import asyncio
import json

import pytest

import memory_relevance
import recall_policy
from recall_policy import RecallPolicy


def test_repeated_planning_fields_normalize_each_candidate_once(monkeypatch):
    policy = RecallPolicy()
    original = policy._normalize_locatable_query_term
    calls = Counter()

    def normalize(value, **kwargs):
        calls[value] += 1
        return original(value, **kwargs)

    monkeypatch.setattr(policy, '_normalize_locatable_query_term', normalize)
    cue = {'candidate_id': 'Orion42', 'description': 'Vega43 实验结果和后续步骤'}
    query = json.dumps({'activity': {'motivation': [cue] * 80}, 'trigger': cue},
                       ensure_ascii=False, sort_keys=True)
    terms = policy.locatable_query_terms(query)
    assert terms and any('42' in term or '43' in term for term in terms)
    assert calls and max(calls.values()) == 1


def test_complete_eight_term_prefix_does_not_parse_discarded_tail(monkeypatch):
    policy = RecallPolicy()
    query = ' '.join(f'"Probe{index:02}Fixture"' for index in range(8))
    query += ' 后面仍然保留完整记录以及其他词语。' * 80

    def unexpected(*_args, **_kwargs):
        pytest.fail('Later candidates cannot replace the already selected eight terms.')

    monkeypatch.setattr(policy, '_pos_structural_locatable_terms', unexpected)
    monkeypatch.setattr(recall_policy, 'content_terms_for_query', unexpected)
    assert policy.locatable_query_terms(query) == [f'Probe{index:02}Fixture' for index in range(8)]


def test_repeated_entity_tokens_keep_all_suffix_occurrences(monkeypatch):
    policy = RecallPolicy()
    query = 'Orion42老师、Orion42姐姐、Orion42老师'
    monkeypatch.setattr(policy, '_posseg_words', lambda _text: [('Orion42', 'nr')] * 80)
    original = policy._expand_entity_title_suffixes
    calls = []

    def expand(raw, entity):
        calls.append(entity)
        return original(raw, entity)

    monkeypatch.setattr(policy, '_expand_entity_title_suffixes', expand)
    terms = policy._pos_structural_locatable_terms(query, content_term_keys=set())
    assert terms == ['Orion42老师', 'Orion42姐姐']
    assert calls == ['Orion42']


def test_entity_keyword_channel_reuses_complete_suffix_scan(monkeypatch):
    policy = RecallPolicy()
    query = 'Orion42老师、Orion42姐姐、Orion42老师'
    monkeypatch.setattr(policy, '_posseg_words', lambda _text: [('Orion42', 'nr')] * 80)
    original = policy._expand_entity_title_suffixes
    calls = []

    def expand(raw, entity):
        calls.append(entity)
        return original(raw, entity)

    monkeypatch.setattr(policy, '_expand_entity_title_suffixes', expand)
    terms = policy.extract_entity_keywords(query)
    assert 'Orion42老师' in terms and 'Orion42姐姐' in terms
    assert calls == ['Orion42']


def test_strip_fragments_observe_current_identity_between_queries(monkeypatch):
    policy = RecallPolicy()
    names = {'ai_name': 'Aster', 'user_aliases': ['Vega']}
    monkeypatch.setattr(recall_policy, 'identity_names', lambda: names)
    assert policy._normalize_locatable_query_term('AsterProbe42') == 'probe42'
    names['ai_name'] = 'Orion'
    assert policy._normalize_locatable_query_term('AsterProbe42') == 'AsterProbe42'


@pytest.mark.parametrize('focus', ['完整查询', '另一个焦点'])
def test_focus_and_full_query_clean_once_per_distinct_view(monkeypatch, focus):
    calls = []
    monkeypatch.setattr(memory_relevance, 'recall_focus_query', lambda *_args: focus)

    def clean(text, *_args):
        calls.append(text)
        return text

    monkeypatch.setattr(memory_relevance, '_strip_query_water_terms', clean)
    expected = '完整查询' if len('完整查询') < len(focus) else focus
    assert memory_relevance.recall_topic_query('完整查询') == expected
    assert calls == ([focus] if focus == '完整查询' else [focus, '完整查询'])


def test_water_fragments_prepare_once_per_full_cleaning(monkeypatch):
    original = memory_relevance._ordered_cjk_water_terms
    calls = []

    def prepare(terms):
        calls.append(terms)
        return original(terms)

    monkeypatch.setattr(memory_relevance, '_ordered_cjk_water_terms', prepare)
    memory_relevance._strip_query_water_terms('我想查询实验观察以及具体条件。' * 100)
    assert len(calls) == 1


def test_explicit_recall_marker_needs_no_entity_scan_to_reject_probe_only(monkeypatch):
    policy = RecallPolicy()

    def unexpected(_query):
        pytest.fail('An explicit recall request already rules out probe-only.')

    monkeypatch.setattr(policy, 'extract_entity_keywords', unexpected)
    assert policy._is_probe_only_query('test the previous memory search') is False


@pytest.mark.parametrize(('entities', 'expected'), [([], True), (['Orion42'], True), (['小葵'], False)])
def test_probe_only_without_recall_markers_retains_entity_check(monkeypatch, entities, expected):
    policy = RecallPolicy()
    monkeypatch.setattr(policy, 'extract_entity_keywords', lambda _query: entities)
    assert policy._is_probe_only_query('test the observation') is expected


def configuration_service(monkeypatch, threshold):
    from gateway import GatewayService
    from memory_system_config import values_from_config

    service = GatewayService.__new__(GatewayService)
    service.config = {}
    service.recall_policy = RecallPolicy()
    values = values_from_config(service.config)
    values['recall']['semantic_threshold'] = threshold
    monkeypatch.setattr(service, '_maybe_reload_runtime_overlay', lambda: None)
    monkeypatch.setattr(service, '_memory_configuration_snapshot',
                        lambda **_kwargs: {'revision': 42, 'values': values})
    return service


@pytest.mark.parametrize('error_type', [RuntimeError, asyncio.CancelledError])
def test_native_prepare_reuses_one_frozen_policy_and_resets_on_failure(monkeypatch, error_type):
    from memory_system_config import REQUEST_CONFIGURATION

    service = configuration_service(monkeypatch, .81)
    original_scope = REQUEST_CONFIGURATION.get()

    async def prepared(*_args, **_kwargs):
        first = service.recall_policy
        await asyncio.sleep(0)
        assert service.recall_policy is first
        assert first.semantic_threshold == .81
        assert service._recall_query_plan('Probe42') is service._recall_query_plan('Probe42')
        raise error_type('offline_expected_failure')

    monkeypatch.setattr(service, '_prepare_payload_with_clock', prepared)
    with pytest.raises(error_type, match='offline_expected_failure'):
        asyncio.run(service.prepare_payload({'messages': [], '_ombre_trace_context': {
            'runtime_scope': 'internal:autonomy_chain'}}, 'offline-scope', autonomy_context={}))
    assert REQUEST_CONFIGURATION.get() is original_scope


def test_native_prepare_settings_are_isolated_between_concurrent_services(monkeypatch):
    from memory_system_config import REQUEST_CONFIGURATION

    services = [configuration_service(monkeypatch, threshold) for threshold in (.71, .86)]

    async def scenario():
        async def call(service, threshold):
            async def prepared(*_args, **_kwargs):
                policy = service.recall_policy
                await asyncio.sleep(0)
                assert service.recall_policy is policy and policy.semantic_threshold == threshold
                assert REQUEST_CONFIGURATION.get()['service'] == id(service)
                return {'messages': []}, []
            monkeypatch.setattr(service, '_prepare_payload_with_clock', prepared)
            return await service.prepare_payload({'messages': [], '_ombre_trace_context': {
                'runtime_scope': 'internal:autonomy_chain'}}, 'offline-scope', autonomy_context={})
        await asyncio.gather(*(call(service, threshold) for service, threshold in zip(services, (.71, .86))))
    asyncio.run(scenario())


def test_native_prepare_same_service_concurrent_revisions_do_not_mix(monkeypatch):
    from memory_system_config import REQUEST_CONFIGURATION, values_from_config

    service = configuration_service(monkeypatch, .81)
    snapshots = []
    for revision, threshold in [(41, .71), (42, .86)]:
        values = values_from_config({})
        values['recall']['semantic_threshold'] = threshold
        snapshots.append({'revision': revision, 'values': values})
    monkeypatch.setattr(service, '_memory_configuration_snapshot', lambda **_kwargs: snapshots.pop(0))

    async def prepared(_payload, session_id, **_kwargs):
        policy, snapshot = service.recall_policy, REQUEST_CONFIGURATION.get()
        await asyncio.sleep(0)
        expected = (41, .71) if session_id == 'first' else (42, .86)
        assert (snapshot['revision'], policy.semantic_threshold) == expected
        assert service.recall_policy is policy and REQUEST_CONFIGURATION.get() is snapshot
        return {'messages': []}, []
    monkeypatch.setattr(service, '_prepare_payload_with_clock', prepared)
    async def scenario():
        await asyncio.gather(*(service.prepare_payload({'messages': [], '_ombre_trace_context': {
            'runtime_scope': 'internal:autonomy_chain'}}, name, autonomy_context={}) for name in ('first', 'second')))
    asyncio.run(scenario())


def test_chat_prepare_keeps_existing_request_snapshot(monkeypatch):
    from memory_system_config import REQUEST_CONFIGURATION, values_from_config

    service = configuration_service(monkeypatch, .81)
    existing = {'service': id(service), 'revision': 9, 'values': values_from_config({}),
                'effective_recall_policy': service.__dict__['_recall_policy']}
    token = REQUEST_CONFIGURATION.set(existing)
    async def prepared(*_args, **_kwargs):
        assert REQUEST_CONFIGURATION.get() is existing
        return {'messages': []}, []
    monkeypatch.setattr(service, '_prepare_payload_with_clock', prepared)
    try:
        asyncio.run(service.prepare_payload({'messages': []}, 'offline-scope'))
        assert REQUEST_CONFIGURATION.get() is existing
    finally:
        REQUEST_CONFIGURATION.reset(token)


def test_native_prepare_nested_call_keeps_outer_policy_and_snapshot(monkeypatch):
    from memory_system_config import REQUEST_CONFIGURATION

    service = configuration_service(monkeypatch, .81)
    async def prepared(payload, session_id, **_kwargs):
        policy, snapshot = service.recall_policy, REQUEST_CONFIGURATION.get()
        if session_id == 'outer':
            await service.prepare_payload(payload, 'inner', autonomy_context={})
            assert service.recall_policy is policy and REQUEST_CONFIGURATION.get() is snapshot
        return {'messages': []}, []
    monkeypatch.setattr(service, '_prepare_payload_with_clock', prepared)
    asyncio.run(service.prepare_payload({'messages': [], '_ombre_trace_context': {
        'runtime_scope': 'internal:autonomy_chain'}}, 'outer', autonomy_context={}))
