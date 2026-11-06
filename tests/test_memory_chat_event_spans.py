"""Selected event quotes retain L0 authority and exact observed codepoint bounds."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from zoneinfo import ZoneInfo

import pytest

from memory_authority import MemoryAuthorityStore
from memory_chat_event_spans import parse_event_selection, event_source_views, event_envelope
from memory_l1_api import prepare_request
from memory_l1_ingestion import prepare_model_document
from memory_l1_runtime import MemoryL1Runtime
from memory_l1_semantic_wire import assemble_document
from memory_l1_service import L1OwnerError, MemoryL1RevisionService
from memory_source_provenance import canonical_source
from memory_system_config import defaults


def selection(anchors, *, context='上下文不是证据', point='同一件事的前后变化'):
    return {'context': context, 'memories': [{'point': point, 'anchors': anchors}]}


def fixture():
    config = {'identity': {'ai_name': '测试助手', 'user_name': '测试主人',
                          'user_display_name': '测试主人', 'user_aliases': ['称呼']},
              'persona': {'profile_id': 'span-profile'},
              'gateway': {'canonical_continuation': {'enabled': True, 'conversation_id': 'span-conversation'}}}
    catalog = defaults()['catalog']
    engine = NS(config=config, memory_semantic_profile_id='span-profile', tz=ZoneInfo('Asia/Shanghai'),
                _memory_l1_revisions=lambda: NS(catalog_provider=lambda: (catalog, 'span-catalog-v1')))
    texts = ['外层未选甲。', '外层未选乙。', '不可见😀前缀。先收货😀。中间无关闲聊。后来改期🐈。不可见后缀。']
    sources = [canonical_source({'event_id': f'span-event-{index}', 'version_id': 'original-v1',
                'role': 'user', 'text': text, 'created_at': '2026-11-06T10:00:00+08:00'},
                conversation_id='span-conversation', profile_id='span-profile') for index, text in enumerate(texts)]
    text = sources[2]['text']
    low, high = text.index('先收货'), text.index('不可见后缀')
    sources[2]['_model_range'] = {'start': low, 'end': high}
    projected = [{'binding_id': 'source:3', 'text': text[low:high],
                  'input_range': {'start': low, 'end': high}, 'source_length': len(text)}]
    raw = selection([{'source': 'source:3', 'quote': '后来改期🐈。'},
                     {'source': 'source:3', 'quote': '先收货😀。'}])
    return engine, sources, projected, raw


def test_unicode_outer_offset_disjoint_views_and_formal_wire_preserve_l0():
    engine, originals, projected, raw = fixture()
    before = deepcopy((originals, projected, raw))
    event, = parse_event_selection(raw, projected)
    text = originals[2]['text']
    assert event == {'point': raw['memories'][0]['point'], 'anchors': [
        {'source': 'source:3', 'start': text.index('先收货'), 'end': text.index('先收货') + len('先收货😀。')},
        {'source': 'source:3', 'start': text.index('后来改期'), 'end': text.index('后来改期') + len('后来改期🐈。')}]}
    views, fragments = event_source_views(event, originals)
    assert [row['source'] for row in fragments] == ['source:1', 'source:2']
    assert [row['original_index'] for row in fragments] == [2, 2]
    assert all(view['text'] == text and view['version_id'] == 'original-v1'
               and view['event_id'] == originals[2]['event_id'] and view['role'] == 'user' for view in views)
    untouched_views = deepcopy(views)
    second = fragments[1]
    envelope, prepared_sources, catalog = event_envelope(engine, views, fragments,
        core_source='source:3', core_range={'start': second['start'] + 1, 'end': second['end'] + 1})
    assert views == untouched_views and (originals, projected, raw) == before
    assert len(prepared_sources) == 2 and catalog[1] == 'span-catalog-v1'
    assert [row['text'] for row in envelope['source_texts']] == ['先收货😀。', '后来改期🐈。']
    assert envelope['segment']['core_source'] == 'source:2'
    assert envelope['segment']['core_range'] == {'start': second['start'] + 1, 'end': second['end']}
    messages, schema = prepare_request([{'role': 'system', 'content': '测试'},
        {'role': 'user', 'content': json.dumps({'l1_contract': envelope}, ensure_ascii=False)}], stage='auto_l1')
    wire = json.dumps(messages, ensure_ascii=False)
    assert '中间无关闲聊' not in wire and '不可见😀前缀' not in wire and '不可见后缀' not in wire
    assert raw['context'] not in wire
    assert schema['$defs']['EvidenceRef']['enum'] == ['source:1', 'source:2']
    assert all(row['_model_range'] == {'start': fragment['start'], 'end': fragment['end']}
               for row, fragment in zip(prepared_sources, fragments))


@pytest.mark.parametrize('text,anchor,code', [
    ('同句😀同句😀', {'source': 'source:1', 'quote': '同句😀'}, 'event_quote_not_unique'),
    ('原话。', {'source': 'source:1', 'quote': '原话!'}, 'event_quote_not_unique'),
    ('原话。', {'source': 'source:1', 'quote': ''}, 'event_evidence_required'),
    ('原话。', {'source': 'source:1', 'quote': '  '}, 'event_evidence_required'),
    ('原话。', {'source': 'source:2', 'quote': '原话'}, 'event_source_not_visible'),
    ('原话。', {'source': 'source:0', 'quote': '原话'}, 'event_source_reference_invalid'),
    ('原话。', {'source': 'source:1', 'quote': '上下文不是证据'}, 'event_quote_not_unique'),
])
def test_invalid_quote_or_unseen_source_rejects_without_context_fallback(text, anchor, code):
    with pytest.raises(L1OwnerError, match=code):
        parse_event_selection(selection([anchor]), [{'binding_id': 'source:1', 'text': text}])


def test_empty_selection_and_missing_effective_evidence_are_distinct():
    assert parse_event_selection({'context': '这句也不能成为候选', 'memories': []}, []) == []
    for raw in (selection([]), selection([{'source': 'source:1', 'quote': '甲'}], point=' ')):
        with pytest.raises(L1OwnerError):
            parse_event_selection(raw, [{'binding_id': 'source:1', 'text': '甲'}])
    with pytest.raises(L1OwnerError, match='event_selection_invalid'):
        parse_event_selection({'context': '只有上下文，没有选材结果'}, [])
    with pytest.raises(L1OwnerError, match='event_evidence_required'):
        event_source_views({'point': '只有摘要', 'anchors': []}, [])
    with pytest.raises(L1OwnerError, match='event_evidence_required'):
        event_envelope(NS(), [], [])


@pytest.mark.parametrize('extent,source_length', [
    ({'start': 4, 'end': 8}, 20),  # Visible text length and absolute width disagree.
    ({'start': -1, 'end': 1}, 20),
    ({'start': False, 'end': 2}, 20),
    ({'start': 10, 'end': 12}, 11),
])
def test_malformed_outer_ranges_are_not_guessed(extent, source_length):
    with pytest.raises(L1OwnerError, match='event_source_range_invalid'):
        parse_event_selection(selection([{'source': 'source:1', 'quote': '😀'}]),
            [{'binding_id': 'source:1', 'text': '甲😀', 'input_range': extent, 'source_length': source_length}])


def test_duplicate_or_overlapping_quote_occurrences_are_not_silently_chosen():
    anchor = {'source': 'source:1', 'quote': '甲'}
    with pytest.raises(L1OwnerError, match='event_anchor_duplicate'):
        parse_event_selection(selection([anchor, anchor]), [{'binding_id': 'source:1', 'text': '甲'}])
    with pytest.raises(L1OwnerError, match='event_quote_not_unique'):
        parse_event_selection(selection([{'source': 'source:1', 'quote': 'aa'}]),
                              [{'binding_id': 'source:1', 'text': 'aaa'}])
    with pytest.raises(L1OwnerError, match='event_source_reference_ambiguous'):
        parse_event_selection(selection([anchor]), [{'binding_id': 'source:1', 'text': '甲'}] * 2)


@pytest.mark.parametrize('start,end', [(-1, 2), (0, 999), (False, 2), (0, 1), (12, 12)])
def test_direct_anchor_cannot_escape_original_or_outer_input_range(start, end):
    _, originals, _, _ = fixture()
    with pytest.raises(L1OwnerError, match='event_source_range_invalid'):
        event_source_views({'point': '越界', 'anchors': [{'source': 'source:3', 'start': start, 'end': end}]}, originals)


def test_outer_context_only_selection_and_forged_view_ranges_are_rejected():
    engine, originals, projected, raw = fixture()
    event, = parse_event_selection(raw, projected)
    views, fragments = event_source_views(event, originals)
    with pytest.raises(L1OwnerError, match='event_core_not_observed'):
        event_envelope(engine, views, fragments, core_source='source:1', core_range={'start': 0, 'end': 2})
    with pytest.raises(L1OwnerError, match='event_source_range_invalid'):
        event_envelope(engine, views, fragments, core_source='source:3')
    forged = deepcopy(fragments)
    forged[0]['end'] += 1
    with pytest.raises(L1OwnerError, match='event_source_range_invalid'):
        event_envelope(engine, views, forged)
    originals[2]['_model_excluded'] = True
    with pytest.raises(L1OwnerError, match='event_source_not_visible'):
        event_source_views(event, originals)


def test_formal_document_accepts_same_l0_views_but_rejects_gap_as_evidence(tmp_path):
    engine, originals, projected, raw = fixture()
    event, = parse_event_selection(raw, projected)
    views, fragments = event_source_views(event, originals)
    _, sources, catalog = event_envelope(engine, views, fragments)
    service = MemoryL1RevisionService(MemoryAuthorityStore(str(tmp_path / 'local.sqlite3')),
                                    None, None, lambda: catalog)
    document = {'format': 'semantic-v1', 'narrative': {'title': '人工边界夹具', 'body': '先收货，后来改期。'},
        'entities': [{'entity_id': 'e1', 'name': '测试主人', 'entity_type': 'user', 'evidence_refs': ['source:1']}],
        'statements': [{'statement_id': 's1', 'content': '先收货，后来改期。', 'subject_ref': 'e1',
            'classification': {'kind': 'fact', 'stability': 'episode', 'modality': 'assertion', 'basis': 'direct_expression'},
            'time': None, 'conditions': None, 'evidence_refs': ['source:1', 'source:2']}],
        'retrieval': {'keywords': ['收货', '改期']}}

    def prepare(doc, key):
        return prepare_model_document(service, memory_id=key,
            document=assemble_document(doc, require_keywords=True), sources=sources,
            generation={'prompt_revision': None, 'model_route_revision': None, 'task_id': key,
                        'actor': 'daily_chat_first_pass'}, request_id=key, catalog_context=catalog,
            config=engine.config, candidate_refs=['source:1', 'source:2'])

    full = prepare(document, 'span-positive')
    bindings = full['evidence_bindings']
    assert len(bindings) == 2 and bindings[0]['source_id'] == bindings[1]['source_id']
    assert bindings[0]['source_revision'] == bindings[1]['source_revision']
    assert [binding['anchors'][0]['start'] for binding in bindings] == [row['start'] for row in fragments]
    bad = deepcopy(document)
    bad['scenes'] = [{'scene_id': 'c1', 'title': '越过未选闲聊', 'statement_refs': ['s1'],
        'source_ranges': [{'source_ref': 'source:1', 'start': fragments[0]['start'], 'end': fragments[1]['end']}]}]
    with pytest.raises(L1OwnerError, match='scene_range_outside_observed_input'):
        prepare(bad, 'span-negative')
