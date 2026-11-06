"""L2 sees selected chat evidence, never an unselected gap via text/title/anchors."""
from copy import deepcopy
import json

import pytest

from memory_l1_contract import make_text_anchor
from memory_l1_schema import document_template, statement_template
from memory_l1_service import L1OwnerError
from memory_l2_service import semantic_input_projection


CHAT_NAMESPACES = ('snapshot.aiz.canonical_event', 'snapshot.ombre.raw_event',
                   'snapshot.ombre.conversation_turn')


def material_for(namespace, text, ranges, *, title='来源标题'):
    anchors = [make_text_anchor(text, target_namespace='l0', target_id='original-l0',
        target_revision='original-v1', text_view_id='memory.source.exact-v1', start=start, end=end)
        for start, end in ranges]
    binding = {'binding_id': 'source:1', 'source_namespace': namespace, 'source_id': 'original-l0',
        'source_revision': 'original-v1', 'author_ref': 'owner', 'speaker_ref': 'owner', 'anchors': anchors}
    document = document_template('所选事件有前后变化。', '所选事件')
    statement = statement_template()
    statement.update(statement_id='s1', content='所选事件有前后变化。', evidence_refs=['source:1'])
    document['statements'] = [statement]
    document['evidence_bindings'] = [binding]
    return {'l1_document': document, 'shared_statements': [], 'rings': [],
        'sources': [{'binding': binding, 'text': text, 'title': title, 'author': 'user', 'speaker': 'user',
                     'recorded_at': '2026-11-06T10:00:00Z', 'eligible_source': True}]}


@pytest.mark.parametrize('namespace', CHAT_NAMESPACES)
def test_two_emoji_spans_in_one_chat_leave_gap_and_l0_unchanged(namespace):
    text = '不可见前缀。先收货😀。__未选中间事实__。后来改期🐈。不可见后缀。'
    left, right = text.index('先收货'), text.index('后来改期')
    ranges = [(right, right + len('后来改期🐈。')), (left, left + len('先收货😀。'))]
    material = material_for(namespace, text, ranges, title=text)
    # A reader's accidental prose sidecar must not bypass the bounded view.
    for anchor in material['l1_document']['evidence_bindings'][0]['anchors']:
        anchor.update(text=text, quote=text, reader_note='__未选中间事实__')
    material['sources'][0]['raw_snapshot'] = text
    before = deepcopy(material)
    projected = semantic_input_projection(material)
    assert material == before
    source, = projected['sources']
    assert 'text' not in source and 'title' not in source and 'raw_snapshot' not in source
    assert source['read_ranges'] == [
        {'start': left, 'end': left + len('先收货😀。'), 'text': '先收货😀。'},
        {'start': right, 'end': right + len('后来改期🐈。'), 'text': '后来改期🐈。'}]
    assert source['source_length'] == len(text)
    assert source['coordinate_unit'] == 'unicode_codepoint_absolute_half_open'
    assert source['author'] == source['speaker'] == 'user'
    wire = json.dumps(projected, ensure_ascii=False)
    assert '__未选中间事实__' not in wire and '不可见前缀' not in wire and '不可见后缀' not in wire
    output_anchors = projected['l1_document']['evidence_bindings'][0]['anchors']
    assert all(set(anchor) == {'start', 'end', 'coordinate_unit'} for anchor in output_anchors)
    assert material['sources'][0]['text'] == text
    assert material['sources'][0]['binding']['source_revision'] == 'original-v1'


@pytest.mark.parametrize('namespace', CHAT_NAMESPACES)
@pytest.mark.parametrize('split', [False, True])
def test_complete_old_chat_stays_readable_text_including_adjacent_full_coverage(namespace, split):
    text = '旧聊天😀保持完整且可读。'
    ranges = [(0, 3), (3, len(text))] if split else [(0, len(text))]
    material = material_for(namespace, text, ranges, title='原有标题')
    projected = semantic_input_projection(material)
    source, = projected['sources']
    assert source['text'] == text and source['title'] == '原有标题'
    assert 'read_ranges' not in source and 'source_length' not in source


@pytest.mark.parametrize('namespace', [None, *CHAT_NAMESPACES])
def test_legacy_rows_without_anchor_ranges_keep_existing_readable_source(namespace):
    material = material_for(namespace, '旧行没有分段边界。', [])
    material['sources'][0]['binding'].pop('anchors')
    if namespace is None:
        material['sources'][0]['binding'].pop('source_namespace')
    before = deepcopy(material)
    assert semantic_input_projection(material)['sources'][0]['text'] == '旧行没有分段边界。'
    assert material == before


def test_existing_browse_projection_keeps_title_and_separate_read_ranges():
    text = '读到😀。__未读秘密__。读到🐈。'
    last = text.index('读到🐈')
    material = material_for('snapshot.aiz.browse.article', text, [(0, 2), (2, 4), (last, len(text))],
                            title='原文章标题')
    before = deepcopy(material)
    projected = semantic_input_projection(material)
    assert material == before
    source, = projected['sources']
    assert 'text' not in source and source['title'] == '原文章标题'
    assert source['read_ranges'] == [{'start': 0, 'end': 4, 'text': '读到😀。'},
                                    {'start': last, 'end': len(text), 'text': '读到🐈。'}]
    assert '__未读秘密__' not in json.dumps(projected, ensure_ascii=False)


@pytest.mark.parametrize('start,end', [(-1, 2), (0, 99), (0, 0), (False, 2), (1, 1.5)])
def test_invalid_chat_bounds_never_fall_back_to_full_text(start, end):
    material = material_for(CHAT_NAMESPACES[0], '原L0全文。', [(0, 2)])
    material['sources'][0]['binding']['anchors'][0].update(start=start, end=end)
    with pytest.raises(L1OwnerError, match='l2_source_range_invalid'):
        semantic_input_projection(material)


def test_non_chat_non_article_projection_is_unchanged():
    material = material_for('snapshot.ombre.diary_mcp', '原日记仍使用既有投影。', [(0, 2)])
    source, = semantic_input_projection(material)['sources']
    assert source['text'] == '原日记仍使用既有投影。' and 'read_ranges' not in source
