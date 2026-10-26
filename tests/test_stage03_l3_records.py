"""L3 uses existing history/notes without promoting comments to L0 facts."""
import asyncio
from copy import deepcopy

import pytest

from memory_authority import IdempotencyConflict, RevisionConflict
from memory_l3_records import MemoryL3Records, relevant_rings
from test_memory_l2_p4 import setup


def run(task):
    return asyncio.run(task)


def test_note_save_edit_retract_history_is_persistent_and_idempotent(tmp_path):
    l2, authority, calls = setup(tmp_path)
    service = MemoryL3Records(authority, l2.revisions.commit)
    payload = {'expected_revision': 2, 'request_id': 'note-1', 'content': '以后再看这段经历时，我想记得这个小细节。'}
    first = run(service.save_note('m1', payload))
    assert first['committed'] is True and first['note']['details']['l3_role'] == 'note'
    assert first['note']['source_refs'] == []
    assert run(service.save_note('m1', payload))['replayed']
    with pytest.raises(IdempotencyConflict):
        run(service.save_note('m1', {**payload, 'content': '不能借同一次请求改成另一份'}))
    edited = run(service.save_note('m1', {**payload, 'request_id': 'note-edit', 'content': '更正这个补充，但不覆盖原历史。',
        'replaces_ring_id': first['ring_id'], 'expected_content_sha256': first['note']['details']['content_sha256']}))
    assert edited['replaced_ring_id'] == first['ring_id']
    restored = MemoryL3Records(authority, l2.revisions.commit)
    timeline = restored.timeline('m1')
    notes = {r['event_id']: r for r in timeline['items'] if r['kind'] == 'note'}
    assert notes[first['ring_id']]['state'] == 'retracted'
    assert notes[edited['ring_id']]['content'] == '更正这个补充，但不覆盖原历史。'
    result = run(restored.retract_note('m1', edited['ring_id'], {'expected_revision': 2,
        'request_id': 'retract', 'expected_content_sha256': edited['note']['details']['content_sha256']}))
    assert result['note']['state'] == 'retracted'
    assert authority.get_memory('m1')['active_revision'] == 2
    assert calls == []


def test_stale_note_cannot_replace_newer_supplement_or_memory_revision(tmp_path):
    l2, authority, _ = setup(tmp_path)
    service = MemoryL3Records(authority, l2.revisions.commit)
    with pytest.raises(RevisionConflict):
        run(service.save_note('m1', {'expected_revision': 1, 'request_id': 'stale', 'content': 'draft kept'}))
    note = run(service.save_note('m1', {'expected_revision': 2, 'request_id': 'first', 'content': 'original'}))
    with pytest.raises(RevisionConflict):
        run(service.retract_note('m1', note['ring_id'], {'expected_revision': 2, 'request_id': 'wrong-source',
            'expected_content_sha256': '0' * 64}))
    assert service._ring('m1', note['ring_id'])['state'] == 'active'


def test_plain_and_legacy_marker_notes_never_become_l2_source_dependencies():
    doc = {'evidence_bindings': [{'binding_id': 'source:A'}]}
    old = {'ring_id': 'legacy', 'state': 'active', 'source_refs': ['source:A'], 'metadata': {}, 'content': 'old sourced followup'}
    marker = {**old, 'ring_id': 'old-entry-name', 'source_refs': ['dashboard']}
    note = {**old, 'ring_id': 'note', 'metadata': {'l3_role': 'note'}}
    pending = {**old, 'ring_id': 'pending', 'metadata': {'l3_role': 'adopted_update'}}
    adopted = {**old, 'ring_id': 'adopted', 'metadata': {'l3_role': 'adopted_update', 'adoption_receipt': {'operation_id': 'op'}}}
    rows = [old, marker, note, pending, adopted]
    original = deepcopy(rows)
    assert [r['ring_id'] for r in relevant_rings(rows, doc)] == ['legacy', 'adopted']
    assert rows == original


def test_timeline_pagination_preserves_all_note_history(tmp_path):
    l2, authority, _ = setup(tmp_path)
    service = MemoryL3Records(authority, l2.revisions.commit)
    for index in range(4):
        run(service.save_note('m1', {'expected_revision': 2, 'request_id': f'note-{index}', 'content': f'补充 {index}'}))
    first, second, third = [service.timeline('m1', offset=offset, limit=2) for offset in (0, 2, 4)]
    ids = [r['event_id'] for page in (first, second, third) for r in page['items']]
    assert len(set(ids)) == len(ids) == first['total'] == 6
