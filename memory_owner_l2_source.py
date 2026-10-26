"""Actual owner correction actions, not unedited model prose promoted to L0."""
from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import json

from memory_l1_contract import canonical_json, make_text_anchor
from memory_l1_service import L1OwnerError

NAMESPACE = 'snapshot.ombre.owner_l2_edit'


def changed_passages(before, after):
    if before == after:
        return []
    # Trim common edges first: normal small corrections in a long owner draft
    # do not compare the unchanged full narrative quadratically.
    left, common = 0, min(len(before), len(after))
    while left < common and before[left] == after[left]:
        left += 1
    right = 0
    while right < common - left and before[-right - 1] == after[-right - 1]:
        right += 1
    a, b = before[left:len(before) - right if right else None], after[left:len(after) - right if right else None]
    return [{'before': a[i:j], 'after': b[k:l], 'before_range': [left + i, left + j],
             'after_range': [left + k, left + l]} for tag, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if tag != 'equal']


class OwnerL2EditSourceReader:
    namespace = NAMESPACE

    def __init__(self, authority):
        self.authority = authority

    async def capture(self, memory_id, presentation_revision):
        locator = canonical_json({'memory_id': memory_id, 'presentation_revision': presentation_revision})
        return await self.read({'source_id': locator, 'source_revision': presentation_revision})

    async def read(self, binding):
        locator = json.loads(binding['source_id'])
        if (not isinstance(locator, dict) or set(locator) != {'memory_id', 'presentation_revision'}
                or type(locator['presentation_revision']) is not int or locator['presentation_revision'] < 1
                or locator['presentation_revision'] != binding['source_revision']):
            return {'state': 'unavailable', 'reason': 'owner_l2_source_identity_mismatch'}
        mid, revision = locator['memory_id'], locator['presentation_revision']
        memory = self.authority.get_memory(mid)
        if not memory or memory['state'] != 'active' or memory['recall_policy'] == 'disabled':
            return {'state': 'withdrawn', 'reason': 'owner_l2_memory_not_eligible'}
        state = self.authority.get_meta('owner_l2_source_state:' + binding['source_id'], {'state': 'active'})
        if state.get('state') != 'active':
            return {'state': 'withdrawn', 'reason': 'owner_l2_correction_withdrawn'}
        db = self.authority._connect()
        try:
            row = db.execute('SELECT actor,document_json,edit_note,created_at FROM memory_l2_revisions '
                'WHERE memory_id=? AND presentation_revision=?', (mid, revision)).fetchone()
            proposal = db.execute('SELECT proposal_json FROM memory_l2_sync_proposals WHERE memory_id=? AND presentation_revision=?',
                (mid, revision)).fetchone()
            if not row or row['actor'] != 'owner' or not proposal:
                return {'state': 'unavailable', 'reason': 'owner_l2_correction_not_recorded'}
            original = json.loads(proposal['proposal_json'])
            document = json.loads(row['document_json'])
            if original.get('proposed_text') != document['full_injection_text']:
                return {'state': 'changed', 'reason': 'owner_l2_correction_hash_mismatch'}
            passages = changed_passages(original.get('before_text', ''), document['full_injection_text'])
            text = canonical_json({'event': 'owner_l2_fact_correction', 'changed_passages': passages,
                'owner_edit_note': row['edit_note'],
                'scope': 'Only the change action and changed passages are owner correction evidence; unchanged model prose is comparison context.'})
            if not passages and not row['edit_note'].strip():
                return {'state': 'unavailable', 'reason': 'owner_correction_has_no_changed_material'}
            return {'state': 'available', 'eligible_source': True, 'source_namespace': NAMESPACE,
                'source_id': binding['source_id'], 'source_revision': revision, 'text_view_id': 'owner-l2-correction.exact-v1',
                'text': text, 'author': 'owner', 'speaker': 'owner',
                'participant': {'subject_id': 'owner:memory-edit', 'kind': 'human', 'display_name': '主人', 'resolution': 'authenticated_owner_edit'},
                'recorded_at': datetime.fromtimestamp(row['created_at'], timezone.utc).isoformat(),
                'limitations': ['仅本次实际修改和修改说明是主人更正依据；未改旧模型正文不因此成为主人原话。']}
        finally:
            db.close()


def bind(snapshot, binding_id='owner-correction'):
    if snapshot.get('state') != 'available' or snapshot.get('eligible_source') is not True:
        raise L1OwnerError('owner_l2_correction_source_unavailable', status=409, field_path='sources')
    return {'binding_id': binding_id, 'source_namespace': NAMESPACE, 'source_id': snapshot['source_id'],
        'source_revision': snapshot['source_revision'], 'author_ref': snapshot['participant']['subject_id'],
        'speaker_ref': snapshot['participant']['subject_id'], 'anchors': [make_text_anchor(snapshot['text'], target_namespace='l0',
            target_id=snapshot['source_id'], target_revision=snapshot['source_revision'], text_view_id=snapshot['text_view_id'],
            start=0, end=len(snapshot['text']))]}
