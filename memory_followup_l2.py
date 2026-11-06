"""L2-only owner feedback on the existing followup and presentation workflow."""
from copy import deepcopy
from urllib.parse import quote

from memory_authority import MemoryProposal, PolicyDecision
from memory_l1_contract import semantic_sha256
from memory_l1_service import L1OwnerError


async def review_context(service, task_id, *, memory_id=None):
    from memory_self_review_source import check_native_sources, native_feedback_layer
    from memory_l2_service import MemoryL2Service
    if not isinstance(task_id, str) or not task_id.startswith('followup_') or len(task_id) > 200:
        raise L1OwnerError('invalid_followup_task_id')
    task = service.tasks.get(task_id)
    if not task or memory_id is not None and task['memory_id'] != memory_id:
        raise L1OwnerError('followup_not_found', status=404)
    envelope = task['envelope']
    if envelope.get('schema') != 'self-review-followup-v1':
        raise L1OwnerError('native_l2_followup_required', status=409)
    update = envelope['updates'][0]
    await service._check_native_delivery(task)
    allowed, actual = await check_native_sources(envelope, service.engine.config,
        service.review_source_reader, with_sources=True)
    if update['update_id'] not in allowed or native_feedback_layer(actual['self_review'], update, actual) != 'l2':
        raise L1OwnerError('qualified_l2_followup_required', status=409)
    ref = update['memory_ref']
    detail = await service.revisions.detail(task['memory_id'])
    if (detail['memory']['state'] != 'active' or detail['l1']['state'] != 'current'
            or detail['current_revision'] != ref['l1_revision']):
        raise L1OwnerError('followup_target_changed', status=409)
    l2 = MemoryL2Service(service.engine, service.revisions).detail(task['memory_id'])['l2']
    if (l2['state'] != 'ready' or type(ref['l2_revision']) is not int
            or l2['current_presentation_revision'] != ref['l2_revision']
            or not l2['presentation'] or l2['presentation']['first_pass_revision'] != ref['l1_revision']):
        raise L1OwnerError('followup_original_l2_changed', status=409)
    candidate = service.authority.get_candidate(task_id)
    if candidate and candidate['status'] in {'rejected', 'committed'}:
        raise L1OwnerError('followup_not_waiting_review', status=409)
    feedback = [actual[key] for key in update['evidence_refs'] if actual[key]['kind'] == 'reply_feedback']
    diagnoses = [d for d in actual['self_review']['feedback_diagnoses']
                 if d['feedback_id'] in {f['feedback_id'] for f in feedback}]
    material = {'original_l2': deepcopy(l2['presentation']['document']),
        'proposed_difference': (task.get('draft_content') or
            (candidate['proposal']['proposed_body'] if candidate else update['content'])),
        'owner_annotations': [{'text': f['text'], 'recorded_at': f['recorded_at']} for f in feedback],
        'diagnoses': [{'layer': d['layer'], 'reason': d['reason']} for d in diagnoses],
        'instruction': '依据已核对的L1、原来源和主人指出的具体差异修订完整L2稿；保留无关细节和条件，不改L1。生成稿必须等待主人审核，不能覆盖现有手稿。'}
    identity = {'task_id': task_id, 'memory_id': task['memory_id'],
        'l1_revision': ref['l1_revision'], 'l2_revision': ref['l2_revision'],
        'candidate_revision': candidate['revision'] if candidate else None,
        'l1_sha256': semantic_sha256(detail['l1']['document']), 'material_sha256': semantic_sha256(material)}
    return {'task': task, 'identity': identity, 'material': material}


def actions(task):
    mid, task_id = task['memory_id'], task['task_id']
    ref = task['envelope']['updates'][0]['memory_ref']
    return {'presentation_url': '/api/memory-authority/memories/' + quote(mid, safe='') + '/presentation',
        'prepare_payload': {'expected_l1_revision': ref['l1_revision'],
            'request_id': 'followup-l2:' + task_id + ':draft:' + str(task['draft_revision']), 'followup_task_id': task_id},
        'save_parameters': {'followup_task_id': task_id, 'change_kind': 'wording'},
        'notice': '在原L2呈现页查看完整旧稿与候选稿，可编辑后按原保存流程采用。刷新只生成待审稿，不替换主人手稿；保存须带此任务ID。'}


def prepare_candidate(service, claimed, sources):
    update = claimed['envelope']['updates'][0]
    content = claimed.get('draft_content') or update['content']
    ref = update['memory_ref']
    fields = claimed.get('draft_fields') or {}
    legacy = {'id': claimed['task_id'], 'title': 'L2呈现修订待审', 'content': content,
        'kind': 'reflection', 'mode': 'review', 'source_verification': 'verified',
        'reason': 'L1保持准确；原批注指出L2呈现差异，完整稿须由主人审核。', **deepcopy(fields)}
    proposal = MemoryProposal.from_mapping({'proposal_id': claimed['task_id'],
        'source_type': 'daily_chat_memory', 'proposed_body': content,
        'original_excerpt': '\n'.join(s['text'] for s in sources).replace('<','‹').replace('>','›'),
        'source_refs': [s['namespace'] + ':' + s['event_id'] + '@' + s['version_id'] for s in sources],
        'source_status': 'verified', 'memory_type': fields.get('kind','reflection'), 'requested_mode': 'review',
        'confidence': None, 'owner_explicit': False, 'metadata': {'legacy_candidate': legacy,
            'l3_followup': {'task_id': claimed['task_id'], 'memory_id': claimed['memory_id'],
                'expected_revision': ref['l1_revision'], 'original_l2_revision': ref['l2_revision'],
                'layer': 'l2', 'kind': update['kind'], 'plan': None, 'compiled': None,
                'review_actions': actions(claimed)}}})
    # This is the existing presentation review boundary, not the L1 auto policy.
    policy = PolicyDecision('queue_review', ('REVIEW_L2_PRESENTATION_REQUIRED',),
        'l2_presentation_review', requires_owner_confirmation=True)
    published, candidate = service.tasks.publish(claimed, proposal, policy,
        generation={'origin': 'qualified_owner_l2_feedback', 'model_calls': 0}, model_io_id=None)
    claimed.clear(); claimed.update(published)
    return {'phase': 'waiting_review', 'candidate_ids': [candidate['candidate_id']],
            'receipt': None, 'draft_content': None}


def acknowledge_save(service, task_id):
    """A real owner L2 save authorizes closing its original pending proposal."""
    current = service.tasks.current_receipt(task_id)
    if current['phase'] != 'adopted':
        return
    candidate = service.authority.get_candidate(task_id)
    if candidate and candidate['status'] in {'pending','deferred','commit_failed'}:
        candidate = service.authority.decide_candidate(task_id, action='accept',
            expected_revision=candidate['revision'], request_id='l2-followup-owner:' + task_id,
            actor='owner', reason_codes=['OWNER_L2_PRESENTATION_SAVED'])
    if candidate and candidate['status'] == 'accepted':
        service.authority.decide_candidate(task_id, action='commit', expected_revision=candidate['revision'],
            request_id='l2-followup-committed:' + task_id, actor='commit_service',
            reason_codes=['TARGET_L2_PRESENTATION_COMMITTED'])
