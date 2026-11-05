"""One tiny synthetic owner-summary through the actual configured model routes.

Default is preparation only. --run-real permits at most one auto_l1 and one
presentation_l2 invocation, SDK retries disabled. All durable writes are under
a newly created --output directory; no live messages, source bridge or memory
database are read. Existing prompt configuration is read and copied, never edited.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bucket_manager import BucketManager
from memory_l0_sources import MemoryL0Sources
from memory_l1_source_readers import configured_source_readers
from memory_l2_service import MemoryL2Service
from memory_owner_summary import OwnerSummaryPreparation, OwnerSummarySourceReader, prepare_owner_summary_tasks
from memory_system_config import merge_config, read_overlay, runtime_overlay_path
from reflection_engine import ReflectionEngine
from utils import load_config


async def validate(args):
    config_path = Path(args.config).resolve()
    if not config_path.is_file(): raise ValueError('validation_config_file_required')
    original = load_config(str(config_path))
    config = merge_config(original, read_overlay(runtime_overlay_path(original)))
    config = copy.deepcopy(config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    mirror = ((config.get('gateway') or {}).get('prompt_plan_mirror_path')
              or str(Path(config.get('buckets_dir') or '.') / 'prompt_plan_mirror.sqlite3'))
    copied_mirror = output / 'prompt-plan.sqlite3'
    if Path(mirror).is_file():
        with closing(sqlite3.connect(Path(mirror).resolve().as_uri() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(copied_mirror)) as target: source.backup(target)
    config.update(state_dir=str(output / 'state'), buckets_dir=str(output / 'buckets'),
        _runtime_config_path=str(output / 'unused-runtime-overlay.yaml'))
    config['memory_authority'] = {**config.get('memory_authority', {}), 'enabled':True,
        'db_path':str(output / 'authority.sqlite3')}
    config['gateway'] = {**config.get('gateway', {}), 'prompt_plan_mirror_path':str(copied_mirror),
        'canonical_continuation':{'enabled':True, 'base_url':'http://synthetic-source.invalid',
            'conversation_id':'unrelated-canonical-scope'}}
    config['reflection'] = {**config.get('reflection', {}), 'daily_chat_memory_mode':'review'}
    policy = config['memory_authority'].setdefault('ingestion_policy', {})
    policy['source_modes'] = {**policy.get('source_modes', {}), 'daily_chat':'review', 'owner_summary':'review'}
    engine = ReflectionEngine(config)
    authority, buckets = engine.memory_authority_store, BucketManager(config)
    revisions = engine._memory_l1_revisions(buckets)
    engine._memory_l1_revisions = lambda manager=None: revisions
    readers = MemoryL0Sources(authority, configured_source_readers(config, authority))
    engine._memory_l1_sources = lambda: readers
    body = '这是隔离验证的虚构资料，不是真实主人谈话。小禾说：以后给我推荐饮料时，别选明显苦的；我还没决定周三是否去看展。'
    summary = {'id':'synthetic-summary-only', 'revision':1, 'body':'虚构人物小禾的饮料边界，以及尚未决定的看展打算。'}
    summary['body_sha256'] = hashlib.sha256(summary['body'].encode()).hexdigest()
    event = {'event_id':'synthetic-owner-summary-event', 'version_id':'synthetic-version-1', 'role':'user',
        'text':body, 'created_at':'2026-11-01T00:00:00+00:00'}
    envelope = {'operation_id':'synthetic.owner-summary.validation', 'summary':summary,
        'context':{'conversation_id':'synthetic-conversation', 'branch_id':'synthetic-branch',
            'profile_id':'synthetic-profile', 'sources':[event]}}
    async def synthetic_capture(self, operation_id, locator, version):
        if operation_id != envelope['operation_id'] or locator != {'event_id':event['event_id']} or version != event['version_id']:
            return {'state':'unavailable','eligible_source':False}
        return {'state':'available','eligible_source':True,'source_revision':event['version_id'],
            'text':event['text'],'role':event['role'],'recorded_at':event['created_at'],
            'conversation_id':'synthetic-conversation','branch_id':'synthetic-branch','profile_id':'synthetic-profile',
            'summary':{k:summary[k] for k in ('id','revision','body_sha256')}}
    # This isolated process owns the fixed synthetic reader only. No source
    # request can reach a real Aiz bridge, canonical event or attachment.
    OwnerSummarySourceReader.capture = synthetic_capture
    os.environ.setdefault('OMBRE_CANONICAL_BRIDGE_TOKEN','synthetic-validation-only')
    tasks = OwnerSummaryPreparation(authority,engine)
    receipt = tasks.enqueue(envelope,budget=12000)
    calls = {'auto_l1':0,'presentation_l2':0}
    original_completion = engine._daily_chat_memory_create_completion
    async def once(client, **kwargs):
        stage = kwargs['task_stage']
        if stage not in calls or calls[stage] >= 1: raise RuntimeError('isolated_call_limit')
        calls[stage] += 1
        client = client.with_options(max_retries=0)
        kwargs['task_route'] = {**kwargs['task_route'],'client':client}
        return await original_completion(client,**kwargs)
    engine._daily_chat_memory_create_completion = once
    routes = {stage:{k:v for k,v in engine._memory_task_route(stage).items()
        if k in ('model','configured','thinking_mode','reasoning_effort','max_tokens','connection','stage')}
        for stage in calls}
    result = {'synthetic_only':True,'production_writes':0,'calls':calls,'routes':routes,'receipt':receipt}
    if args.run_real:
        result['l1_worker'] = await prepare_owner_summary_tasks(engine,buckets,max_jobs=1)
        receipt = tasks.receipt(receipt['task_id'])
        if receipt['candidate_ids']:
            cid = receipt['candidate_ids'][0]
            candidate = authority.get_candidate(cid)
            if candidate['status'] in ('pending','accepted'):
                result['review_adoption'] = await engine._memory_l1_adopt_candidate(candidate,buckets,
                    owner=True,request_id='synthetic-review-only')
                result['l2_worker'] = await MemoryL2Service(engine,revisions).run_once(1)
        result['receipt'] = tasks.receipt(receipt['task_id'])
    (output / 'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Existing effective configuration; read only.')
    parser.add_argument('--output',required=True,help='New isolated directory; must not already exist.')
    parser.add_argument('--run-real',action='store_true',help='Explicitly run <= 1 auto_l1 and <= 1 presentation_l2 call.')
    asyncio.run(validate(parser.parse_args()))
