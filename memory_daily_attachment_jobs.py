"""Independent attachment preparation on the existing daily scheduler/queue.

Never advances raw-chat cursors, creates owner turns, or treats unverified
provider receipts as adopted files. Oversize projections remain pending.
"""
import math
from memory_l1_contract import semantic_sha256
from memory_l1_schema import split_document
from memory_l1_service import L1OwnerError
from memory_daily_attachments import collect_daily_attachments
from memory_attachment_source import semantic_text


async def run_attachment_jobs(engine, bucket_mgr, *, day, mode, options):
    from memory_candidate_preparation import MemoryCandidatePreparation
    from memory_authority import MemoryProposal
    from memory_l1_api import AUTO_KINDS
    if mode == 'off' or engine._daily_chat_memory_candidate_limit(mode) <= 0:
        return {'status': 'disabled', 'results': []}
    sources,issues=await collect_daily_attachments(day,timezone=str(engine.tz),config=engine.config,
        profile_id=engine.memory_semantic_profile_id)
    queue=MemoryCandidatePreparation(engine.memory_authority_store)
    results=[]; calls=0; budget=engine._daily_chat_memory_candidate_limit(mode)
    for source in sources:
        if getattr(engine,'_daily_chat_memory_cancel_requested',False):
            issues.append({'code':'cancelled'});break
        fingerprint=semantic_sha256({'source_key':source['source_key'],'profile':engine.memory_semantic_profile_id})
        key='attachment_l1:'+fingerprint
        receipt=queue.get(key)
        try:
            if not receipt or receipt.get('issue'):
                text=semantic_text(source)
                if len(text)>options['daily_chat_memory_window_max_input_chars']:
                    issue={'code':'attachment_window_pending','projected_chars':len(text),
                        'window_chars':options['daily_chat_memory_window_max_input_chars'],'coverage':'not_processed'}
                    queue.prepare(key,[],engine.memory_ingestion_policy,scope=key,source_fingerprint=fingerprint,
                        issue=issue,replace_failure=bool(receipt))
                    results.append({'key':key,'status':'pending','issue':issue});continue
                if calls>=options['daily_chat_memory_max_windows_per_run'] or budget<=0:
                    queue.prepare(key,[],engine.memory_ingestion_policy,scope=key,source_fingerprint=fingerprint,
                        issue={'code':'attachment_run_budget'},replace_failure=bool(receipt))
                    results.append({'key':key,'status':'pending','reason':'attachment_run_budget'});continue
                route=engine._memory_task_route('auto_l1')
                if not route.get('configured'):
                    raise L1OwnerError('attachment_route_unconfigured',field_path='generation')
                calls+=1  # Failed provider attempts also consume this run's window budget.
                candidates,stats=await engine._extract_window_candidates(day,[],attachment_sources=[source],max_candidates=budget)
                if stats.get('empty_output') or stats.get('parse_failure') or stats.get('schema_error') or stats.get('prompt_disabled'):
                    raise L1OwnerError('attachment_generation_incomplete',field_path='generation')
                if getattr(engine,'_daily_chat_memory_cancel_requested',False):
                    raise L1OwnerError('attachment_job_cancelled',field_path='generation')
                proposals=[]
                for item in candidates:
                    if item.get('_candidate_validation_issue') or not isinstance(item.get('document'),dict):
                        raise L1OwnerError('attachment_invalid_document',field_path='candidates')
                    if item.get('kind') not in AUTO_KINDS:
                        raise L1OwnerError('attachment_invalid_kind',field_path='candidates.kind')
                    confidence=item.get('confidence')
                    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
                        raise L1OwnerError('invalid_confidence',field_path='candidates.confidence')
                    for flag in ('sensitive','transient','generic'):
                        if type(item.get(flag,False)) is not bool:
                            raise L1OwnerError('boolean_required',field_path='candidates.'+flag)
                    ident='attachment_l1_'+semantic_sha256({'key':key,'document':item['document']})[:40]
                    doc=engine._memory_l1_daily_document(item,ident)
                    body,semantic=split_document(doc)
                    # Actual live bindings, not model-declared source verification.
                    for binding in doc['evidence_bindings']:
                        if not (await engine._memory_l1_sources().locate(binding)).get('eligible_source'):
                            raise L1OwnerError('source_not_currently_readable',field_path='evidence_bindings')
                    proposals.append(MemoryProposal.from_mapping({'proposal_id':ident,'source_type':'attachment_document',
                        'proposed_body':body,'source_refs':[source['source_key']],'source_status':'verified',
                        'memory_type':item['kind'],'confidence':confidence,'requested_mode':mode,'owner_explicit':False,
                        'sensitive':item.get('sensitive',False),'transient':item.get('transient',False),'generic':item.get('generic',False),
                        'metadata':{'memory_l1':semantic,'attachment_preparation_key':key,
                            'legacy_candidate':{'title':doc['narrative']['title'],'content':body,'proposed_memory':body}}}))
                # An empty, valid generation is a durable receipt, not a retry loop.
                receipt=queue.prepare(key,proposals,engine.memory_ingestion_policy,scope=key,source_fingerprint=fingerprint,
                    generated_count=len(candidates),replace_failure=bool(receipt))
            handled=[];adoption_issues=[]
            for row in queue.remaining(receipt)[:budget]:
                if getattr(engine,'_daily_chat_memory_cancel_requested',False):
                    adoption_issues.append({'code':'attachment_job_cancelled'});break
                budget-=1
                if row['status']=='accepted':
                    if mode!='auto' and row['proposal'].get('requested_mode')=='auto':continue
                    try:
                        adopted=await engine._memory_l1_adopt_candidate(row,bucket_mgr,request_id=key,owner=False)
                        if adopted['status'] in ('created','exists'):handled.append(row['candidate_id'])
                    except Exception as exc:
                        adoption_issues.append({'code':getattr(exc,'code',type(exc).__name__)})
                else:handled.append(row['candidate_id'])
            queue.acknowledge(key,handled)
            results.append({'key':key,'status':'blocked' if adoption_issues else 'prepared',
                'candidate_ids':receipt['candidate_ids'],'adoption_issues':adoption_issues})
        except Exception as exc:
            problem={'code':getattr(exc,'code',type(exc).__name__)}
            if not receipt or receipt.get('issue'):
                queue.prepare(key,[],engine.memory_ingestion_policy,scope=key,source_fingerprint=fingerprint,
                    issue=problem,replace_failure=bool(receipt))
            results.append({'key':key,'status':'blocked','issue':problem})
    # Pending receipt descriptors never become source text or model input.
    for issue in issues:
        if issue.get('code')=='adoption_receipt_pending':
            key='attachment_pending:'+semantic_sha256(issue)
            if not queue.get(key):
                queue.prepare(key,[],engine.memory_ingestion_policy,scope=key,source_fingerprint=key,issue=issue)
    return {'status':'partial' if issues or any(r['status'] in ('pending','blocked') for r in results) else 'completed',
        'results':results,'issues':issues,'model_calls':calls}
