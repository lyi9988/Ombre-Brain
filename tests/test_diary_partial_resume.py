import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from test_memory_l1_ingestion_p2c import setup
from memory_l1_runtime import MemoryL1Runtime
from memory_l1_ingestion import model_template
from memory_l1_schema import statement_template
from memory_l0_sources import MemoryL0Sources, McpDiarySourceAdapter
from memory_authority import MemoryIngestionPolicy, MemoryProposal, RevisionConflict
from memory_diary_progress import DiaryProgress

DAY = '2026-10-14'
def run(coro): return asyncio.run(coro)

def fixture(tmp_path, *, limit=1, length=600, mode='review'):
    service, _, _, _ = setup(tmp_path)
    state = {'text': 'a' * length, 'fail': None, 'during': None}
    async def read(day): return {'id':'diary-test', 'content':state['text']}
    sources = MemoryL0Sources(service.authority, [McpDiarySourceAdapter('fixture', read)])
    engine = MemoryL1Runtime()
    engine.config = {'reflection':{'diary_l1_input_window_chars':256,'diary_l1_max_windows':limit}}
    engine.memory_authority_store = service.authority
    engine.memory_ingestion_policy = MemoryIngestionPolicy()
    engine.daily_chat_memory_mode = mode
    engine.temperature = 0
    engine._memory_l1_sources = lambda: sources
    engine._memory_l1_revisions = lambda bucket_mgr=None: service
    engine._memory_l1_prompt = lambda _: ('fixture prompt', {'prompt_revision':'r1'})
    engine._memory_task_route = lambda *args: {'configured':True,'client':None,'model':'fixture','connection':'main'}
    engine._completion_finish_reason = lambda r:r.choices[0].finish_reason
    engine._completion_content = lambda r:r.choices[0].message.content
    engine._parse_json_object = json.loads
    calls=[]
    async def complete(*args, **kwargs):
        p=json.loads(kwargs['messages'][1]['content']); calls.append(p['window_index'])
        if state['during']: state['during'](); state['during']=None
        if p['window_index']==state['fail']:
            return NS(choices=[NS(message=NS(content='invalid fixture'),finish_reason='stop')])
        draft=model_template(catalog=service.catalog_provider()[0])
        draft['narrative']={'title':'diary', 'body':'window '+str(p['window_index'])}
        statement=statement_template();statement.pop('statement_revision_id')
        statement.update(content='fixture experience',evidence_refs=['diary-source'])
        draft['statements']=[statement]
        draft['scenes']=[{'scene_id':'new:scene','title':'scene','source_refs':['diary-source'],
            'statement_refs':['new:statement'],'attributes':[]}]
        value={'document':draft,'scene_bodies':{'new:scene':'scene body'},'confidence':0.9}
        return NS(choices=[NS(message=NS(content=json.dumps(value)),finish_reason='stop')])
    engine._daily_chat_memory_create_completion=complete
    return engine,service,state,calls

def task(engine): return run(engine._memory_l1_diary_task(DAY,None))

def test_budget_resume_durable_no_duplicate_scenes(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    a=task(e); assert a['coverage']['state']=='partial'
    b=task(e); assert b['coverage']['state']=='partial'
    c=task(e); assert c['coverage']['state']=='complete'
    assert calls==[0,1,2]
    row=s.authority.get_candidate(a['candidate_id'])
    scenes=row['proposal']['metadata']['memory_l1']['scenes']
    assert len(scenes)==len({x['scene_id'] for x in scenes})==3
    assert row['revision']==3
    assert task(e)['replayed'] and calls==[0,1,2]

def test_valid_result_checkpoint_failure_does_not_allow_paid_replay(tmp_path, monkeypatch):
    e,s,state,calls=fixture(tmp_path)
    original=DiaryProgress.save
    def fail_validated(self, ident, progress, **kwargs):
        if progress.get('input_ranges') and 'uncertain_window' not in progress:
            raise RuntimeError('checkpoint write failed after valid model result')
        return original(self, ident, progress, **kwargs)
    monkeypatch.setattr(DiaryProgress, 'save', fail_validated)
    assert task(e)['reason']=='validated_checkpoint_requires_recovery'
    monkeypatch.setattr(DiaryProgress, 'save', original)
    assert task(e)['reason']=='model_outcome_unknown_requires_retry_decision'
    assert calls==[0]

def test_failure_second_resumes_only_missing(tmp_path):
    e,s,state,calls=fixture(tmp_path,limit=3)
    state['fail']=1
    a=task(e); assert a['coverage']['input_ranges']==[{'start':0,'end':256}]
    state['fail']=None
    b=task(e); assert b['coverage']['state']=='complete'
    assert calls==[0,1,1,2]

def test_restart_and_source_change(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    a=task(e)
    # A new engine/Authority connection reopens the same durable DB.
    e2,s2,state2,calls2=fixture(tmp_path)
    b=task(e2); assert calls2==[1] and b['candidate_id']==a['candidate_id']
    state2['text']='b'*600
    c=task(e2); assert calls2==[1,0] and c['candidate_id']!=a['candidate_id']

def test_checkpoint_before_candidate_failure_survives_restart(tmp_path,monkeypatch):
    from memory_diary_l1 import DiaryL1Service
    original=DiaryL1Service.prepare
    async def fail(*a,**kw): raise RuntimeError('simulate write failure')
    monkeypatch.setattr(DiaryL1Service,'prepare',fail)
    e,s,state,calls=fixture(tmp_path)
    assert task(e)['reason']=='checkpoint_saved_candidate_not_updated'
    monkeypatch.setattr(DiaryL1Service,'prepare',original)
    e2,s2,state2,calls2=fixture(tmp_path)
    assert task(e2)['coverage']['state']=='partial' and calls2==[1]

def test_auto_all_in_one_run_stays_auto(tmp_path):
    e,s,state,calls=fixture(tmp_path,limit=3,mode='auto')
    a=task(e)
    assert a['coverage']['state']=='complete'
    assert s.authority.get_candidate(a['candidate_id'])['status'] in ('accepted','committed')
    assert len(s.authority.list_memories())>0

def test_partial_auto_stays_review_on_resume(tmp_path):
    e,s,state,calls=fixture(tmp_path,mode='auto')
    a=task(e); task(e); c=task(e)
    assert c['coverage']['state']=='complete'
    assert s.authority.get_candidate(a['candidate_id'])['status']=='pending'
    assert not s.authority.list_memories()

def test_accepted_candidate_never_resumes(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    a=task(e); row=s.authority.get_candidate(a['candidate_id'])
    s.authority.decide_candidate(row['candidate_id'], action='accept',expected_revision=row['revision'],
        request_id='owner-accept',actor='owner')
    task(e)
    assert calls==[0]

def owner_edit(s,cid):
    row=s.authority.get_candidate(cid)
    proposal=MemoryProposal(**{**row['proposal'],'proposed_body':'owner text'})
    s.authority.revise_candidate(cid,expected_revision=row['revision'],proposal=proposal,
        request_id='owner-edit',actor='owner')

def test_owner_edit_before_run_is_not_overwritten(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    a=task(e);owner_edit(s,a['candidate_id'])
    assert task(e)['reason']=='candidate_changed_after_checkpoint' and calls==[0]

def test_owner_edit_during_model_await_is_not_overwritten(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    a=task(e);state['during']=lambda:owner_edit(s,a['candidate_id'])
    b=task(e)
    assert b['reason']=='checkpoint_saved_candidate_not_updated'
    assert s.authority.get_candidate(a['candidate_id'])['proposal']['proposed_body']=='owner text'

def test_legacy_partial_requires_decision(tmp_path):
    e,s,state,calls=fixture(tmp_path);a=task(e)
    with s.authority._connect() as db:
        row=s.authority.get_candidate(a['candidate_id']);p=row['proposal'];p['metadata'].pop('diary_l1_checkpoint')
        db.execute('UPDATE candidates SET proposal_json=? WHERE candidate_id=?',(json.dumps(p),a['candidate_id']))
    assert task(e)['reason']=='legacy_partial_requires_rebuild_decision' and calls==[0]

def test_task_store_cas_and_rollback(tmp_path):
    e,s,_,_=fixture(tmp_path)
    store=DiaryProgress(s.authority);store.save('x',{'v':1},expected_revision=0)
    with pytest.raises(RevisionConflict): store.save('x',{'v':2},expected_revision=0)
    assert store.get('x')=={'revision':1,'progress':{'v':1}}

def test_same_engine_overlap_no_second_model_call(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    original=e._daily_chat_memory_create_completion
    async def scenario():
        entered=asyncio.Event();release=asyncio.Event()
        async def wait(*a,**kw):
            entered.set();await release.wait();return await original(*a,**kw)
        e._daily_chat_memory_create_completion=wait
        first=asyncio.create_task(e._memory_l1_diary_task(DAY,None));await entered.wait()
        second=await e._memory_l1_diary_task(DAY,None)
        assert second['status']=='in_progress'
        release.set();await first
    run(scenario());assert calls==[0]


def test_timeout_marks_unknown_no_automatic_second_call(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    async def timeout(*a,**kw): calls.append('timeout'); raise TimeoutError('unknown provider outcome')
    e._daily_chat_memory_create_completion=timeout
    assert task(e)['status']=='failed'
    assert task(e)['reason']=='model_outcome_unknown_requires_retry_decision'
    assert calls==['timeout']


def test_cancelled_run_keeps_durable_unknown_intent(tmp_path):
    e,s,state,calls=fixture(tmp_path)
    async def cancelled(*a,**kw): calls.append('cancel'); raise asyncio.CancelledError()
    e._daily_chat_memory_create_completion=cancelled
    with pytest.raises(asyncio.CancelledError): task(e)
    assert task(e)['reason']=='model_outcome_unknown_requires_retry_decision'
    assert calls==['cancel']


def test_complete_checkpoint_compacts_generated_payload(tmp_path):
    e,s,state,calls=fixture(tmp_path,limit=3)
    a=task(e)
    progress=DiaryProgress(s.authority).get(a['candidate_id'])['progress']
    assert set(progress)=={'completed_candidate_id','candidate_revision'}
    assert task(e)['replayed'] and calls==[0,1,2]

@pytest.mark.parametrize('change', ['prompt','options'])
def test_generation_change_starts_new_task(tmp_path,change):
    e,s,state,calls=fixture(tmp_path);a=task(e)
    if change=='prompt': e._memory_l1_prompt=lambda _:('new prompt',{'prompt_revision':'r2'})
    else: e.config['reflection']['diary_l1_max_tokens']=17000
    b=task(e)
    assert a['candidate_id']!=b['candidate_id'] and calls==[0,0]


def test_accept_during_await_is_not_revised(tmp_path):
    e,s,state,calls=fixture(tmp_path);a=task(e)
    def accept():
        row=s.authority.get_candidate(a['candidate_id'])
        s.authority.decide_candidate(row['candidate_id'],action='accept',expected_revision=row['revision'],
            request_id='during-accept',actor='owner')
    state['during']=accept
    b=task(e)
    row=s.authority.get_candidate(a['candidate_id'])
    assert row['status']=='accepted'
    assert row['proposal']['metadata']['diary_l1_bundle']['coverage']['state']=='partial'
    assert b['reason']=='checkpoint_saved_candidate_not_updated'
