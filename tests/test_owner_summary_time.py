import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from memory_l1_service import L1OwnerError
from memory_owner_summary_time import validate_time
from test_owner_summary_memory import setup, run

SYNTHETIC = '这是隔离验证的虚构资料，不是真实主人谈话。小禾说：以后给我推荐饮料时，别选明显苦的；我还没决定周三是否去看展。'
ORIGINAL = Path(__file__).resolve().parents[3] / 'staging/stage08-summary-memory-20261105/real-1plus1/receipts-real3/auto_l1-synthetic-output.json'


def document(content='小禾还没决定周三是否去看展。',time=None):
    return {'format':'semantic-v1','entities':[], 'narrative':{'title':'看展未决','body':content},
        'retrieval':{'keywords':['小禾','看展']}, 'statements':[{'content':content,'subject_ref':None,
            'classification':{'kind':None,'stability':None,'modality':None,'basis':None},
            'time':time,'conditions':None,'evidence_refs':['source:1']}]}


def test_original_saved_provider_documents_are_rejected_without_mutation():
    if not ORIGINAL.is_file(): pytest.skip('private synthetic provider receipt is not bundled in source')
    original = json.loads(json.loads(ORIGINAL.read_text(encoding='utf-8'))['generated_output'])
    before = copy.deepcopy(original)
    issues=[]
    for candidate in original['candidates']:
        with pytest.raises(L1OwnerError) as caught:
            validate_time(candidate['document'],{'source:1':SYNTHETIC})
        issues.append((caught.value.code,caught.value.field_path))
    assert issues[0][0] == issues[1][0] == 'owner_summary_time_not_supported'
    assert '.time.event.start' in issues[0][1] and issues[1][1].endswith('.content')
    assert original == before


def test_actual_saved_completion_fails_before_any_review_candidate(tmp_path,monkeypatch):
    if not ORIGINAL.is_file(): pytest.skip('private synthetic provider receipt is not bundled in source')
    raw=json.loads(ORIGINAL.read_text(encoding='utf-8'))['generated_output']
    f=setup(tmp_path,monkeypatch,body=SYNTHETIC)
    async def replay(client,**kwargs):
        assert kwargs['task_stage']=='auto_l1'
        return NS(choices=[NS(message=NS(content=raw),finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion=replay
    receipt=run(f)['results'][0]
    assert receipt['status']=='failed' and receipt['candidate_ids']==[]
    assert receipt['issue']['code']=='owner_summary_time_not_supported'
    row=f.tasks.get(receipt['task_id'])
    assert row['windows'][0]['output']['text']==raw
    assert not f.env['authority'].list_memories()


@pytest.mark.parametrize('source,value,precision',[
    ('计划在2026年11月4日去看展。','2026-11-04','day'),
    ('计划在2026-11-04T08:30:45+08:00去看展。','2026-11-04T08:30:45+08:00','instant'),
])
def test_explicit_original_date_and_exact_clock_are_valid(source,value,precision):
    doc=document(source,{'event':{'start':value,'precision':precision,'interpretation':'instant'},'applicability':None})
    validate_time(doc,{'source:1':source})


def test_date_only_source_cannot_gain_clock_seconds_or_timezone():
    doc=document('计划去看展。',{'event':{'start':'2026-11-04T08:30:00+08:00','precision':'instant','interpretation':'instant'},'applicability':None})
    with pytest.raises(L1OwnerError): validate_time(doc,{'source:1':'计划2026年11月4日去看展。'})
    with pytest.raises(L1OwnerError): validate_time(doc,{'source:1':'计划2026年11月4日08:30去看展。'})


def test_statement_cannot_borrow_date_from_an_unreferenced_neighbor():
    doc=document('计划在2026年11月4日去看展。')
    with pytest.raises(L1OwnerError) as caught:
        validate_time(doc,{'source:1':'还没决定周三是否去看展。','source:2':'另一件事发生在2026年11月4日。'})
    assert caught.value.field_path=='document.statements[0].content'


def test_english_date_precision_and_offset_are_source_local():
    with pytest.raises(L1OwnerError):
        validate_time(document('The visit is November 4, 2026.'), {'source:1': 'Maybe Wednesday.'})
    validate_time(document('On November 4, 2026 at 08:30.'),
                  {'source:1': 'The visit is on 4 November 2026 at 08:30:15.'})
    with pytest.raises(L1OwnerError):
        validate_time(document('The visit is at 08:30Z.'), {'source:1': 'The visit is at 08:30 UTC+8.'})


def test_relative_wednesday_remains_usable_and_record_time_stays_in_l0(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch,body=SYNTHETIC)
    async def revised(client,**kwargs):
        payload=json.loads(kwargs['messages'][1]['content'])
        assert all('source_recorded_at' not in s for s in payload['l1_contract']['source_texts'])
        assert 'input_kind为owner_selected_summary' in kwargs['messages'][0]['content']
        return NS(choices=[NS(message=NS(content=json.dumps({'candidates':[{'document':document(),
            'kind':'key_event','confidence':0.9,'source_refs':['source:1'],'reason':'原话保留未决'}]})),finish_reason='stop')])
    f.engine._daily_chat_memory_create_completion=revised
    receipt=run(f)['results'][0]
    assert receipt['status']=='review',receipt
    row=f.env['authority'].get_candidate(receipt['candidate_ids'][0])
    binding=row['proposal']['metadata']['memory_l1']['evidence_bindings'][0]
    assert f.readers.copy(binding)['recorded_at']==f.envelope['context']['sources'][0]['created_at']
    assert row['proposal']['proposed_body']=='小禾还没决定周三是否去看展。'


def test_owner_edited_unsupported_date_is_rechecked_before_commit(tmp_path,monkeypatch):
    from memory_l1_schema import join_document
    f=setup(tmp_path,monkeypatch)
    receipt=run(f)['results'][0]
    row=f.env['authority'].get_candidate(receipt['candidate_ids'][0])
    edited=join_document(row['proposal']['proposed_body'],row['proposal']['metadata']['memory_l1'])
    edited['narrative']['body']='小禾在2026年11月4日已决定去看展。'
    with pytest.raises(L1OwnerError) as caught:
        asyncio.run(f.engine._memory_l1_adopt_candidate(row,f.env['buckets'],owner=True,
            edit={'document':edited},request_id='unsupported-owner-date'))
    assert caught.value.code=='owner_summary_time_not_supported'
    assert not f.env['authority'].list_memories()
