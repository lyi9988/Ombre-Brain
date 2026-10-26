"""Approved bounded inner-state reward through real source and selection paths."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from memory_inner_affinity import (DEFAULT_WEIGHTS, DEFAULT_POLICY, build_inner_context,
    statement_inner_affinity, document_inner_affinity, local_inner_candidates)
from memory_l1_schema import document_template, statement_template, unknown
from memory_recall_semantics import resolve_document
from memory_system_config import values_from_config, REQUEST_CONFIGURATION
from recall_signal_protocol import INNER_DIMENSIONS, snapshot_from_inner
from gateway import memory_use, GatewayService
from memory_score_evidence import score_evidence
from test_memory_recall_gateway_r2 import (_authority_fixture, _natural_candidate,
    _gateway_service, _install_source_http, _base_config, _recall_input, _commit_l1)

NOW = datetime(2026,10,26,12,tzinfo=timezone.utc)
STAMP = NOW.isoformat()

def snapshot(**values):
    source = dict(schema="inner-state-v1",subject_id="guyan",checkpoint_id="cp-test",revision={"aiz":1,"brain":2},
        rules_version="inner-r1",settled_at=STAMP,availability="complete",source_refs=[],recent_changes=[],feelings=[],
        dimensions={key:dict(value=values.get(key),unit="0..1",owner=owner,kind=kind,
            status="known" if key in values else "unknown",reason=None if key in values else "not_observed")
            for key,(owner,kind) in INNER_DIMENSIONS.items()})
    return source

def document(*, content="我想和你聊聊亲密与靠近，也记得受伤后的解释。", person="user", namespace="snapshot.aiz.canonical_event"):
    doc = document_template(content,"synthetic")
    doc["entities"] = [dict(entity_id="who",entity_type=unknown("known",person),name="主人" if person=="user" else "小周",aliases=[])]
    statement = statement_template()
    statement.update(statement_id="s1",content=content,subject_ref=unknown("known","who"),evidence_refs=["source-1"])
    statement["classification"].update(kind=unknown("known","preference"),stability=unknown("known","stable"),
        modality=unknown("known","assertion"),basis=unknown("known","record"))
    statement["conditions"] = dict(state="not_applicable",text=None,predicate=None,exceptions=[],evidence_refs=[])
    statement["time"]["applicability"].update(temporal_scope="time_range",
        start=unknown("known","2020-01-01"),end=unknown("unbounded",evidence_refs=["source-1"]))
    doc["statements"] = [statement]
    doc["evidence_bindings"] = [dict(binding_id="source-1",source_namespace=namespace,
        source_id=json.dumps(dict(event_id="cause-1",conversation_id="conv-1")),source_revision="v1",
        author_ref=None,speaker_ref=None,anchors=[{"synthetic":True}])]
    return doc

def context(settings=None, source=None, query="陪我聊聊亲密与靠近"):
    return build_inner_context(settings or values_from_config({}),[snapshot_from_inner(source or snapshot(libido=.8))],
        NOW,query,scope={"conversation_id":"conv-1"})

def usage(doc):
    uses = resolve_document(doc,{"now":NOW,"query":"陪我聊聊亲密与靠近","purpose":"current","facts":{}},
        [],source_states={"source-1":"available"})
    return {"items":uses,"source_states":{"source-1":"available"}}

def component(doc, ctx, *, selection=None, use=None):
    item = (use or usage(doc))["items"][0]
    return statement_inner_affinity(doc["statements"][0],ctx,document=doc,usage=item,
        selection=selection or {"direct_ids":["s1"]},source_states={"source-1":"available"})

def test_exact_approved_defaults_all35_and_21_active():
    assert DEFAULT_POLICY == {"enabled":True,"max_bonus":.05,"weights":{}}
    assert set(DEFAULT_WEIGHTS)==set(INNER_DIMENSIONS)
    assert sum(value==1 for value in DEFAULT_WEIGHTS.values())==21
    assert DEFAULT_WEIGHTS["rest"]==DEFAULT_WEIGHTS["energy"]==DEFAULT_WEIGHTS["fatigue"]==0

def test_known_unknown_medium_libido_and_max_without_accumulation():
    doc = document()
    row = component(doc,context(source=snapshot(libido=.8,grieve=.5,longing=.6)))
    assert row["bonus"] == pytest.approx(.04)
    assert row["bonus"] != pytest.approx((.8+.5+.6)*.05)
    assert any(c["dimension"]=="tenderness" and c["contribution"] is None for c in row["contributions"])
    medium = component(doc,context(source=snapshot(libido=.5)))
    assert medium["bonus"] == pytest.approx(.025)

def test_security_inverse_is_read_only_and_known_zero_stays_known():
    doc = document(content="我希望和你聊聊安全感与安心。")
    src = snapshot(security=.2); original=deepcopy(src)
    assert component(doc,context(source=src))["bonus"] == pytest.approx(.04)
    assert src == original
    row = component(document(),context(source=snapshot(libido=0)))
    libido = next(c for c in row["contributions"] if c["dimension"]=="libido")
    assert libido["state"]=="known" and libido["intensity"]==0 and row["bonus"]==0

@pytest.mark.parametrize("query",["我现在住在哪里？","日期和地点是什么？","这件事有没有发生？","你几点来？我住哪里？"])
def test_fact_queries_never_receive_state_reward(query):
    row = component(document(),context(query=query))
    assert row["bonus"]==0 and row["reason"]=="fact_query"

@pytest.mark.parametrize("variant",["other_person","browse","topic_only","no_source","condition_false","missing_support"])
def test_person_content_source_conditions_and_support_are_real_requirements(variant):
    doc=document(); use=None; selection=None
    if variant=="other_person": doc=document(person="third_party")
    if variant=="browse": doc=document(namespace="snapshot.aiz.browse.reading")
    if variant=="topic_only":
        doc=document(content="客观记录一篇文章的出版编号。")
        doc["statements"][0]["classification"]["topics"]=["intimacy"]
    if variant=="no_source": doc["statements"][0]["evidence_refs"]=[]
    if variant=="condition_false":
        use=usage(doc); use["items"][0]["condition_match"]=False
    if variant=="missing_support": selection={"direct_ids":["s1"],"missing_support_ids":["s2"]}
    assert component(doc,context(),selection=selection,use=use)["bonus"]==0


def test_canonical_third_party_about_reference_is_not_owner_intimacy():
    doc=document(content="我和你聊过小周与朋友的亲密关系。")
    doc["entities"].append(dict(entity_id="other",entity_type=unknown("known","third_party"),
                               name="小周",aliases=[]))
    doc["statements"][0]["cognition"] = {"role":unknown("known","observation"),
        "epistemic_status":unknown("known","supported"),"about_refs":["other"],"open_questions":[]}
    row=component(doc,context())
    assert row["bonus"]==0
    assert any(c["reason"]=="current_person_unproven" for c in row["contributions"])


def test_real_l1_unspecified_condition_can_be_recalled_without_becoming_current_fact():
    doc=document()
    doc["statements"][0]["conditions"].update(state="not_stated")
    use=usage(doc)
    assert use["items"][0]["condition_match"] is None
    assert use["items"][0]["usage"] == "relationship_reference"
    before=deepcopy(use)
    assert component(doc,context(),use=use)["bonus"]==pytest.approx(.04)
    assert use==before  # Do not turn unknown into a met condition/current fact.
    doc["statements"][0]["conditions"].update(state="known",text="只有她有空时")
    assert component(doc,context())["bonus"]==0
    doc["statements"][0]["conditions"].update(state="not_stated",text=None)
    use=usage(doc);use["items"][0].update(usage="historical_evidence",reason_codes=["superseded_for_current_use"])
    assert component(doc,context(),use=use)["bonus"]==0

def test_stale_missing_and_owner_formula_do_not_double_consume_state():
    doc=document(); source=snapshot(libido=.8); source["availability"]="stale"
    assert component(doc,context(source=source))["bonus"]==0
    ctx=build_inner_context(values_from_config({}),[],NOW,"陪我聊聊亲密")
    assert component(doc,ctx)["bonus"]==0
    settings=values_from_config({})
    settings["recall"]["signal_bindings"]=[dict(id="desire",producer="guyan.inner_state",key="libido")]
    settings["features"].append(dict(id="alias",source="runtime",input_key="signal_desire"))
    settings["formulas"].append(dict(id="owner",expression="relevance + alias*.01",weights={}))
    result=document_inner_affinity(doc,context(settings),usage=usage(doc),selection={"direct_ids":["s1"]},
        calculation={"formula_id":"owner","statements":[]},settings=settings)
    assert result["bonus"]==0 and result["reason"]=="owner_formula_already_consumes_state"

def test_causal_expansion_only_uses_bound_current_event_and_spare_slots():
    doc=document(); source=snapshot(libido=.8)
    event=dict(kind="conversation",event_id="cause-1",occurred_at=STAMP)
    source["recent_changes"]=[dict(key="libido",occurred_at=STAMP,source_refs=[event])]
    settings=values_from_config({})
    records={"b":dict(state="l1",document=doc)}
    buckets={"b":dict(id="b",content=doc["statements"][0]["content"])}
    def candidates(src,limit=1):
        return local_inner_candidates(records,buckets,context(settings,src,query="陪我随便聊聊"),settings,set(),limit=limit)
    assert len(candidates(source))==1
    assert candidates(source,limit=0)==[]
    unbound=deepcopy(source);unbound["recent_changes"]=[];unbound["source_refs"]=[event]
    assert candidates(unbound)==[]
    clock=deepcopy(source);clock["recent_changes"][0]["source_refs"][0]["kind"]="clock"
    assert candidates(clock)==[]
    other=deepcopy(source);other["recent_changes"][0]["source_refs"][0]["event_id"]="different"
    assert candidates(other)==[]

@pytest.mark.parametrize("policy",[{"enabled":False},{"max_bonus":0},{"weights":{"libido":0}}])
def test_off_zero_maintains_component_and_expansion(policy):
    settings=values_from_config({}); settings["recall"]["inner_state_affinity"].update(policy)
    assert component(document(),context(settings))["bonus"]==0

def _run_real(tmp_path,monkeypatch,*,policy=None,query="陪我聊聊亲密与靠近",source_state="available",threshold=.5):
    rows=[("b-neutral","普通合成资料。"),("b-personal","亲密与靠近的合成经历。")]
    authority,view,records,buckets=_authority_fixture(tmp_path,[])
    for bid,body in rows:
        _commit_l1(authority,bid,body,current=("陪我聊聊阅读理解与研究方法。" if bid=="b-neutral"
            else "我想和你聊聊亲密与靠近。"),profile_kind="preference",profile_stability="stable")
        buckets.append(dict(id=bid,content=body,metadata={"name":bid}))
    candidates=[_natural_candidate(buckets[0],score=.62,keyword_score=.62),
                _natural_candidate(buckets[1],score=.60,keyword_score=.60)]
    requests=_install_source_http(monkeypatch,state=source_state)
    config=_base_config(local_threshold=threshold)
    config["recall"]["inner_state_affinity"].update(policy or {})
    config["limits"]["inject_max_cards"]=1
    service,calls=_gateway_service(tmp_path,monkeypatch,view,buckets,candidates,config)
    service.inject_max_cards=1
    # The reused fixture replaces the picker with identity for its older tests.
    # Restore the real final picker for this source -> rank -> choose contract.
    service._pick_dynamic_cards=GatewayService._pick_dynamic_cards.__get__(service)
    request=_recall_input(query);request["_signal_snapshots"]=[snapshot_from_inner(snapshot(libido=.8))]
    request["_signal_sampled_at"]=NOW
    selected,suppressed,debug=asyncio.run(service._finish_natural_selection(query,"test-inner-state",buckets,recall_input=request))
    return selected,suppressed,debug,service,request,requests,calls

def test_real_source_scoring_and_final_selection_component_once(tmp_path,monkeypatch):
    selected,suppressed,debug,service,request,requests,calls=_run_real(tmp_path,monkeypatch)
    assert requests and calls["rerank"]==[]
    scores=debug["configured_computation"]["candidate_scores"]
    assert scores
    for row in scores:
        calculation=row["computation"]; affinity=calculation["inner_state_affinity"]
        assert calculation["score"] == pytest.approx(row["keyword_score"])
        bonus = .04 if row["bucket_id"]=="b-personal" else 0
        assert affinity["applied_bonus"] == pytest.approx(bonus)
        assert row["final_score"] == pytest.approx(row["keyword_score"]+bonus)
    assert [row["id"] for row in selected]==["b-personal"]
    evidence=score_evidence({"values":service.config["memory_system"],"revision":"local-r1"},debug["configured_computation"])
    assert evidence["state"]=="recorded"
    assert evidence["configuration"]["values"]["recall"]["inner_state_affinity"]["max_bonus"]==.05
    target=next(row for row in evidence["candidates"] if row["bucket_id"]=="b-personal")
    assert target["computation"]["inner_state_affinity"]["applied_bonus"]==pytest.approx(.04)
    assert "亲密" not in json.dumps(evidence,ensure_ascii=False)
    # Repeating only final adoption is idempotent; formulas and relevance stay put.
    items=[dict(bucket={"id":"x"},score=.60,_configured_score={"score":.60,
        "inner_state_affinity":{"state":"computed","bonus":.04}})]
    memory_use.apply_inner_ranking(service,request,request["q_context"],items)
    memory_use.apply_inner_ranking(service,request,request["q_context"],items)
    assert items[0]["score"]==pytest.approx(.64)

def test_real_reward_cannot_lift_local_admission_or_original_card_confidence(tmp_path,monkeypatch):
    selected,suppressed,debug,*_=_run_real(tmp_path,monkeypatch,threshold=.63)
    assert selected==[]
    assert all(row["admitted"] is False for row in debug["configured_computation"]["candidate_scores"])

def test_real_withdrawn_source_has_no_component_or_final_memory(tmp_path,monkeypatch):
    selected,suppressed,debug,*_=_run_real(tmp_path,monkeypatch,source_state="withdrawn")
    assert selected==[]
    assert debug["configured_computation"]["candidate_scores"]==[]



@pytest.mark.parametrize("query", ["Where do I live?", "When did I arrive?", "Who was there?", "How many people attended?"])
def test_actual_english_fact_boundaries(query):
    ctx=context(query=query)
    assert ctx["query_mode"]=="fact"
    assert component(document(),ctx)["bonus"]==0


def test_actual_english_open_use_and_current_person():
    doc=document(content="I want intimacy and to be closer to you.")
    ctx=context(query="I want to talk about intimacy with you")
    assert ctx["query_mode"]=="open"
    assert component(doc,ctx)["bonus"]==pytest.approx(.04)


def test_current_causal_source_is_independent_of_open_word_detection():
    doc=document(); src=snapshot(libido=.8)
    event=dict(kind="conversation",event_id="cause-1",occurred_at=STAMP)
    src["recent_changes"]=[dict(key="libido",occurred_at=STAMP,source_refs=[event])]
    ctx=context(source=src,query="嗯")
    assert ctx["query_mode"]=="unproven"
    assert component(doc,ctx,selection={"direct_ids":[]})["bonus"]==pytest.approx(.04)
    settings=values_from_config({})
    candidates=local_inner_candidates({"b":dict(state="l1",document=doc)},
        {"b":dict(id="b")},ctx,settings,set(),limit=1)
    assert len(candidates)==1
    assert candidates[0]["_inner_candidate"]["matches"][0]["reason"]=="current_event_source"
    assert component(doc,context(source=src,query="我住哪里？"))["bonus"]==0


@pytest.mark.parametrize("policy",[{"enabled":False},{"max_bonus":0},{"weights":{"libido":0}}])
def test_real_off_and_zero_keep_original_choice(tmp_path,monkeypatch,policy):
    selected,_,debug,*_=_run_real(tmp_path,monkeypatch,policy=policy)
    assert [row["id"] for row in selected]==["b-neutral"]
    for row in debug["configured_computation"]["candidate_scores"]:
        assert row["final_score"]==pytest.approx(row["keyword_score"])


def test_final_card_confidence_uses_original_score(tmp_path,monkeypatch):
    _,_,_,service,request,*_=_run_real(tmp_path,monkeypatch)
    items=[dict(bucket={"id":"below"},score=.53,_configured_score={"score":.53,"window_score":.53,
        "inner_state_affinity":{"state":"computed","bonus":.04}})]
    memory_use.apply_inner_ranking(service,request,request["q_context"],items)
    assert items[0]["score"]==.53
    assert service._pick_dynamic_cards(items,query=request["q_context"])==[]
