"""Behavioral contracts for query-local claim evidence and scoring scope."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_l1_schema import document_template, statement_template, unknown
from memory_query_evidence import correction_matches, match_statements, query_purpose
from memory_recall_semantics import resolve_document, score_document
from memory_system_config import ConfigurationError, validate_values, values_from_config


NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)


def _known(value):
    return unknown("known", value)


def _claim(statement_id, content, *, kind="fact", stability="phase",
           event_start="2026-10-19T00:00:00+00:00", evidence_refs=()):
    statement = statement_template()
    statement.update(
        statement_id=statement_id,
        statement_revision_id="revision:" + statement_id,
        content=content,
        evidence_refs=list(evidence_refs),
    )
    statement["classification"].update(
        kind=_known(kind),
        stability=_known(stability),
        modality=_known("assertion"),
        basis=_known("record"),
    )
    statement["time"]["event"].update(
        start=_known(event_start),
        end=unknown("not_stated"),
    )
    return statement


def _document(*claims, scenes=(), relations=()):
    document = document_template("Synthetic L1 body")
    document["statements"] = list(claims)
    document["scenes"] = list(scenes)
    document["relations"] = list(relations)
    return document


def _context(query, *, purpose="current", target_time=None):
    return {
        "query": query,
        "purpose": purpose,
        "now": NOW,
        "target_time": target_time,
        "facts": {},
    }


def test_query_purpose_separates_past_current_mixed_and_relationship_intent():
    past = query_purpose("那天我提过胃不舒服，后来怎么样？", now=NOW)
    assert past["purpose"] == "history"
    assert past["temporal_reference"] == "unresolved"
    assert past["target_time"] is None

    conflict = query_purpose("我们刚吵架，你现在还生气吗？", now=NOW)
    assert conflict["purpose"] == "current"
    assert conflict["purpose_reason"] == "explicit_present_reference"

    comparison = query_purpose("你以前觉得逗我笑就够了，现在怎么想？", now=NOW)
    assert comparison["purpose"] == "mixed"
    assert comparison["temporal_reference"] == "unresolved"

    companionship = query_purpose("如果我不开心又不想说话，你应该怎么陪我？", now=NOW)
    assert companionship["purpose"] == "relationship"
    assert companionship["purpose_reason"] == "conditional_interaction_query"


def test_matching_keeps_only_necessary_scene_and_relation_context_as_support():
    direct = _claim("st-date", "苏州出行日期改到11月3日。")
    scene_support = _claim("st-ticket", "机票尚未购买，等排班有结果。")
    relation_support = _claim("st-shift", "11月1日临时加班，原计划需要变化。")
    document = _document(
        direct,
        scene_support,
        relation_support,
        scenes=[{"scene_id": "scene-trip", "statement_refs": ["st-date", "st-ticket"]}],
        relations=[{
            "kind": "revises",
            "confirmation_state": "confirmed",
            "origin": "owner",
            "from": {"memory_id": None, "revision": "this_revision",
                     "object_type": "statement", "object_id": "st-date"},
            "to": {"memory_id": None, "revision": "this_revision",
                   "object_type": "statement", "object_id": "st-shift"},
        }],
    )
    selection = match_statements(document, _context("苏州出行日期是什么？"))

    assert selection["direct_ids"] == ["st-date"]
    assert set(selection["support_ids"]) == {"st-ticket", "st-shift"}
    assert selection["support_reasons"] == {
        "st-ticket": "declared_scene_context",
        "st-shift": "declared_revises",
    }

    scored = score_document(
        document,
        values_from_config({}),
        _context("苏州出行日期是什么？"),
        {"relevance": 0.72, "statement_selection": selection,
         "eligible_statement_ids": {"st-date", "st-ticket", "st-shift"}},
    )
    assert scored["aggregation_statement_ids"] == ["st-date"]
    rows = {row["statement_id"]: row for row in scored["statements"]}
    assert rows["st-date"]["participates_in_aggregation"] is True
    assert rows["st-ticket"]["match_kind"] == "support"
    assert rows["st-shift"]["match_kind"] == "support"
    assert rows["st-ticket"]["participates_in_aggregation"] is False
    assert rows["st-shift"]["participates_in_aggregation"] is False
    assert scored["score"] == pytest.approx(rows["st-date"]["score"])


def test_unverified_or_excluded_claims_cannot_enter_the_scoring_pool():
    claim = _claim("st-source", "苏州日期改到11月3日。", evidence_refs=["source:trip"])
    document = _document(claim)
    context = _context("苏州日期改到哪天？")
    selection = match_statements(document, context)
    assert selection["direct_ids"] == ["st-source"]

    unresolved = resolve_document(
        document, context, source_states={"source:trip": "unverified"}
    )[0]
    assert unresolved["usage"] == "unresolved_evidence"
    assert unresolved["source_state"] == "unverified"
    excluded = resolve_document(
        document, context, source_states={"source:trip": "unavailable"}
    )[0]
    assert excluded["usage"] == "excluded"

    for use in (unresolved, excluded):
        eligible_ids = ({use["statement_id"]}
                        if use["source_state"] == "available" and use["usage"] != "excluded"
                        else set())
        scored = score_document(
            document, values_from_config({}), context,
            {"relevance": 0.8, "statement_selection": selection,
             "eligible_statement_ids": eligible_ids},
        )
        assert scored["aggregation_statement_ids"] == []
        assert scored["state"] == "unassigned_semantic_evidence"
        assert scored["statements"][0]["participates_in_aggregation"] is False


def test_unrelated_stable_coffee_preference_cannot_raise_old_state_score():
    query = "那次胃疼后来怎么样？"
    old_state = _claim(
        "st-stomach", "那次胃疼持续了两天。", kind="state", stability="phase",
        event_start="2025-10-19T00:00:00+00:00",
    )
    coffee = _claim(
        "st-coffee", "我仍然喜欢咖啡的味道。", kind="preference", stability="stable",
        event_start="2024-01-01T00:00:00+00:00",
    )
    values = values_from_config({})
    values["time_profiles"] = [{
        "id": "old_state_decay",
        "selector": {"kind": "state", "query_purpose": "history"},
        "function": "exponential",
        "half_life_days": 1,
        "weight": 9,
    }]
    context = _context(query, purpose="history")
    solo = score_document(
        _document(old_state), values, context, {"relevance": 0.8}
    )
    combined = score_document(
        _document(old_state, coffee), values, context, {"relevance": 0.8}
    )

    assert combined["score"] == pytest.approx(solo["score"])
    assert combined["score"] < 0.1
    rows = {row["statement_id"]: row for row in combined["statements"]}
    assert rows["st-stomach"]["match_kind"] == "direct"
    assert rows["st-stomach"]["participates_in_aggregation"] is True
    assert rows["st-coffee"]["match_kind"] == "unmatched"
    assert rows["st-coffee"]["participates_in_aggregation"] is False


def test_empty_query_uses_explicit_claim_scope_but_semantic_only_hit_stays_unassigned():
    document = _document(
        _claim("st-one", "苏州日期暂定11月3日。"),
        _claim("st-two", "机票尚未购买。"),
    )
    empty_context = _context("")
    explicit = match_statements(document, empty_context)
    assert explicit["state"] == "explicit_claim_scope"
    assert explicit["direct_ids"] == ["st-one", "st-two"]
    assert explicit["support_ids"] == []
    preview = score_document(
        document, values_from_config({}), empty_context,
        {"relevance": 0.4, "statement_selection": explicit},
    )
    assert preview["aggregation_statement_ids"] == ["st-one", "st-two"]

    semantic_context = _context("山间温泉的价格是多少？")
    semantic_selection = match_statements(document, semantic_context)
    assert semantic_selection["state"] == "unassigned_semantic_evidence"
    assert semantic_selection["direct_ids"] == []
    semantic_only = score_document(
        document, values_from_config({}), semantic_context,
        {"relevance": 0.63, "semantic_score": 0.91,
         "statement_selection": semantic_selection},
    )
    assert semantic_only["state"] == "unassigned_semantic_evidence"
    assert semantic_only["score"] == pytest.approx(0.63)
    assert semantic_only["aggregation_statement_ids"] == []
    assert all(not row["participates_in_aggregation"] for row in semantic_only["statements"])


def test_ticket_false_premise_matches_historical_correction_background():
    ticket = _claim(
        "st-ticket-state",
        "截至2026年10月19日，苏州机票尚未买，须等排班确定后再决定。",
        kind="state",
        stability="episode",
        event_start="2026-10-19T00:00:00+00:00",
        evidence_refs=["source:trip"],
    )
    document = _document(ticket)
    document["evidence_bindings"] = [{"binding_id": "source:trip"}]
    query = "你帮我买的机票是几点出发？"
    purpose = query_purpose(query, now=NOW)
    context = _context(query, purpose=purpose["purpose"])
    selection = match_statements(document, context)

    assert purpose["purpose"] == "current"
    assert selection["direct_ids"] == ["st-ticket-state"]
    assert correction_matches(document, context, selection) == ["st-ticket-state"]

    use = resolve_document(
        document,
        context,
        source_states={"source:trip": "available"},
    )[0]
    assert use["usage"] == "historical_evidence"
    assert use["time"]["event"]["start"]["value"] == "2026-10-19T00:00:00+00:00"
    assert use["evidence_refs"] == ["source:trip"]


def test_correction_background_setting_is_an_optional_boolean():
    defaults = values_from_config({})
    assert defaults["recall"]["include_correction_background"] is True

    for enabled in (True, False):
        values = values_from_config({})
        values["recall"]["include_correction_background"] = enabled
        assert validate_values(values)["recall"]["include_correction_background"] is enabled

    invalid = values_from_config({})
    invalid["recall"]["include_correction_background"] = "false"
    with pytest.raises(ConfigurationError) as error:
        validate_values(invalid)
    assert error.value.code == "boolean_required"
    assert error.value.field == "recall.include_correction_background"


@pytest.mark.parametrize("flag", ["exact_anchor_match", "planner_lexical_match", "distinctive_anchor_match", "category_overview_item"])
def test_raw_match_flags_cannot_bypass_an_applied_owner_time_or_class_formula(flag):
    from gateway import GatewayService
    service = object.__new__(GatewayService)
    service.config = {}
    service.inject_max_cards = 2
    service.first_card_min_score = 0.55
    service._pick_axis_diverse_dynamic_cards = lambda *args, **kwargs: []
    candidate = {"bucket": {"id": "old-state"}, "score": 0.08, "keyword_score": 0.99,
                 "semantic_score": 0.99, flag: True,
                 "_configured_score": {"window_score": 0.08}}
    assert service._dynamic_bucket_item_has_reliable_recall_signal("旧状态", candidate) is False
    assert service._pick_dynamic_cards([candidate], query="旧状态") == []


def test_semantic_only_attribution_still_obeys_the_owner_global_formula():
    from types import SimpleNamespace
    from memory_recall_runtime import score_pool
    document = _document(_claim("st-coffee", "仍然喜欢咖啡的味道。", kind="preference", stability="stable"))
    values = values_from_config({})
    values["formulas"][0]["expression"] = "0"
    context = _context("山间温泉的价格是多少？")
    result = score_document(document, values, context, {"relevance": 0.8})
    assert result["state"] == "unassigned_semantic_evidence"
    assert result["formula_id"] == "final" and result["score"] == 0
    assert result["aggregation_statement_ids"] == []
    service = SimpleNamespace(config={"memory_system": values})
    request = {"_use_context": context, "_semantic_records": {"b": {"state": "l1", "document": document}}}
    candidate = {"bucket": {"id": "b"}, "score": 0.8}
    score_pool(service, request, context["query"], [candidate])
    assert candidate["score"] == 0
    assert candidate["_configured_score"]["window_score"] == 0


def test_removed_temporal_and_grammatical_shells_do_not_reappear_as_content_hits():
    symptom = _claim('symptom', '昨晚胃难受。', kind='state', stability='episode')
    recovered = _claim('recovered', '她随后说胃已经不疼了。', kind='state', stability='episode')
    unrelated = _claim('reflection', '我意识到自己有时太急着讲笑话。', kind='interpretation')
    selection = match_statements(_document(symptom, recovered, unrelated),
        _context('那天我提到胃不舒服，后来还疼吗？', purpose='history'))
    assert set(selection['direct_ids']) == {'symptom', 'recovered'}
    assert selection['matches']['reflection']['terms'] == []


@pytest.mark.parametrize('wording', ['票仍未购买。', '票还没买。', '没有购买机票。'])
def test_negated_purchase_forms_are_correction_background_without_claiming_current_truth(wording):
    document = _document(_claim('ticket', wording, kind='state', stability='episode'))
    context = _context('你帮我买的机票是几点出发？')
    selection = match_statements(document, context)
    assert correction_matches(document, context, selection) == ['ticket']


def test_not_saying_an_action_happened_is_not_treated_as_negating_the_action():
    document = _document(_claim('speech', '我没有说购买机票。', kind='experience', stability='episode'))
    context = _context('你帮我买的机票是几点出发？')
    selection = match_statements(document, context)
    assert correction_matches(document, context, selection) == []


def test_l1_fallback_cannot_drop_a_required_condition_without_faithful_prose():
    from types import SimpleNamespace
    from memory_recall_runtime import render
    a = _claim('date', '行程暂定11月3日。', evidence_refs=['source:A'])
    b = _claim('condition', '排班确定后才能买票。', evidence_refs=['source:A'])
    b['conditions'] = {'state': 'known', 'text': None,
        'predicate': {'op': 'eq', 'field': 'schedule_confirmed', 'value': True}, 'exceptions': []}
    document = _document(a, b)
    items = resolve_document(document, _context('行程日期呢？'), source_states={'source:A': 'available'})
    usage = {'state': 'l1', 'record': {'document': document}, 'items': items, 'narrative_allowed': False,
        'statement_selection': {'direct_ids': ['date'], 'support_ids': ['condition']}}
    service = SimpleNamespace(_resolve_gateway_fixed_prompt=lambda _name, runtime_values: runtime_values['narrative'])
    trace = {}
    assert render(service, usage, '', '', 4000, trace=trace) == ''
    assert trace['render_reason'] == 'required_context_not_readable'
    assert trace['unrenderable_required_statement_ids'] == ['condition']
