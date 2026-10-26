"""Evidence-qualified, non-additive final recall reward from the unified state.

No state writes, model calls, topic-only rewards or relevance admission occur.
"""
from __future__ import annotations
import ast
import copy
import math
import re

from recall_signal_protocol import INNER_DIMENSIONS
from memory_affinity import resolve_signals

DEFAULT_POLICY = {"enabled": True, "max_bonus": .05, "weights": {}}
_ACTIVE = frozenset(("curiosity", "novelty", "expression", "creativity", "autonomy",
    "competence", "solitude", "boredom", "duty", "reflection", "tenderness",
    "possessiveness", "longing", "security", "protective_drive", "libido", "grieve",
    "anger", "favored", "session_defensiveness", "cling"))
DEFAULT_WEIGHTS = {key: 1.0 if key in _ACTIVE else 0.0 for key in INNER_DIMENSIONS}

# These are local use cues in the actual statement/cognitive question text.
# A catalog topic/domain alone is intentionally never consulted here.
_USE = {
 "energy": r"精力|体力|energy", "rest": r"休息|睡眠|rest",
 "curiosity": r"好奇|探索|研究|理解|了解|为什么|问题|发现|curious|explor|understand|research",
 "novelty": r"新鲜|尝试|新的|未知|冒险|novel|new experience",
 "expression": r"表达|分享|倾诉|说说|聊聊|诉说|express|share|talk",
 "creativity": r"创作|创造|灵感|写作|绘画|设计|creative|creat|inspiration",
 "autonomy": r"自主|独立|自己决定|选择权|自愿|autonom|independen",
 "competence": r"能力|掌握|学会|胜任|进步|练习|competenc|learn|skill",
 "solitude": r"独处|安静|空间|一个人|solitude|quiet|alone",
 "boredom": r"无聊|没意思|枯燥|找点事|bored",
 "duty": r"责任|完成|目标|承担|成就|承诺|duty|accomplish|commit",
 "reflection": r"反思|回顾|想清楚|复盘|理解自己|reflect",
 "valence": r"心情|快乐|愉快|难过|mood",
 "arousal": r"激动|平静|兴奋|arousal",
 "tenderness": r"温柔|体贴|陪伴|拥抱|照顾|tender|gentle|cuddle",
 "possessiveness": r"占有|吃醋|只属于|独占|嫉妒|possess|jealous",
 "longing": r"想念|靠近|亲近|相见|想你|思念|longing|miss you|closer",
 "security": r"安全感|安心|不安|害怕失去|确定关系|security|insecure|reassur",
 "protective_drive": r"保护|关心|照顾|担心|守护|protect|care for",
 "libido": r"亲密|欲望|身体|亲吻|亲近|拥吻|性爱|感官|缠绵|intimacy|intimate|desire|sensual|kiss",
 "grieve": r"委屈|受伤|伤心|难过|失落|痛苦|悲伤|griev|hurt|sad",
 "anger": r"生气|愤怒|争执|冲突|吵架|不满|angry|anger|argument|conflict",
 "favored": r"被选择|偏爱|回应|重视|在乎|选择我|favored|chosen|respond",
 "session_defensiveness": r"防备|戒备|误解|防御|解释|defens|misunderstand",
 "cling": r"依恋|黏|粘|不舍|陪着|别走|cling|attached|stay with",
 "fatigue": r"疲惫|累了|负担|fatigue",
 "slow_security": r"安全感|security", "slow_confidence": r"信心|confidence",
 "slow_mood": r"心情|mood", "exploration_affinity": r"探索|explor",
 "social_recovery_rate": r"陪伴|恢复|recover", "connection_initiative": r"联系|靠近|connect",
 "creative_satisfaction": r"创作|creat", "challenge_tolerance": r"挑战|challenge",
 "reflection_preference": r"反思|reflect",
}
_RELATIONAL = frozenset(key for key, (_, kind) in INNER_DIMENSIONS.items()
                       if kind == "affect" and key not in {"valence", "arousal"})
_FACT = re.compile(r"哪里|哪儿|哪个|哪天|哪年|几点|几号|多少|谁|是否|有没有|是不是|"
    r"什么时候|何时|何处|什么时间|日期|地点|地址|住址|准确|具体时间|已经.{0,12}(?:了|完成)|"
    r"\b(?:where|when|who|how many|how much|what time|which date|did|has|have)\b", re.I)
_OPEN = re.compile(r"聊聊|聊一聊|随便|陪我|陪你|相处|靠近|亲近|一起|探索|研究|了解|灵感|"
    r"创作|反思|复盘|想念|想你|抱抱|亲密|分享|说说|倾诉|"
    r"\b(?:explore|talk|chat|share|create|reflect|together|closer)\b", re.I)

def _number(value):
    return type(value) in (int, float) and math.isfinite(value)

def _known(value):
    return value.get("value") if isinstance(value, dict) and value.get("state") == "known" else None

def build_inner_context(settings, snapshots, now, query, *, scope=None):
    """Freeze the existing complete checkpoint and approved owner policy once."""
    configured = (settings.get("recall") or {}).get("inner_state_affinity") or {}
    policy = {**DEFAULT_POLICY, **copy.deepcopy(configured)}
    policy["weights"] = {**DEFAULT_WEIGHTS, **(configured.get("weights") or {})}
    text = str(query or "")
    strong = bool(_FACT.search(text) or len(re.findall(r"[?？]", text)) >= 2)
    mode = "fact" if strong else "open" if _OPEN.search(text) else "unproven"
    bindings = [{"id":key, "producer":"guyan.inner_state", "key":key} for key in INNER_DIMENSIONS]
    signals = resolve_signals({"recall":{"signal_bindings":bindings}}, snapshots, now)
    # Legacy fragments are readable by legacy bindings, but cannot stand in for
    # a complete 35-dimensional checkpoint in this new component.
    supplied = any(isinstance(row, dict) and row.get("producer") == "guyan.inner_state" for row in snapshots)
    return {"policy":policy, "signals":signals, "query":text, "query_mode":mode,
            "scope":copy.deepcopy(scope or {}), "state":"available" if any(
                row["state"] == "known" for row in signals.values()) else "unknown" if supplied else "not_provided"}

def _entity_scope(statement, document, query):
    refs = {_known(statement.get("subject_ref"))}
    refs.update((statement.get("cognition") or {}).get("about_refs") or [])
    rows = [row for row in document.get("entities") or [] if row.get("entity_id") in refs]
    explicit, personal, third_party, personal_subject = False, False, False, False
    for row in rows:
        kind = _known(row.get("entity_type"))
        names = [str(name).casefold() for name in [row.get("name"), *(row.get("aliases") or [])] if name]
        named = any(name in query.casefold() for name in names)
        if kind == "user":
            personal_subject = True
            current = named or bool(re.search(r"我|主人|\b(?:i|me|my|we|our)\b", query, re.I))
            personal |= current; explicit |= current
        elif kind == "assistant":
            personal_subject = True
            current = named or bool(re.search(r"你|顾衍|\b(?:you|your|we|our)\b", query, re.I))
            personal |= current; explicit |= current
        else:
            explicit |= named
            third_party |= kind in {"third_party", "person", "group", "organization"}
    return {"explicit":explicit, "personal":personal and not third_party, "third_party":third_party,
            "personal_subject":personal_subject and not third_party}

def _bindings_for(statement, document):
    wanted = set(statement.get("evidence_refs") or [])
    return [row for row in document.get("evidence_bindings") or [] if row.get("binding_id") in wanted]

def _causal_match(statement, document, signal, scope):
    import json
    refs = signal.get("source_refs") or []
    for binding in _bindings_for(statement, document):
        namespace = str(binding.get("source_namespace") or "")
        if namespace not in {"snapshot.aiz.canonical_event", "snapshot.ombre.raw_event",
                             "snapshot.ombre.conversation_turn", "snapshot.aiz.action_receipt"}:
            continue
        try:
            locator = json.loads(binding.get("source_id", ""))
        except (ValueError, TypeError):
            continue
        if not isinstance(locator, dict):
            continue
        if scope.get("conversation_id") and locator.get("conversation_id") != scope["conversation_id"]:
            continue
        event_id = locator.get("event_id")
        for ref in refs:
            expected_kind = "operation" if namespace.endswith("action_receipt") else "conversation"
            if (ref.get("kind") == expected_kind and isinstance(event_id, str)
                    and event_id == ref.get("event_id")):
                return True
    return False

def _source_ready(statement, document, usage, source_states):
    bindings = _bindings_for(statement, document)
    # An unspecified condition is still unknown. A qualified recollection may
    # nevertheless be relevant as a reference; ranking it does not assert that
    # its contents hold now. Explicit unmet/unknown requirements stay gated.
    reference_only = bool(usage and usage.get("condition_match") is None
        and (statement.get("conditions") or {}).get("state") == "not_stated"
        and usage.get("usage") in {"relationship_reference", "historical_evidence"}
        and not ({"superseded_for_current_use", "confirmed_revision_replacement", "state_needs_reconfirmation"}
                 & set(usage.get("reason_codes") or [])))
    return bool(statement.get("content") and bindings and usage
        and usage.get("source_state") == "available"
        and usage.get("usage") not in {"excluded", "unresolved_evidence"}
        and usage.get("scope_match") is not False
        and (usage.get("condition_match") is True or reference_only)
        and all(source_states.get(row["binding_id"]) == "available" for row in bindings))

def statement_inner_affinity(statement, context, *, document, usage=None, selection=None, source_states=None):
    """Return a small independent component; never turn evidence into a fact."""
    policy = context["policy"]
    result = {"state":"unknown","bonus":0.0,"reason":None,"contributions":[],
              "statement_id":statement.get("statement_id"),"matched_by":[]}
    if not policy.get("enabled") or not _number(policy.get("max_bonus")) or policy["max_bonus"] <= 0:
        result.update(state="disabled",reason="disabled")
        return result
    if context["query_mode"] == "fact":
        result["reason"] = "fact_query"
        return result
    if not _source_ready(statement, document, usage, source_states or {}):
        result["reason"] = "statement_source_or_use_unavailable"
        return result
    sid = statement.get("statement_id")
    direct = sid in (selection or {}).get("direct_ids", [])
    if (selection or {}).get("missing_support_ids"):
        result["reason"] = "required_support_unavailable"
        return result
    entity = _entity_scope(statement, document, context["query"])
    text = " ".join([str(statement.get("content") or ""),
                    *((statement.get("cognition") or {}).get("open_questions") or [])])
    for key, signal in context["signals"].items():
        weight = policy["weights"].get(key, 0)
        if not _number(weight) or weight <= 0:
            continue
        # Intensity does not establish what this particular memory is useful for.
        relevant = bool(re.search(_USE[key], text, re.I))
        causal = _causal_match(statement, document, signal, context.get("scope") or {})
        basis = (["current_event_source"] if causal else [] if context["query_mode"] != "open" else
                 ["query_statement_content"] if direct else
                 ["current_entity_use"] if entity["explicit"] else [])
        relational_ok = key not in _RELATIONAL or ((entity["personal"] or causal and entity["personal_subject"]) and not any(
            str(row.get("source_namespace") or "").startswith("snapshot.aiz.browse.")
            for row in _bindings_for(statement, document)))
        row = {"dimension":key,"state":signal.get("state","unknown"),"intensity":None,
               "weight":weight,"contribution":None,"reason":None,"matched_by":basis}
        if not relevant or not basis or not relational_ok:
            row["reason"] = "current_use_unproven" if relational_ok else "current_person_unproven"
        elif signal.get("state") != "known" or not _number(signal.get("value")):
            row["reason"] = signal.get("reason") or "signal_unknown"
        else:
            intensity = 1 - signal["value"] if key == "security" else signal["value"]
            row.update(intensity=intensity,contribution=intensity * weight,
                       state="known",reason=None)
            for name in basis:
                if name not in result["matched_by"]: result["matched_by"].append(name)
        result["contributions"].append(row)
    known = [row["contribution"] for row in result["contributions"] if row["contribution"] is not None]
    if known:
        result.update(state="computed",reason=None,bonus=policy["max_bonus"] * max(known))
    else:
        result["reason"] = "no_qualified_known_dimension"
    return result

def _formula_uses_state(settings, calculation):
    """Avoid a second reward when the owner's actual formula consumes state."""
    bindings = {row["id"]:row for row in (settings.get("recall") or {}).get("signal_bindings", [])}
    state_ids = {key for key,row in bindings.items() if row.get("producer") in
                 {"guyan.inner_state","aiz.needs","ombre.persona"}}
    state_inputs = {"signal_" + key for key in state_ids}
    for rule in (settings.get("recall") or {}).get("affinity_mappings", []):
        if rule.get("signal_id") in state_ids: state_inputs.add(rule.get("target"))
    features = {row["id"]:row for row in settings.get("features", [])}
    formulas = {row["id"]:row for row in settings.get("formulas", [])}
    def names(expression):
        try: return {node.id for node in ast.walk(ast.parse(expression or "0", mode="eval")) if isinstance(node, ast.Name)}
        except (SyntaxError, TypeError, ValueError): return set()
    used = {calculation.get("formula_id")}
    used.update(row.get("formula_id") for row in calculation.get("statements", [])
                if row.get("participates_in_aggregation"))
    pending = set().union(*(names((formulas.get(ident) or {}).get("expression")) for ident in used))
    seen = set()
    while pending:
        ident = pending.pop()
        if ident in state_inputs: return True
        if ident in seen: continue
        seen.add(ident)
        feature = features.get(ident) or {}
        if feature.get("input_key") in state_inputs: return True
        pending.update(names(feature.get("expression")) - seen)
    return False

def document_inner_affinity(document, context, *, usage, selection, calculation, settings):
    result = {"state":"unknown","bonus":0.0,"reason":None,"aggregation":"max","statements":[],
              "source_state":context.get("state", "not_provided")}
    if _formula_uses_state(settings, calculation):
        result.update(state="suppressed",reason="owner_formula_already_consumes_state")
        return result
    uses = {row["statement_id"]:row for row in (usage or {}).get("items", [])}
    for statement in document.get("statements") or []:
        result["statements"].append(statement_inner_affinity(statement,context,document=document,
            usage=uses.get(statement["statement_id"]),selection=selection,
            source_states=(usage or {}).get("source_states") or {}))
    known = [row["bonus"] for row in result["statements"] if row["state"] == "computed"]
    if known:
        result.update(state="computed",reason=None,bonus=max(known))
    else:
        result["reason"] = next((row["reason"] for row in result["statements"] if row.get("reason")), "l1_evidence_not_provided")
    return result

def local_inner_candidates(records, buckets, context, settings, present, *, limit):
    """Use only already-loaded L1 index rows and spare ordinary pool capacity."""
    if (context["query_mode"] == "fact" or not context["policy"].get("enabled")
            or context["policy"].get("max_bonus", 0) <= 0 or limit <= 0):
        return []
    from memory_recall_semantics import lexical_features
    result = []
    for bid, record in records.items():
        document = record.get("document")
        if (bid in present or bid not in buckets or record.get("state") != "l1"
                or record.get("historical") or not document or record.get("sync_scope")):
            continue
        proven = []
        for statement in document.get("statements") or []:
            if not statement.get("content") or not _bindings_for(statement,document): continue
            entity = _entity_scope(statement,document,context["query"])
            for key,signal in context["signals"].items():
                if (context["policy"]["weights"].get(key,0) <= 0 or signal.get("state") != "known"
                        or not re.search(_USE[key],str(statement.get("content") or ""),re.I)):
                    continue
                intensity = 1 - signal["value"] if key == "security" else signal["value"]
                if intensity <= 0: continue
                cause = _causal_match(statement,document,signal,context.get("scope") or {})
                if key in _RELATIONAL and (not (entity["personal"] or cause and entity["personal_subject"]) or any(
                    str(row.get("source_namespace") or "").startswith("snapshot.aiz.browse.")
                    for row in _bindings_for(statement, document))): continue
                if cause or context["query_mode"] == "open" and entity["explicit"]:
                    proven.append((key, "current_event_source" if cause else "current_entity_use"))
        if not proven: continue
        lexical = lexical_features(context["query"],document,settings["catalog"])
        weights = settings["recall"]["local_match_weights"]
        # A source-event exact match uses the existing exact-content cue weight.
        # It remains a candidate and still needs the ordinary source/admission pass.
        causal = any(reason == "current_event_source" for _,reason in proven)
        strength = max((value * weights[key] for key,value in lexical.items()),default=0)
        if causal:
            strength = max(strength,weights["keyword_match"])
        if strength <= 0: continue
        result.append({"bucket":buckets[bid],"score":strength,"keyword_score":0.0,"semantic_score":0.0,
            "l1_local_match":{**lexical,**({"source_event_match":1.0} if causal else {})},
            "_inner_candidate":{"state":"pending_source_verification","matches":[
                {"dimension":key,"reason":reason} for key,reason in proven]}})
    result.sort(key=lambda row:(-row["score"],str(row["bucket"]["id"])))
    return result[:limit]
