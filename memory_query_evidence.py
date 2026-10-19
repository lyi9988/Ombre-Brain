"""Query-local claim matches, not new facts or a second retrieval authority.

Lexical matches identify candidates, including negative evidence. Unmatched
semantic retrieval remains explicitly unassigned rather than inventing which
claim the embedding matched. Supporting context never supplies a ranking score.
"""
from __future__ import annotations

from functools import lru_cache
import math
import re

from query_terms import GENERIC_LEXICAL_STOPWORDS


_SHELL = GENERIC_LEXICAL_STOPWORDS | frozenset({
    "我", "你", "他", "她", "它", "的", "了", "吗", "呢", "呀", "啊", "吧",
    "是", "在", "有", "和", "与", "也", "都", "就", "又", "还", "不", "没",
    "会", "要", "能", "想", "说", "先", "再", "时", "上", "下", "去", "来",
    "着", "过", "很", "有点", "怎么", "怎样", "如何", "为什么", "时候", "时候的",
    "应该", "可以", "如果", "假如", "记得", "还记得", "提到", "当时", "那天", "那次",
    "后来", "以前", "最后", "已经", "尚未", "没有", "还没", "这几天", "一般", "怎么样",
    "年", "月", "日", "号", "天", "分", "秒",
    "前", "后", "前后", "以后", "之后", "那时", "原来",
    "到", "把", "被", "让", "得", "地", "跟", "从", "对", "为", "于", "等", "这", "那",
    "the", "a", "an", "i", "me", "my", "we", "our", "is", "was", "are", "do",
    "did", "what", "how", "when", "why", "to", "of", "and", "not", "yet",
})


def query_purpose(query, *, target_time=None, now=None):
    """Separate temporal intent from the old-memory eligibility policy.

    An unresolved 'that day' is a historical reference, not a guessed date.
    Current conflict/argument words by themselves carry no temporal intent.
    """
    text = str(query or "").casefold()
    # A negated request to discuss the past is not evidence of historical use.
    intent = re.sub(r"(?:不是问|不是说|不问|别提|不要讲)[^，。！？?;；]{0,8}(?:以前|过去|之前|那天)", "", text)
    past = bool(re.search(r"那天|那次|当时|以前|之前|上次|从前|曾经|后来|昨[天晚日]|前天|上[周月年]|去年|"
                          r"\b(?:previously|before|yesterday|last\s+(?:week|month|year)|back\s+then)\b", intent))
    current = bool(re.search(r"现在|目前|此刻|今天|最近|这几天|这周|刚刚|现住址|"
                             r"\b(?:now|currently|today|these\s+days)\b", intent))
    conditional = bool(re.search(r"如果|假如|每当|当.{1,20}时|.{1,20}的时候|\b(?:if|whenever)\b", intent))
    recall = bool(re.search(r"记得|回忆|说过|提过|怎么定的|\b(?:remember|recall)\b", intent))
    if past and current:
        purpose, reason = "mixed", "past_and_present_comparison"
    elif current:
        purpose, reason = "current", "explicit_present_reference"
    elif past:
        purpose, reason = "history", "explicit_past_reference"
    elif conditional:
        purpose, reason = "relationship", "conditional_interaction_query"
    elif target_time is not None:
        purpose, reason = ("history", "resolved_date_reference") if now is None or target_time < now else ("current", "resolved_current_or_future_reference")
    elif recall:
        purpose, reason = "history", "explicit_recollection_query"
    else:
        purpose, reason = "current", "no_explicit_past_reference"
    return {"purpose": purpose, "purpose_reason": reason,
            "temporal_reference": "resolved" if target_time is not None else "unresolved" if past else "not_specified",
            "target_time": target_time}


@lru_cache(maxsize=2048)
def terms(text):
    text = str(text or "").casefold()
    import jieba
    cut = getattr(jieba, "cut_for_search", jieba.cut)
    return frozenset(word.strip() for word in cut(text)
        if word.strip() and word.strip() not in _SHELL
        and re.fullmatch(r"[\u4e00-\u9fff]+|[a-z][a-z0-9_.-]*", word.strip())
        and (not word.isascii() or len(word) > 1))


def _text(statement):
    condition = statement.get("conditions") or {}
    cognition = statement.get("cognition") or {}
    return " ".join(str(value) for value in [statement.get("content") or "", condition.get("text") or "",
        *(item.get("text") or "" for item in condition.get("exceptions") or []),
        *(cognition.get("open_questions") or [])])


@lru_cache(maxsize=512)
def _noun_fragment(term):
    import jieba.posseg as pseg
    return any(part.word == term and part.flag.startswith('n') for part in pseg.cut(term))


def match_statements(document, context):
    """Return direct and explicit supporting refs; never copy a bucket hit to all.

    A complete declared scene/relation may supply context. Coherent L2 sections
    supply their own additional support later, without changing this hit set.
    """
    statements = document.get("statements") or []
    by_id = {row["statement_id"]: row for row in statements if row.get("statement_id")}
    query = str(context.get("query") or "").casefold()
    if not query:
        return {"state": "explicit_claim_scope", "direct_ids": list(by_id), "support_ids": [],
                "matches": {sid: {"score": 1.0, "terms": [], "reason": "explicit_claim_scope"} for sid in by_id}}
    # Mere shared identity is not a content hit on every claim about that person.
    identity = {term for entity in document.get("entities") or []
                for value in [entity.get("name"), *(entity.get("aliases") or [])] for term in terms(str(value or ""))}
    query_terms = set(terms(query)) - identity
    texts = {sid: _text(row).casefold() for sid, row in by_id.items()}
    statement_terms = {sid: set(terms(text)) - identity for sid, text in texts.items()}
    # Allow a declared specific word such as '票' to match a compound '机票'.
    anchors = query_terms | {term for values in statement_terms.values() for term in values
        if any(term in token for token in query_terms)
        and (len(term) > 1 or _noun_fragment(term))}
    hits = {sid: {term for term in anchors if term in text and term in query} for sid, text in texts.items()}
    weights = {term: math.log(1 + len(by_id) / (1 + sum(term in values for values in hits.values()))) for term in anchors}
    denominator = sum(weights.values()) or 1.0
    matches = {sid: {"score": sum(weights[term] for term in values) / denominator,
                     "terms": sorted(values), "reason": "local_content_match" if values else "no_local_content_match"}
               for sid, values in hits.items()}
    direct = {sid for sid, row in matches.items() if row["terms"]}
    selected, support_reasons = set(direct), {}
    # Only explicit local endpoints are traversed here. Cross-memory links keep
    # their existing source/revision qualification in the runtime relation path.
    edges = []
    for scene in document.get("scenes") or []:
        edges.append((set(scene.get("statement_refs") or []) & by_id.keys(), "declared_scene_context"))
    for relation in document.get("relations") or []:
        if relation.get("confirmation_state") == "rejected":
            continue
        if relation.get("kind") not in {"same_event", "supplements", "revises", "limits", "supersedes", "possible_conflict"}:
            continue
        ends = [relation.get(key) or {} for key in ("from", "to")]
        local = {end.get("object_id") for end in ends if end.get("object_type") == "statement"
                 and (not end.get("memory_id") or end.get("memory_id") == context.get("memory_id"))
                 and (end.get("revision") in (None, "this_revision") or end.get("revision") == context.get("revision"))}
        edges.append((local & by_id.keys(), "declared_" + relation["kind"]))
    changed = True
    while changed:
        changed = False
        for refs, reason in edges:
            if refs & selected:
                for sid in refs - selected:
                    selected.add(sid)
                    support_reasons[sid] = reason
                    changed = True
    return {"state": "matched" if direct else "unassigned_semantic_evidence",
            "direct_ids": [sid for sid in by_id if sid in direct],
            "support_ids": [sid for sid in by_id if sid in selected - direct],
            "support_reasons": support_reasons, "matches": matches}


def correction_matches(document, context, selection):
    """A bounded background cue, not an adjudication that a premise is false.

    It requires both a shared content anchor and an explicitly negated action
    that the question presupposes. The source-time qualification remains intact.
    """
    query = str(context.get("query") or "").casefold()
    presupposed = re.search(r"(?:你|我|他|她).{0,18}的.{0,18}(?:是|几点|在哪|怎么样|如何|多少)|已经.{1,12}(?:了|的)|\b(?:you|i)\s+.*\b(?:bought|booked|sent|finished)\b", query)
    if not presupposed:
        return []
    found = []
    by_id = {row.get("statement_id"): row for row in document.get("statements") or []}
    for sid in selection.get("direct_ids") or []:
        row = by_id[sid]
        text = str(row.get("content") or "")
        negations = re.findall(r"(?:尚未|还未|还没|未曾|从未|没有|并未|未|没)([^，。！？；,;.!?]{1,12})", text)
        shared = set(selection["matches"][sid]["terms"])
        import jieba.posseg as pseg
        predicates = []
        for span in negations:
            predicate = next((part.word for part in pseg.cut(span) if part.flag.startswith('v')), None)
            if predicate:
                predicates.append(predicate)
        actions = {term for term in shared if any(term in predicate or predicate in term for predicate in predicates)}
        if actions and shared - actions:
            found.append(sid)
    return found


def semantic_source_refs(value):
    """Explicit semantic dependencies, excluding the source inventory itself."""
    refs = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"evidence_refs", "source_refs"} and isinstance(item, list):
                refs.update(ref for ref in item if isinstance(ref, str))
            elif key not in {"evidence_bindings", "generation"}:
                refs.update(semantic_source_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(semantic_source_refs(item))
    return refs
