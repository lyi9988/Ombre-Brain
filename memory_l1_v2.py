"""Natural memory cards and their program-owned source bindings.

This module is shared with Gateway. It performs no model calls or writes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re

from jsonschema import Draft202012Validator

from memory_l1_contract import L1ContractError, canonical_json, canonical_l1_body


SCHEMA_VERSION = "memory-l1-v2"
TEXT_PROJECTION_VERSION = "memory-l1-v2-text-v1"
CATALOG_VERSION = "memory-l1-v2-catalog-20261114-r2"
KINDS = {
    "preference": "有后续意义的喜好或习惯，可阶段性或持久，持续程度由 stability 表达",
    "boundary": "明确的行为限制、拒绝、允许或授权范围，保留适用条件；普通喜恶属于偏好",
    "shared_language": "有双方明确约定或共同用法支持的隐语、特殊称呼或默契；保留关键原字词及有据异写",
    "commitment": "单方明确承诺或双方确实达成的约定，保留谁承诺、是否接受、期限和履行状态",
    "project_state": "有长期回想价值的项目决定、关键转折或教训；普通进度和工具结果留原记录",
    "relationship_anchor": "有长期关系意义的相处、陪伴、和解或认识变化；按完整经历选类，可涉及第三方，分清亲历与转述",
    "self_insight": "顾衍对自身动机、倾向、局限、价值或矛盾的新理解或修正，允许不确定",
    "life_experience": "有长期回想价值的具体生活经历，如一次特别的旅行、搬家或家庭事件；不记流水账",
    "knowledge_discovery": "顾衍实际探索所得且值得以后运用的外部知识、方法或理解；完整保留同一探索主线",
    "personal_state": "有后续意义的个人现实处境、身心或生活状态，保留适用时间和依据；一次波动不推成长久状态",
    "person_relationship": "主线是确认有后续用途的人物身份、角色或关系事实，例如经谁认识某人。普通提及、同名和猜测留在原材料里；若主线是一次相处或和解，整件事归 relationship_anchor。",
    "expressed_need": "主线是当事人明确向对方提出、以后仍需理解的诉求，例如希望少隔几天才联系。保留对方实际回应。一般喜好归 preference，现实处境归 personal_state，明确承诺或达成的约定归 commitment。",
}
STABILITIES = {
    "episode": "当次经历或当时状态，单次出现",
    "phase": "某一阶段适用，需要结合期限和后续进展",
    "stable": "明确持续适用或有持续来源支持，后来仍可更正",
    "unknown": "材料不足以判断持续程度",
}
TOPICS = {
    "interaction": {"name": "相处方式", "aliases": ["沟通", "相处", "回应方式"],
        "include": "怎样交流、陪伴及相处边界", "exclude": "仅出现一句普通称呼"},
    "people_relationships": {"name": "人物与关系", "aliases": ["人物", "关系", "认识"],
        "include": "人物身份、彼此联系和怎样认识", "exclude": "只因提到一个人名"},
    "intimacy": {"name": "亲密互动", "aliases": ["亲近", "亲密", "拥抱"],
        "include": "具体亲密经历、愿望及相关边界", "exclude": "只因出现身体部位"},
    "learning_career": {"name": "学习与职业", "aliases": ["学习", "职业", "论文", "工作"],
        "include": "学习、考试、论文与职业经历", "exclude": "与学习无关的一次工具测试"},
    "guyan_project": {"name": "顾衍项目", "aliases": ["顾衍系统", "Reality", "Ombre", "交互界面"],
        "include": "顾衍自身系统、Reality及聊天可交互生成物", "exclude": "所有其他个人创作"},
    "health_routine": {"name": "健康与作息", "aliases": ["健康", "作息", "睡眠", "身体状况"],
        "include": "有意义的身体体验、饮食或作息变化", "exclude": "无后续意义的早餐流水账"},
    "daily_life": {"name": "日常生活", "aliases": ["生活", "购物", "住宿", "出行"],
        "include": "具体生活事件、生活安排与变化", "exclude": "只有随口寒暄"},
    "interests_preferences": {"name": "兴趣与偏好", "aliases": ["兴趣", "喜好", "审美"],
        "include": "音乐、电影、穿搭、设计等兴趣与审美", "exclude": "一次提议自动升级成稳定偏好"},
    "emotions_experiences": {"name": "情绪与体验", "aliases": ["感受", "心情", "心理体验"],
        "include": "有具体经历的感受、自我理解及变化", "exclude": "脱离经历的随口情绪词"},
    "projects_creation": {"name": "项目与创作", "aliases": ["创作", "设计", "产品", "工程"],
        "include": "其他项目、作品与创作过程", "exclude": "纯调试输出或源码全文"},
}
MODEL_REQUIRED = {"title", "body", "kind", "stability", "keywords", "people", "sources", "primary_topic", "event_time"}
MODEL_OPTIONAL = {"secondary_topic", "people_links", "scope_notes"}
DOCUMENT_FIELDS = (MODEL_REQUIRED - {"sources"}) | MODEL_OPTIONAL | {
    "schema_version", "source_refs", "evidence_bindings", "generation"}


def catalog():
    return {"version": CATALOG_VERSION, "kinds": deepcopy(KINDS),
            "topics": deepcopy(TOPICS), "stabilities": deepcopy(STABILITIES)}


def is_v2(value):
    return isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION


def current_kind(kind):
    """Interpret historical signal labels without rewriting stored revisions."""
    return "shared_language" if kind == "signal" else kind


def validate_adoptable_kind(kind):
    if not isinstance(kind, str) or kind not in KINDS:
        _error("classification_gap", "document.kind")


def _error(code, path):
    raise L1ContractError(code, path)


def _string_array(values=None, *, minimum=0):
    item = {"type": "string", "minLength": 1}
    result = {"type": "array", "items": item, "uniqueItems": True, "minItems": minimum}
    if values is not None:
        values = list(dict.fromkeys(values))
        if values:
            item["enum"] = values
        else:
            result["maxItems"] = 0
    return result


def model_card_schema(source_refs, people_refs, catalog=None, *, allow_classification_gap=False):
    vocabulary = catalog or globals()["catalog"]()
    text = {"type": "string", "minLength": 1}
    refs = _string_array(source_refs, minimum=1)
    note = {"type": "object", "additionalProperties": False,
            "required": ["focus", "rule", "sources"], "properties": {
                "focus": text, "rule": text, "sources": refs}}
    link = {"type": "object", "additionalProperties": False,
            "required": ["label", "how", "sources"], "properties": {
                "label": text, "how": {"type": "string", "enum": [
                    "mentioned", "learned_from", "handoff_via", "direct_interaction"]},
                "detail": text, "sources": refs}}
    if people_refs:
        link["properties"]["via"] = {"type": "string", "enum": list(people_refs)}
    event = {"type": ["object", "null"], "additionalProperties": False,
             "description": "有依据时填事件实际发生时间及支持来源。提起旧事的时间、计划日期和消息写入时间不等于事件发生时间。发生时间确实无法确定时填 null。",
             "required": ["start", "precision", "sources"], "properties": {
                 "start": text, "end": text,
                 "precision": {"type": "string", "enum": ["year", "month", "day", "minute", "second", "approximate"]},
                 "sources": refs}}
    kind = {"type": "string", "enum": list(vocabulary["kinds"])}
    if allow_classification_gap:
        kind = {"type": ["string", "null"], "enum": [*kind["enum"], None]}
    return {"type": "object", "additionalProperties": False,
            "required": sorted(MODEL_REQUIRED), "properties": {
                "title": text, "body": text,
                "kind": kind,
                "stability": {"type": "string", "enum": list(vocabulary["stabilities"])},
                "keywords": _string_array(minimum=1),
                "people": _string_array(people_refs), "sources": refs,
                "primary_topic": {"type": "string", "enum": list(vocabulary["topics"])},
                "secondary_topic": {"type": "string", "enum": list(vocabulary["topics"])},
                "event_time": event,
                "people_links": {"type": "array", "items": link},
                "scope_notes": {"type": "array", "items": note}}}


def build_model_response_format(source_refs, people_refs, catalog=None, *, include_review=False):
    card = model_card_schema(source_refs, people_refs, catalog, allow_classification_gap=False)
    if include_review:
        flags = ("sensitive", "transient", "generic")
        card["properties"]["review"] = {"type": "object", "additionalProperties": False,
            "required": ["confidence", *flags], "properties": {
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                **{flag: {"type": "boolean"} for flag in flags},
            }}
        card["required"].append("review")
    return {"type": "json_schema", "json_schema": {
        "name": "memory_l1_v2_cards", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": ["cards"], "properties": {"cards": {
                       "type": "array", "items": card}}}}}


def _date(value, path):
    if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2}))?)?)?", value):
        _error("invalid_event_time", path)
    try:
        if len(value) == 4:
            return datetime(int(value), 1, 1)
        if len(value) == 7:
            return datetime.fromisoformat(value + "-01")
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _error("invalid_event_time", path)


def validate_model_card(card, *, source_refs, people_refs, catalog=None, allow_classification_gap=False,
                         require_primary_topic=True, require_event_time=True, require_date_in_body=True):
    schema = model_card_schema(source_refs, people_refs, catalog, allow_classification_gap=allow_classification_gap)
    if not require_primary_topic:
        schema["required"].remove("primary_topic")
    if not require_event_time:
        schema["required"].remove("event_time")
    errors = sorted(Draft202012Validator(schema).iter_errors(card),
                    key=lambda error: tuple(str(part) for part in error.absolute_path))
    if errors:
        error = errors[0]
        _error("invalid_v2_card_" + error.validator,
               "card." + ".".join(str(part) for part in error.absolute_path))
    value = deepcopy(card)
    value["body"] = canonical_l1_body(value["body"])
    for field in ("title", "keywords"):
        items = value[field] if isinstance(value[field], list) else [value[field]]
        if any(not item.strip() for item in items):
            _error("nonempty_string_required", "card." + field)
    if value.get("secondary_topic") and not value.get("primary_topic"):
        _error("primary_topic_required", "card.secondary_topic")
    if value.get("secondary_topic") == value.get("primary_topic") and value.get("primary_topic"):
        _error("duplicate_topic", "card.secondary_topic")
    event = value.get("event_time")
    if event:
        start = _date(event["start"], "card.event_time.start")
        if event.get("end"):
            end = _date(event["end"], "card.event_time.end")
            try:
                if end < start:
                    _error("reversed_event_range", "card.event_time.end")
            except TypeError:
                _error("inconsistent_event_timezone", "card.event_time")
        if require_date_in_body and event["precision"] in {"day", "minute", "second"}:
            date_text = f"{start.year}年{start.month}月{start.day}日"
            compact = re.sub(r"\s+", "", value["body"])
            if date_text not in compact and start.strftime("%Y-%m-%d") not in compact:
                _error("event_date_missing_from_body", "card.body")
    return value


def referenced_sources(card):
    refs = list(card.get("sources", card.get("source_refs", [])))
    if card.get("event_time"):
        refs.extend(card["event_time"].get("sources", []))
    for field in ("scope_notes", "people_links"):
        for item in card.get(field, []):
            refs.extend(item.get("sources", []))
    return list(dict.fromkeys(refs))


def prepare_card(card, *, people, bindings, generation, catalog=None, source_ref_map=None,
                 allow_classification_gap=False):
    vocabulary = deepcopy(catalog or globals()["catalog"]())
    people_by_ref = {item["ref"]: item for item in people}
    binding_by_id = {item["binding_id"]: item for item in bindings}
    mapping = source_ref_map or {ref: ref for ref in binding_by_id}
    # The provider response schema enforces current generation requirements.
    # Assembly also resumes drafts saved before the topic and time fields became required.
    value = validate_model_card(card, source_refs=list(mapping),
                                people_refs=list(people_by_ref), catalog=vocabulary,
                                allow_classification_gap=allow_classification_gap, require_primary_topic=False,
                                require_event_time=False, require_date_in_body=generation.get("origin") != "chat_l1")
    all_refs = referenced_sources(value)
    selected_binding_ids = list(dict.fromkeys(mapping[ref] for ref in all_refs))
    if any(ref not in binding_by_id for ref in selected_binding_ids):
        _error("unknown_source_mapping", "source_refs")
    selected_people = list(value["people"])
    for link in value.get("people_links", []):
        if link.get("via") and link["via"] not in selected_people:
            selected_people.append(link["via"])
    resolved = []
    person_ids = {}
    for ref in selected_people:
        person = people_by_ref[ref]
        person_id = person.get("person_id", person.get("subject_id"))
        if not isinstance(person_id, str) or not person_id.strip():
            _error("confirmed_person_identity_required", "people." + ref)
        person_ids[ref] = person_id
        resolved.append({"person_id": person_id, "name": person["name"],
                         "kind": person["kind"], "aliases": list(person.get("aliases", []))})
    value["people"] = resolved
    value["source_refs"] = selected_binding_ids
    value.pop("sources")
    for field in ("scope_notes", "people_links"):
        for item in value.get(field, []):
            item["sources"] = list(dict.fromkeys(mapping[ref] for ref in item["sources"]))
            if field == "people_links" and item.get("via"):
                item["via"] = person_ids[item["via"]]
    if value.get("event_time"):
        value["event_time"]["sources"] = list(dict.fromkeys(
            mapping[ref] for ref in value["event_time"]["sources"]))
    value["schema_version"] = SCHEMA_VERSION
    value["evidence_bindings"] = [deepcopy(binding_by_id[ref]) for ref in selected_binding_ids]
    value["generation"] = {**deepcopy(generation), "catalog_snapshot": vocabulary,
                           "catalog_revision": vocabulary["version"]}
    return validate_document(value, allow_classification_gap=allow_classification_gap)


def validate_document(document, *, allow_classification_gap=False):
    canonical_json(document)
    required = (MODEL_REQUIRED - {"sources", "primary_topic", "event_time"}) | {
        "schema_version", "source_refs", "evidence_bindings", "generation"}
    if not isinstance(document, dict) or set(document) - DOCUMENT_FIELDS or required - set(document):
        _error("invalid_v2_document_shape", "document")
    if not is_v2(document):
        _error("unsupported_l1_schema", "document.schema_version")
    people = document["people"]
    if not isinstance(people, list):
        _error("array_required", "document.people")
    person_ids = []
    for person in people:
        if (not isinstance(person, dict) or set(person) != {"person_id", "name", "kind", "aliases"}
                or person.get("kind") not in {"human", "ai", "unknown"}
                or any(not isinstance(person.get(key), str) or not person[key].strip()
                       for key in ("person_id", "name"))
                or not isinstance(person.get("aliases"), list)
                or any(not isinstance(alias, str) or not alias.strip() for alias in person["aliases"])):
            _error("invalid_person", "document.people")
        person_ids.append(person["person_id"])
    if len(set(person_ids)) != len(person_ids):
        _error("duplicate_person", "document.people")
    generation = document["generation"]
    if not isinstance(generation, dict) or not isinstance(generation.get("origin"), str) or not generation["origin"]:
        _error("generation_origin_required", "document.generation")
    vocabulary = generation.get("catalog_snapshot")
    if not isinstance(vocabulary, dict) or not {"version", "kinds", "topics", "stabilities"} <= set(vocabulary):
        _error("catalog_snapshot_required", "document.generation.catalog_snapshot")
    bindings = document["evidence_bindings"]
    if not isinstance(bindings, list) or not bindings:
        _error("source_bindings_required", "document.evidence_bindings")
    # Reuse the established exact-view anchor checks, without inventing facts.
    from memory_l1_schema import _anchor, _revision
    binding_ids = []
    for index, binding in enumerate(bindings):
        path = f"document.evidence_bindings[{index}]"
        expected = {"binding_id", "source_namespace", "source_id", "source_revision", "author_ref", "speaker_ref", "anchors"}
        if not isinstance(binding, dict) or set(binding) != expected:
            _error("invalid_source_binding", path)
        for key in ("binding_id", "source_namespace", "source_id"):
            if not isinstance(binding[key], str) or not binding[key].strip():
                _error("nonempty_string_required", path + "." + key)
        _revision(binding["source_revision"], path + ".source_revision")
        for key in ("author_ref", "speaker_ref"):
            if binding[key] is not None and (not isinstance(binding[key], str) or not binding[key].strip()):
                _error("invalid_source_person", path + "." + key)
        if not isinstance(binding["anchors"], list) or not binding["anchors"]:
            _error("source_anchor_required", path)
        for anchor in binding["anchors"]:
            _anchor(anchor, path + ".anchors")
            if (anchor["target_namespace"] != "l0" or anchor["target_id"] != binding["source_id"]
                    or type(anchor["target_revision"]) is not type(binding["source_revision"])
                    or anchor["target_revision"] != binding["source_revision"]):
                _error("source_anchor_mismatch", path)
        binding_ids.append(binding["binding_id"])
    if len(set(binding_ids)) != len(binding_ids):
        _error("duplicate_source_binding", "document.evidence_bindings")
    card = {key: deepcopy(document[key]) for key in (MODEL_REQUIRED | MODEL_OPTIONAL) - {"sources"} if key in document}
    card["people"] = person_ids
    card["sources"] = deepcopy(document["source_refs"])
    validate_model_card(card, source_refs=binding_ids, people_refs=person_ids, catalog=vocabulary,
                        allow_classification_gap=allow_classification_gap, require_primary_topic=False,
                         require_event_time=False, require_date_in_body=generation["origin"] != "chat_l1")
    if set(referenced_sources(card)) != set(binding_ids):
        _error("unreferenced_source_binding", "document.evidence_bindings")
    return deepcopy(document)


def retrieval_text(body, semantic):
    title = str(semantic.get("title") or "").strip()
    return "\n\n".join(part for part in (title, canonical_l1_body(body)) if part)
