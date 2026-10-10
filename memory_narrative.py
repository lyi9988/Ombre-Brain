"""Source-bound narration and time metadata; no new Memory authority or model call.

An extraction batch date is not an event date. Quoted time expressions are
checked against the cited source, but this is not a proof of the event itself.
Relative/imprecise expressions deliberately remain imprecise.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import re
from typing import Any


CONTRACT_VERSION = "chat-narrative-v1"


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _label(value: Any, limit: int = 80) -> str:
    return re.sub(r"[\[\]\r\n\x00-\x1f]", " ", str(value or "")).strip()[:limit]


def iso_day(value: Any) -> str:
    text = str(value or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", text):
        return ""
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return ""


def _source_day(value: Any, tz) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz).date().isoformat()
    except (TypeError, ValueError):
        return ""


def source_event_time(candidate: dict, turns: list[dict], tz) -> dict:
    """Accept only a time expression within one uniquely cited source passage.

    No date is inferred from batch time, model-supplied normalized values,
    titles, or unrelated source turns. More complex dates retain the quote.
    """
    unknown = {"precision": "unknown", "basis": "not_stated"}
    proposed = _mapping(candidate.get("event_time"))
    expression = str(proposed.get("expression") or "").strip()
    evidence = str(proposed.get("evidence") or "").strip()
    role = proposed.get("source_role")
    if not expression:
        return unknown
    unknown = {"precision": "unknown", "basis": "unverified_time_expression"}
    if role not in {"user", "assistant"} or not evidence or len(evidence) > 600 or len(expression) > 60:
        return unknown
    matches = [turn for turn in turns if str(turn.get("id")) == str(proposed.get("source_turn_id"))]
    if len(matches) != 1:
        return unknown
    turn = matches[0]
    source_text = str(turn.get(f"{role}_text") or "")
    if evidence not in source_text or evidence.count(expression) != 1 or source_text.count(evidence) != 1:
        return unknown
    result = {
        "precision": "expression", "basis": "source_expression",
        "expression": expression, "source_role": role,
        "source_ref": f"conversation_turn:{turn['id']}",
        "source_span": [source_text.index(evidence), source_text.index(evidence) + len(evidence)],
        "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "reference_date": _source_day(turn.get("created_at"), tz),
    }
    # Normalize only a complete absolute date; '昨天' and '去年春天' keep
    # their original precision and source-time anchor, not the review clock.
    absolute = re.fullmatch(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})日?", expression)
    if absolute:
        try:
            result.update(precision="day", value=date(*map(int, absolute.groups())).isoformat())
        except ValueError:
            return unknown
    return result


def build_chat_narrative(candidate: dict, turns: list[dict], identity: dict, tz, batch_date: str) -> dict:
    """Called after source-ID resolution, never from arbitrary model metadata."""
    names = {"assistant": str(identity.get("ai_name") or "AI"),
             "user": str(identity.get("user_display_name") or "用户")}
    sources = []
    days = set()
    for turn in turns:
        day = _source_day(turn.get("created_at"), tz)
        if day:
            days.add(day)
        for role in ("user", "assistant"):
            if str(turn.get(f"{role}_text") or "").strip():
                sources.append({"source_ref": f"conversation_turn:{turn.get('id')}",
                                "role": role, "speaker": names[role], "mentioned_date": day,
                                "source_text_sha256": hashlib.sha256(str(turn[f"{role}_text"]).encode("utf-8")).hexdigest()})
    return {
        "version": CONTRACT_VERSION,
        "narrator": {"role": "assistant", "name": names["assistant"]},
        "source_kind": "daily_chat", "sources": sources,
        "mentioned_dates": sorted(days), "extraction_date": iso_day(batch_date),
        "event_time": source_event_time(candidate, turns, tz),
        # This describes a generation process, not a change of narrator or an
        # assertion that the conversational model personally authored the text.
        "generation_kind": "background_extraction",
    }


def chat_commit_metadata(candidate: dict, recorded_at: str) -> dict:
    narrative = dict(_mapping(candidate.get("narrative")))
    if narrative.get("version") == CONTRACT_VERSION:
        body = str(candidate.get("content") or candidate.get("proposed_memory") or "")
        narrative["body_sha256"] = narrative_body_hash(body)
    event = _mapping(narrative.get("event_time"))
    event_day = iso_day(event.get("value")) if event.get("precision") == "day" else ""
    metadata = {
        "narrative": narrative, "date": event_day or None, "event_date": event_day or None,
        "mentioned_date": iso_day(candidate.get("date")),
        "recorded_at": recorded_at,
    }
    if isinstance(candidate.get("semantic_annotations"), dict):
        from memory_semantics import guard_metadata
        metadata["semantic_annotations"] = candidate["semantic_annotations"]
        metadata = guard_metadata(metadata, str(candidate.get("content") or candidate.get("proposed_memory") or ""))
    return metadata


def narrative_body_hash(body: str) -> str:
    text = str(body or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_context_labels(bucket: dict | None = None, moment: dict | None = None) -> list[str]:
    """Shared Brain/Gateway rendering, including truthful legacy fallbacks.

    Labels travel inside the existing Memory block, not an additional injector.
    Bodies/indices stay unchanged; legacy batch dates are labelled as mentions.
    """
    bucket, moment = bucket or {}, moment or {}
    meta, mm = _mapping(bucket.get("metadata")), _mapping(moment.get("metadata"))
    if not bucket:
        meta = {
            "narrative": mm.get("bucket_narrative"), "source": mm.get("bucket_source"),
            "from_daily_chat": mm.get("bucket_from_daily_chat"), "date": mm.get("bucket_date"),
            "event_date": mm.get("bucket_event_date"), "mentioned_date": mm.get("bucket_mentioned_date"),
            "recorded_at": mm.get("bucket_recorded_at"),
        }
    narrative = _mapping(meta.get("narrative"))
    labels = []
    stale = bool(mm.get("bucket_narrative_stale")) if not bucket else (
        bool(narrative.get("body_sha256")) and "content" in bucket and
        narrative_body_hash(bucket["content"]) != narrative["body_sha256"])
    if stale:
        explicit_date = iso_day(meta.get("date"))
        return ["[正文已修订；旧叙述归属与原文时间标注不适用]",
                f"[记忆日期字段:{explicit_date}；与修订正文的对应关系待核对]" if explicit_date else "[事件日期:未知]"]
    if narrative.get("version") == CONTRACT_VERSION:
        narrator = _mapping(narrative.get("narrator"))
        name = _label(narrator.get("name"))
        if name and narrative.get("edited_by") != "owner":
            labels.append(f"[叙述者:{name}；正文的我指叙述者，引文按原说话人]")
        labels.append("[来源:聊天模型当场写下的回忆；非主人逐项确认]" if narrative.get("generation_kind") == "chat_authored"
                      else "[来源:双方聊天的事后整理]")
        if narrative.get("edited_by") == "owner":
            labels.append("[正文经主人编辑；人称按正文明确归属]")
        event = _mapping(narrative.get("event_time"))
        event_day = iso_day(event.get("value")) if event.get("precision") == "day" else ""
        explicit_date = iso_day(meta.get("date"))
        date_overridden = "date" in meta and explicit_date != event_day
        if date_overridden:
            labels.append(f"[事件日期:{explicit_date}；记忆日期字段]" if explicit_date else "[事件日期:未知；日期字段已清空]")
        elif event_day:
            labels.append(f"[事件日期:{event_day}；依据原文陈述]")
        elif event.get("basis") == "source_expression" and event.get("expression"):
            labels.append(f"[事件时间原话:{_label(event['expression'], 60)}]")
            anchor = iso_day(event.get("reference_date"))
            if anchor:
                labels.append(f"[该时间表述的说话日期:{anchor}；不是本次回忆日期]")
        else:
            labels.append("[事件日期:未知；勿用提及或入库日期代替]")
        raw_days = narrative.get("mentioned_dates")
        days = [iso_day(day) for day in raw_days if iso_day(day)] if isinstance(raw_days, list) else []
        if days:
            value = days[0] if len(days) == 1 else f"{min(days)}至{max(days)}"
            labels.append(f"[提及日期:{value}]")
        elif iso_day(meta.get("mentioned_date")):
            labels.append(f"[整理材料日期:{iso_day(meta['mentioned_date'])}；逐条说话时间未记录]")
        recorded = iso_day(meta.get("recorded_at"))
        if recorded:
            labels.append(f"[入库日期:{recorded}]")
        return labels

    # Earlier daily extractors assigned the review day to both date/event_date
    # and backdated created to 23:59:59. None proves the event or write time.
    legacy_auto_moment = not bucket and str(moment.get("bucket_id") or "").startswith("daily_chat_memory_")
    if meta.get("source") == "daily_chat_memory" or meta.get("from_daily_chat") or legacy_auto_moment:
        labels.append("[事件日期:未知；旧自动记忆日期仅为聊天整理日期]")
        mentioned = iso_day(meta.get("mentioned_date") or meta.get("date") or meta.get("event_date"))
        if mentioned:
            labels.append(f"[聊天整理日期:{mentioned}]")
        return labels

    event_day = iso_day(meta.get("date") or meta.get("event_date") or mm.get("bucket_date") or mm.get("date"))
    if event_day:
        return [f"[date:{event_day}]"]
    created = iso_day(meta.get("created") or mm.get("bucket_created") or moment.get("created_at"))
    if created:
        return [f"[created:{created}]", "[事件日期:未知；created仅为记录时间]"]
    return ["[事件日期:未知]"]
