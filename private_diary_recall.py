"""Read the fixed conversation's adopted diary only for explicit dated questions."""
from __future__ import annotations

import asyncio
import hashlib
import re

import httpx

from recall_rerank_window import explicit_query_parts
from turn_clock import reference_now
from utils import count_tokens_approx, parse_human_date_reference, strip_human_date_references


_DIARY = re.compile(r"日记|手记|日誌|diary|journal|(?:当时|那天|那时).{0,8}(?:怎么想|在想什么|什么感受)|(?:写了什么|写下了什么|写过什么)", re.I)
_FULL = re.compile(r"全文|整篇|完整(?:日记|原文|内容)|逐字|full\s+(?:text|diary|entry)|entire\s+entry", re.I)
_NEXT = re.compile(r"(?:从第?\s*(\d+)\s*(?:个)?(?:字|字符)|(?:offset|character)\s+(\d+))", re.I)


def queries(query, *, tz):
    """Keep the date with the question that actually asks for the diary."""
    if not _DIARY.search(query):
        return []
    now = reference_now(tz)
    output = []
    for part in explicit_query_parts(query):
        if not _DIARY.search(part):
            continue
        hint = parse_human_date_reference(part, now=now, tz=tz)
        if not hint:
            continue
        remainder = part.replace(hint["label"], "", 1)
        another = parse_human_date_reference(remainder, now=now, tz=tz)
        if another and another["date"] != hint["date"]:
            continue
        offset = _NEXT.search(remainder)
        offset = max(0, int(next(value for value in offset.groups() if value)) - 1) if offset else 0
        focus = re.sub(r"私人日记|日记|手记|日誌|(?:当时|那天|那时)|(?:怎么想|在想什么|什么感受)|"
                       r"写(?:了|过|下了)什么|你(?:的)?|顾衍|\b(?:your|diary|journal|what|did|write)\b",
                       " ", strip_human_date_references(remainder), flags=re.I).strip()
        item = {"date": hint["date"], "focus": focus[:400], "full": bool(_FULL.search(part)) or offset > 0,
                "offset": offset}
        if item["date"] not in {entry["date"] for entry in output}:
            output.append(item)
    return output


def _checked_source(payload, date, conversation_id):
    source = payload.get("source") or {}
    if (payload.get("conversation_id") != conversation_id
            or not isinstance(payload.get("identity_id"), str) or not payload["identity_id"]
            or source.get("source_namespace") != "snapshot.aiz.diary.private"
            or source.get("source_id") != "diary:private:" + date):
        raise ValueError("diary_scope_mismatch")
    if source.get("state") != "available" or source.get("eligible_source") is not True:
        return source
    permission = source.get("permission") or {}
    if (permission.get("authorized") is not True or permission.get("scope") != "owner.private_diary"
            or type(source.get("content_chars")) is not int or source["content_chars"] <= 0
            or type(source.get("adopted_at_ms")) is not int or source["adopted_at_ms"] <= 0
            or not isinstance(source.get("source_revision"), str) or not source["source_revision"]
            or not isinstance(source.get("content_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", source["content_sha256"])):
        raise ValueError("diary_identity_missing")
    return source


def _render(text, *, date, author, start, total, full):
    label = f"{date} 私人日记（{author}）"
    end = start + len(text)
    if start == 0 and end == total:
        return f"{label}：\n{text}"
    heading = f"{label}，第 {start + 1} 至 {end} 字，共 {total} 字："
    continued = (f"\n\n如需续读，请指定 {date} 日记，从第 {end + 1} 个字继续。"
                 if full and end < total else "")
    return f"{heading}\n{text}{continued}"


async def read(adapter, request, *, budget, timeout):
    """Fetch one metadata view and one bounded, same-revision source span."""
    date = request["date"]
    debug = {"date": date, "state": "unavailable"}
    if budget <= 0:
        return "", {**debug, "reason": "date_recall_budget_disabled"}
    params = {"conversation_id": adapter.conversation_id}
    url = f"{adapter.base_url}/bridge/v1/private-diary/{date}"
    headers = adapter._headers()

    async def get(query):
        response = await adapter.http_client.get(url, params=query, headers=headers)
        response.raise_for_status()
        payload = response.json()
        return payload["identity_id"], _checked_source(payload, date, adapter.conversation_id)

    try:
        async with asyncio.timeout(timeout):
            lookup = dict(params)
            if request["focus"] and not request["full"]:
                lookup["focus"] = request["focus"]
            identity, meta = await get(lookup)
            if meta["state"] != "available" or meta.get("eligible_source") is not True:
                return "", {**debug, "state": "missing" if meta.get("reason") == "diary_adoption_unavailable" else "unavailable"}
            total = meta["content_chars"]
            start, end = request["offset"], total
            if not request["full"]:
                for paragraph in meta.get("paragraphs") or []:
                    left, right = paragraph.get("start"), paragraph.get("end")
                    if type(left) is int and type(right) is int and 0 <= left < right <= total:
                        match = paragraph.get("match_start")
                        start = max(left, match - 40) if type(match) is int and left <= match < right else left
                        end = right
                        break
            # A single read is bounded by both the source API and the existing date budget.
            end = min(end, start + min(12000, budget * 4))
            if start >= end:
                return "", {**debug, "reason": "diary_offset_out_of_range"}
            source_identity, source = await get({**params, "revision": meta["source_revision"], "start": start, "end": end})
            if source.get("state") == "changed":
                return "", {**debug, "state": "changed"}
            if (source.get("state") != "available" or source.get("eligible_source") is not True
                    or source_identity != identity or any(source.get(key) != meta.get(key) for key in
                        ("source_revision", "content_sha256", "content_chars", "adopted_at_ms", "author"))
                    or source.get("range") != {"start": start, "end": end, "coordinate_unit": "unicode_codepoint"}
                    or not isinstance(source.get("text"), str) or len(source["text"]) != end - start):
                return "", {**debug, "reason": "diary_span_mismatch"}
            body = source["text"]
            if start == 0 and end == total and hashlib.sha256(body.encode()).hexdigest() != meta["content_sha256"]:
                return "", {**debug, "reason": "diary_digest_mismatch"}
            author_kind = (source.get("author") or {}).get("kind")
            if author_kind not in {"user", "assistant"}:
                return "", {**debug, "reason": "diary_author_unverified"}
            author = "主人修订" if author_kind == "user" else "顾衍原记"
            parts = {"date": date, "author": author, "start": start, "total": total, "full": request["full"]}
            text = _render(body, **parts)
            if count_tokens_approx(text) > budget:
                low, high = 0, len(body)
                while low < high:
                    length = (low + high + 1) // 2
                    if count_tokens_approx(_render(body[:length], **parts)) <= budget:
                        low = length
                    else:
                        high = length - 1
                body = body[:low]
                text = _render(body, **parts)
            if not body.strip() or count_tokens_approx(text) > budget:
                return "", {**debug, "reason": "no_diary_excerpt_fits"}
            return text, {**debug, "state": "qualified", "source_revision": source["source_revision"],
                          "content_sha256": source["content_sha256"], "excerpt_sha256": hashlib.sha256(body.encode()).hexdigest(),
                          "range": {"start": start, "end": start + len(body)},
                          "complete": start == 0 and len(body) == total, "content_chars": total,
                          "adopted_at_ms": source["adopted_at_ms"]}
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, AttributeError):
        return "", {**debug, "reason": "diary_reader_unavailable"}


async def read_all(adapter, questions, *, budget, timeout):
    per_question = max(0, (budget - len(questions)) // len(questions))
    results = await asyncio.gather(*(read(adapter, question, budget=per_question, timeout=timeout)
                                     for question in questions))
    return "\n\n".join(text for text, _ in results if text), [debug for _, debug in results]
