"""Request-local retrieval input; no conversation or Memory writes."""
from __future__ import annotations

import hashlib
from typing import Any, Callable


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_recall_input(
    messages: list[dict[str, Any]], *, current_query: str,
    text_extractor: Callable[[Any], str], cleaner: Callable[[str], str],
    recent_turns: int = 4, trace_context: dict[str, Any] | None = None,
    context_max_chars: int = 3000,
) -> dict[str, Any]:
    """Use every current user, and up to N completed dialogue turns.

    A completed assistant reply is a boundary; assistant tool-call carriers,
    tool results and instruction projections are not dialogue boundaries.
    History is restricted to the input actually selected by the caller.
    Assistant text is a search cue only, never fact/alias evidence.
    """
    trace = trace_context or {}
    coverage = trace.get("coverage") or {}
    items = coverage.get("items") or []
    indexed = [item for item in items if isinstance(item, dict)
               and isinstance(item.get("message_index"), int)]
    canonical_indices = {item["message_index"] for item in indexed}
    staged_indices = {item["message_index"] for item in indexed if item.get("turn_member")}
    rows = []
    for index, item in enumerate(messages or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"} or item.get("tool_calls"):
            continue
        if indexed and index not in canonical_indices:
            continue
        text = str(text_extractor(item.get("content")) or "")
        if role == "user":
            text = cleaner(text)
        text = text.strip()
        if text:
            rows.append({"index": index, "role": role, "text": text})
    current = []
    # Skip completed assistant output only when scanning a legal continuation;
    # the caller provides the real current query in both request phases.
    last_user = next((i for i in range(len(rows) - 1, -1, -1)
                      if rows[i]["role"] == "user"
                      and rows[i]["text"] == current_query.strip()), None)
    if staged_indices:
        current = [row for row in rows if row["index"] in staged_indices and row["role"] == "user"]
        first_current = min((row["index"] for row in current), default=len(messages))
        history = [row for row in rows if row["index"] < first_current]
    elif last_user is not None:
        i = last_user
        while i >= 0 and rows[i]["role"] == "user":
            current.insert(0, rows[i])
            i -= 1
        history = rows[:i + 1]
    else:
        history = []
    if not current and current_query.strip():
        current = [{"index": -1, "role": "user", "text": current_query.strip()}]
    groups = []
    pending = []
    for row in history:
        if row["role"] == "user":
            pending.append(row)
        elif pending:
            groups.append([*pending, row])
            pending = []
    requested = max(0, int(recent_turns))
    selected = groups[-requested:] if requested else []
    recent = [row for group in selected for row in group]
    # Preserve all current input. Only background is bounded here; long
    # current input is split into embedding units by the embedding engine.
    available = max(0, int(context_max_chars))
    kept = []
    omitted = 0
    for row in reversed(recent):
        text = row["text"]
        if not available:
            omitted += len(text)
            continue
        take = min(len(text), available)
        kept.insert(0, {**row, "text": text[:take]})
        available -= take
        omitted += len(text) - take
    q_current = "\n".join(row["text"] for row in current)
    background = "\n".join(f'{row["role"]}: {row["text"]}' for row in kept)
    q_context = q_current if not background else f"{q_current}\n\nRecent dialogue:\n{background}"
    query_views = list(dict.fromkeys(q for q in (q_current, q_context) if q))
    current_indices = [row["index"] for row in current if row["index"] >= 0]
    staged_ids = [str(item.get("event_id") or "") for item in items
                  if isinstance(item, dict) and item.get("turn_member") and item.get("event_id")]
    metadata = {
        "current_message_indices": current_indices,
        "current_message_count": len(current),
        "staged_user_event_ids": list(dict.fromkeys(staged_ids)),
        "recent_message_indices": [row["index"] for row in kept],
        "recent_turns_requested": requested,
        "recent_turns_selected": len(selected),
        "recent_context_truncated_chars": omitted,
        "current_chars": len(q_current), "context_chars": len(background),
        "query_views": [{"kind": "current" if i == 0 else "context",
                         "chars": len(q), "sha256": _sha(q)}
                        for i, q in enumerate(query_views)],
        "conversation_id": str(trace.get("conversation_id") or ""),
        "turn_id": str(trace.get("turn_id") or ""),
        "context_revision": coverage.get("context_revision"),
        "input_source": "canonical_coverage" if indexed else "legacy_selected_messages",
        "current_input_complete": (
            (last_user is not None and bool(current_indices)
             and (not staged_indices or len(current_indices) == len(staged_indices)))
            or not current_query.strip()
        ),
    }
    return {
        "q_current": q_current, "q_context": q_context,
        "query_views": query_views, "current_indices": current_indices,
        "recent_indices": metadata["recent_message_indices"], "metadata": metadata,
        "current_messages": current, "recent_messages": kept,
    }
