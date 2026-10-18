"""Complete-source daily chat windows; scheduling caps never change the plan.

The character target measures source text, not serialized prompt/token size.
One indivisible turn may exceed it and is retained whole with an explicit flag.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyMemoryWindow:
    start: int
    end: int
    next_start: int
    turns: list[dict]
    source_text_chars: int
    exceeds_char_target: bool


def source_text_chars(turn: dict) -> int:
    """Count the complete raw sources used by L1, when present.

    Legacy turn projections can have shortened display text while retaining full
    raw sources. Counting both would duplicate the same conversation material.
    """
    sources = turn.get("_semantic_sources")
    if isinstance(sources, list) and sources:
        return sum(len(str(source.get("text") or "")) for source in sources
                   if isinstance(source, dict))
    return sum(len(str(turn.get(field) or ""))
               for field in ("user_text", "assistant_text"))


def daily_memory_windows(
    turns: list[dict], *, window_turns: int = 24, stride_turns: int = 12,
    max_input_chars: int = 12000,
) -> list[DailyMemoryWindow]:
    """Plan greedy, overlapping windows without slicing a source or turn.

    Short windows retain the requested overlap when it permits forward progress;
    the next start always advances, and each later window adds unseen turns.
    The same input/config produces the same plan independently of run budgets.
    Callers can stop before a later window and resume at ``next_start`` from the
    last completed one, retaining its overlap. No watermark is written here.
    """
    for name, value in (("window_turns", window_turns), ("stride_turns", stride_turns),
                        ("max_input_chars", max_input_chars)):
        if type(value) is not int or value <= 0:
            raise ValueError("invalid_" + name)
    if stride_turns > window_turns:
        raise ValueError("stride_exceeds_window")
    lengths = [source_text_chars(turn) for turn in turns]
    overlap = window_turns - stride_turns
    windows = []
    start = 0
    while start < len(turns):
        end = start
        chars = 0
        while end < min(start + window_turns, len(turns)):
            if end > start and chars + lengths[end] > max_input_chars:
                break
            chars += lengths[end]
            end += 1
        next_start = (len(turns) if end == len(turns)
                      else max(start + 1, end - overlap))
        if end < len(turns):
            # Retain as much overlap as fits alongside the first unseen turn.
            # This makes next_start the actual next window start, also when a
            # one-window run resumes, rather than repeating only old material.
            next_chars = sum(lengths[next_start:end + 1])
            while next_start < end and next_chars > max_input_chars:
                next_chars -= lengths[next_start]
                next_start += 1
        windows.append(DailyMemoryWindow(
            start=start, end=end, next_start=next_start,
            turns=turns[start:end], source_text_chars=chars,
            exceeds_char_target=chars > max_input_chars,
        ))
        start = next_start
    return windows


def safe_prefix_event_id(turns: list[dict], stop: int) -> int:
    """Largest event watermark that cannot consume a retained suffix source.

    Turn order follows conversation time and may interleave different sessions'
    event ids. Taking the prefix's maximum alone can swallow an unprocessed
    suffix event. Missing ids cannot establish a safe positive watermark.
    """
    if not 0 <= stop <= len(turns):
        raise ValueError("invalid_prefix_stop")
    prefix_ids = []
    suffix_ids = []
    for index, turn in enumerate(turns):
        event_ids = [int(value) for value in turn.get("raw_event_ids") or []
                     if str(value).isdigit() and int(value) > 0]
        if not event_ids:
            return 0
        (prefix_ids if index < stop else suffix_ids).extend(event_ids)
    maximum = max(prefix_ids, default=0)
    return max(0, min(maximum, min(suffix_ids) - 1)) if suffix_ids else maximum
