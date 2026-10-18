"""Offline coverage and safe continuation checks for real-length chat windows."""
from copy import deepcopy

import pytest

from memory_daily_windows import daily_memory_windows, safe_prefix_event_id, source_text_chars


def turn(index, size=280):
    return {"id": index + 1, "user_text": "润" * min(30, size),
            "assistant_text": "衍" * max(0, size - 30),
            "raw_event_ids": [index * 2 + 1, index * 2 + 2]}


def ids(window):
    return [item["id"] for item in window.turns]


@pytest.mark.parametrize("count,expected", [(20, 1), (24, 1), (200, 16)])
def test_normal_day_preserves_24_12_and_covers_tail(count, expected):
    turns = [turn(index) for index in range(count)]
    windows = daily_memory_windows(turns)
    assert len(windows) == expected
    assert {item["id"] for window in windows for item in window.turns} == set(range(1, count + 1))
    assert all(len(window.turns) <= 24 and window.source_text_chars <= 12000 for window in windows)
    if count == 200:
        assert [window.start for window in windows] == list(range(0, 181, 12))
        assert windows[-1].end == 200


def test_real_variable_lengths_not_average_and_no_source_mutation():
    turns = [turn(index, size) for index, size in enumerate([10, 10, 2000, 10, 2900, 100, 30, 2900, 10])]
    before = deepcopy(turns)
    windows = daily_memory_windows(turns, max_input_chars=3000)
    assert all(window.source_text_chars == sum(source_text_chars(item) for item in window.turns)
               for window in windows)
    assert all(window.source_text_chars <= 3000 for window in windows)
    assert {item["id"] for window in windows for item in window.turns} == set(range(1, 10))
    assert all(left.end < right.end for left, right in zip(windows, windows[1:]))
    assert turns == before
    assert all(item is turns[item["id"] - 1] for window in windows for item in window.turns)


def test_complete_raw_sources_counted_instead_of_truncated_projection():
    turns = [turn(0, 30), turn(1, 30)]
    turns[0]["_semantic_sources"] = [{"text": "完整原文" * 2000}]
    assert source_text_chars(turns[0]) == 8000
    windows = daily_memory_windows(turns, max_input_chars=6000)
    assert ids(windows[0]) == [1]
    assert windows[0].exceeds_char_target
    assert windows[0].source_text_chars == 8000
    assert turns[0]["_semantic_sources"][0]["text"] == "完整原文" * 2000
    assert ids(windows[1]) == [2]


def test_indivisible_oversize_turn_retained_whole_and_marked():
    turns = [turn(0, 20), turn(1, 18000), turn(2, 20)]
    windows = daily_memory_windows(turns)
    assert [ids(window) for window in windows] == [[1], [2], [3]]
    assert [window.exceeds_char_target for window in windows] == [False, True, False]
    assert windows[1].turns[0]["assistant_text"] == "衍" * 17970


def test_adjacent_correction_across_nominal_boundary_shares_sources():
    turns = [turn(index) for index in range(50)]
    turns[23]["user_text"] = "以后我们约定周五散步。"
    turns[24]["user_text"] = "改口了，周五不行，改成周六。"
    windows = daily_memory_windows(turns)
    combined = next(window for window in windows if {24, 25} <= set(ids(window)))
    assert combined.turns[11] is turns[23]
    assert combined.turns[12] is turns[24]


@pytest.mark.parametrize("budget", [1, 10])
def test_budgeted_runs_resume_same_windows_and_preserve_overlap(budget):
    turns = [turn(index) for index in range(200)]
    full = daily_memory_windows(turns)
    replayed = []
    cursor = 0
    while cursor < 400:
        remaining = [item for item in turns if min(item["raw_event_ids"]) > cursor]
        plan = daily_memory_windows(remaining)
        completed = plan[:budget]
        replayed.extend(ids(window) for window in completed)
        next_cursor = safe_prefix_event_id(remaining, completed[-1].next_start)
        assert next_cursor > cursor
        cursor = next_cursor
    assert replayed == [ids(window) for window in full]
    assert len(replayed) == 16


def test_variable_length_plan_resumes_even_with_one_window_budget():
    turns = [turn(index, size) for index, size in enumerate([100] * 15 + [2900, 20, 9000, 20, 1500] * 6)]
    full = daily_memory_windows(turns, max_input_chars=3000)
    remaining = turns
    resumed = []
    while remaining:
        window = daily_memory_windows(remaining, max_input_chars=3000)[0]
        resumed.append(ids(window))
        assert window.next_start > 0
        remaining = remaining[window.next_start:]
    assert resumed == [ids(window) for window in full]


def test_safe_watermark_cannot_swallow_interleaved_suffix_event():
    turns = [{"raw_event_ids": [1, 10]}, {"raw_event_ids": [5, 20]}]
    assert safe_prefix_event_id(turns, 1) == 4
    assert safe_prefix_event_id(turns, 2) == 20
    assert safe_prefix_event_id(turns, 0) == 0
    assert safe_prefix_event_id([{}], 1) == 0


@pytest.mark.parametrize("kwargs", [{"window_turns": 0}, {"stride_turns": 25},
                                   {"max_input_chars": 0}, {"stride_turns": True}])
def test_invalid_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        daily_memory_windows([turn(0)], **kwargs)


def configured_engine(tmp_path, **overrides):
    from reflection_engine import ReflectionEngine
    from test_daily_chat_memory_quality_review_v4 import make_config, patch_model
    engine = ReflectionEngine(make_config(tmp_path, **overrides))
    patch_model(engine, '{"candidates": []}')
    return engine


def raw_day(count):
    from datetime import datetime, timedelta
    from test_daily_chat_memory_quality_review_v4 import event_turn
    start = datetime.fromisoformat("2026-08-18T09:00:00+08:00")
    return [event_turn(index + 1, "user", f"我承诺以后一起散步，记住第{index + 1}件事。",
                       (start + timedelta(minutes=index)).isoformat()) for index in range(count)]


def run_day(engine, events, mode="review", buckets=None):
    import asyncio
    from test_daily_chat_memory_quality_review_v4 import FakeBuckets, FakeRawEvents, FakeTurns
    async def batch():
        result = await engine._run_daily_chat_memory_impl(buckets or FakeBuckets(),
            raw_event_store=FakeRawEvents(events), conversation_turn_store=FakeTurns([]),
            key="2026-08-18", mode=mode)
        return engine._ack_daily_chat_memory_preparation(result)
    return asyncio.run(batch())


def candidate_for(event_id):
    return {"kind": "commitment", "content": f"主人约定今后和我一起完成第{event_id}件散步计划。",
            "confidence": 0.9, "title": "一起散步", "source_event_ids": [event_id]}


def test_real_run_window_cap_partial_empty_advances_and_finishes_200_turns(tmp_path):
    engine = configured_engine(tmp_path)
    events = raw_day(200)
    seen = []

    async def extract(key, window, **kwargs):
        seen.append([item["id"] for item in window])
        return [], {}

    engine._extract_window_candidates = extract
    first = run_day(engine, events)
    assert first["status"] == "partial"
    assert first["reason"] == "window_budget_exhausted"
    assert first["cursor_updated"]
    assert first["processed_until_seq"] == 120
    assert len(seen) == 10
    second = run_day(engine, events)
    assert second["status"] == "zero_candidates"
    assert second["last_raw_event_id"] == 200
    assert len(seen) == 16
    assert set(value for window in seen for value in window) == set(range(1, 201))


def test_real_run_candidate_cap_stops_windows_and_retries_unconsumed_sources(tmp_path):
    engine = configured_engine(tmp_path, daily_chat_memory_window_turns=4,
        daily_chat_memory_window_stride_turns=2, daily_chat_memory_review_max_per_day=2)
    events = raw_day(8)
    calls = []

    async def extract(key, window, **kwargs):
        calls.append((window[0]["id"], kwargs["max_candidates"]))
        # The second window exceeds the remaining one-candidate budget.
        selected = [window[0]["id"]] if window[0]["id"] == 1 else [window[0]["id"], window[1]["id"]]
        return [candidate_for(value) for value in selected], {}

    engine._extract_window_candidates = extract
    first = run_day(engine, events)
    assert first["status"] == "partial"
    assert first["reason"] == "candidate_window_exceeds_remaining_budget"
    assert first["last_raw_event_id"] == 2
    assert len(first["candidates"]) == 1
    assert calls == [(1, 2), (3, 1)]
    second = run_day(engine, events)
    assert second["status"] == "partial"
    assert second["reason"] == "candidate_budget_exhausted"
    assert {value for item in second["candidates"] for value in item["source_event_ids"]} == {3, 4}
    assert second["last_raw_event_id"] == 4


def test_single_window_candidate_overflow_is_explicit_and_source_remains_retryable(tmp_path):
    engine = configured_engine(tmp_path, daily_chat_memory_review_max_per_day=1)
    calls = []

    async def extract(key, window, **kwargs):
        calls.append(key)
        return [candidate_for(1), candidate_for(2)], {}

    engine._extract_window_candidates = extract
    result = run_day(engine, raw_day(2))
    assert result["status"] == "partial"
    assert result["reason"] == "candidate_window_exceeds_remaining_budget"
    assert result["processed_until_seq"] == 0
    assert result["cursor_updated"] is False
    assert not engine._load_daily_chat_memory_pending()
    summary_calls = []

    async def summary(*args, **kwargs):
        summary_calls.append(True)
        return []

    engine._summarize_daily_chat_memory_windows = summary
    blocked = run_day(engine, raw_day(2))
    assert blocked["status"] == "partial"
    assert blocked["reason"] == "candidate_window_exceeds_total_budget_requires_change"
    assert blocked["required_candidate_count"] == 2
    assert len(calls) == 1
    assert not summary_calls
    engine.daily_chat_memory_review_max_per_day = 2
    resumed = run_day(engine, raw_day(2))
    assert resumed["status"] == "pending"
    assert resumed["last_raw_event_id"] == 2
    assert len(calls) == 2


def test_persistence_failure_keeps_source_watermark(tmp_path):
    engine = configured_engine(tmp_path, mode="auto")

    async def extract(key, window, **kwargs):
        return [candidate_for(1)], {}

    async def fail_write(candidates, buckets, **kwargs):
        return {"created": 0, "results": [{"id": candidates[0]["id"], "status": "failed"}]}

    engine._extract_window_candidates = extract
    engine._write_daily_chat_memory_candidates = fail_write
    result = run_day(engine, raw_day(2), mode="auto")
    assert result["status"] == "partial"
    assert result["reason"] == "candidate_persistence_failed"
    assert not result["cursor_updated"]
    assert engine.daily_chat_memory_run_cursor()["last_raw_event_id"] == 0


def test_turn_payloads_keep_full_text_before_char_window_planning():
    from reflection_engine import ReflectionEngine
    text = "完整原文" * 1000
    projected = ReflectionEngine._conversation_turn_payloads([
        {"id": 1, "user_text": text, "assistant_text": text}], limit=200)
    assert projected[0]["user_text"] == text
    assert projected[0]["assistant_text"] == text
