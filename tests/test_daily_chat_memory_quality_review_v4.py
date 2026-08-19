"""脱敏专项 V4：高召回 Review 与可解释过滤。

覆盖：窗口全覆盖、软警告保留、hard/soft 分层、echo 判定修正、rejected 语义、
cursor/watermark、run audit、defer、Review/Auto 门禁分离、V3 回归。
合成 fixture，不读取生产正文，不调用真实模型。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from reflection_engine import ReflectionEngine


def make_config(tmp_path: Path, *, mode: str = "review", **overrides) -> dict:
    state_dir = tmp_path / "state"
    buckets_dir = tmp_path / "buckets"
    config = {
        "identity": {
            "ai_name": "Test AI",
            "user_name": "Test User",
            "user_display_name": "Test User",
            "user_aliases": ["owner"],
        },
        "state_dir": str(state_dir),
        "buckets_dir": str(buckets_dir),
        "reflection": {
            "enabled": True,
            "daily_chat_memory_mode": mode,
            "daily_chat_memory_pending_path": str(state_dir / "daily_chat_memory_candidates.json"),
            "daily_chat_memory_requests_path": str(state_dir / "daily_chat_memory_requests.json"),
            "daily_chat_memory_run_audit_path": str(state_dir / "daily_chat_memory_run_audit.json"),
            "daily_chat_memory_summary_enabled": False,
            "daily_chat_memory_max_per_day": 10,
            "daily_chat_memory_review_max_per_day": 10,
            "daily_activity_summary_enabled": True,
            **overrides,
        },
    }
    return config


class FakeBuckets:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.created: list[str] = []

    async def list_all(self, include_archive: bool = False):
        return list(self.items.values())

    async def get(self, bucket_id: str):
        return self.items.get(bucket_id)

    async def create(self, **kwargs):
        bucket_id = str(kwargs["bucket_id"])
        item = {"id": bucket_id, "content": kwargs.get("content") or "", "metadata": dict(kwargs.get("extra_metadata") or {})}
        self.items[bucket_id] = item
        self.created.append(bucket_id)
        return bucket_id


class FakeTurns:
    def __init__(self, turns: list[dict] | None = None):
        self.turns = turns or []

    def list_conversation_turns_between(self, **kwargs):
        return self.turns


class FakeRawEvents:
    def __init__(self, events: list[dict] | None = None):
        self.events = events or []

    def list_events_between(self, **kwargs):
        return self.events


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


def patch_model(engine: ReflectionEngine, content: str):
    calls = []

    async def fake_create_completion(client, *, model, messages, max_tokens, temperature, use_daily_client):
        payload_text = ""
        for message in messages:
            if message.get("role") == "user":
                payload_text = message.get("content") or ""
        calls.append(json.loads(payload_text))
        return FakeResponse(content)

    engine._daily_chat_memory_model_client = lambda *, candidate: (object(), "fake-model", True)
    engine._daily_chat_memory_create_completion = fake_create_completion
    return calls


def model_candidates_json(candidates: list[dict]) -> str:
    return json.dumps({"candidates": candidates}, ensure_ascii=False)


def event_turn(event_id: int, role: str, text: str, when: str) -> dict:
    return {
        "id": event_id,
        "role": role,
        "text": text,
        "session_id": "s",
        "created_at": when,
        "metadata": {"round_id": event_id},
    }


def user_turn(turn_id: int, text: str, when: str) -> dict:
    return {"id": turn_id, "session_id": "s", "created_at": when, "user_text": text, "assistant_text": "收到。"}


def seed_pending(engine: ReflectionEngine, items: list[dict]):
    path = Path(engine.daily_chat_memory_pending_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items, "cursor": {}}, ensure_ascii=False, indent=2), encoding="utf-8")


def pending_bytes(engine: ReflectionEngine) -> bytes:
    return Path(engine.daily_chat_memory_pending_path).read_bytes()


def run_raw_review(engine: ReflectionEngine, events: list[dict]):
    return asyncio.run(
        engine.run_daily_chat_memory(
            FakeBuckets(),
            conversation_turn_store=FakeTurns([]),
            raw_event_store=FakeRawEvents(events),
            key="2026-08-18",
            mode="review",
            force=True,
        )
    )


def run_turn_review(engine: ReflectionEngine, turns: list[dict], mode: str = "review"):
    return asyncio.run(
        engine.run_daily_chat_memory(
            FakeBuckets(),
            conversation_turn_store=FakeTurns(turns),
            key="2026-08-18",
            mode=mode,
            force=True,
        )
    )


# ---------------------------------------------------------------------------
# 1. 全天覆盖：开头/中间/结尾 + 超过 40 条后尾部不丢失
# ---------------------------------------------------------------------------

def test_windows_cover_beginning_middle_end_with_model(tmp_path):
    """有意义内容分布全天开头/中间/结尾，窗口全部检查，候选全部保留。"""
    engine = ReflectionEngine(
        make_config(
            tmp_path,
            mode="review",
            daily_chat_memory_window_turns=4,
            daily_chat_memory_window_stride_turns=2,
            daily_chat_memory_max_windows_per_run=10,
        )
    )
    turns = [
        user_turn(1, "以后默认先说明边界", "2026-08-18T09:00:00+08:00"),
        user_turn(2, "今天天气不错", "2026-08-18T09:05:00+08:00"),
        user_turn(3, "晚上想吃什么", "2026-08-18T09:10:00+08:00"),
        user_turn(4, "随便吧", "2026-08-18T09:15:00+08:00"),
        user_turn(5, "我承诺以后每周二一起散步", "2026-08-18T12:00:00+08:00"),
        user_turn(6, "哦对了记得买牛奶", "2026-08-18T12:05:00+08:00"),
        user_turn(7, "今天下班好晚", "2026-08-18T18:00:00+08:00"),
        user_turn(8, "我发现自己其实很怕孤独", "2026-08-18T18:10:00+08:00"),
    ]
    calls = patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "默认先说明边界",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
                {"should_write": True, "kind": "commitment", "title": "每周二一起散步",
                 "content": "主人承诺每周二晚上一起散步。", "domain": "general", "confidence": 0.8, "source_turn_ids": [5]},
                {"should_write": True, "kind": "self_insight", "title": "怕孤独",
                 "content": "主人意识到自己其实很怕孤独。", "domain": "general", "confidence": 0.7, "source_turn_ids": [8]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    kinds = {candidate["kind"] for candidate in result["candidates"]}
    assert kinds >= {"stable_preference", "commitment", "self_insight"}
    # 开头/中间/结尾候选全部进入 Review（窗口全部被检查）
    assert len(calls) >= 2  # 8 轮 / 窗口4 / stride2 → 至少 2 个模型调用


def test_over_40_turns_tail_content_not_lost(tmp_path):
    """超过 40 条轮次后，尾部重要内容不会因为采样消失。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(n, f"普通聊天内容 {n}", f"2026-08-18T{9 + n // 60}:{(n * 7) % 60:02d}:00+08:00") for n in range(1, 51)]
    # 尾部一条重要承诺（第 50 条）
    turns.append(user_turn(50, "我承诺以后每周日都陪你散步", "2026-08-18T23:50:00+08:00"))
    calls = patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "commitment", "title": "每周日散步",
                 "content": "主人承诺每周日晚上一起散步。", "domain": "general", "confidence": 0.8, "source_turn_ids": [50]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    # 尾部候选被保留
    assert any(candidate["kind"] == "commitment" for candidate in result["candidates"])
    # 全天分窗：多窗口被检查，而不是只挑 40 条
    assert len(calls) >= 2


def test_pure_smalltalk_allows_zero_candidates(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(n, "在吗", f"2026-08-18T10:0{n}:00+08:00") for n in range(1, 4)]
    result = run_turn_review(engine, turns)
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"


# ---------------------------------------------------------------------------
# 2. 软警告保留 / hard reject 分层
# ---------------------------------------------------------------------------

def test_low_confidence_kept_in_review_but_not_applied_in_auto(tmp_path):
    """低置信度候选：Review 保留并显示 flag；Auto 不应用。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "x",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.5, "source_turn_ids": [1]},
            ]
        ),
    )
    review = run_turn_review(engine, turns)
    assert review["status"] == "pending"
    assert "low_confidence" in review["candidates"][0]["soft_flags"]

    auto_engine = ReflectionEngine(make_config(tmp_path, mode="auto"))
    patch_model(
        auto_engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "x",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.5, "source_turn_ids": [1]},
            ]
        ),
    )
    auto = run_turn_review(auto_engine, turns, mode="auto")
    assert auto["status"] == "zero_candidates"


def test_possibly_generic_transient_duplicate_not_silently_dropped(tmp_path):
    """possibly_generic / possibly_transient / possible_duplicate 均保留并打标。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "值得记住的内容",
                 "content": "主人希望以后默认先说明边界，这可能是只在这段时间有效的偏好。",
                 "domain": "general", "confidence": 0.7, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    flags = result["candidates"][0]["soft_flags"]
    assert "possibly_generic" in flags
    assert "possibly_transient" in flags


def test_hard_rejects_still_dropped_with_counts(tmp_path):
    """来源不存在 / 内部转储 / 伪造来源 / 空建议 → 硬拒绝且 audit 计数。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "a",
                 "content": "无来源候选", "domain": "general", "confidence": 0.8},
                {"should_write": True, "kind": "stable_preference", "title": "b",
                 "content": "伪造来源候选", "domain": "general", "confidence": 0.8, "source_turn_ids": [999]},
                {"should_write": True, "kind": "stable_preference", "title": "c",
                 "content": "", "domain": "general", "confidence": 0.8, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "zero_candidates"
    assert result["hard_rejects"].get("missing_or_fabricated_source", 0) >= 1
    assert result["hard_rejects"].get("empty_proposed_memory", 0) >= 1


def test_necessary_factual_overlap_not_echo(tmp_path):
    """必要事实关键词重合不是 echo：建议记忆与原文共享事实词但独立表述 → 正常保留。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "我希望你以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "默认先说明边界",
                 "content": "主人以后默认先说明边界，这是她沟通时的重要偏好。",
                 "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    flags = result["candidates"][0]["soft_flags"]
    assert "needs_owner_edit" not in flags
    assert "excerpt_overlap" not in flags


def test_wholesale_copy_requires_edit_before_approve(tmp_path):
    """整段照抄：Review 保留 + 标记；编辑成独立建议后才可 approve。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "我希望你以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "x",
                 "content": "我希望你以后默认先说明边界", "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    candidate = result["candidates"][0]
    assert "needs_owner_edit" in candidate["soft_flags"]

    buckets = FakeBuckets()
    blocked = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="confirm", request_id="rq-edit-1")
    )
    assert blocked["results"][0]["status"] == "needs_owner_edit"
    assert buckets.created == []

    # 主人编辑后再 approve
    edited = asyncio.run(
        engine.confirm_daily_chat_memory(
            [candidate["id"]],
            buckets,
            action="confirm",
            request_id="rq-edit-2",
            edits={candidate["id"]: {"content": "主人希望以后默认先说明边界，重要决定先商量。"}},
        )
    )
    assert edited["results"][0]["status"] == "created"
    assert buckets.created == [candidate["id"]]


def test_assistant_vague_comfort_no_candidate(tmp_path):
    """assistant 泛泛安慰不产生长期候选（纯安慰窗口被确定性噪音预过滤跳过）。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [{"id": 1, "session_id": "s", "created_at": "2026-08-18T10:00:00+08:00",
              "user_text": "", "assistant_text": "别难过，我在呢，抱抱你。"}]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "key_event", "title": "x",
                 "content": "安慰类内容", "domain": "general", "confidence": 0.7, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "zero_candidates"


def test_natural_language_commitment_enters_review(tmp_path):
    """自然语言表达的明确承诺（无固定关键词）可进入 Review。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后每个周末我都会陪你去看电影，雷打不动", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "commitment", "title": "周末看电影",
                 "content": "主人承诺以后每个周末都陪顾衍看电影。", "domain": "general", "confidence": 0.78, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    assert result["candidates"][0]["kind"] == "commitment"


def test_meaningful_episodic_key_event_enters_review(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "今天是我们第一次一起看日出，特别值得记住", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "key_event", "title": "第一次一起看日出",
                 "content": "主人与顾衍第一次一起看日出，是重要的共同经历。", "domain": "general", "confidence": 0.8, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    assert result["candidates"][0]["kind"] == "key_event"


# ---------------------------------------------------------------------------
# 3. rejected 语义：相似只警告，精确重复幂等抑制
# ---------------------------------------------------------------------------

def test_similar_to_rejected_warns_not_blackhole(tmp_path):
    """与历史 rejected 相似但来源不同的候选：保留并标记，不形成永久黑洞。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    rejected_item = {
        "id": "old-rejected-1",
        "date": "2026-08-17",
        "status": "rejected",
        "created_at": "2026-08-17T16:00:00+00:00",
        "candidate": {
            "id": "old-rejected-1",
            "date": "2026-08-17",
            "kind": "stable_preference",
            "content": "主人希望以后默认先说明边界。",
            "source_turn_ids": [100],
            "source_event_ids": [1000],
            "source_hash": "oldhash1",
            "source_verification": "verified",
            "mode": "review",
        },
    }
    seed_pending(engine, [rejected_item])
    turns = [user_turn(1, "我希望你以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "默认先说明边界",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert "previously_rejected_similar" in candidate["soft_flags"]
    assert candidate["source_turn_ids"] == [1]  # 新来源未被旧 rejected 吞掉


def test_exact_duplicate_suppressed_idempotently(tmp_path):
    """同一 source_hash + 同一 kind 的完全重复被幂等抑制。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "默认先说明边界",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
            ]
        ),
    )
    first = run_turn_review(engine, turns)
    assert first["status"] == "pending"
    assert len(first["candidates"]) == 1

    # 相同区间重跑（force）→ 精确重复被抑制，不重复建 pending
    second = run_turn_review(engine, turns)
    assert second["status"] == "zero_candidates"
    assert second["hard_rejects"].get("exact_duplicate", 0) >= 1


# ---------------------------------------------------------------------------
# 4. cursor / watermark / run audit
# ---------------------------------------------------------------------------

def test_zero_candidates_advances_watermark(tmp_path):
    """zero_candidates 也推进 watermark 到实际 source_end_seq，不再每小时重复扫描。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    events = [
        event_turn(1, "user", "在吗", "2026-08-18T09:00:00+08:00"),
        event_turn(2, "assistant", "在呢", "2026-08-18T09:00:05+08:00"),
        event_turn(3, "user", "晚安", "2026-08-18T23:00:00+08:00"),
    ]
    patch_model(engine, json.dumps({"candidates": []}, ensure_ascii=False))
    result = run_raw_review(engine, events)
    assert result["status"] == "zero_candidates"
    assert result["cursor_updated"] is True
    assert result["last_raw_event_id"] == 3
    cursor = engine.daily_chat_memory_run_cursor("default")
    assert cursor["last_raw_event_id"] == 3


def test_new_messages_same_day_processed_next_run(tmp_path):
    """当天新增消息会在下一 run 处理（基于 seq 的 watermark）。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    patch_model(engine, json.dumps({"candidates": []}, ensure_ascii=False))
    first_events = [event_turn(n, "user", f"内容{n}", f"2026-08-18T10:0{n}:00+08:00") for n in range(1, 4)]
    first = run_raw_review(engine, first_events)
    assert first["status"] == "zero_candidates"
    assert first["last_raw_event_id"] == 3

    # 新增事件 4-5
    more_events = first_events + [
        event_turn(4, "user", "我承诺以后每周二散步", "2026-08-18T20:00:00+08:00"),
        event_turn(5, "assistant", "好", "2026-08-18T20:00:05+08:00"),
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "commitment", "title": "每周二散步",
                 "content": "主人承诺每周二晚上一起散步。", "domain": "general", "confidence": 0.8, "source_event_ids": [4]},
            ]
        ),
    )
    second = asyncio.run(
        engine.run_daily_chat_memory(
            FakeBuckets(),
            conversation_turn_store=FakeTurns([]),
            raw_event_store=FakeRawEvents(more_events),
            key="2026-08-18",
            mode="review",
            force=False,  # 从 watermark 继续
        )
    )
    assert second["status"] == "pending"
    assert second["source_start_seq"] == 3  # 只处理 >3 的新区间
    assert second["candidates"][0]["source_event_ids"] == [4]


def test_model_failure_does_not_advance_failed_range(tmp_path):
    """窗口模型失败：status=partial，cursor 不越过失败区间。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    events = [event_turn(n, "user", f"内容{n}", f"2026-08-18T10:0{n}:00+08:00") for n in range(1, 6)]
    patch_model(engine, json.dumps({"candidates": []}, ensure_ascii=False))

    async def broken_extract_window(**kwargs):
        raise RuntimeError("model timeout")

    engine._extract_window_candidates = broken_extract_window
    result = run_raw_review(engine, events)
    assert result["status"] == "partial"
    assert result["error_category"] == "window_extraction_failed"
    assert result["cursor_updated"] is False
    cursor = engine.daily_chat_memory_run_cursor("default")
    assert cursor["last_raw_event_id"] == 0


def test_partial_not_faked_as_success(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    events = [event_turn(n, "user", f"内容{n}", f"2026-08-18T10:0{n}:00+08:00") for n in range(1, 6)]
    patch_model(engine, json.dumps({"candidates": []}, ensure_ascii=False))

    async def flaky_extract_window(**kwargs):
        raise RuntimeError("rate limit")

    engine._extract_window_candidates = flaky_extract_window
    result = run_raw_review(engine, events)
    assert result["status"] == "partial"
    audit = engine.list_daily_chat_memory_run_audit(limit=1)[-1]
    assert audit["status"] == "partial"
    assert audit["error_category"] == "window_extraction_failed"


def test_retry_same_range_no_double_charge_or_duplicate_pending(tmp_path):
    """同一区间重试：模型调用按窗口计、候选不重复建 pending（精确重复抑制）。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    events = [event_turn(n, "user", "以后默认先说明边界", f"2026-08-18T10:0{n}:00+08:00") for n in range(1, 4)]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "默认先说明边界",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.75, "source_event_ids": [1]},
            ]
        ),
    )
    first = run_raw_review(engine, events)
    assert first["status"] == "pending"

    second = run_raw_review(engine, events)
    # 精确重复被抑制 → 零新增 pending
    assert second["status"] == "zero_candidates"
    assert second["hard_rejects"].get("exact_duplicate", 0) >= 1
    items = engine._load_daily_chat_memory_pending()
    assert len([item for item in items if item.get("status") == "pending"]) == 1


def test_run_audit_counts_match_processing(tmp_path):
    """run audit 计数与实际处理一致。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    patch_model(
        engine,
        model_candidates_json(
            [
                {"should_write": True, "kind": "stable_preference", "title": "a",
                 "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
                {"should_write": True, "kind": "stable_preference", "title": "b",
                 "content": "无来源的候选", "domain": "general", "confidence": 0.75},
            ]
        ),
    )
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    audit = engine.list_daily_chat_memory_run_audit(limit=1)[-1]
    assert audit["date"] == "2026-08-18"
    assert audit["eligible_turn_count"] == 1
    assert audit["model_candidate_count"] == 2
    assert audit["hard_rejects"].get("missing_or_fabricated_source", 0) >= 1
    assert audit["pending_count"] == 1
    assert audit["status"] == "success"
    assert audit["run_id"]


# ---------------------------------------------------------------------------
# 5. Review/Auto 门禁分离 + 只读 + 隔离 + 审计动作
# ---------------------------------------------------------------------------

def test_review_and_auto_use_different_gates(tmp_path):
    """同一候选在 Review 保留（带 flag），在 Auto 被严格丢弃。"""
    candidate_json = model_candidates_json(
        [
            {"should_write": True, "kind": "stable_preference", "title": "x",
             "content": "主人希望以后默认先说明边界。", "domain": "general", "confidence": 0.5, "source_turn_ids": [1]},
        ]
    )
    review_engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    patch_model(review_engine, candidate_json)
    review = run_turn_review(review_engine, [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")])
    assert review["status"] == "pending"

    auto_engine = ReflectionEngine(make_config(tmp_path, mode="auto"))
    patch_model(auto_engine, candidate_json)
    auto = run_turn_review(auto_engine, [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")], mode="auto")
    assert auto["status"] == "zero_candidates"


def test_get_pending_and_runs_read_only(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    run_turn_review(engine, turns)
    before = pending_bytes(engine)
    for _ in range(3):
        assert engine.list_daily_chat_memory_pending()
        assert engine.list_daily_chat_memory_run_audit(limit=10)
        assert engine.daily_chat_memory_run_cursor("default")
    assert pending_bytes(engine) == before


def test_pending_not_in_downstream_materials(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    result = run_turn_review(engine, turns)
    assert result["status"] == "pending"
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-18", daily_chat_memory_candidates=result["candidates"]
    ) == []


def test_approve_writes_bucket_with_source_hash(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    result = run_turn_review(engine, turns)
    candidate = result["candidates"][0]
    buckets = FakeBuckets()
    confirmed = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="confirm", request_id="rq-approve")
    )
    assert confirmed["results"][0]["status"] == "created"
    metadata = buckets.items[candidate["id"]]["metadata"]
    assert metadata["source_hash"] == candidate["source_hash"]
    assert metadata["source_conversation_turn_ids"] == [1]
    # 原文不写入 bucket
    assert "original_excerpt" not in metadata


def test_reject_defer_edit_auditable(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    result = run_turn_review(engine, turns)
    candidate = result["candidates"][0]
    buckets = FakeBuckets()

    deferred = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="defer", request_id="rq-defer")
    )
    assert deferred["results"][0]["status"] == "deferred"
    items = engine._load_daily_chat_memory_pending()
    item = next(i for i in items if i["id"] == candidate["id"])
    assert item["status"] == "deferred"
    assert item["action_source"] == "owner"
    assert item["request_id"] == "rq-defer"
    assert buckets.created == []

    # 第二条候选：reject 可审计
    turns2 = [user_turn(2, "我承诺以后每周二散步", "2026-08-18T10:05:00+08:00")]
    result2 = run_turn_review(engine, turns2)
    candidate2 = result2["candidates"][0]
    rejected = asyncio.run(
        engine.confirm_daily_chat_memory([candidate2["id"]], buckets, action="reject", request_id="rq-reject", reject_reason="too_generic")
    )
    assert rejected["results"][0]["status"] == "rejected"
    item2 = next(i for i in engine._load_daily_chat_memory_pending() if i["id"] == candidate2["id"])
    assert item2["reject_reason"] == "too_generic"
    assert item2["rejected_at"]
    assert buckets.created == []


def test_v3_sanitizer_source_preview_request_id_rate_limit_regression(tmp_path):
    """V3 安全地基回归：sanitizer / source-preview / request_id 幂等 / 限流。"""
    engine = ReflectionEngine(
        make_config(tmp_path, mode="review", daily_chat_memory_confirm_rate_limit_per_minute=1)
    )
    turns = [user_turn(1, "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]
    result = run_turn_review(engine, turns)
    candidate = result["candidates"][0]
    buckets = FakeBuckets()

    # request_id 幂等
    first = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="reject", request_id="rq-reg-1", reject_reason="wrong")
    )
    second = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="reject", request_id="rq-reg-1", reject_reason="wrong")
    )
    assert second["idempotent_replay"] is True
    assert second["rejected"] == 1

    # 限流：rate limit = 1/min，第二次真实操作被限
    turns2 = [user_turn(2, "我承诺以后每周二散步", "2026-08-18T10:05:00+08:00")]
    result2 = run_turn_review(engine, turns2)
    assert result2["status"] == "pending"
    candidate2 = result2["candidates"][0]
    third = asyncio.run(
        engine.confirm_daily_chat_memory([candidate2["id"]], buckets, action="reject", request_id="rq-reg-3")
    )
    assert third["status"] == "rate_limited"

    # sanitizer 回归
    assert engine._daily_chat_memory_owner_text('<silent mood="x" as="y" reason="z"></silent> 你好') == "你好"

    # source-preview 返回净化文本
    seed_pending(
        engine,
        [
            {
                "id": "preview-cand",
                "date": "2026-08-18",
                "status": "pending",
                "created_at": "2026-08-18T16:00:00+00:00",
                "candidate": {
                    "id": "preview-cand",
                    "date": "2026-08-18",
                    "kind": "key_event",
                    "content": "x",
                    "source_event_ids": [1],
                    "source_verification": "verified",
                    "source_hash": "abc",
                    "mode": "review",
                },
            }
        ],
    )
    preview = asyncio.run(
        engine.daily_chat_memory_source_preview(
            "preview-cand",
            raw_event_store=FakeRawEvents([event_turn(1, "user", "以后默认先说明边界", "2026-08-18T10:00:00+08:00")]),
        )
    )
    assert preview["status"] == "ok"
    assert preview["events"][0]["text"] == "以后默认先说明边界"
    assert preview["events"][0]["truncated"] is False
