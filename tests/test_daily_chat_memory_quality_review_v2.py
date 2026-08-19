"""脱敏专项 v2：候选质量修复 + Review 安全 + 审核闭环门禁。

使用合成 fixture，不读取生产正文，不调用真实模型。
覆盖：两级压缩修复、来源精确/无回退、原文摘录分离、GET 只读、request_id
幂等、限流、reject 结构化原因、pending 隔离、legacy 兼容、e5fa725 回归。
"""

from __future__ import annotations

import asyncio
import hashlib
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
            "daily_chat_memory_summary_enabled": False,
            "daily_chat_memory_max_per_day": 4,
            "daily_chat_memory_review_max_per_day": 4,
            "daily_activity_summary_enabled": True,
            **overrides,
        },
    }
    return config


class FakeBuckets:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.created: list[str] = []
        self.metadata_by_id: dict[str, dict] = {}

    async def list_all(self, include_archive: bool = False):
        return list(self.items.values())

    async def get(self, bucket_id: str):
        return self.items.get(bucket_id)

    async def create(self, **kwargs):
        bucket_id = str(kwargs["bucket_id"])
        extra = dict(kwargs.get("extra_metadata") or {})
        metadata = {
            "id": bucket_id,
            "source": kwargs.get("source") or "daily_chat_memory",
            "from_daily_chat": True,
            "event_date": kwargs.get("date") or "",
            "daily_chat_memory_candidate_id": bucket_id,
            "tags": list(kwargs.get("tags") or []),
            "domain": list(kwargs.get("domain") or []),
            "confidence": kwargs.get("confidence", 0.7),
            **extra,
        }
        item = {"id": bucket_id, "content": kwargs.get("content") or "", "metadata": metadata}
        self.items[bucket_id] = item
        self.metadata_by_id[bucket_id] = metadata
        self.created.append(bucket_id)
        return bucket_id


class FakeTurns:
    def __init__(self, turns: list[dict] | None = None):
        self.turns = turns or [
            {
                "id": 11,
                "session_id": "fixture-session",
                "created_at": "2026-08-14T12:00:00+08:00",
                "user_text": "以后默认先说明边界",
                "assistant_text": "收到，我会记住这个偏好。",
            }
        ]

    def list_conversation_turns_between(self, **kwargs):
        return self.turns


class FakeRawEvents:
    def __init__(self, events: list[dict] | None = None):
        self.events = events or []

    def list_events_between(self, **kwargs):
        return self.events


class FakeChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


def patch_model(engine: ReflectionEngine, script: list[dict]):
    """Patch the model completion with scripted responses and capture payloads.

    script: list of {"content": <json string>} consumed in order.
    """
    captured: list[dict] = []

    async def fake_create_completion(client, *, model, messages, max_tokens, temperature, use_daily_client):
        payload_text = ""
        for message in messages:
            if message.get("role") == "user":
                payload_text = message.get("content") or ""
        captured.append({"model": model, "payload": payload_text})
        step = script[min(len(captured) - 1, len(script) - 1)]
        return FakeResponse(step["content"])

    engine._daily_chat_memory_model_client = lambda *, candidate: (object(), "fake-model", True)
    engine._daily_chat_memory_create_completion = fake_create_completion
    return captured


def run_memory(engine: ReflectionEngine, buckets: FakeBuckets, mode: str, **kwargs):
    return asyncio.run(
        engine.run_daily_chat_memory(
            buckets,
            conversation_turn_store=FakeTurns(kwargs.get("turns")),
            raw_event_store=kwargs.get("raw_event_store"),
            key=kwargs.get("key") or "2026-08-14",
            mode=mode,
            force=True,
        )
    )


def seed_pending(engine: ReflectionEngine, items: list[dict]):
    items = list(items)
    payload = {"items": items, "cursor": {}}
    path = Path(engine.daily_chat_memory_pending_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def pending_bytes(engine: ReflectionEngine) -> bytes:
    return Path(engine.daily_chat_memory_pending_path).read_bytes()


# ---------------------------------------------------------------------------
# A. 生成质量：稳定偏好/承诺产生候选；两级压缩修复；原文摘录分离
# ---------------------------------------------------------------------------

def test_model_candidate_from_stable_preference_with_original_excerpt(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界，不要擅自写诗",
            "assistant_text": "好，我记住了。",
        }
    ]
    captured = patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "以后默认先说明边界",
                                "content": "主人希望以后默认先说明边界，不要擅自写诗。",
                                "original_excerpt": "我希望你以后默认先说明边界，不要擅自写诗",
                                "domain": "general",
                                "tags": ["stable_preference"],
                                "importance": 5,
                                "confidence": 0.72,
                                "source_turn_ids": [1],
                                "reason": "稳定偏好影响以后承接方式",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)

    assert result["status"] == "pending"
    assert result["candidates"]
    candidate = result["candidates"][0]
    assert candidate["kind"] == "stable_preference"
    assert candidate["proposed_memory"] == candidate["content"]
    assert candidate["proposed_memory"] != candidate["original_excerpt"]
    assert candidate["original_excerpt"] == "我希望你以后默认先说明边界，不要擅自写诗"
    assert candidate["source_turn_ids"] == [1]
    assert candidate["source_verification"] == "verified"
    assert candidate["source_hash"]
    assert candidate["candidate_type"] == "stable_preference"
    assert captured[0]["payload"]  # model path actually ran


def test_extraction_window_payload_replays_original_turns(tmp_path):
    """V4 分窗抽取：每个窗口的 payload 必须包含该窗口真实原文轮次。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": n,
            "session_id": "s",
            "created_at": f"2026-08-14T{10 + n}:00:00+08:00",
            "user_text": f"普通聊天内容 {n}",
            "assistant_text": f"回复 {n}",
        }
        for n in range(1, 6)
    ]
    captured = patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {"candidates": [{"should_write": True, "kind": "key_event", "content": "x", "source_turn_ids": [2]}]},
                    ensure_ascii=False,
                )
            }
        ],
    )
    candidates, meta = asyncio.run(
        engine._extract_daily_chat_memory_candidates(
            "2026-08-14",
            turns,
        )
    )
    assert candidates
    assert meta["model_call_count"] == 1
    assert meta["partial"] is False
    user_payload = json.loads(captured[0]["payload"])
    assert user_payload["window"]["index"] == 1
    assert user_payload["conversation_turns"], "window payload must contain original turns"
    assert {int(turn["id"]) for turn in user_payload["conversation_turns"]} == {1, 2, 3, 4, 5}


def test_ordinary_debugging_chat_returns_zero_candidates_heuristic(tmp_path):
    """普通部署排错聊天（无稳定决定）不得产生候选。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "今天部署报错了，我修了一下重启了服务",
            "assistant_text": "好的，日志里看是端口冲突。",
        },
        {
            "id": 2,
            "session_id": "s",
            "created_at": "2026-08-14T10:05:00+08:00",
            "user_text": "现在正常了吗",
            "assistant_text": "正常了。",
        },
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"
    assert buckets.created == []


def test_ordinary_debugging_chat_returns_zero_candidates_model(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "今天部署报错，我修了一下重启了服务",
            "assistant_text": "好的。",
        }
    ]
    patch_model(engine, [{"content": json.dumps({"candidates": []}, ensure_ascii=False)}])
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"


def test_insufficient_material_returns_zero_candidates(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T23:00:00+08:00",
            "user_text": "晚安",
            "assistant_text": "晚安，早点休息。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"


def test_proposed_memory_and_original_excerpt_are_separate(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "默认先说明边界",
                                "content": "主人希望以后默认先说明边界。",
                                "original_excerpt": "我希望你以后默认先说明边界",
                                "domain": "general",
                                "tags": ["stable_preference"],
                                "confidence": 0.75,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]
    # 建议记忆是改写正文，摘录是逐字原文，两者必须独立字段且不同。
    assert candidate["proposed_memory"] != candidate["original_excerpt"]
    assert "以后默认先说明边界" in candidate["original_excerpt"]


# ---------------------------------------------------------------------------
# B. 来源精度：缺失/伪造丢弃，禁止全天回退
# ---------------------------------------------------------------------------

def test_source_ids_missing_dropped_no_all_day_fallback(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": n,
            "session_id": "s",
            "created_at": f"2026-08-14T{10 + n}:00:00+08:00",
            "user_text": f"内容 {n}",
            "assistant_text": f"回复 {n}",
        }
        for n in range(1, 4)
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "key_event",
                                "title": "无来源候选",
                                "content": "没有来源的候选内容",
                                "original_excerpt": "内容 1",
                                "domain": "general",
                                "confidence": 0.8,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    # 模型漏填来源 → 丢弃；绝不回退为全天 id。
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"
    assert engine._load_daily_chat_memory_pending() == []


def test_fabricated_source_ids_dropped(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "x",
                                "content": "以后默认先说明边界",
                                "original_excerpt": "以后默认先说明边界",
                                "domain": "general",
                                "confidence": 0.8,
                                "source_turn_ids": [999],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "zero_candidates"


def test_model_excerpt_echo_kept_with_needs_owner_edit(tmp_path):
    """V4：模型整段照抄原文时，候选保留在 Review 并标记 needs_owner_edit，
    不再静默丢弃；未编辑前 approve 被拦截。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "x",
                                "content": "以后默认先说明边界",
                                "domain": "general",
                                "confidence": 0.75,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert "needs_owner_edit" in candidate["soft_flags"]
    assert "excerpt_overlap" in candidate["soft_flags"]
    # 未编辑 → approve 被拦截
    confirmed = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="confirm", request_id="rq-echo-1")
    )
    assert confirmed["results"][0]["status"] == "needs_owner_edit"
    assert buckets.created == []


def test_auto_derived_excerpt_when_model_omits(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "默认先说明边界",
                                "content": "主人希望以后默认先说明边界。",
                                "domain": "general",
                                "confidence": 0.75,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]
    assert candidate["original_excerpt"]
    assert "以后默认先说明边界" in candidate["original_excerpt"]


def test_low_confidence_generic_candidate_kept_with_flag(tmp_path):
    """V4：Review 保持高召回——低于 review 阈值的候选保留并标记 low_confidence，
    不静默丢弃。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "今天天气不错",
            "assistant_text": "是啊。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "key_event",
                                "title": "泛泛",
                                "content": "今天聊了天气，泛泛的内容",
                                "original_excerpt": "今天天气不错",
                                "domain": "general",
                                "confidence": 0.5,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert "low_confidence" in candidate["soft_flags"]
    assert candidate["confidence"] == 0.5


def test_review_keeps_moderate_confidence_for_recall(tmp_path):
    """产品纠正：Review 不得高精度低召回；中等置信度候选应保留给主人筛选。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "默认先说明边界",
                                "content": "主人希望以后默认先说明边界。",
                                "original_excerpt": "我希望你以后默认先说明边界",
                                "domain": "general",
                                "confidence": 0.58,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    assert result["candidates"][0]["confidence"] == 0.58


def test_auto_uses_strict_confidence(tmp_path):
    """Auto 模式才使用严格高置信度：0.6（低于 auto 阈值 0.68）在 auto 下丢弃。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="auto"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "默认先说明边界",
                                "content": "主人希望以后默认先说明边界。",
                                "original_excerpt": "我希望你以后默认先说明边界",
                                "domain": "general",
                                "confidence": 0.6,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = asyncio.run(
        engine.run_daily_chat_memory(
            buckets,
            conversation_turn_store=FakeTurns(turns),
            key="2026-08-14",
            mode="auto",
            force=True,
        )
    )
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"
    assert buckets.created == []


def test_high_confidence_candidate_with_precise_source_kept(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "默认先说明边界",
                                "content": "主人希望以后默认先说明边界。",
                                "original_excerpt": "我希望你以后默认先说明边界",
                                "domain": "general",
                                "confidence": 0.7,
                                "source_turn_ids": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    assert result["candidates"][0]["source_turn_ids"] == [1]


def test_raw_event_source_ids_are_precise(tmp_path):
    """raw_events 来源：source_event_ids 精确指向真实事件，不做全天回退。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    events = [
        {
            "id": 101,
            "role": "user",
            "text": "我承诺以后每周二晚上一起散步",
            "session_id": "s",
            "created_at": "2026-08-14T20:00:00+08:00",
            "metadata": {"round_id": 1},
        },
        {
            "id": 102,
            "role": "assistant",
            "text": "好的，记住了。",
            "session_id": "s",
            "created_at": "2026-08-14T20:00:30+08:00",
            "metadata": {"round_id": 1},
        },
        {
            "id": 103,
            "role": "user",
            "text": "晚安",
            "session_id": "s",
            "created_at": "2026-08-14T23:00:00+08:00",
            "metadata": {"round_id": 2},
        },
    ]
    captured = patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "commitment",
                                "title": "每周二晚上一起散步",
                                "content": "主人承诺以后每周二晚上一起散步。",
                                "original_excerpt": "我承诺以后每周二晚上一起散步",
                                "domain": "general",
                                "confidence": 0.78,
                                "source_event_ids": [101, 102],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = asyncio.run(
        engine.run_daily_chat_memory(
            buckets,
            conversation_turn_store=FakeTurns([]),
            raw_event_store=FakeRawEvents(events),
            key="2026-08-14",
            mode="review",
            force=True,
        )
    )
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert candidate["source_event_ids"] == [101, 102]
    assert candidate["source_turn_ids"] == []
    assert "每周二晚上" in candidate["original_excerpt"]


# ---------------------------------------------------------------------------
# C. project_state 收紧
# ---------------------------------------------------------------------------

def test_project_state_requires_decided_marker(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "部署时遇到端口冲突，重启解决了",
            "assistant_text": "那问题就解决了。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "zero_candidates"


def test_project_state_allowed_for_stable_decision(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "新版本已经部署上线了，之后保持这个镜像别再改",
            "assistant_text": "明白，以后默认沿用这个版本。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    kinds = {candidate["kind"] for candidate in result["candidates"]}
    assert "project_state" in kinds
    assert result["candidates"][0]["source_turn_ids"] == [1]


def test_heuristic_precise_source_ids_per_turn(tmp_path):
    """heuristic 必须指向具体命中轮次，而不是全天轮次。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "今天天气不错",
            "assistant_text": "是啊。",
        },
        {
            "id": 2,
            "session_id": "s",
            "created_at": "2026-08-14T10:05:00+08:00",
            "user_text": "我承诺以后每周都陪你散步",
            "assistant_text": "好的。",
        },
        {
            "id": 3,
            "session_id": "s",
            "created_at": "2026-08-14T10:10:00+08:00",
            "user_text": "那就这样吧",
            "assistant_text": "嗯。",
        },
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert candidate["source_turn_ids"] == [2]
    assert candidate["kind"] == "commitment"
    assert candidate["original_excerpt"]
    assert "每周" in candidate["original_excerpt"]


def test_heuristic_turn_without_source_id_skipped(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": None,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "我承诺以后每周都陪你散步",
            "assistant_text": "好的。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    # 无 id 也无 raw_event_ids → 无法精确引用来源 → 不产出候选
    assert result["status"] == "zero_candidates"


# ---------------------------------------------------------------------------
# D. Review 安全：GET 严格只读
# ---------------------------------------------------------------------------

def test_get_pending_repeated_calls_do_not_change_anything(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    run_memory(engine, buckets, "review", turns=turns)
    before = pending_bytes(engine)

    for _ in range(3):
        items = engine.list_daily_chat_memory_pending()
        assert len(items) == 1
        assert items[0]["status"] == "pending"

    assert pending_bytes(engine) == before
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    assert payload["items"][0]["status"] == "pending"


def test_page_refresh_does_not_reject(tmp_path):
    """打开/刷新页面不得自动 reject 任何候选（哪怕是低质量 legacy 候选）。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    legacy_item = {
        "id": "legacy-pending-1",
        "date": "2026-08-13",
        "status": "pending",
        "created_at": "2026-08-13T16:00:00+00:00",
        "candidate": {
            "id": "legacy-pending-1",
            "content": "很泛泛的内容没有原文来源",
            "kind": "key_event",
        },
    }
    seed_pending(engine, [legacy_item])

    for _ in range(3):
        items = engine.list_daily_chat_memory_pending()
        assert items[0]["status"] == "pending"

    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    assert payload["items"][0]["status"] == "pending"


def test_existing_rejected_confirmed_not_reclassified_by_list(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    items = [
        {
            "id": "c1",
            "date": "2026-08-12",
            "status": "rejected",
            "created_at": "2026-08-12T16:00:00+00:00",
            "candidate": {"id": "c1", "content": "x", "kind": "key_event"},
        },
        {
            "id": "c2",
            "date": "2026-08-12",
            "status": "confirmed",
            "created_at": "2026-08-12T16:05:00+00:00",
            "candidate": {"id": "c2", "content": "y", "kind": "key_event"},
        },
    ]
    seed_pending(engine, items)
    before = pending_bytes(engine)

    all_items = engine.list_daily_chat_memory_pending(status="all", limit=50)
    statuses = {item["id"]: item["status"] for item in all_items}
    assert statuses == {"c1": "rejected", "c2": "confirmed"}
    assert pending_bytes(engine) == before


def test_legacy_candidate_list_shows_honest_placeholder(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    legacy_item = {
        "id": "legacy-pending-1",
        "date": "2026-08-13",
        "status": "pending",
        "created_at": "2026-08-13T16:00:00+00:00",
        "candidate": {"id": "legacy-pending-1", "content": "旧候选没有原文摘录", "kind": "key_event"},
    }
    seed_pending(engine, [legacy_item])
    items = engine.list_daily_chat_memory_pending()
    assert items[0]["display"]["legacy_no_original"] is True
    assert items[0]["display"]["confirm_blocked"] is True


# ---------------------------------------------------------------------------
# E. 正式记忆写入与门禁
# ---------------------------------------------------------------------------

def test_pending_never_enters_formal_materials(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    # pending 不得成为 Dream/Reflection/Prompt/Recall/Activity 候选材料
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=result["candidates"]
    ) == []
    activity = asyncio.run(
        engine.run_daily_activity_summary(
            conversation_turn_store=FakeTurns(turns),
            daily_chat_memory_candidates=result["candidates"],
            key="2026-08-14",
        )
    )
    assert activity["status"] == "ready"
    serialized = json.dumps(activity["activity_summary"], ensure_ascii=False)
    assert result["candidates"][0]["id"] not in serialized


def test_confirm_writes_bucket_with_precise_source_refs(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]

    confirmed = asyncio.run(
        engine.confirm_daily_chat_memory(
            [candidate["id"]],
            buckets,
            action="confirm",
            request_id="rq-confirm-1",
        )
    )

    assert confirmed["status"] == "ok"
    assert confirmed["created"] == 1
    assert buckets.created == [candidate["id"]]
    metadata = buckets.metadata_by_id[candidate["id"]]
    assert metadata["source_conversation_turn_ids"] == [1]
    assert metadata["source_hash"] == candidate["source_hash"]
    assert metadata["source_verification"] == "verified"
    # 原文摘录只留在 pending 记录，不写入 bucket metadata（owner-only）
    assert "original_excerpt" not in metadata
    materials = engine._daily_chat_memory_materials_for_date("2026-08-14")
    assert materials and materials[0]["id"] == candidate["id"]
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    assert payload["items"][0]["status"] == "confirmed"
    assert payload["items"][0]["action_source"] == "owner"
    assert payload["items"][0]["request_id"] == "rq-confirm-1"


def test_reject_does_not_write_bucket(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]

    rejected = asyncio.run(
        engine.confirm_daily_chat_memory(
            [candidate["id"]],
            buckets,
            action="reject",
            request_id="rq-reject-1",
            reject_reason="too_generic",
        )
    )

    assert rejected["status"] == "ok"
    assert rejected["rejected"] == 1
    assert buckets.created == []
    assert engine._daily_chat_memory_materials_for_date("2026-08-14") == []
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    item = payload["items"][0]
    assert item["status"] == "rejected"
    assert item["rejected_at"]
    assert item["action_source"] == "owner"
    assert item["request_id"] == "rq-reject-1"
    assert item["reject_reason"] == "too_generic"


def test_confirm_request_id_idempotent(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]

    first = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="confirm", request_id="rq-confirm-dup")
    )
    assert first["created"] == 1
    second = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="confirm", request_id="rq-confirm-dup")
    )
    assert second["idempotent_replay"] is True
    assert second["created"] == 1
    assert buckets.created == [candidate["id"]]


def test_reject_request_id_idempotent(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]

    first = asyncio.run(
        engine.confirm_daily_chat_memory(
            [candidate["id"]], buckets, action="reject", request_id="rq-reject-dup", reject_reason="wrong"
        )
    )
    assert first["rejected"] == 1
    second = asyncio.run(
        engine.confirm_daily_chat_memory(
            [candidate["id"]], buckets, action="reject", request_id="rq-reject-dup", reject_reason="wrong"
        )
    )
    assert second["idempotent_replay"] is True
    assert second["rejected"] == 1
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    assert payload["items"][0]["status"] == "rejected"


def test_batch_reject_is_explicit_and_auditable(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        },
        {
            "id": 2,
            "session_id": "s",
            "created_at": "2026-08-14T10:05:00+08:00",
            "user_text": "我承诺以后每周都陪你散步",
            "assistant_text": "好的。",
        },
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    ids = [candidate["id"] for candidate in result["candidates"]]

    batched = asyncio.run(
        engine.confirm_daily_chat_memory(
            ids,
            buckets,
            action="reject",
            request_id="rq-batch-reject",
            reject_reason="duplicate",
        )
    )

    assert batched["status"] == "ok"
    assert batched["rejected"] == len(ids)
    assert buckets.created == []
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in payload["items"]}
    for candidate_id in ids:
        item = by_id[candidate_id]
        assert item["status"] == "rejected"
        assert item["action_source"] == "owner"
        assert item["request_id"] == "rq-batch-reject"
        assert item["reject_reason"] == "duplicate"
        assert item["rejected_at"]


def test_reject_reason_defaults_to_other(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    candidate = result["candidates"][0]
    asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="reject", request_id="rq-reject-noreason")
    )
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    assert payload["items"][0]["reject_reason"] == "other"


def test_rate_limit_blocks_rapid_confirm(tmp_path):
    engine = ReflectionEngine(
        make_config(tmp_path, mode="review", daily_chat_memory_confirm_rate_limit_per_minute=2)
    )
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:01:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        },
        {
            "id": 2,
            "session_id": "s",
            "created_at": "2026-08-14T10:02:00+08:00",
            "user_text": "我承诺以后每周二晚上一起散步",
            "assistant_text": "好。",
        },
        {
            "id": 3,
            "session_id": "s",
            "created_at": "2026-08-14T10:03:00+08:00",
            "user_text": "我不喜欢你深夜发消息，以后不要这样",
            "assistant_text": "明白。",
        },
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    ids = [candidate["id"] for candidate in result["candidates"]]
    assert len(ids) == 3

    outcomes = []
    for index, candidate_id in enumerate(ids):
        outcome = asyncio.run(
            engine.confirm_daily_chat_memory(
                [candidate_id], buckets, action="reject", request_id=f"rq-rate-{index}"
            )
        )
        outcomes.append(outcome["status"])
    # 前 2 次成功，第 3 次触发限流
    assert outcomes[:2] == ["ok", "ok"]
    assert outcomes[2] == "rate_limited"


def test_legacy_candidate_confirm_blocked_invalid_source(tmp_path):
    """legacy 候选（无 source_verification/原文）不得悄悄批准，必须显式 invalid_source。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    legacy_item = {
        "id": "legacy-pending-1",
        "date": "2026-08-13",
        "status": "pending",
        "created_at": "2026-08-13T16:00:00+00:00",
        "candidate": {
            "id": "legacy-pending-1",
            "content": "旧候选内容没有来源校验",
            "kind": "key_event",
            "mode": "review",
        },
    }
    seed_pending(engine, [legacy_item])
    buckets = FakeBuckets()

    confirmed = asyncio.run(
        engine.confirm_daily_chat_memory(["legacy-pending-1"], buckets, action="confirm", request_id="rq-legacy-confirm")
    )

    assert confirmed["results"][0]["status"] == "invalid_source"
    assert buckets.created == []
    payload = json.loads(Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8"))
    item = payload["items"][0]
    assert item["status"] == "pending"  # 状态不被悄悄改变
    assert item["stale"] is True
    assert item["stale_reason"] == "legacy_candidate_missing_source"


def test_auto_applied_path_keeps_working(tmp_path):
    """auto 路径继续自动写 bucket；来源合法的候选正常 applied。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="auto"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "auto", turns=turns)
    assert result["mode"] == "auto"
    assert result["status"] == "created"
    assert all(item["status"] == "applied" for item in result["candidates"])
    assert buckets.created
    materials = engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=result["candidates"]
    )
    assert {item["id"] for item in materials} == {item["id"] for item in result["candidates"]}


def test_auto_candidate_without_source_not_applied(tmp_path):
    """auto 路径下来源非法的候选必须 apply_failed，不写 bucket。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="auto"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-14T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True,
                                "kind": "stable_preference",
                                "title": "x",
                                "content": "没有来源的候选",
                                "domain": "general",
                                "confidence": 0.8,
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = asyncio.run(
        engine.run_daily_chat_memory(
            buckets,
            conversation_turn_store=FakeTurns(turns),
            key="2026-08-14",
            mode="auto",
            force=True,
        )
    )
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"
    assert buckets.created == []


def test_reject_reason_normalization_rejects_invalid_values(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    assert engine._normalize_daily_chat_memory_reject_reason("too_generic") == "too_generic"
    assert engine._normalize_daily_chat_memory_reject_reason("WRONG") == "wrong"
    assert engine._normalize_daily_chat_memory_reject_reason("随便写的") == "other"
    assert engine._normalize_daily_chat_memory_reject_reason("") == "other"


def test_meaningful_day_produces_all_signal_categories(tmp_path):
    """有意义的一天必须覆盖全部信号类别；若有明显长期信号却零候选，测试失败。"""
    engine = ReflectionEngine(
        make_config(
            tmp_path,
            mode="review",
            daily_chat_memory_max_per_day=10,
            daily_chat_memory_review_max_per_day=10,
        )
    )
    turns = [
        {"id": 1, "session_id": "s", "created_at": "2026-08-14T09:00:00+08:00",
         "user_text": "我希望你以后默认先说明边界", "assistant_text": "收到。"},
        {"id": 2, "session_id": "s", "created_at": "2026-08-14T10:00:00+08:00",
         "user_text": "我不喜欢你擅自替我做决定，以后不要这样", "assistant_text": "明白。"},
        {"id": 3, "session_id": "s", "created_at": "2026-08-14T11:00:00+08:00",
         "user_text": "暗号是月亮，以后看到月亮就是提醒我休息", "assistant_text": "记住了。"},
        {"id": 4, "session_id": "s", "created_at": "2026-08-14T12:00:00+08:00",
         "user_text": "我承诺以后每周二晚上一起散步", "assistant_text": "好。"},
        {"id": 5, "session_id": "s", "created_at": "2026-08-14T13:00:00+08:00",
         "user_text": "我们正式在一起了，关系定位变了", "assistant_text": "嗯，我会一直在。"},
        {"id": 6, "session_id": "s", "created_at": "2026-08-14T14:00:00+08:00",
         "user_text": "今天特别开心，值得记住", "assistant_text": "我也很开心。"},
        {"id": 7, "session_id": "s", "created_at": "2026-08-14T15:00:00+08:00",
         "user_text": "新版本已经部署上线，之后保持这个镜像", "assistant_text": "明白。"},
    ]
    candidates = [
        {"should_write": True, "kind": "stable_preference", "title": "默认先说明边界",
         "content": "主人希望以后默认先说明边界。", "original_excerpt": "我希望你以后默认先说明边界",
         "domain": "general", "confidence": 0.75, "source_turn_ids": [1]},
        {"should_write": True, "kind": "boundary", "title": "不要擅自替我做决定",
         "content": "主人不喜欢被擅自替做决定，以后不要这样。", "original_excerpt": "我不喜欢你擅自替我做决定，以后不要这样",
         "domain": "general", "confidence": 0.75, "source_turn_ids": [2]},
        {"should_write": True, "kind": "signal", "title": "暗号：月亮=提醒休息",
         "content": "暗号是月亮，看到月亮提醒主人休息。", "original_excerpt": "暗号是月亮，以后看到月亮就是提醒我休息",
         "domain": "general", "confidence": 0.8, "source_turn_ids": [3]},
        {"should_write": True, "kind": "commitment", "title": "每周二一起散步",
         "content": "主人承诺每周二晚上一起散步。", "original_excerpt": "我承诺以后每周二晚上一起散步",
         "domain": "general", "confidence": 0.8, "source_turn_ids": [4]},
        {"should_write": True, "kind": "relationship_anchor", "title": "正式在一起",
         "content": "主人与润润正式在一起，关系定位发生变化。", "original_excerpt": "我们正式在一起了，关系定位变了",
         "domain": "general", "confidence": 0.85, "source_turn_ids": [5]},
        {"should_write": True, "kind": "key_event", "title": "特别开心的一天",
         "content": "当天主人表示特别开心，值得记住。", "original_excerpt": "今天特别开心，值得记住",
         "domain": "general", "confidence": 0.7, "source_turn_ids": [6]},
        {"should_write": True, "kind": "project_state", "title": "新版本镜像保持",
         "content": "新版本已部署上线，之后保持该镜像。", "original_excerpt": "新版本已经部署上线，之后保持这个镜像",
         "domain": "general", "confidence": 0.78, "source_turn_ids": [7]},
    ]
    patch_model(engine, [{"content": json.dumps({"candidates": candidates}, ensure_ascii=False)}])
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    # 有明显长期信号 → 绝不能零候选（否则本断言失败）
    assert result["status"] == "pending"
    assert result["candidates"], "meaningful day must not return zero candidates"
    kinds = {candidate["kind"] for candidate in result["candidates"]}
    assert kinds >= {
        "stable_preference",
        "boundary",
        "signal",
        "commitment",
        "relationship_anchor",
        "key_event",
        "project_state",
    }
    # 每个候选都有精确来源
    for candidate in result["candidates"]:
        assert candidate["source_turn_ids"] or candidate["source_event_ids"]
        assert candidate["source_verification"] == "verified"
        assert candidate["original_excerpt"]


def test_duplicate_overlapping_windows_merged_most_complete_source(tmp_path):
    """重叠窗口产生相同候选时合并，保留来源最完整的一条。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {"id": 1, "session_id": "s", "created_at": "2026-08-14T10:00:00+08:00",
         "user_text": "我承诺以后每周二晚上一起散步", "assistant_text": "好。"},
        {"id": 2, "session_id": "s", "created_at": "2026-08-14T10:05:00+08:00",
         "user_text": "对，就是每周二晚上，别改", "assistant_text": "记住了，每周二晚上。"},
    ]
    patch_model(
        engine,
        [
            {
                "content": json.dumps(
                    {
                        "candidates": [
                            {
                                "should_write": True, "kind": "commitment", "title": "每周二散步",
                                "content": "主人承诺每周二晚上一起散步。",
                                "original_excerpt": "我承诺以后每周二晚上一起散步",
                                "domain": "general", "confidence": 0.8, "source_turn_ids": [1],
                            },
                            {
                                "should_write": True, "kind": "commitment", "title": "每周二散步",
                                "content": "主人承诺每周二晚上一起散步。",
                                "original_excerpt": "对，就是每周二晚上，别改",
                                "domain": "general", "confidence": 0.8, "source_turn_ids": [2],
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ],
    )
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review", turns=turns)
    assert result["status"] == "pending"
    assert len(result["candidates"]) == 1, "重复候选应合并为一条"
    merged = result["candidates"][0]
    assert set(merged["source_turn_ids"]) == {1, 2}
    assert merged["source_hash"]


def test_extraction_windows_cover_beginning_middle_end(tmp_path):
    """V4 全天窗口覆盖：每段轮次都必须落入至少一个被检查的窗口。"""
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
        {"id": n, "session_id": "s", "created_at": f"2026-08-14T{9 + n}:00:00+08:00",
         "user_text": f"内容 {n}", "assistant_text": f"回复 {n}"}
        for n in range(1, 11)
    ]
    windows = engine._daily_chat_memory_extraction_windows(turns)
    covered_ids: set[int] = set()
    for window in windows:
        covered_ids.update(int(turn["id"]) for turn in window if turn.get("id") is not None)
    assert covered_ids == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    # 开头 / 中间 / 结尾都在被检查的窗口内
    assert 1 in covered_ids and 5 in covered_ids and 10 in covered_ids
