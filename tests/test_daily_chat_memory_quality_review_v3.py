"""脱敏专项 V3：V2 验收失败返修门禁。

覆盖：中文长句不中间截断、中英文标点边界、<silent>/内部标签清理、内部素材
转储来源丢弃、建议记忆独立/缺失无效/echo 无效、临时安慰不误判、assistant
内容仅限承诺/关系/稳定边界、完整原文仅经来源展开、API 与落盘一致、GET 严格
只读、pending 隔离门禁。
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
    async def fake_create_completion(client, *, model, messages, max_tokens, temperature, use_daily_client):
        return FakeResponse(content)

    engine._daily_chat_memory_model_client = lambda *, candidate: (object(), "fake-model", True)
    engine._daily_chat_memory_create_completion = fake_create_completion


def model_candidates_json(candidates: list[dict]) -> str:
    return json.dumps({"candidates": candidates}, ensure_ascii=False)


def run_review(engine: ReflectionEngine, turns: list[dict]):
    return asyncio.run(
        engine.run_daily_chat_memory(
            FakeBuckets(),
            conversation_turn_store=FakeTurns(turns),
            key="2026-08-18",
            mode="review",
            force=True,
        )
    )


def seed_pending(engine: ReflectionEngine, items: list[dict]):
    path = Path(engine.daily_chat_memory_pending_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items, "cursor": {}}, ensure_ascii=False, indent=2), encoding="utf-8")


def pending_bytes(engine: ReflectionEngine) -> bytes:
    return Path(engine.daily_chat_memory_pending_path).read_bytes()


# ---------------------------------------------------------------------------
# 1. 中文长句与标点边界：摘录绝不从句子中间截断
# ---------------------------------------------------------------------------

def test_chinese_long_sentence_not_truncated_mid_sentence(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": (
                "我希望你以后默认先说明边界，不要擅自替我决定任何事情。"
                "我们约定好，遇到重要决定一定要先商量。"
            ),
            "assistant_text": "好，我记住了。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "stable_preference",
                    "title": "以后默认先说明边界",
                    "content": "主人希望以后默认先说明边界，重要决定先商量。",
                    "domain": "general",
                    "confidence": 0.75,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "pending"
    excerpt = result["candidates"][0]["original_excerpt"]
    # 摘录必须是完整句子：不以“，”，也不以句中断点结尾。
    assert excerpt.startswith("我希望你") or excerpt.startswith("我们约定好")
    assert excerpt.endswith("。") or excerpt.endswith("！") or excerpt.endswith("？")
    assert "，不要擅自替我决定任何事情。我们约定好" in excerpt


def test_mixed_cn_en_punctuation_boundaries(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "以后默认先说明边界。This is the rule! 记得每周二一起散步。",
            "assistant_text": "OK.",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "stable_preference",
                    "title": "默认先说明边界",
                    "content": "主人希望以后默认先说明边界，并记得每周二散步的约定。",
                    "domain": "general",
                    "confidence": 0.75,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "pending"
    excerpt = result["candidates"][0]["original_excerpt"]
    assert not excerpt.endswith("，")


# ---------------------------------------------------------------------------
# 2. <silent> / 内部控制标记清理；内部素材转储来源丢弃
# ---------------------------------------------------------------------------

def test_silent_and_internal_tags_cleaned(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": '嗯，累了。',
            "assistant_text": (
                '<silent mood="heartache" as="心疼地把你搂进怀里" reason="终于肯承认自己累了"></silent> '
                '累了就对了。我们什么都不想了，钻进被窝里好好睡。晚安。 [语音:累了就对了。晚安。]'
            ),
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "key_event",
                    "title": "主人累了的一天",
                    "content": "主人那天表示很累，需要好好休息。",
                    "domain": "general",
                    "confidence": 0.7,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    # key_event 需要主人明确表达；此处 user 有内容，保留。
    assert result["status"] == "pending"
    excerpt = result["candidates"][0]["original_excerpt"]
    assert "<" not in excerpt and ">" not in excerpt
    assert "as=" not in excerpt and "reason=" not in excerpt and "mood=" not in excerpt
    assert "silent" not in excerpt
    assert "[语音" not in excerpt
    # 摘录只包含干净的完整句子
    assert excerpt.endswith("。") or excerpt.endswith("！") or excerpt.endswith("？")


def test_internal_dump_source_turn_dropped(tmp_path):
    """“近期素材/今天的聊天 [app/bridge]”内部材料转储不是真实来源，候选丢弃。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": '近期素材： 今天的聊天: [app/bridge] 顾衍: 这是内部拼接的素材转储内容',
            "assistant_text": "明白。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "key_event",
                    "title": "x",
                    "content": "内部素材转储不应成为记忆",
                    "domain": "general",
                    "confidence": 0.8,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"


# ---------------------------------------------------------------------------
# 3. proposed_memory 独立、缺失无效、echo 无效、可读
# ---------------------------------------------------------------------------

def test_proposed_memory_missing_invalid(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "stable_preference",
                    "title": "x",
                    "content": "",
                    "domain": "general",
                    "confidence": 0.8,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "zero_candidates"
    assert result["reason"] == "no_candidates"


def test_model_echo_kept_with_needs_owner_edit(tmp_path):
    """V4：模型把原句原样当建议记忆 → 候选保留并标记 needs_owner_edit，
    未编辑前不能 approve（不再静默丢弃）。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "我希望你以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "stable_preference",
                    "title": "x",
                    "content": "我希望你以后默认先说明边界",
                    "domain": "general",
                    "confidence": 0.8,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert "needs_owner_edit" in candidate["soft_flags"]
    assert "excerpt_overlap" in candidate["soft_flags"]
    # 未编辑 → approve 拦截
    buckets = FakeBuckets()
    blocked = asyncio.run(
        engine.confirm_daily_chat_memory([candidate["id"]], buckets, action="confirm", request_id="rq-echo-v3")
    )
    assert blocked["results"][0]["status"] == "needs_owner_edit"
    assert buckets.created == []


def test_proposed_memory_independent_readable(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "以后默认先说明边界，重要决定先商量",
            "assistant_text": "收到。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "stable_preference",
                    "title": "默认先说明边界",
                    "content": "主人希望以后默认先说明边界，遇到重要决定先商量。",
                    "domain": "general",
                    "confidence": 0.75,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "pending"
    candidate = result["candidates"][0]
    assert candidate["proposed_memory"] != candidate["original_excerpt"]
    assert len(candidate["proposed_memory"]) >= 12
    assert candidate["original_excerpt"]
    # 建议记忆不以来源原句开头（脱水和整理过）
    assert not candidate["proposed_memory"].startswith(candidate["original_excerpt"][:8])


# ---------------------------------------------------------------------------
# 4. 临时安慰不误判；assistant 内容仅限承诺/关系/稳定边界
# ---------------------------------------------------------------------------

def test_assistant_comfort_not_boundary_or_key_event(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "",
            "assistant_text": "别难过，我在呢，抱抱你。",
        }
    ]
    for kind in ("boundary", "key_event", "stable_preference"):
        patch_model(
            engine,
            model_candidates_json(
                [
                    {
                        "should_write": True,
                        "kind": kind,
                        "title": "x",
                        "content": "安慰类内容不应成为长期记忆",
                        "domain": "general",
                        "confidence": 0.7,
                        "source_turn_ids": [1],
                    }
                ]
            ),
        )
        result = run_review(engine, turns)
        assert result["status"] == "zero_candidates", f"kind={kind} must be dropped"


def test_assistant_commitment_with_durable_marker_eligible(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "",
            "assistant_text": "我会一直记得我们的约定，每周二晚上一起散步。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "commitment",
                    "title": "每周二散步约定",
                    "content": "润润会一直记得每周二晚上一起散步的约定。",
                    "domain": "general",
                    "confidence": 0.75,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "pending"
    assert result["candidates"][0]["kind"] == "commitment"


def test_assistant_boundary_without_owner_statement_dropped(tmp_path):
    """assistant 单方面说“我不会再让你难过”不能成为主人 boundary。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "",
            "assistant_text": "我不会再让你难过，这是我对自己的要求。",
        }
    ]
    patch_model(
        engine,
        model_candidates_json(
            [
                {
                    "should_write": True,
                    "kind": "boundary",
                    "title": "x",
                    "content": "主人以后不会再难过",
                    "domain": "general",
                    "confidence": 0.75,
                    "source_turn_ids": [1],
                }
            ]
        ),
    )
    result = run_review(engine, turns)
    assert result["status"] == "zero_candidates"


# ---------------------------------------------------------------------------
# 5. 完整原文仅经来源展开（source-preview），API 与落盘一致
# ---------------------------------------------------------------------------

def test_source_preview_returns_sanitized_full_text_and_is_consistent(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    raw_text = (
        '<silent mood="calm" as="轻声说" reason="安抚"></silent> 我们慢慢来，不着急。'
        '先把手头的事情放一放。 晚安。 [语音:我们慢慢来。]'
    )
    events = [
        {"id": 7001, "role": "user", "text": "以后默认先说明边界", "session_id": "s",
         "created_at": "2026-08-18T10:00:00+08:00", "metadata": {}},
        {"id": 7002, "role": "assistant", "text": raw_text, "session_id": "s",
         "created_at": "2026-08-18T10:00:05+08:00", "metadata": {}},
    ]
    seed_pending(
        engine,
        [
            {
                "id": "cand-source-1",
                "date": "2026-08-18",
                "status": "pending",
                "created_at": "2026-08-18T16:00:00+00:00",
                "candidate": {
                    "id": "cand-source-1",
                    "date": "2026-08-18",
                    "kind": "key_event",
                    "content": "主人累了，润润安抚后一起休息。",
                    "proposed_memory": "主人累了，润润安抚后一起休息。",
                    "original_excerpt": "我们慢慢来，不着急。",
                    "source_event_ids": [7001, 7002],
                    "source_verification": "verified",
                    "source_hash": "abc",
                    "mode": "review",
                    "status": "pending",
                },
            }
        ],
    )

    preview = asyncio.run(
        engine.daily_chat_memory_source_preview(
            "cand-source-1",
            raw_event_store=FakeRawEvents(events),
        )
    )
    assert preview["status"] == "ok"
    assert preview["missing_event_ids"] == []
    # 完整原文经来源展开且已净化：无 <silent> / 无 [语音
    for event in preview["events"]:
        assert "<" not in event["text"] and ">" not in event["text"]
        assert "silent" not in event["text"]
        assert "[语音" not in event["text"]
    by_id = {event["id"]: event["text"] for event in preview["events"]}
    assert "我们慢慢来" in by_id[7002]
    assert "以后默认先说明边界" in by_id[7001]


def test_source_preview_unknown_candidate_404(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    seed_pending(engine, [])
    preview = asyncio.run(
        engine.daily_chat_memory_source_preview("does-not-exist", raw_event_store=FakeRawEvents([]))
    )
    assert preview["status"] == "missing"


# ---------------------------------------------------------------------------
# 6. GET 严格只读与 pending 隔离门禁继续
# ---------------------------------------------------------------------------

def test_get_pending_still_read_only(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    run_review(engine, turns)
    before = pending_bytes(engine)
    for _ in range(3):
        items = engine.list_daily_chat_memory_pending()
        assert items and items[0]["status"] == "pending"
    assert pending_bytes(engine) == before


def test_pending_isolation_gate_continues(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    turns = [
        {
            "id": 1,
            "session_id": "s",
            "created_at": "2026-08-18T10:00:00+08:00",
            "user_text": "以后默认先说明边界",
            "assistant_text": "收到。",
        }
    ]
    result = run_review(engine, turns)
    # review 模式：pending 不入正式材料
    assert result["status"] == "pending"
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-18", daily_chat_memory_candidates=result["candidates"]
    ) == []
    assert all(item["mode"] == "review" and item["status"] == "pending" for item in result["candidates"])


def test_malformed_v2_candidate_marked_blocked_not_auto_modified(tmp_path):
    """V2 畸形候选（原文含内部控制标记）标记不可批准，不自动 reject/改写。"""
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    malformed = {
        "id": "v2-malformed-1",
        "date": "2026-08-18",
        "status": "pending",
        "created_at": "2026-08-18T16:00:00+00:00",
        "candidate": {
            "id": "v2-malformed-1",
            "date": "2026-08-18",
            "kind": "key_event",
            "content": 'as="心疼地把你搂进怀里" reason="...',
            "proposed_memory": 'as="心疼地把你搂进怀里" reason="...',
            "original_excerpt": 'as="心疼地把你搂进怀里" r',
            "source_event_ids": [5481, 5482],
            "source_verification": "verified",
            "source_hash": "deadbeef",
            "mode": "review",
        },
    }
    seed_pending(engine, [malformed])
    before = pending_bytes(engine)

    items = engine.list_daily_chat_memory_pending()
    assert items[0]["status"] == "pending"  # 不被自动改写/reject
    assert items[0]["display"]["confirm_blocked"] is True
    assert "candidate_excerpt_unclean" in items[0]["display"]["blocked_reasons"]
    assert pending_bytes(engine) == before  # 文件零改动

    # 尝试批准 → invalid_source，不写 bucket
    buckets = FakeBuckets()
    result = asyncio.run(
        engine.confirm_daily_chat_memory(["v2-malformed-1"], buckets, action="confirm", request_id="rq-v2-malformed")
    )
    assert result["results"][0]["status"] == "invalid_source"
    assert buckets.created == []
