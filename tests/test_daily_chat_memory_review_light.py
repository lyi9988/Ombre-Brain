"""脱敏专项：轻量 daily_chat_memory review/auto candidate-material 门禁。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from reflection_engine import ReflectionEngine


def make_config(tmp_path: Path, *, mode: str = "review") -> dict:
    state_dir = tmp_path / "state"
    buckets_dir = tmp_path / "buckets"
    return {
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
            "daily_chat_memory_summary_enabled": False,
            "daily_chat_memory_max_per_day": 4,
            "daily_chat_memory_review_max_per_day": 4,
            "daily_activity_summary_enabled": True,
        },
    }


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
        date = str(kwargs.get("date") or "")
        metadata = {
            "id": bucket_id,
            "source": kwargs.get("source") or "daily_chat_memory",
            "from_daily_chat": True,
            "event_date": date,
            "daily_chat_memory_candidate_id": bucket_id,
            "tags": list(kwargs.get("tags") or []),
            "domain": list(kwargs.get("domain") or []),
            "confidence": kwargs.get("confidence", 0.7),
        }
        item = {"id": bucket_id, "content": kwargs.get("content") or "", "metadata": metadata}
        self.items[bucket_id] = item
        self.created.append(bucket_id)
        return bucket_id


class FakeTurns:
    def list_conversation_turns_between(self, **kwargs):
        return [
            {
                "id": 11,
                "session_id": "fixture-session",
                "created_at": "2026-08-14T12:00:00+08:00",
                "user_text": "以后默认先说明边界",
                "assistant_text": "收到，我会记住这个偏好。",
            }
        ]


def run_memory(engine: ReflectionEngine, buckets: FakeBuckets, mode: str):
    return asyncio.run(
        engine.run_daily_chat_memory(
            buckets,
            conversation_turn_store=FakeTurns(),
            key="2026-08-14",
            mode=mode,
            force=True,
        )
    )


def test_review_pending_is_generated_and_dashboard_material_gate_blocks_it(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    buckets = FakeBuckets()

    result = run_memory(engine, buckets, "review")

    assert result["mode"] == "review"
    assert result["status"] == "pending"
    assert result["candidates"]
    assert all(item["mode"] == "review" and item["status"] == "pending" for item in result["candidates"])
    assert buckets.created == []
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=result["candidates"]
    ) == []

    payload = json.loads(
        Path(engine.daily_chat_memory_pending_path).read_text(encoding="utf-8")
    )
    assert payload["items"][0]["status"] == "pending"


def test_review_confirm_writes_bucket_and_unlocks_material_path(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review")
    candidate_id = result["candidates"][0]["id"]

    confirmed = asyncio.run(
        engine.confirm_daily_chat_memory([candidate_id], buckets, action="confirm")
    )

    assert confirmed["results"][0]["status"] == "created"
    assert buckets.created == [candidate_id]
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=[]
    )[0]["id"] == candidate_id


def test_review_rejected_does_not_write_or_enter_formal_materials(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="review"))
    buckets = FakeBuckets()
    result = run_memory(engine, buckets, "review")
    candidate_id = result["candidates"][0]["id"]

    rejected = asyncio.run(
        engine.confirm_daily_chat_memory([candidate_id], buckets, action="reject")
    )

    assert rejected["results"][0]["status"] == "rejected"
    assert buckets.created == []
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=[]
    ) == []


def test_auto_keeps_automatic_bucket_write_and_material_projection(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path, mode="auto"))
    buckets = FakeBuckets()

    result = run_memory(engine, buckets, "auto")

    assert result["mode"] == "auto"
    assert result["status"] == "created"
    assert result["candidates"]
    assert all(item["mode"] == "auto" and item["status"] == "applied" for item in result["candidates"])
    assert len(buckets.created) == len(result["candidates"])
    materials = engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=result["candidates"]
    )
    assert {item["id"] for item in materials} == {item["id"] for item in result["candidates"]}
    assert "daily_chat_memory_auto_disabled" not in result.values()


def test_review_status_gate_covers_rejected_and_auto_pending(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path))
    blocked = [
        {"id": "review-pending", "date": "2026-08-14", "mode": "review", "status": "pending", "content": "x"},
        {"id": "review-rejected", "date": "2026-08-14", "mode": "review", "status": "rejected", "content": "x"},
        {"id": "auto-pending", "date": "2026-08-14", "mode": "auto", "status": "pending", "content": "x"},
        {"id": "auto-failed", "date": "2026-08-14", "mode": "auto", "status": "apply_failed", "content": "x"},
    ]
    assert engine._daily_chat_memory_materials_for_date(
        "2026-08-14", daily_chat_memory_candidates=blocked
    ) == []


def test_activity_summary_keeps_raw_conversation_path_without_candidate_material(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path))
    pending = {
        "id": "review-pending",
        "date": "2026-08-14",
        "mode": "review",
        "status": "pending",
        "content": "pending candidate content",
    }

    result = asyncio.run(
        engine.run_daily_activity_summary(
            conversation_turn_store=FakeTurns(),
            daily_chat_memory_candidates=[pending],
            key="2026-08-14",
        )
    )

    assert result["status"] == "ready"
    assert result["turn_source"] == "conversation_turns"
    assert result["turns"] == 1
    assert result["activity_summary"]["evidence"] == [{"session_id": "fixture-session"}]
    assert all("review-pending" not in str(value) for value in result["activity_summary"].values())


def test_historical_reflection_buckets_are_outside_light_fix_scope(tmp_path):
    engine = ReflectionEngine(make_config(tmp_path))
    assert not hasattr(engine, "daily_chat_memory_store")
    assert not (Path(__file__).parents[1] / "daily_chat_memory_store.py").exists()
    assert not (Path(__file__).parents[1] / "scripts" / "dry_run_daily_chat_memory_migration.py").exists()
