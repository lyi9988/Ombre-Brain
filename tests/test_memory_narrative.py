"""Synthetic only: attribution/time survives normalization, review and commit."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from memory_narrative import (
    CONTRACT_VERSION, build_chat_narrative, chat_commit_metadata,
    memory_context_labels, source_event_time,
)
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config, attach_fake_commit_service


TZ = timezone(timedelta(hours=8))
IDENTITY = {"ai_name": "测试叙述者", "user_display_name": "测试用户"}


def turn(text="我在2025年12月3日搬了家，但没有答应以后都住那里。", when="2026-10-08T23:30:00+08:00"):
    return {"id": 1, "raw_event_ids": [101, 102], "created_at": when,
            "user_text": text, "assistant_text": "我明白了，你没有作出长期约定。"}


def candidate(expression="2025年12月3日", source=None):
    source = source or turn()
    return {"title": "搬家时没有长期居住约定", "kind": "key_event", "confidence": .95,
            "content": "她告诉我自己搬了家，但没有答应以后一直住在那里。我不能把这次搬家理解成长期居住约定。",
            "source_turn_ids": [1], "source_event_ids": [101, 102],
            "event_time": {"expression": expression, "evidence": source["user_text"],
                           "source_turn_id": 1, "source_role": "user"}}


def engine(tmp_path, mode="review"):
    config = reflection_config(tmp_path, mode=mode)
    config["identity"] = IDENTITY
    return ReflectionEngine(config)


def normalize(instance, value=None, sources=None, mode="review"):
    return instance._normalize_daily_chat_memory_candidates(
        "2026-10-08", [value or candidate()], sources or [turn()], mode=mode)


def test_source_date_not_batch_or_write_date():
    item = candidate()
    item["narrative"] = build_chat_narrative(item, [turn()], IDENTITY, TZ, "2026-10-08")
    item["date"] = "2026-10-08"
    metadata = chat_commit_metadata(item, "2026-10-09T01:00:00+08:00")
    assert metadata["date"] == "2025-12-03"
    assert metadata["mentioned_date"] == "2026-10-08"
    assert metadata["recorded_at"].startswith("2026-10-09")
    labels = " ".join(memory_context_labels({"metadata": metadata}))
    assert "事件日期:2025-12-03" in labels
    assert "提及日期:2026-10-08" in labels
    assert "入库日期:2026-10-09" in labels


@pytest.mark.parametrize("expression", ["昨天", "前两天", "去年春天", "2025年", "2025年12月"])
def test_imprecise_time_keeps_expression_and_source_anchor(expression):
    source = turn(f"我在{expression}搬家了。", "2026-10-08T17:00:00Z")
    time = source_event_time(candidate(expression, source), [source], TZ)
    assert time["precision"] == "expression"
    assert time["expression"] == expression
    assert time["reference_date"] == "2026-10-09"
    assert "value" not in time


@pytest.mark.parametrize("change", [
    {"source_turn_id": 99}, {"source_role": "system"}, {"source_role": "assistant"},
    {"evidence": "不存在的原话2025年12月3日"}, {"expression": "2026-10-09"},
    {"expression": "2025年12月3日", "evidence": ""},
])
def test_time_evidence_cannot_be_fabricated(change):
    value = candidate()
    value["event_time"].update(change)
    assert source_event_time(value, [turn()], TZ)["precision"] == "unknown"


def test_model_normalized_time_and_narrator_are_not_trusted(tmp_path):
    instance = engine(tmp_path)
    value = candidate()
    value["event_time"]["value"] = "2099-01-01"
    value["narrative"] = {"narrator": {"name": "冒充的作者"}}
    normalized = normalize(instance, value)[0]
    narrative = normalized["narrative"]
    assert narrative["narrator"] == {"role": "assistant", "name": "测试叙述者"}
    assert narrative["event_time"]["value"] == "2025-12-03"
    assert {s["speaker"] for s in narrative["sources"]} == {"测试叙述者", "测试用户"}
    assert narrative["generation_kind"] == "background_extraction"
    assert normalized["content"] == value["content"]
    assert instance._daily_chat_memory_proposal(normalized, mode="review").metadata["narrative"] == narrative


def test_no_time_never_defaults_to_batch_date(tmp_path):
    instance = engine(tmp_path)
    value = candidate()
    value.pop("event_time")
    normalized = normalize(instance, value)[0]
    metadata = chat_commit_metadata(normalized, "2026-10-09T01:00:00+08:00")
    assert metadata["date"] is None and metadata["event_date"] is None
    assert metadata["narrative"]["event_time"]["precision"] == "unknown"
    assert "事件日期:未知" in " ".join(memory_context_labels({"metadata": metadata}))


@pytest.mark.parametrize("mode", ["auto", "review"])
def test_real_authority_commit_keeps_body_narrator_and_three_times(tmp_path, mode):
    instance = engine(tmp_path, mode)
    projection = attach_fake_commit_service(instance)
    item = normalize(instance, mode=mode)[0]
    item["mode"] = mode
    if mode == "review":
        instance._store_daily_chat_memory_pending([item])
        result = asyncio.run(instance.confirm_daily_chat_memory(
            [item["id"]], object(), action="confirm", request_id="review-narrative"))
    else:
        result = asyncio.run(instance._write_daily_chat_memory_candidates([item], object()))
    assert result["created"] == 1
    assert len(projection.revisions) == 1
    committed = projection.revisions[0]
    meta = committed["metadata"]
    assert meta["narrative"]["version"] == CONTRACT_VERSION
    assert meta["date"] == "2025-12-03"
    assert meta["mentioned_date"] == "2026-10-08"
    assert datetime.fromisoformat(meta["created"]) > datetime(2026, 10, 8, tzinfo=TZ)
    assert committed["body"] == item["content"]
    from gateway import GatewayService
    labels = " ".join(GatewayService._bucket_date_meta_parts(None, {"metadata": meta}))
    assert "测试叙述者" in labels and "2025-12-03" in labels


def test_owner_edit_does_not_retain_unreviewed_event_date(tmp_path):
    instance = engine(tmp_path)
    item = normalize(instance)[0]
    edited = instance._apply_daily_chat_memory_candidate_edit(item, {"content": "我在谈另一件事，日期不明。"})
    assert item["narrative"]["event_time"]["precision"] == "day"
    assert edited["narrative"]["event_time"]["precision"] == "unknown"
    assert edited["narrative"]["edited_by"] == "owner"


def test_legacy_auto_date_is_not_presented_as_event_or_write_time():
    meta = {"source": "daily_chat_memory", "date": "2026-10-08", "event_date": "2026-10-08",
            "created": "2026-10-08T23:59:59+08:00"}
    labels = " ".join(memory_context_labels({"metadata": meta}))
    assert "事件日期:未知" in labels and "聊天整理日期:2026-10-08" in labels
    assert "[date:" not in labels and "[created:" not in labels


def test_legacy_explicit_date_and_event_date_remain_usable():
    assert memory_context_labels({"metadata": {"date": "2025-12-03"}}) == ["[date:2025-12-03]"]
    assert memory_context_labels({"metadata": {"event_date": "2025-12-03"}}) == ["[date:2025-12-03]"]
    assert "事件日期:未知" in " ".join(memory_context_labels({"metadata": {"created": "2026-10-09"}}))


@pytest.mark.parametrize("expression", ["2025-02-30", "2025年13月3日"])
def test_invalid_calendar_date_is_not_normalized(expression):
    source = turn(f"我在{expression}搬了家。")
    assert source_event_time(candidate(expression, source), [source], TZ)["precision"] == "unknown"


def test_default_prompts_have_narration_and_time_contract(tmp_path):
    instance = engine(tmp_path)
    prompt = instance._daily_chat_memory_prompt()
    assert "正文默认用 测试叙述者 第一人称" in prompt
    assert "正文优先用第三人称" not in prompt
    assert 'event_time=' in prompt and "事后提炼" in prompt
    assert "事件时间原话与说话时间分开" in instance._daily_chat_memory_summary_prompt()


def test_label_escaping_and_no_source_body_leak():
    value = candidate()
    narrative = build_chat_narrative(value, [turn()], {**IDENTITY, "ai_name": "name]\n[injected"}, TZ, "2026-10-08")
    labels = memory_context_labels({"metadata": {"narrative": narrative}})
    assert all("\n" not in label for label in labels)
    assert all(label.count("[") == 1 and label.count("]") == 1 for label in labels)
    assert turn()["user_text"] not in str(narrative)


def test_fragment_projection_preserves_attribution_without_body_rewrite():
    from memory_moments import parse_bucket_moments
    from gateway import GatewayService
    value = {**candidate(), "date": "2026-10-08"}
    value["narrative"] = build_chat_narrative(value, [turn()], IDENTITY, TZ, "2026-10-08")
    bucket = {"id": "daily_chat_memory_test", "content": value["content"],
              "metadata": chat_commit_metadata(value, "2026-10-09T01:00:00+08:00")}
    moments = parse_bucket_moments(bucket)
    assert moments
    for moment in moments:
        labels = " ".join(GatewayService._bucket_date_meta_parts(None, moment=moment))
        assert "测试叙述者" in labels and "事件日期:2025-12-03" in labels
        assert "提及日期:2026-10-08" in labels
    assert bucket["content"] == value["content"]


def test_old_auto_fragment_does_not_promote_batch_date():
    labels = " ".join(memory_context_labels(moment={"bucket_id": "daily_chat_memory_20261008_abc",
                                                  "metadata": {"bucket_date": "2026-10-08"}}))
    assert "事件日期:未知" in labels and "聊天整理日期:2026-10-08" in labels


def test_brain_and_gateway_use_same_labels():
    import server
    from gateway import GatewayService
    bucket = {"metadata": {"source": "daily_chat_memory", "date": "2026-10-08"}}
    assert server._bucket_date_meta_parts(bucket) == GatewayService._bucket_date_meta_parts(None, bucket)


def test_crash_after_commit_reuses_record_clock_and_revision(tmp_path, monkeypatch):
    import reflection_engine as module
    instance = engine(tmp_path, "auto")
    projection = attach_fake_commit_service(instance)
    item = normalize(instance, mode="auto")[0]
    item["mode"] = "auto"
    decide = instance.memory_authority_store.decide_candidate

    def interrupted(candidate_id, **kwargs):
        if kwargs["action"] == "commit":
            raise SystemExit("simulated process exit before candidate receipt")
        return decide(candidate_id, **kwargs)

    monkeypatch.setattr(instance.memory_authority_store, "decide_candidate", interrupted)
    with pytest.raises(SystemExit):
        asyncio.run(instance._write_daily_chat_memory_candidates([item], object()))
    recorded = projection.revisions[0]["metadata"]["recorded_at"]
    monkeypatch.setattr(instance.memory_authority_store, "decide_candidate", decide)

    class LaterClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2030, 1, 1, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(module, "datetime", LaterClock)
    result = asyncio.run(instance._write_daily_chat_memory_candidates([item], object()))
    assert result["failed"] == 0 and result["exists"] == 1
    assert len(projection.revisions) == 1
    assert projection.revisions[0]["metadata"]["recorded_at"] == recorded
    assert instance.memory_authority_store.get_candidate(item["id"])["status"] == "committed"
