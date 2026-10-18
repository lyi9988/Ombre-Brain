import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from memory_l1_catalog_schema import model_document_schema
from memory_l1_schema import catalog_with_defaults
from memory_l1_semantic_wire import assemble_document, semantic_contract, semantic_sources
from memory_l1_service import L1OwnerError
from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from test_chat_memory_preparation import setup as setup_chat
from identity import identity_names, model_identity


CHAT_OMITTED_RELATIONS = {"revises", "supersedes", "split_from", "merged_from"}


def compact_document():
    return {
        "format": "semantic-v1",
        "entities": [{"name": "主人", "entity_type": "user"}],
        "statements": [{
            "content": "那次想要抱抱。",
            "subject_ref": "e1",
            "classification": {
                "kind": "preference",
                "stability": "episode",
                "modality": "request",
                "basis": "direct_expression",
            },
            "time": None,
            "conditions": None,
            "evidence_refs": ["source:current"],
        }],
    }


def parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def set_test_identity(engine):
    engine.config["identity"] = {
        "ai_name": "配置里的助手名",
        "user_name": "配置里的用户ID",
        "user_display_name": "配置里的显示名",
        "user_aliases": ["小润", "配置昵称"],
    }
    engine.identity = identity_names(engine.config)
    return model_identity(engine.config)


def test_initial_chat_directory_omits_only_four_relation_kinds_and_full_schema_stays_complete():
    catalog = catalog_with_defaults()
    catalog["relation_types"].append("owner_custom_relation")
    catalog["definitions"].extend([
        {"axis": "relation_types", "id": "owner_custom_relation", "label": "自定义关系",
         "description": "主人定义的关系说明", "aliases": [], "examples": [], "parent_id": None},
        {"axis": "cognition_statuses", "id": "revised", "label": "已修订",
         "description": "作为旧理解保留，并沿修订关系看当前版本。",
         "aliases": [], "examples": [], "parent_id": None},
        {"axis": "cognition_statuses", "id": "resolved", "label": "已梳理",
         "description": "主人自定义的已梳理说明", "aliases": [], "examples": [], "parent_id": None},
    ])
    original = copy.deepcopy(catalog)

    chat_contract = semantic_contract(["source:1"], catalog=catalog, initial_chat=True)
    full_contract = semantic_contract(["source:1"], catalog=catalog)
    full_schema = model_document_schema(["source:1"], catalog=catalog)

    chat_relation_kinds = set(chat_contract["catalog"]["relation_types"])
    assert CHAT_OMITTED_RELATIONS.isdisjoint(chat_relation_kinds)
    assert chat_relation_kinds == set(catalog["relation_types"]) - CHAT_OMITTED_RELATIONS
    assert "owner_custom_relation" in chat_relation_kinds
    assert chat_contract["catalog"]["relation_types"]["owner_custom_relation"] == {
        "label": "自定义关系", "description": "主人定义的关系说明"
    }
    # A standard default is reduced to its short label; owner-edited extra
    # descriptions remain explicit objects in the model-facing directory.
    assert chat_contract["catalog"]["cognition_statuses"]["revised"] == "已修订"
    assert chat_contract["catalog"]["cognition_statuses"]["resolved"] == {
        "label": "已梳理", "description": "主人自定义的已梳理说明"
    }
    assert set(full_contract["catalog"]["relation_types"]) == set(catalog["relation_types"])
    assert full_contract["catalog"]["cognition_statuses"]["revised"]["description"] == (
        "作为旧理解保留，并沿修订关系看当前版本。"
    )
    assert full_contract["catalog"]["relation_types"]["owner_custom_relation"]["description"] == (
        "主人定义的关系说明"
    )
    assert "relations" in full_contract["document"]["optional"]

    schema_text = json.dumps(full_schema)
    assert all(kind in schema_text for kind in CHAT_OMITTED_RELATIONS)
    assert catalog == original


def test_require_keywords_is_opt_in_and_valid_keywords_survive_assembly():
    catalog = catalog_with_defaults()
    schema_before = model_document_schema(["source:1"], catalog=catalog)
    default_contract = semantic_contract(["source:1"], catalog=catalog)
    required_contract = semantic_contract(["source:1"], catalog=catalog, require_keywords=True)

    assert "retrieval" not in default_contract["document"]["required"]
    assert "retrieval" in default_contract["document"]["optional"]
    assert "retrieval" in required_contract["document"]["required"]
    assert "retrieval" not in required_contract["document"]["optional"]
    assert required_contract["document"]["retrieval"]["keywords"] == (
        "non-empty string[];由模型根据材料提供"
    )
    assert model_document_schema(["source:1"], catalog=catalog) == schema_before

    keywords = ["那次疲惫", "需要拥抱"]
    draft = compact_document()
    draft["retrieval"] = {"keywords": list(keywords), "tags": ["关系"]}
    original = copy.deepcopy(draft)
    assembled = assemble_document(
        draft,
        narrative={"body": "完整回忆"},
        require_keywords=True,
    )
    assert assembled["retrieval"]["keywords"] == keywords
    assert assembled["retrieval"]["tags"] == ["关系"]
    assert draft == original

    legacy = assemble_document(compact_document(), narrative={"body": "兼容正文"})
    assert legacy["retrieval"] == {"keywords": [], "tags": []}


@pytest.mark.parametrize(
    "case",
    ["missing_retrieval", "missing_keywords", "empty_keywords", "non_string_keyword"],
)
def test_required_keywords_reject_missing_empty_or_non_string_values(case):
    draft = compact_document()
    if case == "missing_retrieval":
        pass
    elif case == "missing_keywords":
        draft["retrieval"] = {"tags": []}
    elif case == "empty_keywords":
        draft["retrieval"] = {"keywords": [], "tags": []}
    else:
        draft["retrieval"] = {"keywords": ["有意义的词", 7], "tags": []}

    expected_error = (
        "semantic_fields_required"
        if case in {"missing_retrieval", "missing_keywords"}
        else "nonempty_keywords_required"
    )
    with pytest.raises(L1OwnerError, match=expected_error):
        assemble_document(
            draft,
            narrative={"body": "完整正文"},
            require_keywords=True,
        )


def test_chat_preparation_sends_scoped_contract_and_minute_timezone_source_view(tmp_path):
    env, engine, envelope, calls = setup_chat(tmp_path)
    expected_identity = set_test_identity(engine)
    engine.tz = timezone(timedelta(hours=8))
    for source in envelope["context"]["sources"]:
        source["created_at"] = "2026-10-17T22:08:33.987654Z"
    raw_sources = copy.deepcopy(envelope["context"]["sources"])

    tasks = ChatMemoryPreparation(env["authority"])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env["buckets"]))

    assert result["results"][0]["memory_status"] == "pending"
    assert len(calls) == 1
    model_input = calls[0]
    assert model_input["identity"] == expected_identity
    semantic = model_input["semantic_contract"]
    for axis in ("subjects", "entity_types"):
        assert semantic["catalog"][axis]["user"] == "润润"
        assert semantic["catalog"][axis]["assistant"] == "顾衍"
    assert semantic["catalog"]["kinds"] == {
        "experience": "经历", "state": "状态", "fact": "事实", "preference": "偏好",
        "boundary": "边界", "intention": "意向", "interpretation": "解释", "procedure": "做法",
    }
    sent_relation_kinds = set(model_input["semantic_contract"]["catalog"]["relation_types"])
    assert CHAT_OMITTED_RELATIONS.isdisjoint(sent_relation_kinds)

    view_time = parse_timestamp(model_input["sources"][0]["source_recorded_at"])
    assert (view_time.year, view_time.month, view_time.day, view_time.hour, view_time.minute) == (
        2026, 10, 18, 6, 8
    )
    assert view_time.second == 0 and view_time.microsecond == 0
    assert view_time.utcoffset() == timedelta(hours=8)
    assert envelope["context"]["sources"] == raw_sources


def test_auto_l1_sends_each_source_once_and_keeps_legacy_window_input(tmp_path, monkeypatch):
    env, engine, _, calls = setup_chat(tmp_path)
    expected_identity = set_test_identity(engine)
    assert expected_identity == {
        "user": {"name": "润润", "aliases": ["小润", "配置昵称"]},
        "assistant": {"name": "顾衍", "aliases": []},
    }
    monkeypatch.setattr(
        engine.memory_authority_store,
        "find_candidates_by_sources",
        lambda _source_keys: [{
            "candidate_id": "comparison-only-existing",
            "revision": 3,
            "status": "pending",
            "proposal": {
                "proposed_body": "仅供比较的既有候选，不含本次对话原文。",
                "source_type": "synthetic-test",
            },
        }],
    )
    engine.memory_authority_enabled = True
    engine.tz = timezone(timedelta(hours=8))
    system_prompt, prompt_context = engine._memory_l1_prompt(
        "ombre.daily_chat_memory_prompt",
        runtime_values={"max_candidates": 1},
    )
    assert "本次对话角色：user=润润，assistant=顾衍。" in system_prompt
    engine._memory_l1_prompt = lambda source_id, runtime_values=None: (system_prompt, prompt_context)

    user_text = "润润说：周末改到周日，票还没买。"
    assistant_text = "顾衍回应：我记下改期和未购票这两件事。"
    created_at = "2026-10-18T10:00:00+08:00"
    system_prompts = []
    async def auto_completion(client, **kwargs):
        calls.append(json.loads(kwargs["messages"][1]["content"]))
        system_prompts.append(kwargs["messages"][0]["content"])
        document = compact_document()
        document["narrative"] = {"title": "周末改期", "body": "合成的完整回忆"}
        document["retrieval"] = {"keywords": ["周末改期", "未购票"], "tags": []}
        document["statements"][0]["evidence_refs"] = ["source:2"]
        response = {"candidates": [{"document": document, "kind": "shared_experience",
                                    "confidence": 0.95}]}
        return NS(choices=[NS(message=NS(content=json.dumps(response, ensure_ascii=False)),
                              finish_reason="stop")])
    engine._daily_chat_memory_create_completion = auto_completion
    window = [{
        "id": 44,
        "session_id": "synthetic-conversation",
        "created_at": created_at,
        "raw_event_ids": [101, 102],
        "user_text": user_text,
        "assistant_text": assistant_text,
        "_semantic_sources": [
            {"id": 101, "session_id": "synthetic-conversation", "role": "user",
             "text": user_text, "created_at": created_at, "metadata": {}},
            {"id": 102, "session_id": "synthetic-conversation", "role": "assistant",
             "text": assistant_text, "created_at": created_at, "metadata": {}},
        ],
    }]
    original_window = copy.deepcopy(window)

    candidates, stats = asyncio.run(engine._extract_window_candidates("2026-10-18", window))

    assert stats["finish_reason"] == "stop"
    assert len(calls) == 1 and len(candidates) == 1
    model_input = calls[0]
    assert "conversation_turns" not in model_input
    assert model_input["identity"] == expected_identity
    assert system_prompts[0] == system_prompt
    assert list(model_input)[-1] == "l1_contract"
    assert model_input["existing_source_memories"][0]["candidate_id"] == "comparison-only-existing"
    l1_contract = model_input["l1_contract"]
    assert list(l1_contract) == ["source_texts", "catalog_revision", "semantic_contract"]
    contract = l1_contract["semantic_contract"]
    assert list(contract)[-1] == "response_format"
    assert "narrative" in contract["document"]["required"]
    assert "narrative" not in contract["document"]["optional"]
    assert "retrieval" in contract["document"]["required"]
    assert contract["document"]["retrieval"]["keywords"] == "non-empty string[];由模型根据材料提供"
    assert contract["catalog"]["subjects"]["user"] == "润润"
    assert contract["catalog"]["subjects"]["assistant"] == "顾衍"
    response_format = contract["response_format"]
    assert response_format["root"] == {"candidates": "Candidate[]"}
    assert response_format["Candidate"] == {
        "kind": "key_event|stable_preference|boundary|signal|commitment|project_state|relationship_anchor|self_insight|love_letter",
        "confidence": "number in [0,1]",
        "source_refs": "source_id[]",
        "reason": "string",
        "document": "Document (document/types above)",
    }
    rules = response_format["rules"]
    assert "subject_ref/about_refs只引用本稿实体ID，例如e1，不能用user/assistant代替。" in rules
    assert "保留主体、改口、否定、条件和未知" in rules
    assert "不同日期的状态分别表达" in rules
    assert "计划目标日不自动作为截止日" in rules
    assert "每份document独立填写完整narrative与非空retrieval.keywords" in rules
    source_texts = l1_contract["source_texts"]
    assert [source["text"] for source in source_texts] == [user_text, assistant_text]
    serialized = json.dumps(model_input, ensure_ascii=False)
    assert serialized.count(user_text) == 1
    assert serialized.count(assistant_text) == 1
    assert candidates[0]["source_turn_ids"] == ["44"]
    assert window == original_window

    engine.memory_authority_enabled = False
    legacy_candidates, legacy_stats = asyncio.run(engine._extract_window_candidates("2026-10-18", window))
    assert legacy_stats["finish_reason"] == "stop"
    assert len(calls) == 2 and len(legacy_candidates) == 1
    legacy_input = calls[1]
    assert "l1_contract" not in legacy_input
    assert legacy_input["conversation_turns"][0]["user_text"] == user_text
    assert legacy_input["conversation_turns"][0]["assistant_text"] == assistant_text
    assert window == original_window


def make_auto_l1_engine(tmp_path, raw_output):
    env, engine, _, calls = setup_chat(tmp_path)
    set_test_identity(engine)
    engine.memory_authority_enabled = True
    engine.tz = timezone(timedelta(hours=8))
    engine._memory_l1_prompt = lambda source_id, runtime_values=None: (
        "synthetic auto L1 prompt",
        {"prompt_revision": "synthetic-prompt-r1"},
    )

    async def completion(client, **kwargs):
        calls.append(json.loads(kwargs["messages"][1]["content"]))
        return NS(choices=[NS(message=NS(content=raw_output), finish_reason="stop")])

    engine._daily_chat_memory_create_completion = completion
    created_at = "2026-10-18T10:00:00+08:00"
    window = [{
        "id": 54,
        "session_id": "synthetic-conversation",
        "created_at": created_at,
        "raw_event_ids": [201, 202],
        "user_text": "润润：周末安排暂时未知。",
        "assistant_text": "顾衍：我会等你确认。",
        "_semantic_sources": [
            {"id": 201, "session_id": "synthetic-conversation", "role": "user",
             "text": "润润：周末安排暂时未知。", "created_at": created_at, "metadata": {}},
            {"id": 202, "session_id": "synthetic-conversation", "role": "assistant",
             "text": "顾衍：我会等你确认。", "created_at": created_at, "metadata": {}},
        ],
    }]
    return engine, window, calls


def test_auto_l1_rejects_bare_semantic_document_without_candidates(tmp_path):
    bare_document = compact_document()
    bare_document["narrative"] = {"title": "周末安排", "body": "合成回忆"}
    bare_document["retrieval"] = {"keywords": ["周末安排"], "tags": []}
    engine, window, calls = make_auto_l1_engine(
        tmp_path,
        json.dumps(bare_document, ensure_ascii=False),
    )

    with pytest.raises(L1OwnerError, match="first_pass_candidates_required"):
        asyncio.run(engine._extract_window_candidates("2026-10-18", window))

    assert len(calls) == 1


def test_auto_l1_accepts_empty_candidates_list(tmp_path):
    engine, window, calls = make_auto_l1_engine(
        tmp_path,
        json.dumps({"candidates": []}, ensure_ascii=False),
    )

    candidates, stats = asyncio.run(engine._extract_window_candidates("2026-10-18", window))

    assert len(calls) == 1
    assert candidates == []
    assert stats["parse_failure"] is False


def test_source_projection_keeps_raw_snapshot_and_binding_unchanged():
    snapshots = [{
        "text": "完整来源原文。" * 80,
        "author": "user",
        "recorded_at": "2026-10-17T22:08:33.987654Z",
        "source_revision": "program-only",
    }]
    bindings = [{"binding_id": "source:1", "source_revision": "program-only",
                 "anchors": [{"sha": "program-only"}]}]
    original = copy.deepcopy((snapshots, bindings))

    projected = semantic_sources(
        snapshots,
        bindings,
        chat_view=True,
        source_timezone=timezone(timedelta(hours=8)),
    )

    view_time = parse_timestamp(projected[0]["source_recorded_at"])
    assert (view_time.year, view_time.month, view_time.day, view_time.hour, view_time.minute) == (
        2026, 10, 18, 6, 8
    )
    assert view_time.second == 0 and view_time.microsecond == 0
    assert view_time.utcoffset() == timedelta(hours=8)
    assert (snapshots, bindings) == original


@pytest.mark.parametrize(
    "path",
    [
        ("time",),
        ("conditions",),
        ("subject_ref",),
        ("evidence_refs",),
        ("classification", "stability"),
    ],
)
def test_missing_required_semantic_judgments_still_fail(path):
    document = compact_document()
    target = document["statements"][0]
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]

    with pytest.raises(L1OwnerError, match="semantic_fields_required"):
        assemble_document(document, narrative={"body": "完整正文"})
