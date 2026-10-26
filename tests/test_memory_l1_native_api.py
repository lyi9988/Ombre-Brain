"""No-network contracts for native L1 schemas, SDK requests and retained wire."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

from jsonschema import Draft202012Validator
import pytest

from memory_l1_api import build_schema, create_completion, prepare_request, validate_response
from memory_l1_schema import catalog_with_defaults
from memory_l1_semantic_wire import semantic_contract
from memory_l1_service import L1OwnerError
from memory_task_routes import resolve_task_route


STAGES = ("auto_l1", "chat_l1", "action_l1", "diary_l1")
SOURCE_IDS = ["source:1", "source:2"]


def catalog():
    return catalog_with_defaults({"attributes": [
        {"id": "pressure", "value_type": "number", "scope": "any",
         "label": "压力", "description": "主人定义的数值属性"}], "topics": ["散步"]})


def document():
    return {"format": "semantic-v1", "entry": {"form": "episode"},
        "narrative": {"title": "完整场景", "body": "起初约定周五散步，后来改成周六，仍待确认地点。"},
        "retrieval": {"keywords": ["散步改口", "待定地点"], "tags": ["约定"]},
        "entities": [{"entity_id": "e1", "name": "主人", "entity_type": "user", "aliases": ["润润"],
                      "evidence_refs": ["source:1"]}],
        "scenes": [{"scene_id": "scene1", "title": "散步约定", "statement_refs": ["s1"],
                    "source_refs": ["source:1", "source:2"],
                    "source_ranges": [{"source_ref": "source:1", "start": 0, "end": 5}],
                    "attributes": [{"definition_id": "pressure", "value": 2}]}],
        "statements": [{"statement_id": "s1", "content": "后来把散步改成周六，地点尚待确认。",
            "subject_ref": "e1", "classification": {"kind": "preference", "stability": "episode",
                "modality": "commitment", "basis": "direct_expression", "topics": ["散步"]},
            "time": {"event": {"precision": "day", "interpretation": "instant",
                "start": "2026-10-17", "end": None, "timezone": "Asia/Shanghai"},
                "applicability": {"temporal_scope": "recurring", "scene_ref": "scene1",
                    "start": None, "end": {"state": "unbounded"}, "recurrence_text": "每周六"}},
            "conditions": {"state": "known", "text": "压力较低时", "predicate": {"op": "lt", "field": "pressure", "value": 5},
                "evidence_refs": ["source:2"], "exceptions": [{"state": "known", "text": "下雨除外"}]},
            "property": {"key": "pressure", "value": 3, "polarity": "qualified"},
            "attributes": [{"definition_id": "pressure", "value": {"state": "unresolved"}}],
            "cognition": {"role": "hypothesis", "epistemic_status": "tentative", "about_refs": ["e1"],
                          "open_questions": ["地点尚未确认，后续需要问她。"]},
            "role_hint": "core", "evidence_refs": ["source:1", "source:2"]}],
        "relations": [{"from": {"object_type": "statement", "object_id": "s1"},
            "to": {"object_type": "scene", "object_id": "scene1"}, "kind": "supplements", "evidence_refs": ["source:2"]}],
        "attributes": [{"definition_id": "pressure", "value": 4}]}


def wrapped(stage, doc=None):
    doc = deepcopy(doc if doc is not None else document())
    if stage == "chat_l1":
        return {"document": doc, "kind": "shared_experience", "confidence": 0.9, "sensitive": False}
    if stage == "diary_l1":
        return {"document": doc, "scene_bodies": {"scene1": "完整场景正文，保留前后改口。"}, "confidence": 0.9}
    return {"candidates": [{"kind": "commitment", "confidence": 0.9, "source_refs": SOURCE_IDS,
                            "reason": "双方明确约定", "document": doc}]}


def messages(*, stage="auto_l1", nested=False):
    payload = {"identity": {"ai_name": "顾衍", "user_display_name": "润润"},
        "sources": [{"binding_id": "source:1", "text": "完整首段，周五散步。", "author": "user", "source_recorded_at": "2026-10-17T10:00+08:00"},
                    {"binding_id": "source:2", "text": "后来改成周六，地点待定。", "author": "user", "source_recorded_at": "2026-10-17T10:01+08:00"}],
        "semantic_contract": semantic_contract(SOURCE_IDS, catalog=catalog(), initial_chat=stage != "diary_l1")}
    if stage == "chat_l1":
        payload.update(narrative_from_draft=True, draft={"narrative": {"body": "聊天中已写好的完整原稿。"}})
    if nested:
        payload = {"date": "2026-10-17", "self_anchor_entry": "人格锚点原文", "l1_contract": payload}
    return [{"role": "system", "content": "保留完整语义。"}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_root_wrapper_keeps_full_optional_semantics(stage):
    schema = build_schema(SOURCE_IDS, catalog(), stage=stage)
    Draft202012Validator.check_schema(schema)
    original = wrapped(stage)
    Draft202012Validator(schema).validate(original)
    assert original == wrapped(stage)
    defs = schema["$defs"]
    assert {"scenes", "relations", "attributes"} <= set(defs["Document"]["properties"])
    assert {"property", "attributes", "cognition", "time", "conditions"} <= set(defs["Statement"]["properties"])
    assert defs["Cognition"]["properties"]["open_questions"]["items"]["type"] == "string"
    wrong_wrapper = {"document": document()} if stage in ("auto_l1", "action_l1") else {"candidates": []}
    assert not Draft202012Validator(schema).is_valid(wrong_wrapper)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("forged", ["source:current", "source:999", "raw-event:123"])
def test_all_evidence_and_scene_bindings_reject_non_source_ids(stage, forged):
    schema = build_schema(SOURCE_IDS, catalog(), stage=stage)
    assert schema["$defs"]["EvidenceRef"]["enum"] == SOURCE_IDS
    assert schema["$defs"]["SceneSourceRef"]["enum"] == SOURCE_IDS
    for mutate in (lambda doc: doc["statements"][0].update(evidence_refs=[forged]),
                   lambda doc: doc["scenes"][0].update(source_refs=[forged]),
                   lambda doc: doc["scenes"][0]["source_ranges"][0].update(source_ref=forged)):
        doc = document()
        mutate(doc)
        assert not Draft202012Validator(schema).is_valid(wrapped(stage, doc))


@pytest.mark.parametrize("stage", STAGES)
def test_prepare_request_retains_context_source_and_owner_catalog(stage):
    original = messages(stage=stage, nested=stage == "auto_l1")
    before = deepcopy(original)
    prepared, schema = prepare_request(original, stage=stage)
    old = json.loads(original[-1]["content"])
    new = json.loads(prepared[-1]["content"])
    old_envelope = old.get("l1_contract", old)
    new_envelope = new.get("l1_contract", new)
    assert original == before
    assert prepared[0] == original[0]
    assert new_envelope["sources"] == old_envelope["sources"]
    assert new_envelope["identity"] == old_envelope["identity"]
    assert new_envelope["semantic_contract"]["catalog"] == old_envelope["semantic_contract"]["catalog"]
    assert new_envelope["semantic_contract"]["source_ids"] == SOURCE_IDS
    assert new_envelope["semantic_contract"]["types"] == old_envelope["semantic_contract"]["types"]
    assert new_envelope["semantic_contract"]["document"] == old_envelope["semantic_contract"]["document"]
    assert schema["$defs"]["Attribute"]["properties"]["definition_id"]["enum"] == ["pressure"]
    assert schema["$defs"]["StatementProperty"]["oneOf"][0]["properties"]["key"]["enum"] == ["pressure"]
    if stage == "chat_l1":
        assert new_envelope["draft"] == old_envelope["draft"]
        assert "narrative" not in schema["$defs"]["Document"]["required"]


def test_scene_range_objects_are_still_rejected_without_repair():
    doc = document()
    doc["scenes"][0]["source_refs"] = [{"source_ref": "source:1", "start": 0, "end": 5}]
    raw = json.dumps(wrapped("chat_l1", doc), ensure_ascii=False)
    response = NS(choices=[NS(message=NS(content=raw), finish_reason="stop")],
                  native_schema=build_schema(SOURCE_IDS, catalog(), stage="chat_l1"))
    with pytest.raises(L1OwnerError) as caught:
        validate_response(response)
    assert caught.value.field_path == "document.scenes.0.source_refs.0"
    assert response.choices[0].message.content == raw


def test_reference_guidance_uses_original_contract_without_adding_source_text():
    original = messages(stage="chat_l1")
    prepared, schema = prepare_request(original, stage="chat_l1")
    payload = json.loads(prepared[-1]["content"])
    assert payload["sources"] == json.loads(original[-1]["content"])["sources"]
    assert payload["semantic_contract"]["types"]["Scene"]["source_refs?"] == "source_id[]"
    assert "source_ranges" in payload["semantic_contract"]["output_rules"]
    assert schema["$defs"]["Scene"]["properties"]["source_refs"]["items"] == {"$ref": "#/$defs/SceneSourceRef"}
    assert schema["$defs"]["SceneSourceRef"]["enum"] == SOURCE_IDS


class FakeClient:
    def __init__(self, raw='{"candidates": []}', *, base_url="https://api.deepseek.com/v1/", error=None):
        self.base_url = base_url
        self.raw = raw
        self.error = error
        self.options = []
        self.response_calls = []
        self.chat_calls = []
        self.responses = NS(create=self.create_response)
        self.chat = NS(completions=NS(create=self.create_chat))

    def with_options(self, **kwargs):
        self.options.append(deepcopy(kwargs))
        if "base_url" in kwargs:
            self.base_url = kwargs["base_url"]
        return self

    async def create_response(self, **kwargs):
        self.response_calls.append(deepcopy(kwargs))
        if self.error:
            raise self.error
        return NS(status="completed", model=kwargs["model"], output=[NS(type="message", content=[NS(type="output_text", text=self.raw)])],
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                   "output_tokens_details": {"reasoning_tokens": 7}})

    async def create_chat(self, **kwargs):
        self.chat_calls.append(deepcopy(kwargs))
        return NS(choices=[NS(message=NS(content=self.raw), finish_reason="stop")], usage=NS(total_tokens=30))


def route(client, stage="auto_l1", config=None):
    engine = NS(config=config or {"reflection": {}}, client=client, model="deepseek-chat",
                daily_chat_memory_client=None, dehydration_client=None, thinking_mode="disabled")
    return resolve_task_route(engine, stage)


@pytest.mark.parametrize("stage", STAGES)
def test_native_deepseek_one_call_root_low_32k_timeout_and_no_retries(stage):
    client = FakeClient(json.dumps(wrapped(stage), ensure_ascii=False))
    resolved = route(client, stage)
    response = asyncio.run(create_completion(client, model=resolved["model"], messages=messages(stage=stage),
        route=resolved, temperature=0.6))
    assert resolved["thinking_mode"] == "enabled"
    assert resolved["reasoning_effort"] == "low"
    assert resolved["max_tokens"] == 32000
    assert resolved["timeout_seconds"] == 180
    assert client.options == [{"timeout": 180, "max_retries": 0}, {"base_url": "https://api.deepseek.com/"}]
    assert len(client.response_calls) == 1 and not client.chat_calls
    request = client.response_calls[0]
    assert request["max_output_tokens"] == 32000
    assert request["reasoning"] == {"effort": "low"}
    assert request["text"]["format"]["type"] == "json_schema"
    assert response.native_request == request
    assert response.usage.prompt_tokens == 10 and response.usage.completion_tokens == 20
    assert response.usage.completion_tokens_details == {"reasoning_tokens": 7}
    validate_response(response)


def test_explicit_owner_route_options_are_preserved():
    client = FakeClient()
    resolved = route(client, config={"reflection": {"daily_chat_memory_timeout_seconds": 240,
        "memory_task_routes": {"auto_l1": {"thinking_mode": "disabled", "reasoning_effort": "high", "max_tokens": 48000}}}})
    response = asyncio.run(create_completion(client, model=resolved["model"], messages=messages(), route=resolved, temperature=0.4))
    assert client.options[0] == {"timeout": 240, "max_retries": 0}
    assert response.native_request["max_output_tokens"] == 48000
    assert response.native_request["reasoning"] == {"effort": "none"}


def test_non_deepseek_uses_single_chat_schema_request():
    client = FakeClient(base_url="https://example.invalid/v1")
    resolved = route(client)
    response = asyncio.run(create_completion(client, model=resolved["model"], messages=messages(), route=resolved, temperature=0.6))
    assert len(client.chat_calls) == 1 and not client.response_calls
    assert response.native_request["response_format"]["type"] == "json_schema"
    assert response.native_request["max_tokens"] == 32000
    assert client.options == [{"timeout": 180, "max_retries": 0}]


def test_native_error_propagates_without_protocol_fallback_or_retry():
    client = FakeClient(error=RuntimeError("synthetic provider failure"))
    resolved = route(client)
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        asyncio.run(create_completion(client, model=resolved["model"], messages=messages(), route=resolved, temperature=0.6))
    assert len(client.response_calls) == 1
    assert not client.chat_calls


@pytest.mark.parametrize("raw,code", [('{"candidates":', "native_l1_invalid_json"),
    ('{"document": {}}', "native_l1_schema_mismatch")])
def test_invalid_native_output_is_preserved_and_validator_rejects(raw, code):
    client = FakeClient(raw)
    resolved = route(client)
    response = asyncio.run(create_completion(client, model=resolved["model"], messages=messages(), route=resolved, temperature=0.6))
    assert response.choices[0].message.content == raw
    with pytest.raises(L1OwnerError, match=code):
        validate_response(response)
    assert response.choices[0].message.content == raw
    assert len(client.response_calls) == 1 and not client.chat_calls


def test_chat_native_model_io_retains_actual_input_and_invalid_original_output(tmp_path):
    from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
    from memory_model_io import MemoryModelIO
    from reflection_engine import ReflectionEngine
    from test_chat_memory_preparation import setup

    env, engine, envelope, _ = setup(tmp_path, compact=True)
    raw = '{"document":'
    client = FakeClient(raw)
    resolved = route(client, "chat_l1")
    engine._memory_task_route = lambda stage: {**resolved, "stage": stage}
    # Exercise the real native boundary instead of the fixture's legacy stub.
    engine._daily_chat_memory_create_completion = ReflectionEngine._daily_chat_memory_create_completion.__get__(engine)
    tasks = ChatMemoryPreparation(env["authority"])
    task_id = tasks.enqueue(envelope)["task_id"]
    asyncio.run(prepare_chat_tasks(engine, env["buckets"]))
    assert tasks.receipt(task_id)["memory_status"] == "preparation_failed"
    assert tasks.receipt(task_id)["issue"]["code"] == "native_l1_invalid_json"
    recorded = MemoryModelIO(env["authority"]).list(task_id)["items"]
    assert len(recorded) == 1
    row = recorded[0]
    assert row["status"] == "failed" and row["raw_output"] == raw
    assert row["validation"]["issue"]["code"] == "native_l1_invalid_json"
    assert len(client.response_calls) == 1 and not client.chat_calls
    assert row["native_request"] == client.response_calls[0]
    assert row["messages"] == client.response_calls[0]["input"]
    assert row["completion_parameters"]["max_output_tokens"] == 32000
    assert row["completion_parameters"]["reasoning"] == {"effort": "low"}
    payload = json.loads(row["messages"][-1]["content"])
    assert payload["draft"] == envelope["arguments"]
    assert [source["text"] for source in payload["sources"]] == [source["text"] for source in envelope["context"]["sources"]]
    assert row["usage"]["reasoning_tokens"] == 7
    assert not env["authority"].list_candidates(status="all")


def test_chat_native_timeout_retains_actual_input_before_dispatch_without_retry(tmp_path):
    from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
    from memory_model_io import MemoryModelIO
    from reflection_engine import ReflectionEngine
    from test_chat_memory_preparation import setup

    env, engine, envelope, _ = setup(tmp_path, compact=True)
    captured_before_timeout = []

    class TimeoutClient(FakeClient):
        async def create_response(self, **kwargs):
            # Read the real persisted row while dispatch is starting. These
            # fields must already exist without receiving a provider response.
            current = MemoryModelIO(env["authority"]).list(task_id)["items"]
            assert len(current) == 1
            assert current[0]["status"] == "running"
            assert current[0]["native_request"] == kwargs
            assert current[0]["messages"] == kwargs["input"]
            captured_before_timeout.append(deepcopy(current[0]))
            return await super().create_response(**kwargs)

    client = TimeoutClient(error=TimeoutError("synthetic provider timeout"))
    resolved = route(client, "chat_l1")
    engine._memory_task_route = lambda stage: {**resolved, "stage": stage}
    engine._daily_chat_memory_create_completion = ReflectionEngine._daily_chat_memory_create_completion.__get__(engine)
    tasks = ChatMemoryPreparation(env["authority"])
    task_id = tasks.enqueue(envelope)["task_id"]
    asyncio.run(prepare_chat_tasks(engine, env["buckets"]))
    assert tasks.receipt(task_id)["memory_status"] == "preparation_failed"
    assert tasks.receipt(task_id)["issue"]["code"] == "TimeoutError"
    rows = MemoryModelIO(env["authority"]).list(task_id)["items"]
    assert len(rows) == len(captured_before_timeout) == len(client.response_calls) == 1
    assert not client.chat_calls
    row = rows[0]
    assert row["status"] == "failed" and row["raw_output"] is None
    assert row["error"]["code"] == "TimeoutError"
    assert row["native_request"] == client.response_calls[0]
    assert row["native_request"]["text"]["format"]["schema"]["$defs"]["EvidenceRef"]["enum"] == ["source:1", "source:2", "source:3"]
    assert row["messages"] == client.response_calls[0]["input"]
    assert row["transport"] == {"timeout_seconds": 180, "max_retries": 0}
    assert row["completion_parameters"]["max_output_tokens"] == 32000
    assert row["completion_parameters"]["reasoning"] == {"effort": "low"}
    payload = json.loads(row["messages"][-1]["content"])
    assert payload["draft"] == envelope["arguments"]
    assert [source["text"] for source in payload["sources"]] == [source["text"] for source in envelope["context"]["sources"]]
    assert not env["authority"].list_candidates(status="all")


def test_authority_daily_run_skips_unconsumed_summary_and_sends_complete_original_sources(tmp_path):
    from reflection_engine import ReflectionEngine
    from test_chat_memory_preparation import setup
    from test_daily_chat_memory_quality_review_v4 import FakeRawEvents, FakeTurns

    env, engine, _, _ = setup(tmp_path, compact=True)
    assert engine.memory_authority_enabled
    client = FakeClient('{"candidates": []}')
    resolved = route(client, "auto_l1")
    engine._memory_task_route = lambda stage: {**resolved, "stage": stage}
    engine._daily_chat_memory_create_completion = ReflectionEngine._daily_chat_memory_create_completion.__get__(engine)
    summary_calls = []

    async def unused_summary(*args, **kwargs):
        summary_calls.append(True)
        raise AssertionError("source-based L1 must not call the unused summary stage")

    engine._summarize_daily_chat_memory_windows = unused_summary
    user_text = "这次约定从完整原文开始。" + "原文内容" * 1000 + "最后改口为周六，不要丢掉尾部。"
    assistant_text = "我记住前后的变化。" + "助手原文" * 500 + "保留待定地点。"
    events = [{"id": 1, "role": "user", "text": user_text, "session_id": "s",
               "created_at": "2026-08-18T09:00:00+08:00", "metadata": {"round_id": 1}},
              {"id": 2, "role": "assistant", "text": assistant_text, "session_id": "s",
               "created_at": "2026-08-18T09:00:01+08:00", "metadata": {"round_id": 1}}]
    result = asyncio.run(engine.run_daily_chat_memory(env["buckets"],
        raw_event_store=FakeRawEvents(events), conversation_turn_store=FakeTurns([]),
        key="2026-08-18", mode="review"))
    assert result["status"] == "zero_candidates"
    assert result["last_raw_event_id"] == 2
    assert not summary_calls
    assert len(client.response_calls) == 1 and not client.chat_calls
    payload = json.loads(client.response_calls[0]["input"][-1]["content"])
    assert [source["text"] for source in payload["l1_contract"]["source_texts"]] == [user_text, assistant_text]
    assert payload["l1_contract"]["semantic_contract"]["source_ids"] == ["source:1", "source:2"]
    assert engine.list_daily_chat_memory_run_audit(limit=1)[0]["summary_stage"] == "not_needed_for_source_based_l1"
