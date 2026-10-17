from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace as NS

from jsonschema import Draft202012Validator

from memory_l1_catalog_schema import model_document_schema
from memory_l1_ingestion import model_template
from memory_l1_schema import BASE_CATALOG, statement_template, unknown
from memory_l1_wire_schema import MODEL_DOCUMENT_SCHEMA


def _known(value):
    return {"state": "known", "value": value, "evidence_refs": []}


def _model_document(catalog=None):
    document = model_template(catalog=catalog)
    statement = statement_template()
    statement.pop("statement_revision_id")
    statement.update(statement_id="claim-1", content="一条合成陈述。")
    document["statements"] = [statement]
    return document


def _validate(schema, document):
    Draft202012Validator(schema).validate(document)


def _rejects(schema, document):
    assert list(Draft202012Validator(schema).iter_errors(document))


class FakeCompletionClient:
    """Capture an actual routed chat completion without network access."""

    def __init__(self, *, content, payload_calls):
        self.content = content
        self.payload_calls = payload_calls
        self.calls = []
        self.chat = NS(completions=self)

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        user_message = next(message for message in kwargs["messages"] if message["role"] == "user")
        self.payload_calls.append(json.loads(user_message["content"]))
        return NS(choices=[NS(message=NS(content=self.content), finish_reason="stop")])


def test_dynamic_schema_is_valid_with_default_and_empty_catalog_axes():
    default_schema = model_document_schema(catalog=None)
    Draft202012Validator.check_schema(default_schema)

    empty_catalog = {
        "kinds": [],
        "topics": [],
        "stabilities": [],
        "modalities": [],
        "bases": [],
        "entity_types": [],
        "relation_types": [],
        "attributes": [],
    }
    empty_schema = model_document_schema(catalog=empty_catalog)
    Draft202012Validator.check_schema(empty_schema)

    document = _model_document(empty_catalog)
    document["entities"] = [{
        "entity_id": "person-1",
        "entity_type": unknown(),
        "name": "合成的人",
        "aliases": [],
    }]
    _validate(empty_schema, document)

    document["statements"][0]["classification"]["kind"] = _known("unsupported")
    _rejects(empty_schema, document)


def test_template_document_without_program_fields_passes_dynamic_schema():
    schema = model_document_schema(catalog=copy.deepcopy(BASE_CATALOG))
    document = _model_document(BASE_CATALOG)

    assert "statement_revision_id" not in document["statements"][0]
    assert set(document) == {
        "entry", "narrative", "entities", "scenes", "statements",
        "retrieval", "relations", "attributes",
    }
    _validate(schema, document)


def test_default_kind_and_topic_vocabularies_constrain_model_fields_only():
    schema = model_document_schema(catalog=copy.deepcopy(BASE_CATALOG))
    document = _model_document(BASE_CATALOG)
    classification = document["statements"][0]["classification"]

    classification["kind"] = _known("interpretation")
    _validate(schema, document)

    classification["kind"] = _known("reflection")
    _rejects(schema, document)

    classification["kind"] = unknown()
    classification["topics"] = ["free-form topic"]
    _rejects(schema, document)

    classification["topics"] = []
    document["retrieval"]["tags"] = ["free-form retrieval tag"]
    _validate(schema, document)


def test_catalog_hot_edit_builds_fresh_schema_without_mutating_old_or_global_schema():
    original_catalog = copy.deepcopy(BASE_CATALOG)
    old_schema = model_document_schema(catalog=original_catalog)
    frozen_old_schema = copy.deepcopy(old_schema)
    frozen_global_schema = copy.deepcopy(MODEL_DOCUMENT_SCHEMA)

    edited_catalog = copy.deepcopy(original_catalog)
    edited_catalog["topics"].append("owner-added-topic")
    edited_catalog["kinds"].append("owner-added-kind")
    frozen_edited_catalog = copy.deepcopy(edited_catalog)
    new_schema = model_document_schema(catalog=edited_catalog)

    document = _model_document(edited_catalog)
    classification = document["statements"][0]["classification"]
    classification["kind"] = _known("owner-added-kind")
    classification["topics"] = ["owner-added-topic"]
    _validate(new_schema, document)
    _rejects(old_schema, document)

    assert old_schema == frozen_old_schema
    assert MODEL_DOCUMENT_SCHEMA == frozen_global_schema
    assert original_catalog == BASE_CATALOG
    assert edited_catalog == frozen_edited_catalog


def test_catalog_ids_relation_kinds_and_predicate_grammar_are_enforced():
    catalog = copy.deepcopy(BASE_CATALOG)
    catalog.update(
        entity_types=["person"],
        relation_types=["related_to"],
        attributes=[{
            "id": "mood",
            "value_type": "enum",
            "options": ["calm", "happy"],
            "scope": "any",
        }],
    )
    schema = model_document_schema(catalog=catalog)
    Draft202012Validator.check_schema(schema)

    document = _model_document(catalog)
    document["entities"] = [{
        "entity_id": "person-1",
        "entity_type": _known("person"),
        "name": "合成的人",
        "aliases": [],
    }]
    document["attributes"] = [{
        "definition_id": "mood",
        "value": _known("calm"),
        "evidence_refs": [],
    }]
    statement = document["statements"][0]
    statement["attributes"] = [{
        "definition_id": "mood",
        "value": _known("happy"),
        "evidence_refs": [],
    }]
    statement["property"] = {
        "key": "mood",
        "value": _known("calm"),
        "polarity": "affirmed",
        "evidence_refs": [],
    }
    statement["conditions"].update({
        "state": "known",
        "text": "心情为 calm",
        "predicate": {"op": "eq", "field": "mood", "value": "calm"},
    })
    document["relations"] = [{
        "relation_id": "relation-1",
        "from": {"object_type": "statement", "object_id": "claim-1"},
        "to": {"object_type": "memory", "object_id": "memory-1"},
        "kind": "related_to",
        "evidence_refs": [],
    }]
    _validate(schema, document)

    for mutate in (
        lambda doc: doc["entities"][0]["entity_type"].update(value="not_listed"),
        lambda doc: doc["attributes"][0].update(definition_id="not_listed"),
        lambda doc: doc["statements"][0]["attributes"][0].update(definition_id="not_listed"),
        lambda doc: doc["statements"][0]["property"].update(key="not_listed"),
        lambda doc: doc["statements"][0]["conditions"].update(
            predicate={"op": "eq", "field": "not_listed", "value": "calm"}),
        lambda doc: doc["relations"][0].update(kind="not_listed"),
    ):
        invalid = copy.deepcopy(document)
        mutate(invalid)
        _rejects(schema, invalid)

    for predicate in (
        {"op": "eval", "expression": "true"},
        {"op": "all", "args": []},
        {"op": "not", "args": []},
        {"op": "eq", "field": "mood", "value": "calm", "args": []},
    ):
        invalid = copy.deepcopy(document)
        invalid["statements"][0]["conditions"]["predicate"] = predicate
        _rejects(schema, invalid)


def test_empty_attribute_catalog_keeps_lists_empty_and_disables_structured_fields():
    catalog = copy.deepcopy(BASE_CATALOG)
    catalog["attributes"] = []
    schema = model_document_schema(catalog=catalog)
    Draft202012Validator.check_schema(schema)

    document = _model_document(catalog)
    _validate(schema, document)

    invalid_attribute = copy.deepcopy(document)
    invalid_attribute["attributes"] = [{
        "definition_id": "free-form",
        "value": unknown(),
        "evidence_refs": [],
    }]
    _rejects(schema, invalid_attribute)

    invalid_predicate = copy.deepcopy(document)
    invalid_predicate["statements"][0]["conditions"]["predicate"] = {
        "op": "eq", "field": "free-form", "value": "x",
    }
    _rejects(schema, invalid_predicate)

    invalid_property = copy.deepcopy(document)
    invalid_property["statements"][0]["property"] = {
        "key": "free-form", "value": unknown(), "polarity": "affirmed", "evidence_refs": [],
    }
    _rejects(schema, invalid_property)


def test_chat_preparation_sends_semantic_contract_from_the_same_live_catalog_snapshot(tmp_path):
    # Reuse the established synthetic model stub and its captured request payload.
    from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
    from memory_l1_semantic_wire import semantic_contract
    from test_chat_memory_preparation import setup as setup_chat_preparation

    env, engine, envelope, calls = setup_chat_preparation(tmp_path)
    document = model_template(catalog=env["catalog"])
    document["narrative"] = {"title": "后台拟题", "body": "模型尝试改写正文"}
    statement = statement_template()
    statement.pop("statement_revision_id")
    statement.update(statement_id="claim1", content="合成的当次请求", evidence_refs=["source:2"])
    document["statements"] = [statement]
    routed_client = FakeCompletionClient(
        content=json.dumps({"document": document, "kind": "shared_experience", "confidence": 0.95}),
        payload_calls=calls,
    )
    engine.config["reflection"]["memory_task_routes"] = {
        "chat_l1": {
            "connection": "dehydration",
            "model": "synthetic-chat-l1-catalog-v1",
            "thinking_mode": "enabled",
            "reasoning_effort": "high",
            "max_tokens": 7654,
        },
    }
    engine.config["dehydration"] = {"thinking_mode": "disabled"}
    engine.dehydration_client = routed_client
    engine.dehydration_model = "synthetic-dehydration-default"
    # The shared setup helper still supports legacy tests. This case exercises
    # the task-route method and its selected fake connection directly.
    del engine._daily_chat_memory_create_completion
    del engine._memory_task_route

    edited_catalog = copy.deepcopy(env["catalog"])
    edited_catalog["topics"].append("hot-edited-topic")
    edited_catalog["kinds"].append("hot-edited-kind")
    catalog_revision = "catalog-hot-edit-r9"
    env["revisions"].catalog_provider = lambda: (edited_catalog, catalog_revision)

    tasks = ChatMemoryPreparation(env["authority"])
    tasks.enqueue(envelope)
    result = asyncio.run(prepare_chat_tasks(engine, env["buckets"]))

    assert result["results"][0]["memory_status"] == "pending"
    assert len(calls) == 1
    assert len(routed_client.calls) == 1
    assert routed_client.calls[0]["model"] == "synthetic-chat-l1-catalog-v1"
    assert routed_client.calls[0]["max_tokens"] == 7654
    assert routed_client.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert routed_client.calls[0]["reasoning_effort"] == "high"
    payload = calls[0]
    assert payload["catalog_revision"] == catalog_revision
    binding_ids = [source["binding_id"] for source in payload["sources"]]
    assert payload["semantic_contract"] == semantic_contract(binding_ids, catalog=edited_catalog)
    assert not {"document_schema", "catalog", "document_template"}.intersection(payload)
    assert "hot-edited-topic" in payload["semantic_contract"]["catalog"]["topics"]
    assert "hot-edited-kind" in payload["semantic_contract"]["catalog"]["kinds"]
