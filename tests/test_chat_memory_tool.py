import asyncio
import copy

import pytest

from chat_memory_tool import ChatMemoryTool, candidate_id, receipt
from memory_authority import IdempotencyConflict
from memory_narrative import memory_context_labels
from reflection_engine import ReflectionEngine
from test_memory_authority_r1 import reflection_config, attach_fake_commit_service


def envelope(**changes):
    value = {'operation_id': 'op-' + 'a' * 32,
             'arguments': {'title': '一次回忆', 'content': '我听她说，她在昨天搬了家，没有答应长期住那里。',
                           'kind': 'shared_experience', 'confidence': 0.9,
                           'event_time': {'expression': '昨天', 'evidence': '我昨天搬了家。'}},
             'context': {'conversation_id': 'c', 'request_id': 'r', 'turn_id': 'evt-1',
                         'sources': [{'event_id': 'evt-1', 'version_id': 'v1', 'role': 'user',
                                      'text': '我昨天搬了家。没有答应长期住那里。', 'created_at': '2026-10-08T17:00:00Z'}]}}
    value['arguments'].update(changes)
    return value


def service(tmp_path, mode='auto'):
    engine = ReflectionEngine(reflection_config(tmp_path, mode=mode))
    projection = attach_fake_commit_service(engine)
    return ChatMemoryTool(engine, object()), projection


def test_auto_body_time_provenance_and_same_operation_replay(tmp_path):
    tool, projection = service(tmp_path)
    async def run():
        value = envelope()
        result = await tool.submit(value)
        assert result['memory_status'] == 'committed'
        assert (await tool.submit(value))['memory_id'] == result['memory_id']
        assert len(projection.revisions) == 1
        row = tool.store.get_candidate(result['memory_id'])
        assert row['proposal']['owner_explicit'] is False
        assert row['proposal']['proposed_body'] == value['arguments']['content']
        narrative = row['proposal']['metadata']['narrative']
        assert narrative['generation_kind'] == 'chat_authored'
        assert narrative['event_time']['precision'] == 'expression'
        assert narrative['event_time']['reference_date'] == '2026-10-09'
        assert narrative['sources'][0]['version_id'] == 'v1'
        assert 'conversation_turn:' not in str(narrative)
        with pytest.raises(IdempotencyConflict):
            await tool.submit(envelope(content='different'))
    asyncio.run(run())


def test_chat_l1_body_over_request_limit_has_specific_error_without_slicing(tmp_path):
    tool, _projection = service(tmp_path)
    document = {
        "entry": {"form": "episode"},
        "narrative": {"title": "超限正文", "body": "界" * (512 * 1024 + 1)},
        "entities": [], "scenes": [], "statements": [],
        "retrieval": {"keywords": [], "tags": []}, "relations": [], "attributes": [],
    }
    request = envelope()
    request["arguments"] = {"document": document}
    with pytest.raises(ValueError, match="l1_narrative_body_too_large"):
        tool._proposal(request)


def test_chat_source_current_requires_the_actual_user_turn(tmp_path):
    tool, _projection = service(tmp_path)
    document = {
        "entry": {"form": "episode"},
        "narrative": {"title": "当前来源", "body": "保留当前说法。"},
        "entities": [], "scenes": [], "statements": [],
        "retrieval": {"keywords": [], "tags": []}, "relations": [], "attributes": [],
    }
    request = envelope()
    request["arguments"] = {"document": document}
    request["context"]["sources"][-1]["role"] = "assistant"
    with pytest.raises(ValueError, match="current_user_source_must_be_last"):
        tool._proposal(request)


def test_aiz_catalog_adapter_commits_first_pass_l1_through_temporary_brain_route(tmp_path, monkeypatch):
    """Use Aiz's real catalog/adapter and Brain's real handler with isolated stores."""
    import ast
    import httpx
    import hmac
    import logging
    import os
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from starlette.applications import Starlette
    from starlette.routing import Route

    brain_root = Path(__file__).resolve().parents[1]
    aiz_root = brain_root.parent / "aizizhu-memory-chat-submit-r1"
    monkeypatch.syspath_prepend(str(aiz_root))
    monkeypatch.syspath_prepend(str(aiz_root / "tests"))
    from core.conversation_service import ConversationService, current_turn_context
    from core.conversation_store import ConversationStore
    from core.memory_tool import MemoryToolAdapter
    from core.sdk_tools import SdkToolCatalog
    from core.tool_operations import ToolOperationStore, ToolOperationManager
    from core.tool_registry import default_tool_registry
    from test_tool_write_r1 import _settings

    config = reflection_config(tmp_path, mode="auto")
    engine = ReflectionEngine(config)
    bucket_manager = __import__("bucket_manager").BucketManager(config)
    engine._memory_authority_service = lambda manager: __import__("memory_commit_service").MemoryCommitService(
        engine.memory_authority_store,
        __import__("memory_commit_service").BucketMemoryProjection(config, bucket_manager=manager),
    )

    server_source = (brain_root / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(server_source, filename=str(brain_root / "server.py"))
    selected = []
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in {"_memory_write_token", "_authorized_memory_write", "api_chat_memory_tool"}):
            node.decorator_list = []
            selected.append(node)
    namespace = {
        "__name__": "temporary_brain_chat_memory_route",
        "os": os,
        "hmac": hmac,
        "config": config,
        "reflection_engine": engine,
        "memory_authority_store": engine.memory_authority_store,
        "bucket_mgr": bucket_manager,
        "logger": logging.getLogger("test.temporary_brain_chat_memory"),
    }
    assert {node.name for node in selected} == {"_memory_write_token", "_authorized_memory_write", "api_chat_memory_tool"}
    monkeypatch.setenv("OMBRE_MEMORY_WRITE_TOKEN", "temporary-brain-token")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(brain_root / "server.py"), "exec"), namespace)
    brain_app = Starlette(routes=[
        Route("/api/memory-authority/chat-tool", namespace["api_chat_memory_tool"], methods=["POST"]),
        Route("/api/memory-authority/chat-tool/{operation_id}", namespace["api_chat_memory_tool"], methods=["GET"]),
    ])
    transport = httpx.ASGITransport(app=brain_app)

    import core.memory_tool as aiz_memory_tool_module
    real_async_client = httpx.AsyncClient

    class RoutedAsyncClient:
        def __init__(self, **kwargs):
            self.client = real_async_client(transport=transport, base_url="http://brain.test", **kwargs)

        async def __aenter__(self):
            await self.client.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self.client.__aexit__(*args)

        async def request(self, *args, **kwargs):
            return await self.client.request(*args, **kwargs)

    monkeypatch.setattr(aiz_memory_tool_module, "httpx", SimpleNamespace(AsyncClient=RoutedAsyncClient))

    registry = default_tool_registry()
    settings = _settings(tmp_path, registry, enabled=["memory.write", "memory.status"], permission="allow")
    operations = ToolOperationStore(tmp_path / "aiz-operations.sqlite3")
    operation_manager = ToolOperationManager(store=operations, registry=registry, settings=settings)
    conversation_store = ConversationStore(tmp_path / "aiz-conversation.sqlite3")
    adapter = MemoryToolAdapter(conversation_store,
        credentials=lambda: ("http://brain.test", "temporary-brain-token"))
    operation_manager.register_adapter("memory.write", adapter)
    catalog = SdkToolCatalog({}, settings=settings, registry=registry, operation_manager=operation_manager)
    catalog.register_executor("memory.status", adapter.status)
    observed = []
    status_observed = []
    document = {
        "entry": {"form": "episode"},
        "narrative": {"title": "一段被认真记住的经历", "body": "我记得这次谈话里那份明确的请求。" * 800},
        "entities": [], "scenes": [],
        "statements": [],
        "retrieval": {"keywords": ["请求"], "tags": ["shared_experience"]},
        "relations": [], "attributes": [],
    }
    from memory_l1_schema import statement_template
    statement = statement_template()
    statement.pop("statement_revision_id")
    statement["statement_id"] = "new:statement"
    statement["content"] = "主人希望这段谈话被认真记住。"
    statement["evidence_refs"] = ["source:current"]
    document["statements"] = [statement]
    original_source = "我希望你把这次谈话认真记住。"

    async def generate(_messages):
        context = current_turn_context()
        event = context["user_event"]
        catalog.bind_turn("logical-request", conversation_id=event["conversation_id"], turn_id=event["id"])
        catalog.bind_tool_call("l1-write-call")
        result = await catalog.execute("memory.write", {"narrative": document["narrative"]}, explicit_intent=True)
        observed.append(result)
        catalog.bind_tool_call("l1-status-call")
        status_result = await catalog.execute(
            "memory.status", {"operation_id": result["operation_id"]}, explicit_intent=True)
        status_observed.append(status_result)
        return "我会记住这段经历。"

    turn = asyncio.run(ConversationService(conversation_store, generate).handle_inbound(
        "temporary-conversation", source="app", source_event_id="temporary-ingress", content=original_source))
    assert turn["status"] == "completed", (turn.get("failure_event"), observed)
    assert observed[0].get("ok"), observed[0]
    assert observed[0]["result"]["memory_status"] == "preparing"
    async def locate(_binding): return {'state': 'available', 'eligible_source': True}
    from memory_l0_sources import MemoryL0Sources
    engine._memory_l1_sources = lambda: SimpleNamespace(locate=locate,
        retain_copy=MemoryL0Sources(engine.memory_authority_store).retain_copy)
    engine._daily_chat_memory_model_client = lambda **kw: (object(), 'synthetic-model', False)
    async def create(_client, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=__import__('json').dumps({'document': document, 'kind': 'key_event', 'confidence': 0.95})), finish_reason='stop')])
    engine._daily_chat_memory_create_completion = create
    from memory_chat_preparation import prepare_chat_tasks
    prepared = asyncio.run(prepare_chat_tasks(engine, bucket_manager))
    observed[0]['result'] = prepared['results'][0]
    assert observed[0]['result']['memory_status'] == 'committed'
    assert observed[0]["result"]["l1_state"] == "current"
    assert observed[0]["result"]["l1_complete"] is True
    assert status_observed[0]["memory_status"] == "preparing"
    status_observed[0] = asyncio.run(adapter.status({"operation_id": observed[0]["operation_id"]}))
    assert status_observed[0]["memory_status"] == "committed"
    assert status_observed[0]["l1_state"] == "current" and status_observed[0]["l1_complete"] is True
    rows = engine.memory_authority_store.list_candidates(status="all", limit=10)
    row = next(item for item in rows if item["proposal"].get("source_type") == "chat_tool")
    candidate_id_value = row["candidate_id"]
    proposal = row["proposal"]
    assert len(proposal["proposed_body"]) > 12000
    assert proposal["proposed_body"] == document["narrative"]["body"]
    semantic = proposal["metadata"]["memory_l1"]
    assert semantic["generation"]["origin"] == "first_pass_model"
    assert semantic["generation"]["prompt_revision"].startswith("prompt_sha256:")
    assert semantic["generation"]["model_route_revision"].startswith("route_sha256:")
    assert semantic["generation"]["task_id"].startswith(candidate_id_value)
    assert semantic["evidence_bindings"][0]["binding_id"] == "source:1"
    assert semantic["evidence_bindings"][0]["source_namespace"] == "snapshot.aiz.canonical_event"
    assert semantic["statements"][0]["evidence_refs"] == ["source:1"]
    revision = engine.memory_authority_store.get_memory_revision(candidate_id_value, 1)
    committed_bucket = asyncio.run(bucket_manager.get(candidate_id_value))
    assert committed_bucket["content"] == document["narrative"]["body"]
    assert revision["metadata"]["memory_l1"] == semantic
    assert revision["metadata"]["memory_l1_content_sha256"]
    assert not revision["metadata"].get("semantic_annotations")
    assert original_source in proposal["metadata"]["runtime_context"]["sources"][0]["text"]


@pytest.mark.parametrize('mode', ['review', 'off'])
def test_policy_not_bypassed(tmp_path, mode):
    tool, projection = service(tmp_path, mode)
    result = asyncio.run(tool.submit(envelope()))
    assert result['memory_status'] == ('pending' if mode == 'review' else 'rejected')
    assert result['memory_id'] is None and not projection.revisions


def test_review_existing_entry_confirms_without_truncating(tmp_path):
    tool, projection = service(tmp_path, 'review')
    content = '我保留完整经历，不把它变成一条短事实。' * 120
    async def run():
        result = await tool.submit(envelope(content=content))
        assert tool.engine._load_daily_chat_memory_pending()[0]['candidate']['content'] == content
        confirmed = await tool.engine.confirm_daily_chat_memory([result['candidate_id']], tool.bucket_mgr, request_id='owner-confirm')
        assert confirmed['created'] == 1
        assert receipt(tool.store, result['candidate_id'])['memory_status'] == 'committed'
        assert tool.store.get_candidate(result['candidate_id'])['proposal']['proposed_body'] == content
    asyncio.run(run())


def test_owner_edit_and_unknown_time(tmp_path):
    tool, projection = service(tmp_path, 'review')
    async def run():
        result = await tool.submit(envelope(event_time={'expression': '2099年1月1日', 'evidence': 'not in source'}))
        row = tool.store.get_candidate(result['candidate_id'])
        assert row['proposal']['metadata']['narrative']['event_time']['precision'] == 'unknown'
        await tool.confirm(row, edit={'content': '主人修改后的正文'}, request_id='edit')
        assert receipt(tool.store, result['candidate_id'])['body_preserved'] is False
        assert tool.store.get_candidate(result['candidate_id'])['proposal']['proposed_body'] == '主人修改后的正文'
    asyncio.run(run())


def test_review_l1_candidate_owner_edit_uses_revision_service_before_commit(tmp_path):
    tool, _fake_projection = service(tmp_path, 'review')
    from bucket_manager import BucketManager
    from memory_commit_service import BucketMemoryProjection, MemoryCommitService
    from memory_l1_schema import join_document, statement_template
    config = tool.engine.config
    bucket_manager = BucketManager(config)
    tool.bucket_mgr = bucket_manager
    tool.engine._memory_authority_service = lambda manager: MemoryCommitService(
        tool.store, BucketMemoryProjection(config, bucket_manager=manager))

    document = {
        'entry': {'form': 'episode'},
        'narrative': {'title': '聊天中的长期回忆', 'body': '初始完整正文，保持原文。'},
        'entities': [], 'scenes': [], 'statements': [],
        'retrieval': {'keywords': ['聊天'], 'tags': []}, 'relations': [], 'attributes': [],
    }
    statement = statement_template()
    statement.pop('statement_revision_id')
    statement['statement_id'] = 'new:statement'
    statement['content'] = '主人希望保留这段谈话。'
    statement['evidence_refs'] = ['source:current']
    document['statements'] = [statement]

    async def locate(_binding): return {'state': 'available', 'eligible_source': True}
    from types import SimpleNamespace
    tool.engine._memory_l1_sources = lambda: SimpleNamespace(locate=locate)

    async def run():
        request = envelope()
        request['arguments'] = {'document': document, 'kind': 'key_event'}
        submitted = await tool.submit(request)
        assert submitted['memory_status'] == 'pending'
        row = tool.store.get_candidate(submitted['candidate_id'])
        owner_document = join_document(row['proposal']['proposed_body'], row['proposal']['metadata']['memory_l1'])
        owner_document['narrative']['body'] = '主人审核后的正文已完整保留。'
        owner_document['retrieval']['tags'] = ['owner-reviewed']
        confirmed = await tool.confirm(row, edit={'document': owner_document}, request_id='owner-l1-confirm')
        final = tool.store.get_candidate(submitted['candidate_id'])
        semantic = final['proposal']['metadata']['memory_l1']
        assert confirmed['status'] == 'created' and final['status'] == 'committed'
        assert final['proposal']['proposed_body'] == '主人审核后的正文已完整保留。'
        assert semantic['retrieval']['tags'] == ['owner-reviewed']
        assert semantic['generation']['origin'] == 'owner_edit'
        assert receipt(tool.store, submitted['candidate_id'])['l1_state'] == 'current'
        bucket = await bucket_manager.get(submitted['candidate_id'])
        assert bucket['content'] == '主人审核后的正文已完整保留。'

    asyncio.run(run())


@pytest.mark.parametrize('field', ['owner_explicit', 'source_status', 'narrative', 'source_refs'])
def test_model_cannot_forge_authority(tmp_path, field):
    tool, projection = service(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(tool.submit(envelope(**{field: True})))
    assert not projection.revisions


def test_sensitive_and_identity_go_to_existing_review(tmp_path):
    tool, projection = service(tmp_path)
    assert asyncio.run(tool.submit(envelope(kind='identity')))['memory_status'] == 'pending'
    assert not projection.revisions


def test_review_display_source_preview_is_read_only_and_revisioned(tmp_path):
    tool, projection = service(tmp_path, 'review')
    async def run():
        result = await tool.submit(envelope())
        before = tool.store.get_candidate(result['candidate_id'])
        display = tool.engine.list_daily_chat_memory_pending()[0]['display']
        assert display['confirm_blocked'] is False and display['has_source_preview'] is True
        preview = await tool.engine.daily_chat_memory_source_preview(result['candidate_id'])
        assert preview['source_authority'] == 'aizizhu_canonical_snapshot'
        assert preview['events'][0]['version_id'] == 'v1'
        assert tool.store.get_candidate(result['candidate_id']) == before
    asyncio.run(run())


def test_crash_after_memory_commit_reconciles_same_revision_only_on_owner_action(tmp_path, monkeypatch):
    tool, projection = service(tmp_path)
    real = tool.store.decide_candidate
    def crash(*args, **kwargs):
        if kwargs.get('action') == 'commit':
            raise RuntimeError('simulated receipt crash')
        return real(*args, **kwargs)
    monkeypatch.setattr(tool.store, 'decide_candidate', crash)
    async def run():
        with pytest.raises(RuntimeError):
            await tool.submit(envelope())
        result = await tool.submit(envelope())
        assert result['memory_status'] == 'accepted'
        assert len(projection.revisions) == 1
        monkeypatch.setattr(tool.store, 'decide_candidate', real)
        await tool.confirm(tool.store.get_candidate(result['candidate_id']), edit=None, request_id='recover')
        assert receipt(tool.store, result['candidate_id'])['memory_status'] == 'committed'
        assert len(projection.revisions) == 1
    asyncio.run(run())


def test_server_auth_read_status_and_validation(tmp_path, monkeypatch):
    import json
    import server
    from starlette.requests import Request
    tool, projection = service(tmp_path, 'review')
    monkeypatch.setattr(server, 'reflection_engine', tool.engine)
    monkeypatch.setattr(server, 'memory_authority_store', tool.store)
    monkeypatch.setattr(server, 'bucket_mgr', tool.bucket_mgr)
    monkeypatch.setattr(server, '_memory_write_token', lambda: 'test-token')
    def request(method='POST', body=None, authorized=True):
        async def receive():
            return {'type': 'http.request', 'body': json.dumps(body or envelope()).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': method, 'path': '/',
                        'path_params': {'operation_id': envelope()['operation_id']},
                        'headers': [(b'authorization', b'Bearer test-token')] if authorized else []}, receive)
    async def run():
        assert (await server.api_chat_memory_tool(request(authorized=False))).status_code == 401
        assert (await server.api_chat_memory_tool(request(body={'owner_explicit': True}))).status_code == 400
        result = await server.api_chat_memory_tool(request())
        assert result.status_code == 200 and result.headers['Cache-Control'] == 'no-store'
        response = await server.api_chat_memory_tool(request('GET'))
        assert json.loads(response.body)['memory_status'] == 'pending'
        assert b'"proposed_body"' not in response.body
        assert envelope()['arguments']['content'].encode() not in response.body
        assert not projection.revisions
    asyncio.run(run())


def test_proven_not_applied_failure_can_be_explicitly_retried(tmp_path):
    tool, projection = service(tmp_path)
    projection.fail_revision = True
    async def run():
        result = await tool.submit(envelope())
        assert result['memory_status'] == 'commit_failed'
        assert not projection.revisions
        projection.fail_revision = False
        await tool.confirm(tool.store.get_candidate(result['candidate_id']), edit=None, request_id='owner-retry')
        assert receipt(tool.store, result['candidate_id'])['memory_status'] == 'committed'
        assert len(projection.revisions) == 1
    asyncio.run(run())
