"""Synthetic lifecycle regression for v2, using actual Authority and Bucket IO."""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore, MemoryIngestionPolicy, MemoryProposal, PolicyDecision
from memory_candidate_edit import save_owner_candidate
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_l0_sources import MemoryL0Sources
from memory_l1_contract import L1ContractError
from memory_l1_features import read_features
from memory_l1_ingestion import bind_sources
from memory_l1_reads import MemoryL1Reads
from memory_l1_schema import catalog_with_defaults, join_document, split_document
from memory_l1_service import MemoryL1RevisionService
from memory_l1_v2 import prepare_card, validate_model_card
from memory_l1_v2_diagnostics import MemoryV2Diagnostics
from memory_l1_v2_sources import AgentMemoryDraftReader, draft_source
from memory_retrieval_text import l1_retrieval_text, text_projection_version


def setup_v2(tmp_path):
    config = {"buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state"),
              "persona": {"profile_id": "test"}, "gateway": {"canonical_continuation": {
                  "enabled": True, "conversation_id": "c1"}}}
    authority = MemoryAuthorityStore(config)
    buckets = BucketManager(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, buckets))
    service = MemoryL1RevisionService(authority, commit, buckets, lambda: (catalog_with_defaults(), "old-catalog"))
    raw = {"system": "aizizhu", "namespace": "canonical_event", "profile_id": "test",
           "conversation_id": "c1", "event_id": "1", "version_id": "1", "role": "assistant",
           "recorded_at": "2026-11-13T10:00:00+08:00", "text": "我想先试两周，之后一起看看。"}
    snapshots, bindings = bind_sources([raw], config=config)
    person = snapshots[0]["participant"]
    people = [{"ref": "p1", "person_id": person["subject_id"], "name": person["display_name"], "kind": person["kind"]}]
    card = {"title": "两周试用的提议", "body": "2026年11月13日，我提出先试两周，之后一起看看，尚未得到回应。",
            "kind": "project_state", "stability": "phase", "keywords": ["两周试用"],
            "people": ["p1"], "sources": ["m1"], "primary_topic": "projects_creation",
            "event_time": {"start": "2026-11-13", "precision": "day", "sources": ["m1"]}}
    document = prepare_card(card, people=people, bindings=bindings, generation={"origin": "auto_l1", "task_id": "test-run"},
                            source_ref_map={"m1": bindings[0]["binding_id"]})
    return config, authority, buckets, service, document, snapshots


def test_v2_commit_source_index_edit_and_no_l2(tmp_path):
    _, authority, _, service, document, snapshots = setup_v2(tmp_path)
    body, semantic = split_document(document)
    result = asyncio.run(service.commit.commit_memory(memory_id="v2", bucket_id="v2", expected_revision=0,
        body=body, metadata={"name": document["title"], "memory_l1": semantic}, source_refs=["test-source"],
        decision_source="test", idempotency_key="commit-v2", actor="owner", l1_write_intent=True))
    assert result["revision"] == 1
    detail = asyncio.run(service.detail("v2"))
    assert detail["l1"]["document"] == document
    assert detail["l1"]["preparation"]["l2"]["state"] == "direct_l1"
    assert authority.pending_outbox()[-1]["payload"]["l2_preparation"]["state"] == "direct_l1"
    assert "body" not in authority.get_memory_revision("v2", 1)["metadata"]["memory_l1"]
    features = read_features(authority, "v2", 1)
    assert any(row["feature_path"] == "stability" and row["value"] == "phase" for row in features)
    assert all(row["object_type"] != "statement" for row in features)
    assert l1_retrieval_text(document) == document["title"] + "\n\n" + body
    assert text_projection_version(document) == "memory-l1-v2-text-v1"

    class Reader:
        namespace = snapshots[0]["source_namespace"]
        available = True

        async def read(self, binding):
            return snapshots[0] if self.available else {"state": "withdrawn"}

    reader = Reader()
    reads = MemoryL1Reads(service, MemoryL0Sources(authority, [reader]))
    request = [{"memory_id": "v2", "revision": 1, "binding_ids": semantic["source_refs"]}]
    available = asyncio.run(reads.recall_source_status(request))["items"][0]
    assert all(item["eligible_source"] for item in available["sources"].values())
    reader.available = False
    withdrawn = asyncio.run(reads.recall_source_status(request))["items"][0]
    assert not any(item["eligible_source"] for item in withdrawn["sources"].values())

    edited = deepcopy(document)
    edited["body"] += " 后来我决定等她回应。"
    saved = asyncio.run(service.save("v2", {"expected_revision": 1, "request_id": "owner-edit", "document": edited}))
    assert saved["revision"] == 2
    assert asyncio.run(service.detail("v2", 1))["body"] == body


def test_retained_v2_auto_candidate_returns_to_review_after_permission_removed(tmp_path):
    from reflection_engine import ReflectionEngine

    config, authority, buckets, _, document, _ = setup_v2(tmp_path)
    config['memory_authority'] = {'enabled': True}
    engine = ReflectionEngine(config)
    body, semantic = split_document(document)
    proposal = MemoryProposal.from_mapping({'proposal_id': 'waiting-auto', 'source_type': 'daily_chat',
        'proposed_body': body, 'source_status': 'verified', 'source_refs': ['test-source'],
        'memory_type': 'project_state', 'requested_mode': 'auto', 'confidence': .9,
        'metadata': {'memory_l1': semantic, 'confidence_source': 'model_review'}})
    old_policy = MemoryIngestionPolicy({'auto_types': ['project_state']})
    row = authority.put_candidate(proposal, old_policy.evaluate(proposal))
    assert row['status'] == 'accepted'
    engine.memory_ingestion_policy = MemoryIngestionPolicy({'auto_types': []})
    result = asyncio.run(engine._memory_l1_adopt_candidate(row, buckets, request_id='retry'))
    current = authority.get_candidate('waiting-auto')
    assert result['status'] == current['status'] == 'pending'
    assert current['revision'] == row['revision'] + 1
    assert current['policy']['requires_owner_confirmation'] is True
    assert authority.get_memory('waiting-auto') is None


def test_v2_retry_honors_saved_owner_acceptance_when_auto_is_disallowed(tmp_path):
    from reflection_engine import ReflectionEngine

    config, authority, buckets, _, document, _ = setup_v2(tmp_path)
    config['memory_authority'] = {'enabled': True}
    engine = ReflectionEngine(config)
    body, semantic = split_document(document)
    proposal = MemoryProposal.from_mapping({'proposal_id': 'owner-accepted', 'source_type': 'daily_chat',
        'proposed_body': body, 'source_status': 'verified', 'source_refs': ['test-source'],
        'memory_type': 'project_state', 'metadata': {'memory_l1': semantic}})
    row = authority.put_candidate(proposal, engine.memory_ingestion_policy.evaluate(proposal))
    row = authority.decide_candidate(row['candidate_id'], action='accept', expected_revision=row['revision'],
        request_id='owner-accept', actor='owner')

    class Reader:
        async def locate(self, binding):
            return {'state': 'available', 'eligible_source': True}

    engine._memory_l1_sources = lambda: Reader()
    result = asyncio.run(engine._memory_l1_adopt_candidate(row, buckets, request_id='recover'))
    assert result['status'] == 'created'
    commit = authority.get_commit_by_idempotency('l1-candidate:owner-accepted:revision:2')
    assert commit['payload']['decision_source'] == 'owner'


def test_v2_owner_candidate_edit_keeps_source_and_updates_type(tmp_path):
    _, authority, _, service, document, _ = setup_v2(tmp_path)
    body, semantic = split_document(document)
    proposal = MemoryProposal.from_mapping({"proposal_id": "candidate-v2", "source_type": "daily_chat",
        "proposed_body": body, "memory_type": "project_state", "requested_mode": "review", "confidence": 1.0,
        "source_status": "verified", "source_refs": ["source"], "metadata": {"memory_l1": semantic}})
    row = authority.put_candidate(proposal, PolicyDecision("queue_review", ["OWNER_REVIEW"], "review"))
    edited = deepcopy(document)
    edited["kind"] = "life_experience"
    edited["keywords"].append("试学")
    saved = save_owner_candidate(authority, row["candidate_id"], edit={"document": edited},
        expected_revision=row["revision"], request_id="edit-candidate", revision_service=service)
    assert saved["proposal"]["memory_type"] == "life_experience"
    restored = join_document(saved["proposal"]["proposed_body"], saved["proposal"]["metadata"]["memory_l1"])
    assert restored["keywords"] == edited["keywords"]
    assert restored["evidence_bindings"] == document["evidence_bindings"]


@pytest.mark.parametrize(('kind', 'evolution'), [('correction', False), ('understanding', True)])
def test_v2_followup_review_commits_same_card_without_an_l2_or_false_evolution(tmp_path, kind, evolution):
    from b1_protocol import digest
    from memory_authority import MemoryIngestionPolicy
    from memory_l3_followup import MemoryFollowups, run_followup_tasks
    from memory_l3_records import MemoryL3Records
    from persona_memory_reaction_context import memory_ref
    from test_stage03_l3_followup import payload

    config, authority, buckets, service, document, snapshots = setup_v2(tmp_path)
    body, semantic = split_document(document)
    asyncio.run(service.commit.commit_memory(memory_id='m1', bucket_id='b1', expected_revision=0,
        body=body, metadata={'memory_l1': semantic, 'name': document['title']},
        source_refs=['original-source'], actor='owner', decision_source='review', idempotency_key='v2-original'))
    reader = MemoryL0Sources(authority)
    reader.retain_copy(snapshots[0])

    class Reader:
        namespace = snapshots[0]['source_namespace']

        async def read(self, binding):
            return reader.copy(binding)

    reader.adapters[Reader.namespace] = Reader()
    new_body = '2026年11月13日，我提出先试两周，后来听到她说想等下个月再决定。'
    calls = []

    async def complete(client, **kwargs):
        calls.append(kwargs)
        material = json.loads(kwargs['messages'][1]['content'])
        evidence = material['source_evidence'][0]['binding']['binding_id']
        assert 'statements' not in material['material']['l1_document']
        return SimpleNamespace(text=json.dumps({'decision': 'revise_l1', 'reason': '新对话澄清了回复。',
            'patch': [{'op': 'replace_text', 'path': '/body', 'old': '之后一起看看，尚未得到回应。',
                'value': '后来听到她说想等下个月再决定。', 'evidence_refs': [evidence]}]}), finish='stop')

    engine = SimpleNamespace(config=config, memory_authority_enabled=True, memory_authority_store=authority,
        memory_ingestion_policy=MemoryIngestionPolicy(), daily_chat_memory_mode='review',
        memory_semantic_profile_id='test', daily_chat_memory_timeout_seconds=180, temperature=.1,
        _memory_l1_revisions=lambda _mgr: service, _memory_l1_sources=lambda: reader,
        _memory_l1_prompt=lambda source: (source, {'prompt_revision': 'test-prompt'}),
        _memory_task_route=lambda stage: {'stage': stage, 'configured': True, 'client': object(),
            'model': 'fixture', 'max_tokens': 1000, 'connection': 'reflection'},
        _daily_chat_memory_create_completion=complete, _completion_content=lambda response: response.text,
        _completion_finish_reason=lambda response: response.finish, _parse_json_object=json.loads)

    async def resolve(context, _turn):
        return {'state': 'available', 'eligible_fragment_ids': [f['fragment_id'] for f in context['fragments']]}

    value = payload(kind=kind, value='specific_future_use' if evolution else 'explicit_owner_correction')
    update = value['updates'][0]
    update['content'] = '后来她说想等下个月再决定。'
    message = '我想等下个月再决定。'
    value['turn']['user_event']['content'] = value['sources'][0]['text'] = message
    update['evidence_spans'][0].update(end=len(message), evidence_hash=digest(message))
    fragment = value['context']['fragments'][0]
    fragment.update(l1_revision=1, text=body, text_sha256=sha256(body.encode()).hexdigest())
    fragment['fragment_id'] = 'fragment_' + digest({k: v for k, v in fragment.items() if k != 'fragment_id'})[:32]
    context = value['context']
    context['delivery_sha256'] = digest({k: v for k, v in context.items() if k not in {'delivery_sha256', 'delivery_receipt_id'}})
    update['memory_ref'] = memory_ref(fragment)
    source = {'conversation_key': digest('c1'), 'user_id': 'u1', 'assistant_id': 'a1',
              'user': message, 'assistant': value['turn']['assistant_event']['content']}
    update['update_id'] = 'ml3:' + digest({'source_hash': digest(source), **{k: v for k, v in update.items() if k != 'update_id'}})

    followup = MemoryFollowups(engine, buckets, resolve)
    asyncio.run(followup.submit(value))
    report = asyncio.run(run_followup_tasks(engine, buckets, max_jobs=1))['results'][0]
    assert report['phase'] == 'prepared', report
    row = authority.get_candidate(report['candidate_ids'][0])
    assert row['status'] == 'pending' and authority.get_memory('m1')['active_revision'] == 1
    records = MemoryL3Records(authority, service.commit)
    assert not any(item['details'].get('display_scope') == 'evolution' for item in records.timeline('m1')['items'])
    accepted = asyncio.run(followup.adopt(row, owner=True, request_id='accept-revision'))
    assert accepted['receipt']['revision'] == 2 and len(calls) == 1
    result = asyncio.run(service.detail('m1'))
    assert result['body'] == new_body
    assert result['l1']['document']['kind'] == document['kind']
    assert len(result['l1']['document']['source_refs']) == 2
    assert asyncio.run(service.detail('m1', 1))['body'] == body
    assert authority.get_memory(row['candidate_id']) is None
    assert all(event['payload']['l2_preparation']['state'] == 'direct_l1' for event in authority.pending_outbox())
    timeline = records.timeline('m1')['items']
    assert all(item['details']['display_scope'] == 'revision_history' for item in timeline if item['kind'] == 'l1_revision')
    assert sum(item['details'].get('display_scope') == 'evolution' for item in timeline) == int(evolution)


def test_v2_requires_keywords_and_preserves_unknown_people(tmp_path):
    _, _, _, _, document, _ = setup_v2(tmp_path)
    card = {key: value for key, value in document.items() if key in {
        "title", "body", "kind", "stability", "keywords", "primary_topic"}}
    card.update(people=["p1"], sources=["m1"], people_links=[
        {"label": "林工", "how": "mentioned", "sources": ["m1"]}])
    validated = validate_model_card(card, source_refs=["m1"], people_refs=["p1"])
    assert validated["people"] == ["p1"]
    card["keywords"] = []
    with pytest.raises(L1ContractError, match="minItems"):
        validate_model_card(card, source_refs=["m1"], people_refs=["p1"])


def test_agent_draft_is_own_source_and_revision_change_invalidates(tmp_path):
    config, authority, _, _, _, _ = setup_v2(tmp_path)
    task = {"task_id": "draft-test", "draft_revision": 1, "phase": "prepared",
            "created_at": "2026-11-13T10:00:00+08:00", "envelope": {
                "context": {"conversation_id": "c1", "sources": []},
                "arguments": {"narrative": {"body": "我当时觉得她也许累了。"}}}}
    authority.set_meta("memory_l1_chat_task:draft-test", task)
    raw = draft_source(task, config)
    snapshots, bindings = bind_sources([raw], config=config)
    sources = MemoryL0Sources(authority, [AgentMemoryDraftReader(config, authority)])
    assert asyncio.run(sources.locate(bindings[0], verify_only=True))["eligible_source"]
    task["draft_revision"] = 2
    task["envelope"]["arguments"]["narrative"]["body"] = "后来我发现自己理解错了。"
    authority.set_meta("memory_l1_chat_task:draft-test", task)
    assert not asyncio.run(sources.locate(bindings[0], verify_only=True))["eligible_source"]


def test_diagnostics_expiry_does_not_remove_memory(tmp_path):
    _, authority, _, _, _, _ = setup_v2(tmp_path)
    recorder = MemoryV2Diagnostics(authority, max_bytes=65536)
    recorder.record("old", "source", {"count": 10, "body": "must not be logged"})
    conn = authority._connect()
    conn.execute("UPDATE memory_v2_diagnostics SET created_at=?",
                 ((datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),))
    conn.commit()
    conn.close()
    recorder.record("new", "model", {"usage": {"input_tokens": 100}, "raw_response": "private"})
    rows = recorder.list()
    assert len(rows) == 1 and rows[0]["run_id"] == "new"
    assert "raw_response" not in rows[0]["data"]
