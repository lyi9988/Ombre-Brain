import asyncio
from copy import deepcopy
import pytest

from memory_authority import MemoryProposal, PolicyDecision
from memory_l1_schema import bind_l1_metadata, join_document
from memory_l1_service import L1OwnerError
from memory_summary_coverage import check_summary_coverage, related_summary_statements, summary_link_receipt
from test_owner_summary_memory import setup, run


def test_current_source_revision_and_all_facts_are_required_for_link(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch)
    cid = run(f)["results"][0]["candidate_ids"][0]
    authority = f.env["authority"]
    row = authority.get_candidate(cid)
    asyncio.run(f.engine._memory_l1_adopt_candidate(
        row, f.env["buckets"], request_id="adopt-first", owner=True))
    proposal = deepcopy(row["proposal"])
    proposal["proposal_id"] = "another-summary-candidate"
    semantic = proposal["metadata"]["memory_l1"]
    assert check_summary_coverage(authority, proposal)["status"] == "linked"
    assert related_summary_statements(authority, semantic["evidence_bindings"])[0]["memory_id"] == cid
    authority.put_candidate(MemoryProposal.from_mapping(proposal),
                            PolicyDecision("queue_review", ("OWNER_REVIEW",), "review"))
    linked = asyncio.run(f.engine._memory_l1_adopt_candidate(
        authority.get_candidate(proposal["proposal_id"]), f.env["buckets"], owner=True, request_id="adopt-second"))
    assert linked["status"] == "linked" and linked["linked_memories"][0]["memory_id"] == cid
    assert summary_link_receipt(authority, proposal["proposal_id"]) == linked["linked_memories"]
    assert authority.get_memory(proposal["proposal_id"]) is None
    proposal["proposed_body"] += " 后来有了一个尚未解决的新疑问。"
    assert check_summary_coverage(authority, proposal)["status"] == "overlap_review"

    # One message can support two different facts. A matching title or source
    # never proves the second condition, time or interpretation.
    original = semantic["statements"][0]
    extra = deepcopy(original)
    extra.update(statement_id="extra", content="还须等预约确认。")
    semantic["statements"].append(extra)
    mixed = check_summary_coverage(authority, proposal)
    assert mixed["status"] == "overlap_review" and mixed["unmatched_statement_ids"] == ["extra"]
    original["conditions"]["text"] = "只有收到书面确认"
    assert not check_summary_coverage(authority, proposal)["covered_statement_ids"]
    semantic["evidence_bindings"][0]["source_revision"] = "changed-source-revision"
    assert check_summary_coverage(authority, proposal)["status"] == "new"


def test_review_can_adopt_only_the_new_condition_from_an_overlapping_summary(tmp_path, monkeypatch):
    f = setup(tmp_path, monkeypatch, body="我在考虑周三去看展，要收到书面确认才会去。")
    cid = run(f)["results"][0]["candidate_ids"][0]
    store = f.env["authority"]
    first = store.get_candidate(cid)
    original = join_document(first["proposal"]["proposed_body"], first["proposal"]["metadata"]["memory_l1"])
    original["statements"][0]["content"] = original["narrative"]["body"] = "我在考虑周三去看展。"
    asyncio.run(f.engine._memory_l1_adopt_candidate(first, f.env["buckets"], owner=True,
        request_id="first", edit={"document": original}))
    draft = deepcopy(store.get_candidate(cid)["proposal"])
    draft["proposal_id"] = "summary-with-new-condition"
    semantic = draft["metadata"]["memory_l1"]
    old = semantic["statements"][0]
    extra = deepcopy(old)
    extra.update(statement_id="s-extra", content="赴约还有书面确认这一条件。")
    semantic["statements"].append(extra)
    draft["proposed_body"] += "\n" + extra["content"]
    draft["metadata"] = bind_l1_metadata(draft["proposed_body"], draft["metadata"])
    draft["metadata"]["owner_summary_coverage"] = check_summary_coverage(store, draft)
    row = store.put_candidate(MemoryProposal.from_mapping(draft), PolicyDecision("queue_review", (), "review"))
    with pytest.raises(L1OwnerError, match="summary_contains_already_adopted_statements"):
        asyncio.run(f.engine._memory_l1_adopt_candidate(row, f.env["buckets"], owner=True, request_id="mixed"))
    assert store.get_candidate(row["candidate_id"])["status"] == "pending"
    edit = join_document(draft["proposed_body"], semantic)
    edit["statements"] = [edit["statements"][1]]
    edit["narrative"]["body"] = extra["content"]
    asyncio.run(f.engine._memory_l1_adopt_candidate(
        row, f.env["buckets"], owner=True, request_id="only-the-addition", edit={"document": edit}))
    assert store.get_memory(cid)["active_revision"] == 1
    assert store.get_memory(row["candidate_id"])["active_revision"] == 1
