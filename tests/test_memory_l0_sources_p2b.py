import asyncio
import copy

import pytest

from memory_authority import MemoryAuthorityStore
from memory_l0_sources import DIARY_NAMESPACE, DIARY_VIEW, McpDiarySourceAdapter, MemoryL0Sources
from memory_l1_contract import L1ContractError


def setup_sources(tmp_path, reader_state):
    config = {"buckets_dir": str(tmp_path / "buckets"), "state_dir": str(tmp_path / "state")}

    async def fake_mcp_read(day):
        """In-memory MCP stand-in; this test never invokes a provider or production service."""
        if reader_state.get("raises"):
            raise RuntimeError("synthetic MCP unavailable")
        return copy.deepcopy(reader_state["record"])

    authority = MemoryAuthorityStore(config)
    adapter = McpDiarySourceAdapter("synthetic-mcp-provider", fake_mcp_read)
    return authority, adapter, MemoryL0Sources(authority, [adapter])


def capture(sources, day="2026-10-14", binding_id="bind-p2b"):
    return asyncio.run(sources.capture_diary(day, binding_id=binding_id))


def locate(sources, binding, **kwargs):
    return asyncio.run(sources.locate(binding, **kwargs))


def test_diary_capture_keeps_full_unicode_text_and_unknown_identity_fields(tmp_path):
    text = "前😀e\u0301后" + ("日记正文段落。" * 8192)
    reader_state = {"record": {"id": "synthetic-diary-1", "content": text}}
    _, adapter, sources = setup_sources(tmp_path, reader_state)

    captured = capture(sources)
    snapshot = captured["source"]
    binding = captured["binding"]
    assert snapshot["state"] == "available"
    assert snapshot["text"] == text
    assert len(snapshot["text"]) > 30_000
    assert snapshot["author"] is None
    assert snapshot["speaker"] is None
    assert snapshot["recorded_at"] is None
    assert snapshot["source_namespace"] == DIARY_NAMESPACE
    assert snapshot["source_namespace"] == "snapshot.ombre.diary_mcp"
    assert snapshot["source_namespace"] not in {"aiz.canonical", "aiz.private_diary", "aiz.shared_diary"}
    assert snapshot["text_view_id"] == DIARY_VIEW
    assert "Aiz" in snapshot["limitations"][0]
    assert binding["author_ref"] is None and binding["speaker_ref"] is None
    anchor = binding["anchors"][0]
    assert anchor["start"] == 0 and anchor["end"] == len(text)
    assert anchor["coordinate_unit"] == "unicode_codepoint"
    assert anchor["text_sha256"]

    # Python code-point slicing treats the supplementary emoji as one code point.
    exact = locate(sources, binding, start=1, end=4)
    assert exact["eligible_source"] is True
    assert exact["text"] == "😀e\u0301"
    assert exact["range"] == {"start": 1, "end": 4, "coordinate_unit": "unicode_codepoint"}


@pytest.mark.parametrize("mutation,expected_state", [
    (lambda record: record.update(content="更新后的原文"), "changed"),
    (lambda record: record.update(revision_id="upstream-revision-2"), "changed"),
    (lambda record: record.update(revoked=True), "withdrawn"),
    (lambda record: record.update(status="deleted"), "withdrawn"),
])
def test_changed_or_withdrawn_sources_are_not_eligible_and_copy_is_labelled(tmp_path, mutation, expected_state):
    original_text = "原始合成日记原文。"
    reader_state = {"record": {"id": "synthetic-diary-2", "content": original_text, "revision_id": "r1"}}
    _, _, sources = setup_sources(tmp_path, reader_state)
    binding = capture(sources)["binding"]

    mutation(reader_state["record"])
    result = locate(sources, binding)
    assert result["eligible_source"] is False
    assert result["view"] == "source_copy"
    assert result["state"] == expected_state
    assert result["text"] == original_text
    assert any("历史原文副本" in note and "不作当前资格依据" in note for note in result["limitations"])


def test_adapter_exception_returns_ineligible_labeled_copy(tmp_path):
    reader_state = {"record": {"id": "synthetic-diary-3", "content": "离线前的原文。"}}
    _, _, sources = setup_sources(tmp_path, reader_state)
    binding = capture(sources)["binding"]
    reader_state["raises"] = True

    result = locate(sources, binding)
    assert result["eligible_source"] is False
    assert result["state"] == "unavailable"
    assert result["view"] == "source_copy"
    assert result["text"] == "离线前的原文。"
    assert any("历史原文副本" in note and "不作当前资格依据" in note for note in result["limitations"])


@pytest.mark.parametrize("hash_field", ["text_sha256", "quote_sha256"])
def test_invalid_codepoint_ranges_and_anchor_hash_tampering_fail_closed(tmp_path, hash_field):
    reader_state = {"record": {"id": "synthetic-diary-4", "content": "甲😀乙丙"}}
    _, _, sources = setup_sources(tmp_path, reader_state)
    binding = capture(sources)["binding"]
    for start, end in ((-1, 2), (1, 1), (0, 99), (True, 2)):
        with pytest.raises(L1ContractError) as exc:
            locate(sources, binding, start=start, end=end)
        assert exc.value.code == "invalid_codepoint_range"

    forged = copy.deepcopy(binding)
    forged["anchors"][0][hash_field] = "0" * 64
    result = locate(sources, forged)
    assert result["eligible_source"] is False
    assert result["state"] == "mismatch"
    assert result["reason"] == "source_anchor_mismatch"
    assert result["view"] == "source_copy"
    assert result["text"] == "甲😀乙丙"


def test_new_source_revision_cannot_be_asserted_by_rewriting_binding_json(tmp_path):
    reader_state = {"record": {"id": "synthetic-diary-5", "content": "同一段合成原文。", "revision_id": "r1"}}
    _, adapter, sources = setup_sources(tmp_path, reader_state)
    binding = capture(sources)["binding"]
    original_revision = binding["source_revision"]

    reader_state["record"]["revision_id"] = "r2"
    current = asyncio.run(adapter.capture("2026-10-14"))
    assert current["source_revision"] != original_revision
    forged = copy.deepcopy(binding)
    # Claiming the newly observed source revision leaves the anchors bound to r1.
    forged["source_revision"] = current["source_revision"]
    result = locate(sources, forged)
    assert result["eligible_source"] is False
    assert result["state"] == "mismatch"
    assert result["reason"] == "source_anchor_mismatch"
    assert result["text"] is None


def test_changed_text_view_identity_is_rejected_even_when_source_revision_matches(tmp_path):
    reader_state = {"record": {"id": "synthetic-diary-6", "content": "原文视图。"}}
    _, _, sources = setup_sources(tmp_path, reader_state)
    binding = capture(sources)["binding"]
    forged = copy.deepcopy(binding)
    forged["anchors"][0]["text_view_id"] = "diary_mcp.content.guessed-v2"

    result = locate(sources, forged)
    assert result["eligible_source"] is False
    assert result["state"] == "mismatch"
    assert result["view"] == "source_copy"
    assert result["text"] == "原文视图。"
