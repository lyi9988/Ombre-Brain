"""Offline controller gates and postimage checks; never contact a provider."""
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_authority import MemoryAuthorityStore
from scripts import repair_recall_r1_embeddings as repair


def run(coro):
    return asyncio.run(coro)


def inventory(**changes):
    state = dict(mode="inventory", stop_reason="inventory_only",
                 path_identity_ok=True, authority_file_present=True,
                 embedding_file_present=True, prompt_mirror_file_present=True,
                 embedding_schema_ready=True, active_revision_sha_mismatch=0,
                 live_body_hash_mismatches=0, upgrade_source_units=1041)
    state.update(changes)
    return state


def config():
    return {"memory_authority": {"enabled": True},
            "embedding": {"enabled": True, "api_key": "SENTINEL_PRIVATE_KEY"}}


@pytest.mark.parametrize("mode,execute,approved,expected", [
    ("inventory", False, False, "inventory_only"),
    ("pilot", False, False, "execute_flag_required"),
    ("full", False, True, "execute_flag_required"),
    ("full", True, False, "pilot_approval_required"),
])
def test_cli_never_applies_without_explicit_authority(monkeypatch, capsys, mode, execute, approved, expected):
    monkeypatch.setattr(repair, "load_config", config)
    monkeypatch.setattr(repair, "_paths", lambda unused: {})
    async def fake_inventory(*args):
        return inventory()
    monkeypatch.setattr(repair, "_inventory", fake_inventory)
    monkeypatch.setattr(repair, "_apply_controller", lambda *args, **kwargs: pytest.fail("apply forbidden"))
    code = run(repair._async_main(SimpleNamespace(mode=mode, execute=execute,
                        pilot_verified=approved, max_total_units=repair.MAX_TOTAL_UNITS)))
    assert code == (0 if mode == "inventory" else 2)
    assert json.loads(capsys.readouterr().out)["stop_reason"] == expected


def test_pilot_corpus_1041_still_uses_one_memory_budget(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(repair, "load_config", config)
    monkeypatch.setattr(repair, "_paths", lambda unused: {})
    async def fake_inventory(*args):
        return inventory()
    async def fake_apply(*args, **kwargs):
        calls.append(kwargs)
        return {"mode": "pilot", "stop_reason": "pilot_complete"}
    monkeypatch.setattr(repair, "_inventory", fake_inventory)
    monkeypatch.setattr(repair, "_apply_controller", fake_apply)
    assert run(repair._async_main(SimpleNamespace(mode="pilot", execute=True,
               pilot_verified=False, max_total_units=repair.MAX_TOTAL_UNITS))) == 0
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "pilot_complete"
    assert len(calls) == 1
    assert calls[0]["total_unit_cap"] == repair.MAX_UNITS_PER_MEMORY == 7


@pytest.mark.parametrize("change,expected", [
    ({"path_identity_ok": False}, "unsafe_path_identity"),
    ({"authority_file_present": False}, "missing_store"),
    ({"prompt_mirror_file_present": False}, "missing_store"),
    ({"embedding_schema_ready": False}, "embedding_schema_not_ready"),
    ({"active_revision_sha_mismatch": 1}, "hash_mismatch"),
    ({"live_body_hash_mismatches": 1}, "hash_mismatch"),
])
def test_preflight_refuses_before_constructing_provider(monkeypatch, change, expected):
    monkeypatch.setattr(repair, "_prepare_apply_components", lambda *args: pytest.fail("provider constructed"))
    assert run(repair._apply_controller(config(), {}, inventory(**change),
               mode="pilot", total_unit_cap=7))["stop_reason"] == expected


def test_full_requires_corpus_cap_before_provider(monkeypatch):
    monkeypatch.setattr(repair, "_prepare_apply_components", lambda *args: pytest.fail("provider constructed"))
    assert run(repair._apply_controller(config(), {}, inventory(), mode="full",
               total_unit_cap=1040))["stop_reason"] == "unit_cap_exceeded"
    class PassedCap(Exception):
        pass
    def passed(*args):
        raise PassedCap()
    monkeypatch.setattr(repair, "_prepare_apply_components", passed)
    with pytest.raises(PassedCap):
        run(repair._apply_controller(config(), {}, inventory(), mode="full", total_unit_cap=1041))


def test_prompt_mirror_is_readonly_and_does_not_initialize_schema(tmp_path):
    path = tmp_path / "mirror.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE prompts (value TEXT)")
        conn.execute("INSERT INTO prompts VALUES ('fixed')")
    mirror = repair.ReadOnlyPromptPlanMirror(path)
    with mirror._open() as conn:
        assert conn.execute("SELECT value FROM prompts").fetchone()[0] == "fixed"
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE prompts SET value='changed'")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE surprise(value TEXT)")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM prompts").fetchone()[0] == "fixed"


def test_candidate_order_matches_worker_authority_order(tmp_path):
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    path = Path(authority.path)
    with sqlite3.connect(path) as conn:
        for ident, stamp, policy in (
            ("a", "2026-09-30T12:00:00Z", "enabled"),
            ("b", "2026-09-30T12:00:00Z", "enabled"),
            ("old", "2026-09-29T12:00:00Z", "enabled"),
            ("excluded", "2026-10-01T12:00:00Z", "disabled"),
        ):
            conn.execute("INSERT INTO memories(memory_id,bucket_id,active_revision,state,recall_policy,updated_at) "
                         "VALUES (?,?,1,'active',?,?)", (ident, ident, policy, stamp))
            conn.execute("INSERT INTO memory_projection_status"
                         "(memory_id,memory_revision,projector,status,details_json,updated_at) "
                         "VALUES (?,1,'embedding','pending_rebuild','{}',?)", (ident, stamp))
    worker_order = [m["memory_id"] for m in authority.list_memories(state="active")
                    if m["recall_policy"] == "enabled"]
    with repair._readonly(path) as conn:
        controller_order = [m["memory_id"] for m in repair._active_enabled_rows(conn)]
    assert controller_order == worker_order == ["a", "b", "old"]
    assert repair._next_candidate(path)["memory_id"] == worker_order[0]


def test_next_candidate_skips_cooling_memory_and_returns_none_when_all_cool(tmp_path, monkeypatch):
    now_ms = 2_000_000_000_000
    monkeypatch.setattr(repair.time, "time", lambda: now_ms / 1000)
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    path = Path(authority.path)
    with sqlite3.connect(path) as conn:
        for ident, retry_after in (("a", now_ms + 60_000), ("b", 0)):
            conn.execute("INSERT INTO memories(memory_id,bucket_id,active_revision,state,recall_policy,updated_at) "
                         "VALUES (?,?,1,'active','enabled','2026-09-30T12:00:00Z')", (ident, ident))
            conn.execute("INSERT INTO memory_projection_status"
                         "(memory_id,memory_revision,projector,status,details_json,updated_at) "
                         "VALUES (?,1,'embedding','pending_rebuild',?,'2026-09-30T12:00:00Z')",
                         (ident, json.dumps({"retry_after_ms": retry_after})))
    assert repair._next_candidate(path)["memory_id"] == "b"
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE memory_projection_status SET details_json=? WHERE memory_id='b'",
                     (json.dumps({"retry_after_ms": now_ms + 60_000}),))
    assert repair._next_candidate(path) is None
    assert repair._upgrade_candidate_count(path) == 2


class FakeEmbedding:
    call_invocations = 0
    call_elapsed_ms = []
    closed = False
    def runtime_debug(self):
        return {"connection_fallback_count": 0}
    async def close(self):
        self.closed = True


@pytest.mark.parametrize("worker_result,expected", [
    ({"embedding_units_budget_remaining": 7, "deferred": 1}, "budget_blocked"),
    ({"embedding_units_budget_remaining": 6, "deferred": 1}, "provider_failure"),
    ({"embedding_units_budget_remaining": 7}, "no_progress"),
    ({"embedding_units_budget_remaining": 0, "projected": 1,
      "embedding_units_completed": 1}, "metadata_failure"),
])
def test_failure_stops_without_query_or_next_memory(monkeypatch, capsys, worker_result, expected):
    calls = []
    embedding = FakeEmbedding()
    if expected == "provider_failure":
        embedding.runtime_debug = lambda: {
            "last_status": "error", "last_error_category": "http_status",
            "last_error_type": "HTTPStatusError", "last_http_status": 503,
            "last_latency_ms": 91,
            "api_key": "SENTINEL_PRIVATE_KEY", "provider_response": "PRIVATE_BODY",
            "connection_fallback_count": 0,
        }
    class Worker:
        async def repair_pending_once(self, **kwargs):
            calls.append(kwargs)
            return worker_result
    candidate = {"memory_id": "m", "revision": 1, "bucket_id": "b", "body_sha256": "sha"}
    seen = []
    monkeypatch.setattr(repair, "_prepare_apply_components", lambda *args: (Worker(), embedding, object()))
    monkeypatch.setattr(repair, "_resolve_prompts", lambda *args: ("instruction", "query"))
    monkeypatch.setattr(repair, "_next_candidate", lambda *args: seen.append(1) or candidate)
    monkeypatch.setattr(repair, "_projection_status", lambda *args: {
        "status": "pending_rebuild", "details": {}, "updated_at": "before"})
    monkeypatch.setattr(repair, "_active_revision", lambda *args: {
        "state": "active", "recall_policy": "enabled", "active_revision": 1, "body_sha256": "sha"})
    monkeypatch.setattr(repair, "_verify_projection_rows", lambda **kwargs: pytest.fail("bad postimage"))
    monkeypatch.setattr(repair, "_verify_new_query", lambda **kwargs: pytest.fail("query forbidden"))
    async def fake_inventory(*args):
        return inventory()
    monkeypatch.setattr(repair, "_inventory", fake_inventory)
    result = run(repair._apply_controller(config(), {"authority": Path("unused")},
                                          inventory(), mode="pilot", total_unit_cap=7))
    assert result["stop_reason"] == expected
    assert seen == [1] and len(calls) == 1
    assert calls[0] == {"limit": 1, "upgrade_legacy_indexes": True,
                        "max_embedding_units": 7, "projectors": ("embedding",)}
    assert embedding.closed
    assert all(json.loads(line)["event"] == "progress" for line in capsys.readouterr().out.splitlines())
    assert result["provider_diagnostics"] == repair._provider_diagnostics(embedding)
    if expected == "provider_failure":
        assert result["provider_diagnostics"] == {
            "last_status": "error", "last_error_category": "http_status",
            "last_error_type": "HTTPStatusError", "last_http_status": 503,
            "last_latency_ms": 91,
        }
        assert "SENTINEL_PRIVATE_KEY" not in json.dumps(result)
        assert "PRIVATE_BODY" not in json.dumps(result)


def test_pilot_completes_one_memory_while_corpus_has_1041_units(monkeypatch, capsys):
    candidate = {"memory_id": "m", "revision": 1, "bucket_id": "b", "body_sha256": "sha"}
    selected = []
    embedding = FakeEmbedding()
    class Worker:
        async def repair_pending_once(self, **kwargs):
            assert kwargs["max_embedding_units"] == 7
            assert kwargs["projectors"] == ("embedding",)
            return {"projected": 1, "embedding_units_completed": 1,
                    "embedding_units_budget_remaining": 6}
    monkeypatch.setattr(repair, "_prepare_apply_components", lambda *args: (Worker(), embedding, object()))
    monkeypatch.setattr(repair, "_resolve_prompts", lambda *args: ("instruction", "query"))
    monkeypatch.setattr(repair, "_next_candidate", lambda *args: selected.append(1) or candidate)
    statuses = iter([
        {"status": "pending_rebuild", "details": {}, "updated_at": "before"},
        {"status": "projected", "details": {"metadata_complete": True}, "updated_at": "after"},
    ])
    monkeypatch.setattr(repair, "_projection_status", lambda *args: next(statuses))
    monkeypatch.setattr(repair, "_active_revision", lambda *args: {
        "state": "active", "recall_policy": "enabled", "active_revision": 1, "body_sha256": "sha"})
    async def verify_rows(**kwargs):
        assert kwargs["candidate"] == candidate
        return {"ok": True, "rows_checked": 1, "source_unit_ids": ["unit-1"]}
    async def verify_query(**kwargs):
        assert kwargs["candidate"] == candidate
        return {"ok": True, "requests": 1, "input_chars": 4,
                "input_sha256": "h" * 64, "cache_status": "miss"}
    async def final_inventory(*args):
        return inventory(upgrade_source_units=1040)
    monkeypatch.setattr(repair, "_verify_projection_rows", verify_rows)
    monkeypatch.setattr(repair, "_verify_new_query", verify_query)
    monkeypatch.setattr(repair, "_inventory", final_inventory)
    result = run(repair._apply_controller(config(), {"authority": Path("unused")},
                                          inventory(), mode="pilot", total_unit_cap=7))
    assert result["stop_reason"] == "pilot_complete"
    assert result["memories_checked"] == result["embedding_units_completed"] == 1
    assert result["verified_vector_rows"] == result["new_query_verified"] == 1
    assert result["embedding_units_remaining"] == 1040
    assert result["provider_diagnostics"] == {
        "last_status": "not_requested", "last_error_category": "",
        "last_error_type": "", "last_http_status": None,
        "last_latency_ms": None,
    }
    assert selected == [1] and embedding.closed
    assert "SENTINEL_PRIVATE_KEY" not in capsys.readouterr().out


def test_progress_has_only_numeric_allowlisted_counters(capsys):
    repair._progress(mode="pilot", started=repair.time.monotonic(), checked=2,
                     completed=3, verified=1, degraded=0, embedding=FakeEmbedding())
    output = capsys.readouterr().out
    row = json.loads(output)
    assert set(row) == {"event", "mode", "memories_checked", "embedding_units_completed",
                        "verified_vector_rows", "degraded_memories", "provider_call_invocations",
                        "connection_fallback_count", "elapsed_ms"}
    assert row["memories_checked"] == 2
    assert "SENTINEL_PRIVATE_KEY" not in output and "memory_body" not in output


@pytest.mark.parametrize("corrupt", [None, "input_sha256", "preparation_sha256",
                                  "provider", "source_unit_id", "embedding",
                                  "nonfinite", "infinite", "boolean", "dimension"])
def test_selected_candidate_vector_metadata_is_verified(tmp_path, monkeypatch, corrupt):
    body = "SENTINEL_PRIVATE_BODY"
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    unit = {"source_unit_id": "unit-1", "text": "SENTINEL_PRIVATE_UNIT"}
    snapshot = {"base_url": "https://fake.test/v1", "model": "fake-model",
                "max_chars": 500, "document_preparation": "p" * 64}
    values = {"embedding": "[0.1,0.2]", "model": "fake-model", "dimension": 2,
              "parent_bucket_id": "bucket-1", "parent_memory_id": "memory-1",
              "source_unit_id": "unit-1", "source_revision": 1,
              "source_sha256": body_sha,
              "input_sha256": hashlib.sha256(unit["text"].encode()).hexdigest(),
              "preparation_sha256": "p" * 64, "provider": "https://fake.test/v1",
              "input_truncated_chars": 0}
    if corrupt:
        field = "embedding" if corrupt in {"nonfinite", "infinite", "boolean"} else corrupt
        values[field] = ("[NaN,0.2]" if corrupt == "nonfinite" else
                         "[Infinity,0.2]" if corrupt == "infinite" else
                         "[true,0.2]" if corrupt == "boolean" else
                         "not-json" if corrupt == "embedding" else
            3 if corrupt == "dimension" else "wrong")
    path = tmp_path / "embeddings.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE embeddings (" + ",".join(f"{key} TEXT" for key in values) + ")")
        conn.execute("INSERT INTO embeddings (" + ",".join(values) + ") VALUES (" +
                     ",".join("?" for _ in values) + ")", list(values.values()))
    permanent = tmp_path / "permanent"
    permanent.mkdir()
    class Buckets:
        permanent_dir = str(permanent)
        dynamic_dir = str(tmp_path / "dynamic")
        async def get(self, bucket_id):
            return {"path": str(permanent / "memory.md"), "content": body}
    class Embedding:
        def _query_config_snapshot(self, **kwargs):
            return snapshot
        def _prepare_document_snapshot(self, text, snapshot):
            return text
    candidate = {"memory_id": "memory-1", "bucket_id": "bucket-1",
                 "revision": 1, "body_sha256": body_sha}
    monkeypatch.setattr(repair, "_active_revision", lambda *args: {
        "state": "active", "recall_policy": "enabled", "active_revision": 1,
        "body_sha256": body_sha})
    monkeypatch.setattr(repair, "_projection_status", lambda *args: {
        "status": "projected", "source_sha256": body_sha,
        "details": {"metadata_complete": True, "source_revision": 1,
                    "source_sha256": body_sha, "source_memory_id": "memory-1",
                    "parent_bucket_id": "bucket-1", "dimension": 2,
                    "source_unit_count": 1, "model": "fake-model", "provider": "fake.test",
                    "preparation_sha256": "p" * 64}})
    monkeypatch.setattr(repair, "_resolve_prompts", lambda *args: ("document", "query"))
    worker = SimpleNamespace(bucket_manager=Buckets(), _embedding_source_units=lambda *args: [unit])
    result = run(repair._verify_projection_rows(
        config={}, paths={"authority": path, "embeddings": path}, worker=worker,
        embedding=Embedding(), mirror=object(), candidate=candidate))
    assert result["ok"] is (corrupt is None)
    assert result["rows_checked"] == (1 if corrupt is None else 0)


def test_query_smoke_requires_verified_candidate_and_successful_transport():
    class Embedding:
        def __init__(self):
            self.results = [{"bucket_id": "bucket-1", "index_status": "verified"}]
            self.status = "ok"
            self.http = 200
            self.calls = []
        def _query_config_snapshot(self, **kwargs):
            return {"max_chars": 500}
        def _prepare_query_snapshot(self, text, snapshot):
            return text
        async def search_similar_queries(self, queries, **kwargs):
            self.calls.append(kwargs)
            kwargs["cache_debug"].update(provider_requests=1, status="miss")
            return self.results
        def runtime_debug(self):
            return {"last_status": self.status, "last_http_status": self.http}
    embedding = Embedding()
    candidate = {"memory_id": "memory-1", "bucket_id": "bucket-1",
                 "revision": 1, "body_sha256": "h" * 64}
    projection = {"ok": True, "source_unit_ids": ["unit-1"],
                  "query_instruction": "query", "document_instruction": "document"}
    assert run(repair._verify_new_query(embedding=embedding, candidate=candidate,
                                        projection=projection))["ok"]
    assert embedding.calls[-1]["index_metadata"]["bucket-1"] == {
        "memory_id": "memory-1", "revision": 1, "body_sha256": "h" * 64}
    embedding.results = [{"bucket_id": "bucket-1", "index_status": "legacy"}]
    assert not run(repair._verify_new_query(embedding=embedding, candidate=candidate,
                                            projection=projection))["ok"]
    embedding.results = [{"bucket_id": "bucket-1", "index_status": "verified"}]
    embedding.http = 503
    assert not run(repair._verify_new_query(embedding=embedding, candidate=candidate,
                                            projection=projection))["ok"]


def test_measured_engine_counts_method_invocations_even_on_exception(monkeypatch):
    # Replace the superclass call; no real client, socket, key, or DB is created.
    calls = []
    async def fake_request(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic provider failure")
        return [0.1, 0.2]
    monkeypatch.setattr(repair.EmbeddingEngine, "_request_embedding", fake_request)
    measured = object.__new__(repair.MeasuredEmbeddingEngine)
    measured.call_invocations = 0
    measured.call_elapsed_ms = []
    assert run(measured._request_embedding("fake-input")) == [0.1, 0.2]
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        run(measured._request_embedding("fake-input"))
    assert measured.call_invocations == len(calls) == 2
    assert len(measured.call_elapsed_ms) == 2
    assert all(isinstance(value, int) and value >= 0 for value in measured.call_elapsed_ms)


def test_provider_diagnostics_allowlists_failure_values_without_arbitrary_runtime_text():
    secret = "SENTINEL_PRIVATE_KEY_AND_BODY"
    class Embedding:
        def runtime_debug(self):
            return {
                "last_status": "error", "last_error_category": "http_status",
                "last_error_type": f"HTTPStatusError-{secret}",
                "last_http_status": 503, "last_latency_ms": 91,
                "base_url": f"https://{secret}.test", "api_key": secret,
                "last_http_timing": {"response_body": secret},
                "provider_response": secret,
            }
    diagnostics = repair._provider_diagnostics(Embedding())
    assert diagnostics == {
        "last_status": "error", "last_error_category": "http_status",
        "last_error_type": "other", "last_http_status": 503,
        "last_latency_ms": 91,
    }
    assert secret not in json.dumps(diagnostics)


@pytest.mark.parametrize("runtime", [
    {"last_status": ["private"], "last_error_category": "private",
     "last_error_type": {"message": "private"},
     "last_http_status": "503 private", "last_latency_ms": -1},
    {"last_status": True, "last_error_category": None,
     "last_error_type": 123, "last_http_status": True,
     "last_latency_ms": 42.0},
    {"last_status": "unknown private", "last_error_category": "read private",
     "last_error_type": "Error private", "last_http_status": 600,
     "last_latency_ms": 86400001},
])
def test_provider_diagnostics_malformed_values_are_bounded(runtime):
    class Embedding:
        def runtime_debug(self):
            return runtime
    assert repair._provider_diagnostics(Embedding()) == {
        "last_status": "other", "last_error_category": "other",
        "last_error_type": "other", "last_http_status": None,
        "last_latency_ms": None,
    }


def test_provider_diagnostics_accepts_minimal_fake_and_unavailable_runtime():
    default = {"last_status": "not_requested", "last_error_category": "",
               "last_error_type": "", "last_http_status": None,
               "last_latency_ms": None}
    assert repair._provider_diagnostics(FakeEmbedding()) == default
    class NonMappingEmbedding:
        def runtime_debug(self):
            return "PRIVATE_BODY"
    class BrokenEmbedding:
        def runtime_debug(self):
            raise RuntimeError("PRIVATE_BODY")
    assert repair._provider_diagnostics(NonMappingEmbedding()) == default
    assert repair._provider_diagnostics(BrokenEmbedding()) == default
