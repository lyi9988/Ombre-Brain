"""Focused coverage for stage-specific memory write model routes.

All providers are synthetic AsyncOpenAI-compatible clients. These tests record
the completion request without making a network call or touching production data.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
from types import SimpleNamespace

import pytest

from memory_task_routes import DEFAULT_ROUTE, STAGES, normalize_task_routes


class RecordingClient:
    def __init__(self, *, api_key, base_url, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[])


def make_engine(tmp_path, monkeypatch, *, routes=None, include_daily=True,
                include_reflection=True, include_dehydration=True):
    import reflection_engine

    for key in (
        "OMBRE_REFLECTION_API_KEY",
        "OMBRE_EMBEDDING_API_KEY",
        "OMBRE_PERSONA_API_KEY",
        "OMBRE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        reflection_engine,
        "AsyncOpenAI",
        lambda **kwargs: RecordingClient(**kwargs),
    )
    reflection = {
        "enabled": True,
        "memory_task_routes": copy.deepcopy(routes or {}),
        "daily_chat_memory_candidate_max_tokens": 6400,
        "daily_chat_memory_summary_max_tokens": 2400,
    }
    if include_reflection:
        reflection.update({
            "api_key": "synthetic-reflection-secret",
            "base_url": "https://reflection.invalid/v1",
            "model": "reflection-default",
            "thinking_mode": "enabled",
        })
    if include_daily:
        reflection.update({
            "daily_chat_memory_api_key": "synthetic-daily-secret",
            "daily_chat_memory_base_url": "https://daily.invalid/v1",
            "daily_chat_memory_candidate_model": "daily-candidate-default",
            "daily_chat_memory_summary_model": "daily-summary-default",
            "daily_chat_memory_thinking_mode": "enabled",
        })
    dehydration = {}
    if include_dehydration:
        dehydration = {
            "api_key": "synthetic-dehydration-secret",
            "base_url": "https://dehydration.invalid/v1",
            "model": "dehydration-default",
            "thinking_mode": "enabled",
        }
    config = {
        "state_dir": str(tmp_path / "state"),
        "buckets_dir": str(tmp_path / "buckets"),
        "reflection": reflection,
        "dehydration": dehydration,
    }
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "buckets").mkdir(parents=True, exist_ok=True)
    from reflection_engine import ReflectionEngine

    return ReflectionEngine(config)


def test_route_normalization_returns_all_stages_and_validates_values():
    routes = normalize_task_routes({"auto_l1": {"model": "  route-model  "}})

    assert set(routes) == set(STAGES)
    assert routes["auto_l1"] == {**DEFAULT_ROUTE, "model": "route-model",
                                  "thinking_mode": "enabled", "reasoning_effort": "low", "max_tokens": 32000}
    for stage in ("chat_l1", "diary_l1", "action_l1"):
        assert routes[stage]["max_tokens"] == 32000
        assert routes[stage]["reasoning_effort"] == "low"
    assert routes["presentation_l2"] == DEFAULT_ROUTE
    assert all(routes[stage] == DEFAULT_ROUTE for stage in ("presentation_l2", "chat_summary", "activity_summary"))

    invalid = [
        [],
        {"unexpected_stage": {}},
        {"auto_l1": {"connection": "unknown"}},
        {"auto_l1": {"thinking_mode": "sometimes"}},
        {"auto_l1": {"reasoning_effort": "xhigh"}},
        {"auto_l1": {"max_tokens": True}},
        {"auto_l1": {"max_tokens": 0}},
        {"auto_l1": {"max_tokens": 1_000_001}},
        {"auto_l1": {"api_key": "must-not-be-a-route-field"}},
        {"presentation_l2": {"api_key": "must-not-be-a-route-field"}},
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            normalize_task_routes(value)


def test_stage_completion_calls_use_the_selected_model_thinking_effort_and_token_limit(
    tmp_path, monkeypatch
):
    routes = {
        "auto_l1": {
            "connection": "reflection",
            "model": "memory-auto-v7",
            "thinking_mode": "enabled",
            "reasoning_effort": "high",
            "max_tokens": 701,
        },
        "chat_l1": {
            "connection": "dehydration",
            "model": "memory-chat-v4",
            "thinking_mode": "disabled",
            "reasoning_effort": "low",
            "max_tokens": 1302,
        },
        "chat_summary": {
            "connection": "daily",
            "model": "memory-summary-v2",
            "thinking_mode": "default",
            "reasoning_effort": "medium",
            "max_tokens": 2403,
        },
        "presentation_l2": {
            "connection": "daily",
            "model": "memory-presentation-v1",
            "thinking_mode": "disabled",
            "reasoning_effort": "high",
            "max_tokens": 12345,
        },
    }
    engine = make_engine(tmp_path, monkeypatch, routes=routes)
    messages = [{"role": "user", "content": "synthetic request"}]

    for stage in ("auto_l1", "chat_l1", "chat_summary", "presentation_l2"):
        asyncio.run(engine._daily_chat_memory_create_completion(
            None,
            model="legacy-caller-model",
            messages=messages,
            max_tokens=55,
            temperature=0.17,
            use_daily_client=False,
            task_stage=stage,
        ))

    auto_call = engine.client.calls[0]
    chat_call = engine.dehydration_client.calls[0]
    summary_call = engine.daily_chat_memory_client.calls[0]
    assert auto_call["model"] == "memory-auto-v7"
    assert auto_call["max_tokens"] == 701
    assert auto_call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert auto_call["reasoning_effort"] == "high"
    assert auto_call["temperature"] == 0.17

    assert chat_call["model"] == "memory-chat-v4"
    assert chat_call["max_tokens"] == 1302
    assert chat_call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert chat_call["reasoning_effort"] == "low"

    assert summary_call["model"] == "memory-summary-v2"
    assert summary_call["max_tokens"] == 2403
    assert "extra_body" not in summary_call
    assert summary_call["reasoning_effort"] == "medium"

    presentation_call = engine.daily_chat_memory_client.calls[1]
    assert presentation_call["model"] == "memory-presentation-v1"
    assert presentation_call["max_tokens"] == 12345
    assert presentation_call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert presentation_call["reasoning_effort"] == "high"


def test_presentation_l2_legacy_route_uses_existing_model_and_full_draft_default(
    tmp_path, monkeypatch
):
    engine = make_engine(tmp_path, monkeypatch)

    route = engine._memory_task_route("presentation_l2")

    assert route["stage"] == "presentation_l2"
    assert route["connection"] == "daily"
    assert route["model"] == "daily-candidate-default"
    assert route["thinking_mode"] == "enabled"
    assert route["max_tokens"] == 16000
    assert route["max_tokens_source"] == "task_default"


def test_explicit_unconfigured_connection_stays_unavailable_without_cross_source_fallback(
    tmp_path, monkeypatch
):
    engine = make_engine(
        tmp_path,
        monkeypatch,
        routes={"auto_l1": {"connection": "reflection"}},
        include_reflection=False,
        include_daily=True,
        include_dehydration=True,
    )

    route = engine._memory_task_route("auto_l1")

    assert route["connection"] == "reflection"
    assert route["configured"] is False
    assert route["unavailable_reason"] == "connection_unconfigured"
    assert route["client"] is None
    # The regular ReflectionEngine constructor can preserve its old shared
    # client fallback; an explicit stage route must still report the source
    # selected by its owner configuration as unavailable.
    assert engine.daily_chat_memory_client is not None
    assert engine.dehydration_client is not None
    with pytest.raises(RuntimeError, match="connection_unconfigured"):
        asyncio.run(engine._daily_chat_memory_create_completion(
            None,
            model="legacy-caller-model",
            messages=[{"role": "user", "content": "synthetic request"}],
            max_tokens=55,
            temperature=0.17,
            use_daily_client=False,
            task_stage="auto_l1",
        ))
    assert engine.daily_chat_memory_client.calls == []
    assert engine.dehydration_client.calls == []


def test_legacy_routes_preserve_existing_provider_selection(tmp_path, monkeypatch):
    with_daily = make_engine(tmp_path / "with-daily", monkeypatch)
    auto = with_daily._memory_task_route("auto_l1")
    diary = with_daily._memory_task_route("diary_l1")
    assert auto["connection"] == "daily"
    assert auto["model"] == "daily-candidate-default"
    assert diary["connection"] == "reflection"
    assert diary["model"] == "reflection-default"

    without_daily = make_engine(
        tmp_path / "without-daily", monkeypatch, include_daily=False
    )
    auto_fallback = without_daily._memory_task_route("auto_l1")
    assert auto_fallback["connection"] == "dehydration"
    assert auto_fallback["model"] == "dehydration-default"


def _nested_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def test_public_effective_routes_exclude_clients_and_provider_credentials(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)

    public = engine.memory_task_models_public()
    encoded = json.dumps(public, ensure_ascii=False).lower()
    keys = set(_nested_keys(public))

    assert {"routes", "effective"} <= set(public)
    assert set(public["routes"]) == set(STAGES)
    assert set(public["effective"]) == set(STAGES)
    assert not ({"client", "api_key", "base_url", "credential", "credentials"} & keys)
    for secret in (
        "synthetic-reflection-secret",
        "synthetic-daily-secret",
        "synthetic-dehydration-secret",
        "https://reflection.invalid/v1",
        "https://daily.invalid/v1",
        "https://dehydration.invalid/v1",
    ):
        assert secret not in encoded


class FakeRequest:
    def __init__(self, body=None):
        self.body = copy.deepcopy(body)

    async def json(self):
        return copy.deepcopy(self.body)


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture
def config_api(monkeypatch, tmp_path):
    """Use the real handlers with runtime config and storage rooted in tmp_path."""
    state_dir = tmp_path / "state"
    runtime_config_path = state_dir / "config.runtime.yaml"
    monkeypatch.setenv("OMBRE_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(tmp_path / "buckets"))
    monkeypatch.setenv("OMBRE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("OMBRE_RUNTIME_CONFIG_PATH", str(runtime_config_path))
    for key in (
        "OMBRE_REFLECTION_API_KEY",
        "OMBRE_EMBEDDING_API_KEY",
        "OMBRE_PERSONA_API_KEY",
        "OMBRE_API_KEY",
        "OMBRE_RERANKER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    server = importlib.import_module("server")
    config = server.load_config()
    config["state_dir"] = str(state_dir)
    config["buckets_dir"] = str(tmp_path / "buckets")
    config["_runtime_config_path"] = str(runtime_config_path)
    config["reflection"] = {
        **(config.get("reflection") or {}),
        "api_key": "synthetic-reflection-secret",
        "base_url": "https://reflection.invalid/v1",
        "model": "reflection-default",
        "daily_chat_memory_api_key": "synthetic-daily-secret",
        "daily_chat_memory_base_url": "https://daily.invalid/v1",
        "daily_chat_memory_candidate_model": "daily-candidate-default",
        "daily_chat_memory_summary_model": "daily-summary-default",
    }
    config["dehydration"] = {
        **(config.get("dehydration") or {}),
        "api_key": "synthetic-dehydration-secret",
        "base_url": "https://dehydration.invalid/v1",
        "model": "dehydration-default",
    }
    monkeypatch.setattr(server, "config", config)
    monkeypatch.setattr(server, "reflection_engine", server.reflection_engine)
    monkeypatch.setattr(server, "memory_authority_store", server.memory_authority_store)
    monkeypatch.setattr(server, "memory_commit_service", server.memory_commit_service)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda _request: None)
    monkeypatch.setattr(
        server,
        "embedding_engine",
        SimpleNamespace(base_url="", dimension=None, reconfigure=lambda _config: None),
    )
    return SimpleNamespace(server=server, config=config, runtime_config_path=runtime_config_path)


def test_real_config_api_persists_and_reads_safe_stage_routes(config_api, monkeypatch):
    import reflection_engine

    monkeypatch.setattr(
        reflection_engine,
        "AsyncOpenAI",
        lambda **kwargs: RecordingClient(**kwargs),
    )
    routes = {
        "auto_l1": {
            "connection": "reflection",
            "model": "api-auto-v3",
            "thinking_mode": "enabled",
            "reasoning_effort": "high",
            "max_tokens": 555,
        },
        "diary_l1": {
            "connection": "daily",
            "model": "api-diary-v1",
            "thinking_mode": "default",
            "max_tokens": 777,
        },
    }

    saved = asyncio.run(config_api.server.api_config_update(
        FakeRequest({"reflection": {"memory_task_routes": routes}, "persist": True})
    ))
    assert saved.status_code == 200, saved.body

    fetched = asyncio.run(config_api.server.api_config_get(FakeRequest()))
    assert fetched.status_code == 200, fetched.body
    payload = response_json(fetched)
    reflection = payload["reflection"]
    assert reflection["memory_task_routes"] == normalize_task_routes(routes)
    assert set(reflection["memory_task_effective"]) == set(STAGES)

    persisted = config_api.runtime_config_path.read_text(encoding="utf-8")
    assert "memory_task_routes" in persisted
    assert "api-auto-v3" in persisted

    public_section = {
        "routes": reflection["memory_task_routes"],
        "effective": reflection["memory_task_effective"],
    }
    public_text = json.dumps(public_section, ensure_ascii=False).lower()
    public_keys = set(_nested_keys(public_section))
    assert not ({"client", "api_key", "base_url", "credential", "credentials"} & public_keys)
    for secret in (
        "synthetic-reflection-secret",
        "synthetic-daily-secret",
        "synthetic-dehydration-secret",
        "https://reflection.invalid/v1",
        "https://daily.invalid/v1",
        "https://dehydration.invalid/v1",
    ):
        assert secret not in public_text

    before_config = copy.deepcopy(config_api.config)
    before_overlay = config_api.runtime_config_path.read_bytes()
    invalid = asyncio.run(config_api.server.api_config_update(
        FakeRequest({
            "reflection": {"memory_task_routes": {"auto_l1": {"api_key": "no"}}},
            "persist": True,
        })
    ))
    assert invalid.status_code == 400
    assert config_api.config == before_config
    assert config_api.runtime_config_path.read_bytes() == before_overlay


def install_temporary_config_runtime(config_api, monkeypatch, tmp_path):
    """Build the currently active runtime from the same temporary config."""
    import reflection_engine
    from bucket_manager import BucketManager
    from memory_commit_service import BucketMemoryProjection, MemoryCommitService
    from memory_task_routes import normalize_task_routes

    clients = []

    def make_client(**kwargs):
        client = RecordingClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(reflection_engine, "AsyncOpenAI", make_client)
    config = config_api.config
    config["state_dir"] = str(tmp_path / "state")
    config["buckets_dir"] = str(tmp_path / "buckets")
    config["_runtime_config_path"] = str(config_api.runtime_config_path)
    config["gateway"] = {
        **(config.get("gateway") or {}),
        "prompt_plan_mirror_path": str(tmp_path / "prompt-plan.sqlite3"),
    }
    config["memory_authority"] = {
        "enabled": True,
        "db_path": str(tmp_path / "state" / "memory-authority.sqlite3"),
    }
    config["reflection"]["enabled"] = True
    config["reflection"]["memory_task_routes"] = normalize_task_routes({
        "auto_l1": {"connection": "daily", "model": "old-owner-route-model"},
    })

    from reflection_engine import ReflectionEngine

    engine = ReflectionEngine(config)
    buckets = BucketManager(config)
    authority = engine.memory_authority_store
    commit = MemoryCommitService(
        authority,
        BucketMemoryProjection(config, bucket_manager=buckets),
    )
    monkeypatch.setattr(config_api.server, "reflection_engine", engine)
    monkeypatch.setattr(config_api.server, "memory_authority_store", authority)
    monkeypatch.setattr(config_api.server, "memory_commit_service", commit)
    monkeypatch.setattr(config_api.server, "bucket_mgr", buckets)
    monkeypatch.setattr(config_api.server, "prompt_plan_mirror", None)
    assert engine._memory_task_route("auto_l1")["model"] == "old-owner-route-model"
    return engine, authority, commit, clients


def test_memory_task_route_persist_failure_restores_old_runtime_then_retry_saves(
    config_api, monkeypatch, tmp_path
):
    import memory_system_config
    from memory_task_routes import normalize_task_routes, task_routes_revision

    old_engine, old_store, old_commit, clients = install_temporary_config_runtime(
        config_api, monkeypatch, tmp_path
    )
    old_config = copy.deepcopy(config_api.config)
    previous_routes = copy.deepcopy(config_api.config["reflection"]["memory_task_routes"])
    expected_revision = task_routes_revision(previous_routes)
    new_routes = normalize_task_routes({
        "auto_l1": {"connection": "daily", "model": "new-owner-route-model"},
    })
    request_body = {
        "reflection": {"memory_task_routes": new_routes},
        "persist": True,
        "expected_memory_task_revision": expected_revision,
    }
    original_update_overlay = memory_system_config.update_overlay

    def fail_persist(*_args, **_kwargs):
        raise OSError("synthetic overlay write failure")

    monkeypatch.setattr(memory_system_config, "update_overlay", fail_persist)
    failed = asyncio.run(config_api.server.api_config_update(FakeRequest(request_body)))

    assert failed.status_code == 500
    assert config_api.config == old_config
    assert config_api.config["reflection"]["memory_task_routes"] == previous_routes
    assert config_api.server.reflection_engine is old_engine
    assert config_api.server.memory_authority_store is old_store
    assert config_api.server.memory_commit_service is old_commit
    assert config_api.server.reflection_engine._memory_task_route("auto_l1")["model"] == "old-owner-route-model"
    assert not config_api.runtime_config_path.exists()
    assert all(client.calls == [] for client in clients)


    monkeypatch.setattr(memory_system_config, "update_overlay", original_update_overlay)
    saved = asyncio.run(config_api.server.api_config_update(FakeRequest(request_body)))

    assert saved.status_code == 200, saved.body
    fetched = asyncio.run(config_api.server.api_config_get(FakeRequest()))
    assert fetched.status_code == 200, fetched.body
    reflection = response_json(fetched)["reflection"]
    assert reflection["memory_task_routes"] == new_routes
    assert reflection["memory_task_effective"]["auto_l1"]["model"] == "new-owner-route-model"
    assert "new-owner-route-model" in config_api.runtime_config_path.read_text(encoding="utf-8")
    assert all(client.calls == [] for client in clients)


def test_l2_schema_config_persists_reads_back_and_rejects_stale_or_invalid(config_api):
    initial = response_json(asyncio.run(config_api.server.api_config_get(FakeRequest())))['reflection']
    schema = copy.deepcopy(initial['memory_l2_output_schema'])
    schema['properties']['full_injection_text']['minLength'] = 2
    saved = asyncio.run(config_api.server.api_config_update(FakeRequest({'reflection':{'memory_l2_output_schema':schema},
        'expected_memory_l2_schema_revision':initial['memory_l2_schema_revision'], 'persist':True})))
    assert saved.status_code == 200, saved.body
    readback = response_json(asyncio.run(config_api.server.api_config_get(FakeRequest())))['reflection']
    assert readback['memory_l2_output_schema'] == schema
    assert readback['memory_l2_schema_revision'] != initial['memory_l2_schema_revision']
    assert 'memory_l2_output_schema' in config_api.runtime_config_path.read_text(encoding='utf-8')
    before = copy.deepcopy(config_api.config)
    stale = asyncio.run(config_api.server.api_config_update(FakeRequest({'reflection':{'memory_l2_output_schema':{}},
        'expected_memory_l2_schema_revision':initial['memory_l2_schema_revision']})))
    assert stale.status_code == 409 and config_api.config == before
    invalid = asyncio.run(config_api.server.api_config_update(FakeRequest({'reflection':{'memory_l2_output_schema':{'type':'not-a-json-schema-type'}},
        'expected_memory_l2_schema_revision':readback['memory_l2_schema_revision']})))
    assert invalid.status_code == 400 and config_api.config == before


def test_l2_scheduler_uses_same_server_cas_as_task_routes(config_api):
    initial = response_json(asyncio.run(config_api.server.api_config_get(FakeRequest())))['reflection']
    first = asyncio.run(config_api.server.api_config_update(FakeRequest({'reflection':{'memory_l2_jobs_per_run':3},
        'expected_memory_task_revision':initial['memory_task_revision'], 'persist':True})))
    assert first.status_code == 200, first.body
    readback = response_json(asyncio.run(config_api.server.api_config_get(FakeRequest())))['reflection']
    assert readback['memory_l2_jobs_per_run'] == 3
    assert readback['memory_task_revision'] != initial['memory_task_revision']
    second = asyncio.run(config_api.server.api_config_update(FakeRequest({'reflection':{'memory_l2_jobs_per_run':7},
        'expected_memory_task_revision':initial['memory_task_revision'], 'persist':True})))
    assert second.status_code == 409 and config_api.config['reflection']['memory_l2_jobs_per_run'] == 3

def test_memory_task_route_expected_revision_conflict_has_no_partial_update(
    config_api, monkeypatch, tmp_path
):
    from memory_task_routes import normalize_task_routes, task_routes_revision

    old_engine, old_store, old_commit, clients = install_temporary_config_runtime(
        config_api, monkeypatch, tmp_path
    )
    old_config = copy.deepcopy(config_api.config)
    old_routes = copy.deepcopy(config_api.config["reflection"]["memory_task_routes"])
    new_routes = normalize_task_routes({
        "auto_l1": {"connection": "daily", "model": "must-not-apply"},
    })
    stale_revision = "mt-" + "0" * 64
    assert stale_revision != task_routes_revision(old_routes)

    response = asyncio.run(config_api.server.api_config_update(FakeRequest({
        "reflection": {"memory_task_routes": new_routes},
        "persist": True,
        "expected_memory_task_revision": stale_revision,
    })))

    assert response.status_code == 409
    assert response_json(response)["error"] == "memory_task_revision_conflict"
    assert config_api.config == old_config
    assert config_api.config["reflection"]["memory_task_routes"] == old_routes
    assert config_api.server.reflection_engine is old_engine
    assert config_api.server.memory_authority_store is old_store
    assert config_api.server.memory_commit_service is old_commit
    assert config_api.server.reflection_engine._memory_task_route("auto_l1")["model"] == "old-owner-route-model"
    assert not config_api.runtime_config_path.exists()
    assert all(client.calls == [] for client in clients)
