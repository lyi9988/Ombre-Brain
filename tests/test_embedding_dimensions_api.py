"""Synthetic dashboard API coverage for embedding output dimensions."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
from types import SimpleNamespace

import pytest
import yaml
from starlette.responses import JSONResponse


class FakeRequest:
    def __init__(self, body=None):
        self._body = copy.deepcopy(body)

    async def json(self):
        return copy.deepcopy(self._body)


class RecordingEmbeddingEngine:
    def __init__(self, dimension):
        self.dimension = dimension
        self.base_url = "https://embedding.invalid/v1"
        self.reconfigured = []

    async def reconfigure(self, config):
        embedding = copy.deepcopy(config["embedding"])
        self.reconfigured.append(embedding)
        self.dimension = embedding.get("dimensions")


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Load the real handlers with all initialized storage rooted in tmp_path."""
    state_dir = tmp_path / "state"
    buckets_dir = tmp_path / "buckets"
    runtime_config_path = state_dir / "config.runtime.yaml"
    monkeypatch.setenv("OMBRE_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("OMBRE_BUCKETS_DIR", str(buckets_dir))
    monkeypatch.setenv("OMBRE_STATE_DIR", str(state_dir))
    monkeypatch.setenv("OMBRE_RUNTIME_CONFIG_PATH", str(runtime_config_path))

    server = importlib.import_module("server")
    config = server.load_config()
    config["state_dir"] = str(state_dir)
    config["_runtime_config_path"] = str(runtime_config_path)
    config["embedding"]["dimensions"] = 8
    monkeypatch.setattr(server, "config", config)

    engine = RecordingEmbeddingEngine(8)
    monkeypatch.setattr(server, "embedding_engine", engine)
    monkeypatch.setattr(server, "_require_dashboard_auth", lambda _request: None)

    async def no_gateway_update(_payload):
        return None

    monkeypatch.setattr(server, "_hot_update_gateway_config", no_gateway_update)
    return SimpleNamespace(
        server=server,
        config=config,
        engine=engine,
        runtime_config_path=runtime_config_path,
    )


@pytest.mark.parametrize(
    ("requested", "normalized"),
    [(17, 17), ("32", 32), (None, None)],
)
def test_dimensions_update_get_runtime_engine_and_runtime_yaml(api, requested, normalized):
    api.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    api.runtime_config_path.write_text(
        yaml.safe_dump({"embedding": {"model": "persisted-model"}}),
        encoding="utf-8",
    )

    updated = asyncio.run(
        api.server.api_config_update(
            FakeRequest({"embedding": {"dimensions": requested}, "persist": True})
        )
    )

    assert updated.status_code == 200
    assert "embedding.dimensions" in response_json(updated)["updated"]
    assert api.config["embedding"]["dimensions"] == normalized
    assert api.engine.reconfigured == [{**api.config["embedding"]}]
    assert api.engine.dimension == normalized

    persisted = yaml.safe_load(api.runtime_config_path.read_text(encoding="utf-8"))
    assert persisted["embedding"]["dimensions"] == normalized
    assert persisted["embedding"]["model"] == "persisted-model"

    fetched = asyncio.run(api.server.api_config_get(FakeRequest()))
    assert fetched.status_code == 200
    embedding = response_json(fetched)["embedding"]
    assert embedding["dimensions"] == normalized
    assert embedding["effective_dimensions"] == normalized


@pytest.mark.parametrize(
    "body",
    [
        {
            "dehydration": {"model": "must-not-apply"},
            "embedding": {"dimensions": 0},
            "persist": True,
        },
        {
            "dehydration": {"model": "must-not-apply"},
            "embedding": None,
            "persist": True,
        },
        ["not", "a", "config-object"],
    ],
)
def test_invalid_dimensions_input_is_rejected_without_partial_mutation(api, body):
    original_config = copy.deepcopy(api.config)
    original_dehydration_model = api.server.dehydrator.model

    response = asyncio.run(api.server.api_config_update(FakeRequest(body)))

    assert response.status_code == 400
    assert api.config == original_config
    assert api.server.dehydrator.model == original_dehydration_model
    assert api.engine.reconfigured == []
    assert not api.runtime_config_path.exists()


def test_config_handlers_enforce_dashboard_auth_before_read_or_update(api, monkeypatch):
    original_config = copy.deepcopy(api.config)
    denied = JSONResponse({"error": "unauthorized"}, status_code=401)
    monkeypatch.setattr(api.server, "_require_dashboard_auth", lambda _request: denied)

    fetched = asyncio.run(api.server.api_config_get(FakeRequest()))
    updated = asyncio.run(
        api.server.api_config_update(
            FakeRequest({"embedding": {"dimensions": 64}, "persist": True})
        )
    )

    assert fetched.status_code == updated.status_code == 401
    assert api.config == original_config
    assert api.engine.reconfigured == []
    assert not api.runtime_config_path.exists()
