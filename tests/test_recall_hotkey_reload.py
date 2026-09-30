"""Offline mounted credential reload regressions."""

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import gateway
import pytest
from gateway import GatewayService
from reranker_engine import RerankerEngine


def _service(tmp_path, monkeypatch, *, mounted_key="mounted-initial"):
    # Redirect only the mounted .env lookup; never touch the real file.
    monkeypatch.delenv("OMBRE_ENV_PATH", raising=False)
    monkeypatch.setattr(gateway, "__file__", str(tmp_path / "gateway.py"))
    env = tmp_path / ".env"
    env.write_text(f"OMBRE_RERANKER_API_KEY={mounted_key}\n", encoding="utf-8")
    overlay = tmp_path / "config.runtime.yaml"
    overlay.write_text("reranker:\n  model: model-initial\n", encoding="utf-8")
    service = GatewayService.__new__(GatewayService)
    service.config = {
        "gateway": {}, "embedding": {},
        "reranker": {
            "enabled": True, "model": "model-startup",
            "base_url": "https://reranker.invalid/v1", "api_key": "startup-stale",
        },
    }
    service.gateway_cfg = service.config["gateway"]
    service.embedding_engine = SimpleNamespace(
        model="embed", base_url="https://embedding.invalid/v1",
        api_key="embedding-test", enabled=True,
    )
    service.reranker_engine = RerankerEngine(service.config)
    service._runtime_overlay_path = str(overlay)
    service._runtime_overlay_signature = None
    service._runtime_overlay_status = {}
    return service, env, overlay


def test_initial_mounted_key_supersedes_stale_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OMBRE_RERANKER_API_KEY", "startup-stale")
    service, _env, _overlay = _service(tmp_path, monkeypatch)
    service._maybe_reload_runtime_overlay(force=True)
    assert service.reranker_engine.model == "model-initial"
    assert service.reranker_engine.api_key == "mounted-initial"
    assert service.reranker_engine.enabled is True
    assert service._runtime_overlay_status["credential_matches_observed_mount"]["OMBRE_RERANKER_API_KEY"] is True
    assert "mounted-initial" not in str(service._runtime_overlay_status)
    assert "startup-stale" not in str(service._runtime_overlay_status)


def test_direct_override_survives_unrelated_env_and_model_until_named_key_changes(
    tmp_path, monkeypatch,
):
    service, env, overlay = _service(tmp_path, monkeypatch)
    service._maybe_reload_runtime_overlay(force=True)
    service._apply_reranker_config({"api_key": "owner-direct"})
    revision = service._rerank_config_revision
    assert service._runtime_overlay_status["credential_matches_observed_mount"]["OMBRE_RERANKER_API_KEY"] is False

    env.write_text("OMBRE_RERANKER_API_KEY=mounted-initial\nUNRELATED=longer\n", encoding="utf-8")
    service._maybe_reload_runtime_overlay()
    assert service.reranker_engine.api_key == "owner-direct"
    assert service._rerank_config_revision == revision

    overlay.write_text("reranker:\n  model: model-new\n", encoding="utf-8")
    service._maybe_reload_runtime_overlay()
    assert service.reranker_engine.model == "model-new"
    assert service.reranker_engine.api_key == "owner-direct"

    env.write_text("OMBRE_RERANKER_API_KEY=mounted-rotated-longer\nUNRELATED=longer\n", encoding="utf-8")
    service._maybe_reload_runtime_overlay()
    assert service.reranker_engine.api_key == "mounted-rotated-longer"
    assert service._rerank_config_revision > revision
    assert service._runtime_overlay_status["credential_matches_observed_mount"]["OMBRE_RERANKER_API_KEY"] is True
    assert "mounted-rotated-longer" not in str(service._runtime_overlay_status)
    assert "owner-direct" not in str(service._runtime_overlay_status)


@pytest.mark.parametrize("replacement", [
    "OMBRE_RERANKER_API_KEY=\n# credential removed\n",
    "UNRELATED=still-present\n",
])
def test_removed_mounted_credential_does_not_resurrect_stale_config(tmp_path, monkeypatch, replacement):
    service, env, _overlay = _service(tmp_path, monkeypatch)
    service._maybe_reload_runtime_overlay(force=True)
    service.config["embedding"]["api_key"] = "embedding-legacy-credential"
    env.write_text(replacement, encoding="utf-8")
    service._maybe_reload_runtime_overlay()
    assert service.reranker_engine.api_key == ""
    assert service.reranker_engine.enabled is False
    service._apply_reranker_config({"model": "model-after-clear"})
    assert service.reranker_engine.model == "model-after-clear"
    assert service.reranker_engine.api_key == ""
    assert service.reranker_engine.enabled is False


def test_initial_explicit_blank_key_overrides_stale_config(tmp_path, monkeypatch):
    service, _env, _overlay = _service(tmp_path, monkeypatch, mounted_key="")
    service._maybe_reload_runtime_overlay(force=True)
    assert service.reranker_engine.api_key == ""
    assert service.reranker_engine.enabled is False


def test_unreadable_env_preserves_last_authorized_credential(tmp_path, monkeypatch):
    service, _env, _overlay = _service(tmp_path, monkeypatch)
    service._maybe_reload_runtime_overlay(force=True)
    service._apply_reranker_config({"api_key": "owner-direct"})
    monkeypatch.setattr(service, "_runtime_env_credentials", lambda: None)
    service._apply_runtime_overlay({"reranker": {"model": "model-new"}}, env_changed=True)
    assert service.reranker_engine.api_key == "owner-direct"


def test_missing_mounted_name_preserves_legacy_config(tmp_path, monkeypatch):
    service, env, _overlay = _service(tmp_path, monkeypatch)
    env.write_text("UNRELATED_KEY=unrelated\n", encoding="utf-8")
    service._maybe_reload_runtime_overlay(force=True)
    assert service.reranker_engine.api_key == "startup-stale"
    assert service.reranker_engine.enabled is True


@pytest.mark.parametrize(("line", "expected"), [
    (r'OMBRE_RERANKER_API_KEY="json\nvalue" # comment', "json\nvalue"),
    ("export OMBRE_RERANKER_API_KEY='single value' # comment", "single value"),
    ("OMBRE_RERANKER_API_KEY=bare-value  # comment", "bare-value"),
    ("OMBRE_RERANKER_API_KEY=${UNSAFE_TEST_ENV} # comment", "${UNSAFE_TEST_ENV}"),
])
def test_custom_env_path_parser_formats_and_no_interpolation(tmp_path, monkeypatch, line, expected):
    monkeypatch.setenv("UNSAFE_TEST_ENV", "must-not-interpolate")
    custom = tmp_path / "mounted-credentials.env"
    custom.write_text(line + "\nOTHER_SECRET=ignored\n", encoding="utf-8")
    monkeypatch.setenv("OMBRE_ENV_PATH", str(custom))
    assert GatewayService._runtime_env_path() == custom
    assert GatewayService._runtime_env_credentials() == {"OMBRE_RERANKER_API_KEY": expected}


@pytest.mark.parametrize("line", [
    'OMBRE_RERANKER_API_KEY="unterminated',
    "OMBRE_RERANKER_API_KEY='unterminated",
])
def test_malformed_quoted_named_key_returns_unavailable(tmp_path, monkeypatch, line):
    custom = tmp_path / "mounted-credentials.env"
    custom.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setenv("OMBRE_ENV_PATH", str(custom))
    assert GatewayService._runtime_env_credentials() is None


def test_unreadable_env_snapshot_retries_without_stat_change(tmp_path, monkeypatch):
    service, _env, _overlay = _service(tmp_path, monkeypatch)
    service._maybe_reload_runtime_overlay(force=True)
    real_reader = service._runtime_env_credentials
    monkeypatch.setattr(service, "_runtime_env_credentials", lambda: None)
    service._maybe_reload_runtime_overlay(force=True)
    assert service._runtime_overlay_status["credential_source_readable"] is False
    assert service._runtime_overlay_status["credential_matches_observed_mount"]["OMBRE_RERANKER_API_KEY"] is None
    assert service._runtime_overlay_status["last_reload_status"] == (
        "credential_source_unavailable_preserved_previous"
    )
    assert service._runtime_overlay_signature is None
    assert service.reranker_engine.api_key == "mounted-initial"

    monkeypatch.setattr(service, "_runtime_env_credentials", real_reader)
    service._maybe_reload_runtime_overlay()
    assert service._runtime_overlay_status["credential_source_readable"] is True
    assert service._runtime_overlay_status["last_reload_status"] == "reloaded"
    assert service._runtime_overlay_signature is not None
    assert service.reranker_engine.api_key == "mounted-initial"


def test_natural_candidate_eligibility_single_pass_preserves_authorized_ids():
    service = GatewayService.__new__(GatewayService)
    service.inject_max_cards = 3
    service.memory_authority_enabled = True
    service.memory_authority_view = SimpleNamespace(
        enabled=True, auto_recallable_bucket_ids=lambda: frozenset({"dynamic", "semantic", "excluded"}),
    )
    service._recall_query_plan = lambda _query: SimpleNamespace(
        skip_reason="", skip_long_term_recall=False,
    )
    service._query_has_relevance_facet = lambda _query: False
    service._is_relevance_suppressed = lambda *_args: False
    service._is_self_anchor_recall_excluded_bucket = lambda bucket: bucket["id"] == "excluded"
    calls = Counter()

    def dynamic(bucket):
        calls[("dynamic", bucket["id"])] += 1
        return bucket["id"] in {"dynamic", "denied"}

    def semantic(bucket):
        calls[("semantic", bucket["id"])] += 1
        return bucket["id"] in {"semantic", "excluded"}

    service._is_dynamic_candidate = dynamic
    service._is_semantic_candidate_bucket = semantic
    service._is_identity_name_candidate_bucket = lambda *_args: False
    service._is_relevance_candidate_bucket = lambda *_args: False
    seen = []

    class ReachedAliases(Exception):
        pass

    def capture_aliases(_query, eligible_ids):
        seen.append(eligible_ids)
        raise ReachedAliases

    service._retrieval_alias_hits = capture_aliases
    buckets = [{"id": name} for name in ("dynamic", "semantic", "excluded", "denied")]
    try:
        asyncio.run(service._dynamic_bucket_candidate_items(
            "ordinary query", "session", buckets, natural_input={"kind": "natural"},
            allow_semantic=False,
        ))
    except ReachedAliases:
        pass
    assert seen == [{"dynamic", "semantic"}]
    assert all(count <= 1 for count in calls.values())
