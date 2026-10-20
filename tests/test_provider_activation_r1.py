"""Offline regression tests for provider activation; all HTTP is MockTransport."""
import ast, asyncio, builtins, copy, json, os
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
import config_activation

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server.py"
ADMIN = "https://gateway.invalid/api/config"
TOKEN = "gateway-test-token"
SECRET = "owner-secret-must-not-leak"


def run(coro):
    return asyncio.run(coro)


def install_transport(monkeypatch, handler):
    real = httpx.AsyncClient
    options = []
    def client(*, timeout, follow_redirects, trust_env):
        assert trust_env is False
        options.append((timeout, follow_redirects))
        return real(timeout=timeout, follow_redirects=follow_redirects, trust_env=False, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(config_activation.httpx, "AsyncClient", client)
    return options


def health(*, enabled=True, model="embed-v2", base_url="https://embed.invalid/v1",
           configured=True, reload="reloaded", readable=True, source="mounted_env", matches=True):
    return {"status": "ok", "gateway": {
        "embedding": {"enabled": enabled, "model": model, "base_url": base_url, "configured": configured},
        "retrieval_runtime": {"runtime_overlay": {
            "last_reload_status": reload, "credential_source_readable": readable,
            "credential_sources": {"OMBRE_EMBEDDING_API_KEY": source},
            "credential_matches_observed_mount": {"OMBRE_EMBEDDING_API_KEY": matches}}}}}


def test_complete_ack_is_exact_and_does_not_verify_provider(monkeypatch):
    calls = []
    opts = install_transport(monkeypatch, lambda r: calls.append(r) or httpx.Response(
        200, json={"ok": True, "updated": ["memory.model", "memory.top_k"]}))
    result = run(config_activation.activate_gateway({"memory": {"model": "m1", "top_k": 4}},
                                                      admin_url=ADMIN, token=TOKEN))
    assert result["state"] == "confirmed"
    assert result["confirmed_fields"] == ["memory.model", "memory.top_k"]
    assert result["unconfirmed_fields"] == [] and result["provider_verified"] is False
    assert len(calls) == 1 and calls[0].method == "POST" and opts == [(2.0, False)]


@pytest.mark.parametrize("ack,reason", [
    ({"ok": False, "updated": ["memory.model"]}, "invalid_acknowledgement"),
    ({"ok": True, "updated": ["memory.model"]}, "incomplete_acknowledgement"),
    ({"ok": True, "updated": ["memory.model", 4]}, "invalid_acknowledgement")])
def test_false_malformed_and_partial_ack_are_unconfirmed(monkeypatch, ack, reason):
    calls = []
    install_transport(monkeypatch, lambda r: calls.append(r) or httpx.Response(200, json=ack))
    result = run(config_activation.activate_gateway({"memory": {"model": "m1", "top_k": 4}},
                                                      admin_url=ADMIN, token=TOKEN))
    assert result["state"] == "unconfirmed" and result["reason"] == reason
    assert result["provider_verified"] is False and len(calls) == 1


def test_not_configured_http_failure_and_secret_redaction(monkeypatch):
    calls = []
    install_transport(monkeypatch, lambda r: calls.append(r) or httpx.Response(503, text=f"{SECRET} body"))
    absent = run(config_activation.activate_gateway({"memory": {"model": "m"}}, admin_url="", token=TOKEN))
    failed = run(config_activation.activate_gateway({"memory": {"model": "m"}}, admin_url=ADMIN, token=TOKEN))
    assert absent["state"] == "not_configured" and absent["reason"] == "gateway_admin_not_configured"
    assert failed["state"] == "failed" and failed["reason"] == "gateway_http_error"
    assert len(calls) == 1 and SECRET not in repr((absent, failed))


def test_timeout_is_single_attempt_and_does_not_echo_exception_or_secrets(monkeypatch):
    calls = []
    def timeout(r):
        calls.append(r)
        raise httpx.ReadTimeout(f"contains {SECRET}", request=r)
    options = install_transport(monkeypatch, timeout)
    result = run(config_activation.activate_gateway(
        {"dehydration": {"api_key": SECRET}}, admin_url=ADMIN, token=TOKEN))
    assert result["reason"] == "gateway_timeout" and result["attempted"] is True
    assert len(calls) == 1 and options == [(2.0, False)]
    assert SECRET not in repr(result) and TOKEN not in repr(result)


def test_embedding_persistence_gate_health_snapshot_and_provider_false(monkeypatch):
    calls = []
    install_transport(monkeypatch, lambda r: calls.append(r) or httpx.Response(200, json=health()))
    fields = ["embedding.enabled", "embedding.model", "embedding.base_url"]
    expected = {"enabled": True, "model": "embed-v2", "base_url": "https://embed.invalid/v1/"}
    no_persist = run(config_activation.activate_gateway(
        {}, admin_url=ADMIN, token=TOKEN, embedding_fields=fields,
        embedding_persisted=False, embedding_expected=expected))
    assert no_persist["reason"] == "embedding_requires_persistence" and calls == []
    confirmed = run(config_activation.activate_gateway(
        {}, admin_url=ADMIN, token=TOKEN, embedding_fields=fields,
        embedding_persisted=True, embedding_expected=expected))
    assert confirmed["state"] == "confirmed"
    assert confirmed["embedding_confirmation"] == "shared_mount_runtime_snapshot"
    assert confirmed["provider_verified"] is False
    assert [r.method for r in calls] == ["GET"] and calls[0].url.path == "/health"


def test_embedding_missing_credentials_and_runtime_mismatch_are_not_confirmed(monkeypatch):
    snapshots = [health(enabled=False, configured=False), health(model="wrong"), health(source="process_env")]
    install_transport(monkeypatch, lambda r: httpx.Response(200, json=snapshots.pop(0)))
    expected = {"enabled": True, "model": "embed-v2", "base_url": "https://embed.invalid/v1"}
    fieldsets = [["embedding.enabled", "embedding.model", "embedding.base_url"],
                 ["embedding.enabled", "embedding.model", "embedding.base_url"],
                 ["embedding.enabled", "embedding.model", "embedding.base_url", "embedding.api_key"]]
    receipts = [run(config_activation.activate_gateway(
        {}, admin_url=ADMIN, token=TOKEN, embedding_fields=fields,
        embedding_persisted=True, embedding_expected=expected)) for fields in fieldsets]
    assert all(r["state"] == "unconfirmed" for r in receipts)
    assert all(r["reason"] == "embedding_runtime_not_confirmed" for r in receipts)


def test_embedding_generation_distinguishes_two_nonempty_mount_values():
    expected = dict(enabled=True, model="embed-v2", base_url="https://embed.invalid/v1",
                    credential_generation="a" * 32)
    snapshot = health()
    overlay = snapshot["gateway"]["retrieval_runtime"]["runtime_overlay"]
    overlay["credential_generations"] = {"OMBRE_EMBEDDING_API_KEY": "b" * 32}
    assert not config_activation._embedding_matches(snapshot, expected, {"embedding.api_key"})
    overlay["credential_generations"]["OMBRE_EMBEDDING_API_KEY"] = "a" * 32
    assert config_activation._embedding_matches(snapshot, expected, {"embedding.api_key"})


def test_local_consumer_receipt_compares_values_without_exposing_them():
    result = config_activation.local_credential_receipt(
        {"reflection.api_key": SECRET, "dream.api_key": SECRET},
        {"reflection.api_key": SimpleNamespace(api_key=SECRET),
         "dream.api_key": SimpleNamespace(api_key="TEST_ONLY_older")})
    assert result["confirmed_fields"] == ["reflection.api_key"]
    assert result["unconfirmed_fields"] == ["dream.api_key"]
    assert result["state"] == "unconfirmed" and result["provider_verified"] is False
    assert SECRET not in repr(result) and "TEST_ONLY_older" not in repr(result)


def test_stale_adoption_nonce_cannot_confirm_current_credential(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"ok": True, "updated": ["reranker.api_key"],
            "credential_adoption": {"generation": "0" * 32, "state": "consumer_snapshot",
                                    "confirmed_fields": ["reranker.api_key"]},
            "reranker": {"api_ready": True}})
    install_transport(monkeypatch, handler)
    result = run(config_activation.activate_gateway({"reranker": {"api_key": SECRET}},
        admin_url=ADMIN, token=TOKEN))
    assert result["reason"] == "credential_consumer_not_confirmed"


def test_upstream_save_preserves_effective_base_backups_in_runtime_overlay(monkeypatch, tmp_path):
    import yaml
    from test_credential_persistence_r1 import HELPERS
    runtime = tmp_path / "runtime.yaml"
    env = tmp_path / "synthetic.env"
    monkeypatch.setenv("OMBRE_ENV_PATH", str(env))
    monkeypatch.delenv("OMBRE_GATEWAY_ADMIN_URL", raising=False)
    existing = dict(name="fixture", base_url="https://old.invalid/v1",
        api_key="TEST_ONLY_old_primary",
        api_keys=[dict(api_key="TEST_ONLY_backup", label="owner-backup")])
    ns, route = load_route({"config": {"_runtime_config_path": str(runtime),
        "state_dir": str(tmp_path), "gateway": {"upstreams": [existing]}}})
    ns.update(HELPERS)
    ns["_dashboard_gateway_upstreams_have_key_values"] = lambda raw: any(
        HELPERS["_dashboard_api_key_values"](item.get("api_key_values", [])) for item in raw)
    raw = dict(name="fixture", base_url="https://new.invalid/v1",
        api_key_envs=["FIRST", "SECOND"], api_key_values=["TEST_ONLY_one", "TEST_ONLY_two"],
        models=["model-fixture"], credential_mode="replace")
    result = response_json(run(route(Request(dict(gateway={"upstreams": [raw]},
                                                  persist=True, persist_env=True)))))
    assert result["ok"] is True
    saved = yaml.safe_load(runtime.read_text(encoding="utf-8"))["gateway"]["upstreams"][0]
    assert "api_key" not in saved
    assert saved["credential_primary_env"] == "FIRST"
    assert saved["api_keys"] == existing["api_keys"]
    assert "api_key_values" not in saved
    assert "TEST_ONLY" not in repr(result)


def test_old_model_settings_page_conflicts_before_mutating_consumer(monkeypatch, tmp_path):
    monkeypatch.delenv("OMBRE_GATEWAY_ADMIN_URL", raising=False)
    ns, route = load_route({"config": {"_runtime_config_path": str(tmp_path / "runtime.yaml"),
        "dehydration": {"model": "before", "api_key": ""}}})
    revision = config_activation.credential_edit_revision(ns["config"])
    first = response_json(run(route(Request({"dehydration": {"model": "first-page"},
                                            "expected_credential_revision": revision}))))
    assert first["ok"] is True and first["credential_revision"] != revision
    second = run(route(Request({"dehydration": {"model": "stale-second-page"},
                                "expected_credential_revision": revision})))
    assert second.status_code == 409
    assert response_json(second)["error"] == "credential_revision_conflict"
    assert ns["config"]["dehydration"]["model"] == "first-page"
    assert ns["dehydrator"].model == "first-page"


def load_route(overrides=None):
    tree = ast.parse(SERVER.read_text(encoding="utf-8"), filename=str(SERVER))
    names = {"api_config_update", "_hot_update_gateway_config"}
    defs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            defs.append(node)
    assert {n.name for n in defs} == names
    module = ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[]))
    ns = {"__file__": str(SERVER), "os": os,
          "reflection_engine": None, "memory_authority_store": None, "memory_commit_service": None,
          "logger": SimpleNamespace(warning=lambda *a, **k: None), "config": {},
          "dehydrator": SimpleNamespace(model="old", base_url="", api_key=""),
          "embedding_engine": SimpleNamespace(enabled=False, model="", base_url="", reconfigure=lambda _: None),
          "_require_dashboard_auth": lambda _: None,
          "_write_dashboard_env_values": lambda updates: list(updates),
          "_dashboard_env_path": lambda: str(Path(ns["config"].get("_runtime_config_path", "synthetic-only.runtime")).with_suffix(".env")),
          "activate_gateway": config_activation.activate_gateway,
          "activation_receipt": config_activation.activation_receipt,
          "config_save_result": config_activation.config_save_result}
    ns.update(overrides or {})
    exec(compile(module, str(SERVER), "exec"), ns)
    return ns, ns["api_config_update"]


class Request:
    def __init__(self, body): self.body = body
    async def json(self): return self.body


def response_json(response):
    return json.loads(response.body)


def test_actual_route_persists_yaml_and_env_before_embedding_health_readback(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime.yaml"
    events = []
    ns, route = load_route({"config": {
        "_runtime_config_path": str(runtime), "state_dir": str(tmp_path),
        "embedding": {"enabled": False, "model": "old", "base_url": "https://old.invalid/v1", "api_key": ""}}})

    async def reconfigure(cfg):
        engine = ns["embedding_engine"]
        engine.enabled, engine.model, engine.base_url = (
            cfg["embedding"]["enabled"], cfg["embedding"]["model"], cfg["embedding"]["base_url"])
    ns["embedding_engine"].reconfigure = reconfigure

    def save_env(updates):
        events.append("env_saved")
        assert runtime.exists() and updates == {"OMBRE_EMBEDDING_API_KEY": SECRET}
        from credential_file import write_values
        write_values(ns["_dashboard_env_path"](), updates, preserve_inode=True)
        return ["OMBRE_EMBEDDING_API_KEY"]
    ns["_write_dashboard_env_values"] = save_env
    monkeypatch.setenv("OMBRE_GATEWAY_ADMIN_URL", ADMIN)
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", TOKEN)

    def handler(req):
        import yaml
        saved = yaml.safe_load(runtime.read_text(encoding="utf-8"))
        assert saved["embedding"] == {"enabled": True, "model": "embed-v2", "base_url": "https://embed.invalid/v1"}
        assert "api_key" not in saved["embedding"]
        assert events == ["env_saved"] and req.method == "GET"
        from credential_file import read_values
        observed = health()
        observed["gateway"]["retrieval_runtime"]["runtime_overlay"]["credential_generations"] = read_values(ns["_dashboard_env_path"]())[1]
        return httpx.Response(200, json=observed)
    install_transport(monkeypatch, handler)
    response = run(route(Request({"embedding": {
        "enabled": True, "model": "embed-v2", "base_url": "https://embed.invalid/v1", "api_key": SECRET},
        "persist": True, "persist_env": True})))
    result = response_json(response)
    assert result["ok"] is True and result["gateway_activation"]["state"] == "confirmed"
    assert result["gateway_activation"]["provider_verified"] is False
    assert SECRET not in response.body.decode("utf-8")


def test_actual_route_omits_empty_dehydration_key_and_nonempty_key_needs_ack(monkeypatch, tmp_path):
    calls = []
    _, route = load_route({"config": {
        "_runtime_config_path": str(tmp_path / "runtime.yaml"), "state_dir": str(tmp_path),
        "dehydration": {"model": "old", "base_url": "https://old.invalid/v1", "api_key": ""}}})
    monkeypatch.setenv("OMBRE_GATEWAY_ADMIN_URL", ADMIN)
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", TOKEN)
    def handler(req):
        payload = json.loads(req.content)
        calls.append(payload)
        fields = [f"{section}.{key}" for section, values in payload.items() for key in values]
        ack = fields if "dehydration.api_key" not in fields else [f for f in fields if f != "dehydration.api_key"]
        return httpx.Response(200, json={"ok": True, "updated": ack})
    install_transport(monkeypatch, handler)
    empty = response_json(run(route(Request({"dehydration": {"model": "empty-key-model", "api_key": ""}}))))
    assert "api_key" not in calls[0]["dehydration"] and empty["gateway_activation"]["state"] == "confirmed"
    keyed = response_json(run(route(Request({"dehydration": {"api_key": SECRET}}))))
    assert calls[1]["dehydration"]["api_key"] == SECRET
    assert keyed["gateway_activation"]["reason"] == "incomplete_acknowledgement"
    assert SECRET not in repr(keyed)


def test_actual_route_yaml_and_env_persistence_failures_never_sync(monkeypatch, tmp_path):
    monkeypatch.setenv("OMBRE_GATEWAY_ADMIN_URL", ADMIN)
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", TOKEN)
    gateway_calls = []
    install_transport(monkeypatch, lambda req: gateway_calls.append(req) or httpx.Response(
        200, json={"ok": True, "updated": []}))

    yaml_path = tmp_path / "yaml-fail.yaml"
    ns, route = load_route({"config": {
        "_runtime_config_path": str(yaml_path), "state_dir": str(tmp_path), "dehydration": {"api_key": ""}}})
    import memory_system_config
    def disk_full(*args, **kwargs):
        raise OSError("disk full")
    with monkeypatch.context() as scoped:
        scoped.setattr(memory_system_config, 'update_overlay', disk_full)
        yaml_failure = response_json(run(route(Request({"dehydration": {"model": "new"}, "persist": True}))))
    assert yaml_failure["error"] == "runtime_persist_failed"
    assert yaml_failure["persistence"]["runtime_yaml"] == "failed" and gateway_calls == []

    env_path = tmp_path / "env-fail.yaml"
    def fail_env(_updates):
        raise OSError("credential store unavailable")
    _, route = load_route({"config": {
        "_runtime_config_path": str(env_path), "state_dir": str(tmp_path), "dehydration": {"api_key": ""}},
        "_write_dashboard_env_values": fail_env})
    env_failure = response_json(run(route(Request({
        "dehydration": {"model": "new", "api_key": SECRET}, "persist": True, "persist_env": True}))))
    assert env_failure["error"] == "credential_persist_failed"
    assert env_failure["persistence"]["runtime_yaml"] == "saved"
    assert env_failure["persistence"]["credentials"] == "unconfirmed"
    assert env_path.exists() and gateway_calls == []


@pytest.mark.parametrize("actual", [768, None, "missing", 1024])
def test_embedding_dimension_is_part_of_runtime_confirmation(monkeypatch, actual):
    snapshot = health()
    if actual != "missing":
        snapshot["gateway"]["embedding"]["requested_dimension"] = actual
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=snapshot))
    receipt = run(config_activation.activate_gateway(
        {}, admin_url=ADMIN, token=TOKEN, embedding_fields=["embedding.dimensions"],
        embedding_persisted=True, embedding_expected={
            "enabled": True, "model": "embed-v2", "base_url": "https://embed.invalid/v1",
            "dimensions": 1024}))
    assert (receipt["state"] == "confirmed") is (actual == 1024)


@pytest.mark.parametrize("observed", [None, {"enabled": False, "model": "r1", "api_ready": True},
                                    {"enabled": True, "model": "wrong", "api_ready": True},
                                    {"enabled": True, "model": "r1", "api_ready": False},
                                    {"enabled": True, "model": "r1", "api_ready": True}])
def test_reranker_ack_must_include_matching_effective_runtime(monkeypatch, observed):
    fields = ["reranker.enabled", "reranker.model", "reranker.api_key"]
    install_transport(monkeypatch, lambda _: httpx.Response(200, json={
        "ok": True, "updated": fields, "reranker": observed}))
    receipt = run(config_activation.activate_gateway(
        {"reranker": {"enabled": True, "model": "r1", "api_key": SECRET}},
        admin_url=ADMIN, token=TOKEN))
    # Existence and even matching parameters cannot prove which credential
    # the consumer adopted. Legacy ACK has no per-operation identity.
    assert receipt["state"] == "unconfirmed"
    assert SECRET not in repr(receipt)
    assert receipt["reason"] == "credential_consumer_not_confirmed"
    assert "reranker.api_key" not in receipt["confirmed_fields"]


def test_actual_reranker_runtime_key_is_saved_before_sync_without_other_key_writes(monkeypatch, tmp_path):
    from reranker_engine import RerankerEngine
    from dotenv import dotenv_values
    import re
    env_path = tmp_path / ".env"
    env_path.write_text("OMBRE_RERANKER_API_KEY=old\nOMBRE_API_KEY=keep-existing\n", encoding="utf-8")
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    names = {"_quote_env_value", "_write_dashboard_env_values"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns, route = load_route({"config": {
        "_runtime_config_path": str(tmp_path / "runtime.yaml"),
        "reranker": {"api_key": SECRET, "base_url": "https://rerank.invalid/v1", "model": "r1"},
        "dehydration": {"api_key": "unsaved-unrelated"}},
        "RerankerEngine": RerankerEngine, "re": re, "_json_lib": json,
        "_dashboard_env_path": lambda: str(env_path)})
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[])), str(SERVER), "exec"), ns)
    monkeypatch.setenv("OMBRE_RERANKER_API_KEY", "before")
    monkeypatch.setenv("OMBRE_GATEWAY_ADMIN_URL", ADMIN)
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", TOKEN)
    def handler(req):
        saved = dotenv_values(env_path, interpolate=False)
        assert saved == {"OMBRE_RERANKER_API_KEY": SECRET, "OMBRE_API_KEY": "keep-existing"}
        sent = json.loads(req.content)
        assert sent["reranker"] == {"api_key": SECRET}
        return httpx.Response(200, json={"ok": True, "updated": ["reranker.api_key"],
            "credential_adoption": {"generation": sent["_credential_generation"], "state": "consumer_snapshot", "confirmed_fields": ["reranker.api_key"]},
            "reranker": {"api_ready": True}})
    install_transport(monkeypatch, handler)
    result = response_json(run(route(Request({"reranker": {}, "persist_env": True}))))
    assert result["gateway_activation"]["state"] == "confirmed"
    assert result["persistence"]["credentials"] == "saved"
    assert SECRET not in repr(result)


def test_embedding_own_runtime_key_saved_on_explicit_save_without_new_input(monkeypatch, tmp_path):
    captured = []
    async def reconfigure(_):
        return None
    _, route = load_route({"config": {
        "_runtime_config_path": str(tmp_path / "runtime.yaml"),
        "embedding": {"api_key": SECRET, "model": "embed-v2", "base_url": "https://embed.invalid/v1"}},
        "embedding_engine": SimpleNamespace(reconfigure=reconfigure),
        "_write_dashboard_env_values": lambda values: captured.append(values) or list(values)})
    monkeypatch.delenv("OMBRE_GATEWAY_ADMIN_URL", raising=False)
    result = response_json(run(route(Request({"embedding": {}, "persist_env": True}))))
    assert captured == [{"OMBRE_EMBEDDING_API_KEY": SECRET}]
    assert result["persistence"]["credentials"] == "saved"
    assert result["gateway_activation"]["state"] == "not_configured"
