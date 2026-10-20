"""Offline tests. Docker test is opt-in and uses only generated fake credentials.

GUYAN_MOUNT_DOCKER=1 GUYAN_GATEWAY_SOURCE=/.../gateway.py python3 -m unittest \
  discover -s tests -p test_credential_directory_mount.py -v
Run on the local Linux Docker host as root (for /proc/PID/root verification).
"""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("credential_mount", ROOT / "scripts/credential_mount.py")
mount = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mount)
from credential_file import read_values, write_values


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="guyan-credential-unit-")
        self.root = Path(self.tmp.name)
        self.operation = self.root / "operation"
        self.operation.mkdir()
        self.credentials = self.root / "credentials"
        self.credentials.mkdir()
        (self.operation / "host-current.env").write_text("OMBRE_EMBEDDING_API_KEY=fake-initial\nBACKUP_KEY=fake-backup\n")
        shutil.copyfile(self.operation / "host-current.env", self.credentials / ".env")
        self.state = {"credentials_dir": str(self.credentials)}
        self.service = "ombre-brain"
        self.model = {"services": {self.service: {
            "command": ["python", "server.py"], "image": "old",
            "env_file": ["old.env"], "environment": {
                "OMBRE_EMBEDDING_API_KEY": "fake-initial", "BACKUP_KEY": "fake-backup",
                "OMBRE_BUCKETS_DIR": "/data", "UNRELATED": "literal$VALUE"},
            "volumes": [{"type": "bind", "source": "/old/env", "target": "/app/.env"},
                        {"type": "bind", "source": "/state", "target": "/state"}],
        }}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_rebuild_and_rollback_do_not_embed_old_credentials(self):
        write_values(self.credentials / ".env", {"OMBRE_EMBEDDING_API_KEY": "fake-new"})
        rendered = mount.render_model(self.model, self.service, self.state, self.operation, "sha256:new")
        service = rendered["services"][self.service]
        self.assertNotIn("OMBRE_EMBEDDING_API_KEY", service["environment"])
        self.assertNotIn("env_file", service)
        self.assertEqual(service["environment"]["UNRELATED"], "literal$$VALUE")
        self.assertEqual(service["environment"]["OMBRE_BUCKETS_DIR"], "/data")
        targets = [v["target"] for v in service["volumes"]]
        self.assertEqual(targets, ["/state", mount.CONTAINER_DIR, mount.RUNTIME_DIR])
        rollback = mount.render_model(self.model, self.service, self.state, self.operation, "sha256:old")
        self.assertEqual(rollback["services"][self.service]["volumes"], service["volumes"])
        self.assertEqual(read_values(self.credentials / ".env")[0]["OMBRE_EMBEDDING_API_KEY"], "fake-new")

    def test_explicit_override_conflict_is_not_silently_lost(self):
        self.model["services"][self.service]["environment"]["BACKUP_KEY"] = "fake-other"
        with self.assertRaises(mount.Refused):
            mount.render_model(self.model, self.service, self.state, self.operation, "new")

    def test_original_model_unchanged_and_missing_mount_refused(self):
        original = deepcopy(self.model)
        mount.render_model(self.model, self.service, self.state, self.operation, "new")
        self.assertEqual(self.model, original)
        self.model["services"][self.service]["volumes"] = []
        with self.assertRaises(mount.Refused):
            mount.render_model(self.model, self.service, self.state, self.operation, "new")

    def test_private_creation_and_no_overwrite(self):
        path = self.operation / "private.env"
        mount.private_write(path, b"FAKE=private\n")
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            mount.private_write(path, b"replacement")

    def test_rollback_can_select_a_compatible_image(self):
        self.state["services"] = {self.service: {"image": "sha256:old", "project": "test"}}
        (self.operation / "state.json").write_text(json.dumps(self.state))
        (self.operation / (self.service + ".original.json")).write_text(json.dumps(self.model))
        with patch.object(sys, "platform", "linux"), patch.object(sys, "argv", [
                "credential_mount.py", "rollback", "--operation-dir", str(self.operation),
                "--service", self.service, "--image", "sha256:compatible"]):
            mount.main()
        rendered = json.loads((self.operation / (self.service + ".rollback.json")).read_text())
        self.assertEqual(rendered["services"][self.service]["image"], "sha256:compatible")

    @unittest.skipUnless(os.name == "posix", "POSIX process root verification")
    def test_verify_requires_runtime_source_and_bootstrap(self):
        self.credentials.chmod(0o700)
        (self.credentials / ".env").chmod(0o600)
        entry = self.root / ".env"
        entry.symlink_to(self.credentials / ".env")
        self.state.update(host_entry=str(entry), runtime_dir=str(self.operation / "runtime"))
        info = {"State": {"Running": True, "Pid": os.getpid()}, "Mounts": [
            {"Destination": str(self.credentials), "Type": "bind", "RW": True, "Source": str(self.credentials)},
            {"Destination": mount.RUNTIME_DIR, "Type": "bind", "RW": False, "Source": self.state["runtime_dir"]}],
            "Config": {"Env": ["OMBRE_ENV_PATH=" + str(self.credentials / ".env")],
                       "Entrypoint": ["python", mount.RUNTIME_DIR + "/credential_bootstrap.py"]}}
        with patch.object(mount, "SERVICES", ("one",)), patch.object(mount, "CONTAINER_DIR", str(self.credentials)), \
                patch.object(mount, "inspect", return_value=info):
            self.assertEqual(mount.verify(self.state)["phase"], "mounts_verified")
            info["Config"]["Entrypoint"] = []
            with self.assertRaises(mount.Refused):
                mount.verify(self.state)
            info["Config"]["Entrypoint"] = ["python", mount.RUNTIME_DIR + "/credential_bootstrap.py"]
            info["Mounts"][1]["RW"] = True
            with self.assertRaises(mount.Refused):
                mount.verify(self.state)

    def test_activate_refuses_live_containers_and_changed_source(self):
        entry = self.root / ".env"
        shutil.copyfile(self.operation / "host-current.env", entry)
        self.state.update(phase="prepared", operation=uuid.uuid4().hex, host_entry=str(entry),
                          host_resolved=str(entry), services={"original": {"container": "id"}}, preimage_checked_at=1)
        with patch.object(mount, "inspect", return_value={"Id": "id", "State": {"Running": True}}):
            with self.assertRaises(mount.Refused):
                mount.activate(self.state, self.operation)
        entry.write_text("OMBRE_EMBEDDING_API_KEY=fake-later-owner-save\n")
        with patch.object(mount, "inspect", return_value={"Id": "id", "State": {"Running": False}}):
            with self.assertRaises(mount.Refused):
                mount.activate(self.state, self.operation)
        self.assertIn("fake-later-owner-save", entry.read_text())
        self.assertFalse(entry.is_symlink())

    @unittest.skipUnless(os.name == "posix", "POSIX symlinks")
    def test_activate_and_unknown_outcome_resume_keep_current_file(self):
        entry = self.root / ".env"
        shutil.copyfile(self.operation / "host-current.env", entry)
        self.state.update(phase="prepared", operation=uuid.uuid4().hex, host_entry=str(entry),
                          host_resolved=str(entry), services={"original": {"container": "id"}}, preimage_checked_at=1)
        with patch.object(mount, "inspect", return_value={"Id": "id", "State": {"Running": False}}):
            mount.activate(self.state, self.operation)
        self.assertEqual(entry.resolve(), self.credentials / ".env")
        write_values(entry, {"OMBRE_EMBEDDING_API_KEY": "fake-after-cutover"})
        mount.activate(self.state, self.operation)
        self.assertEqual(read_values(entry)[0]["OMBRE_EMBEDDING_API_KEY"], "fake-after-cutover")


def extract_gateway(source):
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GatewayService")
    names = {"_runtime_env_path", "_runtime_env_credentials", "_runtime_credential_value",
             "_bool_config_value", "_apply_runtime_overlay"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise RuntimeError("actual Gateway credential methods unavailable")
    return ast.unparse(ast.Module(body=[ast.ClassDef(name="GatewayService", bases=[], keywords=[],
                                                   body=methods, decorator_list=[])], type_ignores=[]))


CONSUMER = '''
import ast, json, os, re, sys, threading, types, urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from credential_file import read_values
from gateway_methods import GatewayService
from credential_bootstrap import current_environment
class EmbeddingEngine:
    @staticmethod
    def _requested_dimension(value): return int(value or 1024)
import gateway_methods
gateway_methods.EmbeddingEngine = EmbeddingEngine
gateway_methods.Path, gateway_methods.os = Path, os
gateway_methods.json, gateway_methods.re = json, re
engine = types.SimpleNamespace(model='fake-model', dimension=1024, base_url='http://127.0.0.1:8090',
                               api_key='', enabled=True, timeout=2, candidate_limit=3, score_weight=1)
service = GatewayService()
service.config = {'embedding': {'enabled': True, 'model': 'fake-model', 'base_url': engine.base_url}}
service.embedding_engine = engine
service.reranker_engine = types.SimpleNamespace(**vars(engine))
service.gateway_cfg = {}
tree = ast.parse(Path('/fixture/server.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
         ('_dashboard_env_path', '_write_dashboard_env_values')]
scope = {'os': os, '__file__': '/app/server.py'}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual_brain_writer', 'exec'), scope)
captured = []
class MockProvider(BaseHTTPRequestHandler):
    def do_POST(self):
        captured.append(self.headers.get('Authorization'))
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def log_message(self, *args): pass
threading.Thread(target=ThreadingHTTPServer(('127.0.0.1', 8090), MockProvider).serve_forever, daemon=True).start()
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        try:
            if self.path == '/save':
                scope['_write_dashboard_env_values'](data['updates'])
            service._apply_runtime_overlay({'embedding': {}, 'reranker': {}}, env_changed=True)
            if self.path == '/outbound':
                request = urllib.request.Request(engine.base_url + '/embeddings', data=b'{}',
                          headers={'Authorization': 'Bearer ' + engine.api_key})
                urllib.request.urlopen(request, timeout=2).read()
            values, generations = read_values(os.environ['OMBRE_ENV_PATH'])
            body = {'values': values, 'generations': generations, 'consumer_key': engine.api_key,
                    'captured': captured, 'startup': os.environ.get('OMBRE_EMBEDDING_API_KEY')}
            self.send_response(200)
        except Exception as error:
            body = {'error': type(error).__name__}; self.send_response(409)
        self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def log_message(self, *args): pass
ThreadingHTTPServer(('127.0.0.1', 8080), Handler).serve_forever()
'''


@unittest.skipUnless(os.environ.get("GUYAN_MOUNT_DOCKER") == "1", "set GUYAN_MOUNT_DOCKER=1 for real Docker bind test")
class RealDockerMountTests(unittest.TestCase):
    def test_migration_save_hot_consumer_atomic_replace_rebuild_and_compatible_rollback(self):
        self.assertEqual(sys.platform, "linux")
        self.assertEqual(os.geteuid(), 0, "test requires local root /proc access")
        image = os.environ.get("GUYAN_MOUNT_IMAGE", "python:3.12-slim")
        subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, check=True)
        gateway = Path(os.environ["GUYAN_GATEWAY_SOURCE"])
        with tempfile.TemporaryDirectory(prefix="guyan-real-credential-bind-") as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            fixture.mkdir()
            for source in (ROOT / "credential_file.py", ROOT / "scripts/credential_bootstrap.py", ROOT / "server.py"):
                shutil.copyfile(source, fixture / source.name)
            (fixture / "gateway_methods.py").write_text("from __future__ import annotations\n" + extract_gateway(gateway))
            (fixture / "consumer.py").write_text(CONSUMER)
            entry = root / ".env"
            write_values(entry, {"OMBRE_EMBEDDING_API_KEY": "fake-initial", "BACKUP_KEY": "fake-backup"})
            project = "guyancred" + uuid.uuid4().hex[:12]
            names = (project + "-gateway", project + "-brain")
            model = {"name": project, "services": {name: {
                "image": image, "container_name": name, "network_mode": "none",
                "command": ["python", "/fixture/consumer.py"],
                "env_file": [str(entry)],
                "environment": {"PYTHONPATH": "/fixture", "OMBRE_ENV_PATH": "/app/.env"},
                "volumes": [{"type": "bind", "source": str(entry), "target": "/app/.env"},
                            {"type": "bind", "source": str(fixture), "target": "/fixture", "read_only": True}]
            } for name in names}}
            original = root / "compose.json"
            original.write_text(json.dumps(model))
            def compose(file, *args):
                return subprocess.run(["docker", "compose", "-p", project, "-f", str(file), *args],
                                      capture_output=True, check=True)
            def request(name, path="/read", data=None, expected=200):
                code = """import json,urllib.request,urllib.error,sys
try:
 r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8080'+sys.argv[1],data=sys.argv[2].encode(),headers={'Content-Type':'application/json'}),timeout=3)
except urllib.error.HTTPError as e: r=e
print(json.dumps({'status':r.status,'body':json.loads(r.read())}))"""
                result = subprocess.run(["docker", "exec", name, "python", "-c", code, path, json.dumps(data or {})],
                                        capture_output=True, check=True)
                body = json.loads(result.stdout)
                self.assertEqual(body["status"], expected)
                return body["body"]
            def ready(name):
                import time
                for _ in range(30):
                    try:
                        return request(name)
                    except (subprocess.CalledProcessError, AssertionError):
                        time.sleep(.1)
                logs = subprocess.run(["docker", "logs", name], capture_output=True)
                self.fail("fake consumer not ready: " + logs.stderr.decode()[-2500:])
            try:
                compose(original, "up", "-d", "--no-build")
                for name in names:
                    ready(name)
                # Actual old file bind pins an inode after host atomic replacement.
                old_inode = entry.stat().st_ino
                write_values(entry, {"OMBRE_EMBEDDING_API_KEY": "fake-host-new"})
                self.assertNotEqual(entry.stat().st_ino, old_inode)
                self.assertEqual(request(names[1])["values"]["OMBRE_EMBEDDING_API_KEY"], "fake-initial")
                with patch.object(mount, "SERVICES", names):
                    args = argparse.Namespace(writers_quiesced=True, host_env=str(entry),
                           credentials_dir=str(root / "credentials"), operation_dir=str(root / "conflict"))
                    with self.assertRaises(mount.Refused):
                        mount.prepare(args)
                    self.assertTrue((root / "conflict" / (names[0] + ".env")).is_file())
                    self.assertFalse((root / "credentials").exists())
                    # Reconcile fake values explicitly; this is not an automatic secret choice.
                    request(names[1], "/save", {"updates": {"OMBRE_EMBEDDING_API_KEY": "fake-host-new"}})
                    args.operation_dir = str(root / "operation")
                    state = mount.prepare(args)
                    operation = Path(args.operation_dir)
                    mount.check_preimage(state, operation)
                    # Late write reaches the orphan inode, leaving host unchanged.
                    request(names[1], "/save", {"updates": {"LATE_SAVE": "fake-late-owner-save"}})
                    self.assertNotIn("LATE_SAVE", read_values(entry)[0])
                    with self.assertRaises(mount.Refused):
                        mount.check_preimage(state, operation)
                    self.assertNotIn("preimage_checked_at", state)
                    with self.assertRaises(mount.Refused):
                        mount.activate(state, operation)
                    self.assertFalse(entry.is_symlink())
                    # Explicit fake-value reconciliation and a new operation; no replay.
                    write_values(entry, {"LATE_SAVE": "fake-late-owner-save"})
                    (root / "credentials").rename(operation / "unused-stage")
                    args.operation_dir = str(root / "operation-current")
                    state = mount.prepare(args)
                    operation = Path(args.operation_dir)
                    mount.check_preimage(state, operation)
                    forward, rollback = {}, {}
                    for name in names:
                        original_model = json.loads((operation / (name + ".original.json")).read_text())
                        for kind, collection in (("directory", forward), ("rollback", rollback)):
                            rendered = mount.render_model(original_model, name, state, operation, image)
                            file = operation / (name + "." + kind + ".json")
                            mount.private_write(file, json.dumps(rendered).encode())
                            collection[name] = file
                            compose(file, "config", "--quiet")
                    with self.assertRaises(mount.Refused):
                        mount.activate(state, operation)
                    compose(original, "stop", *names)
                    mount.activate(state, operation)
                    for name in names:
                        compose(forward[name], "up", "-d", "--no-deps", "--no-build", name)
                        self.assertEqual(ready(name)["startup"], "fake-host-new")
                    mount.verify(state)
                    result = request(names[1], "/save", {"updates": {"OMBRE_EMBEDDING_API_KEY": "fake-saved", "SECOND_KEY": "fake-secondary"}})
                    self.assertEqual(read_values(entry)[0]["OMBRE_EMBEDDING_API_KEY"], "fake-saved")
                    self.assertEqual(result["generations"], read_values(entry)[1])
                    used = request(names[0], "/outbound")
                    self.assertEqual(used["consumer_key"], "fake-saved")
                    self.assertEqual(used["captured"][-1], "Bearer fake-saved")
                    request(names[1], "/save", {"updates": {"OMBRE_EMBEDDING_API_KEY": "bad\nvalue"}}, expected=409)
                    self.assertEqual(read_values(entry)[0]["OMBRE_EMBEDDING_API_KEY"], "fake-saved")
                    # Host symlink and container writers use the same resolved .lock.
                    errors = []
                    def save_host():
                        try:
                            write_values(entry, {"HOST_PARALLEL": "fake-host"})
                        except Exception as e:
                            errors.append(type(e).__name__)
                    thread = threading.Thread(target=save_host)
                    thread.start()
                    request(names[1], "/save", {"updates": {"BRAIN_PARALLEL": "fake-brain"}})
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(errors)
                    values = read_values(entry)[0]
                    self.assertEqual(values["HOST_PARALLEL"], "fake-host")
                    self.assertEqual(values["BRAIN_PARALLEL"], "fake-brain")
                    # Directory bind survives atomic replacement, in both running consumers.
                    old_inode = entry.stat().st_ino
                    write_values(entry, {"OMBRE_EMBEDDING_API_KEY": "fake-atomic-latest"})
                    self.assertNotEqual(entry.stat().st_ino, old_inode)
                    self.assertTrue(entry.is_symlink())
                    for name in names:
                        self.assertEqual(request(name)["consumer_key"], "fake-atomic-latest")
                    mount.verify(state)
                    for name in names:
                        compose(forward[name], "up", "-d", "--force-recreate", "--no-deps", "--no-build", name)
                        self.assertEqual(ready(name)["startup"], "fake-atomic-latest")
                    # Compatible image rollback retains the directory + bootstrap.
                    for name in names:
                        compose(rollback[name], "up", "-d", "--force-recreate", "--no-deps", "--no-build", name)
                        after = ready(name)
                        self.assertEqual(after["startup"], "fake-atomic-latest")
                        self.assertEqual(after["values"]["BACKUP_KEY"], "fake-backup")
                        self.assertEqual(after["values"]["SECOND_KEY"], "fake-secondary")
                        self.assertEqual(after["values"]["LATE_SAVE"], "fake-late-owner-save")
                    mount.verify(state)
            finally:
                subprocess.run(["docker", "rm", "-f", *names], capture_output=True)


if __name__ == "__main__":
    unittest.main()
