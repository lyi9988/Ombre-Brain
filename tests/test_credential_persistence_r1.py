"""Offline contract tests for Dashboard credential persistence helpers.

Every credential-like value is synthetic and every file is temporary.
"""
import ast
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dashboard_helpers():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"), filename=str(SERVER))
    wanted = {
        "_dashboard_env_path",
        "_quote_env_value",
        "_write_dashboard_env_values",
        "_dashboard_split_names",
        "_dashboard_api_key_values",
        "_dashboard_sanitize_env_names",
        "_dashboard_sanitize_upstream_models",
        "_dashboard_normalize_upstream_protocol",
        "_dashboard_sanitize_gateway_upstreams",
        "_dashboard_gateway_upstream_env_updates",
        "_dashboard_gateway_hot_upstreams",
    }
    functions = [
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])).body[0]
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise AssertionError(f"server.py helper set changed: found {sorted(node.name for node in functions)}")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"__file__": str(SERVER), "os": os, "re": re, "_json_lib": json}
    exec(compile(module, str(SERVER), "exec"), namespace)
    return namespace


HELPERS = _load_dashboard_helpers()


def _raw(name="edge", *, mode="replace", envs=None, values=None):
    return [{
        "name": name,
        "protocol": "openai",
        "base_url": "https://synthetic.invalid/v1",
        "credential_mode": mode,
        "api_key_envs": list(envs or []),
        "api_key_values": list(values or []),
    }]


class DashboardCredentialPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="brain_credential_persist_r1_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_replace_uses_new_env_key_first_and_old_inline_does_not_mask_it(self):
        existing = [{
            "name": "edge",
            "api_key": "synthetic-old-inline-primary",
            "api_key_envs": ["OLD_EDGE_KEY"],
            "api_keys": [
                {"api_key": "synthetic-backup-inline", "label": "owner-backup"},
                {"api_key": "synthetic-old-env-copy", "label": "env:NEW_EDGE_KEY"},
            ],
            "credential_primary_env": "OLD_EDGE_KEY",
        }]
        raw = _raw(envs=["NEW_EDGE_KEY"], values=["synthetic-new-primary"])
        sanitized = HELPERS["_dashboard_sanitize_gateway_upstreams"](raw, existing)
        updates = HELPERS["_dashboard_gateway_upstream_env_updates"](raw)
        hot = HELPERS["_dashboard_gateway_hot_upstreams"](sanitized, raw, updates)[0]

        self.assertEqual(sanitized[0]["credential_mode"], "replace")
        self.assertEqual(sanitized[0]["credential_primary_env"], "NEW_EDGE_KEY")
        self.assertNotIn("api_key", sanitized[0])
        self.assertEqual(updates, {"NEW_EDGE_KEY": "synthetic-new-primary"})
        self.assertNotIn("api_key", hot, "old inline primary is removed for explicit replacement")
        self.assertEqual(hot["api_keys"][0], {
            "api_key": "synthetic-new-primary", "label": "env:NEW_EDGE_KEY"})
        self.assertTrue(any(item.get("label") == "owner-backup" for item in hot["api_keys"]))
        self.assertFalse(any(item.get("label") == "env:NEW_EDGE_KEY" and
                             item.get("api_key") == "synthetic-old-env-copy"
                             for item in hot["api_keys"]))

    def test_valid_existing_api_keys_remain_as_backups(self):
        existing = [{
            "name": "edge",
            "api_keys": [
                {"api_key": "synthetic-backup-a", "label": "backup-a"},
                {"api_key": "synthetic-backup-b", "label": "backup-b"},
            ],
        }]
        raw = _raw(envs=["NEW_EDGE_KEY"], values=["synthetic-new"])
        sanitized = HELPERS["_dashboard_sanitize_gateway_upstreams"](raw, existing)
        hot = HELPERS["_dashboard_gateway_hot_upstreams"](
            sanitized, raw, HELPERS["_dashboard_gateway_upstream_env_updates"](raw))[0]
        self.assertEqual([item["label"] for item in hot["api_keys"]],
                         ["env:NEW_EDGE_KEY", "backup-a", "backup-b"])

    def test_append_requires_new_env_name_and_preserves_old_names(self):
        existing = [{"name": "edge", "api_key_envs": ["OLD_A", "OLD_B"],
                     "api_keys": [{"api_key": "synthetic-backup", "label": "backup"}]}]
        raw = _raw(mode="append", envs=["NEW_C"], values=["synthetic-c"])
        sanitized = HELPERS["_dashboard_sanitize_gateway_upstreams"](raw, existing)
        self.assertEqual(sanitized[0]["api_key_envs"], ["OLD_A", "OLD_B", "NEW_C"])
        self.assertEqual(sanitized[0]["api_keys"], existing[0]["api_keys"])
        self.assertEqual(HELPERS["_dashboard_gateway_upstream_env_updates"](raw),
                         {"NEW_C": "synthetic-c"})

        invalid = _raw(mode="append", envs=["OLD_B"], values=["synthetic-replacement"])
        with self.assertRaisesRegex(ValueError, "append requires new credential env names"):
            HELPERS["_dashboard_sanitize_gateway_upstreams"](invalid, existing)

    def test_two_key_blank_first_slot_updates_only_second_env(self):
        from credential_file import read_values

        env_file = self.root / "two-key.synthetic.env"
        env_file.write_text(
            "FIRST_KEY=synthetic-first-old\nSECOND_KEY=synthetic-second-old\n",
            encoding="utf-8",
        )
        raw = _raw(envs=["FIRST_KEY", "SECOND_KEY"], values=["", "synthetic-second-new"])
        values = HELPERS["_dashboard_api_key_values"](["", "synthetic-second-new"])
        self.assertEqual(values, ["", "synthetic-second-new"])
        updates = HELPERS["_dashboard_gateway_upstream_env_updates"](raw)
        self.assertEqual(updates, {"SECOND_KEY": "synthetic-second-new"})
        with patch.dict(os.environ, {"OMBRE_ENV_PATH": str(env_file)}):
            HELPERS["_write_dashboard_env_values"](updates)
        reread, _ = read_values(env_file)
        self.assertEqual(reread["FIRST_KEY"], "synthetic-first-old")
        self.assertEqual(reread["SECOND_KEY"], "synthetic-second-new")

    def test_duplicate_env_names_and_cross_upstream_conflicting_updates_rejected(self):
        with self.assertRaisesRegex(ValueError, "env names must be valid, nonempty and unique"):
            HELPERS["_dashboard_sanitize_env_names"](["DUPLICATE_KEY", "DUPLICATE_KEY"])

        raw = [
            {"name": "upstream-a", "api_key_envs": ["SHARED_KEY"],
             "api_key_values": ["synthetic-value-a"]},
            {"name": "upstream-b", "api_key_envs": ["SHARED_KEY"],
             "api_key_values": ["synthetic-value-b"]},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting credential updates"):
            HELPERS["_dashboard_gateway_upstream_env_updates"](raw)

    def test_nul_validation_happens_before_file_write(self):
        env_file = self.root / "synthetic.env"
        before = b"KEEP=synthetic-existing\nTARGET=synthetic-old\n"
        env_file.write_bytes(before)
        bad = "synthetic\x00invalid"
        with patch.dict(os.environ, {"OMBRE_ENV_PATH": str(env_file)}):
            with self.assertRaises(ValueError):
                HELPERS["_dashboard_gateway_upstream_env_updates"](
                    _raw(envs=["TARGET"], values=[bad]))
            with self.assertRaises(ValueError):
                HELPERS["_write_dashboard_env_values"]({"TARGET": bad})
        self.assertEqual(env_file.read_bytes(), before)

    def test_preserve_inode_target_fsync_failure_retains_old_recovery_and_new_temp(self):
        from credential_file import write_values

        env_file = self.root / "recovery.synthetic.env"
        before = b"KEEP=synthetic-old-file\nBRAIN_CREDENTIAL_WRITE_TEST=synthetic-old-file-value\n"
        env_file.write_bytes(before)
        target_stat = env_file.stat()
        real_fsync = os.fsync
        failed_target_sync = []

        def fail_only_target_fsync(fd):
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) == (target_stat.st_dev, target_stat.st_ino):
                recovery_files = list(self.root.glob("recovery.synthetic.env.*.recovery"))
                staged_files = list(self.root.glob("recovery.synthetic.env.*.tmp"))
                self.assertEqual(len(recovery_files), 1)
                self.assertEqual(len(staged_files), 1)
                if os.name == "posix":
                    self.assertEqual(stat.S_IMODE(recovery_files[0].stat().st_mode), 0o600)
                    self.assertEqual(stat.S_IMODE(staged_files[0].stat().st_mode), 0o600)
                self.assertEqual(recovery_files[0].read_bytes(), before)
                staged = staged_files[0].read_text(encoding="utf-8")
                self.assertIn("KEEP=synthetic-old-file", staged)
                self.assertIn("BRAIN_CREDENTIAL_WRITE_TEST=synthetic-new-file-value", staged)
                failed_target_sync.append(True)
                raise OSError("synthetic destination fsync failure")
            return real_fsync(fd)

        with patch.dict(os.environ, {"BRAIN_CREDENTIAL_WRITE_TEST": "synthetic-process-old-value"}):
            with patch("credential_file.os.fsync", side_effect=fail_only_target_fsync):
                with self.assertRaisesRegex(OSError, "credential write unconfirmed"):
                    write_values(env_file, {
                        "BRAIN_CREDENTIAL_WRITE_TEST": "synthetic-new-file-value",
                    }, preserve_inode=True)
            self.assertEqual(os.environ["BRAIN_CREDENTIAL_WRITE_TEST"],
                             "synthetic-process-old-value")

        self.assertEqual(failed_target_sync, [True])
        recovery_files = list(self.root.glob("recovery.synthetic.env.*.recovery"))
        staged_files = list(self.root.glob("recovery.synthetic.env.*.tmp"))
        self.assertEqual(len(recovery_files), 1)
        self.assertEqual(len(staged_files), 1)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(recovery_files[0].stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(staged_files[0].stat().st_mode), 0o600)
        self.assertEqual(recovery_files[0].read_bytes(), before)
        self.assertIn("BRAIN_CREDENTIAL_WRITE_TEST=synthetic-new-file-value",
                      staged_files[0].read_text(encoding="utf-8"))

    def test_duplicate_env_assignment_is_updated_once_and_unrelated_values_remain(self):
        from credential_file import read_values

        env_file = self.root / "synthetic.env"
        env_file.write_text(
            "KEEP=synthetic-keep\nTARGET=synthetic-old-a\nTARGET=synthetic-old-b\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"OMBRE_ENV_PATH": str(env_file)}):
            self.assertEqual(HELPERS["_write_dashboard_env_values"](
                {"TARGET": "synthetic-new"}), ["env.TARGET"])
        values, _ = read_values(env_file)
        self.assertEqual(values["TARGET"], "synthetic-new")
        self.assertEqual(values["KEEP"], "synthetic-keep")
        self.assertEqual(env_file.read_text(encoding="utf-8").count("TARGET="), 1)

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlink/inode semantics")
    def test_symlink_and_target_inode_survive_dashboard_save(self):
        env_file = self.root / "synthetic.env"
        env_file.write_text("KEEP=synthetic-keep\nTARGET=synthetic-old\n", encoding="utf-8")
        symlink = self.root / "release.env"
        symlink.symlink_to(env_file)
        before_inode = env_file.stat().st_ino
        with patch.dict(os.environ, {"OMBRE_ENV_PATH": str(symlink)}):
            HELPERS["_write_dashboard_env_values"]({"TARGET": "synthetic-new"})
        self.assertTrue(symlink.is_symlink())
        self.assertEqual(symlink.resolve(), env_file.resolve())
        self.assertEqual(env_file.stat().st_ino, before_inode)
        self.assertIn("TARGET=synthetic-new", env_file.read_text(encoding="utf-8"))

    @unittest.skipUnless(os.name == "posix", "WSL/Linux hard-link view evidence")
    def test_atomic_replace_stales_hardlink_view_but_preserve_inode_updates_open_view(self):
        from credential_file import write_values

        target = self.root / "mounted.env"
        target.write_text("VALUE=synthetic-before\n", encoding="utf-8")
        pinned = self.root / "pinned-view.env"
        os.link(target, pinned)
        old_inode = target.stat().st_ino
        with pinned.open("rb") as open_view:
            write_values(target, {"VALUE": "synthetic-after-rename"}, preserve_inode=False)
            self.assertNotEqual(target.stat().st_ino, old_inode)
            self.assertEqual(pinned.stat().st_ino, old_inode)
            self.assertIn(b"synthetic-before", pinned.read_bytes())
            self.assertEqual(open_view.read(), b"VALUE=synthetic-before\n")

        pinned.unlink()
        os.link(target, pinned)
        preserved_inode = target.stat().st_ino
        with pinned.open("rb") as open_view:
            write_values(target, {"VALUE": "synthetic-after-in-place"}, preserve_inode=True)
            self.assertEqual(target.stat().st_ino, preserved_inode)
            self.assertEqual(pinned.stat().st_ino, preserved_inode)
            self.assertIn(b"synthetic-after-in-place", pinned.read_bytes())
            open_view.seek(0)
            opened_content = open_view.read()
            self.assertIn(b"VALUE=synthetic-after-in-place\n", opened_content)
            self.assertIn(b"# guyan-credential-generation VALUE ", opened_content)


if __name__ == "__main__":
    unittest.main()
