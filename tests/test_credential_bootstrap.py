"""Fake credentials only: startup deletion evidence and fail-closed loading."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("credential_bootstrap", ROOT / "scripts/credential_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
from credential_file import read_values, write_values

KEY = "OMBRE_EMBEDDING_API_KEY"
MARKER = "# guyan-credential-generation "


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="guyan-bootstrap-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / ".env"
        self.path.write_text("", encoding="utf-8")
        environment = patch.dict(os.environ, {"OMBRE_ENV_PATH": str(self.path),
                                 KEY: "fake-image-old", "UNRELATED": "fake-preserved",
                                 "BACKUP_KEY": "fake-backup", "TOKEN_BUDGET": "32000"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def start(self):
        captured = []
        with patch.object(bootstrap.os, "execvp", side_effect=lambda *_: captured.append(dict(os.environ))) as execute:
            bootstrap.main(["python", "service.py"])
        execute.assert_called_once_with("python", ["python", "service.py"])
        return captured[0]

    def test_deleted_credential_does_not_revive_from_image_environment(self):
        write_values(self.path, {KEY: "fake-saved"})
        generation = write_values(self.path, {KEY: None})[KEY]
        values, generations = read_values(self.path)
        self.assertNotIn(KEY, values)
        self.assertEqual(generations[KEY], generation)
        os.environ[KEY] = "fake-image-old"  # Simulated fresh Docker creation environment.
        adopted = self.start()
        self.assertNotIn(KEY, adopted)
        self.assertEqual(adopted["UNRELATED"], "fake-preserved")
        self.assertEqual(adopted["BACKUP_KEY"], "fake-backup")
        self.assertEqual(adopted["TOKEN_BUDGET"], "32000")

    def test_current_assignment_wins_even_when_generation_marker_exists(self):
        self.path.write_text(KEY + "=fake-first\n" + MARKER + KEY + " " + "a" * 32 + "\n" +
                             KEY + "=fake-current\n", encoding="utf-8")
        adopted = self.start()
        self.assertEqual(adopted[KEY], "fake-current")
        self.assertEqual(adopted["OMBRE_ENV_PATH"], str(self.path))

    def test_explicit_empty_assignment_remains_defined_and_replaces_inherited_key(self):
        write_values(self.path, {KEY: ""})
        os.environ[KEY] = "fake-image-old"
        adopted = self.start()
        self.assertIn(KEY, adopted)
        self.assertEqual(adopted[KEY], "")

    def test_absent_or_invalid_deletion_evidence_preserves_inherited_values(self):
        self.path.write_text(MARKER + KEY + " not-a-generation\n" +
                             MARKER + "INVALID.NAME " + "b" * 32 + "\n", encoding="utf-8")
        os.environ["INVALID.NAME"] = "fake-inherited"
        adopted = self.start()
        self.assertEqual(adopted[KEY], "fake-image-old")
        self.assertEqual(adopted["UNRELATED"], "fake-preserved")
        self.assertEqual(adopted["INVALID.NAME"], "fake-inherited")

    def test_missing_source_neither_executes_nor_changes_environment(self):
        self.path.unlink()
        before = dict(os.environ)
        with patch.object(bootstrap.os, "execvp") as execute, self.assertRaises(OSError):
            bootstrap.main(["python", "service.py"])
        execute.assert_not_called()
        self.assertEqual(dict(os.environ), before)

    def test_read_failure_neither_clears_inheritance_nor_executes(self):
        for failure in (OSError("fake unreadable file"), UnicodeError("fake invalid encoding")):
            before = dict(os.environ)
            with self.subTest(failure=type(failure).__name__), \
                    patch.object(bootstrap, "read_values", side_effect=failure), \
                    patch.object(bootstrap.os, "execvp") as execute, self.assertRaises(type(failure)):
                bootstrap.main(["python", "service.py"])
            execute.assert_not_called()
            self.assertEqual(dict(os.environ), before)

    def test_runtime_copy_cli_keeps_explicit_failure_message(self):
        runtime = self.path.parent / "runtime"
        runtime.mkdir()
        for source in (ROOT / "scripts/credential_bootstrap.py", ROOT / "credential_file.py"):
            shutil.copyfile(source, runtime / source.name)
        environment = dict(os.environ, OMBRE_ENV_PATH=str(self.path.parent / "missing.env"))
        # Windows Python needs its OS directory, not any inherited credential.
        if os.name == "nt":
            environment["SystemRoot"] = str(Path(sys.executable).anchor / Path("Windows"))
        result = subprocess.run([sys.executable, str(runtime / "credential_bootstrap.py"),
                                 sys.executable, "-c", "print('must not execute')"],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.strip(), "credential bootstrap failed; service was not started")


if __name__ == "__main__":
    unittest.main()
