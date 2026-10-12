"""Append-only execution closure tests; no models or benchmark banks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from dsv3wm.native import sha256

path = Path(__file__).resolve().parents[1] / "scripts/freeze_execution_dependencies.py"
spec = importlib.util.spec_from_file_location("execution_dependencies", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExecutionDependenciesTests(unittest.TestCase):
    def fixture(self, directory):
        member, source = directory / "member.txt", directory / "source.json"
        member.write_bytes(b"original frozen input")
        record = {"instrument": "synthetic-development-test", "confirmation_access": False,
                  "files": {str(member.resolve()): sha256(member)}, "gate": {"minimum": 10}}
        source.write_text(json.dumps(record), encoding="utf-8")
        return member, source, record

    def test_original_bytes_metadata_and_pins_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            member, source, old = self.fixture(Path(temporary))
            before = source.read_bytes()
            new = module.extend(source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(new["gate"], old["gate"])
            self.assertEqual(new["files"][str(member.resolve())], old["files"][str(member.resolve())])
            self.assertEqual(new["files"][str(source.resolve()).replace("\\", "/")], sha256(source))
            self.assertTrue(any(name.endswith("bwmn/planbench_extract.py") for name in new["files"]))
            self.assertTrue(any(name.endswith("bwmn/flops.py") for name in new["files"]))

    def test_drift_refuses_instead_of_refreezing_changed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            member, source, _ = self.fixture(Path(temporary))
            member.write_bytes(b"changed input")
            with self.assertRaisesRegex(RuntimeError, "freeze drift"):
                module.extend(source)

    def test_confirmation_access_must_be_explicitly_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, source, record = self.fixture(Path(temporary))
            del record["confirmation_access"]
            source.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed confirmation"):
                module.extend(source)
