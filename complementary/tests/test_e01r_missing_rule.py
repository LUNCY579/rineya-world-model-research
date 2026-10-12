"""Section 10 missing-cell rule, torn attempt lines and complete bound-code records (E01R freeze review item 6g)."""
import contextlib
import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01R"))

import analyze_e01r as A  # noqa: E402
import native_e01r as M  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


class TestMissingRule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cells = Path(self.tmp.name) / "cells"
        self.e01n = Path(self.tmp.name) / "e01n"
        self.cells.mkdir()
        self.ne = SimpleNamespace(MAX_FURTHER_ATTEMPTS=M.MAX_FURTHER_ATTEMPTS, read_attempts=M.read_attempts,
                                  attempt_summary=M.attempt_summary, E01N_CELLS=self.e01n, ATTEMPTS=M.ATTEMPTS)
        self.tag = "E-R2_s0"
        ad = self.cells / self.tag / "adapter"
        ad.mkdir(parents=True)
        (ad / "adapter_model.safetensors").write_bytes(b"weights")
        self.opened = {"adapters_sha256": {self.tag: sha(b"weights")}}

    def tearDown(self):
        self.tmp.cleanup()

    def fail_eval(self, n):
        for _ in range(n):
            M.attempt_start(self.cells, "eval", self.tag)

    def test_three_failed_attempts_allow_missing(self):
        self.fail_eval(3)
        out = A.missing_attempts(self.ne, self.cells, [self.tag], {self.tag: "no eval"}, self.opened)
        self.assertEqual(out[self.tag]["failed"], 3)

    def test_two_failed_attempts_refused(self):
        self.fail_eval(2)
        with self.assertRaises(SystemExit):
            A.missing_attempts(self.ne, self.cells, [self.tag], {self.tag: "no eval"}, self.opened)

    def test_existing_eval_failing_integrity_without_ok_event(self):
        (self.cells / f"eval_sealed_{self.tag}.json").write_text("{}", encoding="utf-8")
        out = A.missing_attempts(self.ne, self.cells, [self.tag], {self.tag: "raw sha mismatch"}, self.opened)
        self.assertTrue(out[self.tag]["eval_record_exists"])

    def test_adapter_mismatch_needs_no_attempts(self):
        (self.cells / self.tag / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
        out = A.missing_attempts(self.ne, self.cells, [self.tag], {self.tag: "no eval"}, self.opened)
        self.assertTrue(out[self.tag]["adapter_mismatch"])

    def test_control_adapter_path_is_e01n(self):
        tag = "N-CFSD_s1"
        ad = self.e01n / tag / "adapter"
        ad.mkdir(parents=True)
        (ad / "adapter_model.safetensors").write_bytes(b"ctl")
        self.assertFalse(A.adapter_mismatch(self.ne, self.cells, tag, {"adapters_sha256": {tag: sha(b"ctl")}}))
        self.assertTrue(A.adapter_mismatch(self.ne, self.cells, tag, {"adapters_sha256": {tag: sha(b"x")}}))

    def test_torn_line_is_tolerated_and_isolated(self):
        M.attempt_start(self.cells, "eval", self.tag)
        p = M.attempts_path(self.cells)
        with open(p, "ab") as f:
            f.write(b'{"event": "start", "attem')  # torn by a crash: no closing brace, no newline
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(M.read_attempts(self.cells)), 1)
            M.attempt_start(self.cells, "eval", self.tag)   # appended on its own line after the torn one
            ev = M.read_attempts(self.cells)
        self.assertEqual(len(ev), 2)

    def test_code_records_must_be_complete(self):
        commit = {"a.py": "1", "b.py": "2"}
        self.assertEqual(A.code_record_problems({"a.py": "1", "b.py": "2"}, commit, "x"), [])
        self.assertTrue(A.code_record_problems({"a.py": "1"}, commit, "x"))
        self.assertTrue(A.code_record_problems({"a.py": "1", "b.py": "3"}, commit, "x"))
        self.assertTrue(A.code_record_problems(None, commit, "x"))


if __name__ == "__main__":
    unittest.main()
