"""Isolated meaningful guards; never call the runner main or mutate a repository."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent
FREEZE = LOGS / "BELIEF_V4_R2_RESUMPTION_SOURCE_FREEZE.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenResumptionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        freeze = json.loads(FREEZE.read_bytes())
        source = LOGS / "resume_belief_v4_r2_frozen_stage.py"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == freeze["runner_source"]["sha256"]
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == freeze["tests_source"]["sha256"]
        for row in freeze["frozen_inputs"]:
            assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        cls.runner = load("test_frozen_resumption", source)
        cls.base = cls.runner.import_frozen(LOGS / "run_provisional_archival_checkpoint_v2.py",
                                           cls.runner.BASE_SHA, "test_frozen_df03")
        cls.config = json.loads((LOGS / "BELIEF_V4_R2_FROZEN_WHITESPACE_EXCEPTION.json").read_bytes())
        cls.diagnostics = (LOGS / "BELIEF_V4_R2_RESUMPTION_FIXTURES/exact_diff_check_stdout.txt").read_bytes()
        cls.source = (LOGS / "BELIEF_V4_R2_RESUMPTION_FIXTURES/exact_official_test_13990.source").read_bytes()

    def result(self, stdout=None, code=2, stderr=b""):
        return {"returncode": code, "stdout": self.diagnostics if stdout is None else stdout,
                "stderr": stderr}

    def test_exact_five_actual_captured_diagnostics_preserve_nonzero(self):
        out = self.runner.validate_diagnostics(self.result(), self.config)
        self.assertEqual(out["actual_exit_code"], 2)
        self.assertEqual(out["warnings"], 5)
        self.assertEqual(out["strict_diff_check"], "NONZERO_EXIT_2_NOT_AN_UNQUALIFIED_PASS")
        self.assertEqual(hashlib.sha256(self.source).hexdigest(), self.runner.TEST_SHA)
        self.assertEqual(len(self.source), 20749)
        self.assertEqual(self.source.splitlines(keepends=True)[211], base64.b64decode(self.runner.LINE_B64))

    def test_sixth_warning_rejected(self):
        extra = b"datasets/rineya_high_quality_factory_v1/other.py:212: trailing whitespace.\n+x \n"
        with self.assertRaises(ValueError):
            self.runner.validate_diagnostics(self.result(self.diagnostics + extra), self.config)

    def test_wrong_warning_line_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.validate_diagnostics(self.result(self.diagnostics.replace(b":212:", b":213:", 1)), self.config)

    def test_wrong_type_or_payload_rejected(self):
        for changed in (self.diagnostics.replace(b"trailing whitespace.", b"new blank line at EOF.", 1),
                        self.diagnostics.replace(b"+        for path", b"+       for path", 1)):
            with self.assertRaises(ValueError):
                self.runner.validate_diagnostics(self.result(changed), self.config)

    def test_duplicate_warning_rejected(self):
        lines = self.diagnostics.splitlines(keepends=True)
        altered = b"".join(lines[:2] + lines[:2] + lines[4:])
        with self.assertRaises(ValueError):
            self.runner.validate_diagnostics(self.result(altered), self.config)

    def test_missing_warning_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.validate_diagnostics(self.result(b"".join(self.diagnostics.splitlines(keepends=True)[2:])), self.config)

    def test_wrong_path_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.validate_diagnostics(self.result(self.diagnostics.replace(b"tests/oracle_belief", b"tests/other_belief", 1)), self.config)

    def test_untrusted_or_success_diagnostics_rejected(self):
        for altered in (self.result(code=0), self.result(stderr=b"warning: untrusted\n"),
                        self.result(stdout=self.diagnostics.decode()), self.result(code=True),
                        {**self.result(), "trusted": True}):
            with self.assertRaises(ValueError):
                self.runner.validate_diagnostics(altered, self.config)

    def test_config_hash_line_path_type_or_missing_rejected(self):
        variants = []
        for key, value in (("source_sha256", "0" * 64), ("line_number", 211), ("line_number", True),
                           ("paths", list(self.runner.WARNING_PATHS) + ["outside"]),
                           ("required_exit_code", 0), ("grants_data_approval", 0)):
            bad = copy.deepcopy(self.config);bad[key] = value;variants.append(bad)
        bad = copy.deepcopy(self.config);del bad["source_sha256"];variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                self.runner.validate_exception(bad)

    def warning_fixture(self, root):
        factory, worktree = root / "factory", root / "worktree"
        checked = {}
        for name in self.runner.WARNING_PATHS:
            for p in (factory / name, worktree / self.runner.PREFIX / name):
                p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(self.source)
            checked[name] = self.base.proof(factory / name, False)
        index = {name: entry["git_blob"] for name, entry in checked.items()}
        return factory, worktree, checked, index

    def test_actual_five_source_target_and_index_exact_bytes_pass(self):
        with tempfile.TemporaryDirectory(prefix="rineya-resumption-isolated-") as folder:
            factory, worktree, checked, index = self.warning_fixture(Path(folder))
            self.assertEqual(len(self.runner.verify_warning_sources(self.base, factory, worktree,
                                         checked, index, self.config)), 5)

    def test_changed_actual_source_and_target_rejected(self):
        for target in (False, True):
            with tempfile.TemporaryDirectory(prefix="rineya-resumption-isolated-") as folder:
                factory, worktree, checked, index = self.warning_fixture(Path(folder))
                name = self.runner.WARNING_PATHS[0]
                p = worktree / self.runner.PREFIX / name if target else factory / name
                p.write_bytes(self.source[:-1] + b"x")
                with self.assertRaises(ValueError):
                    self.runner.verify_warning_sources(self.base, factory, worktree, checked, index, self.config)

    def test_changed_index_or_inventory_source_hash_rejected(self):
        with tempfile.TemporaryDirectory(prefix="rineya-resumption-isolated-") as folder:
            factory, worktree, checked, index = self.warning_fixture(Path(folder))
            name = self.runner.WARNING_PATHS[0]
            bad_index = dict(index);bad_index[name] = "0" * 40
            with self.assertRaises(ValueError):
                self.runner.verify_warning_sources(self.base, factory, worktree, checked, bad_index, self.config)
            bad = copy.deepcopy(checked);bad[name]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                self.runner.verify_warning_sources(self.base, factory, worktree, bad, index, self.config)

    def index_fixture(self):
        checked = {"new.json": {"git_blob": "a" * 40}, "old.json": {"git_blob": "b" * 40}}
        parent = {"tools/tool.py": ("100755", "c" * 40), self.runner.PREFIX + "/old.json": ("100644", "b" * 40)}
        current = {"tools/tool.py": ("100755", "c" * 40), self.runner.PREFIX + "/old.json": ("100644", "b" * 40),
                   self.runner.PREFIX + "/new.json": ("100644", "a" * 40)}
        return checked, parent, current

    def test_exact_full_index_and_parent_outside_scope_pass(self):
        checked, parent, current = self.index_fixture()
        self.assertEqual(self.runner.validate_whole_index(current, parent, checked), current)

    def test_changed_missing_extra_outside_or_mode_index_rejected(self):
        checked, parent, current = self.index_fixture()
        variants = []
        bad = dict(current);bad["tools/tool.py"] = ("100755", "d" * 40);variants.append(bad)
        bad = dict(current);bad["outside"] = ("100644", "d" * 40);variants.append(bad)
        bad = dict(current);del bad[self.runner.PREFIX + "/new.json"];variants.append(bad)
        bad = dict(current);bad[self.runner.PREFIX + "/new.json"] = ("100755", "a" * 40);variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                self.runner.validate_whole_index(bad, parent, checked)

    def test_exact_changes_pass_but_unstaged_untracked_ignored_deleted_duplicate_rejected(self):
        checked, _, _ = self.index_fixture();previous = {"old.json": "b" * 40}
        staged = [self.runner.PREFIX + "/new.json"]
        self.assertEqual(self.runner.validate_changes(staged, [], [], [], [], checked, previous), staged)
        for changes in ((staged + ["tools/x"], [], [], [], []), (staged, ["x"], [], [], []),
                        (staged, [], ["x"], [], []), (staged, [], [], ["x"], []),
                        (staged, [], [], [], ["x"]), (staged * 2, [], [], [], []), ([], [], [], [], [])):
            with self.assertRaises(ValueError):
                self.runner.validate_changes(*changes, checked, previous)

    def test_index_conflict_duplicate_and_parent_submodule_rejected(self):
        for raw in (b"100644 " + b"a" * 40 + b" 1\tx\0",
                    (b"100644 " + b"a" * 40 + b" 0\tx\0") * 2):
            with self.assertRaises(ValueError):
                self.runner.parse_index(raw)
        with self.assertRaises(ValueError):
            self.runner.parse_tree(b"160000 commit " + b"a" * 40 + b"\tx\0")

    def test_preserved_failure_scope_and_no_commit_required(self):
        failed = json.loads((LOGS / "belief-v4-validated-code-candidate-r2_STAGE.json").read_bytes())
        self.runner.validate_failed_receipt(failed)
        for key, value in (("spec_sha256", "0" * 64), ("expected_parent_sha", "0" * 40),
                           ("error", "different"), ("commit_sha", "a" * 40)):
            bad = dict(failed);bad[key] = value
            with self.assertRaises(ValueError):
                self.runner.validate_failed_receipt(bad)

    def test_spec_exact_counts_and_parent_required(self):
        spec = json.loads((LOGS / "belief-v4-validated-code-candidate-r2_SPEC.json").read_bytes())
        self.runner.validate_spec(spec)
        for key, value in (("expected_parent_sha", "0" * 40), ("raw_lossless_required_count", 24),
                           ("inventory_unique_git_blob_groups", 2152)):
            bad = dict(spec);bad[key] = value
            with self.assertRaises(ValueError):
                self.runner.validate_spec(bad)

    def test_bounded_capture_real_exit_and_metadata_cap(self):
        result = self.runner.bounded_process([sys.executable, "-c", "import sys;sys.stdout.write('bounded');sys.exit(2)"], LOGS)
        self.assertEqual(result, {"returncode": 2, "stdout": b"bounded", "stderr": b""})
        with self.assertRaises(ValueError):
            self.runner.bounded_process([sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'x' * (3*1024*1024))"], LOGS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
