"""Scope and integration safety tests; no real factory inventory is prepared."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
module_spec = importlib.util.spec_from_file_location("rgm_filter", Path(__file__).with_name("prepare_rgm_v1_checkpoint_frozen.py"))
wrapper = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(wrapper)


class ArchivalFilterTests(unittest.TestCase):
    def test_every_release_path_wins_over_v2_patterns(self):
        for path in ["data/RGM-001/releases/planning-gap-v2.jsonl", "data/RGM-001/releases/planning_gap_v2/source.py",
                     "reviews/releases/RGM-planning-v2.json", "tests/releases/test_rgm_planning_controls_v2.py"]:
            with self.subTest(path=path):
                self.assertIsNone(wrapper.defer_reason(path))

    def test_v1_evidence_and_new_shared_gate_sources_stay_included(self):
        for path in ["data/RGM-001/candidate_renderings/planning-gap-v1.jsonl", "data/RGM-001/quarantine/planning-gap-v1.jsonl",
                     "reviews/expert/planning-gap-v1.expert_comparison.preexecution_source_v2.py",
                     "audits/rgm_planning_gap_v1_adversarial_final_checks_v2.py",
                     "reviews/adjudication/planning-gap-v1.jsonl", "factory/planning_controls.py",
                     "factory/pipeline.py", "tests/test_planning_review_corrections.py",
                     "generators/relational/planning_gap.py", "oracles/rgm_planning_controls_independent.py",
                     "reviews/expert/planning-gap-v1-planning-gap-v2-comparison.json"]:
            with self.subTest(path=path):
                self.assertIsNone(wrapper.defer_reason(path))

    def test_public_v2_and_root_freeze_keep_priority(self):
        for path in ["schemas/RGM_PLANNING_CONTROL_PROPERTIES_V2.md", "schemas/planning-gap-v2.json",
                     "research/curricula/RGM-PLANNING-GAP-V2-PROSPECTIVE.md",
                     "research/curricula/RGM-PLANNING-GAP-V2-ROOT_FREEZE.json",
                     "research/research_gap_analysis/PLANNING_GAP_V2_PUBLIC_INPUT_AUDIT.py",
                     "schemas/RGM_PLANNING_V2_RENDERER_CONTROL_ENCODING_ADDENDUM.md",
                     "handoff/RGM_V1_SUPPLEMENTS_AND_V2_PREPARATION.json",
                     "manifests/planning-gap-v2-ROOT_FREEZE.json", "manifests/planning-gap-v2-root-freeze.json"]:
            with self.subTest(path=path):
                self.assertIsNone(wrapper.defer_reason(path))

    def test_intended_formal_v2_paths_are_deferred(self):
        for path in ["oracles/rgm_planning_controls_v2_independent.py", "tests/oracle_rgm_planning_v2_test.py",
                     "tests/oracle_rgm_planning_controls_v2_test.py", "tests/nested/test_rgm_planning_controls_v2_actual.py",
                     "reviews/oracle/RGM-planning-v2-first-execution-20261010T051814Z.json",
                     "reviews/oracle/RGM-v2/audit.json"]:
            with self.subTest(path=path):
                self.assertIsNotNone(wrapper.defer_reason(path))

    def test_future_artifacts_only_match_exact_wave_token(self):
        for path in ["generators/relational/planning_gap_v2.py", "data/RGM-001/candidate_renderings/planning-gap-v2.jsonl",
                     "reviews/adversarial/planning-gap-v2.jsonl", "manifests/planning-gap-v2_SOURCE.py",
                     "audits/planning_gap_v2_integrity.py"]:
            with self.subTest(path=path):
                self.assertIsNotNone(wrapper.defer_reason(path))
        for path in ["generators/relational/planning_gap_v20.py", "reviews/oracle/RGM-v20.json",
                     "reviews/oracle/RGM-planning-v2x.json", "reviews/oracle/RGM-planning-final-encoding-tests.json",
                     "tests/oracle_belief_controls_v2_test.py", "reviews/expert/planning-gap-v20.jsonl",
                     "reproducibility/source_blobs/" + "a" * 64 + ".source",
                     "backup_logs/planning-gap-v2_ARCHIVE_PREPARATION.json"]:
            with self.subTest(path=path):
                self.assertIsNone(wrapper.defer_reason(path))

    def test_factory_walk_hashes_deferred_actual_bytes_only_and_preserves_other_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files = {"oracles/rgm_planning_controls_v2_independent.py": b"unfinished\n",
                     "data/RGM-001/releases/planning-gap-v2.jsonl": b"immutable\n",
                     "schemas/RGM_PLANNING_CONTROL_PROPERTIES_V2.md": b"public\n",
                     "reviews/expert/planning-gap-v1.jsonl": b"review\n"}
            for relative, payload in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            outside = root / "other-project"
            path = outside / "oracles/rgm_planning_controls_v2_independent.py"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"outside\n")
            original = Path.rglob
            deferred = {}
            with patch.object(wrapper, "ROOT", root), wrapper.filter_factory_walk(deferred):
                walked = {item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()}
                self.assertNotIn("oracles/rgm_planning_controls_v2_independent.py", walked)
                self.assertTrue(set(files) - set(deferred) <= walked)
                self.assertIn(path, list(outside.rglob("*")))
            self.assertIs(Path.rglob, original)
            item = deferred["oracles/rgm_planning_controls_v2_independent.py"]
            self.assertEqual(item["sha256"], hashlib.sha256(b"unfinished\n").hexdigest())
            self.assertEqual(item["size"], len(b"unfinished\n"))

    def test_walk_restores_path_method_after_failure(self):
        original = Path.rglob
        with self.assertRaises(RuntimeError):
            with wrapper.filter_factory_walk({}):
                raise RuntimeError("test interruption")
        self.assertIs(Path.rglob, original)

    def test_freeze_signal_and_provisional_scope_required_before_builder(self):
        for arguments in [[], ["--freeze-scope", "--checkpoint-id", "x", "--preparer", "/root", "--runner", "/root", "--approval", "reviewed_release"]]:
            with self.subTest(arguments=arguments), patch.object(wrapper.inventory, "main") as builder:
                with self.assertRaises(ValueError):
                    wrapper.main(arguments)
                builder.assert_not_called()

    def test_fresh_id_preserves_existing_scope_and_never_calls_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "backup_logs").mkdir()
            existing = root / "backup_logs/x_SPEC.json"
            existing.write_bytes(b"prior frozen bytes\n")
            with patch.object(wrapper, "ROOT", root), patch.object(wrapper.inventory, "main") as builder:
                with self.assertRaises(ValueError):
                    wrapper.main(["--freeze-scope", "--checkpoint-id", "x", "--preparer", "/root", "--runner", "/root"])
                builder.assert_not_called()
                self.assertEqual(existing.read_bytes(), b"prior frozen bytes\n")

    def test_wrapper_integration_keeps_builder_args_checks_and_records_actual_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "backup_logs").mkdir()
            builder_path = root / "backup_logs/existing_builder.py"
            builder_path.write_bytes(b"unchanged existing builder\n")
            (root / "backup_logs/prepare_belief_checkpoint_frozen.py").write_bytes(b"unchanged original wrapper\n")
            for relative in ["oracles/rgm_planning_controls_v2_independent.py", "schemas/planning-gap-v2.md",
                             "data/RGM-001/releases/planning-gap-v2.jsonl", "reviews/adjudication/planning-gap-v1.jsonl"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
            calls = []

            def snapshot(path):
                payload = path.read_bytes()
                return {"path": path.relative_to(root).as_posix(), "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()}, payload

            def atomic(path, payload):
                path.write_text(json.dumps(payload))

            def builder_main():
                calls.append(list(sys.argv[1:]))
                entries = [snapshot(path)[0] for path in root.rglob("*") if path.is_file()]
                atomic(root / "backup_logs/synthetic_SPEC.json", {"files": entries, "validation_status": "passed"})

            fake = SimpleNamespace(__file__=str(builder_path), main=builder_main, atomic_json=atomic, snapshot=snapshot)
            arguments = ["--freeze-scope", "--checkpoint-id", "synthetic", "--preparer", "/root/backup_manager",
                         "--runner", "/root", "--validation-status", "passed"]
            original_walk, original_argv = Path.rglob, sys.argv
            with patch.object(wrapper, "ROOT", root), patch.object(wrapper, "inventory", fake):
                wrapper.main(arguments)
            self.assertIs(Path.rglob, original_walk)
            self.assertIs(sys.argv, original_argv)
            self.assertEqual(calls, [[argument for argument in arguments if argument != "--freeze-scope"]])
            spec = json.loads((root / "backup_logs/synthetic_SPEC.json").read_text())
            report = json.loads((root / "backup_logs/synthetic_FROZEN_SCOPE.json").read_text())
            included = {entry["path"] for entry in spec["files"]}
            self.assertNotIn("oracles/rgm_planning_controls_v2_independent.py", included)
            self.assertIn("schemas/planning-gap-v2.md", included)
            self.assertIn("data/RGM-001/releases/planning-gap-v2.jsonl", included)
            self.assertIn("reviews/adjudication/planning-gap-v1.jsonl", included)
            self.assertEqual(report["actual_author_execution"], "/root/backup_manager")
            self.assertEqual(report["actual_runner_execution"], "/root/backup_manager")
            self.assertEqual(report["checkpoint_runner_execution"], "/root")
            self.assertFalse(report["grants_data_approval"])
            self.assertEqual(spec["frozen_scope_report"], snapshot(root / "backup_logs/synthetic_FROZEN_SCOPE.json")[0])


if __name__ == "__main__":
    unittest.main()
