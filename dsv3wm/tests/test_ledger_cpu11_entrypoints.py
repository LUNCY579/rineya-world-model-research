"""Ledger CPU11 copy/source parity and refusal paths; no native model calls."""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))
from dsv3wm.config import Config
from dsv3wm.native import EXPECTED_CORE_SHA, sha256, verify_freeze
from dsv3wm.provenance import verify_bound_inputs
import train_ledger_recovery_cpu11 as train
import evaluate_ledger_recovery_cpu11 as evaluate
import evaluate_rcwm_retention_v2_cpu11 as rp
import freeze_ledger_cpu11 as freezer

ORIGINAL_SHA = {
    "train_ledger_recovery": "425d972a7632fa287f72aa580228fc684fc0c5fbb59c1fb88780e82998d73ee1",
    "evaluate_ledger_recovery": "7a840e9d011822be4092e434cc175d1f2a94bcaa76d1b9dabf73758da848c20a",
    "evaluate_rcwm_retention_v2": "5aa7526a31d39ff30c331c08090265a93721510fcadb97160b8a384cdabe7455",
}
LEDGER_ADDITIONS = '''    paths += [PACKAGE / "scripts/train_ledger_recovery.py",
              PACKAGE / "scripts/evaluate_ledger_recovery_cpu11.py",
              PACKAGE / "scripts/evaluate_rcwm_retention_v2_cpu11.py",
              PACKAGE / "scripts/freeze_ledger_cpu11.py", PACKAGE / "src/dsv3wm/native_cpu11.py",
              PACKAGE / "prereg/RECOVERY_RESOURCE_CLARIFICATION_3.md",
              PACKAGE / "prereg/RECOVERY_RESOURCE_CLARIFICATION_4.md",
              PACKAGE / "prereg/LEDGER_RESOURCE_CLARIFICATION_CPU11_V1.md",
              PACKAGE / "prereg/EXECUTION_SOURCE_CLOSURE_V1.md",
              PACKAGE / "scripts/freeze_execution_dependencies.py", PACKAGE / "src/dsv3wm/__init__.py",
              REPO / "bwm/code/bwmn/__init__.py", REPO / "bwm/code/bwmn/blocks.py",
              REPO / "bwm/code/bwmn/flops.py", REPO / "bwm/code/bwmn/planbench_extract.py"]
'''
RP_ADDITIONS = '''    return [*builder.source_paths(), Path(__file__), ROOT / "src/dsv3wm/native_cpu11.py",
            ROOT / "scripts/freeze_ledger_cpu11.py", ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_3.md",
            ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_4.md",
            ROOT / "prereg/LEDGER_RESOURCE_CLARIFICATION_CPU11_V1.md"]
'''


def approved_changes(name):
    if name == "evaluate_rcwm_retention_v2":
        return [
            ('    return builder.source_paths()\n', RP_ADDITIONS),
            ('    from dsv3wm.native import EXPECTED_CORE_SHA, boundary, load, runtime, sha256, verify_freeze\n',
             '    from dsv3wm.native import EXPECTED_CORE_SHA, boundary, runtime, sha256, verify_freeze\n'
             '    from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load\n'),
            ('    result = {"status": "started",', '    result = {"resource_admission_profile": admission_profile(), "status": "started",'),
            ('            model, tokenizer, lm = load(cfg, args.adapter)\n',
             '            model, tokenizer, lm = load(cfg, args.adapter)\n            result["admission_check"] = last_admission_check()\n'),
            ('    finally:\n        result["duration_s"]',
             '    finally:\n        result["admission_check"] = last_admission_check()\n        result["duration_s"]')]
    if name == "train_ledger_recovery":
        edits = [('from dsv3wm.native import sha256, verify_freeze\n',
                  'from dsv3wm.native import sha256, verify_freeze\n'
                  'from dsv3wm.native_cpu11 import admission_profile, last_admission_check\n'),
                 ('    return paths\n', LEDGER_ADDITIONS + '    return paths\n')]
        final = '    finally:\n        result["duration_s"]'
    else:
        edits = [('from train_ledger_recovery import (', 'from train_ledger_recovery_cpu11 import ('),
                 ('from dsv3wm.native import sha256\n', 'from dsv3wm.native import sha256\n'
                  'from dsv3wm.native_cpu11 import admission_profile, last_admission_check\n')]
        final = '    finally:\n        if predictor is not None:'
    return [*edits,
            ('        from dsv3wm.native import runtime, load, boundary, EXPECTED_CORE_SHA\n',
             '        from dsv3wm.native import runtime, boundary, EXPECTED_CORE_SHA\n'
             '        from dsv3wm.native_cpu11 import load\n'),
            ('    result = {"status": "started",', '    result = {"resource_admission_profile": admission_profile(), "status": "started",'),
            ('            model, tok, lm = load(cfg)\n',
             '            model, tok, lm = load(cfg)\n            result["admission_check"] = last_admission_check()\n'),
            (final, final.replace('    finally:\n', '    finally:\n        result["admission_check"] = last_admission_check()\n'))]


def write_freeze(directory, instrument="tick-compatible-ledger-development-v1"):
    member = directory / "public-input.txt"
    member.write_bytes(b"synthetic non-label test input")
    source = directory / "original.json"
    record = {"instrument": instrument, "confirmation_access": False,
              "qualification_freeze_sha256": "a" * 64, "registration": {"gate": 10, "seed": 0},
              "files": {str(member.resolve()).replace("\\", "/"): sha256(member)}}
    source.write_text(json.dumps(record), encoding="utf-8")
    return source, member, record


class LedgerCPU11EntrypointTests(unittest.TestCase):
    def test_original_registered_entrypoint_bytes_are_unchanged(self):
        for name, expected in ORIGINAL_SHA.items():
            self.assertEqual(hashlib.sha256((SCRIPTS / (name + ".py")).read_bytes()).hexdigest(), expected)

    def test_copies_change_only_explicit_loader_source_pins_and_admission_metadata(self):
        for name in ORIGINAL_SHA:
            with self.subTest(name=name):
                expected = (SCRIPTS / (name + ".py")).read_text(encoding="utf-8")
                for before, after in approved_changes(name):
                    self.assertEqual(expected.count(before), 1)
                    expected = expected.replace(before, after)
                actual = (SCRIPTS / (name + "_cpu11.py")).read_text(encoding="utf-8")
                self.assertEqual(actual, expected)
                self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))

    def test_algorithm_classes_functions_and_nested_counters_are_ast_identical(self):
        for name in ORIGINAL_SHA:
            with self.subTest(name=name):
                def algorithms(path):
                    return {node.name: ast.dump(node) for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                            and node.name not in {"main", "required_sources", "source_paths"}}
                old = algorithms(SCRIPTS / (name + ".py"))
                new = algorithms(SCRIPTS / (name + "_cpu11.py"))
                self.assertEqual(old, new)
                if name == "evaluate_ledger_recovery":
                    self.assertTrue({"LedgerNativePredictor", "generate_dynamics", "score_dynamics", "count_forward",
                                     "validate_child_lineage"} <= set(new))

    def test_new_source_closure_binds_originals_and_shared_dependencies(self):
        paths = set(freezer.source_additions())
        self.assertTrue(set(train.required_sources()) <= paths)
        self.assertTrue(set(rp.source_paths()) <= paths)
        required = [ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/native_cpu11.py",
                    ROOT / "src/dsv3wm/__init__.py", ROOT / "prereg/LEDGER_RESOURCE_CLARIFICATION_CPU11_V1.md",
                    ROOT.parent / "bwm/code/bwmn/__init__.py", ROOT.parent / "bwm/code/bwmn/blocks.py",
                    ROOT.parent / "bwm/code/bwmn/flops.py", ROOT.parent / "bwm/code/bwmn/planbench_extract.py",
                    *[SCRIPTS / (name + suffix + ".py") for name in ORIGINAL_SHA for suffix in ("", "_cpu11")]]
        self.assertTrue(set(required) <= paths)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_extension_preserves_original_pins_metadata_and_freeze_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, member, original = write_freeze(directory)
            source_bytes = source.read_bytes()
            with patch.dict("os.environ", {"BWM_CPU_BATCH_TOKENS": "8000", "BWM_RAM_FLOOR_GB": "8"}):
                extended = freezer.extend(source)
            self.assertEqual(source.read_bytes(), source_bytes)
            for key in original:
                if key != "files":
                    self.assertEqual(extended[key], original[key])
            self.assertTrue(original["files"].items() <= extended["files"].items())
            self.assertEqual(extended["files"][str(source.resolve()).replace("\\", "/")], sha256(source))
            amendment = extended["ledger_cpu11_admission_amendment"]
            self.assertEqual(amendment["original_freeze_sha256"], sha256(source))
            self.assertEqual(amendment["profile"]["load_free_host_ram_floor_gib"], 11)
            output = directory / "new.json"
            output.write_text(json.dumps(extended), encoding="utf-8")
            verify_freeze(output)
            verify_bound_inputs(output, [source, member, *train.required_sources(), *rp.source_paths()])

    def test_extension_rejects_drift_wrong_scope_confirmation_and_reextension(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, member, original = write_freeze(directory)
            member.write_bytes(b"tampered input")
            with self.assertRaisesRegex(RuntimeError, "freeze drift"):
                freezer.extend(source)
            member.write_bytes(b"synthetic non-label test input")
            for change in ({"instrument": "rcwm-RP-retention-v1-B0"}, {"instrument": "recovery-H3"},
                           {"confirmation_access": True}, {"ledger_cpu11_admission_amendment": {}}):
                source.write_text(json.dumps({**original, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    freezer.extend(source)

    def test_extension_cli_refuses_to_overwrite_before_inspecting_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.json"
            output.write_bytes(b"immutable prior output")
            argv = ["freeze_ledger_cpu11.py", "--freeze", str(Path(temporary) / "missing.json"), "--out", str(output)]
            with patch.object(sys, "argv", argv), patch.object(freezer, "extend") as extend:
                with self.assertRaises(FileExistsError):
                    freezer.main()
                extend.assert_not_called()
            self.assertEqual(output.read_bytes(), b"immutable prior output")

    def test_old_source_freeze_cannot_admit_cpu11_source_before_qualification(self):
        from train_ledger_recovery import required_sources as original_sources
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, _, original = write_freeze(directory)
            original["files"].update({str(path.resolve()).replace("\\", "/"): sha256(path) for path in original_sources()})
            source.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input absent from freeze: .*train_ledger_recovery_cpu11"):
                train.bind_execution(source, [])

    def test_training_qualification_refusal_precedes_runtime_and_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / "config.json"
            config.write_text(json.dumps(train.LedgerConfig().dict()), encoding="utf-8")
            argv = ["train_ledger_recovery_cpu11.py", "--config", str(config), "--train", str(directory / "train.jsonl"),
                    "--qualification", str(directory / "B0.json"), "--freeze", str(directory / "freeze.json"),
                    "--out", str(directory / "out")]
            with patch.object(sys, "argv", argv), patch.object(train, "bind_execution", return_value="synthetic-freeze"), \
                 patch.object(train, "bind_qualification", side_effect=ValueError("unqualified B0")), \
                 patch("dsv3wm.native.runtime") as runtime, patch("dsv3wm.native_cpu11.load") as load:
                with self.assertRaisesRegex(ValueError, "unqualified B0"):
                    train.main()
                runtime.assert_not_called(); load.assert_not_called()
            self.assertFalse((directory / "out").exists())

    def test_b0_qualification_cannot_use_an_adapted_adapter_before_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / "config.json"
            config.write_text(json.dumps(train.LedgerConfig().dict()), encoding="utf-8")
            argv = ["evaluate_ledger_recovery_cpu11.py", "--config", str(config), "--data", str(directory / "dev.jsonl"),
                    "--adapter", str(directory / "adapted"), "--baseline-only", "--freeze", str(directory / "freeze.json"),
                    "--out", str(directory / "out.json")]
            with patch.object(sys, "argv", argv), patch.object(evaluate, "bind_execution", return_value="synthetic-freeze"), \
                 patch("dsv3wm.native.runtime") as runtime, patch("dsv3wm.native_cpu11.load") as load:
                with self.assertRaisesRegex(ValueError, "unchanged original T2"):
                    evaluate.main()
                runtime.assert_not_called(); load.assert_not_called()
            self.assertFalse((directory / "out.json").exists())

    def test_rp_refused_admission_is_preserved_without_generation_or_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / "config.json"
            config.write_text(json.dumps(Config(max_length=512, generation_tokens=64).dict()), encoding="utf-8")
            fixtures = directory / "fixtures.json"
            fixtures.write_text(json.dumps({"schema": "dsv3wm-rcwm-retention-v2", "claim_scope": "synthetic scope",
                "parent_counts": {"R": 12, "P": 12}, "graph_group_counts": {"R": 1, "P": 1}}), encoding="utf-8")
            adapter = directory / "adapter"; adapter.mkdir()
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                (adapter / name).write_bytes(b"synthetic artifact")
            contract = {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64}
            admission = {"physical_available_gib": 10.8, "commit_available_gib": 19.0, "passed": False, "status": "refused"}
            argv = ["evaluate_rcwm_retention_v2_cpu11.py", "--config", str(config), "--fixtures", str(fixtures),
                    "--adapter", str(adapter), "--freeze", str(directory / "freeze.json"), "--out", str(directory / "out.json")]
            with patch.object(sys, "argv", argv), patch("dsv3wm.native.verify_freeze", return_value="synthetic-freeze"), \
                 patch("dsv3wm.provenance.verify_bound_inputs"), patch.object(rp, "validate_document", return_value=([], contract)), \
                 patch("dsv3wm.native.runtime", return_value=nullcontext()), patch.dict(sys.modules, {"torch": ModuleType("torch")}), \
                 patch("dsv3wm.native_cpu11.load", side_effect=RuntimeError("CPU11 physical admission refused")) as load, \
                 patch("dsv3wm.native_cpu11.last_admission_check", return_value=admission), \
                 patch.object(rp, "generate_then_score") as generate:
                with self.assertRaisesRegex(RuntimeError, "admission refused"):
                    rp.main()
                load.assert_called_once(); generate.assert_not_called()
            result = json.loads((directory / "out.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["admission_check"], admission)
            self.assertEqual(result["rows"], [])
            self.assertIsNone(result["retention_pass"])
            self.assertEqual(result["resource_admission_profile"]["load_free_host_ram_floor_gib"], 11)

    def test_rp_original_strict_scoring_and_v2_lineage_contract_remain_same_objects(self):
        self.assertIs(rp.strict_state, rp.reference.strict_state)
        self.assertIs(rp.score_all, rp.reference.score_all)
        self.assertIs(rp.generate_then_score, rp.reference.generate_then_score)
        self.assertIs(rp.paired_changes, rp.reference.paired_changes)
        path = SCRIPTS / "freeze_rcwm_retention_adapted_v2.py"
        spec = importlib.util.spec_from_file_location("_ledger_cpu11_lineage_compatibility", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertTrue(set(module.evaluator.source_paths()) <= set(rp.source_paths()))
        hashes = {str(path): sha256(path) for path in rp.source_paths()}
        validated = module.lineage.validate_source_record({"source_sha256": hashes}, module.evaluator.source_paths())
        self.assertIn((SCRIPTS / "evaluate_rcwm_retention_v2_cpu11.py").resolve(), validated)


if __name__ == "__main__":
    unittest.main()
