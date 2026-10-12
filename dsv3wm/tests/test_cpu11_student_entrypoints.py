"""CPU11 CLI source/AST parity and pre-load rejection; no native model calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from dsv3wm.native import EXPECTED_CORE_SHA, sha256
from dsv3wm.recovery_student import StudentConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
IMPORT = "from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load"
NAMES = ("train_recovery_student", "evaluate_recovery_student", "evaluate_recovery_student_guards")
ORIGINAL_SHA = {
    "train_recovery_student": "4fc02324d1965bd03f6eb7442028d08fdbb87e8e900d8718c3984e8e98e0b785",
    "evaluate_recovery_student": "60ba9bed75d6225b2675925b63460a177bbfdca57ee6957141edfde0ddf6a363",
    "evaluate_recovery_student_guards": "11ac658194bc919ce423c265b8ebd0a071ffc1b036a2efdfd9a9c04c8297d58f",
}


def entrypoint(name):
    path = SCRIPTS / (name + "_cpu11.py")
    spec = importlib.util.spec_from_file_location("_cpu11_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def permitted_changes(name):
    guards = name.endswith("_guards")
    train = name.startswith("train_")
    load_import = ("    from dsv3wm.native import runtime, load, sha256, verify_freeze, boundary" if train else
                   "    from dsv3wm.native import load, runtime, sha256, verify_freeze")
    replacement = ("    from dsv3wm.native import runtime, sha256, verify_freeze, boundary" if train else
                   "    from dsv3wm.native import runtime, sha256, verify_freeze") + "\n    " + IMPORT
    native_tuple = '"recovery_planning", "native", "provenance"' if train else '"proposer", "native", "provenance"'
    new_tuple = native_tuple.replace('"native",', '"native", "native_cpu11",')
    if guards:
        source_change = ('    return [Path(__file__), ROOT / "scripts/freeze_recovery_student_guards.py",',
                         '    return [Path(__file__), ROOT / "scripts/evaluate_recovery_student_guards.py", ROOT / "scripts/freeze_recovery_student_guards.py",')
        record_start = '    record = {"schema": SCHEMA, "status": "started",'
        record_change = ('    finally:\n        behavior = generated["behavior"]',
                         '    finally:\n        record["admission_check"] = last_admission_check()\n        behavior = generated["behavior"]')
    else:
        source_change = ('REPO / "bwm/code/scripts/planbench_qwen.py"]',
                         f'REPO / "bwm/code/scripts/planbench_qwen.py", REPO / "dsv3wm/scripts/{name}.py"]')
        record_start = '    record = {"status": "started",'
        record_change = ('    finally:\n        record["duration_s"]',
                         '    finally:\n        record["admission_check"] = last_admission_check()\n        record["duration_s"]')
    return [(load_import, replacement), (native_tuple, new_tuple), source_change,
            (record_start, record_start.replace('record = {', 'record = {"resource_admission_profile": admission_profile(), ')),
            ('            model, tokenizer, lm = load(cfg)\n',
             '            model, tokenizer, lm = load(cfg)\n            record["admission_check"] = last_admission_check()\n'), record_change]


class StudentCPU11EntrypointTests(unittest.TestCase):
    def test_exact_copy_allows_only_loader_source_pins_and_admission_metadata(self):
        for name in NAMES:
            with self.subTest(name=name):
                original = (SCRIPTS / (name + ".py")).read_text(encoding="utf-8")
                expected = original
                for before, after in permitted_changes(name):
                    self.assertEqual(expected.count(before), 1)
                    expected = expected.replace(before, after)
                actual = (SCRIPTS / (name + "_cpu11.py")).read_text(encoding="utf-8")
                self.assertEqual(actual, expected)
                self.assertEqual(ast.dump(ast.parse(actual)), ast.dump(ast.parse(expected)))

    def test_original_registered_entrypoint_bytes_remain_unchanged(self):
        for name, expected in ORIGINAL_SHA.items():
            self.assertEqual(hashlib.sha256((SCRIPTS / (name + ".py")).read_bytes()).hexdigest(), expected)

    def test_algorithm_functions_and_nested_losses_and_generators_are_ast_identical(self):
        for name in NAMES:
            with self.subTest(name=name):
                old = ast.parse((SCRIPTS / (name + ".py")).read_text(encoding="utf-8"))
                new = ast.parse((SCRIPTS / (name + "_cpu11.py")).read_text(encoding="utf-8"))
                def algorithms(tree):
                    return {node.name: ast.dump(node) for node in ast.walk(tree)
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in {"main", "source_paths"}}
                self.assertEqual(algorithms(old), algorithms(new))
                if name.startswith("train_"):
                    self.assertIn("loss_pair", algorithms(new))
                elif name.endswith("_guards"):
                    for function in ("generate_guards", "score_guards", "paired_guards", "TaggedPredictor", "behavior_generator", "count_forward"):
                        self.assertIn(function, algorithms(new))

    def test_guard_source_closure_retains_original_cli_and_adds_cpu11_loader(self):
        paths = entrypoint("evaluate_recovery_student_guards").source_paths()
        self.assertIn(SCRIPTS / "evaluate_recovery_student_guards.py", paths)
        self.assertIn(ROOT / "src/dsv3wm/native.py", paths)
        self.assertIn(ROOT / "src/dsv3wm/native_cpu11.py", paths)
        self.assertIn(SCRIPTS / "evaluate_recovery_student_guards_cpu11.py", paths)

    def test_failed_verified_corpus_refuses_before_runtime_or_cpu11_load(self):
        script = entrypoint("train_recovery_student")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text('{"arm":"A3"}', encoding="utf-8")
            (directory / "manifest.json").write_text('{"output_sha256":{}}', encoding="utf-8")
            (directory / "tasks.jsonl").write_text('{"split":"train"}\n', encoding="utf-8")
            argv = ["train_recovery_student_cpu11.py", "--config", str(directory / "config.json"),
                    "--train", str(directory / "train.jsonl"), "--tasks", str(directory / "tasks.jsonl"),
                    "--teacher-dir", str(directory), "--warm-dir", str(directory / "warm"),
                    "--qualification", str(directory / "qualification.json"), "--freeze", str(directory / "freeze.json"),
                    "--out", str(directory / "output")]
            with patch.object(sys, "argv", argv), patch("dsv3wm.native.verify_freeze", return_value="synthetic-freeze"), \
                 patch("dsv3wm.provenance.verify_bound_inputs"), patch("dsv3wm.data.load_records", return_value=[{"domain": "register"}]), \
                 patch("dsv3wm.recovery_student.load_teacher_corpus", side_effect=ValueError("unverified corpus")), \
                 patch("dsv3wm.native_cpu11.load") as load, patch("dsv3wm.native.runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "unverified corpus"):
                    script.main()
                load.assert_not_called(); runtime.assert_not_called()
            self.assertFalse((directory / "output").exists())

    def test_readout_cannot_alias_trained_control_as_A3_before_model_loading(self):
        script = entrypoint("evaluate_recovery_student")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); adapter = directory / "adapter"; adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"; weights.write_bytes(b"explicit synthetic test artifact")
            config = directory / "config.json"; config.write_text(json.dumps(StudentConfig(arm="A3").dict()))
            actual = StudentConfig(arm="A3", plan_weight=0, control_extra_one_step=True)
            run = directory / "run.json"
            run.write_text(json.dumps({"status": "student-pilot-trained", "config": actual.dict(), "optimizer_steps": 32,
                "core_sha256": EXPECTED_CORE_SHA, "adapter_sha256": sha256(weights), "adapter_config_sha256": "synthetic-config-hash"}))
            argv = ["evaluate_recovery_student_cpu11.py", "--config", str(config), "--adapter", str(adapter),
                    "--run", str(run), "--tasks", str(directory / "tasks.jsonl"), "--freeze", str(directory / "freeze.json"),
                    "--out", str(directory / "out.json")]
            with patch.object(sys, "argv", argv), patch("dsv3wm.native.verify_freeze", return_value="synthetic-freeze"), \
                 patch("dsv3wm.provenance.verify_bound_inputs"), patch("dsv3wm.native_cpu11.load") as load, \
                 patch("dsv3wm.native.runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "full trained"):
                    script.main()
                load.assert_not_called(); runtime.assert_not_called()
            self.assertFalse((directory / "out.json").exists())

    def test_guard_warm_mode_requires_source_train_before_model_loading(self):
        script = entrypoint("evaluate_recovery_student_guards")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); adapter = directory / "adapter"; adapter.mkdir()
            config = directory / "config.json"; config.write_text(json.dumps(StudentConfig(arm="A3").dict()))
            run = directory / "run.json"; run.write_text('{}')
            argv = ["evaluate_recovery_student_guards_cpu11.py", "--config", str(config), "--adapter", str(adapter),
                    "--run", str(run), "--data", str(directory / "dev.jsonl"), "--fixtures", str(directory / "fixtures.json"),
                    "--freeze", str(directory / "freeze.json"), "--out", str(directory / "out.json"), "--warm-baseline"]
            with patch.object(sys, "argv", argv), patch("dsv3wm.native.verify_freeze", return_value="synthetic-freeze"), \
                 patch("dsv3wm.provenance.verify_bound_inputs"), patch("dsv3wm.native_cpu11.load") as load, \
                 patch("dsv3wm.native.runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "explicit source --train"):
                    script.main()
                load.assert_not_called(); runtime.assert_not_called()
            self.assertFalse((directory / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
