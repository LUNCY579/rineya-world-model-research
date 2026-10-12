"""Training-only CPU10.5 admission/algorithm parity with fake modules; no model."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dsv3wm import native, native_cpu11, native_cpu105_training as cpu105
from dsv3wm.recovery_student import StudentConfig
import freeze_recovery_student_cpu105_training as freezer

OLD_SHA = {"src/dsv3wm/native.py": "110f5cfd1ee57fc717703cc36c316954a45ec40d63870b316a397d0da4f9f3e2",
           "src/dsv3wm/native_cpu11.py": "075b7fdea2199a5f8f564022e48742b3b3064c58b8bec877ed7c0198d8c4f884",
           "scripts/train_recovery_student_cpu11.py": "99eb58e55f6f6b7cce71f797c98cc2898812f3798dc05903660b2b87fb013d4e"}
COPY_CHANGES = [
    ('    from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load\n',
     '    from dsv3wm.native_cpu105_training import admission_profile, last_admission_check, load, validate_training_config\n'),
    ('    cfg = StudentConfig.read(args.config)\n', '    cfg = StudentConfig.read(args.config)\n    validate_training_config(cfg)\n'),
    ('"native", "native_cpu11", "provenance"', '"native", "native_cpu11", "native_cpu105_training", "provenance"'),
    ('REPO / "dsv3wm/scripts/train_recovery_student.py"]',
     'REPO / "dsv3wm/scripts/train_recovery_student.py", REPO / "dsv3wm/scripts/train_recovery_student_cpu11.py",\n'
     '              REPO / "dsv3wm/scripts/freeze_recovery_student_cpu105_training.py", REPO / "dsv3wm/prereg/RECOVERY_RESOURCE_CLARIFICATION_5.md"]'),
    ('            model, tokenizer, lm = load(cfg)\n', '            model, tokenizer, lm = load(cfg, training_only=True)\n')]


def cfg(kind):
    return StudentConfig.read(ROOT / "configs" / cpu105.REGISTERED_CONFIGS[kind][0])


class CPU105TrainingTests(unittest.TestCase):
    def fixture(self, free=10.5, commit=10.5, *, changed=None, fp32=True):
        events = []
        torch = ModuleType("torch"); torch.float32 = object()
        torch.set_num_threads = lambda n: events.append(("threads", n))
        torch.set_float32_matmul_precision = lambda value: events.append(("precision", value))
        torch.use_deterministic_algorithms = lambda value, **kwargs: events.append(("determinism", value, kwargs))
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype=torch.float32 if fp32 else "fp16")]))
        tokenizer = object()
        lm = ModuleType("bwmn.lm")
        lm.MODELS = {"titan-sparse": Path("synthetic-core")}
        lm.host_ram_free_gb = lambda: (free, commit)
        lm.set_determinism = lambda seed: events.append(("seed", seed))
        def load_core(core):
            events.append(("core", core)); return model, tokenizer
        lm.load_core = load_core
        bwmn = ModuleType("bwmn"); bwmn.lm = lm
        expected = {"model.safetensors": native.EXPECTED_CORE_SHA, **native.EXPECTED_SIDECARS}
        def digest(path):
            if Path(path).name.startswith("recovery_student_"):
                return native.sha256(path)
            name = Path(path).name
            events.append(("hash", name))
            return "changed bytes" if name == changed else expected[name]
        return events, {"bwmn": bwmn, "bwmn.lm": lm, "torch": torch}, digest, (model, tokenizer, lm)

    def test_exact_registered_a4_c3_c4_only_and_full_config_type_checks(self):
        for kind in ("A4", "C3", "C4"):
            self.assertEqual(cpu105.validate_training_config(cfg(kind)), kind)
        for changed in (StudentConfig(arm="A3"), StudentConfig(arm="A4", seed=1), StudentConfig(arm="A4", threads=1),
                        StudentConfig(arm="A4", ce_chunk=4), StudentConfig(arm="A4", mtp_weight=.2),
                        StudentConfig(arm="A4", plan_weight=1), SimpleNamespace(**cfg("A4").dict())):
            with self.subTest(config=changed), self.assertRaises(ValueError):
                cpu105.validate_training_config(changed)
        with patch.object(cpu105, "sha256", return_value="changed config bytes"), self.assertRaisesRegex(ValueError, "configuration bytes"):
            cpu105.validate_training_config(cfg("A4"))

    def test_explicit_training_flag_and_no_inference_adapter_before_model_work(self):
        for kwargs in ({}, {"training_only": False}, {"training_only": 1}, {"training_only": True, "adapter": "adapter"},
                       {"training_only": True, "adapter": ""}):
            events, modules, digest, _ = self.fixture()
            with self.subTest(kwargs=kwargs), patch.dict(sys.modules, modules), \
                 patch.object(cpu105, "sha256", side_effect=digest), self.assertRaisesRegex(ValueError, "training_only=True"):
                cpu105.load(cfg("A4"), **kwargs)
            self.assertEqual(events, [])
        events, modules, digest, _ = self.fixture()
        with patch.dict(sys.modules, modules), patch.object(cpu105, "sha256", side_effect=digest), self.assertRaises(ValueError):
            cpu105.load(StudentConfig(arm="A3"), training_only=True)
        self.assertEqual(events, [])

    def test_both_memory_floors_and_refusal_record_before_core_work(self):
        for physical, commit in ((10.499999, 20.), (11., 10.499999), (10.49, 10.49)):
            events, modules, digest, _ = self.fixture(physical, commit)
            with self.subTest(physical=physical, commit=commit), patch.dict(sys.modules, modules), \
                 patch.object(cpu105, "sha256", side_effect=digest), self.assertRaisesRegex(RuntimeError, "10.5 GiB"):
                cpu105.load(cfg("A4"), training_only=True)
            self.assertEqual(events, [])
            check = cpu105.last_admission_check()
            self.assertEqual(check["physical_available_gib"], physical)
            self.assertEqual(check["commit_available_gib"], commit)
            self.assertEqual(check["status"], "refused"); self.assertFalse(check["passed"])
        events, modules, digest, expected = self.fixture()
        with patch.dict(sys.modules, modules), patch.object(cpu105, "sha256", side_effect=digest):
            self.assertEqual(cpu105.load(cfg("C3"), training_only=True), expected)
        check = cpu105.last_admission_check()
        self.assertEqual(check["training_kind"], "C3"); self.assertTrue(check["passed"])
        check["passed"] = False
        self.assertTrue(cpu105.last_admission_check()["passed"])

    def test_cpu11_floor_remains_and_accepted_core_call_sequence_is_identical(self):
        events, modules, digest, _ = self.fixture(10.8694, 18.301304)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "11 GiB"):
            native_cpu11.load(cfg("A4"))
        self.assertEqual(events, [])
        calls = []
        for loader in (native_cpu11, cpu105):
            events, modules, digest, _ = self.fixture(12., 20.)
            with patch.dict(sys.modules, modules), patch.object(loader, "sha256", side_effect=digest):
                loader.load(cfg("A4"), **({"training_only": True} if loader is cpu105 else {}))
            calls.append(events)
        self.assertEqual(calls[0], calls[1])

    def test_all_core_sidecars_and_fp32_remain_required(self):
        for name in ("model.safetensors", *native.EXPECTED_SIDECARS):
            events, modules, digest, _ = self.fixture(changed=name)
            with self.subTest(name=name), patch.dict(sys.modules, modules), patch.object(cpu105, "sha256", side_effect=digest), \
                 self.assertRaises(RuntimeError):
                cpu105.load(cfg("C4"), training_only=True)
            self.assertFalse(any(event[0] == "core" for event in events))
        _, modules, digest, _ = self.fixture(fp32=False)
        with patch.dict(sys.modules, modules), patch.object(cpu105, "sha256", side_effect=digest), self.assertRaisesRegex(RuntimeError, "FP32"):
            cpu105.load(cfg("A4"), training_only=True)

    def test_runtime_boundary_and_actual_four_gib_priority_refusals_unchanged(self):
        self.assertIs(cpu105.runtime, native.runtime); self.assertIs(cpu105.boundary, native.boundary)
        with tempfile.TemporaryDirectory() as temporary:
            fake = ModuleType("psutil")
            fake.virtual_memory = lambda: SimpleNamespace(available=4 * 2**30 - 1)
            with patch.dict(os.environ, {"RINEYA_LANE_DIR": temporary}), patch.dict(sys.modules, {"psutil": fake}), \
                 self.assertRaisesRegex(RuntimeError, "below 4 GiB"):
                cpu105.boundary(cfg("A4"))
            fake.virtual_memory = lambda: SimpleNamespace(available=4 * 2**30)
            with patch.dict(os.environ, {"RINEYA_LANE_DIR": temporary}), patch.dict(sys.modules, {"psutil": fake}):
                cpu105.boundary(cfg("A4"))
                (Path(temporary) / "rineya-cpu-lane.waiting.rcx").write_text("synthetic waiter")
                with self.assertRaisesRegex(RuntimeError, "priority waiter"):
                    cpu105.boundary(cfg("A4"))

    def test_sampler_environment_unchanged_and_profile_exclusive_scope(self):
        with patch.dict(os.environ, {}, clear=True):
            profile = cpu105.admission_profile()
        self.assertEqual(profile["allowed_training_kinds"], ["A4", "C3", "C4"])
        self.assertTrue(profile["training_only"]); self.assertFalse(profile["training_cache"])
        self.assertEqual(profile["load_free_host_ram_floor_gib"], 10.5)
        self.assertEqual(profile["unchanged_boundary_free_host_ram_floor_gib"], 4)
        for name, value in (("BWM_CPU_BATCH_TOKENS", "4000"), ("BWM_RAM_FLOOR_GB", "4")):
            events, modules, digest, _ = self.fixture()
            with patch.dict(os.environ, {name: value}, clear=True), patch.dict(sys.modules, modules), \
                 patch.object(cpu105, "sha256", side_effect=digest), self.assertRaises(ValueError):
                cpu105.load(cfg("C4"), training_only=True)
            self.assertEqual(events, [])

    def test_every_old_loader_and_cpu11_trainer_byte_hash_unchanged(self):
        for relative, digest in OLD_SHA.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_cpu11_core_precision_load_tail_is_ast_identical(self):
        def tail(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load")
            index = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "set_num_threads")
            return ast.dump(ast.Module(body=body[index:], type_ignores=[]))
        self.assertEqual(tail(ROOT / "src/dsv3wm/native_cpu11.py"), tail(ROOT / "src/dsv3wm/native_cpu105_training.py"))

    def test_trainer_exact_copy_and_nested_loss_ast_allow_only_admission_changes(self):
        original = (ROOT / "scripts/train_recovery_student_cpu11.py").read_text(encoding="utf-8")
        expected = original
        for before, after in COPY_CHANGES:
            self.assertEqual(expected.count(before), 1); expected = expected.replace(before, after)
        actual = (ROOT / "scripts/train_recovery_student_cpu105.py").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)
        def losses(source):
            return {node.name: ast.dump(node) for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name != "main"}
        self.assertEqual(losses(original), losses(actual)); self.assertIn("loss_pair", losses(actual))
        self.assertEqual(original.count("lm.thermal_guard()"), actual.count("lm.thermal_guard()"))

    def test_a3_cli_repeat_refuses_before_freeze_inputs_output_or_runtime(self):
        path = ROOT / "scripts/train_recovery_student_cpu105.py"
        spec = importlib.util.spec_from_file_location("_h3_cpu105_training_only", path)
        script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            argv = ["train_recovery_student_cpu105.py", "--config", str(ROOT / "configs/recovery_student_A3.json")]
            for flag in ("train", "tasks", "teacher-dir", "warm-dir", "qualification", "freeze", "out"):
                argv += ["--" + flag, str(directory / flag)]
            with patch.object(sys, "argv", argv), patch("dsv3wm.native.verify_freeze") as verify, \
                 patch("dsv3wm.native.runtime") as runtime, patch.object(cpu105, "load") as load:
                with self.assertRaisesRegex(ValueError, "ordinary A3 repeats"):
                    script.main()
                verify.assert_not_called(); runtime.assert_not_called(); load.assert_not_called()
            self.assertFalse((directory / "out").exists())

    def freeze_fixture(self, directory):
        paths = [ROOT / relative for relative in OLD_SHA]
        paths += [ROOT / "scripts/train_recovery_student.py", *cpu105.registered_config_paths()]
        record = {"instrument": "warm-verified-H3-students-controls-v1", "confirmation_access": False,
                  "corpus_gate": {"passed": True, "parents": 27}, "shared_verified_parent_count": 27,
                  "cpu11_phase_amendment": {"profile": native_cpu11.ADMISSION_PROFILE, "original_freeze_sha256": "a" * 64},
                  "files": {str(path.resolve()).replace("\\", "/"): native.sha256(path) for path in paths}}
        source = directory / "synthetic_training_freeze.json"
        source.write_text(json.dumps(record), encoding="utf-8")
        return source, record

    def test_extension_preserves_all_original_metadata_and_members_and_own_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            source, original = self.freeze_fixture(Path(temporary))
            before = source.read_bytes()
            result = freezer.extend(source)
            self.assertEqual(source.read_bytes(), before)
            for key, value in original.items():
                if key != "files": self.assertEqual(result[key], value)
            self.assertTrue(original["files"].items() <= result["files"].items())
            self.assertEqual(result["files"][str(source.resolve()).replace("\\", "/")], native.sha256(source))
            self.assertEqual(result["cpu105_training_amendment"]["profile"]["allowed_training_kinds"], ["A4", "C3", "C4"])
            for path in freezer.extra_paths():
                self.assertEqual(result["files"][str(path.resolve()).replace("\\", "/")], native.sha256(path))

    def test_extension_rejects_inference_other_domains_drift_reextension_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, original = self.freeze_fixture(directory)
            for instrument in ("warm-verified-H3-readout-v1", "recovery-H3-teacher-v1", "tick-compatible-ledger-development-v1",
                               "rcwm-RP-retention-v2-B0", "block-G1-v1"):
                source.write_text(json.dumps({**original, "instrument": instrument}), encoding="utf-8")
                with self.subTest(instrument=instrument), self.assertRaises(ValueError): freezer.extend(source)
            for change in ({"confirmation_access": True}, {"cpu105_training_amendment": {}}, {"cpu11_phase_amendment": {}}):
                source.write_text(json.dumps({**original, **change}), encoding="utf-8")
                with self.assertRaises(ValueError): freezer.extend(source)
            wrong = copy.deepcopy(original); wrong["files"][next(iter(wrong["files"]))] = "0" * 64
            source.write_text(json.dumps(wrong), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "freeze drift"): freezer.extend(source)
            output = directory / "existing.json"; output.write_bytes(b"preserved old freeze")
            argv = ["freeze_recovery_student_cpu105_training.py", "--freeze", str(source), "--out", str(output)]
            with patch.object(sys, "argv", argv), patch.object(freezer, "extend") as extend, self.assertRaises(FileExistsError):
                freezer.main()
            extend.assert_not_called(); self.assertEqual(output.read_bytes(), b"preserved old freeze")


if __name__ == "__main__":
    unittest.main()
