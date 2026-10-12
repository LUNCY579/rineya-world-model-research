"""CPU-only admission and source parity with fake modules; never load a model."""
import ast
import copy
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from dsv3wm import native, native_cpu11

ROOT = Path(__file__).resolve().parents[1]


class CPU11Tests(unittest.TestCase):
    def fixture(self, *, free=12., commit=None, dtype_ok=True, changed=None):
        events = []
        torch = ModuleType("torch")
        torch.float32 = object()
        torch.set_num_threads = lambda n: events.append(("threads", n))
        torch.set_float32_matmul_precision = lambda value: events.append(("precision", value))
        torch.use_deterministic_algorithms = lambda value, **kw: events.append(("determinism", value, kw))
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype=torch.float32 if dtype_ok else "float16")]))
        tokenizer = object()
        lm = ModuleType("bwmn.lm")
        lm.MODELS = {"titan-sparse": Path("synthetic-core")}
        lm.host_ram_free_gb = lambda: (free, free if commit is None else commit)
        lm.set_determinism = lambda seed: events.append(("seed", seed))
        def load_core(core):
            events.append(("core", core))
            return model, tokenizer
        lm.load_core = load_core
        bwmn = ModuleType("bwmn"); bwmn.lm = lm
        peft = ModuleType("peft")
        def from_pretrained(parent, path, **kwargs):
            events.append(("unmerged-peft", path, kwargs))
            self.assertIs(parent, model)
            return model
        peft.PeftModel = SimpleNamespace(from_pretrained=from_pretrained)
        expected = {"model.safetensors": native.EXPECTED_CORE_SHA, **native.EXPECTED_SIDECARS}
        def digest(path):
            name = Path(path).name
            events.append(("hash", name))
            return "changed-parent-bytes" if name == changed else expected[name]
        return events, {"bwmn": bwmn, "bwmn.lm": lm, "torch": torch, "peft": peft}, digest, (model, tokenizer, lm)

    def call(self, loader, *, free=12., dtype_ok=True, changed=None, adapter=None):
        events, modules, digest, expected = self.fixture(free=free, dtype_ok=dtype_ok, changed=changed)
        cfg = SimpleNamespace(device="cpu", threads=4, seed=7, core="titan-sparse")
        with patch.dict(sys.modules, modules), patch.object(loader, "sha256", side_effect=digest):
            result = loader.load(cfg, adapter=adapter)
        self.assertEqual(result, expected)
        return events

    def test_threshold_accepts_eleven_and_preserves_historical_twelve_floor(self):
        for free in (11., 11.9, 12.):
            self.assertIn(("core", "titan-sparse"), self.call(native_cpu11, free=free))
        events, modules, digest, _ = self.fixture(free=10.999)
        cfg = SimpleNamespace(device="cpu", threads=4, seed=0, core="titan-sparse")
        with patch.dict(sys.modules, modules), patch.object(native_cpu11, "sha256", side_effect=digest), \
             self.assertRaisesRegex(RuntimeError, "11 GiB"):
            native_cpu11.load(cfg)
        self.assertEqual(events, [])
        with patch.dict(sys.modules, self.fixture(free=11.9)[1]), self.assertRaisesRegex(RuntimeError, "12 GiB"):
            native.load(cfg)

    def test_commit_floor_refuses_before_core_and_records_actual_admission(self):
        events, modules, digest, _ = self.fixture(free=11.8, commit=10.9)
        cfg = SimpleNamespace(device="cpu", threads=4, seed=0, core="titan-sparse")
        with patch.dict(sys.modules, modules), patch.object(native_cpu11, "sha256", side_effect=digest), \
             self.assertRaisesRegex(RuntimeError, "physical and commit"):
            native_cpu11.load(cfg)
        self.assertEqual(events, [])
        check = native_cpu11.last_admission_check()
        self.assertEqual(check["physical_available_gib"], 11.8)
        self.assertEqual(check["commit_available_gib"], 10.9)
        self.assertTrue(check["physical_qualified"])
        self.assertFalse(check["commit_qualified"])
        self.assertEqual(check["status"], "refused")
        self.call(native_cpu11, free=11.1)
        check = native_cpu11.last_admission_check()
        self.assertEqual(check["status"], "admitted")
        self.assertTrue(check["passed"])
        self.assertEqual(check["commit_available_gib"], 11.1)
        check["passed"] = False
        self.assertTrue(native_cpu11.last_admission_check()["passed"])

    def test_rejects_cuda_and_nonfour_threads_before_import_or_model_work(self):
        for device, threads in (("cuda", 4), ("cpu", 1), ("cpu", 8), ("cpu", True), ("cpu", 4.0)):
            with self.subTest(device=device, threads=threads):
                events, modules, digest, _ = self.fixture()
                cfg = SimpleNamespace(device=device, threads=threads)
                with patch.dict(sys.modules, modules), patch.object(native_cpu11, "sha256", side_effect=digest), \
                     self.assertRaises(ValueError):
                    native_cpu11.load(cfg)
                self.assertEqual(events, [])

    def test_core_and_every_sidecar_drift_refuse_before_load_core(self):
        for name in ("model.safetensors", *native.EXPECTED_SIDECARS):
            with self.subTest(name=name):
                events, modules, digest, _ = self.fixture(changed=name)
                cfg = SimpleNamespace(device="cpu", threads=4, seed=0, core="titan-sparse")
                with patch.dict(sys.modules, modules), patch.object(native_cpu11, "sha256", side_effect=digest), self.assertRaises(RuntimeError):
                    native_cpu11.load(cfg, adapter="synthetic-adapter")
                self.assertFalse(any(event[0] in {"core", "unmerged-peft"} for event in events))

    def test_fp32_rejection_precedes_peft_and_no_merge_is_invoked(self):
        events, modules, digest, _ = self.fixture(dtype_ok=False)
        cfg = SimpleNamespace(device="cpu", threads=4, seed=0, core="titan-sparse")
        with patch.dict(sys.modules, modules), patch.object(native_cpu11, "sha256", side_effect=digest), \
             self.assertRaisesRegex(RuntimeError, "FP32"):
            native_cpu11.load(cfg, adapter="synthetic-adapter")
        self.assertFalse(any(event[0] == "unmerged-peft" for event in events))
        calls = self.call(native_cpu11, adapter="synthetic-adapter")
        self.assertEqual(calls[-1], ("unmerged-peft", "synthetic-adapter", {"is_trainable": False}))

    def test_accepted_load_call_sequence_is_identical_and_runtime_aliases_unchanged(self):
        self.assertEqual(self.call(native, adapter="synthetic-adapter"), self.call(native_cpu11, adapter="synthetic-adapter"))
        self.assertIs(native_cpu11.runtime, native.runtime)
        self.assertIs(native_cpu11.boundary, native.boundary)
        self.assertEqual(native_cpu11.ADMISSION_PROFILE["unchanged_boundary_free_host_ram_floor_gib"], 4)
        self.assertFalse(native_cpu11.ADMISSION_PROFILE["prediction_change"])

    def test_sampler_environment_preserves_geometry_and_records_exact_default_floor(self):
        for batch, floor in ((None, None), ("8000", "8"), ("8000", "8.0")):
            with patch.dict(os.environ, {}, clear=True):
                if batch is not None: os.environ["BWM_CPU_BATCH_TOKENS"] = batch
                if floor is not None: os.environ["BWM_RAM_FLOOR_GB"] = floor
                before = dict(os.environ)
                profile = native_cpu11.admission_profile()
                self.assertEqual(profile["environment"], {"BWM_CPU_BATCH_TOKENS": batch, "BWM_RAM_FLOOR_GB": floor})
                self.assertEqual(dict(os.environ), before)
        for name, value in (("BWM_CPU_BATCH_TOKENS", "4000"), ("BWM_CPU_BATCH_TOKENS", "08000"),
                            ("BWM_RAM_FLOOR_GB", "4"), ("BWM_RAM_FLOOR_GB", "11"), ("BWM_RAM_FLOOR_GB", "nan")):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}, clear=True):
                events, modules, digest, _ = self.fixture()
                cfg = SimpleNamespace(device="cpu", threads=4, seed=0, core="titan-sparse")
                with patch.dict(sys.modules, modules), patch.object(native_cpu11, "sha256", side_effect=digest), self.assertRaises(ValueError):
                    native_cpu11.load(cfg)
                self.assertEqual(events, [])

    def test_original_algorithm_tail_is_exact_after_resource_admission_prefix(self):
        old = ast.parse((ROOT / "src/dsv3wm/native.py").read_text(encoding="utf-8"))
        new = ast.parse((ROOT / "src/dsv3wm/native_cpu11.py").read_text(encoding="utf-8"))
        old_body = copy.deepcopy(next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "load").body)
        new_body = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "load").body
        self.assertEqual(old_body[2].test.values[1].comparators[0].value, 12)
        def tail(body):
            for index, node in enumerate(body):
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "set_num_threads"):
                    return body[index:]
            raise AssertionError("thread/precision/load tail missing")
        self.assertEqual(ast.dump(ast.Module(body=tail(old_body), type_ignores=[])),
                         ast.dump(ast.Module(body=tail(new_body), type_ignores=[])))

    def test_collector_exact_copy_has_only_loader_pins_and_profile_metadata_changes(self):
        old = (ROOT / "scripts/collect_recovery_teacher.py").read_text(encoding="utf-8")
        new = (ROOT / "scripts/collect_recovery_teacher_cpu11.py").read_text(encoding="utf-8")
        allowed = [
            ("from dsv3wm.native import EXPECTED_CORE_SHA, load, runtime, sha256, verify_freeze",
             "from dsv3wm.native import EXPECTED_CORE_SHA, runtime, sha256, verify_freeze\n    from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load"),
            ('REPO / "dsv3wm/src/dsv3wm/native.py", REPO / "dsv3wm/src/dsv3wm/provenance.py",',
             'REPO / "dsv3wm/src/dsv3wm/native.py", REPO / "dsv3wm/src/dsv3wm/native_cpu11.py", REPO / "dsv3wm/src/dsv3wm/provenance.py",'),
            ('REPO / "bwm/code/bwmn/gpu_lease.py", REPO / "bwm/code/scripts/planbench_qwen.py"]',
             'REPO / "bwm/code/bwmn/gpu_lease.py", REPO / "bwm/code/scripts/planbench_qwen.py",\n               REPO / "dsv3wm/scripts/collect_recovery_teacher.py"]'),
            ('"parent_checkpoint_sha256": EXPECTED_CORE_SHA, "parent_checkpoint_verified_by_native_load": False,',
             '"parent_checkpoint_sha256": EXPECTED_CORE_SHA, "parent_checkpoint_verified_by_native_load": False,\n                  "resource_admission_profile": admission_profile(),'),
            ('            model, tokenizer, lm = load(cfg)\n',
             '            model, tokenizer, lm = load(cfg)\n            manifest["admission_check"] = last_admission_check()\n'),
            ('    finally:\n        manifest["duration_s"] = time.perf_counter() - started',
             '    finally:\n        manifest["admission_check"] = last_admission_check()\n        manifest["duration_s"] = time.perf_counter() - started')]
        for before, after in allowed:
            self.assertEqual(old.count(before), 1)
            old = old.replace(before, after)
        self.assertEqual(new, old)


if __name__ == "__main__":
    unittest.main()
