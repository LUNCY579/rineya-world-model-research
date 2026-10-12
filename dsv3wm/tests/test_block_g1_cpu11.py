"""Copy, freeze-preservation and pre-load refusal tests; no model or G1 run."""
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
new = importlib.import_module("evaluate_block_g1_cpu11")
freezer = importlib.import_module("freeze_block_g1_cpu11")
from dsv3wm import native, native_cpu11


class BlockG1CPU11Tests(unittest.TestCase):
    def test_evaluator_copy_changes_only_loader_pins_and_admission_records(self):
        old = (ROOT / "scripts/evaluate_block_g1.py").read_text(encoding="utf-8")
        changed = (ROOT / "scripts/evaluate_block_g1_cpu11.py").read_text(encoding="utf-8")
        allowed = [
            ('from dsv3wm.provenance import verify_bound_inputs\n',
             'from dsv3wm.provenance import verify_bound_inputs\nfrom dsv3wm.native_cpu11 import admission_profile, last_admission_check\n\n\ndef cpu11_sources():\n    return [*required_sources(), Path(__file__), PACKAGE / "src/dsv3wm/native_cpu11.py",\n            PACKAGE / "scripts/freeze_block_g1_cpu11.py", PACKAGE / "prereg/BLOCK_G1_CPU11_V1.md"]\n'),
            ('verify_bound_inputs(args.freeze, [*required_sources(), args.config, args.cases, args.manifest,',
             'verify_bound_inputs(args.freeze, [*cpu11_sources(), args.config, args.cases, args.manifest,'),
            ('result = {"status": "started",', 'result = {"resource_admission_profile": admission_profile(), "status": "started",'),
            ('sha256(p) for p in required_sources()},', 'sha256(p) for p in cpu11_sources()},'),
            ('from dsv3wm.native import runtime, load, EXPECTED_CORE_SHA',
             'from dsv3wm.native import runtime, EXPECTED_CORE_SHA\n        from dsv3wm.native_cpu11 import load'),
            ('            model, tokenizer, lm = load(cfg)\n',
             '            model, tokenizer, lm = load(cfg)\n            result["admission_check"] = last_admission_check()\n'),
            ('    finally:\n        result.update(duration_s=',
             '    finally:\n        result["admission_check"] = last_admission_check()\n        result.update(duration_s=')]
        for before, after in allowed:
            self.assertEqual(old.count(before), 1)
            old = old.replace(before, after)
        self.assertEqual(changed, old)

    def test_original_frozen_source_stays_byte_identical(self):
        freeze = json.loads((ROOT / "prereg/BLOCK_G1_FREEZE_V2.json").read_text(encoding="utf-8"))
        for name in ("evaluate_block_g1.py", "build_block_g1.py", "freeze_block_g1.py"):
            path = ROOT / "scripts" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             freeze["files"][str(path.resolve()).replace("\\", "/")])

    def test_input_validation_and_scoring_objects_are_inherited(self):
        old = importlib.import_module("evaluate_block_g1")
        self.assertEqual(new.validate_cases.__code__.co_code, old.validate_cases.__code__.co_code)
        self.assertIs(new.score_case, old.score_case)
        self.assertIs(new.summarize, old.summarize)
        self.assertIs(new.BlockG1Config, old.BlockG1Config)
        sources = set(new.cpu11_sources())
        self.assertTrue(set(new.required_sources()) <= sources)
        self.assertTrue({ROOT / "scripts/evaluate_block_g1_cpu11.py", ROOT / "scripts/evaluate_block_g1.py",
                         ROOT / "src/dsv3wm/native_cpu11.py", ROOT / "scripts/freeze_block_g1_cpu11.py",
                         ROOT / "prereg/BLOCK_G1_CPU11_V1.md"} <= sources)

    def test_extension_preserves_every_original_pin_metadata_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            member = directory / "fixed_case_bytes"
            member.write_text("unchanged input bytes", encoding="utf-8")
            source = directory / "original.json"
            record = {"instrument": "constructed-original-delta-Block-G1-v1", "confirmation_access": False,
                      "frozen_at_utc": "original-time", "gate": {"format": 11, "tuples": 10, "legal_changes": 5},
                      "files": {str(member.resolve()): native.sha256(member)}}
            source.write_text(json.dumps(record), encoding="utf-8")
            original = source.read_bytes()
            extended = freezer.extend(source)
            self.assertEqual(source.read_bytes(), original)
            for key, value in record.items():
                if key != "files":
                    self.assertEqual(extended[key], value)
            for key, value in record["files"].items():
                self.assertEqual(extended["files"][key], value)
            self.assertEqual(extended["cpu11_resource_amendment"]["original_freeze_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(extended["cpu11_resource_amendment"]["profile"]["load_free_host_ram_floor_gib"], 11)
            self.assertFalse(extended["cpu11_resource_amendment"]["profile"]["prediction_change"])
            self.assertTrue(set(str(p.resolve()).replace("\\", "/") for p in new.cpu11_sources()) <= extended["files"].keys())
            for path, digest in extended["files"].items():
                self.assertEqual(native.sha256(path), digest)

    def test_drift_and_wrong_phase_refuse_without_writing_an_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            member = directory / "input"
            member.write_text("before", encoding="utf-8")
            source = directory / "freeze.json"
            record = {"instrument": "constructed-original-delta-Block-G1-v1", "confirmation_access": False,
                      "files": {str(member.resolve()): native.sha256(member)}}
            source.write_text(json.dumps(record), encoding="utf-8")
            member.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "freeze drift"):
                freezer.extend(source)
            record.update(instrument="other-study")
            source.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Block G1"):
                freezer.extend(source)
            record.update(instrument="constructed-original-delta-Block-G1-v1", confirmation_access=True)
            source.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Block G1"):
                freezer.extend(source)

    def test_entrypoint_refused_admission_preserves_failed_record_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = directory / "config.json"
            cfg = new.BlockG1Config()
            config.write_text(json.dumps(cfg.dict()), encoding="utf-8")
            cases = directory / "cases.jsonl"
            cases.write_text("".join(json.dumps({"id": str(i), "parent_world_id": str(i),
                                               "role": "primary" if i < 12 else "diagnostic"}) + "\n" for i in range(16)), encoding="utf-8")
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"generator": new.VERSION, "cases_sha256": "case-hash", "config": cfg.dict()}), encoding="utf-8")
            out = directory / "new_result.json"
            torch = ModuleType("torch")
            peft = ModuleType("peft")
            peft.PeftModel = SimpleNamespace(from_pretrained=lambda *a, **k: self.fail("adapter/model work forbidden"))
            def digest(path):
                path = Path(path)
                if path == cases: return "case-hash"
                if path == new.BW_ADAPTER / "adapter_model.safetensors": return new.BW_SHA
                if path == new.BW_ADAPTER / "adapter_config.json": return new.BW_CONFIG_SHA
                return native.sha256(path)
            arguments = ["evaluate_block_g1_cpu11.py", "--config", str(config), "--cases", str(cases),
                         "--manifest", str(manifest), "--freeze", "synthetic-freeze", "--out", str(out)]
            with patch.object(sys, "argv", arguments), patch.dict(sys.modules, {"torch": torch, "peft": peft}), \
                 patch.object(new, "sha256", side_effect=digest), patch.object(new, "verify_freeze", return_value="freeze-hash"), \
                 patch.object(new, "verify_bound_inputs"), patch.object(new, "validate_cases"), \
                 patch.object(native, "runtime", return_value=nullcontext()), \
                 patch.object(native, "load", side_effect=AssertionError("old loader forbidden")) as old_load, \
                 patch.object(native_cpu11, "load", side_effect=RuntimeError("CPU11 admission refused")) as new_load, \
                 patch.object(new, "last_admission_check", return_value={"status": "refused"}), \
                 self.assertRaisesRegex(RuntimeError, "CPU11 admission refused"):
                new.main()
            new_load.assert_called_once()
            old_load.assert_not_called()
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["admission_check"], {"status": "refused"})
            self.assertEqual(result["resource_admission_profile"]["load_free_host_ram_floor_gib"], 11)
            self.assertNotIn("outcomes", result)
            self.assertNotIn("summary", result)
            self.assertEqual(result["decoder_forwards"], 0)

    def test_nondefault_sampler_environment_refuses_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "freeze.json"
            source.write_text(json.dumps({"instrument": "constructed-original-delta-Block-G1-v1",
                                          "confirmation_access": False, "files": {}}), encoding="utf-8")
            with patch.dict("os.environ", {"BWM_CPU_BATCH_TOKENS": "4000"}), self.assertRaisesRegex(ValueError, "unchanged"):
                freezer.extend(source)

    def test_existing_cpu11_amendment_refused_without_metadata_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "already_extended.json"
            record = {"instrument": "constructed-original-delta-Block-G1-v1", "confirmation_access": False,
                      "files": {}, "cpu11_resource_amendment": {"original_freeze_sha256": "preserved-reference"}}
            source.write_text(json.dumps(record), encoding="utf-8")
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "already present"):
                freezer.extend(source)
            self.assertEqual(source.read_bytes(), before)

    def test_cli_preserves_existing_freeze_before_reading_source(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "existing.json"
            out.write_text("preserve", encoding="utf-8")
            with patch.object(sys, "argv", ["freeze_block_g1_cpu11.py", "--freeze", "nonexistent", "--out", str(out)]), \
                 self.assertRaises(FileExistsError):
                freezer.main()
            self.assertEqual(out.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
