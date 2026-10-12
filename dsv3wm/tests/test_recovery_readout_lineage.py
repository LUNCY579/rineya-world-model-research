"""Pure recovery readout provenance/tamper fixtures; never load native weights."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dsv3wm.native import EXPECTED_CORE_SHA, sha256
from dsv3wm.recovery import INITIAL_F_SHA, RecoveryConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_recovery_readout.py"
spec = importlib.util.spec_from_file_location("_dsv3wm_readout_lineage_tests", SCRIPT)
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)
RESOURCE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_measured.py"
resource_spec = importlib.util.spec_from_file_location("_dsv3wm_resource_failure_tests", RESOURCE_SCRIPT)
resource = importlib.util.module_from_spec(resource_spec)
resource_spec.loader.exec_module(resource)


class ReadoutLineageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        self.adapter = self.run / "adapter"
        self.adapter.mkdir(parents=True)
        self.config = self.root / "config.json"
        self.train = self.root / "train.jsonl"
        self.qualification = self.root / "qualification.json"
        self.freeze = self.root / "training_freeze.json"
        self.config.write_text(json.dumps(RecoveryConfig().dict()), encoding="utf-8")
        self.train.write_text("explicit unit-test data bytes\n", encoding="utf-8")
        self.qualification.write_text(json.dumps({"status": "development-evaluated", "baseline_qualified": True}), encoding="utf-8")
        self.peft = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     "use_rslora": False, "use_dora": False, "bias": "none", "modules_to_save": None,
                     "rank_pattern": {}, "alpha_pattern": {}, "target_parameters": None,
                     "base_model_name_or_path": "unit-test-core-path", "revision": None,
                     "peft_version": "unit-test-version", "inference_mode": True}
        (self.adapter / "adapter_config.json").write_text(json.dumps(self.peft), encoding="utf-8")
        initial = self.root / "registered"
        initial.mkdir()
        (initial / "adapter_config.json").write_text(json.dumps(self.peft), encoding="utf-8")
        self.initial_patch = patch("dsv3wm.recovery.INITIAL_F", initial)
        self.initial_patch.start()
        # Not a checkpoint: checksum lineage is tested using an explicit byte fixture.
        (self.adapter / "adapter_model.safetensors").write_bytes(b"unit-test adapter bytes, no model")
        self.freeze.write_text(json.dumps({"files": {str(path): sha256(path)
                                                   for path in (self.config, self.train, self.qualification)}}), encoding="utf-8")
        self.record = {"status": "pilot-trained", "initial_adapter_sha256": INITIAL_F_SHA,
                       "command": ["train_recovery.py", "--config", str(self.config), "--freeze", str(self.freeze),
                                   "--qualification", str(self.qualification)],
                       "config": RecoveryConfig.read(self.config).dict(), "config_sha256": sha256(self.config),
                       "data_sha256": sha256(self.train), "qualification_sha256": sha256(self.qualification),
                       "freeze_sha256": sha256(self.freeze), "core_sha256": EXPECTED_CORE_SHA,
                       "adapter_sha256": sha256(self.adapter / "adapter_model.safetensors")}
        (self.run / "run.json").write_text(json.dumps(self.record), encoding="utf-8")

    def tearDown(self):
        self.initial_patch.stop()
        self.directory.cleanup()

    def test_matching_prospective_run_lineage_is_accepted(self):
        arm, paths = readout.validate_run(self.run, self.train)
        self.assertEqual(arm, "A0")
        self.assertIn(self.adapter / "adapter_config.json", paths)

    def test_training_input_or_weight_tamper_is_rejected(self):
        self.train.write_text("changed dataset bytes", encoding="utf-8")
        with self.assertRaises(ValueError):
            readout.validate_run(self.run, self.train)
        self.train.write_text("explicit unit-test data bytes\n", encoding="utf-8")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"changed weight bytes")
        with self.assertRaises(ValueError):
            readout.validate_run(self.run, self.train)

    def test_archived_training_source_drift_is_not_silently_re_frozen(self):
        self.qualification.write_text("changed qualification bytes", encoding="utf-8")
        # Replacing run's reported hash cannot hide drift from the old freeze.
        self.record["qualification_sha256"] = sha256(self.qualification)
        (self.run / "run.json").write_text(json.dumps(self.record), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "freeze drift"):
            readout.validate_run(self.run, self.train)

    def test_saved_adapter_scaling_tamper_is_rejected(self):
        for key, value in (("use_rslora", True), ("use_dora", True), ("rank_pattern", {"q_proj": 8}),
                           ("alpha_pattern", {"q_proj": 64}), ("target_parameters", ["q_proj.weight"]),
                           ("unknown_behavior_field", True)):
            with self.subTest(field=key):
                changed = dict(self.peft, **{key: value})
                (self.adapter / "adapter_config.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    readout.validate_run(self.run, self.train)

    def test_saved_adapter_metadata_changes_require_the_same_core_identity(self):
        changed = dict(self.peft, base_model_name_or_path="another-metadata-path", revision="metadata-revision",
                       peft_version="another-metadata-version", inference_mode=False)
        (self.adapter / "adapter_config.json").write_text(json.dumps(changed), encoding="utf-8")
        arm, _ = readout.validate_run(self.run, self.train)
        self.assertEqual(arm, "A0")
        self.record["core_sha256"] = "wrong-parent-core"
        (self.run / "run.json").write_text(json.dumps(self.record), encoding="utf-8")
        with self.assertRaises(ValueError):
            readout.validate_run(self.run, self.train)


class ResourceFailureTest(unittest.TestCase):
    def test_monitor_failure_stops_mock_child_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "resource.json"
            child = MagicMock()
            child.pid, child.returncode = 12345, None
            child.poll.side_effect = lambda: child.returncode
            child.wait.return_value = -15
            owned = MagicMock()
            owned.pid = child.pid
            owned.create_time.return_value = 100.0
            owned.children.return_value = []
            owned.is_running.side_effect = lambda: child.returncode is None
            owned.memory_info.side_effect = RuntimeError("unit-test monitor failure")
            def terminate():
                child.returncode = -15
            owned.terminate.side_effect = terminate
            owned.kill.side_effect = terminate
            with patch("sys.argv", ["run_measured.py", "--out", str(out), "--", "unit-test-script.py"]), \
                 patch.object(resource.subprocess, "Popen", return_value=child), \
                 patch("psutil.Process", return_value=owned), \
                 patch("psutil.wait_procs", return_value=([owned], [])):
                with self.assertRaises(RuntimeError):
                    resource.main()
            self.assertTrue(owned.terminate.called or owned.kill.called)
            self.assertTrue(child.wait.called)
            record = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["exit_code"], -15)
            self.assertIn("terminated/reaped", record["child_cleanup"])

    def test_spawn_failure_is_recorded_without_running_a_child(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "resource.json"
            with patch("sys.argv", ["run_measured.py", "--out", str(out), "--", "unit-test-script.py"]), \
                 patch.object(resource.subprocess, "Popen", side_effect=OSError("unit-test spawn failure")):
                with self.assertRaises(OSError):
                    resource.main()
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
