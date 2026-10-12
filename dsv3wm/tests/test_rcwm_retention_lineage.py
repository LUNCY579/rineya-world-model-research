"""Adapted retention raw-digest, configuration and qualification edges; no model."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsv3wm.config import Config
from dsv3wm.native import EXPECTED_CORE_SHA, sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_rcwm_retention_adapted.py"
spec = importlib.util.spec_from_file_location("_rcwm_retention_lineage_tests", SCRIPT)
freezer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freezer)


class RCWMRetentionLineageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.initial = self.root / "registered_T2"
        self.initial.mkdir()
        self.peft = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     "use_rslora": False, "use_dora": False, "rank_pattern": {}, "alpha_pattern": {},
                     "target_parameters": None, "bias": "none", "modules_to_save": None,
                     "inference_mode": True, "base_model_name_or_path": "unit-test core metadata"}
        self.write(self.initial / "adapter_config.json", self.peft)
        (self.initial / "adapter_model.safetensors").write_bytes(b"unit fixture only, not native T2 weights")
        self.initial_sha = sha256(self.initial / "adapter_model.safetensors")
        self.initial_config_sha = sha256(self.initial / "adapter_config.json")
        self.patches = [patch("dsv3wm.ledger_recovery.T2_ADAPTER", self.initial),
                        patch("dsv3wm.ledger_recovery.T2_SHA", self.initial_sha),
                        patch("dsv3wm.ledger_recovery.T2_CONFIG_SHA", self.initial_config_sha)]
        for item in self.patches:
            item.start()
        self.trainer = freezer.trainer_module()
        self.config = self.root / "ledger_config.json"
        self.cfg = self.trainer.LedgerConfig()
        self.write(self.config, self.cfg.dict())
        self.train, self.dev = self.root / "train.jsonl", self.root / "dev.jsonl"
        self.train.write_text("TRAIN digest fixture, not native rows\n", encoding="utf-8")
        self.dev.write_text("DEV digest fixture, not native rows\n", encoding="utf-8")
        self.qualification_freeze = self.root / "qualified_b0_freeze.json"
        qualification_pins = [self.config, self.dev, self.initial / "adapter_model.safetensors",
                              self.initial / "adapter_config.json", *self.trainer.required_sources()]
        self.write(self.qualification_freeze, {"files": {str(path): sha256(path) for path in qualification_pins}})
        self.qualification = self.root / "qualified_b0.json"
        self.qualified = {"status": "development-evaluated", "baseline_only": True, "baseline_qualified": True,
                          "adapter_sha256": self.initial_sha, "adapter_config_sha256": self.initial_config_sha,
                          "freeze_sha256": sha256(self.qualification_freeze), "data_sha256": sha256(self.dev),
                          "config": self.cfg.dict(), "config_sha256": sha256(self.config),
                          "core_sha256": EXPECTED_CORE_SHA, "oracle_feedback": False,
                          "source_sha256": {str(path): sha256(path) for path in self.trainer.required_sources()},
                          "command": ["evaluate_ledger_recovery.py", "--freeze", str(self.qualification_freeze),
                                      "--config", str(self.config), "--data", str(self.dev)],
                          "summary": {"canonical_id": {"1": {"n": 12, "rollout_correct": 12, "rollout_malformed": 0}}}}
        self.write(self.qualification, self.qualified)
        self.training_freeze = self.root / "training_freeze.json"
        bound = [self.config, self.train, self.dev, self.qualification, self.initial / "adapter_model.safetensors",
                 self.initial / "adapter_config.json", *self.trainer.required_sources()]
        self.write(self.training_freeze, {"qualification_freeze_sha256": sha256(self.qualification_freeze),
                                          "files": {str(path): sha256(path) for path in bound}})
        self.run = self.root / "A0"
        (self.run / "adapter").mkdir(parents=True)
        self.write(self.run / "adapter/adapter_config.json", self.peft)
        (self.run / "adapter/adapter_model.safetensors").write_bytes(b"unit fixture only, not trained native weights")
        self.record = {"status": "pilot-trained", "config": self.cfg.dict(),
                       "command": ["train_ledger_recovery.py", "--config", str(self.config), "--train", str(self.train),
                                   "--freeze", str(self.training_freeze), "--qualification", str(self.qualification)],
                       "config_sha256": sha256(self.config), "data_sha256": sha256(self.train),
                       "freeze_sha256": sha256(self.training_freeze), "qualification_sha256": sha256(self.qualification),
                       "qualification_freeze_sha256": sha256(self.qualification_freeze),
                       "initial_adapter_sha256": self.initial_sha, "initial_adapter_config_sha256": self.initial_config_sha,
                       "core_sha256": EXPECTED_CORE_SHA, "adapter_sha256": sha256(self.run / "adapter/adapter_model.safetensors"),
                       "adapter_config_sha256": sha256(self.run / "adapter/adapter_config.json"),
                       "source_sha256": {str(path): sha256(path) for path in self.trainer.required_sources()}}
        self.save_record()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def write(self, path, value):
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def save_record(self):
        self.write(self.run / "run.json", self.record)

    def test_completed_ledger_run_with_exact_edges_is_accepted(self):
        result = freezer.validate_run(self.run)
        self.assertEqual(result["arm"], "A0")
        self.assertEqual(result["train"], self.train)
        self.assertIn(self.run / "run.json", result["paths"])

    def test_raw_data_config_weight_and_qualification_digest_tamper_stops(self):
        for field in ("data_sha256", "config_sha256", "adapter_sha256", "adapter_config_sha256", "qualification_sha256", "freeze_sha256"):
            with self.subTest(field=field):
                original = self.record[field]
                self.record[field] = "0" * 64
                self.save_record()
                with self.assertRaises((ValueError, RuntimeError)):
                    freezer.validate_run(self.run)
                self.record[field] = original
                self.save_record()

    def test_full_config_and_qualifier_cannot_be_self_asserted(self):
        self.record["config"]["legality_weight"] = False  # Python False == 0.0 must not pass.
        self.save_record()
        with self.assertRaises(ValueError):
            freezer.validate_run(self.run)
        self.record["config"] = self.cfg.dict()
        self.record["qualification_freeze_sha256"] = "0" * 64
        self.save_record()
        with self.assertRaises(ValueError):
            freezer.validate_run(self.run)

    def test_saved_full_adapter_behavior_and_types_are_checked_after_repinning_hash(self):
        for key, value in (("use_rslora", True), ("use_dora", True), ("rank_pattern", {"q_proj": 8}),
                           ("alpha_pattern", {"q_proj": 64}), ("target_parameters", ["q_proj.weight"]),
                           ("r", 16.0), ("unknown_behavior_field", True)):
            with self.subTest(field=key):
                changed = dict(self.peft, **{key: value})
                self.write(self.run / "adapter/adapter_config.json", changed)
                self.record["adapter_config_sha256"] = sha256(self.run / "adapter/adapter_config.json")
                self.save_record()
                with self.assertRaises(ValueError):
                    freezer.validate_run(self.run)

    def test_target_order_and_metadata_only_changes_are_normalized(self):
        changed = dict(self.peft, target_modules=list(reversed(self.peft["target_modules"])),
                       base_model_name_or_path="different metadata", inference_mode=False)
        self.write(self.run / "adapter/adapter_config.json", changed)
        self.record["adapter_config_sha256"] = sha256(self.run / "adapter/adapter_config.json")
        self.save_record()
        self.assertEqual(freezer.validate_run(self.run)["arm"], "A0")
        changed["target_modules"] = "q_proj|v_proj"
        self.write(self.run / "adapter/adapter_config.json", changed)
        self.record["adapter_config_sha256"] = sha256(self.run / "adapter/adapter_config.json")
        self.save_record()
        with self.assertRaises(ValueError):
            freezer.validate_run(self.run)

    def test_training_source_and_shared_dataset_drift_are_rejected(self):
        path = self.trainer.required_sources()[0]
        self.record["source_sha256"][str(path)] = "0" * 64
        self.save_record()
        with self.assertRaises(ValueError):
            freezer.validate_run(self.run)
        self.record["source_sha256"][str(path)] = sha256(path)
        self.save_record()
        with self.assertRaises(ValueError):
            freezer.validate_run(self.run, expected_train=self.dev)

    def test_unchanged_t2_baseline_requires_matched_fixture_core_and_old_freeze(self):
        config = self.root / "retention_config.json"
        cfg = Config(device="cpu", max_length=512, generation_tokens=64)
        self.write(config, cfg.dict())
        fixtures = self.root / "rp_fixtures.json"
        cases = [{"id": f"case{index}", "family": "register_machine" if index < 12 else "prerequisite_plan",
                  "parent_world_id": f"parent{index}", "semantic_input_id": f"input{index}",
                  "graph_group_id": "register" if index < 12 else "prerequisite"} for index in range(24)]
        contract = {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64}
        self.write(fixtures, {"schema": "dsv3wm-rcwm-retention-v1", "fixtures": cases, "inference_contract": contract})
        old = self.root / "rp_B0_freeze.json"
        pins = [config, fixtures, self.initial / "adapter_model.safetensors", self.initial / "adapter_config.json",
                *freezer.retention_sources()]
        self.write(old, {"files": {str(path): sha256(path) for path in pins}})
        baseline = self.root / "rp_B0.json"
        rows = [{**case, "correct": True, "schema_valid": True} for case in cases]
        record = {"status": "development-evaluated", "adapter_sha256": self.initial_sha,
                  "adapter_config_sha256": self.initial_config_sha, "parent_sha256": EXPECTED_CORE_SHA,
                  "fixture_sha256": sha256(fixtures), "config": cfg.dict(), "config_sha256": sha256(config),
                  "inference_contract": contract, "oracle_feedback": False, "all_generations_before_scoring": True,
                  "freeze_sha256": sha256(old), "rows": rows,
                  "command": ["evaluate_rcwm_retention.py", "--config", str(config), "--fixtures", str(fixtures),
                              "--freeze", str(old), "--adapter", str(self.initial)],
                  "source_sha256": {str(path): sha256(path) for path in freezer.retention_sources()}}
        self.write(baseline, record)
        self.assertEqual(freezer.validate_baseline(baseline)["fixtures"], fixtures)
        for field in ("parent_sha256", "fixture_sha256", "adapter_config_sha256", "freeze_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(record)
                changed[field] = "0" * 64
                self.write(baseline, changed)
                with self.assertRaises((ValueError, RuntimeError)):
                    freezer.validate_baseline(baseline)


if __name__ == "__main__":
    unittest.main()
