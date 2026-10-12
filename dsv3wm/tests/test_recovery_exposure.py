"""Pure synthetic lineage/statistical-unit tests; no native or exact execution."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_exposure_script", ROOT / "scripts/analyze_recovery_exposure.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(path):
    return analysis.Evidence().digest(path)


class RecoveryExposureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.dataset = self.home / "dataset"
        self.artifacts = self.home / "artifacts"
        self.results = self.home / "results"
        self.initial = self.home / "initial"
        self.initial.mkdir()
        self.dataset.mkdir()
        self.results.mkdir()
        (self.initial / "adapter_model.safetensors").write_bytes(b"synthetic initial bytes, not model tensors")
        initial_sha = digest(self.initial / "adapter_model.safetensors")
        pin = patch.object(analysis, "INITIAL_SHA", initial_sha)
        pin.start()
        self.addCleanup(pin.stop)
        peft = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "inference_mode": False, "auto_mapping": None}
        write_json(self.initial / "adapter_config.json", peft)

        def row(index, split, cohort, limit):
            value = {"domain": "register", "split": split, "rules": {"semantics": "bwm-register-v2", "limit": limit},
                     "s0": {"x": index}, "actions": ["add1", "sub1", "double", "halve"],
                     "provenance": {"cohort": cohort}}
            value["parent_world_id"] = analysis.parent_world_key("register", value["s0"], value["rules"])
            return value
        self.train = [row(i, "train", "recovery_train", 40) for i in range(128)]
        self.dev = [row(i, "dev", "recovery_id_teacher_known" if i < 12 else "recovery_ood_limit",
                        48 if i < 12 else 96) for i in range(24)]
        self.tasks = []
        for index in [*range(6), *range(12, 18)]:
            source = self.dev[index]
            task = {key: copy.deepcopy(source[key]) for key in ("domain", "split", "rules", "s0", "parent_world_id")}
            task["goal"] = {"kind": "state", "state": {"x": index + 2}}
            task["id"] = analysis.identity_hash({key: task[key] for key in ("domain", "rules", "s0", "goal")})
            self.tasks.append(task)
        for name, rows in (("train.jsonl", self.train), ("dev.jsonl", self.dev), ("planning_dev.jsonl", self.tasks)):
            (self.dataset / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(self.dataset / "parent_registry.json", {
            row["parent_world_id"]: {"split": row["split"], "cohort": row["provenance"]["cohort"]}
            for row in self.train + self.dev})
        write_json(self.dataset / "manifest.json", {"files": {
            name: {"sha256": digest(self.dataset / name)}
            for name in ("train.jsonl", "dev.jsonl", "planning_dev.jsonl", "parent_registry.json")}})
        self.qualification = self.results / "B0_qualification.json"
        write_json(self.qualification, {"status": "development-evaluated", "baseline_qualified": True,
                                      "oracle_feedback": False, "adapter_sha256": initial_sha, "freeze_sha256": "f" * 64})
        self.configs = {}
        for arm in analysis.ARMS:
            path = self.home / "configs" / f"{arm}.json"
            write_json(path, {"arm": arm, "steps": 96 if arm == "A5" else 32})
            self.configs[arm] = path
        self.training_freeze = self.home / "training_freeze.json"
        sources = [analysis.ROOT / "scripts/train_recovery.py", *[
            analysis.ROOT / f"src/dsv3wm/{name}.py" for name in ("__init__", "config", "recovery", "native", "domains", "stats")]]
        inputs = [*self.configs.values(), self.dataset / "train.jsonl", self.qualification,
                  self.initial / "adapter_model.safetensors", self.initial / "adapter_config.json", *sources]
        write_json(self.training_freeze, {"confirmation_access": False,
                                         "qualification_freeze_sha256": "f" * 64,
                                         "qualification_sha256": digest(self.qualification),
                                         "files": {str(path.resolve()): digest(path) for path in inputs}})
        self.runs = {}
        for arm in analysis.ARMS:
            directory = self.artifacts / f"{arm}_s0"
            adapter = directory / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(arm.encode())
            write_json(adapter / "adapter_config.json", dict(peft, inference_mode=True))
            config = analysis.RecoveryConfig.read(self.configs[arm])
            rng = random.Random(177)
            order = [rng.randrange(128) for _ in range(config.steps)]
            run = {"status": "pilot-trained", "config": config.dict(), "optimizer_steps": config.steps,
                   "command": ["train_recovery.py", "--config", str(self.configs[arm]),
                               "--train", str(self.dataset / "train.jsonl"), "--freeze", str(self.training_freeze),
                               "--qualification", str(self.qualification), "--out", str(directory)],
                   "config_sha256": digest(self.configs[arm]), "data_sha256": digest(self.dataset / "train.jsonl"),
                   "qualification_sha256": digest(self.qualification), "freeze_sha256": digest(self.training_freeze),
                   "initial_adapter_sha256": initial_sha, "core_sha256": analysis.CORE_SHA,
                   "adapter_sha256": digest(adapter / "adapter_model.safetensors"), "row_order": order,
                   "counts": {"forwards": len(order) * len(config.horizons), "transitions": len(order) * len(config.horizons)},
                   "optimizer": {"kind": "AdamW", "lr": config.lr}}
            write_json(directory / "run.json", run)
            self.runs[arm] = run
        self.readout_path = self.home / "readout_freeze.json"
        bound = [self.training_freeze, self.qualification, *self.configs.values(), *[
            self.dataset / name for name in ("train.jsonl", "dev.jsonl", "planning_dev.jsonl", "manifest.json", "parent_registry.json")]]
        for arm in analysis.ARMS:
            bound.extend(self.artifacts / f"{arm}_s0" / name for name in
                         ("run.json", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"))
        self.readout = {"confirmation_access": False, "inherited_adapter_sha256": initial_sha,
                        "runs": {arm: str((self.artifacts / f"{arm}_s0").resolve()) for arm in analysis.ARMS},
                        "files": {str(path.resolve()): digest(path) for path in bound}}
        # Deliberately nonexistent and forbidden reference: analysis must not follow it.
        self.readout["files"][str(self.home / "sealed/confirmation.labels.jsonl")] = "0" * 64
        write_json(self.readout_path, self.readout)
        for arm in analysis.ARMS:
            common = {"status": "development-evaluated", "config": self.runs[arm]["config"],
                      "oracle_feedback": False, "adapter_sha256": self.runs[arm]["adapter_sha256"],
                      "freeze_sha256": digest(self.readout_path)}
            dynamics = dict(common, data_sha256=digest(self.dataset / "dev.jsonl"), per_parent=[])
            for i, row in enumerate(self.dev):
                correct = i not in ({12, 13} if arm == "A1" else {14} if arm == "A5" else set())
                dynamics["per_parent"].append({"parent_world_id": row["parent_world_id"],
                                               "cohort": "id" if i < 12 else "ood",
                                               "horizons": {"4": {"rollout_correct": correct}}})
            write_json(self.results / f"{arm}_dynamics.json", dynamics)
            planning = dict(common, tasks_sha256=digest(self.dataset / "planning_dev.jsonl"),
                            confirmation_access=False, all_choices_fixed_before_scoring=True,
                            config_sha256=self.runs[arm]["config_sha256"],
                            adapter_config_sha256=digest(self.artifacts / f"{arm}_s0/adapter/adapter_config.json"), per_parent=[])
            dev_by_parent = {row["parent_world_id"]: row for row in self.dev}
            for i, task in enumerate(self.tasks):
                planning["per_parent"].append({"parent_world_id": task["parent_world_id"], "task_id": task["id"],
                                               "cohort": dev_by_parent[task["parent_world_id"]]["provenance"]["cohort"],
                                               "success": not (arm == "A5" and i == 11)})
            write_json(self.results / f"{arm}_planning.json", planning)

    def analyze(self):
        return analysis.analyze(self.dataset, self.artifacts, self.results, self.readout_path,
                                repetitions=100, initial_adapter=self.initial)

    def rebind_run(self, arm, run):
        path = self.artifacts / f"{arm}_s0/run.json"
        write_json(path, run)
        self.readout["files"][str(path.resolve())] = digest(path)
        write_json(self.readout_path, self.readout)

    def test_full_synthetic_analysis_counts_and_shared_units(self):
        result = self.analyze()
        expected = {"A0": (32, 29, 29), "A1": (32, 29, 29), "A2": (32, 29, 87), "A5": (96, 64, 64)}
        for arm, (draws, parents, targets) in expected.items():
            item = result["exposure"][arm]
            self.assertEqual((item["trajectory_draws"], item["distinct_parents"], item["unique_parent_horizon_target_instances"]),
                             (draws, parents, targets))
        paired = result["paired_vs_A5"]["A1"]["P1"]["all"]
        self.assertEqual((paired["n"], paired["gains"], paired["losses"]), (24, 1, 2))
        self.assertEqual(paired["parent_bootstrap_difference"]["parent_count"], 24)
        self.assertEqual(result["statistical_units"]["distinct_DEV_parents"], 24)
        self.assertEqual(result["statistical_units"]["shared_endpoint_parents"], 12)
        self.assertTrue(result["original_P1_P2_and_gates_unchanged"])
        self.assertNotIn(str(self.home / "sealed/confirmation.labels.jsonl"), result["provenance"])

    def test_no_model_or_oracle_execution(self):
        with patch("dsv3wm.native.load", side_effect=AssertionError("model load forbidden")) as model, \
             patch("dsv3wm.native.runtime", side_effect=AssertionError("runtime forbidden")) as runtime, \
             patch("dsv3wm.domains._semantics", side_effect=AssertionError("simulator import forbidden")) as semantics, \
             patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("oracle forbidden")) as oracle:
            self.analyze()
            for forbidden in (model, runtime, semantics, oracle):
                forbidden.assert_not_called()

    def test_weight_hash_drift_rejected(self):
        (self.artifacts / "A1_s0/adapter/adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash drift"):
            self.analyze()

    def test_unbound_inputs_rejected_before_opening(self):
        forbidden = self.home / "unbound/confirmation.labels.jsonl"
        evidence = analysis.Evidence()
        with self.assertRaisesRegex(ValueError, "absent from freeze"):
            evidence.bind({"files": {}}, forbidden)
        self.assertEqual(evidence.files, {})

    def test_recorded_data_hash_mismatch_rejected(self):
        run = copy.deepcopy(self.runs["A0"])
        run["data_sha256"] = "0" * 64
        self.rebind_run("A0", run)
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            self.analyze()

    def test_recorded_bool_int_alias_rejected(self):
        run = copy.deepcopy(self.runs["A0"])
        run["config"]["seed"] = False  # False == 0 must not pass config comparison.
        self.rebind_run("A0", run)
        with self.assertRaisesRegex(ValueError, "typed warm"):
            self.analyze()

    def test_draw_order_mutation_rejected_even_when_hash_rebound(self):
        run = copy.deepcopy(self.runs["A0"])
        run["row_order"][0] = (run["row_order"][0] + 1) % 128
        self.rebind_run("A0", run)
        with self.assertRaisesRegex(ValueError, "with-replacement schedule"):
            self.analyze()

    def test_bool_draw_index_and_horizon_count_mismatch_rejected(self):
        run = copy.deepcopy(self.runs["A2"])
        config = analysis.RecoveryConfig(**run["config"])
        run["row_order"][0] = True
        with self.assertRaisesRegex(ValueError, "typed in-range"):
            analysis.audit_draw_order(run, config, self.train)
        run = copy.deepcopy(self.runs["A2"])
        run["counts"]["transitions"] = 32
        with self.assertRaisesRegex(ValueError, "horizon-response count"):
            analysis.audit_draw_order(run, config, self.train)

    def test_canonical_parent_tampering_rejected(self):
        rows = copy.deepcopy(self.train)
        rows[0]["parent_world_id"] = "0" * 64
        path = self.home / "train_bad.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "canonical parent mismatch"):
            analysis.input_metadata(path, "train", 128)

    def test_repeated_parent_rows_cannot_inflate_evaluation_n(self):
        result = analysis.read_json(self.results / "A1_dynamics.json")
        meta = {row["parent_world_id"]: {"cohort": row["provenance"]["cohort"]} for row in self.dev}
        result["per_parent"].append(copy.deepcopy(result["per_parent"][0]))
        with self.assertRaisesRegex(ValueError, "pseudoreplication"):
            analysis.outcome_map(result, "P1", meta)
        result["per_parent"] = result["per_parent"][1:-1]
        with self.assertRaisesRegex(ValueError, "fixed parent set"):
            analysis.outcome_map(result, "P1", meta)

    def test_pairing_parent_and_cohort_mismatch_rejected(self):
        baseline = {"p": {"cohort": "id", "outcome": True}, "q": {"cohort": "ood", "outcome": False}}
        with self.assertRaisesRegex(ValueError, "paired parent"):
            analysis.compare_parents(baseline, {"p": baseline["p"]}, 100)
        candidate = copy.deepcopy(baseline)
        candidate["q"]["cohort"] = "id"
        with self.assertRaisesRegex(ValueError, "paired parent/cohort"):
            analysis.compare_parents(baseline, candidate, 100)

    def test_constant_success_ci_does_not_assert_equivalence(self):
        baseline = {"p": {"cohort": "id", "outcome": True}, "q": {"cohort": "ood", "outcome": True}}
        paired = analysis.compare_parents(baseline, baseline, 100)["all"]
        self.assertGreater(paired["baseline_wilson95"][1] - paired["baseline_wilson95"][0], 0)
        self.assertEqual(paired["parent_bootstrap_difference"]["ci"], [0, 0])
        self.assertIn("not equivalence", paired["bootstrap_scope"])

    def test_adapter_scaling_types_and_behavior_remain_bound(self):
        original = analysis.read_json(self.initial / "adapter_config.json")
        changed = dict(original, inference_mode=True, peft_version="changed")
        self.assertEqual(analysis.adapter_behavior(original), analysis.adapter_behavior(changed))
        self.assertNotEqual(analysis.adapter_behavior(original), analysis.adapter_behavior(dict(original, lora_alpha=32.0)))
        self.assertNotEqual(analysis.adapter_behavior(original), analysis.adapter_behavior(dict(original, auto_mapping={})))

    def test_cli_refuses_overwrite_before_reading_inputs(self):
        path = self.home / "existing.json"
        path.write_text("preserved", encoding="utf-8")
        with patch.object(sys, "argv", ["analyze_recovery_exposure.py", "--freeze", "missing", "--out", str(path)]):
            with self.assertRaises(FileExistsError):
                analysis.main()
        self.assertEqual(path.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
