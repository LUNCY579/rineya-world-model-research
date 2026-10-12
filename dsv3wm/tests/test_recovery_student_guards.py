"""Pure CPU mocks: causal schedule, delayed scoring, paired guards, source freeze."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dsv3wm.native import sha256
from dsv3wm.recovery import legacy_parse
from dsv3wm.recovery_student import StudentConfig

ROOT = Path(__file__).resolve().parents[1]
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

guard = module("h3_guard_test_reference", "evaluate_recovery_student_guards.py")
freezer = module("h3_guard_freezer_test_reference", "freeze_recovery_student_guards.py")


class ForbiddenLabels(dict):
    def __getitem__(self, key):
        raise AssertionError("no labels during generation")


class MockPredictor:
    """Explicit deterministic fixture, not an oracle or native efficacy model."""
    def __init__(self, events, *, illegal_first=False, malformed_first=False, terminate_first=False):
        self.events, self.traces = events, []
        self.illegal_first, self.malformed_first, self.terminate_first = illegal_first, malformed_first, terminate_first
        self.batch_count = 0
    def predict_batch(self, inputs):
        self.batch_count += 1
        self.events.append("dynamics")
        outputs = []
        for item in inputs:
            x = item["s0"]["x"]
            for action in item["actions"]:
                x += 1 if action == "add1" else -1
            special = self.batch_count == 1 and item["s0"]["x"] == 0
            reply = {"x": x, "legal": not (special and self.illegal_first),
                     "terminated": special and self.terminate_first}
            text = "bad mock JSON" if special and self.malformed_first else json.dumps(reply)
            generated = {"text": text, "n_tokens": 12, "hit_eos": True}
            parsed = legacy_parse(text)
            self.traces.append({"conditioning": copy.deepcopy(item), "generated": generated, "parsed": parsed})
            outputs.append(parsed)
        return outputs


class StudentGuardTests(unittest.TestCase):
    def rows(self):
        rows = []
        for index in range(24):
            x = 2 * index
            rows.append({"id": f"source-{index}", "parent_world_id": f"parent-{index}", "split": "dev", "domain": "register",
                         "rules": {"limit": 96}, "s0": {"x": x}, "actions": ["add1", "sub1", "add1", "sub1"],
                         "provenance": {"cohort": "recovery_id_teacher_known" if index < 12 else "recovery_ood_limit"},
                         "labels": {"states": [{"x": x + 1}, {"x": x}, {"x": x + 1}, {"x": x}],
                                    "legal": [True] * 4, "terminated": [False] * 4, "failed": [False] * 4}})
        return rows

    def fixtures(self):
        return [{"id": f"fixture-{index}", "domain": "register" if index < 3 else "ledger",
                 "prompt": f"historical-mock-{index}", "messages": [{"role": "user", "content": str(index)}],
                 "target": '{"x":1}'} for index in range(6)]

    def behavior(self, events, *, wrong_first=False):
        def generate(requests):
            events.append("behavior")
            self.assertEqual(len(requests), 6)
            self.assertTrue(all("target" not in row for row in requests))
            return {"replies": [{"text": '{"x":2}' if wrong_first and index == 0 else '{"x":1}',
                                 "n_tokens": 5, "hit_eos": True} for index in range(6)],
                    "model_queries": 6, "input_tokens": 60, "output_tokens": 30}
        return generate

    def generated(self, rows=None, **predictor_options):
        rows = self.rows() if rows is None else rows
        events = []
        generated = guard.generate_guards(rows, self.fixtures(), MockPredictor(events, **predictor_options), self.behavior(events))
        return generated, events

    def test_all_generation_before_score_no_future_labels_or_oracle(self):
        rows, fixtures, events = self.rows(), self.fixtures(), []
        original_labels = [row["labels"] for row in rows]
        for row in rows:
            row["labels"] = ForbiddenLabels()
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.native.load", side_effect=AssertionError("no native")):
            generated = guard.generate_guards(rows, fixtures, MockPredictor(events), self.behavior(events))
            self.assertEqual(events, ["dynamics"] * 7 + ["behavior"])
            for row, labels in zip(rows, original_labels):
                row["labels"] = labels
            with patch.object(guard, "reference_scorer", wraps=guard.reference_scorer) as scorer:
                result = guard.score_guards(rows, fixtures, generated)
                scorer.assert_called_once()
        self.assertEqual(result["summary"]["all"]["4"]["rollout_correct"], 24)
        self.assertEqual(result["behavior_correct"], 6)
        self.assertEqual(len(generated["dynamics_traces"]), 168)
        self.assertEqual(generated["dynamics_traces"][24]["phase"], "rollout")
        self.assertEqual(generated["dynamics_traces"][24]["step"], 2)
        self.assertEqual(generated["dynamics_traces"][24]["conditioning"]["s0"], {"x": 1})
        self.assertIsNone(result["retention_pass"])
        self.assertIn("unmeasured", result["calibration"])

    def test_attempted_failure_accumulates_instead_of_using_last_legal(self):
        rows = self.rows()
        generated, _ = self.generated(rows, illegal_first=True)
        result = guard.score_guards(rows, self.fixtures(), generated)
        steps = result["per_parent"][0]["steps"]
        self.assertTrue(generated["rollouts"][0][-1]["legal"])
        self.assertEqual([step["predicted_cumulative_attempted_failure"] for step in steps], [True] * 4)
        self.assertFalse(steps[-1]["cumulative_failure_correct"])

    def test_malformed_and_terminated_rows_never_get_oracle_repair(self):
        for options in ({"malformed_first": True}, {"terminate_first": True}):
            with self.subTest(options=options):
                generated, _ = self.generated(**options)
                row_traces = [row for row in generated["dynamics_traces"] if row["source_index"] == 0 and row["phase"] == "rollout"]
                self.assertEqual(len(row_traces), 1)
                result = guard.score_guards(self.rows(), self.fixtures(), generated)
                steps = result["per_parent"][0]["steps"]
                self.assertFalse(steps[-1]["attempted"])
                if options.get("malformed_first"):
                    self.assertIsNone(steps[-1]["predicted_cumulative_attempted_failure"])
                else:
                    self.assertFalse(steps[-1]["predicted_cumulative_attempted_failure"])

    def test_corrupt_raw_reply_or_parent_tags_refuse_scoring(self):
        generated, _ = self.generated()
        for key, value in (("parent_world_id", "another-parent"), ("source_index", 3)):
            bad = copy.deepcopy(generated)
            bad["dynamics_traces"][0][key] = value
            with self.assertRaisesRegex(ValueError, "tags"):
                guard.score_guards(self.rows(), self.fixtures(), bad)
        bad = copy.deepcopy(generated)
        bad["dynamics_traces"][0]["generated"]["text"] = '{"x":999,"legal":true,"terminated":false}'
        with self.assertRaisesRegex(ValueError, "parsed output"):
            guard.score_guards(self.rows(), self.fixtures(), bad)

    def test_source_admission_rejects_confirmation_cohort_and_changed_bytes(self):
        bad = self.rows(); bad[0]["split"] = "confirmation"
        with self.assertRaisesRegex(ValueError, "DEV"):
            guard.validate_rows(bad)
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "dev.jsonl"; fixtures = Path(temporary) / "fixtures.json"
            data.write_text("replacement data"); fixtures.write_text("replacement fixtures")
            with patch("dsv3wm.native.load", side_effect=AssertionError("no native")), self.assertRaisesRegex(ValueError, "bytes changed"):
                guard.load_fixed_inputs(data, fixtures)
        actual = guard.load_fixed_inputs(ROOT / "datasets/recovery-v1/dev.jsonl", ROOT / "results/retention_core_v1.fixtures.json")
        self.assertEqual(len(actual[0]), 24)
        self.assertEqual(len(actual[1]), 6)

    def test_paired_losses_match_actual_warm_parent_and_preserve_unqualified_cases(self):
        generated, _ = self.generated()
        outcomes = guard.score_guards(self.rows(), self.fixtures(), generated)
        baseline = {"schema": guard.SCHEMA, "status": "development-guards-evaluated", "warm_baseline": True,
                    "artifact_arm": "A2", "adapter_sha256": "warm-a2", "data_sha256": guard.DEV_SHA,
                    "fixtures_sha256": guard.FIXTURE_SHA, "outcomes": outcomes,
                    "oracle_feedback": False, "all_generation_fixed_before_scoring": True}
        current = copy.deepcopy(baseline)
        current["outcomes"]["per_parent"][0]["horizons"]["4"]["rollout_correct"] = False
        current["outcomes"]["behavior"][0]["correct"] = False
        paired = guard.paired_guards(current, baseline, warm_arm="A2", warm_adapter_sha="warm-a2")
        self.assertEqual(paired["id_h4_net_loss"], 1)
        self.assertEqual(paired["behavior"]["losses"], ["fixture-0"])
        self.assertTrue(paired["behavior_losses_require_explanation"])
        with self.assertRaisesRegex(ValueError, "warm parent"):
            guard.paired_guards(current, baseline, warm_arm="A0", warm_adapter_sha="warm-a0")
        baseline["all_generation_fixed_before_scoring"] = False
        with self.assertRaisesRegex(ValueError, "warm parent"):
            guard.paired_guards(current, baseline, warm_arm="A2", warm_adapter_sha="warm-a2")

    def test_freezer_reuses_full_student_identity_and_pins_fixed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cfg = StudentConfig(arm="A3", plan_weight=0, control_extra_one_step=True)
            warm, config = directory / "warm", directory / "control-config.json"
            (warm / "adapter").mkdir(parents=True)
            for path in (warm / "run.json", warm / "adapter/adapter_model.safetensors", warm / "adapter/adapter_config.json"):
                path.write_text("synthetic frozen fixture bytes")
            config.write_text(json.dumps(cfg.dict()))
            record = {"config": {"arm": "A2"}}
            selected = [warm / "run.json", warm / "adapter/adapter_model.safetensors", warm / "adapter/adapter_config.json"]
            core = directory / "core.json"; core.write_text("synthetic core already verified fixture")
            with patch("evaluate_recovery_student_guards.admit_artifact", return_value=(record, selected, "warm-parent-proposer-baseline")), \
                 patch("freeze_recovery_student_readout.validate_core", return_value=[core]), \
                 patch("dsv3wm.native.load", side_effect=AssertionError("no native")) as load:
                result = freezer.build_freeze(student_dirs=[], warm_dirs=[warm], warm_configs=[config], train=directory / "train.jsonl",
                    data=ROOT / "datasets/recovery-v1/dev.jsonl", fixtures=ROOT / "results/retention_core_v1.fixtures.json", core=directory)
                load.assert_not_called()
            self.assertEqual(result["artifacts"][0]["kind"], "warm-A2")
            paths = result["files"]
            for path in (config, core, ROOT / "datasets/recovery-v1/dev.jsonl", ROOT / "results/retention_core_v1.fixtures.json",
                         ROOT / "scripts/evaluate_recovery_student_guards.py", ROOT / "scripts/evaluate_recovery_retention.py",
                         ROOT / "scripts/freeze_recovery_student_readout.py"):
                self.assertEqual(paths[str(path.resolve()).replace("\\", "/")], sha256(path))
            with patch("dsv3wm.recovery_student.admit_readout_lineage", side_effect=ValueError("full trained configuration mismatch")), \
                 self.assertRaisesRegex(ValueError, "full trained"):
                guard.admit_artifact(warm / "adapter", warm / "run.json", StudentConfig(arm="A3"))


if __name__ == "__main__":
    unittest.main()
