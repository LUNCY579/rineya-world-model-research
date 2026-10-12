import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "scripts/analyze_recovery_traces.py"
spec = importlib.util.spec_from_file_location("recovery_posthoc_analysis", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def trace(state, action, x, legal=True, terminated=False, text=None):
    text = text if text is not None else json.dumps({"x": x, "legal": legal, "terminated": terminated})
    return {"conditioning": {"domain": "register", "rules": {"limit": 10}, "s0": {"x": state}, "actions": [action]},
            "generated": {"text": text, "n_tokens": 10, "hit_eos": True}, "parsed": analysis.parse_raw(text)}


class RecoveryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"split": "dev", "domain": "register", "parent_world_id": "p-id", "s0": {"x": 1},
                         "rules": {"limit": 10}, "actions": ["halve", "add1", "add1", "add1"],
                         "provenance": {"cohort": "recovery_id_teacher_known"},
                         "labels": {"states": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}], "legal": [False, True, True, True],
                                    "terminated": [False] * 4, "failed": [True] * 4, "executed": [True] * 4}},
                        {"split": "dev", "domain": "register", "parent_world_id": "p-ood", "s0": {"x": 6},
                         "rules": {"limit": 10}, "actions": ["finish", "double", "add1", "halve"],
                         "provenance": {"cohort": "recovery_ood_limit"},
                         "labels": {"states": [{"x": 6}] * 4, "legal": [True, False, False, False],
                                    "terminated": [True] * 4, "failed": [False] * 4, "executed": [True, False, False, False]}}]
        self.traces = [trace(1, "halve", 9, False), trace(6, "finish", 6, True, True),
                       trace(9, "add1", 2), trace(2, "add1", 3), trace(3, "add1", 4),
                       trace(1, "halve", 1, False), trace(6, "finish", 6, True, True)]

    def test_cumulative_failure_is_not_last_legal_and_no_oracle(self):
        with patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")):
            result = analysis.analyze(self.records, self.traces)
        first, second = result["per_parent"]
        self.assertTrue(first["P1_final_state_correct"])
        self.assertTrue(first["correct_final_wrong_intermediate"])
        self.assertTrue(first["steps"][3]["failed"])
        self.assertFalse(self.traces[4]["parsed"]["failed"])
        self.assertTrue(first["steps"][3]["cumulative_failure_correct"])
        self.assertFalse(second["steps"][3]["failed"])
        self.assertTrue(second["steps"][3]["skipped"])
        self.assertFalse(second["steps"][3]["attempted"])
        self.assertEqual(result["calibration"]["status"], "unknown")

    def test_corrupt_trace_and_unknown_order_are_rejected(self):
        corrupt = copy.deepcopy(self.traces)
        corrupt[2]["conditioning"]["s0"] = {"x": 1}  # Would be a gold reset.
        with self.assertRaises(ValueError):
            analysis.analyze(self.records, corrupt)
        corrupt = copy.deepcopy(self.traces)
        corrupt[0]["parsed"]["state"]["x"] = 1
        with self.assertRaises(ValueError):
            analysis.analyze(self.records, corrupt)
        with self.assertRaises(ValueError):
            analysis.analyze(self.records, self.traces + [self.traces[0]])
        with self.assertRaises(ValueError):
            analysis.analyze(list(reversed(self.records)), self.traces)

    def test_malformed_abstention_keeps_unobserved_failure_unknown(self):
        traces = [trace(1, "halve", 0, text="bad"), trace(6, "finish", 6, True, True),
                  trace(1, "halve", 1, False), trace(6, "finish", 6, True, True)]
        result = analysis.analyze(self.records, traces)
        values = result["per_parent"][0]["steps"]
        self.assertIsNone(values[0]["failed"])
        self.assertTrue(values[1]["abstained"])
        self.assertIsNone(values[3]["failed"])
        self.assertFalse(result["per_parent"][0]["P1_final_state_correct"])
        self.assertEqual(result["summary"]["id"]["steps"]["4"]["failure_unknown"], 1)

    def test_confirmation_labels_and_missing_trace_refused(self):
        rows = copy.deepcopy(self.records)
        rows[0]["split"] = "confirmation"
        with self.assertRaises(ValueError):
            analysis.analyze(rows, self.traces)
        with self.assertRaises(ValueError):
            analysis.analyze(self.records, self.traces[:-1])

    def test_identical_inputs_with_conflicting_outputs_reject_mapping(self):
        rows = [copy.deepcopy(self.records[0]), copy.deepcopy(self.records[0])]
        rows[1]["parent_world_id"] = "distinct-reported-parent"
        rows[1]["provenance"]["cohort"] = "recovery_ood_limit"
        traces = [trace(1, "halve", 9, False), trace(1, "halve", 1, False)]
        with self.assertRaisesRegex(ValueError, "uncertain parent mapping"):
            analysis.analyze(rows, traces)


if __name__ == "__main__":
    unittest.main()
