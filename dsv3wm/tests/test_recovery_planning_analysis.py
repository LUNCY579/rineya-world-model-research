import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from dsv3wm.domains import rules_for, parent_world_key
from dsv3wm.recovery import INITIAL_F_SHA, legacy_parse

path = Path(__file__).resolve().parents[1] / "scripts/analyze_recovery_planning_traces.py"
spec = importlib.util.spec_from_file_location("planning_posthoc", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PlanningAuditTests(unittest.TestCase):
    def setUp(self):
        rules = rules_for("register", limit=32)
        parent = parent_world_key("register", {"x": 0}, rules)
        self.task = {"id": "dev-task", "split": "dev", "domain": "register", "rules": rules,
                     "s0": {"x": 0}, "goal": {"kind": "state", "state": {"x": 2}},
                     "parent_world_id": parent, "provenance": {"source": "pure unit fixture"}}
        def trace(prefix, action, before, after, terminated=False):
            text = json.dumps({"x": after, "legal": True, "terminated": terminated})
            return {"task_id": self.task["id"], "parent_world_id": parent, "parent_plan": prefix,
                    "action": action, "depth": len(prefix) + 1, "batch_index": len(prefix), "batch_size": 2 if not prefix else 1,
                    "prompt_tokens": 10, "output_tokens": 15, "padded_processed_positions_upper_bound": 25,
                    "raw": {"conditioning": {"domain": "register", "rules": rules, "s0": {"x": before}, "actions": [action]},
                            "generated": {"text": text, "n_tokens": 15, "hit_eos": True}, "parsed": legacy_parse(text)}}
        # Two wrong learned transitions compensate at the final goal. Query2's
        # own input is3; the true task trajectory's intermediate state is1.
        self.traces = [trace([], "add1", 0, 3), trace([], "finish", 0, 0, True), trace(["add1"], "add1", 3, 2)]
        self.native = {"status": "development-evaluated", "oracle_feedback": False,
                       "all_choices_fixed_before_scoring": True, "adapter_sha256": INITIAL_F_SHA,
                       "config": {"arm": "A0"}, "budget": {"max_model_calls": 30, "max_depth": 6, "max_processed_positions": 120000},
                       "preflight_proof": {"max_prompt_tokens": 512, "generation_tokens": 32},
                       "per_parent": [{"task_id": self.task["id"], "parent_world_id": parent,
                                       "choice": ["add1", "add1"], "predicted_state": {"x": 2}, "predicted_success": True,
                                       "success": True, "model_queries": 3, "input_tokens": 30,
                                       "output_tokens": 45, "processed_positions_upper_bound": 75}]}

    def test_own_input_truth_and_actual_ancestry_are_distinct_without_models(self):
        before = copy.deepcopy((self.task, self.traces, self.native))
        with patch("dsv3wm.native.load", side_effect=AssertionError("no native model")) as load:
            result = audit.analyze([self.task], self.traces, self.native)
            load.assert_not_called()
        self.assertEqual((self.task, self.traces, self.native), before)
        self.assertEqual(result["arm"], "B0")
        self.assertEqual(result["summary"]["queries"], 3)
        self.assertEqual(result["summary"]["state_error"], 2)
        self.assertEqual(result["summary"]["finish"]["term_correct"], 1)
        self.assertEqual(result["summary"]["nonfinish"]["term_correct"], 2)
        query2 = result["query_audits"][2]
        self.assertEqual(query2["expected_on_own_input"]["state"], {"x": 4})
        selected = result["selected_ancestry_audits"][0]
        self.assertEqual(selected["steps"][1]["actual_state"], {"x": 2})
        self.assertTrue(selected["steps"][1]["state_correct_on_actual_trajectory"])
        self.assertTrue(selected["correct_goal_wrong_intermediate"])
        self.assertTrue(selected["nonterminating_goal_only_success"])
        self.assertTrue(result["original_P2_unchanged"])

    def test_ambiguous_or_reset_ancestry_rejected_before_oracle(self):
        def forbidden(*args):
            raise AssertionError("uncertain mappings must not be relabelled by oracle")
        duplicate = copy.deepcopy(self.traces) + [copy.deepcopy(self.traces[0])]
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate"):
            audit.analyze([self.task], duplicate, self.native, exact_step=forbidden, exact_rollout=forbidden)
        corrupt = copy.deepcopy(self.traces)
        corrupt[2]["raw"]["conditioning"]["s0"] = {"x": 1}
        with self.assertRaisesRegex(ValueError, "ancestry mismatch"):
            audit.analyze([self.task], corrupt, self.native, exact_step=forbidden, exact_rollout=forbidden)
        corrupt = copy.deepcopy(self.traces)
        corrupt[0]["raw"]["parsed"]["state"] = {"x": 1}
        with self.assertRaisesRegex(ValueError, "parsing mismatch"):
            audit.analyze([self.task], corrupt, self.native, exact_step=forbidden, exact_rollout=forbidden)

    def test_counters_incomplete_results_and_confirmation_are_rejected(self):
        bad = copy.deepcopy(self.native)
        bad["per_parent"][0]["model_queries"] = 4
        with self.assertRaisesRegex(ValueError, "counters"):
            audit.analyze([self.task], self.traces, bad)
        bad["status"] = "started"
        with self.assertRaisesRegex(ValueError, "completed"):
            audit.analyze([self.task], self.traces, bad)
        bad_traces = copy.deepcopy(self.traces)
        bad_traces[0]["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "broad-batch"):
            audit.analyze([self.task], bad_traces, self.native)
        with self.assertRaises(PermissionError):
            audit.analyze([{**self.task, "split": "confirmation"}], self.traces, self.native)

    def test_wrong_original_P2_is_not_silently_recomputed(self):
        bad = copy.deepcopy(self.native)
        bad["per_parent"][0]["success"] = False
        with self.assertRaisesRegex(ValueError, "original goal-only P2"):
            audit.analyze([self.task], self.traces, bad)

    def test_finish_positive_class_error_is_visible(self):
        traces = copy.deepcopy(self.traces)
        text = json.dumps({"x": 0, "legal": True, "terminated": False})
        traces[1]["raw"]["generated"]["text"] = text
        traces[1]["raw"]["parsed"] = legacy_parse(text)
        result = audit.analyze([self.task], traces, self.native)
        self.assertEqual(result["summary"]["finish"]["term_error"], 1)
        self.assertEqual(result["summary"]["finish"]["queries"], 1)


if __name__ == "__main__":
    unittest.main()
