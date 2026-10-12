"""Recorded-measurement fixtures for cost denominators, missing regret and pairs."""
import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_recovery_planning_costs.py"
spec = importlib.util.spec_from_file_location("_recovery_planning_cost_tests", SCRIPT)
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


def task_bank():
    tasks = []
    for index in range(12):
        limit, x = (32, index + 1) if index < 6 else (80, index + 20)
        rules = {"semantics": "bwm-register-v2", "limit": limit, "text": "explicit test rules"}
        row = {"split": "dev", "domain": "register", "rules": rules, "s0": {"x": x},
               "goal": {"kind": "state", "state": {"x": x + 3}}, "provenance": {"source_trajectory_id": f"test{index}"}}
        row["parent_world_id"] = analysis.object_hash({"domain": "register", "s0": row["s0"], "actions": [],
                                                     "rules": {"semantics": rules["semantics"], "limit": limit}})
        row["id"] = analysis.object_hash({key: row[key] for key in ("domain", "rules", "s0", "goal")})
        tasks.append(row)
    return tasks


def record(arm, tasks):
    rows = []
    for index, task in enumerate(tasks):
        queries, cost = 15, 3
        rows.append({"parent_world_id": task["parent_world_id"], "task_id": task["id"],
                     "cohort": analysis.COHORTS[0 if index < 6 else 1], "success": True, "all_legal": True,
                     "optimal_success": True, "first_action_optimal": True, "first_action_present": True,
                     "optimal_moves": 3, "independently_verified_optimal_moves": 3,
                     "choice": ["add1"] * cost, "actual_move_cost": cost, "normalized_excess_cost": 0.0,
                     "query_attempts": queries, "model_queries": queries, "input_tokens": queries * 10,
                     "output_tokens": queries * 2, "unpadded_processed_positions_upper_bound": queries * 12,
                     "processed_positions_upper_bound": queries * 14, "malformed_predictions": 0,
                     "illegal_predictions": 0, "overlength_refusals": 0, "budget_refusals": 0})
    result = {"status": "development-evaluated", "instrument": "legacy learned batches/W1/depth6/query30/dev-only",
              "oracle_feedback": False, "confirmation_access": False, "all_choices_fixed_before_scoring": True,
              "config": {"arm": "A0" if arm == "B0" else arm, "seed": 0, "device": "cpu", "core": "titan-sparse",
                         "rank": 16, "max_length": 512, "generation_tokens": 32},
              "budget": copy.deepcopy(analysis.BUDGET), "per_parent": rows,
              "source_sha256": {"explicit_fixture_source.py": "1" * 64},
              "tasks_sha256": analysis.object_hash(tasks)}
    for key in ("config_sha256", "scoring_sha256", "adapter_sha256", "adapter_config_sha256", "freeze_sha256", "traces_sha256"):
        result[key] = "a" * 64
    if arm == "B0":
        result["adapter_sha256"] = analysis.INITIAL_F_SHA
    refresh_summary(result)
    return result


def refresh_summary(result):
    result["summary"] = {}
    for cohort in (*analysis.COHORTS, "all"):
        rows = [row for row in result["per_parent"] if cohort == "all" or row["cohort"] == cohort]
        result["summary"][cohort] = {"n": len(rows), "successes": sum(row["success"] for row in rows),
                                    "queries": sum(row["model_queries"] for row in rows),
                                    **{key: sum(row[key] for row in rows) for key in
                                       ("query_attempts", "input_tokens", "output_tokens", "processed_positions_upper_bound")}}


def mark_failure(row):
    row.update(success=False, all_legal=False, optimal_success=False, actual_move_cost=None,
               normalized_excess_cost=None, choice=None, first_action_present=False, first_action_optimal=False)


class PlanningCostsTest(unittest.TestCase):
    def setUp(self):
        self.tasks = task_bank()
        self.baseline, self.candidate = record("A0", self.tasks), record("A1", self.tasks)

    def test_failure_cost_is_included_in_calls_per_solved_but_regret_stays_null(self):
        rows = copy.deepcopy(self.baseline["per_parent"][:2])
        mark_failure(rows[1])
        summary = analysis.summarize(rows)
        self.assertEqual(summary["solved"], 1)
        self.assertEqual(summary["totals_all_searches"]["model_queries"], 30)
        self.assertEqual(summary["model_queries_per_solved_including_failures"], 30)
        self.assertEqual(summary["success_conditional"]["mean_actual_move_cost"], 3)
        self.assertEqual(summary["success_conditional"]["mean_normalized_excess_cost"], 0)
        self.assertIsNone(rows[1]["actual_move_cost"])
        self.assertIsNone(rows[1]["normalized_excess_cost"])
        mark_failure(rows[0])
        none = analysis.summarize(rows)
        self.assertIsNone(none["model_queries_per_solved_including_failures"])
        self.assertIsNone(none["success_conditional"]["mean_normalized_excess_cost"])

    def test_failure_zero_imputation_and_wrong_metric_types_are_rejected(self):
        broken = copy.deepcopy(self.baseline)
        mark_failure(broken["per_parent"][0])
        broken["per_parent"][0]["normalized_excess_cost"] = 0
        refresh_summary(broken)
        with self.assertRaisesRegex(ValueError, "Failures must retain null"):
            analysis.validate_record(broken, self.tasks, tasks_sha256=broken["tasks_sha256"], arm="A0")
        for key, value in (("model_queries", True), ("actual_move_cost", 3.0), ("independently_verified_optimal_moves", 3.0)):
            changed = copy.deepcopy(self.baseline)
            changed["per_parent"][0][key] = value
            with self.assertRaises(ValueError):
                analysis.validate_record(changed, self.tasks, tasks_sha256=changed["tasks_sha256"], arm="A0")

    def test_paired_regret_and_cost_use_only_identical_common_success_parents(self):
        mark_failure(self.baseline["per_parent"][0])
        mark_failure(self.candidate["per_parent"][1])
        self.candidate["per_parent"][2].update(choice=["add1"] * 4, actual_move_cost=4,
                                               normalized_excess_cost=1 / 3, optimal_success=False)
        pairs = analysis.compare_common_success(self.baseline, self.candidate, repetitions=100, random_seed=9)
        all_cases = pairs["all"]
        self.assertEqual(all_cases["n"], 10)
        self.assertNotIn(self.tasks[0]["parent_world_id"], all_cases["common_success_parents"])
        self.assertNotIn(self.tasks[1]["parent_world_id"], all_cases["common_success_parents"])
        cost = all_cases["candidate_minus_baseline"]["actual_move_cost"]
        self.assertAlmostEqual(cost["difference"], .1)
        self.assertEqual(cost["seed_count"], 1)
        self.assertFalse(cost["seed_uncertainty_estimated"])
        regret = all_cases["candidate_minus_baseline"]["normalized_excess_cost"]
        self.assertAlmostEqual(regret["difference"], 1 / 30)
        self.assertEqual(pairs[analysis.COHORTS[0]]["n"], 4)

    def test_pairing_mismatch_and_no_common_success_are_handled(self):
        for key, value in (("tasks_sha256", "b" * 64), ("scoring_sha256", "b" * 64), ("budget", {})):
            changed = copy.deepcopy(self.candidate)
            changed[key] = value
            with self.assertRaises(ValueError):
                analysis.compare_common_success(self.baseline, changed, repetitions=100)
        changed = copy.deepcopy(self.candidate)
        changed["per_parent"][0]["task_id"] = "wrong-task"
        with self.assertRaises(ValueError):
            analysis.compare_common_success(self.baseline, changed, repetitions=100)
        for row in self.candidate["per_parent"]:
            mark_failure(row)
        result = analysis.compare_common_success(self.baseline, self.candidate, repetitions=100)
        self.assertEqual(result["all"]["n"], 0)
        self.assertTrue(all(value is None for value in result["all"]["candidate_minus_baseline"].values()))

    def test_actual_task_parent_and_aggregate_identity_must_match(self):
        analysis.validate_record(self.baseline, self.tasks, tasks_sha256=self.baseline["tasks_sha256"], arm="A0")
        wrong = copy.deepcopy(self.baseline)
        wrong["per_parent"][0]["parent_world_id"] = "bad-parent"
        with self.assertRaises(ValueError):
            analysis.validate_record(wrong, self.tasks, tasks_sha256=wrong["tasks_sha256"], arm="A0")
        wrong = copy.deepcopy(self.baseline)
        wrong["summary"]["all"]["queries"] -= 1
        with self.assertRaisesRegex(ValueError, "aggregate planning summary"):
            analysis.validate_record(wrong, self.tasks, tasks_sha256=wrong["tasks_sha256"], arm="A0")
        wrong = copy.deepcopy(self.tasks)
        wrong[0]["goal"]["state"]["x"] += 1
        with self.assertRaises(ValueError):
            analysis.validate_tasks(wrong)

    def test_all_five_arms_and_descriptive_parent_only_scope_are_required(self):
        records = {arm: record(arm, self.tasks) for arm in analysis.ARMS}
        result = analysis.analyze(records, self.tasks, tasks_sha256=self.baseline["tasks_sha256"], repetitions=100, random_seed=2)
        self.assertEqual(result["summaries"]["B0"]["all"]["solved"], 12)
        self.assertFalse(result["bootstrap"]["training_seed_uncertainty_estimated"])
        self.assertIn("A5_to_A2", result["paired_common_success"])
        with self.assertRaises(ValueError):
            analysis.analyze({"B0": records["B0"]}, self.tasks, tasks_sha256=self.baseline["tasks_sha256"], repetitions=100)


if __name__ == "__main__":
    unittest.main()
