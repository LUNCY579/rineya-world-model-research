"""Pure recorded-label reward/regret cases; no model, torch or simulator."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze_recovery_terminal_regret as audit
from dsv3wm.native import sha256
from dsv3wm.recovery import RecoveryConfig


def task(index=0):
    state, goal = {"x": index}, {"kind": "state", "state": {"x": index + 3}}
    rules = {"semantics": "bwm-register-v2", "limit": 32 if index < 6 else 80, "text": "synthetic fixture rules"}
    value = {"domain": "register", "split": "dev", "rules": rules, "s0": state, "goal": goal,
             "provenance": {"source_trajectory_id": "fixture-" + str(index)}}
    value["parent_world_id"] = audit.prior.object_hash({"domain": "register", "s0": state, "actions": [],
                                                       "rules": {"semantics": rules["semantics"], "limit": rules["limit"]}})
    value["id"] = audit.prior.object_hash({key: value[key] for key in ("domain", "rules", "s0", "goal")})
    return value


def row(source, actions=None, xs=None, legal=None, terminated=None, executed=None):
    actions = ["add1"] * 3 if actions is None else actions
    xs = [source["s0"]["x"] + i + 1 for i in range(len(actions))] if xs is None else xs
    legal = [True] * len(actions) if legal is None else legal
    terminated = [False] * len(actions) if terminated is None else terminated
    executed = [True] * len(actions) if executed is None else executed
    final = {"x": xs[-1]} if xs else copy.deepcopy(source["s0"])
    all_legal = all(legal) and all(executed)
    success = all_legal and final == source["goal"]["state"]
    labels = {"states": [{"x": x} for x in xs], "legal": legal, "terminated": terminated, "executed": executed}
    verification = {"verified": success, "success": success, "all_legal": all_legal, "final_state": final,
                    "labels": labels, "verifier_sha256": {"bwm/code/bwmn/rw2.py": audit.RW2_SHA}}
    count = sum(action != "finish" for action in actions) if success else None
    return {"task_id": source["id"], "parent_world_id": source["parent_world_id"],
            "cohort": audit.prior.COHORTS[0 if source["rules"]["limit"] == 32 else 1], "choice": copy.deepcopy(actions),
            "predicted_state": final, "success": success, "all_legal": all_legal, "optimal_moves": 3,
            "independently_verified_optimal_moves": 3, "actual_move_cost": count,
            "normalized_excess_cost": (count - 3) / 3 if success else None, "optimal_success": success and count == 3,
            "first_action_present": bool(actions), "first_action_optimal": bool(actions), "verification": verification,
            **{key: 0 for key in audit.prior.COUNTERS}}


def record(arm, tasks, panel=None):
    panel = [row(source) for source in tasks] if panel is None else panel
    configuration = RecoveryConfig(arm="A0" if arm == "B0" else arm, steps=96 if arm == "A5" else 32).dict()
    value = {"status": "development-evaluated", "instrument": "legacy learned batches/W1/depth6/query30/dev-only",
             "oracle_feedback": False, "confirmation_access": False, "all_choices_fixed_before_scoring": True,
             "config": configuration, "budget": copy.deepcopy(audit.prior.BUDGET), "per_parent": panel,
             "adapter_sha256": audit.prior.INITIAL_F_SHA if arm == "B0" else "b" * 64,
             "source_sha256": {"synthetic-source": "c" * 64}, "summary": {}}
    for key in ("config_sha256", "tasks_sha256", "scoring_sha256", "adapter_config_sha256", "freeze_sha256", "traces_sha256"):
        value[key] = "a" * 64
    for cohort in (*audit.prior.COHORTS, "all"):
        selected = [item for item in panel if cohort == "all" or item["cohort"] == cohort]
        value["summary"][cohort] = {"n": len(selected), "successes": sum(item["success"] for item in selected),
            "queries": sum(item["model_queries"] for item in selected),
            **{key: sum(item[key] for item in selected) for key in audit.prior.COST_FIELDS if key != "model_queries"}}
    return value


class TerminalRegretTests(unittest.TestCase):
    def test_goal_only_success_is_preserved_but_has_no_terminal_reward(self):
        source, original = task(), row(task())
        before = copy.deepcopy(original)
        result = audit.score_case(source, original)
        self.assertTrue(result["original_P2_goal_only_success"])
        self.assertFalse(result["terminal_goal_success"])
        self.assertAlmostEqual(result["source_terminal_return"], -.03)
        self.assertAlmostEqual(result["normalized_terminal_regret"], (.96 + .03) / 1.62)
        self.assertEqual(original, before)

    def test_optimal_finish_and_wrong_goal_finish_count_actual_finish_step(self):
        source = task()
        complete = row(source, ["add1"] * 3 + ["finish"], [1, 2, 3, 3], terminated=[False] * 3 + [True])
        correct = audit.score_case(source, complete)
        self.assertTrue(correct["terminal_goal_success"]); self.assertEqual(correct["executed_steps"], 4)
        self.assertAlmostEqual(correct["source_terminal_return"], .96)
        self.assertAlmostEqual(correct["normalized_terminal_regret"], 0.)
        wrong = row(source, ["add1", "add1", "finish"], [1, 2, 2], terminated=[False, False, True])
        result = audit.score_case(source, wrong)
        self.assertFalse(result["terminal_goal_success"])
        self.assertEqual(result["executed_steps"], 3)
        self.assertAlmostEqual(result["source_terminal_return"], -.03)
        self.assertIsNone(result["original_success_conditional_move_cost"])

    def test_early_finish_ignores_suffix_but_does_not_change_goal_only_aggregate(self):
        source = task()
        original = row(source, ["add1", "add1", "add1", "finish", "double"], [1, 2, 3, 3, 3],
                       legal=[True] * 4 + [False], terminated=[False] * 3 + [True, True], executed=[True] * 4 + [False])
        result = audit.score_case(source, original)
        self.assertFalse(result["original_P2_goal_only_success"]); self.assertFalse(result["original_P2_all_legal"])
        self.assertTrue(result["terminal_goal_success"])
        self.assertEqual(result["ignored_after_finish"], 1); self.assertEqual(result["executed_illegal_steps"], 0)
        self.assertAlmostEqual(result["source_terminal_return"], .96)

    def test_illegal_executed_noop_penalty_and_universal_lower_bound(self):
        source = task(1); source["goal"]["state"]["x"] = 5
        illegal = row(source, ["halve", "double", "double", "add1", "finish"], [1, 2, 4, 5, 5],
                      legal=[False, True, True, True, True], terminated=[False] * 4 + [True])
        result = audit.score_case(source, illegal)
        self.assertTrue(result["terminal_goal_success"]); self.assertFalse(result["original_P2_goal_only_success"])
        self.assertEqual(result["executed_illegal_steps"], 1); self.assertAlmostEqual(result["source_terminal_return"], .85)
        lower = row(source, ["halve"] * 6, [1] * 6, legal=[False] * 6)
        bounded = audit.score_case(source, lower)
        self.assertAlmostEqual(bounded["source_terminal_return"], -.66)
        self.assertAlmostEqual(bounded["normalized_terminal_regret"], 1.)

    def test_recorded_empty_is_valid_zero_execution_but_none_is_not_imputed(self):
        source = task(); empty = row(source, [], [])
        result = audit.score_case(source, empty)
        self.assertEqual(result["executed_steps"], 0); self.assertEqual(result["source_terminal_return"], 0.)
        absent = {**empty, "choice": None, "verification": None}
        with self.assertRaises(ValueError): audit.score_case(source, absent)

    def test_missing_and_wrong_types_or_stopping_invariants_refuse(self):
        source = task()
        base = row(source, ["add1", "finish", "double"], [1, 1, 1],
                   legal=[True, True, False], terminated=[False, True, True], executed=[True, True, False])
        mutations = []
        for field, index, value in (("executed", 0, False), ("terminated", 0, True), ("terminated", 2, False),
                                     ("legal", 1, False), ("legal", 2, True), ("executed", 2, True),
                                     ("states", 2, {"x": 2}), ("legal", 0, 1), ("states", 0, {"x": True})):
            changed = copy.deepcopy(base); changed["verification"]["labels"][field][index] = value; mutations.append(changed)
        changed = copy.deepcopy(base); changed["verification"]["labels"]["states"] = []; mutations.append(changed)
        changed = copy.deepcopy(base); changed["verification"]["verified"] = True; mutations.append(changed)
        for value in mutations:
            with self.assertRaises(ValueError): audit.score_case(source, value)
        wrong_noop = row(source, ["halve"], [1], legal=[False])
        with self.assertRaisesRegex(ValueError, "no-ops"): audit.score_case(source, wrong_noop)

    def test_all_case_panels_and_paired_bootstrap_include_original_failure(self):
        tasks = [task(index) for index in range(12)]
        records = {arm: record(arm, tasks) for arm in audit.prior.ARMS}
        failed = row(tasks[-1], ["add1", "finish"], [12, 12], terminated=[False, True])
        records["A5"] = record("A5", tasks, [*records["A5"]["per_parent"][:-1], failed])
        before = copy.deepcopy(records)
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no simulator")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no replay")):
            output = audit.analyze(records, tasks, tasks_sha256="a" * 64, repetitions=100)
        self.assertEqual(records, before)
        self.assertEqual(output["summaries"]["A5"]["all"]["n"], 12)
        self.assertEqual(output["summaries"]["A5"]["all"]["original_P2_successes"], 11)
        pair = output["paired_all_cases"]["B0_to_A5"]["all"]
        self.assertEqual(pair["n"], 12)
        self.assertIn(tasks[-1]["parent_world_id"], pair["all_case_parent_ids"])
        self.assertFalse(pair["candidate_minus_baseline"]["normalized_terminal_regret"]["training_seed_uncertainty_estimated"])
        self.assertIn(ROOT / "prereg/POSTHOC_TERMINAL_REWARD_REGRET_V1.md", audit.source_paths())

    def test_arm_namespace_and_existing_output_refuse_before_reading_any_inputs(self):
        valid = [arm + "=fixture" for arm in audit.prior.ARMS]
        self.assertEqual(set(audit.parse_arms(valid)), set(audit.prior.ARMS))
        for specs in (valid + ["A1=other"], valid[:-1], [*valid[:-1], "A3=other"]):
            with self.assertRaises(ValueError): audit.parse_arms(specs)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.json"; output.write_bytes(b"immutable")
            argv = ["analyze_recovery_terminal_regret.py", "--tasks", "missing", "--arm", "B0=missing", "--out", str(output)]
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError): audit.main()
            self.assertEqual(output.read_bytes(), b"immutable")

    def test_hash_or_output_namespace_tamper_is_refused(self):
        tasks = [task(index) for index in range(12)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tasks_path = directory / "planning_dev.jsonl"
            tasks_path.write_text("".join(audit.prior.canonical(value) + "\n" for value in tasks))
            config = directory / "config.json"; config.write_text(json.dumps(RecoveryConfig().dict()))
            result_path = directory / "B0_planning.json"
            data = record("B0", tasks); data["tasks_sha256"] = sha256(tasks_path)
            data["config_sha256"] = "0" * 64
            data["command"] = [str(ROOT / "scripts/evaluate_recovery_planning.py"), "--config", str(config),
                               "--adapter", str(directory / "adapter"), "--tasks", str(tasks_path),
                               "--scoring", str(directory / "scoring.json"), "--freeze", str(directory / "freeze.json"), "--out", str(result_path)]
            result_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "digest drift"):
                audit.load_record(result_path, tasks_path, tasks, "B0")
            data["command"][-1] = str(directory / "other_output.json"); result_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "namespace"):
                audit.load_record(result_path, tasks_path, tasks, "B0")


if __name__ == "__main__":
    unittest.main()
