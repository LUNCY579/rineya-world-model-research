"""Learned search, isolation, preflight budgets and delayed verifier boundaries."""
import copy
import unittest
from unittest.mock import patch

from dsv3wm.domains import parent_world_key, rules_for
from dsv3wm.recovery_planning import (RecoveryPlanningBudget, batched_search, score_choices, validate_tasks)


def task(identifier, x, goal, limit=96):
    rules = rules_for("register", limit=limit)
    return {"id": identifier, "split": "dev", "domain": "register", "rules": rules, "s0": {"x": x},
            "goal": {"kind": "state", "state": {"x": goal}},
            "parent_world_id": parent_world_key("register", {"x": x}, rules), "provenance": {"fresh": True}}


class MockBatchPredictor:
    """Deterministic toy predictor with raw token traces; no repository oracle."""
    def __init__(self, events=None, malformed=False, terminal=False):
        self.traces, self.batches = [], []
        self.events = events if events is not None else []
        self.malformed, self.terminal = malformed, terminal

    def token_length(self, inputs):
        return 10 if inputs["s0"]["x"] >= 20 else 7

    def predict_batch(self, inputs):
        self.events.append("model")
        self.batches.append(copy.deepcopy(inputs))
        result = []
        for item in inputs:
            x, action = item["s0"]["x"], item["actions"][0]
            if self.malformed:
                output = {"malformed": True, "text": "not a state"}
            else:
                value = {"add1": x + 1, "sub1": x - 1, "double": x * 2, "halve": x // 2, "finish": x}[action]
                legal = (action != "halve" or x % 2 == 0) and abs(value) <= item["rules"]["limit"]
                output = {"state": {"x": value if legal else x}, "legal": legal,
                          "terminated": self.terminal or action == "finish", "failed": not legal}
            result.append(output)
            self.traces.append({"conditioning": copy.deepcopy(item),
                                "generated": {"text": "mock raw state", "n_tokens": 2, "hit_eos": True},
                                "parsed": copy.deepcopy(output)})
        return result


class RecoveryPlanningTest(unittest.TestCase):
    def test_batched_independent_states_and_padded_token_accounting(self):
        tasks = [task("one", 10, 13), task("two", 20, 23)]
        before = copy.deepcopy(tasks)
        predictor = MockBatchPredictor()
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("oracle during search")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("oracle during search")), \
             patch("dsv3wm.planning.verify_plan", side_effect=AssertionError("verifier during search")):
            result = batched_search(tasks, predictor)
        self.assertEqual(tasks, before)
        self.assertEqual(len(predictor.batches[0]), 10)
        self.assertEqual([item["s0"]["x"] for item in predictor.batches[0]], [10] * 5 + [20] * 5)
        self.assertEqual([item["s0"]["x"] for item in predictor.batches[1]], [11] * 5 + [21] * 5)
        self.assertTrue(all(set(inputs) == {"domain", "rules", "s0", "actions"}
                            for batch in predictor.batches for inputs in batch))
        for choice in result["choices"]:
            self.assertLessEqual(choice["model_queries"], 30)
            self.assertLessEqual(choice["query_attempts"], 30)
            self.assertLessEqual(choice["processed_positions_upper_bound"], 120000)
            self.assertEqual(choice["output_tokens"], choice["model_queries"] * 2)
        self.assertEqual(result["preflight_proof"]["max_task_positions"], 16320)
        self.assertFalse(result["oracle_feedback"])
        self.assertTrue(result["all_choices_fixed_before_scoring"])
        self.assertTrue(all(trace["task_id"] in {"one", "two"} and "raw" in trace for trace in result["traces"]))

    def test_budget_refuses_before_model_and_query_cap_is_per_task(self):
        tasks = [task("one", 10, 13), task("two", 20, 23)]
        predictor = MockBatchPredictor()
        refused = batched_search(tasks, predictor, budget=RecoveryPlanningBudget(max_processed_positions=10))
        self.assertFalse(predictor.batches)
        self.assertTrue(all(choice["choice"] is None for choice in refused["choices"]))
        self.assertTrue(all(choice["budget_refusals"] == 5 for choice in refused["choices"]))
        bounded = batched_search(tasks, MockBatchPredictor(), budget=RecoveryPlanningBudget(max_model_calls=3))
        self.assertTrue(all(choice["model_queries"] == 3 for choice in bounded["choices"]))
        self.assertTrue(all(choice["query_attempts"] == 3 for choice in bounded["choices"]))
        self.assertTrue(all(choice["choice"] == ["add1"] for choice in bounded["choices"]))
        with self.assertRaises(ValueError):
            RecoveryPlanningBudget(beam_width=2)
        with self.assertRaises(ValueError):
            batched_search(tasks, MockBatchPredictor(), generation_tokens=33)

    def test_malformed_search_never_fabricates_an_empty_plan(self):
        predictor = MockBatchPredictor(malformed=True)
        result = batched_search([task("one", 10, 13)], predictor)
        self.assertEqual(len(predictor.batches), 1)
        self.assertIsNone(result["choices"][0]["choice"])
        self.assertEqual(result["choices"][0]["malformed_predictions"], 5)

    def test_predicted_termination_is_absorbing_and_finish_not_extended(self):
        predictor = MockBatchPredictor(terminal=True)
        result = batched_search([task("one", 10, 13)], predictor)
        self.assertEqual(len(predictor.batches), 1)
        self.assertTrue(all(len(trace["parent_plan"]) == 0 for trace in result["traces"]))
        self.assertEqual(len(result["choices"][0]["choice"]), 1)
        ordinary = batched_search([task("one", 10, 13)], MockBatchPredictor())
        self.assertTrue(all("finish" not in trace["parent_plan"] for trace in ordinary["traces"]))

    def test_all_verification_occurs_after_all_tasks_finish_searching(self):
        events, tasks = [], [task("one", 10, 13), task("two", 20, 23)]
        predictor = MockBatchPredictor(events=events)
        search = batched_search(tasks, predictor)
        choices = copy.deepcopy(search["choices"])
        self.assertTrue(events and set(events) == {"model"})
        metadata = [{"task_id": row["id"], "parent_world_id": row["parent_world_id"], "split": "dev",
                     "optimal_moves": 3, "cohort": "recovery_ood_limit"} for row in tasks]

        def delayed_verifier(inputs, goal, plan):
            self.assertEqual(search["choices"], choices)
            events.append("verify")
            self.assertEqual(inputs["actions"], [])
            self.assertEqual(plan, ["add1"] * 3)
            return {"success": True, "all_legal": True}

        scored = score_choices(tasks, search, metadata, verifier=delayed_verifier)
        self.assertEqual(events[-2:], ["verify", "verify"])
        self.assertFalse("model" in events[events.index("verify"):])
        self.assertTrue(all(row["success"] and row["optimal_success"] for row in scored))
        self.assertTrue(all(row["normalized_excess_cost"] == 0 for row in scored))
        self.assertTrue(all(row["first_action_optimal"] and row["first_action_remaining_distance"] == 2 for row in scored))
        with self.assertRaises(ValueError):
            score_choices(tasks, dict(search, all_choices_fixed_before_scoring=False), metadata, verifier=delayed_verifier)

    def test_independent_first_action_optimality_rejects_bad_and_missing_proposals(self):
        tasks = [task("one", 10, 13)]
        original = batched_search(tasks, MockBatchPredictor())
        metadata = [{"task_id": "one", "parent_world_id": tasks[0]["parent_world_id"], "split": "dev",
                     "optimal_moves": 3, "cohort": "recovery_ood_limit"}]
        detour = copy.deepcopy(original)
        detour["choices"][0]["choice"] = ["sub1", "add1", "add1", "add1", "add1"]
        scored = score_choices(tasks, detour, metadata)
        self.assertTrue(scored[0]["success"])
        self.assertFalse(scored[0]["first_action_optimal"])
        self.assertFalse(scored[0]["optimal_success"])
        self.assertEqual(scored[0]["actual_move_cost"], 5)
        self.assertEqual(scored[0]["first_action_remaining_distance"], 4)
        missing = copy.deepcopy(original)
        missing["choices"][0]["choice"] = None
        absent = score_choices(tasks, missing, metadata)[0]
        self.assertFalse(absent["first_action_optimal"])
        self.assertFalse(absent["first_action_present"])
        self.assertIsNone(absent["first_action_legal"])
        drift = [dict(metadata[0], optimal_moves=4)]
        with self.assertRaisesRegex(ValueError, "shortest distance disagrees"):
            score_choices(tasks, original, drift)

    def test_task_parent_and_privileged_input_boundaries(self):
        valid = task("one", 10, 13)
        validate_tasks([valid])
        contaminated = dict(valid, labels={"oracle": 13})
        with self.assertRaises(ValueError):
            validate_tasks([contaminated])
        with self.assertRaises(ValueError):
            validate_tasks([dict(valid, parent_world_id="wrong")])
        with self.assertRaises(PermissionError):
            validate_tasks([dict(valid, split="confirmation")])
        with self.assertRaises(ValueError):
            validate_tasks([valid, dict(valid, id="different-same-parent")])


if __name__ == "__main__":
    unittest.main()
