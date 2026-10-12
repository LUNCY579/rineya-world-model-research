import copy
import unittest
from unittest.mock import patch

from dsv3wm.domains import rules_for, parent_world_key
from dsv3wm.planning import (PlanningBudget, action_library, distill_ranked, learned_beam_search,
                             rank_candidates, verify_distillation_row, verify_plan)
from dsv3wm.stats import clustered_bootstrap, holm, paired_mcnemar, wilson_interval


class DeterministicRegisterMock:
    """CPU fixture only: a deliberately imperfect fake, not native efficacy."""
    def __init__(self, events=None):
        self.events = events if events is not None else []

    def __call__(self, conditioning):
        self.events.append("learned")
        x, action = conditioning["s0"]["x"], conditioning["actions"][0]
        # Deliberate lie: sub1 also adds one, yielding a verifier-rejected plan.
        if action in {"add1", "sub1"}:
            return {"state": {"x": x + 1}, "legal": True, "terminated": False, "failed": False}
        return {"malformed": True}


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.conditioning = {"domain": "register", "rules": rules_for("register"),
                             "s0": {"x": 0}, "actions": []}
        self.goal = {"kind": "state", "state": {"x": 2}}
        self.row = {**self.conditioning, "split": "train", "parent_world_id": parent_world_key('register', self.conditioning['s0'], self.conditioning['rules'])}

    def test_search_and_ranking_have_no_exact_executor(self):
        with patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("oracle forbidden")):
            ranked = rank_candidates(self.conditioning, self.goal, [["add1", "add1"], ["sub1", "sub1"]],
                                     DeterministicRegisterMock(), PlanningBudget(max_model_calls=4))
            beam = learned_beam_search(self.conditioning, self.goal, DeterministicRegisterMock(),
                                      PlanningBudget(max_model_calls=6, beam_width=2), actions=["add1", "sub1"])
        self.assertEqual(ranked["model_calls"], 4)
        self.assertFalse(ranked["oracle_feedback"])
        self.assertEqual(beam["choice"], ["add1", "add1"])

    def test_verification_follows_learned_ranking_and_retains_negatives(self):
        events = []
        ranked = rank_candidates(self.conditioning, self.goal, [["add1", "add1"], ["sub1", "sub1"]],
                                 DeterministicRegisterMock(events))
        original = verify_plan
        def logged_verify(*args):
            events.append("exact")
            return original(*args)
        with patch("dsv3wm.planning.verify_plan", side_effect=logged_verify):
            retained = distill_ranked(self.row, self.goal, ranked)
        self.assertEqual(events, ["learned"] * 4 + ["exact"] * 2)
        self.assertEqual(len(retained["verified"]), 1)
        self.assertEqual(len(retained["hard_negatives"]), 1)
        self.assertTrue(verify_distillation_row(retained["verified"][0]))
        self.assertTrue(verify_distillation_row(retained['verified'][0], allowed_training_rows=[self.row]))
        self.assertFalse(verify_distillation_row(retained['verified'][0], allowed_training_rows=[]))
        self.assertFalse(verify_distillation_row(retained['verified'][0], allowed_training_rows=[{**self.row,'split':'dev'}]))
        self.assertFalse(verify_distillation_row(retained["hard_negatives"][0]))
        tampered = copy.deepcopy(retained["verified"][0])
        tampered["plan"] = ["sub1", "sub1"]
        self.assertFalse(verify_distillation_row(tampered))
        tampered = copy.deepcopy(retained["verified"][0])
        tampered["split"] = "confirm"
        self.assertFalse(verify_distillation_row(tampered))
        tampered = copy.deepcopy(retained["verified"][0])
        tampered["target"] = '["add1"]'
        self.assertFalse(verify_distillation_row(tampered))
        tampered = copy.deepcopy(retained['verified'][0])
        tampered['parent_world_id'] = 'forged'
        self.assertFalse(verify_distillation_row(tampered))

    def test_budget_and_malformed_fail_closed(self):
        ranked = rank_candidates(self.conditioning, self.goal, [["add1", "add1"]],
                                 DeterministicRegisterMock(), PlanningBudget(max_model_calls=1))
        self.assertIsNone(ranked["choice"])
        self.assertEqual(ranked["model_calls"], 1)
        bad = rank_candidates(self.conditioning, self.goal, [["double"]], DeterministicRegisterMock())
        self.assertIsNone(bad["choice"])
        self.assertEqual(bad["malformed_predictions"], 1)
        with self.assertRaises(ValueError):
            distill_ranked({**self.row, "split": "dev"}, self.goal, bad)

    def test_exact_verifies_all_three_domains(self):
        register = verify_plan(self.conditioning, self.goal, ["add1", "add1"])
        self.assertTrue(register["verified"])
        ledger = {"domain": "ledger", "rules": rules_for("ledger", objects=["A", "B"]),
                  "s0": {"A": 0, "B": 0}, "actions": []}
        event = {"kind": "add", "account": "A", "amount": 2, "status": "posted"}
        self.assertTrue(verify_plan(ledger, {"kind": "state", "state": {"A": 2, "B": 0}}, [event])["verified"])
        blocks = {"domain": "blocks", "rules": rules_for("blocks", objects=["a", "b"]),
                  "s0": {"clear": ["a", "b"], "table": ["a", "b"], "on": [], "hand": "empty"}, "actions": []}
        goal = {"kind": "facts", "facts": [["on", "a", "b"], ["handempty"]]}
        self.assertTrue(verify_plan(blocks, goal, ["pick-up a", "stack a b"])["verified"])
        self.assertFalse(verify_plan(blocks, goal, ["stack a b"])["verified"])
        self.assertIn("unstack a b", action_library(blocks))

    def test_absorbing_finish_rejects_extra_actions(self):
        zero = {"kind": "state", "state": {"x": 0}}
        self.assertTrue(verify_plan(self.conditioning, zero, ["finish"])["verified"])
        self.assertFalse(verify_plan(self.conditioning, zero, ["finish", "add1"])["verified"])


class StatsTests(unittest.TestCase):
    def test_exact_known_counts(self):
        outcome = paired_mcnemar([False] * 9, [True] * 9)
        self.assertEqual(outcome["gains"], 9)
        self.assertAlmostEqual(outcome["p_two_sided"], 2 / 512)
        self.assertAlmostEqual(outcome["p_one_sided_improvement"], 1 / 512)
        self.assertEqual(paired_mcnemar([True, False], [True, False])["p_two_sided"], 1)
        reverse = paired_mcnemar([True] * 9, [False] * 9)
        self.assertEqual(reverse["p_one_sided_improvement"], 1)
        with self.assertRaises(ValueError):
            paired_mcnemar([1], [True])

    def test_wilson_known_interval_and_holm(self):
        lower, upper = wilson_interval(5, 10)
        self.assertAlmostEqual(lower, .236593, places=5)
        self.assertAlmostEqual(upper, .763407, places=5)
        values = holm({"a": .01, "b": .04, "c": .03})
        self.assertAlmostEqual(values["a"]["adjusted_p"], .03)
        self.assertAlmostEqual(values["c"]["adjusted_p"], .06)
        self.assertAlmostEqual(values["b"]["adjusted_p"], .06)
        self.assertTrue(values["a"]["reject"])
        self.assertFalse(values["b"]["reject"])

    def test_crossed_bootstrap_aggregates_parents_and_seeds(self):
        rows = [{"parent_world_id": parent, "seed": seed, "baseline": 0., "candidate": seed}
                for seed in (-1, 1) for parent in ("p", "q")]
        result = clustered_bootstrap(rows, repetitions=300, random_seed=7)
        self.assertEqual(result, clustered_bootstrap(rows, repetitions=300, random_seed=7))
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["ci"], [-1, 1])
        # Duplicating a parent trajectory does not increase its cell's weight.
        duplicate = rows + [copy.deepcopy(rows[0])] * 20
        self.assertEqual(clustered_bootstrap(duplicate, repetitions=300)["difference"], 0)
        with self.assertRaises(ValueError):
            clustered_bootstrap(rows[:-1], repetitions=100)


if __name__ == "__main__":
    unittest.main()
