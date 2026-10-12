"""Teacher TRAIN allowlist, grammar, delayed verification and corpus gates."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from dsv3wm.data import make_record
from dsv3wm.domains import rules_for, sha256
from dsv3wm.planning import verify_plan
from dsv3wm.recovery_planning import batched_search

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_recovery_teacher.py"
spec = importlib.util.spec_from_file_location("_dsv3wm_teacher_tests", SCRIPT)
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)


def fixture(identifier, x, goal, limit):
    row = make_record("register", {"x": x}, ["add1"] * 4, rules_for("register", limit=limit))
    task = {"id": identifier, "split": "train", "domain": "register", "rules": row["rules"], "s0": row["s0"],
            "parent_world_id": row["parent_world_id"], "goal": {"kind": "state", "state": {"x": goal}},
            "provenance": {"source_trajectory_id": row["id"]}}
    return row, task


class ToyPredictor:
    def __init__(self, events):
        self.traces, self.events = [], events

    def token_length(self, inputs):
        return 7

    def predict_batch(self, inputs):
        self.events.append("model")
        outputs = []
        for item in inputs:
            x, action = item["s0"]["x"], item["actions"][0]
            after = {"add1": x + 1, "sub1": x - 1, "double": 2 * x, "halve": x // 2, "finish": x}[action]
            legal = abs(after) <= item["rules"]["limit"] and (action != "halve" or x % 2 == 0)
            parsed = {"state": {"x": after if legal else x}, "legal": legal,
                      "terminated": action == "finish", "failed": not legal}
            outputs.append(parsed)
            self.traces.append({"conditioning": copy.deepcopy(item), "parsed": copy.deepcopy(parsed),
                                "generated": {"n_tokens": 2, "hit_eos": True, "text": "toy state"}})
        return outputs


class RecoveryTeacherTest(unittest.TestCase):
    def setUp(self):
        pairs = [fixture("one", 10, 13, 32), fixture("two", 20, 23, 40)]
        self.rows, self.tasks = [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    def test_train_requires_explicit_split_and_frozen_row_allowlist(self):
        teacher.validate_train_allowlist(self.tasks, self.rows)
        with self.assertRaises(PermissionError):
            batched_search(self.tasks, ToyPredictor([]))
        with self.assertRaises(PermissionError):
            teacher.validate_train_allowlist([dict(self.tasks[0], split="dev")], self.rows)
        with self.assertRaises(PermissionError):
            teacher.validate_train_allowlist(self.tasks, [dict(self.rows[0], split="confirmation")])
        changed = copy.deepcopy(self.tasks)
        changed[0]["provenance"]["source_trajectory_id"] = "unapproved-row"
        with self.assertRaises(ValueError):
            teacher.validate_train_allowlist(changed, self.rows)

    def test_all_searches_and_rankings_finish_before_verification(self):
        events, frozen_rankings = [], []
        predictor = ToyPredictor(events)
        def checkpoint(search, ranked):
            self.assertEqual(set(events), {"model"})
            events.append("rankings-fixed")
            frozen_rankings.append(sha256(ranked))
        def delayed_verifier(inputs, goal, plan):
            self.assertIn("rankings-fixed", events)
            self.assertEqual(events.count("model"), 6)
            events.append("verify")
            return verify_plan(inputs, goal, plan)
        corpus = teacher.collect_corpus(self.tasks, self.rows, predictor, teacher_provenance={"mock": True},
                                        verifier=delayed_verifier, on_rankings_fixed=checkpoint)
        self.assertEqual(events[:6], ["model"] * 6)
        self.assertNotIn("model", events[events.index("verify"):])
        self.assertEqual(frozen_rankings, [corpus["rankings_sha256"]])
        self.assertEqual(len(corpus["verified"]), 2)
        self.assertFalse(corpus["gate"]["passed"])
        self.assertTrue(corpus["hard_negatives"])
        self.assertTrue(all(not row["verified"] and not row["eligible_for_student_ce"] for row in corpus["hard_negatives"]))
        for row in corpus["verified"]:
            self.assertEqual(row["plan"], ["add1"] * 3 + ["finish"])
            self.assertEqual(row["source_plan"], ["add1"] * 3)
            self.assertEqual(json.loads(row["target"]), {"actions": row["plan"]})
            self.assertTrue(row["verification"]["labels"]["terminated"][-1])
            self.assertTrue(row["provenance"]["all_searches_and_rankings_fixed_before_verification"])
            source = next(item for item in self.rows if item["parent_world_id"] == row["parent_world_id"])
            self.assertEqual(row["provenance"]["training_input_sha256"], sha256(source))

    def test_completion_is_preverification_and_overlength_or_interior_finish_is_rejected(self):
        _, task = fixture("one", 10, 13, 32)
        def trace(prefix, action):
            return {"task_id": "one", "parent_world_id": task["parent_world_id"], "parent_plan": prefix, "action": action,
                    "raw": {"parsed": {"state": {"x": 13}, "legal": True, "terminated": False, "failed": False}}}
        search = {"oracle_feedback": False, "all_choices_fixed_before_scoring": True,
                  "traces": [trace(["add1", "add1"], "add1"), trace(["finish", "add1"], "add1"),
                             trace(["add1"] * 5, "add1")]}
        ranked = teacher.rank_computed_candidates([task], search)["one"]
        ordinary = next(row for row in ranked if row["source_plan"] == ["add1"] * 3)
        self.assertEqual(ordinary["plan"], ["add1"] * 3 + ["finish"])
        self.assertTrue(ordinary["eligible_for_positive"])
        self.assertEqual({row["grammar_refusal"] for row in ranked if row["grammar_refusal"]},
                         {"interior_finish", "overlength_completed_plan"})

    def test_non_predicted_goal_success_is_diagnostic_not_positive_or_hard_negative(self):
        task = self.tasks[0]
        # An already-complete learned candidate whose predicted endpoint is
        # wrong must remain diagnostic when independent execution succeeds.
        search = {"oracle_feedback": False, "all_choices_fixed_before_scoring": True, "budget": {"max_depth": 6},
                  "choices": [{"task_id": "one", "model_queries": 1}], "traces": [
                      {"task_id": "one", "parent_world_id": task["parent_world_id"],
                       "parent_plan": ["add1"] * 3, "action": "finish",
                       "raw": {"parsed": {"state": {"x": 14}, "legal": True, "terminated": True, "failed": False}}}]}
        with patch("dsv3wm.recovery_planning.batched_search", return_value=search):
            corpus = teacher.collect_corpus(self.tasks[:1], self.rows, ToyPredictor([]), teacher_provenance={"mock": True})
        self.assertFalse(corpus["verified"])
        self.assertTrue(any(row["reason"] == "teacher_did_not_predict_legal_goal" for row in corpus["verified_unretained"]))
        self.assertTrue(all(not row["verified"] for row in corpus["hard_negatives"]))

    def test_non_goal_prediction_never_gets_finish_completion_or_positive_admission(self):
        predictor = ToyPredictor([])
        original = predictor.predict_batch
        def wrong_goal_claim(inputs):
            outputs = original(inputs)
            for output, trace in zip(outputs, predictor.traces[-len(outputs):]):
                if output["state"]["x"] == 13:
                    output["state"]["x"] = 14
                    trace["parsed"] = copy.deepcopy(output)
            return outputs
        predictor.predict_batch = wrong_goal_claim
        corpus = teacher.collect_corpus(self.tasks[:1], self.rows, predictor, teacher_provenance={"mock": True})
        source = ["add1"] * 3
        candidate = next(row for row in corpus["ranked_candidates"]["one"] if row["source_plan"] == source)
        self.assertFalse(candidate["predicted_goal"])
        self.assertEqual(candidate["plan"], source)
        self.assertEqual(candidate["completion_origin"], "no-completion-learned-goal-not-predicted")
        self.assertFalse(candidate["eligible_for_positive"])
        rejected = next(row for row in corpus["hard_negatives"] if row["source_plan"] == source)
        self.assertEqual(rejected["plan"], source)
        self.assertTrue(rejected["verification"]["success"])  # goal-only contract
        self.assertFalse(rejected["verification"]["labels"]["terminated"][-1])
        self.assertFalse(rejected["verified"])  # H3 requires final finish
        self.assertFalse(any(row["source_plan"] == source for row in corpus["verified"]))

    def test_gate_requires_parent_limit_and_sequence_diversity(self):
        positives = [{"split": "train", "verified": True, "parent_world_id": f"p{index}",
                      "rules": {"limit": 24 if index % 2 else 32},
                      "plan": [["add1", "finish"], ["sub1", "finish"], ["double", "finish"], ["halve", "finish"]][index % 4]}
                     for index in range(16)]
        self.assertTrue(teacher.corpus_gate(positives)["passed"])
        self.assertFalse(teacher.corpus_gate(positives[:15])["passed"])
        repeated = [dict(row, plan=["add1", "finish"]) for row in positives]
        self.assertFalse(teacher.corpus_gate(repeated)["criteria"]["distinct_action_sequences"])
        with self.assertRaises(ValueError):
            teacher.corpus_gate([dict(positives[0], split="dev")])


if __name__ == "__main__":
    unittest.main()
