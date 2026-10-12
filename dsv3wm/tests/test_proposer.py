import json
import random
import unittest
from unittest.mock import patch
from dsv3wm.proposer import generate_plans, parse_plan, score_plans, validate_task
from dsv3wm.domains import rules_for,parent_world_key


class ProposerTests(unittest.TestCase):
    def setUp(self):
        self.task = {"split": "dev", "parent_world_id": "fresh-dev-parent", "domain": "register",
                     "rules": rules_for("register"), "s0": {"x": 0}, "goal": {"kind": "state", "state": {"x": 2}}}
        self.task['parent_world_id']=parent_world_key(self.task['domain'],self.task['s0'],self.task['rules'])

    def test_forged_parent_cannot_hide_duplicate_world(self):
        with self.assertRaises(ValueError):validate_task({**self.task,'parent_world_id':'different-label-same-world'})

    def test_all_generation_before_verifier_no_oracle_rerank(self):
        events, texts = [], iter(['["sub1","sub1"]', '["add1","add1"]'])
        def mock(prompt, seed, cap):
            events.append("generate")
            return {"text": next(texts), "n_tokens": 4, "input_tokens": 10, "model_calls": 1}
        with patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle during generation")):
            generated = generate_plans(self.task, mock, k=2)
        from dsv3wm.planning import verify_plan
        def verifier(*args):
            events.append("verify")
            return verify_plan(*args)
        with patch("dsv3wm.planning.verify_plan", side_effect=verifier):
            result = score_plans(self.task, generated)
        self.assertEqual(events, ["generate", "generate", "verify", "verify"])
        self.assertFalse(result["top1_success"])
        self.assertTrue(result["pass_at_k"])
        self.assertEqual(result["selected_plan"], ["sub1", "sub1"])
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["unique_valid_plans"], 2)
        self.assertEqual(result["model_queries"], 2)
        self.assertEqual(result["output_tokens"], 8)

    def test_seed_reproducibility_and_exact_training_prompt(self):
        from dsv3wm.planning import planning_prompt
        prompts = []
        def deterministic_mock(prompt, seed, cap):
            prompts.append(prompt)
            return {"text": json.dumps([random.Random(seed).choice(["add1", "sub1"])]), "n_tokens": 2}
        first = generate_plans(self.task, deterministic_mock, k=4, base_seed=5)
        second = generate_plans(self.task, deterministic_mock, k=4, base_seed=5)
        self.assertEqual(first, second)
        self.assertEqual(len({row["seed"] for row in first["candidates"]}), 4)
        self.assertEqual(prompts[0], planning_prompt(validate_task(self.task), self.task["goal"]))

    def test_strict_syntax_does_not_oracle_check_legality(self):
        for text in ['{"plan":[]}', '```json\n[]\n```', '[1]', '["teleport"]', '[] trailing']:
            self.assertTrue(parse_plan(text, "register", self.task["rules"])["malformed"])
        # Odd halving and an unsupported stack are syntactically valid and must
        # remain candidates until the separate exact scorer rejects them.
        self.assertFalse(parse_plan('["halve"]', "register", self.task["rules"])["malformed"])
        blocks = rules_for("blocks", objects=["a", "b"])
        self.assertFalse(parse_plan('["stack a b"]', "blocks", blocks)["malformed"])
        self.assertTrue(parse_plan('["stack a c"]', "blocks", blocks)["malformed"])
        ledger = rules_for("ledger", objects=["A", "B"])
        valid = '[{"kind":"add","account":"A","amount":1,"status":"posted"}]'
        self.assertFalse(parse_plan(valid, "ledger", ledger)["malformed"])
        for invalid in [valid.replace('"amount":1', '"amount":true'), valid.replace('"amount":1', '"amount":NaN'),
                        valid.replace('"amount":1', '"amount":1,"amount":2')]:
            self.assertTrue(parse_plan(invalid, "ledger", ledger)["malformed"])

    def test_dev_only_and_malformed_top1_never_falls_back(self):
        with self.assertRaises(ValueError):
            validate_task({**self.task, "split": "confirm"})
        with self.assertRaises(ValueError):
            validate_task({**self.task, "labels": {"states": []}})
        texts = iter(['bad', '["add1","add1"]'])
        generated = generate_plans(self.task, lambda *args: {"text": next(texts)}, k=2)
        result = score_plans(self.task, generated)
        self.assertFalse(result["top1_success"])
        self.assertTrue(result["pass_at_k"])
        self.assertIsNone(result["selected_plan"])


if __name__ == "__main__":
    unittest.main()
