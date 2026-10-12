"""Fresh R/P fixture, exposure, strict schema and late-score boundaries; no model."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsv3wm.domains import _semantics

def module(name, filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

builder = module("_rcwm_retention_builder_tests", "build_rcwm_retention.py")
evaluator = module("_rcwm_retention_evaluator_tests", "evaluate_rcwm_retention.py")


class RCWMRetentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = builder.build_fixtures(234)

    def test_fresh_deterministic_original_contract_and_shared_graph_groups(self):
        env = _semantics("ledger")
        with patch.object(env, "load_marginals", side_effect=AssertionError("historical marginal access")), \
             patch.object(env, "fds_oracles", side_effect=AssertionError("historical oracle access")):
            self.assertEqual(builder.build_fixtures(234), self.document)
        self.assertEqual(self.document["family_counts"], {"register_machine": 12, "prerequisite_plan": 12})
        self.assertEqual(self.document["parent_counts"], {"register_machine": 12, "prerequisite_plan": 12})
        self.assertEqual(self.document["graph_group_counts"], {"register_machine": 1, "prerequisite_plan": 1})
        for row in self.document["fixtures"]:
            self.assertEqual(row["messages"][0]["content"], env.SYSTEM_ONE[row["family"]])
            self.assertEqual(json.loads(row["messages"][1]["content"]), row["input"])
            self.assertEqual(row["original_train_input_exposure"], "unknown")
            self.assertTrue(row["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
            if row["family"] == "prerequisite_plan":
                self.assertTrue(env.applicable(row["input"]["before"], row["input"]["event"]))

    def test_numerical_conditional_overlap_and_noop_counterexamples(self):
        rows = {row["cohort"]: row for row in self.document["fixtures"]}
        positive = rows["positive_modular_wrap"]
        self.assertEqual(json.loads(positive["target"])["X"], positive["input"]["before"]["X"] + 17 - 97)
        negative = rows["negative_modular_wrap"]
        self.assertEqual(json.loads(negative["target"])["Z"], negative["input"]["before"]["Z"] - 9 + 97)
        equality = rows["conditional_equality_false"]
        self.assertEqual(json.loads(equality["target"])["Y"], (equality["input"]["before"]["Y"] + 4) % 97)
        true = rows["conditional_true"]
        self.assertEqual(json.loads(true["target"])["X"], (true["input"]["before"]["X"] + 7) % 97)
        for cohort in ("conditional_zero_value_noop", "add_zero_noop", "self_swap_noop",
                       "delete_then_add_overlap_noop", "empty_effect_noop", "add_existing_bit_noop", "delete_absent_bit_noop"):
            self.assertEqual(json.loads(rows[cohort]["target"]), rows[cohort]["input"]["before"])

    def test_sha_bound_train_input_keys_ignore_labels_and_nuisance_tokens(self):
        original = self.document["fixtures"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t2_train.jsonl"
            row = {"family": original["family"], **original["input"],
                   "target": {"poisoned_old_label_never_used": "DO NOT USE"}}
            raw = (json.dumps(row) + "\n").encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            audit = builder.training_input_audit(path, expected_sha=digest)
            rebuilt = builder.build_fixtures(234, audit=audit)
            self.assertNotIn(original["semantic_input_id"], {item["semantic_input_id"] for item in rebuilt["fixtures"]})
            self.assertTrue(all(item["original_train_input_exposure"] == "audited_semantically_novel"
                                for item in rebuilt["fixtures"]))
            self.assertFalse(audit["source"]["labels_used"])
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                builder.training_input_audit(path)
            with self.assertRaises(PermissionError):
                builder.training_input_audit(path.with_name("sealed_eval.jsonl"), expected_sha=digest)
        p = next(row for row in self.document["fixtures"] if row["family"] == "prerequisite_plan")
        altered = dict(p["input"]["event"], name="new_name", cost=999, tick=456)
        self.assertEqual(builder.input_key(p["family"], p["input"]["before"], p["input"]["event"]),
                         builder.input_key(p["family"], p["input"]["before"], altered))

    def test_strict_bare_integer_and_register_schema_no_repair(self):
        self.assertEqual(evaluator.strict_state("prerequisite_plan", " 17\n"), 17)
        for text in ("true", "17.0", '"17"', '{"after":17}', "17 explanation", "NaN", "256", "-1"):
            self.assertIsNone(evaluator.strict_state("prerequisite_plan", text))
        self.assertEqual(evaluator.strict_state("register_machine", '{"X":1,"Y":2,"Z":3}'), {"X": 1, "Y": 2, "Z": 3})
        for text in ('{"X":1,"Y":2,"Z":3,"extra":4}', '{"X":1,"X":2,"Y":2,"Z":3}',
                     '{"X":true,"Y":2,"Z":3}', '{"X":NaN,"Y":2,"Z":3}', '{"X":97,"Y":2,"Z":3}',
                     '```json\n{"X":1,"Y":2,"Z":3}\n```'):
            self.assertIsNone(evaluator.strict_state("register_machine", text))

    def test_all_generation_finishes_before_frozen_target_scoring(self):
        events = []
        class WatchedFixture(dict):
            def __getitem__(self, key):
                if key == "target":
                    self_test.assertEqual(events[0], "all-generation-finished")
                    events.append("score")
                return super().__getitem__(key)
        self_test = self
        fixtures = [WatchedFixture(row) for row in self.document["fixtures"]]
        def generate(prompts):
            self.assertEqual(len(prompts), 24)
            self.assertFalse(events)
            events.append("all-generation-finished")
            return [{"text": "0", "n_tokens": 1, "hit_eos": True} for _ in prompts]
        with patch.object(_semantics("ledger"), "step", side_effect=AssertionError("oracle during inference")):
            replies, scored = evaluator.generate_then_score(fixtures, generate)
        self.assertEqual(len(replies), len(scored))
        self.assertEqual(events, ["all-generation-finished", *(["score"] * 24)])

    def test_paired_fixture_parent_and_graph_identity_is_enforced(self):
        fixtures = self.document["fixtures"]
        replies = [{"text": row["target"], "n_tokens": 2, "hit_eos": True} for row in fixtures]
        rows = evaluator.score_all(fixtures, replies)
        baseline = {"status": "development-evaluated", "fixture_sha256": "frozen", "parent_sha256": "parent",
                    "inference_contract": {"max_new_tokens": 64}, "rows": rows}
        current = copy.deepcopy(baseline)
        current["rows"][0]["correct"] = False
        pairs = evaluator.paired_changes(current, baseline)
        self.assertTrue(pairs[0]["loss"])
        self.assertFalse(any(row["gain"] for row in pairs))
        changed = copy.deepcopy(current)
        changed["rows"][0]["graph_group_id"] = "wrong-graph"
        with self.assertRaises(ValueError):
            evaluator.paired_changes(changed, baseline)


if __name__ == "__main__":
    unittest.main()
