"""Pure constructed Block G1 and raw-before-support diagnostics; no native run."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_block_g1 import B, BlockG1Config, canonical_parent, construct_cases, diagnose_delta, score_case, summarize
from evaluate_block_g1 import validate_cases


class BlockG1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = construct_cases()
    def test_twelve_independent_primary_six_changed_six_noop(self):
        primary = self.cases[:12]
        self.assertEqual(len({case["parent_world_id"] for case in primary}), 12)
        self.assertEqual(sum(case["collection_legal"] for case in primary), 6)
        self.assertEqual({case["action"][0] for case in primary}, {"pick-up", "put-down", "stack", "unstack"})
        validate_cases(self.cases)
        for case in primary:
            facts = frozenset(tuple(f) for f in case["before"])
            after, legal = B().step(facts, tuple(case["action"]))
            self.assertEqual(case["prompt"], B().forward_prompt_delta(facts, tuple(case["action"])))
            self.assertEqual(case["target"], B().forward_target_delta(facts, after, legal))
            self.assertEqual(after != facts, legal)
    def test_diagnostics_are_separate_repeated_parent_variants(self):
        primary = {case["id"]: case for case in self.cases[:12]}
        self.assertEqual(len(self.cases[12:]), 4)
        for diagnostic in self.cases[12:]:
            base = primary[diagnostic["source_case_id"]]
            self.assertEqual(diagnostic["parent_world_id"], base["parent_world_id"])
            self.assertNotEqual(diagnostic["prompt"], base["prompt"])
            self.assertEqual(canonical_parent(frozenset(tuple(f) for f in diagnostic["before"]), diagnostic["objects"]), base["parent_world_id"])
    def test_correct_source_targets_pass_without_oracle_during_diagnostics(self):
        with patch.object(B(), "step", side_effect=AssertionError("oracle in diagnostic parser")):
            for case in self.cases:
                value = diagnose_delta(case["target"], case["before"], case["objects"])
                self.assertTrue(value["accepted"])
        outcomes = [score_case(case, case["target"]) for case in self.cases]
        self.assertTrue(summarize(outcomes)["g1_qualified"])
        self.assertEqual(summarize(outcomes)["primary"]["raw_legal_correct"], 12)
    def test_unsupported_deletion_keeps_raw_candidate_before_filter(self):
        case = self.cases[0]
        # Missing holding deletion is unsupported, but naive set subtraction is
        # still a no-op. It must never become accepted through oracle repair.
        raw = json.dumps({"add": [], "del": ["holding a"], "legal": True})
        value = diagnose_delta(raw, case["before"], case["objects"])
        self.assertTrue(value["format_valid"])
        self.assertEqual(value["raw_state_facts"], case["before"])
        self.assertEqual(value["raw_legal"], True)
        self.assertEqual(value["unsupported_deletions"], [["holding", "a"]])
        self.assertFalse(value["source_parser_accepted"]); self.assertFalse(value["accepted"])
    def test_structure_failure_is_not_collapsed_into_an_invented_valid_state(self):
        case = self.cases[0]
        raw = json.dumps({"add": ["holding a"], "del": [], "legal": True})
        value = diagnose_delta(raw, case["before"], case["objects"])
        self.assertTrue(value["support_ok"]); self.assertTrue(value["source_parser_accepted"])
        self.assertFalse(value["structural_state_valid"]); self.assertFalse(value["accepted"])
        self.assertIn(["holding", "a"], value["raw_state_facts"])
        self.assertIn(["handempty"], value["raw_state_facts"])
    def test_type_duplicate_and_prose_failures_abstain(self):
        case = self.cases[0]
        for raw in ('{"add":[],"del":[],"legal":1}', '{"add":[1],"del":[],"legal":true}',
                    '{"add":[],"del":[],"legal":true,"legal":false}',
                    '{"add":[],"del":[],"legal":true} prose', '{"add":["invented a"],"del":[],"legal":true}'):
            value = diagnose_delta(raw, case["before"], case["objects"])
            self.assertFalse(value["format_valid"]); self.assertFalse(value["accepted"])
            self.assertIsNotNone(value["type_or_format_failure"])
    def test_new_six_object_canonicalization_does_not_change_old_domains(self):
        from dsv3wm.domains import parent_world_key, rules_for
        case = self.cases[5]; facts = frozenset(tuple(f) for f in case["before"])
        self.assertEqual(canonical_parent(facts, list(reversed(case["objects"]))), case["parent_world_id"])
        with self.assertRaises(ValueError):
            parent_world_key("blocks", B().state_json(facts), rules_for("blocks", objects=case["objects"]))
    def test_config_types_and_complete_list_guard(self):
        for kwargs in ({"threads": True}, {"threads": 8}, {"device": "cuda"}, {"generation_tokens": True}):
            with self.assertRaises(ValueError):
                BlockG1Config(**kwargs)
        with self.assertRaises(ValueError):
            validate_cases(self.cases[:12])


if __name__ == "__main__":
    unittest.main()
