"""Producer regression tests; independent approval remains a separate process."""
from collections import defaultdict
import importlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factory.common import record_hash, split_for, digest
from factory.reference import solve
from factory.reference import graph_run


class ProducerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.relational = importlib.import_module("generators.relational.generate")
        cls.critic = importlib.import_module("generators.critic.generate")
        cls.rows = {}
        for module, track in [(cls.relational, "RGM-001"), (cls.critic, "RVM-001")]:
            with patch.object(module, "ROOT", Path(cls.workspace.name)):
                cls.rows[track] = module.generate(batch="regression-v2")

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_parent_rendering_equivalence_and_hashes(self):
        for track, rows in self.rows.items():
            self.assertGreaterEqual(len(rows),60)
            groups = defaultdict(list)
            for row in rows:
                self.assertEqual(row["record_hash"], record_hash(row))
                body = row["rendered_input"].split("Task JSON:\n")
                self.assertEqual(2, len(body))
                self.assertEqual(row["semantic_task"], json.loads(body[1]))
                self.assertNotIn(row["sample_id"], row["rendered_input"])
                self.assertNotIn(row["semantic_parent_id"], row["rendered_input"])
                self.assertEqual(row["difficulty"], row["structural_difficulty"])
                self.assertLessEqual(row["structural_difficulty"], 4)
                groups[row["semantic_parent_id"],digest(row["semantic_task"])].append(row)
            self.assertEqual(30, len({r["semantic_parent_id"] for r in rows}))
            for pair in groups.values():
                self.assertEqual(2, len(pair))
                a, b = pair
                self.assertEqual(a["verified_target"], b["verified_target"])
                self.assertEqual(a["semantic_task"], b["semantic_task"])
                self.assertEqual(a["split"], b["split"])
                self.assertNotEqual(a["rendered_input"], b["rendered_input"])

    def test_genuinely_heldout_names_and_larger_counts(self):
        for track, rows in self.rows.items():
            vocabulary = defaultdict(set)
            counts = defaultdict(set)
            unique_parents={r["semantic_parent_id"]:r for r in rows}
            for row in unique_parents.values():
                index=int(row["semantic_parent_id"].rsplit('-p',1)[1])
                self.assertEqual(split_for(index), row["split"])
                task = row["semantic_task"]
                world = task["world"] if task["kind"] == "critic" else task
                kind = world["kind"]
                names = world["nodes"] if kind == "graph" else list(world["on"])
                vocabulary[kind, row["split"]].update(names)
                counts[kind, row["split"]].add(len(names))
                self.assertEqual(["heldout_names", "larger_graph_or_block_count"]
                                 if row["split"] == "dev_challenge" else [], row["challenge_axes"])
            for kind in {key[0] for key in vocabulary}:
                self.assertFalse(vocabulary[kind, "train"] & vocabulary[kind, "dev_challenge"])
                self.assertLess(max(counts[kind, "train"]), min(counts[kind, "dev_challenge"]))
                if kind == "graph":
                    self.assertEqual({3, 4, 5, 6, 7, 8}, counts[kind, "train"] | counts[kind, "dev_challenge"])
            if track == "RGM-001":
                self.assertEqual(18, sum(r["semantic_task"]["kind"] == "graph" for r in unique_parents.values()))
                self.assertEqual({3, 4, 5}, counts["blocks", "train"] | counts["blocks", "dev_challenge"])

    def test_graph_stop_at_first_illegal_and_require_whole_plan(self):
        task = dict(kind="graph", nodes=["a", "b", "g"], edges=[["a", "b", 2], ["b", "g", 2]],
                    start="a", goal="g", fuel=5, plan=["b", "b", "g"])
        result = solve(task)
        self.assertEqual([{"legal": True, "after": {"node": "b", "fuel": 3}},
                          {"legal": False, "after": {"node": "b", "fuel": 3}}], result["steps"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["reaches_goal"])
        self.assertEqual(["b", "g"], result["shortest_plan"])
        task["plan"] = ["b", "g", "g"]
        result = solve(task)
        self.assertEqual("g", result["after"]["node"])
        self.assertFalse(result["reaches_goal"])

    def test_lexicographic_tie_tag_has_real_minimal_paths(self):
        for row in self.rows["RGM-001"][::2]:
            if "lexicographic_tie" not in row["skill_tags"]:
                continue
            task = row["semantic_task"]
            optimum = row["verified_target"]["shortest_plan"]
            tied = [list(plan) for plan in itertools.product(task["nodes"], repeat=len(optimum))
                    if graph_run(task, list(plan))["reaches_goal"]]
            self.assertGreaterEqual(len(tied), 2)
            self.assertEqual(min(tied), optimum)

    def test_block_stop_at_first_illegal_after_goal(self):
        task = dict(kind="blocks", on={"a": "TABLE", "b": "TABLE", "c": "TABLE"},
                    goal_on={"a": "b"}, plan=[{"block": "a", "to": "b"},
                                              {"block": "a", "to": "b"}, {"block": "a", "to": "TABLE"}])
        result = solve(task)
        self.assertEqual(2, len(result["steps"]))
        self.assertEqual({"a": "b", "b": "TABLE", "c": "TABLE"}, result["after"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["reaches_goal"])
        self.assertEqual([{"block": "a", "to": "b"}], result["shortest_plan"])

    def test_critic_equal_length_and_length_counterexamples(self):
        for row in self.rows["RVM-001"][::2]:
            task = row["semantic_task"]
            evaluations = {evaluation["candidate_id"]: evaluation for evaluation in row["verified_target"]["evaluations"]}
            candidates = task["candidates"]
            self.assertTrue(any(len(c["plan"]) == 1 and evaluations[c["candidate_id"]]["valid"] for c in candidates))
            self.assertTrue(any(len(c["plan"]) == 1 and not evaluations[c["candidate_id"]]["valid"] for c in candidates))
            self.assertTrue(any(len(c["plan"]) >= 2 and evaluations[c["candidate_id"]]["valid"] for c in candidates))
            self.assertTrue(any(e["valid"] and not e["claim_correct"] and e["first_error"] == "claim"
                                for e in evaluations.values()))
            repairs = [e["repair_plan"] for e in evaluations.values()]
            self.assertTrue(all(repair == repairs[0] for repair in repairs))
            self.assertIn("not an edit-distance repair", row["rendered_input"])
            rank_index = {cid: i for i, group in enumerate(row["verified_target"]["ranking"]) for cid in group}
            good = next(c for c in candidates if len(c["plan"]) >= 2 and evaluations[c["candidate_id"]]["valid"]
                        and evaluations[c["candidate_id"]]["claim_correct"])
            bad = next(c for c in candidates if len(c["plan"]) == 1 and not evaluations[c["candidate_id"]]["valid"])
            self.assertLess(rank_index[good["candidate_id"]], rank_index[bad["candidate_id"]])

    def test_candidate_batch_immutability(self):
        with patch.object(self.relational, "ROOT", Path(self.workspace.name)):
            self.assertEqual(self.rows["RGM-001"], self.relational.generate(batch="regression-v2"))
            with self.assertRaisesRegex(ValueError, "new batch version"):
                self.relational.generate(count=31, batch="regression-v2")


if __name__ == "__main__":
    unittest.main()
