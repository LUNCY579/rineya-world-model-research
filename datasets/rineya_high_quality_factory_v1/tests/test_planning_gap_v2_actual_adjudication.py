"""Owned hand cases for actual adjudication joins/semantic and renderer diagnostics."""
from pathlib import Path
import copy
import importlib.util
import unittest

PATH = Path(__file__).resolve().parents[1] / "reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py"
SPEC = importlib.util.spec_from_file_location("owned_actual_planning_v2", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class HandProperties(unittest.TestCase):
    def setUp(self):
        self.task = {"kind":"graph", "nodes":["a","b","c"], "edges":[["a","b",1],["a","c",1],["c","b",1]],
                     "start":"a", "goal":"b", "fuel":2, "plan":["c","b"]}

    def test_semantic_node_edge_permutation_not_action_permutation(self):
        changed = copy.deepcopy(self.task)
        changed["nodes"].reverse(); changed["edges"].reverse()
        self.assertEqual(A.semantic_key(changed), A.semantic_key(self.task))
        changed["plan"].reverse()
        self.assertNotEqual(A.semantic_key(changed), A.semantic_key(self.task))

    def test_typed_budget_bool_never_collapses_to_integer(self):
        changed = copy.deepcopy(self.task); changed["fuel"] = True
        self.assertNotEqual(A.semantic_key(changed), A.semantic_key(self.task))

    def test_measured_graph_move_objective_not_fuel_objective(self):
        task = copy.deepcopy(self.task); task["edges"][0][2] = 2
        path, search = A.measured_shortest(task)
        self.assertEqual(path, ["b"]); self.assertIs(search["complete"], True)

    def test_measured_graph_full_sequence_lexical_tie(self):
        task = {"kind":"graph","nodes":["a","b","c","z"],"edges":[["a","c",1],["a","b",1],["c","z",1],["b","z",1]],"start":"a","goal":"z","fuel":2,"plan":[]}
        path, _ = A.measured_shortest(task)
        self.assertEqual(path,["b","z"])

    def test_unreachable_search_is_exhausted_not_timeout(self):
        task = copy.deepcopy(self.task); task["fuel"] = 0
        path, search = A.measured_shortest(task)
        self.assertIsNone(path); self.assertEqual(search["expanded"],search["visited"])

    def test_initial_goal_empty_minimum(self):
        task = copy.deepcopy(self.task); task["goal"] = task["start"]
        path, search = A.measured_shortest(task)
        self.assertEqual(path,[]); self.assertEqual(search["expanded"],1)

    def test_shared_goal_order_is_separate_from_other_styles(self):
        task = {"kind":"blocks","on":{"a":"b","b":"TABLE","c":"TABLE"},"goal_on":{"b":"TABLE","a":"c"},"plan":[{"block":"a","to":"c"}]}
        other = copy.deepcopy(task); other["goal_on"] = {"a":"c","b":"TABLE"}
        self.assertEqual(A.renderer_projection(task),A.renderer_projection(other))
        self.assertNotEqual(A.renderer_projection(task,set(task["goal_on"])),A.renderer_projection(other,set(task["goal_on"])))
        self.assertEqual(A.semantic_key(task), A.semantic_key(other))

    def test_actual_action_object_key_style_detected(self):
        task = {"kind":"blocks","on":{"a":"TABLE","b":"TABLE"},"goal_on":{"a":"b"},"plan":[{"block":"a","to":"b"}]}
        other = copy.deepcopy(task); other["plan"] = [{"to":"b","block":"a"}]
        self.assertNotEqual(A.renderer_projection(task),A.renderer_projection(other))
        self.assertEqual(A.semantic_key(task),A.semantic_key(other))


if __name__ == "__main__":
    unittest.main()
