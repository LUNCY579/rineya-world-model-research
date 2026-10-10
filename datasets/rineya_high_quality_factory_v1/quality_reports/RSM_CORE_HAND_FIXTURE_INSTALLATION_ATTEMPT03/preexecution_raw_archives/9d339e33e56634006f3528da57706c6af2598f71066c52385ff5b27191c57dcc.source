"""Independent factual-tag tests; no stored target or producer is an input."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracles.metadata_independent import check, goal_route_counts


def graph(nodes=None, edges=None, start="a", goal="g", fuel=4):
    return {"kind": "graph", "nodes": ["a", "g"] if nodes is None else nodes,
            "edges": [["a", "g", 1]] if edges is None else edges,
            "start": start, "goal": goal, "fuel": fuel, "plan": []}


def blocks(on, goal):
    return {"kind": "blocks", "on": on, "goal_on": goal, "plan": []}


def row(task, *tags):
    return {"semantic_task": task, "skill_tags": list(tags)}


def brute_counts(world):
    """Enumerate actual full destination sequences, independently of DP frontiers."""
    edges = {(left, right): cost for left, right, cost in world["edges"]}
    bound = world["fuel"] // min(edges.values()) if edges else 0
    counts = []
    for depth in range(bound + 1):
        count = 0
        for sequence in product(world["nodes"], repeat=depth):
            node, spent = world["start"], 0
            for destination in sequence:
                if (node, destination) not in edges:
                    break
                spent += edges[(node, destination)]
                node = destination
            else:
                if spent <= world["fuel"] and node == world["goal"]:
                    count += 1
        counts.append(min(count, 2))
    return counts


class RouteMultiplicityTests(unittest.TestCase):
    def test_goal_departures_revisits_and_equal_nonsmallest_routes(self):
        world = graph(nodes=["a", "g"], edges=[["a", "g", 1], ["g", "a", 1], ["g", "g", 1]], fuel=3)
        self.assertEqual(goal_route_counts(world), [0, 1, 1, 2])
        self.assertEqual(check(row(world, "equal_length_goal_routes")), [])
        self.assertTrue(check(row(world, "no_equal_length_goal_routes")))
        # An absorbing-goal search would miss both routes after the initial goal.
        world["start"] = "g"; world["fuel"] = 2
        self.assertEqual(goal_route_counts(world), [1, 1, 2])

    def test_no_equal_routes_with_goal_loop_and_with_irrelevant_branch_loop(self):
        world = graph(nodes=["g"], edges=[["g", "g", 1]], start="g", fuel=6)
        self.assertEqual(goal_route_counts(world), [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(check(row(world, "single_successor_resource_graph", "no_equal_length_goal_routes")), [])
        world = graph(nodes=["a", "b", "g"], edges=[["a", "b", 1], ["b", "b", 1], ["a", "g", 1]])
        self.assertEqual(goal_route_counts(world), [0, 1, 0, 0, 0])
        self.assertEqual(check(row(world, "branched_resource_graph", "no_equal_length_goal_routes")), [])

    def test_resource_bound_excludes_second_equal_length_route(self):
        world = graph(nodes=["a", "b", "c", "g"], edges=[["a", "b", 3], ["b", "g", 3],
                       ["a", "c", 1], ["c", "g", 1]], fuel=5)
        self.assertEqual(check(row(world, "no_equal_length_goal_routes")), [])
        world["fuel"] = 6
        self.assertEqual(goal_route_counts(world)[2], 2)
        self.assertEqual(check(row(world, "equal_length_goal_routes")), [])

    def test_merged_frontier_preserves_distinct_prefix_multiplicity(self):
        world = graph(nodes=["a", "b", "c", "d", "g"], edges=[["a", "b", 1], ["a", "c", 1],
                       ["b", "d", 1], ["c", "d", 1], ["d", "g", 1]])
        self.assertEqual(goal_route_counts(world), [0, 0, 0, 2, 0])

    def test_empty_success_and_empty_edges(self):
        world = graph(edges=[], start="g", fuel=0)
        self.assertEqual(goal_route_counts(world), [1])
        self.assertEqual(check(row(world, "initial_goal_reachable", "no_equal_length_goal_routes")), [])
        self.assertTrue(check(row(world, "initially_unreachable")))
        world["start"] = "a"
        self.assertEqual(goal_route_counts(world), [0])
        self.assertEqual(check(row(world, "initial_goal_unreachable", "initially_unreachable")), [])

    def test_exhaustive_weighted_two_node_frontiers_vs_full_sequences(self):
        names = ["a", "b"]
        pairs = list(product(names, repeat=2))
        checked = 0
        for costs in product((0, 1, 2), repeat=4):
            edges = [[left, right, cost] for (left, right), cost in zip(pairs, costs) if cost]
            for start, goal, fuel in product(names, names, range(6)):
                world = graph(nodes=names, edges=edges, start=start, goal=goal, fuel=fuel)
                expected = brute_counts(world)
                self.assertEqual(goal_route_counts(world), expected)
                route_tag = "equal_length_goal_routes" if 2 in expected else "no_equal_length_goal_routes"
                reach_tag = "initial_goal_reachable" if any(expected) else "initial_goal_unreachable"
                self.assertEqual(check(row(world, route_tag, reach_tag)), [])
                checked += 1
        self.assertEqual(checked, 1944)

    def test_exhaustive_three_node_nonsimple_routes(self):
        names = ["a", "b", "c"]
        pairs = [(left, right) for left in names for right in names if left != right]
        checked = 0
        for inclusion in product((False, True), repeat=6):
            edges = [[left, right, 1] for (left, right), present in zip(pairs, inclusion) if present]
            for start, goal, fuel in product(names, names, range(5)):
                world = graph(nodes=names, edges=edges, start=start, goal=goal, fuel=fuel)
                self.assertEqual(goal_route_counts(world), brute_counts(world))
                checked += 1
        self.assertEqual(checked, 2880)

    def test_renaming_edges_and_nodes_do_not_change_counts(self):
        world = graph(nodes=["a", "b", "g"], edges=[["a", "b", 1], ["a", "g", 2],
                      ["b", "g", 1], ["g", "a", 1]], fuel=5)
        original = goal_route_counts(world)
        rename = {"a": "z", "b": "n", "g": "m"}
        changed = graph(nodes=[rename[name] for name in reversed(world["nodes"])],
                        edges=[[rename[left], rename[right], cost] for left, right, cost in reversed(world["edges"])],
                        start=rename["a"], goal=rename["g"], fuel=5)
        self.assertEqual(goal_route_counts(changed), original)


class FactualMetadataTests(unittest.TestCase):
    def test_topology_uses_every_node_even_unreachable_branches(self):
        world = graph(nodes=["a", "b", "c", "g"], edges=[["a", "g", 1], ["b", "c", 1], ["b", "g", 1]])
        self.assertEqual(check(row(world, "branched_resource_graph")), [])
        self.assertTrue(check(row(world, "single_successor_resource_graph")))
        world["edges"] = [["a", "a", 1]]
        self.assertEqual(check(row(world, "single_successor_resource_graph")), [])

    def test_critic_initial_world_and_colon_composed_tags(self):
        task = {"kind": "critic", "world": graph(), "candidates": []}
        self.assertEqual(check(row(task, "single_successor_resource_graph:initial_goal_reachable")), [])
        failures = check(row(task, "branched_resource_graph:initial_goal_unreachable"))
        self.assertEqual(len(failures), 2)
        self.assertTrue(any(failure.startswith("branched_resource_graph:") for failure in failures))
        task["world"]["fuel"] = 0
        self.assertEqual(check(row(task, "single_successor_resource_graph:initially_unreachable")), [])

    def test_block_literal_roots_partial_goals_and_legacy_or(self):
        task = blocks({"a": "TABLE", "b": "a", "c": "TABLE"}, {"a": "b"})
        self.assertEqual(check(row(task, "multiple_initial_table_stacks", "multi_stack_goal_composition", "block_goal_composition")), [])
        self.assertTrue(check(row(task, "multiple_explicit_goal_table_roots")))
        task = blocks({"a": "TABLE", "b": "a", "c": "b"}, {"a": "TABLE", "b": "TABLE"})
        self.assertEqual(check(row(task, "multiple_explicit_goal_table_roots", "multi_stack_goal_composition")), [])
        self.assertTrue(check(row(task, "multiple_initial_table_stacks")))
        task["goal_on"] = {"a": "TABLE"}
        self.assertTrue(check(row(task, "multi_stack_goal_composition")))
        self.assertTrue(check(row(task, "multiple_explicit_goal_table_roots")))
        self.assertEqual(check(row(task, "block_goal_composition")), [])

    def test_block_order_and_consistent_name_renaming(self):
        task = blocks({"a": "TABLE", "b": "a", "c": "TABLE"}, {"b": "TABLE", "c": "TABLE"})
        tags = ["multiple_initial_table_stacks", "multiple_explicit_goal_table_roots"]
        self.assertEqual(check(row(task, *tags)), [])
        rename = {"a": "z", "b": "y", "c": "x", "TABLE": "TABLE"}
        changed = blocks({rename[key]: rename[value] for key, value in reversed(list(task["on"].items()))},
                         {rename[key]: rename[value] for key, value in reversed(list(task["goal_on"].items()))})
        self.assertEqual(check(row(changed, *reversed(tags))), [])

    def test_target_and_verdict_metadata_are_not_evidence(self):
        class ForbiddenEvidence:
            def __repr__(self): raise AssertionError("stored target inspected")
            def __eq__(self, other): raise AssertionError("stored target compared")
            def __bool__(self): raise AssertionError("stored target used")
        item = row(graph(), "initial_goal_unreachable")
        original = check(item)
        item["verified_target"] = ForbiddenEvidence()
        item["review_status"] = ForbiddenEvidence()
        self.assertEqual(check(item), original)
        self.assertTrue(original)

    def test_legacy_equal_routes_matches_current_property(self):
        world = graph(nodes=["a", "g"], edges=[["a", "g", 1], ["g", "a", 1], ["g", "g", 1]], fuel=3)
        self.assertEqual(check(row(world, "equal_length_routes")), [])
        world["fuel"] = 2
        self.assertTrue(check(row(world, "equal_length_routes")))

    def test_wrong_domain_and_malformed_worlds_are_failures(self):
        self.assertTrue(check(row(blocks({"a": "TABLE"}, {}), "branched_resource_graph")))
        self.assertTrue(check(row(graph(), "block_goal_composition")))
        world = graph(); world["edges"] = [["a", "g", 0]]
        self.assertTrue(check(row(world, "initial_goal_reachable")))
        task = blocks({"a": "b", "b": "a"}, {})
        self.assertTrue(check(row(task, "block_goal_composition")))
        for item in [None, {}, {"semantic_task": {}}, {"semantic_task": {}, "skill_tags": True},
                     {"semantic_task": {}, "skill_tags": [1]}]:
            self.assertTrue(check(item))

    def test_unknown_nonfactual_tags_and_input_immutability(self):
        task = row(graph(), "unlisted_competency", "initial_goal_reachable")
        original = deepcopy(task)
        self.assertEqual(check(task), [])
        self.assertEqual(task, original)


if __name__ == "__main__":
    unittest.main()
