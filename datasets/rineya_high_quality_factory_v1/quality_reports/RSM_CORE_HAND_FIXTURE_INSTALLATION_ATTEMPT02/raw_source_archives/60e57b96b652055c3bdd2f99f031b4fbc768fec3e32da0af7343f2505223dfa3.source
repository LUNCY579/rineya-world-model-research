"""Boundary and independently enumerated checks for the reference oracle.

Run: python3 -m unittest discover -s High-Quality_Dataset/tests -p 'oracle*.py'
No generator, model call, expected dataset target, or third-party package is used.
"""

from collections import deque
from copy import deepcopy
from itertools import permutations, product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracles.independent import OracleError, solve


def const(value):
    return {"const": value}


def var(name):
    return {"var": name}


def comparison(operation, left, right):
    return {"op": operation, "args": [left, right]}


def event(operation="add", amount=1, source="x", destination="y", status="ok"):
    return {"status": status, "amount": amount, "op": operation,
            "source": source, "destination": destination}


def causal(events=None, rule=None, precondition=None):
    return {"kind": "causal", "state": {"x": 5, "y": -2},
            "rule": const(True) if rule is None else rule,
            "precondition": const(True) if precondition is None else precondition,
            "events": [] if events is None else events}


def graph(edges=None, fuel=5, plan=None, start="a", goal="g", nodes=None):
    return {"kind": "graph", "nodes": ["a", "b", "c", "g"] if nodes is None else nodes,
            "edges": [["a", "b", 1], ["b", "g", 1]] if edges is None else edges,
            "start": start, "goal": goal, "fuel": fuel, "plan": [] if plan is None else plan}


def blocks(on=None, goal=None, plan=None):
    return {"kind": "blocks", "on": {"a": "TABLE", "b": "TABLE", "c": "TABLE"}
            if on is None else on, "goal_on": {} if goal is None else goal,
            "plan": [] if plan is None else plan}


def enumerated_graph_plan(world):
    """Brute force all name sequences up to the fuel bound, without oracle helpers."""
    edges = {(source, destination): cost for source, destination, cost in world["edges"]}
    for depth in range(world["fuel"] + 1):
        for sequence in product(sorted(world["nodes"]), repeat=depth):
            node, fuel = world["start"], world["fuel"]
            for destination in sequence:
                if (node, destination) not in edges or edges[(node, destination)] > fuel:
                    break
                fuel -= edges[(node, destination)]
                node = destination
            else:
                if node == world["goal"]:
                    return list(sequence)
    return None


def stack_worlds(names):
    """Build every valid world as a partition of an ordered physical stack layout."""
    worlds = set()
    for ordered in permutations(names):
        for cuts in product((False, True), repeat=max(0, len(names) - 1)):
            support = "TABLE"
            on = {}
            for index, block in enumerate(ordered):
                if index and cuts[index - 1]:
                    support = "TABLE"
                on[block] = support
                support = block
            worlds.add(tuple(on[name] for name in sorted(names)))
    return worlds


def as_stacks(names, state):
    supports = dict(zip(names, state))
    above = {support: block for block, support in supports.items() if support != "TABLE"}
    stacks = []
    for base in sorted(block for block, support in supports.items() if support == "TABLE"):
        tower = [base]
        while tower[-1] in above:
            tower.append(above[tower[-1]])
        stacks.append(tuple(tower))
    return tuple(sorted(stacks))


def enumerated_block_plan(names, initial, goal):
    """BFS a physical stack representation independently from support-map execution."""
    start = as_stacks(names, initial)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        towers, path = queue.popleft()
        supports = {}
        for tower in towers:
            for index, block in enumerate(tower):
                supports[block] = "TABLE" if index == 0 else tower[index - 1]
        if all(supports[block] == support for block, support in goal.items()):
            return [{"block": block, "to": target} for block, target in path]
        moves = []
        for source_index, tower in enumerate(towers):
            moving = tower[-1]
            if len(tower) > 1:
                updated = list(towers)
                updated[source_index] = tower[:-1]
                updated.append((moving,))
                moves.append(((moving, "TABLE"), tuple(sorted(updated))))
            for target_index, target_tower in enumerate(towers):
                if target_index == source_index:
                    continue
                updated = [part for index, part in enumerate(towers)
                           if index not in (source_index, target_index)]
                if len(tower) > 1:
                    updated.append(tower[:-1])
                updated.append(target_tower + (moving,))
                moves.append(((moving, target_tower[-1]), tuple(sorted(updated))))
        for action, following in sorted(moves):
            if following not in visited:
                visited.add(following)
                queue.append((following, path + (action,)))
    return None


class CausalTests(unittest.TestCase):
    def test_simultaneous_swap_and_transfer_conserve(self):
        task = causal([event("swap"), event("transfer", 8), event("transfer", 3, "y", "x")])
        answer = solve(task)
        self.assertEqual([step["after"] for step in answer["steps"]],
                         [{"x": -2, "y": 5}, {"x": -10, "y": 13}, {"x": -7, "y": 10}])
        for step in answer["steps"]:
            self.assertEqual(sum(step["after"].values()), 3)
        self.assertEqual(task["state"], {"x": 5, "y": -2})

    def test_sequential_dynamic_precondition_and_both_gates(self):
        task = causal([event("transfer", 5), event("transfer", 1), event("add", status="bad")],
                      comparison("eq", var("event.status"), const("ok")),
                      comparison("ge", var("state.x"), var("event.amount")))
        answer = solve(task)
        self.assertEqual(answer["after"], {"x": 0, "y": 3})
        self.assertEqual(answer["steps"][1]["calc"], {"operation": "identity"})
        self.assertEqual(answer["steps"][2]["cmp"], {"rule": False, "precondition": False})

    def test_boolean_boundaries_and_empty_groups(self):
        task = causal([event()], {"op": "all", "args": []}, {"op": "any", "args": []})
        self.assertEqual(solve(task)["after"], task["state"])
        task["precondition"] = {"op": "not", "args": [const(False)]}
        self.assertEqual(solve(task)["after"], {"x": 6, "y": -2})
        task["rule"] = {"op": "all", "args": [var("event.status")]}
        task["events"][0]["status"] = True
        self.assertEqual(solve(task)["after"], {"x": 6, "y": -2})

    def test_type_sensitive_equality(self):
        for left, right, expected in [(True, 1, False), (None, None, True), ("1", 1, False),
                                      (False, 0, False), (1, 1, True)]:
            answer = solve(causal([event()], comparison("eq", const(left), const(right))))
            self.assertIs(answer["steps"][0]["cmp"]["rule"], expected)
            answer = solve(causal([event()], comparison("ne", const(left), const(right))))
            self.assertIs(answer["steps"][0]["cmp"]["rule"], not expected)

    def test_all_comparison_boundaries(self):
        for operation, values in {"ge": [False, True, True], "gt": [False, False, True],
                                  "le": [True, True, False], "lt": [True, False, False]}.items():
            for left, expected in zip((2, 3, 4), values):
                answer = solve(causal([event()], comparison(operation, const(left), const(3))))
                self.assertIs(answer["steps"][0]["cmp"]["rule"], expected)
        task = causal([event()], comparison("lt", const("a"), const("b")))
        self.assertTrue(solve(task)["steps"][0]["cmp"]["rule"])

    def test_identity_scale_and_add(self):
        initial = causal([event("scale", 1), event("add", 3, "y", "x")])
        self.assertEqual(solve(initial)["after"], {"x": 5, "y": 1})
        initial["rule"] = const(False)
        answer = solve(initial)
        self.assertEqual(answer["after"], initial["state"])
        self.assertTrue(all(step["calc"]["operation"] == "identity" for step in answer["steps"]))

    def test_renaming_coordinates(self):
        task = causal([event("swap"), event("transfer", 2), event("scale", 3, "y", "x")],
                      comparison("le", var("state.y"), const(10)))
        renamed = deepcopy(task)
        renamed["state"] = {"x": task["state"]["y"], "y": task["state"]["x"]}
        renamed["rule"]["args"][0] = var("state.x")
        for action in renamed["events"]:
            action["source"] = "y" if action["source"] == "x" else "x"
            action["destination"] = "y" if action["destination"] == "x" else "x"
        original = solve(task)
        changed = solve(renamed)
        for before, after in zip(original["steps"], changed["steps"]):
            self.assertEqual(before["cmp"], after["cmp"])
            self.assertEqual(before["calc"], after["calc"])
            self.assertEqual(before["after"], {"x": after["after"]["y"], "y": after["after"]["x"]})

    def test_ast_malformed_rejected_including_dead_branches(self):
        invalid = [{"const": 1}, {"var": "state.z"}, {"op": "unknown", "args": []},
                   {"const": True, "var": "state.x"}, {"const": []}, {"const": 1.0},
                   {"op": "not", "args": []}, {"op": "eq", "args": [const(1)]},
                   {"op": "all", "args": [const(1)]},
                   comparison("gt", const(True), const(False)),
                   comparison("lt", const(1), const("2")),
                   {"op": "any", "args": [const(True), {"var": "state.nope"}]}]
        for rule in invalid:
            with self.subTest(rule=rule), self.assertRaises(OracleError):
                solve(causal([event()], rule))
        cyclical = {"op": "not", "args": []}
        cyclical["args"].append(cyclical)
        with self.assertRaises(OracleError):
            solve(causal(rule=cyclical))
        with self.assertRaises(OracleError):
            solve(causal([event(status=4)], var("event.status")))

    def test_malformed_events_and_states(self):
        for field, value in [("amount", True), ("amount", 0), ("op", "divide"),
                             ("source", "z"), ("status", {}), ("destination", 7)]:
            task = causal([event()])
            task["events"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(OracleError):
                solve(task)
        for operation in ("transfer", "swap"):
            with self.assertRaises(OracleError):
                solve(causal([event(operation, source="x", destination="x")]))
        task = causal()
        task["state"]["x"] = True
        with self.assertRaises(OracleError):
            solve(task)


class GraphTests(unittest.TestCase):
    def test_first_illegal_stops_and_reaches_goal_requires_full_validity(self):
        task = graph(plan=["b", "g", "missing", "a"])
        answer = solve(task)
        self.assertEqual(len(answer["steps"]), 3)
        self.assertFalse(answer["valid"])
        self.assertFalse(answer["reaches_goal"])
        self.assertEqual(answer["after"], {"node": "g", "fuel": 3})
        self.assertEqual(answer["steps"][-1], {"legal": False, "after": answer["after"]})

    def test_fuel_boundary_empty_goal_and_unreachable(self):
        self.assertTrue(solve(graph(fuel=2, plan=["b", "g"]))["reaches_goal"])
        limited = solve(graph(fuel=1, plan=["b", "g"]))
        self.assertIsNone(limited["shortest_plan"])
        self.assertEqual(limited["after"], {"node": "b", "fuel": 0})
        answer = solve(graph(fuel=0, goal="a"))
        self.assertEqual(answer["shortest_plan"], [])
        self.assertTrue(answer["valid"] and answer["reaches_goal"])
        self.assertIsNone(solve(graph(edges=[], fuel=100))["shortest_plan"])

    def test_canonical_shortest_tie_is_full_destination_sequence(self):
        task = graph(edges=[["a", "c", 1], ["c", "g", 1], ["a", "b", 2], ["b", "g", 2]])
        self.assertEqual(solve(task)["shortest_plan"], ["b", "g"])
        task["edges"] = list(reversed(task["edges"]))
        task["nodes"] = list(reversed(task["nodes"]))
        self.assertEqual(solve(task)["shortest_plan"], ["b", "g"])

    def test_same_node_different_fuel_states_are_distinct(self):
        task = graph(nodes=["a", "b", "c", "d", "g"], fuel=4,
                     edges=[["a", "b", 2], ["b", "c", 2], ["a", "d", 1],
                            ["d", "c", 1], ["c", "g", 2]])
        self.assertEqual(solve(task)["shortest_plan"], ["d", "c", "g"])

    def test_exhaustive_three_node_graphs_vs_sequence_enumeration(self):
        names = ["a", "b", "c"]
        possible = [(left, right) for left in names for right in names if left != right]
        checked = 0
        for inclusion in product((False, True), repeat=len(possible)):
            edges = [[left, right, 1 + (left > right)]
                     for (left, right), present in zip(possible, inclusion) if present]
            for start, goal, fuel in product(names, names, range(4)):
                task = graph(nodes=names, edges=edges, start=start, goal=goal, fuel=fuel)
                self.assertEqual(solve(task)["shortest_plan"], enumerated_graph_plan(task))
                checked += 1
        self.assertEqual(checked, 2304)

    def test_rename_and_permute(self):
        task = graph(plan=["b"])
        rename = {"a": "node_a", "b": "node_b", "c": "node_c", "g": "node_g"}
        altered = {"kind": "graph", "nodes": [rename[node] for node in reversed(task["nodes"])],
                   "edges": [[rename[left], rename[right], cost] for left, right, cost in reversed(task["edges"])],
                   "start": rename[task["start"]], "goal": rename[task["goal"]],
                   "fuel": task["fuel"], "plan": [rename[node] for node in task["plan"]]}
        original, changed = solve(task), solve(altered)
        self.assertEqual(changed["shortest_plan"], [rename[node] for node in original["shortest_plan"]])
        self.assertEqual(changed["after"]["fuel"], original["after"]["fuel"])
        self.assertEqual(changed["after"]["node"], rename[original["after"]["node"]])
        self.assertEqual(changed["valid"], original["valid"])

    def test_graph_malformed_rejected(self):
        for key, value in [("nodes", ["a", "a", "g"]), ("start", "missing"), ("goal", None),
                           ("fuel", -1), ("fuel", True), ("plan", [1]),
                           ("edges", [["a", "b", 0]]), ("edges", [["a", "b", True]]),
                           ("edges", [["missing", "b", 1]]),
                           ("edges", [["a", "b", 1], ["a", "b", 2]])]:
            task = graph()
            task[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(OracleError):
                solve(task)


class BlockTests(unittest.TestCase):
    def test_stop_first_illegal_and_clear_boundaries(self):
        task = blocks({"a": "b", "b": "TABLE", "c": "TABLE"}, {"a": "c"},
                      [{"block": "b", "to": "c"}, {"block": "a", "to": "c"}])
        answer = solve(task)
        self.assertFalse(answer["valid"])
        self.assertEqual(answer["after"], task["on"])
        self.assertEqual(len(answer["steps"]), 1)
        task["plan"] = [{"block": "a", "to": "c"}, {"block": "a", "to": "c"},
                        {"block": "a", "to": "TABLE"}]
        answer = solve(task)
        self.assertEqual(len(answer["steps"]), 2)
        self.assertFalse(answer["reaches_goal"])
        self.assertEqual(answer["after"], {"a": "c", "b": "TABLE", "c": "TABLE"})

    def test_illegal_action_types_are_semantic_failures(self):
        for action in [{"block": "missing", "to": "a"}, {"block": "a", "to": "missing"},
                       {"block": "a", "to": "a"}, {"block": "a", "to": "TABLE"}]:
            answer = solve(blocks(plan=[action]))
            self.assertFalse(answer["valid"])
            self.assertEqual(answer["after"], blocks()["on"])

    def test_canonical_tie(self):
        task = blocks(goal={"a": "b", "c": "TABLE"})
        self.assertEqual(solve(task)["shortest_plan"], [{"block": "a", "to": "b"}])
        task = blocks({"a": "b", "b": "TABLE", "c": "d", "d": "TABLE"},
                      {"a": "TABLE", "c": "TABLE"})
        self.assertEqual(solve(task)["shortest_plan"],
                         [{"block": "a", "to": "TABLE"}, {"block": "c", "to": "TABLE"}])

    def test_exhaustive_valid_worlds_and_all_goals_vs_stack_bfs(self):
        names = ("a", "b", "c")
        worlds = sorted(stack_worlds(names))
        self.assertEqual(len(worlds), 13)
        checked = 0
        for initial, final in product(worlds, repeat=2):
            goal = dict(zip(names, final))
            task = blocks(dict(zip(names, initial)), goal)
            expected = enumerated_block_plan(names, initial, goal)
            answer = solve(task)
            self.assertEqual(answer["shortest_plan"], expected)
            task["plan"] = answer["shortest_plan"]
            completed = solve(task)
            self.assertTrue(completed["valid"] and completed["reaches_goal"])
            self.assertEqual(completed["after"], goal)
            checked += 1
        self.assertEqual(checked, 169)

    def test_four_blocks_partial_goals_and_conservation(self):
        names = ("a", "b", "c", "d")
        worlds = sorted(stack_worlds(names))
        self.assertEqual(len(worlds), 73)
        for index, initial in enumerate(worlds[::4]):
            goal = {"a": worlds[(index * 11) % len(worlds)][0]}
            task = blocks(dict(zip(names, initial)), goal)
            expected = enumerated_block_plan(names, initial, goal)
            answer = solve(task)
            self.assertEqual(answer["shortest_plan"], expected)
            task["plan"] = expected
            for step in solve(task)["steps"]:
                state = step["after"]
                self.assertEqual(set(state), set(names))
                self.assertIn(tuple(state[name] for name in names), worlds)

    def test_empty_world_identity_and_rename_permutation(self):
        answer = solve(blocks({}, {}))
        self.assertTrue(answer["valid"] and answer["reaches_goal"])
        self.assertEqual(answer["shortest_plan"], [])
        original = blocks({"a": "b", "b": "TABLE", "c": "TABLE"}, {"a": "c"})
        rename = {"a": "block_a", "b": "block_b", "c": "block_c", "TABLE": "TABLE"}
        changed = blocks({rename[key]: rename[value] for key, value in reversed(list(original["on"].items()))},
                         {rename[key]: rename[value] for key, value in original["goal_on"].items()})
        expected = [{"block": rename[action["block"]], "to": rename[action["to"]]}
                    for action in solve(original)["shortest_plan"]]
        self.assertEqual(solve(changed)["shortest_plan"], expected)

    def test_cycles_invalid_supports_and_contradictory_goals(self):
        invalid_states = [{"a": "a"}, {"a": "b", "b": "a"}, {"a": "missing"},
                          {"a": "c", "b": "c", "c": "TABLE"}, {"TABLE": "TABLE"}]
        for state in invalid_states:
            with self.subTest(state=state), self.assertRaises(OracleError):
                solve(blocks(state))
        invalid_goals = [{"a": "a"}, {"a": "b", "b": "a"}, {"a": "c", "b": "c"},
                         {"missing": "TABLE"}, {"a": "missing"}]
        for goal in invalid_goals:
            with self.subTest(goal=goal), self.assertRaises(OracleError):
                solve(blocks(goal=goal))
        with self.assertRaises(OracleError):
            solve(blocks(plan=[{"block": "a"}]))


class CriticTests(unittest.TestCase):
    def critic(self, candidates, world=None):
        return {"kind": "critic", "world": graph() if world is None else world,
                "candidates": candidates}

    def candidate(self, identity, plan, node="a", fuel=5):
        return {"candidate_id": identity, "plan": plan, "claimed_after": {"node": node, "fuel": fuel}}

    def test_labels_claim_separation_first_error_and_ranking(self):
        candidates = [self.candidate("wrong_goal_claim", ["b", "g"], "g", 4),
                      self.candidate("good_goal", ["b", "g"], "g", 3),
                      self.candidate("progress", ["b"], "b", 4),
                      self.candidate("identity", []),
                      self.candidate("invalid", ["c", "g"]),
                      self.candidate("after_goal_invalid", ["b", "g", "missing"], "g", 3)]
        result = solve(self.critic(candidates))
        by_id = {row["candidate_id"]: row for row in result["evaluations"]}
        self.assertEqual(result["ranking"], [["good_goal"], ["progress"], ["identity"],
                                              ["wrong_goal_claim"], ["after_goal_invalid"], ["invalid"]])
        self.assertEqual([by_id[name]["usefulness"] for name in ("good_goal", "progress", "identity", "invalid")],
                         [2, 1, 0, -1])
        wrong = by_id["wrong_goal_claim"]
        self.assertTrue(wrong["valid"] and wrong["reaches_goal"])
        self.assertFalse(wrong["claim_correct"])
        self.assertEqual(wrong["first_error"], "claim")
        self.assertEqual(wrong["progress"], 2)
        self.assertEqual(by_id["invalid"]["first_error"], 0)
        self.assertEqual(by_id["after_goal_invalid"]["first_error"], 2)
        self.assertEqual(by_id["after_goal_invalid"]["remaining_moves"], 0)
        self.assertFalse(by_id["after_goal_invalid"]["reaches_goal"])
        for row in result["evaluations"]:
            self.assertEqual(row["repair_plan"], ["b", "g"])

    def test_ranking_exact_tie_groups_sorted_and_candidate_independence(self):
        candidates = [self.candidate("z", ["b"], "b", 4), self.candidate("a", ["b"], "b", 4),
                      self.candidate("goal", ["b", "g"], "g", 3)]
        result = solve(self.critic(candidates))
        self.assertEqual(result["ranking"], [["goal"], ["a", "z"]])
        permuted = solve(self.critic(list(reversed(candidates))))
        self.assertEqual(permuted["ranking"], result["ranking"])
        self.assertEqual(sorted(permuted["evaluations"], key=lambda row: row["candidate_id"]),
                         sorted(result["evaluations"], key=lambda row: row["candidate_id"]))

    def test_null_progress_initial_or_final_unreachable(self):
        unreachable = graph(fuel=1)
        result = solve(self.critic([self.candidate("stay", [], fuel=1),
                                   self.candidate("move", ["b"], "b", 0)], unreachable))
        for row in result["evaluations"]:
            self.assertIsNone(row["remaining_moves"])
            self.assertIsNone(row["progress"])
            self.assertIsNone(row["repair_plan"])
            self.assertEqual(row["usefulness"], 0)
        self.assertEqual(result["ranking"], [["stay"], ["move"]])
        world = graph(edges=[["a", "b", 1], ["b", "g", 1], ["a", "c", 5]])
        result = solve(self.critic([self.candidate("dead_end", ["c"], "c", 0)], world))
        row = result["evaluations"][0]
        self.assertIsNone(row["remaining_moves"])
        self.assertIsNone(row["progress"])
        self.assertEqual(row["usefulness"], 0)
        self.assertEqual(row["repair_plan"], ["b", "g"])

    def test_initial_goal_progress_zero_and_world_plan_ignored(self):
        world = graph(goal="a")
        world["plan"] = {"deliberately": "ignored"}
        row = solve(self.critic([self.candidate("stay", [])], world))["evaluations"][0]
        self.assertEqual((row["remaining_moves"], row["progress"], row["usefulness"]), (0, 0, 2))
        self.assertEqual(row["repair_plan"], [])

    def test_invalid_claim_can_be_correct_and_null_distance_ranks_last(self):
        world = graph(edges=[["a", "b", 1], ["b", "g", 1], ["a", "c", 5]])
        candidates = [self.candidate("null", ["c"], "c", 0),
                      self.candidate("finite", []), self.candidate("bad_step", ["missing"])]
        result = solve(self.critic(candidates, world))
        self.assertEqual(result["ranking"], [["finite"], ["null"], ["bad_step"]])
        row = result["evaluations"][2]
        self.assertTrue(row["claim_correct"])
        self.assertFalse(row["valid"])
        self.assertIsNone(row["progress"])

    def test_duplicate_ids_and_malformed_claims(self):
        duplicate = [self.candidate("same", []), self.candidate("same", [])]
        with self.assertRaises(OracleError):
            solve(self.critic(duplicate))
        for claim in [{"node": "a", "fuel": True}, {"node": "missing", "fuel": 1},
                      {"node": "a", "fuel": -1}, {"node": "a"}]:
            candidate = self.candidate("id", [])
            candidate["claimed_after"] = claim
            with self.subTest(claim=claim), self.assertRaises(OracleError):
                solve(self.critic([candidate]))
        for candidate in [{"candidate_id": "id", "plan": []},
                          {"candidate_id": 1, "plan": [], "claimed_after": {"node": "a", "fuel": 5}},
                          {"candidate_id": "id", "plan": [False], "claimed_after": {"node": "a", "fuel": 5}}]:
            with self.subTest(candidate=candidate), self.assertRaises(OracleError):
                solve(self.critic([candidate]))


class DispatchTests(unittest.TestCase):
    def test_input_unmodified_and_output_snapshot_not_aliased(self):
        task = causal([event(), event()])
        original = deepcopy(task)
        answer = solve(task)
        self.assertEqual(task, original)
        answer["steps"][0]["after"]["x"] = 999
        self.assertEqual(answer["steps"][1]["after"]["x"], 7)
        self.assertEqual(answer["after"]["x"], 7)

    def test_unknown_and_missing_fields(self):
        for task in [None, [], {}, {"kind": "unknown"}, {"kind": "graph"}, {"kind": True}]:
            with self.subTest(task=task), self.assertRaises(OracleError):
                solve(task)


if __name__ == "__main__":
    unittest.main()
