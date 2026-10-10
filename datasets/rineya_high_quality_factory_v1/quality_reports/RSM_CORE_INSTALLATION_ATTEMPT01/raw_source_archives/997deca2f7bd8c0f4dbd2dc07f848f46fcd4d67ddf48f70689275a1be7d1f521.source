"""Independent boundary, exhaustive, and metamorphic tests for extended tasks."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracles.extended_independent import solve
from oracles.independent import OracleError, solve as dispatch


def add(register, value):
    return {"op": "add", "register": register, "value": value}


def call(name):
    return {"op": "call", "name": name}


def branch(register="x", cmp="lt", value=0, then=None, otherwise=None):
    return {"op": "if", "test": {"register": register, "cmp": cmp, "value": value},
            "then": [] if then is None else then, "else": [] if otherwise is None else otherwise}


def skill_task():
    return {"kind": "skills", "state": {"x": 2, "y": 7},
            "procedures": {"inner": [add("x", 3)], "outer": [call("inner"), {"op": "swap"}]},
            "main": [call("outer"), call("inner"), branch("x", "ge", 10, [add("y", 1)], [add("y", -1)])]}


def module_task():
    return {"kind": "modules", "modules": [
        {"module_id": "z", "capabilities": ["read"], "priority": 1},
        {"module_id": "a", "capabilities": ["read", "write"], "priority": 1},
        {"module_id": "low", "capabilities": ["read"], "priority": 0}],
        "memory": {"protected": True, "x": 0}, "protected": ["protected"],
        "requests": [{"request_id": "r", "needs": ["read"], "writes": [
            {"key": "protected", "value": 1}, {"key": "x", "value": 1},
            {"key": "x", "value": 2}, {"key": "protected", "value": None}]},
            {"request_id": "no", "needs": ["missing"], "writes": [{"key": "protected", "value": False},
                                                                     {"key": "new", "value": "not attempted"}]}]}


def memory_task():
    return {"kind": "episodic", "events": [
        {"key": "k", "version": 4, "at": 3, "op": "put", "value": "v4"},
        {"key": "k", "version": 2, "at": 8, "op": "put", "value": "late lower"},
        {"key": "k", "version": 5, "at": 10, "op": "delete", "value": None},
        {"key": "k", "version": 4, "at": 12, "op": "put", "value": "lower after delete"},
        {"key": "null", "version": 0, "at": 0, "op": "put", "value": None}],
        "queries": [{"query_id": "before", "key": "k", "as_of": 2},
                    {"query_id": "v4", "key": "k", "as_of": 8},
                    {"query_id": "deleted", "key": "k", "as_of": 20},
                    {"query_id": "null", "key": "null", "as_of": 0}]}


def belief_task():
    return {"kind": "belief", "worlds": [
        {"world_id": "bool", "facts": {"f": True, "g": "left"}},
        {"world_id": "int", "facts": {"f": 1, "g": "left"}},
        {"world_id": "null", "facts": {"f": None, "g": "right"}},
        {"world_id": "string", "facts": {"f": "é", "g": "right"}}],
        "observations": [], "queries": [{"query_id": "f", "fact": "f"}, {"query_id": "g", "fact": "g"}]}


def quotient_task():
    return {"kind": "predictive_abstraction", "states": ["a", "b", "c", "d"], "actions": ["z", "x"],
            "observable": {"a": 0, "b": 0, "c": 0, "d": 1},
            "transition": {"a": {"x": "b", "z": "a"}, "b": {"x": "c", "z": "b"},
                           "c": {"x": "d", "z": "c"}, "d": {"x": "d", "z": "d"}}}


def table_partition(task):
    """Independent refinement check; production oracle uses pair automata instead."""
    def value_key(state):
        value = task["observable"][state]
        return (type(value).__name__, value)

    buckets = {}
    for state in task["states"]:
        buckets.setdefault(value_key(state), []).append(state)
    partition = sorted(sorted(group) for group in buckets.values())
    while True:
        membership = {state: index for index, group in enumerate(partition) for state in group}
        buckets = {}
        for state in task["states"]:
            key = (value_key(state), tuple(membership[task["transition"][state][action]]
                                           for action in sorted(task["actions"])))
            buckets.setdefault(key, []).append(state)
        updated = sorted(sorted(group) for group in buckets.values())
        if updated == partition:
            return partition
        partition = updated


def repair_task():
    return {"kind": "constraint_repair", "alphabet": ["b", "a"], "slots": 3,
            "constraints": [{"kind": "count", "value": "b", "min": 1, "max": 1}],
            "candidates": [{"candidate_id": "candidate", "assignment": ["b", "b", "b"]}],
            "min_distance": 2}


def history_task():
    return {"kind": "agent_history", "objects": ["coin", "key"], "places": ["box", "bag", "shelf"],
            "initial_locations": {"coin": "box", "key": "bag"},
            "agents": [{"agent_id": "observer", "initial_beliefs": {"coin": "box", "key": None},
                        "object_priority": ["key", "coin"]},
                       {"agent_id": "absent", "initial_beliefs": {"coin": "box", "key": "shelf"},
                        "object_priority": ["key", "coin"]},
                       {"agent_id": "unknown", "initial_beliefs": {"coin": None, "key": None},
                        "object_priority": ["coin", "key"]}],
            "events": [{"object": "coin", "to": "shelf", "observed_by": ["observer"]},
                       {"object": "key", "to": "box", "observed_by": ["observer"]}]}


class SkillsTests(unittest.TestCase):
    def test_nested_repeated_calls_condition_order_and_simultaneous_swap(self):
        answer = solve(skill_task())
        self.assertEqual(answer, {"after": {"x": 10, "y": 6}, "executed_primitives": 4,
                                  "call_order": ["outer", "inner", "inner"], "conditions": [True]})
        task = skill_task()
        task["procedures"]["inner"] = [branch("x", "eq", 2, [add("x", 3)], [add("x", 1)])]
        answer = solve(task)
        self.assertEqual(answer["conditions"], [True, False, False])
        self.assertEqual(answer["after"], {"x": 8, "y": 4})

    def test_cycle_in_unchosen_branch_and_unused_procedure_rejected(self):
        task = skill_task()
        task["procedures"]["unused"] = [branch(then=[], otherwise=[call("unused")])]
        with self.assertRaises(OracleError):
            solve(task)
        task = skill_task()
        task["procedures"]["inner"] = [call("outer")]
        with self.assertRaises(OracleError):
            solve(task)
        task = skill_task()
        task["main"] = [branch(then=[], otherwise=[call("missing")])]
        with self.assertRaises(OracleError):
            solve(task)

    def test_inlining_and_procedure_order_permutation(self):
        task = skill_task()
        original = solve(task)
        task["main"] = [add("x", 3), {"op": "swap"}, add("x", 3), task["main"][-1]]
        flattened = solve(task)
        for key in ("after", "conditions", "executed_primitives"):
            self.assertEqual(original[key], flattened[key])
        task = skill_task()
        task["procedures"] = dict(reversed(list(task["procedures"].items())))
        self.assertEqual(solve(task), original)

    def test_register_rename_scale_and_signed_values(self):
        task = skill_task()
        task["main"].append({"op": "scale", "register": "x", "value": -2})
        changed = deepcopy(task)
        changed["state"] = {"x": task["state"]["y"], "y": task["state"]["x"]}

        def rename(instructions):
            for instruction in instructions:
                if "register" in instruction:
                    instruction["register"] = {"x": "y", "y": "x"}[instruction["register"]]
                if instruction["op"] == "if":
                    instruction["test"]["register"] = {"x": "y", "y": "x"}[instruction["test"]["register"]]
                    rename(instruction["then"])
                    rename(instruction["else"])

        rename(changed["main"])
        for body in changed["procedures"].values():
            rename(body)
        original, renamed = solve(task), solve(changed)
        self.assertEqual(original["after"], {"x": renamed["after"]["y"], "y": renamed["after"]["x"]})
        for key in ("call_order", "conditions", "executed_primitives"):
            self.assertEqual(original[key], renamed[key])

    def test_malformed_instructions(self):
        for instruction in [{"op": "swap", "extra": 1}, add("z", 1), add("x", True),
                            {"op": "if", "test": {"register": "x", "cmp": "gt", "value": 1}, "then": [], "else": []},
                            {"op": "divide"}]:
            task = skill_task()
            task["main"] = [instruction]
            with self.subTest(instruction=instruction), self.assertRaises(OracleError):
                solve(task)


class ModuleTests(unittest.TestCase):
    def test_competence_abstention_and_protected_write_indices(self):
        answer = solve(module_task())
        self.assertEqual(answer["routes"][0], {"request_id": "r", "selected_module": "low",
                                               "eligible_modules": ["a", "low", "z"], "blocked_writes": [0, 3]})
        self.assertEqual(answer["routes"][1], {"request_id": "no", "selected_module": None,
                                               "eligible_modules": [], "blocked_writes": []})
        self.assertEqual(answer["memory"], {"protected": True, "x": 2})
        self.assertIs(type(answer["memory"]["protected"]), bool)

    def test_tie_empty_needs_and_module_order(self):
        task = module_task()
        task["modules"] = task["modules"][:2]
        task["requests"][0]["needs"] = []
        original = solve(task)
        self.assertEqual(original["routes"][0]["selected_module"], "a")
        task["modules"].reverse()
        self.assertEqual(solve(task), original)
        task["modules"] = []
        self.assertTrue(all(row["selected_module"] is None for row in solve(task)["routes"]))

    def test_request_write_order_and_capability_monotonicity(self):
        task = module_task()
        task["requests"].append({"request_id": "last", "needs": ["write"], "writes": [{"key": "x", "value": None}]})
        self.assertIsNone(solve(task)["memory"]["x"])
        baseline = solve(module_task())["routes"][0]["eligible_modules"]
        changed = module_task()
        changed["modules"].append({"module_id": "extra", "capabilities": ["read"], "priority": 100})
        self.assertEqual(set(solve(changed)["routes"][0]["eligible_modules"]), set(baseline) | {"extra"})
        self.assertEqual(solve(changed)["memory"], solve(module_task())["memory"])

    def test_malformed_modules_requests_and_protection(self):
        tasks = []
        task = module_task(); task["protected"] = ["missing"]; tasks.append(task)
        task = module_task(); task["modules"][0]["priority"] = False; tasks.append(task)
        task = module_task(); task["modules"][0]["capabilities"] = ["read", "read"]; tasks.append(task)
        task = module_task(); task["requests"][1]["request_id"] = "r"; tasks.append(task)
        task = module_task(); task["memory"]["x"] = 1.0; tasks.append(task)
        for task in tasks:
            with self.assertRaises(OracleError): solve(task)


class EpisodicTests(unittest.TestCase):
    def test_version_precedes_chronology_and_tombstone(self):
        answer = solve(memory_task())["results"]
        self.assertEqual(answer[0]["status"], "absent")
        self.assertEqual((answer[1]["value"], answer[1]["chosen_event"]), ("v4", 0))
        self.assertEqual((answer[2]["status"], answer[2]["chosen_version"], answer[2]["chosen_event"]), ("deleted", 5, 2))
        self.assertEqual((answer[3]["status"], answer[3]["value"]), ("present", None))

    def test_same_version_time_then_input_order(self):
        task = memory_task()
        task["events"] = [{"key": "k", "version": 1, "at": at, "op": "put", "value": value}
                          for at, value in [(5, "first"), (3, "earlier"), (5, "last")]]
        task["queries"] = [{"query_id": "q", "key": "k", "as_of": 5}]
        row = solve(task)["results"][0]
        self.assertEqual((row["value"], row["chosen_event"]), ("last", 2))

    def test_irrelevant_future_events_and_key_rename(self):
        task = memory_task()
        original = solve(task)
        task["events"].extend([{"key": "other", "version": 99, "at": 0, "op": "put", "value": 1},
                               {"key": "k", "version": 99, "at": 21, "op": "put", "value": 2}])
        self.assertEqual(solve(task), original)
        for event in task["events"]: event["key"] = "prefix_" + event["key"]
        for query in task["queries"]: query["key"] = "prefix_" + query["key"]
        renamed = solve(task)
        for before, after in zip(original["results"], renamed["results"]):
            self.assertEqual(after["key"], "prefix_" + before["key"])
            self.assertEqual({k: v for k, v in before.items() if k != "key"},
                             {k: v for k, v in after.items() if k != "key"})

    def test_empty_memory_and_malformed_events(self):
        task = memory_task(); task["events"] = []
        self.assertTrue(all(row["status"] == "absent" for row in solve(task)["results"]))
        for key, value in [("version", -1), ("at", True), ("op", "update"), ("value", [])]:
            task = memory_task(); task["events"][0][key] = value
            with self.subTest(key=key), self.assertRaises(OracleError): solve(task)
        task = memory_task(); task["events"][2]["value"] = "not null"
        with self.assertRaises(OracleError): solve(task)


class BeliefTests(unittest.TestCase):
    def test_boolean_integer_separation_and_canonical_unicode_counts(self):
        task = belief_task()
        answer = solve(task)
        self.assertEqual(answer["queries"][0]["counts"], [{"value": "é", "count": 1},
                         {"value": 1, "count": 1}, {"value": None, "count": 1}, {"value": True, "count": 1}])
        task["observations"] = [{"fact": "f", "equals": True}]
        answer = solve(task)
        self.assertEqual(answer["surviving_worlds"], ["bool"])
        self.assertIs(answer["queries"][0]["guaranteed"], True)
        task["observations"][0]["equals"] = 1
        answer = solve(task)
        self.assertEqual(answer["surviving_worlds"], ["int"])
        self.assertIs(type(answer["queries"][0]["guaranteed"]), int)

    def test_empty_belief_stays_empty_and_null_entailment(self):
        task = belief_task()
        task["observations"] = [{"fact": "f", "equals": "absent"}, {"fact": "g", "equals": "left"}]
        answer = solve(task)
        self.assertEqual(answer["belief_sizes"], [0, 0])
        self.assertEqual(answer["surviving_worlds"], [])
        for row in answer["queries"]:
            self.assertEqual(row["counts"], [])
            self.assertFalse(row["entailed"] or row["can_reduce_uncertainty"])
        task["observations"] = [{"fact": "f", "equals": None}]
        row = solve(task)["queries"][0]
        self.assertTrue(row["entailed"])
        self.assertIsNone(row["guaranteed"])

    def test_filter_order_world_order_and_duplicate_observation(self):
        task = belief_task()
        task["observations"] = [{"fact": "g", "equals": "left"}, {"fact": "f", "equals": 1}]
        original = solve(task)
        self.assertEqual(original["belief_sizes"], [2, 1])
        task["worlds"].reverse(); task["observations"].reverse()
        changed = solve(task)
        self.assertEqual(changed["surviving_worlds"], original["surviving_worlds"])
        self.assertEqual(changed["queries"], original["queries"])
        self.assertEqual(changed["belief_sizes"], [1, 1])
        task["observations"].append(task["observations"][-1])
        self.assertEqual(solve(task)["belief_sizes"], [1, 1, 1])

    def test_malformed_world_and_fact_queries(self):
        tasks = []
        task = belief_task(); del task["worlds"][0]["facts"]["g"]; tasks.append(task)
        task = belief_task(); task["worlds"][1]["world_id"] = "bool"; tasks.append(task)
        task = belief_task(); task["queries"][0]["fact"] = "missing"; tasks.append(task)
        task = belief_task(); task["observations"] = [{"fact": "f", "equals": 1.0}]; tasks.append(task)
        for task in tasks:
            with self.assertRaises(OracleError): solve(task)


class PredictiveTests(unittest.TestCase):
    def test_multistep_distinguishing_sequence_and_homomorphism(self):
        task = quotient_task()
        answer = solve(task)
        self.assertEqual(answer["classes"], [["a"], ["b"], ["c"], ["d"]])
        self.assertEqual(answer["abstract_transitions"][0], {"class_id": 0, "action": "x", "next_class": 1})
        for row in answer["abstract_transitions"]:
            for member in answer["classes"][row["class_id"]]:
                self.assertEqual(answer["state_to_class"][task["transition"][member][row["action"]]], row["next_class"])

    def test_exhaustive_three_state_two_action_quotients(self):
        states, actions = ["a", "b", "c"], ["x", "y"]
        checked = 0
        for values in product((0, 1), repeat=3):
            for next_states in product(states, repeat=6):
                task = {"kind": "predictive_abstraction", "states": states, "actions": actions,
                        "observable": dict(zip(states, values)),
                        "transition": {state: dict(zip(actions, next_states[2 * index:2 * index + 2]))
                                       for index, state in enumerate(states)}}
                self.assertEqual(solve(task)["classes"], table_partition(task))
                checked += 1
        self.assertEqual(checked, 5832)

    def test_exhaustive_typed_observables(self):
        states, actions = ["a", "b"], ["x", "y"]
        checked = 0
        for values in product((True, 1, None), repeat=2):
            for next_states in product(states, repeat=4):
                task = {"kind": "predictive_abstraction", "states": states, "actions": actions,
                        "observable": dict(zip(states, values)),
                        "transition": {state: dict(zip(actions, next_states[2 * index:2 * index + 2]))
                                       for index, state in enumerate(states)}}
                self.assertEqual(solve(task)["classes"], table_partition(task))
                checked += 1
        self.assertEqual(checked, 144)

    def test_order_permutation_and_state_action_rename(self):
        task = quotient_task()
        original = solve(task)
        task["states"].reverse(); task["actions"].reverse()
        task["transition"] = dict(reversed(list(task["transition"].items())))
        self.assertEqual(solve(task), original)
        renamed = {"kind": task["kind"], "states": ["S_" + name for name in task["states"]],
                   "actions": ["A_" + name for name in task["actions"]],
                   "observable": {"S_" + name: value for name, value in task["observable"].items()},
                   "transition": {"S_" + state: {"A_" + action: "S_" + successor for action, successor in successors.items()}
                                  for state, successors in task["transition"].items()}}
        changed = solve(renamed)
        self.assertEqual(changed["classes"], [["S_" + state for state in group] for group in original["classes"]])

    def test_missing_transitions_unknown_states_and_float_rejected(self):
        tasks = []
        task = quotient_task(); del task["transition"]["a"]["x"]; tasks.append(task)
        task = quotient_task(); task["transition"]["a"]["x"] = "missing"; tasks.append(task)
        task = quotient_task(); task["observable"]["a"] = 1.0; tasks.append(task)
        task = quotient_task(); task["actions"] = []; tasks.append(task)
        task = quotient_task(); task["states"] = ["a", "a"]; tasks.append(task)
        for task in tasks:
            with self.assertRaises(OracleError): solve(task)


class RepairTests(unittest.TestCase):
    def test_minimum_hamming_then_full_lexical_tie(self):
        answer = solve(repair_task())
        self.assertEqual(answer, {"feasible_count": 3, "repairs": [{"candidate_id": "candidate", "original_valid": False,
                                 "edit_distance": 2, "repaired": ["a", "a", "b"]}], "diverse_selection": ["candidate"]})

    def test_greedy_diversity_is_ordered_and_zero_permits_duplicates(self):
        task = repair_task(); task["constraints"] = []
        task["candidates"] = [{"candidate_id": name, "assignment": list(value)}
                              for name, value in [("c", "bba"), ("b", "aab"), ("a", "aaa"), ("d", "aaa")]]
        self.assertEqual(solve(task)["diverse_selection"], ["a", "c"])
        task["min_distance"] = 0
        self.assertEqual(solve(task)["diverse_selection"], ["a", "b", "c", "d"])
        task["min_distance"] = 4
        self.assertEqual(solve(task)["diverse_selection"], ["a"])

    def test_impossible_constraints_and_same_position_boundaries(self):
        task = repair_task()
        task["constraints"] = [{"kind": "allowed", "position": 0, "values": []}]
        answer = solve(task)
        self.assertEqual(answer["feasible_count"], 0)
        self.assertEqual(answer["diverse_selection"], [])
        self.assertIsNone(answer["repairs"][0]["repaired"])
        task["constraints"] = [{"kind": "different", "positions": [0, 0]}]
        self.assertEqual(solve(task)["feasible_count"], 0)
        task["constraints"] = [{"kind": "count", "value": "a", "min": 2, "max": 1}]
        self.assertEqual(solve(task)["feasible_count"], 0)
        task["constraints"] = [{"kind": "equal", "positions": [0, 0]}]
        self.assertEqual(solve(task)["feasible_count"], 8)

    def test_exhaustive_binary_repair_and_diversity(self):
        choices = list(product(("a", "b"), repeat=3))
        options = [({"kind": "equal", "positions": [0, 1]}, lambda a: a[0] == a[1]),
                   ({"kind": "different", "positions": [1, 2]}, lambda a: a[1] != a[2]),
                   ({"kind": "allowed", "position": 0, "values": ["a"]}, lambda a: a[0] == "a"),
                   ({"kind": "allowed", "position": 1, "values": ["b"]}, lambda a: a[1] == "b"),
                   ({"kind": "count", "value": "a", "min": 0, "max": 0}, lambda a: "a" not in a),
                   ({"kind": "count", "value": "a", "min": 1, "max": 1}, lambda a: len([x for x in a if x == "a"]) == 1),
                   ({"kind": "count", "value": "a", "min": 2, "max": 3}, lambda a: len([x for x in a if x == "a"]) >= 2),
                   ({"kind": "allowed", "position": 2, "values": []}, lambda a: False)]
        checked = 0
        for inclusion in product((False, True), repeat=8):
            selected = [item for item, included in zip(options, inclusion) if included]
            feasible = [assignment for assignment in choices if all(predicate(assignment) for _, predicate in selected)]
            for separation in range(5):
                task = repair_task(); task["constraints"] = [rule for rule, _ in selected]
                task["min_distance"] = separation
                task["candidates"] = [{"candidate_id": "c" + str(index), "assignment": list(value)}
                                      for index, value in enumerate(choices)]
                answer = solve(task)
                self.assertEqual(answer["feasible_count"], len(feasible))
                accepted = []
                accepted_ids = []
                expected_rows = []
                for index, original in enumerate(choices):
                    ranking = sorted((sum(x != y for x, y in zip(original, final)), final) for final in feasible)
                    best_distance, best = ranking[0] if ranking else (None, None)
                    row = answer["repairs"][index]
                    self.assertEqual(row["original_valid"], original in feasible)
                    self.assertEqual(row["edit_distance"], best_distance)
                    self.assertEqual(row["repaired"], None if best is None else list(best))
                    if best is not None: expected_rows.append((best_distance, row["candidate_id"], best))
                for _, identity, repaired in sorted(expected_rows):
                    pair_distances = [len([pair for pair in zip(repaired, prior) if pair[0] != pair[1]]) for prior in accepted]
                    if not pair_distances or min(pair_distances) >= separation:
                        accepted.append(repaired); accepted_ids.append(identity)
                self.assertEqual(answer["diverse_selection"], accepted_ids)
                checked += 1
        self.assertEqual(checked, 1280)

    def test_constraint_alphabet_candidate_permutations(self):
        task = repair_task()
        task["constraints"].append({"kind": "equal", "positions": [0, 1]})
        task["candidates"].append({"candidate_id": "other", "assignment": ["a", "b", "a"]})
        original = solve(task)
        task["alphabet"].reverse(); task["constraints"].reverse(); task["candidates"].reverse()
        changed = solve(task)
        self.assertEqual(changed["feasible_count"], original["feasible_count"])
        self.assertEqual(changed["diverse_selection"], original["diverse_selection"])
        self.assertEqual(list(reversed(changed["repairs"])), original["repairs"])

    def test_malformed_constraint_candidate_and_positions(self):
        tasks = []
        task = repair_task(); task["slots"] = True; tasks.append(task)
        task = repair_task(); task["min_distance"] = -1; tasks.append(task)
        task = repair_task(); task["constraints"] = [{"kind": "equal", "positions": [0, 3]}]; tasks.append(task)
        task = repair_task(); task["constraints"] = [{"kind": "allowed", "position": False, "values": []}]; tasks.append(task)
        task = repair_task(); task["candidates"][0]["assignment"] = ["a"]; tasks.append(task)
        task = repair_task(); task["constraints"] = [{"kind": "count", "value": "unknown", "min": 0, "max": 1}]; tasks.append(task)
        for task in tasks:
            with self.assertRaises(OracleError): solve(task)


class AgentHistoryTests(unittest.TestCase):
    def test_observer_false_belief_unknown_and_policy(self):
        answer = solve(history_task())
        self.assertEqual(answer["actual_locations"], {"coin": "shelf", "key": "box"})
        observer, absent, unknown = answer["agents"]
        self.assertEqual(observer["false_belief_objects"], [])
        self.assertEqual(observer["action"], {"type": "search", "object": "key", "place": "box"})
        self.assertEqual(absent["false_belief_objects"], ["coin", "key"])
        self.assertEqual(absent["action"], {"type": "search", "object": "key", "place": "shelf"})
        self.assertEqual(unknown["unknown_objects"], ["coin", "key"])
        self.assertEqual(unknown["action"], {"type": "abstain"})

    def test_last_observed_location_and_unobserved_actual_move(self):
        task = history_task()
        task["events"].append({"object": "coin", "to": "bag", "observed_by": []})
        answer = solve(task)
        self.assertEqual(answer["agents"][0]["beliefs"]["coin"], "shelf")
        self.assertEqual(answer["agents"][0]["false_belief_objects"], ["coin"])
        self.assertEqual(answer["actual_locations"]["coin"], "bag")

    def test_empty_events_preserve_explicit_initial_false_beliefs(self):
        task = history_task(); task["events"] = []
        answer = solve(task)
        self.assertEqual(answer["actual_locations"], task["initial_locations"])
        self.assertEqual(answer["agents"][1]["false_belief_objects"], ["key"])
        for row, agent in zip(answer["agents"], task["agents"]):
            self.assertEqual(row["beliefs"], agent["initial_beliefs"])

    def test_object_place_agent_renaming_and_observer_permutation(self):
        task = history_task(); task["events"][0]["observed_by"].append("unknown")
        original = solve(task)
        task["events"][0]["observed_by"].reverse(); task["objects"].reverse(); task["places"].reverse()
        self.assertEqual(solve(task), original)
        changed = deepcopy(task)
        changed["objects"] = ["o_" + name for name in task["objects"]]
        changed["places"] = ["p_" + name for name in task["places"]]
        changed["initial_locations"] = {"o_" + key: "p_" + value for key, value in task["initial_locations"].items()}
        for agent in changed["agents"]:
            agent["agent_id"] = "a_" + agent["agent_id"]
            agent["initial_beliefs"] = {"o_" + key: None if value is None else "p_" + value for key, value in agent["initial_beliefs"].items()}
            agent["object_priority"] = ["o_" + name for name in agent["object_priority"]]
        for event in changed["events"]:
            event["object"] = "o_" + event["object"]; event["to"] = "p_" + event["to"]
            event["observed_by"] = ["a_" + name for name in event["observed_by"]]
        renamed = solve(changed)
        self.assertEqual(renamed["actual_locations"], {"o_" + key: "p_" + value for key, value in original["actual_locations"].items()})
        for before, after in zip(original["agents"], renamed["agents"]):
            self.assertEqual(after["false_belief_objects"], ["o_" + name for name in before["false_belief_objects"]])
            self.assertEqual(after["unknown_objects"], ["o_" + name for name in before["unknown_objects"]])

    def test_malformed_observers_and_initial_beliefs(self):
        tasks = []
        task = history_task(); task["events"][0]["observed_by"] = ["missing"]; tasks.append(task)
        task = history_task(); task["events"][0]["observed_by"] = ["observer", "observer"]; tasks.append(task)
        task = history_task(); del task["agents"][0]["initial_beliefs"]["coin"]; tasks.append(task)
        task = history_task(); task["agents"][0]["initial_beliefs"]["coin"] = True; tasks.append(task)
        task = history_task(); task["agents"][0]["object_priority"] = ["coin", "coin"]; tasks.append(task)
        for task in tasks:
            with self.assertRaises(OracleError): solve(task)


class ExtendedDispatchTests(unittest.TestCase):
    def test_dispatch_agrees_input_is_unchanged_and_results_do_not_alias(self):
        for task in [skill_task(), module_task(), memory_task(), belief_task(), quotient_task(), repair_task(), history_task()]:
            before = deepcopy(task)
            answer = solve(task)
            self.assertEqual(dispatch(task), answer)
            self.assertEqual(task, before)
            if "after" in answer: answer["after"]["x"] = 999
            if "memory" in answer: answer["memory"]["protected"] = 999
            if "actual_locations" in answer: answer["actual_locations"]["coin"] = "changed"
            self.assertEqual(task, before)

    def test_unknown_kind_and_empty_structures(self):
        for task in [None, {}, {"kind": "unknown"}, {"kind": "skills"}]:
            with self.assertRaises(OracleError): solve(task)
        task = {"kind": "skills", "state": {"x": 0, "y": 0}, "procedures": {}, "main": []}
        self.assertEqual(solve(task)["executed_primitives"], 0)
        self.assertEqual(solve({"kind": "belief", "worlds": [], "observations": [], "queries": []}),
                         {"surviving_worlds": [], "belief_sizes": [], "queries": []})


if __name__ == "__main__":
    unittest.main()
