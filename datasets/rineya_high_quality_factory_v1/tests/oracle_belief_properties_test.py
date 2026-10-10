"""Fresh belief properties derived from the public finite-world contract.

Expected results use typed atoms and independently intersected bit masks. The
expected mechanism and hand fixtures were authored before invoking either solver.
No candidate, stored target, producer implementation, or verdict is an input.
"""

from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
EXHAUSTIVE_STATS = {}
FIRST_MISMATCH = None


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atom(value):
    if type(value) not in (bool, int, str, type(None)):
        raise TypeError("Public belief scalars are null, boolean, integer, or string")
    return type(value).__name__, value


def enumerated_expected(task):
    """Intersect independently indexed world masks, then partition typed atoms."""
    worlds = task["worlds"]
    possible = (1 << len(worlds)) - 1
    fact_masks = {}
    for index, world in enumerate(worlds):
        for fact, value in world["facts"].items():
            key = fact, atom(value)
            fact_masks[key] = fact_masks.get(key, 0) | (1 << index)
    sizes = []
    for observation in task["observations"]:
        possible &= fact_masks.get((observation["fact"], atom(observation["equals"])), 0)
        sizes.append(possible.bit_count())
    surviving_indices = [index for index in range(len(worlds)) if possible & (1 << index)]
    queries = []
    for query in task["queries"]:
        partition = {}
        for index in surviving_indices:
            value = worlds[index]["facts"][query["fact"]]
            key = atom(value)
            if key not in partition:
                partition[key] = {"value": value, "count": 0}
            partition[key]["count"] += 1
        counts = sorted(partition.values(), key=lambda entry: canonical(entry["value"]))
        entailed = bool(surviving_indices) and len(counts) == 1
        queries.append({"query_id": query["query_id"], "counts": counts,
            "distinct_values": len(counts), "entailed": entailed,
            "guaranteed": counts[0]["value"] if entailed else None,
            "can_reduce_uncertainty": len(counts) > 1})
    return {"surviving_worlds": sorted(worlds[index]["world_id"] for index in surviving_indices),
            "belief_sizes": sizes, "queries": queries}


def make_task(facts, observations=(), queries=None):
    return {"kind": "belief", "worlds": [
        {"world_id": "w" + str(index), "facts": deepcopy(item)} for index, item in enumerate(facts)],
        "observations": [{"fact": fact, "equals": value} for fact, value in observations],
        "queries": [{"query_id": name, "fact": fact} for name, fact in
            (queries if queries is not None else [("q", next(iter(facts[0])))]) ]}


def cross_check(task, expected=None):
    # Expected answers exist before importing or invoking existing mechanisms.
    global FIRST_MISMATCH
    expected = enumerated_expected(task) if expected is None else expected
    from oracles.independent import solve as independent_solve
    from generators.other_tracks.reference import solve as reference_solve
    for name, solver in [("independent", independent_solve), ("reference", reference_solve)]:
        before = canonical(task)
        try:
            observed = solver(task)
        except Exception as error:
            mismatch = {"mechanism": name, "task": deepcopy(task), "expected": expected,
                        "exception": type(error).__name__ + ": " + str(error)}
            if FIRST_MISMATCH is None:
                FIRST_MISMATCH = mismatch
            raise AssertionError(canonical(mismatch)) from error
        if canonical(observed) != canonical(expected) or canonical(task) != before:
            mismatch = {"mechanism": name, "task": deepcopy(task), "expected": expected,
                        "observed": observed, "input_modified": canonical(task) != before}
            if FIRST_MISMATCH is None:
                FIRST_MISMATCH = mismatch
            raise AssertionError(canonical(mismatch))
    return expected


class FreshBeliefProperties(unittest.TestCase):
    def test_exact_typed_partition_encoding_and_count_conservation(self):
        values = [True, 1, False, 0, None, "1", "true", "0", "false", "null", "é", True, 1]
        task = make_task([{"x": value} for value in values])
        expected = enumerated_expected(task)
        counts = expected["queries"][0]["counts"]
        self.assertEqual([canonical(item["value"]) for item in counts],
                         ['"0"', '"1"', '"false"', '"null"', '"true"', '"é"', '0', '1', 'false', 'null', 'true'])
        self.assertEqual([item["count"] for item in counts], [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2])
        self.assertEqual(sum(item["count"] for item in counts), len(values))
        cross_check(task, expected)

    def test_unobserved_invariant_is_entailed_with_multiple_survivors(self):
        task = make_task([{"seen": 1, "fixed": None, "unknown": "a"},
                          {"seen": 1, "fixed": None, "unknown": "b"},
                          {"seen": 0, "fixed": "changed", "unknown": "c"}],
                         [("seen", 1)], [("constant", "fixed"), ("useful", "unknown")])
        expected = enumerated_expected(task)
        self.assertEqual(expected["belief_sizes"], [2])
        constant, useful = expected["queries"]
        self.assertTrue(constant["entailed"])
        self.assertIsNone(constant["guaranteed"])
        self.assertFalse(constant["can_reduce_uncertainty"])
        self.assertTrue(useful["can_reduce_uncertainty"])
        cross_check(task, expected)

    def test_conjunctive_evidence_can_leave_a_useful_unobserved_query(self):
        task = make_task([{"a": True, "b": 0, "x": True}, {"a": True, "b": 0, "x": 1},
                          {"a": True, "b": 1, "x": None}, {"a": False, "b": 0, "x": "1"}],
                         [("a", True), ("b", 0)], [("decision", "x")])
        expected = enumerated_expected(task)
        self.assertEqual(expected["belief_sizes"], [3, 2])
        self.assertEqual(expected["queries"][0]["distinct_values"], 2)
        self.assertTrue(expected["queries"][0]["can_reduce_uncertainty"])
        self.assertEqual([canonical(item["value"]) for item in expected["queries"][0]["counts"]], ["1", "true"])
        cross_check(task, expected)

    def test_same_cardinality_observation_changes_query_partition(self):
        facts = [{"group": "left", "x": 0}, {"group": "left", "x": 0},
                 {"group": "right", "x": True}, {"group": "right", "x": 1}]
        left = make_task(facts, [("group", "left")], [("q", "x")])
        right = make_task(facts, [("group", "right")], [("q", "x")])
        le, re = enumerated_expected(left), enumerated_expected(right)
        self.assertEqual(le["belief_sizes"], re["belief_sizes"])
        self.assertEqual(le["belief_sizes"], [2])
        self.assertTrue(le["queries"][0]["entailed"])
        self.assertEqual(type(le["queries"][0]["guaranteed"]), int)
        self.assertFalse(re["queries"][0]["entailed"])
        self.assertTrue(re["queries"][0]["can_reduce_uncertainty"])
        cross_check(left, le); cross_check(right, re)

    def test_redundant_observation_preserves_partition_and_prefix_size(self):
        task = make_task([{"a": 1, "b": True}, {"a": 1, "b": 1}, {"a": 0, "b": None}],
                         [("a", 1), ("a", 1), ("a", 1)], [("q", "b")])
        expected = enumerated_expected(task)
        self.assertEqual(expected["belief_sizes"], [2, 2, 2])
        once = deepcopy(task); once["observations"] = once["observations"][:1]
        self.assertEqual(expected["queries"], enumerated_expected(once)["queries"])
        cross_check(task, expected); cross_check(once)

    def test_reordered_observations_keep_final_set_but_not_prefix_sizes(self):
        facts = [{"a": 1, "b": 1}, {"a": 1, "b": 0}, {"a": 1, "b": 0}, {"a": 0, "b": 1}]
        first = make_task(facts, [("a", 1), ("b", 1)], [("q", "a")])
        reversed_task = deepcopy(first); reversed_task["observations"].reverse()
        one, two = enumerated_expected(first), enumerated_expected(reversed_task)
        self.assertEqual(one["belief_sizes"], [3, 1])
        self.assertEqual(two["belief_sizes"], [2, 1])
        self.assertEqual(one["surviving_worlds"], two["surviving_worlds"])
        self.assertEqual(one["queries"], two["queries"])
        cross_check(first, one); cross_check(reversed_task, two)

    def test_individually_supported_cross_fact_observations_can_contradict(self):
        facts = [{"a": 0, "b": 1}, {"a": 1, "b": 0}]
        for observation in [("a", 0), ("b", 0)]:
            self.assertEqual(enumerated_expected(make_task(facts, [observation]))["belief_sizes"], [1])
        task = make_task(facts, [("a", 0), ("b", 0)], [("qa", "a"), ("qb", "b")])
        expected = enumerated_expected(task)
        self.assertEqual(expected["belief_sizes"], [1, 0])
        self.assertEqual(expected["surviving_worlds"], [])
        self.assertTrue(all(query["counts"] == [] and not query["entailed"] for query in expected["queries"]))
        cross_check(task, expected)

    def test_post_empty_absorption_cannot_revive_original_worlds(self):
        task = make_task([{"a": 0, "b": 1}, {"a": 1, "b": 0}],
                         [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 0)], [("q", "b")])
        expected = enumerated_expected(task)
        self.assertEqual(expected["belief_sizes"], [1, 0, 0, 0, 0])
        cross_check(task, expected)

    def test_null_entailment_ambiguous_null_and_empty_null_are_distinct(self):
        tasks = [make_task([{"x": None}, {"x": None}]),
                 make_task([{"x": None}, {"x": "null"}]),
                 make_task([{"x": None}], [("x", "null")])]
        queries = [enumerated_expected(task)["queries"][0] for task in tasks]
        self.assertTrue(all(query["guaranteed"] is None for query in queries))
        self.assertEqual([query["entailed"] for query in queries], [True, False, False])
        self.assertEqual([query["distinct_values"] for query in queries], [1, 2, 0])
        self.assertEqual([query["can_reduce_uncertainty"] for query in queries], [False, True, False])
        for task in tasks: cross_check(task)

    def test_each_boolean_integer_string_observation_selects_exact_atom(self):
        values = [True, 1, "true", "1", False, 0, "false", "0", None, "null"]
        for value in values:
            task = make_task([{"x": other} for other in values], [("x", value)])
            expected = enumerated_expected(task)
            self.assertEqual(expected["belief_sizes"], [1])
            self.assertEqual(atom(expected["queries"][0]["guaranteed"]), atom(value))
            cross_check(task, expected)

    def test_world_id_order_fact_name_and_query_name_metamorphics(self):
        task = make_task([{"a": True, "b": None}, {"a": 1, "b": "é"},
                          {"a": True, "b": "é"}], [("a", True)], [("first", "a"), ("second", "b")])
        expected = enumerated_expected(task)
        renamed = deepcopy(task)
        world_names = {"w0": "zeta", "w1": "alpha", "w2": "beta"}
        fact_names = {"a": "observed-name", "b": "unobserved-name"}
        for world in renamed["worlds"]:
            world["world_id"] = world_names[world["world_id"]]
            world["facts"] = {fact_names[name]: value for name, value in reversed(list(world["facts"].items()))}
        renamed["worlds"].reverse()
        for observation in renamed["observations"]: observation["fact"] = fact_names[observation["fact"]]
        for query in renamed["queries"]:
            query["fact"] = fact_names[query["fact"]]
            query["query_id"] = "renamed-" + query["query_id"]
        renamed["queries"].reverse()
        re = enumerated_expected(renamed)
        self.assertEqual(re["surviving_worlds"], sorted(world_names[name] for name in expected["surviving_worlds"]))
        self.assertEqual(re["belief_sizes"], expected["belief_sizes"])
        self.assertEqual(re["queries"], [{**query, "query_id": "renamed-" + query["query_id"]} for query in reversed(expected["queries"])])
        cross_check(task, expected); cross_check(renamed, re)

    def test_duplicate_fact_worlds_count_separately_without_changing_atoms(self):
        task = make_task([{"a": True, "b": None}, {"a": True, "b": 1}], [("a", True)], [("q", "b")])
        doubled = deepcopy(task)
        doubled["worlds"] += [{"world_id": "copy-" + world["world_id"], "facts": deepcopy(world["facts"])} for world in task["worlds"]]
        original, repeated = enumerated_expected(task), enumerated_expected(doubled)
        self.assertEqual(repeated["belief_sizes"], [2 * size for size in original["belief_sizes"]])
        self.assertEqual(repeated["queries"][0]["counts"], [{**entry, "count": 2 * entry["count"]} for entry in original["queries"][0]["counts"]])
        self.assertEqual(repeated["queries"][0]["distinct_values"], original["queries"][0]["distinct_values"])
        self.assertEqual(repeated["queries"][0]["entailed"], original["queries"][0]["entailed"])
        cross_check(task, original); cross_check(doubled, repeated)

    def test_prefix_results_match_separately_queried_prefixes(self):
        task = make_task([{"a": True, "b": 0}, {"a": True, "b": 1}, {"a": 1, "b": 0}],
                         [("a", True), ("b", 0), ("b", 1), ("a", 1)], [("q", "a")])
        expected = enumerated_expected(task)
        sizes = []
        for stop in range(1, len(task["observations"]) + 1):
            prefix = deepcopy(task); prefix["observations"] = prefix["observations"][:stop]
            pe = enumerated_expected(prefix)
            sizes.append(len(pe["surviving_worlds"]))
            cross_check(prefix, pe)
        self.assertEqual(sizes, expected["belief_sizes"])
        cross_check(task, expected)

    def test_bounded_exhaustive_typed_world_tables_and_ordered_histories(self):
        digest = hashlib.sha256()
        stats = {"typed_grids": 2, "world_tables": 0, "generated_histories": 0,
                 "query_observations": 0, "solver_cross_checks": 0, "observation_prefixes": 0,
                 "maximum_worlds": 3, "maximum_observations": 2}
        for domain in [(True, 1, None, "1"), (False, 0, None, "0")]:
            assignments = list(itertools.product(domain, repeat=2))
            observation_atoms = list(itertools.product(("a", "b"), domain))
            for size in range(4):
                for indices in itertools.combinations_with_replacement(range(len(assignments)), size):
                    facts = [{"a": assignments[index][0], "b": assignments[index][1]} for index in indices]
                    stats["world_tables"] += 1
                    # With no initial worlds there are no existing fact names.
                    # Empty queried beliefs arise instead from contradictions.
                    for length in range(3 if size else 1):
                        for observations in itertools.product(observation_atoms, repeat=length):
                            task = make_task(facts, observations, [("qa", "a"), ("qb", "b")] if size else [])
                            expected = enumerated_expected(task)
                            # Independent mathematical invariants, before mechanism calls.
                            sizes = [size] + expected["belief_sizes"]
                            self.assertTrue(all(left >= right >= 0 for left, right in zip(sizes, sizes[1:])))
                            for query in expected["queries"]:
                                self.assertEqual(sum(item["count"] for item in query["counts"]), len(expected["surviving_worlds"]))
                                self.assertTrue(all(item["count"] > 0 for item in query["counts"]))
                                self.assertEqual(query["distinct_values"], len(query["counts"]))
                                self.assertEqual(query["entailed"], bool(expected["surviving_worlds"]) and len(query["counts"]) == 1)
                                self.assertEqual(query["can_reduce_uncertainty"], len(query["counts"]) > 1)
                            if length == 2:
                                reverse = deepcopy(task); reverse["observations"].reverse()
                                reverse_expected = enumerated_expected(reverse)
                                self.assertEqual(expected["surviving_worlds"], reverse_expected["surviving_worlds"])
                                self.assertEqual(expected["queries"], reverse_expected["queries"])
                            digest.update((canonical({"task": task, "expected": expected}) + "\n").encode("utf-8"))
                            cross_check(task, expected)
                            stats["generated_histories"] += 1
                            stats["query_observations"] += len(task["queries"])
                            stats["solver_cross_checks"] += 2
                            stats["observation_prefixes"] += length
        self.assertEqual(stats["world_tables"], 1938)
        self.assertEqual(stats["generated_histories"], 141330)
        self.assertEqual(stats["query_observations"], 282656)
        stats["expected_stream_sha256"] = digest.hexdigest()
        EXHAUSTIVE_STATS.update(stats)


if __name__ == "__main__":
    unittest.main()
