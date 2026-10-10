"""Independent public-contract probes for JSON boolean/integer identity.

Targets in these tests are hand-derived or computed from newly authored task
inputs; no dataset labels, producer code, or reviewer verdicts are read.
"""

from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracles.independent import OracleError, solve


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ScalarIdentityProbes(unittest.TestCase):
    def test_public_scalar_encodings_are_distinct(self):
        self.assertEqual(True, 1)  # Documents why untyped Python equality is insufficient.
        self.assertEqual(False, 0)
        self.assertEqual([canonical(value) for value in (True, 1, False, 0)], ["true", "1", "false", "0"])
        self.assertEqual(len({canonical(value) for value in (True, 1, False, 0)}), 4)

    def test_episodic_exact_values_and_synthetic_bad_targets(self):
        for boolean, integer in [(True, 1), (False, 0)]:
            payload = {"kind": "episodic", "events": [{"key": "flag", "version": 0, "at": 0,
                       "op": "put", "value": boolean}],
                       "queries": [{"query_id": "q", "key": "flag", "as_of": 0}]}
            expected = {"results": [{"query_id": "q", "key": "flag", "status": "present",
                       "value": boolean, "chosen_version": 0, "chosen_event": 0}]}
            actual = solve(payload)
            self.assertEqual(canonical(actual), canonical(expected))
            self.assertIs(type(actual["results"][0]["value"]), bool)
            corrupted = deepcopy(expected); corrupted["results"][0]["value"] = integer
            self.assertEqual(actual, corrupted)
            self.assertNotEqual(canonical(actual), canonical(corrupted))
            corrupted_task = deepcopy(payload); corrupted_task["events"][0]["value"] = integer
            self.assertEqual(payload, corrupted_task)
            self.assertNotEqual(canonical(payload), canonical(corrupted_task))
            self.assertIs(type(solve(corrupted_task)["results"][0]["value"]), int)

    def test_belief_filtering_and_counts_separate_all_four_scalars(self):
        values = [("true", True), ("one", 1), ("false", False), ("zero", 0)]
        payload = {"kind": "belief", "worlds": [{"world_id": identity, "facts": {"flag": value}}
                   for identity, value in values], "observations": [], "queries": [{"query_id": "q", "fact": "flag"}]}
        result = solve(payload)["queries"][0]
        self.assertEqual([canonical(row["value"]) for row in result["counts"]], ["0", "1", "false", "true"])
        self.assertEqual(result["distinct_values"], 4)
        for identity, value in values:
            payload["observations"] = [{"fact": "flag", "equals": value}]
            result = solve(payload)
            self.assertEqual(result["surviving_worlds"], [identity])
            self.assertEqual(canonical(result["queries"][0]["guaranteed"]), canonical(value))

    def test_causal_equality_never_coerces_boolean_to_integer(self):
        for boolean, integer in [(True, 1), (False, 0)]:
            payload = {"kind": "causal", "state": {"x": 2, "y": 3},
                       "rule": {"op": "eq", "args": [{"const": boolean}, {"const": integer}]},
                       "precondition": {"const": True},
                       "events": [{"status": "ok", "amount": 2, "op": "add", "source": "x", "destination": "y"}]}
            result = solve(payload)
            self.assertIs(result["steps"][0]["cmp"]["rule"], False)
            self.assertEqual(result["after"], {"x": 2, "y": 3})
            payload["rule"]["op"] = "ne"
            self.assertIs(solve(payload)["steps"][0]["cmp"]["rule"], True)

    def test_predictive_partition_keeps_boolean_integer_observables_distinct(self):
        names = ["true", "one", "false", "zero"]
        payload = {"kind": "predictive_abstraction", "states": names, "actions": ["stay"],
                   "observable": dict(zip(names, [True, 1, False, 0])),
                   "transition": {name: {"stay": name} for name in names}}
        result = solve(payload)
        self.assertEqual(result["classes"], [["false"], ["one"], ["true"], ["zero"]])
        self.assertEqual([canonical(row["value"]) for row in result["abstract_observable"]], ["false", "1", "true", "0"])

    def test_protected_module_memory_preserves_exact_boolean_type(self):
        for boolean, integer in [(True, 1), (False, 0)]:
            payload = {"kind": "modules", "modules": [{"module_id": "m", "capabilities": [], "priority": 0}],
                       "memory": {"flag": boolean}, "protected": ["flag"],
                       "requests": [{"request_id": "r", "needs": [], "writes": [{"key": "flag", "value": integer}]}]}
            result = solve(payload)
            self.assertIs(type(result["memory"]["flag"]), bool)
            self.assertEqual(canonical(result["memory"]["flag"]), canonical(boolean))
            self.assertEqual(result["routes"][0]["blocked_writes"], [0])

    def test_booleans_are_rejected_in_integer_only_semantic_fields(self):
        for boolean in [True, False]:
            payload = {"kind": "episodic", "events": [{"key": "k", "version": boolean, "at": 0, "op": "put", "value": None}], "queries": []}
            with self.assertRaises(OracleError): solve(payload)
            payload = {"kind": "skills", "state": {"x": boolean, "y": 0}, "procedures": {}, "main": []}
            with self.assertRaises(OracleError): solve(payload)


if __name__ == "__main__":
    unittest.main()
