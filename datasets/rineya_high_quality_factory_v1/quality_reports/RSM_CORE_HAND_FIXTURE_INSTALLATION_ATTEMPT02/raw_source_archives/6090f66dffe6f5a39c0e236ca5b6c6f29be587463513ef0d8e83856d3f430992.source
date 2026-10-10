"""Fresh, independently derived properties for the existing episodic contract.

No producer, dataset label, stored target, or review is read. Exhaustive small
histories use pairwise dominance to identify the unique contract winner.
"""

from copy import deepcopy
from itertools import product
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracles.extended_independent import solve


def event(version, at, value=None, operation="put", key="k"):
    return {"key": key, "version": version, "at": at, "op": operation, "value": value}


def query(identity, as_of, key="k"):
    return {"query_id": identity, "key": key, "as_of": as_of}


def task(events, queries):
    return {"kind": "episodic", "events": events, "queries": queries}


def pairwise_winner(events, request):
    """Find the undominated eligible event; do not sort or call oracle helpers."""
    eligible = [(index, item) for index, item in enumerate(events)
                if item["key"] == request["key"] and item["at"] <= request["as_of"]]
    winners = []
    for index, item in eligible:
        dominated = False
        for other_index, other in eligible:
            if other["version"] > item["version"]:
                dominated = True
            elif other["version"] == item["version"] and other["at"] > item["at"]:
                dominated = True
            elif other["version"] == item["version"] and other["at"] == item["at"] and other_index > index:
                dominated = True
        if not dominated:
            winners.append((index, item))
    if not eligible:
        return {"query_id": request["query_id"], "key": request["key"], "status": "absent",
                "value": None, "chosen_version": None, "chosen_event": None}
    assert len(winners) == 1
    index, selected = winners[0]
    return {"query_id": request["query_id"], "key": request["key"],
            "status": "deleted" if selected["op"] == "delete" else "present",
            "value": selected["value"], "chosen_version": selected["version"], "chosen_event": index}


class EpisodicFreshProperties(unittest.TestCase):
    def assert_typed_equal(self, observed, expected):
        self.assertEqual(json.dumps(observed, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                         json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def test_delayed_winner_eligible_exactly_at_as_of(self):
        events = [event(8, 20, "delayed"), event(7, 1, "old high"), event(3, 19, "later low")]
        result = solve(task(events, [query("twenty", 20), query("nineteen", 19), query("early", 0)]))["results"]
        self.assertEqual([row["chosen_event"] for row in result], [0, 1, None])
        self.assertEqual([row["value"] for row in result], ["delayed", "old high", None])
        self.assertEqual([row["status"] for row in result], ["present", "present", "absent"])

    def test_filtering_precedes_version_priority(self):
        events = [event(999, 9, "future"), event(2, 3, "eligible"), event(1, 4, "lower later")]
        row = solve(task(events, [query("before_future", 8)]))["results"][0]
        self.assertEqual((row["chosen_event"], row["chosen_version"], row["value"]), (1, 2, "eligible"))

    def test_highest_version_overrides_later_lower_version(self):
        events = [event(9, 1, "authoritative"), event(8, 80, "late"), event(0, 100, "latest")]
        row = solve(task(events, [query("all", 100)]))["results"][0]
        self.assertEqual((row["value"], row["chosen_version"], row["chosen_event"]), ("authoritative", 9, 0))

    def test_same_version_uses_time_before_input_index(self):
        events = [event(4, 7, "latest time"), event(4, 2, "latest index")]
        self.assertEqual(solve(task(events, [query("q", 7)]))["results"][0]["chosen_event"], 0)

    def test_equal_version_time_index_ties_delete_and_present_null(self):
        events = [event(2, 5, "first"), event(2, 5, operation="delete")]
        row = solve(task(events, [query("q", 5)]))["results"][0]
        self.assertEqual((row["status"], row["value"], row["chosen_event"]), ("deleted", None, 1))
        events.append(event(2, 5, None))
        row = solve(task(events, [query("q", 5)]))["results"][0]
        self.assertEqual((row["status"], row["value"], row["chosen_event"]), ("present", None, 2))

    def test_high_version_tombstone_survives_late_lower_put(self):
        events = [event(6, 1, operation="delete"), event(5, 99, "late lower")]
        row = solve(task(events, [query("q", 100)]))["results"][0]
        self.assertEqual((row["status"], row["chosen_version"], row["chosen_event"]), ("deleted", 6, 0))

    def test_absent_deleted_and_put_null_are_distinct_results(self):
        events = [event(1, 0, None, key="present_null"), event(1, 0, operation="delete", key="deleted")]
        result = solve(task(events, [query("absent", 0, "missing"), query("null", 0, "present_null"),
                                    query("tombstone", 0, "deleted")]))["results"]
        self.assertEqual([row["status"] for row in result], ["absent", "present", "deleted"])
        self.assertEqual([row["value"] for row in result], [None, None, None])
        self.assertEqual([row["chosen_version"] for row in result], [None, 1, 1])
        self.assertEqual([row["chosen_event"] for row in result], [None, 0, 1])

    def test_exact_key_and_irrelevant_append_suppression(self):
        original = task([event(1, 0, "correct")], [query("q", 2), query("missing", 2, "K")])
        baseline = solve(original)
        changed = deepcopy(original)
        changed["events"].extend([event(100, 0, "different case", key="K2"),
                                  event(100, 0, "prefix", key="k-more"), event(100, 3, "future")])
        self.assert_typed_equal(solve(changed), baseline)

    def test_irrelevant_insertion_shifts_only_chosen_event_indices(self):
        original = task([event(1, 0, True)], [query("q", 1), query("missing", 1, "missing")])
        baseline = solve(original)
        original["events"].insert(0, event(99, 0, "irrelevant", key="other"))
        changed = solve(original)
        expected = deepcopy(baseline); expected["results"][0]["chosen_event"] = 1
        self.assert_typed_equal(changed, expected)

    def test_query_identifiers_and_input_order_do_not_change_winners(self):
        requests = [query("z-last-lexically", 5), query("", 0), query("é\nquery", 3), query("a", 5)]
        original = task([event(1, 1, False), event(2, 4, 1)], requests)
        result = solve(original)["results"]
        self.assertEqual([row["query_id"] for row in result], [item["query_id"] for item in requests])
        changed = deepcopy(original); changed["queries"].reverse()
        self.assert_typed_equal(solve(changed)["results"], list(reversed(result)))
        changed["queries"] = [query("renamed_" + str(index), item["as_of"], item["key"])
                              for index, item in enumerate(requests)]
        renamed = solve(changed)["results"]
        for before, after in zip(result, renamed):
            self.assert_typed_equal({key: value for key, value in before.items() if key != "query_id"},
                                    {key: value for key, value in after.items() if key != "query_id"})

    def test_unsorted_history_and_reordered_nonconflicting_events(self):
        events = [event(3, 7, "future high"), event(2, 4, "middle"), event(1, 0, "initial")]
        requests = [query("zero", 0), query("four", 4), query("seven", 7)]
        rows = solve(task(events, requests))["results"]
        self.assertEqual([row["chosen_event"] for row in rows], [2, 1, 0])
        reordered = solve(task(list(reversed(events)), requests))["results"]
        self.assertEqual([row["chosen_event"] for row in reordered], [0, 1, 2])
        for before, after in zip(rows, reordered):
            self.assert_typed_equal({key: value for key, value in before.items() if key != "chosen_event"},
                                    {key: value for key, value in after.items() if key != "chosen_event"})

    def test_exact_scalar_types_preserved(self):
        for value in [True, 1, False, 0, None, "1", "é"]:
            row = solve(task([event(0, 0, value)], [query("q", 0)]))["results"][0]
            self.assertIs(type(row["value"]), type(value))
            self.assert_typed_equal(row["value"], value)

    def test_exhaustive_small_histories_pairwise_dominance_and_monotonic_version(self):
        variants = [event(0, 0, None), event(0, 1, True), event(1, 0, operation="delete"),
                    event(1, 1, 1), event(1, 1, "tie"), event(2, 2, False),
                    event(3, 0, "other", key="noise"), event(2, 3, operation="delete"), event(2, 3, None)]
        requests = [query("query_" + str(index), time, key)
                    for index, (key, time) in enumerate(product(("k", "noise", "absent"), range(5)))]
        histories, query_checks = 0, 0
        for length in range(4):
            for events in product(variants, repeat=length):
                payload = task(list(events), requests)
                before = deepcopy(payload)
                result = solve(payload)["results"]
                expected = [pairwise_winner(events, request) for request in requests]
                self.assert_typed_equal(result, expected)
                self.assertEqual(payload, before)
                for key in ("k", "noise", "absent"):
                    versions = [row["chosen_version"] for row in result if row["key"] == key and row["chosen_version"] is not None]
                    self.assertEqual(versions, sorted(versions))
                for request, row in zip(requests, result):
                    if row["chosen_event"] is not None:
                        chosen = events[row["chosen_event"]]
                        self.assertEqual(chosen["key"], request["key"])
                        self.assertLessEqual(chosen["at"], request["as_of"])
                histories += 1; query_checks += len(requests)
        self.assertEqual(histories, 820)
        self.assertEqual(query_checks, 12300)


if __name__ == "__main__":
    unittest.main()
