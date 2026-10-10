#!/opt/anaconda3/bin/python3
"""Fresh expert implementation from public episodic semantics; no factory imports.

This file intentionally does not reuse prior reviewer solvers.  It is prepared
before memory-gap-v1 candidates are available.  Canonical gold is compared only
after deriving an answer from the student-visible prompt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

EXECUTION_ID = "/root/expert_reviewer"
BASE = Path(__file__).resolve().parents[2]


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def exact_equal(left, right):
    """Avoid Python's bool/int equality for public type-sensitive JSON values."""
    return encode(left) == encode(right)


def scalar(value):
    return value is None or type(value) in (bool, int, str)


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def validate_task(task):
    if not isinstance(task, dict) or task.get("kind") != "episodic":
        raise ValueError("Task is not an episodic object")
    if not isinstance(task.get("events"), list) or not isinstance(task.get("queries"), list):
        raise ValueError("events and queries must be lists")
    for index, event in enumerate(task["events"]):
        if set(event) != {"key", "version", "at", "op", "value"}:
            raise ValueError(f"Event {index} has unexpected fields")
        if type(event["key"]) is not str:
            raise ValueError(f"Event {index} key is not a string")
        if not all(nonnegative_integer(event[name]) for name in ("version", "at")):
            raise ValueError(f"Event {index} version/time is not a nonnegative integer")
        if event["op"] not in ("put", "delete") or not scalar(event["value"]):
            raise ValueError(f"Event {index} has invalid operation/value")
        if event["op"] == "delete" and event["value"] is not None:
            raise ValueError(f"Event {index} delete value must be null")
    seen = set()
    for index, query in enumerate(task["queries"]):
        if set(query) != {"query_id", "key", "as_of"}:
            raise ValueError(f"Query {index} has unexpected fields")
        if type(query["query_id"]) is not str or type(query["key"]) is not str:
            raise ValueError(f"Query {index} ID/key is not a string")
        if not nonnegative_integer(query["as_of"]):
            raise ValueError(f"Query {index} cutoff is not a nonnegative integer")
        if query["query_id"] in seen:
            raise ValueError(f"Query {index} repeats an ID")
        seen.add(query["query_id"])


def result_from_event(query, selected, index):
    result = {"query_id": query["query_id"], "key": query["key"],
              "status": "absent", "value": None,
              "chosen_version": None, "chosen_event": None}
    if selected is not None:
        result.update(status="deleted" if selected["op"] == "delete" else "present",
                      value=None if selected["op"] == "delete" else selected["value"],
                      chosen_version=selected["version"], chosen_event=index)
    return result


def solve_scan(task):
    """Primary solver: a query-local running maximum of the contract tuple."""
    validate_task(task)
    results = []
    evidence = []
    for query in task["queries"]:
        selected, selected_index, selected_rank = None, None, None
        eligible = []
        for index, event in enumerate(task["events"]):
            if event["key"] != query["key"] or event["at"] > query["as_of"]:
                continue
            eligible.append(index)
            rank = (event["version"], event["at"], index)
            if selected_rank is None or rank > selected_rank:
                selected, selected_index, selected_rank = event, index, rank
        results.append(result_from_event(query, selected, selected_index))
        evidence.append({"query_id": query["query_id"], "key": query["key"],
                         "as_of": query["as_of"], "eligible_indices": eligible,
                         "selected_index": selected_index,
                         "selected_tuple": list(selected_rank) if selected_rank else None})
    return {"results": results}, evidence


def solve_sorted(task):
    """Secondary solver: sort once by precedence and select first eligible."""
    validate_task(task)
    order = sorted(range(len(task["events"])),
                   key=lambda i: (task["events"][i]["version"], task["events"][i]["at"], i),
                   reverse=True)
    output = []
    for query in task["queries"]:
        for index in order:
            event = task["events"][index]
            if event["key"] == query["key"] and event["at"] <= query["as_of"]:
                output.append(result_from_event(query, event, index))
                break
        else:
            output.append(result_from_event(query, None, None))
    return {"results": output}


def public_prompt_task(prompt, public_text):
    if type(prompt) is not str or prompt.count("Task JSON:\n") != 1:
        raise ValueError("Prompt must have one exact Task JSON delimiter")
    contract, serialized = prompt.split("Task JSON:\n")
    section = public_text.split("## `kind=episodic`\n", 1)[1].split("\n## ", 1)[0].strip()
    if section not in contract:
        raise ValueError("Prompt does not contain the full public episodic contract")
    task = json.loads(serialized)
    validate_task(task)
    return task


def fresh_string(existing, stem):
    candidate = stem
    while candidate in existing:
        candidate += "_"
    return candidate


def metamorphic_checks(task):
    expected, _ = solve_scan(task)
    checks = []

    def check(name, transformed, transformed_expected):
        answer, _ = solve_scan(transformed)
        checks.append({"name": name, "passed": exact_equal(answer, transformed_expected)})
        if not exact_equal(answer, solve_sorted(transformed)):
            raise AssertionError("Independent episodic implementations disagree")

    changed = copy.deepcopy(task)
    id_map = {q["query_id"]: f"expert_opaque_{len(task['queries']) - i:04d}"
              for i, q in enumerate(task["queries"])}
    for query in changed["queries"]:
        query["query_id"] = id_map[query["query_id"]]
    answer = copy.deepcopy(expected)
    for result in answer["results"]:
        result["query_id"] = id_map[result["query_id"]]
    check("query_id_renaming", changed, answer)

    changed = copy.deepcopy(task)
    changed["queries"].reverse()
    check("query_order_reversal", changed, {"results": list(reversed(expected["results"]))})

    keys = sorted({e["key"] for e in task["events"]} | {q["key"] for q in task["queries"]})
    key_map = {key: f"expert_key_{len(keys) - i:04d}" for i, key in enumerate(keys)}
    changed, answer = copy.deepcopy(task), copy.deepcopy(expected)
    for item in changed["events"] + changed["queries"]:
        item["key"] = key_map[item["key"]]
    for result in answer["results"]:
        result["key"] = key_map[result["key"]]
    check("key_name_bijection", changed, answer)

    changed = copy.deepcopy(task)
    irrelevant = fresh_string(set(keys), "expert_irrelevant")
    highest = max([e["version"] for e in task["events"]] + [0]) + 1
    changed["events"].append({"key": irrelevant, "version": highest,
                              "at": 0, "op": "put", "value": "noise"})
    check("append_irrelevant_key", changed, expected)

    changed = copy.deepcopy(task)
    future = max([q["as_of"] for q in task["queries"]] + [0]) + 1
    for key in keys:
        changed["events"].append({"key": key, "version": highest,
                                  "at": future, "op": "put", "value": "future"})
    check("append_future_higher_versions", changed, expected)

    # Permute precedence groups but preserve order within each exact (key,v,t)
    # tie group.  This cannot change winner semantics, only selected indices.
    groups = {}
    for index, event in enumerate(task["events"]):
        groups.setdefault((event["key"], event["version"], event["at"]), []).append(index)
    perm = [i for key in sorted(groups, reverse=True) for i in groups[key]]
    index_map = {old: new for new, old in enumerate(perm)}
    changed = copy.deepcopy(task)
    changed["events"] = [copy.deepcopy(task["events"][i]) for i in perm]
    answer = copy.deepcopy(expected)
    for result in answer["results"]:
        if result["chosen_event"] is not None:
            result["chosen_event"] = index_map[result["chosen_event"]]
    check("precedence_group_permutation_with_index_remap", changed, answer)

    absent = fresh_string(set(keys), "expert_absent")
    new_id = fresh_string({q["query_id"] for q in task["queries"]}, "expert_new_query")
    changed = copy.deepcopy(task)
    query = {"query_id": new_id, "key": absent, "as_of": future}
    changed["queries"].append(query)
    check("append_absent_query", changed,
          {"results": expected["results"] + [result_from_event(query, None, None)]})

    # JSON object order/whitespace are presentation controls, never semantics.
    changed = json.loads(json.dumps(task, indent=3, ensure_ascii=False))
    check("json_serialization_invariance", changed, expected)
    if not all(c["passed"] for c in checks):
        raise AssertionError("An episodic metamorphic check failed")
    return checks


def query_diagnostics(task):
    answer, traces = solve_scan(task)
    diagnostics = []
    for position, (query, result, trace) in enumerate(zip(task["queries"], answer["results"], traces)):
        index = result["chosen_event"]
        row = dict(trace, query_position=position, status=result["status"], value=result["value"])
        if index is not None:
            event = task["events"][index]
            eligible = [task["events"][i] for i in trace["eligible_indices"]]
            row.update(events_after_winner=len(task["events"]) - index - 1,
                       later_input_lower_version=any(i > index and task["events"][i]["version"] < event["version"]
                                                     for i in trace["eligible_indices"]),
                       later_time_lower_version=any(e["at"] > event["at"] and e["version"] < event["version"]
                                                    for e in eligible),
                       version_tie=any(i != index and task["events"][i]["version"] == event["version"]
                                       for i in trace["eligible_indices"]),
                       version_time_tie=any(i != index and task["events"][i]["version"] == event["version"]
                                            and task["events"][i]["at"] == event["at"]
                                            for i in trace["eligible_indices"]))
            changed = copy.deepcopy(task)
            del changed["events"][index]
            replacement = solve_scan(changed)[0]["results"][position]
            row["without_winner"] = replacement
            row["winner_payload_dependency"] = not exact_equal(
                {k: result[k] for k in ("status", "value")},
                {k: replacement[k] for k in ("status", "value")})
            row["winner_version_dependency"] = result["chosen_version"] != replacement["chosen_version"]
        diagnostics.append(row)
    return diagnostics


def toy_task(events, queries):
    return {"kind": "episodic", "events": events, "queries": queries}


def preflight():
    # Manually specified public-contract examples, not producer expectations.
    events = [
        {"key": "a", "version": 4, "at": 2, "op": "put", "value": "old-high"},
        {"key": "a", "version": 3, "at": 9, "op": "put", "value": "late-low"},
        {"key": "a", "version": 4, "at": 5, "op": "put", "value": None},
        {"key": "a", "version": 4, "at": 5, "op": "delete", "value": None},
        {"key": "a", "version": 8, "at": 21, "op": "put", "value": "future"},
        {"key": "bool", "version": 0, "at": 0, "op": "put", "value": True},
        {"key": "int", "version": 0, "at": 0, "op": "put", "value": 1},
        {"key": "other", "version": 100, "at": 0, "op": "put", "value": "noise"},
    ]
    queries = [
        {"query_id": "late", "key": "a", "as_of": 20},
        {"query_id": "before", "key": "a", "as_of": 1},
        {"query_id": "early", "key": "a", "as_of": 4},
        {"query_id": "absent", "key": "missing", "as_of": 100},
        {"query_id": "future_now", "key": "a", "as_of": 21},
        {"query_id": "truth", "key": "bool", "as_of": 0},
        {"query_id": "number", "key": "int", "as_of": 0},
    ]
    task = toy_task(events, queries)
    expected = {"results": [
        {"query_id": "late", "key": "a", "status": "deleted", "value": None, "chosen_version": 4, "chosen_event": 3},
        {"query_id": "before", "key": "a", "status": "absent", "value": None, "chosen_version": None, "chosen_event": None},
        {"query_id": "early", "key": "a", "status": "present", "value": "old-high", "chosen_version": 4, "chosen_event": 0},
        {"query_id": "absent", "key": "missing", "status": "absent", "value": None, "chosen_version": None, "chosen_event": None},
        {"query_id": "future_now", "key": "a", "status": "present", "value": "future", "chosen_version": 8, "chosen_event": 4},
        {"query_id": "truth", "key": "bool", "status": "present", "value": True, "chosen_version": 0, "chosen_event": 5},
        {"query_id": "number", "key": "int", "status": "present", "value": 1, "chosen_version": 0, "chosen_event": 6},
    ]}
    assert exact_equal(solve_scan(task)[0], expected)
    assert exact_equal(solve_sorted(task), expected)
    null_task = toy_task([{ "key": "null-key", "version": 0, "at": 0,
                           "op": "put", "value": None}],
                         [{"query_id": "n", "key": "null-key", "as_of": 0}])
    assert solve_scan(null_task)[0]["results"][0]["status"] == "present"
    # One same-version later-time put supersedes an earlier delete; version wins
    # over a much later lower-version delete, and input order resolves exact ties.
    tie_task = toy_task([
        {"key": "t", "version": 2, "at": 2, "op": "delete", "value": None},
        {"key": "t", "version": 2, "at": 3, "op": "put", "value": False},
        {"key": "t", "version": 1, "at": 9, "op": "delete", "value": None}],
        [{"query_id": "tq", "key": "t", "as_of": 10}])
    tie_result = solve_scan(tie_task)[0]["results"][0]
    assert tie_result["chosen_event"] == 1 and tie_result["status"] == "present" and tie_result["value"] is False
    rejection_cases = []
    for label, altered in [("boolean_version", copy.deepcopy(task)),
                           ("non_null_delete", copy.deepcopy(task)),
                           ("duplicate_query_id", copy.deepcopy(task))]:
        if label == "boolean_version":
            altered["events"][0]["version"] = True
        elif label == "non_null_delete":
            altered["events"][3]["value"] = 0
        else:
            altered["queries"][1]["query_id"] = altered["queries"][0]["query_id"]
        try:
            validate_task(altered)
        except ValueError:
            rejection_cases.append(label)
        else:
            raise AssertionError(f"Invalid public task accepted: {label}")
    public_path = BASE / "schemas/EXTENDED_SEMANTICS.md"
    public_text = public_path.read_text()
    section = public_text.split("## `kind=episodic`\n", 1)[1].split("\n## ", 1)[0].strip()
    parsed = public_prompt_task(section + "\nTask JSON:\n" + json.dumps(task), public_text)
    assert exact_equal(parsed, task)
    properties = BASE / "schemas/MEMORY_CONTROL_PROPERTIES.md"
    report = {
        "reviewer_execution_id": EXECUTION_ID,
        "stage": "PREPARED_PUBLIC_CONTRACT_ONLY",
        "dataset_verdict_issued": False,
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_contract_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "memory_properties_available": properties.exists(),
        "memory_properties_sha256": hashlib.sha256(properties.read_bytes()).hexdigest() if properties.exists() else None,
        "primary_method": "Per-query scan of eligible events with lexicographic (version,time,input-index) maximum",
        "secondary_method": "Descending global event precedence ordering and first query-eligible event",
        "manual_contract_examples": {"precedence_boundary": expected, "put_null_status": "present", "later_time_and_lower_version": tie_result},
        "metamorphics": metamorphic_checks(task) + metamorphic_checks(null_task) + metamorphic_checks(tie_task),
        "invalid_examples_rejected": rejection_cases,
        "query_dependency_evidence": query_diagnostics(task),
        "read_scope": ["schemas/EXTENDED_SEMANTICS.md", "schemas/MEMORY_CONTROL_PROPERTIES.md if present", "this new own script"],
        "blindness": "No candidates, generators, reference/oracles, prior reviewer verdicts or producer coverage reports read at preparation stage",
        "pending": ["actual candidate path", "public delayed-dependency/tag/difficulty definitions", "canonical row payload and hash binding", "actual per-record judgments and semantic inspection"],
    }
    path = BASE / "reviews/expert/memory-gap-v1.preparation.json"
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(path), "manual_example_sets": 3, "metamorphic_checks": len(report["metamorphics"]), "all_passed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        parser.error("Candidate runner will be added after the public batch and properties exist")
