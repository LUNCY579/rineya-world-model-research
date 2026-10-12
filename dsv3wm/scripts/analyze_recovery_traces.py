"""POST HOC supplementary recovery trace audit; stdlib only, no oracle/model.

Replay the declared frozen batched-rollout schedule using predicted states, then
consume declared direct batches. Never match a trace using a gold state, silently
repair an uncertain mapping, or revise the original P1 result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_raw(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite constant")
    try:
        value = json.loads(text.strip(), object_pairs_hook=unique, parse_constant=constant)
        if (not isinstance(value, dict) or set(value) != {"x", "legal", "terminated"}
                or type(value["x"]) is not int or type(value["legal"]) is not bool
                or type(value["terminated"]) is not bool):
            raise ValueError("invalid legacy response")
        return {"state": {"x": value["x"]}, "legal": value["legal"],
                "terminated": value["terminated"], "failed": not value["legal"]}
    except (ValueError, TypeError, AttributeError):
        return {"malformed": True, "text": text}


def analyze(records, traces, *, direct_horizons=(1,), horizon=4):
    if not records or horizon != 4 or direct_horizons not in ((1,), (1, 2, 4)):
        raise ValueError("fixed four-step cohort and explicit B0/adapted direct schedule required")
    parents = [record.get("parent_world_id") for record in records]
    if any(not isinstance(parent, str) or not parent for parent in parents) or len(set(parents)) != len(parents):
        raise ValueError("unique named source parents required")
    for record in records:
        if record.get("split") != "dev" or record.get("domain") != "register" or len(record.get("actions", [])) != 4:
            raise ValueError("only four-step RegisterWorld DEV rows allowed")
        if (not isinstance(record.get("s0"), dict) or set(record["s0"]) != {"x"} or type(record["s0"]["x"]) is not int
                or type(record.get("rules", {}).get("limit")) is not int or record["rules"]["limit"] <= 0):
            raise ValueError("strict initial state/rule representation required")
        labels = record["labels"]
        if any(len(labels.get(key, [])) != 4 for key in ("states", "legal", "terminated", "failed", "executed")):
            raise ValueError("complete DEV labels required")
        if (any(set(state) != {"x"} or type(state["x"]) is not int for state in labels["states"])
                or any(type(flag) is not bool for key in ("legal", "terminated", "failed", "executed") for flag in labels[key])):
            raise ValueError("strict state/flag labels required")
        if record["provenance"].get("cohort") not in {"recovery_id_teacher_known", "recovery_ood_limit"}:
            raise ValueError("unknown cohort cannot be silently pooled")
    cursor = 0
    def consume_batch(expected):
        nonlocal cursor
        if cursor + len(expected) > len(traces):
            raise ValueError("missing traces; schedule cannot be reconstructed")
        parsed, duplicates = [], {}
        for inputs, trace in zip(expected, traces[cursor:cursor + len(expected)]):
            if set(trace) != {"conditioning", "generated", "parsed"} or canonical(trace["conditioning"]) != canonical(inputs):
                raise ValueError(f"conditioning/order mismatch at trace {cursor + len(parsed)}")
            value = parse_raw(trace["generated"].get("text"))
            if canonical(value) != canonical(trace["parsed"]):
                raise ValueError("raw output and preserved parsed output disagree")
            signature, output = canonical(inputs), canonical(value)
            if signature in duplicates and duplicates[signature] != output:
                raise ValueError("indistinguishable conditionings with different outputs: uncertain parent mapping")
            duplicates[signature] = output
            parsed.append(value)
        cursor += len(expected)
        return parsed
    states = [copy.deepcopy(record["s0"]) for record in records]
    dead, terminated, failed = [False] * len(records), [False] * len(records), [False] * len(records)
    outputs = [[] for _ in records]
    for step in range(4):
        active = [index for index in range(len(records)) if not dead[index] and not terminated[index]]
        inputs = [{"domain": "register", "rules": records[index]["rules"], "s0": states[index],
                   "actions": [records[index]["actions"][step]]} for index in active]
        batch = consume_batch(inputs)
        for index, value in zip(active, batch):
            if value.get("malformed"):
                dead[index] = True
                # An already observed illegal attempt remains known failure.
                # Otherwise malformed execution leaves cumulative status unknown.
                failed[index] = True if failed[index] is True else None
                outputs[index].append({"state": None, "legal": None, "terminated": None,
                                       "attempted": True, "malformed": True, "failed": failed[index]})
            else:
                states[index] = copy.deepcopy(value["state"])
                terminated[index] = value["terminated"]
                failed[index] = failed[index] or not value["legal"]
                outputs[index].append({**value, "failed": failed[index], "attempted": True, "malformed": False})
        for index in range(len(records)):
            if index in active:
                continue
            if terminated[index]:
                outputs[index].append({"state": copy.deepcopy(states[index]), "legal": False, "terminated": True,
                                       "failed": failed[index], "attempted": False, "skipped": True, "malformed": False})
            elif dead[index]:
                outputs[index].append({"state": None, "legal": None, "terminated": None, "failed": failed[index],
                                       "attempted": False, "abstained": True, "malformed": True})
    direct = {}
    for depth in direct_horizons:
        inputs = [{"domain": "register", "rules": record["rules"], "s0": record["s0"],
                   "actions": record["actions"][:depth]} for record in records]
        direct[str(depth)] = consume_batch(inputs)
    if cursor != len(traces):
        raise ValueError("trailing traces imply an undeclared/uncertain evaluation schedule")
    per_parent = []
    for index, record in enumerate(records):
        steps = []
        for step, value in enumerate(outputs[index]):
            labels = record["labels"]
            known = type(value["failed"]) is bool
            steps.append({**value, "state_correct": value["state"] == labels["states"][step],
                          "legality_correct": type(value["legal"]) is bool and value["legal"] == labels["legal"][step],
                          "termination_correct": type(value["terminated"]) is bool and value["terminated"] == labels["terminated"][step],
                          "failure_known": known, "cumulative_failure_correct": known and value["failed"] == labels["failed"][step]})
        per_parent.append({"parent_world_id": record["parent_world_id"],
                           "cohort": "id" if record["provenance"]["cohort"] == "recovery_id_teacher_known" else "ood",
                           "steps": steps, "P1_final_state_correct": steps[3]["state_correct"],
                           "correct_final_wrong_intermediate": steps[3]["state_correct"] and any(not row["state_correct"] for row in steps[:3]),
                           "direct_diagnostics": {depth: {"malformed": values[index].get("malformed", False),
                               "state_correct": not values[index].get("malformed") and values[index]["state"] == record["labels"]["states"][int(depth) - 1],
                               "cumulative_failure": values[index].get("failed") if depth == "1" else None,
                               "cumulative_failure_scope": "single attempted prediction" if depth == "1" else
                                   "unknown: direct last-action legality is not multi-step cumulative failure"}
                               for depth, values in direct.items()}})
    summary = {}
    for cohort in ("id", "ood", "all"):
        selected = [row for row in per_parent if cohort == "all" or row["cohort"] == cohort]
        summary[cohort] = {"n": len(selected), "correct_final_wrong_intermediate": sum(row["correct_final_wrong_intermediate"] for row in selected),
                           "steps": {}}
        for step in range(4):
            values = [row["steps"][step] for row in selected]
            fields = ("state_correct", "legality_correct", "termination_correct", "failure_known", "cumulative_failure_correct", "malformed", "attempted")
            metrics = {field: sum(bool(row.get(field)) for row in values) for field in fields}
            metrics.update(n=len(values), failure_unknown=len(values) - metrics["failure_known"],
                           skipped=sum(bool(row.get("skipped")) for row in values),
                           attempted_legality_correct=sum(row["legality_correct"] for row in values if row["attempted"]))
            summary[cohort]["steps"][str(step + 1)] = metrics
    return {"analysis_status": "posthoc supplementary; no native efficacy/confirmation inference",
            "scope": "preserved generated text and DEV labels; no model, simulator, oracle transition or repair",
            "P1_scope": "original four-step final-state-only metric unchanged; this is additional trace analysis",
            "mapping": {"policy": "strict source order / active batched rollout / declared direct schedule",
                        "trace_count": cursor, "direct_horizons": list(direct_horizons),
                        "ambiguous_mapping": "reject mismatches, unknown cohorts, missing/trailing traces and conflicting identical inputs"},
            "calibration": {"status": "unknown", "reason": "no probability/logit captures; boolean legality is not calibrated confidence"},
            "summary": summary, "per_parent": per_parent}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="labelled recovery DEV JSONL only")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--result", required=True, help="preserved native recovery result to bind hashes and P1")
    parser.add_argument("--out", required=True, help="new supplementary posthoc JSON path")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = json.loads(Path(args.traces).read_text(encoding="utf-8"))
    original = json.loads(Path(args.result).read_text(encoding="utf-8"))
    if original.get("status") != "development-evaluated" or original.get("data_sha256") != file_hash(args.data) or original.get("traces_sha256") != file_hash(args.traces):
        raise ValueError("native result/data/trace provenance mismatch")
    horizons = (1,) if "--baseline-only" in original["command"] else (1, 2, 4)
    result = analyze(records, traces, direct_horizons=horizons)
    preserved = {row["parent_world_id"]: row["horizons"]["4"]["rollout_correct"] for row in original["per_parent"]}
    if preserved != {row["parent_world_id"]: row["P1_final_state_correct"] for row in result["per_parent"]}:
        raise ValueError("supplementary reconstruction disagrees with preserved P1; do not revise original outcomes")
    result["provenance"] = {"data_sha256": file_hash(args.data), "traces_sha256": file_hash(args.traces),
                            "original_result_sha256": file_hash(args.result), "analysis_sha256": file_hash(__file__),
                            "original_P1_verified_unchanged": True}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
