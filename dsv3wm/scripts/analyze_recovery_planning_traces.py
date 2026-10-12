"""POST HOC DEV planning audit. No models, new predictions or P2 changes.

Query truth is conditional on that query's OWN input state. Selected-path truth
is independently replayed from the actual task initial state. These are distinct
instruments; repeated search queries are not independent statistical trials.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(tasks, traces, native, *, exact_step=None, exact_rollout=None):
    from dsv3wm.recovery_planning import validate_tasks
    from dsv3wm.recovery import legacy_parse, INITIAL_F_SHA
    validate_tasks(tasks)
    if (native.get("status") != "development-evaluated" or native.get("oracle_feedback") is not False
            or native.get("all_choices_fixed_before_scoring") is not True):
        raise ValueError("only completed fixed-choice DEV planning records may be audited")
    lookup = {task["id"]: task for task in tasks}
    choices = native["per_parent"]
    if len(choices) != len(tasks) or {row["task_id"] for row in choices} != set(lookup):
        raise ValueError("native choice/task identity mismatch")
    by_choice = {row["task_id"]: row for row in choices}
    parsed, nodes, per_task = [], {}, {task["id"]: [] for task in tasks}
    moves = {"add1", "sub1", "double", "halve", "finish"}
    for index, trace in enumerate(traces):
        task = lookup.get(trace["task_id"])
        if task is None or trace["parent_world_id"] != task["parent_world_id"]:
            raise ValueError("query task/parent identity mismatch")
        parent, action, depth = trace["parent_plan"], trace["action"], trace["depth"]
        if (not isinstance(parent, list) or any(type(move) is not str or move not in moves for move in parent)
                or type(action) is not str or action not in moves or type(depth) is not int or depth != len(parent) + 1
                or trace["batch_index"] != depth - 1):
            raise ValueError("invalid query prefix/depth/action tags")
        prefix = tuple(parent + [action])
        identity = task["id"], prefix
        if identity in nodes:
            raise ValueError("ambiguous duplicate query prefix; cannot reconstruct ancestry")
        raw, inputs = trace["raw"], trace["raw"]["conditioning"]
        if (set(inputs) != {"domain", "rules", "s0", "actions"} or inputs["domain"] != "register"
                or canonical(inputs["rules"]) != canonical(task["rules"]) or inputs["actions"] != [action]
                or not isinstance(inputs["s0"], dict) or set(inputs["s0"]) != {"x"} or type(inputs["s0"]["x"]) is not int):
            raise ValueError("invalid own-conditioned query input")
        value = legacy_parse(raw["generated"].get("text"))
        if canonical(value) != canonical(raw["parsed"]):
            raise ValueError("query raw text/preserved parsing mismatch")
        for field in ("prompt_tokens", "output_tokens", "padded_processed_positions_upper_bound", "batch_size"):
            if type(trace[field]) is not int or trace[field] < (1 if field in ("prompt_tokens", "batch_size") else 0):
                raise ValueError("invalid raw query counter")
        if trace["output_tokens"] != raw["generated"]["n_tokens"]:
            raise ValueError("query output token counter mismatch")
        if trace["prompt_tokens"] > native["preflight_proof"]["max_prompt_tokens"] or trace["output_tokens"] > native["preflight_proof"]["generation_tokens"]:
            raise ValueError("query exceeded the recorded prompt/generation bound")
        item = {"trace_index": index, "trace": trace, "value": value, "inputs": inputs, "prefix": prefix}
        nodes[identity] = item
        per_task[task["id"]].append(item)
        parsed.append(item)
    batches = {}
    for item in parsed:
        batches.setdefault(item["trace"]["batch_index"], []).append(item["trace"])
    for batch in batches.values():
        padded = max(row["prompt_tokens"] for row in batch) + max(row["output_tokens"] for row in batch)
        if any(row["batch_size"] != len(batch) or row["padded_processed_positions_upper_bound"] != padded for row in batch):
            raise ValueError("broad-batch size/padding counter mismatch")
    # Resolve every query's learned ancestry before any exact scoring. Its input
    # must be the stored parent prediction, never a guessed/gold reset.
    for item in parsed:
        trace, task = item["trace"], lookup[item["trace"]["task_id"]]
        parent = tuple(trace["parent_plan"])
        if not parent:
            expected = task["s0"]
        else:
            ancestor = nodes.get((task["id"], parent))
            if (ancestor is None or ancestor["trace_index"] >= item["trace_index"]
                    or ancestor["value"].get("malformed") or not ancestor["value"]["legal"]
                    or ancestor["value"]["failed"] or ancestor["value"]["terminated"]):
                raise ValueError("missing/refused/terminated or out-of-order learned ancestor")
            expected = ancestor["value"]["state"]
        if canonical(item["inputs"]["s0"]) != canonical(expected):
            raise ValueError("own predicted-state ancestry mismatch; no relabelling permitted")
    paths = {}
    for task in tasks:
        choice, query_rows = by_choice[task["id"]], per_task[task["id"]]
        if choice["parent_world_id"] != task["parent_world_id"]:
            raise ValueError("selected parent identity mismatch")
        checks = {"model_queries": len(query_rows), "input_tokens": sum(row["trace"]["prompt_tokens"] for row in query_rows),
                  "output_tokens": sum(row["trace"]["output_tokens"] for row in query_rows),
                  "processed_positions_upper_bound": sum(row["trace"]["padded_processed_positions_upper_bound"] for row in query_rows)}
        if any(choice[key] != value for key, value in checks.items()):
            raise ValueError("native counters disagree with preserved raw queries")
        if checks["model_queries"] > native["budget"]["max_model_calls"] or checks["processed_positions_upper_bound"] > native["budget"]["max_processed_positions"]:
            raise ValueError("recorded planning budget exceeded")
        plan = choice["choice"]
        if plan is None:
            paths[task["id"]] = None
            continue
        if not isinstance(plan, list) or len(plan) > native["budget"]["max_depth"]:
            raise ValueError("selected plan violates recorded depth contract")
        path = []
        for depth in range(1, len(plan) + 1):
            node = nodes.get((task["id"], tuple(plan[:depth])))
            if node is None or node["value"].get("malformed") or not node["value"]["legal"] or node["value"]["failed"]:
                raise ValueError("selected plan lacks unique accepted learned ancestry")
            path.append(node)
        endpoint = path[-1]["value"]["state"] if path else task["s0"]
        if (canonical(endpoint) != canonical(choice["predicted_state"])
                or choice["predicted_success"] is not (endpoint == task["goal"]["state"])):
            raise ValueError("selected endpoint disagrees with preserved choice")
        paths[task["id"]] = path
    # Inputs, paths and native counters are now fixed. Exact rules only add
    # diagnostics below; no generation/search function exists in this module.
    if exact_step is None or exact_rollout is None:
        from dsv3wm.domains import exact_transition, rollout_labels
        exact_step = exact_transition if exact_step is None else exact_step
        exact_rollout = rollout_labels if exact_rollout is None else exact_rollout
    query_audits = []
    for item in parsed:
        trace, inputs, value = item["trace"], item["inputs"], item["value"]
        state, legal, terminated, invalid = exact_step("register", copy.deepcopy(inputs["s0"]), trace["action"], copy.deepcopy(inputs["rules"]))
        valid = not value.get("malformed")
        audit = {"task_id": trace["task_id"], "parent_world_id": trace["parent_world_id"],
                 "trace_index": item["trace_index"], "depth": trace["depth"], "parent_plan": trace["parent_plan"],
                 "action": trace["action"], "own_conditioning": copy.deepcopy(inputs), "schema_valid": valid,
                 "expected_on_own_input": {"state": state, "legal": legal, "terminated": terminated, "invalid": invalid},
                 "predicted": copy.deepcopy(value), "raw_text": trace["raw"]["generated"]["text"],
                 "state_correct": valid and value["state"] == state, "state_error": valid and value["state"] != state,
                 "legality_correct": valid and value["legal"] == legal, "legality_error": valid and value["legal"] != legal,
                 "termination_correct": valid and value["terminated"] == terminated,
                 "termination_error": valid and value["terminated"] != terminated,
                 "out_of_range_conditioning": abs(inputs["s0"]["x"]) > inputs["rules"]["limit"],
                 "prompt_tokens": trace["prompt_tokens"], "output_tokens": trace["output_tokens"],
                 "padded_processed_positions_upper_bound": trace["padded_processed_positions_upper_bound"]}
        query_audits.append(audit)
    def aggregate(rows):
        return {"queries": len(rows), "schema_valid": sum(row["schema_valid"] for row in rows),
                "malformed_unknown": sum(not row["schema_valid"] for row in rows),
                **{field: sum(row[field] for row in rows) for field in
                   ("state_correct", "state_error", "legality_correct", "legality_error", "termination_correct", "termination_error", "out_of_range_conditioning")},
                "finish": {"queries": sum(row["action"] == "finish" for row in rows),
                           "term_correct": sum(row["termination_correct"] for row in rows if row["action"] == "finish"),
                           "term_error": sum(row["termination_error"] for row in rows if row["action"] == "finish"),
                           "term_unknown": sum(not row["schema_valid"] for row in rows if row["action"] == "finish")},
                "nonfinish": {"queries": sum(row["action"] != "finish" for row in rows),
                              "term_correct": sum(row["termination_correct"] for row in rows if row["action"] != "finish"),
                              "term_error": sum(row["termination_error"] for row in rows if row["action"] != "finish"),
                              "term_unknown": sum(not row["schema_valid"] for row in rows if row["action"] != "finish")}}
    selected_audits = []
    for task in tasks:
        choice, path = by_choice[task["id"]], paths[task["id"]]
        if path is None:
            if choice["success"] is not False:
                raise ValueError("no-choice native success mismatch")
            selected_audits.append({"task_id": task["id"], "parent_world_id": task["parent_world_id"],
                                    "choice": None, "original_P2_success": False, "steps": [],
                                    "correct_goal_wrong_intermediate": False, "nonterminating_goal_only_success": False})
            continue
        plan = choice["choice"]
        labels = exact_rollout("register", copy.deepcopy(task["s0"]), copy.deepcopy(plan), copy.deepcopy(task["rules"]))
        actual_final = labels["states"][-1] if plan else task["s0"]
        all_legal = all(labels["legal"]) and all(labels["executed"])
        goal_success = all_legal and actual_final == task["goal"]["state"]
        actual_term = labels["terminated"][-1] if plan else False
        if goal_success is not choice["success"]:
            raise ValueError("exact posthoc replay disagrees with original goal-only P2; original must not be rewritten")
        steps = [{"step": index + 1, "trace_index": node["trace_index"], "action": plan[index],
                  "predicted": copy.deepcopy(node["value"]), "actual_state": labels["states"][index],
                  "actual_legal": labels["legal"][index], "actual_terminated": labels["terminated"][index],
                  "actual_executed": labels["executed"][index],
                  "state_correct_on_actual_trajectory": node["value"]["state"] == labels["states"][index],
                  "legality_correct_on_actual_trajectory": node["value"]["legal"] == labels["legal"][index],
                  "termination_correct_on_actual_trajectory": node["value"]["terminated"] == labels["terminated"][index]}
                 for index, node in enumerate(path)]
        selected_audits.append({"task_id": task["id"], "parent_world_id": task["parent_world_id"], "choice": plan,
                                "original_P2_success": goal_success, "actual_final_state": actual_final,
                                "actual_terminated": actual_term, "steps": steps,
                                "correct_goal_wrong_intermediate": goal_success and any(not step["state_correct_on_actual_trajectory"] for step in steps[:-1]),
                                "nonterminating_goal_only_success": goal_success and not actual_term})
    arm = "B0" if native.get("adapter_sha256") == INITIAL_F_SHA else native["config"]["arm"]
    return {"analysis_status": "posthoc supplementary DEV audit; no new inference/confirmation claim", "arm": arm,
            "query_truth_scope": "exact single step on each query's own conditioned state, not ground-truth task trajectory",
            "statistical_unit_warning": "repeated branch queries/parents are dependent; no binomial trial inference",
            "counter_scope": "logical transition queries and recorded broad-batch padding upper bounds; not FLOPs or exact internal decoder calls",
            "original_P2_unchanged": True, "oracle_feedback": False,
            "summary": {**aggregate(query_audits), "unique_conditionings": len({canonical(row["own_conditioning"]) for row in query_audits}),
                        "selected_parents": len(tasks), "goal_only_successes": sum(row["original_P2_success"] for row in selected_audits),
                        "correct_goal_wrong_intermediate": sum(row["correct_goal_wrong_intermediate"] for row in selected_audits),
                        "nonterminating_goal_only_successes": sum(row["nonterminating_goal_only_success"] for row in selected_audits)},
            "per_parent_queries": [{"task_id": task["id"], "parent_world_id": task["parent_world_id"],
                                     **aggregate([row for row in query_audits if row["task_id"] == task["id"]])} for task in tasks],
            "query_audits": query_audits, "selected_ancestry_audits": selected_audits,
            "calibration": {"status": "unknown", "reason": "no predictive probabilities captured"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "traces", "tasks", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    native = json.loads(args.result.read_text(encoding="utf-8"))
    if native.get("status") != "development-evaluated" or native.get("tasks_sha256") != file_hash(args.tasks) or native.get("traces_sha256") != file_hash(args.traces):
        raise ValueError("incomplete native result or result/tasks/traces hash mismatch")
    from dsv3wm import domains
    oracle_sources = [Path(domains.__file__).resolve(), domains.SOURCES["register"].resolve()]
    recorded_sources = {Path(name).resolve(): digest for name, digest in native.get("source_sha256", {}).items()}
    if any(recorded_sources.get(source) != file_hash(source) for source in oracle_sources):
        raise ValueError("posthoc exact-rule source differs from native source record")
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = json.loads(args.traces.read_text(encoding="utf-8"))
    result = analyze(tasks, traces, native)
    result["provenance"] = {"original_result_sha256": file_hash(args.result), "tasks_sha256": file_hash(args.tasks),
                            "traces_sha256": file_hash(args.traces), "analysis_sha256": file_hash(__file__),
                            "exact_rule_source_sha256": {str(source): file_hash(source) for source in oracle_sources}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
