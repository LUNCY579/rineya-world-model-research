"""Posthoc source-terminal reward/regret from preserved recovery P2 verification.

Never runs a model, transition rule, solver, or plan repair. Original goal-only
P2 success and success-conditional excess cost remain unchanged.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze_recovery_planning_costs as prior
from dsv3wm.stats import clustered_bootstrap

SCHEMA = "dsv3wm-posthoc-source-terminal-regret-v1"
TASKS_SHA = "7cc5b60412ef1d9793dd38ebeb05067b722e3b54010ff1113695e915594cc69c"
RW2_SHA = "1195ea10044b28cecb7f30244aa2e51628222748c1e15d6bebb68ad5d11d3da9"
ACTIONS = {"add1", "sub1", "double", "halve", "finish"}
CONTRACT = {"reward_id": "register-v2-success-step-violation", "terminal_goal_reward": 1.,
            "executed_step_cost": .01, "executed_illegal_cost": .1, "action_cap": 6,
            "universal_return_lower_bound": -.66, "optimal_return": "1-.01*(optimal_moves+1)",
            "normalized_regret": "(optimal_return-source_return)/(optimal_return+.66)",
            "normalization_scope": "universal conservative cap6 bound, not attainable worst for every parent; not RCWM candidate-mean regret",
            "missing_plan_policy": "strict refusal; no null/malformed/incomplete to empty-plan substitution",
            "empty_plan_policy": "recorded [] with complete empty labels and initial final_state has zero executed reward",
            "original_P2_unchanged": True, "posthoc": True, "numerical_bound_tolerance": 1e-12}


def source_paths():
    return [Path(__file__), ROOT / "scripts/analyze_recovery_planning_costs.py", ROOT / "scripts/freeze_recovery_readout.py",
            ROOT / "prereg/POSTHOC_TERMINAL_REWARD_REGRET_V1.md",
            ROOT / "src/dsv3wm/stats.py", ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/provenance.py",
            ROOT / "src/dsv3wm/recovery.py", ROOT / "src/dsv3wm/config.py", ROOT / "src/dsv3wm/__init__.py",
            REPO / "bwm/code/bwmn/rw2.py"]


def command_path(record, flag):
    command = record.get("command")
    if not isinstance(command, list) or any(type(value) is not str for value in command) or command.count(flag) != 1:
        raise ValueError("Unambiguous original planning command required")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"): raise ValueError("Missing command path")
    return (REPO / command[index + 1]).resolve()


def parse_arms(specs):
    selected = {}
    for specification in specs:
        if "=" not in specification: raise ValueError("Use --arm NAME=path")
        arm, path = specification.split("=", 1)
        if arm not in prior.ARMS or arm in selected or not path: raise ValueError("Unique known five original arm names required")
        selected[arm] = Path(path).resolve()
    if set(selected) != set(prior.ARMS): raise ValueError("Exactly B0/A0/A1/A2/A5 required")
    return selected


def load_record(path, tasks_path, tasks, arm):
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import RecoveryConfig, INITIAL_F, INITIAL_F_SHA
    from freeze_recovery_readout import validate_run
    path, tasks_path = Path(path).resolve(), Path(tasks_path).resolve()
    record = prior.read_json(path.read_text(encoding="utf-8"))
    prior.validate_record(record, tasks, tasks_sha256=sha256(tasks_path), arm=arm)
    config, adapter, task_input, scoring, freeze, output = [command_path(record, flag) for flag in
             ("--config", "--adapter", "--tasks", "--scoring", "--freeze", "--out")]
    program = (REPO / record["command"][0]).resolve()
    if program != (ROOT / "scripts/evaluate_recovery_planning.py").resolve() or output != path or task_input != tasks_path:
        raise ValueError("Original P2 execution namespace/result/task path mismatch")
    if prior.canonical(RecoveryConfig.read(config).dict()) != prior.canonical(record["config"]):
        raise ValueError("Actual expanded planning configuration differs")
    bound = [config, tasks_path, scoring, adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
    for actual, field in zip(bound, ("config_sha256", "tasks_sha256", "scoring_sha256", "adapter_sha256", "adapter_config_sha256")):
        if sha256(actual) != record[field]: raise ValueError("Original planning input/artifact digest drift: " + field)
    if verify_freeze(freeze) != record["freeze_sha256"]: raise ValueError("Original planning freeze changed")
    traces = path.with_suffix(".traces.json")
    if sha256(traces) != record["traces_sha256"]: raise ValueError("Preserved planning trace digest changed")
    validate_trace_identities(prior.read_json(traces.read_text(encoding="utf-8")), record, tasks)
    for name, expected in record["source_sha256"].items():
        actual = (REPO / name).resolve()
        if sha256(actual) != expected: raise ValueError("Executed planning source changed")
        bound.append(actual)
    if program not in bound or (REPO / "bwm/code/bwmn/rw2.py").resolve() not in bound:
        raise ValueError("Original evaluation and source reward membership required")
    if sha256(REPO / "bwm/code/bwmn/rw2.py") != RW2_SHA: raise ValueError("Registered reward source changed")
    if arm == "B0":
        if adapter != INITIAL_F.resolve() or sha256(adapter / "adapter_model.safetensors") != INITIAL_F_SHA:
            raise ValueError("Original registered F B0 artifact required")
    else:
        actual_arm, history = validate_run(adapter.parent, tasks_path.with_name("train.jsonl"))
        if actual_arm != arm: raise ValueError("Actual trained arm differs")
        bound += history
    verify_bound_inputs(freeze, bound)
    labels = [row for row in prior.read_json(scoring.read_text(encoding="utf-8")) if row.get("split") == "dev"]
    indexed = {row["task_id"]: row for row in labels}
    if len(indexed) != 12 or len(labels) != 12 or set(indexed) != {task["id"] for task in tasks}:
        raise ValueError("Frozen twelve DEV scoring-metadata identities required")
    for row in record["per_parent"]:
        label = indexed[row["task_id"]]
        if any(label.get(key) != row[key] for key in ("parent_world_id", "cohort", "optimal_moves")):
            raise ValueError("Published task/parent/cohort/distance metadata differs")
        verification = row.get("verification")
        if not isinstance(verification, dict): raise ValueError("Every original chosen plan needs complete recorded verification")
        hashes = verification.get("verifier_sha256")
        if not isinstance(hashes, dict) or hashes.get("bwm/code/bwmn/rw2.py") != RW2_SHA:
            raise ValueError("Recorded independent verifier source identity required")
        for name, expected in hashes.items():
            actual = (REPO / name).resolve()
            if not prior.is_sha(expected) or sha256(actual) != expected: raise ValueError("Recorded verifier source drift")
            bound.append(actual)
    return record, sorted(set([path, traces, freeze, *bound]))


def validate_trace_identities(traces, record, tasks):
    """Check recorded learned ancestry/counters only; never compute a successor."""
    from dsv3wm.recovery import legacy_parse
    lookup = {task["id"]: task for task in tasks}
    if not isinstance(traces, list) or len(traces) != sum(row["model_queries"] for row in record["per_parent"]):
        raise ValueError("Complete recorded learned-query trace panel required")
    nodes, counts = {}, {task["id"]: [0, 0, 0] for task in tasks}
    for trace in traces:
        task = lookup.get(trace.get("task_id"))
        if task is None or trace.get("parent_world_id") != task["parent_world_id"]: raise ValueError("Trace task/parent identity differs")
        parent, action, depth = trace.get("parent_plan"), trace.get("action"), trace.get("depth")
        if (not isinstance(parent, list) or any(type(move) is not str or move not in ACTIONS for move in parent)
                or type(action) is not str or action not in ACTIONS or type(depth) is not int or depth != len(parent) + 1
                or not 1 <= depth <= 6): raise ValueError("Recorded learned trace action/depth prefix differs")
        key = task["id"], tuple([*parent, action])
        if key in nodes: raise ValueError("Duplicate learned-query prefix")
        raw = trace["raw"]; inputs = raw["conditioning"]
        if (set(inputs) != {"domain", "rules", "s0", "actions"} or inputs["domain"] != "register"
                or prior.canonical(inputs["rules"]) != prior.canonical(task["rules"]) or inputs["actions"] != [action]):
            raise ValueError("Trace causal conditioning differs")
        expected_state = task["s0"] if not parent else nodes.get((task["id"], tuple(parent)), {}).get("state")
        if expected_state is None or inputs["s0"] != expected_state: raise ValueError("Trace own-predicted ancestry differs")
        parsed = legacy_parse(raw["generated"].get("text"))
        if prior.canonical(parsed) != prior.canonical(raw["parsed"]): raise ValueError("Raw learned reply/parse changed")
        if (type(trace.get("prompt_tokens")) is not int or not 1 <= trace["prompt_tokens"] <= 512
                or type(trace.get("output_tokens")) is not int or not 0 <= trace["output_tokens"] <= 32
                or type(raw["generated"].get("n_tokens")) is not int or raw["generated"]["n_tokens"] != trace["output_tokens"]
                or type(raw["generated"].get("hit_eos")) is not bool): raise ValueError("Trace token counters differ")
        nodes[key] = parsed
        counts[task["id"]][0] += 1; counts[task["id"]][1] += trace["prompt_tokens"]; counts[task["id"]][2] += trace["output_tokens"]
    for row in record["per_parent"]:
        if counts[row["task_id"]] != [row["model_queries"], row["input_tokens"], row["output_tokens"]]:
            raise ValueError("Trace/per-parent query work differs")
        choice = row["choice"]
        if isinstance(choice, list) and choice:
            prediction = nodes.get((row["task_id"], tuple(choice)))
            if prediction is None or prediction.get("state") != row["predicted_state"]:
                raise ValueError("Recorded chosen plan/prediction absent from learned traces")


def score_case(task, row):
    """Algebra over recorded execution flags/states; no transition recomputation."""
    plan, verification = row.get("choice"), row.get("verification")
    if not isinstance(plan, list) or len(plan) > 6 or any(type(action) is not str or action not in ACTIONS for action in plan):
        raise ValueError("Complete recorded action list required; null/malformed is not an empty plan")
    if not isinstance(verification, dict) or not isinstance(verification.get("labels"), dict):
        raise ValueError("Complete recorded verification required")
    labels = verification["labels"]
    for field in ("states", "legal", "terminated", "executed"):
        if not isinstance(labels.get(field), list) or len(labels[field]) != len(plan):
            raise ValueError("Recorded labels must have exact chosen-plan cardinality")
    if any(type(value) is not bool for field in ("legal", "terminated", "executed") for value in labels[field]):
        raise ValueError("Strict boolean execution/legality/termination flags required")
    optimum = row.get("optimal_moves")
    if (type(optimum) is not int or not 3 <= optimum <= 5
            or type(row.get("independently_verified_optimal_moves")) is not int or row["independently_verified_optimal_moves"] != optimum):
        raise ValueError("Published3..5 optimal moves must fit cap6 including finish")
    if any(type(row.get(field)) is not bool for field in ("success", "all_legal")):
        raise ValueError("Original P2 outcome flags required")
    initial, goal, limit = task["s0"]["x"], task["goal"]["state"]["x"], task["rules"]["limit"]
    previous, done, executed, illegal = initial, False, 0, 0
    for action, state, legal, terminated, attempted in zip(plan, labels["states"], labels["legal"], labels["terminated"], labels["executed"]):
        if not isinstance(state, dict) or set(state) != {"x"} or type(state["x"]) is not int or abs(state["x"]) > limit:
            raise ValueError("Strict bounded recorded register state required")
        after = state["x"]
        if done:
            if attempted or legal or not terminated or after != previous:
                raise ValueError("Recorded post-finish labels must be unexecuted and absorbing")
        else:
            if not attempted: raise ValueError("Unexecuted step before finish is incomplete verification")
            executed += 1; illegal += not legal
            if action == "finish":
                if not legal or not terminated or after != previous: raise ValueError("Finish must be legal, terminating and state-preserving")
                done = True
            elif terminated or (not legal and after != previous):
                raise ValueError("Nonfinish cannot terminate; illegal executed moves must be no-ops")
        previous = after
    final = verification.get("final_state")
    if not isinstance(final, dict) or set(final) != {"x"} or type(final["x"]) is not int or final != {"x": previous}:
        raise ValueError("Recorded final state differs from complete labels/initial empty-plan state")
    original_all_legal = all(labels["legal"]) and all(labels["executed"])
    original_success = original_all_legal and previous == goal
    if (type(verification.get("all_legal")) is not bool or type(verification.get("success")) is not bool
            or type(verification.get("verified")) is not bool or verification["verified"] != original_success
            or verification["all_legal"] != original_all_legal or verification["success"] != original_success
            or row["all_legal"] != original_all_legal or row["success"] != original_success):
        raise ValueError("Recorded goal-only P2 flags disagree with their preserved verification")
    terminal_success = done and previous == goal
    reward = float(terminal_success) - .01 * executed - .1 * illegal
    optimal_reward = 1 - .01 * (optimum + 1)
    regret = optimal_reward - reward
    normalized = regret / (optimal_reward + .66)
    if reward < -.66 - 1e-12 or reward > optimal_reward + 1e-12 or not -1e-12 <= normalized <= 1 + 1e-12:
        raise ValueError("Recorded return/optimality violates source cap6 reward bounds")
    return {"task_id": row["task_id"], "parent_world_id": row["parent_world_id"], "cohort": row["cohort"],
            "recorded_choice": copy.deepcopy(plan), "original_P2_goal_only_success": row["success"],
            "original_P2_all_legal": row["all_legal"], "original_success_conditional_move_cost": row.get("actual_move_cost"),
            "original_success_conditional_excess_cost": row.get("normalized_excess_cost"),
            "terminal_goal_success": terminal_success, "terminated": done, "final_x": previous,
            "executed_steps": executed, "executed_illegal_steps": illegal, "ignored_after_finish": len(plan) - executed,
            "optimal_moves": optimum, "optimal_terminal_return": optimal_reward, "source_terminal_return": reward,
            "source_terminal_return_regret": regret, "normalized_terminal_regret": normalized}


def analyze(records, tasks, *, tasks_sha256, repetitions=2000, random_seed=20261012):
    if set(records) != set(prior.ARMS): raise ValueError("Exactly the five original completed planning arms required")
    if type(repetitions) is not int or repetitions < 100 or type(random_seed) is not int: raise ValueError("Deterministic >=100 bootstrap repetitions required")
    tasks_by_id = {task["id"]: task for task in tasks}
    panels, summaries = {}, {}
    for arm in prior.ARMS:
        original = prior.validate_record(records[arm], tasks, tasks_sha256=tasks_sha256, arm=arm)
        panels[arm] = [score_case(tasks_by_id[row["task_id"]], row) for row in original]
        summaries[arm] = {}
        for cohort in (*prior.COHORTS, "all"):
            selected = [row for row in panels[arm] if cohort == "all" or row["cohort"] == cohort]
            summaries[arm][cohort] = {"n": len(selected), "evaluable": len(selected), "unevaluable": 0,
                "original_P2_successes": sum(row["original_P2_goal_only_success"] for row in selected),
                "terminal_goal_successes": sum(row["terminal_goal_success"] for row in selected),
                "executed_steps": sum(row["executed_steps"] for row in selected),
                **{"mean_" + field: sum(row[field] for row in selected) / len(selected) for field in
                   ("source_terminal_return", "source_terminal_return_regret", "normalized_terminal_regret")}}
    paired = {}
    for baseline, candidate in prior.COMPARISONS:
        for field in ("tasks_sha256", "scoring_sha256", "budget", "instrument", "source_sha256"):
            if prior.canonical(records[baseline].get(field)) != prior.canonical(records[candidate].get(field)):
                raise ValueError("Paired original planning instrument/provenance mismatch")
        old, new = [{row["parent_world_id"]: row for row in panels[arm]} for arm in (baseline, candidate)]
        if set(old) != set(new) or any(any(old[parent][field] != new[parent][field] for field in ("task_id", "cohort", "optimal_moves")) for parent in old):
            raise ValueError("Complete matched parent/task/cohort/distance panel required")
        contrast = {}
        for cohort in (*prior.COHORTS, "all"):
            selected = sorted(parent for parent in old if cohort == "all" or old[parent]["cohort"] == cohort)
            estimates = {}
            for field in ("source_terminal_return", "source_terminal_return_regret", "normalized_terminal_regret"):
                cells = [{"parent_world_id": parent, "seed": 0, "baseline": old[parent][field], "candidate": new[parent][field]} for parent in selected]
                result = clustered_bootstrap(cells, repetitions=repetitions, random_seed=random_seed)
                result.update(unit="all matched fixed DEV parents, one seed; descriptive parent bootstrap",
                              training_seed_uncertainty_estimated=False, limitations="posthoc; no equivalence, promotion or confirmatory family")
                estimates[field] = result
            contrast[cohort] = {"n": len(selected), "all_case_parent_ids": selected, "candidate_minus_baseline": estimates}
        paired[baseline + "_to_" + candidate] = contrast
    return {"schema": SCHEMA, "status": "offline posthoc terminal-contract audit", "contract": CONTRACT,
            "per_arm": panels, "summaries": summaries, "paired_all_cases": paired,
            "confirmation_access": False, "oracle_called": False, "model_loaded": False, "original_P2_unchanged": True,
            "scope": "derived source terminal reward, not original goal-only P2 or stricter H3 whole-plan validity",
            "limitations": ["Conservative universal bound is not an attainable worst return for every parent.",
                            "One seed and12 fixed DEV parents; bootstrap does not estimate training-seed uncertainty or prove equivalence.",
                            "Reward is reconstructed from recorded execution labels only; no finish appended and no simulator replay.",
                            "Null/malformed/incomplete records are refused rather than imputed as abstentions or penalties."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--arm", required=True, action="append", help="NAME=original result path; all five required")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError("Terminal audit exists; use a fresh namespace")
    if args.tasks.resolve() != (ROOT / "datasets/recovery-v1/planning_dev.jsonl").resolve() or prior.file_hash(args.tasks) != TASKS_SHA:
        raise ValueError("Only fixed original public planning DEV tasks are admitted")
    selected = parse_arms(args.arm)
    tasks = [prior.read_json(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    records, bound = {}, [args.tasks]
    for arm, path in selected.items():
        records[arm], files = load_record(path, args.tasks, tasks, arm); bound += files
    output = analyze(records, tasks, tasks_sha256=prior.file_hash(args.tasks))
    output.update(command=sys.argv, input_sha256={str(path.resolve()): prior.file_hash(path) for path in sorted(set(bound))},
                  source_sha256={str(path.resolve()): prior.file_hash(path) for path in source_paths()},
                  execution_provenance={arm: {key: records[arm][key] for key in ("config_sha256", "tasks_sha256", "scoring_sha256",
                        "adapter_sha256", "adapter_config_sha256", "freeze_sha256", "traces_sha256", "source_sha256")} for arm in prior.ARMS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream: json.dump(output, stream, indent=2, allow_nan=False); stream.write("\n")
    print(prior.canonical({"out": str(args.out), "status": output["status"], "scope": output["scope"]}))


if __name__ == "__main__":
    main()
