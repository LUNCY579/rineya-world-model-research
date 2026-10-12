"""Offline, single-seed planning cost/regret descriptions on fixed recovery DEV.

No model, exact transition, solver or sealed labels are loaded. Recorded outcome
flags are consumed as measurements; this analysis never rescores a plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from dsv3wm.stats import clustered_bootstrap

ARMS = ("B0", "A0", "A1", "A2", "A5")
INITIAL_F_SHA = "c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466"
COHORTS = ("recovery_id_teacher_known", "recovery_ood_limit")
BUDGET = {"max_depth": 6, "max_model_calls": 30, "beam_width": 1, "max_processed_positions": 120000}
COST_FIELDS = ("query_attempts", "model_queries", "input_tokens", "output_tokens", "processed_positions_upper_bound")
PAIR_METRICS = ("actual_move_cost", "normalized_excess_cost", *COST_FIELDS)
COUNTERS = (*COST_FIELDS, "unpadded_processed_positions_upper_bound", "malformed_predictions", "illegal_predictions",
            "overlength_refusals", "budget_refusals")
COMPARISONS = (("B0", "A0"), ("B0", "A1"), ("B0", "A2"), ("B0", "A5"),
               ("A0", "A1"), ("A0", "A2"), ("A0", "A5"), ("A5", "A1"), ("A5", "A2"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def validate_tasks(tasks):
    identities = {}
    for task in tasks:
        if (task.get("split") != "dev" or task.get("domain") != "register"
                or set(task) != {"id", "split", "domain", "rules", "s0", "goal", "parent_world_id", "provenance"}):
            raise ValueError("Bare fixed RegisterWorld DEV task inputs required")
        state, rules, goal = task["s0"], task["rules"], task["goal"]
        limit = rules.get("limit")
        if type(limit) is not int or limit <= 0 or rules.get("semantics") != "bwm-register-v2":
            raise ValueError("Invalid task rule specification")
        if set(goal) != {"kind", "state"} or goal["kind"] != "state":
            raise ValueError("Explicit goal state required")
        for value in (state, goal["state"]):
            if set(value) != {"x"} or type(value["x"]) is not int or abs(value["x"]) > limit:
                raise ValueError("Invalid task initial/goal state")
        operative = {key: rules[key] for key in ("semantics", "limit", "objects") if key in rules}
        parent = object_hash({"domain": "register", "s0": state, "actions": [], "rules": operative})
        identity = object_hash({key: task[key] for key in ("domain", "rules", "s0", "goal")})
        if task["parent_world_id"] != parent or task["id"] != identity or parent in identities:
            raise ValueError("Task ID/canonical parent identity is invalid or duplicated")
        cohort = task["provenance"].get("source_trajectory_id")
        if not isinstance(cohort, str) or not cohort:
            raise ValueError("Task training/development source provenance required")
        identities[parent] = task["id"]
    if len(tasks) != 12:
        raise ValueError("Recovery planning analysis binds the full twelve-parent DEV task bank")
    return identities


def validate_record(record, tasks, *, tasks_sha256, arm):
    identities = validate_tasks(tasks)
    if (arm not in ARMS or record.get("status") != "development-evaluated"
            or record.get("instrument") != "legacy learned batches/W1/depth6/query30/dev-only"
            or record.get("oracle_feedback") is not False or record.get("confirmation_access") is not False
            or record.get("all_choices_fixed_before_scoring") is not True
            or canonical(record.get("budget")) != canonical(BUDGET)):
        raise ValueError("Completed learned-only recovery planning record and fixed budget required")
    cfg = record["config"]
    if (type(cfg.get("seed")) is not int or cfg["seed"] != 0 or cfg.get("arm") != ("A0" if arm == "B0" else arm)
            or cfg.get("device") != "cpu" or type(cfg.get("max_length")) is not int or cfg["max_length"] != 512
            or cfg.get("core") != "titan-sparse" or type(cfg.get("rank")) is not int or cfg["rank"] != 16
            or type(cfg.get("generation_tokens")) is not int or cfg["generation_tokens"] != 32):
        raise ValueError("Single seed 0 / CPU / 512-input / 32-output planning record required")
    if record.get("tasks_sha256") != tasks_sha256:
        raise ValueError("Actual task-file hash differs from recorded planning inputs")
    for key in ("config_sha256", "tasks_sha256", "scoring_sha256", "adapter_sha256", "adapter_config_sha256", "freeze_sha256", "traces_sha256"):
        if not is_sha(record.get(key)):
            raise ValueError(f"Missing or invalid provenance hash: {key}")
    if arm == "B0" and record["adapter_sha256"] != INITIAL_F_SHA:
        raise ValueError("B0 must retain the recorded registered-F adapter identity")
    sources = record.get("source_sha256")
    if not isinstance(sources, dict) or not sources or any(not is_sha(value) for value in sources.values()):
        raise ValueError("Execution source hash provenance required")
    rows, seen = record.get("per_parent"), set()
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise ValueError("Complete one-record-per-task parent panel required")
    for row in rows:
        parent = row.get("parent_world_id")
        if parent in seen or parent not in identities or row.get("task_id") != identities[parent]:
            raise ValueError("Planning result/task-parent identity mismatch")
        seen.add(parent)
        if row.get("cohort") not in COHORTS:
            raise ValueError("Unknown fixed DEV cohort")
        task = next(task for task in tasks if task["parent_world_id"] == parent)
        expected_cohort = "recovery_id_teacher_known" if task["rules"]["limit"] in (32, 48, 64) else "recovery_ood_limit"
        if row["cohort"] != expected_cohort:
            raise ValueError("Result cohort disagrees with fixed task rule range")
        for key in COUNTERS:
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Nonnegative integer planning metric required: {key}")
        if (row["query_attempts"] > 30 or row["model_queries"] > row["query_attempts"]
                or row["query_attempts"] != row["model_queries"] + row["overlength_refusals"] + row["budget_refusals"]
                or row["malformed_predictions"] + row["illegal_predictions"] > row["model_queries"]
                or row["input_tokens"] > row["model_queries"] * 512 or row["output_tokens"] > row["model_queries"] * 32
                or row["unpadded_processed_positions_upper_bound"] != row["input_tokens"] + row["output_tokens"]
                or not row["unpadded_processed_positions_upper_bound"] <= row["processed_positions_upper_bound"] <= 16320):
            raise ValueError("Planning cost/query accounting or frozen cap is inconsistent")
        for key in ("success", "all_legal", "optimal_success", "first_action_optimal", "first_action_present"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"Boolean recorded outcome required: {key}")
        optimum = row.get("optimal_moves")
        if (type(optimum) is not int or not 3 <= optimum <= 5
                or type(row.get("independently_verified_optimal_moves")) is not int
                or row["independently_verified_optimal_moves"] != optimum):
            raise ValueError("Recorded post-choice optimal move metadata is inconsistent")
        choice = row.get("choice")
        if choice is not None and (not isinstance(choice, list) or len(choice) > 6
                or any(action not in ("add1", "sub1", "double", "halve", "finish") for action in choice)):
            raise ValueError("Invalid recorded choice syntax/length")
        cost, regret = row.get("actual_move_cost"), row.get("normalized_excess_cost")
        if row["success"]:
            if (choice is None or not row["all_legal"] or type(cost) is not int or not optimum <= cost <= 6
                    or cost != sum(action != "finish" for action in choice)
                    or isinstance(regret, bool) or not isinstance(regret, (int, float)) or not math.isfinite(regret)
                    or not math.isclose(regret, (cost - optimum) / optimum, rel_tol=1e-12, abs_tol=1e-12)
                    or row["optimal_success"] != (cost == optimum)):
                raise ValueError("Successful plan cost/conditional normalized excess cost is inconsistent")
        elif cost is not None or regret is not None or row["optimal_success"]:
            raise ValueError("Failures must retain null move cost/regret; zero is not a failure substitute")
    if seen != set(identities) or any(sum(row["cohort"] == cohort for row in rows) != 6 for cohort in COHORTS):
        raise ValueError("Full paired six-ID/six-OOD parent panel required")
    reported = record.get("summary")
    if not isinstance(reported, dict):
        raise ValueError("Recorded aggregate summary required")
    for cohort in (*COHORTS, "all"):
        selected = [row for row in rows if cohort == "all" or row["cohort"] == cohort]
        expected = {"n": len(selected), "successes": sum(row["success"] for row in selected),
                    "queries": sum(row["model_queries"] for row in selected),
                    "query_attempts": sum(row["query_attempts"] for row in selected),
                    "input_tokens": sum(row["input_tokens"] for row in selected), "output_tokens": sum(row["output_tokens"] for row in selected),
                    "processed_positions_upper_bound": sum(row["processed_positions_upper_bound"] for row in selected)}
        if cohort not in reported or any(type(reported[cohort].get(key)) is not int or reported[cohort][key] != value
                                         for key, value in expected.items()):
            raise ValueError("Recorded aggregate planning summary disagrees with per-parent costs/outcomes")
    return rows


def summarize(rows):
    solved = [row for row in rows if row["success"]]
    totals = {field: sum(row[field] for row in rows) for field in COST_FIELDS}
    return {"n": len(rows), "solved": len(solved), "success_rate": len(solved) / len(rows) if rows else None,
            "totals_all_searches": totals,
            "model_queries_per_solved_including_failures": totals["model_queries"] / len(solved) if solved else None,
            "model_queries_per_solved_definition": "all completed searches' model queries / solved problems; failed-search cost included",
            "success_conditional": {"n": len(solved), "mean_actual_move_cost": sum(row["actual_move_cost"] for row in solved) / len(solved) if solved else None,
                                    "mean_normalized_excess_cost": sum(row["normalized_excess_cost"] for row in solved) / len(solved) if solved else None,
                                    "cost_regret_failure_policy": "null; failures excluded from these conditional means"}}


def descriptive_bootstrap(rows, *, repetitions, random_seed):
    if not rows:
        return None
    result = clustered_bootstrap(rows, repetitions=repetitions, random_seed=random_seed)
    result.update(unit="descriptive equal-weight common-success parents; singleton seed fixed",
                  seed_uncertainty_estimated=False,
                  limitations="One training seed, fixed selected DEV bank, conditioned on success in both arms. No training-seed uncertainty or equivalence inference.")
    return result


def compare_common_success(baseline, candidate, *, repetitions=2000, random_seed=20261012):
    for key in ("tasks_sha256", "scoring_sha256", "budget", "instrument", "source_sha256"):
        if canonical(baseline.get(key)) != canonical(candidate.get(key)):
            raise ValueError(f"Paired planning instrument/input mismatch: {key}")
    old = {row["parent_world_id"]: row for row in baseline["per_parent"]}
    new = {row["parent_world_id"]: row for row in candidate["per_parent"]}
    if set(old) != set(new) or len(old) != len(baseline["per_parent"]) or len(new) != len(candidate["per_parent"]):
        raise ValueError("Matched complete unique parent sets required")
    if any(old[parent]["task_id"] != new[parent]["task_id"] or old[parent]["cohort"] != new[parent]["cohort"]
           or old[parent]["optimal_moves"] != new[parent]["optimal_moves"] for parent in old):
        raise ValueError("Paired task/cohort/optimality identity mismatch")
    output = {}
    for cohort in (*COHORTS, "all"):
        selected = sorted(parent for parent in old if old[parent]["success"] and new[parent]["success"]
                          and (cohort == "all" or old[parent]["cohort"] == cohort))
        metrics = {}
        for field in PAIR_METRICS:
            pairs = [{"parent_world_id": parent, "seed": 0, "baseline": old[parent][field], "candidate": new[parent][field]}
                     for parent in selected]
            metrics[field] = descriptive_bootstrap(pairs, repetitions=repetitions, random_seed=random_seed)
        output[cohort] = {"common_success_parents": selected, "n": len(selected), "candidate_minus_baseline": metrics,
                          "selection_scope": "same parents solved by both arms; failures never imputed or included in paired regret/cost differences"}
    return output


def analyze(records, tasks, *, tasks_sha256, repetitions=2000, random_seed=20261012):
    if set(records) != set(ARMS):
        raise ValueError("All completed B0/A0/A1/A2/A5 planning arms required")
    if type(repetitions) is not int or repetitions < 100 or type(random_seed) is not int:
        raise ValueError("At least 100 deterministic bootstrap repetitions required")
    for arm in ARMS:
        validate_record(records[arm], tasks, tasks_sha256=tasks_sha256, arm=arm)
    summaries = {}
    for arm in ARMS:
        rows = records[arm]["per_parent"]
        summaries[arm] = {cohort: summarize([row for row in rows if cohort == "all" or row["cohort"] == cohort])
                          for cohort in (*COHORTS, "all")}
        summaries[arm]["recorded_batch_position_bound"] = records[arm].get("legacy_batch_processed_positions_upper_bound")
        summaries[arm]["batch_position_scope"] = "separate recorded batch estimator; per-parent padding upper bounds above are the paired cost measure"
    comparisons = {f"{baseline}_to_{candidate}": compare_common_success(records[baseline], records[candidate],
                    repetitions=repetitions, random_seed=random_seed) for baseline, candidate in COMPARISONS}
    return {"schema": "dsv3wm-recovery-planning-costs-v1", "status": "offline descriptive analysis",
            "summaries": summaries, "paired_common_success": comparisons,
            "bootstrap": {"repetitions": repetitions, "random_seed": random_seed, "training_seeds": [0], "training_seed_uncertainty_estimated": False},
            "definitions": {"regret": "normalized_excess_cost=(actual legal move count-optimal moves)/optimal moves, conditional on success",
                            "failure_regret": "null; no zero, penalty or oracle rescore is substituted",
                            "processed_positions": "recorded conservative cached-decoder padding upper bounds, not measured FLOPs"},
            "limitations": ["One seed and twelve fixed DEV parents; descriptive post-hoc analysis only.",
                            "Success-conditional arm means have different selected sets; only matched common-success differences are compared.",
                            "Parent-bootstrap intervals do not estimate training-seed uncertainty or prove equivalence, including degenerate zero intervals.",
                            "Model queries are transition predictions; batching and padding differ from raw decoder forward counts.",
                            "No model, simulator, solver, sealed labels or changed evaluation records are used."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/recovery-v1")
    parser.add_argument("--tasks", type=Path, default=ROOT / "datasets/recovery-v1/planning_dev.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Planning cost analysis already exists")
    if args.tasks.name != "planning_dev.jsonl":
        raise ValueError("Only the known public recovery planning DEV task filename is accepted")
    tasks = [read_json(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = {arm: args.results / (arm + "_planning.json") for arm in ARMS}
    records = {arm: read_json(path.read_text(encoding="utf-8")) for arm, path in paths.items()}
    output = analyze(records, tasks, tasks_sha256=file_hash(args.tasks), repetitions=args.repetitions, random_seed=args.bootstrap_seed)
    output.update(command=sys.argv, inputs_sha256={str(path.resolve()): file_hash(path) for path in [args.tasks, *paths.values()]},
                  source_sha256={str(Path(__file__).resolve()): file_hash(__file__),
                                 str((ROOT / "src/dsv3wm/stats.py").resolve()): file_hash(ROOT / "src/dsv3wm/stats.py"),
                                 str((ROOT / "src/dsv3wm/__init__.py").resolve()): file_hash(ROOT / "src/dsv3wm/__init__.py")},
                  execution_provenance={arm: {key: record[key] for key in ("config_sha256", "tasks_sha256", "scoring_sha256", "adapter_sha256",
                                               "adapter_config_sha256", "freeze_sha256", "traces_sha256", "source_sha256")}
                                        for arm, record in records.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
