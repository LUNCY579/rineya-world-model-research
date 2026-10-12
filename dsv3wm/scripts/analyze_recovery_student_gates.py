"""Offline fixed H3 continuation gates; never generates or executes transitions.

Completed proposer/guard records and their actual run metadata are required.
Operational thresholds and descriptive uncertainty do not establish efficacy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dsv3wm.recovery_student import StudentConfig, parse_registered_plan, registered_prompt, shared_orders
from dsv3wm.domains import sha256 as object_hash
from dsv3wm.recovery import RecoveryConfig, INITIAL_F_SHA
from dsv3wm.native import EXPECTED_CORE_SHA
from dsv3wm.stats import paired_mcnemar, wilson_interval
from dsv3wm.recovery_planning import validate_tasks
from analyze_recovery_student_costs import reduce_record
from evaluate_recovery_student_guards import DEV_SHA, FIXTURE_SHA, SCHEMA as GUARD_SCHEMA, paired_guards, validate_rows

SCHEMA = "dsv3wm-h3-offline-operational-gates-v1"
CONTROL_FIELDS = {"plan_weight", "control_extra_one_step"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("missing/invalid SHA256 identity: " + name)
    return value


def kind(cfg):
    return ("C3" if cfg.arm == "A3" else "C4") if cfg.control_extra_one_step else cfg.arm


def identity(cfg):
    """Only registered control-specific training fields may differ from warm config."""
    return {key: value for key, value in cfg.dict().items() if key not in CONTROL_FIELDS}


def validate_proposer(record, tasks):
    if record.get("schema") is not None:
        raise ValueError("unknown proposer schema; registered legacy readout has no schema field")
    cfg = StudentConfig(**record["config"])
    if cfg.seed != 0:
        raise ValueError("registered one-seed recovery pilot requires seed0")
    # Reuse the existing raw-candidate/selection/budget/aggregate validator.
    costs = reduce_record(record)
    if costs["learned_reranked_readout"] is None:
        raise ValueError("matched fixed-B0 hybrid readout required; missing results cannot pass a gate")
    indexed = {row["task_id"]: row for row in record["outcomes"]}
    if set(indexed) != {task["id"] for task in tasks}:
        raise ValueError("proposer task allowlist differs from the fixed twelve DEV tasks")
    for task in tasks:
        row = indexed[task["id"]]
        if (row["parent_world_id"] != task["parent_world_id"] or row.get("prompt") != registered_prompt(task)
                or row.get("selected_index") != 0):
            raise ValueError("fixed parent/prompt/greedy choice identity differs")
        valid = 0
        for index, candidate in enumerate(row["candidates"]):
            expected_seed = None if index == 0 else int(hashlib.sha256(
                f"{cfg.seed}|{task['parent_world_id']}|0|{index}".encode()).hexdigest()[:15], 16)
            if candidate.get("seed") != expected_seed or type(candidate.get("seed")) is bool:
                raise ValueError("registered greedy/CRN candidate seed differs")
            parsed = parse_registered_plan(candidate["raw"].get("text"))
            if (type(candidate.get("malformed")) is not bool or candidate["malformed"] != parsed["malformed"]
                    or candidate.get("plan") != parsed["plan"]):
                raise ValueError("raw registered plan syntax differs from preserved parse")
            valid += not parsed["malformed"]
            score = row["scores"][index]
            if (type(score.get("all_legal")) is not bool or (score["success"] and not score["all_legal"])
                    or (parsed["malformed"] and (score["success"] or score["all_legal"]))):
                raise ValueError("malformed proposals cannot be exact-positive scored plans")
            verification = score.get("verification")
            if not parsed["malformed"]:
                if (not isinstance(verification, dict) or type(verification.get("success")) is not bool
                        or type(verification.get("all_legal")) is not bool):
                    raise ValueError("preserved independent full-plan verification required")
                termination = verification.get("labels", {}).get("terminated")
                if (not isinstance(termination, list) or len(termination) != len(parsed["plan"])
                        or any(type(flag) is not bool for flag in termination)
                        or score["all_legal"] != verification["all_legal"]
                        or score["success"] != (verification["success"] and verification["all_legal"] and termination[-1])):
                    raise ValueError("fixed final-finish success differs from preserved verification")
        if row.get("valid_syntax_count") != valid or type(row.get("valid_syntax_count")) is not int:
            raise ValueError("syntax count differs from strict raw parse")
    if (type(record.get("valid_syntax_count")) is not int
            or record["valid_syntax_count"] != sum(row["valid_syntax_count"] for row in indexed.values())):
        raise ValueError("aggregate K4 syntax count differs")
    for key in ("tasks_sha256", "run_sha256", "adapter_sha256", "adapter_config_sha256"):
        digest(record.get(key), key)
    return cfg, indexed, costs


def validate_guard(record, cfg, *, warm=False):
    if (record.get("schema") != GUARD_SCHEMA or record.get("status") != "development-guards-evaluated"
            or record.get("oracle_feedback") is not False or record.get("confirmation_access") is not False
            or record.get("all_generation_fixed_before_scoring") is not True
            or record.get("warm_baseline") is not warm or record.get("retention_pass") is not None
            or record.get("data_sha256") != DEV_SHA or record.get("fixtures_sha256") != FIXTURE_SHA):
        raise ValueError("completed unchanged DEV24/six-fixture guard schema and scoring boundary required")
    actual = StudentConfig(**record["config"])
    if canonical(actual.dict()) != canonical(cfg.dict()):
        raise ValueError("guard full config differs from its actual proposer config")
    outcomes = record["outcomes"]
    parents = outcomes.get("per_parent")
    if not isinstance(parents, list) or len(parents) != 24:
        raise ValueError("complete twenty-four-parent guard readout required")
    indexed = {row["parent_world_id"]: row for row in parents}
    if len(indexed) != 24 or sum(row.get("cohort") == "id" for row in parents) != 12 or sum(row.get("cohort") == "ood" for row in parents) != 12:
        raise ValueError("unique unchanged twelve ID/twelve OOD guard parents required")
    fields = ("rollout_correct", "rollout_malformed", "legality_correct", "termination_correct",
              "direct_correct", "direct_malformed", "direct_legality_correct", "direct_termination_correct")
    for row in parents:
        if set(row.get("horizons", {})) != {"1", "2", "4"} or any(type(row["horizons"][h].get(field)) is not bool
                for h in ("1", "2", "4") for field in fields):
            raise ValueError("typed complete state/flag guard outcomes required")
    for cohort in ("id", "ood", "all"):
        selected = [row for row in parents if cohort == "all" or row["cohort"] == cohort]
        for horizon in ("1", "2", "4"):
            summary = outcomes["summary"][cohort][horizon]
            if (type(summary.get("n")) is not int or summary["n"] != len(selected)
                    or any(type(summary.get(field)) is not int or summary[field] != sum(row["horizons"][horizon][field] for row in selected)
                           for field in fields)):
                raise ValueError("guard aggregate differs from paired parent outcomes")
    behavior = outcomes.get("behavior")
    if (not isinstance(behavior, list) or len(behavior) != 6 or len({row["id"] for row in behavior}) != 6
            or sum(row.get("domain") == "register" for row in behavior) != 3 or sum(row.get("domain") == "ledger" for row in behavior) != 3
            or any(type(row.get("correct")) is not bool for row in behavior)
            or type(outcomes.get("behavior_correct")) is not int or outcomes["behavior_correct"] != sum(row["correct"] for row in behavior)
            or type(outcomes.get("behavior_n")) is not int or outcomes["behavior_n"] != 6 or outcomes.get("retention_pass") is not None):
        raise ValueError("unchanged six-case narrow behavioral guard required")
    for key in ("run_sha256", "adapter_sha256", "adapter_config_sha256"):
        digest(record.get(key), key)
    return indexed


def validate_lineage(proposer, guards, run, *, warm=False):
    cfg = StudentConfig(**proposer["config"])
    for key in ("run_sha256", "adapter_sha256", "adapter_config_sha256"):
        if proposer.get(key) != guards.get(key):
            raise ValueError("proposer and guard do not bind the same actual run/adapter")
    if (run.get("core_sha256") != EXPECTED_CORE_SHA or run.get("optimizer_steps") != 32
            or run.get("adapter_sha256") != proposer["adapter_sha256"]):
        raise ValueError("actual32-step artifact/core/weight identity differs")
    if warm:
        actual = RecoveryConfig(**run["config"])
        if (actual.arm != cfg.warm_arm or actual.seed != cfg.seed or run.get("status") != "pilot-trained"
                or run.get("initial_adapter_sha256") != INITIAL_F_SHA
                or proposer.get("readout_kind") != "warm-parent-proposer-baseline"
                or guards.get("readout_kind") != "warm-parent-proposer-baseline"
                or proposer.get("artifact_training_status") != "pilot-trained" or guards.get("artifact_training_status") != "pilot-trained"
                or proposer.get("artifact_arm") != cfg.warm_arm or guards.get("artifact_arm") != cfg.warm_arm
                or canonical(proposer.get("artifact_training_config")) != canonical(run["config"])):
            raise ValueError("explicit unchanged actual own warm-parent readouts required")
    else:
        actual = StudentConfig(**run["config"])
        expected = "h3-continuation-control-" + kind(cfg) if cfg.control_extra_one_step else "h3-trained-student-proposer"
        if (canonical(actual.dict()) != canonical(cfg.dict()) or run.get("status") != "student-pilot-trained"
                or run.get("training_kind") != kind(cfg) or guards.get("artifact_training_kind") != kind(cfg)
                or proposer.get("readout_kind") != expected or guards.get("readout_kind") != expected
                or proposer.get("artifact_training_status") != "student-pilot-trained" or guards.get("artifact_training_status") != "student-pilot-trained"
                or proposer.get("artifact_arm") != cfg.arm or guards.get("artifact_arm") != cfg.arm
                or canonical(proposer.get("artifact_training_config")) != canonical(run["config"])
                or run.get("adapter_config_sha256") != proposer["adapter_config_sha256"]
                or run.get("warm_arm") != cfg.warm_arm or run.get("corpus_gate", {}).get("passed") is not True):
            raise ValueError("full actual student/control identity differs; controls cannot masquerade as students")
        for key in ("train_sha256", "tasks_sha256", "teacher_manifest_sha256", "verified_corpus_sha256",
                    "warm_run_sha256", "warm_adapter_sha256", "warm_adapter_config_sha256"):
            digest(run.get(key), "actual run " + key)
        for key in ("trajectory_order", "plan_order"):
            if (not isinstance(run.get(key), list) or len(run[key]) != 32
                    or any(type(index) is not int or index < 0 for index in run[key])):
                raise ValueError("actual registered source/plan order of all32 updates required")
    return cfg


def validate_sampling(run, *, training_rows, corpus_rows, corpus_manifest, bound_hashes):
    """Counts come from hash-bound TRAIN/corpus files, never supplied count claims.

    Positive verification is preserved evidence; no transition oracle executes.
    The native trainer already independently reverified these same bound bytes.
    """
    for name, key in (("train", "train_sha256"), ("verified", "verified_corpus_sha256"),
                      ("manifest", "teacher_manifest_sha256"), ("tasks", "tasks_sha256")):
        if digest(bound_hashes.get(name), "sampling " + name) != run.get(key):
            raise ValueError("actual hash-bound sampling input differs: " + name)
    if (corpus_manifest.get("status") != "corpus-qualified" or corpus_manifest.get("gate", {}).get("passed") is not True
            or corpus_manifest.get("train_file_sha256") != bound_hashes["train"]
            or corpus_manifest.get("tasks_file_sha256") != bound_hashes["tasks"]
            or corpus_manifest.get("output_sha256", {}).get("verified.jsonl") != bound_hashes["verified"]):
        raise ValueError("qualified corpus manifest does not bind the actual sampling files")
    if (len(training_rows) != 128 or len(corpus_rows) != 27
            or any(row.get("split") != "train" or row.get("domain") != "register" for row in training_rows)
            or any(row.get("split") != "train" or row.get("domain") != "register" or row.get("verified") is not True for row in corpus_rows)):
        raise ValueError("fixed128 TRAIN rows and qualified27 verified positives required")
    training = {row["parent_world_id"]: row for row in training_rows}
    parents = {row["parent_world_id"] for row in corpus_rows}
    if len(training) != 128 or len(parents) != 27 or not parents <= set(training):
        raise ValueError("unique source/positive TRAIN parent allowlists required")
    for positive in corpus_rows:
        source = training[positive["parent_world_id"]]
        if (any(canonical(positive[key]) != canonical(source[key]) for key in ("domain", "rules", "s0"))
                or positive.get("provenance", {}).get("training_input_sha256") != object_hash(source)
                or corpus_manifest.get("training_row_allowlist", {}).get(positive["parent_world_id"]) != object_hash(source)):
            raise ValueError("positive sampling row differs from its bound source TRAIN parent")
    limits = sorted({row["rules"]["limit"] for row in corpus_rows})
    sequences = len({object_hash(row["plan"]) for row in corpus_rows})
    expected_gate = {"passed": len(parents) >= 16 and len(limits) >= 2 and sequences >= 4,
                     "parents": len(parents), "limits": limits, "sequences": sequences}
    manifest_gate = corpus_manifest["gate"]
    if (canonical(run.get("corpus_gate")) != canonical(expected_gate)
            or manifest_gate.get("accepted_train_parents") != len(parents)
            or manifest_gate.get("covered_limits") != limits or manifest_gate.get("distinct_action_sequences") != sequences):
        raise ValueError("sampling counts disagree with actual qualified corpus rows")
    cfg = StudentConfig(**run["config"])
    trajectories, plans = shared_orders(len(training_rows), len(corpus_rows), cfg.steps, cfg.seed)
    if run.get("trajectory_order") != trajectories or run.get("plan_order") != plans:
        raise ValueError("actual draw orders must equal deterministic shared_orders for bound128/27 pools")
    return {"training_rows": len(training_rows), "positive_rows": len(corpus_rows), "distinct_positive_plans": sequences,
            "limits": limits, "hashes": dict(bound_hashes), "draw_orders": "exact deterministic shared_orders; seed0/32 updates"}


def paired_endpoint(before, after, parents, field):
    b, a = [before[parent][field] for parent in parents], [after[parent][field] for parent in parents]
    result = paired_mcnemar(b, a)
    result.update(gained_parent_ids=[p for p, x, y in zip(parents, b, a) if not x and y],
                  lost_parent_ids=[p for p, x, y in zip(parents, b, a) if x and not y],
                  baseline_wilson95=list(wilson_interval(sum(b), len(b))),
                  candidate_wilson95=list(wilson_interval(sum(a), len(a))),
                  net_count=result["gains"] - result["losses"],
                  inference="descriptive unadjusted comparison; one training seed; not confirmation")
    return result


def analyze(student_proposer, warm_proposer, student_guards, warm_guards, *, student_run, warm_run, tasks, dev_rows,
            training_rows, corpus_rows, corpus_manifest, bound_hashes,
            control_proposer=None, control_guards=None, control_run=None):
    validate_tasks(tasks)
    validate_rows(dev_rows)
    if len(tasks) != 12:
        raise ValueError("fixed twelve DEV planning tasks required")
    student_cfg, student_rows, costs = validate_proposer(student_proposer, tasks)
    warm_cfg, warm_rows, warm_costs = validate_proposer(warm_proposer, tasks)
    student_parents = validate_guard(student_guards, student_cfg)
    warm_parents = validate_guard(warm_guards, warm_cfg, warm=True)
    validate_lineage(student_proposer, student_guards, student_run)
    validate_lineage(warm_proposer, warm_guards, warm_run, warm=True)
    sampling = validate_sampling(student_run, training_rows=training_rows, corpus_rows=corpus_rows,
                                 corpus_manifest=corpus_manifest, bound_hashes=bound_hashes)
    if (canonical(identity(student_cfg)) != canonical(identity(warm_cfg))
            or student_proposer["tasks_sha256"] != warm_proposer["tasks_sha256"]
            or student_run.get("warm_adapter_sha256") != warm_proposer["adapter_sha256"]
            or student_run.get("warm_adapter_config_sha256") != warm_proposer["adapter_config_sha256"]
            or student_run.get("warm_run_sha256") != warm_proposer["run_sha256"]
            or student_run.get("train_sha256") != digest(warm_run.get("data_sha256"), "warm source TRAIN")
            or set(student_parents) != set(warm_parents)):
        raise ValueError("decoding/task/full own warm-weight lineage differs")
    task_parents = [task["parent_world_id"] for task in tasks]
    source_cohorts = {row["parent_world_id"]: "id" if row["provenance"]["cohort"] == "recovery_id_teacher_known" else "ood" for row in dev_rows}
    if set(source_cohorts) != set(warm_parents) or any(source_cohorts[p] != warm_parents[p]["cohort"] for p in source_cohorts):
        raise ValueError("guard parent/cohort mapping differs from the fixed DEV source rows")
    if (not set(task_parents) <= set(warm_parents)
            or sum(warm_parents[parent]["cohort"] == "id" for parent in task_parents) != 6
            or any(student_parents[p]["cohort"] != warm_parents[p]["cohort"] for p in warm_parents)):
        raise ValueError("planning parents must be the fixed six ID/six OOD guard subset")
    before = {row["parent_world_id"]: {"top1": row["top1_success"], "hybrid": row["learned_reranking"]["success"]} for row in warm_rows.values()}
    after = {row["parent_world_id"]: {"top1": row["top1_success"], "hybrid": row["learned_reranking"]["success"]} for row in student_rows.values()}
    pure = paired_endpoint(before, after, task_parents, "top1")
    hybrid = paired_endpoint(before, after, task_parents, "hybrid")
    guards = paired_guards(student_guards, warm_guards, warm_arm=student_cfg.warm_arm, warm_adapter_sha=student_run["warm_adapter_sha256"])
    greedy_syntax = sum(not student_rows[task["id"]]["candidates"][0]["malformed"] for task in tasks)
    criteria = {"greedy_syntax_at_least_9_of_12": greedy_syntax >= 9,
                "paired_top1_net_gain_at_least_2_of_12": pure["net_count"] >= 2,
                "no_lost_matched_hybrid_success": hybrid["losses"] == 0,
                "id_h4_net_loss_at_most_2_of_12": guards["id_h4_net_loss_at_most_two"]}
    quantitative = all(criteria.values())
    behavior_losses = guards["behavior"]["losses"]
    classification = ("hold: operational numerical condition failed" if not quantitative else
                      "review required: retained behavior loss needs explanation" if behavior_losses else
                      "numerical and measured behavior conditions met; runtime review required")
    if student_cfg.control_extra_one_step:
        classification = "control diagnostic only; " + classification
    result = {"schema": SCHEMA, "status": "offline measured-gate reduction; no new native outcome", "artifact_kind": kind(student_cfg),
              "config": student_cfg.dict(), "warm_arm": student_cfg.warm_arm, "n_planning_parents": 12,
              "greedy_syntax": {"correct": greedy_syntax, "n": 12, "wilson95": list(wilson_interval(greedy_syntax, 12))},
              "pure_top1_vs_own_warm": pure, "matched_fixed_B0_hybrid_vs_own_warm": hybrid,
              "paired_dynamics_behavior": guards, "criteria": criteria, "quantitative_conditions_met": quantitative,
              "behavior_review_pending": bool(behavior_losses), "classification": classification,
              "gate_applies_to_distilled_student": not student_cfg.control_extra_one_step,
              "offline_pass4_coverage": {"student": student_proposer["pass_at_4_successes"], "warm": warm_proposer["pass_at_4_successes"],
                  "scope": "offline oracle coverage only; never substitutes for fixed greedy efficacy or deployment selection"},
              "costs": {"student": costs, "warm": warm_costs}, "control_contrast": None,
              "sampling_input_binding": sampling,
              "confirmed": False, "confirmation_access": False, "retention_pass": None,
              "limitations": ["One training seed and12 paired planning parents; Wilson intervals are descriptive and do not establish equivalence.",
                  "Top1 and hybrid McNemar p-values are separate unadjusted descriptive contrasts, not a confirmation claim family.",
                  "Planning12 parents are a subset of dynamics24; these endpoints are dependent.",
                  "Measured query/position caps are validated; runtime feasibility and explanations for behavior losses require review.",
                  "This operational threshold report does not bypass the failed RegisterWorld predictive gain gate or qualify G4/G5/G7."]}
    supplied = (control_proposer is not None, control_guards is not None, control_run is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("optional control needs proposer, guards and actual run together")
    if all(supplied):
        control_cfg, control_rows, control_costs = validate_proposer(control_proposer, tasks)
        control_parents = validate_guard(control_guards, control_cfg)
        validate_lineage(control_proposer, control_guards, control_run)
        if (set(control_parents) != set(student_parents)
                or any(control_parents[p]["cohort"] != source_cohorts[p] for p in control_parents)):
            raise ValueError("control guard must have the exact fixed DEV24 parents/cohorts")
        control_behavior = {row["id"]: row["domain"] for row in control_guards["outcomes"]["behavior"]}
        expected_behavior = {row["id"]: row["domain"] for row in warm_guards["outcomes"]["behavior"]}
        student_behavior = {row["id"]: row["domain"] for row in student_guards["outcomes"]["behavior"]}
        if control_behavior != expected_behavior or control_behavior != student_behavior:
            raise ValueError("control behavior must have the exact warm/student six-fixture IDs/domains")
        validate_sampling(control_run, training_rows=training_rows, corpus_rows=corpus_rows,
                          corpus_manifest=corpus_manifest, bound_hashes=bound_hashes)
        if (student_cfg.control_extra_one_step or not control_cfg.control_extra_one_step
                or canonical(identity(control_cfg)) != canonical(identity(student_cfg))
                or control_proposer["tasks_sha256"] != student_proposer["tasks_sha256"]
                or any(control_run.get(key) != student_run.get(key) for key in
                       ("warm_adapter_sha256", "warm_adapter_config_sha256", "warm_run_sha256", "train_sha256", "tasks_sha256",
                        "teacher_manifest_sha256", "verified_corpus_sha256", "trajectory_order", "plan_order"))):
            raise ValueError("actual registered own matched continuation control differs")
        control_index = {row["parent_world_id"]: {"top1": row["top1_success"], "hybrid": row["learned_reranking"]["success"]} for row in control_rows.values()}
        result["control_contrast"] = {"kind": kind(control_cfg), "pure_top1": paired_endpoint(control_index, after, task_parents, "top1"),
                                      "matched_hybrid": paired_endpoint(control_index, after, task_parents, "hybrid"), "costs": control_costs,
                                      "scope": "descriptive actual matched continuation-control contrast; no additional gain threshold is invented"}
    return result


def source_argument(record, flag):
    command = record.get("command", [])
    if not isinstance(command, list) or command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise ValueError("explicit source command argument required: " + flag)
    path = Path(command[command.index(flag) + 1])
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_run(proposer, guards):
    path = source_argument(proposer, "--run")
    if path != source_argument(guards, "--run") or file_hash(path) != proposer["run_sha256"] or file_hash(path) != guards["run_sha256"]:
        raise ValueError("actual run metadata path/hash differs across proposer/guard records")
    return json.loads(path.read_text(encoding="utf-8")), path


def read_sampling_inputs(run):
    train = source_argument(run, "--train")
    tasks = source_argument(run, "--tasks")
    teacher = source_argument(run, "--teacher-dir")
    paths = {"train": train, "tasks": tasks, "manifest": teacher / "manifest.json", "verified": teacher / "verified.jsonl"}
    hashes = {key: file_hash(path) for key, path in paths.items()}
    for name, key in (("train", "train_sha256"), ("tasks", "tasks_sha256"),
                      ("manifest", "teacher_manifest_sha256"), ("verified", "verified_corpus_sha256")):
        if hashes[name] != run.get(key):
            raise ValueError("recorded sampling input hash differs from actual file: " + name)
    rows = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"training_rows": rows(train), "corpus_rows": rows(paths["verified"]),
            "corpus_manifest": json.loads(paths["manifest"].read_text(encoding="utf-8")), "bound_hashes": hashes}, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student-proposer", "warm-proposer", "student-guards", "warm-guards", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--control-proposer", type=Path)
    parser.add_argument("--control-guards", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("gate reduction exists; refusing overwrite")
    if bool(args.control_proposer) != bool(args.control_guards):
        raise ValueError("provide both optional control records")
    paths = {name: getattr(args, name) for name in ("student_proposer", "warm_proposer", "student_guards", "warm_guards")}
    if args.control_proposer:
        paths.update(control_proposer=args.control_proposer, control_guards=args.control_guards)
    records = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    task_path = ROOT / "datasets/recovery-v1/planning_dev.jsonl"
    for name in ("student_proposer", "warm_proposer", "control_proposer"):
        if name in records and (source_argument(records[name], "--tasks") != task_path.resolve()
                                or file_hash(task_path) != records[name].get("tasks_sha256")):
            raise ValueError("registered DEV task file/path/hash differs")
    for name in ("student_guards", "warm_guards", "control_guards"):
        if name in records and (source_argument(records[name], "--data") != (ROOT / "datasets/recovery-v1/dev.jsonl").resolve()
                or source_argument(records[name], "--fixtures") != (ROOT / "results/retention_core_v1.fixtures.json").resolve()):
            raise ValueError("fixed DEV/behavior input paths required; no confirmation-bank access")
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_path, fixture_path = ROOT / "datasets/recovery-v1/dev.jsonl", ROOT / "results/retention_core_v1.fixtures.json"
    if file_hash(dev_path) != DEV_SHA or file_hash(fixture_path) != FIXTURE_SHA:
        raise ValueError("fixed source DEV/behavior bytes changed")
    dev_rows = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    student_run, student_path = read_run(records["student_proposer"], records["student_guards"])
    warm_run, warm_path = read_run(records["warm_proposer"], records["warm_guards"])
    sampling, sampling_paths = read_sampling_inputs(student_run)
    paths.update({"sampling_" + name: path for name, path in sampling_paths.items()})
    control_run = None
    if args.control_proposer:
        control_run, control_path = read_run(records["control_proposer"], records["control_guards"])
        control_sampling, control_sampling_paths = read_sampling_inputs(control_run)
        if control_sampling["bound_hashes"] != sampling["bound_hashes"]:
            raise ValueError("actual control sampling files differ from shared student inputs")
        paths.update({"control_sampling_" + name: path for name, path in control_sampling_paths.items()})
        paths["control_run"] = control_path
    result = analyze(**records, student_run=student_run, warm_run=warm_run, tasks=tasks, dev_rows=dev_rows,
                     control_run=control_run, **sampling)
    paths.update(student_run=student_path, warm_run=warm_path, tasks=task_path, dev=dev_path, fixtures=fixture_path)
    sources = [Path(__file__), ROOT / "src/dsv3wm/stats.py", ROOT / "src/dsv3wm/recovery_student.py",
               ROOT / "scripts/analyze_recovery_student_costs.py", ROOT / "scripts/evaluate_recovery_student_guards.py"]
    result.update(input_sha256={str(path.resolve()): file_hash(path) for path in paths.values()},
                  source_sha256={str(path.resolve()): file_hash(path) for path in sources})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
