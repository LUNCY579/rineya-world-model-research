"""Offline exposure and paired DEV supplement; no prediction or oracle calls.

Read selected TRAIN/DEV metadata and preserved results only. Never follow prior
public-source/vault references or read scoring banks/traces. Original P1/P2,
analyses, and gates stay unchanged; write to a fresh supplemental output.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from dsv3wm.domains import parent_world_key, sha256 as identity_hash
from dsv3wm.recovery import RecoveryConfig
from dsv3wm.stats import clustered_bootstrap, holm, paired_mcnemar, wilson_interval

ARMS = ("A0", "A1", "A2", "A5")
INITIAL_SHA = "c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466"
CORE_SHA = "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"
INITIAL_ADAPTER = REPO.parent / "Rineya_BWM_001_Research_Bundle/native_v1/runs/F_titan_main/adapter"
COHORTS = {"recovery_id_teacher_known": "id", "recovery_ood_limit": "ood"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=_invalid_constant)


class Evidence:
    """Selected byte bindings only; no traversal of a freeze's other inputs."""
    def __init__(self):
        self.files = {}

    def digest(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            self.files[str(path)] = value.hexdigest()
        return self.files[str(path)]

    def freeze(self, path):
        value = read_json(path)
        if value.get("confirmation_access") is not False or not isinstance(value.get("files"), dict):
            raise ValueError("development-only freeze required")
        self.digest(path)
        return value

    def bind(self, freeze, path):
        path = Path(path).resolve()
        entries = {}
        for name, digest in freeze["files"].items():
            resolved = (REPO / name).resolve()
            if resolved in entries and entries[resolved] != digest:
                raise ValueError("conflicting freeze path aliases")
            entries[resolved] = digest
        if path not in entries:
            raise ValueError(f"input absent from freeze: {path}")
        if entries[path] != self.digest(path):
            raise ValueError(f"input hash drift: {path}")


def command_path(command, flag):
    if not isinstance(command, list) or command.count(flag) != 1:
        raise ValueError(f"single recorded {flag} required")
    index = command.index(flag)
    if index + 1 == len(command) or not isinstance(command[index + 1], str):
        raise ValueError(f"missing recorded {flag} value")
    return (REPO / command[index + 1]).resolve()


def input_metadata(path, split, expected_count):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        if (row.get("split") != split or row.get("domain") != "register"
                or not isinstance(row.get("rules"), dict)
                or type(row["rules"].get("limit")) is not int
                or not isinstance(row.get("s0"), dict) or set(row["s0"]) != {"x"}
                or type(row["s0"]["x"]) is not int
                or row.get("parent_world_id") != parent_world_key("register", row["s0"], row["rules"])):
            raise ValueError("TRAIN/DEV metadata or canonical parent mismatch")
        cohort = row.get("provenance", {}).get("cohort")
        if (split == "train" and cohort != "recovery_train") or (split != "train" and cohort not in COHORTS):
            raise ValueError("unexpected recovery cohort")
        if not isinstance(row.get("actions"), list) or len(row["actions"]) != 4:
            raise ValueError("four-action trajectory metadata required")
        rows.append({key: row[key] for key in ("parent_world_id", "rules", "s0", "actions")}
                    | {"cohort": cohort})
    parents = [row["parent_world_id"] for row in rows]
    if len(rows) != expected_count or len(set(parents)) != len(parents):
        raise ValueError("fixed independent parent count required; no pseudoreplication")
    return rows


def audit_draw_order(record, config, train):
    order = record.get("row_order")
    if (not isinstance(order, list) or len(order) != config.steps * config.batch_size
            or any(type(index) is not int or not 0 <= index < len(train) for index in order)):
        raise ValueError("typed in-range training draw order required")
    rng = random.Random(177 + config.seed)
    if order != [rng.randrange(len(train)) for _ in order]:
        raise ValueError("recorded order differs from frozen with-replacement schedule")
    horizons = config.horizons
    uses = len(order) * len(horizons)
    if any(type(record.get("counts", {}).get(key)) is not int or record["counts"][key] != uses
           for key in ("forwards", "transitions")):
        raise ValueError("nominal horizon-response count differs from order/config")
    parents = [train[index]["parent_world_id"] for index in order]
    targets = sorted({(parent, horizon) for parent in parents for horizon in horizons})
    return {"trajectory_draws": len(order), "distinct_parents": len(set(parents)),
            "target_instance_uses": uses, "unique_parent_horizon_target_instances": len(targets),
            "target_instance_identity_unit": "(parent_world_id,horizon), not distinct response text or independent trials",
            "repeated_target_instance_uses": uses - len(targets), "horizons": list(horizons),
            "parent_draw_frequencies": dict(sorted(Counter(parents).items())),
            "draws_by_rule_limit": dict(sorted(Counter(str(train[i]["rules"]["limit"]) for i in order).items())),
            "ordered_parent_draws_sha256": identity_hash(parents),
            "unique_parent_horizon_targets_sha256": identity_hash(targets),
            "counts_transitions_unit": "one supervised horizon response per draw and horizon; equals nominal objective forwards, not distinct parents, independent trials, or simulator action executions"}


def adapter_behavior(value):
    value = {key: item for key, item in value.items()
             if key not in {"base_model_name_or_path", "revision", "peft_version", "inference_mode"}}
    targets = value.get("target_modules")
    if not isinstance(targets, list) or any(not isinstance(item, str) for item in targets):
        raise ValueError("string adapter target modules required")
    value["target_modules"] = sorted(set(targets))
    return canonical(value)


def audit_run(directory, train_path, train, readout, evidence, initial_adapter):
    directory = Path(directory).resolve()
    run_path = directory / "run.json"
    evidence.bind(readout, run_path)
    run = read_json(run_path)
    config_path = command_path(run["command"], "--config")
    training_freeze_path = command_path(run["command"], "--freeze")
    qualification_path = command_path(run["command"], "--qualification")
    for path in (config_path, training_freeze_path, qualification_path):
        evidence.bind(readout, path)
    config = RecoveryConfig.read(config_path)
    if (run.get("status") != "pilot-trained" or config.arm not in ARMS or config.seed != 0
            or config.steps != (96 if config.arm == "A5" else 32) or config.batch_size != 1
            or canonical(config.dict()) != canonical(run["config"])
            or run.get("optimizer_steps") != config.steps
            or command_path(run["command"], "--train") != Path(train_path).resolve()
            or command_path(run["command"], "--out") != directory
            or run.get("config_sha256") != evidence.digest(config_path)
            or run.get("data_sha256") != evidence.digest(train_path)
            or run.get("qualification_sha256") != evidence.digest(qualification_path)
            or run.get("freeze_sha256") != evidence.digest(training_freeze_path)
            or run.get("initial_adapter_sha256") != INITIAL_SHA or run.get("core_sha256") != CORE_SHA):
        raise ValueError("typed warm training config/hash lineage mismatch")
    if Path(readout.get("runs", {}).get(config.arm, "")).resolve() != directory:
        raise ValueError("run absent from selected readout freeze")
    training = evidence.freeze(training_freeze_path)
    for path in (config_path, train_path, qualification_path, initial_adapter / "adapter_model.safetensors",
                 initial_adapter / "adapter_config.json", ROOT / "scripts/train_recovery.py",
                 *[ROOT / f"src/dsv3wm/{name}.py" for name in ("__init__", "config", "recovery", "native", "domains", "stats")]):
        evidence.bind(training, path)
    qualification = read_json(qualification_path)
    if (qualification.get("status") != "development-evaluated" or qualification.get("baseline_qualified") is not True
            or qualification.get("oracle_feedback") is not False or qualification.get("adapter_sha256") != INITIAL_SHA
            or training.get("qualification_freeze_sha256") != qualification.get("freeze_sha256")
            or training.get("qualification_sha256") != evidence.digest(qualification_path)):
        raise ValueError("qualification lineage mismatch")
    adapter = directory / "adapter"
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        evidence.bind(readout, adapter / filename)
    if (run.get("adapter_sha256") != evidence.digest(adapter / "adapter_model.safetensors")
            or evidence.digest(initial_adapter / "adapter_model.safetensors") != INITIAL_SHA
            or adapter_behavior(read_json(adapter / "adapter_config.json"))
            != adapter_behavior(read_json(initial_adapter / "adapter_config.json"))):
        raise ValueError("adapter weight/configuration lineage mismatch")
    exposure = audit_draw_order(run, config, train)
    exposure["recorded_counts"] = run["counts"]
    exposure["config"] = config.dict()
    return config.arm, exposure, run


def outcome_map(result, metric, metadata):
    if not isinstance(result.get("per_parent"), list):
        raise ValueError("per-parent outcomes required")
    output = {}
    for row in result["per_parent"]:
        parent = row.get("parent_world_id")
        if parent in output or parent not in metadata:
            raise ValueError("duplicate or unexpected outcome parent; no pseudoreplication")
        cohort = COHORTS[metadata[parent]["cohort"]]
        if metric == "P1":
            outcome = row["horizons"]["4"]["rollout_correct"]
            if row.get("cohort") != cohort:
                raise ValueError("P1 ID/OOD cohort mismatch")
        else:
            outcome = row.get("success")
            if row.get("cohort") != metadata[parent]["cohort"] or row.get("task_id") != metadata[parent]["task_id"]:
                raise ValueError("planning task/cohort mismatch")
        if type(outcome) is not bool:
            raise ValueError("boolean parent-level outcome required")
        output[parent] = {"outcome": outcome, "cohort": cohort}
    if output.keys() != metadata.keys():
        raise ValueError("fixed parent set mismatch")
    return output


def compare_parents(baseline, candidate, repetitions=2000):
    if baseline.keys() != candidate.keys() or any(baseline[p]["cohort"] != candidate[p]["cohort"] for p in baseline):
        raise ValueError("paired parent/cohort mismatch")
    output = {}
    for cohort in ("all", "id", "ood"):
        parents = sorted(p for p in baseline if cohort == "all" or baseline[p]["cohort"] == cohort)
        if not parents:
            raise ValueError("nonempty fixed ID/OOD cohorts required")
        a = [baseline[p]["outcome"] for p in parents]
        b = [candidate[p]["outcome"] for p in parents]
        paired = paired_mcnemar(a, b)
        paired["baseline_wilson95"] = list(wilson_interval(sum(a), len(a)))
        paired["candidate_wilson95"] = list(wilson_interval(sum(b), len(b)))
        paired["parent_bootstrap_difference"] = clustered_bootstrap(
            [{"parent_world_id": p, "seed": 0, "baseline": int(x), "candidate": int(y)}
             for p, x, y in zip(parents, a, b)], repetitions=repetitions, random_seed=20261012)
        paired["bootstrap_scope"] = "empirical one-seed parent resampling; no training-seed uncertainty; a degenerate interval is not equivalence or a population accuracy bound"
        paired["parent_set_sha256"] = identity_hash(parents)
        output[cohort] = paired
    return output


def analyze(dataset, artifacts, results, freeze_path, repetitions=2000, *, initial_adapter=None):
    dataset, artifacts, results = map(lambda p: Path(p).resolve(), (dataset, artifacts, results))
    initial_adapter = Path(initial_adapter or INITIAL_ADAPTER).resolve()
    evidence = Evidence()
    readout = evidence.freeze(freeze_path)
    if readout.get("inherited_adapter_sha256") != INITIAL_SHA:
        raise ValueError("inherited adapter pin differs")
    selected = ("train.jsonl", "dev.jsonl", "planning_dev.jsonl", "manifest.json", "parent_registry.json")
    for filename in selected:
        evidence.bind(readout, dataset / filename)
    manifest = read_json(dataset / "manifest.json")
    registry = read_json(dataset / "parent_registry.json")
    for filename in selected:
        if filename != "manifest.json" and manifest["files"][filename]["sha256"] != evidence.digest(dataset / filename):
            raise ValueError("manifest selected-file hash mismatch")
    train = input_metadata(dataset / "train.jsonl", "train", 128)
    dev = input_metadata(dataset / "dev.jsonl", "dev", 24)
    train_parents = {row["parent_world_id"] for row in train}
    dev_meta = {row["parent_world_id"]: row for row in dev}
    if (train_parents & dev_meta.keys()
            or train_parents != {p for p, value in registry.items() if value.get("split") == "train"}
            or Counter(row["cohort"] for row in dev) != Counter({cohort: 12 for cohort in COHORTS})):
        raise ValueError("fixed TRAIN128/DEV24 independent parent registry mismatch")
    for row in train + dev:
        expected = {"split": "train" if row["cohort"] == "recovery_train" else "dev", "cohort": row["cohort"]}
        if registry.get(row["parent_world_id"]) != expected:
            raise ValueError("parent registry cohort/split mismatch")
    plan_meta = {}
    for line in (dataset / "planning_dev.jsonl").read_text(encoding="utf-8").splitlines():
        task = json.loads(line, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        parent = task.get("parent_world_id")
        if (task.get("split") != "dev" or task.get("domain") != "register" or parent not in dev_meta
                or parent in plan_meta or canonical(task["s0"]) != canonical(dev_meta[parent]["s0"])
                or canonical(task["rules"]) != canonical(dev_meta[parent]["rules"])
                or task.get("id") != identity_hash({key: task[key] for key in ("domain", "rules", "s0", "goal")})):
            raise ValueError("fixed planning parent/input identity mismatch")
        plan_meta[parent] = {"task_id": task["id"], "cohort": dev_meta[parent]["cohort"]}
    if len(plan_meta) != 12 or Counter(v["cohort"] for v in plan_meta.values()) != Counter({c: 6 for c in COHORTS}):
        raise ValueError("fixed twelve planning parents required")
    exposures, runs, metrics = {}, {}, {"P1": {}, "P2": {}}
    for arm in ARMS:
        identity, exposures[arm], runs[arm] = audit_run(artifacts / f"{arm}_s0", dataset / "train.jsonl",
                                                     train, readout, evidence, initial_adapter)
        if identity != arm:
            raise ValueError("arm directory/config mismatch")
        for metric, suffix, data_name, meta in (("P1", "dynamics", "dev.jsonl", dev_meta),
                                                ("P2", "planning", "planning_dev.jsonl", plan_meta)):
            path = results / f"{arm}_{suffix}.json"
            result = read_json(path)
            evidence.digest(path)
            if (result.get("status") != "development-evaluated" or result.get("oracle_feedback") is not False
                    or result.get("freeze_sha256") != evidence.digest(freeze_path)
                    or canonical(result.get("config")) != canonical(runs[arm]["config"])
                    or result.get("adapter_sha256") != runs[arm]["adapter_sha256"]
                    or result.get("data_sha256" if metric == "P1" else "tasks_sha256") != evidence.digest(dataset / data_name)):
                raise ValueError("DEV result config/input/adapter/freeze lineage mismatch")
            if metric == "P2" and (result.get("confirmation_access") is not False
                                    or result.get("all_choices_fixed_before_scoring") is not True
                                    or result.get("config_sha256") != runs[arm]["config_sha256"]
                                    or result.get("adapter_config_sha256") != evidence.digest(artifacts / f"{arm}_s0/adapter/adapter_config.json")):
                raise ValueError("planning generation/scoring or config lineage mismatch")
            metrics[metric][arm] = outcome_map(result, metric, meta)
    reference = {k: v for k, v in runs["A0"]["config"].items() if k not in {"arm", "steps"}}
    prefix = runs["A0"]["row_order"]
    for arm in ARMS:
        if (runs[arm]["row_order"][:len(prefix)] != prefix
                or canonical({k: v for k, v in runs[arm]["config"].items() if k not in {"arm", "steps"}}) != canonical(reference)
                or runs[arm]["freeze_sha256"] != runs["A0"]["freeze_sha256"]
                or canonical(runs[arm]["optimizer"]) != canonical(runs["A0"]["optimizer"])):
            raise ValueError("shared first32 schedule/config/optimizer/training-freeze drift")
    comparisons, pvalues = {}, {}
    for arm in ("A1", "A2"):
        comparisons[arm] = {}
        for metric in ("P1", "P2"):
            comparisons[arm][metric] = compare_parents(metrics[metric]["A5"], metrics[metric][arm], repetitions)
            pvalues[f"{arm}_{metric}_vs_A5"] = comparisons[arm][metric]["all"]["p_two_sided"]
    return {"analysis_status": "posthoc-descriptive", "scope": "fixed development, seed0; no native/oracle execution, confirmation, efficacy or equivalence claim",
            "original_P1_P2_and_gates_unchanged": True,
            "P1": "four-step own-state free-running exact final state",
            "P2": "registered nonterminating goal-state achievement, not H3 finish-contract success",
            "available_train_parents": len(train_parents), "train_parent_set_sha256": identity_hash(sorted(train_parents)),
            "sampling": "random.Random(177+seed), trajectory draws with replacement; common first32 verified",
            "exposure": exposures, "paired_vs_A5": comparisons,
            "Holm_supplemental_A5_four_test_family": holm(pvalues),
            "multiplicity_scope": "separate descriptive supplement; does not replace the original A0 family or prospective gates",
            "statistical_units": {"P1_parents": 24, "P2_parents": 12, "shared_endpoint_parents": len(dev_meta.keys() & plan_meta.keys()),
                                  "distinct_DEV_parents": len(dev_meta.keys() | plan_meta.keys()), "training_seeds": 1,
                                  "warning": "do not pool horizons, repeated draws, shared endpoints or queries as independent trials"},
            "core_hash_scope": "recorded native core digest matches registered pin; core weights not reread by this offline analysis",
            "provenance": evidence.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/recovery-v1")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/recovery-v1")
    parser.add_argument("--results", type=Path, default=ROOT / "results/recovery-v1")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve prior analysis; choose a new output")
    report = analyze(args.dataset, args.artifacts, args.results, args.freeze, args.bootstrap_repetitions)
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  analysis_source_sha256=Evidence().digest(Path(__file__)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
