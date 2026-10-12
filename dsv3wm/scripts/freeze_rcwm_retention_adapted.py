"""Freeze adapted R/P retention with complete Ledger training and T2 lineage.

The original B0 freezer/freeze remain immutable. This new prospective freeze
inherits their verified source pins and binds every completed Ledger run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def trainer_module():
    path = ROOT / "scripts/train_ledger_recovery.py"
    name = "_rcwm_adapted_retention_training_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Ledger training contract unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def command_path(record, flag):
    command = record.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command) or command.count(flag) != 1:
        raise ValueError(f"Exactly one recorded {flag} path is required")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"Recorded {flag} path is missing")
    return (REPO / command[index + 1]).resolve()


def typed_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def retention_sources():
    return [ROOT / "scripts/evaluate_rcwm_retention.py", ROOT / "scripts/build_rcwm_retention.py",
            ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/provenance.py", ROOT / "src/dsv3wm/config.py",
            ROOT / "src/dsv3wm/domains.py", REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py",
            REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/gpu_lease.py",
            REPO / "bwm/code/scripts/planbench_qwen.py"]


def validate_source_record(record, required):
    from dsv3wm.native import sha256
    sources = record.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("Recorded execution source hashes required")
    identities = {(REPO / name).resolve(): digest for name, digest in sources.items()}
    if not {Path(path).resolve() for path in required} <= set(identities):
        raise ValueError("Recorded execution source membership is incomplete")
    for path, expected in identities.items():
        if sha256(path) != expected:
            raise ValueError(f"Recorded execution source drift: {path}")
    return list(identities)


def validate_baseline(path):
    """Validate unchanged T2 result and its original source/data/config freeze."""
    from dsv3wm.config import Config
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    from dsv3wm.native import EXPECTED_CORE_SHA, sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding="utf-8"))
    config, fixtures, original_freeze, adapter = [command_path(result, flag)
                                                 for flag in ("--config", "--fixtures", "--freeze", "--adapter")]
    cfg = Config.read(config)
    contract = {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64}
    if (result.get("status") != "development-evaluated" or result.get("adapter_sha256") != T2_SHA
            or result.get("adapter_config_sha256") != T2_CONFIG_SHA or result.get("parent_sha256") != EXPECTED_CORE_SHA
            or adapter != T2_ADAPTER.resolve() or cfg.device != "cpu" or cfg.max_length != 512 or cfg.generation_tokens != 64
            or typed_json(result.get("inference_contract")) != typed_json(contract)
            or result.get("oracle_feedback") is not False or result.get("all_generations_before_scoring") is not True
            or typed_json(result.get("config")) != typed_json(cfg.dict())):
        raise ValueError("Completed unchanged T2 baseline with the fixed R/P inference contract required")
    if (sha256(config) != result.get("config_sha256") or sha256(fixtures) != result.get("fixture_sha256")
            or sha256(adapter / "adapter_model.safetensors") != T2_SHA
            or sha256(adapter / "adapter_config.json") != T2_CONFIG_SHA
            or verify_freeze(original_freeze) != result.get("freeze_sha256")):
        raise ValueError("Original T2 baseline raw config/data/model/freeze digest drift")
    sources = validate_source_record(result, retention_sources())
    verify_bound_inputs(original_freeze, [config, fixtures, adapter / "adapter_model.safetensors",
                                          adapter / "adapter_config.json", *retention_sources()])
    document = json.loads(fixtures.read_text(encoding="utf-8"))
    frozen_cases = {row["id"]: row for row in document["fixtures"]}
    rows = result.get("rows")
    if (document.get("schema") != "dsv3wm-rcwm-retention-v1" or len(frozen_cases) != 24
            or len(document["fixtures"]) != 24
            or typed_json({key: document.get("inference_contract", {}).get(key) for key in contract}) != typed_json(contract)
            or any(sum(row["family"] == family for row in document["fixtures"]) != 12
                   for family in ("register_machine", "prerequisite_plan"))
            or not isinstance(rows, list) or len(rows) != 24 or len({row["id"] for row in rows}) != 24
            or {row["id"] for row in rows} != set(frozen_cases)):
        raise ValueError("Completed baseline must preserve all 24 fixed fixture identities")
    for row in rows:
        fixture = frozen_cases[row["id"]]
        if (any(row.get(key) != fixture[key] for key in ("family", "parent_world_id", "semantic_input_id", "graph_group_id"))
                or type(row.get("correct")) is not bool or type(row.get("schema_valid")) is not bool):
            raise ValueError("Baseline per-case identity/type drift")
    frozen = json.loads(original_freeze.read_text(encoding="utf-8"))
    return {"result": result, "config": config, "fixtures": fixtures, "original_freeze": original_freeze,
            "paths": [path, config, fixtures, original_freeze, *sources], "files": frozen["files"]}


def validate_run(directory, *, expected_train=None):
    """Bind one completed A0/A1/A2/A5 run to all original training edges."""
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    from dsv3wm.native import EXPECTED_CORE_SHA, sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    trainer = trainer_module()
    directory = Path(directory).resolve()
    record_path = directory / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config, train, training_freeze, qualification = [command_path(record, flag)
                                                    for flag in ("--config", "--train", "--freeze", "--qualification")]
    cfg = trainer.LedgerConfig.read(config)
    if (record.get("status") != "pilot-trained" or record.get("initial_adapter_sha256") != T2_SHA
            or record.get("initial_adapter_config_sha256") != T2_CONFIG_SHA
            or record.get("core_sha256") != EXPECTED_CORE_SHA
            or typed_json(record.get("config")) != typed_json(cfg.dict())):
        raise ValueError("Completed fixed-recipe Ledger run with unchanged T2/core initialization required")
    if expected_train is not None and train != Path(expected_train).resolve():
        raise ValueError("Adapted retention runs must share the selected training dataset")
    dev, adapter = train.with_name("dev.jsonl"), directory / "adapter"
    for path, field in ((config, "config_sha256"), (train, "data_sha256"), (qualification, "qualification_sha256"),
                        (adapter / "adapter_model.safetensors", "adapter_sha256"),
                        (adapter / "adapter_config.json", "adapter_config_sha256")):
        if sha256(path) != record.get(field):
            raise ValueError(f"Recorded Ledger {field} digest drift")
    if verify_freeze(training_freeze) != record.get("freeze_sha256"):
        raise ValueError("Original Ledger training freeze digest drift")
    if sha256(T2_ADAPTER / "adapter_model.safetensors") != T2_SHA or sha256(T2_ADAPTER / "adapter_config.json") != T2_CONFIG_SHA:
        raise ValueError("Registered T2 initializer bytes drifted")
    sources = validate_source_record(record, trainer.required_sources())
    verify_bound_inputs(training_freeze, [config, train, dev, qualification, T2_ADAPTER / "adapter_model.safetensors",
                                         T2_ADAPTER / "adapter_config.json", *trainer.required_sources()])
    qualified, qualification_freeze_sha = trainer.bind_qualification(training_freeze, qualification, dev)
    if record.get("qualification_freeze_sha256") != qualification_freeze_sha:
        raise ValueError("Recorded qualification-to-training freeze linkage drift")
    original_qualification_freeze = command_path(qualified, "--freeze")
    if verify_freeze(original_qualification_freeze) != qualification_freeze_sha:
        raise ValueError("Original qualified B0 source/data freeze drift")
    qualification_config, qualification_data = [command_path(qualified, flag) for flag in ("--config", "--data")]
    qualification_cfg = trainer.LedgerConfig.read(qualification_config)
    if (qualification_data != dev or sha256(qualification_config) != qualified.get("config_sha256")
            or typed_json(qualified.get("config")) != typed_json(qualification_cfg.dict())
            or qualified.get("core_sha256") != EXPECTED_CORE_SHA or qualified.get("oracle_feedback") is not False):
        raise ValueError("Original qualified B0 full config/data/core contract drift")
    qualification_sources = validate_source_record(qualified, trainer.required_sources())
    verify_bound_inputs(original_qualification_freeze, [qualification_config, qualification_data,
                       T2_ADAPTER / "adapter_model.safetensors", T2_ADAPTER / "adapter_config.json", *trainer.required_sources()])
    actual = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    trainer.validate_adapter_config(actual)
    qualification_trace = qualification.with_suffix(".traces.json")
    paths = [record_path, config, train, dev, training_freeze, qualification, original_qualification_freeze,
             qualification_config, qualification_data, *qualification_sources,
             adapter / "adapter_model.safetensors", adapter / "adapter_config.json", *sources]
    if qualified.get("traces_sha256") is not None:
        paths.append(qualification_trace)
    return {"arm": cfg.arm, "record": record, "train": train, "paths": paths,
            "adapter": adapter, "training_freeze": training_freeze}


def main():
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    from dsv3wm.native import EXPECTED_CORE_SHA, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True, help="completed Ledger run directory; repeat")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Adapted R/P retention freeze exists; create a new freeze")
    baseline = validate_baseline(args.baseline)
    paths = [*baseline["paths"], Path(__file__), ROOT / "src/dsv3wm/ledger_recovery.py",
             ROOT / "scripts/train_ledger_recovery.py", T2_ADAPTER / "adapter_model.safetensors", T2_ADAPTER / "adapter_config.json"]
    runs, selected_train = {}, None
    for directory in args.run:
        validated = validate_run(directory, expected_train=selected_train)
        if validated["arm"] in runs:
            raise ValueError("Duplicate Ledger arm in adapted retention freeze")
        selected_train = validated["train"]
        runs[validated["arm"]] = str(Path(directory).resolve())
        paths.extend(validated["paths"])
    # Copy already-verified B0 pins, preserving old identities/digests; do not
    # reinterpret a changed B0 source as the original baseline instrument.
    files = dict(baseline["files"])
    files.update({str(path.resolve()).replace("\\", "/"): sha256(path) for path in sorted(set(paths))})
    record = {"instrument": "rcwm-RP-retention-adapted-lineage-v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "files": files, "ledger_runs": runs, "shared_train": str(selected_train),
              "baseline_result": str(args.baseline.resolve()), "baseline_result_sha256": sha256(args.baseline),
              "baseline_original_freeze": str(baseline["original_freeze"]),
              "baseline_original_freeze_sha256": sha256(baseline["original_freeze"]),
              "fixtures": str(baseline["fixtures"]), "fixtures_sha256": sha256(baseline["fixtures"]),
              "parent_sha256": EXPECTED_CORE_SHA, "initial_T2_weight_sha256": T2_SHA, "initial_T2_config_sha256": T2_CONFIG_SHA,
              "confirmation_access": False, "scope": "prospective supplementary adapted R/P behavior; original B0 freeze remains unchanged"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
