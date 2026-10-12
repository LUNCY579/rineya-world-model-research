"""Prospective adapted R/P v2 freeze; inherit v2 B0 pins and strict Ledger lineage."""
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


def reference(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


lineage = reference("_rp_v1_complete_ledger_lineage_reference", ROOT / "scripts/freeze_rcwm_retention_adapted.py")
evaluator = reference("_rp_v2_baseline_inference_contract", ROOT / "scripts/evaluate_rcwm_retention_v2.py")
validate_run = lineage.validate_run


def validate_baseline(path):
    from dsv3wm.config import Config
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    from dsv3wm.native import EXPECTED_CORE_SHA, sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding="utf-8"))
    config, fixtures, old_freeze, adapter = [lineage.command_path(result, flag)
                                            for flag in ("--config", "--fixtures", "--freeze", "--adapter")]
    cfg, contract = Config.read(config), {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64}
    if (result.get("status") != "development-evaluated" or result.get("fixture_schema") != "dsv3wm-rcwm-retention-v2"
            or result.get("adapter_sha256") != T2_SHA or result.get("adapter_config_sha256") != T2_CONFIG_SHA
            or result.get("parent_sha256") != EXPECTED_CORE_SHA or adapter != T2_ADAPTER.resolve()
            or cfg.device != "cpu" or cfg.max_length != 512 or cfg.generation_tokens != 64
            or lineage.typed_json(result.get("inference_contract")) != lineage.typed_json(contract)
            or lineage.typed_json(result.get("config")) != lineage.typed_json(cfg.dict())
            or result.get("oracle_feedback") is not False or result.get("all_generations_before_scoring") is not True):
        raise ValueError("Completed unchanged T2 v2 B0 required; v1 outcome cannot replace it")
    if (sha256(config) != result.get("config_sha256") or sha256(fixtures) != result.get("fixture_sha256")
            or sha256(adapter / "adapter_model.safetensors") != T2_SHA
            or sha256(adapter / "adapter_config.json") != T2_CONFIG_SHA
            or verify_freeze(old_freeze) != result.get("freeze_sha256")):
        raise ValueError("Original v2 baseline config/fixture/model/freeze digest drift")
    source_paths = lineage.validate_source_record(result, evaluator.source_paths())
    verify_bound_inputs(old_freeze, [config, fixtures, adapter / "adapter_model.safetensors",
                                    adapter / "adapter_config.json", *evaluator.source_paths()])
    cases, _ = evaluator.validate_document(json.loads(fixtures.read_text(encoding="utf-8")))
    lookup, rows = {case["id"]: case for case in cases}, result.get("rows")
    if not isinstance(rows, list) or len(rows) != 24 or len({row["id"] for row in rows}) != 24 or {row["id"] for row in rows} != set(lookup):
        raise ValueError("Complete paired v2 per-case outcomes required")
    for row in rows:
        if (any(row.get(key) != lookup[row["id"]][key] for key in ("family", "parent_world_id", "semantic_input_id", "graph_group_id"))
                or type(row.get("correct")) is not bool or type(row.get("schema_valid")) is not bool):
            raise ValueError("V2 baseline case identity/type drift")
    return {"result": result, "fixtures": fixtures, "original_freeze": old_freeze,
            "files": json.loads(old_freeze.read_text(encoding="utf-8"))["files"],
            "paths": [path, config, fixtures, old_freeze, *source_paths]}


def main():
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    from dsv3wm.native import EXPECTED_CORE_SHA, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Adapted R/P v2 freeze exists")
    baseline = validate_baseline(args.baseline)
    paths = [*baseline["paths"], Path(__file__), ROOT / "scripts/freeze_rcwm_retention_adapted.py",
             ROOT / "scripts/train_ledger_recovery.py", ROOT / "src/dsv3wm/ledger_recovery.py",
             T2_ADAPTER / "adapter_model.safetensors", T2_ADAPTER / "adapter_config.json"]
    runs, train = {}, None
    for directory in args.run:
        validated = validate_run(directory, expected_train=train)
        if validated["arm"] in runs:
            raise ValueError("Duplicate Ledger arm")
        runs[validated["arm"]] = str(directory.resolve())
        train = validated["train"]
        paths += validated["paths"]
    files = dict(baseline["files"])
    files.update({str(path.resolve()).replace("\\", "/"): sha256(path) for path in sorted(set(paths))})
    record = {"instrument": "rcwm-RP-retention-adapted-lineage-v2", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "files": files, "ledger_runs": runs, "shared_train": str(train),
              "baseline_result": str(args.baseline.resolve()), "baseline_result_sha256": sha256(args.baseline),
              "baseline_original_freeze": str(baseline["original_freeze"]),
              "baseline_original_freeze_sha256": sha256(baseline["original_freeze"]),
              "fixtures": str(baseline["fixtures"]), "fixtures_sha256": sha256(baseline["fixtures"]),
              "parent_sha256": EXPECTED_CORE_SHA, "initial_T2_weight_sha256": T2_SHA, "initial_T2_config_sha256": T2_CONFIG_SHA,
              "confirmation_access": False, "scope": "prospective adapted tick-complete R/P behavior; all v1 artifacts unchanged"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
