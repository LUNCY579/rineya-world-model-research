"""Tick-complete R/P v2 behavior with unchanged strict scoring and expanded pins."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def reference_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = reference_module("_rcwm_retention_v1_score_reference", ROOT / "scripts/evaluate_rcwm_retention.py")
builder = reference_module("_rcwm_retention_v2_input_contract", ROOT / "scripts/build_rcwm_retention_v2.py")
strict_state = reference.strict_state
score_all = reference.score_all
generate_then_score = reference.generate_then_score
paired_changes = reference.paired_changes


def source_paths():
    return builder.source_paths()


def validate_document(document):
    fixtures = document.get("fixtures")
    contract = {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64}
    if (document.get("schema") != "dsv3wm-rcwm-retention-v2" or not isinstance(fixtures, list) or len(fixtures) != 24
            or len({row["id"] for row in fixtures}) != 24
            or any(sum(row["family"] == family for row in fixtures) != 12
                   for family in ("register_machine", "prerequisite_plan"))
            or any(type(document["inference_contract"].get(key)) is not type(value)
                   or document["inference_contract"][key] != value for key, value in contract.items())):
        raise ValueError("Fixed tick-complete v2 12 R/12 P and unchanged inference caps required")
    for row in fixtures:
        event = row["input"]["event"]
        fields = ({"kind", "other", "register", "threshold", "tick", "value"} if row["family"] == "register_machine"
                  else {"add", "cost", "delete", "kind", "name", "pre_off", "pre_on"})
        if set(event) != fields or (row["family"] == "register_machine" and type(event["tick"]) is not int):
            raise ValueError("Original R/P event field contract required")
    return fixtures, contract


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import EXPECTED_CORE_SHA, boundary, load, runtime, sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter", "fixtures", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--model-label", default="unmerged-adapter")
    args = parser.parse_args()
    cfg = Config.read(args.config)
    if cfg.device != "cpu" or cfg.max_length != 512 or cfg.generation_tokens != 64:
        raise ValueError("R/P v2 requires CPU, 512 input tokens and greedy 64 generated tokens")
    if args.out.exists():
        raise FileExistsError("R/P v2 output exists")
    freeze_sha = verify_freeze(args.freeze)
    bound = [args.config, args.fixtures, args.adapter / "adapter_model.safetensors", args.adapter / "adapter_config.json", *source_paths()]
    if args.baseline:
        bound.append(args.baseline)
    verify_bound_inputs(args.freeze, bound)
    document = json.loads(args.fixtures.read_text(encoding="utf-8"))
    fixtures, contract = validate_document(document)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    fixture_sha = sha256(args.fixtures)
    if baseline is not None and (baseline.get("status") != "development-evaluated"
                                 or baseline.get("fixture_sha256") != fixture_sha
                                 or baseline.get("parent_sha256") != EXPECTED_CORE_SHA
                                 or baseline.get("inference_contract") != contract
                                 or baseline.get("fixture_schema") != "dsv3wm-rcwm-retention-v2"):
        raise ValueError("Matched completed v2 B0 required; v1 is not a substitute")
    result = {"status": "started", "command": sys.argv, "model_label": args.model_label, "config": cfg.dict(),
              "config_sha256": sha256(args.config), "fixture_sha256": fixture_sha, "fixture_schema": document["schema"],
              "freeze_sha256": freeze_sha, "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"), "parent_sha256": EXPECTED_CORE_SHA,
              "source_sha256": {str(path): sha256(path) for path in source_paths()}, "inference_contract": contract,
              "oracle_feedback": False, "all_generations_before_scoring": True, "retention_pass": None,
              "scope": document["claim_scope"], "parent_counts": document["parent_counts"],
              "graph_group_counts": document["graph_group_counts"], "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    started = time.perf_counter()
    try:
        with runtime(cfg, "rcwm-rp-v2-" + args.model_label):
            import torch
            model, tokenizer, lm = load(cfg, args.adapter)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise ValueError("FP32 unmerged core and adapter required")
            model.eval(); lm.DUTY_RATIO = 0.0
            prompts = [fixture["prompt"] for fixture in fixtures]
            lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
            if any(length > 512 for length in lengths):
                raise ValueError("Overlength v2 input; no truncation")
            for fixture in fixtures:
                rendered = tokenizer.apply_chat_template(fixture["messages"], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                if rendered != fixture["prompt"]:
                    raise ValueError("Original chat template drift")
            boundary(cfg)
            replies, result["rows"] = generate_then_score(
                fixtures, lambda inputs: lm.sample_crn(model, tokenizer, inputs, [None] * 24, max_new_tokens=64))
            summary = {}
            for family in ("register_machine", "prerequisite_plan"):
                rows = [row for row in result["rows"] if row["family"] == family]
                summary[family] = {"n": len(rows), "parents": len({row["parent_world_id"] for row in rows}),
                                   "shared_graph_groups": len({row["graph_group_id"] for row in rows}),
                                   "correct": sum(row["correct"] for row in rows), "schema_valid": sum(row["schema_valid"] for row in rows)}
            result.update(status="development-evaluated", summary=summary, model_queries=24,
                          input_tokens=sum(lengths), output_tokens=sum(reply["n_tokens"] for reply in replies),
                          processed_positions_upper_bound=24 * (max(lengths) + 64),
                          unpadded_processed_positions_upper_bound=sum(n + reply["n_tokens"] for n, reply in zip(lengths, replies)),
                          position_scope="conservative standard-KV padded token-position bound, not measured FLOPs")
            if baseline is not None:
                result.update(paired_changes=paired_changes(result, baseline), baseline_sha256=sha256(args.baseline))
    except BaseException as error:
        result.update(status="failed", error=repr(error)); raise
    finally:
        result["duration_s"] = time.perf_counter() - started
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": result["summary"], "retention_pass": None}), flush=True)


if __name__ == "__main__":
    main()
