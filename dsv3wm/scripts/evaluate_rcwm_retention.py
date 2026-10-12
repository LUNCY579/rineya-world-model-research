"""Original-format supplementary R/P behavior on frozen fresh fixtures, CPU only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def strict_state(family, text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant")
    try:
        value = json.loads(text.strip(), object_pairs_hook=unique, parse_constant=constant)
        if family == "prerequisite_plan":
            return value if type(value) is int and 0 <= value <= 255 else None
        if family == "register_machine":
            return value if (isinstance(value, dict) and set(value) == {"X", "Y", "Z"}
                             and all(type(item) is int and 0 <= item < 97 for item in value.values())) else None
        return None
    except (ValueError, TypeError, AttributeError):
        return None


def score_all(fixtures, replies):
    """All generation must already be complete before this offline label access."""
    if len(fixtures) != len(replies):
        raise ValueError("Fixture/reply cardinality mismatch")
    rows = []
    for fixture, reply in zip(fixtures, replies):
        parsed = strict_state(fixture["family"], reply["text"])
        expected = json.loads(fixture["target"])
        rows.append({"id": fixture["id"], "family": fixture["family"], "cohort": fixture["cohort"],
                     "parent_world_id": fixture["parent_world_id"], "semantic_input_id": fixture["semantic_input_id"],
                     "graph_group_id": fixture["graph_group_id"],
                     "original_train_input_exposure": fixture["original_train_input_exposure"],
                     "original_train_parent_exposure": fixture["original_train_parent_exposure"],
                     "generated": reply, "parsed": parsed, "schema_valid": parsed is not None,
                     "correct": parsed is not None and parsed == expected})
    return rows


def generate_then_score(fixtures, generate):
    """Pass prompts only to generation; expose frozen targets after it returns."""
    replies = generate([fixture["prompt"] for fixture in fixtures])
    return replies, score_all(fixtures, replies)


def paired_changes(current, baseline):
    for key in ("fixture_sha256", "parent_sha256", "inference_contract"):
        if current[key] != baseline[key]:
            raise ValueError(f"Matched baseline drift: {key}")
    if baseline.get("status") != "development-evaluated":
        raise ValueError("Completed baseline behavior record required")
    previous = {row["id"]: row for row in baseline["rows"]}
    if set(previous) != {row["id"] for row in current["rows"]}:
        raise ValueError("Matched fixture identities differ")
    pairs = []
    for row in current["rows"]:
        old = previous[row["id"]]
        if row["parent_world_id"] != old["parent_world_id"] or row["semantic_input_id"] != old["semantic_input_id"]:
            raise ValueError("Matched parent/input identity drift")
        if row["graph_group_id"] != old["graph_group_id"]:
            raise ValueError("Matched shared rule-graph group drift")
        pairs.append({"id": row["id"], "family": row["family"], "parent_world_id": row["parent_world_id"],
                      "baseline_correct": old["correct"], "current_correct": row["correct"],
                      "gain": row["correct"] and not old["correct"], "loss": old["correct"] and not row["correct"]})
    return pairs


def main():
    from dsv3wm import REPO
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
        raise ValueError("RCWM R/P supplement binds CPU, 512 input tokens and greedy 64 output tokens")
    if args.out.exists():
        raise FileExistsError("RCWM retention result already exists")
    freeze_sha = verify_freeze(args.freeze)
    sources = [Path(__file__), REPO / "dsv3wm/scripts/build_rcwm_retention.py", REPO / "dsv3wm/src/dsv3wm/native.py",
               REPO / "dsv3wm/src/dsv3wm/provenance.py", REPO / "dsv3wm/src/dsv3wm/config.py",
               REPO / "dsv3wm/src/dsv3wm/domains.py", REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py",
               REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/gpu_lease.py",
               REPO / "bwm/code/scripts/planbench_qwen.py"]
    bound = [args.config, args.fixtures, args.adapter / "adapter_model.safetensors", args.adapter / "adapter_config.json", *sources]
    if args.baseline:
        bound.append(args.baseline)
    verify_bound_inputs(args.freeze, bound)
    document = json.loads(args.fixtures.read_text(encoding="utf-8"))
    fixtures = document["fixtures"]
    contract = {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64}
    if (document.get("schema") != "dsv3wm-rcwm-retention-v1" or len(fixtures) != 24
            or len({row["id"] for row in fixtures}) != 24
            or any(sum(row["family"] == family for row in fixtures) != 12
                   for family in ("register_machine", "prerequisite_plan"))
            or any(document["inference_contract"].get(key) != value for key, value in contract.items())):
        raise ValueError("Frozen 12-register/12-prerequisite fixture contract required")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    result = {"status": "started", "command": sys.argv, "model_label": args.model_label, "config": cfg.dict(),
              "config_sha256": sha256(args.config), "fixture_sha256": sha256(args.fixtures), "freeze_sha256": freeze_sha,
              "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"), "parent_sha256": EXPECTED_CORE_SHA,
              "source_sha256": {str(path): sha256(path) for path in sources}, "inference_contract": contract,
              "oracle_feedback": False, "all_generations_before_scoring": True, "retention_pass": None,
              "scope": document["claim_scope"], "parent_counts": document["parent_counts"],
              "graph_group_counts": document["graph_group_counts"], "rows": []}
    if baseline is not None and (baseline.get("status") != "development-evaluated"
                                 or baseline.get("fixture_sha256") != result["fixture_sha256"]
                                 or baseline.get("parent_sha256") != EXPECTED_CORE_SHA
                                 or baseline.get("inference_contract") != contract):
        raise ValueError("Matched completed B0 fixture/core/inference contract required before model loading")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    started = time.perf_counter()
    try:
        with runtime(cfg, "rcwm-rp-retention-" + args.model_label):
            import torch
            model, tokenizer, lm = load(cfg, args.adapter)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise RuntimeError("FP32 core and unmerged adapter required")
            model.eval()
            lm.DUTY_RATIO = 0.0
            prompts = [row["prompt"] for row in fixtures]
            lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
            if any(length > 512 for length in lengths):
                raise ValueError("Overlength retention input; no truncation")
            for fixture in fixtures:
                rendered = tokenizer.apply_chat_template(fixture["messages"], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                if rendered != fixture["prompt"]:
                    raise ValueError("Original prompt/chat template drift")
            boundary(cfg)
            replies, result["rows"] = generate_then_score(
                fixtures, lambda inputs: lm.sample_crn(model, tokenizer, inputs, [None] * 24, max_new_tokens=64))
            summary = {}
            for family in ("register_machine", "prerequisite_plan"):
                rows = [row for row in result["rows"] if row["family"] == family]
                summary[family] = {"n": len(rows), "parents": len({row["parent_world_id"] for row in rows}),
                                   "shared_graph_groups": len({row["graph_group_id"] for row in rows}),
                                   "correct": sum(row["correct"] for row in rows),
                                   "schema_valid": sum(row["schema_valid"] for row in rows)}
            result.update(status="development-evaluated", summary=summary, model_queries=24,
                          input_tokens=sum(lengths), output_tokens=sum(reply["n_tokens"] for reply in replies),
                          processed_positions_upper_bound=24 * (max(lengths) + 64),
                          unpadded_processed_positions_upper_bound=sum(n + reply["n_tokens"] for n, reply in zip(lengths, replies)),
                          position_scope="conservative standard-KV token-position bound including batch padding and capped final forward; not measured FLOPs")
            if baseline is not None:
                result["paired_changes"] = paired_changes(result, baseline)
                result["baseline_sha256"] = sha256(args.baseline)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["duration_s"] = time.perf_counter() - started
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": result["summary"], "retention_pass": None}), flush=True)


if __name__ == "__main__":
    main()
