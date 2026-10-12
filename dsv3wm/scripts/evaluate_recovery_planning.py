"""Frozen batched learned RegisterWorld planning development readout, W1/depth6."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from dsv3wm import REPO
    from dsv3wm.native import load, runtime, sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import LegacyPredictor, RecoveryConfig, load_warm
    from dsv3wm.recovery_planning import batched_search, score_choices, validate_tasks
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter", "tasks", "scoring", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = RecoveryConfig.read(args.config)
    if cfg.device != "cpu" or cfg.max_length > 512 or cfg.generation_tokens > 32:
        raise ValueError("Frozen recovery planning readout requires CPU, prompt<=512, generation<=32")
    out, traces_path = args.out, args.out.with_suffix(".traces.json")
    if out.exists() or traces_path.exists():
        raise FileExistsError("Refusing to overwrite recovery planning result/traces")
    freeze_sha = verify_freeze(args.freeze)
    sources = [Path(__file__), REPO / "dsv3wm/src/dsv3wm/recovery_planning.py", REPO / "dsv3wm/src/dsv3wm/recovery.py",
               REPO / "dsv3wm/src/dsv3wm/native.py", REPO / "dsv3wm/src/dsv3wm/provenance.py",
               REPO / "dsv3wm/src/dsv3wm/planning.py", REPO / "dsv3wm/src/dsv3wm/domains.py",
               REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py",
               REPO / "bwm/code/bwmn/gpu_lease.py", REPO / "bwm/code/scripts/planbench_qwen.py"]
    bound = [args.config, args.tasks, args.scoring, args.adapter / "adapter_model.safetensors",
             args.adapter / "adapter_config.json", *sources]
    verify_bound_inputs(args.freeze, bound)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_tasks(tasks)
    if len(tasks) != 12:
        raise ValueError("Recovery planning readout binds twelve independent development parents")
    result = {"status": "started", "instrument": "legacy learned batches/W1/depth6/query30/dev-only",
              "command": sys.argv, "config": cfg.dict(), "config_sha256": sha256(args.config),
              "tasks_sha256": sha256(args.tasks), "scoring_sha256": sha256(args.scoring),
              "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"), "freeze_sha256": freeze_sha,
              "source_sha256": {str(path): sha256(path) for path in sources}, "oracle_feedback": False,
              "confirmation_access": False, "scope": "G3 development planning, not efficacy confirmation"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    started = time.perf_counter()
    try:
        with runtime(cfg, "recovery-planning-" + cfg.arm):
            model, tokenizer, lm = load(cfg)
            model = load_warm(model, args.adapter)
            predictor = LegacyPredictor(model, tokenizer, cfg)
            search = batched_search(tasks, predictor, max_prompt_tokens=cfg.max_length,
                                    generation_tokens=cfg.generation_tokens)
            # Only now parse the scoring metadata and invoke exact verification.
            scoring = json.loads(args.scoring.read_text(encoding="utf-8"))
            scored = score_choices(tasks, search, scoring)
            traces_path.write_text(json.dumps(search["traces"], indent=2) + "\n", encoding="utf-8")
            summaries = {}
            for cohort in ("recovery_id_teacher_known", "recovery_ood_limit", "all"):
                selected = [row for row in scored if cohort == "all" or row["cohort"] == cohort]
                summaries[cohort] = {"n": len(selected), "successes": sum(row["success"] for row in selected),
                                     "optimal_successes": sum(row["optimal_success"] for row in selected),
                                     "first_action_optimal": sum(row["first_action_optimal"] for row in selected),
                                     "first_action_present": sum(row["first_action_present"] for row in selected),
                                     "queries": sum(row["model_queries"] for row in selected),
                                     "query_attempts": sum(row["query_attempts"] for row in selected),
                                     "input_tokens": sum(row["input_tokens"] for row in selected),
                                     "output_tokens": sum(row["output_tokens"] for row in selected),
                                     "processed_positions_upper_bound": sum(row["processed_positions_upper_bound"] for row in selected)}
            result.update(status="development-evaluated", per_parent=scored, summary=summaries,
                          budget=search["budget"], preflight_proof=search["preflight_proof"],
                          all_choices_fixed_before_scoring=True, traces_sha256=sha256(traces_path),
                          legacy_batch_processed_positions_upper_bound=predictor.decoder_token_positions,
                          legacy_decoder_forwards_lower_bound=predictor.decoder_forwards,
                          legacy_decoder_forwards_scope="Lower bound/estimate: lm.sample_crn can internally split broad batches; processed-position padding bound remains conservative.")
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["duration_s"] = time.perf_counter() - started
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": result["summary"], "duration_s": result["duration_s"]}), flush=True)


if __name__ == "__main__":
    main()
