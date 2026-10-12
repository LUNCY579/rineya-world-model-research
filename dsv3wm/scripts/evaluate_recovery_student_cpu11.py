"""DEV-only legacy student top1/pass4 and optional fixed-B0 learned reranking."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from dsv3wm import REPO
    from dsv3wm.native import runtime, sha256, verify_freeze
    from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import load_warm, LegacyPredictor, INITIAL_F, INITIAL_F_SHA
    from dsv3wm.recovery_student import (StudentConfig, generate_students, rerank_students, score_students,
                                         fixed_forward_adapter, admit_readout_lineage)
    from dsv3wm.proposer import NativePlanGenerator
    from dsv3wm.recovery_planning import validate_tasks
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter", "run", "tasks", "freeze", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--rerank", action="store_true", help="B0 learned ranking; charged after four proposals")
    parser.add_argument("--warm-baseline", action="store_true", help="unchanged final32 A2/A0 before H3 adaptation")
    parser.add_argument("--train", type=Path, help="original source TRAIN file; required only for --warm-baseline")
    args = parser.parse_args()
    cfg = StudentConfig.read(args.config)
    freeze_sha = verify_freeze(args.freeze)
    sources = [Path(__file__), *[REPO / ("dsv3wm/src/dsv3wm/" + name + ".py") for name in
              ("recovery_student", "recovery", "recovery_planning", "proposer", "native", "native_cpu11", "provenance", "domains", "planning", "config")],
              REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/gpu_lease.py",
              REPO / "bwm/code/scripts/planbench_qwen.py", REPO / "dsv3wm/scripts/evaluate_recovery_student.py"]
    bound = [args.config, args.adapter / "adapter_model.safetensors", args.adapter / "adapter_config.json",
             args.run, args.tasks, INITIAL_F / "adapter_config.json", *sources]
    if args.rerank:
        bound.extend([INITIAL_F / "adapter_model.safetensors", INITIAL_F / "adapter_config.json"])
        if sha256(INITIAL_F / "adapter_model.safetensors") != INITIAL_F_SHA:
            raise ValueError("fixed B0 forward identity mismatch")
    verify_bound_inputs(args.freeze, bound)
    lineage, extra_bound, readout_kind = admit_readout_lineage(args.adapter, args.run, cfg,
        warm_baseline=args.warm_baseline, train_path=args.train)
    verify_bound_inputs(args.freeze, extra_bound)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_tasks(tasks)
    if len(tasks) != 12:
        raise ValueError("fixed twelve DEV planning parents required")
    record = {"resource_admission_profile": admission_profile(), "status": "started", "config": cfg.dict(), "command": sys.argv, "freeze_sha256": freeze_sha,
              "tasks_sha256": sha256(args.tasks), "config_sha256": sha256(args.config), "run_sha256": sha256(args.run),
              "adapter_sha256": lineage["adapter_sha256"], "source_sha256": {str(path): sha256(path) for path in sources},
              "readout_kind": readout_kind, "artifact_arm": lineage["config"]["arm"],
              "artifact_training_config": lineage["config"], "artifact_training_status": lineage["status"],
              "artifact_training_freeze_sha256": lineage.get("freeze_sha256"),
              "artifact_source_inputs_sha256": {str(path): sha256(path) for path in extra_bound},
              "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
              "registered_adapter_config_sha256": sha256(INITIAL_F / "adapter_config.json"),
              "oracle_feedback": False, "confirmation_access": False, "scope": "H3 development feasibility; not efficacy confirmation",
              "with_search_status": "fixed-B0 learned candidate reranking enabled" if args.rerank else "deferred; pure direct readout only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    started, generated = time.perf_counter(), []
    try:
        with runtime(cfg, "recovery-student-readout"):
            model, tokenizer, lm = load(cfg)
            record["admission_check"] = last_admission_check()
            model = load_warm(model, args.adapter)
            generator = NativePlanGenerator(model, tokenizer, cfg,
                                             temperature=cfg.student_temperature, top_p=cfg.student_top_p)
            generated = generate_students(tasks, generator, base_seed=cfg.seed)
            reranked = None
            if args.rerank:
                # All student outputs are fixed before the explicit hybrid phase.
                with fixed_forward_adapter(model, INITIAL_F) as adapter_switch:
                    predictor = LegacyPredictor(model, tokenizer, cfg)
                    reranked = rerank_students(tasks, generated, predictor)
                record["forward_adapter_sha256"] = INITIAL_F_SHA
                record["adapter_switch"] = {**adapter_switch, "student_adapter_restored": True}
                record["inference_kind"] = "hybrid " + readout_kind + " + fixed-B0 learned dynamics; no exact selection"
            else:
                record["inference_kind"] = "pure " + readout_kind + "; fixed greedy top1"
            # No native model call occurs after this exact scoring boundary.
            outcomes = score_students(tasks, generated, reranked)
            record.update(status="development-evaluated", n=len(outcomes), outcomes=outcomes,
                          all_generation_and_learned_reranking_fixed_before_exact_scoring=True,
                          top1_successes=sum(row["top1_success"] for row in outcomes),
                          pass_at_4_successes=sum(row["pass_at_4"] for row in outcomes),
                          first_action_optimal=sum(row["first_action_optimal"] for row in outcomes),
                          valid_syntax_count=sum(row["valid_syntax_count"] for row in outcomes),
                          proposal_queries=sum(row["model_queries"] for row in outcomes),
                          proposal_input_tokens=sum(row["input_tokens"] for row in outcomes),
                          proposal_output_tokens=sum(row["output_tokens"] for row in outcomes),
                          proposal_processed_positions=sum(row["processed_positions"] for row in outcomes),
                          with_search_successes=sum(row["learned_reranking"]["success"] for row in outcomes) if reranked is not None else None,
                          reranking_forward_queries=sum(row["forward_queries"] for row in reranked) if reranked is not None else None)
    except BaseException as error:
        record.update(status="failed", error=repr(error), partial_generations=generated)
        raise
    finally:
        record["admission_check"] = last_admission_check()
        record["duration_s"] = time.perf_counter() - started
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": record["status"], "top1": record.get("top1_successes")}), flush=True)


if __name__ == "__main__":
    main()
