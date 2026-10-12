"""Development-only direct student evaluation; never a learned beam result."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime, load, sha256
    from dsv3wm.proposer import NativePlanGenerator, generate_plans, score_plans, validate_task
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--tasks", required=True, help="DEV-only JSONL initial states/goals, no labels/plans")
    parser.add_argument("--out", required=True, help="new JSON output path")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-p", type=float, default=.95)
    args = parser.parse_args()
    cfg = Config.read(args.config)
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tasks:
        raise ValueError("nonempty DEV task cohort required")
    for task in tasks:
        validate_task(task)
    if len({task["parent_world_id"] for task in tasks}) != len(tasks):
        raise ValueError("one task per parent required for this development instrument")
    if not 1 <= args.k <= 64 or args.max_actions <= 0:
        raise ValueError("invalid candidate/action budget")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), package / "src/dsv3wm/proposer.py", package / "src/dsv3wm/planning.py",
             package / "src/dsv3wm/domains.py", package / "src/dsv3wm/native.py",
             package.parent / "bwm/code/bwmn/lm.py"]
    result = {"status": "started", "instrument": "pure direct proposer; fixed top1, offline pass@k",
              "qualification_scope": "implementation and development outcomes do not establish A3/A4 efficacy; consult training records",
              "config": cfg.dict(), "config_sha256": sha256(args.config), "tasks_sha256": sha256(args.tasks),
              "adapter_sha256": sha256(Path(args.adapter) / "adapter_model.safetensors"),
              "source_sha256": {str(p.relative_to(package.parent)): sha256(p) for p in paths},
              "command": sys.argv, "k": args.k, "max_actions": args.max_actions,
              "temperature": args.temperature, "top_p": args.top_p, "oracle_feedback": False,
              "cache_policy": "existing BWM standard FP32 KV; seeded CRN; no compression"}
    # Reserve the output exclusively before any native work. Hash/input errors
    # above perform no native work and leave no misleading started-run artifact.
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")
    started, generated = time.perf_counter(), []
    try:
        with runtime(cfg, "evaluate-student"):
            model, tokenizer, lm = load(cfg, args.adapter)
            result["parent_sha256"] = sha256(lm.MODELS[cfg.core] / "model.safetensors")
            generator = NativePlanGenerator(model, tokenizer, cfg, temperature=args.temperature, top_p=args.top_p)
            for task in tasks:
                generated.append(generate_plans(task, generator, k=args.k, base_seed=cfg.seed,
                                                max_new_tokens=cfg.generation_tokens, max_actions=args.max_actions))
            # Every task/candidate is generated before any independent exact score.
            outcomes = [score_plans(task, row) for task, row in zip(tasks, generated)]
            result.update(status="development-evaluated", n=len(outcomes),
                          top1_successes=sum(row["top1_success"] for row in outcomes),
                          pass_at_k_successes=sum(row["pass_at_k"] for row in outcomes), outcomes=outcomes,
                          model_queries=sum(row["model_queries"] for row in outcomes),
                          input_tokens=sum(row["input_tokens"] for row in outcomes),
                          output_tokens=sum(row["output_tokens"] for row in outcomes),
                          decoder_forwards=sum(row["decoder_forwards"] for row in outcomes),
                          decoder_token_positions=sum(row["decoder_token_positions"] for row in outcomes))
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), partial_generations=generated)
        raise
    finally:
        result["duration_s"] = time.perf_counter() - started
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"out": str(out), "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
