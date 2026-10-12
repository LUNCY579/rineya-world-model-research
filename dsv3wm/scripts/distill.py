"""TRAIN-only learned candidate search followed by independent exact verification.

There is deliberately no mock/oracle CLI predictor. Tests inject an explicitly
named deterministic mock. An optional adapter omission is recorded as base-only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from dsv3wm.config import Config
    from dsv3wm.data import load_records
    from dsv3wm.domains import canonical, sha256
    from dsv3wm.native import Predictor, load, runtime, sha256 as file_sha256
    from dsv3wm.planning import PlanningBudget, distill_ranked, learned_beam_search, rank_candidates
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="explicit native adapter; omission runs the frozen base")
    parser.add_argument("--candidates", help="TRAIN-only JSONL: parent_world_id and candidates action arrays")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-model-calls", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--keep", type=int, default=1)
    args = parser.parse_args()
    if args.limit <= 0 or args.keep <= 0:
        raise ValueError("positive limit and keep required")
    rows = load_records(args.train, expected_split="train")
    # Goals come only from already-labelled TRAIN trajectories. They are kept
    # separate from learned dynamics inputs and never serve as candidate plans.
    eligible = [row for row in rows if row["actions"] and all(row["labels"]["legal"])
                and all(row["labels"]["executed"])
                and canonical(row["s0"]) != canonical(row["labels"]["states"][-1])]
    selected = []
    seen = set()
    for row in eligible:
        if row["parent_world_id"] not in seen:
            seen.add(row["parent_world_id"])
            selected.append(row)
        if len(selected) == args.limit:
            break
    if not selected:
        raise ValueError("no nontrivial all-legal TRAIN goal trajectories")
    candidates = {}
    if args.candidates:
        for line in Path(args.candidates).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != "train" or not isinstance(row.get("candidates"), list):
                raise ValueError("candidates must explicitly be TRAIN-only")
            if row["parent_world_id"] not in seen or row["parent_world_id"] in candidates:
                raise ValueError("unknown/duplicate candidate parent")
            candidates[row["parent_world_id"]] = row["candidates"]
        if set(candidates) != seen:
            raise ValueError("candidate parents must match selected training parents")
    cfg = Config.read(args.config)
    budget = PlanningBudget(max_model_calls=args.max_model_calls, max_depth=args.max_depth,
                            beam_width=args.beam_width)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "started", "config": cfg.dict(), "config_sha256": file_sha256(args.config),
                "train_sha256": file_sha256(args.train), "adapter": args.adapter,
                "predictor": "native-unmerged-adapter" if args.adapter else "native-frozen-base",
                "budget": budget.__dict__, "command": sys.argv, "selected_parents": sorted(seen),
                "goal_source": "nontrivial all-legal labelled training trajectories",
                "goal_pool_sha256": sha256([{ "parent": r["parent_world_id"],
                    "goal": r["labels"]["states"][-1]} for r in selected])}
    if args.candidates:
        manifest["candidates_sha256"] = file_sha256(args.candidates)
    if args.adapter:
        manifest["adapter_sha256"] = file_sha256(Path(args.adapter) / "adapter_model.safetensors")
    positive, negative, searches = [], [], []
    try:
        with runtime(cfg, "distill"):
            model, tokenizer, _ = load(cfg, args.adapter)
            predictor = Predictor(model, tokenizer, cfg)
            for row in selected:
                conditioning = {key: row[key] for key in ("domain", "rules", "s0")}
                conditioning["actions"] = []
                goal = {"kind": "state", "state": row["labels"]["states"][-1]}
                if args.candidates:
                    ranked = rank_candidates(conditioning, goal, candidates[row["parent_world_id"]], predictor, budget)
                else:
                    ranked = learned_beam_search(conditioning, goal, predictor, budget)
                # All learned calls for this problem are complete before exact
                # verification sees any ranked candidate. No feedback is reused.
                retained = distill_ranked(row, goal, ranked, keep=args.keep)
                positive.extend(retained["verified"])
                negative.extend(retained["hard_negatives"])
                searches.append({"parent_world_id": row["parent_world_id"], "goal": goal, **ranked})
            manifest.update(status="pilot-distilled", verified_count=len(positive),
                            hard_negative_count=len(negative), model_calls=predictor.calls,
                            input_tokens=predictor.input_tokens, output_tokens=predictor.output_tokens)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        for name, values in (("verified.jsonl", positive), ("hard_negatives.jsonl", negative), ("searches.jsonl", searches)):
            (out / name).write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
