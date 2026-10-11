"""Blocksworld evaluation: PlanBench Task 1 (test, run once per frozen config) or generated validation instances.

  python scripts/blocks_eval.py --set val --proposer oracle --fwd exact --out runs/BW_EVAL_harness
  python scripts/blocks_eval.py --set planbench --proposer lm --pa runs/BW_Pa_s1/adapter --fwd lm --fwd-adapter runs/BW_F_s0/adapter --K 8 --out runs/BW_EVAL_x
Every submitted plan is scored by the exact executor; with --val, every plan is also checked by VAL (PlanBench's
validator) and disagreements are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import blocks_planner as BP  # noqa: E402


TEST_SETS = ("planbench", "hard", "seeds_fresh")  # seeds_fresh: item 7 (PREREG_ITEM7_SEEDS.md, D-65)


def _json_set(path: Path, prefix: str):
    return [{"id": f"{prefix}/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"])), "path": None} for i, r in enumerate(json.loads(path.read_text()))]


def load_set(name: str, n: int):
    out = []
    if name == "hard":
        for p in sorted((B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1])):
            objs, init, goal = B.parse_problem(p)
            out.append({"id": f"hard/{p.stem}", "objs": objs, "init": init, "goal": goal, "path": str(p),
                        "domain": str(B.BENCH / "blocksworld_hard" / "generated_domain.pddl")})
    elif name == "val_v2":
        out = _json_set(ROOT / "runs/BW_data_v2/instances_val.json", "val_v2")
    elif name == "val_large":
        out = _json_set(ROOT / "runs/BW_val_large/instances_val_large.json", "val_large")
    elif name == "mirror":
        out = _json_set(ROOT / "runs/BW_val_large/instances_mirror_hard.json", "mirror")
    elif name == "seeds_fresh":
        out = _json_set(ROOT / "runs/BW_val_large/instances_seeds_fresh.json", "seeds_fresh")
    elif name == "planbench":
        for sub in ("generated_basic_3", "generated_basic"):
            for p in sorted((B.BENCH / "blocksworld" / sub).glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1])):
                objs, init, goal = B.parse_problem(p)
                out.append({"id": f"{sub}/{p.stem}", "objs": objs, "init": init, "goal": goal, "path": str(p)})
    elif name == "val":
        for i, r in enumerate(json.loads((ROOT / "runs/BW_data/instances_val.json").read_text())):
            out.append({"id": f"val/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
                        "goal": frozenset(map(tuple, r["goal"])), "path": None})
    return out[:n] if n else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["planbench", "hard", "val", "val_v2", "val_large", "mirror", "seeds_fresh"], required=True)
    ap.add_argument("--planner", choices=["replan", "beam", "direct"], default="replan")
    ap.add_argument("--W", type=int, default=4, help="beam width (beam planner)")
    ap.add_argument("--E", type=int, default=48, help="expansion cap per instance (beam planner)")
    ap.add_argument("--max-depth", type=int, default=80)
    ap.add_argument("--step-cap", type=int, default=24, help="replan planner step cap")
    ap.add_argument("--R", type=int, default=0, help="beam planner rollout cap (0 = whole candidate)")
    ap.add_argument("--fwd-fmt", choices=["state", "delta"], default="state")
    ap.add_argument("--max-batch-tokens", type=int, default=5000)
    ap.add_argument("--M", type=int, default=0, help="proposer token cap (0 = default 120)")
    ap.add_argument("--P", type=int, default=0, help="1 = accept truncated plan prefixes (beam rollout cap)")
    ap.add_argument("--prop-batch-tokens", type=int, default=6000)
    ap.add_argument("--freeze", help="freeze record (JSON) required for test sets; its 'args' must match this run")
    ap.add_argument("--shard", default="", help="i/k: run instances with index % k == i (D-50; outputs do not depend on "
                                                 "shard composition: per-instance CRN seeds, batch-independent sampling)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--proposer", choices=["oracle", "lm"], required=True)
    ap.add_argument("--pa", help="action-only proposer adapter")
    ap.add_argument("--mode", default="actions")
    ap.add_argument("--fwd", choices=["exact", "lm"], required=True)
    ap.add_argument("--fwd-adapter")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--val", action="store_true", help="also check every plan with VAL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    if a.set in TEST_SETS:
        # Test sets run once per frozen configuration (PREREG_PLANBENCH.md): settings must match the freeze record.
        if not a.freeze:
            sys.exit(f"--set {a.set} is a test set: pass --freeze <record> (see PREREG_PLANBENCH.md)")
        frz = json.loads((ROOT / a.freeze).read_text())
        keys = ("proposer", "pa", "mode", "fwd", "fwd_adapter", "fwd_fmt", "K", "eval_seed", "planner", "W", "E",
                "max_depth", "step_cap", "R", "M", "P")
        bad = {k: (getattr(a, k), frz["args"].get(k)) for k in keys if getattr(a, k) != frz["args"].get(k)}
        if bad:
            sys.exit(f"settings differ from the freeze record {a.freeze}: {bad}")
        if (out / "summary.json").exists():
            sys.exit(f"{out}/summary.json exists: test sets are run once")
    out.mkdir(parents=True, exist_ok=True)
    inst = load_set(a.set, a.n)
    if a.shard:
        si, sk = map(int, a.shard.split("/"))
        inst = [it for j, it in enumerate(inst) if j % sk == si]
    model = tok = None
    if a.proposer == "lm" or a.fwd == "lm":
        from bwmn import lm
        from peft import PeftModel
        base, tok = lm.load_core("titan-sparse")
        first = True
        for name, d in (("fwd", a.fwd_adapter), ("pa", a.pa)):
            if not d:
                continue
            if first:
                model = PeftModel.from_pretrained(base, str(ROOT / d), adapter_name=name, is_trainable=False)
                first = False
            else:
                model.load_adapter(str(ROOT / d), adapter_name=name, is_trainable=False)
        model.eval()
    prop = BP.OracleProposer() if a.proposer == "oracle" else BP.LMProposerBW(
        model, tok, "pa", a.mode, K=a.K, max_new=a.M or None, prefix_ok=bool(a.P), max_batch_tokens=a.prop_batch_tokens)
    fwd = BP.ExactF() if a.fwd == "exact" else BP.LMForwardBW(model, tok, "fwd", fmt=a.fwd_fmt,
                                                              cache=out / "fcache.jsonl",
                                                              max_batch_tokens=a.max_batch_tokens)
    t0 = time.time()
    prog = open(out / "progress.txt", "a", encoding="utf-8")
    plog = lambda m: (prog.write(time.strftime("%H:%M:%S ") + m + "\n"), prog.flush())
    if a.planner == "beam":
        eps = BP.plan_beam(inst, prop, fwd, eval_seed=a.eval_seed, width=a.W, max_expansions=a.E,
                           max_depth=a.max_depth, max_rollout=a.R, log=plog)
    elif a.planner == "direct":
        eps = BP.plan_direct(inst, prop, eval_seed=a.eval_seed)
    else:
        eps = BP.plan_instances(inst, prop, fwd, eval_seed=a.eval_seed, step_cap=a.step_cap, log=plog)
    summ, rows = BP.score(inst, eps)
    summ["wall_s"] = round(time.time() - t0, 1)
    summ["f_model_queries"] = getattr(fwd, "model_queries", getattr(fwd, "calls", None))
    summ["f_queries_per_instance"] = round((summ["f_model_queries"] or 0) / max(1, len(inst)), 1)
    summ["proposer_tokens_per_instance"] = round(sum(e["gen_tokens"] for e in eps) / max(1, len(inst)), 1)
    summ["expansions_per_instance"] = round(sum(e.get("expansions", e.get("rounds", 0)) for e in eps) / max(1, len(inst)), 2)
    if a.val:
        dis = 0
        for it, r in zip(inst, rows):
            if it["path"] is None:
                p = out / "tmp_problem.pddl"
                p.write_text(B.write_problem(it["objs"], it["init"], it["goal"]))
            else:
                p = Path(it["path"])
            dom = Path(it["domain"]) if it.get("domain") else B.DOMAIN_PDDL
            v = B.val_check(p, [tuple(x.split()) for x in r["plan"]], domain_path=dom)
            r["val"] = v
            dis += (v != r["success"])
        summ["val_success"] = sum(r["val"] for r in rows) / len(rows)
        summ["val_disagreements"] = dis
    by_blocks = {}
    for it, r in zip(inst, rows):
        by_blocks.setdefault(len(it["objs"]), []).append(r["success"])
    summ["success_by_blocks"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_blocks.items())}
    print(json.dumps(summ), flush=True)
    (out / "summary.json").write_text(json.dumps({"args": vars(a), "summary": summ}, indent=1))
    with open(out / "plans.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
