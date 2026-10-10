"""Joint roadmap item 4: search distillation (expert iteration) for the Blocksworld proposer.

Stage `search` (GPU): fresh generated hard-style problems (6-10 blocks, whole-configuration goals; seed disjoint from
val_large/mirror, hard-set signatures/invariants excluded as in blocks_val_large.py) are solved by the frozen B2
system's planner (learned F + proposer, beam search in imagination). Only plans that F VERIFIED (predicted
goal-reaching with every step legal) are kept, so the teacher signal comes from the model's own world model, never from
the exact simulator; executor validity is recorded for analysis only (label precision).
Stage `rows` (CPU): every suffix of each verified plan becomes a proposer row (state from F's predicted trajectory,
goal, outcome=success, target = remaining actions), mixed with the Pa_v2 rows for continued training.

Usage:
  python scripts/bw_distill.py search --n 240 --out runs/BW_DISTILL_r1        (GPU; resumable per chunk)
  python scripts/bw_distill.py rows --out runs/BW_DISTILL_r1                 (CPU)
Smoke test without models: --exact-smoke (exact F + scripted proposer; harness only, never used for training).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B  # noqa: E402
from bwmn import blocks_planner as BP  # noqa: E402
import blocks_val_large as VL  # noqa: E402

SEED = 20261010  # disjoint from val_large (20261008) and mirror (20261009)


def problems(n: int):
    hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
    sigs = B.hard_signatures()
    inv = {VL.invariant(*h) for h in hard}
    for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
        for r in json.loads((ROOT / "runs/BW_val_large" / fn).read_text()):
            inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    rng = random.Random(SEED)
    sizes = [6, 7, 8, 8, 9, 9, 10, 10]
    out = VL.make(rng, [sizes[i % len(sizes)] for i in range(n)], sigs, inv)
    return [{"id": f"distill/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"]))} for i, r in enumerate(out)]


def search(a):
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    inst = problems(a.n)
    done = set()
    sol = out / "solutions.jsonl"
    if sol.exists():
        done = {json.loads(l)["id"] for l in sol.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [it for it in inst if it["id"] not in done]
    if a.exact_smoke:
        sys.path.insert(0, str(Path(a.smoke_dir)))
        from test_plan_beam import MyopicProposer  # scratch harness proposer
        prop, fwd = MyopicProposer(), BP.ExactF()
    else:
        from bwmn import lm
        from peft import PeftModel
        frz = json.loads((ROOT / "runs/FREEZE_B2_beam.json").read_text())["args"]
        base, tok = lm.load_core("titan-sparse")
        model = PeftModel.from_pretrained(base, str(ROOT / frz["fwd_adapter"]), adapter_name="fwd", is_trainable=False)
        model.load_adapter(str(ROOT / frz["pa"]), adapter_name="pa", is_trainable=False)
        model.eval()
        fwd = BP.LMForwardBW(model, tok, "fwd", fmt=frz["fwd_fmt"], cache=out / "fcache.jsonl", max_batch_tokens=5000)
        prop = BP.LMProposerBW(model, tok, "pa", "actions", K=a.K, max_new=frz["M"] or None, prefix_ok=bool(frz["P"]),
                               max_batch_tokens=6000)
        a.R = frz["R"]
    for c0 in range(0, len(todo), a.chunk):
        part = todo[c0:c0 + a.chunk]
        traj_f = {}

        class Recording:  # keep F's predicted states along each submitted plan (for the training rows)
            def predict(self, keys):
                res = fwd.predict(keys)
                for k, r in zip(keys, res):
                    traj_f[k] = r
                return res

        eps = BP.plan_beam(part, prop, Recording(), eval_seed=a.eval_seed, width=a.W, max_expansions=a.E,
                           max_rollout=a.R)
        with open(sol, "a", encoding="utf-8") as fh:
            for it, e in zip(part, eps):
                plan = e["plan"] or []
                states, s = [], it["init"]
                for act in plan:
                    r = traj_f.get((s, act)) or fwd.predict([(s, act)])[0]
                    if r is None or not r[1]:
                        break
                    s = r[0]
                    states.append(s)
                ex = B.execute(it["init"], it["goal"], plan)
                fh.write(json.dumps({"id": it["id"], "n": len(it["objs"]), "objs": it["objs"],
                                     "init": sorted(map(list, it["init"])), "goal": sorted(map(list, it["goal"])),
                                     "verified": e["verified"], "plan": [list(x) for x in plan],
                                     "f_states": [sorted(map(list, x)) for x in states],
                                     "exact_valid_for_analysis_only": bool(ex["success"])}) + "\n")
        print(f"chunk {c0 // a.chunk}: {sum(e['verified'] for e in eps)}/{len(part)} verified", flush=True)


def rows(a):
    out = ROOT / a.out
    sols = [json.loads(l) for l in (out / "solutions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ver = [r for r in sols if r["verified"]]
    new = []
    for r in ver:
        goal = frozenset(map(tuple, r["goal"]))
        states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
        plan = [tuple(x) for x in r["plan"]]
        for t in range(len(plan)):
            new.append({"prompt": B.proposer_prompt(states[t], goal), "target": B.target_actions(plan[t:]),
                        "outcome": "success", "n_blocks": r["n"]})
    prec = sum(r["exact_valid_for_analysis_only"] for r in ver) / max(1, len(ver))
    with open(out / "distill_rows.jsonl", "w", encoding="utf-8") as fh:
        for x in new:
            fh.write(json.dumps(x) + "\n")
    meta = {"problems": len(sols), "verified": len(ver), "label_precision_analysis_only": round(prec, 4),
            "rows": len(new), "by_blocks": dict(Counter(r["n"] for r in ver))}
    (out / "DISTILL_META.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "rows"])
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--K", type=int, default=4, help="proposals per node (cheaper than the frozen K=8)")
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--E", type=int, default=24)
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--exact-smoke", action="store_true")
    ap.add_argument("--smoke-dir", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    search(a) if a.stage == "search" else rows(a)


if __name__ == "__main__":
    main()
