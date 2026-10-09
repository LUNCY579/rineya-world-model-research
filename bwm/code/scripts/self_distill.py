"""HC2: world-model-verified self-distillation on unlabeled parents.

For each unlabeled TRAIN-goal parent of a pool (default: OOD-range), sample K candidate plans from the current
proposer, verify them with a verifier, and keep the shortest verified plan. Verifier:
  --verifier f      F_LM in imagination (no executor, no teacher)          -> Q_FD
  --verifier exact  exact executor (upper bound, executor-verified data)    -> Q_XD
Rows (every suffix, outcome=success; states from the verifier's predicted states) are mixed 1:1 with the original
training rows, and the proposer adapter is fine-tuned further for --steps updates. Verification precision (fraction
of F-verified plans that the executor confirms) is computed for analysis only and never used for selection.

Usage: python scripts/self_distill.py --fwd runs/F_titan_main/adapter --prop runs/P_states_s1/adapter --mode states \
           --verifier f --parents 3000 --K 8 --steps 300 --out runs/SD_states_s1_f
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import lm, planner, rw2  # noqa: E402
from run_eval import load_f_table  # noqa: E402
from train_proposer import build_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--prop", required=True)
    ap.add_argument("--mode", choices=["actions", "states"], required=True)
    ap.add_argument("--verifier", choices=["f", "exact"], required=True)
    ap.add_argument("--pool", default="range")
    ap.add_argument("--parents", type=int, default=3000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mix", type=float, default=0.5, help="fraction of new verified rows in each batch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    lm.set_determinism(a.seed)
    base, tok = lm.load_core(a.core)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(ROOT / a.fwd), adapter_name="fwd", is_trainable=False)
    model.load_adapter(str(ROOT / a.prop), adapter_name="prop", is_trainable=True)
    model.eval()  # no LoRA dropout while proposing
    ftable = load_f_table(model, tok, ROOT / a.fwd, P)
    verifier = planner.TableForward(ftable, fallback=planner.LMForward(model, tok, "fwd")) if a.verifier == "f" \
        else planner.ExactForward()

    # 1) unlabeled parents (train goals only): start, goal, limit and a budget drawn like evaluation (slack 0-6)
    pools = rw2.build_pools()
    rng = random.Random(a.seed)
    fams = pools[a.pool]["train"][:a.parents]
    states = []
    tasks = []
    for s, g, L, o in fams:
        b = o + 1 + rng.choice(rw2.TRAIN_SLACKS)  # the budget is part of the problem statement, not a label
        t = rw2.Task(s, g, L, b)
        tasks.append(t)
        states.append({"x": s, "goal": g, "limit": L, "remaining": b, "family": t.family, "t": 0})
    t0 = time.time()
    model.set_adapter("prop")
    prop = planner.LMProposer(model, tok, "prop", a.mode, K=a.K, temperature=1.0, top_p=0.95)
    props = prop.propose(states, eval_seed=777 + a.seed)
    P(f"proposed {len(states)} x {a.K} in {time.time()-t0:.0f}s")
    starts, plans, keys = [], [], []
    for i, (s, p) in enumerate(zip(states, props)):
        for ci, c in enumerate(p["cands"]):
            if c["plan"] is not None:
                starts.append((s["x"], s["limit"]))
                plans.append(c["plan"])
                keys.append((i, ci))
    rolls = planner.rollout_batch(verifier, starts, plans)
    best = {}
    for (i, ci), plan, r in zip(keys, plans, rolls):
        succ, u = planner.predicted_utility(plan, r, states[i]["goal"], states[i]["remaining"])
        if succ:
            used = r["terminated_at"] + 1
            if i not in best or used < best[i][0]:
                best[i] = (used, plan[:used], r["pred_states"][:used])
    # verification precision (analysis only)
    tp = sum(rw2.execute(tasks[i], v[1])["success"] for i, v in best.items())
    cov_true = sum(any(c["plan"] is not None and rw2.execute(tasks[i], c["plan"])["success"] for c in props[i]["cands"])
                   for i in range(len(tasks)))
    stats = {"parents": len(tasks), "verified": len(best), "verified_true": tp,
             "precision": tp / max(1, len(best)), "true_coverage": cov_true / len(tasks),
             "verified_rate": len(best) / len(tasks)}
    P("verification", json.dumps(stats))

    # 2) rows from verified plans: every suffix, with the verifier's predicted states
    new = []
    for i, (used, plan, pst) in best.items():
        t = tasks[i]
        xs = [t.start] + pst
        for k in range(len(plan)):
            r = {"x": xs[k], "goal": t.goal, "limit": t.limit, "budget": t.budget - k, "outcome": "success",
                 "actions": plan[k:], "steps": [{"action": a_, "next_x": nx} for a_, nx in zip(plan[k:], pst[k:])]}
            new.append(r)
    old = build_rows(5121, 1)
    sft_new = rw2.proposer_sft(new, a.mode)
    sft_old = rw2.proposer_sft(old, a.mode)
    rows = sft_new + sft_old
    n_new = len(sft_new)
    orng = random.Random(a.seed * 31 + 7)
    order = []
    for _ in range(a.steps):
        for _ in range(a.batch):
            if orng.random() < a.mix and n_new:
                order.append(orng.randrange(n_new))
            else:
                order.append(n_new + orng.randrange(len(sft_old)))
    P(f"rows: new {n_new} (from {len(best)} verified plans), old {len(sft_old)}")
    model.set_adapter("prop")
    for n_, p_ in model.named_parameters():
        p_.requires_grad = ".prop." in n_ and "lora_" in n_
    trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    P(f"trainable (prop adapter) {trainable}")
    log = lm.train_sft(model, tok, rows, steps=a.steps, batch_size=a.batch, lr=a.lr, seed=a.seed, logger=P,
                       row_order=order)
    model.save_pretrained(str(out / "adapter_tmp"), selected_adapters=["prop"])
    src = out / "adapter_tmp" / "prop"
    dst = out / "adapter"
    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    src.rename(dst)
    (out / "adapter_tmp").rmdir()
    meta = {"args": vars(a), "verification": stats, "train": log}
    (dst / "RUN_META.json").write_text(json.dumps(meta, indent=1))
    (out / "metrics.json").write_text(json.dumps(meta, indent=1))
    P("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
