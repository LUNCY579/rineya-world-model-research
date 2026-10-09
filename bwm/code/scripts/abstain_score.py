"""Proposer-intrinsic infeasibility detection on paired twins (same parent; feasible vs budget cut below optimum).

Signals per task (higher = more likely infeasible), all computed without the exact simulator:
  lr_fail   : log Q(plan|failure) - log Q(plan|success) for the greedy success-conditioned plan (outcome likelihood ratio)
  overrun   : 1 if the greedy plan is longer than the budget
  nll       : -log Q(plan|success) per token
  f_reject  : 1 if F_LM predicts that no candidate among K=4 succeeds (the gate rule's abstention)
Reports AUROC per signal on the requested split pair. Thresholds would be chosen on validation, never on test.

Usage: python scripts/abstain_score.py --fwd runs/F_titan_main/adapter --pa runs/P_actions_s1/adapter --split test --out runs/ABST_s1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import lm, planner, rw2  # noqa: E402
from run_eval import load_f_table  # noqa: E402


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--pa", required=True)
    ap.add_argument("--mode", default="actions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, tok = lm.load_core(a.core)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(ROOT / a.fwd), adapter_name="fwd", is_trainable=False)
    model.load_adapter(str(ROOT / a.pa), adapter_name="pa", is_trainable=False)
    model.eval()
    ftable = load_f_table(model, tok, ROOT / a.fwd, print)
    pools = rw2.build_pools()
    feas = rw2.eval_tasks(pools, "id", a.split, a.n, seed=0)
    inf = rw2.eval_tasks(pools, "id", a.split, a.n, seed=0, infeasible=True)
    tasks = feas + inf
    model.set_adapter("pa")
    prompts = [rw2.proposer_prompt(t.start, t.goal, t.limit, t.budget, "success") for t in tasks]
    gens = lm.sample_crn(model, tok, prompts, [None] * len(prompts), max_new_tokens=160)
    plans = [rw2.parse_plan(g["text"], a.mode) for g in gens]
    tgt = [g["text"] for g in gens]
    ps = lm.score_targets(model, tok, list(zip(prompts, tgt)))
    pf = lm.score_targets(model, tok, [(rw2.proposer_prompt(t.start, t.goal, t.limit, t.budget, "failure"), x)
                                       for t, x in zip(tasks, tgt)])
    # F-gate signal with K=4 candidates at the first decision
    prop = planner.LMProposer(model, tok, "pa", a.mode, K=4)
    states = [{"x": t.start, "goal": t.goal, "limit": t.limit, "remaining": t.budget, "family": t.family, "t": 0}
              for t in tasks]
    props = prop.propose(states, eval_seed=0)
    f = planner.TableForward(ftable, fallback=planner.LMForward(model, tok, "fwd"))
    rows = []
    for t, g, pl, s_, f_, pr in zip(tasks, gens, plans, ps, pf, props):
        valid = [c for c in pr["cands"] if c["plan"] is not None]
        rolls = planner.rollout_batch(f, [(t.start, t.limit)] * len(valid), [c["plan"] for c in valid])
        any_pred = any(planner.predicted_utility(c["plan"], r, t.goal, t.budget)[0] for c, r in zip(valid, rolls))
        ntok = max(1, g["n_tokens"])
        rows.append({"family": t.family, "feasible": t.feasible, "budget": t.budget, "opt": t.opt,
                     "lr_fail": f_ - s_, "nll": -s_ / ntok,
                     "overrun": int(pl is not None and len(pl[0]) > t.budget),
                     "f_reject": int(not any_pred), "malformed": pl is None})
    res = {}
    for sig in ("lr_fail", "nll", "overrun", "f_reject"):
        pos = [r[sig] for r in rows if not r["feasible"]]
        neg = [r[sig] for r in rows if r["feasible"]]
        res[sig] = auroc(pos, neg)
    print(json.dumps(res, indent=1))
    Path(ROOT / a.out).write_text(json.dumps({"args": vars(a), "auroc": res, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
