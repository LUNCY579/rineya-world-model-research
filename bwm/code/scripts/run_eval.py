"""Stage E: evaluate arms with execute-one-and-replan on fixed task sets.

The frozen backbone is loaded once; adapters are attached by name. F is frozen as a function: F_LM's predictions
over all (x, action, limit) for limits 32/48/64/96/128 are computed once per F adapter (cached next to the adapter)
and every arm reads that same table, so F errors are identical across arms.

Arms (proposer adapter, mode, F, rule, K, outcome):
  A00/A10 no F; A01/A11 F_LM rerank (primary); *g = F_LM gate (strict abstain operating point);
  *x = exact-F oracle diagnostic; A11sc = state-consistency rejection; A01k8/A01k12 = more action-only candidates;
  A00flip = outcome 'failure' requested (command-use control); BASE_* = frozen core without proposer adapter.
Baselines (no proposer): G1F/G2F greedy under F_LM, BFS16F/BFS64F/BFS256F breadth-first search under F_LM.

Usage:
  python scripts/run_eval.py --fwd runs/F_titan_main/adapter --pa runs/P_actions_s1/adapter --ps runs/P_states_s1/adapter \
      --arms A00,A10,A01,A11 --splits id_val --out runs/EVAL_pilot_s1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import lm, planner, rw2  # noqa: E402

# name: (pool, split, n, infeasible, budget_delta)
SPLITS = {
    "id_val": ("id", "validation", 400, False, 0),
    "long_val": ("long", "validation", 200, False, 0),
    "range_val": ("range", "validation", 300, False, 0),
    "id": ("id", "test", 1000, False, 0),
    "long": ("long", "test", 530, False, 0),
    "range": ("range", "test", 600, False, 0),
    "limitonly": ("limitonly", "test", 300, False, 0),
    "infeasible": ("id", "test", 300, True, 0),
    "id_b4": ("id", "test", 300, False, 4),
}

ARMS = {
    # name: (adapter, mode, F, rule, K, outcome, consistency_reject)
    "A00": ("pa", "actions", None, "rerank", 4, "success", False),
    "A10": ("ps", "states", None, "rerank", 4, "success", False),
    "A01": ("pa", "actions", "lm", "rerank", 4, "success", False),
    "A11": ("ps", "states", "lm", "rerank", 4, "success", False),
    "A01g": ("pa", "actions", "lm", "gate", 4, "success", False),
    "A11g": ("ps", "states", "lm", "gate", 4, "success", False),
    "A01x": ("pa", "actions", "exact", "rerank", 4, "success", False),
    "A11x": ("ps", "states", "exact", "rerank", 4, "success", False),
    "A11sc": ("ps", "states", "lm", "rerank", 4, "success", True),
    "A01k8": ("pa", "actions", "lm", "rerank", 8, "success", False),
    "A01k12": ("pa", "actions", "lm", "rerank", 12, "success", False),
    "A11k8": ("ps", "states", "lm", "rerank", 8, "success", False),
    "A00k1": ("pa", "actions", None, "rerank", 1, "success", False),
    "A10k1": ("ps", "states", None, "rerank", 1, "success", False),
    "A00flip": ("pa", "actions", None, "rerank", 1, "failure", False),
    "A10flip": ("ps", "states", None, "rerank", 1, "failure", False),
    "BASE_a": (None, "actions", None, "rerank", 4, "success", False),
    "BASE_aF": (None, "actions", "lm", "rerank", 4, "success", False),
    "RAND_F": ("random", None, "lm", "rerank", 4, "success", False),
}
BASELINES = {"G1F": ("greedy", 1), "G2F": ("greedy", 2), "BFS16F": ("bfs", 16), "BFS64F": ("bfs", 64),
             "BFS256F": ("bfs", 256)}


def full_keys():
    return [(x, a, L) for L in rw2.ID_LIMITS + rw2.OOD_RANGE_LIMITS for x in range(-L, L + 1) for a in rw2.ACTIONS]


def load_f_table(model, tok, fwd_dir: Path, log):
    cache = fwd_dir.parent / "F_TABLE.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        return {tuple(k): (tuple(v) if v is not None else None) for k, v in rows}
    t0 = time.time()
    f = planner.LMForward(model, tok, "fwd")
    keys = full_keys()
    vals = f._query(keys)
    cache.write_text(json.dumps([[list(k), (list(v) if v is not None else None)] for k, v in zip(keys, vals)]))
    acc = sum(v == rw2.exact_step(*k) for k, v in zip(keys, vals)) / len(keys)
    log(f"F table: {len(keys)} keys in {time.time()-t0:.0f}s, exact agreement {acc:.4f}")
    return dict(zip(keys, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--fwd", required=True, help="forward adapter dir (F_LM)")
    ap.add_argument("--pa", help="action-only proposer adapter dir")
    ap.add_argument("--ps", help="state-and-action proposer adapter dir")
    ap.add_argument("--arms", default="A00,A10,A01,A11")
    ap.add_argument("--baselines", default="")
    ap.add_argument("--splits", default="id_val")
    ap.add_argument("--n-scale", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke test: exact table stands in for F_LM")
    ap.add_argument("--slice", default="", help="START:END task slice (crash-resilient chunks; merge with merge_parts.py)")
    ap.add_argument("--share-prefix", action="store_true", help="shared-prompt prefill across the K candidates")
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

    lm.set_determinism(a.eval_seed)
    base, tok = lm.load_core(a.core)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(ROOT / a.fwd), adapter_name="fwd", is_trainable=False)
    for name, d in (("pa", a.pa), ("ps", a.ps)):
        if d:
            model.load_adapter(str(ROOT / d), adapter_name=name, is_trainable=False)
    model.eval()
    ftable = {k: rw2.exact_step(*k) for k in full_keys()} if a.smoke else load_f_table(model, tok, ROOT / a.fwd, P)
    lmf = planner.LMForward(model, tok, "fwd")

    def make_fwd(kind):
        if kind == "lm":
            return planner.TableForward(ftable, fallback=lmf)
        if kind == "exact":
            return planner.ExactForward()
        return None

    pools = rw2.build_pools()
    tasksets = {}
    for s in a.splits.split(","):
        pool, split, n, inf, bd = SPLITS[s]
        tasksets[s] = rw2.eval_tasks(pools, pool, split, int(n * a.n_scale), seed=0, infeasible=inf, budget_delta=bd)
        if a.slice:
            lo, hi = (int(v) for v in a.slice.split(":"))
            tasksets[s] = tasksets[s][lo:hi]
    part = f"__{a.slice.replace(':', '-')}" if a.slice else ""
    summ_path = out / "summary.json"
    summary = json.loads(summ_path.read_text())["summary"] if summ_path.exists() else {}

    def record(name, s, eps, t0):
        sm = planner.summarize(eps)
        sm["wall_s"] = round(time.time() - t0, 1)
        summary[f"{name}/{s}{part}"] = sm
        with open(out / f"episodes_{name}_{s}{part}.jsonl", "w", encoding="utf-8") as fh:
            for e in eps:
                fh.write(json.dumps(e) + "\n")
        P(f"{a.tag} {name:8s} {s:10s} succ {sm['success']:.3f} open-loop {sm['open_loop_success'] if sm['open_loop_success'] is None else round(sm['open_loop_success'],3)} "
          f"cov1 {sm['coverage_first'] if sm['coverage_first'] is None else round(sm['coverage_first'],3)} "
          f"fcalls {sm['f_calls_per_task']:.0f} gen_tok {sm['gen_tokens_per_task']:.0f} {sm['outcomes']} {sm['wall_s']}s")
        summ_path.write_text(json.dumps({"args": vars(a), "summary": summary}, indent=1))

    for bname in [b for b in a.baselines.split(",") if b]:
        kind, par = BASELINES[bname]
        for s, tasks in tasksets.items():
            t0 = time.time()
            f = make_fwd("lm")
            eps = planner.greedy_under_f(tasks, f, par) if kind == "greedy" else planner.bfs_under_f(tasks, f, par)
            record(bname, s, eps, t0)
    for arm in [x for x in a.arms.split(",") if x]:
        adapter, mode, fk, rule, K, outcome, sc = ARMS[arm]
        for s, tasks in tasksets.items():
            t0 = time.time()
            if adapter == "random":
                prop = planner.RandomProposer(K=K)
            else:
                prop = planner.LMProposer(model, tok, adapter, mode, K=K, temperature=a.temperature, top_p=a.top_p,
                                          outcome=outcome, share_prefix=a.share_prefix)
            prog = open(out / "progress.txt", "a", encoding="utf-8")
            prog.write(f"{time.strftime('%H:%M:%S')} start {arm}/{s} n={len(tasks)}" + chr(10))
            prog.flush()

            def plog(msg, _f=prog, _a=arm, _s=s):
                _f.write(f"{time.strftime('%H:%M:%S')} {_a}/{_s} {msg}" + chr(10))
                _f.flush()
            eps = planner.run_episodes(tasks, prop, make_fwd(fk), eval_seed=a.eval_seed, rule=rule,
                                       consistency_reject=sc, log=plog)
            record(arm, s, eps, t0)


if __name__ == "__main__":
    main()
