"""FlagWorld stages: forward (train+grade F_LM), proposer (train), eval (arms), floors (CPU).

  python scripts/flags_run.py forward --out runs/FL_F_main
  python scripts/flags_run.py proposer --mode states --seed 1 --out runs/FL_P_states_s1
  python scripts/flags_run.py eval --fwd runs/FL_F_main/adapter --pa ... --ps ... --arms A00,A10,A01,A11 --splits id_val --out runs/FL_EVAL_pilot
  python scripts/flags_run.py floors
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import flags as FW  # noqa: E402
from bwmn import flags_planner as FP  # noqa: E402
from bwmn import lm  # noqa: E402

POOL_N, POOL_SEED = 6000, 11
TRANSFER_N, TRANSFER_SEED = 600, 12


def pools():
    feas, imp = FW.build_pool(POOL_N, POOL_SEED, imp_frac=0.1)
    tfeas, timp = FW.build_pool(TRANSFER_N, TRANSFER_SEED, transfer=True, imp_frac=0.1)
    return feas, imp, tfeas


def tasks_for(split_name, n, seed=0):
    feas, imp, tfeas = pools()
    rng = random.Random(f"flags-{split_name}-{seed}")
    if split_name in ("id_val", "id"):
        src = feas["validation" if split_name == "id_val" else "test"]
        return [FW.world_with_budget(w, o + 1 + rng.choice((0, 1, 2))) for w, o in src[:n]]
    if split_name == "transfer":
        src = tfeas["test"] + tfeas["validation"]
        return [FW.world_with_budget(w, o + 1 + rng.choice((0, 1, 2))) for w, o in src[:n]]
    if split_name == "infeasible":
        return [FW.world_with_budget(w, 8) for w in imp["test"][:n]]
    raise ValueError(split_name)


def logger(out):
    f = open(out / "log.txt", "a", encoding="utf-8")

    def P(*s):
        m = " ".join(str(x) for x in s)
        print(m, flush=True)
        f.write(m + "\n")
        f.flush()
    return P


def cmd_forward(a):
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    P = logger(out)
    feas, imp, tfeas = pools()
    _, fwd = FW.build_training_rows(feas["train"][:a.worlds], seed=1)
    uniq = {}
    for r in fwd:
        uniq.setdefault((r["state"], r["action"]), r)
    keys = sorted(uniq)
    random.Random(a.seed).shuffle(keys)
    rows = [uniq[k] for k in keys[:a.rows]]
    P(f"unique transitions {len(uniq)}; training {len(rows)}; illegal {sum(not r['legal'] for r in rows)}")
    model, tok = lm.load_core(a.core)
    pm = lm.attach_lora(model, seed=a.seed)
    log = lm.train_sft(pm, tok, FW.forward_sft(rows), steps=a.steps, batch_size=16, lr=1e-4, seed=a.seed, logger=P)
    lm.save_adapter(pm, out / "adapter", {"core": a.core, "args": vars(a), "train": log})
    res = grade(pm, tok, feas, tfeas, set(keys[:a.rows]))
    P(json.dumps(res))
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "train": log, "grade": res}, indent=1))


def grade(model, tok, feas, tfeas, seen):
    """Held-out transitions from validation worlds (n=5-7) and transfer worlds (n=8), all keys on real trajectories."""
    res = {}
    for name, worlds in (("val", feas["validation"][:300]), ("transfer", tfeas["validation"][:57] + tfeas["test"][:119])):
        _, fwd = FW.build_training_rows(worlds, seed=99)
        uniq = {}
        for r in fwd:
            uniq.setdefault((r["state"], r["action"]), r)
        ks = sorted(uniq)
        gens = lm.sample_crn(model, tok, [FW.forward_prompt(s, a_) for s, a_ in ks], [None] * len(ks), max_new_tokens=32)
        c = Counter()
        for k, g in zip(ks, gens):
            r = uniq[k]
            ok = FW.parse_forward(g["text"], len(k[0])) == (r["next"], r["legal"], r["terminated"])
            grp = "seen" if k in seen else "unseen"
            for s in ("all", grp, "illegal" if not r["legal"] else "legal", "finish" if k[1] == "finish" else "move"):
                c[f"n_{s}"] += 1
                c[f"ok_{s}"] += ok
        res[name] = {s[2:]: round(c["ok_" + s[2:]] / c[s], 4) for s in c if s.startswith("n_")}
        res[name]["n"] = c["n_all"]
    return res


def cmd_proposer(a):
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    P = logger(out)
    feas, imp, tfeas = pools()
    prop, _ = FW.build_training_rows(feas["train"][:a.worlds], seed=a.data_seed)
    if a.shuffle_states:
        rng = random.Random(a.seed)
        bank = {}
        for r in prop:
            for i, st in enumerate(r["steps"]):
                bank.setdefault((len(st["state"]), i), []).append(st["next"])
        prop = [{**r, "steps": [{**st, "next": rng.choice(bank[(len(st["state"]), i)])} for i, st in enumerate(r["steps"])]}
                for r in prop]
    sft = FW.proposer_sft(prop, a.mode)
    P(f"rows {len(sft)} outcomes {dict(Counter(r['outcome'] for r in prop))}")
    orng = random.Random(10007 * a.seed + 17)
    order = []
    while len(order) < a.steps * a.batch:
        idx = list(range(len(sft)))
        orng.shuffle(idx)
        order.extend(idx)
    model, tok = lm.load_core(a.core)
    pm = lm.attach_lora(model, seed=a.seed)
    log = lm.train_sft(pm, tok, sft, steps=a.steps, batch_size=a.batch, lr=1e-4, seed=a.seed, logger=P,
                       row_order=order, max_len=512)
    lm.save_adapter(pm, out / "adapter", {"core": a.core, "args": vars(a), "train": log})
    (out / "metrics.json").write_text(json.dumps({"args": vars(a), "train": log}, indent=1))


ARMS = {
    "A00": ("pa", "actions", None, "rerank", 4, "success"),
    "A10": ("ps", "states", None, "rerank", 4, "success"),
    "A01": ("pa", "actions", "lm", "rerank", 4, "success"),
    "A11": ("ps", "states", "lm", "rerank", 4, "success"),
    "A01g": ("pa", "actions", "lm", "gate", 4, "success"),
    "A11g": ("ps", "states", "lm", "gate", 4, "success"),
    "A01x": ("pa", "actions", "exact", "rerank", 4, "success"),
    "A11x": ("ps", "states", "exact", "rerank", 4, "success"),
    "A00flip": ("pa", "actions", None, "rerank", 1, "failure"),
}


def cmd_eval(a):
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    P = logger(out)
    lm.set_determinism(a.eval_seed)
    base, tok = lm.load_core(a.core)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(ROOT / a.fwd), adapter_name="fwd", is_trainable=False)
    for name, d in (("pa", a.pa), ("ps", a.ps)):
        if d:
            model.load_adapter(str(ROOT / d), adapter_name=name, is_trainable=False)
    model.eval()
    lmf = FP.LMForwardFlags(model, tok, "fwd")  # shared memo = frozen function across arms
    sp = out / "summary.json"
    summary = json.loads(sp.read_text())["summary"] if sp.exists() else {}
    for arm in a.arms.split(","):
        adapter, mode, fk, rule, K, outcome = ARMS[arm]
        for s in a.splits.split(","):
            worlds = tasks_for(s, int({"id_val": 300, "id": 800, "transfer": 176, "infeasible": 200}[s] * a.n_scale))
            t0 = time.time()
            prop = FP.LMProposerFlags(model, tok, adapter, mode, K=K, outcome=outcome)
            fwd = lmf if fk == "lm" else FP.ExactForwardFlags() if fk == "exact" else None
            eps = FP.run_episodes(worlds, prop, fwd, eval_seed=a.eval_seed, rule=rule)
            sm = FP.summarize(eps)
            sm["wall_s"] = round(time.time() - t0, 1)
            summary[f"{arm}/{s}"] = sm
            with open(out / f"episodes_{arm}_{s}.jsonl", "w", encoding="utf-8") as fh:
                for e in eps:
                    fh.write(json.dumps(e) + "\n")
            P(f"{arm:8s} {s:10s} succ {sm['success']:.3f} open-loop {sm['open_loop_success']} cov1 {sm['coverage_first']} "
              f"fcalls {sm['f_calls_per_task']:.0f} gen {sm['gen_tokens_per_task']:.0f} {sm['outcomes']} {sm['wall_s']}s")
            sp.write_text(json.dumps({"args": vars(a), "summary": summary}, indent=1))


def cmd_floors(a):
    for s in ("id_val", "id", "transfer"):
        ws = tasks_for(s, {"id_val": 300, "id": 800, "transfer": 176}[s])
        g1 = sum(FW.greedy_episode(w, w.budget, 1)["success"] for w in ws) / len(ws)
        g2 = sum(FW.greedy_episode(w, w.budget, 2)["success"] for w in ws) / len(ws)
        print(s, len(ws), "greedy1", round(g1, 3), "greedy2", round(g2, 3),
              "opt dist", sorted(Counter(FW.opt_moves(w) for w in ws).items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forward")
    f.add_argument("--core", default="titan-sparse")
    f.add_argument("--worlds", type=int, default=2500)
    f.add_argument("--rows", type=int, default=8000)
    f.add_argument("--steps", type=int, default=500)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", required=True)
    p = sub.add_parser("proposer")
    p.add_argument("--core", default="titan-sparse")
    p.add_argument("--mode", choices=["actions", "states"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-seed", type=int, default=1)
    p.add_argument("--worlds", type=int, default=3879)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--shuffle-states", action="store_true")
    p.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--core", default="titan-sparse")
    e.add_argument("--fwd", required=True)
    e.add_argument("--pa")
    e.add_argument("--ps")
    e.add_argument("--arms", default="A00,A10,A01,A11")
    e.add_argument("--splits", default="id_val")
    e.add_argument("--n-scale", type=float, default=1.0)
    e.add_argument("--eval-seed", type=int, default=0)
    e.add_argument("--out", required=True)
    sub.add_parser("floors")
    a = ap.parse_args()
    {"forward": cmd_forward, "proposer": cmd_proposer, "eval": cmd_eval, "floors": cmd_floors}[a.cmd](a)


if __name__ == "__main__":
    main()
