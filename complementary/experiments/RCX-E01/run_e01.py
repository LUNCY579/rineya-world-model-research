"""RCX-E01 runner (CPU mechanism tier): train one or more arms, evaluate on a battery, write summaries.

Usage (from complementary/):
  python -B experiments/RCX-E01/run_e01.py --arms VR-ans,CFSD-ans --seed 0 --out runs/RCX-E01/pilot \
      --battery dev --n-train 32000 --steps 3000
Every arm writes runs/.../<arm>_s<seed>/{train_log.jsonl, history.json, preds.jsonl.gz, summary.json}.
Checkpoints (*.pt) are git-ignored. The battery is generated once per (split, n_parents, seed) and cached
with its sha256 in the output directory.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.cuda.is_available = lambda: False  # CPU-only recipe (RCWM I-21)
torch.cuda.device_count = lambda: 0
torch.set_num_threads(int(os.environ.get("RCX_THREADS", "4")))

from rcx.guard import ThermalGuard  # noqa: E402
from rcx.models import char_lm as LM  # noqa: E402
from rcx.rcl import gen, score as S, world as W  # noqa: E402

ARMS = {
    "FX-ans": ("FX", "ans"), "FXSD-ans": ("FX_SD", "ans"),
    "VR-ans": ("VR", "ans"), "VR-trace": ("VR", "trace"),
    "CFSD-ans": ("CF_SD", "ans"), "CFSD-trace": ("CF_SD", "trace"), "CFSD-fill": ("CF_SD", "fill"),
    "VRSD-ans": ("VR_SD", "ans"), "CF-ans": ("CF", "ans"),
    "VRSD-trace": ("VR_SD", "trace"), "CF-trace": ("CF", "trace"),
}
CANON_ARMS = {"VR-ans", "VR-trace", "CF-ans", "CFSD-ans", "FX-ans"}
REGIME_SEED = {"FX": 1, "FX_SD": 2, "VR": 3, "VR_SD": 4, "CF": 5, "CF_SD": 6}
BATTERY_SEEDS = {"dev": 20261220}


def train_seed(regime, seed):
    return 20261300 + 100 * REGIME_SEED[regime] + seed


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_battery(out_dir, split, n_parents, seed):
    path = Path(out_dir) / f"battery_{split}_{n_parents}_{seed}.jsonl"
    if not path.exists():
        recs = gen.make_battery(split, n_parents, seed)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    return recs, sha_file(path)


def model_config(args):
    return LM.CharLMConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                           d_ff=args.d_ff, dropout=args.dropout, max_len=args.max_len)


def predict(model, recs, fmt, max_new, batch, canon=False):
    """Greedy outputs -> parsed after dicts (None when malformed). With canon, inputs are canonicalized
    (tool tier B) and outputs mapped back to the original names and key order."""
    if canon:
        conv = [W.canonicalize(r["input"]) for r in recs]
        texts = [c[0] for c in conv]
    else:
        texts = [r["input"] for r in recs]
    outs = LM.generate(model, texts, max_new=max_new, batch_size=batch)
    parsed = []
    for i, (r, o) in enumerate(zip(recs, outs)):
        if canon:
            ctext, corder, back = conv[i]
            p = W.parse_output(o, fmt, corder)
            parsed.append(back(p["after"]) if p else None)
        else:
            p = W.parse_output(o, fmt, r["state_order"])
            parsed.append(p["after"] if p else None)
    return outs, parsed


def evaluate(model, recs, fmt, max_new, batch, canon=False):
    outs, parsed = predict(model, recs, fmt, max_new, batch, canon)
    res = [S.classify(r, a) for r, a in zip(recs, parsed)]
    summ = S.aggregate(recs, res)
    comp = {s: S.composite(summ, s) for s in gen.SURFACES}
    return outs, res, summ, comp


def probe_table(recs, res, surface):
    """Per-stratum accuracy on probe (non-posted) vs posted-twin variants of F strata, for verdicts."""
    from collections import defaultdict
    t = defaultdict(lambda: [0, 0])
    for r, x in zip(recs, res):
        if r["surface"] == surface and r["group"].startswith("F"):
            key = f"{r['group']}:{r['variant']}"
            t[key][0] += x["exact"]
            t[key][1] += 1
    return {k: v[0] / v[1] for k, v in t.items()}


def run_arm(arm, args, battery, battery_sha):
    regime, fmt = ARMS[arm]
    out = Path(args.out) / f"{arm}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.force:
        print(f"[{arm}] summary exists, skipping", flush=True)
        return json.loads((out / "summary.json").read_text())
    tseed = train_seed(regime, args.seed)
    train = gen.make_train(regime, args.n_train, tseed)
    units = None
    if fmt == "fill":
        units = gen.fill_units(train)
        train = [gen.with_fill(r, units) for r in train]
        battery = [gen.with_fill(r, units) for r in battery]
    records = [{"input": r["input"], "target": r["targets"][fmt]} for r in train]
    max_tgt = max(len(r["target"]) for r in records)
    cfg = LM.TrainConfig(model=model_config(args), steps=args.steps, batch_size=args.batch, lr=args.lr,
                         warmup=args.warmup, weight_decay=0.1, grad_clip=1.0, bucket=args.bucket,
                         log_every=50, guard_every=10, checkpoint_path=str(out / "ckpt.pt"),
                         checkpoint_every=500)
    guard = ThermalGuard(log=lambda m: print(f"[{arm}] guard: {m}", flush=True))
    t0 = time.time()
    print(f"[{arm}] regime={regime} fmt={fmt} n_train={len(records)} max_target={max_tgt} seed={tseed}", flush=True)
    model, hist = LM.train(records, cfg, seed=args.seed, log_path=out / "train_log.jsonl", guard=guard)
    t_train = time.time() - t0
    max_new = max_tgt + 60
    t1 = time.time()
    outs, res, summ, comp = evaluate(model, battery, fmt, max_new, args.eval_batch)
    t_eval = time.time() - t1
    summary = {
        "arm": arm, "regime": regime, "fmt": fmt, "seed": args.seed, "train_seed": tseed,
        "n_train": len(records), "train_parents": len({r["parent_id"] for r in train}),
        "fill_units": units, "battery": args.battery, "battery_sha256": battery_sha,
        "model_config": vars(model_config(args)), "train_config": {k: v for k, v in vars(cfg).items() if k != "model"},
        "params": hist["params"], "tokens": hist["tokens"], "target_tokens": hist["target_tokens"],
        "train_wall_s": t_train, "eval_wall_s": t_eval, "guard_wait_s": hist["guard_wait_s"],
        "final_loss": hist["log"][-1]["loss"] if hist["log"] else None,
        "composite_exact": comp,
        "composite_decision": {s: S.composite(summ, s, field="decision") for s in gen.SURFACES},
        "cells": summ, "probe_in": probe_table(battery, res, "in"), "probe_ho": probe_table(battery, res, "ho"),
        "code_sha256": {p: sha_file(ROOT / p) for p in ("rcx/rcl/world.py", "rcx/rcl/gen.py", "rcx/rcl/score.py",
                                                        "rcx/rcl/templates.py", "rcx/rcl/vocab.py",
                                                        "rcx/models/char_lm.py", "experiments/RCX-E01/run_e01.py")},
    }
    if arm in CANON_ARMS and fmt != "fill":
        _, res_c, summ_c, comp_c = evaluate(model, battery, fmt, max_new, args.eval_batch, canon=True)
        summary["canon"] = {"composite_exact": comp_c, "cells": summ_c}
        with gzip.open(out / "preds_canon.jsonl.gz", "wt", encoding="utf-8") as f:
            for i, x in enumerate(res_c):
                f.write(json.dumps({"i": i, **x}) + "\n")
    with gzip.open(out / "preds.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (o, x) in enumerate(zip(outs, res)):
            f.write(json.dumps({"i": i, "out": o, **x}) + "\n")
    hist_small = {k: v for k, v in hist.items() if k != "log"}
    (out / "history.json").write_text(json.dumps(hist_small, indent=1, default=str), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    ck = out / "ckpt.pt"
    if ck.exists() and not args.keep_ckpt:
        LM.save(model, out / "model.pt")
        ck.unlink()
    print(f"[{arm}] done train {t_train:.0f}s eval {t_eval:.0f}s  P-in {comp['in']:.3f}  P-ho {comp['ho']:.3f}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--battery", default="dev", choices=["dev"])
    ap.add_argument("--dev-parents", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=32000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--bucket", type=int, default=8)
    ap.add_argument("--d-model", dest="d_model", type=int, default=128)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=3)
    ap.add_argument("--n-heads", dest="n_heads", type=int, default=4)
    ap.add_argument("--d-ff", dest="d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-len", dest="max_len", type=int, default=640)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-ckpt", action="store_true")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    battery, bsha = load_battery(args.out, args.battery, args.dev_parents, BATTERY_SEEDS[args.battery])
    print(f"battery {args.battery}: {len(battery)} records sha {bsha[:12]}", flush=True)
    for arm in args.arms.split(","):
        run_arm(arm.strip(), args, battery, bsha)


if __name__ == "__main__":
    main()
