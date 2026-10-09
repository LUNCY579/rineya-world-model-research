"""Stage P: train an outcome-conditioned proposer adapter (action-only or state-and-action) on a frozen core.

Matched design: for a given --seed, both modes use the SAME rows, the SAME row order, the SAME number of updates and
the SAME LoRA initialisation seed. Only the target rendering differs (mechanism view; state targets cost more tokens).

Usage: python scripts/train_proposer.py --mode actions --seed 11 --out runs/P_actions_s11
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
from bwmn import lm, rw2  # noqa: E402


def build_rows(families: int, data_seed: int, filter_success: bool = False, shuffle_states: bool = False,
               shuffle_seed: int = 0, pool: str = "id"):
    pools = rw2.build_pools()
    data = rw2.build_training_rows(pools, families, seed=data_seed, pool=pool)
    rows = data["proposer"]
    if filter_success:
        rows = [r for r in rows if r["outcome"] == "success"]
    if shuffle_states:
        # Control: keep actions intact, replace each claimed state by a state from another row at the same
        # position (compatible shape, semantically wrong).
        rng = random.Random(shuffle_seed)
        bank = {}
        for r in rows:
            for i, st in enumerate(r["steps"]):
                bank.setdefault(i, []).append(st["next_x"])
        new = []
        for r in rows:
            steps = [{"action": st["action"], "next_x": rng.choice(bank[i])} for i, st in enumerate(r["steps"])]
            new.append({**r, "steps": steps})
        rows = new
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--mode", choices=["actions", "states"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--families", type=int, default=2000)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--filter-success", action="store_true", help="success-only SFT baseline (filtered BC)")
    ap.add_argument("--shuffle-states", action="store_true", help="shuffled-state-target control (states mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save-at", default="", help="comma-separated steps at which to save intermediate adapters")
    ap.add_argument("--pool", default="id")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a", encoding="utf-8")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    rows = build_rows(a.families, a.data_seed, a.filter_success, a.shuffle_states, a.seed, a.pool)
    sft = rw2.proposer_sft(rows, a.mode)
    P(f"rows {len(sft)} outcomes {dict(Counter(r['outcome'] for r in rows))} policies {dict(Counter(r['policy'] for r in rows))}")
    # identical order across modes for the same seed
    order_rng = random.Random(10007 * a.seed + 17)
    order = []
    while len(order) < a.steps * a.batch:
        idx = list(range(len(sft)))
        order_rng.shuffle(idx)
        order.extend(idx)
    model, tok = lm.load_core(a.core)
    pm = lm.attach_lora(model, seed=a.seed)
    t, n = lm.count_trainable(pm)
    P(f"trainable {t} / {n}")
    t0 = time.time()
    save_at = {int(x) for x in a.save_at.split(",") if x}

    def on_step(step, m):
        if step in save_at and step != a.steps:
            lm.save_adapter(m, out / f"adapter_step{step}", {"core": a.core, "args": vars(a), "step": step})
            P(f"saved intermediate adapter at step {step}")

    trainlog = lm.train_sft(pm, tok, sft, steps=a.steps, batch_size=a.batch, lr=a.lr, seed=a.seed, logger=P,
                            row_order=order, grad_ckpt=not a.no_ckpt, on_step=on_step,
                            ckpt_dir=out / "ckpt")
    P("train done", json.dumps({k: v for k, v in trainlog.items() if k != "loss_curve_every50"}))
    meta = {"core": a.core, "args": vars(a), "rows": len(sft), "train": trainlog,
            "row_order_sha": lm.hashlib.sha256(json.dumps(order[:a.steps * a.batch]).encode()).hexdigest()}
    lm.save_adapter(pm, out / "adapter", meta)
    __import__("shutil").rmtree(out / "ckpt", ignore_errors=True)
    (out / "metrics.json").write_text(json.dumps(meta, indent=1))
    P("total s", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
