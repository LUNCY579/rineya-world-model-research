"""Addendum B3 (PREREG_B3_DISTILL.md) data preparation, CPU only.

  fresh  runs/BW_val_large/instances_b3_fresh.json: 110 generated hard-style instances with the hard set's block-count
         distribution (mirror generator, seed 20261011), excluding the invariants of hard, val_large, mirror and the 240
         distillation problems. Written once; refuses to overwrite.
  mix    runs/BW_DISTILL_r1/mix_v3.jsonl  = distillation rows D + |D| v2 proposer rows (seed 0)
         runs/BW_DISTILL_r1/mix_v2c.jsonl = 2|D| v2 proposer rows (seed 0; the first |D| are the v2 half of mix_v3)
         (distillation rows over MAX_LEN tokens are dropped explicitly and counted in MIX_META.json)
Usage: python scripts/bw_b3_prep.py fresh | mix [--out runs/BW_DISTILL_r1]
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
import blocks_val_large as VL  # noqa: E402
import bw_distill as BD  # noqa: E402

FRESH_SEED = 20261011
V2_ROWS = ROOT / "runs/BW_data_v2/proposer_actions_train.jsonl"
MAX_LEN = 640  # train_generic --max-len for both B3 arms


def fresh(a):
    dst = ROOT / "runs/BW_val_large/instances_b3_fresh.json"
    if dst.exists():
        sys.exit(f"{dst} exists (written once)")
    hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
    inv = {VL.invariant(*h) for h in hard}
    for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
        for r in json.loads((ROOT / "runs/BW_val_large" / fn).read_text()):
            inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    for it in BD.problems(240):
        inv.add(VL.invariant(it["objs"], it["init"], it["goal"]))
    sizes = sorted(len(h[0]) for h in hard)
    out = VL.make(random.Random(FRESH_SEED), sizes, B.hard_signatures(), inv)
    dst.write_text(json.dumps(out))
    print("b3_fresh", len(out), sorted(Counter(r["n"] for r in out).items()), "excluded invariants", len(inv))


def mix(a):
    out = ROOT / a.out
    from transformers import AutoTokenizer
    from bwmn import lm

    tok = AutoTokenizer.from_pretrained(lm.MODELS["titan-sparse"])
    d_all = [json.loads(l) for l in (out / "distill_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n_tok = lambda x: len(tok(x, add_special_tokens=False).input_ids)  # noqa: E731  (as lm.encode_pair: separately)
    d = [r for r in d_all if n_tok(r["prompt"]) + n_tok(r["target"]) + 1 <= MAX_LEN]
    v2 = [json.loads(l) for l in V2_ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(0)
    pick = rng.sample(range(len(v2)), 2 * len(d))
    v3 = d + [v2[i] for i in pick[:len(d)]]
    v2c = [v2[i] for i in pick]
    for name, rows in (("mix_v3.jsonl", v3), ("mix_v2c.jsonl", v2c)):
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"prompt": r["prompt"], "target": r["target"]}) + "\n")
    meta = {"distill_rows": len(d), "distill_rows_dropped_over_max_len": len(d_all) - len(d), "max_len": MAX_LEN,
            "v2_rows_available": len(v2), "mix_v3": len(v3), "mix_v2c": len(v2c), "seed": 0}
    (out / "MIX_META.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fresh", "mix"])
    ap.add_argument("--out", default="runs/BW_DISTILL_r1")
    a = ap.parse_args()
    fresh(a) if a.stage == "fresh" else mix(a)


if __name__ == "__main__":
    main()
