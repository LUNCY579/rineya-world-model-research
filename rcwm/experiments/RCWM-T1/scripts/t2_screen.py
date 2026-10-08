"""RCWM-T2 seed-0 screen (DECISIONS D-19): decides whether seeds 1 and 2 get GPU time. Fixed before any
T2 output existed. Exit 0 = pass (replicate), 1 = fail.

Pass iff, on data/t2_eval.jsonl with ON minus same-session OFF accuracy:
  * the mean gain over the 10 targeted strata is >= 0.15, and
  * no untouched stratum (ledger */posted, register swap, register add/nowrap) drops by more than 0.10.

  python scripts/t2_screen.py --tag t2_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETED = ["ledger|add/void", "ledger|scale/void", "ledger|transfer/void", "regist|mix/wrap", "regist|mix/nowrap",
            "regist|conditional/gt", "regist|conditional/le", "regist|add/wrap", "prereq|applicable", "prereq|inapplicable"]
UNTOUCHED = ["ledger|add/posted", "ledger|scale/posted", "ledger|transfer/posted", "regist|swap", "regist|add/nowrap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t2_s0")
    a = ap.parse_args()
    res = json.loads((ROOT / "runs" / "t2" / a.tag / "eval.json").read_text(encoding="utf-8"))
    b, t = res["per_op_t2_eval_base"], res["per_op_t2_eval_t2"]
    assert set(TARGETED + UNTOUCHED) <= set(b), sorted(set(TARGETED + UNTOUCHED) - set(b))
    gain = {k: t[k]["acc"] - b[k]["acc"] for k in TARGETED + UNTOUCHED}
    mean_t = sum(gain[k] for k in TARGETED) / len(TARGETED)
    worst_u = min(gain[k] for k in UNTOUCHED)
    ok = mean_t >= 0.15 and worst_u >= -0.10
    for k in TARGETED + UNTOUCHED:
        print(f"[t2-screen] {'T' if k in TARGETED else 'U'} {k:<24} {b[k]['acc']:.3f} -> {t[k]['acc']:.3f} ({gain[k]:+.3f})")
    print(f"[t2-screen] {a.tag}: mean targeted gain {mean_t:+.3f} (>= 0.15), worst untouched {worst_u:+.3f} (>= -0.10) -> "
          f"{'PASS: replicate seeds 1, 2' if ok else 'FAIL: no replication'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
