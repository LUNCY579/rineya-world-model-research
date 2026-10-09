"""Render eval summaries as Markdown tables.

Usage: python scripts/report.py runs/EVAL_pilot_s1 [runs/EVAL_x ...] > table.md
Columns: success, open-loop success (round-1 plan executed in full), coverage@K at round 1, first-action optimality,
F calls/task (per-episode accounting), generated tokens/task, would-abstain rate, main outcomes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    for d in sys.argv[1:]:
        p = Path(d) / "summary.json"
        if not p.exists():
            continue
        S = json.loads(p.read_text())["summary"]
        splits = sorted({k.split("/")[1] for k in S})
        arms = []
        for k in S:
            a = k.split("/")[0]
            if a not in arms:
                arms.append(a)
        print(f"\n### {d}\n")
        for sp in splits:
            print(f"\n**{sp}**\n")
            print("| arm | n | success | open-loop | coverage@K r1 | 1st-action opt | F calls | gen tok | would-abstain | outcomes |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for a in arms:
                s = S.get(f"{a}/{sp}")
                if not s:
                    continue
                oc = ", ".join(f"{k}:{v}" for k, v in sorted(s["outcomes"].items(), key=lambda kv: -kv[1]))
                print(f"| {a} | {s['n']} | **{fmt(s['success'])}** | {fmt(s.get('open_loop_success'))} | "
                      f"{fmt(s.get('coverage_first'))} | {fmt(s.get('first_action_optimal'))} | "
                      f"{fmt(s.get('f_calls_per_task'), 1)} | {fmt(s.get('gen_tokens_per_task'), 0)} | "
                      f"{fmt(s.get('would_abstain_any'))} | {oc} |")


if __name__ == "__main__":
    main()
