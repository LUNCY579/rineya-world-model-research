"""Paired analysis across seeds: contrasts between arms on one split, clustered by held-out goal.

Usage: python scripts/analyze.py --runs runs/EVAL_s1,runs/EVAL_s2 --split id --contrast A11:A01 --contrast A01:A00
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import stats  # noqa: E402

def load(run, arm, split, metric):
    p = ROOT / run / f"episodes_{arm}_{split}.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return {r["family"]: (float(bool(r[metric])), r) for r in rows}

def strata_of(r, pool):
    t = r["task"]
    s = ["all", f"slack{t['budget'] - r['opt'] - 1}" if r["feasible"] else "infeasible"]
    if pool.startswith("range"):
        s.append("goal>64" if abs(t["goal"]) > 64 else "start_only>64")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated eval dirs, one per training seed")
    ap.add_argument("--split", required=True)
    ap.add_argument("--contrast", action="append", required=True, help="X:Y means X minus Y")
    ap.add_argument("--metric", default="success")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = a.runs.split(",")
    res = {}
    for c in a.contrast:
        X, Y = c.split(":")
        per = {}
        for si, run in enumerate(runs):
            dx, dy = load(run, X, a.split, a.metric), load(run, Y, a.split, a.metric)
            fams = sorted(set(dx) & set(dy))
            per[si] = (fams, dx, dy)
        fams = sorted(set.intersection(*[set(v[0]) for v in per.values()]))
        out = {}
        anyrow = per[0][1]
        strata = sorted({s for f in fams for s in strata_of(anyrow[f][1], a.split)})
        for st in strata:
            fs = [f for f in fams if st in strata_of(anyrow[f][1], a.split)]
            if len(fs) < 20:
                continue
            diff = {si: [per[si][1][f][0] - per[si][2][f][0] for f in fs] for si in per}
            goals = [anyrow[f][1]["task"]["goal"] for f in fs]
            bs = stats.crossed_cluster_bootstrap(diff, goals)
            tabs = {si: stats.paired_table([int(per[si][1][f][0]) for f in fs], [int(per[si][2][f][0]) for f in fs]) for si in per}
            out[st] = {"n": len(fs), "rate_X": sum(t["rate_a"] for t in tabs.values()) / len(tabs),
                       "rate_Y": sum(t["rate_b"] for t in tabs.values()) / len(tabs), **bs,
                       "per_seed_mcnemar": {si: {k: tabs[si][k] for k in ("only_a", "only_b", "mcnemar_p")} for si in tabs},
                       "seeds_positive": sum(v > 0 for v in bs["per_seed"].values())}
        res[c] = out
        o = out.get("all")
        if o:
            print(f"{c:14s} {a.split:9s} n={o['n']} X={o['rate_X']:.3f} Y={o['rate_Y']:.3f} diff={o['mean']:+.4f} "
                  f"CI[{o['ci_crossed_cluster'][0]:+.4f},{o['ci_crossed_cluster'][1]:+.4f}] seed+ {o['seeds_positive']}/{o['n_seeds']} "
                  f"seed-t {None if o['seed_t_ci'] is None else [round(x,4) for x in o['seed_t_ci']]}")
            for st, v in out.items():
                if st != "all":
                    print(f"    {st:14s} n={v['n']} X={v['rate_X']:.3f} Y={v['rate_Y']:.3f} diff={v['mean']:+.4f} CI[{v['ci_crossed_cluster'][0]:+.4f},{v['ci_crossed_cluster'][1]:+.4f}]")
    if a.out:
        Path(ROOT / a.out).write_text(json.dumps(res, indent=1, default=str))

if __name__ == "__main__":
    main()
