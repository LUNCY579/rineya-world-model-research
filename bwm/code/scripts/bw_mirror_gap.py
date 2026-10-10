"""Post-hoc, descriptive analysis of the pre-registered hard-minus-mirror gap (frozen B2 S-beam; DECISIONS P-10).

The registered flag (gap > 10 pp) stands as computed; this script only describes where the gap comes from:
  - a lower bound on moves (2 x blocks that must move: not settled on their goal support or above such a block);
  - goal and initial tower structure;
  - success by the tallest goal tower (<= 6, 7-9, >= 10) in each set, and the mirror's success reweighted to the
    hard set's mix of those bins (direct standardization).
Writes runs/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json. CPU, no model.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B  # noqa: E402
import blocks_val_large as VL  # noqa: E402


def must_move(objs, init, goal) -> int:
    on_i = {f[1]: f[2] for f in init if f[0] == "on"}
    on_g = {f[1]: f[2] for f in goal if f[0] == "on"}
    memo = {}

    def settled(x):
        if x not in memo:
            below = on_i.get(x)
            if x in on_g and on_g[x] != below:
                memo[x] = False
            else:
                memo[x] = True if below is None else settled(below)
        return memo[x]

    return sum(not settled(x) for x in objs)


def bin_of(h: int) -> str:
    return "<=6" if h <= 6 else "7-9" if h <= 9 else ">=10"


def main():
    hard = []
    for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"):
        o, i, g = B.parse_problem(p)
        hard.append((f"hard/{p.stem}", o, i, g))
    mirror = [(f"mirror/{k}", r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"])))
              for k, r in enumerate(json.loads((ROOT / "runs/BW_val_large/instances_mirror_hard.json").read_text()))]
    succ = {}
    for t in ("hard", "mirror"):
        for line in open(ROOT / f"runs/BW_TEST_B2/{t}_beam/plans.jsonl", encoding="utf-8"):
            r = json.loads(line)
            succ[r["id"]] = int(bool(r.get("val", r["success"])))
    out = {"note": "post-hoc, descriptive; the registered flag stands as computed"}
    bins = {}
    for name, insts in (("hard", hard), ("mirror", mirror)):
        lb, gmax, gcount, imax, by = [], [], [], [], defaultdict(list)
        for iid, o, i, g in insts:
            gt = [h for h in VL.goal_towers(g, o) if h > 1]
            it = [h for h in VL.towers_of(i, o) if h > 1]
            lb.append(2 * must_move(o, i, g))
            gmax.append(max(gt) if gt else 1)
            gcount.append(len(gt))
            imax.append(max(it) if it else 1)
            by[bin_of(gmax[-1])].append(succ[iid])
        bins[name] = by
        out[name] = {"n": len(insts), "success": round(sum(succ[x[0]] for x in insts) / len(insts), 4),
                     "mean_move_lower_bound": round(st.mean(lb), 2), "mean_max_goal_tower": round(st.mean(gmax), 2),
                     "mean_goal_towers": round(st.mean(gcount), 2), "mean_max_init_tower": round(st.mean(imax), 2),
                     "success_by_max_goal_tower": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())}}
    n_h = sum(len(v) for v in bins["hard"].values())
    std = sum(len(bins["hard"][k]) * sum(bins["mirror"][k]) / len(bins["mirror"][k]) for k in bins["hard"]) / n_h
    out["mirror_standardized_to_hard_goal_tower_mix"] = round(std, 4)
    out["gap_pp_raw"] = round(100 * (out["hard"]["success"] - out["mirror"]["success"]), 1)
    out["gap_pp_standardized"] = round(100 * (out["hard"]["success"] - std), 1)
    (ROOT / "runs/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
