"""Generated hard-style Blocksworld instances (never PlanBench): 6-15 blocks, goal = every on-pair of a tall-tower
target (as in PlanBench Blocksworld-hard). Two disjoint sets from different seeds:
  runs/BW_val_large/instances_val_large.json    planner selection for Track B2 (K, width, expansion cap)
  runs/BW_val_large/instances_mirror_hard.json  contamination mirror of the 110 hard instances (same block counts)
Exclusion from the hard set: exact renaming signatures for <= 7 blocks; for larger instances a conservative invariant
(block count, sorted init tower heights, sorted goal tower heights) — any match is dropped.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402


def towers_of(s: frozenset, objs) -> list[int]:
    on = {f[1]: f[2] for f in s if f[0] == "on"}
    below = set(on.values())
    heights = []
    for top in (b for b in objs if b not in below):
        h, x = 1, top
        while x in on:
            x = on[x]
            h += 1
        heights.append(h)
    return sorted(heights)


def goal_towers(goal: frozenset, objs) -> list[int]:
    return towers_of(frozenset(f for f in goal if f[0] == "on"), objs)


def invariant(objs, init, goal):
    return (len(objs), tuple(towers_of(init, objs)), tuple(goal_towers(goal, objs)))


def make(rng, sizes, hard_sigs, hard_inv):
    out = []
    for n in sizes:
        while True:
            g = B.generate(rng, n, full_goal_frac=1.0)
            if g is None:
                continue
            objs, init, goal = g
            if n <= 7 and B.signature(objs, init, goal) in hard_sigs:
                continue
            if invariant(objs, init, goal) in hard_inv:
                continue
            out.append({"objs": objs, "init": sorted(map(list, init)), "goal": sorted(map(list, goal)), "n": n})
            break
    return out


def main():
    hard = []
    for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"):
        hard.append(B.parse_problem(p))
    hard_sigs = B.hard_signatures()
    hard_inv = {invariant(*h) for h in hard}
    sizes = [len(h[0]) for h in hard]
    out = ROOT / "runs" / "BW_val_large"
    out.mkdir(parents=True, exist_ok=True)
    val = make(random.Random(20261008), sorted(sizes)[::1][:110] + sorted(sizes)[:10], hard_sigs, hard_inv)
    mirror = make(random.Random(20261009), sorted(sizes), hard_sigs, hard_inv)
    (out / "instances_val_large.json").write_text(json.dumps(val))
    (out / "instances_mirror_hard.json").write_text(json.dumps(mirror))
    print("val_large", len(val), sorted(Counter(r["n"] for r in val).items()))
    print("mirror", len(mirror), sorted(Counter(r["n"] for r in mirror).items()))
    print("goal pairs val_large", sorted(Counter(len(r["goal"]) for r in val).items()))


if __name__ == "__main__":
    main()
