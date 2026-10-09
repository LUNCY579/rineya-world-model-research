"""Blocksworld training data from OUR generator only (PlanBench instances are test-only and excluded up to renaming).

Writes runs/BW_data/:
  forward_train.jsonl / forward_val.jsonl  : goal-blind transitions (state, action) -> {legal, state}; ~50% illegal
  proposer_actions_train.jsonl / _val      : outcome-conditioned plans (action-only targets)
  proposer_states_train.jsonl / _val       : same rows, state-and-action targets
  instances_val.json                       : generated validation instances (planner selection; never PlanBench)
Block names are sampled from a..o so models cannot rely on PlanBench's a..e naming; 3-5 blocks (plus 6-block rows
for size robustness). Proposer rows: every suffix of a BFS-optimal plan (success), plus noisy-plan suffixes with their
achieved outcome (failure), as in RW2.
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
from bwmn import blocks as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-instances", type=int, default=6000)
    ap.add_argument("--val-instances", type=int, default=400)
    ap.add_argument("--forward-rows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out", default="runs/BW_data")
    ap.add_argument("--full-goal-frac", type=float, default=0.0,
                    help="v2: fraction of instances whose goal is the whole target configuration (hard-set goal style)")
    ap.add_argument("--proposer-only", action="store_true", help="v2: skip forward rows (BW_F keeps its v1 data)")
    ap.add_argument("--exclude-hard", action="store_true", help="v2: also exclude Blocksworld-hard signatures (<=7 blocks)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    test_sigs = B.test_signatures() | (B.hard_signatures() if a.exclude_hard else set())
    seen = set()
    insts = []
    while len(insts) < a.train_instances + a.val_instances:
        n = rng.choice([3, 4, 4, 5, 5, 5, 6])
        g = B.generate(rng, n, full_goal_frac=a.full_goal_frac)
        if g is None:
            continue
        sig = B.signature(*g) if n <= 6 else None
        if sig in test_sigs:
            continue
        objs, init, goal = g
        plan = B.bfs_plan(init, goal, objs)
        if plan is None or len(plan) == 0:
            continue
        insts.append((objs, init, goal, plan, sig))
    rng.shuffle(insts)
    # split by renaming-signature so isomorphic instances never straddle train/val
    val_sigs = set()
    train, val = [], []
    for it in insts:
        if len(val) < a.val_instances and (it[4] in val_sigs or rng.random() < 0.08):
            val.append(it)
            val_sigs.add(it[4])
        elif it[4] not in val_sigs:
            train.append(it)

    leak_dropped = [0]

    def proposer_rows(items):
        rows = []
        for objs, init, goal, plan, _ in items:
            # optimal-plan suffixes (success)
            s = init
            states = [init]
            for act in plan:
                s, _ = B.step(s, act)
                states.append(s)
            for t in range(len(plan)):
                # Suffix leakage guard (review of 2026-10-06): an intermediate (state, goal) can be a renaming of a
                # PlanBench test problem even when the instance's own (init, goal) is not. Drop such rows.
                if t > 0 and B.signature(objs, states[t], goal) in test_sigs:
                    leak_dropped[0] += 1
                    continue
                rows.append({"state": states[t], "goal": goal, "outcome": "success", "plan": plan[t:],
                             "next": states[t + 1:], "objs": objs})
            # one noisy plan per instance: replace one action by a random ground action; record achieved outcome
            noisy = list(plan)
            i = rng.randrange(len(noisy))
            noisy[i] = rng.choice(B.ground_actions(objs))
            s = init
            seq = []
            for act in noisy:
                s, _ = B.step(s, act)
                seq.append(s)
            ok = B.goal_met(s, goal) and B.execute(init, goal, noisy)["valid"]
            rows.append({"state": init, "goal": goal, "outcome": "success" if ok else "failure", "plan": noisy,
                         "next": seq, "objs": objs})
        return rows

    def forward_rows(items, n):
        rows = []
        keys = set()
        acts_cache = {}
        while len(rows) < n:
            objs, init, goal, plan, _ = items[rng.randrange(len(items))]
            # states along the optimal plan and a short random walk
            s = init
            for _ in range(rng.randrange(0, 8)):
                legal = [x for x in B.ground_actions(objs) if B.applicable(s, x)]
                s, _ = B.step(s, rng.choice(legal))
            ga = acts_cache.setdefault(tuple(objs), B.ground_actions(objs))
            legal = [x for x in ga if B.applicable(s, x)]
            illegal = [x for x in ga if not B.applicable(s, x)]
            act = rng.choice(legal) if (rng.random() < 0.5 or not illegal) else rng.choice(illegal)
            key = (s, act)
            if key in keys:
                continue
            keys.add(key)
            ns, ok = B.step(s, act)
            rows.append({"prompt": B.forward_prompt(s, act), "target": B.forward_target(ns, ok), "legal": ok,
                         "n_blocks": len(objs)})
        return rows

    def dump(name, rows):
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    for split, items in (("train", train), ("val", val)):
        pr = proposer_rows(items)
        dump(f"proposer_actions_{split}.jsonl",
             [{"prompt": B.proposer_prompt(r["state"], r["goal"], r["outcome"]), "target": B.target_actions(r["plan"]),
               "outcome": r["outcome"]} for r in pr])
        dump(f"proposer_states_{split}.jsonl",
             [{"prompt": B.proposer_prompt(r["state"], r["goal"], r["outcome"]),
               "target": B.target_states(list(zip(r["plan"], r["next"]))), "outcome": r["outcome"]} for r in pr])
        if not a.proposer_only:
            dump(f"forward_{split}.jsonl", forward_rows(items, a.forward_rows if split == "train" else 1500))
        print(split, "instances", len(items), "proposer rows", len(pr), Counter(r["outcome"] for r in pr),
              "blocks", Counter(len(it[0]) for it in items))
    (out / "instances_val.json").write_text(json.dumps(
        [{"objs": o, "init": sorted(map(list, i)), "goal": sorted(map(list, g)), "opt": len(p)} for o, i, g, p, _ in val]))
    print("suffix rows dropped by the test-signature guard:", leak_dropped[0])
    (out / "META.json").write_text(json.dumps({"args": vars(a), "train": len(train), "val": len(val),
                                               "test_signatures_excluded": len(test_sigs),
                                               "suffix_rows_dropped_test_signature": leak_dropped[0]}, indent=1))


if __name__ == "__main__":
    main()
