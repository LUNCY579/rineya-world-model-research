"""Fresh RCX Blocksworld instances, BWM exclusion filter, exact-label datasets split by instance (RCX-E07/E08).

Seeds: this work uses only 20261270-20261289 (SEED_RANGE, enforced); BWM uses 20261005-20261013. One seed per purpose
(SEEDS); a purpose never reuses another's seed.

Generator: B.generate(rng, n, full_goal_frac=FULL_GOAL_FRAC) with rng = random.Random(seed), sizes in the given order.
FULL_GOAL_FRAC = 1.0 (CX choice): every goal is the full on-pair configuration of a tall-tower target, i.e. the
PlanBench Blocksworld-hard style that BWM's val_large / mirror / b3fresh / distill sets use and where BWM's 37 hard-110
failures occur (the E08 target regime). BWM's default 0.0 (1-4 random goal pairs) stays available as a parameter and
must be stated in each run's config. Caveat: a full on-pair goal is fixed up to renaming by its multiset of tower
heights, so there are only p(n) - 1 goal classes (2, 4, 6, 10, 14 for n = 3..7; p = integer partitions), and the
number of distinct (init, goal) instances up to renaming is small for n <= 4 (fresh_instances raises when exhausted).

Filters, in order, per draw: generate() returned None (BWM's own rejection) -> BWM exclusion recipe (excl.py) ->
duplicate within this call or `seen` under `unique` ('instance': renaming class of (init, goal); 'goal': renaming class
of the goal alone). Pass one `seen` set across train/val/test calls so no instance (or goal) is shared across sets.

Dataset rows (build_dataset; one per sampled state): inst, n, objs, goal, goal_key, state, h (exact h*, inf = dead end),
dead, opt (sorted optimal BWM actions a: h*(step(s, a)) = h*(s) - 1). Splits are by INSTANCE (split_by_instance), never
by state. When rows cover all (or most) states of each instance, two instances with renaming-equivalent goals yield the
same rows, so split with group_key=goal_key (or generate with unique='goal'). Blocksworld with a consistent goal has no
dead ends (its state graph is strongly connected); build_dataset counts them anyway.
"""
from __future__ import annotations

import math
import random
from collections import Counter

from . import bwm_import as BI
from . import excl as EX
from . import space as SP

SEEDS = {
    "train_instances": 20261270,
    "val_instances": 20261271,
    "test_ood_instances": 20261272,   # E07 structural OOD, 7-12 blocks
    "e08_instances": 20261273,        # E08 search, 8-15 blocks
    "state_sampling": 20261274,
    "split": 20261275,
    "model_init": 20261276,           # model seeds 20261276-20261279 (replicates)
    "tests": 20261280,                # unit tests only (20261280-20261289), never used for reported numbers
}
SEED_RANGE = (20261270, 20261289)
FULL_GOAL_FRAC = 1.0


def goal_key(objs, goal) -> str:
    """Renaming class of a goal. On-pair-only goals: exact via sorted goal chain lengths (VL.goal_towers); otherwise
    B.signature with an empty init (n <= 7) or the raw goal."""
    B, VL = BI.load()
    if all(f[0] == "on" for f in goal):
        tops, bottoms = [f[1] for f in goal], [f[2] for f in goal]
        heights = VL.goal_towers(goal, list(objs))
        if len(set(tops)) == len(tops) and len(set(bottoms)) == len(bottoms) and sum(heights) == len(objs):
            return f"chains:{len(objs)}:" + ",".join(map(str, heights))  # disjoint chains: exact renaming class
    if len(objs) <= 7:
        return "gsig:" + B.signature(list(objs), frozenset(), goal)
    return "graw:" + B.canonical([list(objs), sorted(map(list, goal))])


def instance_key(objs, init, goal, sig: str | None = None) -> str:
    """Renaming class of (init, goal): B.signature for n <= 7; exact raw key above (renamed duplicates at n >= 8 are
    not detected, but are vanishingly rare)."""
    B, _ = BI.load()
    if len(objs) <= 7:
        return "sig:" + (sig or B.signature(list(objs), init, goal))
    return "raw:" + B.canonical([list(objs), sorted(map(list, init)), sorted(map(list, goal))])


def fresh_instances(seed: int, sizes, full_goal_frac: float = FULL_GOAL_FRAC, seen: set | None = None,
                    ex: dict | None = None, prefix: str = "rcx", unique: str = "instance",
                    max_draws_per_item: int = 2000, accept=None, strict: bool = True):
    """Return (instances, stats). instances: dicts(id, n, objs, init, goal, goal_key, seed).

    accept(goal_key) -> bool optionally restricts goal classes (e.g. a class-level split); rejected draws are counted
    as 'rejected_class'. An item that finds no instance within max_draws_per_item raises (strict) or is skipped and
    counted as 'exhausted' (strict=False). Each rejected draw is counted under the first filter it fails, in the order
    generate -> invariant -> class -> goal duplicate -> signature -> instance duplicate (the exclusion decision itself
    is order-independent).
    """
    if not SEED_RANGE[0] <= seed <= SEED_RANGE[1]:
        raise ValueError(f"seed {seed} is outside the RCX-E07/E08 range {SEED_RANGE}")
    if unique not in ("instance", "goal"):
        raise ValueError("unique must be 'instance' or 'goal'")
    B, VL = BI.load()
    ex = ex or EX.load_exclusions()
    seen = set() if seen is None else seen
    rng = random.Random(seed)
    stats = Counter()
    out = []
    for k, n in enumerate(sizes):
        for _ in range(max_draws_per_item):
            g = B.generate(rng, n, full_goal_frac=full_goal_frac)
            stats["draws"] += 1
            if g is None:
                stats["generate_none"] += 1
                continue
            objs, init, goal = g
            if VL.invariant(objs, init, goal) in ex["invariants"]:
                stats["excluded_invariant"] += 1
                continue
            gk = goal_key(objs, goal)  # cheap filters first: B.signature costs ~0.1 s at n = 7
            if accept is not None and not accept(gk):
                stats["rejected_class"] += 1
                continue
            if unique == "goal" and gk in seen:
                stats["duplicate"] += 1
                continue
            sig = B.signature(objs, init, goal) if n <= 7 else None
            if EX.exclusion_reason(objs, init, goal, ex, sig=sig):
                stats["excluded_signature"] += 1
                continue
            key = instance_key(objs, init, goal, sig) if unique == "instance" else gk
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            out.append({"id": f"{prefix}/{seed}/{k}", "n": n, "objs": list(objs), "init": init, "goal": goal,
                        "goal_key": gk, "seed": seed})
            break
        else:
            if strict:
                raise RuntimeError(f"no fresh instance with n={n} after {max_draws_per_item} draws (pool exhausted?)")
            stats["exhausted"] += 1
    stats["kept"] = len(out)
    return out, dict(stats)


def split_by_instance(instances, fractions: dict, seed: int = SEEDS["split"], group_key=None) -> dict:
    """Deterministic instance-level split {name: [instances]}; the last name takes the rest.

    group_key(instance) -> hashable keeps every group in one split (e.g. lambda it: it['goal_key']); groups are
    shuffled and assigned greedily so that split sizes (in instances) approach the fractions.
    """
    groups = {}
    for i, it in enumerate(instances):
        groups.setdefault(group_key(it) if group_key else i, []).append(i)
    order = sorted(groups, key=str)
    random.Random(seed).shuffle(order)
    names = list(fractions)
    out = {name: [] for name in names}
    total, j, filled = len(instances), 0, 0
    for gk in order:
        while j < len(names) - 1 and filled >= round(sum(fractions[x] for x in names[:j + 1]) * total):
            j += 1
        out[names[j]].extend(groups[gk])
        filled += len(groups[gk])
    return {name: [instances[i] for i in sorted(ix)] for name, ix in out.items()}


def build_dataset(instances, states_per_instance: int | None = None, seed: int = SEEDS["state_sampling"],
                  include_init: bool = True, with_opt: bool = True, exclude_rows: bool = True, ex: dict | None = None):
    """Exact-label rows for every instance (n <= 8). states_per_instance None = all states, else a uniform sample.

    exclude_rows: drop rows whose (state, goal), read as an instance, falls under the BWM recipe
    (excl.row_exclusion_reason), so no row is a reserved instance's root with its label (counted in stats).
    Returns (rows, stats). Rows of one instance are contiguous and carry `inst`, so any split stays by instance.
    """
    rng = random.Random(seed)
    ex = (ex or EX.load_exclusions()) if exclude_rows else None
    rows, stats = [], Counter()
    for it in instances:
        L = SP.GoalLabels(it["objs"], it["goal"])
        N = len(L.space)
        if states_per_instance is None or states_per_instance >= N:
            ids = list(range(N))
        else:
            ids = rng.sample(range(N), states_per_instance)
            if include_init:
                i0 = L.sid(it["init"])
                if i0 not in ids:
                    ids[0] = i0
        gk = it.get("goal_key") or goal_key(it["objs"], it["goal"])
        for i in ids:
            s = L.state(i)
            if exclude_rows:
                why = EX.row_exclusion_reason(it["objs"], s, it["goal"], ex)
                if why:
                    stats[f"rows_excluded_{why}"] += 1
                    continue
            h = L.h_id(i)
            rows.append({"inst": it["id"], "n": it["n"], "objs": it["objs"], "goal": it["goal"], "goal_key": gk,
                         "state": s, "h": h, "dead": math.isinf(h), "opt": L.opt_actions_id(i) if with_opt else None})
            stats["dead"] += math.isinf(h)
            stats["rows"] += 1
        stats["instances"] += 1
        stats["dead_states_in_space"] += int((L.dist < 0).sum())
    return rows, dict(stats)
