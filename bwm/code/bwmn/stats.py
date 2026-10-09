"""Paired statistics over task parents and training seeds.

- Unit of analysis: the task parent (one task per family per split). Seeds are a second, crossed source of variation.
- crossed_bootstrap: resample parents AND seeds with replacement (independently), recompute the mean paired
  difference. This scopes the interval to both task sampling and training variation (ENG-12).
"""
from __future__ import annotations

import math
import random
from statistics import mean


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_table(a: list[int], b: list[int]) -> dict:
    assert len(a) == len(b)
    both = sum(x and y for x, y in zip(a, b))
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    return {"n": len(a), "both": both, "only_a": only_a, "only_b": only_b,
            "rate_a": sum(a) / len(a), "rate_b": sum(b) / len(b), "diff": (sum(a) - sum(b)) / len(a),
            "mcnemar_p": mcnemar_exact(only_a, only_b)}


def crossed_bootstrap(diff_by_seed: dict[int, list[float]], B: int = 4000, seed: int = 12345,
                      alpha: float = 0.05) -> dict:
    """diff_by_seed[s][i] = per-parent paired difference (arm X minus arm Y) for seed s, parent i.
    All seeds must share the same parent list. Returns mean and percentile CI."""
    seeds = sorted(diff_by_seed)
    n = len(diff_by_seed[seeds[0]])
    assert all(len(diff_by_seed[s]) == n for s in seeds)
    point = mean(mean(diff_by_seed[s]) for s in seeds)
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        ss = [rng.choice(seeds) for _ in seeds]
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(mean(mean(diff_by_seed[s][i] for i in idx) for s in ss))
    stats.sort()
    lo = stats[int(alpha / 2 * B)]
    hi = stats[int((1 - alpha / 2) * B) - 1]
    # parent-only bootstrap (conditional on the fitted seeds)
    pst = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        pst.append(mean(mean(diff_by_seed[s][i] for i in idx) for s in seeds))
    pst.sort()
    return {"mean": point, "ci_crossed": [lo, hi], "ci_parent_only": [pst[int(alpha / 2 * B)], pst[int((1 - alpha / 2) * B) - 1]],
            "per_seed": {s: mean(diff_by_seed[s]) for s in seeds}, "n_parents": n, "n_seeds": len(seeds), "B": B}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def crossed_cluster_bootstrap(diff_by_seed: dict, clusters: list, B: int = 4000, seed: int = 2026,
                              alpha: float = 0.05) -> dict:
    """Resample seeds and CLUSTERS (e.g. held-out goal values) with replacement, independently.

    diff_by_seed[s][i] is the paired difference for parent i under training seed s; clusters[i] is parent i's
    cluster label. The statistic is the mean over resampled seeds of the mean over parents in resampled clusters."""
    seeds = sorted(diff_by_seed)
    n = len(clusters)
    groups: dict = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    labels = sorted(groups)
    point = mean(mean(diff_by_seed[s]) for s in seeds)
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        ss = [rng.choice(seeds) for _ in seeds]
        idx = [i for _ in labels for i in groups[rng.choice(labels)]]
        stats.append(mean(mean(diff_by_seed[s][i] for i in idx) for s in ss))
    stats.sort()
    lo, hi = stats[int(alpha / 2 * B)], stats[int((1 - alpha / 2) * B) - 1]
    per_seed = [mean(diff_by_seed[s]) for s in seeds]
    k = len(per_seed)
    t_ci = None
    if k >= 2:
        m = mean(per_seed)
        sd = math.sqrt(sum((x - m) ** 2 for x in per_seed) / (k - 1))
        tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}.get(k - 1, 2.0)
        t_ci = [m - tq * sd / math.sqrt(k), m + tq * sd / math.sqrt(k)]
    return {"mean": point, "ci_crossed_cluster": [lo, hi], "seed_t_ci": t_ci, "per_seed": dict(zip(seeds, per_seed)),
            "n_parents": n, "n_clusters": len(labels), "n_seeds": len(seeds), "B": B}
