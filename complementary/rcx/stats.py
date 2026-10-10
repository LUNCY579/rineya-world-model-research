"""Statistics for parent-clustered paired comparisons (RCX).

Data convention: an arm is a mapping parent_id -> list of per-variant scores (a bare number counts as one
variant). Parents are the independent units; variants are averaged within a parent first (parent macro) and
parents get equal weight. Differences are always b - a (candidate minus baseline); the sign follows the score,
so positive is not necessarily better for regret or cost.
"""
from __future__ import annotations

import math
import numbers
import random
from typing import Hashable, Mapping, Sequence, TypeVar

K = TypeVar('K', bound=Hashable)
Arm = Mapping[Hashable, Sequence[float]]


# Provenance: mcnemar_exact and _quantile reproduce the logic of mcnemar_exact and quantile in
# rineya_complementary_research_2026-10-09/08_reference_tools/paired_metrics.py (bundle RCI-001 v1.0,
# read-only reference tool), copied rather than imported so RCX never imports from outside complementary/.
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p on discordant counts b (a-only) and c (b-only); no continuity correction."""
    if type(b) is not int or type(c) is not int or min(b, c) < 0:
        raise ValueError('Discordant counts must be nonnegative integers')
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if k == n // 2:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    maximum = max(logs)
    log_tail = maximum + math.log(sum(math.exp(x - maximum) for x in logs))
    return min(1.0, math.exp(math.log(2) + log_tail))


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - index) + values[upper] * (index - lower) if upper != lower else values[lower]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials; (0, 1) when n == 0."""
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f'need 0 <= k <= n, got k={k}, n={n}')
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, centre - half)), (1.0 if k == n else min(1.0, centre + half))


def holm(pvals: Mapping[K, float]) -> dict[K, float]:
    """Holm step-down adjusted p-values for a family {name: p}; keys and order are preserved."""
    for key, p in pvals.items():
        if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
            raise ValueError(f'p-value for {key!r} must be in [0, 1], got {p!r}')
    m, running, adjusted = len(pvals), 0.0, {}
    for i, (key, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return {key: adjusted[key] for key in pvals}


def _values(parent: Hashable, raw) -> list[float]:
    if isinstance(raw, (str, bytes)):
        raise ValueError(f'parent {parent!r}: scores must be numbers or a list of numbers, got {raw!r}')
    values = [float(raw)] if isinstance(raw, numbers.Real) else [float(x) for x in raw]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError(f'parent {parent!r}: scores must be a non-empty list of finite numbers')
    return values


def _parent_means(arm: Arm) -> dict[Hashable, float]:
    if not arm:
        raise ValueError('no parents supplied')
    return {parent: sum(v) / len(v) for parent, v in ((p, _values(p, raw)) for p, raw in arm.items())}


def _order(keys) -> list:
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=repr)


def _same_parents(*arms: Arm) -> None:
    ref = set(arms[0])
    for arm in arms[1:]:
        if set(arm) != ref:
            only_a, only_b = _order(ref - set(arm))[:5], _order(set(arm) - ref)[:5]
            raise ValueError(f'parent sets differ: only in first {only_a}, only in other {only_b}')


def _check_boot(n_parents: int, resamples: int, alpha: float) -> None:
    if n_parents < 2:
        raise ValueError('At least two independent parents are required')
    if not 100 <= resamples <= 100000:
        raise ValueError('resamples must be between 100 and 100000')
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must be in (0, 1)')


def _boot_means(values: list[float], resamples: int, seed: int) -> list[float]:
    rng, n = random.Random(seed), len(values)
    return [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]


def _interval(boot: list[float], alpha: float) -> tuple[float, float]:
    return _quantile(boot, alpha / 2), _quantile(boot, 1 - alpha / 2)


def parent_macro(arm: Arm) -> float:
    """Equal-weight mean over parents of each parent's mean over variants."""
    means = _parent_means(arm)
    return sum(means[p] for p in _order(means)) / len(means)


def cluster_bootstrap_mean(a: Arm, resamples: int = 5000, seed: int = 17, alpha: float = 0.05) -> dict:
    """Parent-cluster percentile bootstrap CI of one arm's parent macro: dict(mean, lo, hi, n_parents)."""
    means = _parent_means(a)
    _check_boot(len(means), resamples, alpha)
    values = [means[p] for p in _order(means)]
    lo, hi = _interval(_boot_means(values, resamples, seed), alpha)
    return dict(mean=sum(values) / len(values), lo=lo, hi=hi, n_parents=len(values))


def cluster_bootstrap_diff(a: Arm, b: Arm, resamples: int = 5000, seed: int = 17, alpha: float = 0.05) -> dict:
    """Paired parent-cluster bootstrap of macro(b) - macro(a): dict(mean_a, mean_b, diff, lo, hi, n_parents).

    Variants are averaged within parent, then parents are resampled with replacement (the same parents for
    both arms). With the default seed and alpha it reproduces the bundle's percentile_interval_95.
    """
    _same_parents(a, b)
    ma, mb = _parent_means(a), _parent_means(b)
    keys = _order(ma)
    _check_boot(len(keys), resamples, alpha)
    diffs = [mb[k] - ma[k] for k in keys]
    lo, hi = _interval(_boot_means(diffs, resamples, seed), alpha)
    n = len(keys)
    return dict(mean_a=sum(ma[k] for k in keys) / n, mean_b=sum(mb[k] for k in keys) / n,
                diff=sum(diffs) / n, lo=lo, hi=hi, n_parents=n)


def _seed_matrix(per_seed: Sequence[Arm], keys: list) -> list[list[float]]:
    if not per_seed:
        raise ValueError('per-seed list is empty')
    rows = []
    for arm in per_seed:
        means = _parent_means(arm)
        rows.append([means[k] for k in keys])
    return rows


def seed_parent_bootstrap(per_seed_a: Sequence[Arm], per_seed_b: Sequence[Arm] | None = None,
                          resamples: int = 5000, seed: int = 17, alpha: float = 0.05,
                          resample_seeds: bool = False) -> dict:
    """Multi-seed parent bootstrap. Each per_seed_* is a list (one entry per training seed) of arms.

    Within a parent, variants are averaged per seed, then seeds are averaged; parents are resampled with
    replacement. With only per_seed_a: dict(mean, lo, hi, n_parents, n_seeds). With both arms (paired on
    parents, seed counts may differ): dict(mean_a, mean_b, diff, lo, hi, n_parents, n_seeds_a, n_seeds_b) for
    b - a. resample_seeds=True also resamples each arm's seeds once per replicate (two-level bootstrap).
    """
    arms = list(per_seed_a) + list(per_seed_b or [])
    if not per_seed_a or (per_seed_b is not None and not per_seed_b):
        raise ValueError('each arm needs at least one seed')
    _same_parents(*arms)
    keys = _order(per_seed_a[0])
    _check_boot(len(keys), resamples, alpha)
    n = len(keys)
    mat_a = _seed_matrix(per_seed_a, keys)
    mat_b = _seed_matrix(per_seed_b, keys) if per_seed_b is not None else None

    def seed_avg(mat: list[list[float]], seeds: Sequence[int]) -> list[float]:
        return [sum(mat[s][j] for s in seeds) / len(seeds) for j in range(n)]

    pa = seed_avg(mat_a, range(len(mat_a)))
    pb = seed_avg(mat_b, range(len(mat_b))) if mat_b is not None else None
    point = [y - x for x, y in zip(pa, pb)] if pb is not None else pa
    if not resample_seeds:
        boot = _boot_means(point, resamples, seed)
    else:
        rng, boot = random.Random(seed), []
        for _ in range(resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            ra = seed_avg(mat_a, [rng.randrange(len(mat_a)) for _ in mat_a])
            if mat_b is None:
                boot.append(sum(ra[j] for j in idx) / n)
            else:
                rb = seed_avg(mat_b, [rng.randrange(len(mat_b)) for _ in mat_b])
                boot.append(sum(rb[j] - ra[j] for j in idx) / n)
    lo, hi = _interval(boot, alpha)
    if pb is None:
        return dict(mean=sum(pa) / n, lo=lo, hi=hi, n_parents=n, n_seeds=len(mat_a))
    return dict(mean_a=sum(pa) / n, mean_b=sum(pb) / n, diff=sum(point) / n, lo=lo, hi=hi,
                n_parents=n, n_seeds_a=len(mat_a), n_seeds_b=len(mat_b))


def paired_binary_table(a01: Arm, b01: Arm, threshold: float = 0.5) -> tuple[int, int]:
    """Discordant parent counts (b, c) for mcnemar_exact: b = only arm a correct, c = only arm b correct.

    A parent is correct when the mean of its variant scores is >= threshold. Parents must match.
    """
    _same_parents(a01, b01)
    ma, mb = _parent_means(a01), _parent_means(b01)
    ca = {k: ma[k] >= threshold for k in ma}
    cb = {k: mb[k] >= threshold for k in mb}
    return sum(ca[k] and not cb[k] for k in ca), sum(cb[k] and not ca[k] for k in ca)
