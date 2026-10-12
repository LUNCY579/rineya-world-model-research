"""Paired artifact-level statistics and crossed parent/seed bootstrap.

Repeated seeds on a common bank must never be pooled as independent instances.
The bootstrap aggregates trajectories within parent/seed cells, then resamples
parent IDs and seeds independently. Small seed counts remain a limitation.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Iterable


def paired_mcnemar(baseline: Iterable[bool], candidate: Iterable[bool]) -> dict:
    baseline, candidate = list(baseline), list(candidate)
    if not baseline or len(baseline) != len(candidate) or any(type(x) is not bool for x in baseline + candidate):
        raise ValueError("nonempty equal-length boolean pairs required")
    gains = sum(not a and b for a, b in zip(baseline, candidate))
    losses = sum(a and not b for a, b in zip(baseline, candidate))
    discordant = gains + losses
    def cdf(k):
        return sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return {"n": len(baseline), "baseline_success": sum(baseline), "candidate_success": sum(candidate),
            "gains": gains, "losses": losses, "discordant": discordant,
            "difference": (gains - losses) / len(baseline),
            "p_two_sided": min(1.0, 2 * cdf(min(gains, losses))) if discordant else 1.0,
            "p_one_sided_improvement": cdf(losses) if discordant else 1.0}


mcnemar_exact = paired_mcnemar


def wilson_interval(successes: int, n: int, confidence: float = .95) -> tuple[float, float]:
    if type(successes) is not int or type(n) is not int or n <= 0 or not 0 <= successes <= n:
        raise ValueError("valid success count and positive n required")
    if not 0 < confidence < 1:
        raise ValueError("invalid confidence")
    z, p = NormalDist().inv_cdf((1 + confidence) / 2), successes / n
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    half = z / divisor * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0., center - half), min(1., center + half)


def holm(pvalues: dict[str, float], alpha: float = .05) -> dict:
    if not pvalues or not 0 < alpha < 1 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("valid nonempty p-value family required")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    previous, output = 0., {}
    for index, key in enumerate(ordered):
        adjusted = min(1., max(previous, (len(ordered) - index) * pvalues[key]))
        output[key] = {"p": pvalues[key], "adjusted_p": adjusted, "reject": adjusted <= alpha}
        previous = adjusted
    return output


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def clustered_bootstrap(rows: list[dict], *, repetitions: int = 2000, random_seed: int = 0,
                        confidence: float = .95, baseline_field: str = "baseline",
                        candidate_field: str = "candidate") -> dict:
    if not rows or type(repetitions) is not int or repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("rows, >=100 replicates and valid confidence required")
    cells: dict[tuple, list[float]] = {}
    for row in rows:
        parent, seed = row["parent_world_id"], row["seed"]
        a, b = row[baseline_field], row[candidate_field]
        if (not isinstance(parent, str) or not parent or type(seed) is not int
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (a, b))):
            raise ValueError("finite paired measures and parent/seed identities required")
        cells.setdefault((seed, parent), []).append(b - a)
    seeds = sorted({seed for seed, _ in cells})
    parents = sorted({parent for _, parent in cells})
    if len(cells) != len(seeds) * len(parents):
        raise ValueError("complete crossed seed/parent panel required; missing cells cannot be silently imputed")
    means = {key: sum(values) / len(values) for key, values in cells.items()}
    estimate = sum(means.values()) / len(means)
    rng, samples = random.Random(random_seed), []
    for _ in range(repetitions):
        sampled_seeds = rng.choices(seeds, k=len(seeds))
        sampled_parents = rng.choices(parents, k=len(parents))
        samples.append(sum(means[seed, parent] for seed in sampled_seeds for parent in sampled_parents) / len(means))
    samples.sort()
    tail = (1 - confidence) / 2
    return {"difference": estimate, "ci": [_quantile(samples, tail), _quantile(samples, 1 - tail)],
            "confidence": confidence, "parent_count": len(parents), "seed_count": len(seeds),
            "trajectory_count": len(rows), "repetitions": repetitions, "random_seed": random_seed,
            "unit": "equal-weight parent/seed cell means; crossed resampling",
            "limitations": "fewer than five seeds; seed uncertainty is weakly estimated" if len(seeds) < 5 else None}
