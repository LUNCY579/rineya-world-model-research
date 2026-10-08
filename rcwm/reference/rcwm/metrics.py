"""Small evaluation primitives; final scientific analysis still needs a contract."""
from __future__ import annotations
import numpy as np


def candidate_regret(actual_utilities, selected: int) -> float:
    x = np.asarray(actual_utilities, float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty candidate utility vector required")
    if not isinstance(selected, (int, np.integer)) or not 0 <= selected < len(x):
        raise ValueError("selected index out of range")
    return float(x.max() - x[selected])


def brier_score(probabilities, targets) -> float:
    p, y = np.asarray(probabilities, float), np.asarray(targets, float)
    if p.shape != y.shape or p.size == 0 or not np.isfinite(p).all():
        raise ValueError("matching finite nonempty arrays required")
    if ((p < 0) | (p > 1)).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("probabilities in [0,1] and binary targets required")
    return float(np.mean((p-y)**2))


def paired_parent_bootstrap(parent_ids, baseline, candidate, *, seed=0,
                            draws=2000, confidence=.95) -> dict:
    """Equal-weight parent means; percentile CI conditional on this trained model.

    Not a training-seed CI, sequential test, or a noninferiority certification.
    """
    p = np.asarray(parent_ids, str)
    a, b = np.asarray(baseline, float), np.asarray(candidate, float)
    if p.ndim != 1 or p.shape != a.shape or a.shape != b.shape or not len(p):
        raise ValueError("matching nonempty 1D arrays required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite scores required")
    if not isinstance(draws, int) or draws < 100 or not 0 < confidence < 1:
        raise ValueError("draws >=100 and confidence in (0,1) required")
    ids = np.unique(p)
    if len(ids) < 2:
        raise ValueError("at least two independent parents required")
    d = np.array([(b[p==i]-a[p==i]).mean() for i in ids])
    rng = np.random.default_rng(seed)
    # Streaming keeps the helper safe for larger parent sets.
    means = np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(draws)])
    alpha = (1-confidence)/2
    lo, hi = np.quantile(means,[alpha,1-alpha])
    return {"difference":float(d.mean()),"ci_low":float(lo),"ci_high":float(hi),
            "n_parents":len(ids),"draws":draws,"confidence":confidence,
            "scope":"conditional_on_fixed_models"}
