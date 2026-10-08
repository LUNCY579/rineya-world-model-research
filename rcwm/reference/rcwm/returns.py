"""Finite-horizon consequence accounting and explicit linear utility."""
from __future__ import annotations
import numpy as np


def discounted_features(features, gamma: float = 1.0, *, durations=None,
                        terminated=None, truncated=None) -> np.ndarray:
    """Return discounted observed features through the first terminal/truncated event.

    Features of that final event are included. Truncation stops the observed window;
    this function does NOT infer that continuation value is zero at truncation.
    With durations, the discount exponent is cumulative elapsed time BEFORE a step.
    Without durations, each action consumes one discount unit.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be nonempty [time, feature]")
    if not np.isfinite(x).all() or not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("finite features and gamma in [0,1] required")
    n = len(x)
    dt = np.ones(n) if durations is None else np.asarray(durations, dtype=float)
    if dt.shape != (n,) or not np.isfinite(dt).all() or (dt < 0).any():
        raise ValueError("durations must be finite nonnegative [time]")
    stop = np.zeros(n, dtype=bool)
    for flags in (terminated, truncated):
        if flags is not None:
            f = np.asarray(flags)
            if f.shape != (n,) or f.dtype != np.bool_:
                raise ValueError("termination flags must be Boolean [time]")
            stop |= f
    if stop.any():
        end = np.flatnonzero(stop)[0] + 1
        x, dt = x[:end], dt[:end]
    elapsed = np.concatenate(([0.0], np.cumsum(dt[:-1])))
    return (x * np.power(gamma, elapsed)[:, None]).sum(axis=0)


def linear_utility(features, weights) -> np.ndarray:
    x, w = np.asarray(features, float), np.asarray(weights, float)
    if x.ndim < 1 or w.ndim != 1 or x.shape[-1] != len(w):
        raise ValueError("last feature dimension must match one weight vector")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("features and weights must be finite")
    return x @ w
