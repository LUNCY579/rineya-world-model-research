"""Goals, utilities, regret, symbolic encodings and parent-level statistics."""
from __future__ import annotations

import numpy as np

from . import env

KINDS_L = ("add", "scale", "transfer")
KINDS_R = ("add", "conditional", "mix", "swap")


def goals(dim: int, n: int, seed: int) -> np.ndarray:
    """Gaussian goal weights over φ, unit-normalised."""
    g = np.random.default_rng(seed).standard_normal((n, dim))
    return (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)


def onehot(i: int, n: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def symbolic(fam: str, s0, events: list[dict], kmax: int = 4) -> np.ndarray:
    """Exact structured encoding of (start state, plan) for the no-LLM learner. Unused event slots
    are zero with a 'present' flag of 0."""
    if fam == "prerequisite_plan":
        x = [float(b) for b in env.bits(s0)]
        for t in range(kmax):
            if t < len(events):
                e = events[t]
                x += [1.0] + [float(b) for f in ("pre_on", "pre_off", "delete", "add") for b in env.bits(e[f])]
            else:
                x += [0.0] * 33
        return np.array(x, dtype=np.float32)
    if fam == "ledger_replay":
        x = [s0[a] / 100.0 for a in env.ACCOUNTS]
        for t in range(kmax):
            if t < len(events):
                e = events[t]
                x += ([1.0] + onehot(KINDS_L.index(e["kind"]), 3) + onehot(env.ACCOUNTS.index(e["account"]), 4)
                      + onehot(env.ACCOUNTS.index(e["other"]), 4) + [e["amount"] / 10.0, float(e["status"] == "posted")])
            else:
                x += [0.0] * 14
        return np.array(x, dtype=np.float32)
    if fam == "register_machine":
        x = [s0[r] / 97.0 for r in env.REGISTERS]
        for t in range(kmax):
            if t < len(events):
                e = events[t]
                x += ([1.0] + onehot(KINDS_R.index(e["kind"]), 4) + onehot(env.REGISTERS.index(e["register"]), 3)
                      + onehot(env.REGISTERS.index(e["other"]), 3) + [e["value"] / 10.0, e["threshold"] / 97.0])
            else:
                x += [0.0] * 13
        return np.array(x, dtype=np.float32)
    raise ValueError(fam)


def goals_in_subspace(dim: int, n: int, r: int, seed: int) -> np.ndarray:
    """Training goals confined to a seeded random r-dimensional subspace of goal space."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((dim, r)))
    g = rng.standard_normal((n, r)) @ basis.T
    return (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)


def regret_table(U_true: np.ndarray, U_pred: np.ndarray, rng: np.random.Generator) -> dict:
    """U_true, U_pred: [P parents, K candidates, G goals]. Ties in U_pred broken at random.

    Per parent: normalised regret = Σ_g (U* − U_sel) / Σ_g (U* − Ū)  (ratio of sums over goals;
    a parent whose candidates are indistinguishable under every goal has a zero denominator and is
    excluded, preregistered). Optimal = share of goals with U_sel >= U* − 1e-9."""
    P, K, G = U_true.shape
    noise = rng.random(U_pred.shape) * 1e-9
    sel = np.argmax(U_pred + noise, axis=1)                                # [P, G]
    u_sel = np.take_along_axis(U_true, sel[:, None, :], axis=1)[:, 0, :]
    u_star = U_true.max(1)
    u_mean = U_true.mean(1)
    num = (u_star - u_sel).sum(1)
    den = (u_star - u_mean).sum(1)
    valid = den > 1e-9
    nreg = np.where(valid, num / np.where(valid, den, 1.0), np.nan)
    opt = np.where(valid, (u_sel >= u_star - 1e-9).mean(1), np.nan)
    return {"nregret": nreg, "optimal": opt}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 17) -> dict:
    """Parent-level paired bootstrap of mean(a - b). a, b: per-parent values (nan-dropped jointly)."""
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def bootstrap_mean(a: np.ndarray, n_boot: int = 10000, seed: int = 17) -> dict:
    a = a[~np.isnan(a)]
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    return {"n": int(len(a)), "mean": float(a.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
