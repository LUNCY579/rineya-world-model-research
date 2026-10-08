"""Probe-verified rollout (PVR): use a consequence probe on the frozen model's own answer-slot
state to choose among sampled next-state answers.

Verifier = independent logistic probes (L2, LBFGS) for the binary next-state features plus
Gaussian ridge heads for continuous ones, trained on ONE-STEP items only (no multi-step labels).
A candidate next state is scored by its log-likelihood under the verifier's posterior.
"""
from __future__ import annotations

import numpy as np
import torch

from . import data, env, heads


VERIFY_SPEC = {"prerequisite_plan": (8, 0), "ledger_replay": (12, 4), "register_machine": (9, 3)}


def next_phi(fam: str, before, state) -> np.ndarray:
    """Verifier features of a (candidate) next state relative to `before`: binary block first.
      P: 8 bits
      L: sign_neg x4, changed x4, parity x4 | clip(value/100, ±3) x4
      R: hi>48 x3, parity x3, changed x3    | value/97 x3
    Parity is included because Titan's one-step errors are mostly a single wrong value."""
    if fam == "prerequisite_plan":
        return np.array(env.bits(state), dtype=np.float32)
    if fam == "ledger_replay":
        v = [state[a] for a in env.ACCOUNTS]
        return np.array([int(x < 0) for x in v] + [int(state[a] != before[a]) for a in env.ACCOUNTS]
                        + [x % 2 for x in v] + [float(np.clip(x / 100.0, -3, 3)) for x in v], dtype=np.float32)
    if fam == "register_machine":
        v = [state[r] for r in env.REGISTERS]
        return np.array([int(x > 48) for x in v] + [x % 2 for x in v] + [int(state[r] != before[r]) for r in env.REGISTERS]
                        + [x / 97.0 for x in v], dtype=np.float32)
    raise ValueError(fam)


class LogisticProbe:
    """Multi-output independent L2 logistic regression; C chosen on DEV log-loss."""
    CS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

    def fit(self, Xtr, Ytr, Xdv, Ydv):
        self.std = heads.Standardizer(Xtr)
        A = torch.tensor(self.std(Xtr), dtype=torch.float64)
        Ad = torch.tensor(self.std(Xdv), dtype=torch.float64)
        Y = torch.tensor(Ytr, dtype=torch.float64)
        Yd = torch.tensor(Ydv, dtype=torch.float64)
        best = None
        for C in self.CS:
            W = torch.zeros(A.shape[1], Y.shape[1], dtype=torch.float64, requires_grad=True)
            b = torch.zeros(Y.shape[1], dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ W + b, Y) \
                    + (W ** 2).sum() / (2 * C * A.shape[0])
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                dl = float(torch.nn.functional.binary_cross_entropy_with_logits(Ad @ W + b, Yd))
            if best is None or dl < best[0]:
                best = (dl, C, W.detach().clone(), b.detach().clone())
        self.dev_logloss, self.C, self.W, self.b = best
        return self

    def proba(self, X):
        A = torch.tensor(self.std(X), dtype=torch.float64)
        return torch.sigmoid(A @ self.W + self.b).numpy()


class Verifier:
    """Binary features: LogisticProbe. Continuous: ridge mean + DEV residual sigma (Gaussian)."""

    def __init__(self, fam: str):
        self.fam = fam
        self.nbin = VERIFY_SPEC[fam][0]

    def fit(self, Xtr, Ptr, Xdv, Pdv):
        nb = self.nbin
        self.lp = LogisticProbe().fit(Xtr, Ptr[:, :nb], Xdv, Pdv[:, :nb])
        self.cont = Ptr.shape[1] > nb
        if self.cont:
            self.rg = heads.Ridge().fit(Xtr, Ptr[:, nb:], Xdv, Pdv[:, nb:], 0)
            res = Pdv[:, nb:] - self.rg.predict(Xdv)
            self.sigma = np.maximum(res.std(0), 0.05)
        return self

    def posterior(self, X):
        out = {"p": np.clip(self.lp.proba(X), 1e-4, 1 - 1e-4)}
        if self.cont:
            out["mu"] = self.rg.predict(X)
        return out

    def loglik(self, post_row: dict, phi: np.ndarray, i: int = 0) -> float:
        nb = self.nbin
        p, y = post_row["p"][i], phi[:nb]
        ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
        if self.cont:
            z = (phi[nb:] - post_row["mu"][i]) / self.sigma
            ll += float(-0.5 * (z ** 2).sum())
        return ll

    def argmax_state_bits(self, post_row: dict, i: int = 0) -> np.ndarray:
        return (post_row["p"][i] > 0.5).astype(int)


def p_state_from_bits(b: np.ndarray) -> int:
    return int(sum(int(v) << j for j, v in enumerate(b)))
