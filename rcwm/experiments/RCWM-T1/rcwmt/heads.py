"""Readouts on frozen features.

  ridge      closed-form ridge on standardized features; binary targets as a linear probability
             model thresholded at 0.5 (the standard linear probe); λ chosen on DEV
  mlp        consequence head (B3): 1 hidden layer, BCE on binary + Huber on continuous targets;
             LR chosen on DEV, early stopping on DEV loss
  scalar     matched scalar evaluator (B2): MLP on [h; w] -> U, same hidden width, trained on
             train-goal utilities only
  bilinear   strong scalar evaluator (B2b): U = w · (M h + b), trained on utilities only — the
             same functional form as a linear consequence readout, but supervised by scalars

All heads run on CPU (the GPU belongs to capture jobs and the peer session).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import os as _os
torch.set_num_threads(int(_os.environ.get("RCWM_THREADS", "8")))


class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-3

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def _score(P: np.ndarray, Y: np.ndarray, nbin: int) -> float:
    """Higher is better: mean binary accuracy and mean continuous R² (equal weight per target)."""
    parts = []
    if nbin:
        parts.extend(list(((P[:, :nbin] > 0.5) == (Y[:, :nbin] > 0.5)).mean(0)))
    if Y.shape[1] > nbin:
        yc, pc = Y[:, nbin:], P[:, nbin:]
        r2 = 1 - ((yc - pc) ** 2).sum(0) / (((yc - yc.mean(0)) ** 2).sum(0) + 1e-9)
        parts.extend(list(r2))
    return float(np.mean(parts))


class Ridge:
    LAMBDAS = (1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)

    def fit(self, Xtr, Ytr, Xdv, Ydv, nbin: int):
        self.std = Standardizer(Xtr)
        A = torch.tensor(self.std(Xtr), dtype=torch.float64)
        Y = torch.tensor(Ytr, dtype=torch.float64)
        self.ym = Y.mean(0, keepdim=True)
        A1, Yc = A, Y - self.ym
        G = A1.T @ A1
        XtY = A1.T @ Yc
        Ad = torch.tensor(self.std(Xdv), dtype=torch.float64)
        best = None
        eye = torch.eye(G.shape[0], dtype=torch.float64)
        for lam in self.LAMBDAS:
            W = torch.linalg.solve(G + lam * eye, XtY)
            P = (Ad @ W + self.ym).numpy()
            s = _score(P, Ydv, nbin)
            if best is None or s > best[0]:
                best = (s, lam, W)
        self.dev_score, self.lam, self.W = best
        return self

    def predict(self, X):
        return (torch.tensor(self.std(X), dtype=torch.float64) @ self.W + self.ym).numpy()


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def _loss(out, Y, nbin):
    l = 0.0
    if nbin:
        l = l + nn.functional.binary_cross_entropy_with_logits(out[:, :nbin], Y[:, :nbin])
    if Y.shape[1] > nbin:
        l = l + nn.functional.huber_loss(out[:, nbin:], Y[:, nbin:], delta=1.0)
    return l


def _to_pred(out: torch.Tensor, nbin: int) -> np.ndarray:
    o = out.detach().clone()
    if nbin:
        o[:, :nbin] = torch.sigmoid(o[:, :nbin])
    return o.numpy()


class MLPHead:
    LRS = (3e-4, 1e-3, 3e-3)

    def __init__(self, hidden=256, epochs=300, wd=1e-2, seed=17, batch=256):
        self.hidden, self.epochs, self.wd, self.seed, self.batch = hidden, epochs, wd, seed, batch

    def _train(self, Xtr, Ytr, Xdv, Ydv, nbin, lr):
        torch.manual_seed(self.seed)
        m = _MLP(Xtr.shape[1], Ytr.shape[1], self.hidden)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=self.wd)
        best, best_state, bad = None, None, 0
        n = Xtr.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            m.train()
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, self.batch):
                idx = perm[s: s + self.batch]
                opt.zero_grad()
                _loss(m(Xtr[idx]), Ytr[idx], nbin).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                dl = float(_loss(m(Xdv), Ydv, nbin))
            if best is None or dl < best - 1e-5:
                best, bad = dl, 0
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 25:
                    break
        m.load_state_dict(best_state)
        m.eval()
        return m, best

    def fit(self, Xtr, Ytr, Xdv, Ydv, nbin: int):
        self.std = Standardizer(Xtr)
        self.nbin = nbin
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        xtr, ytr, xdv, ydv = T(self.std(Xtr)), T(Ytr), T(self.std(Xdv)), T(Ydv)
        best = None
        for lr in self.LRS:
            m, dl = self._train(xtr, ytr, xdv, ydv, nbin, lr)
            with torch.no_grad():
                s = _score(_to_pred(m(xdv), nbin), Ydv, nbin)
            if best is None or s > best[0]:
                best = (s, lr, m)
        self.dev_score, self.lr, self.m = best
        return self

    def predict(self, X):
        with torch.no_grad():
            return _to_pred(self.m(torch.tensor(self.std(X), dtype=torch.float32)), self.nbin)


class ScalarHead:
    """B2: MLP([h; w]) -> U (kind='mlp') or bilinear U = w·(Mh+b) (kind='bilinear')."""
    LRS = (3e-4, 1e-3, 3e-3)

    def __init__(self, kind="mlp", hidden=256, epochs=200, wd=1e-2, seed=17, batch=512):
        self.kind, self.hidden, self.epochs, self.wd, self.seed, self.batch = kind, hidden, epochs, wd, seed, batch

    def _model(self, d_h, d_w):
        if self.kind == "mlp":
            return nn.Sequential(nn.Linear(d_h + d_w, self.hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, 1))
        return nn.Sequential(nn.Linear(d_h, self.hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.hidden, d_w))

    def _fwd(self, m, h, w):
        if self.kind == "mlp":
            return m(torch.cat([h, w], 1)).squeeze(1)
        return (m(h) * w).sum(1)

    def fit(self, Htr, Wtr, Utr, Hdv, Wdv, Udv):
        """H: [n, d] features, W: [n, |φ|] goal weights, U: [n] utilities (rows = candidate x goal)."""
        self.std = Standardizer(Htr)
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        htr, wtr, utr = T(self.std(Htr)), T(Wtr), T(Utr)
        hdv, wdv, udv = T(self.std(Hdv)), T(Wdv), T(Udv)
        best = None
        for lr in self.LRS:
            torch.manual_seed(self.seed)
            m = self._model(htr.shape[1], wtr.shape[1])
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=self.wd)
            g = torch.Generator().manual_seed(self.seed)
            bl, bs, bad = None, None, 0
            for ep in range(self.epochs):
                m.train()
                perm = torch.randperm(htr.shape[0], generator=g)
                for s in range(0, htr.shape[0], self.batch):
                    i = perm[s: s + self.batch]
                    opt.zero_grad()
                    nn.functional.huber_loss(self._fwd(m, htr[i], wtr[i]), utr[i], delta=1.0).backward()
                    opt.step()
                m.eval()
                with torch.no_grad():
                    dl = float(nn.functional.huber_loss(self._fwd(m, hdv, wdv), udv, delta=1.0))
                if bl is None or dl < bl - 1e-5:
                    bl, bad, bs = dl, 0, {k: v.clone() for k, v in m.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 20:
                        break
            m.load_state_dict(bs)
            m.eval()
            if best is None or bl < best[0]:
                best = (bl, lr, m)
        self.dev_loss, self.lr, self.m = best
        return self

    def predict(self, H, W):
        with torch.no_grad():
            T = lambda a: torch.tensor(a, dtype=torch.float32)
            return self._fwd(self.m, T(self.std(H)), T(W)).numpy()
