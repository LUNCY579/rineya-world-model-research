"""Verification helpers (scratch; read-only use of rcwmt). CPU only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rcwmt import controls, data, env, fidelity, heads, decide  # noqa: E402,F401

FAMS = env.FAMILIES
LAYERS = (4, 8, 12, 16, 20, 24, 28)


_ROWS = None


def rows_all():
    global _ROWS
    if _ROWS is None:
        _ROWS = data.candidate_table(data.load_bank("bank_pilot"))
    return _ROWS


def split_rows(fam, frame, split):
    """frame: 'one' (Titan one-step frame, k=1 transitions), or int k (multi format)."""
    rs = rows_all()
    if frame == "one":
        return [dict(r, item_id=r["cand_id"] + "-one") for r in rs
                if r["family"] == fam and r["k"] == 1 and r["split"] == split]
    return [r for r in rs if r["family"] == fam and r["k"] == frame and r["split"] == split]


def sp3(fam, frame):
    return {s: split_rows(fam, frame, s) for s in ("train", "dev", "test")}


_ARMS = {}


def arm(name):
    if name not in _ARMS:
        _ARMS[name] = data.Arm("bank_pilot", name)
    return _ARMS[name]


def ridge_fit(X, Y, Xd, Yd, nbin):
    return heads.Ridge().fit(X, Y, Xd, Yd, nbin)


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    return float(np.mean(1 - ((y - p) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)))


def probe(armname, sp, target_fn, nbin, layer=None, layers=LAYERS, sub=None):
    """Fit ridge on TRAIN (λ on DEV, layer on DEV unless fixed); return dict with test preds.
    sub: optional predicate on rows restricting all splits."""
    A = arm(armname)
    spx = {s: [r for r in x if (sub is None or sub(r))] for s, x in sp.items()}
    Y = {s: np.stack([np.atleast_1d(target_fn(r)).astype(np.float32) for r in x]) for s, x in spx.items()}
    ids = {s: [r["item_id"] for r in x] for s, x in spx.items()}
    best = None
    for L in ([layer] if layer is not None else layers):
        rg = ridge_fit(A.X_last(ids["train"], L), Y["train"], A.X_last(ids["dev"], L), Y["dev"], nbin)
        if best is None or rg.dev_score > best[0]:
            best = (rg.dev_score, L, rg)
    _, L, rg = best
    P = rg.predict(A.X_last(ids["test"], L))
    return {"layer": L, "P": P, "Y": Y["test"], "rows": spx["test"], "lam": rg.lam}


def exact_items(P, Y):
    return ((P > 0.5) == (Y > 0.5)).all(1).astype(float)


def balanced_acc(P, Y):
    p, y = (P > 0.5).ravel(), (Y > 0.5).ravel()
    tpr = (p & y).sum() / max(y.sum(), 1)
    tnr = (~p & ~y).sum() / max((~y).sum(), 1)
    return float((tpr + tnr) / 2)


def parent_means(rows, v):
    pids = sorted({r["parent_id"] for r in rows})
    ix = {p: i for i, p in enumerate(pids)}
    s, n = np.zeros(len(pids)), np.zeros(len(pids))
    for r, x in zip(rows, v):
        s[ix[r["parent_id"]]] += x
        n[ix[r["parent_id"]]] += 1
    return s / n


def paired_boot(rows, a, b, n_boot=10000, seed=17):
    return decide.paired_bootstrap(parent_means(rows, a), parent_means(rows, b), n_boot, seed)


def boot_stat_diff(rows, fa, fb, n_boot=2000, seed=17):
    """Parent-cluster bootstrap of stat_a - stat_b where fa/fb map an index array -> statistic."""
    pids = sorted({r["parent_id"] for r in rows})
    by = {p: [] for p in pids}
    for i, r in enumerate(rows):
        by[r["parent_id"]].append(i)
    rng = np.random.default_rng(seed)
    full = np.arange(len(rows))
    point = fa(full) - fb(full)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(pids), len(pids))
        idx = np.concatenate([by[pids[j]] for j in pick])
        bs.append(fa(idx) - fb(idx))
    return {"diff": float(point), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
