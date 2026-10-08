"""Audit 8: selection/evaluation metric mismatch. heads.Ridge picks lambda (and fidelity picks the layer)
by mean per-feature dev accuracy, but the reported metric is exact-Δ. Re-select by dev exact-Δ
(same probe, same splits, same grid + a wider grid) and compare TEST exact-Δ."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, fidelity, heads
rows = data.candidate_table(data.load_bank("bank_pilot"))
rows_one = [dict(r, item_id=r["cand_id"] + "-one") for r in rows if r["k"] == 1]
GRID = (1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)
WIDE = (1e-2,) + GRID + (1e6, 1e7)

def fit_all(Xtr, Ytr, lams):
    std = heads.Standardizer(Xtr)
    A = torch.tensor(std(Xtr), dtype=torch.float64); Y = torch.tensor(Ytr, dtype=torch.float64)
    ym = Y.mean(0, keepdim=True); G = A.T @ A; XtY = A.T @ (Y - ym); eye = torch.eye(G.shape[0], dtype=torch.float64)
    return std, ym, {lam: torch.linalg.solve(G + lam * eye, XtY) for lam in lams}

def pred(std, ym, W, X):
    return (torch.tensor(std(X), dtype=torch.float64) @ W + ym).numpy()

for armname in sys.argv[1:]:
    arm = data.Arm("bank_pilot", armname)
    for fam in ("ledger_replay", "prerequisite_plan", "register_machine"):
        for kk, rr, k in (("one", rows_one, 1), ("k1", rows, 1), ("k2", rows, 2), ("k4", rows, 4)):
            sp = {s: [r for r in rr if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
            D = {s: np.stack([fidelity.delta_of(fam, r["s0"], r["states"][-1]) for r in x]) for s, x in sp.items()}
            ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
            nb = D["train"].shape[1]
            res = {"feat": None, "exact": None, "exact_wide": None}
            for L in arm.layers:
                if L == 0: continue
                Xtr, Xdv, Xte = (arm.X_last(ids[s], L) for s in ("train", "dev", "test"))
                std, ym, Ws = fit_all(Xtr, D["train"], WIDE)
                for lam, W in Ws.items():
                    Pd = pred(std, ym, W, Xdv)
                    feat = heads._score(Pd, D["dev"], nb)
                    ex = float(((Pd > 0.5) == (D["dev"] > 0.5)).all(1).mean())
                    te = None
                    def test_exact():
                        return float(((pred(std, ym, W, Xte) > 0.5) == (D["test"] > 0.5)).all(1).mean())
                    if lam in GRID and (res["feat"] is None or feat > res["feat"][0]):
                        res["feat"] = (feat, L, lam, test_exact())
                    if lam in GRID and (res["exact"] is None or ex > res["exact"][0]):
                        res["exact"] = (ex, L, lam, test_exact())
                    if res["exact_wide"] is None or ex > res["exact_wide"][0]:
                        res["exact_wide"] = (ex, L, lam, test_exact())
            copy = float((D["test"] == 0).all(1).mean())
            print(f"{armname:<6} {fam[:6]} {kk:<3} copy={copy:.3f} | sel-by-featacc: test={res['feat'][3]:.3f} (L{res['feat'][1]},lam={res['feat'][2]:g}) "
                  f"| sel-by-exact: test={res['exact'][3]:.3f} (L{res['exact'][1]},lam={res['exact'][2]:g}) "
                  f"| wide grid: test={res['exact_wide'][3]:.3f} (L{res['exact_wide'][1]},lam={res['exact_wide'][2]:g})", flush=True)
