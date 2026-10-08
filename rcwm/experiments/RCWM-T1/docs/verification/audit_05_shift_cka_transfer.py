"""Audit 5: C3 numbers (rel L2 w/o massive dims, median |z|, linear CKA), CKA sensitivity to massive
dims, and transfer sanity (identical layers must transfer perfectly)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, controls
sys.path.insert(0, str(ROOT / "scripts"))
from mechanism_report import cka, shift

sp, su = data.Arm("bank_pilot", "sparse"), data.Arm("bank_pilot", "supa")
on_ids = [it["item_id"] for it in su.items if it["split"] == "test" and it["item_id"].endswith("-one")]
L_ids = [i for i in on_ids if i.startswith("T1L")]
print("n test one-frame items", len(on_ids), "ledger only", len(L_ids))

def cka_ref(X, Y):  # HSIC formulation with centering matrix, independent of mechanism_report.cka
    n = X.shape[0]; H = np.eye(n) - 1.0 / n
    K, Lm = X @ X.T, Y @ Y.T
    hs = lambda A, B: np.trace(A @ H @ B @ H)
    return hs(K, Lm) / np.sqrt(hs(K, K) * hs(Lm, Lm))

for L in (12, 16, 20, 24, 28):
    A = su.X_last(on_ids, L).astype(np.float64); B = sp.X_last(on_ids, L).astype(np.float64)
    massive = np.argsort(-np.abs(A).mean(0))[:5]
    s = shift(A, B, massive)
    keep = np.setdiff1d(np.arange(1024), massive)
    var = A.var(0); top_var_share = np.sort(var)[::-1][:5].sum() / var.sum()
    Az = (A - A.mean(0)) / (A.std(0) + 1e-6); Bz = (B - B.mean(0)) / (A.std(0) + 1e-6)
    Al, Bl = su.X_last(L_ids, L).astype(np.float64), sp.X_last(L_ids, L).astype(np.float64)
    print(f"L{L}: relL2_nomassive={s['rel_l2_nomassive']:.3f} median|z|={s['median_abs_z']:.2f} cka={s['cka']:.3f} "
          f"cka_ref={cka_ref(A, B):.3f} | top5-variance share={top_var_share:.2f} cka_no_massive={cka(A[:, keep], B[:, keep]):.3f} "
          f"cka_standardized={cka(Az, Bz):.3f} | ledger-only cka={cka(Al, Bl):.3f} ledger-only cka_std={cka((Al-Al.mean(0))/(Al.std(0)+1e-6), (Bl-Bl.mean(0))/(Al.std(0)+1e-6)):.3f}")

# Transfer sanity: at layer 8 states are bit-identical, so A_on_B must equal A_on_A
rows = data.candidate_table(data.load_bank("bank_pilot"))
rows_one = [dict(r, item_id=r["cand_id"] + "-one") for r in rows if r["k"] == 1]
from rcwmt import fidelity
fn = lambda r: fidelity.delta_of("ledger_replay", r["s0"], r["states"][-1])
for L in (8, 12, 16):
    _, aa = controls._fit_eval(sp, sp, rows_one, "ledger_replay", 1, fn, layer=L)
    _, ab = controls._fit_eval(sp, su, rows_one, "ledger_replay", 1, fn, layer=L)
    _, abr = controls._fit_eval(sp, su, rows_one, "ledger_replay", 1, fn, layer=L, restandardize=True)
    _, bb = controls._fit_eval(su, su, rows_one, "ledger_replay", 1, fn, layer=L)
    _, ba = controls._fit_eval(su, sp, rows_one, "ledger_replay", 1, fn, layer=L)
    _, bar = controls._fit_eval(su, sp, rows_one, "ledger_replay", 1, fn, layer=L, restandardize=True)
    print(f"transfer L{L}: sparse->sparse {aa['exact_delta']:.3f} sparse->supa raw {ab['exact_delta']:.3f} restd {abr['exact_delta']:.3f} | "
          f"supa->supa {bb['exact_delta']:.3f} supa->sparse raw {ba['exact_delta']:.3f} restd {bar['exact_delta']:.3f}")
