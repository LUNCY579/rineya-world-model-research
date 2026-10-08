"""Which item set / definition reproduces the R5 table? (independent; numpy only)"""
import json, numpy as np, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RUN = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/runs/bank_pilot")
idx = json.loads((RUN / "supa/index.json").read_text(encoding="utf-8"))["items"]
A_ = np.load(RUN / "supa/last.npy", mmap_mode="r"); B_ = np.load(RUN / "sparse/last.npy", mmap_mode="r")
LAY = [0, 4, 8, 12, 16, 20, 24, 28]
def cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))
def metrics(rows):
    rows = np.sort(np.array(rows))
    XA = np.asarray(A_[rows], np.float64); XB = np.asarray(B_[rows], np.float64)
    out = []
    for L in (12, 16, 20, 24, 28):
        A, B = XA[:, LAY.index(L)], XB[:, LAY.index(L)]
        d = B - A
        m = np.argsort(-np.abs(A).mean(0))[:5]; keep = np.setdiff1d(np.arange(1024), m)
        out.append((L, round(float(np.linalg.norm(d) / np.linalg.norm(A)), 3), round(float(np.linalg.norm(d[:, keep]) / np.linalg.norm(A[:, keep])), 3),
                    round(float(np.median(np.abs(d / (A.std(0) + 1e-6)))), 2), round(cka(A, B), 3)))
    return out
sets = {
 "test one all-fam": [i for i, it in enumerate(idx) if it["split"] == "test" and it["item_id"].endswith("-one")],
 "test k1 all-fam": [i for i, it in enumerate(idx) if it["split"] == "test" and it["item_id"].endswith("-k1")],
 "test all items": [i for i, it in enumerate(idx) if it["split"] == "test"],
 "all one": [i for i, it in enumerate(idx) if it["item_id"].endswith("-one")],
}
for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    sets[f"test one {fam[:6]}"] = [i for i, it in enumerate(idx) if it["split"] == "test" and it["item_id"].endswith("-one") and it["family"] == fam]
print("claimed R5: L12 (0.16,0.24,1.03,0.991) L16 (0.40,0.53,1.58,0.953) L20 (0.65,0.81,1.61,0.965) L24 (0.75,1.15,1.61,0.925) L28 (0.57,0.73,1.36,0.937)")
print("columns: layer, relL2, relL2_nomassive, median|z|, CKA")
for n, r in sets.items():
    print(f"{n:<22} n={len(r):<5}", metrics(r))
