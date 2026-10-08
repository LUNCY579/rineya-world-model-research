"""Audit 3: unit checks of X_last ordering, ridge closed form, restandardized transfer, delta_of,
step_cf rotation, exact metric, parent bootstrap. CPU only."""
import json, sys, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, env, fidelity, controls, heads, decide

rng = np.random.default_rng(0)
# --- X_last with shuffled / duplicated ids vs direct memmap indexing
A = data.Arm("bank_pilot", "supa")
ids = [A.items[i]["item_id"] for i in rng.choice(len(A.items), 500, replace=False)] + [A.items[5]["item_id"]] * 3
X = A.X_last(ids, 24)
ref = np.stack([np.asarray(A.last[A.pos[i], A.layer_index(24)]).astype(np.float32) for i in ids])
print("X_last shuffled ids exact match:", np.array_equal(X, ref))
ev_ids = [A.items[i]["item_id"] for i in rng.choice(A.event_rows, 200, replace=False)]
Xe = A.X_event(ev_ids, 3, 20)
refe = np.stack([np.asarray(A.events[A.ev_pos[i], 2, A.layer_index(20)]).astype(np.float32) for i in ev_ids])
print("X_event shuffled ids exact match:", np.array_equal(Xe, refe))

# --- Ridge closed form vs sklearn-free reference: augmented-intercept normal equations with unpenalized intercept
n, d, q = 300, 50, 6
Xtr = rng.normal(size=(n, d)).astype(np.float32) * rng.uniform(0.1, 10, d) + rng.normal(size=d) * 5
W0 = rng.normal(size=(d, q))
Ytr = ((Xtr - Xtr.mean(0)) @ W0 + rng.normal(size=(n, q)) > 0).astype(np.float32)
Xdv = rng.normal(size=(100, d)).astype(np.float32) * rng.uniform(0.1, 10, d)
Ydv = rng.integers(0, 2, (100, q)).astype(np.float32)
rg = heads.Ridge().fit(Xtr, Ytr, Xdv, Ydv, q)
Z = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-3)
Za = np.hstack([Z, np.ones((n, 1))]).astype(np.float64)
P = np.eye(d + 1) * rg.lam; P[-1, -1] = 0
Wa = np.linalg.solve(Za.T @ Za + P, Za.T @ Ytr)
Zt = (Xdv - Xtr.mean(0)) / (Xtr.std(0) + 1e-3)
pred_ref = np.hstack([Zt, np.ones((100, 1))]) @ Wa
print("ridge vs reference (unpenalized intercept) max abs diff:", float(np.abs(rg.predict(Xdv) - pred_ref).max()), "lam", rg.lam)

# --- restandardized transfer math: if B = A*s + t per dim (pure affine), restd transfer must equal A-on-A
class FakeArm:
    def __init__(self, M, ids): self.M, self.p, self.layers = M, {i: k for k, i in enumerate(ids)}, [0, 1]
    def X_last(self, ids, L): return self.M[[self.p[i] for i in ids]]
rows = []
ids = []
for pi in range(60):
    for c in range(8):
        s0 = {a: int(rng.integers(-50, 50)) for a in env.ACCOUNTS}
        ev = {"account": "A", "other": "B", "kind": "add", "amount": int(rng.integers(-5, 5)), "status": "posted", "tick": 1}
        st = env.rollout("ledger_replay", s0, [ev])
        iid = f"p{pi}-c{c}"
        rows.append({"family": "ledger_replay", "k": 1, "split": ["train", "train", "train", "dev", "test"][pi % 5],
                     "item_id": iid, "s0": s0, "events": [ev], "states": st, "parent_id": f"p{pi}"})
        ids.append(iid)
T = np.stack([fidelity.delta_of("ledger_replay", r["s0"], r["states"][-1]) for r in rows])
MA = (np.hstack([T, rng.normal(size=(len(rows), 40))]) @ rng.normal(size=(48, 64))).astype(np.float32) + rng.normal(size=(len(rows), 64)).astype(np.float32) * 0.1
s, t = rng.uniform(0.2, 5, 64).astype(np.float32), rng.normal(size=64).astype(np.float32) * 20
MB = MA * s + t
fa, fb = FakeArm(MA, ids), FakeArm(MB, ids)
class Sel:  # bypass fidelity.select_layer_ridge layer loop over 'layers' 0,1 (layer 0 excluded) -> use 1
    pass
fa.layers = fb.layers = [1]
fn = lambda r: fidelity.delta_of("ledger_replay", r["s0"], r["states"][-1])
La, aa = controls._fit_eval(fa, fa, rows, "ledger_replay", 1, fn)
_, ab = controls._fit_eval(fa, fb, rows, "ledger_replay", 1, fn, layer=La)
_, abr = controls._fit_eval(fa, fb, rows, "ledger_replay", 1, fn, layer=La, restandardize=True)
print("synthetic affine B: A_on_A", aa["exact_delta"], "A_on_B raw", ab["exact_delta"], "A_on_B restd", abr["exact_delta"],
      "(restd must equal A_on_A up to train-vs-same-stats float error)")

# --- delta_of independent re-implementation over the whole bank
bank = data.load_bank("bank_pilot")
crow = data.candidate_table(bank)
bad = 0
for r in crow:
    fam, s0, sk = r["family"], r["s0"], r["states"][-1]
    if fam == "ledger_replay":
        ref = [int((sk[a] < 0) != (s0[a] < 0)) for a in "ABCD"] + [int(sk[a] != s0[a]) for a in "ABCD"]
    elif fam == "prerequisite_plan":
        ref = [((sk ^ s0) >> b) & 1 for b in range(8)]
    else:
        ref = [int((sk[x] > 48) != (s0[x] > 48)) for x in "XYZ"] + [int((sk[x] % 2) != (s0[x] % 2)) for x in "XYZ"] + [int(sk[x] != s0[x]) for x in "XYZ"]
    if not np.array_equal(np.array(ref, np.float32), fidelity.delta_of(fam, s0, sk)):
        bad += 1
print("delta_of vs independent formula mismatches:", bad, "of", len(crow))

# --- step_cf ledger rotation semantics
s0 = {"A": 1, "B": 2, "C": 3, "D": 4}
for acct, oth in (("A", "B"), ("D", "C")):
    ev = {"account": acct, "other": oth, "kind": "transfer", "amount": 10, "status": "posted", "tick": 1}
    print("cf", acct, "->", oth, controls.step_cf("ledger_replay", s0, ev), "true", env.step("ledger_replay", s0, ev))
# rotation cf vs true: base rates over the bank L k=1
L1 = [r for r in crow if r["family"] == "ledger_replay" and r["k"] == 1]
Dt = np.stack([fidelity.delta_of("ledger_replay", r["s0"], r["states"][-1]) for r in L1])
Dc = np.stack([fidelity.delta_of("ledger_replay", r["s0"], controls.rollout_cf("ledger_replay", r["s0"], r["events"])) for r in L1])
print("L k1 base rates true", Dt.mean(0).round(3).tolist())
print("L k1 base rates cf  ", Dc.mean(0).round(3).tolist())

# --- exact metric
Dp = np.array([[0.6, 0.4], [0.49, 0.51], [0.2, 0.8]]); Dtt = np.array([[1, 0], [1, 1], [0, 1]], np.float32)
m = fidelity.delta_metrics(Dp, Dtt); print("exact metric toy (expect 2/3):", m["exact_delta"], m["per_item_exact"])

# --- paired bootstrap: parent means and CI with known answer
a = np.array([1.0, 1.0, 0.0, 1.0]); b = np.array([0.0, 1.0, 0.0, 0.0])
print("paired bootstrap toy:", decide.paired_bootstrap(a, b))
rows_ = [{"parent_id": p} for p in ["x", "x", "y", "z", "z", "z"]]
print("_parent_mean toy (x=.5,y=1,z=1/3):", fidelity._parent_mean(rows_, np.array([1, 0, 1, 1, 0, 0.0])))
