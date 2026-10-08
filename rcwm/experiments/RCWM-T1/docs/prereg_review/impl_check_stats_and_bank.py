import os, sys, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RCWM_THREADS"] = "2"
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
import numpy as np
import confirm_analysis as ca
from rcwmt import data, decide, decisions, env

print("== Holm")
for ps in ({"a": 0.01, "b": 0.02, "c": 0.021}, {"a": 0.02, "b": 0.03, "c": 0.04}, {"a": 0.0, "b": 0.0, "c": 0.03}):
    print(ps, ca.holm(ps))

print("== strat_boot")
rng = np.random.default_rng(0)
dL = rng.normal(-0.3, 0.3, 150); dR = rng.normal(-0.6, 0.3, 50)
r = ca.strat_boot({"L": dL, "R": dR}); print(r, "pooled-unweighted", np.concatenate([dL, dR]).mean(), "eq-w", (dL.mean()+dR.mean())/2)
r = ca.strat_boot({"L": rng.normal(-0.02, 0.3, 150), "R": rng.normal(-0.02, 0.3, 150)}); print("near null", r["p_one_sided"], r["ci95"])
dn = dL.copy(); dn[:5] = np.nan
print("nan", ca.strat_boot({"L": dn, "R": dR})["n_parents"])
try:
    print(ca.strat_boot({"L": np.full(10, np.nan), "R": dR}))
except Exception as e:
    print("all-nan family ->", type(e).__name__, e)

print("== confirmation candidate_table")
bc = data.load_bank("bank_confirm")
rows = data.candidate_table(bc)
print(len(rows), collections.Counter(r["split"] for r in rows), any("-t0" in r["cand_id"] for r in rows))
for fam in env.FAMILIES:
    for k in (1, 2, 4):
        te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "test"]
        pids = sorted({r["parent_id"] for r in te})
        K = 1 + max(int(r["cand_id"].rsplit("-c", 1)[1]) for r in te)
        pos = {r["item_id"]: i for i, r in enumerate(te)}
        Y = np.stack([r["phi"] for r in te]); G = decide.goals(Y.shape[1], 256, 1001)
        U = decisions._cube(pids, K, k, pos, Y @ G.T)
        t = decide.regret_table(U, U, np.random.default_rng(17))
        print(fam[:6], k, len(te), len(pids), K, U.shape, "nan-den parents", int(np.isnan(t["nregret"]).sum()))
# twin items from bank.items
from rcwmt import bank as bankmod
its = bankmod.items(bc)
print("items", len(its), collections.Counter(it["format"] for it in its), sum(1 for it in its if it.get("twin")))
print("twin ids sample", [it["item_id"] for it in its if it.get("twin")][:2])

print("== tie check: identical phi rows -> identical U?")
for dim in (8, 9, 12):
    G = decide.goals(dim, 256, 1001)
    base = (np.random.default_rng(1).random((50, dim)) > 0.5).astype(np.float32)
    if dim == 12: base[:, 8:] = np.random.default_rng(2).normal(0, 1, (50, 4)).astype(np.float32)
    X = base[np.random.default_rng(3).integers(0, 50, 1200)]
    U = X @ G.T
    mism = 0
    for i in range(50):
        idx = np.where((X == base[i]).all(1))[0]
        if len(idx) > 1 and not (U[idx] == U[idx[0]]).all(): mism += 1
    print(dim, "rows w/ non-bitwise-identical U among duplicates:", mism)
