"""S with abundant simulator labels (no model involved): is C2 a label-budget statement?"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["RCWM_THREADS"] = "8"
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, data, decide, decisions, env, heads
rows = data.candidate_table(data.load_bank("bank_pilot"))
t0 = time.time()
big = bankmod.build(n_parents=4000, k_cand=8, horizon=4, seed=777001, exclusions=None)
for p in big["parents"]:
    p["split"] = "train"
brow = data.candidate_table(big)
print("generated", len(big["parents"]), f"{time.time()-t0:.0f}s", flush=True)


def regret_of(split_rows, P, k):
    pids = sorted({r["parent_id"] for r in split_rows})
    pos = {r["item_id"]: i for i, r in enumerate(split_rows)}
    G = decide.goals(P.shape[1], 256, 1001)
    Y = np.stack([r["phi"] for r in split_rows])
    Ut = decisions._cube(pids, 8, k, pos, Y @ G.T)
    Up = decisions._cube(pids, 8, k, pos, P @ G.T)
    return float(np.nanmean(decide.regret_table(Ut, Up, np.random.default_rng(17))["nregret"]))


out = {}
for fam in ("ledger_replay", "register_machine"):
    k = 4
    nb = data.PHI_SPEC[fam]["bin"]
    sym = lambda xs: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in xs])
    dv = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "dev"]
    te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "test"]
    pilot_tr = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "train"]
    btr = [r for r in brow if r["family"] == fam and r["k"] == k]
    Ydv = np.stack([r["phi"] for r in dv])
    res = {}
    for n in (len({r['parent_id'] for r in pilot_tr}), 500, 1000, 4000):
        if n <= 200:
            tr = pilot_tr
        else:
            keep = set(sorted({r["parent_id"] for r in btr})[:n])
            tr = [r for r in btr if r["parent_id"] in keep]
        Ytr = np.stack([r["phi"] for r in tr])
        sm = heads.MLPHead(seed=17).fit(sym(tr), Ytr, sym(dv), Ydv, nb)
        P = sm.predict(sym(te))
        Yte = np.stack([r["phi"] for r in te])
        ch = list(range(4, 8)) if fam == "ledger_replay" else list(range(6, 9))
        res[n] = {"S_mlp_test_nregret": regret_of(te, P, k),
                  "changed_flag_acc": float(((P[:, ch] > 0.5) == (Yte[:, ch] > 0.5)).mean()),
                  "bin_acc": float(((P[:, :nb] > 0.5) == (Yte[:, :nb] > 0.5)).mean())}
        print(fam, n, res[n], f"{time.time()-t0:.0f}s", flush=True)
    out[fam] = res
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
