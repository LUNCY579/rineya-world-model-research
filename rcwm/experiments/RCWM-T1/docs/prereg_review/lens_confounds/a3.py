"""B3 fairness: per-layer DEV/TEST regret for answer-slot and event-boundary states; learning curves."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["RCWM_THREADS"] = "8"
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import data, decide, decisions, env, fidelity, heads
rows = data.candidate_table(data.load_bank("bank_pilot"))
arm = data.Arm("bank_pilot", "sparse")


def regret_of(split_rows, P, k):
    fam = split_rows[0]["family"]
    pids = sorted({r["parent_id"] for r in split_rows})
    pos = {r["item_id"]: i for i, r in enumerate(split_rows)}
    G = decide.goals(P.shape[1], 256, 1001)
    Y = np.stack([r["phi"] for r in split_rows])
    Ut = decisions._cube(pids, 8, k, pos, Y @ G.T)
    Up = decisions._cube(pids, 8, k, pos, P @ G.T)
    return float(np.nanmean(decide.regret_table(Ut, Up, np.random.default_rng(17))["nregret"]))


out = {}
for fam in ("ledger_replay", "register_machine"):
    for k in (4,):
        t0 = time.time()
        sp = {s: [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
        nb = data.PHI_SPEC[fam]["bin"]
        ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
        Y = {s: np.stack([r["phi"] for r in x]) for s, x in sp.items()}
        D = {s: np.stack([fidelity.delta_of(fam, r["s0"], r["states"][-1]) for r in x]) for s, x in sp.items()}
        Lsel, _, by = fidelity.select_layer_ridge(arm, ids["train"], D["train"], ids["dev"], D["dev"], nb)
        res = {"ridgeDelta_selected_layer": Lsel, "ridgeDelta_dev_score_by_layer": by, "per_layer": {}}
        for L in arm.layers:
            if L == 0:
                continue
            row = {}
            for src in ("slot", "event4"):
                if src == "slot":
                    X = {s: arm.X_last(ids[s], L) for s in sp}
                else:
                    X = {s: arm.X_event(ids[s], 4, L) for s in sp}
                mh = heads.MLPHead(seed=17).fit(X["train"], Y["train"], X["dev"], Y["dev"], nb)
                rg = heads.Ridge().fit(X["train"], Y["train"], X["dev"], Y["dev"], nb)
                row[f"{src}_mlp_dev"] = regret_of(sp["dev"], mh.predict(X["dev"]), k)
                row[f"{src}_mlp_test"] = regret_of(sp["test"], mh.predict(X["test"]), k)
                row[f"{src}_ridge_dev"] = regret_of(sp["dev"], np.clip(rg.predict(X["dev"]), -5, 5), k)
                row[f"{src}_ridge_test"] = regret_of(sp["test"], np.clip(rg.predict(X["test"]), -5, 5), k)
                row[f"{src}_mlp_devscore"] = mh.dev_score
            res["per_layer"][L] = row
            print(fam, k, L, {a: round(b, 3) for a, b in row.items()}, f"{time.time()-t0:.0f}s", flush=True)
        # learning curve at the registered layer (slot, MLP) and for S
        tr_p = sorted({r["parent_id"] for r in sp["train"]})
        sym = lambda xs: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in xs])
        lc = {}
        for n in (16, 35, 70, len(tr_p)):
            vals = {"B3": [], "S": []}
            for sub in range(3 if n < len(tr_p) else 1):
                keep = set(np.random.default_rng(100 + sub).permutation(tr_p)[:n])
                tr = [r for r in sp["train"] if r["parent_id"] in keep]
                Ytr = np.stack([r["phi"] for r in tr])
                mh = heads.MLPHead(seed=17).fit(arm.X_last([r["item_id"] for r in tr], Lsel), Ytr, arm.X_last(ids["dev"], Lsel), Y["dev"], nb)
                vals["B3"].append(regret_of(sp["test"], mh.predict(arm.X_last(ids["test"], Lsel)), k))
                sm = heads.MLPHead(seed=17).fit(sym(tr), Ytr, sym(sp["dev"]), Y["dev"], nb)
                vals["S"].append(regret_of(sp["test"], sm.predict(sym(sp["test"])), k))
            lc[n] = {m: float(np.mean(v)) for m, v in vals.items()}
            print(fam, k, "LC", n, lc[n], flush=True)
        res["learning_curve"] = lc
        out[f"{fam}|k{k}"] = res
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1, default=str)
