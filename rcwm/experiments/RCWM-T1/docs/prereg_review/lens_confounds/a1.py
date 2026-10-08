import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["RCWM_THREADS"] = "4"
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import data, decide, decisions, env, fidelity
from analyze import load_gen
rows = data.candidate_table(data.load_bank("bank_pilot"))
gen = {x: load_gen(ROOT / "runs" / "bank_pilot" / x) for x in ("sparse", "supa")}


def cube(te, pids, K, k, vals):
    pos = {r["item_id"]: i for i, r in enumerate(te)}
    return decisions._cube(pids, K, k, pos, vals)


def nreg(U_true, U):
    return decide.regret_table(U_true, U, np.random.default_rng(17))["nregret"]


def fillmean(Uc):
    fill = np.nanmean(Uc, axis=1, keepdims=True)
    return np.where(np.isnan(Uc), np.where(np.isnan(fill), 0.0, fill), Uc)


def touched(fam, ev):
    named = set()
    for e in ev:
        if fam == "ledger_replay":
            if e["status"] == "posted":
                named.add(e["account"])
                if e["kind"] == "transfer":
                    named.add(e["other"])
        else:
            named.add(e["register"])
            if e["kind"] == "swap":
                named.add(e["other"])
    return named


out = {}
for fam in ("ledger_replay", "register_machine"):
    keys = env.ACCOUNTS if fam == "ledger_replay" else env.REGISTERS
    nb = data.PHI_SPEC[fam]["bin"]
    for k in (1, 4):
        te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "test"]
        tr = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "train"]
        pids = sorted({r["parent_id"] for r in te}); K = 8
        dim = len(te[0]["phi"]); G = decide.goals(dim, 256, 1001)
        Y = np.stack([r["phi"] for r in te]); U_true = cube(te, pids, K, k, Y @ G.T)
        res = {}
        ch = list(range(4, 8)) if fam == "ledger_replay" else list(range(6, 9))
        oth = [i for i in range(dim) if i not in ch]
        Ych = np.zeros_like(Y); Ych[:, ch] = Y[:, ch]
        res["oracle_changed_only"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, Ych @ G.T))))
        Yo = np.zeros_like(Y); Yo[:, oth] = Y[:, oth]
        res["oracle_nonchanged_only"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, Yo @ G.T))))
        trY = np.stack([r["phi"] for r in tr])

        def touched_phi(r, mode):
            s0 = r["s0"]; nm = touched(fam, r["events"])
            ph = data.phi_from_state(fam, s0, r["events"], s0).copy()
            for i, a in enumerate(keys):
                if a in nm:
                    ph[ch[i]] = 1.0
                    if mode == "mean":
                        if fam == "ledger_replay":
                            for d in (i, 8 + i):
                                ph[d] = trY[:, d].mean()
                        else:
                            for d in (i, 3 + i):
                                ph[d] = 0.5
            return ph

        for mode in ("copy", "mean"):
            P = np.stack([touched_phi(r, mode) for r in te])
            res[f"F_touched_{mode}"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, P @ G.T))))
            P2 = np.zeros_like(P); P2[:, ch] = P[:, ch]
            res["F_touched_changedonly"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, P2 @ G.T))))
        for arm in ("sparse", "supa"):
            g = gen[arm]
            for mode in ("chain_one", "e2e"):
                if mode not in g:
                    continue
                sts = [fidelity._gen_state(g, mode, r) for r in te]
                pr = float(np.mean([s is not None for s in sts]))
                ph = np.stack([np.full(dim, np.nan, np.float32) if s is None else data.phi_from_state(fam, r["s0"], r["events"], s) for r, s in zip(te, sts)])
                exact = float(np.mean([s is not None and env.dumps(s) == r["gold"] for r, s in zip(te, sts)]))
                Uc = fillmean(cube(te, pids, K, k, ph @ G.T))
                base = nreg(U_true, Uc)
                d = {"parse": pr, "exact": exact, "nreg_parentmean": float(np.nanmean(base))}
                ph_s0 = np.stack([data.phi_from_state(fam, r["s0"], r["events"], r["s0"]) if s is None else data.phi_from_state(fam, r["s0"], r["events"], s) for r, s in zip(te, sts)])
                d["nreg_impute_s0"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, ph_s0 @ G.T))))
                ph_t = np.stack([touched_phi(r, "mean") if s is None else data.phi_from_state(fam, r["s0"], r["events"], s) for r, s in zip(te, sts)])
                d["nreg_impute_touched"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, ph_t @ G.T))))
                if mode.startswith("chain"):
                    def lastparsed(r):
                        gg = g[mode].get(r["cand_id"])
                        ss = [None if x is None else json.loads(x) for x in gg["states"][: k + 1]]
                        ss = [x for x in ss if x is not None]
                        return ss[-1]
                    ph_l = np.stack([data.phi_from_state(fam, r["s0"], r["events"], lastparsed(r)) for r in te])
                    d["nreg_impute_lastparsed"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, ph_l @ G.T))))
                ph0 = np.where(np.isnan(ph), 0, ph)
                P2 = np.zeros_like(ph0); P2[:, ch] = ph0[:, ch]
                d["changed_dims_only"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, P2 @ G.T))))
                P3 = np.zeros_like(ph0); P3[:, oth] = ph0[:, oth]
                d["nonchanged_dims_only"] = float(np.nanmean(nreg(U_true, cube(te, pids, K, k, P3 @ G.T))))
                ok = np.array([s is not None for s in sts])
                d["changed_flag_acc"] = float((ph[ok][:, ch] == Y[ok][:, ch]).mean())
                d["changed_exact"] = float((ph[ok][:, ch] == Y[ok][:, ch]).all(1).mean())
                ob = [i for i in oth if i < nb]
                d["other_bin_acc"] = float((ph[ok][:, ob] == Y[ok][:, ob]).mean())
                res[f"G_{arm}_{mode}"] = d
        Pt = np.stack([touched_phi(r, "copy") for r in te])
        res["touched_changed_flag_acc"] = float((Pt[:, ch] == Y[:, ch]).mean())
        res["touched_changed_exact"] = float((Pt[:, ch] == Y[:, ch]).all(1).mean())
        u_star = U_true.max(1); u_mean = U_true.mean(1); den = (u_star - u_mean).sum(1)
        res["den_min_max_median"] = [float(den.min()), float(den.max()), float(np.median(den))]
        res["n_excluded"] = int((den <= 1e-9).sum())
        res["n_distinct_final_per_parent"] = [len({r["gold"] for r in te if r["parent_id"] == p}) for p in pids]
        out[f"{fam}|k{k}"] = res
        print(fam, k, json.dumps(res, indent=0), flush=True)
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
