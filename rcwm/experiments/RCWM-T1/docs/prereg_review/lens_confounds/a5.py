"""(i) Contrasts under goals that ignore the 'changed' dims; (ii) B3 on the trained one-step frame at k=1;
(iii) seed variance of B3/S; (iv) raw (unnormalised) regret."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["RCWM_THREADS"] = "8"
from pathlib import Path
import numpy as np
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import data, decide, decisions, env, fidelity, heads
from analyze import load_gen
rows = data.candidate_table(data.load_bank("bank_pilot"))
arm = data.Arm("bank_pilot", "sparse")
gen = {x: load_gen(ROOT / "runs" / "bank_pilot" / x) for x in ("sparse", "supa")}


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


def per_parent(te, P, k, G, raw=False):
    pids = sorted({r["parent_id"] for r in te})
    pos = {r["item_id"]: i for i, r in enumerate(te)}
    Y = np.stack([r["phi"] for r in te])
    Ut = decisions._cube(pids, 8, k, pos, Y @ G.T)
    Up = decisions._cube(pids, 8, k, pos, P @ G.T)
    fill = np.nanmean(Up, axis=1, keepdims=True)
    Up = np.where(np.isnan(Up), np.where(np.isnan(fill), 0.0, fill), Up)
    t = decide.regret_table(Ut, Up, np.random.default_rng(17))
    if raw:
        noise = np.random.default_rng(17).random(Up.shape) * 1e-9
        sel = np.argmax(Up + noise, axis=1)
        u_sel = np.take_along_axis(Ut, sel[:, None, :], axis=1)[:, 0, :]
        return (Ut.max(1) - u_sel).mean(1)
    return t["nregret"]


out = {}
for fam in ("ledger_replay", "register_machine"):
    keys = env.ACCOUNTS if fam == "ledger_replay" else env.REGISTERS
    nb = data.PHI_SPEC[fam]["bin"]
    ch = list(range(4, 8)) if fam == "ledger_replay" else list(range(6, 9))
    for k in (1, 4):
        t0 = time.time()
        sp = {s: [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
        te = sp["test"]
        dim = len(te[0]["phi"])
        ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
        Y = {s: np.stack([r["phi"] for r in x]) for s, x in sp.items()}
        D = {s: np.stack([fidelity.delta_of(fam, r["s0"], r["states"][-1]) for r in x]) for s, x in sp.items()}
        Lsel, _, _ = fidelity.select_layer_ridge(arm, ids["train"], D["train"], ids["dev"], D["dev"], nb)
        preds = {}
        for seed in (17, 18, 19, 20, 21):
            mh = heads.MLPHead(seed=seed).fit(arm.X_last(ids["train"], Lsel), Y["train"], arm.X_last(ids["dev"], Lsel), Y["dev"], nb)
            preds[f"B3_seed{seed}"] = mh.predict(arm.X_last(ids["test"], Lsel))
            sym = lambda xs: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in xs])
            sm = heads.MLPHead(seed=seed).fit(sym(sp["train"]), Y["train"], sym(sp["dev"]), Y["dev"], nb)
            preds[f"S_seed{seed}"] = sm.predict(sym(te))
        if k == 1:
            # B3 on Titan's trained one-step frame ('-one' items share the k=1 target)
            one = {s: [r["cand_id"] + "-one" for r in x] for s, x in sp.items()}
            best = None
            for L in arm.layers:
                if L == 0:
                    continue
                rg = heads.Ridge().fit(arm.X_last(one["train"], L), D["train"], arm.X_last(one["dev"], L), D["dev"], nb)
                if best is None or rg.dev_score > best[0]:
                    best = (rg.dev_score, L)
            L1 = best[1]
            mh = heads.MLPHead(seed=17).fit(arm.X_last(one["train"], L1), Y["train"], arm.X_last(one["dev"], L1), Y["dev"], nb)
            preds["B3one"] = mh.predict(arm.X_last(one["test"], L1))
        for a in ("sparse", "supa"):
            sts = [fidelity._gen_state(gen[a], "chain_one", r) for r in te]
            preds[f"G_{a}"] = np.stack([np.full(dim, np.nan, np.float32) if s is None else data.phi_from_state(fam, r["s0"], r["events"], s) for r, s in zip(te, sts)])
        Pt = []
        for r in te:
            nm = touched(fam, r["events"])
            ph = data.phi_from_state(fam, r["s0"], r["events"], r["s0"]).copy()
            for i, a in enumerate(keys):
                if a in nm:
                    ph[ch[i]] = 1.0
            Pt.append(ph)
        preds["F_touched"] = np.stack(Pt)
        Gfull = decide.goals(dim, 256, 1001)
        Gnc = Gfull.copy(); Gnc[:, ch] = 0; Gnc /= np.linalg.norm(Gnc, axis=1, keepdims=True)
        res = {"layer": Lsel}
        for gname, G in (("full", Gfull), ("no_changed_dims", Gnc)):
            pp = {m: per_parent(te, P, k, G) for m, P in preds.items()}
            pp["B3_avg5"] = np.nanmean(np.stack([pp[f"B3_seed{s}"] for s in (17, 18, 19, 20, 21)]), 0)
            pp["S_avg5"] = np.nanmean(np.stack([pp[f"S_seed{s}"] for s in (17, 18, 19, 20, 21)]), 0)
            res[gname] = {m: float(np.nanmean(v)) for m, v in pp.items()}
            res[gname]["B3_seed_range"] = [min(res[gname][f"B3_seed{s}"] for s in (17, 18, 19, 20, 21)), max(res[gname][f"B3_seed{s}"] for s in (17, 18, 19, 20, 21))]
            res[gname]["S_seed_range"] = [min(res[gname][f"S_seed{s}"] for s in (17, 18, 19, 20, 21)), max(res[gname][f"S_seed{s}"] for s in (17, 18, 19, 20, 21))]
            res[gname]["G-B3_seed17"] = decide.paired_bootstrap(pp["G_sparse"], pp["B3_seed17"])
            res[gname]["G-S_seed17"] = decide.paired_bootstrap(pp["G_sparse"], pp["S_seed17"])
            res[gname]["G-F_touched"] = decide.paired_bootstrap(pp["G_sparse"], pp["F_touched"])
            res[gname]["G-Gsupa"] = decide.paired_bootstrap(pp["G_sparse"], pp["G_supa"])
            res[gname]["sd_G-B3"] = float(np.nanstd(pp["G_sparse"] - pp["B3_seed17"]))
        rawpp = {m: per_parent(te, P, k, Gfull, raw=True) for m, P in preds.items() if m in ("G_sparse", "B3_seed17", "S_seed17", "F_touched", "G_supa")}
        res["raw_regret"] = {m: float(np.nanmean(v)) for m, v in rawpp.items()}
        out[f"{fam}|k{k}"] = res
        print(fam, k, json.dumps(res), f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1, default=str)
