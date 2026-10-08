"""Goal-conditioned plan selection on held-out parents under held-out goals (RCWM H3).

Selection among the K precommitted candidates of each TEST parent. Utility U_g = w_g · φ(final).
Test goals: 256 Gaussian unit vectors over the full φ space (seed 1001).
Training-goal regimes for the scalar evaluators:
  full16 — 16 Gaussian goals over the full space (seed 2002): new test goals are in span
  sub4   — 16 goals inside a seeded random 4-dim subspace (seed 3003): test goals mostly lie
           outside what the scalar evaluators were ever asked to value
Goal-agnostic readouts (B3, S) never see goals in training, so they are regime-independent.

Methods
  oracle / random
  B3_ridge, B3_mlp     consequence head on frozen h -> φ̂ (sigmoid probs), U = w·φ̂
  B2_mlp, B2_bilinear  matched scalar evaluators trained on train-goal utilities
  S_mlp                no-LLM consequence learner on the exact symbolic plan encoding
  G_<mode>             generative: parsed predicted final state -> φ, unparseable -> parent mean
  F_<heuristic>        answer-free floors: φ of a heuristic final state (copy, first-only, ...)
"""
from __future__ import annotations

import numpy as np

from . import data, decide, env, fidelity, heads

N_TEST_GOALS = 256
N_TRAIN_GOALS = 16


def _cube(pids, K, k, pos, Urows):
    out = np.zeros((len(pids), K, Urows.shape[1]), np.float32)
    for pi, pid in enumerate(pids):
        for c in range(K):
            out[pi, c] = Urows[pos[f"{pid}-c{c}-k{k}"]]
    return out


def run(arm, rows, fam, k, layer, gen, budgets=(8, 16, 35, None), seeds=(17,), subsets=1, return_per_parent=False) -> dict:
    dim = len(rows[0]["phi"]) if rows[0]["family"] == fam else len(next(r for r in rows if r["family"] == fam)["phi"])
    nbin = data.PHI_SPEC[fam]["bin"]
    Gte = decide.goals(dim, N_TEST_GOALS, 1001)
    regimes = {"full16": decide.goals(dim, N_TRAIN_GOALS, 2002),
               "sub4": decide.goals_in_subspace(dim, N_TRAIN_GOALS, min(4, dim - 1), 3003)}
    sp = {s: [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
    te = sp["test"]
    pids = sorted({r["parent_id"] for r in te})
    K = 1 + max(int(r["cand_id"].rsplit("-c", 1)[1]) for r in te)
    pos = {r["item_id"]: i for i, r in enumerate(te)}
    Yte = np.stack([r["phi"] for r in te])
    U_true = _cube(pids, K, k, pos, Yte @ Gte.T)
    rng = np.random.default_rng(17)
    per_parent, summary = {}, {}

    def add(name, U_pred):
        t = decide.regret_table(U_true, U_pred, rng)
        per_parent[name] = t
        summary[name] = {"nregret": decide.bootstrap_mean(t["nregret"]), "optimal": decide.bootstrap_mean(t["optimal"])}

    add("oracle", U_true)
    add("random", np.zeros_like(U_true))
    # answer-free floors
    heur = {}
    for r in te:
        for name, st in fidelity.heuristic_states(fam, r["s0"], r["events"]).items():
            heur.setdefault(name, []).append(data.phi_from_state(fam, r["s0"], r["events"], st))
    for name, v in heur.items():
        add(f"F_{name}", _cube(pids, K, k, pos, np.stack(v) @ Gte.T))
    # generative
    for mode in ("e2e", "chain_one", "chain_m1"):
        if gen and mode in gen:
            ph = []
            for r in te:
                st = fidelity._gen_state(gen, mode, r)
                ph.append(np.full(dim, np.nan, np.float32) if st is None else data.phi_from_state(fam, r["s0"], r["events"], st))
            Uc = _cube(pids, K, k, pos, np.stack(ph) @ Gte.T)
            fill = np.nanmean(np.where(np.isnan(Uc), np.nan, Uc), axis=1, keepdims=True)
            Uc = np.where(np.isnan(Uc), np.where(np.isnan(fill), 0.0, fill), Uc)
            add(f"G_{mode}", Uc)

    # trained readouts across label budgets / seeds
    tr_all, dv = sp["train"], sp["dev"]
    tr_pids = sorted({r["parent_id"] for r in tr_all})
    Xdv, Xte = arm.X_last([r["item_id"] for r in dv], layer), arm.X_last([r["item_id"] for r in te], layer)
    Ydv = np.stack([r["phi"] for r in dv])
    sym = lambda xs: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in xs])
    Sdv, Ste = sym(dv), sym(te)

    def expand(X, Y, G):
        return np.repeat(X, len(G), 0), np.tile(G, (len(X), 1)), (Y @ G.T).reshape(-1)

    sweep = {}
    for n in budgets:
        nn_ = n or len(tr_pids)
        for sub in range(subsets if n else 1):
            order = np.random.default_rng(100 + sub).permutation(tr_pids) if n else tr_pids
            keep = set(order[:nn_])
            tr = [r for r in tr_all if r["parent_id"] in keep]
            Xtr, Ytr = arm.X_last([r["item_id"] for r in tr], layer), np.stack([r["phi"] for r in tr])
            for seed in seeds:
                tag = f"n{nn_}_s{sub}_seed{seed}"
                res = {}
                rg = heads.Ridge().fit(Xtr, Ytr, Xdv, Ydv, nbin)
                mh = heads.MLPHead(seed=seed).fit(Xtr, Ytr, Xdv, Ydv, nbin)
                sm = heads.MLPHead(seed=seed).fit(sym(tr), Ytr, Sdv, Ydv, nbin)
                for name, P in (("B3_ridge", np.clip(rg.predict(Xte), -5, 5)), ("B3_mlp", mh.predict(Xte)), ("S_mlp", sm.predict(Ste))):
                    res[name] = _cube(pids, K, k, pos, P @ Gte.T)
                for regime, Gtr in regimes.items():
                    Htr, Wtr, Utr = expand(Xtr, Ytr, Gtr)
                    Hdv, Wdv, Udv = expand(Xdv, Ydv, Gtr)
                    Hte, Wte, _ = expand(Xte, Yte, Gte)
                    for kind in ("mlp", "bilinear"):
                        b2 = heads.ScalarHead(kind, seed=seed).fit(Htr, Wtr, Utr, Hdv, Wdv, Udv)
                        res[f"B2_{kind}@{regime}"] = _cube(pids, K, k, pos, b2.predict(Hte, Wte).reshape(len(te), N_TEST_GOALS))
                entry = {}
                for name, U in res.items():
                    t = decide.regret_table(U_true, U, rng)
                    per_parent[f"{name}|{tag}"] = t
                    entry[name] = float(np.nanmean(t["nregret"]))
                sweep[tag] = entry
    # aggregate across subsets/seeds at each budget
    agg = {}
    for n in budgets:
        nn_ = n or len(tr_pids)
        tags = [t for t in sweep if t.startswith(f"n{nn_}_")]
        names = sweep[tags[0]].keys()
        agg[f"n{nn_}"] = {m: {"mean": float(np.mean([sweep[t][m] for t in tags])),
                              "sd": float(np.std([sweep[t][m] for t in tags]))} for m in names}

    def avg_pp(name, nn_):
        tags = [t for t in per_parent if t.startswith(f"{name}|n{nn_}_")]
        return np.nanmean(np.stack([per_parent[t]["nregret"] for t in tags]), 0)

    contrasts = {}
    for n in budgets:
        nn_ = n or len(tr_pids)
        for regime in regimes:
            for kind in ("mlp", "bilinear"):
                contrasts[f"n{nn_}|B3_mlp - B2_{kind}@{regime}"] = decide.paired_bootstrap(avg_pp("B3_mlp", nn_), avg_pp(f"B2_{kind}@{regime}", nn_))
        contrasts[f"n{nn_}|B3_mlp - S_mlp"] = decide.paired_bootstrap(avg_pp("B3_mlp", nn_), avg_pp("S_mlp", nn_))
    full = len(tr_pids)
    for g in [m for m in summary if m.startswith("G_") or m.startswith("F_")]:
        contrasts[f"n{full}|B3_mlp - {g}"] = decide.paired_bootstrap(avg_pp("B3_mlp", full), per_parent[g]["nregret"])
    res = {"n_test_parents": len(pids), "K": K, "layer": layer, "fixed": summary, "budgets": agg,
           "contrasts": contrasts, "n_train_parents_full": full}
    if return_per_parent:
        res["pids"] = pids
        res["per_parent"] = {name: t["nregret"] for name, t in per_parent.items()}
    return res


def g_only(rows, fam, k, gen, modes=("e2e", "chain_one", "chain_m1"), split="test") -> dict:
    """Per-parent normalised regret of generative consequence models only (fast; no training).
    Returns {mode: {"pids": [...], "nregret": array}} on the same test parents and goals as run()."""
    dim = len(next(r for r in rows if r["family"] == fam)["phi"])
    Gte = decide.goals(dim, N_TEST_GOALS, 1001)
    te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == split]
    pids = sorted({r["parent_id"] for r in te})
    K = 1 + max(int(r["cand_id"].rsplit("-c", 1)[1]) for r in te)
    pos = {r["item_id"]: i for i, r in enumerate(te)}
    U_true = _cube(pids, K, k, pos, np.stack([r["phi"] for r in te]) @ Gte.T)
    rng = np.random.default_rng(17)
    out = {}
    for mode in modes:
        if not gen or mode not in gen:
            continue
        ph = []
        for r in te:
            st = fidelity._gen_state(gen, mode, r)
            ph.append(np.full(dim, np.nan, np.float32) if st is None else data.phi_from_state(fam, r["s0"], r["events"], st))
        Uc = _cube(pids, K, k, pos, np.stack(ph) @ Gte.T)
        fill = np.nanmean(Uc, axis=1, keepdims=True)
        Uc = np.where(np.isnan(Uc), np.where(np.isnan(fill), 0.0, fill), Uc)
        out[mode] = {"pids": pids, "nregret": decide.regret_table(U_true, Uc, rng)["nregret"]}
    return out
