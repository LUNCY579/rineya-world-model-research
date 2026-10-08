"""PREREG-1 evaluation for one (family, k): every method's per-parent normalised regret on the
confirmation parents. Fitting uses pilot TRAIN only; every selection uses pilot DEV only, scored by
decision regret under SELECTION goals (seed 1002), never the test goals (seed 1001).
"""
from __future__ import annotations

import numpy as np

from . import data, decide, fidelity, heads

N_TEST_GOALS = 256
SEL_GOAL_SEED = 1002
TEST_GOAL_SEED = 1001
SEEDS = (17, 18, 19, 20, 21)
CHANGED = {"ledger_replay": slice(4, 8), "register_machine": slice(6, 9)}


def _cube(pids, K, k, pos, Urows):
    out = np.zeros((len(pids), K, Urows.shape[1]), np.float32)
    for pi, pid in enumerate(pids):
        for c in range(K):
            out[pi, c] = Urows[pos[f"{pid}-c{c}-k{k}"]]
    return out


class Split:
    """Decision set for one split: parents x K candidates, true utilities under goals G."""

    def __init__(self, rows, G, k):
        self.rows, self.G, self.k = rows, G, k
        self.pids = sorted({r["parent_id"] for r in rows})
        self.K = 1 + max(int(r["cand_id"].rsplit("-c", 1)[1]) for r in rows)
        self.pos = {r["item_id"]: i for i, r in enumerate(rows)}
        self.Y = np.stack([r["phi"] for r in rows])
        self.U_true = self.cube(self.Y @ G.T)

    def cube(self, Urows):
        return _cube(self.pids, self.K, self.k, self.pos, Urows)

    def regret(self, U_pred, seed=17):
        return decide.regret_table(self.U_true, U_pred, np.random.default_rng(seed))["nregret"]

    def regret_phi(self, P):
        return self.regret(self.cube(P @ self.G.T))


def _gen_phi(fam, rows, gen, mode, fill="parent_mean"):
    """φ̂ rows from generations; unparseable -> NaN (parent-mean fill) or the last parsed chain
    state (sensitivity). Returns (phi array with NaN rows, parse mask)."""
    dim = len(rows[0]["phi"])
    out, ok = [], []
    for r in rows:
        st = fidelity._gen_state(gen, mode, r)
        if st is None and fill == "last_parsed" and mode != "e2e":
            g = gen[mode].get(r["cand_id"])
            prev = [s for s in (g["states"][: r["k"] + 1] if g else []) if s is not None]
            st = None if not prev else __import__("json").loads(prev[-1])
        ok.append(st is not None)
        out.append(np.full(dim, np.nan, np.float32) if st is None else data.phi_from_state(r["family"], r["s0"], r["events"], st))
    return np.stack(out), np.array(ok)


def _fill_parent_mean(Uc):
    fill = np.nanmean(Uc, axis=1, keepdims=True)
    return np.where(np.isnan(Uc), np.where(np.isnan(fill), 0.0, fill), Uc)


def _fit_predict(kind, Xtr, Ytr, Xdv, Ydv, Xte_list, nbin, seed):
    if kind == "ridge":
        m = heads.Ridge().fit(Xtr, Ytr, Xdv, Ydv, nbin)
        return [np.clip(m.predict(X), -5, 5) for X in Xte_list]
    m = heads.MLPHead(seed=seed).fit(Xtr, Ytr, Xdv, Ydv, nbin)
    return [m.predict(X) for X in Xte_list]


def evaluate(arm, rows, fam, k, gen_titan, gen_supa, s_extra_rows=None) -> dict:
    """rows: pilot train/dev rows + confirmation rows with split 'test'."""
    nbin = data.PHI_SPEC[fam]["bin"]
    sp = {s: [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
    dim = len(sp["test"][0]["phi"])
    Gte, Gsel = decide.goals(dim, N_TEST_GOALS, TEST_GOAL_SEED), decide.goals(dim, N_TEST_GOALS, SEL_GOAL_SEED)
    TE, DV = Split(sp["test"], Gte, k), Split(sp["dev"], Gsel, k)
    out = {"pids": TE.pids, "per_parent": {}, "parse_rate": {}, "selection": {}}
    pp = out["per_parent"]

    # floors (answer-free) and partial simulators (diagnostics)
    heur = {}
    for r in sp["test"]:
        for name, st in fidelity.heuristic_states(fam, r["s0"], r["events"]).items():
            heur.setdefault(name, []).append(data.phi_from_state(fam, r["s0"], r["events"], st))
    for name, v in heur.items():
        pp[f"F_{name}"] = TE.regret_phi(np.stack(v))
    if fam in CHANGED:
        ft = []
        for r in sp["test"]:
            ph = data.phi_from_state(fam, r["s0"], r["events"], r["s0"]).copy()
            ph[CHANGED[fam]] = fidelity.touched_delta(fam, r["s0"], r["events"])[CHANGED[fam]]
            ft.append(ph)
        pp["F_touched"] = TE.regret_phi(np.stack(ft))

    # generative consequence models
    for arm_name, gen in (("Gtitan", gen_titan), ("Gsupa", gen_supa)):
        for mode in ("e2e", "chain_one", "chain_m1"):
            if not gen or mode not in gen:
                continue
            P, ok = _gen_phi(fam, sp["test"], gen, mode)
            pp[f"{arm_name}_{mode}"] = TE.regret(_fill_parent_mean(TE.cube(P @ Gte.T)))
            out["parse_rate"][f"{arm_name}_{mode}"] = float(ok.mean())
            if mode != "e2e":
                P2, ok2 = _gen_phi(fam, sp["test"], gen, mode, fill="last_parsed")
                pp[f"{arm_name}_{mode}_fill_lastparsed"] = TE.regret(_fill_parent_mean(TE.cube(P2 @ Gte.T)))
        modes = [m for m in (f"{arm_name}_e2e", f"{arm_name}_chain_one", f"{arm_name}_chain_m1") if m in pp]
        if modes:
            best = min(modes, key=lambda m: np.nanmean(pp[m]))
            pp[f"{arm_name}_best"] = pp[best]
            out[f"{arm_name}_best_mode"] = best

    # B3*: joint (layer != 0, head) selection by pilot-DEV regret under selection goals
    ids = {s: [r["item_id"] for r in sp[s]] for s in sp}
    Ytr, Ydv = np.stack([r["phi"] for r in sp["train"]]), DV.Y
    best, ub = None, []
    for L in arm.layers:
        if L == 0:
            continue
        Xtr, Xdv, Xte = arm.X_last(ids["train"], L), arm.X_last(ids["dev"], L), arm.X_last(ids["test"], L)
        for kind in ("ridge", "mlp"):
            Pdv, Pte = _fit_predict(kind, Xtr, Ytr, Xdv, Ydv, [Xdv, Xte], nbin, 17)
            sc = float(np.nanmean(DV.regret_phi(Pdv)))
            out["selection"][f"B3|L{L}|{kind}"] = sc
            ub.append(float(np.nanmean(TE.regret_phi(Pte))))     # best-on-test: upper bound only
            if best is None or sc < best[0]:
                best = (sc, L, kind)
    _, Lb, kb = best
    out["selection"]["B3_chosen"] = {"layer": Lb, "head": kb}
    out["B3_best_on_test_upper_bound"] = min(ub)
    Xtr, Xdv, Xte = arm.X_last(ids["train"], Lb), arm.X_last(ids["dev"], Lb), arm.X_last(ids["test"], Lb)
    seeds = (17,) if kb == "ridge" else SEEDS
    pp["B3"] = np.nanmean(np.stack([TE.regret_phi(_fit_predict(kb, Xtr, Ytr, Xdv, Ydv, [Xte], nbin, s)[0]) for s in seeds]), 0)

    # S*: head selection by pilot-DEV regret; equal label budget; plus S@abundant (secondary)
    from . import decide as _d
    sym = lambda xs: np.stack([_d.symbolic(fam, r["s0"], r["events"]) for r in xs])
    Str, Sdv, Ste = sym(sp["train"]), sym(sp["dev"]), sym(sp["test"])
    sbest = None
    for kind in ("ridge", "mlp"):
        (Pdv,) = _fit_predict(kind, Str, Ytr, Sdv, Ydv, [Sdv], nbin, 17)
        sc = float(np.nanmean(DV.regret_phi(Pdv)))
        out["selection"][f"S|{kind}"] = sc
        if sbest is None or sc < sbest[0]:
            sbest = (sc, kind)
    ks = sbest[1]
    out["selection"]["S_chosen"] = ks
    seeds = (17,) if ks == "ridge" else SEEDS
    pp["S"] = np.nanmean(np.stack([TE.regret_phi(_fit_predict(ks, Str, Ytr, Sdv, Ydv, [Ste], nbin, s)[0]) for s in seeds]), 0)
    if s_extra_rows:
        xr = [r for r in s_extra_rows if r["family"] == fam and r["k"] == k]
        if xr:
            pp["S_abundant"] = np.nanmean(np.stack([TE.regret_phi(_fit_predict(ks, sym(xr), np.stack([r["phi"] for r in xr]), Sdv, Ydv, [Ste], nbin, s)[0])
                                                    for s in seeds]), 0)
            out["S_abundant_n_candidates"] = len(xr)

    # B2 bilinear scalar (secondary), chunked prediction (verification SHOULD-1)
    regimes = {"full16": decide.goals(dim, 16, 2002), "sub4": decide.goals_in_subspace(dim, 16, min(4, dim - 1), 3003)}
    Xtr_b, Xdv_b, Xte_b = Xtr, Xdv, Xte
    for regime, Gtr in regimes.items():
        expand = lambda X, Y: (np.repeat(X, len(Gtr), 0), np.tile(Gtr, (len(X), 1)), (Y @ Gtr.T).reshape(-1))
        b2 = heads.ScalarHead("bilinear", seed=17).fit(*expand(Xtr_b, Ytr), *expand(Xdv_b, Ydv))
        U = np.zeros((len(sp["test"]), N_TEST_GOALS), np.float32)
        for s0 in range(0, len(sp["test"]), 64):
            Xc = Xte_b[s0: s0 + 64]
            U[s0: s0 + len(Xc)] = b2.predict(np.repeat(Xc, N_TEST_GOALS, 0), np.tile(Gte, (len(Xc), 1))).reshape(len(Xc), N_TEST_GOALS)
        pp[f"B2_bilinear@{regime}"] = TE.regret(TE.cube(U))
    return out
