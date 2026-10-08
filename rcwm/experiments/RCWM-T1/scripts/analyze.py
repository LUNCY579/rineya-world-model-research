"""Analyse captured arms: consequence fidelity (latent vs generative vs floors), latent tracking at
action boundaries, and goal-conditioned decisions (B3 vs matched scalar B2 vs generative vs no-LLM).

  python scripts/analyze.py --bank bank_pilot --arms sparse,supa [--decisions] [--tracking]

Selection (layer, λ, LR) uses DEV only; every reported number is on TEST parents.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import data, decide, env, heads  # noqa: E402

FAMS = env.FAMILIES
KS = (1, 2, 4)
N_GOALS = 16


def nbin_of(fam):
    return data.PHI_SPEC[fam]["bin"]


def fidelity_metrics(P: np.ndarray, Y: np.ndarray, fam: str) -> dict:
    nb = nbin_of(fam)
    out = {}
    if nb:
        acc = ((P[:, :nb] > 0.5) == (Y[:, :nb] > 0.5))
        out["bin_acc"] = float(acc.mean())
        out["bin_all_exact"] = float(acc.all(1).mean())
        out["bin_acc_per_feature"] = [float(x) for x in acc.mean(0)]
    if Y.shape[1] > nb:
        yc, pc = Y[:, nb:], P[:, nb:]
        out["cont_r2"] = float(np.mean(1 - ((yc - pc) ** 2).sum(0) / (((yc - yc.mean(0)) ** 2).sum(0) + 1e-9)))
        out["cont_mae"] = float(np.abs(yc - pc).mean())
    return out


def split_rows(rows, fam, k, split):
    return [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == split]


def floors(rows, fam, k):
    tr, te = split_rows(rows, fam, k, "train"), split_rows(rows, fam, k, "test")
    Ytr = np.stack([r["phi"] for r in tr]); Yte = np.stack([r["phi"] for r in te])
    nb = nbin_of(fam)
    maj = np.tile(Ytr.mean(0, keepdims=True), (len(te), 1))
    maj[:, :nb] = (maj[:, :nb] > 0.5).astype(np.float32)
    copy = np.stack([data.phi_from_state(fam, r["s0"], r["events"], r["s0"]) for r in te])
    return {"majority": fidelity_metrics(maj, Yte, fam), "copy_start": fidelity_metrics(copy, Yte, fam)}


def latent_fidelity(arm: data.Arm, rows, fam, k, use_mlp=True):
    tr, dv, te = (split_rows(rows, fam, k, s) for s in ("train", "dev", "test"))
    Ytr, Ydv, Yte = (np.stack([r["phi"] for r in x]) for x in (tr, dv, te))
    nb = nbin_of(fam)
    per_layer = {}
    best = None
    for L in arm.layers:
        Xtr, Xdv = arm.X_last([r["item_id"] for r in tr], L), arm.X_last([r["item_id"] for r in dv], L)
        rg = heads.Ridge().fit(Xtr, Ytr, Xdv, Ydv, nb)
        per_layer[L] = rg.dev_score
        if best is None or rg.dev_score > best[0]:
            best = (rg.dev_score, L, rg)
    _, Lb, rg = best
    Xte = arm.X_last([r["item_id"] for r in te], Lb)
    res = {"layer": Lb, "dev_score_by_layer": per_layer, "ridge": fidelity_metrics(rg.predict(Xte), Yte, fam),
           "ridge_lambda": rg.lam}
    preds = {"ridge": rg.predict(Xte)}
    if use_mlp:
        Xtr, Xdv = arm.X_last([r["item_id"] for r in tr], Lb), arm.X_last([r["item_id"] for r in dv], Lb)
        mh = heads.MLPHead().fit(Xtr, Ytr, Xdv, Ydv, nb)
        preds["mlp"] = mh.predict(Xte)
        res["mlp"] = fidelity_metrics(preds["mlp"], Yte, fam)
        res["mlp_lr"] = mh.lr
    return res, preds, Lb


def symbolic_fidelity(rows, fam, k, n_train_parents=None, seed=17):
    tr, dv, te = (split_rows(rows, fam, k, s) for s in ("train", "dev", "test"))
    if n_train_parents:
        keep = sorted({r["parent_id"] for r in tr})[:n_train_parents]
        tr = [r for r in tr if r["parent_id"] in set(keep)]
    enc = lambda x: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in x])
    Ytr, Ydv, Yte = (np.stack([r["phi"] for r in x]) for x in (tr, dv, te))
    mh = heads.MLPHead(seed=seed).fit(enc(tr), Ytr, enc(dv), Ydv, nbin_of(fam))
    P = mh.predict(enc(te))
    return fidelity_metrics(P, Yte, fam), P


def load_gen(arm_dir: Path):
    out = {}
    p = arm_dir / "gen_e2e.jsonl"
    if p.exists():
        out["e2e"] = {r["item_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    for mode in ("chain_one", "chain_m1"):
        p = arm_dir / f"gen_{mode}.jsonl"
        if p.exists():
            out[mode] = {r["cand_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    return out


def gen_phi(gen, mode, r):
    """φ̂ from a generative arm for test row r, or None if unparseable."""
    fam = r["family"]
    if mode == "e2e":
        g = gen["e2e"].get(r["item_id"])
        st = None if g is None else env.parse_state(fam, g["text"])
    else:
        g = gen[mode].get(r["cand_id"])
        s = None if g is None else g["states"][r["k"]]
        st = None if s is None else json.loads(s)
    return None if st is None else data.phi_from_state(fam, r["s0"], r["events"], st)


def gen_fidelity(gen, rows, fam, k):
    te = split_rows(rows, fam, k, "test")
    Y = np.stack([r["phi"] for r in te])
    out = {}
    for mode in ("e2e", "chain_one", "chain_m1"):
        if mode not in gen:
            continue
        ph = [gen_phi(gen, mode, r) for r in te]
        parsed = np.array([p is not None for p in ph])
        # unparseable -> counts as wrong on every binary feature (score against an impossible value)
        P = np.stack([p if p is not None else np.full(Y.shape[1], -9.0, np.float32) for p in ph])
        m = fidelity_metrics(P, Y, fam)
        m["parse_rate"] = float(parsed.mean())
        m["state_exact"] = float(np.mean([(p is not None) and (env.dumps(json.loads(gs)) == r["gold"])
                                          for p, gs, r in zip(ph, [gen_state_str(gen, mode, r) for r in te], te)]))
        out[mode] = m
    return out


def gen_state_str(gen, mode, r):
    if mode == "e2e":
        g = gen["e2e"].get(r["item_id"])
        st = None if g is None else env.parse_state(r["family"], g["text"])
        return "null" if st is None else env.dumps(st)
    g = gen[mode].get(r["cand_id"])
    s = None if g is None else g["states"][r["k"]]
    return "null" if s is None else s


def tracking(arm: data.Arm, rows, fam, H=4):
    """Latent state at action boundary j (k=H item) -> φ of the prefix state after j events.
    Ridge per layer; layer chosen on DEV per j. Compared with the answer-slot readout at k=j."""
    cands = {s: [r for r in rows if r["family"] == fam and r["k"] == H and r["split"] == s] for s in ("train", "dev", "test")}
    out = {}
    for j in range(1, H + 1):
        Y = {s: np.stack([data.phi_from_state(fam, r["s0"], r["events"][:j], r["states"][j]) for r in cands[s]]) for s in cands}
        best = None
        for L in arm.layers:
            X = {s: arm.X_event([r["item_id"] for r in cands[s]], j, L) for s in ("train", "dev")}
            rg = heads.Ridge().fit(X["train"], Y["train"], X["dev"], Y["dev"], nbin_of(fam))
            if best is None or rg.dev_score > best[0]:
                best = (rg.dev_score, L, rg)
        _, Lb, rg = best
        Xte = arm.X_event([r["item_id"] for r in cands["test"]], j, Lb)
        out[j] = {"layer": Lb, **fidelity_metrics(rg.predict(Xte), Y["test"], fam)}
    return out


def decisions(arm: data.Arm, rows, fam, k, Lb, preds_te, gen, label_sizes=(8, 16, 32, None)):
    """Goal-conditioned selection among K candidates on TEST parents under held-out goals."""
    dim = data.PHI_SPEC[fam]["bin"] + data.PHI_SPEC[fam]["cont"]
    Gtr, Gte = decide.goals(dim, N_GOALS, 2002), decide.goals(dim, N_GOALS, 1001)
    te = split_rows(rows, fam, k, "test")
    pids = sorted({r["parent_id"] for r in te})
    K = max(int(r["cand_id"].rsplit("-c", 1)[1]) for r in te) + 1
    pos = {r["item_id"]: i for i, r in enumerate(te)}

    def cube(phi_rows: np.ndarray) -> np.ndarray:            # [n_test_rows, dim] -> [P, K, G]
        U = phi_rows @ Gte.T
        out = np.zeros((len(pids), K, N_GOALS), np.float32)
        for pi, pid in enumerate(pids):
            for c in range(K):
                out[pi, c] = U[pos[f"{pid}-c{c}-k{k}"]]
        return out

    Ytrue = np.stack([r["phi"] for r in te])
    U_true = cube(Ytrue)
    rng = np.random.default_rng(17)
    res = {"n_test_parents": len(pids), "K": K, "methods": {}}
    per_parent = {}

    def add(name, U_pred):
        t = decide.regret_table(U_true, U_pred, rng)
        per_parent[name] = t
        res["methods"][name] = {"nregret": decide.bootstrap_mean(t["nregret"]), "optimal": decide.bootstrap_mean(t["optimal"])}

    add("oracle", U_true)
    add("random", np.zeros_like(U_true))
    for hname, P in preds_te.items():
        add(f"B3_{hname}", cube(P))
    for mode in ("e2e", "chain_one", "chain_m1"):
        if mode in gen:
            ph = [gen_phi(gen, mode, r) for r in te]
            U = np.stack([p if p is not None else np.full(dim, np.nan, np.float32) for p in ph]) @ Gte.T
            Uc = np.zeros((len(pids), K, N_GOALS), np.float32)
            for pi, pid in enumerate(pids):
                for c in range(K):
                    Uc[pi, c] = U[pos[f"{pid}-c{c}-k{k}"]]
            fill = np.nanmean(Uc, axis=1, keepdims=True)
            Uc = np.where(np.isnan(Uc), np.where(np.isnan(fill), 0.0, fill), Uc)
            add(f"G_{mode}", Uc)

    # Matched scalar evaluators and label sweep (train parents subsampled in a fixed order).
    tr_all, dv = split_rows(rows, fam, k, "train"), split_rows(rows, fam, k, "dev")
    tr_pids = sorted({r["parent_id"] for r in tr_all})
    Xdv = arm.X_last([r["item_id"] for r in dv], Lb)
    Xte = arm.X_last([r["item_id"] for r in te], Lb)
    Ydv = np.stack([r["phi"] for r in dv])

    def expand(X, Y, G):                                       # rows x goals
        n, g = X.shape[0], G.shape[0]
        return np.repeat(X, g, 0), np.tile(G, (n, 1)), (Y @ G.T).reshape(-1)

    sweep = {}
    for n in label_sizes:
        keep = set(tr_pids[:n]) if n else set(tr_pids)
        tr = [r for r in tr_all if r["parent_id"] in keep]
        Xtr = arm.X_last([r["item_id"] for r in tr], Lb)
        Ytr = np.stack([r["phi"] for r in tr])
        tag = f"n{n or len(tr_pids)}"
        mh = heads.MLPHead().fit(Xtr, Ytr, Xdv, Ydv, nbin_of(fam))
        rg = heads.Ridge().fit(Xtr, Ytr, Xdv, Ydv, nbin_of(fam))
        Htr, Wtr, Utr = expand(Xtr, Ytr, Gtr)
        Hdv, Wdv, Udv = expand(Xdv, Ydv, Gtr)
        b2 = heads.ScalarHead("mlp").fit(Htr, Wtr, Utr, Hdv, Wdv, Udv)
        b2b = heads.ScalarHead("bilinear").fit(Htr, Wtr, Utr, Hdv, Wdv, Udv)
        sym_m, sym_P = symbolic_fidelity(rows, fam, k, n_train_parents=n)
        Hte, Wte, _ = expand(Xte, Ytrue, Gte)
        U2 = b2.predict(Hte, Wte).reshape(len(te), N_GOALS)
        U2b = b2b.predict(Hte, Wte).reshape(len(te), N_GOALS)

        def cubeU(Urows):
            out = np.zeros((len(pids), K, N_GOALS), np.float32)
            for pi, pid in enumerate(pids):
                for c in range(K):
                    out[pi, c] = Urows[pos[f"{pid}-c{c}-k{k}"]]
            return out

        entry = {}
        for name, U in (("B3_mlp", cube(mh.predict(Xte))), ("B3_ridge", cube(rg.predict(Xte))),
                        ("B2_mlp", cubeU(U2)), ("B2_bilinear", cubeU(U2b)), ("S_symbolic", cube(sym_P))):
            t = decide.regret_table(U_true, U, rng)
            per_parent[f"{name}@{tag}"] = t
            entry[name] = {"nregret": decide.bootstrap_mean(t["nregret"]), "optimal": decide.bootstrap_mean(t["optimal"])}
        entry["train_parents"] = len(keep)
        sweep[tag] = entry
    res["label_sweep"] = sweep
    full = f"n{len(tr_pids)}"
    contrasts = {}
    for a_, b_ in ((f"B3_mlp@{full}", f"B2_mlp@{full}"), (f"B3_mlp@{full}", f"B2_bilinear@{full}"),
                   ("B3_mlp", "G_e2e"), ("B3_mlp", "G_chain_one"), (f"B3_mlp@{full}", f"S_symbolic@{full}")):
        if a_ in per_parent and b_ in per_parent:
            contrasts[f"{a_} - {b_}"] = decide.paired_bootstrap(per_parent[a_]["nregret"], per_parent[b_]["nregret"])
    for n in label_sizes[:-1]:
        tag = f"n{n}"
        a_, b_ = f"B3_mlp@{tag}", f"B2_bilinear@{tag}"
        contrasts[f"{a_} - {b_}"] = decide.paired_bootstrap(per_parent[a_]["nregret"], per_parent[b_]["nregret"])
    res["contrasts_nregret"] = contrasts
    return res, per_parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--families", default=",".join(FAMS))
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--decisions", action="store_true")
    ap.add_argument("--tracking", action="store_true")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    bank = data.load_bank(a.bank)
    rows = data.candidate_table(bank)
    outdir = ROOT / "runs" / a.bank / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    fams = a.families.split(",")
    ks = [int(x) for x in a.ks.split(",")]
    for armname in a.arms.split(","):
        t0 = time.time()
        arm = data.Arm(a.bank, armname)
        gen = load_gen(arm.dir)
        report = {"arm": armname, "bank": a.bank, "fidelity": {}, "tracking": {}, "decisions": {}}
        for fam in fams:
            for k in ks:
                key = f"{fam}|k{k}"
                lat, preds, Lb = latent_fidelity(arm, rows, fam, k, use_mlp=not a.no_mlp)
                entry = {"latent": lat, "floors": floors(rows, fam, k)}
                if gen:
                    entry["generative"] = gen_fidelity(gen, rows, fam, k)
                if armname in ("sparse",) or a.tag == "sym":
                    entry["symbolic"], _ = symbolic_fidelity(rows, fam, k)
                report["fidelity"][key] = entry
                if a.decisions:
                    d, _ = decisions(arm, rows, fam, k, Lb, preds, gen)
                    report["decisions"][key] = d
                print(f"[{armname}] {key} L{Lb} ridge {json.dumps({x: round(v, 4) for x, v in lat['ridge'].items() if not isinstance(v, list)})}"
                      + (f" mlp {json.dumps({x: round(v, 4) for x, v in lat['mlp'].items() if not isinstance(v, list)})}" if 'mlp' in lat else ""), flush=True)
            if a.tracking:
                report["tracking"][fam] = tracking(arm, rows, fam, bank["horizon"])
        name = f"{armname}{('_' + a.tag) if a.tag else ''}.json"
        (outdir / name).write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"[{armname}] analysis written in {time.time() - t0:.0f}s -> {outdir / name}", flush=True)


if __name__ == "__main__":
    main()
