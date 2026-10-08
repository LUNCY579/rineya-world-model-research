"""Follow-ups (CPU only):
  C2a  matched-difficulty: symbolic ridge / L8 ridge (identical across arms) on true vs cf targets
  C2b  selectivity CI on discriminating items + expected selectivity if the model wrote the true Δ
  C1d  R new value of the instructed register, by instruction kind, all layers, both arms
  C1e  P applicable-only Δ: copy floor; per-layer both arms
  C4g  L decay split into changed bits vs sign bits, vs touched rule
  C4h  per-event (single-event) Δ decodable from the k=2/k=4 answer slot
  C3r  reference fragility: within-arm probe transfer one-frame -> k1 (same transitions)
"""
from __future__ import annotations

import json
import sys

import numpy as np

import vh
from vh import env, controls, fidelity, heads, decide

OUT = vh.ROOT / "docs" / "verification" / "c_more_results.json"
res = {}
ARMS = ("sparse", "supa")


def log(k, v):
    res[k] = v
    print(k, json.dumps(v, default=float), flush=True)


def delta_true(fam):
    return lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1])


def delta_cf(fam):
    return lambda r: fidelity.delta_of(fam, r["s0"], controls.rollout_cf(fam, r["s0"], r["events"]))


def sym_ridge(sp, fam, tgt, nb):
    enc = {s: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in x]) for s, x in sp.items()}
    Y = {s: np.stack([tgt(r) for r in x]) for s, x in sp.items()}
    rg = heads.Ridge().fit(enc["train"], Y["train"], enc["dev"], Y["dev"], nb)
    return vh.exact_items(rg.predict(enc["test"]), Y["test"])


def sym_mlp(sp, fam, tgt, nb):
    enc = {s: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in x]) for s, x in sp.items()}
    Y = {s: np.stack([tgt(r) for r in x]) for s, x in sp.items()}
    m = heads.MLPHead().fit(enc["train"], Y["train"], enc["dev"], Y["dev"], nb)
    return vh.exact_items(m.predict(enc["test"]), Y["test"])


which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["C2a", "C2b", "C1d", "C1e", "C4g", "C4h", "C3r"]

if "C2a" in which:
    for fam in vh.FAMS:
        for frame in ("one", 1):
            sp = vh.sp3(fam, frame)
            te = sp["test"]
            tt, tc = delta_true(fam), delta_cf(fam)
            nb = len(tt(te[0]))
            disc = np.array([not np.array_equal(tt(r), tc(r)) for r in te])
            e = {"n_disc": int(disc.sum()),
                 "base_rate_true": [round(float(x), 3) for x in np.stack([tt(r) for r in sp["train"]]).mean(0)],
                 "base_rate_cf": [round(float(x), 3) for x in np.stack([tc(r) for r in sp["train"]]).mean(0)]}
            for nm, fn in (("S_ridge", sym_ridge), ("S_mlp", sym_mlp)):
                a, b = fn(sp, fam, tt, nb), fn(sp, fam, tc, nb)
                e[nm] = {"true": float(a.mean()), "cf": float(b.mean()), "true_disc": float(a[disc].mean()), "cf_disc": float(b[disc].mean())}
            for L in (4, 8):
                pa = vh.probe("supa", sp, tt, nb, layer=L); pb = vh.probe("supa", sp, tc, nb, layer=L)
                a, b = vh.exact_items(pa["P"], pa["Y"]), vh.exact_items(pb["P"], pb["Y"])
                e[f"L{L}_ridge(identical arms)"] = {"true": float(a.mean()), "cf": float(b.mean()), "true_disc": float(a[disc].mean()), "cf_disc": float(b[disc].mean())}
            log(f"C2a|{fam}|{frame}", e)

if "C2b" in which:
    for fam in vh.FAMS:
        for frame in ("one", 1):
            sp = vh.sp3(fam, frame)
            te = sp["test"]
            tt, tc = delta_true(fam), delta_cf(fam)
            nb = len(tt(te[0]))
            disc = np.array([not np.array_equal(tt(r), tc(r)) for r in te])
            rows_d = [r for r, d in zip(te, disc) if d]
            e = {"n_disc": int(disc.sum()), "n_disc_parents": len({r["parent_id"] for r in rows_d})}
            sel = {}
            for a in ARMS:
                pt, pc = vh.probe(a, sp, tt, nb), vh.probe(a, sp, tc, nb)
                xt, xc = vh.exact_items(pt["P"], pt["Y"])[disc], vh.exact_items(pc["P"], pc["Y"])[disc]
                sel[a] = (xt, xc)
                e[a] = {"true_disc": float(xt.mean()), "cf_disc": float(xc.mean()),
                        "selectivity_ci": vh.paired_boot(rows_d, xt, xc),
                        "max_attainable_selectivity(1-cf_disc)": float(1 - xc.mean())}
            e["sel_diff_sparse_minus_supa"] = vh.paired_boot(rows_d, sel["sparse"][0] - sel["sparse"][1], sel["supa"][0] - sel["supa"][1])
            # expected selectivity if the model wrote the TRUE Δ linearly into its state (oracle injection)
            if fam == "ledger_replay":
                rng = np.random.default_rng(0)
                A = vh.arm("sparse")
                L = 16
                ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
                X = {s: A.X_last(ids[s], L).astype(np.float64) for s in sp}
                sdX = X["train"].std(0).mean()
                M = rng.standard_normal((nb, 1024))
                Yt = {s: np.stack([tt(r) for r in x]) for s, x in sp.items()}
                Yc = {s: np.stack([tc(r) for r in x]) for s, x in sp.items()}
                inj = {}
                for g in (0.0, 0.02, 0.05, 0.1, 0.2):
                    Xg = {s: X[s] + g * sdX * ((Yt[s] - 0.5) @ M) for s in sp}
                    rt = heads.Ridge().fit(Xg["train"], Yt["train"], Xg["dev"], Yt["dev"], nb)
                    rc = heads.Ridge().fit(Xg["train"], Yc["train"], Xg["dev"], Yc["dev"], nb)
                    xt = vh.exact_items(rt.predict(Xg["test"]), Yt["test"])[disc]
                    xc = vh.exact_items(rc.predict(Xg["test"]), Yc["test"])[disc]
                    inj[str(g)] = {"true_disc": float(xt.mean()), "cf_disc": float(xc.mean()), "sel": float(xt.mean() - xc.mean())}
                e["oracle_injection_sparse_L16"] = inj
            log(f"C2b|{fam}|{frame}", e)

if "C1d" in which:
    fam = "register_machine"
    for frame in ("one", 1):
        sp = vh.sp3(fam, frame)
        for kind in ("swap", "add", "conditional", "mix", None):
            sub = (lambda r, k=kind: r["events"][0]["kind"] == k) if kind else None
            e = {}
            for target_name, fn in (("new", lambda r: r["states"][-1][r["events"][0]["register"]] / 97.0),
                                    ("old", lambda r: r["s0"][r["events"][0]["register"]] / 97.0),
                                    ("other_old", lambda r: r["s0"][r["events"][0]["other"]] / 97.0)):
                for a in ARMS:
                    curve = {}
                    for L in (8, 12, 16, 20, 24, 28):
                        p = vh.probe(a, sp, fn, 0, layer=L, sub=sub)
                        curve[L] = round(vh.r2(p["Y"], p["P"]), 3)
                    e[f"{target_name}|{a}"] = curve
                e["n_test"] = len(p["Y"])
            log(f"C1d|R|{frame}|{kind or 'all'}", e)

if "C1e" in which:
    fam = "prerequisite_plan"
    for frame in ("one", 1):
        sp = vh.sp3(fam, frame)
        appl = lambda r: env.applicable(r["s0"], r["events"][0])
        te = [r for r in sp["test"] if appl(r)]
        copy = float(np.mean([not delta_true(fam)(r).any() for r in te]))
        tr = [r for r in sp["train"] if appl(r)]
        D = np.stack([delta_true(fam)(r) for r in tr])
        maj = (D.mean(0) > 0.5).astype(float)
        majacc = float(np.mean([np.array_equal(delta_true(fam)(r), maj) for r in te]))
        e = {"n_test_applicable": len(te), "copy_floor": copy, "majority_floor": majacc}
        for a in ARMS:
            curve = {}
            pp = {}
            for L in (4, 8, 12, 16, 20, 24, 28):
                p = vh.probe(a, sp, delta_true(fam), 8, layer=L, sub=appl)
                curve[L] = round(float(vh.exact_items(p["P"], p["Y"]).mean()), 3)
            e[a] = curve
        log(f"C1e|P|{frame}|applicable_only", e)

if "C4g" in which:
    fam = "ledger_replay"
    for k in (1, 2, 4):
        sp = vh.sp3(fam, k)
        te = sp["test"]
        e = {}
        touched = np.stack([fidelity.touched_delta(fam, r["s0"], r["events"]) for r in te])[:, 4:]
        Dt = np.stack([delta_true(fam)(r) for r in te])
        e["touched_rule_changed_bits_exact"] = float(((touched > 0.5) == (Dt[:, 4:] > 0.5)).all(1).mean())
        e["copy_changed_bits_exact"] = float((Dt[:, 4:] < 0.5).all(1).mean())
        e["copy_sign_bits_exact"] = float((Dt[:, :4] < 0.5).all(1).mean())
        for a in ARMS:
            for nm, sl in (("changed", slice(4, 8)), ("sign", slice(0, 4))):
                p = vh.probe(a, sp, lambda r, sl=sl: delta_true(fam)(r)[sl], 4)
                e[f"{a}|{nm}"] = {"layer": p["layer"], "exact": round(float(vh.exact_items(p["P"], p["Y"]).mean()), 3),
                                  "balanced_acc": round(vh.balanced_acc(p["P"], p["Y"]), 3)}
        log(f"C4g|L|k{k}", e)

if "C4h" in which:
    # From the answer slot of the k-event prompt, decode the change mask each event j would cause
    # ON ITS OWN applied to s0 (no composition needed). High = every event is still encoded.
    for fam in ("ledger_replay", "register_machine"):
        for k in (2, 4):
            sp = vh.sp3(fam, k)
            e = {}
            for j in range(k):
                fn = lambda r, j=j: fidelity.delta_of(fam, r["s0"], env.step(fam, r["s0"], r["events"][j]))
                nb = len(fn(sp["train"][0]))
                for a in ARMS:
                    p = vh.probe(a, sp, fn, nb)
                    e[f"event{j + 1}_alone|{a}"] = {"layer": p["layer"], "exact": round(float(vh.exact_items(p["P"], p["Y"]).mean()), 3)}
            # reference: the same single-event target at k=1 (sparse)
            log(f"C4h|{fam}|k{k}", e)

if "C3r" in which:
    # Fragility reference: within-arm, one-frame probe applied to k1 items (same transitions,
    # different prompt wording) and vice versa.
    for fam in ("ledger_replay", "register_machine"):
        e = {}
        for a in ARMS:
            A = vh.arm(a)
            spo, spk = vh.sp3(fam, "one"), vh.sp3(fam, 1)
            tt = delta_true(fam)
            nb = len(tt(spo["train"][0]))
            for L in (16, 20):
                Y = {s: np.stack([tt(r) for r in x]) for s, x in spo.items()}
                rg = heads.Ridge().fit(A.X_last([r["item_id"] for r in spo["train"]], L), Y["train"],
                                       A.X_last([r["item_id"] for r in spo["dev"]], L), Y["dev"], nb)
                Xk = A.X_last([r["item_id"] for r in spk["test"]], L)
                Xo = A.X_last([r["item_id"] for r in spo["test"]], L)
                Yk = np.stack([tt(r) for r in spk["test"]])
                std_k = heads.Standardizer(A.X_last([r["item_id"] for r in spk["train"]], L))
                e[f"{a}|L{L}"] = {"one_on_one": float(vh.exact_items(rg.predict(Xo), Y["test"]).mean()),
                                  "one_on_k1_raw": float(vh.exact_items(rg.predict(Xk), Yk).mean()),
                                  "one_on_k1_restd": float(vh.exact_items(rg.predict(std_k(Xk) * rg.std.sd + rg.std.mu), Yk).mean())}
        log(f"C3r|{fam}", e)

tag = sys.argv[2] if len(sys.argv) > 2 else "all"
OUT.with_name(f"c_more_results_{tag}.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
