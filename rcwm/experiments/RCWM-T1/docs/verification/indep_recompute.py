"""Independent recomputation of RCWM-T1 pilot claims C1-C4 (verification scratch; does not import
rcwmt probe/metric code; transitions reimplemented here and cross-checked against rcwmt.env.step).

Run with the sklearn venv, CUDA_VISIBLE_DEVICES=''.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
RUN = ROOT / "runs" / "bank_pilot"
OUT = ROOT / "docs" / "verification"
ACC = ("A", "B", "C", "D")
REG = ("X", "Y", "Z")
MOD = 97
PROBE_LAYERS = (4, 8, 12, 16, 20, 24, 28)   # layer 0 = embedding of a constant token at the answer slot
RNG = np.random.default_rng(12345)
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ independent transitions
def step_L(s, e, rot=False):
    s = dict(s)
    if e["status"] != "posted":
        return s
    a, o = e["account"], e["other"]
    if rot:
        r = {x: ACC[(i + 1) % 4] for i, x in enumerate(ACC)}
        a, o = r[a], r[o]
    if e["kind"] == "add":
        s[a] += e["amount"]
    elif e["kind"] == "scale":
        s[a] *= e["amount"]
    elif e["kind"] == "transfer":
        s[a] -= e["amount"]
        s[o] += e["amount"]
    else:
        raise ValueError(e)
    return s


def step_R(s, e):
    s = dict(s)
    k, r, o, v = e["kind"], e["register"], e["other"], e.get("value", 0)
    if k == "swap":
        s[r], s[o] = s[o], s[r]
    elif k == "add":
        s[r] = (s[r] + v) % MOD
    elif k == "mix":
        s[r] = (s[r] + 2 * s[o] + v) % MOD
    elif k == "conditional":
        s[r] = (s[r] + (v if s[o] > e["threshold"] else -v)) % MOD
    else:
        raise ValueError(e)
    return s


def step_P(s, e):
    if (s & e["pre_on"]) == e["pre_on"] and (s & e["pre_off"]) == 0:
        return (s & ~e["delete"]) | e["add"]
    return s


STEP = {"ledger_replay": step_L, "register_machine": step_R, "prerequisite_plan": step_P}


def delta(fam, s0, s1):
    """Binary change mask: per-feature flip of the binary consequence features vs s0."""
    if fam == "ledger_replay":
        return np.array([int((s1[a] < 0) != (s0[a] < 0)) for a in ACC] + [int(s1[a] != s0[a]) for a in ACC], np.float32)
    if fam == "register_machine":
        return np.array([int((s1[r] > 48) != (s0[r] > 48)) for r in REG] + [int(s1[r] % 2 != s0[r] % 2) for r in REG]
                        + [int(s1[r] != s0[r]) for r in REG], np.float32)
    if fam == "prerequisite_plan":
        return np.array([((s1 >> b) & 1) != ((s0 >> b) & 1) for b in range(8)], np.float32)
    raise ValueError(fam)


def rollout(fam, s0, evs, rot=False):
    st = [s0]
    for e in evs:
        st.append(step_L(st[-1], e, rot) if fam == "ledger_replay" else STEP[fam](st[-1], e))
    return st


# ------------------------------------------------------------------ data
bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
cands = []   # one row per candidate
for p in bank["parents"]:
    for c in p["candidates"]:
        cands.append({"fam": p["family"], "split": p["split"], "pid": p["parent_id"], "cid": c["cand_id"],
                      "s0": p["s0"], "events": c["events"]})

# cross-check reimplemented transitions against rcwmt.env.step (oracle only)
sys.path.insert(0, str(ROOT))
from rcwmt import env as _env  # noqa: E402
nchk = 0
for c in cands:
    s = c["s0"]
    for e in c["events"]:
        a = STEP[c["fam"]](s, e) if c["fam"] != "ledger_replay" else step_L(s, e)
        b = _env.step(c["fam"], s, e)
        assert a == b, (c["cid"], s, e, a, b)
        s = a
        nchk += 1
log(f"transition cross-check vs rcwmt.env.step: {nchk} transitions identical")

idx = {arm: json.loads((RUN / arm / "index.json").read_text(encoding="utf-8")) for arm in ("sparse", "supa")}
assert idx["sparse"]["layers"] == idx["supa"]["layers"] == [0, 4, 8, 12, 16, 20, 24, 28]
ids_sp = [it["item_id"] for it in idx["sparse"]["items"]]
ids_su = [it["item_id"] for it in idx["supa"]["items"]]
log(f"index: sparse {len(ids_sp)} supa {len(ids_su)} same order: {ids_sp == ids_su}")
POS = {arm: {it["item_id"]: i for i, it in enumerate(idx[arm]["items"])} for arm in idx}
# metadata consistency with bank
meta_bad = sum(1 for it in idx["supa"]["items"] if it["split"] != {c["cid"]: c["split"] for c in cands}.get(it["cand_id"], it["split"])) if False else 0
cid_split = {c["cid"]: c["split"] for c in cands}
meta_bad = sum(1 for it in idx["supa"]["items"] if cid_split[it["cand_id"]] != it["split"])
log(f"index split vs bank split mismatches: {meta_bad}")
LAST = {arm: np.load(RUN / arm / "last.npy", mmap_mode="r") for arm in ("sparse", "supa")}
log("last.npy shapes", {a: (v.shape, str(v.dtype)) for a, v in LAST.items()})
LI = {L: i for i, L in enumerate(idx["supa"]["layers"])}


def X_of(arm, item_ids):
    """[n, 8, 1024] float32 in the order of item_ids."""
    rows = np.array([POS[arm][i] for i in item_ids])
    o = np.argsort(rows)
    out = np.empty((len(rows), 8, 1024), np.float32)
    out[o] = np.asarray(LAST[arm][rows[o]], dtype=np.float32)
    return out


def table(fam, frame, k=1, cf=False):
    """item ids, targets, parent ids per split for family/frame."""
    res = {}
    for sp in ("train", "dev", "test"):
        rs = [c for c in cands if c["fam"] == fam and c["split"] == sp]
        ids = [c["cid"] + ("-one" if frame == "one" else f"-k{k}") for c in rs]
        Y = np.stack([delta(fam, c["s0"], rollout(fam, c["s0"], c["events"][:k], rot=cf)[-1]) for c in rs])
        res[sp] = {"ids": ids, "Y": Y, "pid": [c["pid"] for c in rs]}
    return res


def standardize(Xtr, *others):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-3
    return [(Xtr - mu) / sd] + [(x - mu) / sd for x in others], (mu, sd)


def exact(P, Y):
    return ((P > 0.5) == (Y > 0.5)).all(1)


ALPHAS = (1e-1, 1, 10, 1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def ridge_probe(Xs, T, select="exact", layers=PROBE_LAYERS):
    """Xs: dict split->[n,8,1024]; T: table. Returns dict with chosen layer/alpha and test per-item exact."""
    best = None
    for L in layers:
        (Xtr, Xdv, Xte), _ = standardize(Xs["train"][:, LI[L]], Xs["dev"][:, LI[L]], Xs["test"][:, LI[L]])
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(Xtr, T["train"]["Y"])
            Pd = m.predict(Xdv)
            s = exact(Pd, T["dev"]["Y"]).mean() if select == "exact" else ((Pd > 0.5) == (T["dev"]["Y"] > 0.5)).mean()
            if best is None or s > best[0] + 1e-12:
                best = (s, L, a, m)
    s, L, a, m = best
    (Xtr, Xdv, Xte), _ = standardize(Xs["train"][:, LI[L]], Xs["dev"][:, LI[L]], Xs["test"][:, LI[L]])
    pe = exact(m.predict(Xte), T["test"]["Y"])
    return {"layer": L, "alpha": a, "dev": float(s), "test_exact": float(pe.mean()), "per_item": pe, "model": m}


def logit_probe(Xs, T, layers=PROBE_LAYERS):
    """Independent per-feature L2 logistic regression; (layer, C) chosen jointly on DEV exact match."""
    Ytr, Ydv, Yte = T["train"]["Y"], T["dev"]["Y"], T["test"]["Y"]
    best = None
    for L in layers:
        (Xtr, Xdv, Xte), _ = standardize(Xs["train"][:, LI[L]], Xs["dev"][:, LI[L]], Xs["test"][:, LI[L]])
        for C in CS:
            Pd, Pt = np.zeros_like(Ydv), np.zeros_like(Yte)
            for j in range(Ytr.shape[1]):
                y = Ytr[:, j]
                if y.min() == y.max():
                    Pd[:, j], Pt[:, j] = y[0], y[0]
                    continue
                m = LogisticRegression(C=C, max_iter=3000, tol=1e-5).fit(Xtr, y)
                Pd[:, j] = m.predict_proba(Xdv)[:, 1]
                Pt[:, j] = m.predict_proba(Xte)[:, 1]
            s = exact(Pd, Ydv).mean()
            if best is None or s > best[0] + 1e-12:
                best = (s, L, C, exact(Pt, Yte))
    s, L, C, pe = best
    return {"layer": L, "C": C, "dev": float(s), "test_exact": float(pe.mean()), "per_item": pe}


def mlp_probe(Xs, T, L):
    (Xtr, Xdv, Xte), _ = standardize(Xs["train"][:, LI[L]], Xs["dev"][:, LI[L]], Xs["test"][:, LI[L]])
    Ytr = T["train"]["Y"].astype(int)
    best = None
    for alpha in (1e-2, 1e-1, 1.0, 10.0):
        for seed in (0,):
            m = MLPClassifier(hidden_layer_sizes=(256,), alpha=alpha, max_iter=300, random_state=seed,
                              learning_rate_init=1e-3, batch_size=128).fit(Xtr, Ytr)
            s = exact(m.predict_proba(Xdv), T["dev"]["Y"]).mean()
            if best is None or s > best[0]:
                best = (s, alpha, exact(m.predict_proba(Xte), T["test"]["Y"]))
    s, alpha, pe = best
    return {"layer": L, "alpha": alpha, "dev": float(s), "test_exact": float(pe.mean()), "per_item": pe}


def parent_means(pids, v):
    u = sorted(set(pids))
    ix = {p: i for i, p in enumerate(u)}
    s, n = np.zeros(len(u)), np.zeros(len(u))
    for p, x in zip(pids, v):
        s[ix[p]] += x
        n[ix[p]] += 1
    return s / n


def paired_boot(pids, va, vb, B=20000):
    a, b = parent_means(pids, va), parent_means(pids, vb)
    d = a - b
    bs = RNG.integers(0, len(d), (B, len(d)))
    m = d[bs].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "p_two_sided": float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))}


report = {"probe_layers": PROBE_LAYERS, "alphas": ALPHAS, "Cs": CS}

# ------------------------------------------------------------------ (1) one-step decodability, (2) cf control
for fam in ("ledger_replay", "register_machine"):
    for frame in ("one", "k1"):
        key = f"{fam}|{frame}"
        T = table(fam, frame)
        Tcf = table(fam, frame, cf=True) if fam == "ledger_replay" else None
        Xs = {arm: {sp: X_of(arm, T[sp]["ids"]) for sp in T} for arm in ("sparse", "supa")}
        ent = {"n": {sp: len(T[sp]["ids"]) for sp in T},
               "copy_floor_test": float((T["test"]["Y"].sum(1) == 0).mean()),
               "majority_test": float(exact(np.repeat((T["train"]["Y"].mean(0, keepdims=True) > 0.5).astype(float), len(T["test"]["Y"]), 0), T["test"]["Y"]).mean()),
               "base_rates_train": T["train"]["Y"].mean(0).round(3).tolist()}
        res = {}
        for arm in ("sparse", "supa"):
            r = {}
            rr = ridge_probe(Xs[arm], T, "exact")
            r["ridge_selExact"] = rr
            r["ridge_selFeatAcc"] = ridge_probe(Xs[arm], T, "featacc")
            r["logit"] = logit_probe(Xs[arm], T)
            if fam == "ledger_replay" and frame == "one":
                r["mlp"] = mlp_probe(Xs[arm], T, rr["layer"])
            if Tcf is not None:
                rc = ridge_probe(Xs[arm], Tcf, "exact")
                rcf = ridge_probe(Xs[arm], Tcf, "featacc")
                lc = logit_probe(Xs[arm], Tcf)
                disc = ~(T["test"]["Y"] == Tcf["test"]["Y"]).all(1)
                r["cf"] = {"n_disc": int(disc.sum()), "share_same": float(1 - disc.mean()),
                           "copy_floor_cf": float((Tcf["test"]["Y"].sum(1) == 0).mean())}
                for nm, (pt, pc) in {"ridge_selExact": (rr, rc), "ridge_selFeatAcc": (r["ridge_selFeatAcc"], rcf),
                                     "logit": (r["logit"], lc)}.items():
                    r["cf"][nm] = {"true_all": pt["test_exact"], "cf_all": pc["test_exact"],
                                   "true_disc": float(pt["per_item"][disc].mean()), "cf_disc": float(pc["per_item"][disc].mean()),
                                   "sel_disc": float(pt["per_item"][disc].mean() - pc["per_item"][disc].mean()),
                                   "cf_layer": pc["layer"],
                                   "sel_disc_boot": paired_boot([p for p, d in zip(T["test"]["pid"], disc) if d],
                                                                pt["per_item"][disc].astype(float), pc["per_item"][disc].astype(float))}
            res[arm] = r
            log(key, arm, " ".join(f"{nm}={v['test_exact']:.3f}(L{v['layer']})" for nm, v in r.items() if isinstance(v, dict) and "test_exact" in v),
                ("| cf " + " ".join(f"{nm}: true_disc {v['true_disc']:.3f} cf_disc {v['cf_disc']:.3f} sel {v['sel_disc']:+.3f}" for nm, v in r["cf"].items() if isinstance(v, dict)) + f" n_disc {r['cf']['n_disc']}") if "cf" in r else "")
        # paired sparse - supa
        ent["sparse_minus_supa"] = {nm: paired_boot(T["test"]["pid"], res["sparse"][nm]["per_item"].astype(float), res["supa"][nm]["per_item"].astype(float))
                                    for nm in res["sparse"] if isinstance(res["sparse"][nm], dict) and "per_item" in res["sparse"][nm]}
        # cross-arm transfer (ridge, selExact layer of source arm)
        tr = {}
        for a_, b_ in (("sparse", "supa"), ("supa", "sparse")):
            src = res[a_]["ridge_selExact"]
            L = src["layer"]
            _, (mu, sd) = standardize(Xs[a_]["train"][:, LI[L]])
            Xb = Xs[b_]["test"][:, LI[L]]
            raw = exact(src["model"].predict((Xb - mu) / sd), T["test"]["Y"]).mean()
            _, (mub, sdb) = standardize(Xs[b_]["train"][:, LI[L]])
            restd = exact(src["model"].predict((Xb - mub) / sdb), T["test"]["Y"]).mean()
            tr[f"{a_}->{b_}"] = {"layer": L, "raw": float(raw), "restd": float(restd)}
        ent["transfer_ridge"] = tr
        log(key, "sparse-supa", {nm: (round(v["diff"], 3), [round(x, 3) for x in v["ci95"]]) for nm, v in ent["sparse_minus_supa"].items()}, "transfer", tr)
        for arm in res:
            for nm, v in list(res[arm].items()):
                if isinstance(v, dict):
                    v.pop("model", None)
                    if "per_item" in v:
                        v["per_item"] = v["per_item"].astype(int).tolist()
                    if nm == "cf":
                        for v2 in v.values():
                            if isinstance(v2, dict):
                                v2.pop("model", None)
        ent["arms"] = res
        report[key] = ent
        del Xs

# ------------------------------------------------------------------ (C4 extra) multi-step decay, ridge, sparse arm
decay = {}
for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    for k in (1, 2, 4):
        T = table(fam, f"k{k}", k=k)
        Xs = {sp: X_of("sparse", T[sp]["ids"]) for sp in T}
        rr = ridge_probe(Xs, T, "exact")
        rf = ridge_probe(Xs, T, "featacc")
        cp = float((T["test"]["Y"].sum(1) == 0).mean())
        maj = float(exact(np.repeat((T["train"]["Y"].mean(0, keepdims=True) > 0.5).astype(float), len(T["test"]["Y"]), 0), T["test"]["Y"]).mean())
        decay[f"{fam}|k{k}"] = {"ridge_selExact": rr["test_exact"], "ridge_selExact_layer": rr["layer"],
                                "ridge_selFeatAcc": rf["test_exact"], "ridge_selFeatAcc_layer": rf["layer"],
                                "copy_floor": cp, "majority": maj}
        log("decay", fam, k, decay[f"{fam}|k{k}"])
        del Xs
report["decay_sparse"] = decay

# ------------------------------------------------------------------ (3) state shift supa -> sparse
def lin_cka_gram(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    K, Lm = A @ A.T, B @ B.T
    hs = (K * Lm).sum()
    return float(hs / np.sqrt((K * K).sum() * (Lm * Lm).sum()))


shift = {}
te_one = [c["cid"] + "-one" for c in cands if c["split"] == "test"]
fam_of = [c["fam"] for c in cands if c["split"] == "test"]
Xa, Xb = X_of("supa", te_one).astype(np.float64), X_of("sparse", te_one).astype(np.float64)
for L in (0, 4, 8, 12, 16, 20, 24, 28):
    A, B = Xa[:, LI[L]], Xb[:, LI[L]]
    d = B - A
    massive = np.argsort(-np.abs(A).mean(0))[:5]
    massive_max = np.argsort(-np.abs(A).max(0))[:5]
    keep = np.setdiff1d(np.arange(1024), massive)
    keep2 = np.setdiff1d(np.arange(1024), massive_max)
    sdA = A.std(0)
    ent = {"max_abs_diff": float(np.abs(d).max()), "bit_identical": bool((d == 0).all()),
           "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(A)),
           "rel_l2_nomassive": float(np.linalg.norm(d[:, keep]) / np.linalg.norm(A[:, keep])),
           "rel_l2_nomassive_bymax": float(np.linalg.norm(d[:, keep2]) / np.linalg.norm(A[:, keep2])),
           "massive_dims": massive.tolist(),
           "median_abs_z": float(np.median(np.abs(d) / (sdA + 1e-6))) if L > 0 else 0.0,
           "median_abs_meanshift_z": float(np.median(np.abs(d.mean(0)) / (sdA + 1e-6))) if L > 0 else 0.0,
           "cka_linear": lin_cka_gram(A, B) if L > 0 else 1.0,
           "cka_linear_nomassive": lin_cka_gram(A[:, keep], B[:, keep]) if L > 0 else 1.0,
           "rel_l2_mean_part_nomassive": float(np.linalg.norm(d.mean(0)[keep]) * np.sqrt(len(d)) / np.linalg.norm(A[:, keep]))}
    per = {}
    for f in ("ledger_replay", "register_machine", "prerequisite_plan"):
        m = np.array([x == f for x in fam_of])
        Af, Bf = A[m], B[m]
        df = Bf - Af
        per[f] = {"rel_l2_nomassive": float(np.linalg.norm(df[:, keep]) / np.linalg.norm(Af[:, keep])),
                  "cka_linear": lin_cka_gram(Af, Bf) if L > 0 else 1.0}
    ent["per_family"] = per
    shift[L] = ent
    log("shift", L, {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in ent.items() if k_ not in ("per_family", "massive_dims")}, "massive", ent["massive_dims"])
report["state_shift_test_one"] = {"n_items": len(te_one), "layers": shift}

# ------------------------------------------------------------------ (4) Supa generative exact accuracy
gen = {}
for line in (RUN / "supa" / "gen_e2e.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        gen[r["item_id"]] = r


def parse(fam, text):
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    try:
        v = json.loads(t)
    except Exception:
        return None
    if fam == "prerequisite_plan":
        return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 256 else None
    keys = ACC if fam == "ledger_replay" else REG
    if not isinstance(v, dict) or set(v) != set(keys) or not all(isinstance(v[q], int) and not isinstance(v[q], bool) for q in keys):
        return None
    return {q: v[q] for q in keys}


gres = {}
disagree_flag = 0
for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    for frame in ("one", "k1", "k2", "k4"):
        k = 1 if frame == "one" else int(frame[1:])
        for scope in ("test", "all"):
            rs = [c for c in cands if c["fam"] == fam and (scope == "all" or c["split"] == scope)]
            n = parsed = ex = missing = recorded_exact = 0
            for c in rs:
                iid = c["cid"] + ("-one" if frame == "one" else f"-k{k}")
                g = gen.get(iid)
                n += 1
                if g is None:
                    missing += 1
                    continue
                st = parse(fam, g["text"])
                gold = rollout(fam, c["s0"], c["events"][:k])[-1]
                parsed += st is not None
                e = st is not None and st == gold
                ex += e
                recorded_exact += bool(g.get("exact"))
                disagree_flag += int(bool(g.get("exact")) != e)
            gres[f"{fam}|{frame}|{scope}"] = {"n": n, "missing": missing, "parse_rate": parsed / n, "exact": ex / n,
                                              "recorded_exact": recorded_exact / n}
            if scope == "test":
                log("gen supa", fam, frame, gres[f"{fam}|{frame}|{scope}"])
report["gen_supa_e2e"] = gres
report["gen_supa_e2e_recorded_vs_recomputed_disagreements"] = disagree_flag
log("gen recorded-vs-recomputed exact disagreements:", disagree_flag)

(OUT / "indep_recompute_results.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
log("wrote", OUT / "indep_recompute_results.json")
