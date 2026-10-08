"""C3: what kind of shift does the delta make at the answer slot? (CPU only)

Per layer, Supa (A) -> Sparse (B), same items:
  - share of the shift energy ||B-A||^2 that is a constant offset (global mean, per-family mean,
    per family x frame mean) = input-independent part
  - concentration of the shift in a handful of dims beyond the top-5 massive dims
  - PCA of the item-dependent shift
  - how much of B is predicted from A by (i) A + mean shift, (ii) per-dim affine (the
    restandardisation used in R5), (iii) a full ridge linear map fit on TRAIN items
  - direction vs norm: shift after per-row RMS normalisation (what the next RMSNorm sees)
  - probe transfer after the full linear alignment (Sparse probe on mapped Supa states)
"""
from __future__ import annotations

import json

import numpy as np
import torch

import vh
from vh import fidelity

OUT = vh.ROOT / "docs" / "verification" / "c3_results.json"
A_ARM, B_ARM = vh.arm("supa"), vh.arm("sparse")
items = A_ARM.items
fam_of = np.array([it["family"] for it in items])
frame_of = np.array([it["item_id"].rsplit("-", 1)[1] for it in items])
split_of = np.array([it["split"] for it in items])
test_one = np.where((split_of == "test") & (frame_of == "one"))[0]
train_all = np.where(split_of == "train")[0]
test_all = np.where(split_of == "test")[0]


def load(armobj, rows_idx, li):
    order = np.argsort(rows_idx)
    out = np.empty((len(rows_idx), 1024), np.float64)
    out[order] = armobj.last[rows_idx[order], li, :].astype(np.float64)
    return out


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def group_mean_share(d, groups):
    """Share of ||d||^2 explained by replacing each row with its group mean."""
    tot = (d ** 2).sum()
    m = np.zeros_like(d)
    for g in np.unique(groups):
        ix = groups == g
        m[ix] = d[ix].mean(0)
    return float((m ** 2).sum() / tot)


def ridge_map(Xtr, Ytr, lam):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = torch.tensor(Xtr - mx), torch.tensor(Ytr - my)
    G = Xc.T @ Xc
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), Xc.T @ Yc).numpy()
    return lambda X: (X - mx) @ W + my


res = {"n_test_one": int(len(test_one)), "layers": {}}
for li, L in enumerate(A_ARM.layers):
    if L < 12:
        continue
    A, B = load(A_ARM, test_one, li), load(B_ARM, test_one, li)
    massive = np.argsort(-np.abs(A).mean(0))[:5]
    keep = np.setdiff1d(np.arange(1024), massive)
    e = {"massive_dims": massive.tolist()}
    for nm, cols in (("all", np.arange(1024)), ("nomassive", keep)):
        a, b = A[:, cols], B[:, cols]
        d = b - a
        m = d.mean(0)
        tot = (d ** 2).sum()
        sd = a.std(0) + 1e-6
        dc = d - m
        ev = np.linalg.svd(dc, compute_uv=False) ** 2
        cum = np.cumsum(ev) / ev.sum()
        per_dim = (d ** 2).sum(0)
        srt = np.sort(per_dim)[::-1]
        e[nm] = {
            "rel_l2": float(np.sqrt(tot) / np.linalg.norm(a)),
            "share_global_mean": float(len(d) * (m ** 2).sum() / tot),
            "share_family_mean": group_mean_share(d, fam_of[test_one]),
            "centered_shift_over_supa_centered": float(np.linalg.norm(dc) / np.linalg.norm(a - a.mean(0))),
            "median_abs_z_total": float(np.median(np.abs(d / sd))),
            "median_abs_z_meanshift_only": float(np.median(np.abs(m / sd))),
            "median_abs_z_centered_part": float(np.median(np.abs(dc / sd))),
            "share_top5_dims": float(srt[:5].sum() / tot), "share_top20_dims": float(srt[:20].sum() / tot),
            "share_top50_dims": float(srt[:50].sum() / tot), "share_top100_dims": float(srt[:100].sum() / tot),
            "centered_pcs_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
            "centered_pcs_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
            "centered_participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "cka": cka(a, b),
            "median_cos_rowwise": float(np.median((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))),
            "median_norm_ratio_B_over_A": float(np.median(np.linalg.norm(b, axis=1) / np.linalg.norm(a, axis=1))),
        }
        if nm == "all":
            e["share_of_shift_in_massive_dims"] = float((d[:, massive] ** 2).sum() / tot)
            # direction-only shift: RMS-normalise each row over all dims (what the next RMSNorm sees)
            an = a / np.sqrt((a ** 2).mean(1, keepdims=True)); bn = b / np.sqrt((b ** 2).mean(1, keepdims=True))
            dn = bn - an
            e["rmsnormed"] = {"rel_l2_all": float(np.linalg.norm(dn) / np.linalg.norm(an)),
                              "rel_l2_nomassive": float(np.linalg.norm(dn[:, keep]) / np.linalg.norm(an[:, keep])),
                              "share_global_mean_nomassive": float(len(dn) * (dn[:, keep].mean(0) ** 2).sum() / (dn[:, keep] ** 2).sum())}
    # all-items version (all frames/families, test split) of the mean-share
    At, Bt = load(A_ARM, test_all, li)[:, keep], load(B_ARM, test_all, li)[:, keep]
    dt = Bt - At
    tot = (dt ** 2).sum()
    e["all_test_items_nomassive"] = {
        "n": int(len(test_all)), "rel_l2": float(np.sqrt(tot) / np.linalg.norm(At)),
        "share_global_mean": float(len(dt) * (dt.mean(0) ** 2).sum() / tot),
        "share_family_mean": group_mean_share(dt, fam_of[test_all]),
        "share_family_x_frame_mean": group_mean_share(dt, np.char.add(fam_of[test_all], frame_of[test_all])),
    }
    # predictability of B from A (fit on TRAIN items of all frames, evaluate on TEST one-frame)
    Atr, Btr = load(A_ARM, train_all, li), load(B_ARM, train_all, li)
    shift_energy = ((B - A) ** 2).sum()
    var_B = ((B - B.mean(0)) ** 2).sum()
    preds = {"identity": A, "A_plus_train_mean_shift": A + (Btr - Atr).mean(0),
             "per_dim_affine": Btr.mean(0) + (A - Atr.mean(0)) * (Btr.std(0) + 1e-6) / (Atr.std(0) + 1e-6)}
    for lam in (1e2, 1e3, 1e4):
        preds[f"ridge_linear_map_lam{lam:g}"] = ridge_map(Atr, Btr, lam)(A)
    e["predict_B_from_A"] = {k: {"share_of_shift_explained": float(1 - ((B - p) ** 2).sum() / shift_energy),
                                 "r2_of_B": float(1 - ((B - p) ** 2).sum() / var_B)} for k, p in preds.items()}
    res["layers"][L] = e
    print(L, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in e.items() if k != "massive_dims"}), flush=True)
    del Atr, Btr

# ---------------------------------------------------------------- transfer after full linear alignment
fam = "ledger_replay"
for famx in ("ledger_replay", "register_machine"):
    sp = vh.sp3(famx, "one")
    tgt = lambda r, f=famx: fidelity.delta_of(f, r["s0"], r["states"][-1])
    nb = len(tgt(sp["train"][0]))
    for L in (16, 20):
        li = A_ARM.layers.index(L)
        ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
        Y = {s: np.stack([tgt(r) for r in x]) for s, x in sp.items()}
        out = {}
        Atr, Btr = load(A_ARM, train_all, li), load(B_ARM, train_all, li)
        for src, dst, Xs_tr, Xd_tr, src_arm, dst_arm in (("sparse", "supa", Btr, Atr, B_ARM, A_ARM), ("supa", "sparse", Atr, Btr, A_ARM, B_ARM)):
            rg = vh.ridge_fit(src_arm.X_last(ids["train"], L), Y["train"], src_arm.X_last(ids["dev"], L), Y["dev"], nb)
            Xte_dst = dst_arm.X_last(ids["test"], L).astype(np.float64)
            Xte_src = src_arm.X_last(ids["test"], L).astype(np.float64)
            r = {"in_arm": float(vh.exact_items(rg.predict(Xte_src), Y["test"]).mean()),
                 "raw_transfer": float(vh.exact_items(rg.predict(Xte_dst), Y["test"]).mean())}
            mshift = (Xs_tr - Xd_tr).mean(0)
            r["mean_shift_aligned"] = float(vh.exact_items(rg.predict(Xte_dst + mshift), Y["test"]).mean())
            for lam in (1e3, 1e4):
                mp = ridge_map(Xd_tr, Xs_tr, lam)
                r[f"linear_map_aligned_lam{lam:g}"] = float(vh.exact_items(rg.predict(mp(Xte_dst)), Y["test"]).mean())
            out[f"{src}_probe_on_{dst}"] = r
        res.setdefault("transfer_after_alignment", {})[f"{famx}|one|L{L}"] = out
        print("transfer", famx, L, json.dumps(out), flush=True)
        del Atr, Btr

OUT.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
