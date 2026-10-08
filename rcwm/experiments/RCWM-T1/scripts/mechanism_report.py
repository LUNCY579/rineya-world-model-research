"""Mechanism experiment M: how much does each weight delta move the model ON-task (transition
prompts) vs OFF-task (neutral preservation-slice prompts)?

  python scripts/mechanism_report.py --bank bank_pilot --arms sparse,exp,transition,randdelta,sparse_a0.5,sparse_a2.0,qwen3

Fixed after verification bug B1: all shift metrics are computed WITHIN a prompt group (each
transition family's TEST one-frame items; the off-task set as one group), never pooled across
families (between-family variance is 92-97% of the answer-slot variance and dominated pooled CKA).

Per arm X vs Supa, per group and layer:
  rel_l2_nomassive   ||X - S|| / ||S|| over dims excluding Supa's top-5 massive dims
  offset_share       fraction of shift energy explained by the group-constant mean shift
  median_abs_z       median |X - S| in units of Supa's within-group across-item SD
  cka_centered       linear CKA within the group (centered)
  lin_r2             held-out R^2 of a ridge map S -> X (half/half split), i.e. is X ~ linear(S)?
Off-task behaviour: first-token KL(Supa || X) and top-1 agreement on the neutral prompts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import data  # noqa: E402


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def lin_r2(S, X, lam=10.0):
    n = len(S) // 2
    A, B = S[:n], X[:n]
    mu_a, mu_b = A.mean(0), B.mean(0)
    W = np.linalg.solve((A - mu_a).T @ (A - mu_a) + lam * np.eye(A.shape[1]), (A - mu_a).T @ (B - mu_b))
    P = (S[n:] - mu_a) @ W + mu_b
    return float(1 - ((X[n:] - P) ** 2).sum() / ((X[n:] - X[n:].mean(0)) ** 2).sum())


def group_shift(S, X, massive):
    keep = np.setdiff1d(np.arange(S.shape[1]), massive)
    d = (X - S)[:, keep]
    mean = d.mean(0, keepdims=True)
    return {"rel_l2_nomassive": float(np.linalg.norm(d) / np.linalg.norm(S[:, keep])),
            "offset_share": float((np.linalg.norm(mean) ** 2 * len(d)) / (np.linalg.norm(d) ** 2 + 1e-12)),
            "median_abs_z": float(np.median(np.abs(d / (S[:, keep].std(0) + 1e-6)))),
            "cka_centered": cka(S[:, keep], X[:, keep]),
            "lin_r2": lin_r2(S[:, keep], X[:, keep])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--base", default="supa")
    a = ap.parse_args()
    run = ROOT / "runs" / a.bank
    base = data.Arm(a.bank, a.base)
    groups = {}
    for fam in ("ledger_replay", "prerequisite_plan", "register_machine"):
        groups[fam] = [it["item_id"] for it in base.items if it["split"] == "test" and it["family"] == fam and it["item_id"].endswith("-one")]
    off_b = run / a.base / "offtask_last.npy"
    Hb_off = np.load(off_b).astype(np.float64) if off_b.exists() else None
    LPb = np.load(run / a.base / "offtask_logprobs.npy", mmap_mode="r") if (run / a.base / "offtask_logprobs.npy").exists() else None
    report = {"base": a.base, "groups": {g: len(v) for g, v in groups.items()}, "arms": {}}
    for armname in a.arms.split(","):
        if not (run / armname / "index.json").exists():
            print(f"{armname}: not captured yet"); continue
        X = data.Arm(a.bank, armname)
        prov = json.loads((run / armname / "provenance.json").read_text(encoding="utf-8")) if (run / armname / "provenance.json").exists() else {}
        rep = {"delta_stats": prov.get("delta_stats"), "on": {}, "off": {}}
        for li, L in enumerate(base.layers):
            if L == 0:
                continue
            for g, ids in groups.items():
                S = base.X_last(ids, L).astype(np.float64)
                massive = np.argsort(-np.abs(S).mean(0))[:5]
                rep["on"].setdefault(g, {})[L] = group_shift(S, X.X_last(ids, L).astype(np.float64), massive)
            if Hb_off is not None and (run / armname / "offtask_last.npy").exists():
                Ho = np.load(run / armname / "offtask_last.npy").astype(np.float64)
                S = Hb_off[:, li]
                massive = np.argsort(-np.abs(S).mean(0))[:5]
                rep["off"][L] = group_shift(S, Ho[:, li], massive)
        if LPb is not None and (run / armname / "offtask_logprobs.npy").exists():
            LPx = np.load(run / armname / "offtask_logprobs.npy", mmap_mode="r")
            kls, agree = [], []
            for i in range(LPb.shape[0]):
                p = LPb[i].astype(np.float64); q = LPx[i].astype(np.float64)
                kls.append(float((np.exp(p) * (p - q)).sum()))
                agree.append(int(np.argmax(p) == np.argmax(q)))
            rep["offtask_first_token"] = {"kl_mean": float(np.mean(kls)), "kl_median": float(np.median(kls)),
                                          "kl_p90": float(np.quantile(kls, 0.9)), "top1_agreement": float(np.mean(agree))}
        report["arms"][armname] = rep
        for L in (16, 24):
            on = {g[:6]: round(v[L]["rel_l2_nomassive"], 3) for g, v in rep["on"].items()}
            onr = {g[:6]: round(v[L]["offset_share"], 2) for g, v in rep["on"].items()}
            msg = f"{armname:<12} L{L} on relL2 {on} offset {onr}"
            if L in rep["off"]:
                o = rep["off"][L]
                msg += f" | off relL2 {o['rel_l2_nomassive']:.3f} offset {o['offset_share']:.2f} cka {o['cka_centered']:.3f}"
            print(msg, flush=True)
        if "offtask_first_token" in rep:
            f = rep["offtask_first_token"]
            print(f"{armname:<12} off-task first-token KL mean {f['kl_mean']:.4f} median {f['kl_median']:.4f} p90 {f['kl_p90']:.4f} top1 {f['top1_agreement']:.3f}", flush=True)
    (run / "analysis" / "mechanism.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
