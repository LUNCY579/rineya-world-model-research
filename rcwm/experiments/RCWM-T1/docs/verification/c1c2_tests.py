"""C1/C2 alternative-explanation tests (CPU only).

C1: are the arms equal only because the probe saturates on surface-computable features?
    - L8 baseline: layers 0-8 are bit-identical across arms, so whatever is decodable at L8 is
      decodable from both arms by construction.
    - arm-sensitive targets: ledger sign-flip bits and post-event value of the named account;
      register new value of the instructed register (mod-97 arithmetic) and its parity/hi bits;
      prerequisite applicability.
C2: matched-difficulty checks of the counterfactual rules and selectivity on arithmetic targets.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import vh
from vh import env, controls, fidelity

OUT = vh.ROOT / "docs" / "verification" / "c1c2_results.json"
ROT = {a: env.ACCOUNTS[(i + 1) % 4] for i, a in enumerate(env.ACCOUNTS)}
ARMS = ("sparse", "supa")
res = {}


def clip100(v):
    return float(np.clip(v / 100.0, -3, 3))


def cf_final(r):
    return controls.rollout_cf(r["family"], r["s0"], r["events"])


def run_binary(tag, fam, frame, target, sub=None, layer=None):
    sp = vh.sp3(fam, frame)
    out = {}
    per = {}
    for a in ARMS:
        p = vh.probe(a, sp, target, nbin=len(np.atleast_1d(target(sp["train"][0]))), layer=layer, sub=sub)
        ex = vh.exact_items(p["P"], p["Y"])
        out[a] = {"layer": p["layer"], "exact": float(ex.mean()), "balanced_acc": vh.balanced_acc(p["P"], p["Y"]),
                  "base_rate_pos": float((p["Y"] > 0.5).mean()), "n": len(ex)}
        per[a] = (p["rows"], ex)
    rows = per["sparse"][0]
    out["diff_sparse_minus_supa"] = vh.paired_boot(rows, per["sparse"][1], per["supa"][1])
    # majority floor on test (train majority)
    out["agreement_items"] = float((per["sparse"][1] == per["supa"][1]).mean())
    res[tag] = out
    print(tag, json.dumps({a: out[a] for a in ARMS}), "diff", out["diff_sparse_minus_supa"], flush=True)
    return per


def run_cont(tag, fam, frame, target, sub=None):
    sp = vh.sp3(fam, frame)
    out, keep = {}, {}
    for a in ARMS:
        p = vh.probe(a, sp, target, nbin=0, sub=sub)
        out[a] = {"layer": p["layer"], "r2": vh.r2(p["Y"], p["P"]), "n": len(p["Y"])}
        keep[a] = p
    rows = keep["sparse"]["rows"]
    Ys, Ps, Pu = keep["sparse"]["Y"], keep["sparse"]["P"], keep["supa"]["P"]
    out["diff_sparse_minus_supa"] = vh.boot_stat_diff(rows, lambda i: vh.r2(Ys[i], Ps[i]), lambda i: vh.r2(Ys[i], Pu[i]))
    res[tag] = out
    print(tag, json.dumps({a: out[a] for a in ARMS}), "diff", out["diff_sparse_minus_supa"], flush=True)
    return keep


def selectivity_cont(tag, keep_true, keep_cf):
    """R2(true) - R2(cf) per arm, with parent bootstrap, and arm difference of selectivity."""
    out = {}
    rows = keep_true["sparse"]["rows"]
    for a in ARMS:
        Yt, Pt, Yc, Pc = keep_true[a]["Y"], keep_true[a]["P"], keep_cf[a]["Y"], keep_cf[a]["P"]
        out[a] = vh.boot_stat_diff(rows, lambda i: vh.r2(Yt[i], Pt[i]), lambda i: vh.r2(Yc[i], Pc[i]))
    st = {a: (keep_true[a]["Y"], keep_true[a]["P"], keep_cf[a]["Y"], keep_cf[a]["P"]) for a in ARMS}
    f = lambda a: (lambda i: vh.r2(st[a][0][i], st[a][1][i]) - vh.r2(st[a][2][i], st[a][3][i]))
    out["sel_diff_sparse_minus_supa"] = vh.boot_stat_diff(rows, f("sparse"), f("supa"))
    res[tag] = out
    print(tag, json.dumps(out), flush=True)


def main(which):
    t0 = time.time()
    # ---------------------------------------------------------------- per-layer exact-Δ curve (L8 baseline)
    if "layers" in which:
        for fam in vh.FAMS:
            for frame in ("one", 1):
                sp = vh.sp3(fam, frame)
                tgt = lambda r, fam=fam: fidelity.delta_of(fam, r["s0"], r["states"][-1])
                nb = len(tgt(sp["train"][0]))
                curve = {}
                for a in ARMS:
                    curve[a] = {}
                    for L in vh.LAYERS:
                        p = vh.probe(a, sp, tgt, nb, layer=L)
                        curve[a][L] = round(float(vh.exact_items(p["P"], p["Y"]).mean()), 4)
                res[f"layer_curve|{fam}|{frame}"] = curve
                print("layer_curve", fam, frame, curve, flush=True)

    # ---------------------------------------------------------------- ledger
    if "L" in which:
        fam = "ledger_replay"
        for frame in ("one", 1):
            run_binary(f"L|{frame}|sign_bits", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1])[:4])
            run_binary(f"L|{frame}|changed_bits", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1])[4:])
            run_binary(f"L|{frame}|cf_sign_bits", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], cf_final(r))[:4])
            run_binary(f"L|{frame}|cf_changed_bits", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], cf_final(r))[4:])
            posted = lambda r: r["events"][0]["status"] == "posted"
            kt = run_cont(f"L|{frame}|value_named_acc_after", fam, frame,
                          lambda r: clip100(r["states"][-1][r["events"][0]["account"]]), sub=posted)
            kc = run_cont(f"L|{frame}|cf_value_rot_acc_after", fam, frame,
                          lambda r: clip100(cf_final(r)[ROT[r["events"][0]["account"]]]), sub=posted)
            selectivity_cont(f"L|{frame}|SEL_value_after", kt, kc)
            kt = run_cont(f"L|{frame}|value_named_acc_before", fam, frame,
                          lambda r: clip100(r["s0"][r["events"][0]["account"]]), sub=posted)
            kc = run_cont(f"L|{frame}|value_rot_acc_before", fam, frame,
                          lambda r: clip100(r["s0"][ROT[r["events"][0]["account"]]]), sub=posted)
            selectivity_cont(f"L|{frame}|SEL_value_before(retrieval only)", kt, kc)
            kt = run_cont(f"L|{frame}|dvalue_named", fam, frame,
                          lambda r: clip100(r["states"][-1][r["events"][0]["account"]] - r["s0"][r["events"][0]["account"]]), sub=posted)
            kc = run_cont(f"L|{frame}|cf_dvalue_rot", fam, frame,
                          lambda r: clip100(cf_final(r)[ROT[r["events"][0]["account"]]] - r["s0"][ROT[r["events"][0]["account"]]]), sub=posted)
            selectivity_cont(f"L|{frame}|SEL_dvalue", kt, kc)
            # all four final balances (the continuous part of phi), and s0 balances (copy)
            run_cont(f"L|{frame}|final_all4", fam, frame, lambda r: np.array([clip100(r["states"][-1][a]) for a in env.ACCOUNTS]))
            run_cont(f"L|{frame}|s0_all4", fam, frame, lambda r: np.array([clip100(r["s0"][a]) for a in env.ACCOUNTS]))

    # ---------------------------------------------------------------- register
    if "R" in which:
        fam = "register_machine"
        arith = lambda r: r["events"][0]["kind"] in ("mix", "conditional")
        mix = lambda r: r["events"][0]["kind"] == "mix"
        for frame in ("one", 1):
            reg = lambda r: r["events"][0]["register"]
            run_cont(f"R|{frame}|new_value_instructed", fam, frame, lambda r: r["states"][-1][reg(r)] / 97.0)
            run_cont(f"R|{frame}|old_value_instructed", fam, frame, lambda r: r["s0"][reg(r)] / 97.0)
            kt = run_cont(f"R|{frame}|mix|new_value", fam, frame, lambda r: r["states"][-1][reg(r)] / 97.0, sub=mix)
            kc = run_cont(f"R|{frame}|mix|cf_new_value", fam, frame, lambda r: cf_final(r)[reg(r)] / 97.0, sub=mix)
            selectivity_cont(f"R|{frame}|mix|SEL_new_value", kt, kc)
            run_binary(f"R|{frame}|arith|parity_new", fam, frame, lambda r: np.array([r["states"][-1][reg(r)] % 2]), sub=arith)
            run_binary(f"R|{frame}|arith|cf_parity_new", fam, frame, lambda r: np.array([cf_final(r)[reg(r)] % 2]), sub=arith)
            run_binary(f"R|{frame}|arith|hi_new", fam, frame, lambda r: np.array([int(r["states"][-1][reg(r)] > 48)]), sub=arith)
            run_binary(f"R|{frame}|arith|cf_hi_new", fam, frame, lambda r: np.array([int(cf_final(r)[reg(r)] > 48)]), sub=arith)
            run_binary(f"R|{frame}|arith|delta", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1]), sub=arith)
            run_binary(f"R|{frame}|arith|cf_delta", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], cf_final(r)), sub=arith)
            run_binary(f"R|{frame}|delta_hi_par_only", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1])[:6])

    # ---------------------------------------------------------------- prerequisite
    if "P" in which:
        fam = "prerequisite_plan"
        for frame in ("one", 1):
            run_binary(f"P|{frame}|applicable", fam, frame, lambda r: np.array([int(env.applicable(r["s0"], r["events"][0]))]))
            run_binary(f"P|{frame}|cf_applicable", fam, frame,
                       lambda r: np.array([int(env.applicable(r["s0"], dict(r["events"][0], pre_on=r["events"][0]["pre_off"], pre_off=r["events"][0]["pre_on"])))]))
            appl = lambda r: env.applicable(r["s0"], r["events"][0])
            run_binary(f"P|{frame}|applicable_only|delta", fam, frame, lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1]), sub=appl)
            run_binary(f"P|{frame}|final_bits", fam, frame, lambda r: np.array(env.bits(r["states"][-1]), np.float32))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "layers,L,R,P"
    main(which.split(","))
    tag = sys.argv[2] if len(sys.argv) > 2 else "all"
    p = OUT.with_name(f"c1c2_results_{tag}.json")
    p.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
