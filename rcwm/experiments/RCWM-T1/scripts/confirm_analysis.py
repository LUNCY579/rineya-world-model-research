"""PREREG-1 analysis (frozen with PREREG_1.md). Fitting on pilot TRAIN, selection on pilot DEV
(selection goals, seed 1002); every reported number is on the sealed confirmation parents (test
goals, seed 1001).

  python scripts/confirm_analysis.py                                  # the registered run
  python scripts/confirm_analysis.py --dry-run --gen-stem bank_pilot_regen   # code check on pilot TEST
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import data, decide, env, prereg1  # noqa: E402
from analyze import load_gen  # noqa: E402

N_BOOT = 10000
MIN_EFFECT = -0.10
CONFIRM_SHA = "8bc6e2f8a8c7cae5b7b66293ef12b0efc440c281f820eca9d384a659b26a6a16"
FP = {"sparse": "818be9c6b6a3", "supa": "a9952c55f5a1"}
PRIMARY_FAMS = ("ledger_replay", "register_machine")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strat_boot(diffs_by_fam: dict, seed: int = 17) -> dict:
    """Pooled = mean of family means (equal weight); parents resampled within family.
    One-sided p = (count(pooled* >= 0) + 1) / (N + 1)."""
    rng = np.random.default_rng(seed)
    clean = {f: d[~np.isnan(d)] for f, d in diffs_by_fam.items()}
    for f, d in clean.items():
        assert len(d) >= 10, f"family {f}: only {len(d)} non-NaN parents"
    fams = sorted(clean)
    point = float(np.mean([clean[f].mean() for f in fams]))
    bs = np.zeros(N_BOOT)
    for f in fams:
        d = clean[f]
        bs += d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    bs /= len(fams)
    assert np.isfinite(bs).all()
    return {"point": point, "p_one_sided": float(((bs >= 0).sum() + 1) / (N_BOOT + 1)),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "upper95_one_sided": float(np.quantile(bs, 0.95)),
            "family_points": {f: float(clean[f].mean()) for f in fams},
            "n_parents": {f: int(len(clean[f])) for f in fams}}


def one_family(d: np.ndarray, seed: int = 17) -> dict:
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"point": float(d.mean()), "p_one_sided": float(((bs >= 0).sum() + 1) / (N_BOOT + 1)),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "n": int(len(d))}


def holm(ps: dict, alpha: float = 0.05) -> dict:
    """Monotone Holm step-down."""
    order = sorted(ps, key=ps.get)
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        prev = max(prev, min(1.0, ps[k] * (m - i)))
        out[k] = {"p_holm": prev, "holm_significant": prev < alpha}
    return out


def bh(ps: dict, q: float = 0.10) -> dict:
    keys = sorted(ps, key=ps.get)
    m = len(keys)
    adj, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[keys[i]] * m / (i + 1))
        adj[keys[i]] = prev
    return {k: {"p_bh": adj[k], "bh_significant": adj[k] < q} for k in keys}


def exact_rates(rows_conf, gen):
    out = {}
    for fam in env.FAMILIES:
        for k in (1, 2, 4):
            te = [r for r in rows_conf if r["family"] == fam and r["k"] == k]
            for mode in ("e2e", "chain_one", "chain_m1"):
                if mode not in gen:
                    continue
                ok = parsed = 0
                for r in te:
                    if mode == "e2e":
                        g = gen["e2e"].get(r["item_id"])
                        st = None if g is None else env.parse_state(fam, g["text"], lenient=True)
                        st = None if st is None else env.dumps(st)
                    else:
                        g = gen[mode].get(r["cand_id"])
                        st = None if g is None else g["states"][k]
                    parsed += st is not None
                    ok += st == r["gold"]
                out[f"{fam}|k{k}|{mode}"] = {"exact": ok / len(te), "parse": parsed / len(te), "n": len(te)}
    return out


def twin_consistency(bank, gen):
    """P(exact(c0,k4) and exact(t0,k4)); P(pred(c0) != pred(t0)) for e2e and chain_one."""
    res = {}
    for fam in env.FAMILIES:
        ps = [p for p in bank["parents"] if p["family"] == fam and "twin" in p]
        for mode in ("e2e", "chain_one"):
            both = differ = n = 0
            for p in ps:
                c0, t0 = p["candidates"][0], p["twin"]
                g0 = env.dumps(env.rollout(fam, p["s0"], c0["events"])[-1])
                gt = env.dumps(env.rollout(fam, p["s0"], t0["events"])[-1])
                if mode == "e2e":
                    a = gen["e2e"].get(f"{c0['cand_id']}-k4"); b = gen["e2e"].get(f"{t0['cand_id']}-k4")
                    pa = None if a is None else env.parse_state(fam, a["text"], lenient=True)
                    pb = None if b is None else env.parse_state(fam, b["text"], lenient=True)
                    pa = None if pa is None else env.dumps(pa); pb = None if pb is None else env.dumps(pb)
                else:
                    a = gen[mode].get(c0["cand_id"]); b = gen[mode].get(t0["cand_id"])
                    pa = None if a is None else a["states"][4]; pb = None if b is None else b["states"][4]
                n += 1
                both += (pa == g0) and (pb == gt)
                differ += (pa is not None and pb is not None and pa != pb)
            if n:
                res[f"{fam}|{mode}"] = {"n": n, "both_exact": both / n, "predictions_differ": differ / n}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="bank_pilot")
    ap.add_argument("--confirm", default="bank_confirm")
    ap.add_argument("--s-abundant", default="bank_s4000")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gen-stem", default=None, help="dry run only: where the test generations live")
    a = ap.parse_args()
    t0 = time.time()
    pilot_bank = data.load_bank(a.pilot)
    pilot_rows = [r for r in data.candidate_table(pilot_bank) if r["split"] in ("train", "dev")]
    if a.dry_run:
        conf_bank = pilot_bank
        conf_rows = [r for r in data.candidate_table(pilot_bank) if r["split"] == "test"]
        gen_dir = ROOT / "runs" / (a.gen_stem or a.pilot)
        arm = data.Arm(a.pilot, "sparse")
    else:
        conf_path = ROOT / "data" / f"{a.confirm}.json"
        assert sha(conf_path) == CONFIRM_SHA, "confirmation bank hash mismatch"
        conf_bank = data.load_bank(a.confirm)
        conf_rows = data.candidate_table(conf_bank)
        assert conf_rows and all(r["split"] == "test" for r in conf_rows)
        gen_dir = ROOT / "runs" / a.confirm
        for x in ("sparse", "supa"):
            d = gen_dir / x
            assert (d / "DONE.json").exists(), f"{x}: confirmation run incomplete"
            assert json.loads((d / "index.json").read_text(encoding="utf-8"))["bank_sha256"] == CONFIRM_SHA
            assert json.loads((d / "provenance.json").read_text(encoding="utf-8"))["param_fingerprint"].startswith(FP[x]), f"{x}: fingerprint"
        arm = data.ArmUnion([data.Arm(a.pilot, "sparse"), data.Arm(a.confirm, "sparse")])
    gen = {x: load_gen(gen_dir / x) for x in ("sparse", "supa")}
    for x, g in gen.items():                       # integrity: every confirmation item / chain generated
        for mode in ("chain_one", "chain_m1"):
            assert {r["cand_id"] for r in conf_rows} <= set(g[mode]), f"{x} {mode}: missing chains"
            assert all(len(v["states"]) == 5 for v in g[mode].values()), f"{x} {mode}: chain length"
        assert {r["item_id"] for r in conf_rows} <= set(g["e2e"]), f"{x} e2e: missing items"
    rows = pilot_rows + conf_rows
    s_rows = None
    sp = ROOT / "data" / f"{a.s_abundant}.json"
    if sp.exists():
        sb = json.loads(sp.read_text(encoding="utf-8"))
        s_rows = [r for r in data.candidate_table(sb) if r["k"] == 4 and r["family"] in PRIMARY_FAMS]
    results = {"dry_run": a.dry_run, "per_family": {}, "inputs": {}}
    per_parent = {}
    for fam in env.FAMILIES:
        for k in (1, 2, 4):
            ev = prereg1.evaluate(arm, rows, fam, k, gen["sparse"], gen["supa"], s_extra_rows=s_rows if k == 4 else None)
            per_parent[(fam, k)] = ev
            results["per_family"][f"{fam}|k{k}"] = {
                "n_test_parents": len(ev["pids"]), "means": {m: float(np.nanmean(v)) for m, v in ev["per_parent"].items()},
                "parse_rate": ev["parse_rate"], "selection": ev["selection"], "Gsupa_best_mode": ev.get("Gsupa_best_mode"),
                "B3_best_on_test_upper_bound": ev["B3_best_on_test_upper_bound"]}
            print(f"{fam[:6]} k{k}: " + " ".join(f"{m}={np.nanmean(v):.3f}" for m, v in sorted(ev["per_parent"].items())), flush=True)

    pp4 = {f: per_parent[(f, 4)]["per_parent"] for f in PRIMARY_FAMS}
    prim = {"C0": ("Gtitan_chain_one", "F_touched"), "C1": ("Gtitan_chain_one", "B3"),
            "C2": ("Gtitan_chain_one", "S"), "C3": ("Gtitan_chain_one", "Gsupa_best"),
            "C4": ("Gtitan_chain_one", "Gtitan_e2e")}
    out = {n: strat_boot({f: pp4[f][x] - pp4[f][y] for f in PRIMARY_FAMS}) for n, (x, y) in prim.items()}
    for n, h in holm({n: v["p_one_sided"] for n, v in out.items()}).items():
        out[n].update(h)
    for n, (x, y) in prim.items():
        v = out[n]
        v["contrast"] = f"{x} - {y}"
        v["PASS"] = bool(v["holm_significant"] and v["point"] <= MIN_EFFECT)
        fam_tests = {f: one_family(pp4[f][x] - pp4[f][y]) for f in PRIMARY_FAMS}
        v["per_family"] = fam_tests
        v["holds_in_both_families"] = all(t["point"] <= MIN_EFFECT and t["p_one_sided"] < 0.05 for t in fam_tests.values())
        if n in ("C1", "C2"):
            comp = np.mean([np.nanmean(pp4[f][y]) for f in PRIMARY_FAMS])
            floor = np.mean([np.nanmean(pp4[f]["F_touched"]) for f in PRIMARY_FAMS])
            v["comparator_at_or_below_floor"] = bool(comp >= floor)
        print(f"PRIMARY {n} {v['contrast']}: {v['point']:+.3f} CI {[round(c, 3) for c in v['ci95']]} p1 {v['p_one_sided']:.4g} "
              f"holm {v['p_holm']:.4g} -> {'PASS' if v['PASS'] else 'FAIL'}; both families {v['holds_in_both_families']}"
              + (f"; comparator<=floor {v['comparator_at_or_below_floor']}" if 'comparator_at_or_below_floor' in v else ""), flush=True)
    results["primary"] = out
    results["claim_confirmed"] = all(v["PASS"] for v in out.values())

    sec, ps = {}, {}
    for (fam, k), ev in per_parent.items():
        p = ev["per_parent"]
        pairs = {"G_chain - B3": ("Gtitan_chain_one", "B3"), "G_chain - S": ("Gtitan_chain_one", "S"),
                 "G_chain - F_touched": ("Gtitan_chain_one", "F_touched"), "G_chain - G_e2e": ("Gtitan_chain_one", "Gtitan_e2e"),
                 "G_chain - Gsupa_best": ("Gtitan_chain_one", "Gsupa_best"), "G_chain - S_abundant": ("Gtitan_chain_one", "S_abundant"),
                 "G_chain - F_first_only (partial simulator)": ("Gtitan_chain_one", "F_first_only"),
                 "B3 - B2 out-of-span (sub4)": ("B3", "B2_bilinear@sub4"), "B3 - B2 in-span (full16)": ("B3", "B2_bilinear@full16"),
                 "G_chain(fill last-parsed) - Gsupa_chain_one(fill last-parsed)": ("Gtitan_chain_one_fill_lastparsed", "Gsupa_chain_one_fill_lastparsed")}
        for name, (x, y) in pairs.items():
            if x in p and y in p and (fam, k, name) != (None,):
                key = f"{fam}|k{k}|{name}"
                sec[key] = one_family(p[x] - p[y])
                ps[key] = sec[key]["p_one_sided"]
    for kk, v in bh(ps).items():
        sec[kk].update(v)
    results["secondary"] = sec
    results["secondary_note"] = ("Descriptive; one-sided p for 'difference < 0'; BH q=0.10 across all listed. "
                                 "Pre-declared directional prediction: P at k=4, G_chain - B3 >= 0 and chain exact < e2e exact.")
    results["exact_state"] = {x: exact_rates(conf_rows, gen[x]) for x in gen}
    results["twins"] = {x: twin_consistency(conf_bank, gen[x]) for x in gen} if not a.dry_run else "n/a (pilot has no twins)"
    import torch
    results["inputs"] = {"code": {str(p.relative_to(ROOT)): sha(p) for p in sorted(list((ROOT / "rcwmt").glob("*.py")) + [ROOT / "scripts" / "confirm_analysis.py", ROOT / "scripts" / "analyze.py"])},
                         "pilot_bank": sha(ROOT / "data" / f"{a.pilot}.json"),
                         "confirm_bank": None if a.dry_run else CONFIRM_SHA,
                         "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
                         "finished_unix": time.time(), "seconds": round(time.time() - t0, 1)}
    out_dir = ROOT / "runs" / (a.pilot if a.dry_run else a.confirm) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "PREREG_1_DRYRUN" if a.dry_run else "PREREG_1_RESULTS"
    mode = "w" if a.dry_run else "x"
    with (out_dir / f"{stem}.json").open(mode, encoding="utf-8") as f:
        f.write(json.dumps(results, indent=1, default=float) + "\n")
    np.savez(out_dir / f"{stem}_PER_PARENT.npz",
             **{f"{fam}|k{k}|{m}": v for (fam, k), ev in per_parent.items() for m, v in ev["per_parent"].items()},
             **{f"{fam}|k{k}|pids": np.array(ev["pids"]) for (fam, k), ev in per_parent.items()})
    print(f"wrote {out_dir / stem}.json; claim confirmed: {results['claim_confirmed']}", flush=True)


if __name__ == "__main__":
    main()
