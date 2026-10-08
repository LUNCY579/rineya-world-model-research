"""Consequence fidelity per arm (H1) and across arms (H2), amended metrics (exact change mask).

  python scripts/fidelity_report.py --bank bank_pilot --arms sparse,supa [--ks 1,2,4] [--no-mlp]
  python scripts/fidelity_report.py --bank bank_pilot --compare sparse:supa,sparse:randdelta
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
from rcwmt import data, decide, decisions, env, fidelity  # noqa: E402
from scripts.analyze import load_gen  # noqa: E402


def run_arm(bank_stem, armname, ks, fams, mlp, symbolic, decision_ks=(), track=False, one=True):
    rows = data.candidate_table(data.load_bank(bank_stem))
    arm = data.Arm(bank_stem, armname)
    gen = load_gen(arm.dir)
    rep = {"arm": armname, "bank": bank_stem, "results": {}}
    if one:
        # Titan's trained one-step frame ("-one" items): same transitions as k=1, verbatim prompt.
        rows_one = [dict(r, item_id=r["cand_id"] + "-one") for r in rows if r["k"] == 1]
        for fam in fams:
            t0 = time.time()
            r = fidelity.analyse(arm, rows_one, fam, 1, gen, mlp=mlp, symbolic=symbolic)
            rep["results"][f"{fam}|one"] = r
            g = {m: round(v["exact_delta"], 3) for m, v in r.get("generative", {}).items() if m == "e2e"}
            print(f"[{armname}] {fam[:6]} one L{r['latent_ridge']['layer']:>2} ridgeΔ {r['latent_ridge']['exact_delta']:.3f}"
                  + (f" mlpΔ {r['latent_mlp']['exact_delta']:.3f}" if "latent_mlp" in r else "")
                  + f" | gen {g} | comp {r['comparator']['best']} {r['comparator']['exact_delta']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    for fam in fams:
        for k in ks:
            t0 = time.time()
            r = fidelity.analyse(arm, rows, fam, k, gen, mlp=mlp, symbolic=symbolic)
            rep["results"][f"{fam}|k{k}"] = r
            if track and k == max(ks):
                rep.setdefault("tracking", {})[fam] = fidelity.tracking(arm, rows, fam, 4)
                print(f"    tracking {fam[:6]}: " + " ".join(f"j{j}={v['exact_delta']:.3f}(copy {v['copy_floor_exact']:.3f},L{v['layer']})" for j, v in rep["tracking"][fam].items()), flush=True)
            if k in decision_ks:
                t1 = time.time()
                d = decisions.run(arm, rows, fam, k, r["latent_ridge"]["layer"], gen, subsets=2)
                rep.setdefault("decisions", {})[f"{fam}|k{k}"] = d
                fx = {m: round(v["nregret"]["mean"], 3) for m, v in d["fixed"].items()}
                full = d["budgets"][f"n{d['n_train_parents_full']}"]
                print(f"    decisions {fam[:6]} k{k}: fixed {fx}", flush=True)
                print(f"    full-budget {({m: round(v['mean'], 3) for m, v in full.items()})} ({time.time() - t1:.0f}s)", flush=True)
            g = {m: round(v["exact_delta"], 3) for m, v in r.get("generative", {}).items()}
            print(f"[{armname}] {fam[:6]} k{k} L{r['latent_ridge']['layer']:>2} "
                  f"ridgeΔ {r['latent_ridge']['exact_delta']:.3f}"
                  + (f" mlpΔ {r['latent_mlp']['exact_delta']:.3f}" if "latent_mlp" in r else "")
                  + f" | best comp {r['comparator']['best']} {r['comparator']['exact_delta']:.3f}"
                  + f" | gen {g} | input {r['input_decodability']['feat_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = ROOT / "runs" / bank_stem / "analysis" / f"fidelity_{armname}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, default=float) + "\n", encoding="utf-8")
    return rep


def compare(bank_stem, a, b):
    ra = json.loads((ROOT / "runs" / bank_stem / "analysis" / f"fidelity_{a}.json").read_text(encoding="utf-8"))
    rb = json.loads((ROOT / "runs" / bank_stem / "analysis" / f"fidelity_{b}.json").read_text(encoding="utf-8"))
    res = {}
    for key, x in ra["results"].items():
        if key not in rb["results"]:
            continue
        y = rb["results"][key]
        assert x["_test_parents"] == y["_test_parents"]
        rows = [{"parent_id": p} for p in x["_test_parents"]]
        entry = {}
        for metric in ("latent_ridge", "latent_mlp"):
            if metric in x["_per_item"] and metric in y["_per_item"]:
                pa = fidelity._parent_mean(rows, np.array(x["_per_item"][metric]))
                pb = fidelity._parent_mean(rows, np.array(y["_per_item"][metric]))
                entry[metric] = decide.paired_bootstrap(pa, pb)
        entry["input_decodability_diff"] = x["input_decodability"]["feat_acc"] - y["input_decodability"]["feat_acc"]
        res[key] = entry
        lr = entry.get("latent_ridge", {})
        print(f"{a}-{b} {key:<24} ridgeΔ diff {lr.get('mean_diff', float('nan')):+.3f} CI {[round(c, 3) for c in lr.get('ci95', [])]}"
              f" | input diff {entry['input_decodability_diff']:+.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--arms", default="")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--families", default=",".join(env.FAMILIES))
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--no-symbolic", action="store_true")
    ap.add_argument("--compare", default="")
    ap.add_argument("--decision-ks", default="")
    ap.add_argument("--tracking", action="store_true")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    for armname in [x for x in a.arms.split(",") if x]:
        run_arm(a.bank, armname, ks, a.families.split(","), not a.no_mlp, not a.no_symbolic,
                tuple(int(x) for x in a.decision_ks.split(",") if x), a.tracking)
    if a.compare:
        allres = {}
        for pair in a.compare.split(","):
            x, y = pair.split(":")
            allres[pair] = compare(a.bank, x, y)
        (ROOT / "runs" / a.bank / "analysis" / "fidelity_compare.json").write_text(json.dumps(allres, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
