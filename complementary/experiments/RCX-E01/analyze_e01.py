"""RCX-E01 analysis: per-arm tables, PREREG-3-style verdicts and parent-clustered paired contrasts.

Usage (from complementary/):
  python -B experiments/RCX-E01/analyze_e01.py runs/RCX-E01/pilot [--seeds 0] [--out REPORT.md]
Reads <run>/<arm>_s<seed>/summary.json and preds*.jsonl.gz plus the cached battery file.
Descriptive only unless a frozen preregistration names the comparison.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcx import stats as ST  # noqa: E402
from rcx.rcl import gen, score as S  # noqa: E402

COMPOSITE = S.PRIMARY_STRATA
CONTRASTS = [
    ("CFSD-ans", "VR-ans", "data factor (counterfactual+SD vs ordinary), answer-only [bundle 2x2 row]"),
    ("CFSD-trace", "VR-trace", "data factor, trace [bundle 2x2 row]"),
    ("VR-trace", "VR-ans", "trace vs answer, ordinary data [bundle 2x2 column]"),
    ("CFSD-trace", "CFSD-ans", "trace vs answer, counterfactual+SD data [bundle 2x2 column]"),
    ("CFSD-trace", "CFSD-fill", "trace vs constant filler (4b control)"),
    ("CF-ans", "VR-ans", "twins only (canonical surface)"),
    ("VRSD-ans", "VR-ans", "surface diversity + nonce only (no twins)"),
    ("CFSD-ans", "VRSD-ans", "twins given SD"),
    ("CFSD-ans", "FXSD-ans", "varied+CF rules vs fixed rule with SD (T2b analog)"),
    ("FXSD-ans", "FX-ans", "fixed rule: SD vs none (T2b vs T2 analog)"),
]


def load_arm(run, arm, seed):
    d = Path(run) / f"{arm}_s{seed}"
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())
    preds = [json.loads(l) for l in gzip.open(d / "preds.jsonl.gz", "rt", encoding="utf-8")]
    canon = None
    if (d / "preds_canon.jsonl.gz").exists():
        canon = [json.loads(l) for l in gzip.open(d / "preds_canon.jsonl.gz", "rt", encoding="utf-8")]
    return {"summary": summ, "preds": preds, "canon": canon}


def load_battery(run, summ):
    files = glob.glob(os.path.join(run, "battery_*.jsonl"))
    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        import hashlib
        if hashlib.sha256(Path(f).read_bytes()).hexdigest() == summ["battery_sha256"]:
            return recs
    raise FileNotFoundError("battery with matching sha not found")


def parent_dict(battery, preds, surface, strata=COMPOSITE, field="exact"):
    d = defaultdict(list)
    for r, p in zip(battery, preds):
        if r["surface"] == surface and r["group"] in strata:
            d[r["parent_id"]].append(p[field])
    return dict(d)


def probe(battery, preds, surface):
    t = defaultdict(lambda: [0, 0])
    for r, p in zip(battery, preds):
        if r["surface"] == surface and r["group"].startswith("F"):
            k = (r["group"], r["variant"])
            t[k][0] += p["exact"]
            t[k][1] += 1
    return {k: v[0] / v[1] for k, v in t.items()}


def errors_on_probes(battery, preds, surface, groups=("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike")):
    c = defaultdict(int)
    for r, p in zip(battery, preds):
        if r["surface"] == surface and r["group"] in groups and r["variant"] == "probe" and p["err"]:
            c[p["err"]] += 1
    return dict(c)


def verdict(pr, errs, bar=0.80, low=0.30):
    """PREREG-3 definitions adapted: RULE = every unseen probe stratum >= bar and posted twin guard >= bar;
    TRIGGER = every unseen probe <= low, void probe >= bar and most probe errors applied_when_noop."""
    unseen = [pr.get((g, "probe")) for g in ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike")]
    void = pr.get(("F1_void", "probe"))
    twins = [v for (g, var), v in pr.items() if var == "twin_posted"]
    guard = min(twins) if twins else None
    if None in unseen or void is None:
        return "NA"
    if all(u >= bar for u in unseen) and guard is not None and guard >= bar:
        return "RULE"
    tot = sum(errs.values())
    if all(u <= low for u in unseen) and void >= bar and tot and errs.get("applied_when_noop", 0) > tot / 2:
        return "TRIGGER"
    return "MIXED"


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resamples", type=int, default=5000)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = {}
    for d in sorted(glob.glob(os.path.join(args.run, "*_s*"))):
        name, s = os.path.basename(d).rsplit("_s", 1)
        if int(s) in seeds:
            a = load_arm(args.run, name, int(s))
            if a:
                arms.setdefault(name, {})[int(s)] = a
    if not arms:
        print("no arms found")
        return
    any_arm = next(iter(next(iter(arms.values())).values()))
    battery = load_battery(args.run, any_arm["summary"])
    L = [f"# RCX-E01 analysis: `{args.run}`, seeds {seeds}", "",
         "**Tier: mechanism** (small character-level transformers trained from scratch on CPU). These are not Rineya-core results.",
         f"Battery sha256 `{any_arm['summary']['battery_sha256'][:16]}…`, n_parents per stratum "
         f"{len({r['parent_id'] for r in battery if r['group'] == 'V1_pos'})}.",
         "", "## Per-arm summary (seed-averaged parent-macro exact accuracy)", "",
         "| arm | iid exact | P-in | P-ho | ho_tpl | ho_key | ho_name | ho_many | dec-in | dec-ho | verdict-in | verdict-ho | canon P-ho |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["FX-ans", "FXSD-ans", "VR-ans", "VR-trace", "VRSD-ans", "CF-ans", "CFSD-ans", "CFSD-trace", "CFSD-fill",
             "VRSD-trace", "CF-trace"]
    for arm in [a for a in order if a in arms] + [a for a in arms if a not in order]:
        runs = arms[arm]
        m = lambda f: sum(f(r) for r in runs.values()) / len(runs)
        comp = lambda s: m(lambda r: r["summary"]["composite_exact"][s])
        dec = lambda s: m(lambda r: r["summary"]["composite_decision"][s])
        r0 = next(iter(runs.values()))
        v_in = verdict(probe(battery, r0["preds"], "in"), errors_on_probes(battery, r0["preds"], "in"))
        v_ho = verdict(probe(battery, r0["preds"], "ho"), errors_on_probes(battery, r0["preds"], "ho"))
        canon = r0["summary"].get("canon", {}).get("composite_exact", {}).get("ho") if r0["summary"].get("canon") else None
        L.append(f"| {arm} | {fmt(m(lambda r: r['summary']['iid']['exact']))} | {fmt(comp('in'))} | {fmt(comp('ho'))} | "
                 f"{fmt(comp('ho_tpl'))} | {fmt(comp('ho_key'))} | {fmt(comp('ho_name'))} | {fmt(comp('ho_many'))} | "
                 f"{fmt(dec('in'))} | {fmt(dec('ho'))} | {v_in} | {v_ho} | {fmt(canon)} |")
    L += ["", "P-in / P-ho: composite over " + ", ".join(COMPOSITE) + " under the canonical (`in`) and combined "
          "held-out (`ho`) surfaces. dec = decision (apply vs no-op intent) accuracy. Verdicts use PREREG-3-style "
          "definitions on F-strata probes (seed 0).", "", "## Per-stratum exact accuracy (seed-averaged), surface `in` / `ho`", ""]
    hdr = "| arm | " + " | ".join(s.split("_")[0] for s in gen.STRATA) + " |"
    L += [hdr, "|---|" + "---|" * len(gen.STRATA)]
    for arm in [a for a in order if a in arms]:
        runs = arms[arm]
        cells = []
        for s in gen.STRATA:
            vi = sum(r["summary"]["cells"][f"{s}|in"]["exact"] for r in runs.values()) / len(runs)
            vh = sum(r["summary"]["cells"][f"{s}|ho"]["exact"] for r in runs.values()) / len(runs)
            cells.append(f"{vi:.2f}/{vh:.2f}")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "## Error directions on unseen-status probes (F2, F3, F5), seed 0, surface `in`", "",
          "| arm | applied_when_noop | false_copy | arith | format | V3 both-twins exact (in) |", "|---|---|---|---|---|---|"]
    for arm in [a for a in order if a in arms]:
        r0 = next(iter(arms[arm].values()))
        e = errors_on_probes(battery, r0["preds"], "in")
        v3 = r0["summary"]["cells"]["V3_inverted|in"]["parent_all_exact"]
        L.append(f"| {arm} | {e.get('applied_when_noop', 0)} | {e.get('false_copy', 0)} | {e.get('arith', 0)} | "
                 f"{e.get('format', 0)} | {v3:.3f} |")
    L += ["", "## Paired contrasts (parent-clustered bootstrap over seeds × parents; descriptive)", "",
          "| A | B | contrast | surface | mean A | mean B | diff | 95% CI | n parents |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b, desc in CONTRASTS:
        if a not in arms or b not in arms:
            continue
        common = sorted(set(arms[a]) & set(arms[b]))
        if not common:
            continue
        for surf in ("in", "ho"):
            pa = [parent_dict(battery, arms[a][s]["preds"], surf) for s in common]
            pb = [parent_dict(battery, arms[b][s]["preds"], surf) for s in common]
            # stats.seed_parent_bootstrap(x, y) reports y - x; pass (B, A) so diff = A - B.
            res = ST.seed_parent_bootstrap(pb, pa, resamples=args.resamples, seed=20261230)
            L.append(f"| {a} | {b} | {desc} | {surf} | {res['mean_b']:.3f} | {res['mean_a']:.3f} | "
                     f"{res['diff']:+.3f} | [{res['lo']:+.3f}, {res['hi']:+.3f}] | {res['n_parents']} |")
    # canonicalizer control
    L += ["", "## Canonicalizer control (tool tier B): same model, inputs canonicalized", "",
          "| arm | surface | plain | +canon |", "|---|---|---|---|"]
    for arm in [a for a in order if a in arms]:
        r0 = next(iter(arms[arm].values()))
        if not r0["summary"].get("canon"):
            continue
        for surf in ("in", "ho", "ho_tpl", "ho_key", "ho_name", "ho_many"):
            L.append(f"| {arm} | {surf} | {fmt(r0['summary']['composite_exact'][surf])} | "
                     f"{fmt(r0['summary']['canon']['composite_exact'][surf])} |")
    # floors
    L += ["", "## Shortcut floors and ceiling on the same battery", "", "| floor | P-in | P-ho |", "|---|---|---|"]
    tr = {reg: gen.make_train(reg, 4000, 20261300 + 100 * i) for i, reg in enumerate(gen.REGIMES, 1)}
    ctx = S.FloorContext(tr)
    for fl in S.FLOORS:
        res = [S.classify(x, S.floor_after(fl, x, ctx, "VR_SD")) for x in battery]
        sm = S.aggregate(battery, res)
        L.append(f"| {fl} | {fmt(S.composite(sm, 'in'))} | {fmt(S.composite(sm, 'ho'))} |")
    txt = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8", newline="\n")
    print(txt)


if __name__ == "__main__":
    main()
