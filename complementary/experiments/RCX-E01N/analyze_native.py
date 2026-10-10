"""RCX-E01N analysis: compare native arms (OFF, T2, SUPA from probe1; N-* cells) on the same DEV battery.

Usage (from complementary/):
  python -B experiments/RCX-E01N/analyze_native.py --probe runs/RCX-E01N/probe1 --cells runs/RCX-E01N/cells \
      --out runs/RCX-E01N/ANALYSIS_DEV.md
All arms must have been scored on the identical battery (checked by sha256). Descriptive unless a frozen
preregistration names the comparison. Unit of independence: the semantic parent.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcx import stats as ST  # noqa: E402

COMPOSITE = ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike", "V1_pos", "V2_neg", "V3_inverted",
             "V5_posted_override")
VARIED = ("V1_pos", "V2_neg", "V3_inverted", "V5_posted_override", "V6_compose", "V7_bigset")
STRATA = ("F1_void", "F2_unseen_real", "F3_unseen_nonce", "F4_nearmiss", "F5_postedlike", "V1_pos", "V2_neg",
          "V3_inverted", "V5_posted_override", "V6_compose", "V7_bigset", "M1_bal", "M2_amt")
CONTRASTS = [("N-CFSD", "N-VR", "PRIMARY: counterfactual/SD vs ordinary rule-varied data"),
             ("N-VR", "N-FX", "rule-varied vs fixed-rule data (matched)"),
             ("N-CFSD", "T2", "RCX treatment vs registered T2"),
             ("N-VR", "OFF", "training effect (C - A), ordinary"),
             ("N-CFSD", "OFF", "training effect (C - A), counterfactual"),
             ("N-FX", "T2", "RCX fixed-rule recipe vs T2 (sanity)")]


def load_preds(path):
    return [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]


def load_arms(probe, cells):
    battery = [json.loads(l) for l in open(Path(probe) / "battery.jsonl", encoding="utf-8")]
    bsha = hashlib.sha256((Path(probe) / "battery.jsonl").read_bytes()).hexdigest()
    arms = {}
    for arm in ("OFF", "T2", "SUPA"):
        p = Path(probe) / f"preds_{arm}.jsonl.gz"
        if p.exists():
            arms[arm] = [x for x in load_preds(p) if x.get("frame") != "verbatim"]
    for p in sorted(glob.glob(os.path.join(cells, "preds_N-*_s*.jsonl.gz"))):
        tag = os.path.basename(p)[len("preds_"):-len(".jsonl.gz")]
        ev = json.loads((Path(cells) / f"eval_{tag}.json").read_text())
        if ev["battery_sha256"] != bsha:
            raise SystemExit(f"{tag}: battery sha mismatch ({ev['battery_sha256'][:12]} vs {bsha[:12]})")
        arms[tag] = load_preds(p)
    for a, P in arms.items():
        assert len(P) == len(battery), (a, len(P), len(battery))
    return battery, bsha, arms


def cell(battery, P, stratum, surface, field="exact"):
    by = defaultdict(list)
    for r, p in zip(battery, P):
        if r["group"] == stratum and r["surface"] == surface:
            by[r["parent_id"]].append(p[field])
    vals = [sum(v) / len(v) for v in by.values()]
    return sum(vals) / len(vals) if vals else None


def parents(battery, P, groups, surface, field="exact"):
    by = defaultdict(list)
    for r, p in zip(battery, P):
        if r["group"] in groups and r["surface"] == surface:
            by[r["parent_id"]].append(p[field])
    return dict(by)


def noop_rate(battery, P, surface):
    k = n = 0
    for r, p in zip(battery, P):
        if r["surface"] == surface and r["group"] in VARIED and r["gold"]["decision"] == "noop":
            n += 1
            k += p["intent"] == "noop"
    return k, n


def rule_sensitivity(battery, P, surface):
    by = defaultdict(dict)
    for r, p in zip(battery, P):
        if r["group"] == "V3_inverted" and r["surface"] == surface:
            by[r["parent_id"]][r["variant"]] = (p["intent"], p["exact"])
    sens = sum(v["pos"][0] != v["neg"][0] for v in by.values()) / len(by)
    both = sum(v["pos"][1] and v["neg"][1] for v in by.values()) / len(by)
    return sens, both


def verdict(battery, P, surface):
    t = defaultdict(lambda: [0, 0])
    for r, p in zip(battery, P):
        if r["surface"] == surface and r["group"].startswith("F"):
            t[(r["group"], r["variant"])][0] += p["intent"] == "noop" if r["variant"] == "probe" else p["exact"]
            t[(r["group"], r["variant"])][1] += 1
    acc = {k: a / n for k, (a, n) in t.items()}
    unseen = [acc.get((g, "probe")) for g in ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike")]
    void = acc.get(("F1_void", "probe"))
    near = acc.get(("F4_nearmiss", "probe"))
    if None in unseen or void is None:
        return "NA", acc
    if all(u <= 0.3 for u in unseen) and void <= 0.3:
        v = "never no-op (ignores the rule)"
    elif all(u >= 0.8 for u in unseen):
        v = "generalises to unseen statuses (RULE-like)"
    elif all(u <= 0.3 for u in unseen) and void >= 0.8:
        v = "TRIGGER" + (" (trigger widened: near-misses)" if near is not None and near >= 0.8 else "")
    else:
        v = "partial"
    return v, acc


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="runs/RCX-E01N/probe1")
    ap.add_argument("--cells", default="runs/RCX-E01N/cells")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    battery, bsha, arms = load_arms(a.probe, a.cells)
    order = [x for x in ("OFF", "SUPA", "T2") if x in arms] + sorted(k for k in arms if k.startswith("N-"))
    L = [f"# RCX-E01N native analysis (DEV battery sha `{bsha[:16]}…`, {len(battery)} records)", "",
         "**Tiers:** OFF and SUPA are tier A (frozen cores). T2 and N-* are trained students (component adapters on frozen "
         "FP32 Titan-Sparse, unmerged). **DEV, descriptive.** Confirmation requires the frozen PREREG_E01N on a SEALED "
         "battery, 3 seeds, and E33 retention.", "",
         "## Summary", "",
         "| arm | composite in | composite ho_tpl | decision in | no-op when required (varied, in) | V3 sensitivity | "
         "V3 both-twins | verdict (F strata, in) |", "|---|---|---|---|---|---|---|---|"]
    for arm in order:
        P = arms[arm]
        comp = lambda s, f="exact": (sum(cell(battery, P, x, s, f) for x in COMPOSITE) / len(COMPOSITE))  # noqa: E731
        k, n = noop_rate(battery, P, "in")
        lo, hi = ST.wilson(k, n)
        sens, both = rule_sensitivity(battery, P, "in")
        v, _ = verdict(battery, P, "in")
        L.append(f"| {arm} | {fmt(comp('in'))} | {fmt(comp('ho_tpl'))} | {fmt(comp('in', 'decision_ok'))} | "
                 f"{k}/{n} [{lo:.3f}, {hi:.3f}] | {sens:.3f} | {both:.3f} | {v} |")
    L += ["", "## Per-stratum exact / decision (surface `in`)", "",
          "| arm | " + " | ".join(s.split("_")[0] for s in STRATA) + " |", "|---|" + "---|" * len(STRATA)]
    for arm in order:
        P = arms[arm]
        L.append(f"| {arm} | " + " | ".join(
            f"{cell(battery, P, s, 'in'):.2f}/{cell(battery, P, s, 'in', 'decision_ok'):.2f}" for s in STRATA) + " |")
    L += ["", "## Paired contrasts on the primary composite (parent-clustered bootstrap; descriptive on DEV)", "",
          "| A | B | contrast | surface | A | B | A−B | 95% CI | n parents |", "|---|---|---|---|---|---|---|---|---|"]
    for A, Bm, desc in CONTRASTS:
        ka = [k for k in arms if k == A or k.startswith(A + "_s")]
        kb = [k for k in arms if k == Bm or k.startswith(Bm + "_s")]
        if not ka or not kb:
            continue
        for surf in ("in", "ho_tpl"):
            pa = [parents(battery, arms[k], COMPOSITE, surf) for k in ka]
            pb = [parents(battery, arms[k], COMPOSITE, surf) for k in kb]
            res = ST.seed_parent_bootstrap(pb, pa, resamples=a.resamples, seed=20261231)
            L.append(f"| {A} | {Bm} | {desc} | {surf} | {res['mean_b']:.3f} | {res['mean_a']:.3f} | "
                     f"{res['diff']:+.3f} | [{res['lo']:+.3f}, {res['hi']:+.3f}] | {res['n_parents']} |")
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8", newline="\n")
    print(txt)


if __name__ == "__main__":
    main()
