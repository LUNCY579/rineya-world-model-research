"""F-04 follow-up: gates selected with a below-range calibration limit (no model calls beyond the stored passes).

Calibration keys: doubles at limits 80/112 (above the training range; used for the original band) plus 24/28
(below it; runs/A1_lowcal, used by no task and no grading table). Gate families, each selected on the calibration
keys only (max accuracy, ties -> fewer verifier calls):
  band      defer if lo < P(legal) < hi, hi <= 0.5          (the original family, re-selected)
  band2     defer if lo < P(legal) < hi, hi up to 0.999     (may also defer confident 'legal' calls)
  sym       defer if |P - 0.5| < t
Evaluated on the test strata (flips at 96/128, odd limits 21-61, ID, range, illegal) and on the validation planners.
Writes runs/A1_lowcal/twosided.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import rw2  # noqa: E402
import f03_ladder as L3  # noqa: E402
import f_factorized as FF  # noqa: E402

LOW = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(ROOT / "runs/A1_lowcal/keys.jsonl", encoding="utf-8")}


def recs_for(rows):
    out = []
    for k, r in rows.items():
        if k[1] != "double":
            continue
        ref = rw2.exact_step(*k)
        fr = rw2.parse_forward(r.get("free", "") or "")
        lm_ok = (fr if fr is not None else L3.G.flm.get(k)) == ref
        vp = L3.ver_pred(k, r)
        out.append({"k": k, "p": r["p_true"], "lm": lm_ok, "ver": (vp == ref) if vp is not None else lm_ok})
    return out


def acc(rs, rule):
    return sum((x["ver"] if rule(x["p"]) else x["lm"]) for x in rs) / len(rs), sum(rule(x["p"]) for x in rs) / len(rs)


def select(cal, rules):
    return max(rules, key=lambda nr: (acc(cal, nr[1])[0], -acc(cal, nr[1])[1]))


def main():
    a1 = recs_for(L3.G.rows)
    low = recs_for(LOW)
    cal_hi = [x for x in a1 if x["k"][2] in FF.CALIB_LIMITS]
    cal = cal_hi + low
    los = (0.001, 0.005, 0.01, 0.02, 0.05)
    fam = {
        "band": [(f"band({lo},{hi})", (lambda p, lo=lo, hi=hi: lo < p < hi)) for lo in los for hi in (0.3, 0.4, 0.5)],
        "band2": [(f"band2({lo},{hi})", (lambda p, lo=lo, hi=hi: lo < p < hi)) for lo in los
                  for hi in (0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999)],
        "sym": [(f"sym({t})", (lambda p, t=t: abs(p - 0.5) < t)) for t in (0.3, 0.4, 0.45, 0.47, 0.48, 0.49, 0.495, 0.499)],
    }
    chosen = {"original band(0.01,0.5) [calib 80/112 only]": ("band(0.01,0.5)", lambda p: 0.01 < p < 0.5),
              "never": ("never", lambda p: False)}
    for f, rules in fam.items():
        chosen[f"{f} [calib 80/112/24/28]"] = select(cal, rules)
    strata = defaultdict(list)
    for x in a1:
        for s in FF.stratum(*x["k"]) + ["all"]:
            strata[s].append(x)
    strata["lowcal(calib)"] = low
    out = {"selected": {}, "strata": {}}
    print(f"{'gate':44s} " + " ".join(f"{s:>10s}" for s in ("all", "flip_type", "odd_limit", "id", "range", "illegal", "lowcal(calib)")))
    for name, (rid, rule) in chosen.items():
        out["selected"][name] = rid
        row = {}
        for s in ("all", "flip_type", "odd_limit", "id", "range", "illegal", "lowcal(calib)"):
            a, r = acc(strata[s], rule)
            row[s] = {"acc": round(a, 4), "call_rate": round(r, 4)}
        out["strata"][name] = row
        print(f"{name[:44]:44s} " + " ".join(f"{row[s]['acc']:.3f}/{row[s]['call_rate']:.2f}" for s in row))
    # validation planners under each selected gate (tables over task limits only)
    from bwmn import planner
    pools = rw2.build_pools()
    out["planning"] = {}
    for name, (rid, rule) in chosen.items():
        tab = {}
        for Lm in FF.TABLE_LIMITS:
            for x in range(-Lm, Lm + 1):
                for act in rw2.ACTIONS:
                    k = (x, act, Lm)
                    r = L3.G.rows.get(k)
                    pred = L3.G.lm_pred(k)
                    if act == "double" and r is not None and rule(r["p_true"]):
                        vp = L3.ver_pred(k, r)
                        pred = vp if vp is not None else pred
                    tab[k] = pred
        res = {}
        for split, (pool, n) in {"id_val": ("id", 400), "range_val": ("range", 300), "limitonly_val": ("limitonly", 300)}.items():
            tasks = rw2.eval_tasks(pools, pool, "validation", n, seed=0)
            f = planner.TableForward(tab)
            res[f"{split}/G2F"] = round(planner.summarize(planner.greedy_under_f(tasks, f, 2))["success"], 3)
            res[f"{split}/BFS256F"] = round(planner.summarize(planner.bfs_under_f(tasks, f, 256))["success"], 3)
        out["planning"][name] = res
        print(name[:44], res, flush=True)
    (ROOT / "runs/A1_lowcal/twosided.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
