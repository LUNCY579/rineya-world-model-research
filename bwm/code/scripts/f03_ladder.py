"""F-03 deferral baseline ladder (no model calls; reuses runs/A1_factorized/keys.jsonl).

Question: is the uncertainty band (defer a double to the frozen-base verifier only when 0.01 < P(legal) < 0.5) better
than generic deferral at the same verifier-call rate? Policies, all over the same double-action keys:
  never     F_LM alone                         always   verifier on every double
  random    defer a random fraction r (exact expectation)
  conf      defer the r least-confident keys (smallest |P - 0.5|), the standard selective-prediction rule
  band      the pre-selected band (0.01, 0.5) (chosen on calibration limits 80/112 only)
  oracle    defer exactly the keys where F_LM is wrong and the verifier right (upper bound)
Reported per stratum (id, flip_type, range non-flip, odd/below-range limits, calib) at the band's own call rate, plus
risk-coverage curves (accuracy vs deferral rate) for random and conf. Also: verifier accuracy on F_LM's confident
errors (P outside the band), i.e. whether any gate could repair them.
Writes runs/A1_factorized/ladder.json.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import rw2  # noqa: E402
import f_factorized as FF  # noqa: E402

LO, HI = 0.01, 0.5


class G:
    """Same F_LM prediction rule as scripts/a1_gated.py (copied: importing it reruns its planning validation)."""
    rows = {tuple(json.loads(l)["key"]): json.loads(l)
            for l in open(ROOT / "runs/A1_factorized/keys.jsonl", encoding="utf-8")}
    flm = {tuple(k): (tuple(v) if v else None)
           for k, v in json.loads((ROOT / "runs/F_titan_main/F_TABLE.json").read_text())}

    @staticmethod
    def lm_pred(key):
        r = G.rows.get(key)
        if r is None:
            return G.flm.get(key)
        fr = rw2.parse_forward(r.get("free", "") or "")
        return fr if fr is not None or key not in G.flm else G.flm.get(key)


def ver_pred(key, r):
    x = key[0]
    leg = FF._bool(r["ver1"])
    if leg is None or (leg and r["cand"] is None):
        return None
    return (r["cand"], True, False) if leg else (x, False, False)


def main():
    recs = []
    for k, r in G.rows.items():
        if k[1] != "double":
            continue
        ref = rw2.exact_step(*k)
        lm_ok = G.lm_pred(k) == ref
        vp = ver_pred(k, r)
        ver_ok = vp == ref if vp is not None else lm_ok  # unusable verifier answer -> keep F_LM (as in the gate)
        recs.append({"k": k, "p": r["p_true"], "lm": lm_ok, "ver": ver_ok, "strata": FF.stratum(*k) + ["all"]})
    by = defaultdict(list)
    for x in recs:
        for s in x["strata"]:
            by[s].append(x)

    def acc(rs, defer):
        return sum((x["ver"] if d else x["lm"]) for x, d in zip(rs, defer)) / len(rs)

    out = {"n_doubles": len(recs), "band": [LO, HI], "strata": {}, "curves": {}}
    # deployable symmetric rule: defer when |P - 0.5| < t, with t set on the calibration limits (80/112) to match the
    # band's call rate there (the band was also selected on those limits only)
    cal = by["calib"]
    cal_rate = sum(LO < x["p"] < HI for x in cal) / len(cal)
    ds = sorted(abs(x["p"] - 0.5) for x in cal)
    t_conf = ds[min(len(ds) - 1, round(cal_rate * len(ds)))]
    out["conf_global_threshold"] = t_conf
    for s, rs in sorted(by.items()):
        band = [LO < x["p"] < HI for x in rs]
        cg = [abs(x["p"] - 0.5) < t_conf for x in rs]
        rate = sum(band) / len(rs)
        m = round(rate * len(rs))
        order = sorted(range(len(rs)), key=lambda i: abs(rs[i]["p"] - 0.5))
        conf = [False] * len(rs)
        for i in order[:m]:
            conf[i] = True
        rnd = (1 - rate) * sum(x["lm"] for x in rs) / len(rs) + rate * sum(x["ver"] for x in rs) / len(rs)
        oracle = [(not x["lm"]) and x["ver"] for x in rs]
        conf_err = [x for x, b in zip(rs, band) if not b and not x["lm"]]
        out["strata"][s] = {
            "n": len(rs), "band_call_rate": round(rate, 4),
            "never": round(acc(rs, [False] * len(rs)), 4), "always": round(acc(rs, [True] * len(rs)), 4),
            "random@rate": round(rnd, 4), "conf@rate": round(acc(rs, conf), 4), "band": round(acc(rs, band), 4),
            "oracle": round(acc(rs, oracle), 4), "oracle_call_rate": round(sum(oracle) / len(rs), 4),
            "conf_global": round(acc(rs, cg), 4), "conf_global_call_rate": round(sum(cg) / len(rs), 4),
            "confident_errors": len(conf_err),
            "verifier_right_on_confident_errors": sum(x["ver"] for x in conf_err),
        }
    for s in ("all", "id", "flip_type", "odd_limit", "calib"):
        rs = by.get(s)
        if not rs:
            continue
        order = sorted(range(len(rs)), key=lambda i: abs(rs[i]["p"] - 0.5))
        curve = []
        for f in [i / 20 for i in range(21)]:
            m = round(f * len(rs))
            d = [False] * len(rs)
            for i in order[:m]:
                d[i] = True
            rnd = (1 - f) * sum(x["lm"] for x in rs) / len(rs) + f * sum(x["ver"] for x in rs) / len(rs)
            curve.append({"rate": f, "conf": round(acc(rs, d), 4), "random": round(rnd, 4)})
        out["curves"][s] = curve
    (ROOT / "runs/A1_factorized/ladder.json").write_text(json.dumps(out, indent=1))
    print(f"{'stratum':12s} {'n':>5s} {'rate':>6s} {'never':>6s} {'always':>6s} {'random':>6s} {'conf':>6s} "
          f"{'band':>6s} {'oracle':>6s} {'confG':>6s} {'cGrate':>6s}  conf.err (verifier right)")
    for s, v in out["strata"].items():
        print(f"{s:12s} {v['n']:5d} {v['band_call_rate']:6.3f} {v['never']:6.3f} {v['always']:6.3f} "
              f"{v['random@rate']:6.3f} {v['conf@rate']:6.3f} {v['band']:6.3f} {v['oracle']:6.3f} "
              f"{v['conf_global']:6.3f} {v['conf_global_call_rate']:6.3f}  "
              f"{v['confident_errors']} ({v['verifier_right_on_confident_errors']})")


if __name__ == "__main__" and "--planning" not in sys.argv:
    random.seed(0)
    main()


def planning(t_conf: float) -> dict:
    """Validation planners (as in a1_gated.py) under three legality sources for doubles: F_LM, band, symmetric gate."""
    from bwmn import planner

    def table_for(rule):
        tab = {}
        for L in FF.TABLE_LIMITS:
            for x in range(-L, L + 1):
                for act in rw2.ACTIONS:
                    k = (x, act, L)
                    r = G.rows.get(k)
                    pred = G.lm_pred(k)
                    if act == "double" and r is not None and rule(r["p_true"]):
                        vp = ver_pred(k, r)
                        pred = vp if vp is not None else pred
                    tab[k] = pred
        return tab

    rules = {"never": lambda p: False, "band": lambda p: LO < p < HI, "confG": lambda p: abs(p - 0.5) < t_conf}
    pools = rw2.build_pools()
    res = {}
    for split, (pool, n) in {"id_val": ("id", 400), "long_val": ("long", 200), "range_val": ("range", 300),
                             "limitonly_val": ("limitonly", 300)}.items():
        tasks = rw2.eval_tasks(pools, pool, "validation", n, seed=0)
        for name, rule in rules.items():
            f = planner.TableForward(table_for(rule))
            res[f"{split}/G2F/{name}"] = round(planner.summarize(planner.greedy_under_f(tasks, f, 2))["success"], 3)
            res[f"{split}/BFS256F/{name}"] = round(planner.summarize(planner.bfs_under_f(tasks, f, 256))["success"], 3)
        print(split, {k.split("/", 1)[1]: v for k, v in res.items() if k.startswith(split)}, flush=True)
    return res


if __name__ == "__main__" and "--planning" in sys.argv:
    d = json.loads((ROOT / "runs/A1_factorized/ladder.json").read_text())
    d["planning_validation"] = planning(d["conf_global_threshold"])
    (ROOT / "runs/A1_factorized/ladder.json").write_text(json.dumps(d, indent=1))
