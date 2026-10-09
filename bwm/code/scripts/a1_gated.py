"""A1 follow-up (no model calls): uncertainty-gated verification. For doubles, use the frozen-backbone verifier only
when F_LM's own P(legal=true) lies in a band (lo, hi); otherwise trust F_LM. The band is chosen on calibration limits
80/112 only (no task uses them), then evaluated on the FLIP-EVAL strata and validation planners."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import planner, rw2
import f_factorized as FF

rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(ROOT / "runs/A1_factorized/keys.jsonl", encoding="utf-8")}
flm = {tuple(k): (tuple(v) if v else None) for k, v in json.loads((ROOT / "runs/F_titan_main/F_TABLE.json").read_text())}

def lm_pred(key):
    r = rows.get(key)
    if r is None:
        return flm.get(key)
    fr = rw2.parse_forward(r.get("free", "") or "")
    return fr if fr is not None or key not in flm else flm.get(key)

def gated(key, lo, hi):
    x, act, L = key
    r = rows.get(key)
    if act != "double" or r is None or not (lo < r["p_true"] < hi):
        return lm_pred(key)
    leg = FF._bool(r["ver1"])
    if leg is None or (leg and r["cand"] is None):
        return lm_pred(key)
    return (r["cand"], True, False) if leg else (x, False, False)

calib = [k for k in rows if k[2] in FF.CALIB_LIMITS]
grid = [(lo, hi) for lo in (0.001, 0.005, 0.01, 0.02, 0.05) for hi in (0.5, 0.7, 0.9, 0.95, 0.99)]
best = max(grid, key=lambda b: (sum(gated(k, *b) == rw2.exact_step(*k) for k in calib), -(b[1] - b[0])))
lo, hi = best
print("band chosen on calibration limits:", best, "calib acc", round(sum(gated(k, lo, hi) == rw2.exact_step(*k) for k in calib) / len(calib), 4))
strata = defaultdict(Counter)
for k in rows:
    ref = rw2.exact_step(*k)
    for s in FF.stratum(*k) + ["all"]:
        strata[s]["n"] += 1; strata[s]["lm"] += lm_pred(k) == ref; strata[s]["gated"] += gated(k, lo, hi) == ref
        strata[s]["verified_calls"] += (k[1] == "double" and lo < rows[k]["p_true"] < hi)
for s, c in sorted(strata.items()):
    print(f"{s:12s} n={c['n']:5d} lm={c['lm']/c['n']:.4f} gated={c['gated']/c['n']:.4f} verifier-calls={c['verified_calls']}")
table = {}
for L in FF.TABLE_LIMITS:
    for x in range(-L, L + 1):
        for act in rw2.ACTIONS:
            table[(x, act, L)] = gated((x, act, L), lo, hi)
pools = rw2.build_pools()
res = {}
for split, (pool, n) in {"id_val": ("id", 400), "long_val": ("long", 200), "range_val": ("range", 300), "limitonly_val": ("limitonly", 300)}.items():
    tasks = rw2.eval_tasks(pools, pool, "validation", n, seed=0)
    for name, fn in (("G2F", lambda f: planner.greedy_under_f(tasks, f, 2)), ("BFS64F", lambda f: planner.bfs_under_f(tasks, f, 64)),
                     ("BFS256F", lambda f: planner.bfs_under_f(tasks, f, 256))):
        res[f"{split}/{name}"] = round(planner.summarize(fn(planner.TableForward(table)))["success"], 3)
    print(split, {k.split("/")[1]: v for k, v in res.items() if k.startswith(split)})
(ROOT / "runs/A1_factorized/gated.json").write_text(json.dumps({"band": best, "strata": {s: dict(c) for s, c in strata.items()}, "planning_validation": res}, indent=1))
