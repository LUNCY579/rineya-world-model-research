"""T2 review, evaluation-validity lens, part 2 (CPU only, read-only on project data).

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/evalvalidity_check2.py

1. Instrument noise of the stored Titan-Sparse pilot generations: the e2e '-k1' / '-one' items use
   the same prompts as chain_m1 / chain_one step 1 but were batched differently -> greedy flip rate.
2. P applicability: accuracy of surface shortcuts (drop pre_off, drop pre_on, pre_on==0) on T2
   train/eval vs pilot-TEST chain transitions (oracle states).
3. Ledger magnitude shift: share of pilot-TEST L transitions with |balance|>99 (before or after),
   per stratum, vs T2 data; P before-state bit counts.
4. Per-op MDE at n=40 (paired, McNemar-style) and chain/regret paired MDE upper bounds.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rcwmt import env  # noqa: E402

spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "make_t2_data.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)
CODE = {"ledger_replay": "L", "register_machine": "R", "prerequisite_plan": "P"}

bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
test = [p for p in bank["parents"] if p["split"] == "test"]
fam_of = {c["cand_id"]: p["family"] for p in bank["parents"] for c in p["candidates"]}
split_of = {c["cand_id"]: p["split"] for p in bank["parents"] for c in p["candidates"]}
rows = {s: [json.loads(l) for l in (ROOT / "data" / f"t2_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
        for s in ("train", "eval")}
G = ROOT / "runs" / "bank_pilot" / "sparse"

print("== 1. batch-composition flip rate (old instrument, same prompt, different batch)")
e2e = {}
for l in (G / "gen_e2e.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(l)
    e2e[r["item_id"]] = r["text"]
for mode, suf in (("chain_m1", "k1"), ("chain_one", "one")):
    n = same = 0
    by = Counter()
    for l in (G / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        k = f"{r['cand_id']}-{suf}"
        if k in e2e and r["texts"]:
            n += 1
            ok = e2e[k] == r["texts"][0]
            same += ok
            if not ok:
                by[CODE[fam_of[r["cand_id"]]]] += 1
    print(f"   {mode} step1 vs e2e -{suf}: n={n} identical={same} flips={n - same} by family {dict(by)}")

print("== 2. P applicability shortcuts (accuracy of predicting applicable/inapplicable)")
H = {"true": lambda s, e: env.applicable(s, e),
     "drop_pre_off": lambda s, e: (s & e["pre_on"]) == e["pre_on"],
     "drop_pre_on": lambda s, e: (s & e["pre_off"]) == 0,
     "pre_on==0": lambda s, e: e["pre_on"] == 0,
     "always_app": lambda s, e: True}


def shortcut_table(pairs, name):
    n = len(pairs)
    acc = {h: sum(f(s, e) == env.applicable(s, e) for s, e in pairs) / n for h, f in H.items()}
    inapp = sum(not env.applicable(s, e) for s, e in pairs) / n
    print(f"   {name:<26} n={n:4d} inapp={inapp:.3f} " + " ".join(f"{h}={v:.3f}" for h, v in acc.items() if h != "true"))


for s in ("train", "eval"):
    shortcut_table([(r["before"], r["event"]) for r in rows[s] if r["family"] == "prerequisite_plan"], f"t2_{s}")
pairs_all, pairs_s1, pairs_later = [], [], []
seen = set()
for p in test:
    if p["family"] != "prerequisite_plan":
        continue
    for c in p["candidates"]:
        st = env.rollout(p["family"], p["s0"], c["events"])
        for t, ev in enumerate(c["events"]):
            key = env.dumps({"b": st[t], "e": ev})
            if key in seen:
                continue
            seen.add(key)
            pairs_all.append((st[t], ev))
            (pairs_s1 if t == 0 else pairs_later).append((st[t], ev))
shortcut_table(pairs_all, "pilotTEST all steps uniq")
shortcut_table(pairs_s1, "pilotTEST step1 uniq")
shortcut_table(pairs_later, "pilotTEST steps2-4 uniq")
bc = lambda x: bin(x).count("1")
print("   before-state popcount: t2_eval mean %.2f | pilotTEST steps2-4 mean %.2f" % (
    np.mean([bc(r["before"]) for r in rows["eval"] if r["family"] == "prerequisite_plan"]),
    np.mean([bc(s) for s, _ in pairs_later])))

print("== 3. ledger magnitude: share of transitions touching |balance|>99 (before or after)")
big = defaultdict(lambda: [0, 0])
for p in test:
    if p["family"] != "ledger_replay":
        continue
    for c in p["candidates"]:
        st = env.rollout(p["family"], p["s0"], c["events"])
        for t, ev in enumerate(c["events"]):
            s = mk.stratum("L", st[t], ev)
            v = list(st[t].values()) + list(st[t + 1].values())
            big[s][0] += 1
            big[s][1] += any(abs(x) > 99 for x in v)
for s, (n, b) in sorted(big.items()):
    print(f"   pilotTEST L {s:<16} n={n:4d} big={b / n:.3f}")
for sp in ("train", "eval"):
    rr = [r for r in rows[sp] if r["family"] == "ledger_replay"]
    b = sum(any(abs(x) > 99 for x in list(r["before"].values()) + list(json.loads(r["target"]).values())) for r in rr)
    print(f"   t2_{sp} L: n={len(rr)} big={b / len(rr):.3f}")

print("== 4. minimum detectable effects")
def wilson_hw(p, n):
    z = 1.96
    d = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
for p0 in (0.0, 0.1, 0.45, 0.85):
    print(f"   per-op n=40, p={p0:.2f}: Wilson 95% half-width {wilson_hw(p0, 40):.3f}")
# exact McNemar (one-sided 0.025) with all discordant pairs in one direction: need b>=6
from math import comb
for b in range(1, 12):
    pv = sum(comb(b, i) for i in range(b, b + 1)) / 2 ** b
    if 2 * pv < 0.05:
        print(f"   McNemar two-sided p<0.05 needs >= {b} one-directional discordant items of 40 (= +{b / 40:.3f})")
        break
