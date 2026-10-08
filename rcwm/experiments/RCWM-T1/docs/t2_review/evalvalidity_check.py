"""T2 review, evaluation-validity lens (CPU only, read-only on project data).

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/evalvalidity_check.py

1. Distribution shift between data/t2_eval.jsonl rows and pilot-TEST chain transitions (stratum mix,
   tick range, ledger balance range).
2. Instrument of the stored Titan-Sparse pilot chains (runs/bank_pilot/sparse/gen_chain_*.jsonl):
   strict-parse breaks that lenient parsing would have kept, and outputs at/near the old caps.
3. Baseline chain exact accuracy at k=1,2,4 per family (TEST, candidates only) with parent-cluster
   bootstrap SE, and G_chain regret SE via decisions.g_only -> minimum detectable effects.
4. Oracle-anchored per-stratum transition counts available on pilot TEST (OOD per-op check size).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rcwmt import data, decisions, env, runtime  # noqa: E402

spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "make_t2_data.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)
CODE = {"ledger_replay": "L", "register_machine": "R", "prerequisite_plan": "P"}

bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
test = [p for p in bank["parents"] if p["split"] == "test"]
ev_rows = [json.loads(l) for l in (ROOT / "data" / "t2_eval.jsonl").read_text(encoding="utf-8").splitlines()]
tr_rows = [json.loads(l) for l in (ROOT / "data" / "t2_train.jsonl").read_text(encoding="utf-8").splitlines()]

print("== 1. stratum mix: pilot TEST chain transitions (all steps, oracle states) vs t2_eval")
mix, ticks, bal = Counter(), defaultdict(list), {"chain": [], "t2eval": []}
for p in test:
    fam = p["family"]
    for c in p["candidates"]:
        st = env.rollout(fam, p["s0"], c["events"])
        for t, ev in enumerate(c["events"]):
            mix[(CODE[fam], t, mk.stratum(CODE[fam], st[t], ev))] += 1
            if fam != "prerequisite_plan":
                ticks["chain"].append(ev["tick"])
            if fam == "ledger_replay":
                bal["chain"] += list(st[t].values())
for r in ev_rows:
    if r["family"] != "prerequisite_plan":
        ticks["t2eval"].append(r["event"]["tick"])
    if r["family"] == "ledger_replay":
        bal["t2eval"] += list(r["before"].values())
agg = Counter()
for (f, t, s), n in mix.items():
    agg[(f, s)] += n
tot = Counter()
for (f, s), n in agg.items():
    tot[f] += n
for (f, s), n in sorted(agg.items()):
    print(f"   {f} {s:<16} n={n:4d}  share={n / tot[f]:.3f}")
for k, v in ticks.items():
    print(f"   tick {k}: min {min(v)} max {max(v)} median {int(np.median(v))}")
for k, v in bal.items():
    v = np.array(v)
    print(f"   ledger before-balance {k}: min {v.min()} max {v.max()} frac |b|>99 {np.mean(np.abs(v) > 99):.3f}")

print("== 2. stored Titan-Sparse pilot chains: parse/cap instrument")
tok = runtime.tokenizer()
gen = {}
for mode in ("chain_one", "chain_m1"):
    g = {}
    for l in (ROOT / "runs" / "bank_pilot" / "sparse" / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        g[r["cand_id"]] = r
    gen[mode] = g
fam_of = {c["cand_id"]: p["family"] for p in bank["parents"] for c in p["candidates"] + ([p["twin"]] if "twin" in p else [])}
for mode, g in gen.items():
    strict_breaks, lens = Counter(), defaultdict(list)
    for cid, r in g.items():
        fam = fam_of[cid]
        for t, txt in enumerate(r["texts"]):
            n = len(tok(txt, add_special_tokens=False)["input_ids"])
            lens[CODE[fam]].append(n)
            if r["states"][t + 1] is None and env.parse_state(fam, txt, lenient=True) is not None:
                strict_breaks[CODE[fam]] += 1
    print(f"   {mode}: chains {len(g)}; lenient-parsable steps recorded as None (strict break): {dict(strict_breaks)}")
    for f, v in sorted(lens.items()):
        v = np.array(v)
        cap = 40 if f == "P" else 64
        print(f"      {f}: answer tokens max {v.max()} p99 {np.percentile(v, 99):.0f}; >= old cap {cap}: {int((v >= cap).sum())}")

print("== 3. baseline chain exact accuracy (TEST candidates) and parent-cluster bootstrap SE")
rows = data.candidate_table(bank)
rng = np.random.default_rng(0)
for mode in ("chain_one", "chain_m1"):
    for fam in env.FAMILIES:
        line = []
        for k in (1, 2, 4):
            te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "test"]
            byp = defaultdict(list)
            for r in te:
                s = gen[mode][r["cand_id"]]["states"][k]
                byp[r["parent_id"]].append(float(s is not None and s == r["gold"]))
            pm = np.array([np.mean(v) for v in byp.values()])
            bs = [pm[rng.integers(0, len(pm), len(pm))].mean() for _ in range(2000)]
            line.append(f"k{k} {pm.mean():.3f}±{np.std(bs):.3f}")
        print(f"   {mode} {CODE[fam]} (n_par={len(pm)}): " + "  ".join(line))
print("   G regret (decisions.g_only, k=4), parent bootstrap SE")
for fam in env.FAMILIES:
    out = decisions.g_only(rows, fam, 4, gen, modes=("chain_one", "chain_m1"))
    for mode, v in out.items():
        x = np.asarray(v["nregret"])
        bs = [np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
        print(f"      {CODE[fam]} {mode}: nregret {np.nanmean(x):.3f} SE {np.std(bs):.3f} (n={len(x)}, nan={int(np.isnan(x).sum())})")

print("== 4. oracle-anchored pilot-TEST transitions per T2 stratum (unique payloads)")
uniq = defaultdict(set)
for p in test:
    fam = p["family"]
    for c in p["candidates"]:
        st = env.rollout(fam, p["s0"], c["events"])
        for t, ev in enumerate(c["events"]):
            uniq[(CODE[fam], mk.stratum(CODE[fam], st[t], ev))].add(env.dumps({"b": st[t], "e": ev}))
print("   " + ", ".join(f"{f}|{s}={len(v)}" for (f, s), v in sorted(uniq.items())))
print("== 5. t2 train/eval stratum counts")
print("   train", dict(Counter((CODE[r["family"]], r["stratum"]) for r in tr_rows)))
print("   eval ", dict(Counter((CODE[r["family"]], r["stratum"]) for r in ev_rows)))
