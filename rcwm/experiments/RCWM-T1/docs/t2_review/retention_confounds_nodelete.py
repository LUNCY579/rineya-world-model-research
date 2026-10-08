"""RCWM-T2 review (retention / confounds), part 3. CPU only, read-only.

Partial-rule shortcuts that a T2 adapter could learn without learning the full rule, scored on
t2_eval (per sub-stratum) and on pilot TEST chains (accuracy + g_only regret):
  P gate_or    oracle applicability gate, then state | add  (ignores delete)
  L/R          n/a (copy is covered in part 1)

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/retention_confounds_nodelete.py
"""
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, decisions, env  # noqa: E402

FAM = "prerequisite_plan"


def gate_or(s, ev):
    return s | ev["add"] if env.applicable(s, ev) else s


def sub(s, ev):
    if not env.applicable(s, ev):
        return "inapp"
    a = (s & ~ev["delete"]) | ev["add"]
    return "app_same" if a == s else ("app_del" if s & ev["delete"] else "app_addonly")


for split in ("train", "eval"):
    rows = [json.loads(l) for l in (ROOT / "data" / f"t2_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["family"] != FAM:
            continue
        k = sub(r["before"], r["event"])
        c[k][0] += 1
        c[k][1] += env.dumps(gate_or(r["before"], r["event"])) == r["target"]
    n = sum(v[0] for v in c.values())
    print(f"== t2_{split} P: gate_or acc {sum(v[1] for v in c.values()) / n:.3f}  " +
          "  ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(c.items())))

bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
gen = {"chain_one": {}}
steps = collections.Counter()
acc = collections.defaultdict(lambda: [0, 0])
for p in bank["parents"]:
    if p["family"] != FAM:
        continue
    for c in p["candidates"]:
        st = [p["s0"]]
        for ev in c["events"]:
            st.append(gate_or(st[-1], ev))
        gen["chain_one"][c["cand_id"]] = {"states": [env.dumps(x) for x in st]}
        if p["split"] == "test":
            g = env.rollout(FAM, p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                steps[sub(g[t], ev)] += 1
            for k in (1, 2, 4):
                acc[k][0] += 1
                acc[k][1] += st[k] == g[k]
print("== pilot TEST P chain steps by sub-stratum (oracle states):", dict(steps))
print("== pilot TEST P chain acc, gate_or: " + " ".join(f"k{k}={v[1] / v[0]:.3f}" for k, v in sorted(acc.items())))
rows = data.candidate_table(bank)
for k in (1, 4):
    t = decisions.g_only(rows, FAM, k, gen, modes=("chain_one",))
    print(f"   g_only nregret k={k}: gate_or {np.nanmean(t['chain_one']['nregret']):.3f}")
