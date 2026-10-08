"""RCWM-T2 review (lens: retention / confounds), part 2. CPU only, read-only.

  A. decision regret (decisions.g_only) of answer-free generators on pilot TEST: copy-chain
     (every step returns the input state) vs Titan-Sparse stored chains -> does the copy shortcut
     move regret, or only accuracy?
  B. Titan-Sparse's own false-copy rate (output == input although gold != input) and true-copy
     rate (output == input when gold == input), step 1, unique items, both frames -> the baseline
     a T2 "copy shortcut" check must be compared against.
  C. off-task first-token instrument: argmax first token of Titan-Sparse per prompt; how many are a
     single digit / '{' / '[' (where the first token cannot reveal a wrong number or JSON bleed).

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/retention_confounds_extra.py
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
from rcwmt import data, decisions, env, runtime  # noqa: E402

bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
cand = {c["cand_id"]: (p, c) for p in bank["parents"] for c in p["candidates"]}
gen = {}
for mode in ("chain_one", "chain_m1"):
    gen[mode] = {}
    for l in (ROOT / "runs/bank_pilot/sparse" / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        gen[mode][r["cand_id"]] = r

# ---- A. regret of copy-chain vs Titan-Sparse
copy = {"chain_one": {}}
for cid, (p, c) in cand.items():
    copy["chain_one"][cid] = {"states": [env.dumps(p["s0"])] * (len(c["events"]) + 1)}
rows = data.candidate_table(bank)
print("== A. g_only normalised regret (TEST parents; random selection = 1.0)")
for k in (1, 4):
    for fam in env.FAMILIES:
        t = decisions.g_only(rows, fam, k, gen, modes=("chain_one", "chain_m1"))
        cpy = [decisions.g_only(rows, fam, k, copy, modes=("chain_one",))["chain_one"]["nregret"] for _ in range(1)]
        print(f"   k={k} {fam[:1]}: Sparse chain_one {np.nanmean(t['chain_one']['nregret']):.3f}  "
              f"chain_m1 {np.nanmean(t['chain_m1']['nregret']):.3f}  copy-chain {np.nanmean(cpy[0]):.3f}")

# ---- B. Titan-Sparse false-copy / true-copy rates at step 1 (unique items)
print("== B. Titan-Sparse step-1 copy behaviour (unique (s0,e0) on TEST)")
for mode in ("chain_one", "chain_m1"):
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])  # n_goldcopy, copied_when_goldcopy, n_change, copied_when_change
    seen = set()
    for cid, r in gen[mode].items():
        p, c = cand[cid]
        if p["split"] != "test":
            continue
        key = env.dumps({"b": p["s0"], "e": c["events"][0]})
        if key in seen:
            continue
        seen.add(key)
        fam = p["family"]
        gold = env.step(fam, p["s0"], c["events"][0])
        out_copy = r["states"][1] == env.dumps(p["s0"])
        a = acc[fam[:1]]
        if gold == p["s0"]:
            a[0] += 1; a[1] += out_copy
        else:
            a[2] += 1; a[3] += out_copy
    print(f"   {mode}: " + "  ".join(f"{f}: copy|gold=input {v[1]}/{v[0]}  copy|gold!=input {v[3]}/{v[2]}"
                                      for f, v in sorted(acc.items())))

# ---- C. off-task first-token instrument
print("== C. off-task first token (Titan-Sparse argmax)")
tok = runtime.tokenizer()
LP = np.load(ROOT / "runs/bank_pilot/sparse/offtask_logprobs.npy", mmap_mode="r")
meta = json.loads((ROOT / "runs/bank_pilot/sparse/offtask_meta.json").read_text(encoding="utf-8"))
cls = meta["reply_class"]
arg = [int(np.argmax(LP[i])) for i in range(LP.shape[0])]
txt = [tok.decode([a]) for a in arg]
digit = [t.strip().isdigit() for t in txt]
brace = [t.strip()[:1] in ("{", "[") for t in txt]
print(f"   n={len(arg)}  argmax is a digit: {sum(digit)}  starts with {{ or [: {sum(brace)}")
dc = collections.Counter(c for c, d in zip(cls, digit) if d)
print("   digit-first classes:", dict(dc.most_common()))
print("   top argmax tokens:", collections.Counter(txt).most_common(15))
abst = {"insufficient_information", "uncertainty", "knowledge_boundary", "clarification", "clarification_request",
        "safety_boundary", "boundary_setting", "complex_escalation"}
print("   abstention/boundary prompts:", sum(c in abst for c in cls), "argmax tokens:",
      collections.Counter(t for c, t in zip(cls, txt) if c in abst).most_common(8))
