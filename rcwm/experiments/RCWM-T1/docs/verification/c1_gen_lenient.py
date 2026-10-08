"""C1: is Supa's ~0 generative accuracy a format/parse artifact? Lenient re-parse of gen_e2e.jsonl
and gen_chain_one.jsonl (supa). Extract an 'after' state from wrapper objects; report strict vs
lenient parse rate, state-exact and exact change-mask, plus how often the output equals s0 (copy)."""
from __future__ import annotations

import json
import re

import numpy as np

import vh
from vh import env, fidelity

D = vh.ROOT / "runs" / "bank_pilot" / "supa"
rows = {r["item_id"]: r for r in vh.rows_all()}
for r in list(rows.values()):
    if r["k"] == 1:
        rows[r["cand_id"] + "-one"] = dict(r, item_id=r["cand_id"] + "-one")


def lenient(fam, text):
    s = env.parse_state(fam, text)
    if s is not None:
        return s, "strict"
    t = text.strip().strip("`").strip()
    try:
        v = json.loads(t)
    except Exception:
        v = None
    if isinstance(v, dict):
        for key in ("after", "state", "result", "registers", "balances"):
            if key in v:
                s = env.parse_state(fam, json.dumps(v[key]))
                if s is not None:
                    return s, "wrapped"
    if fam == "prerequisite_plan":
        m = re.search(r'"after"\s*:\s*(\d+)\s*[},]', t)
        if m and 0 <= int(m.group(1)) < 256:
            return int(m.group(1)), "regex"
    else:
        m = re.search(r'"after"\s*:\s*(\{[^{}]*\})', t)
        if m:
            s = env.parse_state(fam, m.group(1))
            if s is not None:
                return s, "regex"
    return None, "fail"


out = {}
gen = [json.loads(l) for l in (D / "gen_e2e.jsonl").read_text(encoding="utf-8").splitlines()]
for g in gen:
    r = rows[g["item_id"]]
    fam = r["family"]
    frame = g["item_id"].rsplit("-", 1)[1]
    s, how = lenient(fam, g["text"])
    key = f"{env.SHORT[fam]}|{frame}"
    o = out.setdefault(key, {"n": 0, "strict": 0, "lenient": 0, "state_exact_strict": 0, "state_exact_lenient": 0,
                             "delta_exact_lenient_parsed": 0, "copy_s0": 0, "truncated_or_other": 0})
    o["n"] += 1
    o["strict"] += how == "strict"
    o["lenient"] += s is not None
    gold = r["states"][-1]
    if s is not None:
        ok = env.dumps(s) == env.dumps(gold)
        o["state_exact_strict"] += ok and how == "strict"
        o["state_exact_lenient"] += ok
        o["delta_exact_lenient_parsed"] += bool(np.array_equal(fidelity.delta_of(fam, r["s0"], s), fidelity.delta_of(fam, r["s0"], gold)))
        o["copy_s0"] += env.dumps(s) == env.dumps(r["s0"])
    else:
        o["truncated_or_other"] += 1
for k, o in sorted(out.items()):
    n = o["n"]
    o2 = {"n": n, "parse_strict": round(o["strict"] / n, 3), "parse_lenient": round(o["lenient"] / n, 3),
          "state_exact_strict": round(o["state_exact_strict"] / n, 3),
          "state_exact_lenient": round(o["state_exact_lenient"] / n, 3),
          "state_exact_among_lenient_parsed": round(o["state_exact_lenient"] / max(o["lenient"], 1), 3),
          "delta_exact_among_parsed": round(o["delta_exact_lenient_parsed"] / max(o["lenient"], 1), 3),
          "output_equals_s0_among_parsed": round(o["copy_s0"] / max(o["lenient"], 1), 3)}
    out[k] = o2
    print(k, o2)
# gold copy rate (how often truth == s0) for reference
for fam in vh.FAMS:
    te = [r for r in vh.rows_all() if r["family"] == fam and r["k"] == 1 and r["split"] == "test"]
    print(fam, "truth==s0 at k1:", round(np.mean([env.dumps(r["states"][-1]) == env.dumps(r["s0"]) for r in te]), 3))
(vh.ROOT / "docs" / "verification" / "c1_gen_lenient.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
