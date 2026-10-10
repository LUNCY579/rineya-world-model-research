"""Surface-form audit of the Blocksworld delta world model (the RCWM PREREG-3 'rule or trigger?' question, BWM side).

Same (state, action) keys rendered five ways; exact match against the simulator (delta answers, state-specific parse):
  canon        the training format (sorted JSON keys, sorted lists, block names from a..o)
  unseen_names block names p..z plus two-letter names (never seen in training)
  key_order    JSON keys of the user message and of the state object in reversed order
  list_order   facts inside each state list shuffled (clear/on/table)
  both         unseen names + reversed key order + shuffled lists
Keys: 5- and 10-block states (random towers + short walks), ~50% illegal actions. CPU, shared-prefix greedy decoding.
A rule-level model is invariant to these surface changes; a trigger-level model is not.
Usage: python scripts/bw_surface_audit.py --adapter runs/BW_Fdelta_s0/adapter --out runs/BW_Fdelta_s0/surface_audit.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import lm  # noqa: E402

UNSEEN = list("pqrstuvwxyz") + ["ab", "kx", "mo", "zq", "ty", "uv"]


def render(s, a, mode, rng):
    st = B.state_json(s)
    if mode in ("list_order", "both"):
        for k in ("clear", "on", "table"):
            rng.shuffle(st[k])
    user = {"action": B.action_str(a), "state": st}
    if mode in ("key_order", "both"):
        user = {"state": dict(reversed(list(st.items()))), "action": B.action_str(a)}
        text = json.dumps(user, separators=(",", ":"))  # explicit, non-sorted order
    elif mode == "list_order":
        text = json.dumps(user, separators=(",", ":"), sort_keys=True)  # keys sorted, lists shuffled
    else:
        text = B.canonical(user)
    return B.render_chat(B.SYS_FORWARD_DELTA, text)


def rename(s, a, objs, names):
    m = dict(zip(objs, names))
    rs = frozenset(tuple([f[0]] + [m[x] for x in f[1:]]) for f in s)
    ra = tuple([a[0]] + [m[x] for x in a[1:]])
    return rs, ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    keys = []
    for n in (5, 10):
        for _ in range(a.per_cell):
            objs = sorted(rng.sample(B.NAMES, n))
            s = B.towers_to_state(B.random_towers(objs, rng))
            for _ in range(rng.randrange(0, 5)):
                s, _ = B.step(s, rng.choice([x for x in B.ground_actions(objs) if B.applicable(s, x)]))
            ga = B.ground_actions(objs)
            leg = [x for x in ga if B.applicable(s, x)]
            ill = [x for x in ga if not B.applicable(s, x)]
            act = rng.choice(leg) if (rng.random() < 0.5 or not ill) else rng.choice(ill)
            keys.append((n, objs, s, act))
    rows = []
    for mode in ("canon", "unseen_names", "key_order", "list_order", "both"):
        for n, objs, s, act in keys:
            if mode in ("unseen_names", "both"):
                s2, a2 = rename(s, act, objs, rng.sample(UNSEEN, len(objs)))
            else:
                s2, a2 = s, act
            rows.append({"mode": mode, "n": n, "s": s2, "a": a2, "prompt": render(s2, a2, mode, rng)})
    model, tok = lm.load_core("titan-sparse")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ROOT / a.adapter), is_trainable=False)
    model.eval()
    gens = lm.greedy_shared_prefix(model, tok, [r["prompt"] for r in rows], max_new_tokens=40)
    n_, c_ = Counter(), Counter()
    for r, g in zip(rows, gens):
        ok = B.parse_forward_delta(g["text"], r["s"]) == B.step(r["s"], r["a"])
        for k in (r["mode"], f"{r['mode']}_n{r['n']}"):
            n_[k] += 1
            c_[k] += ok
    res = {k: {"n": n_[k], "exact": round(c_[k] / n_[k], 4)} for k in sorted(n_)}
    print(json.dumps(res, indent=0), flush=True)
    Path(ROOT / a.out).write_text(json.dumps({"adapter": a.adapter, "args": vars(a), "strata": res}, indent=1))


if __name__ == "__main__":
    main()
