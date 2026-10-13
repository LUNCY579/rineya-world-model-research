"""Read-only: for >= 8-block eval instances, list every K-prefilter candidate among distill rows (the only training
rows with >= 7 blocks), decide each by find_iso, and confirm n == 8 candidates with the repo B.signature (40,320
permutations). Also prints the examples behind the test-set init-only / goal-only / F-state counts."""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
SP = Path(sys.argv[1])
sys.path.insert(0, str(SP))
from audit_common import B, ROOT, state_key, goal_key, find_iso, user_json, objs_of  # noqa: E402
import audit_main as AM  # noqa: E402

t0 = time.time()
S = AM.load_eval_sets()
TEST = ("hard", "mirror", "b3fresh", "val_large", "seeds_fresh")
EV = defaultdict(list)
for s in TEST + ("distill240",):
    for e in S[s]:
        e["sk"], e["gk"] = state_key(e["init"], e["objs"]), goal_key(e["goal"], e["objs"])
        EV[(e["sk"], e["gk"])].append((s, e))
cand = Counter()
iso = Counter()
sig8 = Counter()
for ln, line in enumerate(open(ROOT / "runs/BW_DISTILL_r1/distill_rows.jsonl", encoding="utf-8"), 1):
    r = json.loads(line)
    d = user_json(r["prompt"])
    st = B.state_from_json(d["state"])
    g = frozenset(("on", x, y) for x, y in d["goal"])
    objs = objs_of(st, g)
    K = (state_key(st, objs), goal_key(g, objs))
    for s, e in EV.get(K, []):
        n = len(objs)
        cand[(s, n)] += 1
        m = find_iso((e["objs"], e["init"], e["goal"]), (objs, st, g))
        if m is not None:
            iso[(s, n)] += 1
        if n == 8 and s in TEST:
            eq = B.signature(objs, st, g) == B.signature(e["objs"], e["init"], e["goal"])
            sig8[(s, "sig_equal" if eq else "sig_differs", "iso" if m else "non-iso")] += 1
print("K-candidates (set, n) among distill rows:", dict(sorted(cand.items())))
print("find_iso positives:", dict(sorted(iso.items())))
print("n=8 test-set candidates, repo signature vs find_iso:", dict(sig8))
R = json.load(open(SP / "audit_results.json"))
for s in TEST:
    for f in ("forward_delta_train", "v2_proposer_actions_train", "distill_rows", "mix_v3"):
        P = R["pairs"][f"{s} x {f}"]
        keys = [k for k in P if k.endswith("first20")]
        for k in keys:
            ids = sorted({x[0] if isinstance(x, list) else x["eval_id"] for x in P[k]})
            if ids:
                print(f"{s} x {f} {k}: {ids[:20]} e.g. {P[k][:3]}")
print(f"{time.time() - t0:.1f}s")
