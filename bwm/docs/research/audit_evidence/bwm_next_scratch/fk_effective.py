# Re-score P-12 items as the PLANNER would use them (parse_forward_delta's support check turns some wrong
# answers into a stop). Rebuilds the eval keys exactly as bw_keyorder.eval_ does; no model.
import sys, json, random, collections
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
import bw_keyorder as K
from bwmn import blocks as B
rng = random.Random(K.EVAL_SEED)
ks = K.keys(rng)
rows = []
for mode in ("canon", "train_order", "heldout_rev", "heldout_other", "heldout_both"):
    for i, (n, objs, s, act) in enumerate(ks):
        s2, a2 = (K.rename(s, act, objs, rng.sample(K.UNSEEN, len(objs))) if mode == "heldout_both" else (s, act))
        if mode == "canon":
            p = B.forward_prompt_delta(s2, a2)
        elif mode == "train_order":
            p = K.render(s2, a2, rng.choice([o for o in K.TRAIN_ORDERS if o != K.STATE_KEYS]), rng.random() < 0.5, True, rng)
        elif mode == "heldout_rev":
            p = K.render(s2, a2, ("table", "on", "hand", "clear"), True, False, rng)
        elif mode == "heldout_other":
            p = K.render(s2, a2, rng.choice([("hand", "table", "clear", "on"), ("on", "clear", "table", "hand")]), rng.random() < 0.5, False, rng)
        else:
            p = K.render(s2, a2, ("table", "on", "hand", "clear"), True, True, rng)
        import hashlib
        rows.append({"mode": mode, "i": i, "s": s2, "a": a2, "sha": hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]})
key = {(r["mode"], r["i"]): r for r in rows}
items = [json.loads(l) for l in open(R + "/runs/BW_KEYORDER/items.jsonl", encoding="utf-8")]
stats = collections.defaultdict(collections.Counter)
shamis = 0
for it in items:
    r = key[(it["mode"], it["i"])]
    if r["sha"] != it["prompt_sha"]: shamis += 1
    truth = B.step(r["s"], r["a"])
    p = B.parse_forward_delta(it["text"], r["s"])
    c = stats[(it["arm"], it["mode"])]
    c["n"] += 1
    if p == truth: c["exact"] += 1; c["eff_ok"] += 1; continue
    if truth[1]:  # legal action
        if p is None or not p[1]: c["false_stop_on_legal"] += 1
        else: c["wrong_state_on_legal"] += 1
    else:         # illegal action
        if p is None or not p[1]: c["eff_ok"] += 1; c["illegal_caught_by_parser_or_label"] += 1
        else: c["ILLEGAL_ACCEPTED"] += 1
print("prompt sha mismatches:", shamis)
for k in sorted(stats):
    c = stats[k]; n = c["n"]
    print(k, "exact %.3f eff %.3f" % (c["exact"]/n, c["eff_ok"]/n), {x: v for x, v in c.items() if x not in ("n", "exact", "eff_ok")})
