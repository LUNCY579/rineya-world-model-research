"""Self-test of the METHOD-B canonical form: (1) renaming invariance, (2) agreement with a brute-force
canonical form (min over all n! relabellings) as an equivalence relation, for n <= 6; (3) planted contamination."""
import itertools
import json
import random
import sys
import time

sys.argv = ["x", "NUL"]
src = open(__file__.replace("validate_b.py", "audit_b.py"), encoding="utf-8").read()
exec(src.replace('if __name__ == "__main__":\n    main()', ""))

rng = random.Random(12345)


def brute(objs, parts):
    best = None
    for perm in itertools.permutations(range(len(objs))):
        m = {o: perm[i] for i, o in enumerate(objs)}
        key = tuple(tuple(sorted((f[0],) + tuple(m[a] for a in f[1:]) for f in fs)) for _, fs in parts)
        if best is None or key < best:
            best = key
    return best


def rename_random(objs, i, g):
    names = ["z%02d" % k for k in range(len(objs) + 5)]
    rng.shuffle(names)
    m = dict(zip(objs, names))
    return sorted(m.values()), ren(i, m), ren(g, m)


t0 = time.time()
sets = load_evals()
probs = [(o, i, g, iid) for insts in sets.values() for iid, o, i, g in insts]
# sample training rows (all n) from three files
for path, every in (("runs/BW_data_v2/proposer_actions_train.jsonl", 9), ("runs/BW_data/proposer_actions_train.jsonl", 23),
                    ("runs/BW_DISTILL_r1/distill_rows.jsonl", 3)):
    for ln, line in enumerate(open(ROOT + "/" + path, encoding="utf-8"), 1):
        if ln % every:
            continue
        r = json.loads(line)
        _, u = user_json(r["prompt"])
        s = state_from(u["state"])
        g = frozenset(("on", x, y) for x, y in u["goal"])
        probs.append((sorted(blocks_in(s)), s, g, path + ":" + str(ln)))
print("problems", len(probs), flush=True)
# (1) invariance under random renaming (3 renamings each), all three forms
bad = 0
for o, i, g, pid in probs:
    ref = [c for c, _ in forms(o, i, g)]
    for _ in range(3):
        o2, i2, g2 = rename_random(o, i, g)
        if [c for c, _ in forms(o2, i2, g2)] != ref:
            bad += 1
            print("INVARIANCE FAIL", pid)
print("invariance: problems", len(probs), "x3 renamings, failures", bad, round(time.time() - t0, 1), "s", flush=True)
# (2) equivalence relation agrees with brute force for n <= 6 (cap 6000 problems)
small = [p for p in probs if len(p[0]) <= 6]
rng.shuffle(small)
small = small[:6000]
res = {}
for form, parts_of in (("full", lambda i, g: [("I", i), ("G", g)]), ("init", lambda i, g: [("I", i)]),
                       ("goal", lambda i, g: [("G", g)])):
    c2b, b2c = {}, {}
    viol = 0
    sub = small if form == "full" else small[:2500]
    for o, i, g, pid in sub:
        c = H(canon(o, parts_of(i, g))[0])
        b = (len(o), brute(o, parts_of(i, g)))
        if c2b.setdefault(c, b) != b or b2c.setdefault(b, c) != c:
            viol += 1
    res[form] = {"n": len(sub), "classes_canon": len(c2b), "classes_brute": len(b2c), "violations": viol}
    print(form, res[form], round(time.time() - t0, 1), "s", flush=True)
# (3) planted: rename a known eval instance and a known training row; must be found equal
o, i, g, pid = probs[0]
o2, i2, g2 = rename_random(o, i, g)
print("planted equal:", forms(o, i, g)[0][0] == forms(o2, i2, g2)[0][0])
print("STATS", STATS, "wall", round(time.time() - t0, 1))
