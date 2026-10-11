import sys, random
from collections import Counter
sys.path.insert(0, ".")
from fuzz_reader import cases, val_listing, L
tot = Counter(); diffs = []
for seed in (2, 3):
    rng = random.Random(seed)
    for iid, init, goal, text, tags in cases(rng, 1000):
        v = L.val_run(L.planbench_path(iid), text, verbose=True)
        vl = val_listing(v["stdout"]); ours = L.read_val_plan(text); ex = L.execute_text(init, goal, text)
        okv = ex["success"] == v["valid"]
        okr = vl == "other" or (vl is None and ours is None) or (vl is not None and ours is not None and vl == sorted(ours))
        for t in set(tags) | {"all"}:
            tot[(t, "n")] += 1; tot[(t, "valid")] += v["valid"]; tot[(t, "bad")] += vl is None
        if not (okv and okr):
            diffs.append((seed, iid, tags, ascii(text[:200]), v["valid"], ex["success"]))
print("diffs", len(diffs))
for d in diffs[:20]: print(d)
for t in sorted({k[0] for k in tot}):
    print(f"{t:12s} n {tot[(t, 'n')]:5d} VAL-valid {tot[(t, 'valid')]:4d} bad-plan {tot[(t, 'bad')]:4d}")
