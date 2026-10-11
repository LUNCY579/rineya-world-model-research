import json, sys, glob
from collections import Counter, defaultdict
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
for d in ("runs/BW_TEST_B2/hard_beam", "runs/BW_B3_HARD/h2_beam", "runs/BW_TEST_B2/mirror_beam", "runs/BW_TEST_B2/planbench_beam", "runs/BW_DISTILL_r1"):
    files = glob.glob(f"{ROOT}/{d}/shard_*/fcache.jsonl") + glob.glob(f"{ROOT}/{d}/*fcache*.jsonl")
    keys = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            keys[(frozenset(tuple(x) for x in r["s"]), tuple(r["a"]))] = r["text"]
    il = defaultdict(Counter); typ = Counter()
    for (s, a), t in keys.items():
        ns, legal = B.step(s, a)
        if legal: continue
        try: lab = json.loads(t).get("legal")
        except Exception: lab = None
        n = len(B.blocks_of(s))
        il[n]["claimed_legal" if lab is True else "said_illegal"] += 1
        if lab is True: typ[a[0]] += 1
    tot = sum(c["claimed_legal"] for c in il.values()); alln = sum(sum(c.values()) for c in il.values())
    print(d, len(files), "files; illegal keys", alln, "F claimed legal", tot, "(%.3f)" % (tot / max(1, alln)), dict(typ))
    print("   by n:", {n: "%d/%d" % (c["claimed_legal"], sum(c.values())) for n, c in sorted(il.items())})
