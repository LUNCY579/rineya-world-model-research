import json, sys, glob
from collections import Counter, defaultdict
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
sets = {"b2_hard": "runs/BW_TEST_B2/hard_beam", "h2_hard": "runs/BW_B3_HARD/h2_beam",
        "b2_mirror": "runs/BW_TEST_B2/mirror_beam", "b2_pb600": "runs/BW_TEST_B2/planbench_beam"}
for name, d in sets.items():
    files = glob.glob(f"{ROOT}/{d}/shard_*/fcache.jsonl") or glob.glob(f"{ROOT}/{d}/fcache.jsonl")
    seen = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            k = (frozenset(tuple(x) for x in r["s"]), tuple(r["a"]))
            seen[k] = r["text"]
    c = Counter(); bysize = defaultdict(Counter)
    for (s, a), text in seen.items():
        ns, legal = B.step(s, a)
        o = B.parse_forward_delta(text, s)
        n = len(B.blocks_of(s))
        if o is None: kind = "unparsed_true_legal" if legal else "unparsed_true_illegal"
        else:
            ps, pl = o
            if pl and legal: kind = "ok" if ps == ns else "wrong_state"
            elif pl and not legal: kind = "false_legal"
            elif (not pl) and legal: kind = "false_illegal"
            else: kind = "ok"
        c[kind] += 1; c["illegal_true" if not legal else "legal_true"] += 1
        bysize[n][kind] += 1
    print(name, len(files), "files", len(seen), "keys", dict(c))
    errs = {n: {k: v for k, v in cc.items() if k != "ok"} for n, cc in sorted(bysize.items())}
    print("  errors by n:", {n: e for n, e in errs.items() if e})
